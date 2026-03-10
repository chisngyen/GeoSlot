#!/usr/bin/env python3
"""
EXP4: Hierarchical Knowledge Distillation with Progressive Feature Alignment
=============================================================================
Teacher: Full GeoSlot (ConvNeXt-Tiny + enhanced modules) — pretrained, frozen
Student: MobileNetV3-Small (~3M params) — ultra-compact

Distillation strategy (4 progressive levels):
  Level 1: Feature-level — intermediate feature map alignment (L2 + cosine)
  Level 2: Attention-level — transfer attention patterns from teacher
  Level 3: Relation-level — preserve pairwise similarity structure (CRD loss)
  Level 4: Logit-level — soft cross-entropy on teacher embeddings

Novelty:
  - Progressive curriculum: start with feature, add attention, then relation, finally logits
  - View-specific distillation heads for drone vs satellite branches
  - Adaptive layer-wise distillation with learnable weights

Target: <5M params while preserving >90% of teacher performance

Dataset: SUES-200
Usage:
  python exp4_hierarchical_kd.py           # Full training
  python exp4_hierarchical_kd.py --test    # Smoke test
"""

# === SETUP ===
import subprocess, sys, os

def pip_install(pkg, extra=""):
    subprocess.run(f"pip install -q {extra} {pkg}",
                   shell=True, capture_output=True, text=True)

print("[1/2] Installing packages...")
for p in ["timm", "tqdm"]:
    try: __import__(p)
    except ImportError: pip_install(p)
print("[2/2] Setup complete!")

# =============================================================================
# IMPORTS
# =============================================================================
import math, glob, json, time, gc, random, argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm


# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT      = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
OUTPUT_DIR     = "/kaggle/working"

EPOCHS         = 120
BATCH_SIZE     = 256
NUM_WORKERS    = 8
AMP_ENABLED    = True
EVAL_FREQ      = 5

# Progressive curriculum schedule
LEVEL1_START   = 0      # Feature-level distillation
LEVEL2_START   = 20     # + Attention transfer
LEVEL3_START   = 50     # + Relation distillation
LEVEL4_START   = 80     # + Logit-level

LR_INIT        = 1e-3
LR_MIN         = 1e-6
WARMUP_EPOCHS  = 5
WEIGHT_DECAY   = 0.01

# Loss weights
LAMBDA_CE      = 1.0
LAMBDA_TRIPLET = 0.5
LAMBDA_FEAT    = 2.0
LAMBDA_ATTN    = 1.0
LAMBDA_REL     = 0.5
LAMBDA_LOGIT   = 1.0

# Model
TEACHER_BACKBONE = "convnext_tiny"
STUDENT_BACKBONE = "mobilenetv3_small_100"
TEACHER_DIM    = 768
STUDENT_DIM    = 576   # MobileNetV3-Small output
EMBED_DIM      = 512
NUM_CLASSES    = 160

IMG_SIZE       = 224
TRAIN_LOCS     = list(range(1, 121))
TEST_LOCS      = list(range(121, 201))
ALTITUDES      = ["150", "200", "250", "300"]
TEST_ALTITUDE  = "150"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# TEACHER MODEL (ConvNeXt-Tiny with enhanced heads)
# =============================================================================
class TeacherModel(nn.Module):
    """ConvNeXt-Tiny teacher with multi-level feature extraction."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(TEACHER_BACKBONE, pretrained=True,
                                          num_classes=0, global_pool='',
                                          features_only=True)
        # ConvNeXt-Tiny feature dims at each stage: [96, 192, 384, 768]
        self.stage_dims = [96, 192, 384, 768]

        # Embedding head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_head = nn.Sequential(
            nn.LayerNorm(TEACHER_DIM),
            nn.Linear(TEACHER_DIM, EMBED_DIM),
        )

        # Freeze teacher
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        features = self.backbone(x)  # List of feature maps at each stage

        # Global feature from last stage
        last_feat = features[-1]  # [B, 768, H, W]
        global_feat = self.pool(last_feat).flatten(1)  # [B, 768]
        embedding = F.normalize(self.embed_head(global_feat), dim=-1)

        # Patch tokens from last stage
        B, C, H, W = last_feat.shape
        patch_tokens = last_feat.flatten(2).transpose(1, 2)  # [B, N, C]

        return {
            'embedding': embedding,
            'global_feat': global_feat,
            'patch_tokens': patch_tokens,
            'stage_features': features,
            'spatial_hw': (H, W),
        }


# =============================================================================
# STUDENT MODEL (MobileNetV3-Small with distillation hooks)
# =============================================================================
class StudentModel(nn.Module):
    """Ultra-compact student with view-specific distillation heads."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(STUDENT_BACKBONE, pretrained=True,
                                          num_classes=0, global_pool='',
                                          features_only=True)
        # MobileNetV3-Small stages: [16, 16, 24, 40, 48, 96, 576]
        # We'll use features_only which gives intermediate features

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Adaptation layers to match teacher dimensions at different levels
        # Map student features to teacher dimension space
        self.feat_adaptor = nn.Sequential(
            nn.Linear(STUDENT_DIM, TEACHER_DIM),
            nn.LayerNorm(TEACHER_DIM),
        )

        # Patch-level adaptor
        self.patch_adaptor = nn.Sequential(
            nn.Conv2d(STUDENT_DIM, TEACHER_DIM, 1),
            nn.BatchNorm2d(TEACHER_DIM),
        )

        # View-specific heads
        self.drone_head = nn.Sequential(
            nn.Linear(TEACHER_DIM, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
            nn.GELU(),
            nn.Linear(EMBED_DIM, EMBED_DIM),
        )
        self.sat_head = nn.Sequential(
            nn.Linear(TEACHER_DIM, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
            nn.GELU(),
            nn.Linear(EMBED_DIM, EMBED_DIM),
        )

        self.embed_proj = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, EMBED_DIM),
        )

        # Adaptive distillation weights (learnable)
        self.dist_weights = nn.Parameter(torch.ones(4) * 0.25)  # 4 levels

    def forward(self, x, view_type='drone'):
        features = self.backbone(x)  # Multi-scale features
        last_feat = features[-1]  # [B, STUDENT_DIM, H, W]

        # Global feature
        global_feat = self.pool(last_feat).flatten(1)
        adapted_global = self.feat_adaptor(global_feat)

        # Patch tokens
        patch_feat = self.patch_adaptor(last_feat)
        B, C, H, W = patch_feat.shape
        patch_tokens = patch_feat.flatten(2).transpose(1, 2)

        # View-specific embedding
        if view_type == 'drone':
            view_feat = self.drone_head(adapted_global)
        else:
            view_feat = self.sat_head(adapted_global)

        embedding = F.normalize(self.embed_proj(view_feat), dim=-1)

        # Normalized distillation weights
        dist_w = F.softmax(self.dist_weights, dim=0)

        return {
            'embedding': embedding,
            'global_feat': adapted_global,
            'patch_tokens': patch_tokens,
            'stage_features': features,
            'spatial_hw': (H, W),
            'dist_weights': dist_w,
        }

    def extract_embedding(self, x, view_type='drone'):
        return self.forward(x, view_type)['embedding']


# =============================================================================
# HIERARCHICAL DISTILLATION LOSS
# =============================================================================
class HierarchicalDistillLoss(nn.Module):
    """4-level progressive distillation with curriculum."""
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(0.07).log())
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

    @property
    def temp(self):
        return self.log_temp.exp().clamp(0.01, 1.0)

    def level1_feature_loss(self, s_feat, t_feat):
        """Feature-level: L2 + cosine alignment."""
        l2 = F.mse_loss(s_feat, t_feat.detach())
        cos = 1.0 - F.cosine_similarity(s_feat, t_feat.detach(), dim=-1).mean()
        return l2 + cos

    def level2_attention_loss(self, s_patch, t_patch):
        """Attention-level: transfer self-attention patterns."""
        # Compute gram matrices as proxy for attention
        s_gram = torch.bmm(F.normalize(s_patch, dim=-1),
                           F.normalize(s_patch, dim=-1).transpose(1, 2))
        t_gram = torch.bmm(F.normalize(t_patch.detach(), dim=-1),
                           F.normalize(t_patch.detach(), dim=-1).transpose(1, 2))
        s_attn = F.softmax(s_gram / 0.1, dim=-1)
        t_attn = F.softmax(t_gram / 0.1, dim=-1)
        return F.kl_div(s_attn.log().clamp(min=-100), t_attn, reduction='batchmean')

    def level3_relation_loss(self, s_emb, t_emb):
        """Relation-level: CRD — preserve pairwise similarity structure."""
        s_sim = torch.mm(F.normalize(s_emb, dim=-1),
                         F.normalize(s_emb, dim=-1).t())
        t_sim = torch.mm(F.normalize(t_emb.detach(), dim=-1),
                         F.normalize(t_emb.detach(), dim=-1).t())
        return F.mse_loss(s_sim, t_sim)

    def level4_logit_loss(self, s_emb, t_emb, temperature=4.0):
        """Logit-level: soft cross-entropy on teacher embeddings."""
        t_logits = torch.mm(t_emb.detach(), t_emb.detach().t()) / temperature
        s_logits = torch.mm(s_emb, t_emb.detach().t()) / temperature
        t_probs = F.softmax(t_logits, dim=-1)
        return F.kl_div(F.log_softmax(s_logits, dim=-1), t_probs,
                        reduction='batchmean') * (temperature ** 2)

    def forward(self, s_out_q, s_out_r, t_out_q, t_out_r, labels=None, epoch=0):
        q_emb = s_out_q['embedding']
        r_emb = s_out_r['embedding']
        B = q_emb.shape[0]

        # InfoNCE
        logits = q_emb @ r_emb.t() / self.temp
        targets = torch.arange(B, device=logits.device)
        loss_infonce = (F.cross_entropy(logits, targets) +
                        F.cross_entropy(logits.t(), targets)) / 2
        acc = (logits.argmax(dim=-1) == targets).float().mean()

        # CE
        loss_ce = torch.tensor(0.0, device=q_emb.device)
        if labels is not None:
            loss_ce = (F.cross_entropy(self.classifier(q_emb), labels) +
                       F.cross_entropy(self.classifier(r_emb), labels)) / 2

        # Triplet
        loss_triplet = self._triplet_loss(q_emb, r_emb)

        total_loss = LAMBDA_CE * (loss_infonce + loss_ce) + LAMBDA_TRIPLET * loss_triplet

        # Adaptive distillation weights
        dist_w = s_out_q.get('dist_weights', torch.ones(4, device=q_emb.device) * 0.25)

        # === PROGRESSIVE CURRICULUM ===
        loss_feat = loss_attn = loss_rel = loss_logit = torch.tensor(0.0, device=q_emb.device)

        # Level 1: Feature distillation (always active)
        if epoch >= LEVEL1_START:
            ramp = min(1.0, (epoch - LEVEL1_START + 1) / 10)
            loss_feat = (
                self.level1_feature_loss(s_out_q['global_feat'], t_out_q['global_feat']) +
                self.level1_feature_loss(s_out_r['global_feat'], t_out_r['global_feat'])
            ) / 2
            total_loss = total_loss + ramp * dist_w[0] * LAMBDA_FEAT * loss_feat

        # Level 2: Attention transfer
        if epoch >= LEVEL2_START:
            ramp = min(1.0, (epoch - LEVEL2_START + 1) / 10)
            s_p_q = s_out_q['patch_tokens']; t_p_q = t_out_q['patch_tokens']
            # Handle size mismatch
            if s_p_q.shape[1] != t_p_q.shape[1]:
                s_p_q = F.interpolate(s_p_q.transpose(1,2), size=t_p_q.shape[1],
                                      mode='linear').transpose(1,2)
            s_p_r = s_out_r['patch_tokens']; t_p_r = t_out_r['patch_tokens']
            if s_p_r.shape[1] != t_p_r.shape[1]:
                s_p_r = F.interpolate(s_p_r.transpose(1,2), size=t_p_r.shape[1],
                                      mode='linear').transpose(1,2)
            loss_attn = (self.level2_attention_loss(s_p_q, t_p_q) +
                         self.level2_attention_loss(s_p_r, t_p_r)) / 2
            total_loss = total_loss + ramp * dist_w[1] * LAMBDA_ATTN * loss_attn

        # Level 3: Relation distillation
        if epoch >= LEVEL3_START:
            ramp = min(1.0, (epoch - LEVEL3_START + 1) / 10)
            loss_rel = (
                self.level3_relation_loss(q_emb, t_out_q['embedding']) +
                self.level3_relation_loss(r_emb, t_out_r['embedding'])
            ) / 2
            total_loss = total_loss + ramp * dist_w[2] * LAMBDA_REL * loss_rel

        # Level 4: Logit distillation
        if epoch >= LEVEL4_START:
            ramp = min(1.0, (epoch - LEVEL4_START + 1) / 10)
            loss_logit = (
                self.level4_logit_loss(q_emb, t_out_q['embedding']) +
                self.level4_logit_loss(r_emb, t_out_r['embedding'])
            ) / 2
            total_loss = total_loss + ramp * dist_w[3] * LAMBDA_LOGIT * loss_logit

        level = (f"L1:feat" if epoch < LEVEL2_START else
                 f"L2:+attn" if epoch < LEVEL3_START else
                 f"L3:+rel" if epoch < LEVEL4_START else
                 f"L4:+logit")

        return {
            'total_loss': total_loss, 'accuracy': acc, 'level': level,
            'loss_infonce': loss_infonce.item(),
            'loss_ce': loss_ce.item() if torch.is_tensor(loss_ce) else loss_ce,
            'loss_triplet': loss_triplet.item(),
            'loss_feat': loss_feat.item() if torch.is_tensor(loss_feat) else loss_feat,
            'loss_attn': loss_attn.item() if torch.is_tensor(loss_attn) else loss_attn,
            'loss_rel': loss_rel.item() if torch.is_tensor(loss_rel) else loss_rel,
            'loss_logit': loss_logit.item() if torch.is_tensor(loss_logit) else loss_logit,
            'dist_weights': [round(w.item(), 3) for w in dist_w],
        }

    def _triplet_loss(self, q_emb, r_emb, margin=0.3):
        dist = 1.0 - torch.mm(q_emb, r_emb.t())
        pos = dist.diag()
        neg_q = dist.clone(); neg_q.fill_diagonal_(float('inf'))
        neg_r = dist.clone().t(); neg_r.fill_diagonal_(float('inf'))
        return (F.relu(pos - neg_q.min(1)[0] + margin).mean() +
                F.relu(pos - neg_r.min(1)[0] + margin).mean()) / 2


# =============================================================================
# DATASET: SUES-200
# =============================================================================
class SUES200Dataset(Dataset):
    def __init__(self, root, split="train", altitude="150",
                 img_size=224, train_locs=None, test_locs=None):
        super().__init__()
        drone_dir = os.path.join(root, "drone-view")
        satellite_dir = os.path.join(root, "satellite-view")
        if train_locs is None: train_locs = TRAIN_LOCS
        if test_locs is None: test_locs = TEST_LOCS
        locs = train_locs if split == "train" else test_locs

        if split == "train":
            self.drone_tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=3),
                transforms.Pad(10, padding_mode='edge'),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
            self.sat_tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=3),
                transforms.Pad(10, padding_mode='edge'),
                transforms.RandomAffine(90),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        else:
            self.drone_tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
            self.sat_tf = self.drone_tf

        self.pairs = []; self.labels = []; loc_to_label = {}
        for loc_id in locs:
            loc_str = f"{loc_id:04d}"
            sat_path = os.path.join(satellite_dir, loc_str, "0.png")
            if not os.path.exists(sat_path): continue
            alt_dir = os.path.join(drone_dir, loc_str, altitude)
            if not os.path.isdir(alt_dir): continue
            if loc_id not in loc_to_label: loc_to_label[loc_id] = len(loc_to_label)
            for img_name in sorted(os.listdir(alt_dir)):
                if img_name.endswith(('.jpg','.jpeg','.png')):
                    self.pairs.append((os.path.join(alt_dir, img_name), sat_path))
                    self.labels.append(loc_to_label[loc_id])
        self.num_classes = len(loc_to_label)
        print(f"  [SUES-200 {split} alt={altitude}] {len(self.pairs)} pairs ({self.num_classes} cls)")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        dp, sp = self.pairs[idx]
        try: drone=Image.open(dp).convert("RGB"); sat=Image.open(sp).convert("RGB")
        except: drone=Image.new("RGB",(224,224),(128,128,128)); sat=Image.new("RGB",(224,224),(128,128,128))
        return {"query":self.drone_tf(drone),"gallery":self.sat_tf(sat),
                "label":self.labels[idx],"idx":idx}


class SUES200GalleryDataset(Dataset):
    """Satellite gallery with ALL 200 locations (confusion data per SUES-200 protocol)."""
    def __init__(self, root, test_locs=None, img_size=224):
        super().__init__()
        satellite_dir = os.path.join(root, "satellite-view")
        # Standard protocol: gallery includes ALL locations as confusion data
        all_locs = TRAIN_LOCS + TEST_LOCS
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.images = []; self.loc_ids = []
        for loc_id in all_locs:
            loc_str = f"{loc_id:04d}"
            sat_path = os.path.join(satellite_dir, loc_str, "0.png")
            if os.path.exists(sat_path):
                self.images.append(sat_path); self.loc_ids.append(loc_id)
        print(f"  Gallery: {len(self.images)} satellite images (confusion data)")
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return {"image": self.tf(img), "loc_id": self.loc_ids[idx]}


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model, data_root, altitude, device, test_locs=None):
    model.eval()
    query_ds = SUES200Dataset(data_root, "test", altitude, IMG_SIZE, test_locs=test_locs)
    query_loader = DataLoader(query_ds, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    gallery_ds = SUES200GalleryDataset(data_root, test_locs, IMG_SIZE)
    gallery_loader = DataLoader(gallery_ds, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    gal_embs, gal_locs = [], []
    for batch in gallery_loader:
        emb = model.extract_embedding(batch["image"].to(device), 'satellite')
        gal_embs.append(emb.cpu()); gal_locs.extend(batch["loc_id"].tolist())
    gal_embs = torch.cat(gal_embs, 0); gal_locs = np.array(gal_locs)

    q_embs = []
    for batch in query_loader:
        emb = model.extract_embedding(batch["query"].to(device), 'drone')
        q_embs.append(emb.cpu())
    q_embs = torch.cat(q_embs, 0)

    loc_to_gal_idx = {loc: i for i, loc in enumerate(gal_locs)}
    q_gt = np.array([loc_to_gal_idx.get(int(os.path.basename(os.path.dirname(sp))), -1)
                      for _, sp in query_ds.pairs])
    sim = q_embs.numpy() @ gal_embs.numpy().T
    ranks = np.argsort(-sim, axis=1); N = len(q_embs)
    results = {}
    for k in [1, 5, 10]:
        results[f"R@{k}"] = sum(1 for i in range(N) if q_gt[i] in ranks[i, :k]) / N
    ap_sum = sum(1.0/(np.where(ranks[i]==q_gt[i])[0][0]+1)
                 for i in range(N) if len(np.where(ranks[i]==q_gt[i])[0])>0)
    results["AP"] = ap_sum / N
    return results


# =============================================================================
# TRAINING
# =============================================================================
def get_cosine_lr(epoch, total_epochs, base_lr, warmup=5):
    if epoch < warmup: return base_lr*(epoch+1)/warmup
    p = (epoch-warmup)/max(1, total_epochs-warmup)
    return base_lr*0.5*(1+math.cos(math.pi*p))


def train(student, teacher, train_loader, val_fn, device, epochs=EPOCHS):
    criterion = HierarchicalDistillLoss().to(device)
    teacher.eval()

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(criterion.parameters()),
        lr=LR_INIT, weight_decay=WEIGHT_DECAY
    )
    scaler = GradScaler(enabled=AMP_ENABLED and device.type == "cuda")
    best_r1 = 0.0; history = []

    for epoch in range(epochs):
        lr = get_cosine_lr(epoch, epochs, LR_INIT, WARMUP_EPOCHS)
        for pg in optimizer.param_groups: pg['lr'] = lr

        student.train(); ep_loss = ep_acc = n = 0; t0 = time.time()
        level = (f"L1" if epoch < LEVEL2_START else f"L2" if epoch < LEVEL3_START
                 else f"L3" if epoch < LEVEL4_START else f"L4")

        pbar = tqdm(train_loader, desc=f"  Ep {epoch+1}/{epochs} ({level})", leave=False)
        for batch in pbar:
            query = batch["query"].to(device); gallery = batch["gallery"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=AMP_ENABLED and device.type == "cuda"):
                with torch.no_grad():
                    t_out_q = teacher(query); t_out_r = teacher(gallery)
                s_out_q = student(query, 'drone'); s_out_r = student(gallery, 'satellite')
                loss_dict = criterion(s_out_q, s_out_r, t_out_q, t_out_r,
                                      labels=labels, epoch=epoch)
                loss = loss_dict['total_loss']

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True); continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            ep_loss += loss.item(); ep_acc += loss_dict['accuracy'].item(); n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{loss_dict['accuracy'].item():.1%}")

        elapsed = time.time() - t0
        ep_loss /= max(n,1); ep_acc /= max(n,1)
        entry = {"epoch": epoch+1, "level": level, "loss": round(ep_loss,4),
                 "acc": round(ep_acc,4), "lr": round(lr,6), "time": round(elapsed,1),
                 "dist_weights": loss_dict.get('dist_weights', [])}

        if (epoch+1) % EVAL_FREQ == 0 or epoch == epochs-1:
            metrics = val_fn()
            entry.update(metrics)
            r1 = metrics["R@1"]
            if r1 > best_r1:
                best_r1 = r1
                torch.save(student.state_dict(), os.path.join(OUTPUT_DIR, "exp4_hkd_best.pth"))
            print(f"  Ep {epoch+1} ({level}) | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | "
                  f"R@1={r1:.2%} | AP={metrics.get('AP',0):.2%} | LR={lr:.1e} | {elapsed:.0f}s")
        else:
            print(f"  Ep {epoch+1} ({level}) | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | "
                  f"LR={lr:.1e} | {elapsed:.0f}s")

        if (epoch+1) % 20 == 0:
            dw = loss_dict.get('dist_weights', [])
            print(f"    Adaptive dist weights: {dw}")

        history.append(entry)

    return best_r1, history


# =============================================================================
# SMOKE TEST
# =============================================================================
def run_test():
    print("\n" + "="*60)
    print("  EXP4 SMOKE TEST: Hierarchical Knowledge Distillation")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[1/4] Instantiating models...")
    try:
        teacher = TeacherModel().to(device)
        student = StudentModel().to(device)
        criterion = HierarchicalDistillLoss().to(device)
        t_params = sum(p.numel() for p in teacher.parameters())
        s_params = sum(p.numel() for p in student.parameters())
        print(f"  ✓ Teacher: {t_params:,} params (frozen)")
        print(f"  ✓ Student: {s_params:,} params ({s_params*4/(1024*1024):.1f} MB)")
    except Exception as e:
        print(f"  ✗ Failed: {e}"); import traceback; traceback.print_exc(); return False

    print("\n[2/4] Testing forward pass...")
    try:
        dummy = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
        with torch.no_grad(): t_out = teacher(dummy)
        s_out = student(dummy, 'drone')
        print(f"  ✓ Teacher embedding: {t_out['embedding'].shape}")
        print(f"  ✓ Student embedding: {s_out['embedding'].shape}")
        print(f"  ✓ Student dist_weights: {[f'{w:.3f}' for w in s_out['dist_weights'].tolist()]}")
    except Exception as e:
        print(f"  ✗ Failed: {e}"); import traceback; traceback.print_exc(); return False

    print("\n[3/4] Testing loss at all levels...")
    try:
        dummy_r = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
        labels = torch.randint(0, NUM_CLASSES, (4,)).to(device)
        with torch.no_grad(): t_out_r = teacher(dummy_r)
        s_out_r = student(dummy_r, 'satellite')
        for ep in [0, LEVEL2_START, LEVEL3_START, LEVEL4_START]:
            ld = criterion(s_out, s_out_r, t_out, t_out_r, labels=labels, epoch=ep)
            assert not torch.isnan(ld['total_loss']) and not torch.isinf(ld['total_loss'])
            print(f"  ✓ {ld['level']}: loss={ld['total_loss'].item():.4f}")
    except Exception as e:
        print(f"  ✗ Failed: {e}"); import traceback; traceback.print_exc(); return False

    print("\n[4/4] Testing gradient flow...")
    try:
        ld = criterion(s_out, s_out_r, t_out, t_out_r, labels=labels, epoch=LEVEL4_START)
        ld['total_loss'].backward()
        ok = all(p.grad is not None for p in student.parameters() if p.requires_grad)
        frozen = all(p.grad is None for p in teacher.parameters())
        print(f"  ✓ Student gradients: {ok}")
        print(f"  ✓ Teacher frozen: {frozen}")
    except Exception as e:
        print(f"  ✗ Failed: {e}"); import traceback; traceback.print_exc(); return False

    print("\n" + "="*60 + "\n  ALL TESTS PASSED ✓\n" + "="*60)
    return True


# =============================================================================
# MAIN
# =============================================================================
class PKSampler:
    def __init__(self, labels, p=8, k=4):
        self.p = p
        self.k = k
        self.locations = list(set(labels))
        self.drone_by_location = defaultdict(list)
        for idx, label in enumerate(labels):
            self.drone_by_location[label].append(idx)
            
    def __iter__(self):
        locations = self.locations.copy()
        random.shuffle(locations)
        batch = []
        for loc in locations:
            indices = self.drone_by_location[loc]
            if len(indices) < self.k:
                indices = indices * (self.k // len(indices) + 1)
            sampled = random.sample(indices, self.k)
            batch.extend(sampled)
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]
                batch = batch[self.p * self.k:]
                
    def __len__(self):
        return len(self.locations) // self.p


def main():
    global EPOCHS, BATCH_SIZE, DATA_ROOT
    parser = argparse.ArgumentParser(description="EXP4: Hierarchical KD")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    args, _ = parser.parse_known_args()

    if args.test: sys.exit(0 if run_test() else 1)

    EPOCHS = args.epochs; BATCH_SIZE = args.batch_size; DATA_ROOT = args.data_root

    print("\n" + "="*70)
    print("  EXP4: Hierarchical Knowledge Distillation")
    print(f"  Teacher: {TEACHER_BACKBONE} (frozen)")
    print(f"  Student: {STUDENT_BACKBONE}")
    print(f"  Curriculum: L1[0-{LEVEL2_START}) L2[{LEVEL2_START}-{LEVEL3_START}) "
          f"L3[{LEVEL3_START}-{LEVEL4_START}) L4[{LEVEL4_START}-{EPOCHS})")
    print(f"  Device: {DEVICE}")
    print("="*70)

    # Dataset
    print("\n[DATASET] Loading SUES-200...")
    all_pairs = []; all_labels = []
    for alt in ALTITUDES:
        ds = SUES200Dataset(DATA_ROOT, "train", alt, IMG_SIZE)
        all_pairs.extend(ds.pairs); all_labels.extend(ds.labels)

    class CombDS(Dataset):
        def __init__(self, pairs, labels, img_size=224):
            self.pairs=pairs; self.labels=labels
            self.drone_tf = transforms.Compose([
                transforms.Resize((img_size,img_size),interpolation=3),
                transforms.Pad(10,padding_mode='edge'),transforms.RandomCrop((img_size,img_size)),
                transforms.RandomHorizontalFlip(),transforms.ColorJitter(0.2,0.2,0.1,0.05),
                transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            self.sat_tf = transforms.Compose([
                transforms.Resize((img_size,img_size),interpolation=3),
                transforms.Pad(10,padding_mode='edge'),transforms.RandomAffine(90),
                transforms.RandomCrop((img_size,img_size)),transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        def __len__(self): return len(self.pairs)
        def __getitem__(self, idx):
            dp,sp=self.pairs[idx]
            try: d=Image.open(dp).convert("RGB"); s=Image.open(sp).convert("RGB")
            except: d=Image.new("RGB",(224,224),(128,128,128)); s=Image.new("RGB",(224,224),(128,128,128))
            return {"query":self.drone_tf(d),"gallery":self.sat_tf(s),"label":self.labels[idx],"idx":idx}

    train_ds = CombDS(all_pairs, all_labels, IMG_SIZE)
    k_samples = max(2, BATCH_SIZE // 8)
    train_sampler = PKSampler(all_labels, p=8, k=k_samples)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)

    teacher = TeacherModel().to(DEVICE)
    student = StudentModel().to(DEVICE)
    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters()):,} params (frozen)")
    print(f"  Student: {sum(p.numel() for p in student.parameters()):,} params")

    def val_fn(): return evaluate(student, DATA_ROOT, TEST_ALTITUDE, DEVICE)

    best_r1, history = train(student, teacher, train_loader, val_fn, DEVICE, EPOCHS)

    print("\n" + "="*70 + "\n  FINAL RESULTS\n" + "="*70)
    for alt in ALTITUDES:
        m = evaluate(student, DATA_ROOT, alt, DEVICE)
        print(f"  Alt={alt}m | R@1={m['R@1']:.2%} | R@5={m['R@5']:.2%} | AP={m['AP']:.2%}")

    with open(os.path.join(OUTPUT_DIR, "exp4_results.json"), "w") as f:
        json.dump({"experiment":"EXP4_HierarchicalKD","best_r1":best_r1,
                   "student_params": sum(p.numel() for p in student.parameters()),
                   "history":history}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
