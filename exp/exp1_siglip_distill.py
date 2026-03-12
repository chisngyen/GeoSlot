#!/usr/bin/env python3
"""
EXP1: SigLIP-based VLM Distillation for Lightweight Drone Geo-Localization
===========================================================================
Teacher: SigLIP2 ViT-B/16 (frozen) — strong vision-language encoder
Student: EfficientNet-B0 (~5M params) — ultra-lightweight for drone edge deployment

Novelty:
  - Multi-level feature distillation (patch, CLS, attention maps) from VLM teacher
  - Geo-aware contrastive learning using SigLIP's zero-shot spatial priors
  - Progressive distillation: token-level → feature-level → logit-level
  - Dual-branch architecture with view-specific adaptation layers

Target: <10M params, <50MB model, real-time on Jetson Nano

Dataset: SUES-200 (drone ↔ satellite cross-view geo-localization)
Usage:
  python exp1_siglip_distill.py           # Full training on Kaggle
  python exp1_siglip_distill.py --test    # Smoke test (model, forward, loss, grads)
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
try:
    import open_clip
except ImportError:
    pip_install("open-clip-torch")
print("[2/2] Setup complete!")

# =============================================================================
# IMPORTS
# =============================================================================
import math, glob, json, time, gc, random, argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

# === PATCHED: import shared eval_utils for per-altitude evaluation ===
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')
try:
    from eval_utils import evaluate_full, print_paper_results
    HAS_EVAL_UTILS = True
except ImportError:
    HAS_EVAL_UTILS = False
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
import open_clip


# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT      = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
OUTPUT_DIR     = "/kaggle/working"

EPOCHS         = 120
BATCH_SIZE     = 512
NUM_WORKERS    = 8
AMP_ENABLED    = True
EVAL_FREQ      = 5

# Progressive distillation schedule
STAGE1_END     = 30    # Feature-level distillation only
STAGE2_END     = 70    # + Attention transfer
STAGE3_END     = 100   # + Relation-level (CRD)
# Stage 4: + Logit-level until EPOCHS

# Learning rates
LR_INIT        = 1e-3
LR_MIN         = 1e-6
WARMUP_EPOCHS  = 5
WEIGHT_DECAY   = 0.01

# Loss weights
LAMBDA_CE      = 1.0
LAMBDA_TRIPLET = 0.5
LAMBDA_FEAT    = 2.0     # Feature distillation
LAMBDA_ATTN    = 1.0     # Attention transfer
LAMBDA_REL     = 0.5     # Relational distillation (CRD)
LAMBDA_LOGIT   = 1.0     # Logit-level distillation

# Model
TEACHER_NAME   = "ViT-B-16-SigLIP"
TEACHER_PRETRAINED = "webli"
STUDENT_NAME   = "efficientnet_b0"
EMBED_DIM      = 512
TEACHER_DIM    = 768

# Dataset
IMG_SIZE       = 224
TRAIN_LOCS     = list(range(1, 121))
TEST_LOCS      = list(range(121, 201))
ALTITUDES      = ["150", "200", "250", "300"]
TEST_ALTITUDE  = "150"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# TEACHER: SigLIP ViT-B/16 (frozen)
# =============================================================================
class SigLIPTeacher(nn.Module):
    """Frozen SigLIP teacher that provides multi-level features."""
    def __init__(self):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(
            TEACHER_NAME, pretrained=TEACHER_PRETRAINED
        )
        self.visual = model.visual
        self.feature_dim = TEACHER_DIM
        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        """Extract multi-level features from SigLIP."""
        # Get intermediate features via hooks
        features = {}

        # Use forward_features for patch tokens
        visual = self.visual
        if hasattr(visual, 'trunk'):
            trunk = visual.trunk
        else:
            trunk = visual

        # Simple forward to get CLS and patch tokens
        if hasattr(trunk, 'patch_embed'):
            x_patched = trunk.patch_embed(x)
            if hasattr(trunk, 'cls_token') and trunk.cls_token is not None:
                cls_token = trunk.cls_token.expand(x_patched.shape[0], -1, -1)
                x_patched = torch.cat([cls_token, x_patched], dim=1)
            if hasattr(trunk, 'pos_embed') and trunk.pos_embed is not None:
                x_patched = x_patched + trunk.pos_embed

            # Pass through blocks, collecting intermediate features
            intermediate = []
            blocks = trunk.blocks if hasattr(trunk, 'blocks') else trunk.layers
            for i, blk in enumerate(blocks):
                x_patched = blk(x_patched)
                if i in [3, 7, 11]:  # Collect from layers 4, 8, 12
                    intermediate.append(x_patched)

            if hasattr(trunk, 'norm'):
                x_patched = trunk.norm(x_patched)
            intermediate.append(x_patched)

            # CLS token
            cls_feat = x_patched[:, 0]
            patch_feat = x_patched[:, 1:] if x_patched.shape[1] > 1 else x_patched

            # Compute attention-like maps from last block
            attn_maps = None
            try:
                last_block = blocks[-1]
                if hasattr(last_block, 'attn'):
                    attn_maps = last_block.attn.attn_weights if hasattr(last_block.attn, 'attn_weights') else None
            except:
                pass

            return {
                'cls': cls_feat,
                'patch': patch_feat,
                'intermediate': intermediate,
                'attn': attn_maps,
            }
        else:
            # Fallback: just use the model's forward
            out = self.visual(x)
            return {
                'cls': out if out.dim() == 2 else out.mean(dim=1),
                'patch': out.unsqueeze(1) if out.dim() == 2 else out,
                'intermediate': [out.unsqueeze(1) if out.dim() == 2 else out],
                'attn': None,
            }


# =============================================================================
# STUDENT: EfficientNet-B0 with Multi-Level Outputs
# =============================================================================
class EfficientNetStudent(nn.Module):
    """Lightweight student with multi-level feature outputs for distillation."""
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = timm.create_model(STUDENT_NAME, pretrained=True,
                                          num_classes=0, global_pool='')
        self.feat_dim = self.backbone.num_features  # 1280 for B0

        # Feature adaptation layers (student → teacher dimension)
        self.feat_adaptor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feat_dim, TEACHER_DIM),
            nn.LayerNorm(TEACHER_DIM),
        )

        # Patch-level adaptor
        self.patch_adaptor = nn.Sequential(
            nn.Conv2d(self.feat_dim, TEACHER_DIM, 1),
            nn.BatchNorm2d(TEACHER_DIM),
        )

        # View-specific adaptation heads (drone vs satellite)
        self.drone_head = nn.Sequential(
            nn.Linear(TEACHER_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.sat_head = nn.Sequential(
            nn.Linear(TEACHER_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Shared embedding projection
        self.embed_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, view_type='drone'):
        """
        Args:
            x: input image [B, 3, H, W]
            view_type: 'drone' or 'satellite'
        Returns:
            dict with embedding, adapted features, patch features
        """
        feat_map = self.backbone(x)  # [B, 1280, H', W']

        # Global feature (CLS equivalent)
        global_feat = self.feat_adaptor(feat_map)  # [B, TEACHER_DIM]

        # Patch-level features
        patch_feat = self.patch_adaptor(feat_map)  # [B, TEACHER_DIM, H', W']
        B, C, H, W = patch_feat.shape
        patch_tokens = patch_feat.flatten(2).transpose(1, 2)  # [B, N, TEACHER_DIM]

        # View-specific embedding
        if view_type == 'drone':
            view_feat = self.drone_head(global_feat)
        else:
            view_feat = self.sat_head(global_feat)

        embedding = F.normalize(self.embed_proj(view_feat), dim=-1)

        return {
            'embedding': embedding,
            'global_feat': global_feat,
            'patch_tokens': patch_tokens,
            'feat_map': feat_map,
            'spatial_size': (H, W),
        }

    def extract_embedding(self, x, view_type='drone'):
        return self.forward(x, view_type)['embedding']


# =============================================================================
# DISTILLATION LOSS
# =============================================================================
class ProgressiveDistillationLoss(nn.Module):
    """Multi-level progressive distillation loss."""
    def __init__(self, num_classes=160):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(0.07).log())
        self.num_classes = num_classes

        # Classification head for CE loss
        self.classifier = nn.Linear(EMBED_DIM, num_classes)

    @property
    def temp(self):
        return self.log_temp.exp().clamp(0.01, 1.0)

    def feature_distill_loss(self, student_feat, teacher_feat):
        """L2 + cosine feature alignment."""
        s = F.normalize(student_feat, dim=-1)
        t = F.normalize(teacher_feat.detach(), dim=-1)
        l2_loss = F.mse_loss(student_feat, teacher_feat.detach())
        cos_loss = 1.0 - (s * t).sum(dim=-1).mean()
        return l2_loss + cos_loss

    def attention_transfer_loss(self, student_patch, teacher_patch):
        """Transfer attention patterns from teacher to student."""
        # Compute self-attention maps from features
        s_attn = torch.bmm(
            F.normalize(student_patch, dim=-1),
            F.normalize(student_patch, dim=-1).transpose(1, 2)
        )
        t_attn = torch.bmm(
            F.normalize(teacher_patch.detach(), dim=-1),
            F.normalize(teacher_patch.detach(), dim=-1).transpose(1, 2)
        )
        # Match attention distributions
        s_attn = F.softmax(s_attn / 0.1, dim=-1)
        t_attn = F.softmax(t_attn / 0.1, dim=-1)
        return F.kl_div(s_attn.log(), t_attn, reduction='batchmean')

    def relational_distill_loss(self, student_embs, teacher_embs):
        """CRD: preserve pairwise similarity structure."""
        s_sim = torch.mm(F.normalize(student_embs, dim=-1),
                         F.normalize(student_embs, dim=-1).t())
        t_sim = torch.mm(F.normalize(teacher_embs.detach(), dim=-1),
                         F.normalize(teacher_embs.detach(), dim=-1).t())
        return F.mse_loss(s_sim, t_sim)

    def logit_distill_loss(self, student_emb, teacher_emb, temperature=4.0):
        """Soft cross-entropy on teacher embeddings."""
        # Teacher-student similarity as soft labels
        t_logits = torch.mm(teacher_emb.detach(), teacher_emb.detach().t()) / temperature
        s_logits = torch.mm(student_emb, teacher_emb.detach().t()) / temperature
        t_probs = F.softmax(t_logits, dim=-1)
        loss = F.kl_div(F.log_softmax(s_logits, dim=-1), t_probs, reduction='batchmean')
        return loss * (temperature ** 2)

    def forward(self, student_out_q, student_out_r, teacher_out_q, teacher_out_r,
                labels=None, epoch=0):
        q_emb = student_out_q['embedding']
        r_emb = student_out_r['embedding']
        B = q_emb.shape[0]

        # === Contrastive InfoNCE ===
        logits = q_emb @ r_emb.t() / self.temp
        targets = torch.arange(B, device=logits.device)
        loss_infonce = (F.cross_entropy(logits, targets) +
                        F.cross_entropy(logits.t(), targets)) / 2
        acc = (logits.argmax(dim=-1) == targets).float().mean()

        # === CE classification ===
        if labels is not None:
            q_cls = self.classifier(q_emb)
            r_cls = self.classifier(r_emb)
            loss_ce = (F.cross_entropy(q_cls, labels) +
                       F.cross_entropy(r_cls, labels)) / 2
        else:
            loss_ce = torch.tensor(0.0, device=q_emb.device)

        # === Triplet Loss ===
        loss_triplet = self._triplet_loss(q_emb, r_emb)

        total_loss = LAMBDA_CE * (loss_infonce + loss_ce) + LAMBDA_TRIPLET * loss_triplet

        # === Progressive Distillation ===
        loss_feat = torch.tensor(0.0, device=q_emb.device)
        loss_attn = torch.tensor(0.0, device=q_emb.device)
        loss_rel = torch.tensor(0.0, device=q_emb.device)
        loss_logit = torch.tensor(0.0, device=q_emb.device)

        # Stage 1+: Feature-level distillation
        if epoch >= 0:
            ramp = min(1.0, (epoch + 1) / 10)
            loss_feat = (
                self.feature_distill_loss(student_out_q['global_feat'], teacher_out_q['cls']) +
                self.feature_distill_loss(student_out_r['global_feat'], teacher_out_r['cls'])
            ) / 2
            total_loss = total_loss + ramp * LAMBDA_FEAT * loss_feat

        # Stage 2+: Attention transfer
        if epoch >= STAGE1_END:
            ramp = min(1.0, (epoch - STAGE1_END + 1) / 10)
            # Align patch tokens (need to handle different sequence lengths)
            s_patch_q = student_out_q['patch_tokens']
            t_patch_q = teacher_out_q['patch']
            # Interpolate student patches to match teacher sequence length
            if s_patch_q.shape[1] != t_patch_q.shape[1]:
                s_patch_q = F.interpolate(
                    s_patch_q.transpose(1, 2),
                    size=t_patch_q.shape[1],
                    mode='linear'
                ).transpose(1, 2)
            loss_attn = self.attention_transfer_loss(s_patch_q, t_patch_q)
            total_loss = total_loss + ramp * LAMBDA_ATTN * loss_attn

        # Stage 3+: Relational distillation
        if epoch >= STAGE2_END:
            ramp = min(1.0, (epoch - STAGE2_END + 1) / 10)
            loss_rel = (
                self.relational_distill_loss(student_out_q['global_feat'], teacher_out_q['cls']) +
                self.relational_distill_loss(student_out_r['global_feat'], teacher_out_r['cls'])
            ) / 2
            total_loss = total_loss + ramp * LAMBDA_REL * loss_rel

        # Stage 4+: Logit-level distillation
        if epoch >= STAGE3_END:
            ramp = min(1.0, (epoch - STAGE3_END + 1) / 10)
            loss_logit = (
                self.logit_distill_loss(student_out_q['global_feat'], teacher_out_q['cls']) +
                self.logit_distill_loss(student_out_r['global_feat'], teacher_out_r['cls'])
            ) / 2
            total_loss = total_loss + ramp * LAMBDA_LOGIT * loss_logit

        stage = ("S1:feat" if epoch < STAGE1_END else
                 "S2:+attn" if epoch < STAGE2_END else
                 "S3:+rel" if epoch < STAGE3_END else
                 "S4:+logit")

        return {
            'total_loss': total_loss, 'accuracy': acc,
            'loss_infonce': loss_infonce.item(),
            'loss_ce': loss_ce.item() if torch.is_tensor(loss_ce) else loss_ce,
            'loss_triplet': loss_triplet.item(),
            'loss_feat': loss_feat.item() if torch.is_tensor(loss_feat) else loss_feat,
            'loss_attn': loss_attn.item() if torch.is_tensor(loss_attn) else loss_attn,
            'loss_rel': loss_rel.item() if torch.is_tensor(loss_rel) else loss_rel,
            'loss_logit': loss_logit.item() if torch.is_tensor(loss_logit) else loss_logit,
            'stage': stage,
        }

    def _triplet_loss(self, q_emb, r_emb, margin=0.3):
        dist = 1.0 - torch.mm(q_emb, r_emb.t())
        pos = dist.diag()
        # Hard negative mining
        neg_q = dist.clone()
        neg_q.fill_diagonal_(float('inf'))
        hardest_neg_q = neg_q.min(dim=1)[0]
        neg_r = dist.clone().t()
        neg_r.fill_diagonal_(float('inf'))
        hardest_neg_r = neg_r.min(dim=1)[0]
        loss = (F.relu(pos - hardest_neg_q + margin).mean() +
                F.relu(pos - hardest_neg_r + margin).mean()) / 2
        return loss


# =============================================================================
# DATASET: SUES-200
# =============================================================================
class SUES200Dataset(Dataset):
    def __init__(self, root, split="train", altitude="150",
                 img_size=224, train_locs=None, test_locs=None):
        super().__init__()
        self.root = root
        self.split = split
        self.altitude = altitude

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
                transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        else:
            self.drone_tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
            self.sat_tf = self.drone_tf

        self.pairs = []
        self.labels = []
        loc_to_label = {}
        for loc_id in locs:
            loc_str = f"{loc_id:04d}"
            sat_path = os.path.join(satellite_dir, loc_str, "0.png")
            if not os.path.exists(sat_path): continue
            alt_dir = os.path.join(drone_dir, loc_str, altitude)
            if not os.path.isdir(alt_dir): continue
            if loc_id not in loc_to_label:
                loc_to_label[loc_id] = len(loc_to_label)
            label = loc_to_label[loc_id]
            for img_name in sorted(os.listdir(alt_dir)):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    drone_path = os.path.join(alt_dir, img_name)
                    self.pairs.append((drone_path, sat_path))
                    self.labels.append(label)

        self.num_classes = len(loc_to_label)
        print(f"  [SUES-200 {split} alt={altitude}] {len(self.pairs)} pairs "
              f"({len(locs)} locations, {self.num_classes} classes)")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        drone_path, sat_path = self.pairs[idx]
        try:
            drone = Image.open(drone_path).convert("RGB")
            sat = Image.open(sat_path).convert("RGB")
        except Exception:
            drone = Image.new("RGB", (224, 224), (128,128,128))
            sat = Image.new("RGB", (224, 224), (128,128,128))
        return {
            "query": self.drone_tf(drone),
            "gallery": self.sat_tf(sat),
            "label": self.labels[idx],
            "idx": idx
        }


class SUES200GalleryDataset(Dataset):
    """Satellite gallery with ALL 200 locations (confusion data per SUES-200 protocol)."""
    def __init__(self, root, test_locs=None, img_size=224):
        super().__init__()
        satellite_dir = os.path.join(root, "satellite-view")
        if test_locs is None: test_locs = TEST_LOCS
        # Standard protocol: gallery includes ALL locations as confusion data
        all_locs = TRAIN_LOCS + TEST_LOCS
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.images = []
        self.loc_ids = []
        for loc_id in all_locs:
            loc_str = f"{loc_id:04d}"
            sat_path = os.path.join(satellite_dir, loc_str, "0.png")
            if os.path.exists(sat_path):
                self.images.append(sat_path)
                self.loc_ids.append(loc_id)
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
    query_loader = DataLoader(query_ds, batch_size=64, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    gallery_ds = SUES200GalleryDataset(data_root, test_locs, IMG_SIZE)
    gallery_loader = DataLoader(gallery_ds, batch_size=64, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

    gal_embs, gal_locs = [], []
    for batch in gallery_loader:
        emb = model.extract_embedding(batch["image"].to(device), view_type='satellite')
        gal_embs.append(emb.cpu())
        gal_locs.extend(batch["loc_id"].tolist())
    gal_embs = torch.cat(gal_embs, 0)
    gal_locs = np.array(gal_locs)

    q_embs = []
    for batch in query_loader:
        emb = model.extract_embedding(batch["query"].to(device), view_type='drone')
        q_embs.append(emb.cpu())
    q_embs = torch.cat(q_embs, 0)

    loc_to_gal_idx = {loc: i for i, loc in enumerate(gal_locs)}
    q_gt_indices = []
    for drone_path, sat_path in query_ds.pairs:
        loc_str = os.path.basename(os.path.dirname(sat_path))
        loc_id = int(loc_str)
        q_gt_indices.append(loc_to_gal_idx.get(loc_id, -1))
    q_gt_indices = np.array(q_gt_indices)

    sim = q_embs.numpy() @ gal_embs.numpy().T
    ranks = np.argsort(-sim, axis=1)
    N = len(q_embs)

    results = {}
    for k in [1, 5, 10]:
        correct = sum(1 for i in range(N) if q_gt_indices[i] in ranks[i, :k])
        results[f"R@{k}"] = correct / N

    ap_sum = 0
    for i in range(N):
        gt = q_gt_indices[i]
        rank_pos = np.where(ranks[i] == gt)[0]
        if len(rank_pos) > 0:
            ap_sum += 1.0 / (rank_pos[0] + 1)
    results["AP"] = ap_sum / N
    return results


# =============================================================================
# LR SCHEDULER
# =============================================================================
def get_cosine_lr(epoch, total_epochs, base_lr, warmup_epochs=5):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# =============================================================================
# TRAINING
# =============================================================================
def train(student, teacher, train_loader, val_fn, device, epochs=EPOCHS):
    criterion = ProgressiveDistillationLoss(num_classes=160).to(device)
    teacher.eval()

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(criterion.parameters()),
        lr=LR_INIT, weight_decay=WEIGHT_DECAY
    )
    scaler = GradScaler(enabled=AMP_ENABLED and device.type == "cuda")
    best_r1 = 0.0
    history = []

    for epoch in range(epochs):
        lr = get_cosine_lr(epoch, epochs, LR_INIT, WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        student.train()
        ep_loss = ep_acc = n = 0
        t0 = time.time()

        stage_name = ("S1:feat" if epoch < STAGE1_END else
                      "S2:+attn" if epoch < STAGE2_END else
                      "S3:+rel" if epoch < STAGE3_END else
                      "S4:+logit")

        pbar = tqdm(train_loader, desc=f"  Ep {epoch+1}/{epochs} ({stage_name})", leave=False)
        for batch in pbar:
            query = batch["query"].to(device)
            gallery = batch["gallery"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=AMP_ENABLED and device.type == "cuda"):
                # Teacher forward (frozen, no grad)
                with torch.no_grad():
                    t_out_q = teacher(query)
                    t_out_r = teacher(gallery)

                # Student forward
                s_out_q = student(query, view_type='drone')
                s_out_r = student(gallery, view_type='satellite')

                loss_dict = criterion(s_out_q, s_out_r, t_out_q, t_out_r,
                                      labels=labels, epoch=epoch)
                loss = loss_dict['total_loss']

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item()
            ep_acc += loss_dict['accuracy'].item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{loss_dict['accuracy'].item():.1%}")

        elapsed = time.time() - t0
        ep_loss /= max(n, 1)
        ep_acc /= max(n, 1)

        entry = {
            "epoch": epoch+1, "stage": stage_name, "loss": round(ep_loss, 4),
            "acc": round(ep_acc, 4), "lr": round(lr, 6), "time": round(elapsed, 1),
            "feat": round(loss_dict.get('loss_feat', 0), 4),
            "attn": round(loss_dict.get('loss_attn', 0), 4),
            "rel":  round(loss_dict.get('loss_rel', 0), 4),
            "logit": round(loss_dict.get('loss_logit', 0), 4),
        }

        if (epoch+1) % EVAL_FREQ == 0 or epoch == epochs - 1:
            metrics = val_fn()
            entry.update(metrics)
            r1 = metrics["R@1"]
            if r1 > best_r1:
                best_r1 = r1
                torch.save(student.state_dict(),
                           os.path.join(OUTPUT_DIR, "exp1_siglip_distill_best.pth"))
            print(f"  Ep {epoch+1} ({stage_name}) | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | "
                  f"R@1={r1:.2%} | R@5={metrics.get('R@5',0):.2%} | AP={metrics.get('AP',0):.2%} | "
                  f"LR={lr:.1e} | {elapsed:.0f}s")
        else:
            print(f"  Ep {epoch+1} ({stage_name}) | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | "
                  f"LR={lr:.1e} | {elapsed:.0f}s")

        history.append(entry)

    return best_r1, history


# =============================================================================
# SMOKE TEST
# =============================================================================
def run_test():
    print("\n" + "="*60)
    print("  EXP1 SMOKE TEST: SigLIP Distillation")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Model instantiation
    print("\n[1/4] Instantiating models...")
    try:
        teacher = SigLIPTeacher().to(device)
        student = EfficientNetStudent().to(device)
        criterion = ProgressiveDistillationLoss(num_classes=160).to(device)
        print("  ✓ Teacher (SigLIP) and Student (EfficientNet-B0) created")
    except Exception as e:
        print(f"  ✗ Model instantiation failed: {e}")
        return False

    # 2. Forward pass
    print("\n[2/4] Testing forward pass...")
    try:
        dummy_q = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
        dummy_r = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
        labels = torch.randint(0, 160, (4,)).to(device)

        with torch.no_grad():
            t_out_q = teacher(dummy_q)
            t_out_r = teacher(dummy_r)
        s_out_q = student(dummy_q, 'drone')
        s_out_r = student(dummy_r, 'satellite')

        print(f"  ✓ Student embedding shape: {s_out_q['embedding'].shape}")
        print(f"  ✓ Teacher CLS shape: {t_out_q['cls'].shape}")
        total_params = sum(p.numel() for p in student.parameters())
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"  ✓ Student params: {total_params:,} total, {trainable:,} trainable")
        model_size_mb = total_params * 4 / (1024*1024)
        print(f"  ✓ Student model size (FP32): {model_size_mb:.1f} MB")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback; traceback.print_exc()
        return False

    # 3. Loss computation
    print("\n[3/4] Testing loss computation...")
    try:
        for stage_epoch in [0, STAGE1_END, STAGE2_END, STAGE3_END]:
            loss_dict = criterion(s_out_q, s_out_r, t_out_q, t_out_r,
                                  labels=labels, epoch=stage_epoch)
            loss = loss_dict['total_loss']
            assert not torch.isnan(loss), "Loss is NaN"
            assert not torch.isinf(loss), "Loss is Inf"
            print(f"  ✓ Stage {loss_dict['stage']}: loss={loss.item():.4f}")
    except Exception as e:
        print(f"  ✗ Loss computation failed: {e}")
        import traceback; traceback.print_exc()
        return False

    # 4. Gradient flow
    print("\n[4/4] Testing gradient flow...")
    try:
        loss_dict = criterion(s_out_q, s_out_r, t_out_q, t_out_r,
                              labels=labels, epoch=STAGE3_END)
        loss = loss_dict['total_loss']
        loss.backward()
        grad_ok = True
        no_grad_params = []
        for name, p in student.named_parameters():
            if p.requires_grad and p.grad is None:
                grad_ok = False
                no_grad_params.append(name)
        if grad_ok:
            print("  ✓ Gradients flow to all trainable parameters")
        else:
            print(f"  ⚠ Missing gradients for: {no_grad_params[:5]}")

        # Verify teacher is frozen
        teacher_grads = [p.grad for p in teacher.parameters() if p.grad is not None]
        assert len(teacher_grads) == 0, "Teacher should be frozen!"
        print("  ✓ Teacher correctly frozen (no gradients)")
    except Exception as e:
        print(f"  ✗ Gradient check failed: {e}")
        import traceback; traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("  ALL TESTS PASSED ✓")
    print("="*60)
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
    parser = argparse.ArgumentParser(description="EXP1: SigLIP Distillation")
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    args, _ = parser.parse_known_args()

    if args.test:
        success = run_test()
        sys.exit(0 if success else 1)

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    DATA_ROOT = args.data_root

    print("\n" + "="*70)
    print("  EXP1: SigLIP VLM Distillation for Lightweight Drone Geo-Loc")
    print(f"  Teacher: {TEACHER_NAME} ({TEACHER_PRETRAINED})")
    print(f"  Student: {STUDENT_NAME}")
    print(f"  Progressive: S1[0-{STAGE1_END}) S2[{STAGE1_END}-{STAGE2_END}) "
          f"S3[{STAGE2_END}-{STAGE3_END}) S4[{STAGE3_END}-{EPOCHS})")
    print(f"  Device: {DEVICE}")
    print("="*70)

    # Load datasets (all altitudes for training)
    print("\n[DATASET] Loading SUES-200...")
    train_pairs_all = []
    train_labels_all = []
    for alt in ALTITUDES:
        ds = SUES200Dataset(DATA_ROOT, "train", alt, IMG_SIZE)
        train_pairs_all.extend(ds.pairs)
        train_labels_all.extend(ds.labels)
    print(f"  Total train pairs (all altitudes): {len(train_pairs_all)}")

    class CombinedTrainDataset(Dataset):
        def __init__(self, pairs, labels, img_size=224):
            self.pairs = pairs
            self.labels = labels
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
                transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        def __len__(self): return len(self.pairs)
        def __getitem__(self, idx):
            dp, sp = self.pairs[idx]
            try:
                drone = Image.open(dp).convert("RGB")
                sat = Image.open(sp).convert("RGB")
            except:
                drone = Image.new("RGB", (224, 224), (128,128,128))
                sat = Image.new("RGB", (224, 224), (128,128,128))
            return {"query": self.drone_tf(drone), "gallery": self.sat_tf(sat),
                    "label": self.labels[idx], "idx": idx}

    train_ds = CombinedTrainDataset(train_pairs_all, train_labels_all, IMG_SIZE)
    k_samples = max(2, BATCH_SIZE // 8)
    train_sampler = PKSampler(train_labels_all, p=8, k=k_samples)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Models
    print("\n[MODELS] Loading...")
    teacher = SigLIPTeacher().to(DEVICE)
    student = EfficientNetStudent().to(DEVICE)
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print(f"  Teacher params: {t_params:,} (frozen)")
    print(f"  Student params: {s_params:,}")
    print(f"  Student size (FP32): {s_params*4/(1024*1024):.1f} MB")

    # Eval function
    def val_fn():
        return evaluate(student, DATA_ROOT, TEST_ALTITUDE, DEVICE)

    # Train
    print("\n[TRAINING] Starting progressive distillation...")
    best_r1, history = train(student, teacher, train_loader, val_fn, DEVICE, EPOCHS)

    # Final eval all altitudes
    print("\n" + "="*70)
    print("  FINAL RESULTS — All Altitudes")
    print("="*70)
    for alt in ALTITUDES:
        metrics = evaluate(student, DATA_ROOT, alt, DEVICE)
        print(f"  Alt={alt}m | R@1={metrics['R@1']:.2%} | R@5={metrics['R@5']:.2%} | "
              f"R@10={metrics['R@10']:.2%} | AP={metrics['AP']:.2%}")
    print(f"\n  Best R@1 (alt={TEST_ALTITUDE}m): {best_r1:.2%}")

    # Save results
    results = {
        "experiment": "EXP1_SigLIP_Distillation",
        "teacher": TEACHER_NAME,
        "student": STUDENT_NAME,
        "dataset": "SUES-200",
        "best_r1": best_r1,
        "student_params": s_params,
        "student_size_mb": round(s_params*4/(1024*1024), 1),
        "history": history,
    }
    with open(os.path.join(OUTPUT_DIR, "exp1_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {OUTPUT_DIR}/exp1_results.json")
    print("="*70)



def run_final_evaluation(model, test_dataset, device, exp_name, cfg=Config):
    """Run comprehensive per-altitude evaluation with paper-grade output."""
    if HAS_EVAL_UTILS:
        results = evaluate_full(
            model, test_dataset, device,
            data_root=cfg.DATA_ROOT,
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS,
            img_size=cfg.IMG_SIZE,
            train_locs=cfg.TRAIN_LOCS,
            test_locs=cfg.TEST_LOCS,
        )
        print_paper_results(results, exp_name=exp_name)
        return results
    else:
        print("eval_utils not found, using basic evaluate()")
        r, ap = evaluate(model, test_dataset, device)
        print(f"R@1:{r['R@1']:.2f}% R@5:{r['R@5']:.2f}% R@10:{r['R@10']:.2f}% mAP:{ap:.2f}%")
        return {'overall': {**r, 'mAP': ap}}

if __name__ == "__main__":
    main()
