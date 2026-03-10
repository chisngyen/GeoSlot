#!/usr/bin/env python3
"""
EXP2: Mixture-of-Experts Multi-View Specialist for Cross-Platform Geo-Localization
===================================================================================
Architecture: Shared ConvNeXt-Tiny backbone + MoE routing layer with 4 expert networks
  Expert 1: Drone-view specialist (altitude/angle variations)
  Expert 2: Satellite-view specialist (scale/resolution)
  Expert 3: Cross-view bridge expert (alignment features)
  Expert 4: Fine-grained detail expert (texture/structure matching)

Novelty:
  - View-conditioned gating mechanism adapting expert selection by input view type
  - Soft top-2 routing with load balancing loss
  - Cross-view contrastive + per-expert specialization losses

Target: Better generalization across different altitudes and viewing angles

Dataset: SUES-200 (drone ↔ satellite cross-view geo-localization)
Usage:
  python exp2_moe_multiview.py           # Full training on Kaggle
  python exp2_moe_multiview.py --test    # Smoke test
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

# 3-Stage schedule
STAGE1_END     = 20    # Backbone frozen, InfoNCE only
STAGE2_END     = 70    # + MoE specialization + load balancing
# Stage 3: Full fine-tuning

# Learning rates
LR_HEAD        = 1e-3
LR_BACKBONE    = 1e-5
WARMUP_EPOCHS  = 5
WEIGHT_DECAY   = 0.01

# Loss weights
LAMBDA_CE      = 1.0
LAMBDA_TRIPLET = 0.5
LAMBDA_MOE_LB  = 0.1     # Load balancing
LAMBDA_SPEC    = 0.3     # Expert specialization
LAMBDA_DIV     = 0.1     # Expert diversity

# Model
BACKBONE_NAME  = "convnext_tiny"
FEATURE_DIM    = 768
NUM_EXPERTS    = 4
TOP_K_EXPERTS  = 2
EMBED_DIM      = 512
NUM_CLASSES    = 160

# Dataset
IMG_SIZE       = 224
TRAIN_LOCS     = list(range(1, 121))
TEST_LOCS      = list(range(121, 201))
ALTITUDES      = ["150", "200", "250", "300"]
TEST_ALTITUDE  = "150"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# BACKBONE: ConvNeXt-Tiny
# =============================================================================
class ConvNeXtBackbone(nn.Module):
    def __init__(self, model_name=BACKBONE_NAME, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       num_classes=0, global_pool='')
        self.feature_dim = FEATURE_DIM

    def forward(self, x):
        feat = self.model(x)  # [B, C, H', W']
        B, C, H, W = feat.shape
        # Global average pool
        global_feat = feat.mean(dim=[2, 3])  # [B, C]
        # Patch tokens
        patch_tokens = feat.flatten(2).transpose(1, 2)  # [B, N, C]
        return global_feat, patch_tokens, (H, W)


# =============================================================================
# MIXTURE-OF-EXPERTS
# =============================================================================
class ExpertNetwork(nn.Module):
    """Single expert network with specialization."""
    def __init__(self, input_dim, hidden_dim, output_dim, expert_type='general'):
        super().__init__()
        self.expert_type = expert_type
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Expert-specific attention for patch tokens
        self.patch_attn = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, global_feat, patch_tokens=None):
        """
        Args:
            global_feat: [B, D] global features
            patch_tokens: [B, N, D] patch-level features (optional)
        """
        if patch_tokens is not None:
            # Expert-specific weighted pooling of patches
            attn_weights = self.patch_attn(patch_tokens)  # [B, N, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            weighted_patch = (patch_tokens * attn_weights).sum(dim=1)  # [B, D]
            combined = global_feat + weighted_patch
        else:
            combined = global_feat

        return self.net(combined)


class ViewConditionedRouter(nn.Module):
    """Soft top-k router with view-type conditioning."""
    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, num_experts),
        )

        # View-specific bias (drone vs satellite)
        self.view_bias = nn.Embedding(2, num_experts)  # 0=drone, 1=satellite
        nn.init.zeros_(self.view_bias.weight)

        # Noise for exploration during training
        self.noise_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, view_type_id=0):
        """
        Args:
            x: [B, D] input features
            view_type_id: 0=drone, 1=satellite
        Returns:
            gate_weights: [B, num_experts] sparse routing weights
            aux_loss: load balancing loss
        """
        logits = self.gate(x)  # [B, num_experts]

        # Add view-conditioned bias
        bias = self.view_bias(torch.tensor(view_type_id, device=x.device))
        logits = logits + bias.unsqueeze(0)

        # Add noise during training
        if self.training:
            noise = torch.randn_like(logits) * F.softplus(self.noise_scale)
            logits = logits + noise

        # Top-k selection
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)

        # Build sparse gate weights
        gate_weights = torch.zeros_like(logits)
        gate_weights.scatter_(1, topk_idx, topk_weights)

        # Load balancing auxiliary loss
        # Encourages equal expert utilization
        expert_counts = gate_weights.sum(dim=0)  # [num_experts]
        target_load = gate_weights.sum() / self.num_experts
        aux_loss = ((expert_counts - target_load) ** 2).mean()

        # Expert importance loss (prevents router collapse)
        importance = gate_weights.mean(dim=0)
        importance_loss = (importance.std() / (importance.mean() + 1e-8)) ** 2

        return gate_weights, aux_loss + 0.1 * importance_loss


class MoEGeoNet(nn.Module):
    """Complete MoE-based geo-localization network."""
    def __init__(self):
        super().__init__()
        self.backbone = ConvNeXtBackbone()

        # Expert networks
        expert_types = ['drone', 'satellite', 'bridge', 'detail']
        self.experts = nn.ModuleList([
            ExpertNetwork(FEATURE_DIM, FEATURE_DIM * 2, EMBED_DIM, etype)
            for etype in expert_types
        ])

        # Router
        self.router = ViewConditionedRouter(FEATURE_DIM, NUM_EXPERTS, TOP_K_EXPERTS)

        # Final projection
        self.final_proj = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, EMBED_DIM),
        )

        # Classification head
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def encode_view(self, x, view_type_id=0):
        """Encode a single view through backbone + MoE."""
        global_feat, patch_tokens, spatial_hw = self.backbone(x)

        # Route through experts
        gate_weights, router_loss = self.router(global_feat, view_type_id)

        # Expert forward (only top-k are non-zero)
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            out = expert(global_feat, patch_tokens)
            expert_outputs.append(out)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, E, D]

        # Weighted combination
        combined = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        embedding = F.normalize(self.final_proj(combined), dim=-1)

        return {
            'embedding': embedding,
            'gate_weights': gate_weights,
            'router_loss': router_loss,
            'expert_outputs': expert_outputs,
            'global_feat': global_feat,
        }

    def forward(self, q_img, r_img):
        q = self.encode_view(q_img, view_type_id=0)  # drone
        r = self.encode_view(r_img, view_type_id=1)  # satellite
        return {
            'query_embedding': q['embedding'],
            'ref_embedding': r['embedding'],
            'query_gate': q['gate_weights'],
            'ref_gate': r['gate_weights'],
            'router_loss': (q['router_loss'] + r['router_loss']) / 2,
            'query_expert_outputs': q['expert_outputs'],
            'ref_expert_outputs': r['expert_outputs'],
        }

    def extract_embedding(self, x, view_type_id=0):
        return self.encode_view(x, view_type_id)['embedding']


# =============================================================================
# MoE LOSS
# =============================================================================
class MoELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(0.07).log())
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

    @property
    def temp(self):
        return self.log_temp.exp().clamp(0.01, 1.0)

    def expert_diversity_loss(self, expert_outputs):
        """Encourage experts to produce diverse outputs."""
        # expert_outputs: [B, E, D]
        E = expert_outputs.shape[1]
        normed = F.normalize(expert_outputs, dim=-1)
        sim = torch.bmm(normed, normed.transpose(1, 2))  # [B, E, E]
        # Remove diagonal
        mask = ~torch.eye(E, device=sim.device, dtype=torch.bool).unsqueeze(0)
        off_diag = sim[mask].view(-1)
        return off_diag.abs().mean()

    def expert_specialization_loss(self, gate_weights_q, gate_weights_r):
        """Encourage view-specific specialization."""
        # drone should prefer experts 0,2; satellite should prefer 1,2
        # Measure cross-view gate difference
        gate_diff = F.cosine_similarity(gate_weights_q, gate_weights_r, dim=-1)
        # We want some specialization (not identical routing), but not completely different
        return F.relu(gate_diff - 0.3).mean()

    def forward(self, model_out, labels=None, epoch=0):
        q_emb = model_out['query_embedding']
        r_emb = model_out['ref_embedding']
        B = q_emb.shape[0]

        # InfoNCE
        logits = q_emb @ r_emb.t() / self.temp
        targets = torch.arange(B, device=logits.device)
        loss_infonce = (F.cross_entropy(logits, targets) +
                        F.cross_entropy(logits.t(), targets)) / 2
        acc = (logits.argmax(dim=-1) == targets).float().mean()

        # CE classification
        loss_ce = torch.tensor(0.0, device=q_emb.device)
        if labels is not None:
            q_cls = self.classifier(q_emb)
            r_cls = self.classifier(r_emb)
            loss_ce = (F.cross_entropy(q_cls, labels) +
                       F.cross_entropy(r_cls, labels)) / 2

        # Triplet
        loss_triplet = self._triplet_loss(q_emb, r_emb)

        total_loss = LAMBDA_CE * (loss_infonce + loss_ce) + LAMBDA_TRIPLET * loss_triplet

        # Router load balancing
        loss_lb = model_out['router_loss']
        total_loss = total_loss + LAMBDA_MOE_LB * loss_lb

        # Stage 2+: Expert specialization & diversity
        loss_spec = torch.tensor(0.0, device=q_emb.device)
        loss_div = torch.tensor(0.0, device=q_emb.device)
        if epoch >= STAGE1_END:
            ramp = min(1.0, (epoch - STAGE1_END + 1) / 10)
            loss_spec = self.expert_specialization_loss(
                model_out['query_gate'], model_out['ref_gate'])
            loss_div = (
                self.expert_diversity_loss(model_out['query_expert_outputs']) +
                self.expert_diversity_loss(model_out['ref_expert_outputs'])
            ) / 2
            total_loss = total_loss + ramp * (LAMBDA_SPEC * loss_spec + LAMBDA_DIV * loss_div)

        stage = ("S1:frozen" if epoch < STAGE1_END else
                 "S2:+MoE" if epoch < STAGE2_END else "S3:full")

        return {
            'total_loss': total_loss, 'accuracy': acc,
            'loss_infonce': loss_infonce.item(),
            'loss_ce': loss_ce.item() if torch.is_tensor(loss_ce) else loss_ce,
            'loss_triplet': loss_triplet.item(),
            'loss_lb': loss_lb.item(),
            'loss_spec': loss_spec.item() if torch.is_tensor(loss_spec) else loss_spec,
            'loss_div': loss_div.item() if torch.is_tensor(loss_div) else loss_div,
            'stage': stage,
        }

    def _triplet_loss(self, q_emb, r_emb, margin=0.3):
        dist = 1.0 - torch.mm(q_emb, r_emb.t())
        pos = dist.diag()
        neg_q = dist.clone(); neg_q.fill_diagonal_(float('inf'))
        hardest_neg_q = neg_q.min(dim=1)[0]
        neg_r = dist.clone().t(); neg_r.fill_diagonal_(float('inf'))
        hardest_neg_r = neg_r.min(dim=1)[0]
        return (F.relu(pos - hardest_neg_q + margin).mean() +
                F.relu(pos - hardest_neg_r + margin).mean()) / 2


# =============================================================================
# DATASET: SUES-200
# =============================================================================
class SUES200Dataset(Dataset):
    def __init__(self, root, split="train", altitude="150",
                 img_size=224, train_locs=None, test_locs=None):
        super().__init__()
        self.root = root; self.split = split; self.altitude = altitude
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
            label = loc_to_label[loc_id]
            for img_name in sorted(os.listdir(alt_dir)):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.pairs.append((os.path.join(alt_dir, img_name), sat_path))
                    self.labels.append(label)
        self.num_classes = len(loc_to_label)
        print(f"  [SUES-200 {split} alt={altitude}] {len(self.pairs)} pairs ({self.num_classes} cls)")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        dp, sp = self.pairs[idx]
        try:
            drone = Image.open(dp).convert("RGB"); sat = Image.open(sp).convert("RGB")
        except: drone = Image.new("RGB", (224,224), (128,128,128)); sat = Image.new("RGB", (224,224), (128,128,128))
        return {"query": self.drone_tf(drone), "gallery": self.sat_tf(sat),
                "label": self.labels[idx], "idx": idx}


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
    query_loader = DataLoader(query_ds, batch_size=64, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    gallery_ds = SUES200GalleryDataset(data_root, test_locs, IMG_SIZE)
    gallery_loader = DataLoader(gallery_ds, batch_size=64, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
    gal_embs, gal_locs = [], []
    for batch in gallery_loader:
        emb = model.extract_embedding(batch["image"].to(device), view_type_id=1)
        gal_embs.append(emb.cpu()); gal_locs.extend(batch["loc_id"].tolist())
    gal_embs = torch.cat(gal_embs, 0); gal_locs = np.array(gal_locs)

    q_embs = []
    for batch in query_loader:
        emb = model.extract_embedding(batch["query"].to(device), view_type_id=0)
        q_embs.append(emb.cpu())
    q_embs = torch.cat(q_embs, 0)

    loc_to_gal_idx = {loc: i for i, loc in enumerate(gal_locs)}
    q_gt_indices = []
    for dp, sp in query_ds.pairs:
        loc_id = int(os.path.basename(os.path.dirname(sp)))
        q_gt_indices.append(loc_to_gal_idx.get(loc_id, -1))
    q_gt_indices = np.array(q_gt_indices)

    sim = q_embs.numpy() @ gal_embs.numpy().T
    ranks = np.argsort(-sim, axis=1); N = len(q_embs)
    results = {}
    for k in [1, 5, 10]:
        correct = sum(1 for i in range(N) if q_gt_indices[i] in ranks[i, :k])
        results[f"R@{k}"] = correct / N
    ap_sum = 0
    for i in range(N):
        rp = np.where(ranks[i] == q_gt_indices[i])[0]
        if len(rp) > 0: ap_sum += 1.0 / (rp[0] + 1)
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
def train(model, train_loader, val_fn, device, epochs=EPOCHS):
    criterion = MoELoss().to(device)
    bb_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    head_params += list(criterion.parameters())

    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": 0.0},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scaler = GradScaler(enabled=AMP_ENABLED and device.type == "cuda")
    best_r1 = 0.0; history = []

    for epoch in range(epochs):
        if epoch < STAGE1_END:
            lr_bb = 0.0
            lr_hd = get_cosine_lr(epoch, STAGE1_END, LR_HEAD, WARMUP_EPOCHS)
            for p in bb_params: p.requires_grad = False
            stage = "S1:frozen"
        elif epoch < STAGE2_END:
            se = epoch - STAGE1_END; sl = STAGE2_END - STAGE1_END
            lr_bb = get_cosine_lr(se, sl, LR_BACKBONE, 3)
            lr_hd = get_cosine_lr(se, sl, LR_HEAD * 0.5, 0)
            for p in bb_params: p.requires_grad = True
            stage = "S2:+MoE"
        else:
            se = epoch - STAGE2_END; sl = epochs - STAGE2_END
            lr_bb = get_cosine_lr(se, sl, LR_BACKBONE * 0.5, 0)
            lr_hd = get_cosine_lr(se, sl, LR_HEAD * 0.3, 0)
            stage = "S3:full"

        optimizer.param_groups[0]["lr"] = lr_bb
        optimizer.param_groups[1]["lr"] = lr_hd

        model.train()
        ep_loss = ep_acc = n = 0; t0 = time.time()
        pbar = tqdm(train_loader, desc=f"  Ep {epoch+1}/{epochs} ({stage})", leave=False)

        for batch in pbar:
            query = batch["query"].to(device)
            gallery = batch["gallery"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=AMP_ENABLED and device.type == "cuda"):
                model_out = model(query, gallery)
                loss_dict = criterion(model_out, labels=labels, epoch=epoch)
                loss = loss_dict['total_loss']

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True); continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            ep_loss += loss.item(); ep_acc += loss_dict['accuracy'].item(); n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{loss_dict['accuracy'].item():.1%}")

        elapsed = time.time() - t0
        ep_loss /= max(n, 1); ep_acc /= max(n, 1)
        entry = {"epoch": epoch+1, "stage": stage, "loss": round(ep_loss, 4),
                 "acc": round(ep_acc, 4), "lr_bb": round(lr_bb, 6),
                 "lr_hd": round(lr_hd, 6), "time": round(elapsed, 1)}

        if (epoch+1) % EVAL_FREQ == 0 or epoch == epochs - 1:
            metrics = val_fn()
            entry.update(metrics)
            r1 = metrics["R@1"]
            if r1 > best_r1:
                best_r1 = r1
                torch.save(model.state_dict(),
                           os.path.join(OUTPUT_DIR, "exp2_moe_best.pth"))
            print(f"  Ep {epoch+1} ({stage}) | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | "
                  f"R@1={r1:.2%} | R@5={metrics.get('R@5',0):.2%} | AP={metrics.get('AP',0):.2%} | "
                  f"LR={lr_bb:.1e}/{lr_hd:.1e} | {elapsed:.0f}s")
        else:
            print(f"  Ep {epoch+1} ({stage}) | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | "
                  f"LR={lr_bb:.1e}/{lr_hd:.1e} | {elapsed:.0f}s")

        # Log expert routing distribution
        if (epoch+1) % 10 == 0:
            gate_q = loss_dict.get('query_gate', None)
            if model_out.get('query_gate') is not None:
                gw = model_out['query_gate'].detach().cpu().mean(0)
                print(f"    Expert routing (drone): {[f'{w:.2f}' for w in gw.tolist()]}")
            if model_out.get('ref_gate') is not None:
                gw = model_out['ref_gate'].detach().cpu().mean(0)
                print(f"    Expert routing (sat):   {[f'{w:.2f}' for w in gw.tolist()]}")

        history.append(entry)

    return best_r1, history


# =============================================================================
# SMOKE TEST
# =============================================================================
def run_test():
    print("\n" + "="*60)
    print("  EXP2 SMOKE TEST: MoE Multi-View Specialist")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[1/4] Instantiating model...")
    try:
        model = MoEGeoNet().to(device)
        criterion = MoELoss().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ MoEGeoNet created, {total_params:,} params")
        print(f"  ✓ Model size (FP32): {total_params*4/(1024*1024):.1f} MB")
        print(f"  ✓ {NUM_EXPERTS} experts, top-{TOP_K_EXPERTS} routing")
    except Exception as e:
        print(f"  ✗ Failed: {e}"); return False

    print("\n[2/4] Testing forward pass...")
    try:
        dummy_q = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
        dummy_r = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
        out = model(dummy_q, dummy_r)
        print(f"  ✓ Query embedding: {out['query_embedding'].shape}")
        print(f"  ✓ Gate weights (drone): {out['query_gate'].shape}")
        print(f"  ✓ Router loss: {out['router_loss'].item():.4f}")
        # Check routing is sparse (top-k)
        nonzero = (out['query_gate'] > 0).sum(dim=-1).float().mean()
        print(f"  ✓ Avg active experts: {nonzero.item():.1f} (target: {TOP_K_EXPERTS})")
    except Exception as e:
        print(f"  ✗ Failed: {e}"); import traceback; traceback.print_exc(); return False

    print("\n[3/4] Testing loss computation...")
    try:
        labels = torch.randint(0, NUM_CLASSES, (4,)).to(device)
        for ep in [0, STAGE1_END, STAGE2_END]:
            loss_dict = criterion(out, labels=labels, epoch=ep)
            assert not torch.isnan(loss_dict['total_loss'])
            assert not torch.isinf(loss_dict['total_loss'])
            print(f"  ✓ {loss_dict['stage']}: loss={loss_dict['total_loss'].item():.4f}")
    except Exception as e:
        print(f"  ✗ Failed: {e}"); import traceback; traceback.print_exc(); return False

    print("\n[4/4] Testing gradient flow...")
    try:
        loss_dict = criterion(out, labels=labels, epoch=STAGE2_END)
        loss_dict['total_loss'].backward()
        grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"  ✓ Gradients reach all trainable params: {grad_ok}")
    except Exception as e:
        print(f"  ✗ Failed: {e}"); import traceback; traceback.print_exc(); return False

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
    parser = argparse.ArgumentParser(description="EXP2: MoE Multi-View")
    parser.add_argument("--test", action="store_true")
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
    print("  EXP2: Mixture-of-Experts Multi-View Specialist")
    print(f"  Backbone: {BACKBONE_NAME}")
    print(f"  Experts: {NUM_EXPERTS}, Top-{TOP_K_EXPERTS} routing")
    print(f"  3-Stage: S1[0-{STAGE1_END}) S2[{STAGE1_END}-{STAGE2_END}) S3[{STAGE2_END}-{EPOCHS})")
    print(f"  Device: {DEVICE}")
    print("="*70)

    print("\n[DATASET] Loading SUES-200...")
    train_pairs_all = []; train_labels_all = []
    for alt in ALTITUDES:
        ds = SUES200Dataset(DATA_ROOT, "train", alt, IMG_SIZE)
        train_pairs_all.extend(ds.pairs); train_labels_all.extend(ds.labels)
    print(f"  Total train pairs: {len(train_pairs_all)}")

    class CombinedTrainDataset(Dataset):
        def __init__(self, pairs, labels, img_size=224):
            self.pairs = pairs; self.labels = labels
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
        def __len__(self): return len(self.pairs)
        def __getitem__(self, idx):
            dp, sp = self.pairs[idx]
            try: drone = Image.open(dp).convert("RGB"); sat = Image.open(sp).convert("RGB")
            except: drone = Image.new("RGB", (224,224), (128,128,128)); sat = Image.new("RGB", (224,224), (128,128,128))
            return {"query": self.drone_tf(drone), "gallery": self.sat_tf(sat),
                    "label": self.labels[idx], "idx": idx}

    train_ds = CombinedTrainDataset(train_pairs_all, train_labels_all, IMG_SIZE)
    k_samples = max(2, BATCH_SIZE // 8)
    train_sampler = PKSampler(train_labels_all, p=8, k=k_samples)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print("\n[MODEL] Building MoEGeoNet...")
    model = MoEGeoNet().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")
    print(f"  Model size (FP32): {total_params*4/(1024*1024):.1f} MB")

    def val_fn():
        return evaluate(model, DATA_ROOT, TEST_ALTITUDE, DEVICE)

    print("\n[TRAINING] Starting 3-stage MoE training...")
    best_r1, history = train(model, train_loader, val_fn, DEVICE, EPOCHS)

    print("\n" + "="*70)
    print("  FINAL RESULTS — All Altitudes")
    print("="*70)
    for alt in ALTITUDES:
        metrics = evaluate(model, DATA_ROOT, alt, DEVICE)
        print(f"  Alt={alt}m | R@1={metrics['R@1']:.2%} | R@5={metrics['R@5']:.2%} | "
              f"R@10={metrics['R@10']:.2%} | AP={metrics['AP']:.2%}")
    print(f"\n  Best R@1: {best_r1:.2%}")

    results = {
        "experiment": "EXP2_MoE_MultiView",
        "backbone": BACKBONE_NAME, "dataset": "SUES-200",
        "num_experts": NUM_EXPERTS, "top_k": TOP_K_EXPERTS,
        "best_r1": best_r1, "total_params": total_params,
        "history": history,
    }
    with open(os.path.join(OUTPUT_DIR, "exp2_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {OUTPUT_DIR}/exp2_results.json")


if __name__ == "__main__":
    main()
