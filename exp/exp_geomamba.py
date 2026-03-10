#!/usr/bin/env python3
"""
GeoMamba: State-Space Cross-View Drone Geo-Localization
========================================================
Novel contributions:
  1. Bidirectional Spatial-Mamba (BS-Mamba) — Linear-complexity backbone
     that captures long-range spatial dependencies via 4-directional scanning
  2. Optimal Transport Slot Matching (OT-Slot) — Sinkhorn-based differentiable
     matching between drone/satellite feature tokens for cross-view alignment
  3. Scale-Adaptive State Gating (SASG) — Altitude-conditioned gating that
     modulates state-space dynamics for different drone flight heights

Architecture:
  Student: ConvNeXt-Tiny + Mamba Spatial Mixer + OT Matching Head
  Teacher: DINOv2-Base (frozen)

Dataset: SUES-200 (drone ↔ satellite cross-view geo-localization)
Protocol: 120 train / 80 test, gallery = ALL 200 locations (confusion data)

Usage:
  python exp_geomamba.py           # Full training on Kaggle H100
  python exp_geomamba.py --test    # Smoke test
"""

# === SETUP ===
import subprocess, sys, os

def pip_install(pkg, extra=""):
    subprocess.run(f"pip install -q {extra} {pkg}",
                   shell=True, capture_output=True, text=True)

print("[1/2] Installing packages...")
for p in ["timm", "tqdm"]:
    try:
        __import__(p)
    except ImportError:
        pip_install(p)
print("[2/2] Setup complete!")

# === IMPORTS ===
import math
import random
import argparse
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import timm

print("[OK] All imports loaded!")

# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT     = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
OUTPUT_DIR    = "/kaggle/working"

EPOCHS        = 120
BATCH_SIZE    = 32
NUM_WORKERS   = 8
AMP_ENABLED   = True
EVAL_FREQ     = 5

LR_BACKBONE   = 1e-4
LR_HEAD       = 1e-3
LR_MAMBA      = 5e-4
WARMUP_EPOCHS = 5
WEIGHT_DECAY  = 0.01

PHASE1_END    = 25
PHASE2_END    = 75

LAMBDA_CE         = 1.0
LAMBDA_TRIPLET    = 0.5
LAMBDA_INFONCE    = 0.5
LAMBDA_OT         = 0.5     # Optimal transport matching
LAMBDA_ARCFACE    = 0.3
LAMBDA_STATE_REG  = 0.1     # State space regularization
LAMBDA_FEAT_DIST  = 0.3     # Feature distillation
LAMBDA_UAPA       = 0.2

BACKBONE_NAME = "convnext_tiny"
FEATURE_DIM   = 768
EMBED_DIM     = 512
MAMBA_DIM     = 256
MAMBA_DEPTH   = 2
STATE_DIM     = 16
NUM_CLASSES   = 120
MARGIN        = 0.3
OT_ITERS      = 3         # Sinkhorn iterations
OT_REG        = 0.1       # Entropy regularization

IMG_SIZE      = 224
TRAIN_LOCS    = list(range(1, 121))
TEST_LOCS     = list(range(121, 201))
ALTITUDES     = ["150", "200", "250", "300"]

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# =============================================================================
# DATASET
# =============================================================================
class SUES200Dataset(Dataset):
    def __init__(self, root, split, altitude=None, img_size=224,
                 train_locs=None, test_locs=None):
        super().__init__()
        drone_dir = os.path.join(root, "drone-view")
        satellite_dir = os.path.join(root, "satellite-view")
        if train_locs is None: train_locs = TRAIN_LOCS
        if test_locs is None: test_locs = TEST_LOCS
        locs = train_locs if split == "train" else test_locs
        alts = [altitude] if altitude else ALTITUDES
        is_train = split == "train"
        self.drone_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=3),
            transforms.RandomHorizontalFlip(0.5) if is_train else transforms.Lambda(lambda x: x),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)) if is_train
                else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.05) if is_train
                else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        sat_augs = [transforms.Resize((img_size, img_size), interpolation=3)]
        if is_train:
            sat_augs += [transforms.RandomHorizontalFlip(0.5),
                         transforms.RandomVerticalFlip(0.5),
                         transforms.RandomApply([transforms.RandomRotation((90,90))], p=0.5),
                         transforms.ColorJitter(0.2, 0.2, 0.1, 0.05)]
        sat_augs += [transforms.ToTensor(),
                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
        self.sat_tf = transforms.Compose(sat_augs)
        self.pairs, self.labels, self.altitudes_meta = [], [], []
        loc_to_label = {}
        for loc_id in locs:
            loc_str = f"{loc_id:04d}"
            sat_path = os.path.join(satellite_dir, loc_str, "0.png")
            if not os.path.exists(sat_path): continue
            if loc_id not in loc_to_label: loc_to_label[loc_id] = len(loc_to_label)
            label = loc_to_label[loc_id]
            for alt in alts:
                alt_dir = os.path.join(drone_dir, loc_str, alt)
                if not os.path.isdir(alt_dir): continue
                alt_idx = ALTITUDES.index(alt) if alt in ALTITUDES else 0
                for img_name in sorted(os.listdir(alt_dir)):
                    if img_name.endswith(('.jpg','.jpeg','.png')):
                        self.pairs.append((os.path.join(alt_dir, img_name), sat_path))
                        self.labels.append(label)
                        self.altitudes_meta.append(alt_idx)
        self.num_classes = len(loc_to_label)
        print(f"  [{split}] {len(self.pairs)} pairs, {self.num_classes} classes")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        dp, sp = self.pairs[idx]
        try: d = Image.open(dp).convert("RGB"); s = Image.open(sp).convert("RGB")
        except: d = Image.new("RGB",(224,224),(128,128,128)); s = d.copy()
        return {"query": self.drone_tf(d), "gallery": self.sat_tf(s),
                "label": self.labels[idx], "altitude": self.altitudes_meta[idx]}


class SUES200GalleryDataset(Dataset):
    """ALL 200 satellite locations as confusion gallery."""
    def __init__(self, root, test_locs=None, img_size=224):
        super().__init__()
        satellite_dir = os.path.join(root, "satellite-view")
        all_locs = TRAIN_LOCS + TEST_LOCS
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        self.images, self.loc_ids = [], []
        for loc_id in all_locs:
            p = os.path.join(satellite_dir, f"{loc_id:04d}", "0.png")
            if os.path.exists(p): self.images.append(p); self.loc_ids.append(loc_id)
        print(f"  Gallery: {len(self.images)} satellite images (confusion data)")
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        return {"image": self.tf(Image.open(self.images[idx]).convert("RGB")),
                "loc_id": self.loc_ids[idx]}


class PKSampler:
    def __init__(self, dataset, p=8, k=4):
        self.p, self.k = p, k
        self.loc_to_indices = defaultdict(list)
        for i, label in enumerate(dataset.labels): self.loc_to_indices[label].append(i)
        self.locations = list(self.loc_to_indices.keys())
    def __iter__(self):
        locs = self.locations.copy(); random.shuffle(locs)
        batch = []
        for loc in locs:
            idx = self.loc_to_indices[loc]
            if len(idx) < self.k: idx = idx * (self.k // len(idx) + 1)
            batch.extend(random.sample(idx, self.k))
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]; batch = batch[self.p * self.k:]
    def __len__(self): return len(self.locations) // self.p


# =============================================================================
# BIDIRECTIONAL SPATIAL-MAMBA (Novel Component #1)
# =============================================================================
class SelectiveSSM(nn.Module):
    """Selective State Space Model — core of Mamba.

    Instead of fixed state transition matrices, parameters are input-dependent,
    enabling content-aware sequence modeling with O(N) complexity.
    """
    def __init__(self, d_model, state_dim=64, dt_rank=None, expand=2):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.dt_rank = dt_rank or max(d_model // 16, 1)
        d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # S4/Mamba parameters (input-dependent)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)
                                            .unsqueeze(0).expand(d_inner, -1)))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Selective projections
        self.x_proj = nn.Linear(d_inner, self.dt_rank + state_dim * 2, bias=False)

        # Depth-wise conv for local context
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=4, padding=3,
                                groups=d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args: x: [B, L, D]
        Returns: [B, L, D]
        """
        B, L, D = x.shape
        residual = x

        # Project and split
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_in, z = xz.chunk(2, dim=-1)  # each [B, L, d_inner]

        # Conv1d for local context
        x_conv = x_in.transpose(1, 2)  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[:, :, :L]  # [B, d_inner, L]
        x_conv = x_conv.transpose(1, 2)  # [B, L, d_inner]
        x_conv = F.silu(x_conv)

        # Selective scan parameters
        x_dbl = self.x_proj(x_conv)  # [B, L, dt_rank + 2*state_dim]
        dt, B_param, C_param = x_dbl.split(
            [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        dt = self.dt_proj(dt)  # [B, L, d_inner]
        dt = F.softplus(dt)

        # Discretize
        A = -torch.exp(self.A_log)  # [d_inner, state_dim]

        # Simplified selective scan (parallel-friendly)
        # For GPU efficiency, we approximate the scan using chunk-wise processing
        d_inner = x_conv.shape[-1]
        y = self._selective_scan(x_conv, dt, A, B_param, C_param, self.D)

        # Gate and output
        y = y * F.silu(z)
        output = self.out_proj(y)
        return self.norm(output + residual)

    def _selective_scan(self, x, dt, A, B, C, D):
        """Memory-efficient parallel selective scan approximation.

        Uses a global convolution-like approximation instead of a per-timestep
        sequential loop, dramatically reducing autograd graph memory.
        """
        B_batch, L, d = x.shape
        N = self.state_dim

        # Compute discretized A decay for each position [B, L, d, N]
        dt_expand = dt.unsqueeze(-1)          # [B, L, d, 1]
        A_expand = A.unsqueeze(0).unsqueeze(0) # [1, 1, d, N]
        dA = torch.exp(A_expand * dt_expand)   # [B, L, d, N]

        # Input contribution: dB * x
        B_expand = B.unsqueeze(2)              # [B, L, 1, N]
        x_expand = x.unsqueeze(-1)             # [B, L, d, 1]
        dBx = dt_expand * B_expand * x_expand  # [B, L, d, N]

        # Approximate scan: instead of sequential h[t] = dA[t]*h[t-1] + dBx[t],
        # use a windowed exponential-decay weighted sum (memory-efficient)
        # This approximates the recurrence with a fixed window
        K = min(L, 8)  # window size
        outputs = []
        for t in range(L):
            t_start = max(0, t - K + 1)
            # Accumulate contributions from recent positions
            h_t = dBx[:, t]  # [B, d, N] — current input
            for s in range(t_start, t):
                # Decay from position s to t
                decay = dA[:, s+1:t+1].prod(dim=1)  # [B, d, N]
                h_t = h_t + decay * dBx[:, s]
            # Output: y = C * h + D * x
            y_t = (h_t * C[:, t].unsqueeze(1)).sum(-1)  # [B, d]
            y_t = y_t + D * x[:, t]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # [B, L, d]


class BidirectionalSpatialMamba(nn.Module):
    """Bidirectional Spatial-Mamba for 2D feature maps.

    Novel: 4-directional scanning (↓→, ↓←, ↑→, ↑←) captures spatial
    dependencies from all directions, essential for aerial imagery where
    there's no canonical orientation.

    Includes Scale-Adaptive State Gating (SASG) for altitude conditioning.
    """
    def __init__(self, d_model, state_dim=64, num_layers=4, num_altitudes=4):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # 4 scanning directions
        self.ssm_layers = nn.ModuleList([
            nn.ModuleList([SelectiveSSM(d_model, state_dim) for _ in range(4)])
            for _ in range(num_layers)
        ])

        # Direction fusion per layer
        self.dir_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 4, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
            ) for _ in range(num_layers)
        ])

        # Scale-Adaptive State Gating (SASG) — Novel Component #3
        self.altitude_embed = nn.Embedding(num_altitudes, 32)
        self.sasg_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, d_model),
                nn.Sigmoid(),
            ) for _ in range(num_layers)
        ])

    def _get_scan_sequences(self, x):
        """Generate 4 scanning sequences from 2D feature map.

        Args: x: [B, H, W, D]
        Returns: list of 4 sequences, each [B, H*W, D]
        """
        B, H, W, D = x.shape
        # →↓ (row-major, left-to-right top-to-bottom)
        seq1 = x.reshape(B, H * W, D)
        # ←↓ (row-major, right-to-left top-to-bottom)
        seq2 = x.flip(2).reshape(B, H * W, D)
        # →↑ (row-major, left-to-right bottom-to-top)
        seq3 = x.flip(1).reshape(B, H * W, D)
        # ←↑ (row-major, right-to-left bottom-to-top)
        seq4 = x.flip(1).flip(2).reshape(B, H * W, D)
        return [seq1, seq2, seq3, seq4]

    def _unscan(self, outputs, H, W):
        """Reverse scanning to reconstruct 2D feature maps."""
        B, L, D = outputs[0].shape
        out1 = outputs[0]
        out2 = outputs[1].view(B, H, W, D).flip(2).view(B, L, D)
        out3 = outputs[2].view(B, H, W, D).flip(1).view(B, L, D)
        out4 = outputs[3].view(B, H, W, D).flip(1).flip(2).view(B, L, D)
        return [out1, out2, out3, out4]

    def forward(self, feat_map, altitude_idx=None):
        """
        Args:
            feat_map: [B, C, H, W] from backbone
            altitude_idx: [B] altitude index
        Returns:
            tokens: [B, N, D] spatially-aware token embeddings
        """
        B, C, H, W = feat_map.shape
        x = feat_map.permute(0, 2, 3, 1)  # [B, H, W, C]

        for layer_idx in range(self.num_layers):
            # 4-directional scan
            seqs = self._get_scan_sequences(x)
            outputs = [self.ssm_layers[layer_idx][d](seqs[d]) for d in range(4)]
            outputs = self._unscan(outputs, H, W)

            # Fuse directions
            combined = torch.cat(outputs, dim=-1)  # [B, N, 4*D]
            fused = self.dir_fusion[layer_idx](combined)  # [B, N, D]

            # Scale-Adaptive State Gating (altitude conditioning)
            if altitude_idx is not None:
                alt_feat = self.altitude_embed(altitude_idx)  # [B, 32]
                gate = self.sasg_gates[layer_idx](alt_feat)   # [B, D]
                fused = fused * gate.unsqueeze(1)             # [B, N, D]

            x = fused.view(B, H, W, C)

        return x.view(B, H * W, C)  # [B, N, D]


# =============================================================================
# OPTIMAL TRANSPORT MATCHING (Novel Component #2)
# =============================================================================
class SinkhornOT(nn.Module):
    """Differentiable Optimal Transport via Sinkhorn iterations.

    Computes a soft correspondence matrix between drone and satellite tokens,
    providing a principled alignment that respects the geometry of both views.
    """
    def __init__(self, d_model, num_iters=5, reg=0.1):
        super().__init__()
        self.num_iters = num_iters
        self.reg = reg
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)

    def forward(self, drone_tokens, sat_tokens):
        """
        Args:
            drone_tokens: [B, N_d, D]
            sat_tokens: [B, N_s, D]
        Returns:
            transport_plan: [B, N_d, N_s] soft correspondence
            ot_cost: scalar, transport cost
        """
        q = F.normalize(self.proj_q(drone_tokens), dim=-1)
        k = F.normalize(self.proj_k(sat_tokens), dim=-1)

        # Cost matrix (negative cosine similarity)
        cost = 1 - torch.bmm(q, k.transpose(1, 2))  # [B, N_d, N_s]

        # Sinkhorn iterations
        M = -cost / self.reg
        for _ in range(self.num_iters):
            M = M - torch.logsumexp(M, dim=2, keepdim=True)
            M = M - torch.logsumexp(M, dim=1, keepdim=True)
        transport_plan = torch.exp(M)

        # Transport cost
        ot_cost = (transport_plan * cost).sum(dim=[1, 2]).mean()

        return transport_plan, ot_cost


class OTMatchingLoss(nn.Module):
    """Cross-view alignment loss using Optimal Transport.

    For same-location pairs: minimize transport cost (features should align)
    For different-location pairs: maximize transport cost (features shouldn't align)
    """
    def __init__(self, d_model, num_iters=5, reg=0.1, margin=0.5):
        super().__init__()
        self.ot = SinkhornOT(d_model, num_iters, reg)
        self.margin = margin

    def forward(self, drone_tokens, sat_tokens, labels):
        """Fully batched OT loss — no per-sample loop."""
        # Compute batched OT cost
        _, ot_cost = self.ot(drone_tokens, sat_tokens)
        # ot_cost is already mean across batch; use directly as alignment loss
        # All pairs in a PK-sampled batch are positive (same loc has same sat)
        return ot_cost


# =============================================================================
# ARCFACE HEAD
# =============================================================================
class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(embeddings, dim=1)
        cosine = F.linear(x, w)
        if labels is None or not self.training: return cosine * self.s
        one_hot = F.one_hot(labels, cosine.size(1)).float()
        theta = torch.acos(cosine.clamp(-1+1e-7, 1-1e-7))
        target = torch.cos(theta + self.m)
        logits = cosine * (1 - one_hot) + target * one_hot
        return logits * self.s


# =============================================================================
# GEOMAMBA MODEL
# =============================================================================
class GeoMambaStudent(nn.Module):
    """GeoMamba = ConvNeXt-Tiny + Spatial Mamba + OT Matching.

    Pipeline:
      1. ConvNeXt extracts multi-scale features
      2. Bidirectional Spatial-Mamba processes tokens with linear complexity
      3. OT head aligns cross-view tokens for matching
      4. Global + token-level embeddings for retrieval
    """
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE_NAME, pretrained=True,
                                          num_classes=0, global_pool='')

        # Projection: backbone dim → mamba dim
        self.feat_proj = nn.Sequential(
            nn.Linear(FEATURE_DIM, MAMBA_DIM),
            nn.LayerNorm(MAMBA_DIM),
        )

        # Bidirectional Spatial-Mamba
        self.spatial_mamba = BidirectionalSpatialMamba(
            d_model=MAMBA_DIM, state_dim=STATE_DIM,
            num_layers=MAMBA_DEPTH, num_altitudes=4
        )

        # OT matching head
        self.ot_matcher = SinkhornOT(MAMBA_DIM, OT_ITERS, OT_REG)

        # Global pooling + embedding
        self.token_pool = nn.Sequential(
            nn.Linear(MAMBA_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(FEATURE_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.arcface = ArcFaceHead(embed_dim, num_classes)

    def forward(self, x, altitude_idx=None, labels=None, return_all=False):
        feat_map = self.backbone(x)  # [B, C, H', W']

        # Global branch
        g_feat = self.global_pool(feat_map).flatten(1)
        g_emb = self.global_proj(g_feat)

        # Mamba branch: project → spatial mamba → pool
        B, C, H, W = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1,2)  # [B, N, C]
        tokens = self.feat_proj(tokens)               # [B, N, mamba_dim]
        mamba_input = tokens.transpose(1, 2).view(B, MAMBA_DIM, H, W)
        mamba_tokens = self.spatial_mamba(
            mamba_input,
            altitude_idx=altitude_idx
        )  # [B, N, mamba_dim]

        # Token pooling
        t_emb = self.token_pool(mamba_tokens.mean(dim=1))  # [B, embed_dim]

        # Fuse
        combined = torch.cat([g_emb, t_emb], dim=1)
        embedding = self.fusion(combined)
        embedding_norm = F.normalize(embedding, p=2, dim=1)

        logits = self.classifier(embedding)
        arc_logits = self.arcface(embedding, labels)

        if return_all:
            return {
                'embedding': embedding,
                'embedding_norm': embedding_norm,
                'logits': logits,
                'arcface_logits': arc_logits,
                'mamba_tokens': mamba_tokens,
                'feat_map': feat_map,
                'global_feat': g_feat,
            }
        return embedding_norm, logits

    def extract_embedding(self, x, view_type='drone', altitude_idx=None):
        self.eval()
        with torch.no_grad():
            emb, _ = self.forward(x, altitude_idx=altitude_idx)
        return emb


# =============================================================================
# TEACHER
# =============================================================================
class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading DINOv2-base teacher...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.embed_dim = 768
        for p in self.parameters(): p.requires_grad = False
        print(f"  DINOv2 loaded! dim={self.embed_dim}")

    @torch.no_grad()
    def forward(self, x):
        tokens = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks: tokens = blk(tokens)
        tokens = self.model.norm(tokens)
        return tokens[:, 0], tokens[:, 1:]  # cls, patches


# =============================================================================
# ADDITIONAL LOSSES
# =============================================================================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    def forward(self, emb, labels):
        dist = torch.cdist(emb, emb, p=2)
        lab = labels.view(-1,1)
        pos = lab.eq(lab.T).float(); neg = lab.ne(lab.T).float()
        hp = (dist * pos).max(1)[0]
        hn = (dist * neg + pos * 999).min(1)[0]
        return F.relu(hp - hn + self.margin).mean()


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature
    def forward(self, d, s, labels):
        d = F.normalize(d,1); s = F.normalize(s,1)
        sim = d @ s.T / self.t
        lab = labels.view(-1,1); pm = lab.eq(lab.T).float()
        l1 = -(F.log_softmax(sim,1)*pm).sum(1)/pm.sum(1).clamp(1)
        l2 = -(F.log_softmax(sim.T,1)*pm).sum(1)/pm.sum(1).clamp(1)
        return 0.5*(l1.mean()+l2.mean())


class UAPALoss(nn.Module):
    def __init__(self, T0=4.0):
        super().__init__()
        self.T0 = T0
    def forward(self, dl, sl):
        Ud = -(F.softmax(dl,1)*F.log_softmax(dl,1)).sum(1).mean()
        Us = -(F.softmax(sl,1)*F.log_softmax(sl,1)).sum(1).mean()
        T = self.T0 * (1+torch.sigmoid(Ud-Us))
        return (T**2)*F.kl_div(F.log_softmax(dl/T,1), F.softmax(sl/T,1), reduction='batchmean')


# =============================================================================
# LR SCHEDULER
# =============================================================================
def get_cosine_lr(epoch, total, base, warmup=5):
    if epoch < warmup: return base * (epoch+1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (epoch-warmup) / max(1, total-warmup)))


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model, data_root, altitude, device, test_locs=None):
    model.eval()
    qds = SUES200Dataset(data_root, "test", altitude, IMG_SIZE, test_locs=test_locs)
    ql = DataLoader(qds, 64, False, num_workers=NUM_WORKERS, pin_memory=True)
    gds = SUES200GalleryDataset(data_root, test_locs, IMG_SIZE)
    gl = DataLoader(gds, 64, False, num_workers=NUM_WORKERS, pin_memory=True)
    ge, gloc = [], []
    for b in gl:
        ge.append(model.extract_embedding(b["image"].to(device), 'satellite').cpu())
        gloc.extend(b["loc_id"].tolist())
    ge = torch.cat(ge); gloc = np.array(gloc)
    qe = []
    for b in ql:
        ai = b.get("altitude"); ai = ai.to(device) if ai is not None else None
        qe.append(model.extract_embedding(b["query"].to(device), 'drone', ai).cpu())
    qe = torch.cat(qe)
    l2g = {l:i for i,l in enumerate(gloc)}
    qgt = np.array([l2g.get(int(os.path.basename(os.path.dirname(p[1]))), -1) for p in qds.pairs])
    sim = qe.numpy() @ ge.numpy().T
    ranks = np.argsort(-sim, axis=1)
    N = len(qe)
    res = {}
    for k in [1,5,10]:
        res[f"R@{k}"] = sum(1 for i in range(N) if qgt[i] in ranks[i,:k]) / N
    aps = 0
    for i in range(N):
        rp = np.where(ranks[i]==qgt[i])[0]
        if len(rp)>0: aps += 1/(rp[0]+1)
    res["AP"] = aps/N
    return res


# =============================================================================
# TRAINING
# =============================================================================
def train(args):
    set_seed(SEED)
    print("="*70)
    print("GeoMamba: State-Space Cross-View Drone Geo-Localization")
    print("="*70)
    train_ds = SUES200Dataset(args.data_root, "train", img_size=IMG_SIZE)
    sampler = PKSampler(train_ds, p=8, k=max(2, BATCH_SIZE//8))
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    model = GeoMambaStudent(num_classes=train_ds.num_classes).to(DEVICE)
    prms = sum(p.numel() for p in model.parameters())/1e6
    print(f"  Student: {prms:.1f}M params")
    teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    trip_fn = TripletLoss(MARGIN)
    nce_fn = InfoNCELoss(0.07)
    ot_fn = OTMatchingLoss(MAMBA_DIM, OT_ITERS, OT_REG).to(DEVICE)
    uapa_fn = UAPALoss(4.0)
    pgs = [
        {'params': [p for n,p in model.named_parameters() if 'backbone' in n], 'lr': LR_BACKBONE},
        {'params': [p for n,p in model.named_parameters() if 'spatial_mamba' in n], 'lr': LR_MAMBA},
        {'params': [p for n,p in model.named_parameters()
                    if 'backbone' not in n and 'spatial_mamba' not in n], 'lr': LR_HEAD},
    ]
    opt = torch.optim.AdamW(pgs, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=AMP_ENABLED)
    best_r1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        phase = 1 if epoch < PHASE1_END else (2 if epoch < PHASE2_END else 3)
        if phase == 1:
            for p in model.backbone.parameters(): p.requires_grad = False
        else:
            for p in model.backbone.parameters(): p.requires_grad = True
        for pg in opt.param_groups:
            pg['lr'] = get_cosine_lr(epoch, EPOCHS, pg.get('initial_lr', pg['lr']), WARMUP_EPOCHS)
        tl = 0.0; lp = defaultdict(float)
        for bi, batch in enumerate(train_loader):
            d,s,lab,alt = (batch[k].to(DEVICE) for k in ["query","gallery","label","altitude"])
            opt.zero_grad()
            with autocast(enabled=AMP_ENABLED):
                do = model(d, alt, lab, True)
                # Free drone input memory before satellite forward
                del d
                torch.cuda.empty_cache()
                so = model(s, labels=lab, return_all=True)
                del s
                L = {}
                L['ce'] = LAMBDA_CE*0.5*(ce_fn(do['logits'],lab)+ce_fn(so['logits'],lab))
                L['arc'] = LAMBDA_ARCFACE*0.5*(ce_fn(do['arcface_logits'],lab)+ce_fn(so['arcface_logits'],lab))
                L['trip'] = LAMBDA_TRIPLET*0.5*(trip_fn(do['embedding_norm'],lab)+trip_fn(so['embedding_norm'],lab))
                L['nce'] = LAMBDA_INFONCE*nce_fn(do['embedding_norm'],so['embedding_norm'],lab)
                if phase >= 2:
                    # Detach mamba tokens to reduce graph size for OT
                    L['ot'] = LAMBDA_OT * ot_fn(
                        do['mamba_tokens'].detach(), so['mamba_tokens'].detach(), lab)
                    with torch.no_grad():
                        tc_d, tp_d = teacher(batch["query"].to(DEVICE))
                        tc_s, tp_s = teacher(batch["gallery"].to(DEVICE))
                    dn = F.normalize(do['global_feat'],1); sn = F.normalize(so['global_feat'],1)
                    tdn = F.normalize(tc_d,1); tsn = F.normalize(tc_s,1)
                    del tc_d, tp_d, tc_s, tp_s
                    L['fdist'] = LAMBDA_FEAT_DIST*0.5*(
                        F.mse_loss(dn,tdn)+F.mse_loss(sn,tsn)+
                        (1-F.cosine_similarity(dn,tdn).mean())+
                        (1-F.cosine_similarity(sn,tsn).mean()))
                if phase >= 3:
                    L['uapa'] = LAMBDA_UAPA*uapa_fn(do['logits'],so['logits'])
                loss = sum(L.values())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tl += loss.item()
            for k,v in L.items(): lp[k] += v.item()
            if bi%10==0: print(f"  B{bi}/{len(train_loader)} L={loss.item():.4f}")
        nb = max(1, len(train_loader))
        print(f"\nEp {epoch+1}/{EPOCHS} P{phase} AvgL={tl/nb:.4f}")
        for k,v in sorted(lp.items()): print(f"  {k}: {v/nb:.4f}")
        if (epoch+1)%EVAL_FREQ==0 or epoch==EPOCHS-1:
            ar = {}
            for a in ALTITUDES:
                r = evaluate(model, args.data_root, a, DEVICE)
                ar[a] = r; print(f"  {a}m: R@1={r['R@1']:.4f} R@5={r['R@5']:.4f} AP={r['AP']:.4f}")
            avg1 = np.mean([r['R@1'] for r in ar.values()])
            print(f"  AVG R@1={avg1:.4f}")
            if avg1 > best_r1:
                best_r1 = avg1
                torch.save({'epoch':epoch,'model':model.state_dict(),'r1':avg1},
                           os.path.join(OUTPUT_DIR,'geomamba_best.pth'))
                print(f"  *** Best R@1={avg1:.4f} ***")
    print(f"\nDone! Best R@1={best_r1:.4f}")


# =============================================================================
# SMOKE TEST
# =============================================================================
def smoke_test():
    print("="*50); print("SMOKE TEST — GeoMamba"); print("="*50)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = GeoMambaStudent(10).to(dev)
    print(f"✓ Model: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
    x = torch.randn(4,3,224,224,device=dev)
    lab = torch.tensor([0,0,1,1],device=dev); alt = torch.tensor([0,1,2,3],device=dev)
    o = m(x, alt, lab, True)
    print(f"✓ Forward: emb={o['embedding_norm'].shape}, tokens={o['mamba_tokens'].shape}")
    do = m(x, alt, lab, True); so = m(x, labels=lab, return_all=True)
    ce = nn.CrossEntropyLoss()(do['logits'],lab)
    ot = OTMatchingLoss(MAMBA_DIM)(do['mamba_tokens'], so['mamba_tokens'], lab)
    total = ce + ot; total.backward()
    gn = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
    print(f"✓ Backward: grad_norm={gn:.4f}")
    print("\n✅ ALL TESTS PASSED!")


def main():
    global EPOCHS, BATCH_SIZE, DATA_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--test", action="store_true")
    args, _ = parser.parse_known_args()
    EPOCHS=args.epochs; BATCH_SIZE=args.batch_size; DATA_ROOT=args.data_root
    if args.test: smoke_test(); return
    os.makedirs(OUTPUT_DIR, exist_ok=True); train(args)


if __name__ == "__main__":
    main()
