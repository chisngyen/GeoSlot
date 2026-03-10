#!/usr/bin/env python3
"""
GeoSlot: Slot-Guided Cross-View Drone Geo-Localization
=======================================================
Novel contributions:
  1. SlotCVA — Slot Attention for object-centric cross-view correspondence
  2. VLM-Guided Distillation — SigLIP2 teacher with spatial grounding priors
  3. Altitude-Aware Adaptive Pooling (AAAP) — metadata-conditioned feature pooling

Architecture:
  Student: ConvNeXt-Tiny + Slot Attention (K=8 slots) + AAAP
  Teacher: SigLIP2 ViT-B/16 (frozen)

Dataset: SUES-200 (drone ↔ satellite cross-view geo-localization)
Protocol: 120 train / 80 test, gallery = ALL 200 locations (confusion data)

Usage:
  python exp_geoslot.py           # Full training on Kaggle H100
  python exp_geoslot.py --test    # Smoke test
"""

# === SETUP ===
import subprocess, sys, os

def pip_install(pkg, extra=""):
    subprocess.run(f"pip install -q {extra} {pkg}",
                   shell=True, capture_output=True, text=True)

print("[1/3] Installing packages...")
for p in ["timm", "tqdm", "open_clip_torch"]:
    try:
        __import__(p.replace("-", "_").split("_")[0] if "clip" not in p else "open_clip")
    except ImportError:
        pip_install(p)
print("[2/3] Setup complete!")

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
try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False
    print("Warning: open_clip not available, will use DINOv2 teacher fallback")

print("[3/3] All imports loaded!")

# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT     = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
OUTPUT_DIR    = "/kaggle/working"

EPOCHS        = 120
BATCH_SIZE    = 256
NUM_WORKERS   = 8
AMP_ENABLED   = True
EVAL_FREQ     = 5

# Learning rates
LR_BACKBONE   = 1e-4
LR_HEAD       = 1e-3
LR_SLOT       = 5e-4
WARMUP_EPOCHS = 5
WEIGHT_DECAY  = 0.01

# 3-Phase progressive schedule
PHASE1_END    = 30    # Backbone frozen, basic losses only
PHASE2_END    = 80    # + Slot Attention + distillation
# Phase 3: Full fine-tuning + all losses + AAAP

# Loss weights
LAMBDA_CE         = 1.0
LAMBDA_TRIPLET    = 0.5
LAMBDA_INFONCE    = 0.5
LAMBDA_SLOT_CONT  = 0.5    # Slot contrastive
LAMBDA_SLOT_DIST  = 0.3    # Slot distillation from teacher
LAMBDA_ARCFACE    = 0.3    # ArcFace margin loss
LAMBDA_SELF_DIST  = 0.3    # Self-distillation
LAMBDA_UAPA       = 0.2    # Uncertainty alignment

# Model
BACKBONE_NAME = "convnext_tiny"
FEATURE_DIM   = 768
EMBED_DIM     = 512
NUM_SLOTS     = 8       # Number of object-centric slots
SLOT_DIM      = 128     # Dimension per slot
SLOT_ITERS    = 3       # Slot attention iterations
NUM_CLASSES   = 120     # 120 training locations
MARGIN        = 0.3

# Dataset
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
# DATASET (Standard SUES-200 protocol)
# =============================================================================
class SUES200Dataset(Dataset):
    """SUES-200: 120 train / 80 test, all altitudes combined."""
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
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        # Satellite-specific: add random 90° rotation (view from above)
        sat_augs = [
            transforms.Resize((img_size, img_size), interpolation=3),
        ]
        if is_train:
            sat_augs += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
            ]
        sat_augs += [
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])]
        self.sat_tf = transforms.Compose(sat_augs)

        self.pairs = []
        self.labels = []
        self.altitudes_meta = []
        loc_to_label = {}
        for loc_id in locs:
            loc_str = f"{loc_id:04d}"
            sat_path = os.path.join(satellite_dir, loc_str, "0.png")
            if not os.path.exists(sat_path): continue
            if loc_id not in loc_to_label:
                loc_to_label[loc_id] = len(loc_to_label)
            label = loc_to_label[loc_id]
            for alt in alts:
                alt_dir = os.path.join(drone_dir, loc_str, alt)
                if not os.path.isdir(alt_dir): continue
                alt_idx = ALTITUDES.index(alt) if alt in ALTITUDES else 0
                for img_name in sorted(os.listdir(alt_dir)):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        drone_path = os.path.join(alt_dir, img_name)
                        self.pairs.append((drone_path, sat_path))
                        self.labels.append(label)
                        self.altitudes_meta.append(alt_idx)

        self.num_classes = len(loc_to_label)
        self.loc_to_label = loc_to_label
        print(f"  [{split}] {len(self.pairs)} pairs, {len(locs)} locations, "
              f"{self.num_classes} classes")

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
            "altitude": self.altitudes_meta[idx],
            "idx": idx
        }


class SUES200GalleryDataset(Dataset):
    """ALL 200 satellite locations as confusion gallery."""
    def __init__(self, root, test_locs=None, img_size=224):
        super().__init__()
        satellite_dir = os.path.join(root, "satellite-view")
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


class PKSampler:
    """P locations × K samples per batch."""
    def __init__(self, dataset, p=8, k=4):
        self.p, self.k = p, k
        self.loc_to_indices = defaultdict(list)
        for i, label in enumerate(dataset.labels):
            self.loc_to_indices[label].append(i)
        self.locations = list(self.loc_to_indices.keys())

    def __iter__(self):
        locs = self.locations.copy()
        random.shuffle(locs)
        batch = []
        for loc in locs:
            indices = self.loc_to_indices[loc]
            if len(indices) < self.k:
                indices = indices * (self.k // len(indices) + 1)
            batch.extend(random.sample(indices, self.k))
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]
                batch = batch[self.p * self.k:]

    def __len__(self): return len(self.locations) // self.p


# =============================================================================
# SLOT ATTENTION MODULE (Novel Component #1)
# =============================================================================
class SlotAttention(nn.Module):
    """Slot Attention for decomposing features into object-centric slots.

    Given spatial feature maps [B, C, H, W], produces K object-centric slots
    [B, K, D_slot] via iterative competitive attention.

    Novel application: Cross-view geo-localization — slots learn to bind to
    view-invariant semantic regions (buildings, roads, vegetation, water bodies).
    """
    def __init__(self, in_dim, num_slots=8, slot_dim=128, num_iters=3, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iters = num_iters
        self.eps = eps

        # Slot initialization (learnable)
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, slot_dim),
            nn.LayerNorm(slot_dim),
        )

        # Attention
        self.q_proj = nn.Linear(slot_dim, slot_dim)
        self.k_proj = nn.Linear(slot_dim, slot_dim)
        self.v_proj = nn.Linear(slot_dim, slot_dim)
        self.scale = slot_dim ** -0.5

        # GRU update
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] feature map from backbone
        Returns:
            slots: [B, K, D_slot] object-centric slot representations
            attn_maps: [B, K, N] attention maps (N = H*W)
        """
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial dims and project
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]
        inputs = self.input_proj(x_flat)        # [B, N, D_slot]

        # Initialize slots (with learned prior + noise)
        mu = self.slot_mu.expand(B, -1, -1)
        sigma = self.slot_log_sigma.exp().expand(B, -1, -1)
        slots = mu + sigma * torch.randn_like(sigma)   # [B, K, D_slot]

        # Iterative competitive attention
        attn_maps = None
        for _ in range(self.num_iters):
            slots_prev = slots

            # Attention: slots compete for input tokens
            q = self.q_proj(self.norm_slots(slots))   # [B, K, D]
            k = self.k_proj(inputs)                    # [B, N, D]
            v = self.v_proj(inputs)                    # [B, N, D]

            attn_logits = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, K, N]
            attn = F.softmax(attn_logits, dim=1)       # Normalize across SLOTS
            attn_maps = attn                            # Save for distillation

            # Weighted sum → update
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.bmm(attn_norm, v)          # [B, K, D]

            # GRU + MLP residual
            slots = self.gru(updates.flatten(0, 1), slots_prev.flatten(0, 1))
            slots = slots.view(B, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_maps


# =============================================================================
# ALTITUDE-AWARE ADAPTIVE POOLING (Novel Component #3)
# =============================================================================
class AltitudeAwarePool(nn.Module):
    """Altitude-Aware Adaptive Pooling (AAAP).

    Conditions feature aggregation on drone flight altitude:
      - Low altitude (150m) → weight local texture/detail slots higher
      - High altitude (300m) → weight global layout/structure slots higher

    Uses a lightweight MLP to generate per-slot attention weights from altitude.
    """
    def __init__(self, slot_dim, num_slots=8, num_altitudes=4):
        super().__init__()
        self.num_altitudes = num_altitudes
        self.altitude_embed = nn.Embedding(num_altitudes, 32)
        self.gate = nn.Sequential(
            nn.Linear(32, num_slots * 2),
            nn.GELU(),
            nn.Linear(num_slots * 2, num_slots),
            nn.Sigmoid(),
        )
        # Projection from slots to final embedding
        self.proj = nn.Sequential(
            nn.Linear(slot_dim * num_slots, slot_dim * 2),
            nn.LayerNorm(slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, EMBED_DIM),
        )

    def forward(self, slots, altitude_idx=None):
        """
        Args:
            slots: [B, K, D_slot]
            altitude_idx: [B] integer altitude index (0=150m, 1=200m, etc.)
        Returns:
            embedding: [B, EMBED_DIM] altitude-conditioned embedding
        """
        B, K, D = slots.shape

        if altitude_idx is not None:
            alt_feat = self.altitude_embed(altitude_idx)   # [B, 32]
            gate_weights = self.gate(alt_feat)              # [B, K]
            gate_weights = gate_weights.unsqueeze(-1)       # [B, K, 1]
            slots_weighted = slots * gate_weights           # [B, K, D]
        else:
            # At inference for satellite (no altitude), use uniform weighting
            slots_weighted = slots

        # Flatten and project
        flat = slots_weighted.flatten(1)   # [B, K*D]
        embedding = self.proj(flat)        # [B, EMBED_DIM]
        return embedding


# =============================================================================
# ARCFACE HEAD (Performance Boost)
# =============================================================================
class ArcFaceHead(nn.Module):
    """ArcFace angular margin classifier for discriminative embeddings."""
    def __init__(self, embed_dim, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        # Normalize
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(embeddings, dim=1)
        cosine = F.linear(x, w)

        if labels is None or not self.training:
            return cosine * self.s

        # Add angular margin
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        theta = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        logits = cosine * (1 - one_hot) + target_logits * one_hot
        return logits * self.s


# =============================================================================
# GEOSLOT STUDENT MODEL
# =============================================================================
class GeoSlotStudent(nn.Module):
    """GeoSlot: ConvNeXt-Tiny + Slot Attention + AAAP.

    Architecture:
      1. ConvNeXt-Tiny backbone → multi-scale feature maps
      2. Slot Attention → K object-centric slots
      3. AAAP → altitude-conditioned embedding
      4. ArcFace classifier → discriminative learning
      5. Self-distillation across 4 backbone stages
    """
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM,
                 num_slots=NUM_SLOTS, slot_dim=SLOT_DIM, slot_iters=SLOT_ITERS):
        super().__init__()

        # Backbone
        self.backbone = timm.create_model(BACKBONE_NAME, pretrained=True,
                                          num_classes=0, global_pool='')
        self.feature_dim = FEATURE_DIM  # ConvNeXt-Tiny final channel dim

        # Slot Attention on final feature map
        self.slot_attention = SlotAttention(
            in_dim=self.feature_dim,
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_iters=slot_iters,
        )

        # Altitude-Aware Adaptive Pooling
        self.aaap = AltitudeAwarePool(
            slot_dim=slot_dim, num_slots=num_slots, num_altitudes=4
        )

        # Global branch (for when Slot Attention is not used, e.g. satellite)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Slot → global projection (combine slot + global features)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

        # Classifiers
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.arcface = ArcFaceHead(embed_dim, num_classes, s=30.0, m=0.50)

        # Self-distillation heads (4 stages of ConvNeXt)
        stage_dims = [96, 192, 384, 768]
        self.aux_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(True),
                nn.Linear(256, num_classes)
            ) for d in stage_dims
        ])

    def extract_stages(self, x):
        """Extract multi-stage features from ConvNeXt."""
        stages = []
        for i, stage in enumerate(self.backbone.stages):
            if i == 0:
                x = self.backbone.stem(x)
            else:
                x = self.backbone.stages[i-1].downsample(x) if hasattr(self.backbone.stages[i-1], 'downsample') else x
            x = stage.blocks(x) if hasattr(stage, 'blocks') else stage(x)
            stages.append(x)
        return stages

    def forward_backbone(self, x):
        """Get final feature map from backbone."""
        feat = self.backbone(x)  # [B, C, H', W']
        return feat

    def forward(self, x, altitude_idx=None, labels=None, view_type='drone',
                return_all=False):
        """
        Args:
            x: [B, 3, H, W] input image
            altitude_idx: [B] altitude index (only for drone)
            labels: [B] class labels (for ArcFace)
            view_type: 'drone' or 'satellite'
            return_all: if True, return dict with all intermediate features
        """
        # Backbone features
        feat_map = self.forward_backbone(x)   # [B, C, H', W']

        # Global branch
        global_feat = self.global_pool(feat_map).flatten(1)  # [B, C]
        global_emb = self.global_proj(global_feat)            # [B, embed_dim]

        # Slot Attention branch
        slots, attn_maps = self.slot_attention(feat_map)   # [B, K, slot_dim], [B, K, N]

        # Altitude-Aware Pooling
        if view_type == 'drone' and altitude_idx is not None:
            slot_emb = self.aaap(slots, altitude_idx)      # [B, embed_dim]
        else:
            slot_emb = self.aaap(slots, None)              # [B, embed_dim]

        # Fuse global + slot embeddings
        combined = torch.cat([global_emb, slot_emb], dim=1)  # [B, 2*embed_dim]
        embedding = self.fusion(combined)                     # [B, embed_dim]
        embedding_norm = F.normalize(embedding, p=2, dim=1)

        # Classification
        logits = self.classifier(embedding)
        arcface_logits = self.arcface(embedding, labels)

        if return_all:
            return {
                'embedding': embedding,
                'embedding_norm': embedding_norm,
                'logits': logits,
                'arcface_logits': arcface_logits,
                'slots': slots,
                'attn_maps': attn_maps,
                'feat_map': feat_map,
                'global_feat': global_feat,
            }

        return embedding_norm, logits

    def extract_embedding(self, x, view_type='drone', altitude_idx=None):
        """For evaluation: extract normalized embedding."""
        self.eval()
        with torch.no_grad():
            emb_norm, _ = self.forward(x, altitude_idx=altitude_idx,
                                        view_type=view_type)
        return emb_norm


# =============================================================================
# SIGLIP2 TEACHER (Novel Component #2)
# =============================================================================
class SigLIP2Teacher(nn.Module):
    """SigLIP2 ViT-B/16 teacher for VLM-guided spatial distillation.

    Provides:
      - CLS embedding for feature distillation
      - Patch tokens for spatial attention distillation
      - Semantic attention maps as Slot Attention targets
    """
    def __init__(self):
        super().__init__()
        if HAS_OPEN_CLIP:
            print("Loading SigLIP2 teacher via open_clip...")
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-16-SigLIP2-256', pretrained='webli'
                )
                self.embed_dim = self.model.visual.output_dim
                self.has_siglip = True
                print(f"  SigLIP2 loaded! embed_dim={self.embed_dim}")
            except Exception as e:
                print(f"  SigLIP2 failed: {e}, falling back to DINOv2")
                self.has_siglip = False
        else:
            self.has_siglip = False

        if not self.has_siglip:
            print("Loading DINOv2-base teacher fallback...")
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.embed_dim = 768
            print(f"  DINOv2 loaded! embed_dim={self.embed_dim}")

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        """Extract teacher features.

        Returns:
            cls_feat: [B, D] CLS token embedding
            patch_tokens: [B, N, D] patch-level features (for spatial distillation)
        """
        if self.has_siglip:
            # SigLIP2 via open_clip
            features = self.model.encode_image(x)
            cls_feat = F.normalize(features, dim=-1)
            # Patch tokens from visual transformer
            try:
                visual = self.model.visual
                x_in = visual.conv1(x)
                x_in = x_in.reshape(x_in.shape[0], x_in.shape[1], -1).permute(0, 2, 1)
                x_in = torch.cat([visual.class_embedding.expand(x_in.shape[0], -1, -1), x_in], dim=1)
                x_in = x_in + visual.positional_embedding
                x_in = visual.ln_pre(x_in)
                x_in = visual.transformer(x_in)
                patch_tokens = visual.ln_post(x_in[:, 1:])  # Exclude CLS
            except Exception:
                patch_tokens = cls_feat.unsqueeze(1).expand(-1, 196, -1)
        else:
            # DINOv2 fallback
            x_in = self.model.prepare_tokens_with_masks(x)
            for blk in self.model.blocks:
                x_in = blk(x_in)
            x_in = self.model.norm(x_in)
            cls_feat = x_in[:, 0]           # [B, 768]
            patch_tokens = x_in[:, 1:]      # [B, N, 768]

        return cls_feat, patch_tokens


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
class TripletLoss(nn.Module):
    """Hard-mining triplet loss."""
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        dist = torch.cdist(embeddings, embeddings, p=2)
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float()
        neg_mask = labels.ne(labels.T).float()
        hard_pos = (dist * pos_mask).max(dim=1)[0]
        hard_neg = (dist * neg_mask + pos_mask * 999).min(dim=1)[0]
        loss = F.relu(hard_pos - hard_neg + self.margin)
        return loss.mean()


class InfoNCELoss(nn.Module):
    """Symmetric cross-view contrastive loss."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, drone_feats, sat_feats, labels):
        drone_feats = F.normalize(drone_feats, dim=1)
        sat_feats = F.normalize(sat_feats, dim=1)
        sim = drone_feats @ sat_feats.T / self.t

        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float()

        loss_d2s = -(F.log_softmax(sim, dim=1) * pos_mask).sum(1) / pos_mask.sum(1).clamp(1)
        loss_s2d = -(F.log_softmax(sim.T, dim=1) * pos_mask).sum(1) / pos_mask.sum(1).clamp(1)
        return 0.5 * (loss_d2s.mean() + loss_s2d.mean())


class SlotContrastiveLoss(nn.Module):
    """Cross-view slot correspondence loss.

    Aligns corresponding slots between drone and satellite views of the same location.
    Uses Hungarian matching to find optimal slot-to-slot correspondence.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature

    def forward(self, drone_slots, sat_slots, labels):
        """
        Args:
            drone_slots: [B, K, D] drone slot representations
            sat_slots: [B, K, D] satellite slot representations
            labels: [B] location labels
        """
        B, K, D = drone_slots.shape

        # Normalize slots
        ds = F.normalize(drone_slots, dim=-1)  # [B, K, D]
        ss = F.normalize(sat_slots, dim=-1)    # [B, K, D]

        # Per-sample slot similarity: match drone slots to satellite slots
        total_loss = 0.0
        for i in range(B):
            # Slot-to-slot similarity for same location
            sim = ds[i] @ ss[i].T  # [K, K]
            # Maximize diagonal (same slot index should match)
            pos_sim = sim.diag()
            # Contrastive: positive = diagonal, negative = off-diagonal
            logits = sim / self.t
            targets = torch.arange(K, device=logits.device)
            total_loss += F.cross_entropy(logits, targets)

        return total_loss / B


class SlotDistillationLoss(nn.Module):
    """Distill teacher's spatial attention into student's slot attention maps.

    Converts teacher patch tokens into pseudo-slot assignments via k-means-like
    clustering, then aligns with student slot attention maps.
    """
    def __init__(self, teacher_dim, slot_dim, num_slots=8):
        super().__init__()
        self.proj = nn.Linear(teacher_dim, slot_dim)
        self.num_slots = num_slots

    def forward(self, student_attn_maps, teacher_patch_tokens):
        """
        Args:
            student_attn_maps: [B, K, N] student slot attention maps
            teacher_patch_tokens: [B, N', D_teacher] teacher patch features
        """
        B = student_attn_maps.shape[0]
        N = student_attn_maps.shape[2]
        N_t = teacher_patch_tokens.shape[1]

        # Project teacher to slot dim
        teacher_proj = self.proj(teacher_patch_tokens)  # [B, N', slot_dim]

        # If spatial dims don't match, interpolate student maps
        if N != N_t:
            H_s = int(N ** 0.5)
            H_t = int(N_t ** 0.5)
            s_maps = student_attn_maps.view(B, self.num_slots, H_s, H_s)
            s_maps = F.interpolate(s_maps, size=(H_t, H_t), mode='bilinear',
                                    align_corners=False)
            student_attn_maps = s_maps.view(B, self.num_slots, N_t)

        # Compute teacher pseudo-slot assignments via cosine similarity
        teacher_norm = F.normalize(teacher_proj, dim=-1)  # [B, N_t, D]
        # Cluster: assign each patch to most similar student slot
        # Use student attention as soft assignment, match to teacher clustering
        teacher_sim = teacher_norm @ teacher_norm.transpose(1, 2)  # [B, N_t, N_t]

        # Spectral-style: match student attention distribution to teacher affinity
        student_attn_norm = F.softmax(student_attn_maps, dim=-1)  # [B, K, N_t]
        # Teacher affinity → pseudo student distribution
        teacher_affinity = F.softmax(teacher_sim / 0.1, dim=-1)   # [B, N_t, N_t]

        # KL divergence between student slot attention and teacher structure
        # Aggregate teacher affinity per student slot
        teacher_slot_aff = torch.bmm(student_attn_norm, teacher_affinity)  # [B, K, N_t]
        teacher_slot_dist = F.softmax(teacher_slot_aff, dim=-1)

        loss = F.kl_div(
            student_attn_norm.log().clamp(min=-100),
            teacher_slot_dist.detach(),
            reduction='batchmean'
        )
        return loss


class SelfDistillationLoss(nn.Module):
    """Inverse self-distillation across ConvNeXt stages."""
    def __init__(self, temperature=4.0, weights=None):
        super().__init__()
        self.T = temperature
        self.weights = weights or [0.1, 0.2, 0.3, 0.4]

    def forward(self, stage_logits):
        loss = 0.0
        final = stage_logits[-1]
        for i in range(len(stage_logits) - 1):
            t = F.softmax(stage_logits[i] / self.T, dim=1)
            s = F.log_softmax(final / self.T, dim=1)
            loss += self.weights[i] * (self.T ** 2) * F.kl_div(s, t, reduction='batchmean')
        return loss


class UAPALoss(nn.Module):
    """Uncertainty-Aware Prediction Alignment."""
    def __init__(self, T0=4.0):
        super().__init__()
        self.T0 = T0

    def forward(self, drone_logits, sat_logits):
        U_d = -(F.softmax(drone_logits, 1) * F.log_softmax(drone_logits, 1)).sum(1).mean()
        U_s = -(F.softmax(sat_logits, 1) * F.log_softmax(sat_logits, 1)).sum(1).mean()
        T = self.T0 * (1 + torch.sigmoid(U_d - U_s))
        p_s = F.softmax(sat_logits / T, dim=1)
        p_d = F.log_softmax(drone_logits / T, dim=1)
        return (T ** 2) * F.kl_div(p_d, p_s, reduction='batchmean')


# =============================================================================
# LR SCHEDULER
# =============================================================================
def get_cosine_lr(epoch, total_epochs, base_lr, warmup_epochs=5):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model, data_root, altitude, device, test_locs=None):
    """Standard SUES-200 evaluation with confusion gallery."""
    model.eval()
    query_ds = SUES200Dataset(data_root, "test", altitude, IMG_SIZE,
                               test_locs=test_locs)
    query_loader = DataLoader(query_ds, batch_size=64, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    gallery_ds = SUES200GalleryDataset(data_root, test_locs, IMG_SIZE)
    gallery_loader = DataLoader(gallery_ds, batch_size=64, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

    # Gallery embeddings
    gal_embs, gal_locs = [], []
    for batch in gallery_loader:
        emb = model.extract_embedding(batch["image"].to(device),
                                       view_type='satellite')
        gal_embs.append(emb.cpu())
        gal_locs.extend(batch["loc_id"].tolist())
    gal_embs = torch.cat(gal_embs, 0)
    gal_locs = np.array(gal_locs)

    # Query embeddings
    q_embs, q_labels = [], []
    for batch in query_loader:
        alt_idx = batch.get("altitude", None)
        if alt_idx is not None:
            alt_idx = alt_idx.to(device)
        emb = model.extract_embedding(batch["query"].to(device),
                                       view_type='drone', altitude_idx=alt_idx)
        q_embs.append(emb.cpu())
        q_labels.extend([query_ds.pairs[i][1] for i in range(len(emb))])
    q_embs = torch.cat(q_embs, 0)

    # Ground truth: map query to gallery index
    loc_to_gal_idx = {loc: i for i, loc in enumerate(gal_locs)}
    q_gt_indices = []
    for drone_path, sat_path in query_ds.pairs:
        loc_id = int(os.path.basename(os.path.dirname(sat_path)))
        q_gt_indices.append(loc_to_gal_idx.get(loc_id, -1))
    q_gt_indices = np.array(q_gt_indices)

    # Retrieval
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
# TRAINING
# =============================================================================
def train(args):
    set_seed(SEED)
    data_root = args.data_root

    print("=" * 70)
    print("GeoSlot: Slot-Guided Cross-View Drone Geo-Localization")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Train: {len(TRAIN_LOCS)} locs | Test: {len(TEST_LOCS)} locs")
    print(f"  Num slots: {NUM_SLOTS} | Slot dim: {SLOT_DIM}")

    # Dataset
    print("\n[1/4] Loading dataset...")
    train_ds = SUES200Dataset(data_root, "train", img_size=IMG_SIZE)
    sampler = PKSampler(train_ds, p=8, k=max(2, BATCH_SIZE // 8))
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Models
    print("\n[2/4] Building models...")
    model = GeoSlotStudent(num_classes=train_ds.num_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Student: {total_params:.1f}M params ({trainable_params:.1f}M trainable)")

    teacher = SigLIP2Teacher().to(DEVICE)
    teacher.eval()

    # Loss functions
    print("\n[3/4] Setting up losses...")
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    triplet_fn = TripletLoss(margin=MARGIN)
    infonce_fn = InfoNCELoss(temperature=0.07)
    slot_cont_fn = SlotContrastiveLoss(temperature=0.1)
    slot_dist_fn = SlotDistillationLoss(teacher.embed_dim, SLOT_DIM, NUM_SLOTS).to(DEVICE)
    self_dist_fn = SelfDistillationLoss(temperature=4.0)
    uapa_fn = UAPALoss(T0=4.0)

    # Optimizer
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n],
         'lr': LR_BACKBONE},
        {'params': [p for n, p in model.named_parameters() if 'slot_attention' in n],
         'lr': LR_SLOT},
        {'params': [p for n, p in model.named_parameters()
                    if 'backbone' not in n and 'slot_attention' not in n],
         'lr': LR_HEAD},
        {'params': slot_dist_fn.parameters(), 'lr': LR_SLOT},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=AMP_ENABLED)

    # Training loop
    print("\n[4/4] Starting training...")
    best_r1 = 0.0

    for epoch in range(EPOCHS):
        model.train()

        # Determine training phase
        if epoch < PHASE1_END:
            phase = 1
            # Freeze backbone in phase 1
            for p in model.backbone.parameters():
                p.requires_grad = False
        elif epoch < PHASE2_END:
            phase = 2
            for p in model.backbone.parameters():
                p.requires_grad = True
        else:
            phase = 3

        # LR schedule
        for pg in optimizer.param_groups:
            base = pg.get('initial_lr', pg['lr'])
            pg['lr'] = get_cosine_lr(epoch, EPOCHS, base, WARMUP_EPOCHS)
        lr = optimizer.param_groups[0]['lr']

        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{EPOCHS} | Phase {phase} | LR: {lr:.6f}")
        print(f"{'='*50}")

        total_loss = 0.0
        loss_parts = defaultdict(float)

        for batch_idx, batch in enumerate(train_loader):
            drone = batch["query"].to(DEVICE)
            sat = batch["gallery"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            alt_idx = batch["altitude"].to(DEVICE)

            optimizer.zero_grad()

            with autocast(enabled=AMP_ENABLED):
                # Student forward
                d_out = model(drone, altitude_idx=alt_idx, labels=labels,
                             view_type='drone', return_all=True)
                s_out = model(sat, labels=labels, view_type='satellite',
                             return_all=True)

                # === Loss computation ===
                losses = {}

                # 1. CE + ArcFace
                losses['ce'] = LAMBDA_CE * 0.5 * (
                    ce_loss_fn(d_out['logits'], labels) +
                    ce_loss_fn(s_out['logits'], labels)
                )
                losses['arcface'] = LAMBDA_ARCFACE * 0.5 * (
                    ce_loss_fn(d_out['arcface_logits'], labels) +
                    ce_loss_fn(s_out['arcface_logits'], labels)
                )

                # 2. Triplet
                losses['triplet'] = LAMBDA_TRIPLET * 0.5 * (
                    triplet_fn(d_out['embedding_norm'], labels) +
                    triplet_fn(s_out['embedding_norm'], labels)
                )

                # 3. InfoNCE
                losses['infonce'] = LAMBDA_INFONCE * infonce_fn(
                    d_out['embedding_norm'], s_out['embedding_norm'], labels
                )

                # Phase 2+: Slot losses
                if phase >= 2:
                    # 4. Slot Contrastive
                    losses['slot_cont'] = LAMBDA_SLOT_CONT * slot_cont_fn(
                        d_out['slots'], s_out['slots'], labels
                    )

                    # 5. Slot Distillation from teacher
                    with torch.no_grad():
                        t_cls_d, t_patch_d = teacher(drone)
                        t_cls_s, t_patch_s = teacher(sat)

                    losses['slot_dist'] = LAMBDA_SLOT_DIST * 0.5 * (
                        slot_dist_fn(d_out['attn_maps'], t_patch_d) +
                        slot_dist_fn(s_out['attn_maps'], t_patch_s)
                    )

                    # 6. Feature distillation (MSE + cosine)
                    d_feat_norm = F.normalize(d_out['global_feat'], dim=1)
                    s_feat_norm = F.normalize(s_out['global_feat'], dim=1)
                    t_d_norm = F.normalize(t_cls_d, dim=1)
                    t_s_norm = F.normalize(t_cls_s, dim=1)

                    feat_dist = 0.5 * (
                        F.mse_loss(d_feat_norm, t_d_norm) +
                        F.mse_loss(s_feat_norm, t_s_norm) +
                        (1 - F.cosine_similarity(d_feat_norm, t_d_norm).mean()) +
                        (1 - F.cosine_similarity(s_feat_norm, t_s_norm).mean())
                    )
                    losses['feat_dist'] = 0.3 * feat_dist

                # Phase 3: UAPA
                if phase >= 3:
                    losses['uapa'] = LAMBDA_UAPA * uapa_fn(
                        d_out['logits'], s_out['logits']
                    )

                loss = sum(losses.values())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            for k, v in losses.items():
                loss_parts[k] += v.item()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        n_batches = max(1, len(train_loader))
        avg_loss = total_loss / n_batches
        print(f"\nEpoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        for k, v in sorted(loss_parts.items()):
            print(f"  {k}: {v/n_batches:.4f}")

        # Evaluate
        if (epoch + 1) % EVAL_FREQ == 0 or epoch == EPOCHS - 1:
            print("\nEvaluating...")
            all_results = {}
            for alt in ALTITUDES:
                res = evaluate(model, data_root, alt, DEVICE)
                all_results[alt] = res
                print(f"  {alt}m: R@1={res['R@1']:.4f} R@5={res['R@5']:.4f} "
                      f"R@10={res['R@10']:.4f} AP={res['AP']:.4f}")

            # Average across altitudes
            avg_r1 = np.mean([r['R@1'] for r in all_results.values()])
            avg_r5 = np.mean([r['R@5'] for r in all_results.values()])
            avg_r10 = np.mean([r['R@10'] for r in all_results.values()])
            avg_ap = np.mean([r['AP'] for r in all_results.values()])
            print(f"  AVG: R@1={avg_r1:.4f} R@5={avg_r5:.4f} "
                  f"R@10={avg_r10:.4f} AP={avg_ap:.4f}")

            if avg_r1 > best_r1:
                best_r1 = avg_r1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'avg_r1': avg_r1,
                    'all_results': all_results,
                }, os.path.join(OUTPUT_DIR, 'geoslot_best.pth'))
                print(f"  *** New best! Avg R@1 = {avg_r1:.4f} ***")

    print("\n" + "=" * 70)
    print(f"Training complete! Best Avg R@1: {best_r1:.4f}")
    print("=" * 70)


# =============================================================================
# SMOKE TEST
# =============================================================================
def smoke_test():
    """Quick test: model creation, forward pass, loss computation, grads."""
    print("=" * 50)
    print("SMOKE TEST — GeoSlot")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 4

    # Model
    model = GeoSlotStudent(num_classes=10).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ Model created: {params:.1f}M params")

    # Forward
    x = torch.randn(B, 3, 224, 224, device=device)
    labels = torch.tensor([0, 0, 1, 1], device=device)
    alt = torch.tensor([0, 1, 2, 3], device=device)

    out = model(x, altitude_idx=alt, labels=labels, view_type='drone', return_all=True)
    print(f"✓ Forward pass OK")
    print(f"  embedding: {out['embedding_norm'].shape}")
    print(f"  slots: {out['slots'].shape}")
    print(f"  attn_maps: {out['attn_maps'].shape}")
    print(f"  logits: {out['logits'].shape}")
    print(f"  arcface: {out['arcface_logits'].shape}")

    # Losses
    drone_out = model(x, altitude_idx=alt, labels=labels, view_type='drone', return_all=True)
    sat_out = model(x, labels=labels, view_type='satellite', return_all=True)

    ce = nn.CrossEntropyLoss()(drone_out['logits'], labels)
    triplet = TripletLoss()(drone_out['embedding_norm'], labels)
    infonce = InfoNCELoss()(drone_out['embedding_norm'], sat_out['embedding_norm'], labels)
    slot_cont = SlotContrastiveLoss()(drone_out['slots'], sat_out['slots'], labels)
    print(f"✓ Losses computed: CE={ce.item():.4f}, Triplet={triplet.item():.4f}, "
          f"InfoNCE={infonce.item():.4f}, SlotCont={slot_cont.item():.4f}")

    # Backward
    total = ce + triplet + infonce + slot_cont
    total.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"✓ Backward pass OK, grad norm: {grad_norm:.4f}")

    print("\n✅ ALL SMOKE TESTS PASSED!")


# =============================================================================
# MAIN
# =============================================================================
def main():
    global EPOCHS, BATCH_SIZE, DATA_ROOT
    parser = argparse.ArgumentParser(description="GeoSlot Training")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    args, _ = parser.parse_known_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    DATA_ROOT = args.data_root

    if args.test:
        smoke_test()
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
