#!/usr/bin/env python3
"""
GeoCVCA: Cross-View Cross-Attention for Drone Geo-Localization
===============================================================
Novel contributions:
  1. CVCAM (Cross-View Cross-Attention Module) — Iterative cross-attention
     between drone & satellite feature tokens for explicit correspondence
  2. MHSAM (Multi-Head Spatial Attention Module) — 3 attention heads with
     different kernel sizes for multi-scale spatial refinement
  3. Geometry-Aware Positional Encoding — 2D sinusoidal positional embeddings
     injected before cross-attention for spatial grounding

Inspired by: AttenGeo (ICLR 2025) — SOTA cross-view object geo-localization

Architecture:
  Student: ConvNeXt-Tiny + CVCAM + MHSAM + DINOv2 distillation
  Teacher: DINOv2-Base (frozen)

Dataset: SUES-200 (drone ↔ satellite cross-view geo-localization)
Protocol: 120 train / 80 test, gallery = ALL 200 locations (confusion data)

Usage:
  python exp_geocvca.py           # Full training on Kaggle H100
  python exp_geocvca.py --test    # Smoke test
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
from typing import Dict, Any
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
import timm

print("[OK] All imports loaded!")

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

LR            = 0.001
WARMUP_EPOCHS = 5

NUM_CLASSES   = 120
EMBED_DIM     = 768
IMG_SIZE      = 224
MARGIN        = 0.3

# CVCAM config
CVCAM_DEPTH   = 3      # Number of cross-attention iterations
CVCAM_HEADS   = 8      # Number of attention heads
CVCAM_DIM     = 256    # Cross-attention dimension

# MHSAM config
MHSAM_KERNELS = [3, 5, 7]  # Multi-scale kernel sizes

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
# DATASET (same structure as baseline)
# =============================================================================
class SUES200Dataset(Dataset):
    def __init__(self, root, mode="train", altitudes=None, transform=None,
                 train_locs=None, test_locs=None):
        self.root = root
        self.mode = mode
        self.altitudes = altitudes or ALTITUDES
        self.transform = transform
        drone_dir = os.path.join(root, "drone-view")
        satellite_dir = os.path.join(root, "satellite-view")
        if train_locs is None: train_locs = TRAIN_LOCS
        if test_locs is None: test_locs = TEST_LOCS
        loc_ids = train_locs if mode == "train" else test_locs
        self.locations = [f"{loc:04d}" for loc in loc_ids]
        self.location_to_idx = {loc: idx for idx, loc in enumerate(self.locations)}
        self.samples = []
        self.drone_by_location = defaultdict(list)
        for loc in self.locations:
            loc_idx = self.location_to_idx[loc]
            sat_path = os.path.join(satellite_dir, loc, "0.png")
            if not os.path.exists(sat_path): continue
            for alt in self.altitudes:
                alt_dir = os.path.join(drone_dir, loc, alt)
                if not os.path.isdir(alt_dir): continue
                for img_name in sorted(os.listdir(alt_dir)):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        drone_path = os.path.join(alt_dir, img_name)
                        self.samples.append((drone_path, sat_path, loc_idx, alt))
                        self.drone_by_location[loc_idx].append(len(self.samples) - 1)
        print(f"[{mode}] Loaded {len(self.samples)} samples from {len(self.locations)} locations")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        drone_path, sat_path, loc_idx, altitude = self.samples[idx]
        drone_img = Image.open(drone_path).convert('RGB')
        sat_img = Image.open(sat_path).convert('RGB')
        if self.transform:
            drone_img = self.transform(drone_img)
            sat_img = self.transform(sat_img)
        return {'drone': drone_img, 'satellite': sat_img,
                'label': loc_idx, 'altitude': int(altitude)}


class PKSampler:
    def __init__(self, dataset, p=8, k=4):
        self.p, self.k = p, k
        self.locations = list(dataset.drone_by_location.keys())
        self.dataset = dataset
    def __iter__(self):
        locations = self.locations.copy(); random.shuffle(locations)
        batch = []
        for loc in locations:
            indices = self.dataset.drone_by_location[loc]
            if len(indices) < self.k:
                indices = indices * (self.k // len(indices) + 1)
            batch.extend(random.sample(indices, self.k))
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]
                batch = batch[self.p * self.k:]
    def __len__(self): return len(self.locations) // self.p


def get_transforms(mode="train"):
    if mode == "train":
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(0.5),
            T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


# =============================================================================
# CONVNEXT-TINY BACKBONE (from baseline)
# =============================================================================
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, dp=None):
        super().__init__()
        self.dp = dp
    def forward(self, x):
        return drop_path(x, self.dp, self.training)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path_rate=0., layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
    def forward(self, x):
        shortcut = x; x = self.dwconv(x); x = x.permute(0, 2, 3, 1)
        x = self.norm(x); x = self.pwconv1(x); x = self.act(x); x = self.pwconv2(x)
        if self.gamma is not None: x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXtTiny(nn.Module):
    def __init__(self, in_chans=3, depths=[3,3,9,3], dims=[96,192,384,768],
                 drop_path_rate=0., layer_scale_init=1e-6):
        super().__init__()
        self.dims = dims
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], 4, 4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], 2, 2)))
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock(dims[i], dp_rates[cur+j], layer_scale_init)
                for j in range(depths[i])]))
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        stage_outputs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            stage_outputs.append(x)
        final_feat = x.mean([-2, -1])
        final_feat = self.norm(final_feat)
        return final_feat, stage_outputs


def load_convnext_pretrained(model):
    url = "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth"
    try:
        ckpt = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)
        sd = {k: v for k, v in ckpt["model"].items() if not k.startswith('head')}
        model.load_state_dict(sd, strict=False)
        print("Loaded ConvNeXt-Tiny pretrained (ImageNet-22K)")
    except Exception as e:
        print(f"Could not load pretrained: {e}")
    return model


# =============================================================================
# NOVEL COMPONENT 1: GEOMETRY-AWARE POSITIONAL ENCODING
# =============================================================================
class GeometryAwarePositionalEncoding(nn.Module):
    """2D sinusoidal positional encoding for spatial feature maps.

    Unlike 1D positional encoding used in NLP, this generates position-aware
    embeddings that encode both row and column position, critical for
    maintaining spatial relationships during cross-view attention.
    """
    def __init__(self, d_model, max_h=14, max_w=14):
        super().__init__()
        pe = torch.zeros(d_model, max_h, max_w)
        d_half = d_model // 2

        # Row encoding
        pos_h = torch.arange(0, max_h).unsqueeze(1).float()
        div_h = torch.exp(torch.arange(0, d_half, 2).float() * -(math.log(10000.0) / d_half))
        pe[0:d_half:2, :, :] = torch.sin(pos_h * div_h).T.unsqueeze(2).expand(-1, -1, max_w)
        pe[1:d_half:2, :, :] = torch.cos(pos_h * div_h).T.unsqueeze(2).expand(-1, -1, max_w)

        # Column encoding
        pos_w = torch.arange(0, max_w).unsqueeze(1).float()
        div_w = torch.exp(torch.arange(0, d_half, 2).float() * -(math.log(10000.0) / d_half))
        pe[d_half::2, :, :] = torch.sin(pos_w * div_w).T.unsqueeze(1).expand(-1, max_h, -1)
        pe[d_half+1::2, :, :] = torch.cos(pos_w * div_w).T.unsqueeze(1).expand(-1, max_h, -1)

        self.register_buffer('pe', pe.unsqueeze(0))  # [1, D, H, W]

    def forward(self, x):
        """Add positional encoding to feature map [B, D, H, W]."""
        B, D, H, W = x.shape
        return x + self.pe[:, :D, :H, :W]


# =============================================================================
# NOVEL COMPONENT 2: CVCAM (Cross-View Cross-Attention Module)
# =============================================================================
class CrossAttentionBlock(nn.Module):
    """Single cross-attention block.

    Query attends to key-value from the other view, establishing
    implicit spatial correspondences between drone and satellite features.
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query, context):
        """
        Args:
            query: [B, N_q, D] — tokens from one view
            context: [B, N_c, D] — tokens from the other view
        Returns: [B, N_q, D] — cross-attended query
        """
        # Cross-attention: query attends to context
        attn_out, _ = self.attn(query, context, context)
        query = self.norm1(query + attn_out)

        # FFN
        query = self.norm2(query + self.ffn(query))

        return query


class CVCAM(nn.Module):
    """Cross-View Cross-Attention Module.

    Facilitates deep, iterative information exchange between drone and satellite
    features through N rounds of cross-attention. Each round establishes
    implicit spatial correspondences between the drastic viewpoint changes.

    Novel: Bidirectional — drone-to-satellite AND satellite-to-drone attention
    are computed in each iteration, allowing mutual contextual refinement.
    """
    def __init__(self, d_model, num_heads=8, depth=3, dropout=0.1):
        super().__init__()
        self.depth = depth

        # Drone attends to satellite
        self.d2s_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, dropout)
            for _ in range(depth)
        ])

        # Satellite attends to drone
        self.s2d_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, dropout)
            for _ in range(depth)
        ])

        # Projection layers
        self.drone_proj = nn.Linear(d_model, d_model)
        self.sat_proj = nn.Linear(d_model, d_model)

    def forward(self, drone_tokens, sat_tokens):
        """
        Args:
            drone_tokens: [B, N, D] — flattened drone feature map
            sat_tokens: [B, N, D] — flattened satellite feature map
        Returns:
            drone_refined: [B, N, D]
            sat_refined: [B, N, D]
        """
        d = self.drone_proj(drone_tokens)
        s = self.sat_proj(sat_tokens)

        for i in range(self.depth):
            # Bidirectional cross-attention
            d_new = self.d2s_blocks[i](d, s)
            s_new = self.s2d_blocks[i](s, d)
            d, s = d_new, s_new

        return d, s


# =============================================================================
# NOVEL COMPONENT 3: MHSAM (Multi-Head Spatial Attention Module)
# =============================================================================
class MHSAM(nn.Module):
    """Multi-Head Spatial Attention Module.

    Deploys 3 attention heads with Conv/Deconv kernels of varying sizes to
    extract multi-scale spatial features from the cross-attended representations.

    Each head captures spatial structure at a different receptive field,
    enabling scale-invariant feature refinement.
    """
    def __init__(self, d_model, kernels=[3, 5, 7]):
        super().__init__()
        self.num_heads = len(kernels)

        # Conv heads with different kernel sizes
        self.conv_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=k, padding=k//2, groups=d_model),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
                nn.Conv2d(d_model, d_model, kernel_size=1),
                nn.BatchNorm2d(d_model),
            ) for k in kernels
        ])

        # Deconv (transposed conv) heads for upscaling receptive field
        self.deconv_heads = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(d_model, d_model, kernel_size=k, padding=k//2,
                                   output_padding=0, groups=d_model),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
            ) for k in kernels
        ])

        # Spatial attention gates
        self.attention_gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(inplace=True),
                nn.Linear(d_model // 4, d_model),
                nn.Sigmoid(),
            ) for _ in kernels
        ])

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(d_model * self.num_heads, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

    def forward(self, x):
        """
        Args: x: [B, C, H, W] — cross-attended feature map
        Returns: [B, C, H, W] — multi-scale refined features
        """
        head_outputs = []
        for i in range(self.num_heads):
            # Conv branch
            conv_out = self.conv_heads[i](x)

            # Deconv branch
            deconv_out = self.deconv_heads[i](x)
            # Ensure same spatial size
            if deconv_out.shape[-2:] != x.shape[-2:]:
                deconv_out = F.interpolate(deconv_out, size=x.shape[-2:],
                                           mode='bilinear', align_corners=False)

            # Combine with spatial attention gate
            combined = conv_out + deconv_out
            gate = self.attention_gates[i](combined).unsqueeze(-1).unsqueeze(-1)
            head_outputs.append(combined * gate)

        # Concatenate and fuse
        multi_scale = torch.cat(head_outputs, dim=1)  # [B, C*num_heads, H, W]
        return self.fusion(multi_scale) + x  # Residual connection


# =============================================================================
# GEOCVCA STUDENT MODEL
# =============================================================================
class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes))
    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


class GeoCVCAStudent(nn.Module):
    """GeoCVCA = ConvNeXt-Tiny + CVCAM + MHSAM.

    Pipeline:
      1. ConvNeXt backbone extracts multi-scale features
      2. Geometry-Aware Positional Encoding adds 2D spatial awareness
      3. CVCAM performs iterative cross-view cross-attention
      4. MHSAM refines with multi-scale spatial attention
      5. Global + cross-attended embeddings for retrieval

    NOTE: During inference (single-view), CVCAM uses self-attention fallback
    since we only have ONE view available (query or gallery separately).
    """
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = ConvNeXtTiny(drop_path_rate=0.1)
        self.backbone = load_convnext_pretrained(self.backbone)
        self.dims = [96, 192, 384, 768]

        # Stage-wise heads
        self.aux_heads = nn.ModuleList([
            ClassificationHead(dim, num_classes) for dim in self.dims])

        # Project to cross-attention dim
        self.feat_proj = nn.Sequential(
            nn.Conv2d(768, CVCAM_DIM, 1),
            nn.BatchNorm2d(CVCAM_DIM),
            nn.GELU())

        # Geometry-Aware Positional Encoding
        self.pos_enc = GeometryAwarePositionalEncoding(CVCAM_DIM, 7, 7)

        # *** NOVEL: CVCAM ***
        self.cvcam = CVCAM(
            d_model=CVCAM_DIM, num_heads=CVCAM_HEADS,
            depth=CVCAM_DEPTH, dropout=0.1)

        # *** NOVEL: MHSAM ***
        self.mhsam = MHSAM(CVCAM_DIM, MHSAM_KERNELS)

        # Embeddings
        self.cross_embed = nn.Sequential(
            nn.Linear(CVCAM_DIM, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True))

        self.global_embed = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True))

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim))

        self.classifier = nn.Linear(embed_dim, num_classes)

    def _extract_features(self, x):
        """Extract backbone features and project to CVCAM dim."""
        final_feat, stage_outputs = self.backbone(x)
        feat_map = stage_outputs[-1]  # [B, 768, H, W]

        # Project to cross-attention dim
        projected = self.feat_proj(feat_map)  # [B, CVCAM_DIM, H, W]

        # Add positional encoding
        projected = self.pos_enc(projected)

        return final_feat, stage_outputs, projected

    def forward_pair(self, drone_x, sat_x, labels=None):
        """Forward pass with PAIRED drone-satellite input (training).

        Both views are available → use full cross-attention via CVCAM.
        """
        # Extract features
        d_feat, d_stages, d_proj = self._extract_features(drone_x)
        s_feat, s_stages, s_proj = self._extract_features(sat_x)

        # Stage logits
        d_stage_logits = [h(f) for h, f in zip(self.aux_heads, d_stages)]
        s_stage_logits = [h(f) for h, f in zip(self.aux_heads, s_stages)]

        # Flatten for cross-attention: [B, C, H, W] → [B, H*W, C]
        B, C, H, W = d_proj.shape
        d_tokens = d_proj.flatten(2).transpose(1, 2)  # [B, N, C]
        s_tokens = s_proj.flatten(2).transpose(1, 2)

        # *** CVCAM: Cross-view cross-attention ***
        d_cross, s_cross = self.cvcam(d_tokens, s_tokens)

        # Reshape back to 2D for MHSAM
        d_cross_2d = d_cross.transpose(1, 2).view(B, C, H, W)
        s_cross_2d = s_cross.transpose(1, 2).view(B, C, H, W)

        # *** MHSAM: Multi-scale spatial refinement ***
        d_refined = self.mhsam(d_cross_2d)
        s_refined = self.mhsam(s_cross_2d)

        # Pool cross-attended features
        d_cross_pool = d_refined.mean([-2, -1])  # [B, CVCAM_DIM]
        s_cross_pool = s_refined.mean([-2, -1])

        # Cross-attended embeddings
        d_cross_emb = self.cross_embed(d_cross_pool)
        s_cross_emb = self.cross_embed(s_cross_pool)

        # Global embeddings
        d_global_emb = self.global_embed(d_feat)
        s_global_emb = self.global_embed(s_feat)

        # Fuse global + cross-attended
        d_combined = self.fusion(torch.cat([d_global_emb, d_cross_emb], 1))
        s_combined = self.fusion(torch.cat([s_global_emb, s_cross_emb], 1))

        d_norm = F.normalize(d_combined, p=2, dim=1)
        s_norm = F.normalize(s_combined, p=2, dim=1)

        d_logits = self.classifier(d_combined)
        s_logits = self.classifier(s_combined)

        return {
            'drone': {
                'embedding_normed': d_norm,
                'logits': d_logits,
                'stage_logits': d_stage_logits,
                'final_feature': d_feat,
                'cross_tokens': d_cross,
            },
            'sat': {
                'embedding_normed': s_norm,
                'logits': s_logits,
                'stage_logits': s_stage_logits,
                'final_feature': s_feat,
                'cross_tokens': s_cross,
            },
        }

    def forward(self, x, return_all=False):
        """Single-view forward (for inference/evaluation).

        No cross-view partner available → CVCAM uses self-attention fallback.
        """
        final_feat, stage_outputs, projected = self._extract_features(x)

        # Stage logits
        stage_logits = [h(f) for h, f in zip(self.aux_heads, stage_outputs)]

        # Self-attention fallback (query = key = value from same view)
        B, C, H, W = projected.shape
        tokens = projected.flatten(2).transpose(1, 2)

        # Use CVCAM with same input for both views
        cross_tokens, _ = self.cvcam(tokens, tokens)
        cross_2d = cross_tokens.transpose(1, 2).view(B, C, H, W)
        refined = self.mhsam(cross_2d)
        cross_pool = refined.mean([-2, -1])

        cross_emb = self.cross_embed(cross_pool)
        global_emb = self.global_embed(final_feat)
        combined = self.fusion(torch.cat([global_emb, cross_emb], 1))
        combined_norm = F.normalize(combined, p=2, dim=1)
        logits = self.classifier(combined)

        if return_all:
            return {
                'embedding_normed': combined_norm,
                'logits': logits,
                'stage_logits': stage_logits,
                'final_feature': final_feat,
            }
        return combined_norm, logits

    def extract_embedding(self, x):
        self.eval()
        with torch.no_grad():
            emb, _ = self.forward(x)
        return emb


# =============================================================================
# DINOV2 TEACHER
# =============================================================================
class DINOv2Teacher(nn.Module):
    def __init__(self, num_trainable_blocks=2):
        super().__init__()
        print("Loading DINOv2-base teacher...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        for p in self.parameters(): p.requires_grad = False
        for blk in self.model.blocks[-num_trainable_blocks:]:
            for p in blk.parameters(): p.requires_grad = True
        print(f"  DINOv2 loaded! Last {num_trainable_blocks} blocks trainable.")

    @torch.no_grad()
    def forward(self, x):
        tokens = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks: tokens = blk(tokens)
        tokens = self.model.norm(tokens)
        return tokens[:, 0]


# =============================================================================
# LOSSES
# =============================================================================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    def forward(self, emb, labels):
        dist = torch.cdist(emb, emb, p=2)
        lab = labels.view(-1, 1)
        pos = lab.eq(lab.T).float(); neg = lab.ne(lab.T).float()
        hp = (dist * pos).max(1)[0]
        hn = (dist * neg + pos * 999).min(1)[0]
        return F.relu(hp - hn + self.margin).mean()


class SymmetricInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature
    def forward(self, d, s, labels):
        d = F.normalize(d, 1); s = F.normalize(s, 1)
        sim = d @ s.T / self.t
        lab = labels.view(-1, 1); pm = lab.eq(lab.T).float()
        l1 = -(F.log_softmax(sim, 1) * pm).sum(1) / pm.sum(1).clamp(1)
        l2 = -(F.log_softmax(sim.T, 1) * pm).sum(1) / pm.sum(1).clamp(1)
        return 0.5 * (l1.mean() + l2.mean())


class SelfDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super().__init__()
        self.T = temperature
    def forward(self, stage_logits):
        loss = 0.0; final = stage_logits[-1]
        weights = [0.1, 0.2, 0.3, 0.4]
        for i in range(len(stage_logits) - 1):
            pt = F.softmax(stage_logits[i] / self.T, 1)
            ps = F.log_softmax(final / self.T, 1)
            loss += weights[i] * (self.T ** 2) * F.kl_div(ps, pt, reduction='batchmean')
        return loss


class UAPALoss(nn.Module):
    def __init__(self, T0=4.0):
        super().__init__()
        self.T0 = T0
    def forward(self, dl, sl):
        Ud = -(F.softmax(dl, 1) * F.log_softmax(dl, 1)).sum(1).mean()
        Us = -(F.softmax(sl, 1) * F.log_softmax(sl, 1)).sum(1).mean()
        T = self.T0 * (1 + torch.sigmoid(Ud - Us))
        return (T**2) * F.kl_div(F.log_softmax(dl/T, 1), F.softmax(sl/T, 1), reduction='batchmean')


class CrossViewCorrespondenceLoss(nn.Module):
    """Loss that encourages CVCAM to learn meaningful correspondences.

    Measures whether cross-attended tokens from matching locations are
    more similar than tokens from non-matching locations.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature

    def forward(self, drone_tokens, sat_tokens, labels):
        """
        Args:
            drone_tokens: [B, N, D] cross-attended drone tokens
            sat_tokens: [B, N, D] cross-attended satellite tokens
            labels: [B] location labels
        """
        # Pool tokens to get per-sample descriptors
        d = F.normalize(drone_tokens.mean(1), dim=1)  # [B, D]
        s = F.normalize(sat_tokens.mean(1), dim=1)
        sim = d @ s.T / self.t
        lab = labels.view(-1, 1); pm = lab.eq(lab.T).float()
        loss = -(F.log_softmax(sim, 1) * pm).sum(1) / pm.sum(1).clamp(1)
        return loss.mean()


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate(model, test_dataset, device, data_root=None):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    all_feats, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            feats, _ = model(batch['drone'].to(device))
            all_feats.append(feats.cpu()); all_labels.append(batch['label'])
    all_feats = torch.cat(all_feats); all_labels = torch.cat(all_labels)

    transform = get_transforms("test")
    root = data_root or test_dataset.root
    sat_dir = os.path.join(root, "satellite-view")
    all_locs = [f"{loc:04d}" for loc in TRAIN_LOCS + TEST_LOCS]

    sat_feats, sat_labels, gnames = [], [], []
    for loc in all_locs:
        sp = os.path.join(sat_dir, loc, "0.png")
        if not os.path.exists(sp): continue
        img = Image.open(sp).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f, _ = model(t)
        sat_feats.append(f.cpu())
        sat_labels.append(test_dataset.location_to_idx[loc]
                          if loc in test_dataset.location_to_idx else -1 - len(gnames))
        gnames.append(loc)

    sat_feats = torch.cat(sat_feats); sat_labels = torch.tensor(sat_labels)
    sim = torch.mm(all_feats, sat_feats.T)
    _, indices = sim.sort(1, descending=True)
    N = all_feats.size(0)
    r1 = r5 = r10 = 0; ap_total = 0.0
    for i in range(N):
        ranked = sat_labels[indices[i]]
        correct = torch.where(ranked == all_labels[i])[0]
        if len(correct) == 0: continue
        fc = correct[0].item()
        if fc < 1: r1 += 1
        if fc < 5: r5 += 1
        if fc < 10: r10 += 1
        ap_total += sum((j+1)/(p.item()+1) for j, p in enumerate(correct)) / len(correct)
    recall = {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100}
    ap = ap_total / N * 100
    print(f"  Gallery: {len(sat_feats)} sats, Queries: {N}")
    return recall, ap


# =============================================================================
# TRAINING
# =============================================================================
def train(args):
    set_seed(SEED)
    print("=" * 70)
    print("GeoCVCA: Cross-View Cross-Attention for Drone Geo-Localization")
    print("=" * 70)

    train_ds = SUES200Dataset(args.data_root, "train", transform=get_transforms("train"))
    test_ds = SUES200Dataset(args.data_root, "test", transform=get_transforms("test"))
    num_classes = len(TRAIN_LOCS)
    K = max(2, BATCH_SIZE // 8)
    sampler = PKSampler(train_ds, p=8, k=K)
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = GeoCVCAStudent(num_classes=num_classes).to(DEVICE)
    prms = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Student: {prms:.1f}M params")

    try:
        teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except Exception as e:
        print(f"  Could not load DINOv2: {e}"); teacher = None

    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    trip_fn = TripletLoss(MARGIN)
    nce_fn = SymmetricInfoNCELoss(0.07)
    sd_fn = SelfDistillationLoss(4.0)
    uapa_fn = UAPALoss(4.0)
    cvca_fn = CrossViewCorrespondenceLoss(0.1)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler(enabled=AMP_ENABLED)
    best_r1 = 0.0

    for epoch in range(EPOCHS):
        model.train()

        # LR schedule
        if epoch < WARMUP_EPOCHS:
            lr = LR * (epoch + 1) / WARMUP_EPOCHS
        else:
            progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
            lr = 1e-6 + 0.5 * (LR - 1e-6) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups: pg['lr'] = lr

        tl = 0.0; lp = defaultdict(float); nb = 0

        for bi, batch in enumerate(train_loader):
            drone = batch['drone'].to(DEVICE)
            sat = batch['satellite'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            with autocast(enabled=AMP_ENABLED):
                # *** PAIRED forward pass with full cross-attention ***
                out = model.forward_pair(drone, sat, labels)
                do, so = out['drone'], out['sat']

                L = {}
                # CE
                ce = ce_fn(do['logits'], labels) + ce_fn(so['logits'], labels)
                for sl in do['stage_logits']: ce += 0.25 * ce_fn(sl, labels)
                for sl in so['stage_logits']: ce += 0.25 * ce_fn(sl, labels)
                L['ce'] = ce

                # Triplet
                L['trip'] = trip_fn(do['embedding_normed'], labels) + \
                            trip_fn(so['embedding_normed'], labels)

                # InfoNCE
                L['nce'] = nce_fn(do['embedding_normed'], so['embedding_normed'], labels)

                # Self-distillation
                L['sd'] = 0.5 * (sd_fn(do['stage_logits']) + sd_fn(so['stage_logits']))

                # UAPA
                L['uapa'] = 0.2 * uapa_fn(do['logits'], so['logits'])

                # *** NOVEL: Cross-view correspondence loss ***
                L['cvca'] = 0.5 * cvca_fn(do['cross_tokens'], so['cross_tokens'], labels)

                # Cross-distillation
                if teacher is not None:
                    with torch.no_grad():
                        td = teacher(drone); ts = teacher(sat)
                    df = F.normalize(do['final_feature'], 1)
                    sf = F.normalize(so['final_feature'], 1)
                    tdn = F.normalize(td, 1); tsn = F.normalize(ts, 1)
                    L['cdist'] = 0.3 * (F.mse_loss(df, tdn) + F.mse_loss(sf, tsn) +
                                        (1 - F.cosine_similarity(df, tdn).mean()) +
                                        (1 - F.cosine_similarity(sf, tsn).mean()))

                total = sum(L.values())

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()

            tl += total.item(); nb += 1
            for k, v in L.items(): lp[k] += v.item()
            if bi % 10 == 0: print(f"  B{bi}/{len(train_loader)} L={total.item():.4f}")

        nb = max(1, nb)
        print(f"\nEp {epoch+1}/{EPOCHS} LR={lr:.6f} AvgL={tl/nb:.4f}")
        for k, v in sorted(lp.items()): print(f"  {k}: {v/nb:.4f}")

        if (epoch + 1) % EVAL_FREQ == 0 or epoch == EPOCHS - 1:
            print("\nEvaluating...")
            recall, ap = evaluate(model, test_ds, DEVICE)
            print(f"  R@1: {recall['R@1']:.2f}%  R@5: {recall['R@5']:.2f}%  "
                  f"R@10: {recall['R@10']:.2f}%  AP: {ap:.2f}%")
            if recall['R@1'] > best_r1:
                best_r1 = recall['R@1']
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'r1': best_r1, 'ap': ap},
                           os.path.join(OUTPUT_DIR, 'geocvca_best.pth'))
                print(f"  *** New best R@1={best_r1:.2f}% ***")

    print(f"\nDone! Best R@1={best_r1:.2f}%")


# =============================================================================
# SMOKE TEST
# =============================================================================
def smoke_test():
    print("=" * 50); print("SMOKE TEST — GeoCVCA"); print("=" * 50)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = GeoCVCAStudent(10).to(dev)
    print(f"✓ Model: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")

    d = torch.randn(4, 3, 224, 224, device=dev)
    s = torch.randn(4, 3, 224, 224, device=dev)
    lab = torch.tensor([0, 0, 1, 1], device=dev)

    # Test paired forward
    out = m.forward_pair(d, s)
    print(f"✓ Paired: drone_emb={out['drone']['embedding_normed'].shape}, "
          f"cross_tokens={out['drone']['cross_tokens'].shape}")

    # Test single forward
    emb, logits = m(d)
    print(f"✓ Single: emb={emb.shape}, logits={logits.shape}")

    # Backward
    loss = nn.CrossEntropyLoss()(out['drone']['logits'], lab)
    loss.backward()
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
    EPOCHS = args.epochs; BATCH_SIZE = args.batch_size; DATA_ROOT = args.data_root
    args.data_root = DATA_ROOT
    if args.test: smoke_test(); return
    os.makedirs(OUTPUT_DIR, exist_ok=True); train(args)


if __name__ == "__main__":
    main()
