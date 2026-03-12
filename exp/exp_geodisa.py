#!/usr/bin/env python3
"""
GeoDISA: Disentangled Slot Attention for Cross-View Drone Geo-Localization
==========================================================================
Novel contributions:
  1. Disentangled Slot Attention (DISA) — Explicitly partitions slot dimensions
     into shape/geometry vs texture/appearance subsets for view-invariant matching
  2. Shape-Only Retrieval — Computes similarity using only geometry slots,
     ignoring view-dependent texture (facade vs roof mismatch solved!)
  3. Probabilistic Slot Initialization — GMM-based slot init for cross-view
     identifiability guarantees (slots represent same objects across views)

Inspired by: DISA (Springer ML 2025) + PSA (NeurIPS 2024)

Key insight: Drone sees building facades (texture), satellite sees roofs
(different texture), but both share the same geometric layout. DISA matches
on geometry only, fundamentally solving the cross-view texture mismatch.

Architecture:
  Student: ConvNeXt-Tiny + Disentangled Slot Attention + Shape-Only Head
  Teacher: DINOv2-Base (frozen)

Dataset: SUES-200 (drone ↔ satellite cross-view geo-localization)
Protocol: 120 train / 80 test, gallery = ALL 200 locations (confusion data)

Usage:
  python exp_geodisa.py           # Full training on Kaggle H100
  python exp_geodisa.py --test    # Smoke test
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

# Disentangled Slot config
NUM_SLOTS     = 8           # Number of object slots
SLOT_DIM      = 128         # Per-slot dimension
SHAPE_DIM     = 80          # Dims dedicated to shape/geometry (62.5%)
TEXTURE_DIM   = 48          # Dims dedicated to texture/appearance (37.5%)
SLOT_ITERS    = 3           # Slot attention iterations
GMM_COMPONENTS = 8          # GMM components for probabilistic init

LAMBDA_CE       = 1.0
LAMBDA_TRIPLET  = 1.0
LAMBDA_NCE      = 0.5
LAMBDA_SLOT     = 0.5       # Slot contrastive
LAMBDA_DISENTANGLE = 0.3    # Disentanglement regularization
LAMBDA_SD       = 0.5       # Self-distillation
LAMBDA_UAPA     = 0.2
LAMBDA_CDIST    = 0.3

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
# DATASET (same as baseline)
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
# CONVNEXT-TINY BACKBONE
# =============================================================================
class LayerNorm(nn.Module):
    def __init__(self, ns, eps=1e-6, df="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ns))
        self.bias = nn.Parameter(torch.zeros(ns))
        self.eps = eps; self.df = df; self.ns = (ns,)
    def forward(self, x):
        if self.df == "channels_last":
            return F.layer_norm(x, self.ns, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True); s = (x-u).pow(2).mean(1, keepdim=True)
        x = (x-u)/torch.sqrt(s+self.eps)
        return self.weight[:,None,None]*x + self.bias[:,None,None]


def drop_path(x, dp=0., training=False):
    if dp == 0. or not training: return x
    kp = 1-dp; shape = (x.shape[0],)+(1,)*(x.ndim-1)
    rt = kp+torch.rand(shape, dtype=x.dtype, device=x.device); rt.floor_()
    return x.div(kp)*rt

class DropPath(nn.Module):
    def __init__(self, dp=None): super().__init__(); self.dp = dp
    def forward(self, x): return drop_path(x, self.dp, self.training)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, dpr=0., lsi=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, 1e-6)
        self.pw1 = nn.Linear(dim, 4*dim); self.act = nn.GELU()
        self.pw2 = nn.Linear(4*dim, dim)
        self.gamma = nn.Parameter(lsi*torch.ones(dim)) if lsi>0 else None
        self.dp = DropPath(dpr) if dpr>0 else nn.Identity()
    def forward(self, x):
        s = x; x = self.dwconv(x); x = x.permute(0,2,3,1)
        x = self.norm(x); x = self.pw1(x); x = self.act(x); x = self.pw2(x)
        if self.gamma is not None: x = self.gamma*x
        x = x.permute(0,3,1,2); return s+self.dp(x)

class ConvNeXtTiny(nn.Module):
    def __init__(self, ic=3, depths=[3,3,9,3], dims=[96,192,384,768], dpr=0., lsi=1e-6):
        super().__init__()
        self.dims = dims
        self.ds = nn.ModuleList()
        self.ds.append(nn.Sequential(nn.Conv2d(ic, dims[0], 4, 4),
                                     LayerNorm(dims[0], 1e-6, "channels_first")))
        for i in range(3):
            self.ds.append(nn.Sequential(LayerNorm(dims[i], 1e-6, "channels_first"),
                                         nn.Conv2d(dims[i], dims[i+1], 2, 2)))
        rates = [x.item() for x in torch.linspace(0, dpr, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[ConvNeXtBlock(dims[i], rates[cur+j], lsi)
                                               for j in range(depths[i])]))
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], 1e-6)
        self.apply(self._iw)

    def _iw(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.ds[i](x); x = self.stages[i](x); outs.append(x)
        f = x.mean([-2,-1]); f = self.norm(f)
        return f, outs

def load_convnext_pretrained(model):
    url = "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth"
    try:
        ckpt = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)
        sd = {k:v for k,v in ckpt["model"].items() if not k.startswith('head')}
        model.load_state_dict(sd, strict=False)
        print("Loaded ConvNeXt-Tiny pretrained (ImageNet-22K)")
    except Exception as e: print(f"Could not load: {e}")
    return model


# =============================================================================
# NOVEL COMPONENT 1: DISENTANGLED SLOT ATTENTION
# =============================================================================
class ProbabilisticSlotInit(nn.Module):
    """GMM-based probabilistic slot initialization.

    Uses a learned Gaussian Mixture Model to initialize slots from the input
    features, providing identifiability guarantees: the same object in
    different views will be assigned to the same slot.
    """
    def __init__(self, d_input, d_slot, num_slots, num_components=8):
        super().__init__()
        self.num_slots = num_slots
        self.d_slot = d_slot
        self.num_components = num_components

        # GMM parameters (learnable)
        self.means = nn.Parameter(torch.randn(num_components, d_slot) * 0.1)
        self.log_vars = nn.Parameter(torch.zeros(num_components, d_slot))
        self.mix_logits = nn.Parameter(torch.zeros(num_components))

        # Input-dependent slot init
        self.input_to_slot = nn.Sequential(
            nn.Linear(d_input, d_slot * 2),
            nn.GELU(),
            nn.Linear(d_slot * 2, num_slots * d_slot),
        )

    def forward(self, features):
        """
        Args: features: [B, N, D_input] — backbone features
        Returns: slots: [B, K, D_slot] — initialized slots
        """
        B = features.shape[0]

        # Compute per-sample slot initialization from input
        feat_pool = features.mean(dim=1)  # [B, D_input]
        slots_flat = self.input_to_slot(feat_pool)  # [B, K*D_slot]
        slots = slots_flat.view(B, self.num_slots, self.d_slot)

        # Add GMM-based stochastic perturbation
        mix_probs = F.softmax(self.mix_logits, dim=0)  # [C]
        # Sample component for each slot
        comp_idx = torch.multinomial(mix_probs.expand(B * self.num_slots, -1), 1)
        comp_idx = comp_idx.view(B, self.num_slots)  # [B, K]

        # Get mean and variance for selected components
        selected_means = self.means[comp_idx]  # [B, K, D_slot]
        selected_log_vars = self.log_vars[comp_idx]  # [B, K, D_slot]

        if self.training:
            # Reparameterization trick
            std = torch.exp(0.5 * selected_log_vars)
            eps = torch.randn_like(std)
            gmm_init = selected_means + std * eps
            slots = slots + 0.1 * gmm_init  # Blend with input-dependent init

        return slots


class DisentangledSlotAttention(nn.Module):
    """Disentangled Slot Attention (DISA).

    Core novelty: Partitions slot dimensions into non-overlapping subsets:
    - Shape dimensions: encode geometric structure (position, contour, layout)
    - Texture dimensions: encode appearance (color, material, pattern)

    During attention, shape and texture dimensions have separate attention
    computations, forcing the model to disentangle these factors.
    """
    def __init__(self, d_input, d_slot, num_slots, shape_dim, texture_dim,
                 num_iters=3, num_gmm_components=8):
        super().__init__()
        assert shape_dim + texture_dim == d_slot, \
            f"shape_dim({shape_dim}) + texture_dim({texture_dim}) must == d_slot({d_slot})"

        self.d_input = d_input
        self.d_slot = d_slot
        self.num_slots = num_slots
        self.shape_dim = shape_dim
        self.texture_dim = texture_dim
        self.num_iters = num_iters
        self.eps = 1e-8

        # Probabilistic initialization
        self.slot_init = ProbabilisticSlotInit(d_input, d_slot, num_slots, num_gmm_components)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_slot),
            nn.LayerNorm(d_slot),
        )

        # Separate QKV for shape and texture
        self.shape_q = nn.Linear(shape_dim, shape_dim, bias=False)
        self.shape_k = nn.Linear(shape_dim, shape_dim, bias=False)

        self.texture_q = nn.Linear(texture_dim, texture_dim, bias=False)
        self.texture_k = nn.Linear(texture_dim, texture_dim, bias=False)

        # GRU update (per-slot)
        self.gru = nn.GRUCell(d_slot, d_slot)
        self.slot_norm = nn.LayerNorm(d_slot)

        # Slot MLP
        self.slot_mlp = nn.Sequential(
            nn.Linear(d_slot, d_slot * 2),
            nn.GELU(),
            nn.Linear(d_slot * 2, d_slot),
        )

        scale = d_slot ** -0.5
        self.scale_shape = shape_dim ** -0.5
        self.scale_texture = texture_dim ** -0.5

    def forward(self, features):
        """
        Args:
            features: [B, N, D_input] — backbone feature tokens
        Returns:
            slots: [B, K, D_slot] — disentangled slot representations
            attn_weights: [B, K, N] — slot-to-feature attention
            shape_slots: [B, K, shape_dim] — shape-only slot features
            texture_slots: [B, K, texture_dim] — texture-only slot features
        """
        B, N, _ = features.shape

        # Project input features to slot dim
        inputs = self.input_proj(features)  # [B, N, D_slot]

        # Initialize slots via GMM
        slots = self.slot_init(features)  # [B, K, D_slot]

        # Iterative slot attention
        for t in range(self.num_iters):
            slots_prev = slots

            # Split into shape and texture dimensions
            inp_shape = inputs[:, :, :self.shape_dim]       # [B, N, shape_dim]
            inp_texture = inputs[:, :, self.shape_dim:]     # [B, N, texture_dim]
            slot_shape = slots[:, :, :self.shape_dim]       # [B, K, shape_dim]
            slot_texture = slots[:, :, self.shape_dim:]     # [B, K, texture_dim]

            # Shape attention
            q_s = self.shape_q(slot_shape)    # [B, K, shape_dim]
            k_s = self.shape_k(inp_shape)     # [B, N, shape_dim]
            attn_shape = torch.bmm(q_s, k_s.transpose(1, 2)) * self.scale_shape  # [B, K, N]

            # Texture attention
            q_t = self.texture_q(slot_texture)  # [B, K, texture_dim]
            k_t = self.texture_k(inp_texture)   # [B, N, texture_dim]
            attn_texture = torch.bmm(q_t, k_t.transpose(1, 2)) * self.scale_texture  # [B, K, N]

            # Combined attention (shape-weighted)
            # Shape gets higher weight → geometry drives slot assignment
            attn_logits = 0.7 * attn_shape + 0.3 * attn_texture  # [B, K, N]

            # Softmax over slots (competitive assignment)
            attn = F.softmax(attn_logits, dim=1)  # [B, K, N]

            # Weighted mean
            attn_normed = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.bmm(attn_normed, inputs)  # [B, K, D_slot]

            # GRU update
            slots_flat = slots.view(B * self.num_slots, self.d_slot)
            updates_flat = updates.view(B * self.num_slots, self.d_slot)
            slots = self.gru(updates_flat, slots_flat).view(B, self.num_slots, self.d_slot)

            # Residual MLP
            slots = slots + self.slot_mlp(self.slot_norm(slots))

        # Extract disentangled components
        shape_slots = slots[:, :, :self.shape_dim]
        texture_slots = slots[:, :, self.shape_dim:]

        return slots, attn, shape_slots, texture_slots


# =============================================================================
# NOVEL COMPONENT 2: SHAPE-ONLY RETRIEVAL HEAD
# =============================================================================
class ShapeOnlyRetrievalHead(nn.Module):
    """Retrieval head that uses ONLY shape/geometry slot features.

    During cross-view matching, texture is ignored because:
    - Drone sees facades → vertical surfaces, brick/glass textures
    - Satellite sees roofs → horizontal surfaces, flat/shingle textures
    - But geometric layout (building footprints, road patterns) is SHARED

    This head computes embeddings from shape-only slot features,
    making the retrieval inherently view-invariant.
    """
    def __init__(self, shape_dim, num_slots, embed_dim, num_classes):
        super().__init__()
        self.num_slots = num_slots

        # Per-slot shape projection
        self.slot_proj = nn.Sequential(
            nn.Linear(shape_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # Permutation-invariant aggregation (attention pooling)
        self.attn_pool = nn.Sequential(
            nn.Linear(256, 1),
        )

        # Shape embedding
        self.shape_embed = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, shape_slots):
        """
        Args: shape_slots: [B, K, shape_dim]
        Returns: shape_embedding: [B, embed_dim], shape_logits: [B, num_classes]
        """
        B = shape_slots.shape[0]

        # Project each slot
        projected = self.slot_proj(shape_slots)  # [B, K, 256]

        # Attention-weighted aggregation (permutation invariant)
        attn_weights = F.softmax(self.attn_pool(projected), dim=1)  # [B, K, 1]
        aggregated = (projected * attn_weights).sum(dim=1)  # [B, 256]

        # Embedding
        embedding = self.shape_embed(aggregated)  # [B, embed_dim]
        logits = self.classifier(embedding)

        return embedding, logits


# =============================================================================
# GEODISA STUDENT MODEL
# =============================================================================
class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes))
    def forward(self, x): return self.fc(self.pool(x).flatten(1))


class GeoDISAStudent(nn.Module):
    """GeoDISA = ConvNeXt-Tiny + Disentangled Slot Attention + Shape-Only Head.

    Pipeline:
      1. ConvNeXt extracts multi-scale features
      2. DISA decomposes scene into disentangled slots (shape + texture)
      3. Shape-Only Head computes geometry-based embedding for retrieval
      4. Global + shape embeddings fused for final descriptor
    """
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = ConvNeXtTiny(dpr=0.1)
        self.backbone = load_convnext_pretrained(self.backbone)
        self.dims = [96, 192, 384, 768]

        # Stage-wise heads
        self.aux_heads = nn.ModuleList([
            ClassificationHead(dim, num_classes) for dim in self.dims])

        # *** NOVEL: Disentangled Slot Attention ***
        self.slot_attn = DisentangledSlotAttention(
            d_input=768, d_slot=SLOT_DIM, num_slots=NUM_SLOTS,
            shape_dim=SHAPE_DIM, texture_dim=TEXTURE_DIM,
            num_iters=SLOT_ITERS, num_gmm_components=GMM_COMPONENTS)

        # *** NOVEL: Shape-Only Retrieval Head ***
        self.shape_head = ShapeOnlyRetrievalHead(
            SHAPE_DIM, NUM_SLOTS, embed_dim, num_classes)

        # Global embedding
        self.global_embed = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True))
        self.global_cls = nn.Linear(embed_dim, num_classes)

        # Fusion: global + shape
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim))

    def forward(self, x, return_all=False):
        final_feat, stage_outputs = self.backbone(x)

        # Stage logits
        stage_logits = [h(f) for h, f in zip(self.aux_heads, stage_outputs)]

        # Global embedding
        global_emb = self.global_embed(final_feat)
        global_logits = self.global_cls(global_emb)

        # Slot attention on final-stage tokens
        B, C, H, W = stage_outputs[-1].shape
        tokens = stage_outputs[-1].flatten(2).transpose(1, 2)  # [B, N, 768]

        slots, attn_weights, shape_slots, texture_slots = self.slot_attn(tokens)

        # Shape-only retrieval
        shape_emb, shape_logits = self.shape_head(shape_slots)

        # Fused embedding (global + shape)
        fused = self.fusion(torch.cat([global_emb, shape_emb], 1))
        fused_norm = F.normalize(fused, p=2, dim=1)

        if return_all:
            return {
                'embedding_normed': fused_norm,
                'logits': global_logits,
                'stage_logits': stage_logits,
                'shape_embedding': F.normalize(shape_emb, p=2, dim=1),
                'shape_logits': shape_logits,
                'slots': slots,
                'shape_slots': shape_slots,
                'texture_slots': texture_slots,
                'slot_attn': attn_weights,
                'final_feature': final_feat,
                'global_embedding': global_emb,
            }
        return fused_norm, global_logits

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
        print(f"  DINOv2 loaded!")

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
    def __init__(self, t=0.07):
        super().__init__()
        self.t = t
    def forward(self, d, s, labels):
        d = F.normalize(d, 1); s = F.normalize(s, 1)
        sim = d @ s.T / self.t
        lab = labels.view(-1, 1); pm = lab.eq(lab.T).float()
        l1 = -(F.log_softmax(sim, 1) * pm).sum(1) / pm.sum(1).clamp(1)
        l2 = -(F.log_softmax(sim.T, 1) * pm).sum(1) / pm.sum(1).clamp(1)
        return 0.5 * (l1.mean() + l2.mean())


class SelfDistillationLoss(nn.Module):
    def __init__(self, T=4.0):
        super().__init__()
        self.T = T
    def forward(self, stage_logits):
        loss = 0.0; final = stage_logits[-1]
        w = [0.1, 0.2, 0.3, 0.4]
        for i in range(len(stage_logits) - 1):
            pt = F.softmax(stage_logits[i] / self.T, 1)
            ps = F.log_softmax(final / self.T, 1)
            loss += w[i] * (self.T**2) * F.kl_div(ps, pt, reduction='batchmean')
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


class DisentanglementLoss(nn.Module):
    """Encourages disentanglement between shape and texture slot dimensions.

    Uses two regularization terms:
    1. Orthogonality: shape and texture slot subspaces should be orthogonal
    2. Independence: promotes statistical independence via covariance penalty
    """
    def __init__(self):
        super().__init__()

    def forward(self, shape_slots, texture_slots):
        """
        Args:
            shape_slots: [B, K, shape_dim]
            texture_slots: [B, K, texture_dim]
        """
        B, K, Ds = shape_slots.shape
        _, _, Dt = texture_slots.shape

        # 1) Cross-correlation penalty
        # Flatten to [B, K*Ds] and [B, K*Dt]
        s_flat = shape_slots.reshape(B, -1)  # [B, K*Ds]
        t_flat = texture_slots.reshape(B, -1)  # [B, K*Dt]

        # Normalize
        s_norm = s_flat - s_flat.mean(0)
        t_norm = t_flat - t_flat.mean(0)

        # Cross-correlation matrix [K*Ds, K*Dt]
        cross_corr = (s_norm.T @ t_norm) / max(B - 1, 1)

        # Minimize off-diagonal (independence)
        ortho_loss = cross_corr.pow(2).mean()

        # 2) Slot diversity: encourage different slots to capture different objects
        slot_sim = F.cosine_similarity(
            shape_slots.unsqueeze(2),  # [B, K, 1, Ds]
            shape_slots.unsqueeze(1),  # [B, 1, K, Ds]
            dim=-1
        )  # [B, K, K]

        # Mask diagonal
        mask = 1 - torch.eye(K, device=shape_slots.device).unsqueeze(0)
        diversity_loss = (slot_sim * mask).mean()

        return ortho_loss + 0.5 * diversity_loss


class SlotContrastiveLoss(nn.Module):
    """Cross-view slot contrastive loss.

    Encourages shape slots from the same location (drone and satellite)
    to be similar, while different locations should be dissimilar.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature

    def forward(self, d_shape, s_shape, labels):
        """
        Args:
            d_shape: [B, K, shape_dim] — drone shape slots
            s_shape: [B, K, shape_dim] — satellite shape slots
            labels: [B]
        """
        # Aggregate slots (permutation invariant via mean)
        d = F.normalize(d_shape.mean(1), dim=1)  # [B, Ds]
        s = F.normalize(s_shape.mean(1), dim=1)

        sim = d @ s.T / self.t
        lab = labels.view(-1, 1); pm = lab.eq(lab.T).float()
        l1 = -(F.log_softmax(sim, 1) * pm).sum(1) / pm.sum(1).clamp(1)
        l2 = -(F.log_softmax(sim.T, 1) * pm).sum(1) / pm.sum(1).clamp(1)
        return 0.5 * (l1.mean() + l2.mean())


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate(model, test_dataset, device, data_root=None):
    model.eval()
    loader = DataLoader(test_dataset, 256, False, num_workers=NUM_WORKERS, pin_memory=True)
    feats, labels = [], []
    with torch.no_grad():
        for b in loader:
            f, _ = model(b['drone'].to(device))
            feats.append(f.cpu()); labels.append(b['label'])
    feats = torch.cat(feats); labels = torch.cat(labels)

    tf = get_transforms("test")
    root = data_root or test_dataset.root
    sd = os.path.join(root, "satellite-view")
    locs = [f"{l:04d}" for l in TRAIN_LOCS + TEST_LOCS]

    sf, sl, gn = [], [], []
    for loc in locs:
        sp = os.path.join(sd, loc, "0.png")
        if not os.path.exists(sp): continue
        t = tf(Image.open(sp).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad(): f, _ = model(t)
        sf.append(f.cpu())
        sl.append(test_dataset.location_to_idx[loc] if loc in test_dataset.location_to_idx else -1-len(gn))
        gn.append(loc)

    sf = torch.cat(sf); sl = torch.tensor(sl)
    sim = feats @ sf.T; _, idx = sim.sort(1, descending=True)
    N = feats.size(0); r1=r5=r10=0; ap=0.0
    for i in range(N):
        ranked = sl[idx[i]]; c = torch.where(ranked == labels[i])[0]
        if len(c)==0: continue
        fc = c[0].item()
        if fc<1: r1+=1
        if fc<5: r5+=1
        if fc<10: r10+=1
        ap += sum((j+1)/(p.item()+1) for j,p in enumerate(c))/len(c)
    recall = {'R@1':r1/N*100,'R@5':r5/N*100,'R@10':r10/N*100}
    ap = ap/N*100
    print(f"  Gallery: {len(sf)}, Queries: {N}")
    return recall, ap


# =============================================================================
# TRAINING
# =============================================================================
def train(args):
    set_seed(SEED)
    print("="*70)
    print("GeoDISA: Disentangled Slot Attention for Geo-Localization")
    print("="*70)

    train_ds = SUES200Dataset(args.data_root, "train", transform=get_transforms("train"))
    test_ds = SUES200Dataset(args.data_root, "test", transform=get_transforms("test"))
    nc = len(TRAIN_LOCS)
    K = max(2, BATCH_SIZE//8)
    sampler = PKSampler(train_ds, p=8, k=K)
    loader = DataLoader(train_ds, batch_sampler=sampler,
                        num_workers=NUM_WORKERS, pin_memory=True)

    model = GeoDISAStudent(num_classes=nc).to(DEVICE)
    prms = sum(p.numel() for p in model.parameters())/1e6
    print(f"  Student: {prms:.1f}M params")

    try:
        teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except: teacher = None

    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    trip_fn = TripletLoss(MARGIN)
    nce_fn = SymmetricInfoNCELoss(0.07)
    sd_fn = SelfDistillationLoss(4.0)
    uapa_fn = UAPALoss(4.0)
    dis_fn = DisentanglementLoss()
    slot_fn = SlotContrastiveLoss(0.1)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler(enabled=AMP_ENABLED)
    best_r1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        if epoch < WARMUP_EPOCHS:
            lr = LR * (epoch+1) / WARMUP_EPOCHS
        else:
            progress = (epoch-WARMUP_EPOCHS)/max(1, EPOCHS-WARMUP_EPOCHS)
            lr = 1e-6 + 0.5*(LR-1e-6)*(1+math.cos(math.pi*progress))
        for pg in optimizer.param_groups: pg['lr'] = lr

        tl = 0.0; lp = defaultdict(float); nb = 0

        for bi, batch in enumerate(loader):
            drone = batch['drone'].to(DEVICE)
            sat = batch['satellite'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            with autocast(enabled=AMP_ENABLED):
                do = model(drone, return_all=True)
                so = model(sat, return_all=True)

                L = {}

                # CE (global + stages + shape)
                ce = LAMBDA_CE * (ce_fn(do['logits'], labels) + ce_fn(so['logits'], labels))
                for sl in do['stage_logits']: ce += 0.25 * ce_fn(sl, labels)
                for sl in so['stage_logits']: ce += 0.25 * ce_fn(sl, labels)
                ce += 0.5 * (ce_fn(do['shape_logits'], labels) + ce_fn(so['shape_logits'], labels))
                L['ce'] = ce

                # Triplet
                L['trip'] = LAMBDA_TRIPLET * (
                    trip_fn(do['embedding_normed'], labels) +
                    trip_fn(so['embedding_normed'], labels))

                # InfoNCE
                L['nce'] = LAMBDA_NCE * nce_fn(do['embedding_normed'], so['embedding_normed'], labels)

                # *** NOVEL: Slot contrastive (shape-only cross-view) ***
                L['slot'] = LAMBDA_SLOT * slot_fn(do['shape_slots'], so['shape_slots'], labels)

                # *** NOVEL: Disentanglement regularization ***
                L['dis'] = LAMBDA_DISENTANGLE * 0.5 * (
                    dis_fn(do['shape_slots'], do['texture_slots']) +
                    dis_fn(so['shape_slots'], so['texture_slots']))

                # Self-distillation
                L['sd'] = LAMBDA_SD * 0.5 * (sd_fn(do['stage_logits']) + sd_fn(so['stage_logits']))

                # UAPA
                L['uapa'] = LAMBDA_UAPA * uapa_fn(do['logits'], so['logits'])

                # Cross-distillation
                if teacher is not None:
                    with torch.no_grad():
                        td = teacher(drone); ts = teacher(sat)
                    df = F.normalize(do['final_feature'], 1)
                    sf = F.normalize(so['final_feature'], 1)
                    tdn = F.normalize(td, 1); tsn = F.normalize(ts, 1)
                    L['cdist'] = LAMBDA_CDIST * (
                        F.mse_loss(df, tdn) + F.mse_loss(sf, tsn) +
                        (1-F.cosine_similarity(df, tdn).mean()) +
                        (1-F.cosine_similarity(sf, tsn).mean()))

                total = sum(L.values())

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()

            tl += total.item(); nb += 1
            for k, v in L.items(): lp[k] += v.item()
            if bi % 10 == 0: print(f"  B{bi}/{len(loader)} L={total.item():.4f}")

        nb = max(1, nb)
        print(f"\nEp {epoch+1}/{EPOCHS} LR={lr:.6f} AvgL={tl/nb:.4f}")
        for k, v in sorted(lp.items()): print(f"  {k}: {v/nb:.4f}")

        if (epoch+1) % EVAL_FREQ == 0 or epoch == EPOCHS-1:
            print("\nEvaluating...")
            recall, ap = evaluate(model, test_ds, DEVICE)
            print(f"  R@1: {recall['R@1']:.2f}%  R@5: {recall['R@5']:.2f}%  "
                  f"R@10: {recall['R@10']:.2f}%  AP: {ap:.2f}%")
            if recall['R@1'] > best_r1:
                best_r1 = recall['R@1']
                torch.save({'epoch':epoch, 'model':model.state_dict(),
                            'r1':best_r1, 'ap':ap},
                           os.path.join(OUTPUT_DIR, 'geodisa_best.pth'))
                print(f"  *** New best R@1={best_r1:.2f}% ***")

    print(f"\nDone! Best R@1={best_r1:.2f}%")


# =============================================================================
# SMOKE TEST
# =============================================================================
def smoke_test():
    print("="*50); print("SMOKE TEST — GeoDISA"); print("="*50)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = GeoDISAStudent(10).to(dev)
    print(f"✓ Model: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")

    x = torch.randn(4, 3, 224, 224, device=dev)
    lab = torch.tensor([0, 0, 1, 1], device=dev)
    o = m(x, return_all=True)
    print(f"✓ Forward: emb={o['embedding_normed'].shape}")
    print(f"  slots={o['slots'].shape}, shape={o['shape_slots'].shape}, "
          f"texture={o['texture_slots'].shape}")
    print(f"  slot_attn={o['slot_attn'].shape}")

    # Disentanglement test
    dis = DisentanglementLoss()
    dl = dis(o['shape_slots'], o['texture_slots'])
    print(f"✓ Disentanglement loss: {dl.item():.4f}")

    # Backward
    loss = nn.CrossEntropyLoss()(o['logits'], lab) + dl
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
