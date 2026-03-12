#!/usr/bin/env python3
"""
GeoAGEN: Adaptive Error-Controlled Cross-View Drone Geo-Localization
=====================================================================
Novel contributions:
  1. Fuzzy PID Loss Controller — dynamically modulates loss weights each epoch
     using Proportional/Integral/Derivative error signals for stable convergence
  2. Multi-Branch Local Classifiers — 4 spatial partition branches (quadrants)
     for fine-grained local feature extraction alongside global features
  3. Adaptive Error-Guided Temperature — adjusts distillation temperature
     based on training convergence velocity

Inspired by: AGEN (IEEE Sensors 2025) — 94-97% R@1 on SUES-200

Architecture:
  Student: ConvNeXt-Tiny + Multi-Branch Local Heads + Fuzzy PID Controller
  Teacher: DINOv2-Base (frozen)

Dataset: SUES-200 (drone ↔ satellite cross-view geo-localization)
Protocol: 120 train / 80 test, gallery = ALL 200 locations (confusion data)

Usage:
  python exp_geoagen.py           # Full training on Kaggle H100
  python exp_geoagen.py --test    # Smoke test
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

# Fuzzy PID coefficients
PID_KP        = 1.0    # Proportional gain
PID_KI        = 0.1    # Integral gain
PID_KD        = 0.05   # Derivative gain

# Number of local branches (spatial partitions)
NUM_BRANCHES  = 4

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
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
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
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)))
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[ConvNeXtBlock(dims[i], dp_rates[cur+j], layer_scale_init)
                                    for j in range(depths[i])])
            self.stages.append(stage); cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        stage_outputs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            stage_outputs.append(x)
        final_feat = x.mean([-2, -1])
        final_feat = self.norm(final_feat)
        return final_feat, stage_outputs

    def forward(self, x):
        return self.forward_features(x)


def load_convnext_pretrained(model):
    url = "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth"
    try:
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)
        state_dict = {k: v for k, v in checkpoint["model"].items() if not k.startswith('head')}
        model.load_state_dict(state_dict, strict=False)
        print("Loaded ConvNeXt-Tiny pretrained weights (ImageNet-22K)")
    except Exception as e:
        print(f"Could not load pretrained weights: {e}")
    return model


# =============================================================================
# NOVEL COMPONENT 1: MULTI-BRANCH LOCAL CLASSIFIERS
# =============================================================================
class LocalBranchHead(nn.Module):
    """Local classifier branch for spatial partition.

    Splits the final feature map into spatial regions and applies
    independent classifiers to learn fine-grained local features.
    """
    def __init__(self, in_dim, num_classes, num_parts=4):
        super().__init__()
        self.num_parts = num_parts
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            ) for _ in range(num_parts)
        ])
        self.local_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
            ) for _ in range(num_parts)
        ])

    def _partition_feature_map(self, feat_map):
        """Split feature map into spatial parts.

        For num_parts=4: top-left, top-right, bottom-left, bottom-right
        """
        B, C, H, W = feat_map.shape
        h_mid, w_mid = H // 2, W // 2
        parts = [
            feat_map[:, :, :h_mid, :w_mid],         # top-left
            feat_map[:, :, :h_mid, w_mid:],          # top-right
            feat_map[:, :, h_mid:, :w_mid],          # bottom-left
            feat_map[:, :, h_mid:, w_mid:],           # bottom-right
        ]
        return parts

    def forward(self, feat_map):
        """
        Args: feat_map: [B, C, H, W] from backbone stage 4
        Returns: local_logits: list of [B, num_classes], local_embeds: list of [B, 256]
        """
        parts = self._partition_feature_map(feat_map)
        local_logits = []
        local_embeds = []
        for i, part in enumerate(parts):
            pooled = self.pool(part).flatten(1)  # [B, C]
            local_logits.append(self.classifiers[i](pooled))
            local_embeds.append(self.local_embed[i](pooled))
        return local_logits, local_embeds


# =============================================================================
# NOVEL COMPONENT 2: FUZZY PID LOSS CONTROLLER
# =============================================================================
class FuzzyPIDController:
    """Fuzzy PID controller for adaptive loss weight modulation.

    Monitors training error trajectory and dynamically adjusts loss weights
    using Proportional-Integral-Derivative control signals.

    The Fuzzy component maps continuous error signals to discrete fuzzy sets
    (NB, NS, ZE, PS, PB) and applies fuzzy inference rules to determine
    the optimal gain adjustments.
    """
    def __init__(self, kp=1.0, ki=0.1, kd=0.05, num_losses=6):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.num_losses = num_losses

        # Track per-loss error history
        self.error_history = [[] for _ in range(num_losses)]
        self.integral = [0.0] * num_losses
        self.prev_error = [0.0] * num_losses

        # Base weights for each loss
        self.base_weights = [1.0] * num_losses

        # Fuzzy membership boundaries
        self.fuzzy_sets = {
            'NB': (-1.0, -0.5),   # Negative Big
            'NS': (-0.5, -0.1),   # Negative Small
            'ZE': (-0.1,  0.1),   # Zero
            'PS': ( 0.1,  0.5),   # Positive Small
            'PB': ( 0.5,  1.0),   # Positive Big
        }

    def _fuzzify(self, value):
        """Map continuous value to fuzzy membership degrees."""
        memberships = {}
        for name, (low, high) in self.fuzzy_sets.items():
            center = (low + high) / 2
            width = (high - low) / 2
            memberships[name] = max(0, 1 - abs(value - center) / max(width, 1e-8))
        return memberships

    def _defuzzify(self, memberships):
        """Defuzzify using Center of Gravity method."""
        centers = {'NB': -0.75, 'NS': -0.3, 'ZE': 0.0, 'PS': 0.3, 'PB': 0.75}
        num = sum(memberships[k] * centers[k] for k in memberships)
        den = sum(memberships.values()) + 1e-8
        return num / den

    def _fuzzy_inference(self, error, derivative):
        """Apply fuzzy inference rules.

        Rules capture expert knowledge:
        - If error is PB and derivative is PB → increase weight a lot
        - If error is ZE and derivative is ZE → keep weight stable
        - If error is NB → decrease weight (loss is too dominant)
        """
        e_fuzzy = self._fuzzify(error)
        d_fuzzy = self._fuzzify(derivative)

        # Simplified rule base (output fuzzy set for weight adjustment)
        output_memberships = {'NB': 0, 'NS': 0, 'ZE': 0, 'PS': 0, 'PB': 0}

        # Rules: (error_set, derivative_set) → output_set
        rules = [
            ('PB', 'PB', 'PB'), ('PB', 'PS', 'PB'), ('PB', 'ZE', 'PS'),
            ('PS', 'PB', 'PB'), ('PS', 'PS', 'PS'), ('PS', 'ZE', 'ZE'),
            ('ZE', 'PB', 'PS'), ('ZE', 'PS', 'ZE'), ('ZE', 'ZE', 'ZE'),
            ('ZE', 'NS', 'ZE'), ('ZE', 'NB', 'NS'),
            ('NS', 'ZE', 'NS'), ('NS', 'NS', 'NS'), ('NS', 'NB', 'NB'),
            ('NB', 'ZE', 'NB'), ('NB', 'NS', 'NB'), ('NB', 'NB', 'NB'),
        ]

        for e_set, d_set, out_set in rules:
            activation = min(e_fuzzy.get(e_set, 0), d_fuzzy.get(d_set, 0))
            output_memberships[out_set] = max(output_memberships[out_set], activation)

        return self._defuzzify(output_memberships)

    def update(self, epoch, loss_values, target_loss=None):
        """Update PID state and return adjusted loss weights.

        Args:
            epoch: current epoch number
            loss_values: list of current loss values for each component
            target_loss: optional target total loss (if None, uses moving average)

        Returns:
            adjusted_weights: list of adjusted loss weights
        """
        adjusted_weights = []

        for i, loss_val in enumerate(loss_values):
            # Record history
            self.error_history[i].append(loss_val)

            if len(self.error_history[i]) < 2:
                adjusted_weights.append(self.base_weights[i])
                continue

            # Compute normalized error (how far from moving average)
            recent = self.error_history[i][-min(10, len(self.error_history[i])):]
            moving_avg = sum(recent) / len(recent)
            error = (loss_val - moving_avg) / max(moving_avg, 1e-8)
            error = max(-1, min(1, error))  # Clamp to [-1, 1]

            # Derivative (rate of change)
            derivative = error - self.prev_error[i]
            derivative = max(-1, min(1, derivative))

            # Integral (accumulated error)
            self.integral[i] += error
            self.integral[i] = max(-10, min(10, self.integral[i]))  # Anti-windup

            # Standard PID output
            pid_output = (
                self.kp * error +
                self.ki * self.integral[i] +
                self.kd * derivative
            )

            # Fuzzy adjustment
            fuzzy_adj = self._fuzzy_inference(error, derivative)

            # Combine PID and fuzzy
            adjustment = 0.7 * pid_output + 0.3 * fuzzy_adj

            # Apply to base weight with bounds
            new_weight = self.base_weights[i] * (1.0 + 0.3 * adjustment)
            new_weight = max(0.1, min(3.0, new_weight))

            adjusted_weights.append(new_weight)
            self.prev_error[i] = error

        return adjusted_weights


# =============================================================================
# NOVEL COMPONENT 3: ADAPTIVE ERROR-GUIDED TEMPERATURE
# =============================================================================
class AdaptiveTemperature(nn.Module):
    """Dynamically adjusts distillation temperature based on convergence.

    When training loss is high (early training) → higher temperature for softer targets
    When training loss plateaus → lower temperature for sharper targets
    """
    def __init__(self, T0=4.0, T_min=1.0, T_max=10.0):
        super().__init__()
        self.T0 = T0
        self.T_min = T_min
        self.T_max = T_max
        self.loss_history = []

    def get_temperature(self, current_loss):
        """Compute adaptive temperature."""
        self.loss_history.append(current_loss)

        if len(self.loss_history) < 3:
            return self.T0

        # Compute loss velocity (derivative)
        recent = self.loss_history[-5:]
        velocity = (recent[-1] - recent[0]) / len(recent)

        # Compute loss acceleration
        if len(self.loss_history) >= 5:
            older = self.loss_history[-10:-5] if len(self.loss_history) >= 10 else self.loss_history[:5]
            old_vel = (older[-1] - older[0]) / len(older)
            accel = velocity - old_vel
        else:
            accel = 0.0

        # Higher temperature when loss is decreasing rapidly (helpful distillation)
        # Lower temperature when loss plateaus (need sharper signal)
        if velocity < -0.01:  # Decreasing
            T = self.T0 * (1.0 + 0.5 * abs(velocity))
        elif abs(velocity) < 0.001:  # Plateau
            T = self.T0 * 0.7
        else:  # Increasing (unstable)
            T = self.T0 * 1.2

        return max(self.T_min, min(self.T_max, T))


# =============================================================================
# GEOAGEN STUDENT MODEL
# =============================================================================
class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.fc(x)


class GeneralizedMeanPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


class GeoAGENStudent(nn.Module):
    """GeoAGEN = ConvNeXt-Tiny + Multi-Branch Local + Fuzzy PID.

    Pipeline:
      1. ConvNeXt extracts multi-scale features (4 stages)
      2. Global head produces classification + embedding
      3. Multi-Branch Local heads produce fine-grained per-region features
      4. All losses are dynamically weighted by the Fuzzy PID controller
    """
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = ConvNeXtTiny(drop_path_rate=0.1)
        self.backbone = load_convnext_pretrained(self.backbone)
        self.dims = [96, 192, 384, 768]

        # Stage-wise auxiliary heads (self-distillation)
        self.aux_heads = nn.ModuleList([
            ClassificationHead(dim, num_classes) for dim in self.dims
        ])

        # Global embedding
        self.bottleneck = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True))
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.gem = GeneralizedMeanPooling()

        # *** NOVEL: Multi-Branch Local Classifiers ***
        self.local_branches = LocalBranchHead(768, num_classes, NUM_BRANCHES)

        # Local embeddings fusion → global
        self.local_fusion = nn.Sequential(
            nn.Linear(256 * NUM_BRANCHES, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim))

    def forward(self, x, return_all=False):
        final_feat, stage_outputs = self.backbone(x)

        # Stage logits (for self-distillation)
        stage_logits = [head(feat) for head, feat in zip(self.aux_heads, stage_outputs)]

        # Global embedding
        embedding = self.bottleneck(final_feat)
        embedding_normed = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)

        # Local branch features from stage-4 feature map
        local_logits, local_embeds = self.local_branches(stage_outputs[-1])

        # Fuse local embeddings into a global-local descriptor
        local_concat = torch.cat(local_embeds, dim=1)  # [B, 256*4]
        local_global = self.local_fusion(local_concat)  # [B, embed_dim]
        local_global_normed = F.normalize(local_global, p=2, dim=1)

        # Combined embedding (global + local)
        combined = F.normalize(embedding + local_global, p=2, dim=1)

        if return_all:
            return {
                'embedding': embedding,
                'embedding_normed': embedding_normed,
                'combined_normed': combined,
                'local_global_normed': local_global_normed,
                'logits': logits,
                'stage_logits': stage_logits,
                'local_logits': local_logits,
                'local_embeds': local_embeds,
                'stage_features': stage_outputs,
                'final_feature': final_feat,
            }
        return combined, logits

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
        self.num_channels = 768
        for param in self.parameters(): param.requires_grad = False
        for blk in self.model.blocks[-num_trainable_blocks:]:
            for param in blk.parameters(): param.requires_grad = True
        print(f"  DINOv2 loaded! Last {num_trainable_blocks} blocks trainable.")

    @torch.no_grad()
    def forward(self, x):
        tokens = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks: tokens = blk(tokens)
        tokens = self.model.norm(tokens)
        return tokens[:, 0]  # cls token


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    def forward(self, embeddings, labels):
        dist = torch.cdist(embeddings, embeddings, p=2)
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
    def __init__(self, temperature=4.0, weights=[0.1, 0.2, 0.3, 0.4]):
        super().__init__()
        self.temperature = temperature
        self.weights = weights
    def forward(self, stage_logits, adaptive_T=None):
        T = adaptive_T if adaptive_T is not None else self.temperature
        loss = 0.0
        final_logits = stage_logits[-1]
        for i in range(len(stage_logits) - 1):
            p_teacher = F.softmax(stage_logits[i] / T, dim=1)
            p_student = F.log_softmax(final_logits / T, dim=1)
            loss += self.weights[i] * (T ** 2) * F.kl_div(p_student, p_teacher, reduction='batchmean')
        return loss


class UAPALoss(nn.Module):
    def __init__(self, base_temperature=4.0):
        super().__init__()
        self.T0 = base_temperature
    def forward(self, drone_logits, sat_logits):
        Ud = -(F.softmax(drone_logits, 1) * F.log_softmax(drone_logits, 1)).sum(1).mean()
        Us = -(F.softmax(sat_logits, 1) * F.log_softmax(sat_logits, 1)).sum(1).mean()
        T = self.T0 * (1 + torch.sigmoid(Ud - Us))
        return (T ** 2) * F.kl_div(
            F.log_softmax(drone_logits / T, 1),
            F.softmax(sat_logits / T, 1), reduction='batchmean')


class CrossDistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, student_feat, teacher_feat):
        s = F.normalize(student_feat, 1)
        t = F.normalize(teacher_feat, 1)
        return F.mse_loss(s, t) + (1 - F.cosine_similarity(s, t).mean())


# =============================================================================
# EVALUATION (same as baseline — 200-gallery protocol)
# =============================================================================
def evaluate(model, test_dataset, device, data_root=None):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    all_drone_feats, all_drone_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            feats, _ = model(batch['drone'].to(device))
            all_drone_feats.append(feats.cpu())
            all_drone_labels.append(batch['label'])
    all_drone_feats = torch.cat(all_drone_feats)
    all_drone_labels = torch.cat(all_drone_labels)

    # Build full gallery (ALL 200 satellites)
    transform = get_transforms("test")
    root = data_root or test_dataset.root
    satellite_dir = os.path.join(root, "satellite-view")
    all_locs = [f"{loc:04d}" for loc in TRAIN_LOCS + TEST_LOCS]

    sat_feats_list, sat_labels_list, gallery_names = [], [], []
    for loc in all_locs:
        sat_path = os.path.join(satellite_dir, loc, "0.png")
        if not os.path.exists(sat_path): continue
        sat_img = Image.open(sat_path).convert('RGB')
        sat_tensor = transform(sat_img).unsqueeze(0).to(device)
        with torch.no_grad():
            sat_feat, _ = model(sat_tensor)
        sat_feats_list.append(sat_feat.cpu())
        if loc in test_dataset.location_to_idx:
            sat_labels_list.append(test_dataset.location_to_idx[loc])
        else:
            sat_labels_list.append(-1 - len(gallery_names))
        gallery_names.append(loc)

    sat_feats = torch.cat(sat_feats_list)
    sat_labels = torch.tensor(sat_labels_list)

    # Compute metrics
    sim = torch.mm(all_drone_feats, sat_feats.T)
    _, indices = sim.sort(dim=1, descending=True)
    N = all_drone_feats.size(0)
    r1 = r5 = r10 = ap_sum = 0
    for i in range(N):
        ranked = sat_labels[indices[i]]
        correct = torch.where(ranked == all_drone_labels[i])[0]
        if len(correct) == 0: continue
        fc = correct[0].item()
        if fc < 1: r1 += 1
        if fc < 5: r5 += 1
        if fc < 10: r10 += 1
        for j, pos in enumerate(correct):
            ap_sum += (j + 1) / (pos.item() + 1)
        ap_sum_temp = 0
        for j, pos in enumerate(correct):
            ap_sum_temp += (j + 1) / (pos.item() + 1)
        # Already accumulated above

    recall = {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100}
    # Recompute AP properly
    ap_total = 0.0
    for i in range(N):
        ranked = sat_labels[indices[i]]
        correct = torch.where(ranked == all_drone_labels[i])[0]
        if len(correct) == 0: continue
        prec_sum = sum((j+1)/(pos.item()+1) for j, pos in enumerate(correct))
        ap_total += prec_sum / len(correct)
    ap = ap_total / N * 100

    print(f"  Gallery: {len(sat_feats)} sats, Queries: {N} drones")
    return recall, ap


# =============================================================================
# TRAINING WITH FUZZY PID
# =============================================================================
def train(args):
    set_seed(SEED)
    print("=" * 70)
    print("GeoAGEN: Adaptive Error-Controlled Cross-View Geo-Localization")
    print("=" * 70)

    # Data
    train_ds = SUES200Dataset(args.data_root, "train", transform=get_transforms("train"))
    test_ds = SUES200Dataset(args.data_root, "test", transform=get_transforms("test"))
    num_classes = len(TRAIN_LOCS)
    K = max(2, BATCH_SIZE // 8)
    sampler = PKSampler(train_ds, p=8, k=K)
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Models
    model = GeoAGENStudent(num_classes=num_classes).to(DEVICE)
    prms = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Student: {prms:.1f}M params")

    try:
        teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except Exception as e:
        print(f"  Could not load DINOv2: {e}"); teacher = None

    # Losses
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    trip_fn = TripletLoss(MARGIN)
    nce_fn = SymmetricInfoNCELoss(0.07)
    sd_fn = SelfDistillationLoss(4.0)
    uapa_fn = UAPALoss(4.0)
    cd_fn = CrossDistillationLoss()

    # *** NOVEL: Fuzzy PID Controller ***
    # Loss indices: 0=CE, 1=Triplet, 2=NCE, 3=SelfDist, 4=UAPA, 5=CrossDist
    pid_controller = FuzzyPIDController(
        kp=PID_KP, ki=PID_KI, kd=PID_KD, num_losses=6
    )

    # *** NOVEL: Adaptive Temperature ***
    adaptive_temp = AdaptiveTemperature(T0=4.0)

    # Optimizer (SGD like baseline)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler(enabled=AMP_ENABLED)

    best_r1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        if teacher is not None: teacher.eval()

        # LR schedule: warmup + cosine
        if epoch < WARMUP_EPOCHS:
            lr = LR * (epoch + 1) / WARMUP_EPOCHS
        else:
            progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
            lr = 1e-6 + 0.5 * (LR - 1e-6) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        tl = 0.0
        loss_components = defaultdict(float)
        batch_count = 0

        for bi, batch in enumerate(train_loader):
            drone = batch['drone'].to(DEVICE)
            sat = batch['satellite'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            with autocast(enabled=AMP_ENABLED):
                do = model(drone, return_all=True)
                so = model(sat, return_all=True)

                # Compute individual losses
                L = {}

                # 1) CE loss (global + stages + local branches)
                ce = ce_fn(do['logits'], labels) + ce_fn(so['logits'], labels)
                for sl in do['stage_logits']:
                    ce += 0.25 * ce_fn(sl, labels)
                for sl in so['stage_logits']:
                    ce += 0.25 * ce_fn(sl, labels)
                # Local branch CE
                for ll in do['local_logits']:
                    ce += 0.25 * ce_fn(ll, labels)
                for ll in so['local_logits']:
                    ce += 0.25 * ce_fn(ll, labels)
                L['ce'] = ce

                # 2) Triplet (global + local)
                trip = trip_fn(do['combined_normed'], labels) + trip_fn(so['combined_normed'], labels)
                L['triplet'] = trip

                # 3) Cross-view InfoNCE
                L['nce'] = nce_fn(do['combined_normed'], so['combined_normed'], labels)

                # 4) Self-distillation with adaptive temperature
                adaptive_T = adaptive_temp.get_temperature(tl / max(1, batch_count))
                sd = sd_fn(do['stage_logits'], adaptive_T) + sd_fn(so['stage_logits'], adaptive_T)
                L['self_dist'] = sd

                # 5) UAPA
                L['uapa'] = uapa_fn(do['logits'], so['logits'])

                # 6) Cross-distillation (with teacher)
                if teacher is not None:
                    with torch.no_grad():
                        td = teacher(drone)
                        ts = teacher(sat)
                    L['cross_dist'] = cd_fn(do['final_feature'], td) + cd_fn(so['final_feature'], ts)
                else:
                    L['cross_dist'] = torch.tensor(0.0, device=DEVICE)

                # Get PID-adjusted weights
                loss_vals = [L['ce'].item(), L['triplet'].item(), L['nce'].item(),
                             L['self_dist'].item(), L['uapa'].item(), L['cross_dist'].item()]
                pid_weights = pid_controller.update(epoch, loss_vals)

                loss_keys = ['ce', 'triplet', 'nce', 'self_dist', 'uapa', 'cross_dist']
                total = sum(pid_weights[i] * L[loss_keys[i]] for i in range(6))

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()

            tl += total.item()
            for k, v in L.items():
                loss_components[k] += v.item()
            batch_count += 1

            if bi % 10 == 0:
                w_str = " ".join([f"w{i}={pid_weights[i]:.2f}" for i in range(6)])
                print(f"  B{bi}/{len(train_loader)} L={total.item():.4f} [{w_str}]")

        nb = max(1, batch_count)
        print(f"\nEp {epoch+1}/{EPOCHS} LR={lr:.6f} AvgL={tl/nb:.4f} T={adaptive_T:.2f}")
        for k, v in sorted(loss_components.items()):
            print(f"  {k}: {v/nb:.4f}")

        # Evaluate
        if (epoch + 1) % EVAL_FREQ == 0 or epoch == EPOCHS - 1:
            print("\nEvaluating...")
            recall, ap = evaluate(model, test_ds, DEVICE)
            print(f"  R@1: {recall['R@1']:.2f}%  R@5: {recall['R@5']:.2f}%  "
                  f"R@10: {recall['R@10']:.2f}%  AP: {ap:.2f}%")

            if recall['R@1'] > best_r1:
                best_r1 = recall['R@1']
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'r1': best_r1, 'ap': ap},
                           os.path.join(OUTPUT_DIR, 'geoagen_best.pth'))
                print(f"  *** New best R@1={best_r1:.2f}% ***")

    print(f"\nDone! Best R@1={best_r1:.2f}%")


# =============================================================================
# SMOKE TEST
# =============================================================================
def smoke_test():
    print("=" * 50); print("SMOKE TEST — GeoAGEN"); print("=" * 50)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = GeoAGENStudent(10).to(dev)
    print(f"✓ Model: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
    x = torch.randn(4, 3, 224, 224, device=dev)
    lab = torch.tensor([0, 0, 1, 1], device=dev)
    o = m(x, return_all=True)
    print(f"✓ Forward: combined={o['combined_normed'].shape}, "
          f"local_logits={len(o['local_logits'])}x{o['local_logits'][0].shape}")

    # Test PID controller
    pid = FuzzyPIDController(num_losses=6)
    w = pid.update(0, [1.0, 0.5, 0.3, 0.2, 0.1, 0.4])
    print(f"✓ PID weights: {[f'{wi:.3f}' for wi in w]}")

    # Backward
    loss = nn.CrossEntropyLoss()(o['logits'], lab)
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
