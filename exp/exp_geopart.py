#!/usr/bin/env python3
"""
GeoPart: Multi-Granularity Part Representations for Cross-View Geo-Localization
================================================================================
Built on: baseline.py (MobileGeo — 82.35% R@1)
Novel contribution: Multi-Granularity Part Pooling (MGPP)
  - Systematic spatial partitioning: 1x1 + 1x2 + 2x2 + 3x3 = 16 parts
  - Altitude-conditioned part attention
  - Parts added ADDITIVELY to baseline embedding (gate=0 at init → baseline behavior)
  - Part classifiers weighted at 0.05 to avoid dominating gradient

Key learning from failures:
  - Do NOT concatenate novel features with backbone features (random projection kills NCE)
  - Do NOT use GradScaler with SAM optimizer (gradient scale conflicts with two-pass)
  - START from baseline architecture, then add novel components as gated residuals
"""

# === SETUP ===
import subprocess, sys, os

def pip_install(pkg):
    subprocess.run(f"pip install -q {pkg}", shell=True, capture_output=True, text=True)

print("[1/2] Installing packages...")
for p in ["timm", "tqdm"]:
    try: __import__(p)
    except ImportError: pip_install(p)
print("[2/2] Setup complete!")

import math, random, argparse
import numpy as np
from PIL import Image
from collections import defaultdict
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchvision.transforms as T

print("[OK] All imports loaded!")

# =============================================================================
# CONFIG (same as baseline)
# =============================================================================
class Config:
    DATA_ROOT     = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    DRONE_DIR     = "drone-view"
    SATELLITE_DIR = "satellite-view"
    OUTPUT_DIR    = "/kaggle/working"

    NUM_WORKERS   = 8
    P             = 8
    K             = 4
    BATCH_SIZE    = 256
    NUM_EPOCHS    = 120
    LR            = 0.001
    WARMUP_EPOCHS = 5

    IMG_SIZE      = 224
    NUM_CLASSES   = 120
    EMBED_DIM     = 768
    DROP_PATH_RATE= 0.1

    TEMPERATURE   = 4.0
    BASE_TEMPERATURE = 4.0

    LAMBDA_TRIPLET   = 1.0
    LAMBDA_CSC       = 0.5
    LAMBDA_SELF_DIST = 0.5
    LAMBDA_CROSS_DIST= 0.3
    LAMBDA_ALIGN     = 0.2
    MARGIN           = 0.3

    ALTITUDES  = ["150", "200", "250", "300"]
    TRAIN_LOCS = list(range(1, 121))
    TEST_LOCS  = list(range(121, 201))
    USE_AMP    = True
    SEED       = 42

    # GeoPart novel config
    PART_DIM        = 256    # embedding dim per part
    NUM_PARTS       = 16     # 1+2+4+9 granularities
    LAMBDA_PART_CE  = 0.05   # very small to avoid dominating gradient


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark = True


# =============================================================================
# DATASET (identical to baseline)
# =============================================================================
class SUES200Dataset(Dataset):
    def __init__(self, root, mode="train", altitudes=None, transform=None,
                 train_locs=None, test_locs=None):
        self.root = root; self.mode = mode
        self.altitudes = altitudes or Config.ALTITUDES
        self.transform = transform
        self.drone_dir = os.path.join(root, Config.DRONE_DIR)
        self.satellite_dir = os.path.join(root, Config.SATELLITE_DIR)
        if train_locs is None: train_locs = Config.TRAIN_LOCS
        if test_locs is None:  test_locs  = Config.TEST_LOCS
        loc_ids = train_locs if mode == "train" else test_locs
        self.locations = [f"{loc:04d}" for loc in loc_ids]
        self.location_to_idx = {loc: idx for idx, loc in enumerate(self.locations)}
        self.samples = []; self.drone_by_location = defaultdict(list)
        for loc in self.locations:
            loc_idx = self.location_to_idx[loc]
            sat_path = os.path.join(self.satellite_dir, loc, "0.png")
            if not os.path.exists(sat_path): continue
            for alt in self.altitudes:
                alt_dir = os.path.join(self.drone_dir, loc, alt)
                if not os.path.isdir(alt_dir): continue
                for img_name in sorted(os.listdir(alt_dir)):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(alt_dir, img_name), sat_path, loc_idx, alt))
                        self.drone_by_location[loc_idx].append(len(self.samples) - 1)
        print(f"[{mode}] {len(self.samples)} samples, {len(self.locations)} locations")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        dp, sp, li, alt = self.samples[idx]
        d = Image.open(dp).convert('RGB'); s = Image.open(sp).convert('RGB')
        if self.transform: d = self.transform(d); s = self.transform(s)
        return {'drone': d, 'satellite': s, 'label': li, 'altitude': int(alt)}


class PKSampler:
    def __init__(self, dataset, p=8, k=4):
        self.dataset = dataset; self.p = p; self.k = k
        self.locations = list(dataset.drone_by_location.keys())

    def __iter__(self):
        locations = self.locations.copy(); random.shuffle(locations); batch = []
        for loc in locations:
            indices = self.dataset.drone_by_location[loc]
            if len(indices) < self.k: indices = indices * (self.k // len(indices) + 1)
            batch.extend(random.sample(indices, self.k))
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]; batch = batch[self.p * self.k:]

    def __len__(self): return len(self.locations) // self.p


def get_transforms(mode="train"):
    if mode == "train":
        return T.Compose([T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)), T.RandomHorizontalFlip(0.5),
            T.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0)), T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return T.Compose([T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# =============================================================================
# CONVNEXT-TINY BACKBONE (identical to baseline)
# =============================================================================
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps; self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True); s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * ((x - u) / torch.sqrt(s + self.eps)) + self.bias[:, None, None]


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rt = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    rt.floor_(); return x.div(keep_prob) * rt


class DropPath(nn.Module):
    def __init__(self, drop_prob=None): super().__init__(); self.drop_prob = drop_prob
    def forward(self, x): return drop_path(x, self.drop_prob, self.training)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path_rate=0., layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim); self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        shortcut = x; x = self.dwconv(x); x = x.permute(0, 2, 3, 1)
        x = self.norm(x); x = self.pwconv1(x); x = self.act(x); x = self.pwconv2(x)
        if self.gamma is not None: x = self.gamma * x
        return shortcut + self.drop_path(x.permute(0, 3, 2, 1).permute(0, 1, 3, 2))


class ConvNeXtTiny(nn.Module):
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.):
        super().__init__()
        self.dims = dims; self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], 4, 4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], 2, 2)))
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0; self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[ConvNeXtBlock(dims[i], dp_rates[cur + j]) for j in range(depths[i])]))
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6); self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        stage_outputs = []
        for i in range(4):
            x = self.downsample_layers[i](x); x = self.stages[i](x); stage_outputs.append(x)
        return self.norm(x.mean([-2, -1])), stage_outputs


def load_convnext_pretrained(model):
    try:
        ckpt = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth",
            map_location="cpu", check_hash=True)
        model.load_state_dict({k: v for k, v in ckpt["model"].items() if not k.startswith('head')}, strict=False)
        print("Loaded ConvNeXt-Tiny pretrained (ImageNet-22K)")
    except Exception as e: print(f"Could not load pretrained: {e}")
    return model


# =============================================================================
# BASELINE: ClassificationHead + GeneralizedMeanPooling (identical)
# =============================================================================
class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                nn.ReLU(True), nn.Dropout(0.5), nn.Linear(hidden_dim, num_classes))
    def forward(self, x): return self.fc(self.pool(x).flatten(1))


# =============================================================================
# NOVEL: MULTI-GRANULARITY PART POOLING
# =============================================================================
class MultiGranularityPartPooling(nn.Module):
    """Systematic spatial partitioning at 4 granularities → 16 part descriptors.
    Each part is separately pooled and projected to part_dim.
    """
    def __init__(self, in_dim=768, part_dim=256):
        super().__init__()
        self.granularities = [(1, 1), (1, 2), (2, 2), (3, 3)]  # 1+2+4+9 = 16 parts
        self.num_parts = sum(r * c for r, c in self.granularities)
        self.part_projs = nn.ModuleList([nn.Linear(in_dim, part_dim) for _ in range(self.num_parts)])
        self.part_bns   = nn.ModuleList([nn.BatchNorm1d(part_dim) for _ in range(self.num_parts)])

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        parts = []; idx = 0
        for rows, cols in self.granularities:
            hs, ws = H // rows, W // cols
            for r in range(rows):
                for c in range(cols):
                    h0, h1 = r * hs, (r + 1) * hs if r < rows - 1 else H
                    w0, w1 = c * ws, (c + 1) * ws if c < cols - 1 else W
                    pooled = feat_map[:, :, h0:h1, w0:w1].mean(dim=[-2, -1])  # [B, C]
                    proj = self.part_bns[idx](self.part_projs[idx](pooled))
                    parts.append(proj); idx += 1
        return parts  # list of 16 × [B, part_dim]


class AltitudePartAttention(nn.Module):
    """Learnable attention weights over 16 parts per altitude (gate-style)."""
    def __init__(self, num_parts=16, alt_values=[150, 200, 250, 300]):
        super().__init__()
        self.alt_to_idx = {a: i for i, a in enumerate(alt_values)}
        n = len(alt_values) + 1  # +1 for satellite
        self.attention = nn.Parameter(torch.ones(n, num_parts))
        self.default_idx = len(alt_values)
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, altitudes=None):
        if altitudes is None:
            w = self.attention[self.default_idx].unsqueeze(0)
        else:
            B = altitudes.shape[0]
            indices = torch.full((B,), self.default_idx, device=altitudes.device, dtype=torch.long)
            for val, idx in self.alt_to_idx.items():
                indices[altitudes == val] = idx
            w = self.attention[indices]
        return F.softmax(w / self.temp.abs().clamp(min=0.1), dim=-1)  # [B or 1, num_parts]


# =============================================================================
# STUDENT MODEL: BASELINE + MGPP (added as gated residual)
# =============================================================================
class GeoPartStudent(nn.Module):
    """
    Architecture = Baseline MobileGeo + MGPP (gated residual)

    Forward path:
      1. Backbone → final_feat [B, 768]          (same as baseline)
      2. Bottleneck → base_emb [B, 768]           (same as baseline)
      3. MGPP → 16 parts → weighted sum → part_emb [B, 768]  (NOVEL)
      4. embedding = base_emb + gate * part_emb   (gate=0 at init → baseline)
      5. L2 normalize → embedding_normed
      6. classifier → logits

    Why gated residual?
      - At init: gate=0 → embedding = base_emb → NCE/triplet see pretrained features
      - During training: gate grows → part features gradually contribute
      - NCE is NOT killed by random projections from novel components
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, embed_dim=Config.EMBED_DIM):
        super().__init__()
        self.backbone = ConvNeXtTiny(drop_path_rate=Config.DROP_PATH_RATE)
        self.backbone = load_convnext_pretrained(self.backbone)
        self.dims = [96, 192, 384, 768]

        # --- Baseline components (identical) ---
        self.aux_heads = nn.ModuleList([ClassificationHead(d, num_classes) for d in self.dims])
        self.bottleneck = nn.Sequential(
            nn.Linear(768, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(True))
        self.classifier = nn.Linear(embed_dim, num_classes)

        # --- Novel: MGPP ---
        self.mgpp = MultiGranularityPartPooling(in_dim=768, part_dim=Config.PART_DIM)
        self.part_attention = AltitudePartAttention(num_parts=Config.NUM_PARTS)
        self.part_classifiers = nn.ModuleList(
            [nn.Linear(Config.PART_DIM, num_classes) for _ in range(Config.NUM_PARTS)])

        # Project weighted parts to embed_dim for addition
        self.part_proj = nn.Sequential(
            nn.Linear(Config.PART_DIM, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(True))
        # Gate starts at 0 → baseline behavior at start of training
        self.part_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, altitudes=None, return_all=False):
        final_feat, stage_outputs = self.backbone(x)
        stage_logits = [head(feat) for head, feat in zip(self.aux_heads, stage_outputs)]

        # Baseline embedding path
        base_emb = self.bottleneck(final_feat)  # [B, embed_dim]

        # Novel: MGPP part features
        feat_map = stage_outputs[-1]              # [B, 768, H, W]
        parts = self.mgpp(feat_map)               # list of 16 × [B, part_dim]
        part_weights = self.part_attention(altitudes)  # [B, 16]
        stacked = torch.stack(parts, dim=1)       # [B, 16, part_dim]
        weighted_part = (stacked * part_weights.unsqueeze(-1)).sum(1)  # [B, part_dim]
        part_emb = self.part_proj(weighted_part)  # [B, embed_dim]

        # Gated residual addition
        embedding = base_emb + self.part_gate * part_emb  # gate=0 → baseline at init
        embedding_normed = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)

        # Part classifiers (for novel loss, but weighted very lightly)
        part_logits = [clf(p) for clf, p in zip(self.part_classifiers, parts)]

        if return_all:
            return {'embedding': embedding, 'embedding_normed': embedding_normed,
                    'logits': logits, 'stage_logits': stage_logits,
                    'stage_features': stage_outputs, 'final_feature': final_feat,
                    'part_logits': part_logits, 'part_gate': self.part_gate.item()}
        return embedding_normed, logits


# =============================================================================
# DINOV2 TEACHER (identical to baseline)
# =============================================================================
class DINOv2Teacher(nn.Module):
    def __init__(self, num_trainable_blocks=2):
        super().__init__()
        print("Loading DINOv2-base teacher...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        for p in self.model.parameters(): p.requires_grad = False
        for blk in self.model.blocks[-num_trainable_blocks:]:
            for p in blk.parameters(): p.requires_grad = True

    @torch.no_grad()
    def forward(self, x):
        x = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks: x = blk(x)
        return self.model.norm(x)[:, 0]


# =============================================================================
# LOSS FUNCTIONS (baseline losses + light part loss)
# =============================================================================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__(); self.margin = margin
    def forward(self, embeddings, labels):
        dist = torch.cdist(embeddings, embeddings, p=2)
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float(); neg_mask = labels.ne(labels.T).float()
        hard_pos = (dist * pos_mask).max(1)[0]
        hard_neg = (dist * neg_mask + pos_mask * 1e9).min(1)[0]
        return F.relu(hard_pos - hard_neg + self.margin).mean()


class SymmetricInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07): super().__init__(); self.temperature = temperature
    def forward(self, drone_feats, sat_feats, labels):
        d = F.normalize(drone_feats, dim=1); s = F.normalize(sat_feats, dim=1)
        sim = torch.mm(d, s.T) / self.temperature
        labels = labels.view(-1, 1); pos_mask = labels.eq(labels.T).float()
        # Baseline formula: -log(sum(softmax*pos)/num_pos)
        l1 = -torch.log((F.softmax(sim, 1) * pos_mask).sum(1) / pos_mask.sum(1).clamp(1)).mean()
        l2 = -torch.log((F.softmax(sim.T, 1) * pos_mask).sum(1) / pos_mask.sum(1).clamp(1)).mean()
        return 0.5 * (l1 + l2)


class SelfDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, weights=[0.1, 0.2, 0.3, 0.4]):
        super().__init__(); self.temperature = temperature; self.weights = weights
    def forward(self, stage_logits):
        loss = 0.0; final = stage_logits[-1]
        for i in range(len(stage_logits) - 1):
            loss += self.weights[i] * (self.temperature ** 2) * F.kl_div(
                F.log_softmax(final / self.temperature, 1),
                F.softmax(stage_logits[i] / self.temperature, 1), reduction='batchmean')
        return loss


class UAPALoss(nn.Module):
    def __init__(self, base_temperature=4.0): super().__init__(); self.T0 = base_temperature
    def forward(self, drone_logits, sat_logits):
        Ud = -(F.softmax(drone_logits, 1) * F.log_softmax(drone_logits, 1)).sum(1).mean()
        Us = -(F.softmax(sat_logits, 1) * F.log_softmax(sat_logits, 1)).sum(1).mean()
        T = self.T0 * (1 + torch.sigmoid(Ud - Us))
        return (T ** 2) * F.kl_div(F.log_softmax(drone_logits / T, 1),
                                    F.softmax(sat_logits / T, 1), reduction='batchmean')


class CrossDistillationLoss(nn.Module):
    def forward(self, student_feat, teacher_feat):
        sf = F.normalize(student_feat, dim=1); tf = F.normalize(teacher_feat, dim=1)
        return F.mse_loss(sf, tf) + (1 - F.cosine_similarity(sf, tf).mean())


class GeoPartLoss(nn.Module):
    """Baseline losses + tiny part CE contribution."""
    def __init__(self, cfg=Config):
        super().__init__(); self.cfg = cfg
        self.ce           = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.triplet      = TripletLoss(cfg.MARGIN)
        self.nce          = SymmetricInfoNCELoss()
        self.self_dist    = SelfDistillationLoss(cfg.TEMPERATURE)
        self.uapa         = UAPALoss(cfg.BASE_TEMPERATURE)
        self.cross_dist   = CrossDistillationLoss()

    def forward(self, drone_out, sat_out, labels, teacher_drone=None, teacher_sat=None):
        L: Dict[str, Any] = {}

        # --- Baseline CE (same formula as baseline) ---
        ce = 0.0
        for logits in drone_out['stage_logits']: ce += 0.25 * self.ce(logits, labels)
        ce += self.ce(drone_out['logits'], labels)
        for logits in sat_out['stage_logits']:   ce += 0.25 * self.ce(logits, labels)
        ce += self.ce(sat_out['logits'], labels)
        L['ce'] = ce

        # --- Baseline triplet ---
        L['triplet'] = self.cfg.LAMBDA_TRIPLET * (
            self.triplet(drone_out['embedding_normed'], labels) +
            self.triplet(sat_out['embedding_normed'], labels))

        # --- Baseline NCE (symmetric contrastive) ---
        L['csc'] = self.cfg.LAMBDA_CSC * self.nce(
            drone_out['embedding_normed'], sat_out['embedding_normed'], labels)

        # --- Baseline self-distillation ---
        L['self_dist'] = self.cfg.LAMBDA_SELF_DIST * (
            self.self_dist(drone_out['stage_logits']) +
            self.self_dist(sat_out['stage_logits']))

        # --- Baseline UAPA ---
        L['uapa'] = self.cfg.LAMBDA_ALIGN * self.uapa(drone_out['logits'], sat_out['logits'])

        # --- Baseline cross-distillation ---
        if teacher_drone is not None:
            L['cross_dist'] = self.cfg.LAMBDA_CROSS_DIST * (
                self.cross_dist(drone_out['final_feature'], teacher_drone) +
                self.cross_dist(sat_out['final_feature'], teacher_sat))

        # --- NOVEL: Part CE (very lightly weighted) ---
        part_ce = 0.0
        for pl in drone_out['part_logits']: part_ce += self.ce(pl, labels)
        for pl in sat_out['part_logits']:   part_ce += self.ce(pl, labels)
        L['part_ce'] = self.cfg.LAMBDA_PART_CE * part_ce  # 0.05 * 32 terms

        total = sum(L.values()); L['total'] = total
        return total, L


# =============================================================================
# EVALUATION (identical to baseline)
# =============================================================================
def evaluate(model, test_dataset, device):
    model.eval()
    loader = DataLoader(test_dataset, Config.BATCH_SIZE, False,
                        num_workers=Config.NUM_WORKERS, pin_memory=True)
    drone_feats, drone_labels = [], []
    with torch.no_grad():
        for batch in loader:
            f, _ = model(batch['drone'].to(device), batch['altitude'].to(device))
            drone_feats.append(f.cpu()); drone_labels.append(batch['label'])
    drone_feats = torch.cat(drone_feats); drone_labels = torch.cat(drone_labels)

    transform = get_transforms("test")
    sat_dir = os.path.join(test_dataset.root, Config.SATELLITE_DIR)
    sat_feats, sat_labels = [], []
    for loc in [f"{l:04d}" for l in Config.TRAIN_LOCS + Config.TEST_LOCS]:
        sp = os.path.join(sat_dir, loc, "0.png")
        if not os.path.exists(sp): continue
        t = transform(Image.open(sp).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad(): f, _ = model(t)
        sat_feats.append(f.cpu())
        sat_labels.append(test_dataset.location_to_idx.get(loc, -1 - len(sat_labels)))
    sat_feats = torch.cat(sat_feats); sat_labels = torch.tensor(sat_labels)

    _, idx = (drone_feats @ sat_feats.T).sort(1, descending=True)
    N = len(drone_feats); r1 = r5 = r10 = 0; ap = 0.
    for i in range(N):
        correct = torch.where(sat_labels[idx[i]] == drone_labels[i])[0]
        if len(correct) == 0: continue
        fc = correct[0].item()
        if fc < 1: r1 += 1
        if fc < 5: r5 += 1
        if fc < 10: r10 += 1
        ap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(correct)) / len(correct)
    return {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100}, ap/N*100


# =============================================================================
# TRAINING (baseline structure, no GradScaler changes since no SAM)
# =============================================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer; self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs; self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups: pg['lr'] = lr
        return lr


def train_one_epoch(model, teacher, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    if teacher is not None: teacher.eval()
    total = 0.; loss_sum = defaultdict(float)

    for bi, batch in enumerate(loader):
        drone = batch['drone'].to(device); sat = batch['satellite'].to(device)
        labels = batch['label'].to(device); alts = batch['altitude'].to(device)
        optimizer.zero_grad()

        with autocast(enabled=Config.USE_AMP):
            drone_out = model(drone, alts, return_all=True)
            sat_out   = model(sat,   None, return_all=True)
            td, ts = None, None
            if teacher is not None:
                with torch.no_grad(): td = teacher(drone); ts = teacher(sat)
            loss, loss_dict = criterion(drone_out, sat_out, labels, td, ts)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()

        total += loss.item()
        for k, v in loss_dict.items():
            loss_sum[k] += v.item() if torch.is_tensor(v) else v
        if bi % 10 == 0:
            gate = drone_out['part_gate']
            print(f"  B{bi}/{len(loader)} L={loss.item():.4f} gate={gate:.4f}")

    n = max(1, len(loader))
    return total / n, {k: v / n for k, v in loss_sum.items()}


def main():
    global Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=Config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=Config.BATCH_SIZE)
    parser.add_argument("--data_root",  type=str,   default=Config.DATA_ROOT)
    parser.add_argument("--test",       action="store_true")
    args, _ = parser.parse_known_args()
    Config.NUM_EPOCHS = args.epochs; Config.BATCH_SIZE = args.batch_size
    Config.DATA_ROOT  = args.data_root

    if args.test:
        Config.NUM_EPOCHS = 1; Config.NUM_WORKERS = 0
        Config.BATCH_SIZE = 8; Config.P = 2
    Config.K = max(2, Config.BATCH_SIZE // Config.P)

    print("=" * 60); print("GeoPart: Multi-Granularity Parts — SUES-200"); print("=" * 60)
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    train_ds = SUES200Dataset(Config.DATA_ROOT, "train", transform=get_transforms("train"))
    test_ds  = SUES200Dataset(Config.DATA_ROOT, "test",  transform=get_transforms("test"))
    num_classes = len(Config.TRAIN_LOCS)
    loader = DataLoader(train_ds, batch_sampler=PKSampler(train_ds, Config.P, Config.K),
                        num_workers=Config.NUM_WORKERS, pin_memory=True)

    model = GeoPartStudent(num_classes).to(device)
    print(f"  Student: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    try:    teacher = DINOv2Teacher().to(device); teacher.eval()
    except: teacher = None; print("No teacher")

    criterion = GeoPartLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.LR, momentum=0.9, weight_decay=5e-4)
    scheduler = WarmupCosineScheduler(optimizer, Config.WARMUP_EPOCHS, Config.NUM_EPOCHS)
    scaler    = GradScaler(enabled=Config.USE_AMP)

    best_r1 = 0.
    for epoch in range(Config.NUM_EPOCHS):
        lr = scheduler.step(epoch)
        print(f"\n{'='*40}\nEpoch {epoch+1}/{Config.NUM_EPOCHS}, LR={lr:.6f}\n{'='*40}")

        avg_loss, loss_dict = train_one_epoch(model, teacher, loader, criterion,
                                              optimizer, scaler, device, epoch)
        print(f"  AvgL={avg_loss:.4f}")
        for k, v in sorted(loss_dict.items()):
            if k != 'total': print(f"  {k}: {v:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == Config.NUM_EPOCHS - 1:
            rec, ap = evaluate(model, test_ds, device)
            print(f"  R@1:{rec['R@1']:.2f}% R@5:{rec['R@5']:.2f}% R@10:{rec['R@10']:.2f}% AP:{ap:.2f}%")
            if rec['R@1'] > best_r1:
                best_r1 = rec['R@1']
                torch.save({'epoch': epoch, 'model': model.state_dict(), 'r1': best_r1},
                           os.path.join(Config.OUTPUT_DIR, 'geopart_best.pth'))
                print(f"  *** Best R@1={best_r1:.2f}% ***")

    print(f"\nDone! Best R@1={best_r1:.2f}%")


if __name__ == "__main__": main()
