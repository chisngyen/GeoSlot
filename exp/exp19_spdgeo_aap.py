# =============================================================================
# EXP19: SPDGeo-AAP — Altitude-Adaptive Parts
# =============================================================================
# Base:    SPDGeo-D (90.36% R@1) — DINOv2-S + DINOv2-B teacher + 7 losses
# Novel:   1) FiLM-Conditioned Part Discovery — altitude modulates part prototypes
#             via Feature-wise Linear Modulation (γ, β per altitude per part)
#          2) Altitude-Consistency Loss — same location at different altitudes
#             should produce similar GLOBAL embeddings despite part-weight shifts
#          3) Altitude-Aware Salience Reweighting — adjusts part importance
#             based on altitude (low alt → local detail parts matter more,
#             high alt → global layout parts matter more)
#
# Motivation:
#   SPDGeo-D treats all altitudes identically: the same 8 prototypes weight
#   patches the same way whether the drone is at 150m or 300m. But the
#   visual content at different altitudes is fundamentally different:
#     • 150m: Fine-grained textures (windows, cars, trees) are visible
#     • 300m: Global layout (road networks, building footprints) dominates
#   By conditioning the part discovery on altitude, we can:
#     (a) Shift prototype semantics to match altitude-specific content
#     (b) Reweight salience so detail parts dominate low-alt, layout parts dominate high-alt
#     (c) Still produce altitude-invariant embeddings via a consistency loss
#   This directly addresses the 15% R@1 gap (150m vs 300m) seen in SPDGeo-D.
#
# Key insight: FiLM conditioning (Perez et al., 2018) is a lightweight way to
#   modulate features without adding separate branches — just γ*x + β per altitude.
#   Applied to part features AFTER aggregation, so the shared prototypes still
#   learn general semantics while the modulation handles altitude-specific emphasis.
#   Inference for satellite (no altitude) uses the MEAN of all altitude modulations.
#
# Architecture: SPDGeo-D + AltitudeFiLM + AltitudeSalienceReweight
# Total losses: 8 (7 from SPDGeo-D + 1 new AltitudeConsistencyLoss)
# =============================================================================

import subprocess, sys
for _p in ["timm", "tqdm"]:
    try: __import__(_p)
    except ImportError: subprocess.run(f"pip install -q {_p}", shell=True, capture_output=True)

import os, math, json, gc, random, copy
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    SUES_ROOT       = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    DRONE_DIR       = "drone-view"
    SAT_DIR         = "satellite-view"
    OUTPUT_DIR      = "/kaggle/working"
    ALTITUDES       = ["150", "200", "250", "300"]
    TRAIN_LOCS      = list(range(1, 121))
    TEST_LOCS       = list(range(121, 201))
    NUM_CLASSES     = 120

    IMG_SIZE        = 336
    N_PARTS         = 8
    PART_DIM        = 256
    EMBED_DIM       = 512
    TEACHER_DIM     = 768
    CLUSTER_TEMP    = 0.07
    UNFREEZE_BLOCKS = 4

    NUM_EPOCHS      = 120
    P_CLASSES       = 16
    K_SAMPLES       = 4
    LR              = 3e-4
    BACKBONE_LR     = 3e-5
    WEIGHT_DECAY    = 0.01
    WARMUP_EPOCHS   = 5
    USE_AMP         = True
    SEED            = 42

    # Base loss weights (from SPDGeo-D)
    LAMBDA_CE           = 1.0
    LAMBDA_INFONCE      = 1.0
    LAMBDA_TRIPLET      = 0.5
    LAMBDA_CONSISTENCY  = 0.1
    LAMBDA_CROSS_DIST   = 0.3
    LAMBDA_SELF_DIST    = 0.3
    LAMBDA_UAPA         = 0.2

    # NEW: Altitude-adaptive weights
    LAMBDA_ALT_CONSIST  = 0.2   # altitude consistency loss
    NUM_ALTITUDES       = 4     # 150, 200, 250, 300

    DISTILL_TEMP        = 4.0
    EVAL_INTERVAL       = 5
    NUM_WORKERS         = 2

CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Altitude index mapping
ALT_TO_IDX = {150: 0, 200: 1, 250: 2, 300: 3}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


# =============================================================================
# DATASET
# =============================================================================
class SUES200Dataset(Dataset):
    def __init__(self, root, mode="train", altitudes=None, transform=None):
        self.root = root; self.mode = mode
        self.altitudes = altitudes or CFG.ALTITUDES
        self.transform = transform
        self.drone_dir = os.path.join(root, CFG.DRONE_DIR)
        self.sat_dir   = os.path.join(root, CFG.SAT_DIR)
        loc_ids = CFG.TRAIN_LOCS if mode == "train" else CFG.TEST_LOCS
        self.locations       = [f"{l:04d}" for l in loc_ids]
        self.location_to_idx = {l: i for i, l in enumerate(self.locations)}
        self.samples = []; self.drone_by_location = defaultdict(list)
        for loc in self.locations:
            li = self.location_to_idx[loc]
            sp = os.path.join(self.sat_dir, loc, "0.png")
            if not os.path.exists(sp): continue
            for alt in self.altitudes:
                ad = os.path.join(self.drone_dir, loc, alt)
                if not os.path.isdir(ad): continue
                for img in sorted(os.listdir(ad)):
                    if img.endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(ad, img), sp, li, alt))
                        self.drone_by_location[li].append(len(self.samples) - 1)
        print(f"  [{mode}] {len(self.samples)} samples, {len(self.locations)} locations")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        dp, sp, li, alt = self.samples[idx]
        try:
            d = Image.open(dp).convert('RGB'); s = Image.open(sp).convert('RGB')
        except Exception:
            sz = CFG.IMG_SIZE
            d = Image.new('RGB', (sz, sz), (128, 128, 128))
            s = Image.new('RGB', (sz, sz), (128, 128, 128))
        if self.transform: d = self.transform(d); s = self.transform(s)
        return {'drone': d, 'satellite': s, 'label': li, 'altitude': int(alt)}


class PKSampler:
    def __init__(self, ds, p, k): self.ds = ds; self.p = p; self.k = k; self.locs = list(ds.drone_by_location.keys())
    def __iter__(self):
        locs = self.locs.copy(); random.shuffle(locs); batch = []
        for l in locs:
            idx = self.ds.drone_by_location[l]
            if len(idx) < self.k: idx = idx * (self.k // len(idx) + 1)
            batch.extend(random.sample(idx, self.k))
            if len(batch) >= self.p * self.k: yield batch[:self.p * self.k]; batch = batch[self.p * self.k:]
    def __len__(self): return len(self.locs) // self.p


def get_transforms(mode="train"):
    sz = CFG.IMG_SIZE
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((sz, sz)), transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(sz, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05), transforms.RandomGrayscale(p=0.02),
            transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((sz, sz)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])


# =============================================================================
# BACKBONE — DINOv2 ViT-S/14
# =============================================================================
class DINOv2Backbone(nn.Module):
    def __init__(self, unfreeze_blocks=4):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.feature_dim = 384; self.patch_size = 14
        for p in self.model.parameters(): p.requires_grad = False
        for blk in self.model.blocks[-unfreeze_blocks:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in self.model.norm.parameters(): p.requires_grad = True
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  DINOv2 ViT-S/14: {frozen/1e6:.1f}M frozen, {trainable/1e6:.1f}M trainable")

    def forward(self, x):
        features = self.model.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']
        cls_token = features['x_norm_clstoken']
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return patch_tokens, cls_token, (H, W)


# =============================================================================
# TEACHER — DINOv2 ViT-B/14
# =============================================================================
class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        print("  Loading DINOv2 ViT-B/14 teacher …")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        self.output_dim = 768
        for p in self.model.parameters(): p.requires_grad = False
        print("  DINOv2 ViT-B/14 teacher loaded (all frozen).")

    @torch.no_grad()
    def forward(self, x):
        return self.model.forward_features(x)['x_norm_clstoken']


# =============================================================================
# SEMANTIC PART DISCOVERY
# =============================================================================
class SemanticPartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07):
        super().__init__()
        self.n_parts = n_parts; self.temperature = temperature
        self.feat_proj = nn.Sequential(nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU())
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim * 2),
                                    nn.GELU(), nn.Linear(part_dim * 2, part_dim))
        self.salience_head = nn.Sequential(nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, patch_features, spatial_hw):
        B, N, _ = patch_features.shape; H, W = spatial_hw
        feat = self.feat_proj(patch_features)
        feat_norm = F.normalize(feat, dim=-1); proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum('bnd,kd->bnk', feat_norm, proto_norm) / self.temperature
        assign = F.softmax(sim, dim=-1)
        assign_t = assign.transpose(1, 2)
        mass = assign_t.sum(-1, keepdim=True).clamp(min=1e-6)
        part_feat = torch.bmm(assign_t, feat) / mass
        part_feat = part_feat + self.refine(part_feat)
        device = feat.device
        gy = torch.arange(H, device=device).float() / max(H - 1, 1)
        gx = torch.arange(W, device=device).float() / max(W - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(B, -1, -1)) / mass
        salience = self.salience_head(part_feat).squeeze(-1)
        return {'part_features': part_feat, 'part_positions': part_pos,
                'assignment': assign, 'salience': salience}


# =============================================================================
# PART-AWARE POOLING
# =============================================================================
class PartAwarePooling(nn.Module):
    def __init__(self, part_dim=256, embed_dim=512):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(part_dim, part_dim // 2), nn.Tanh(), nn.Linear(part_dim // 2, 1))
        self.proj = nn.Sequential(nn.Linear(part_dim * 3, embed_dim), nn.LayerNorm(embed_dim),
                                  nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, part_features, salience=None):
        B, K, D = part_features.shape
        aw = self.attn(part_features)
        if salience is not None: aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        attn_pool = (aw * part_features).sum(1)
        mean_pool = part_features.mean(1); max_pool = part_features.max(1)[0]
        combined = torch.cat([attn_pool, mean_pool, max_pool], dim=-1)
        return F.normalize(self.proj(combined), dim=-1)


# =============================================================================
# NEW: Altitude FiLM Conditioning
# =============================================================================
class AltitudeFiLM(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on altitude.

    For each altitude a, learns γ_a and β_a vectors that scale and shift
    part features: output = γ_a * part_feat + β_a

    This allows the same 8 semantic prototypes to emphasize different aspects
    at different altitudes without needing separate part discovery branches.

    For satellite images (no altitude), uses the MEAN of all altitude
    modulation parameters, providing an altitude-agnostic representation.
    """
    def __init__(self, n_altitudes=4, feat_dim=256):
        super().__init__()
        # Learnable per-altitude scale and shift
        self.gamma = nn.Parameter(torch.ones(n_altitudes, feat_dim))
        self.beta  = nn.Parameter(torch.zeros(n_altitudes, feat_dim))

    def forward(self, part_features, altitude_idx=None):
        """
        part_features: [B, K, D]
        altitude_idx:  [B] int tensor (0–3) or None (satellite → use mean)
        Returns:       [B, K, D] modulated features
        """
        if altitude_idx is None:
            # Satellite: use mean across all altitudes
            gamma = self.gamma.mean(0)  # [D]
            beta  = self.beta.mean(0)   # [D]
            return part_features * gamma.unsqueeze(0).unsqueeze(0) + beta.unsqueeze(0).unsqueeze(0)
        else:
            # Drone: per-sample altitude conditioning
            gamma = self.gamma[altitude_idx]  # [B, D]
            beta  = self.beta[altitude_idx]   # [B, D]
            return part_features * gamma.unsqueeze(1) + beta.unsqueeze(1)


# =============================================================================
# NEW: Altitude-Aware Salience Reweighting
# =============================================================================
class AltitudeSalienceReweight(nn.Module):
    """
    Learns an altitude-dependent bias on per-part salience scores.

    Intuition: at 150m, fine-detail parts (vegetation texture, building edges)
    should have higher salience; at 300m, layout parts (road network, block shape)
    should dominate. This module adds a learned altitude-specific bias to the
    base salience, steering the pooling attention without overwriting learned salience.
    """
    def __init__(self, n_altitudes=4, n_parts=8):
        super().__init__()
        # Learnable per-altitude, per-part salience bias (init small)
        self.bias = nn.Parameter(torch.zeros(n_altitudes, n_parts))

    def forward(self, salience, altitude_idx=None):
        """
        salience:     [B, K] — base salience from SemanticPartDiscovery
        altitude_idx: [B] or None
        Returns:      [B, K] — reweighted salience (still in [0, 1])
        """
        if altitude_idx is None:
            # Satellite: mean bias across altitudes
            bias = self.bias.mean(0)  # [K]
            return torch.sigmoid(salience.logit().clamp(-10, 10) + bias.unsqueeze(0))
        else:
            bias = self.bias[altitude_idx]  # [B, K]
            return torch.sigmoid(salience.logit().clamp(-10, 10) + bias)


# =============================================================================
# STUDENT MODEL — SPDGeo-AAP
# =============================================================================
class SPDGeoAAPModel(nn.Module):
    """
    SPDGeo-D + Altitude-Adaptive Parts.

    Key changes from SPDGeo-D:
    1. AltitudeFiLM modulates part features based on altitude BEFORE pooling
    2. AltitudeSalienceReweight adjusts salience based on altitude
    3. Both are lightweight (4×D + 4×K params) and inference-compatible
       (satellite uses mean-altitude params, no altitude input needed)
    """
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone  = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = SemanticPartDiscovery(384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP)
        self.pool      = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)

        # NEW: altitude conditioning modules
        self.alt_film     = AltitudeFiLM(cfg.NUM_ALTITUDES, cfg.PART_DIM)
        self.alt_salience = AltitudeSalienceReweight(cfg.NUM_ALTITUDES, cfg.N_PARTS)

        # Classification branches (same as SPDGeo-D)
        self.bottleneck = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                        nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj   = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
                                            nn.LayerNorm(cfg.TEACHER_DIM))
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-AAP student: {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x, altitude_idx=None):
        """Fused embedding for retrieval. altitude_idx=None → satellite."""
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        # Apply altitude conditioning
        mod_feats = self.alt_film(parts['part_features'], altitude_idx)
        mod_sal   = self.alt_salience(parts['salience'], altitude_idx)
        emb = self.pool(mod_feats, mod_sal)
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return F.normalize(0.7 * emb + 0.3 * cls_emb, dim=-1)

    def forward(self, x, altitude_idx=None, return_parts=False):
        """
        altitude_idx: [B] int tensor with values in {0,1,2,3} or None (satellite).
        """
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)

        # Altitude-adaptive modulation
        mod_feats = self.alt_film(parts['part_features'], altitude_idx)
        mod_sal   = self.alt_salience(parts['salience'], altitude_idx)

        emb = self.pool(mod_feats, mod_sal)
        bn = self.bottleneck(emb); logits = self.classifier(bn)
        cls_emb = self.cls_proj(cls_tok); cls_logits = self.cls_classifier(cls_emb)
        fused = F.normalize(0.7 * emb + 0.3 * F.normalize(cls_emb, dim=-1), dim=-1)
        projected_feat = self.teacher_proj(emb)
        out = {'embedding': fused, 'logits': logits, 'cls_logits': cls_logits,
               'projected_feat': projected_feat}
        if return_parts:
            out['parts'] = parts
            out['mod_part_features'] = mod_feats
            out['mod_salience'] = mod_sal
        return out


# =============================================================================
# BASE LOSSES (from SPDGeo-D)
# =============================================================================
class SupInfoNCELoss(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(temp).log())

    def forward(self, q_emb, r_emb, labels):
        t = self.log_t.exp().clamp(0.01, 1.0)
        sim = q_emb @ r_emb.t() / t
        labels = labels.view(-1, 1); pos_mask = labels.eq(labels.T).float()
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        return (-(log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)).mean()


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3): super().__init__(); self.m = margin
    def forward(self, emb, labels):
        d = torch.cdist(emb, emb, p=2); labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float(); neg_mask = labels.ne(labels.T).float()
        hard_pos = (d * pos_mask).max(1)[0]
        hard_neg = (d * neg_mask + pos_mask * 1e9).min(1)[0]
        return F.relu(hard_pos - hard_neg + self.m).mean()


class PartConsistencyLoss(nn.Module):
    def forward(self, assign_q, assign_r):
        dist_q = assign_q.mean(dim=1); dist_r = assign_r.mean(dim=1)
        kl_qr = F.kl_div((dist_q + 1e-8).log(), dist_r, reduction='batchmean', log_target=False)
        kl_rq = F.kl_div((dist_r + 1e-8).log(), dist_q, reduction='batchmean', log_target=False)
        return 0.5 * (kl_qr + kl_rq)


class CrossDistillationLoss(nn.Module):
    def forward(self, student_feat, teacher_feat):
        s = F.normalize(student_feat, dim=-1); t = F.normalize(teacher_feat, dim=-1)
        return F.mse_loss(s, t) + (1.0 - F.cosine_similarity(s, t).mean())


class SelfDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0): super().__init__(); self.T = temperature
    def forward(self, weak_logits, strong_logits):
        p_teacher = F.softmax(strong_logits / self.T, dim=1).detach()
        p_student = F.log_softmax(weak_logits / self.T, dim=1)
        return (self.T ** 2) * F.kl_div(p_student, p_teacher, reduction='batchmean')


class UAPALoss(nn.Module):
    def __init__(self, base_temperature=4.0): super().__init__(); self.T0 = base_temperature
    @staticmethod
    def _entropy(logits):
        probs = F.softmax(logits, dim=1)
        return -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
    def forward(self, drone_logits, sat_logits):
        delta_U = self._entropy(drone_logits) - self._entropy(sat_logits)
        T = self.T0 * (1 + torch.sigmoid(delta_U))
        p_sat = F.softmax(sat_logits / T, dim=1).detach()
        return (T ** 2) * F.kl_div(F.log_softmax(drone_logits / T, dim=1), p_sat, reduction='batchmean')


# =============================================================================
# NEW LOSS — Altitude Consistency
# =============================================================================
class AltitudeConsistencyLoss(nn.Module):
    """
    Encourages embeddings from the SAME location at DIFFERENT altitudes
    to be close in feature space.

    Within each PK batch, groups samples by location. For locations that have
    multiple altitudes in the batch, computes mean embedding per altitude
    and minimizes pairwise cosine distance between them.

    This prevents the FiLM conditioning from pushing different-altitude views
    of the same location apart — the final embedding should be altitude-invariant
    even though intermediate part representations are altitude-adapted.
    """
    def forward(self, embeddings, labels, altitudes):
        """
        embeddings: [B, D] L2-normed
        labels:     [B] location indices
        altitudes:  [B] altitude values (150/200/250/300)
        """
        device = embeddings.device
        unique_locs = labels.unique()
        loss = torch.tensor(0.0, device=device)
        count = 0

        for loc in unique_locs:
            loc_mask = labels == loc
            if loc_mask.sum() < 2:
                continue
            loc_embs = embeddings[loc_mask]   # [n, D]
            loc_alts = altitudes[loc_mask]    # [n]

            # Get unique altitudes for this location
            unique_alts = loc_alts.unique()
            if len(unique_alts) < 2:
                continue

            # Mean embedding per altitude
            alt_means = []
            for alt in unique_alts:
                alt_mask = loc_alts == alt
                alt_means.append(loc_embs[alt_mask].mean(0))
            alt_means = torch.stack(alt_means)  # [A, D]
            alt_means = F.normalize(alt_means, dim=-1)

            # Pairwise cosine distance between altitude means
            A = alt_means.size(0)
            for i in range(A):
                for j in range(i + 1, A):
                    loss += (1 - F.cosine_similarity(alt_means[i:i+1], alt_means[j:j+1]).squeeze())
                    count += 1

        if count > 0:
            loss = loss / count
        return loss


# =============================================================================
# EVALUATION — full 200-location gallery
# =============================================================================
@torch.no_grad()
def evaluate(model, test_ds, device):
    model.eval()
    test_tf = get_transforms("test")

    loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)
    drone_feats, drone_labels, drone_alts = [], [], []
    for b in loader:
        alt_idx = torch.tensor([ALT_TO_IDX[a.item()] for a in b['altitude']], device=device)
        feat = model.extract_embedding(b['drone'].to(device), altitude_idx=alt_idx).cpu()
        drone_feats.append(feat); drone_labels.append(b['label']); drone_alts.append(b['altitude'])
    drone_feats = torch.cat(drone_feats); drone_labels = torch.cat(drone_labels)
    drone_alts = torch.cat(drone_alts)

    # Satellite gallery — no altitude → altitude_idx=None
    all_locs = [f"{l:04d}" for l in range(1, 201)]
    sat_img_list, sat_label_list = [], []; distractor_cnt = 0
    for loc in all_locs:
        sp = os.path.join(test_ds.sat_dir, loc, "0.png")
        if not os.path.exists(sp): continue
        sat_img_list.append(test_tf(Image.open(sp).convert('RGB')))
        if loc in test_ds.location_to_idx: sat_label_list.append(test_ds.location_to_idx[loc])
        else: sat_label_list.append(-1000 - distractor_cnt); distractor_cnt += 1

    sat_feats = []
    for i in range(0, len(sat_img_list), 64):
        batch = torch.stack(sat_img_list[i:i+64]).to(device)
        sat_feats.append(model.extract_embedding(batch, altitude_idx=None).cpu())
    sat_feats = torch.cat(sat_feats); sat_labels = torch.tensor(sat_label_list)

    print(f"  Gallery: {len(sat_feats)} sat imgs | Queries: {len(drone_feats)} drone imgs")

    sim = drone_feats @ sat_feats.T; _, rank = sim.sort(1, descending=True)
    N = drone_feats.size(0); r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(sat_labels[rank[i]] == drone_labels[i])[0]
        if len(matches) == 0: continue
        first = matches[0].item()
        if first < 1: r1 += 1
        if first < 5: r5 += 1
        if first < 10: r10 += 1
        ap += sum((j+1)/(p.item()+1) for j, p in enumerate(matches)) / len(matches)

    overall = {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100, 'mAP': ap/N*100}

    # Per-altitude breakdown
    altitudes_list = sorted(drone_alts.unique().tolist())
    per_alt = {}
    for alt in altitudes_list:
        mask = drone_alts == alt
        if mask.sum() == 0: continue
        af = drone_feats[mask]; al = drone_labels[mask]
        s = af @ sat_feats.T; _, rk = s.sort(1, descending=True)
        n = af.size(0); a1 = a5 = a10 = aap = 0
        for i in range(n):
            m = torch.where(sat_labels[rk[i]] == al[i])[0]
            if len(m) == 0: continue
            f = m[0].item()
            if f < 1: a1 += 1
            if f < 5: a5 += 1
            if f < 10: a10 += 1
            aap += sum((j+1)/(p.item()+1) for j, p in enumerate(m)) / len(m)
        per_alt[int(alt)] = {'R@1': a1/n*100, 'R@5': a5/n*100, 'R@10': a10/n*100, 'mAP': aap/n*100, 'n': n}

    print(f"\n{'='*75}")
    print(f"  Gallery: {len(sat_feats)} satellite images | Queries: {len(drone_feats)} drone images")
    print(f"{'='*75}")
    print(f"  {'Altitude':>8s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'#Query':>6s}")
    print(f"  {'-'*50}")
    for alt in altitudes_list:
        a = per_alt[int(alt)]
        print(f"  {int(alt):>6d}m  {a['R@1']:6.2f}%  {a['R@5']:6.2f}%  {a['R@10']:6.2f}%  {a['mAP']:6.2f}%  {a['n']:>6d}")
    print(f"  {'-'*50}")
    print(f"  {'Overall':>8s}  {overall['R@1']:6.2f}%  {overall['R@5']:6.2f}%  {overall['R@10']:6.2f}%  {overall['mAP']:6.2f}%  {N:>6d}")
    print(f"{'='*75}\n")

    return overall, per_alt


# =============================================================================
# TRAINING
# =============================================================================
def train_one_epoch(model, teacher, loader, losses, alt_consist_loss, optimizer,
                    scaler, device, epoch):
    model.train()
    if teacher: teacher.eval()

    infonce, triplet, ce, consist, cross_dist, self_dist, uapa = losses

    total_sum = 0; n = 0; loss_sums = defaultdict(float)

    for batch in tqdm(loader, desc=f'Ep{epoch:3d}', leave=False):
        drone  = batch['drone'].to(device)
        sat    = batch['satellite'].to(device)
        labels = batch['label'].to(device)
        alts   = batch['altitude'].to(device)

        # Map altitude values → indices
        alt_idx = torch.zeros_like(alts)
        for val, idx in ALT_TO_IDX.items():
            alt_idx[alts == val] = idx

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            d_out = model(drone, altitude_idx=alt_idx, return_parts=True)
            s_out = model(sat, altitude_idx=None, return_parts=True)  # sat → no altitude

            # === Base losses (7 from SPDGeo-D) ===
            l_ce = (ce(d_out['logits'], labels) + ce(s_out['logits'], labels))
            l_ce += 0.3 * (ce(d_out['cls_logits'], labels) + ce(s_out['cls_logits'], labels))
            l_nce = infonce(d_out['embedding'], s_out['embedding'], labels)
            l_tri = 0.5 * (triplet(d_out['embedding'], labels) + triplet(s_out['embedding'], labels))
            l_con = consist(d_out['parts']['assignment'], s_out['parts']['assignment'])

            if teacher is not None:
                with torch.no_grad():
                    t_drone = teacher(drone); t_sat = teacher(sat)
                l_cross = cross_dist(d_out['projected_feat'], t_drone) + cross_dist(s_out['projected_feat'], t_sat)
            else:
                l_cross = torch.tensor(0.0, device=device)

            l_self = self_dist(d_out['cls_logits'], d_out['logits']) + self_dist(s_out['cls_logits'], s_out['logits'])
            l_uapa = uapa(d_out['logits'], s_out['logits'])

            # === NEW: Altitude consistency loss ===
            l_alt = alt_consist_loss(d_out['embedding'], labels, alts)

            # === Total loss ===
            loss = (CFG.LAMBDA_CE          * l_ce +
                    CFG.LAMBDA_INFONCE     * l_nce +
                    CFG.LAMBDA_TRIPLET     * l_tri +
                    CFG.LAMBDA_CONSISTENCY * l_con +
                    CFG.LAMBDA_CROSS_DIST  * l_cross +
                    CFG.LAMBDA_SELF_DIST   * l_self +
                    CFG.LAMBDA_UAPA        * l_uapa +
                    CFG.LAMBDA_ALT_CONSIST * l_alt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()

        total_sum += loss.item(); n += 1
        loss_sums['ce']    += l_ce.item()
        loss_sums['nce']   += l_nce.item()
        loss_sums['tri']   += l_tri.item()
        loss_sums['con']   += l_con.item()
        loss_sums['cross'] += l_cross.item() if torch.is_tensor(l_cross) else l_cross
        loss_sums['self']  += l_self.item()
        loss_sums['uapa']  += l_uapa.item()
        loss_sums['alt']   += l_alt.item()

    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EXP19: SPDGeo-AAP — Altitude-Adaptive Parts")
    print(f"  Base: SPDGeo-D (90.36% R@1)")
    print(f"  Novel: AltitudeFiLM + AltitudeSalienceReweight + AltitudeConsistencyLoss")
    print(f"  Dataset: SUES-200 | Epochs: {CFG.NUM_EPOCHS} | Device: {DEVICE}")
    print(f"  Parts: {CFG.N_PARTS} | Img: {CFG.IMG_SIZE} | Embed: {CFG.EMBED_DIM}")
    print(f"  Losses: 7 base + 1 new = 8 total")
    print("=" * 65)

    print('\nLoading SUES-200 …')
    train_ds = SUES200Dataset(CFG.SUES_ROOT, 'train', transform=get_transforms("train"))
    test_ds  = SUES200Dataset(CFG.SUES_ROOT, 'test', transform=get_transforms("test"))
    train_loader = DataLoader(train_ds, batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print('\nBuilding models …')
    model = SPDGeoAAPModel(CFG.NUM_CLASSES).to(DEVICE)
    teacher = None
    try:
        teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except Exception as e:
        print(f"  [WARN] Could not load teacher: {e}")

    # Base losses
    infonce    = SupInfoNCELoss(temp=0.05).to(DEVICE)
    triplet    = TripletLoss(margin=0.3)
    ce         = nn.CrossEntropyLoss(label_smoothing=0.1)
    consist    = PartConsistencyLoss()
    cross_dist = CrossDistillationLoss()
    self_dist  = SelfDistillationLoss(temperature=CFG.DISTILL_TEMP)
    uapa_loss  = UAPALoss(base_temperature=CFG.DISTILL_TEMP)
    base_losses = (infonce, triplet, ce, consist, cross_dist, self_dist, uapa_loss)

    # New loss
    alt_consist = AltitudeConsistencyLoss()

    # Optimizer
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith('backbone')]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params,      'lr': CFG.BACKBONE_LR},
        {'params': head_params,          'lr': CFG.LR},
        {'params': infonce.parameters(), 'lr': CFG.LR},
    ], weight_decay=CFG.WEIGHT_DECAY)

    scaler = torch.amp.GradScaler('cuda', enabled=CFG.USE_AMP)
    best_r1 = 0.0; results_log = []

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        if epoch <= CFG.WARMUP_EPOCHS:
            lr_scale = epoch / CFG.WARMUP_EPOCHS
        else:
            progress = (epoch - CFG.WARMUP_EPOCHS) / (CFG.NUM_EPOCHS - CFG.WARMUP_EPOCHS)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        lr_scale = max(lr_scale, 0.01)
        optimizer.param_groups[0]['lr'] = CFG.BACKBONE_LR * lr_scale
        optimizer.param_groups[1]['lr'] = CFG.LR * lr_scale
        optimizer.param_groups[2]['lr'] = CFG.LR * lr_scale
        cur_lr = optimizer.param_groups[1]['lr']

        avg_loss, ld = train_one_epoch(model, teacher, train_loader, base_losses,
                                       alt_consist, optimizer, scaler, DEVICE, epoch)

        print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
              f"CE {ld['ce']:.3f}  NCE {ld['nce']:.3f}  Tri {ld['tri']:.3f}  "
              f"Con {ld['con']:.3f}  Crs {ld['cross']:.3f}  Slf {ld['self']:.3f}  "
              f"UAPA {ld['uapa']:.3f}  Alt {ld['alt']:.3f} | LR {cur_lr:.2e}")

        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            metrics, per_alt = evaluate(model, test_ds, DEVICE)
            results_log.append({'epoch': epoch, **metrics})
            print(f"  ► R@1: {metrics['R@1']:.2f}%  R@5: {metrics['R@5']:.2f}%  "
                  f"R@10: {metrics['R@10']:.2f}%  mAP: {metrics['mAP']:.2f}%")
            if metrics['R@1'] > best_r1:
                best_r1 = metrics['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'metrics': metrics, 'per_alt': per_alt},
                           os.path.join(CFG.OUTPUT_DIR, 'exp19_aap_best.pth'))
                print(f"  ★ New best R@1: {best_r1:.2f}%!")

    print(f'\n{"="*65}')
    print(f'  EXP19: SPDGeo-AAP COMPLETE — Best R@1: {best_r1:.2f}%')
    print(f'{"="*65}')
    print(f'  {"Epoch":>6} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"mAP":>8}')
    print(f'  {"-"*44}')
    for r in results_log:
        print(f'  {r["epoch"]:6d} {r["R@1"]:8.2f} {r["R@5"]:8.2f} {r["R@10"]:8.2f} {r["mAP"]:8.2f}')
    print(f'{"="*65}')

    with open(os.path.join(CFG.OUTPUT_DIR, 'exp19_aap_results.json'), 'w') as f:
        json.dump({'results_log': results_log, 'best_r1': best_r1,
                   'config': {k: v for k, v in vars(CFG).items() if not k.startswith('_')}}, f, indent=2)


if __name__ == '__main__':
    main()
