# =============================================================================
# EXP25: SPDGeo-HYP — Hyperbolic Part Embeddings
# =============================================================================
# Base:    SPDGeo-D (90.36% R@1) — DINOv2-S + DINOv2-B teacher + 7 losses
# Novel:   1) Poincaré Ball Embedding — map L2-normalized Euclidean embeddings
#             into the Poincaré ball model of hyperbolic space with LEARNABLE
#             curvature c (initialized at 1.0)
#          2) Hyperbolic InfoNCE — replace cosine similarity with Poincaré
#             distance for contrastive learning; hyperbolic distance is more
#             expressive for hierarchical/tree-like structures
#          3) Curvature-Scheduled Training — start with low curvature (near
#             Euclidean) and gradually increase to full hyperbolic; prevents
#             unstable early training in curved space
#
# Motivation:
#   Geo-localization has natural HIERARCHICAL structure:
#   - Same location → different altitudes (150/200/250/300m) form a tree
#   - Nearby locations share visual features → neighborhood hierarchy
#   - Drone vs satellite → different abstraction levels
#
#   Euclidean space cannot efficiently embed hierarchical structures (requires
#   exponentially growing space). Hyperbolic space has exponential volume
#   growth → can embed trees with bounded distortion.
#
#   Peng et al. (ICML 2021), Ermolov et al. (NeurIPS 2022) showed hyperbolic
#   embeddings improve retrieval by 2-5% on hierarchical datasets.
#
#   For SUES-200: the altitude hierarchy (150→200→250→300m) creates a natural
#   tree where same-location images at different altitudes should cluster
#   hierarchically — images at 250 and 300m are "closer" than 150 and 300m.
#
# Key insight: We keep the Euclidean backbone and only map the FINAL embedding
#   to hyperbolic space. This is safe because:
#   1. DINOv2 features are already well-organized in Euclidean space
#   2. The exponential map is differentiable and numerically stable
#   3. Learnable curvature c adapts the "hyper-bolicity" to the data
#
# Architecture: SPDGeo-D + exponential map + Poincaré distance head
# Total losses: 9 (5 base + HypInfoNCE + HypTriplet + Consistency + UAPA)
#   Note: Replace Euclidean InfoNCE/Triplet with hyperbolic versions;
#         keep CE, Cross-Distill, Self-Distill in Euclidean (logit space)
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
    LAMBDA_HYP_NCE      = 1.0    # replaces LAMBDA_INFONCE
    LAMBDA_HYP_TRI      = 0.5    # replaces LAMBDA_TRIPLET
    LAMBDA_CONSISTENCY  = 0.1
    LAMBDA_CROSS_DIST   = 0.3
    LAMBDA_SELF_DIST    = 0.3
    LAMBDA_UAPA         = 0.2

    # NEW: Hyperbolic parameters
    HYP_INIT_C          = 1.0     # initial curvature
    HYP_CLIP_R          = 2.3     # max radius in Poincaré ball (tanh(2.3)≈0.98)
    CURV_WARMUP_EPOCHS  = 20      # epochs before full curvature

    DISTILL_TEMP        = 4.0
    EVAL_INTERVAL       = 5
    NUM_WORKERS         = 2

CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# HYPERBOLIC MATH — Poincaré Ball Operations
# =============================================================================

class PoincareBall:
    """
    Operations in the Poincaré ball model of hyperbolic space.

    The Poincaré ball B_c^n = {x ∈ R^n : c||x||^2 < 1} with curvature -c.

    Key properties:
    - Points near origin ≈ Euclidean (safe for initialization)
    - Points near boundary = "far in hyperbolic space" (infinite depth)
    - Distance grows logarithmically near boundary → efficient hierarchy

    All operations handle curvature c as a parameter (not fixed at 1).
    """

    @staticmethod
    def lambda_x(x, c):
        """Conformal factor λ_c(x) = 2 / (1 - c||x||^2)"""
        x_sqnorm = (x * x).sum(dim=-1, keepdim=True).clamp(max=1.0/c.item() - 1e-5)
        return 2.0 / (1.0 - c * x_sqnorm).clamp(min=1e-5)

    @staticmethod
    def expmap0(v, c):
        """
        Exponential map from tangent space at origin to Poincaré ball.
        exp_0^c(v) = tanh(√c · ||v|| / 2) · v / (√c · ||v||)

        Maps Euclidean vector v ∈ T_0 B → point on Poincaré ball.
        """
        sqrt_c = c.sqrt()
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        return torch.tanh(sqrt_c * v_norm / 2.0) * v / (sqrt_c * v_norm)

    @staticmethod
    def logmap0(y, c):
        """
        Logarithmic map from Poincaré ball to tangent space at origin.
        log_0^c(y) = atanh(√c · ||y||) · y / (√c · ||y||)
        """
        sqrt_c = c.sqrt()
        y_norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        return torch.atanh((sqrt_c * y_norm).clamp(max=1.0 - 1e-5)) * y / (sqrt_c * y_norm)

    @staticmethod
    def mobius_add(x, y, c):
        """
        Möbius addition in the Poincaré ball.
        x ⊕_c y
        """
        x_sqnorm = (x * x).sum(dim=-1, keepdim=True)
        y_sqnorm = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1 + 2*c*xy + c*y_sqnorm) * x + (1 - c*x_sqnorm) * y
        denom = 1 + 2*c*xy + c*c*x_sqnorm*y_sqnorm
        return num / denom.clamp(min=1e-5)

    @staticmethod
    def dist(x, y, c):
        """
        Poincaré distance: d_c(x, y) = (2/√c) · arctanh(√c · ||-x ⊕_c y||)

        Key property: grows logarithmically near boundary → captures hierarchy.
        """
        sqrt_c = c.sqrt()
        diff = PoincareBall.mobius_add(-x, y, c)
        diff_norm = diff.norm(dim=-1).clamp(max=1.0/sqrt_c.item() - 1e-5)
        return (2.0 / sqrt_c) * torch.atanh(sqrt_c * diff_norm)

    @staticmethod
    def proj(x, c, eps=1e-5):
        """Project to Poincaré ball (ensure ||x||^2 < 1/c)."""
        max_norm = (1.0 / c.sqrt()) - eps
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
        cond = x_norm > max_norm
        return torch.where(cond, x * max_norm / x_norm, x)


# =============================================================================
# STUDENT MODEL — SPDGeo-HYP
# =============================================================================
class SPDGeoHYPModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone  = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = SemanticPartDiscovery(384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP)
        self.pool      = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.bottleneck = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                        nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj   = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
                                            nn.LayerNorm(cfg.TEACHER_DIM))

        # Learnable curvature (initialized such that c = HYP_INIT_C)
        self.log_c = nn.Parameter(torch.tensor(math.log(cfg.HYP_INIT_C)))

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-HYP student: {total/1e6:.1f}M trainable parameters")
        print(f"  Hyperbolic: Poincaré ball, init c={cfg.HYP_INIT_C}")

    @property
    def c(self):
        """Curvature (always positive)."""
        return self.log_c.exp().clamp(0.1, 10.0)

    def to_hyperbolic(self, euclidean_emb):
        """
        Map Euclidean embedding → Poincaré ball via exponential map at origin.
        euclidean_emb: [B, D] L2-normalized Euclidean embedding
        Returns: [B, D] point in Poincaré ball
        """
        c = self.c
        hyp = PoincareBall.expmap0(euclidean_emb, c)
        hyp = PoincareBall.proj(hyp, c)
        return hyp

    def extract_embedding(self, x, return_hyp=False):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        eucl = F.normalize(0.7 * emb + 0.3 * cls_emb, dim=-1)
        if return_hyp:
            return eucl, self.to_hyperbolic(eucl)
        return eucl  # Euclidean for standard eval (cosine similarity)

    def forward(self, x, return_parts=False):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        bn = self.bottleneck(emb); logits = self.classifier(bn)
        cls_emb = self.cls_proj(cls_tok); cls_logits = self.cls_classifier(cls_emb)
        fused = F.normalize(0.7 * emb + 0.3 * F.normalize(cls_emb, dim=-1), dim=-1)
        projected_feat = self.teacher_proj(emb)

        # Map to hyperbolic space
        hyp_emb = self.to_hyperbolic(fused)

        out = {'embedding': fused, 'hyp_embedding': hyp_emb,
               'logits': logits, 'cls_logits': cls_logits,
               'projected_feat': projected_feat}
        if return_parts: out['parts'] = parts
        return out


# =============================================================================
# BASE LOSSES (adapted from SPDGeo-D)
# =============================================================================
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
# NEW: Hyperbolic Losses
# =============================================================================

class HyperbolicInfoNCELoss(nn.Module):
    """
    InfoNCE loss using Poincaré distance instead of cosine similarity.

    sim(x, y) = -d_c(x, y) / τ  (negative distance as similarity)

    Poincaré distance is more discriminative near the boundary,
    providing better separation for hard negatives.
    """
    def __init__(self, temp=0.1):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(temp).log())

    def forward(self, q_hyp, r_hyp, labels, c):
        """
        q_hyp: [B, D] queries in Poincaré ball
        r_hyp: [B, D] references in Poincaré ball
        labels: [B] class labels
        c: scalar curvature
        """
        t = self.log_t.exp().clamp(0.01, 1.0)
        B = q_hyp.size(0)

        # Pairwise hyperbolic distances
        q_exp = q_hyp.unsqueeze(1).expand(B, B, -1)  # [B, B, D]
        r_exp = r_hyp.unsqueeze(0).expand(B, B, -1)  # [B, B, D]
        dists = PoincareBall.dist(
            q_exp.reshape(-1, q_hyp.size(-1)),
            r_exp.reshape(-1, r_hyp.size(-1)),
            c
        ).reshape(B, B)  # [B, B]

        # Similarity = negative distance / temperature
        sim = -dists / t

        labels = labels.view(-1, 1); pos_mask = labels.eq(labels.T).float()
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        return (-(log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)).mean()


class HyperbolicTripletLoss(nn.Module):
    """
    Triplet loss using Poincaré distance.

    d_c(anchor, positive) + margin < d_c(anchor, negative)

    Hyperbolic distance provides better gradient signal for hard negatives
    near the boundary (where Euclidean distance saturates).
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, hyp_emb, labels, c):
        """
        hyp_emb: [B, D] embeddings in Poincaré ball
        labels:  [B] class labels
        c: scalar curvature
        """
        B = hyp_emb.size(0)

        # Pairwise distances
        h_exp_i = hyp_emb.unsqueeze(1).expand(B, B, -1)
        h_exp_j = hyp_emb.unsqueeze(0).expand(B, B, -1)
        d = PoincareBall.dist(
            h_exp_i.reshape(-1, hyp_emb.size(-1)),
            h_exp_j.reshape(-1, hyp_emb.size(-1)),
            c
        ).reshape(B, B)

        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float()
        neg_mask = labels.ne(labels.T).float()

        hard_pos = (d * pos_mask).max(1)[0]
        hard_neg = (d * neg_mask + pos_mask * 1e9).min(1)[0]

        return F.relu(hard_pos - hard_neg + self.margin).mean()


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model, test_ds, device):
    model.eval()
    test_tf = get_transforms("test")

    loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)
    drone_feats, drone_labels, drone_alts = [], [], []
    for b in loader:
        feat = model.extract_embedding(b['drone'].to(device)).cpu()
        drone_feats.append(feat); drone_labels.append(b['label']); drone_alts.append(b['altitude'])
    drone_feats = torch.cat(drone_feats); drone_labels = torch.cat(drone_labels)
    drone_alts = torch.cat(drone_alts)

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
        sat_feats.append(model.extract_embedding(batch).cpu())
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
def train_one_epoch(model, teacher, loader, losses, optimizer, scaler, device, epoch):
    model.train()
    if teacher: teacher.eval()

    hyp_nce, hyp_tri, ce, consist, cross_dist, self_dist, uapa = losses
    total_sum = 0; n = 0; loss_sums = defaultdict(float)

    # Curvature scheduling: start near Euclidean, ramp up
    if epoch <= CFG.CURV_WARMUP_EPOCHS:
        curv_scale = epoch / CFG.CURV_WARMUP_EPOCHS
    else:
        curv_scale = 1.0

    for batch in tqdm(loader, desc=f'Ep{epoch:3d}', leave=False):
        drone  = batch['drone'].to(device)
        sat    = batch['satellite'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            d_out = model(drone, return_parts=True)
            s_out = model(sat, return_parts=True)

            c = model.c * curv_scale  # scheduled curvature

            # === Euclidean losses (CE, Cross-Distill, Self-Distill, UAPA) ===
            l_ce = (ce(d_out['logits'], labels) + ce(s_out['logits'], labels))
            l_ce += 0.3 * (ce(d_out['cls_logits'], labels) + ce(s_out['cls_logits'], labels))
            l_con = consist(d_out['parts']['assignment'], s_out['parts']['assignment'])

            if teacher is not None:
                with torch.no_grad():
                    t_drone = teacher(drone); t_sat = teacher(sat)
                l_cross = cross_dist(d_out['projected_feat'], t_drone) + cross_dist(s_out['projected_feat'], t_sat)
            else:
                l_cross = torch.tensor(0.0, device=device)

            l_self = self_dist(d_out['cls_logits'], d_out['logits']) + self_dist(s_out['cls_logits'], s_out['logits'])
            l_uapa = uapa(d_out['logits'], s_out['logits'])

            # === Hyperbolic losses (replace Euclidean InfoNCE + Triplet) ===
            l_hyp_nce = hyp_nce(d_out['hyp_embedding'], s_out['hyp_embedding'], labels, c)
            l_hyp_tri = 0.5 * (hyp_tri(d_out['hyp_embedding'], labels, c) +
                                hyp_tri(s_out['hyp_embedding'], labels, c))

            # === Total loss ===
            loss = (CFG.LAMBDA_CE          * l_ce +
                    CFG.LAMBDA_HYP_NCE     * l_hyp_nce +
                    CFG.LAMBDA_HYP_TRI     * l_hyp_tri +
                    CFG.LAMBDA_CONSISTENCY * l_con +
                    CFG.LAMBDA_CROSS_DIST  * l_cross +
                    CFG.LAMBDA_SELF_DIST   * l_self +
                    CFG.LAMBDA_UAPA        * l_uapa)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()

        total_sum += loss.item(); n += 1
        loss_sums['ce']       += l_ce.item()
        loss_sums['hyp_nce']  += l_hyp_nce.item()
        loss_sums['hyp_tri']  += l_hyp_tri.item()
        loss_sums['con']      += l_con.item()
        loss_sums['cross']    += l_cross.item() if torch.is_tensor(l_cross) else l_cross
        loss_sums['self']     += l_self.item()
        loss_sums['uapa']     += l_uapa.item()

    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EXP25: SPDGeo-HYP — Hyperbolic Part Embeddings")
    print(f"  Base: SPDGeo-D (90.36% R@1)")
    print(f"  Novel: Poincaré ball + Hyperbolic InfoNCE/Triplet + Learnable c")
    print(f"  Init curvature c={CFG.HYP_INIT_C}, warmup={CFG.CURV_WARMUP_EPOCHS} epochs")
    print(f"  Dataset: SUES-200 | Epochs: {CFG.NUM_EPOCHS} | Device: {DEVICE}")
    print(f"  Parts: {CFG.N_PARTS} | Img: {CFG.IMG_SIZE} | Embed: {CFG.EMBED_DIM}")
    print(f"  Losses: CE + HypNCE + HypTri + Con + CrossDist + SelfDist + UAPA = 7")
    print("=" * 65)

    print('\nLoading SUES-200 …')
    train_ds = SUES200Dataset(CFG.SUES_ROOT, 'train', transform=get_transforms("train"))
    test_ds  = SUES200Dataset(CFG.SUES_ROOT, 'test', transform=get_transforms("test"))
    train_loader = DataLoader(train_ds, batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print('\nBuilding models …')
    model = SPDGeoHYPModel(CFG.NUM_CLASSES).to(DEVICE)
    teacher = None
    try:
        teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except Exception as e:
        print(f"  [WARN] Could not load teacher: {e}")

    # Losses
    hyp_nce    = HyperbolicInfoNCELoss(temp=0.1).to(DEVICE)
    hyp_tri    = HyperbolicTripletLoss(margin=0.5)
    ce         = nn.CrossEntropyLoss(label_smoothing=0.1)
    consist    = PartConsistencyLoss()
    cross_dist = CrossDistillationLoss()
    self_dist  = SelfDistillationLoss(temperature=CFG.DISTILL_TEMP)
    uapa_loss  = UAPALoss(base_temperature=CFG.DISTILL_TEMP)
    losses = (hyp_nce, hyp_tri, ce, consist, cross_dist, self_dist, uapa_loss)

    # Optimizer
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith('backbone')]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params,       'lr': CFG.BACKBONE_LR},
        {'params': head_params,           'lr': CFG.LR},
        {'params': hyp_nce.parameters(),  'lr': CFG.LR},
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
        for pg in optimizer.param_groups:
            if pg is optimizer.param_groups[0]: pg['lr'] = CFG.BACKBONE_LR * lr_scale
            else: pg['lr'] = CFG.LR * lr_scale
        cur_lr = optimizer.param_groups[1]['lr']

        avg_loss, ld = train_one_epoch(model, teacher, train_loader, losses,
                                       optimizer, scaler, DEVICE, epoch)

        c_val = model.c.item()
        print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
              f"CE {ld['ce']:.3f}  HypNCE {ld['hyp_nce']:.3f}  HypTri {ld['hyp_tri']:.3f}  "
              f"Con {ld['con']:.3f}  Crs {ld['cross']:.3f}  Slf {ld['self']:.3f}  "
              f"UAPA {ld['uapa']:.3f} | c={c_val:.3f} | LR {cur_lr:.2e}")

        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            metrics, per_alt = evaluate(model, test_ds, DEVICE)
            results_log.append({'epoch': epoch, **metrics, 'curvature': c_val})
            print(f"  ► R@1: {metrics['R@1']:.2f}%  R@5: {metrics['R@5']:.2f}%  "
                  f"R@10: {metrics['R@10']:.2f}%  mAP: {metrics['mAP']:.2f}%  c={c_val:.3f}")
            if metrics['R@1'] > best_r1:
                best_r1 = metrics['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'metrics': metrics, 'per_alt': per_alt, 'curvature': c_val},
                           os.path.join(CFG.OUTPUT_DIR, 'exp25_hyp_best.pth'))
                print(f"  ★ New best R@1: {best_r1:.2f}%!")

    print(f'\n{"="*65}')
    print(f'  EXP25: SPDGeo-HYP COMPLETE — Best R@1: {best_r1:.2f}%')
    print(f'  Final curvature c = {model.c.item():.4f}')
    print(f'{"="*65}')
    print(f'  {"Epoch":>6} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"mAP":>8} {"c":>7}')
    print(f'  {"-"*50}')
    for r in results_log:
        print(f'  {r["epoch"]:6d} {r["R@1"]:8.2f} {r["R@5"]:8.2f} {r["R@10"]:8.2f} {r["mAP"]:8.2f} {r["curvature"]:7.3f}')
    print(f'{"="*65}')

    with open(os.path.join(CFG.OUTPUT_DIR, 'exp25_hyp_results.json'), 'w') as f:
        json.dump({'results_log': results_log, 'best_r1': best_r1,
                   'final_curvature': model.c.item(),
                   'config': {k: v for k, v in vars(CFG).items() if not k.startswith('_')}}, f, indent=2)


if __name__ == '__main__':
    main()
