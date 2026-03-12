# =============================================================================
# EXP29: SPDGeo-MSP — Multi-Scale Part Discovery
# =============================================================================
# Base:    SPDGeo-DPE (93.59% R@1) — THE CHAMPION
# Novel:   Hierarchical part discovery at 2 scales with gated fusion:
#          1) Fine-grain parts (K_fine=4, T=0.05) — capture small objects
#             (roads, buildings, trees) — sharper assignment
#          2) Coarse-grain parts (K_coarse=4, T=0.10) — capture regions
#             (urban blocks, parks, water bodies) — softer assignment
#          3) ScaleAwareGatedFusion — learned gating between scales
#          4) PartScaleConsistency — ensures fine parts are contained in
#             coarse parts spatially (hierarchical constraint)
#
# Motivation:
#   All experiments use K=8 parts at a single granularity. But drone-satellite
#   matching inherently involves multi-scale reasoning:
#   - Fine scale: specific landmarks (distinctive buildings, road intersections)
#   - Coarse scale: layout patterns (grid vs organic, density gradients)
#
#   A single-scale K=8 part model cannot simultaneously capture both:
#   either it fragments into fine parts (losing layout) or clusters into
#   coarse parts (losing landmarks). HierarchicalParts captures both.
#
#   Total parts = K_fine + K_coarse = 8 (same total as DPE, different structure)
#
# Architecture: SPDGeo-DPE + HierarchicalPartDiscovery + ScaleAwareFusion
# Total losses: 9 (6 base + Proxy + EMA + PartScaleConsist)
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
    PART_DIM        = 256
    EMBED_DIM       = 512
    TEACHER_DIM     = 768
    UNFREEZE_BLOCKS = 4

    # Multi-scale part config
    K_FINE          = 4      # Fine-grain parts
    K_COARSE        = 4      # Coarse-grain parts
    N_PARTS         = 8      # Total (K_FINE + K_COARSE)
    TEMP_FINE       = 0.05   # Sharper assignment for fine parts
    TEMP_COARSE     = 0.10   # Softer assignment for coarse parts

    NUM_EPOCHS      = 120
    P_CLASSES       = 16
    K_SAMPLES       = 4
    LR              = 3e-4
    BACKBONE_LR     = 3e-5
    WEIGHT_DECAY    = 0.01
    WARMUP_EPOCHS   = 5
    USE_AMP         = True
    SEED            = 42

    # Base loss weights
    LAMBDA_CE           = 1.0
    LAMBDA_INFONCE      = 1.0
    LAMBDA_CONSISTENCY  = 0.1
    LAMBDA_CROSS_DIST   = 0.3
    LAMBDA_SELF_DIST    = 0.3
    LAMBDA_UAPA         = 0.2

    # DPE components
    LAMBDA_PROXY        = 0.5
    PROXY_MARGIN        = 0.1
    PROXY_ALPHA         = 32
    LAMBDA_EMA_DIST     = 0.2
    EMA_DECAY           = 0.999

    # NEW: Part scale consistency
    LAMBDA_SCALE_CONSIST = 0.15

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
# BACKBONE
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
# TEACHER
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
# NEW: Hierarchical Part Discovery (Fine + Coarse)
# =============================================================================
class HierarchicalPartDiscovery(nn.Module):
    """
    Two-scale part discovery with separate prototype sets and temperatures.

    Fine scale (K_fine=4, T=0.05):
      - Sharp assignment → each patch goes to ~1 part strongly
      - Captures: specific landmarks, distinctive objects
      - Low temperature forces patches into tight clusters

    Coarse scale (K_coarse=4, T=0.10):
      - Soft assignment → each patch contributes to multiple parts
      - Captures: layout patterns, regional characteristics
      - Higher temperature allows broader, overlapping regions

    Both scales share the same feat_proj (efficiency),
    but have independent prototypes and refinement layers.

    Total parts = K_fine + K_coarse = 8 (same budget as DPE's K=8)
    """
    def __init__(self, feat_dim=384, k_fine=4, k_coarse=4, part_dim=256,
                 temp_fine=0.05, temp_coarse=0.10):
        super().__init__()
        self.k_fine = k_fine; self.k_coarse = k_coarse
        self.temp_fine = temp_fine; self.temp_coarse = temp_coarse
        self.n_parts = k_fine + k_coarse

        # Shared feature projection
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU()
        )

        # Fine-scale prototypes and refinement
        self.fine_prototypes = nn.Parameter(torch.randn(k_fine, part_dim) * 0.02)
        self.fine_refine = nn.Sequential(
            nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim * 2),
            nn.GELU(), nn.Linear(part_dim * 2, part_dim)
        )

        # Coarse-scale prototypes and refinement
        self.coarse_prototypes = nn.Parameter(torch.randn(k_coarse, part_dim) * 0.02)
        self.coarse_refine = nn.Sequential(
            nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim * 2),
            nn.GELU(), nn.Linear(part_dim * 2, part_dim)
        )

        # Salience per-scale
        self.fine_salience = nn.Sequential(
            nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.coarse_salience = nn.Sequential(
            nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def _discover_parts(self, feat, prototypes, temperature, refine_net, salience_head,
                        spatial_hw, B, N):
        """Discover parts at a single scale."""
        H, W = spatial_hw
        device = feat.device

        feat_norm = F.normalize(feat, dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)
        sim = torch.einsum('bnd,kd->bnk', feat_norm, proto_norm) / temperature
        assign = F.softmax(sim, dim=-1)  # [B, N, K]
        assign_t = assign.transpose(1, 2)  # [B, K, N]
        mass = assign_t.sum(-1, keepdim=True).clamp(min=1e-6)
        part_feat = torch.bmm(assign_t, feat) / mass
        part_feat = part_feat + refine_net(part_feat)

        # Spatial positions
        gy = torch.arange(H, device=device).float() / max(H - 1, 1)
        gx = torch.arange(W, device=device).float() / max(W - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(B, -1, -1)) / mass

        salience = salience_head(part_feat).squeeze(-1)

        return part_feat, assign, part_pos, salience

    def forward(self, patch_features, spatial_hw):
        B, N, _ = patch_features.shape
        feat = self.feat_proj(patch_features)  # [B, N, part_dim]

        # Fine-scale discovery
        fine_feat, fine_assign, fine_pos, fine_sal = self._discover_parts(
            feat, self.fine_prototypes, self.temp_fine, self.fine_refine,
            self.fine_salience, spatial_hw, B, N
        )

        # Coarse-scale discovery
        coarse_feat, coarse_assign, coarse_pos, coarse_sal = self._discover_parts(
            feat, self.coarse_prototypes, self.temp_coarse, self.coarse_refine,
            self.coarse_salience, spatial_hw, B, N
        )

        # Concatenate both scales: [B, K_fine+K_coarse, part_dim]
        part_features = torch.cat([fine_feat, coarse_feat], dim=1)
        assignment = torch.cat([fine_assign, coarse_assign], dim=-1)  # [B, N, K_total]
        part_positions = torch.cat([fine_pos, coarse_pos], dim=1)
        salience = torch.cat([fine_sal, coarse_sal], dim=1)

        return {
            'part_features': part_features,
            'assignment': assignment,
            'part_positions': part_positions,
            'salience': salience,
            # Keep per-scale info for PartScaleConsistency loss
            'fine_assign': fine_assign,
            'coarse_assign': coarse_assign,
            'fine_positions': fine_pos,
            'coarse_positions': coarse_pos,
        }


# =============================================================================
# Scale-Aware Gated Pooling
# =============================================================================
class ScaleAwarePooling(nn.Module):
    """
    Pools fine and coarse part features separately, then learns a per-sample
    gate to fuse them. This allows the model to weight fine vs coarse parts
    based on image content.

    Easy images (distinctive landmarks visible) → more fine weight
    Hard images (ambiguous, need layout reasoning) → more coarse weight
    """
    def __init__(self, part_dim=256, embed_dim=512, k_fine=4, k_coarse=4):
        super().__init__()
        self.k_fine = k_fine
        # Per-scale attention pooling
        self.fine_attn = nn.Sequential(nn.Linear(part_dim, part_dim // 2), nn.Tanh(), nn.Linear(part_dim // 2, 1))
        self.coarse_attn = nn.Sequential(nn.Linear(part_dim, part_dim // 2), nn.Tanh(), nn.Linear(part_dim // 2, 1))

        # Per-scale projections
        self.fine_proj = nn.Sequential(nn.Linear(part_dim * 3, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.coarse_proj = nn.Sequential(nn.Linear(part_dim * 3, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

        # Scale fusion gate
        self.scale_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1), nn.Sigmoid()
        )
        # Final projection
        self.final_proj = nn.Linear(embed_dim, embed_dim)

    def _pool_scale(self, feats, salience, attn_net):
        """Attention + mean + max pooling for one scale."""
        aw = attn_net(feats)
        if salience is not None:
            aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        attn_pool = (aw * feats).sum(1)
        mean_pool = feats.mean(1)
        max_pool = feats.max(1)[0]
        return torch.cat([attn_pool, mean_pool, max_pool], dim=-1)

    def forward(self, part_features, salience=None):
        B, K, D = part_features.shape
        fine_feats = part_features[:, :self.k_fine]
        coarse_feats = part_features[:, self.k_fine:]
        fine_sal = salience[:, :self.k_fine] if salience is not None else None
        coarse_sal = salience[:, self.k_fine:] if salience is not None else None

        # Pool each scale
        fine_pooled = self.fine_proj(self._pool_scale(fine_feats, fine_sal, self.fine_attn))
        coarse_pooled = self.coarse_proj(self._pool_scale(coarse_feats, coarse_sal, self.coarse_attn))

        # Learned gate between scales
        combined = torch.cat([fine_pooled, coarse_pooled], dim=-1)
        alpha = self.scale_gate(combined)  # [B, 1]
        fused = alpha * fine_pooled + (1 - alpha) * coarse_pooled
        return F.normalize(self.final_proj(fused), dim=-1)


# =============================================================================
# DYNAMIC FUSION GATE
# =============================================================================
class DynamicFusionGate(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        combined = torch.cat([part_emb, cls_emb], dim=-1)
        alpha = torch.sigmoid(self.gate(combined))
        fused = alpha * part_emb + (1 - alpha) * cls_emb
        return F.normalize(fused, dim=-1)


# =============================================================================
# STUDENT MODEL
# =============================================================================
class SPDGeoMSPModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = HierarchicalPartDiscovery(
            384, cfg.K_FINE, cfg.K_COARSE, cfg.PART_DIM,
            cfg.TEMP_FINE, cfg.TEMP_COARSE
        )
        self.pool = ScaleAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM, cfg.K_FINE, cfg.K_COARSE)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)

        self.bottleneck     = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj   = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
                                            nn.LayerNorm(cfg.TEACHER_DIM))
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-MSP student: {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)

    def forward(self, x, return_parts=False):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        bn = self.bottleneck(emb); logits = self.classifier(bn)
        cls_emb_raw = self.cls_proj(cls_tok); cls_logits = self.cls_classifier(cls_emb_raw)
        cls_emb_norm = F.normalize(cls_emb_raw, dim=-1)
        fused = self.fusion_gate(emb, cls_emb_norm)
        projected_feat = self.teacher_proj(emb)
        out = {'embedding': fused, 'logits': logits, 'cls_logits': cls_logits,
               'projected_feat': projected_feat, 'part_emb': emb, 'cls_emb': cls_emb_norm}
        if return_parts: out['parts'] = parts
        return out


# =============================================================================
# EMA Model
# =============================================================================
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = copy.deepcopy(model); self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def forward(self, x):
        return self.model.extract_embedding(x)


# =============================================================================
# BASE LOSSES
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


class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, embed_dim, margin=0.1, alpha=32):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.margin = margin; self.alpha = alpha; self.num_classes = num_classes

    def forward(self, embeddings, labels):
        P = F.normalize(self.proxies, dim=-1)
        sim = embeddings @ P.T
        one_hot = F.one_hot(labels, self.num_classes).float()
        pos_exp = torch.exp(-self.alpha * (sim * one_hot - self.margin)) * one_hot
        P_plus = one_hot.sum(0); has_pos = P_plus > 0
        pos_term = torch.log(1 + pos_exp.sum(0))
        pos_loss = pos_term[has_pos].mean() if has_pos.sum() > 0 else torch.tensor(0.0, device=embeddings.device)
        neg_mask = 1 - one_hot
        neg_exp = torch.exp(self.alpha * (sim * neg_mask + self.margin)) * neg_mask
        neg_loss = torch.log(1 + neg_exp.sum(0)).mean()
        return pos_loss + neg_loss


class EMADistillationLoss(nn.Module):
    def forward(self, student_emb, ema_emb):
        return (1 - F.cosine_similarity(student_emb, ema_emb)).mean()


# =============================================================================
# NEW: Part Scale Consistency Loss
# =============================================================================
class PartScaleConsistencyLoss(nn.Module):
    """
    Ensures hierarchical spatial consistency: each fine part's spatial support
    should be a subset of exactly one coarse part's spatial support.

    Concretely: for each fine part f, compute overlap with each coarse part c
    as the dot product of their assignment maps. The fine part should have
    maximal overlap with exactly one coarse part (encourage peaky assignment).

    L = -Σ_f max_c (overlap(f, c)) + entropy_regularization

    This prevents fine and coarse parts from covering entirely different
    regions — they should form a hierarchy.
    """
    def forward(self, fine_assign, coarse_assign):
        """
        fine_assign:   [B, N, K_fine]
        coarse_assign: [B, N, K_coarse]
        """
        # Compute per-patch overlap: [B, K_fine, K_coarse]
        # fine_assign.T @ coarse_assign
        overlap = torch.bmm(fine_assign.transpose(1, 2), coarse_assign)  # [B, K_fine, K_coarse]

        # Normalize per fine part
        overlap_norm = overlap / overlap.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # Encourage peaky distribution → each fine part should map to ~1 coarse part
        # Use negative entropy: low entropy = good (peaky)
        entropy = -(overlap_norm * (overlap_norm + 1e-8).log()).sum(dim=-1)  # [B, K_fine]

        return entropy.mean()


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
def train_one_epoch(model, teacher, ema, loader, losses, new_losses, optimizer,
                    scaler, device, epoch):
    model.train()
    if teacher: teacher.eval()

    infonce, ce, consist, cross_dist, self_dist, uapa = losses
    proxy_anchor, ema_dist, scale_consist = new_losses

    total_sum = 0; n = 0; loss_sums = defaultdict(float)

    for batch in tqdm(loader, desc=f'Ep{epoch:3d}', leave=False):
        drone  = batch['drone'].to(device)
        sat    = batch['satellite'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            d_out = model(drone, return_parts=True)
            s_out = model(sat, return_parts=True)

            # Base losses (6)
            l_ce = (ce(d_out['logits'], labels) + ce(s_out['logits'], labels))
            l_ce += 0.3 * (ce(d_out['cls_logits'], labels) + ce(s_out['cls_logits'], labels))
            l_nce = infonce(d_out['embedding'], s_out['embedding'], labels)
            l_con = consist(d_out['parts']['assignment'], s_out['parts']['assignment'])

            if teacher is not None:
                with torch.no_grad():
                    t_drone = teacher(drone); t_sat = teacher(sat)
                l_cross = cross_dist(d_out['projected_feat'], t_drone) + \
                          cross_dist(s_out['projected_feat'], t_sat)
            else:
                l_cross = torch.tensor(0.0, device=device)

            l_self = self_dist(d_out['cls_logits'], d_out['logits']) + \
                     self_dist(s_out['cls_logits'], s_out['logits'])
            l_uapa = uapa(d_out['logits'], s_out['logits'])

            # ProxyAnchor
            l_proxy = 0.5 * (proxy_anchor(d_out['embedding'], labels) +
                             proxy_anchor(s_out['embedding'], labels))

            # EMA Distillation
            with torch.no_grad():
                ema_drone_emb = ema.forward(drone)
                ema_sat_emb   = ema.forward(sat)
            l_ema = 0.5 * (ema_dist(d_out['embedding'], ema_drone_emb) +
                           ema_dist(s_out['embedding'], ema_sat_emb))

            # Part Scale Consistency
            l_scale = 0.5 * (
                scale_consist(d_out['parts']['fine_assign'], d_out['parts']['coarse_assign']) +
                scale_consist(s_out['parts']['fine_assign'], s_out['parts']['coarse_assign'])
            )

            loss = (CFG.LAMBDA_CE            * l_ce +
                    CFG.LAMBDA_INFONCE       * l_nce +
                    CFG.LAMBDA_CONSISTENCY   * l_con +
                    CFG.LAMBDA_CROSS_DIST    * l_cross +
                    CFG.LAMBDA_SELF_DIST     * l_self +
                    CFG.LAMBDA_UAPA          * l_uapa +
                    CFG.LAMBDA_PROXY         * l_proxy +
                    CFG.LAMBDA_EMA_DIST      * l_ema +
                    CFG.LAMBDA_SCALE_CONSIST * l_scale)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()
        ema.update(model)

        total_sum += loss.item(); n += 1
        loss_sums['ce']    += l_ce.item()
        loss_sums['nce']   += l_nce.item()
        loss_sums['con']   += l_con.item()
        loss_sums['cross'] += l_cross.item() if torch.is_tensor(l_cross) else l_cross
        loss_sums['self']  += l_self.item()
        loss_sums['uapa']  += l_uapa.item()
        loss_sums['proxy'] += l_proxy.item()
        loss_sums['ema']   += l_ema.item()
        loss_sums['scale'] += l_scale.item()

    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EXP29: SPDGeo-MSP — Multi-Scale Part Discovery")
    print(f"  Base: SPDGeo-DPE (93.59% R@1)")
    print(f"  Novel: K_fine={CFG.K_FINE}(T={CFG.TEMP_FINE}) + "
          f"K_coarse={CFG.K_COARSE}(T={CFG.TEMP_COARSE}) + ScaleGate")
    print(f"  Dataset: SUES-200 | Epochs: {CFG.NUM_EPOCHS} | Device: {DEVICE}")
    print(f"  Total Parts: {CFG.N_PARTS} | Img: {CFG.IMG_SIZE} | Embed: {CFG.EMBED_DIM}")
    print(f"  Losses: 6 base + Proxy + EMA + ScaleConsist = 9 total")
    print("=" * 65)

    print('\nLoading SUES-200 …')
    train_ds = SUES200Dataset(CFG.SUES_ROOT, 'train', transform=get_transforms("train"))
    test_ds  = SUES200Dataset(CFG.SUES_ROOT, 'test', transform=get_transforms("test"))
    train_loader = DataLoader(train_ds, batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print('\nBuilding models …')
    model = SPDGeoMSPModel(CFG.NUM_CLASSES).to(DEVICE)
    teacher = None
    try:
        teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except Exception as e:
        print(f"  [WARN] Could not load teacher: {e}")

    ema = EMAModel(model, decay=CFG.EMA_DECAY)
    print(f"  EMA model initialized (decay={CFG.EMA_DECAY})")

    # Base losses
    infonce    = SupInfoNCELoss(temp=0.05).to(DEVICE)
    ce         = nn.CrossEntropyLoss(label_smoothing=0.1)
    consist    = PartConsistencyLoss()
    cross_dist = CrossDistillationLoss()
    self_dist  = SelfDistillationLoss(temperature=CFG.DISTILL_TEMP)
    uapa_loss  = UAPALoss(base_temperature=CFG.DISTILL_TEMP)
    base_losses = (infonce, ce, consist, cross_dist, self_dist, uapa_loss)

    # New losses
    proxy_anchor  = ProxyAnchorLoss(CFG.NUM_CLASSES, CFG.EMBED_DIM,
                                    margin=CFG.PROXY_MARGIN, alpha=CFG.PROXY_ALPHA).to(DEVICE)
    ema_dist      = EMADistillationLoss()
    scale_consist = PartScaleConsistencyLoss()
    new_losses = (proxy_anchor, ema_dist, scale_consist)

    # Optimizer
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith('backbone')]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params,            'lr': CFG.BACKBONE_LR},
        {'params': head_params,                'lr': CFG.LR},
        {'params': infonce.parameters(),       'lr': CFG.LR},
        {'params': proxy_anchor.parameters(),  'lr': CFG.LR},
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
        optimizer.param_groups[3]['lr'] = CFG.LR * lr_scale
        cur_lr = optimizer.param_groups[1]['lr']

        avg_loss, ld = train_one_epoch(model, teacher, ema, train_loader, base_losses,
                                       new_losses, optimizer, scaler, DEVICE, epoch)

        print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
              f"CE {ld['ce']:.3f}  NCE {ld['nce']:.3f}  "
              f"Con {ld['con']:.3f}  Crs {ld['cross']:.3f}  Slf {ld['self']:.3f}  "
              f"UAPA {ld['uapa']:.3f} | "
              f"Proxy {ld['proxy']:.3f}  EMA {ld['ema']:.3f}  Scale {ld['scale']:.3f} | "
              f"LR {cur_lr:.2e}")

        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            metrics, per_alt = evaluate(model, test_ds, DEVICE)
            results_log.append({'epoch': epoch, **metrics})
            print(f"  ► R@1: {metrics['R@1']:.2f}%  R@5: {metrics['R@5']:.2f}%  "
                  f"R@10: {metrics['R@10']:.2f}%  mAP: {metrics['mAP']:.2f}%")
            if metrics['R@1'] > best_r1:
                best_r1 = metrics['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'metrics': metrics, 'per_alt': per_alt},
                           os.path.join(CFG.OUTPUT_DIR, 'exp29_msp_best.pth'))
                print(f"  ★ New best R@1: {best_r1:.2f}%!")

            ema_metrics, _ = evaluate(ema.model, test_ds, DEVICE)
            print(f"  ► EMA R@1: {ema_metrics['R@1']:.2f}%")
            if ema_metrics['R@1'] > best_r1:
                best_r1 = ema_metrics['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': ema.model.state_dict(),
                            'metrics': ema_metrics, 'is_ema': True},
                           os.path.join(CFG.OUTPUT_DIR, 'exp29_msp_best.pth'))
                print(f"  ★ New best R@1 (EMA): {best_r1:.2f}%!")

    print(f'\n{"="*65}')
    print(f'  EXP29: SPDGeo-MSP COMPLETE — Best R@1: {best_r1:.2f}%')
    print(f'{"="*65}')
    print(f'  {"Epoch":>6} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"mAP":>8}')
    print(f'  {"-"*44}')
    for r in results_log:
        print(f'  {r["epoch"]:6d} {r["R@1"]:8.2f} {r["R@5"]:8.2f} {r["R@10"]:8.2f} {r["mAP"]:8.2f}')
    print(f'{"="*65}')

    with open(os.path.join(CFG.OUTPUT_DIR, 'exp29_msp_results.json'), 'w') as f:
        json.dump({'results_log': results_log, 'best_r1': best_r1,
                   'config': {k: v for k, v in vars(CFG).items() if not k.startswith('_')}}, f, indent=2)


if __name__ == '__main__':
    main()
