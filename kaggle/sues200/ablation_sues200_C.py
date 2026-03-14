# =============================================================================
# ABLATION STUDY — SPDGeo-DPEA-MAR (EXP35)
# =============================================================================
# Full model = EXP35: 6 base + Proxy + EMA + AltConsist + MaskRecon + AltPred + Diversity
#              + DynamicFusionGate + DeepAltitudeFiLM
#
# Removes ONE component at a time to measure its contribution.
# ABLATION_GROUP controls which pair of ablations to run:
#   A: w/o ProxyAnchor, w/o FusionGate
#   B: w/o EMA, w/o DeepAltFiLM
#   C: w/o AltConsistLoss, w/o MaskRecon
#   D: w/o AltPred, w/o Diversity
#
# Each ablation runs for 60 epochs. Compare against full EXP35 results.
# =============================================================================

ABLATION_GROUP = "C"  # ← "A", "B", "C", or "D"
ABL_EPOCHS = 60
ABL_EVAL_INTERVAL = 10

# --- Ablation configs ---
_FULL = {"use_proxy": True, "use_fusion_gate": True, "use_ema": True,
         "use_alt_film": True, "use_alt_consist": True,
         "use_mask_recon": True, "use_alt_pred": True, "use_diversity": True}

ABLATION_RUNS = {
    "A": [
        {**_FULL, "name": "w/o_ProxyAnchor", "use_proxy": False},
        {**_FULL, "name": "w/o_FusionGate", "use_fusion_gate": False},
    ],
    "B": [
        {**_FULL, "name": "w/o_EMA", "use_ema": False},
        {**_FULL, "name": "w/o_DeepAltFiLM", "use_alt_film": False},
    ],
    "C": [
        {**_FULL, "name": "w/o_AltConsistLoss", "use_alt_consist": False},
        {**_FULL, "name": "w/o_MaskRecon", "use_mask_recon": False},
    ],
    "D": [
        {**_FULL, "name": "w/o_AltPred", "use_alt_pred": False},
        {**_FULL, "name": "w/o_Diversity", "use_diversity": False},
    ],
}

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
    ALT_TO_IDX      = {"150": 0, "200": 1, "250": 2, "300": 3}
    NUM_ALTITUDES   = 4
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

    P_CLASSES       = 16
    K_SAMPLES       = 4
    LR              = 3e-4
    BACKBONE_LR     = 3e-5
    WEIGHT_DECAY    = 0.01
    WARMUP_EPOCHS   = 5
    USE_AMP         = True
    SEED            = 42

    LAMBDA_CE           = 1.0
    LAMBDA_INFONCE      = 1.0
    LAMBDA_CONSISTENCY  = 0.1
    LAMBDA_CROSS_DIST   = 0.3
    LAMBDA_SELF_DIST    = 0.3
    LAMBDA_UAPA         = 0.2
    LAMBDA_PROXY        = 0.5
    PROXY_MARGIN        = 0.1
    PROXY_ALPHA         = 32
    LAMBDA_EMA_DIST     = 0.2
    EMA_DECAY           = 0.996
    LAMBDA_ALT_CONSIST  = 0.2
    LAMBDA_TRIPLET      = 0.5

    MASK_RATIO          = 0.30
    LAMBDA_MASK_RECON   = 0.3
    RECON_WARMUP        = 10
    LAMBDA_ALT_PRED     = 0.15
    LAMBDA_DIVERSITY    = 0.05

    DISTILL_TEMP        = 4.0
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
            d = Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128,128,128))
            s = Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128,128,128))
        if self.transform: d = self.transform(d); s = self.transform(s)
        alt_idx = CFG.ALT_TO_IDX.get(alt, 0)
        alt_norm = (int(alt) - 150) / 150.0
        return {'drone': d, 'satellite': s, 'label': li,
                'altitude': int(alt), 'alt_idx': alt_idx, 'alt_norm': alt_norm}


class PKSampler:
    def __init__(self, ds, p, k):
        self.ds = ds; self.p = p; self.k = k
        self.locs = list(ds.drone_by_location.keys())
    def __iter__(self):
        locs = self.locs.copy(); random.shuffle(locs); batch = []
        for l in locs:
            idx = self.ds.drone_by_location[l]
            if len(idx) < self.k: idx = idx * (self.k // len(idx) + 1)
            batch.extend(random.sample(idx, self.k))
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]; batch = batch[self.p * self.k:]
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
# BACKBONES
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

    def forward(self, x):
        features = self.model.forward_features(x)
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return features['x_norm_patchtokens'], features['x_norm_clstoken'], (H, W)


class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        self.output_dim = 768
        for p in self.model.parameters(): p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        return self.model.forward_features(x)['x_norm_clstoken']


# =============================================================================
# PART DISCOVERY — Vanilla (for w/o DeepAltFiLM)
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

    def forward(self, patch_features, spatial_hw, alt_idx=None):
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
                'assignment': assign, 'salience': salience, 'projected_patches': feat}


# =============================================================================
# PART DISCOVERY — with Deep Altitude FiLM
# =============================================================================
class DeepAltitudeFiLM(nn.Module):
    def __init__(self, num_altitudes=4, feat_dim=256):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_altitudes, feat_dim))
        self.beta  = nn.Parameter(torch.zeros(num_altitudes, feat_dim))

    def forward(self, feat, alt_idx=None):
        if alt_idx is None:
            gamma = self.gamma.mean(0, keepdim=True); beta = self.beta.mean(0, keepdim=True)
            return feat * gamma.unsqueeze(0) + beta.unsqueeze(0)
        else:
            return feat * self.gamma[alt_idx].unsqueeze(1) + self.beta[alt_idx].unsqueeze(1)


class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07, num_altitudes=4):
        super().__init__()
        self.n_parts = n_parts; self.temperature = temperature
        self.feat_proj = nn.Sequential(nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU())
        self.altitude_film = DeepAltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim * 2),
                                    nn.GELU(), nn.Linear(part_dim * 2, part_dim))
        self.salience_head = nn.Sequential(nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, patch_features, spatial_hw, alt_idx=None):
        B, N, _ = patch_features.shape; H, W = spatial_hw
        feat = self.feat_proj(patch_features)
        feat = self.altitude_film(feat, alt_idx)
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
                'assignment': assign, 'salience': salience, 'projected_patches': feat}


# =============================================================================
# POOLING + FUSION GATE
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
        return F.normalize(self.proj(torch.cat([attn_pool, mean_pool, max_pool], dim=-1)), dim=-1)


class DynamicFusionGate(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        alpha = torch.sigmoid(self.gate(torch.cat([part_emb, cls_emb], dim=-1)))
        return F.normalize(alpha * part_emb + (1 - alpha) * cls_emb, dim=-1)


# =============================================================================
# NEW MODULES (from EXP34)
# =============================================================================
class MaskedPartReconstruction(nn.Module):
    def __init__(self, part_dim=256, mask_ratio=0.30):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Linear(part_dim, part_dim * 2), nn.GELU(),
            nn.Linear(part_dim * 2, part_dim), nn.LayerNorm(part_dim),
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, part_dim) * 0.02)

    def forward(self, projected_patches, part_features, assignment):
        B, N, D = projected_patches.shape
        num_mask = int(N * self.mask_ratio)
        noise = torch.rand(B, N, device=projected_patches.device)
        ids_shuffle = noise.argsort(dim=1)
        mask_indices = ids_shuffle[:, :num_mask]
        target = projected_patches.detach()
        mask_expand = mask_indices.unsqueeze(-1).expand(-1, -1, D)
        masked_targets = torch.gather(target, 1, mask_expand)
        K = part_features.shape[1]
        mask_expand_k = mask_indices.unsqueeze(-1).expand(-1, -1, K)
        masked_assign = torch.gather(assignment, 1, mask_expand_k)
        recon = torch.bmm(masked_assign, part_features)
        recon = self.decoder(recon)
        recon_norm = F.normalize(recon, dim=-1)
        target_norm = F.normalize(masked_targets, dim=-1)
        return (1 - (recon_norm * target_norm).sum(dim=-1)).mean()


class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, embedding, alt_target):
        pred = self.head(embedding).squeeze(-1)
        return F.smooth_l1_loss(pred, alt_target)


class PrototypeDiversityLoss(nn.Module):
    def forward(self, prototypes):
        P = F.normalize(prototypes, dim=-1)
        sim = P @ P.T
        K = sim.size(0)
        mask = 1 - torch.eye(K, device=sim.device)
        return (sim * mask).abs().sum() / (K * (K - 1))


# =============================================================================
# ABLATION MODEL — supports all flags
# =============================================================================
class AblationModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG, use_fusion_gate=True, use_alt_film=True,
                 use_mask_recon=True, use_alt_pred=True):
        super().__init__()
        self.use_fusion_gate = use_fusion_gate
        self.use_alt_film = use_alt_film
        self.backbone = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)

        if use_alt_film:
            self.part_disc = AltitudeAwarePartDiscovery(
                384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES)
        else:
            self.part_disc = SemanticPartDiscovery(
                384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP)

        self.pool = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)

        if use_fusion_gate:
            self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)

        self.bottleneck     = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj   = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
                                            nn.LayerNorm(cfg.TEACHER_DIM))

        if use_mask_recon:
            self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)
        if use_alt_pred:
            self.alt_pred = AltitudePredictionHead(cfg.EMBED_DIM)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        flags = f"FG={use_fusion_gate} FiLM={use_alt_film} MR={use_mask_recon} AP={use_alt_pred}"
        print(f"  Model: {total/1e6:.1f}M params | {flags}")

    def _fuse(self, part_emb, cls_emb):
        if self.use_fusion_gate:
            return self.fusion_gate(part_emb, cls_emb)
        return F.normalize(0.7 * part_emb + 0.3 * cls_emb, dim=-1)

    def extract_embedding(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx if self.use_alt_film else None)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self._fuse(emb, cls_emb)

    def forward(self, x, alt_idx=None, return_parts=False):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx if self.use_alt_film else None)
        emb = self.pool(parts['part_features'], parts['salience'])
        bn = self.bottleneck(emb); logits = self.classifier(bn)
        cls_emb_raw = self.cls_proj(cls_tok)
        cls_logits = self.cls_classifier(cls_emb_raw)
        cls_emb_norm = F.normalize(cls_emb_raw, dim=-1)
        fused = self._fuse(emb, cls_emb_norm)
        projected_feat = self.teacher_proj(emb)
        out = {'embedding': fused, 'logits': logits, 'cls_logits': cls_logits,
               'projected_feat': projected_feat, 'part_emb': emb, 'cls_emb': cls_emb_norm}
        if return_parts: out['parts'] = parts
        return out


# =============================================================================
# EMA MODEL
# =============================================================================
class EMAModel:
    def __init__(self, model, decay=0.996):
        self.decay = decay
        self.model = copy.deepcopy(model); self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def forward(self, x, alt_idx=None):
        return self.model.extract_embedding(x, alt_idx=alt_idx)


# =============================================================================
# ALL LOSSES
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
        return (self.T ** 2) * F.kl_div(F.log_softmax(weak_logits / self.T, dim=1), p_teacher, reduction='batchmean')

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
        P = F.normalize(self.proxies, dim=-1); sim = embeddings @ P.T
        one_hot = F.one_hot(labels, self.num_classes).float()
        pos_exp = torch.exp(-self.alpha * (sim * one_hot - self.margin)) * one_hot
        has_pos = one_hot.sum(0) > 0
        pos_term = torch.log(1 + pos_exp.sum(0))
        pos_loss = pos_term[has_pos].mean() if has_pos.sum() > 0 else torch.tensor(0.0, device=embeddings.device)
        neg_mask = 1 - one_hot
        neg_exp = torch.exp(self.alpha * (sim * neg_mask + self.margin)) * neg_mask
        return pos_loss + torch.log(1 + neg_exp.sum(0)).mean()

class EMADistillationLoss(nn.Module):
    def forward(self, student_emb, ema_emb):
        return (1 - F.cosine_similarity(student_emb, ema_emb)).mean()

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__(); self.margin = margin
    def forward(self, embeddings, labels):
        dist = torch.cdist(embeddings, embeddings)
        lbl = labels.view(-1, 1)
        pos_mask = lbl.eq(lbl.T).float() - torch.eye(labels.size(0), device=labels.device)
        neg_mask = lbl.ne(lbl.T).float()
        hard_pos = (dist * pos_mask + (1 - pos_mask) * -1).max(dim=1)[0]
        hard_neg = (dist + (1 - neg_mask) * 999).min(dim=1)[0]
        return F.relu(hard_pos - hard_neg + self.margin).mean()

class AltitudeConsistencyLoss(nn.Module):
    def forward(self, embeddings, labels, altitudes):
        B = embeddings.size(0)
        if B < 2: return torch.tensor(0.0, device=embeddings.device)
        lbl = labels.view(-1, 1); alt = altitudes.view(-1, 1)
        mask = lbl.eq(lbl.T) & alt.ne(alt.T)
        if mask.sum() == 0: return torch.tensor(0.0, device=embeddings.device)
        cos_dist = 1 - (F.normalize(embeddings, dim=-1) @ F.normalize(embeddings, dim=-1).T)
        return (cos_dist * mask.float()).sum() / mask.float().sum().clamp(min=1)


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model, test_ds, device, use_alt_film=True):
    model.eval()
    test_tf = get_transforms("test")
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    drone_feats, drone_labels = [], []
    for b in loader:
        alt_idx = b['alt_idx'].to(device) if use_alt_film else None
        feat = model.extract_embedding(b['drone'].to(device), alt_idx=alt_idx).cpu()
        drone_feats.append(feat); drone_labels.append(b['label'])
    drone_feats = torch.cat(drone_feats); drone_labels = torch.cat(drone_labels)

    all_locs = [f"{l:04d}" for l in range(1, 201)]
    sat_imgs, sat_lbls = [], []; dc = 0
    for loc in all_locs:
        sp = os.path.join(test_ds.sat_dir, loc, "0.png")
        if not os.path.exists(sp): continue
        sat_imgs.append(test_tf(Image.open(sp).convert('RGB')))
        if loc in test_ds.location_to_idx: sat_lbls.append(test_ds.location_to_idx[loc])
        else: sat_lbls.append(-1000 - dc); dc += 1

    sat_feats = []
    for i in range(0, len(sat_imgs), 64):
        batch = torch.stack(sat_imgs[i:i+64]).to(device)
        sat_feats.append(model.extract_embedding(batch, alt_idx=None).cpu())
    sat_feats = torch.cat(sat_feats); sat_labels = torch.tensor(sat_lbls)

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
    return {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100, 'mAP': ap/N*100}


# =============================================================================
# TRAINING — respects ablation flags
# =============================================================================
def train_one_epoch(model, teacher, ema, loader, losses, optimizer, scaler,
                    device, epoch, abl):
    model.train()
    if teacher: teacher.eval()
    infonce, ce, consist, cross_dist, self_dist, uapa = losses['base']
    total_sum = 0; n = 0; loss_sums = defaultdict(float)
    recon_active = abl['use_mask_recon'] and epoch >= CFG.RECON_WARMUP

    for batch in tqdm(loader, desc=f'Ep{epoch:3d}', leave=False):
        drone  = batch['drone'].to(device)
        sat    = batch['satellite'].to(device)
        labels = batch['label'].to(device)
        alt_idx = batch['alt_idx'].to(device) if abl['use_alt_film'] else None
        alts   = batch['altitude'].to(device)
        alt_norm = batch['alt_norm'].to(device).float()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            d_out = model(drone, alt_idx=alt_idx, return_parts=True)
            s_out = model(sat, alt_idx=None, return_parts=True)

            l_ce = (ce(d_out['logits'], labels) + ce(s_out['logits'], labels))
            l_ce += 0.3 * (ce(d_out['cls_logits'], labels) + ce(s_out['cls_logits'], labels))
            l_nce = infonce(d_out['embedding'], s_out['embedding'], labels)
            l_con = consist(d_out['parts']['assignment'], s_out['parts']['assignment'])
            if teacher is not None:
                with torch.no_grad():
                    t_d = teacher(drone); t_s = teacher(sat)
                l_cross = cross_dist(d_out['projected_feat'], t_d) + cross_dist(s_out['projected_feat'], t_s)
            else:
                l_cross = torch.tensor(0.0, device=device)
            l_self = self_dist(d_out['cls_logits'], d_out['logits']) + self_dist(s_out['cls_logits'], s_out['logits'])
            l_uapa = uapa(d_out['logits'], s_out['logits'])

            loss = (CFG.LAMBDA_CE * l_ce + CFG.LAMBDA_INFONCE * l_nce +
                    CFG.LAMBDA_CONSISTENCY * l_con + CFG.LAMBDA_CROSS_DIST * l_cross +
                    CFG.LAMBDA_SELF_DIST * l_self + CFG.LAMBDA_UAPA * l_uapa)

            # ProxyAnchor or Triplet
            if abl['use_proxy']:
                l_px = 0.5 * (losses['proxy'](d_out['embedding'], labels) +
                              losses['proxy'](s_out['embedding'], labels))
                loss += CFG.LAMBDA_PROXY * l_px
                loss_sums['proxy'] += l_px.item()
            else:
                l_tri = 0.5 * (losses['triplet'](d_out['embedding'], labels) +
                               losses['triplet'](s_out['embedding'], labels))
                loss += CFG.LAMBDA_TRIPLET * l_tri
                loss_sums['triplet'] += l_tri.item()

            # EMA distillation
            if abl['use_ema'] and ema is not None:
                with torch.no_grad():
                    ema_d = ema.forward(drone, alt_idx=alt_idx)
                    ema_s = ema.forward(sat, alt_idx=None)
                l_ema = 0.5 * (losses['ema_dist'](d_out['embedding'], ema_d) +
                               losses['ema_dist'](s_out['embedding'], ema_s))
                loss += CFG.LAMBDA_EMA_DIST * l_ema
                loss_sums['ema'] += l_ema.item()

            # Altitude consistency
            if abl['use_alt_consist']:
                l_alt = losses['alt_consist'](d_out['embedding'], labels, alts)
                loss += CFG.LAMBDA_ALT_CONSIST * l_alt
                loss_sums['alt_c'] += l_alt.item()

            # Masked Part Reconstruction
            if recon_active and hasattr(model, 'mask_recon'):
                l_rec_d = model.mask_recon(d_out['parts']['projected_patches'],
                                           d_out['parts']['part_features'],
                                           d_out['parts']['assignment'])
                l_rec_s = model.mask_recon(s_out['parts']['projected_patches'],
                                           s_out['parts']['part_features'],
                                           s_out['parts']['assignment'])
                l_rec = 0.5 * (l_rec_d + l_rec_s)
                loss += CFG.LAMBDA_MASK_RECON * l_rec
                loss_sums['recon'] += l_rec.item()

            # Altitude Prediction
            if abl['use_alt_pred'] and hasattr(model, 'alt_pred'):
                l_ap = model.alt_pred(d_out['embedding'].detach(), alt_norm)
                loss += CFG.LAMBDA_ALT_PRED * l_ap
                loss_sums['alt_p'] += l_ap.item()

            # Prototype Diversity
            if abl['use_diversity']:
                l_div = losses['diversity'](model.part_disc.prototypes)
                loss += CFG.LAMBDA_DIVERSITY * l_div
                loss_sums['div'] += l_div.item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()
        if abl['use_ema'] and ema is not None:
            ema.update(model)

        total_sum += loss.item(); n += 1
        loss_sums['ce'] += l_ce.item(); loss_sums['nce'] += l_nce.item()
        loss_sums['con'] += l_con.item()
        loss_sums['cross'] += l_cross.item() if torch.is_tensor(l_cross) else l_cross
        loss_sums['self'] += l_self.item(); loss_sums['uapa'] += l_uapa.item()

    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}


# =============================================================================
# RUN SINGLE ABLATION
# =============================================================================
def run_ablation(abl_cfg, test_ds_shared):
    abl_name = abl_cfg['name']
    print(f"\n{'#'*65}")
    print(f"  ABLATION: {abl_name}")
    print(f"  Disabled: {[k for k, v in abl_cfg.items() if k.startswith('use_') and not v]}")
    print(f"  Epochs: {ABL_EPOCHS}")
    print(f"{'#'*65}\n")

    set_seed(CFG.SEED)
    num_classes = CFG.NUM_CLASSES

    # Dataset
    train_ds = SUES200Dataset(CFG.SUES_ROOT, 'train', transform=get_transforms("train"))
    train_loader = DataLoader(train_ds, batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # Model
    model = AblationModel(num_classes, CFG,
                          use_fusion_gate=abl_cfg['use_fusion_gate'],
                          use_alt_film=abl_cfg['use_alt_film'],
                          use_mask_recon=abl_cfg['use_mask_recon'],
                          use_alt_pred=abl_cfg['use_alt_pred']).to(DEVICE)

    # Teacher
    teacher = None
    try:
        teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except Exception as e:
        print(f"  [WARN] Teacher failed: {e}")

    # EMA
    ema = None
    if abl_cfg['use_ema']:
        ema = EMAModel(model, decay=CFG.EMA_DECAY)

    # Losses
    infonce = SupInfoNCELoss(temp=0.05).to(DEVICE)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    consist = PartConsistencyLoss()
    cross_dist = CrossDistillationLoss()
    self_dist = SelfDistillationLoss(temperature=CFG.DISTILL_TEMP)
    uapa = UAPALoss(base_temperature=CFG.DISTILL_TEMP)

    losses = {'base': (infonce, ce, consist, cross_dist, self_dist, uapa)}

    if abl_cfg['use_proxy']:
        losses['proxy'] = ProxyAnchorLoss(num_classes, CFG.EMBED_DIM,
                                           margin=CFG.PROXY_MARGIN, alpha=CFG.PROXY_ALPHA).to(DEVICE)
    else:
        losses['triplet'] = TripletLoss(margin=0.3).to(DEVICE)

    losses['ema_dist'] = EMADistillationLoss()
    losses['alt_consist'] = AltitudeConsistencyLoss()
    losses['diversity'] = PrototypeDiversityLoss()

    # Optimizer
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith('backbone')]
    opt_groups = [
        {'params': backbone_params, 'lr': CFG.BACKBONE_LR},
        {'params': head_params,     'lr': CFG.LR},
        {'params': infonce.parameters(), 'lr': CFG.LR},
    ]
    if abl_cfg['use_proxy']:
        opt_groups.append({'params': losses['proxy'].parameters(), 'lr': CFG.LR})
    optimizer = torch.optim.AdamW(opt_groups, weight_decay=CFG.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=CFG.USE_AMP)

    best_r1 = 0.0; results_log = []

    for epoch in range(1, ABL_EPOCHS + 1):
        if epoch <= CFG.WARMUP_EPOCHS:
            lr_scale = epoch / CFG.WARMUP_EPOCHS
        else:
            progress = (epoch - CFG.WARMUP_EPOCHS) / (ABL_EPOCHS - CFG.WARMUP_EPOCHS)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        lr_scale = max(lr_scale, 0.01)
        for i, pg in enumerate(optimizer.param_groups):
            pg['lr'] = (CFG.BACKBONE_LR if i == 0 else CFG.LR) * lr_scale

        avg_loss, ld = train_one_epoch(model, teacher, ema, train_loader, losses,
                                        optimizer, scaler, DEVICE, epoch, abl_cfg)
        loss_str = " ".join(f"{k}={v:.3f}" for k, v in ld.items())
        print(f"Ep {epoch:3d}/{ABL_EPOCHS} | Loss {avg_loss:.4f} | {loss_str}")

        if epoch % ABL_EVAL_INTERVAL == 0 or epoch == ABL_EPOCHS:
            metrics = evaluate(model, test_ds_shared, DEVICE,
                               use_alt_film=abl_cfg['use_alt_film'])
            results_log.append({'epoch': epoch, **metrics})
            print(f"  ► R@1: {metrics['R@1']:.2f}%  R@5: {metrics['R@5']:.2f}%  "
                  f"R@10: {metrics['R@10']:.2f}%  mAP: {metrics['mAP']:.2f}%")
            if metrics['R@1'] > best_r1:
                best_r1 = metrics['R@1']
                print(f"  ★ New best R@1: {best_r1:.2f}%")

    # Cleanup
    del model, teacher, ema, optimizer, scaler, train_ds, train_loader
    for k in list(losses.keys()):
        del losses[k]
    gc.collect(); torch.cuda.empty_cache()

    return {'name': abl_name, 'best_r1': best_r1, 'log': results_log}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    runs = ABLATION_RUNS[ABLATION_GROUP]

    print("=" * 65)
    print(f"  ABLATION STUDY — SPDGeo-DPEA-MAR EXP35 (Group {ABLATION_GROUP})")
    print(f"  Ablations: {[r['name'] for r in runs]}")
    print(f"  Epochs per ablation: {ABL_EPOCHS}")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    # Shared test dataset
    test_ds = SUES200Dataset(CFG.SUES_ROOT, 'test', transform=get_transforms("test"))

    all_results = []
    for i, abl_cfg in enumerate(runs):
        print(f"\n{'='*65}")
        print(f"  [{i+1}/{len(runs)}] Starting ablation: {abl_cfg['name']}")
        print(f"{'='*65}")
        result = run_ablation(abl_cfg, test_ds)
        all_results.append(result)

    # Final summary
    print(f"\n{'='*65}")
    print(f"  ABLATION RESULTS — Group {ABLATION_GROUP}")
    print(f"{'='*65}")
    print(f"  {'Ablation':<25s}  {'Best R@1':>8s}  {'Ep10':>6s}  {'Ep20':>6s}  {'Ep30':>6s}  {'Ep40':>6s}  {'Ep50':>6s}  {'Ep60':>6s}")
    print(f"  {'-'*85}")
    for res in all_results:
        ep_vals = {r['epoch']: r['R@1'] for r in res['log']}
        row = f"  {res['name']:<25s}  {res['best_r1']:7.2f}%"
        for ep in [10, 20, 30, 40, 50, 60]:
            row += f"  {ep_vals.get(ep, 0):5.1f}%"
        print(row)
    print(f"{'='*65}")
    print(f"\n  Compare against full EXP35 to see each component's contribution.")

    # Save
    with open(os.path.join(CFG.OUTPUT_DIR, f'ablation_group{ABLATION_GROUP}_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == '__main__':
    main()

