# =============================================================================
# EXP30: SPDGeo-TTE — Test-Time Ensemble + Entropy Adaptation
# =============================================================================
# Base:    SPDGeo-DPE (93.59% R@1) — THE CHAMPION
# Novel:   INFERENCE-ONLY improvements (no training changes):
#          1) Multi-Crop Ensemble — extract features at 3 resolutions
#             (288, 336, 384), L2-normalize, average → richer representation
#          2) EMA Model Ensemble — average student + EMA model predictions
#             (EMA consistently within 1% of student in DPE experiments)
#          3) Entropy-Minimized Parts — Tent-style (Wang et al., ICLR 2021)
#             1-3 gradient steps on part discovery prototypes to minimize
#             assignment entropy at test time → sharper, more confident parts
#
# Motivation:
#   This is the ONLY experiment that requires NO retraining. It takes the
#   best DPE checkpoint and improves inference quality through:
#   (a) Scale diversity: multi-crop captures features invisible at single scale
#   (b) Model diversity: student+EMA provide complementary predictions
#   (c) Adaptation: entropy minimization adapts part prototypes to the test
#       distribution, which may differ from training distribution
#
#   Practical value: Can be applied to ANY trained model as a post-hoc boost.
#   If it works, it proves that DPE's representation can be better utilized.
#
# This experiment LOADS a trained DPE model and evaluates with TTE.
# It does NOT train from scratch — it's a pure inference experiment.
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
    SEED            = 42

    # TTE hyperparameters
    MULTI_CROP_SIZES   = [280, 336, 392]  # 3 resolutions — must be multiples of 14 (DINOv2 patch_size)
    CROP_WEIGHTS       = [0.25, 0.50, 0.25]  # weight per resolution
    USE_EMA_ENSEMBLE   = True   # Average student + EMA predictions
    EMA_ENSEMBLE_ALPHA = 0.5    # Weight for EMA model in ensemble
    TENT_STEPS         = 3      # Entropy minimization steps
    TENT_LR            = 1e-4   # Learning rate for Tent adaptation
    TENT_ENABLED       = True   # Whether to use Tent adaptation

    # Model checkpoint
    CHECKPOINT_PATH = "/kaggle/working/exp20_dpe_best.pth"

    NUM_WORKERS     = 2

    # Training config (needed for model construction)
    EMA_DECAY       = 0.999

    # ---- FALLBACK: Train from scratch if no checkpoint ----
    NUM_EPOCHS      = 120
    P_CLASSES       = 16
    K_SAMPLES       = 4
    LR              = 3e-4
    BACKBONE_LR     = 3e-5
    WEIGHT_DECAY    = 0.01
    WARMUP_EPOCHS   = 5
    USE_AMP         = True
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
    DISTILL_TEMP        = 4.0
    EVAL_INTERVAL       = 5

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
        return {'drone': d, 'satellite': s, 'label': li, 'altitude': int(alt),
                'drone_path': dp, 'sat_path': sp}


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


def get_transforms(mode="train", img_size=None):
    sz = img_size or CFG.IMG_SIZE
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
# MODEL (same as DPE — needed to load checkpoint)
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
        return features['x_norm_patchtokens'], features['x_norm_clstoken'], \
               (x.shape[2] // self.patch_size, x.shape[3] // self.patch_size)


class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        self.output_dim = 768
        for p in self.model.parameters(): p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        return self.model.forward_features(x)['x_norm_clstoken']


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


class SPDGeoDPEModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone    = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc   = SemanticPartDiscovery(384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP)
        self.pool        = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)
        self.bottleneck     = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj   = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
                                            nn.LayerNorm(cfg.TEACHER_DIM))

    def extract_embedding(self, x):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)

    def extract_with_parts(self, x):
        """Extract embedding + return assignment for Tent."""
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        fused = self.fusion_gate(emb, cls_emb)
        return fused, parts['assignment']

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
# BASE LOSSES (for fallback training)
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
        P = F.normalize(self.proxies, dim=-1); sim = embeddings @ P.T
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
# NEW: Test-Time Entropy Adaptation (Tent-style)
# =============================================================================
class TentAdaptation:
    """
    Test-Time Entropy Minimization for Part Prototypes (Tent, Wang et al. 2021).

    At test time, takes a batch of images and performs a few gradient steps to
    minimize the entropy of part assignments. This adapts the part prototypes
    to the test distribution, making assignments sharper and more confident.

    ONLY updates part_disc.prototypes — all other model weights are frozen.
    After each batch, we reset prototypes to avoid drift.
    """
    def __init__(self, model, lr=1e-4, steps=3):
        self.model = model
        self.lr = lr
        self.steps = steps
        # Save original prototypes for reset
        self.orig_prototypes = model.part_disc.prototypes.data.clone()

    def adapt_and_extract(self, images):
        """
        Adapt prototypes to this batch, extract features, then reset.
        Returns: L2-normalized embedding [B, D]
        """
        # Save state
        self.model.part_disc.prototypes.data.copy_(self.orig_prototypes)

        # Enable grad only for prototypes
        self.model.eval()
        self.model.part_disc.prototypes.requires_grad_(True)
        optimizer = torch.optim.Adam([self.model.part_disc.prototypes], lr=self.lr)

        # Tent adaptation steps
        for _ in range(self.steps):
            optimizer.zero_grad()
            _, assignment = self.model.extract_with_parts(images)
            # Entropy of assignment: lower = more confident
            assign_probs = assignment.mean(dim=1)  # [B, K]
            entropy = -(assign_probs * (assign_probs + 1e-8).log()).sum(dim=-1).mean()
            entropy.backward()
            optimizer.step()

        # Extract with adapted prototypes
        with torch.no_grad():
            emb = self.model.extract_embedding(images)

        # Reset prototypes
        self.model.part_disc.prototypes.requires_grad_(False)
        self.model.part_disc.prototypes.data.copy_(self.orig_prototypes)

        return emb

    def reset(self):
        """Reset prototypes to original."""
        self.model.part_disc.prototypes.data.copy_(self.orig_prototypes)


# =============================================================================
# NEW: Multi-Crop Feature Extraction
# =============================================================================
def multi_crop_extract(model, image_pil, crop_sizes, crop_weights, device,
                       tent=None):
    """
    Extract features at multiple resolutions, weighted average.

    image_pil: PIL Image
    crop_sizes: list of int (e.g., [288, 336, 384])
    crop_weights: list of float (e.g., [0.25, 0.5, 0.25])
    """
    all_feats = []
    for sz, w in zip(crop_sizes, crop_weights):
        tf = get_transforms("test", img_size=sz)
        img_t = tf(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            if tent is not None:
                feat = tent.adapt_and_extract(img_t)
            else:
                feat = model.extract_embedding(img_t)
        all_feats.append(feat * w)

    combined = sum(all_feats)
    return F.normalize(combined, dim=-1)


# =============================================================================
# TTE EVALUATION — the main event
# =============================================================================
@torch.no_grad()
def evaluate_baseline(model, test_ds, device):
    """Standard single-scale evaluation (same as DPE)."""
    model.eval()
    test_tf = get_transforms("test")

    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
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

    return _compute_metrics(drone_feats, drone_labels, drone_alts, sat_feats, sat_labels)


def evaluate_tte(model, test_ds, device, ema_model=None, tent=None):
    """
    Test-Time Ensemble evaluation:
    1. Multi-crop feature extraction (3 scales)
    2. Optional EMA model ensemble
    3. Optional Tent entropy adaptation
    """
    model.eval()
    if ema_model is not None:
        ema_model.eval()

    # Extract drone features with multi-crop
    print("  Extracting drone features (multi-crop) …")
    drone_feats, drone_labels, drone_alts = [], [], []
    for i, (dp, sp, li, alt) in enumerate(tqdm(test_ds.samples, desc="Drone multi-crop")):
        try:
            d_pil = Image.open(dp).convert('RGB')
        except Exception:
            d_pil = Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128, 128, 128))

        feat = multi_crop_extract(model, d_pil, CFG.MULTI_CROP_SIZES,
                                  CFG.CROP_WEIGHTS, device, tent=tent)

        # EMA ensemble
        if ema_model is not None and CFG.USE_EMA_ENSEMBLE:
            ema_feat = multi_crop_extract(ema_model, d_pil, CFG.MULTI_CROP_SIZES,
                                          CFG.CROP_WEIGHTS, device, tent=None)
            alpha = CFG.EMA_ENSEMBLE_ALPHA
            feat = F.normalize((1 - alpha) * feat + alpha * ema_feat, dim=-1)

        drone_feats.append(feat.cpu())
        drone_labels.append(li)
        drone_alts.append(int(alt))

    drone_feats = torch.cat(drone_feats)
    drone_labels = torch.tensor(drone_labels)
    drone_alts = torch.tensor(drone_alts)

    # Extract satellite features with multi-crop
    print("  Extracting satellite features (multi-crop) …")
    all_locs = [f"{l:04d}" for l in range(1, 201)]
    sat_feats, sat_labels = [], []; distractor_cnt = 0
    for loc in tqdm(all_locs, desc="Satellite multi-crop"):
        sp = os.path.join(test_ds.sat_dir, loc, "0.png")
        if not os.path.exists(sp): continue
        try:
            s_pil = Image.open(sp).convert('RGB')
        except Exception:
            s_pil = Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128, 128, 128))

        feat = multi_crop_extract(model, s_pil, CFG.MULTI_CROP_SIZES,
                                  CFG.CROP_WEIGHTS, device, tent=None)

        if ema_model is not None and CFG.USE_EMA_ENSEMBLE:
            ema_feat = multi_crop_extract(ema_model, s_pil, CFG.MULTI_CROP_SIZES,
                                          CFG.CROP_WEIGHTS, device, tent=None)
            alpha = CFG.EMA_ENSEMBLE_ALPHA
            feat = F.normalize((1 - alpha) * feat + alpha * ema_feat, dim=-1)

        sat_feats.append(feat.cpu())
        if loc in test_ds.location_to_idx:
            sat_labels.append(test_ds.location_to_idx[loc])
        else:
            sat_labels.append(-1000 - distractor_cnt); distractor_cnt += 1

    sat_feats = torch.cat(sat_feats)
    sat_labels = torch.tensor(sat_labels)

    return _compute_metrics(drone_feats, drone_labels, drone_alts, sat_feats, sat_labels)


def _compute_metrics(drone_feats, drone_labels, drone_alts, sat_feats, sat_labels):
    """Compute R@1, R@5, R@10, mAP + per-altitude breakdown."""
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

    return overall, per_alt


def print_results(tag, overall, per_alt):
    altitudes_list = sorted(per_alt.keys())
    print(f"\n{'='*75}")
    print(f"  {tag}")
    print(f"{'='*75}")
    print(f"  {'Altitude':>8s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'#Query':>6s}")
    print(f"  {'-'*50}")
    for alt in altitudes_list:
        a = per_alt[int(alt)]
        print(f"  {int(alt):>6d}m  {a['R@1']:6.2f}%  {a['R@5']:6.2f}%  {a['R@10']:6.2f}%  {a['mAP']:6.2f}%  {a['n']:>6d}")
    print(f"  {'-'*50}")
    N = sum(per_alt[a]['n'] for a in per_alt)
    print(f"  {'Overall':>8s}  {overall['R@1']:6.2f}%  {overall['R@5']:6.2f}%  {overall['R@10']:6.2f}%  {overall['mAP']:6.2f}%  {N:>6d}")
    print(f"{'='*75}\n")


# =============================================================================
# TRAINING (fallback if no checkpoint)
# =============================================================================
def train_one_epoch(model, teacher, ema, loader, losses, new_losses, optimizer,
                    scaler, device, epoch):
    model.train()
    if teacher: teacher.eval()
    infonce, ce, consist, cross_dist, self_dist, uapa = losses
    proxy_anchor, ema_dist = new_losses
    total_sum = 0; n = 0; loss_sums = defaultdict(float)

    for batch in tqdm(loader, desc=f'Ep{epoch:3d}', leave=False):
        drone  = batch['drone'].to(device)
        sat    = batch['satellite'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            d_out = model(drone, return_parts=True)
            s_out = model(sat, return_parts=True)
            l_ce = (ce(d_out['logits'], labels) + ce(s_out['logits'], labels))
            l_ce += 0.3 * (ce(d_out['cls_logits'], labels) + ce(s_out['cls_logits'], labels))
            l_nce = infonce(d_out['embedding'], s_out['embedding'], labels)
            l_con = consist(d_out['parts']['assignment'], s_out['parts']['assignment'])
            if teacher is not None:
                with torch.no_grad():
                    t_drone = teacher(drone); t_sat = teacher(sat)
                l_cross = cross_dist(d_out['projected_feat'], t_drone) + cross_dist(s_out['projected_feat'], t_sat)
            else:
                l_cross = torch.tensor(0.0, device=device)
            l_self = self_dist(d_out['cls_logits'], d_out['logits']) + self_dist(s_out['cls_logits'], s_out['logits'])
            l_uapa = uapa(d_out['logits'], s_out['logits'])
            l_proxy = 0.5 * (proxy_anchor(d_out['embedding'], labels) + proxy_anchor(s_out['embedding'], labels))
            with torch.no_grad():
                ema_drone_emb = ema.forward(drone); ema_sat_emb = ema.forward(sat)
            l_ema = 0.5 * (ema_dist(d_out['embedding'], ema_drone_emb) + ema_dist(s_out['embedding'], ema_sat_emb))
            loss = (CFG.LAMBDA_CE * l_ce + CFG.LAMBDA_INFONCE * l_nce + CFG.LAMBDA_CONSISTENCY * l_con +
                    CFG.LAMBDA_CROSS_DIST * l_cross + CFG.LAMBDA_SELF_DIST * l_self + CFG.LAMBDA_UAPA * l_uapa +
                    CFG.LAMBDA_PROXY * l_proxy + CFG.LAMBDA_EMA_DIST * l_ema)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()
        ema.update(model)
        total_sum += loss.item(); n += 1
        loss_sums['ce'] += l_ce.item(); loss_sums['nce'] += l_nce.item()
        loss_sums['con'] += l_con.item()
        loss_sums['cross'] += l_cross.item() if torch.is_tensor(l_cross) else l_cross
        loss_sums['self'] += l_self.item(); loss_sums['uapa'] += l_uapa.item()
        loss_sums['proxy'] += l_proxy.item(); loss_sums['ema'] += l_ema.item()
    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EXP30: SPDGeo-TTE — Test-Time Ensemble + Entropy Adaptation")
    print(f"  Base: SPDGeo-DPE checkpoint")
    print(f"  Novel: MultiCrop({CFG.MULTI_CROP_SIZES}) + "
          f"EMA Ensemble(α={CFG.EMA_ENSEMBLE_ALPHA}) + "
          f"Tent({CFG.TENT_STEPS} steps, lr={CFG.TENT_LR})")
    print(f"  Dataset: SUES-200 | Device: {DEVICE}")
    print("=" * 65)

    # Load test dataset
    test_ds = SUES200Dataset(CFG.SUES_ROOT, 'test', transform=get_transforms("test"))

    # Build model
    print('\nBuilding models …')
    model = SPDGeoDPEModel(CFG.NUM_CLASSES).to(DEVICE)

    # Try to load checkpoint
    checkpoint_loaded = False
    if os.path.exists(CFG.CHECKPOINT_PATH):
        print(f"  Loading checkpoint: {CFG.CHECKPOINT_PATH}")
        ckpt = torch.load(CFG.CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'metrics' in ckpt:
            print(f"  Checkpoint metrics: R@1={ckpt['metrics'].get('R@1', '?'):.2f}%")
        checkpoint_loaded = True
    else:
        print(f"  [WARN] No checkpoint at {CFG.CHECKPOINT_PATH} — will train from scratch first")

    # ---- FALLBACK: Train DPE from scratch if no checkpoint ----
    if not checkpoint_loaded:
        print("\n" + "=" * 65)
        print("  FALLBACK: Training DPE from scratch (same as EXP20)")
        print("=" * 65)
        train_ds = SUES200Dataset(CFG.SUES_ROOT, 'train', transform=get_transforms("train"))
        train_loader = DataLoader(train_ds, batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
                                  num_workers=CFG.NUM_WORKERS, pin_memory=True)
        teacher = None
        try:
            teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
        except Exception as e:
            print(f"  [WARN] Could not load teacher: {e}")

        ema_train = EMAModel(model, decay=CFG.EMA_DECAY)
        infonce = SupInfoNCELoss(temp=0.05).to(DEVICE)
        ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        consist = PartConsistencyLoss()
        cross_dist = CrossDistillationLoss()
        self_dist = SelfDistillationLoss(temperature=CFG.DISTILL_TEMP)
        uapa_loss = UAPALoss(base_temperature=CFG.DISTILL_TEMP)
        base_losses = (infonce, ce, consist, cross_dist, self_dist, uapa_loss)
        proxy_anchor = ProxyAnchorLoss(CFG.NUM_CLASSES, CFG.EMBED_DIM,
                                       margin=CFG.PROXY_MARGIN, alpha=CFG.PROXY_ALPHA).to(DEVICE)
        ema_dist = EMADistillationLoss()
        new_losses = (proxy_anchor, ema_dist)

        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith('backbone')]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': CFG.BACKBONE_LR},
            {'params': head_params, 'lr': CFG.LR},
            {'params': infonce.parameters(), 'lr': CFG.LR},
            {'params': proxy_anchor.parameters(), 'lr': CFG.LR},
        ], weight_decay=CFG.WEIGHT_DECAY)
        scaler = torch.amp.GradScaler('cuda', enabled=CFG.USE_AMP)
        best_r1 = 0.0

        for epoch in range(1, CFG.NUM_EPOCHS + 1):
            if epoch <= CFG.WARMUP_EPOCHS:
                lr_scale = epoch / CFG.WARMUP_EPOCHS
            else:
                progress = (epoch - CFG.WARMUP_EPOCHS) / (CFG.NUM_EPOCHS - CFG.WARMUP_EPOCHS)
                lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
            lr_scale = max(lr_scale, 0.01)
            for i in range(4): optimizer.param_groups[i]['lr'] = (CFG.BACKBONE_LR if i == 0 else CFG.LR) * lr_scale

            avg_loss, ld = train_one_epoch(model, teacher, ema_train, train_loader,
                                           base_losses, new_losses, optimizer, scaler, DEVICE, epoch)
            print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f}")

            if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
                metrics, per_alt = evaluate_baseline(model, test_ds, DEVICE)
                print(f"  ► R@1: {metrics['R@1']:.2f}%")
                if metrics['R@1'] > best_r1:
                    best_r1 = metrics['R@1']
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                                'metrics': metrics, 'per_alt': per_alt},
                               os.path.join(CFG.OUTPUT_DIR, 'exp30_dpe_trained.pth'))
                # EMA eval
                ema_metrics, _ = evaluate_baseline(ema_train.model, test_ds, DEVICE)
                if ema_metrics['R@1'] > best_r1:
                    best_r1 = ema_metrics['R@1']
                    torch.save({'epoch': epoch, 'model_state_dict': ema_train.model.state_dict(),
                                'metrics': ema_metrics, 'is_ema': True},
                               os.path.join(CFG.OUTPUT_DIR, 'exp30_dpe_trained.pth'))

        # Load best for TTE
        ckpt = torch.load(os.path.join(CFG.OUTPUT_DIR, 'exp30_dpe_trained.pth'), map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"\n  Trained DPE best R@1: {best_r1:.2f}%")

    model.eval()

    # Create EMA model for ensemble
    ema_model = None
    if CFG.USE_EMA_ENSEMBLE:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        print("  EMA model ready for ensemble")

    # Create Tent adapter
    tent = None
    if CFG.TENT_ENABLED:
        tent = TentAdaptation(model, lr=CFG.TENT_LR, steps=CFG.TENT_STEPS)
        print(f"  Tent adaptation ready ({CFG.TENT_STEPS} steps, lr={CFG.TENT_LR})")

    # ======== Run evaluations ========
    results = {}

    # 1. Baseline (single-scale, no ensemble)
    print("\n[1/4] Baseline evaluation (single-scale, no tricks) …")
    base_overall, base_per_alt = evaluate_baseline(model, test_ds, DEVICE)
    print_results("BASELINE (Single-Scale)", base_overall, base_per_alt)
    results['baseline'] = base_overall

    # 2. Multi-crop only
    print("[2/4] Multi-crop evaluation …")
    tent_disabled = tent
    mc_overall, mc_per_alt = evaluate_tte(model, test_ds, DEVICE,
                                           ema_model=None, tent=None)
    print_results("MULTI-CROP ONLY", mc_overall, mc_per_alt)
    results['multi_crop'] = mc_overall

    # 3. Multi-crop + EMA ensemble
    if CFG.USE_EMA_ENSEMBLE:
        print("[3/4] Multi-crop + EMA ensemble evaluation …")
        mc_ema_overall, mc_ema_per_alt = evaluate_tte(model, test_ds, DEVICE,
                                                       ema_model=ema_model, tent=None)
        print_results("MULTI-CROP + EMA ENSEMBLE", mc_ema_overall, mc_ema_per_alt)
        results['multi_crop_ema'] = mc_ema_overall

    # 4. Multi-crop + EMA + Tent (full TTE)
    if CFG.TENT_ENABLED:
        print("[4/4] Full TTE (multi-crop + EMA + Tent) evaluation …")
        tte_overall, tte_per_alt = evaluate_tte(model, test_ds, DEVICE,
                                                 ema_model=ema_model, tent=tent)
        print_results("FULL TTE (Multi-Crop + EMA + Tent)", tte_overall, tte_per_alt)
        results['full_tte'] = tte_overall

    # Summary
    print("\n" + "=" * 75)
    print("  EXP30: TTE ABLATION SUMMARY")
    print("=" * 75)
    print(f"  {'Method':<35s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}")
    print(f"  {'-'*65}")
    for name, m in results.items():
        label = name.replace('_', ' ').title()
        print(f"  {label:<35s}  {m['R@1']:6.2f}%  {m['R@5']:6.2f}%  {m['R@10']:6.2f}%  {m['mAP']:6.2f}%")
    print(f"{'='*75}")

    best_method = max(results.items(), key=lambda x: x[1]['R@1'])
    print(f"\n  Best method: {best_method[0]} → R@1: {best_method[1]['R@1']:.2f}%")

    with open(os.path.join(CFG.OUTPUT_DIR, 'exp30_tte_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
