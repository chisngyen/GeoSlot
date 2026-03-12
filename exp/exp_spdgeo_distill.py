# =============================================================================
# SPDGeo-D: Semantic Part Discovery + Multi-Level Distillation
# =============================================================================
# Merges:
#   • SPDGeo  — DINOv2-S backbone + SemanticPartDiscovery + PartAwarePooling
#   • MobileGeo baseline — Cross-distillation (DINOv2-B teacher),
#                          Self-distillation (part-aware → CLS branch),
#                          Uncertainty-Aware Prediction Alignment (UAPA)
#
# Architecture:
#   Teacher  — DINOv2 ViT-B/14 (fully frozen, 768-dim CLS token)
#   Student  — DINOv2 ViT-S/14 (fine-tune last 4 blocks, 384-dim patches)
#            + SemanticPartDiscovery (K=8 semantic parts via soft clustering)
#            + PartAwarePooling    (salience-weighted → 512-dim global embed)
#            + CLS branch          (DINOv2 CLS token → 512-dim auxiliary head)
#            + TeacherProjector    (512 → 768, bridges to teacher space)
#
# Losses (7 components):
#   1. CE Loss          — classification, both branches, drone & satellite
#   2. SupInfoNCE       — label-aware contrastive, drone ↔ satellite (learnable T)
#   3. Triplet          — hard-negative mining within batch
#   4. PartConsistency  — same-location → similar part distribution (sym-KL)
#   5. CrossDistill     — DINOv2-B CLS → student projected embed (MSE + Cosine)
#   6. SelfDistill      — part-aware logits → CLS logits (KD, T=4)
#   7. UAPA             — Uncertainty-Aware Prediction Alignment (drone ↔ sat)
#
# Eval:
#   Full 200-location satellite gallery (80 test + 120 train distractors)
#   matches the standard SUES-200 confusion-data protocol from the baseline.
#
# Training:
#   120 epochs, cosine warmup, PK sampling (P=16 × K=4)
#   Differential LR: backbone 0.1× vs heads 1× (AdamW)
#   Mixed-precision AMP (torch.amp new API)
#
# Self-contained for Kaggle — no external imports beyond timm/tqdm
# =============================================================================

import subprocess, sys
for _p in ["timm", "tqdm"]:
    try:
        __import__(_p)
    except ImportError:
        subprocess.run(f"pip install -q {_p}", shell=True, capture_output=True)

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
    # ── Data ──────────────────────────────────────────────────────────────────
    SUES_ROOT       = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    DRONE_DIR       = "drone-view"
    SAT_DIR         = "satellite-view"
    OUTPUT_DIR      = "/kaggle/working"
    ALTITUDES       = ["150", "200", "250", "300"]
    TRAIN_LOCS      = list(range(1, 121))    # 120 train locations
    TEST_LOCS       = list(range(121, 201))  # 80 test  locations
    NUM_CLASSES     = 120

    # ── Model ─────────────────────────────────────────────────────────────────
    IMG_SIZE        = 336        # 24×24 = 576 patches for DINOv2-S/14
    N_PARTS         = 8          # semantic parts to discover
    PART_DIM        = 256        # part feature dimension
    EMBED_DIM       = 512        # final student embedding dimension
    TEACHER_DIM     = 768        # DINOv2-B CLS dimension
    CLUSTER_TEMP    = 0.07       # clustering temperature
    UNFREEZE_BLOCKS = 4          # DINOv2-S blocks to fine-tune

    # ── Training ──────────────────────────────────────────────────────────────
    NUM_EPOCHS      = 120
    P_CLASSES       = 16         # classes per PK batch
    K_SAMPLES       = 4          # samples per class
    LR              = 3e-4       # base learning rate (heads)
    BACKBONE_LR     = 3e-5       # 0.1× for backbone fine-tuning
    WEIGHT_DECAY    = 0.01
    WARMUP_EPOCHS   = 5
    USE_AMP         = True
    SEED            = 42

    # ── Loss weights ──────────────────────────────────────────────────────────
    LAMBDA_CE           = 1.0    # cross-entropy (both branches)
    LAMBDA_INFONCE      = 1.0    # SupInfoNCE cross-view contrastive
    LAMBDA_TRIPLET      = 0.5    # hard-triplet
    LAMBDA_CONSISTENCY  = 0.1    # part-distribution consistency
    LAMBDA_CROSS_DIST   = 0.3    # DINOv2-B → student cross-distillation
    LAMBDA_SELF_DIST    = 0.3    # part-aware → CLS self-distillation
    LAMBDA_UAPA         = 0.2    # uncertainty-aware drone↔sat alignment

    # Distillation temperatures
    DISTILL_TEMP        = 4.0    # KD temperature for self-distill & UAPA

    # ── Eval ──────────────────────────────────────────────────────────────────
    EVAL_INTERVAL   = 5
    NUM_WORKERS     = 2

CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


# =============================================================================
# DATASET — SUES-200
# =============================================================================

class SUES200Dataset(Dataset):
    """
    SUES-200 Dataset.  Standard benchmark split: 120 train / 80 test (fixed).
    Each sample returns one drone image + its paired satellite image + label.
    """

    def __init__(self, root, mode="train", altitudes=None, transform=None):
        self.root      = root
        self.mode      = mode
        self.altitudes = altitudes or CFG.ALTITUDES
        self.transform = transform
        self.drone_dir = os.path.join(root, CFG.DRONE_DIR)
        self.sat_dir   = os.path.join(root, CFG.SAT_DIR)

        loc_ids              = CFG.TRAIN_LOCS if mode == "train" else CFG.TEST_LOCS
        self.locations       = [f"{l:04d}" for l in loc_ids]
        self.location_to_idx = {l: i for i, l in enumerate(self.locations)}

        self.samples            = []
        self.drone_by_location  = defaultdict(list)

        for loc in self.locations:
            li   = self.location_to_idx[loc]
            sp   = os.path.join(self.sat_dir, loc, "0.png")
            if not os.path.exists(sp):
                continue
            for alt in self.altitudes:
                ad = os.path.join(self.drone_dir, loc, alt)
                if not os.path.isdir(ad):
                    continue
                for img in sorted(os.listdir(ad)):
                    if img.endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(ad, img), sp, li, alt))
                        self.drone_by_location[li].append(len(self.samples) - 1)

        print(f"  [{mode}] {len(self.samples)} samples, {len(self.locations)} locations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dp, sp, li, alt = self.samples[idx]
        try:
            d = Image.open(dp).convert('RGB')
            s = Image.open(sp).convert('RGB')
        except Exception:
            sz = CFG.IMG_SIZE
            d = Image.new('RGB', (sz, sz), (128, 128, 128))
            s = Image.new('RGB', (sz, sz), (128, 128, 128))
        if self.transform:
            d = self.transform(d)
            s = self.transform(s)
        return {'drone': d, 'satellite': s, 'label': li, 'altitude': int(alt)}


class PKSampler:
    """P-K Sampler: P locations × K samples per batch."""

    def __init__(self, ds, p, k):
        self.ds   = ds
        self.p    = p
        self.k    = k
        self.locs = list(ds.drone_by_location.keys())

    def __iter__(self):
        locs  = self.locs.copy()
        random.shuffle(locs)
        batch = []
        for l in locs:
            idx = self.ds.drone_by_location[l]
            if len(idx) < self.k:
                idx = idx * (self.k // len(idx) + 1)
            batch.extend(random.sample(idx, self.k))
            if len(batch) >= self.p * self.k:
                yield batch[: self.p * self.k]
                batch = batch[self.p * self.k :]

    def __len__(self):
        return len(self.locs) // self.p


def get_transforms(mode="train"):
    sz = CFG.IMG_SIZE
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(sz, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.RandomGrayscale(p=0.02),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


# =============================================================================
# BACKBONE — DINOv2 ViT-S/14 (Student)
# =============================================================================

class DINOv2Backbone(nn.Module):
    """DINOv2 ViT-S/14 with selective block unfreezing."""

    def __init__(self, unfreeze_blocks=4):
        super().__init__()
        self.model       = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                                          pretrained=True)
        self.feature_dim = 384
        self.patch_size  = 14

        for p in self.model.parameters():
            p.requires_grad = False

        for blk in self.model.blocks[-unfreeze_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True

        for p in self.model.norm.parameters():
            p.requires_grad = True

        frozen    = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  DINOv2 ViT-S/14: {frozen/1e6:.1f}M frozen, {trainable/1e6:.1f}M trainable")

    def forward(self, x):
        features     = self.model.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']   # [B, N, 384]
        cls_token    = features['x_norm_clstoken']      # [B, 384]
        H = x.shape[2] // self.patch_size
        W = x.shape[3] // self.patch_size
        return patch_tokens, cls_token, (H, W)


# =============================================================================
# TEACHER — DINOv2 ViT-B/14 (fully frozen)
# =============================================================================

class DINOv2Teacher(nn.Module):
    """
    DINOv2 ViT-B/14 teacher — fully frozen.
    Returns the norm'd CLS token [B, 768] for cross-distillation.
    """

    def __init__(self):
        super().__init__()
        print("  Loading DINOv2 ViT-B/14 teacher …")
        self.model      = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14',
                                         pretrained=True)
        self.output_dim = 768

        for p in self.model.parameters():
            p.requires_grad = False

        print("  DINOv2 ViT-B/14 teacher loaded (all frozen).")

    @torch.no_grad()
    def forward(self, x):
        return self.model.forward_features(x)['x_norm_clstoken']   # [B, 768]


# =============================================================================
# SEMANTIC PART DISCOVERY
# =============================================================================

class SemanticPartDiscovery(nn.Module):
    """
    Differentiable soft-clustering of patch features into K semantic parts.
    Each prototype represents one semantic concept (building, road, vegetation…).
    """

    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07):
        super().__init__()
        self.n_parts     = n_parts
        self.temperature = temperature

        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, part_dim),
            nn.LayerNorm(part_dim),
            nn.GELU(),
        )

        # Learnable semantic prototypes
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)

        # Refinement FFN
        self.refine = nn.Sequential(
            nn.LayerNorm(part_dim),
            nn.Linear(part_dim, part_dim * 2),
            nn.GELU(),
            nn.Linear(part_dim * 2, part_dim),
        )

        # Per-part salience estimator
        self.salience_head = nn.Sequential(
            nn.Linear(part_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, patch_features, spatial_hw):
        B, N, _ = patch_features.shape
        H, W    = spatial_hw

        feat      = self.feat_proj(patch_features)               # [B, N, D]
        feat_norm  = F.normalize(feat, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)

        # Soft assignment: [B, N, K]
        sim    = torch.einsum('bnd,kd->bnk', feat_norm, proto_norm) / self.temperature
        assign = F.softmax(sim, dim=-1)

        # Part features: mass-weighted aggregation + residual refinement
        assign_t = assign.transpose(1, 2)                         # [B, K, N]
        mass     = assign_t.sum(-1, keepdim=True).clamp(min=1e-6)
        part_feat = torch.bmm(assign_t, feat) / mass              # [B, K, D]
        part_feat = part_feat + self.refine(part_feat)

        # Part spatial centre-of-mass (for potential spatial loss extensions)
        device      = feat.device
        gy          = torch.arange(H, device=device).float() / max(H - 1, 1)
        gx          = torch.arange(W, device=device).float() / max(W - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords      = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos    = torch.bmm(assign_t, coords.unsqueeze(0).expand(B, -1, -1)) / mass

        salience = self.salience_head(part_feat).squeeze(-1)      # [B, K]

        return {
            'part_features': part_feat,    # [B, K, D]
            'part_positions': part_pos,    # [B, K, 2]
            'assignment':     assign,      # [B, N, K]
            'salience':       salience,    # [B, K]
        }


# =============================================================================
# PART-AWARE POOLING
# =============================================================================

class PartAwarePooling(nn.Module):
    """
    Aggregates K part features into a single normalized embedding.
    Salience is incorporated as a log-prior before the softmax:
        softmax(logit + log(salience)) ∝ salience · exp(logit)
    i.e. high-salience parts are boosted multiplicatively.
    """

    def __init__(self, part_dim=256, embed_dim=512):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(part_dim, part_dim // 2),
            nn.Tanh(),
            nn.Linear(part_dim // 2, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(part_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, part_features, salience=None):
        B, K, D = part_features.shape
        aw = self.attn(part_features)                          # [B, K, 1]
        if salience is not None:
            # log-domain multiplicative prior (clamp avoids -inf)
            aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)

        attn_pool = (aw * part_features).sum(1)                # [B, D]
        mean_pool = part_features.mean(1)
        max_pool  = part_features.max(1)[0]

        combined = torch.cat([attn_pool, mean_pool, max_pool], dim=-1)   # [B, 3D]
        return F.normalize(self.proj(combined), dim=-1)                  # [B, E]


# =============================================================================
# STUDENT MODEL — SPDGeoDistillModel
# =============================================================================

class SPDGeoDistillModel(nn.Module):
    """
    SPDGeo student model augmented for multi-level distillation.

    Outputs (forward):
        embedding      — L2-normalized fused embedding for contrastive losses
        logits         — part-aware classification logits (main branch)
        cls_logits     — CLS-token classification logits (auxiliary branch)
        projected_feat — teacher_proj(part_emb) ∈ R^768 (cross-distillation)
        parts          — SemanticPartDiscovery output dict (if return_parts=True)
    """

    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone  = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = SemanticPartDiscovery(384, cfg.N_PARTS,
                                               cfg.PART_DIM, cfg.CLUSTER_TEMP)
        self.pool      = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)

        # Part-aware classification branch
        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
            nn.BatchNorm1d(cfg.EMBED_DIM),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(cfg.EMBED_DIM, num_classes)

        # CLS-token auxiliary branch
        self.cls_proj = nn.Sequential(
            nn.Linear(384, cfg.EMBED_DIM),
            nn.BatchNorm1d(cfg.EMBED_DIM),
            nn.ReLU(inplace=True),
        )
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)

        # ── NEW: project student embed → teacher space for cross-distillation
        self.teacher_proj = nn.Sequential(
            nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
            nn.LayerNorm(cfg.TEACHER_DIM),
        )

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeoDistill student: {total/1e6:.1f}M trainable parameters")

    # ------------------------------------------------------------------
    def extract_embedding(self, x):
        """Fused embedding for retrieval (eval only, no grad tracking)."""
        patches, cls_tok, hw = self.backbone(x)
        parts    = self.part_disc(patches, hw)
        emb      = self.pool(parts['part_features'], parts['salience'])
        cls_emb  = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return F.normalize(0.7 * emb + 0.3 * cls_emb, dim=-1)

    # ------------------------------------------------------------------
    def forward(self, x, return_parts=False):
        patches, cls_tok, hw = self.backbone(x)
        parts    = self.part_disc(patches, hw)
        emb      = self.pool(parts['part_features'], parts['salience'])   # normalized

        # Part-aware branch
        bn          = self.bottleneck(emb)       # unnorm BN features for CE
        logits      = self.classifier(bn)

        # CLS auxiliary branch
        cls_emb     = self.cls_proj(cls_tok)     # unnorm for CE
        cls_logits  = self.cls_classifier(cls_emb)

        # Fused L2-norm embedding for contrastive losses
        fused = F.normalize(0.7 * emb + 0.3 * F.normalize(cls_emb, dim=-1), dim=-1)

        # Project to teacher feature space (for cross-distillation)
        projected_feat = self.teacher_proj(emb)  # [B, TEACHER_DIM], unnormalized

        out = {
            'embedding':      fused,
            'logits':         logits,
            'cls_logits':     cls_logits,
            'projected_feat': projected_feat,
        }
        if return_parts:
            out['parts'] = parts
        return out


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class SupInfoNCELoss(nn.Module):
    """Label-aware InfoNCE with a learnable log-temperature."""

    def __init__(self, temp=0.05):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(temp).log())

    def forward(self, q_emb, r_emb, labels):
        t        = self.log_t.exp().clamp(0.01, 1.0)
        sim      = q_emb @ r_emb.t() / t
        labels   = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float()
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        loss     = -(log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
        return loss.mean()


class TripletLoss(nn.Module):
    """Batch-hard triplet loss."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.m = margin

    def forward(self, emb, labels):
        d        = torch.cdist(emb, emb, p=2)
        labels   = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float()
        neg_mask = labels.ne(labels.T).float()
        hard_pos = (d * pos_mask).max(1)[0]
        hard_neg = (d * neg_mask + pos_mask * 1e9).min(1)[0]
        return F.relu(hard_pos - hard_neg + self.m).mean()


class PartConsistencyLoss(nn.Module):
    """
    Symmetric KL-divergence on mean part-assignment distributions.
    Encourages drone and satellite of the same location to share
    similar semantic part activations.
    """

    def forward(self, assign_q, assign_r):
        dist_q  = assign_q.mean(dim=1)   # [B, K] — mean over patches
        dist_r  = assign_r.mean(dim=1)
        kl_qr   = F.kl_div((dist_q + 1e-8).log(), dist_r,
                            reduction='batchmean', log_target=False)
        kl_rq   = F.kl_div((dist_r + 1e-8).log(), dist_q,
                            reduction='batchmean', log_target=False)
        return 0.5 * (kl_qr + kl_rq)


class CrossDistillationLoss(nn.Module):
    """
    Feature-level distillation from DINOv2-B teacher to student.
    Combines MSE and cosine distance on L2-normalized features.
    Identical to the baseline's CrossDistillationLoss.
    """

    def forward(self, student_feat, teacher_feat):
        s = F.normalize(student_feat, dim=-1)
        t = F.normalize(teacher_feat, dim=-1)
        mse     = F.mse_loss(s, t)
        cosine  = 1.0 - F.cosine_similarity(s, t).mean()
        return mse + cosine


class SelfDistillationLoss(nn.Module):
    """
    KD-style self-distillation: the stronger part-aware branch (teacher)
    supervises the weaker CLS branch (student), tightening the auxiliary head.

    Adapted from MobileGeo baseline's SelfDistillationLoss (FIGISDI).
    Temperature-scaled KL divergence × T² (Hinton et al., 2015).
    """

    def __init__(self, temperature=4.0):
        super().__init__()
        self.T = temperature

    def forward(self, weak_logits, strong_logits):
        """
        strong_logits — part-aware branch (teacher signal)
        weak_logits   — CLS branch (student being trained)
        """
        p_teacher = F.softmax(strong_logits / self.T, dim=1).detach()
        p_student = F.log_softmax(weak_logits / self.T, dim=1)
        return (self.T ** 2) * F.kl_div(p_student, p_teacher,
                                         reduction='batchmean')


class UAPALoss(nn.Module):
    """
    Uncertainty-Aware Prediction Alignment.
    Satellite (lower uncertainty) teaches drone (higher uncertainty)
    at an adaptive temperature derived from their entropy gap.

    Identical to the baseline's UAPALoss.
    """

    def __init__(self, base_temperature=4.0):
        super().__init__()
        self.T0 = base_temperature

    @staticmethod
    def _entropy(logits):
        probs = F.softmax(logits, dim=1)
        return -(probs * (probs + 1e-8).log()).sum(dim=1).mean()

    def forward(self, drone_logits, sat_logits):
        U_drone = self._entropy(drone_logits)
        U_sat   = self._entropy(sat_logits)
        delta_U = U_drone - U_sat
        T       = self.T0 * (1 + torch.sigmoid(delta_U))

        p_sat   = F.softmax(sat_logits / T, dim=1).detach()
        p_drone = F.log_softmax(drone_logits / T, dim=1)
        return (T ** 2) * F.kl_div(p_drone, p_sat, reduction='batchmean')


# =============================================================================
# EVALUATION — full 200-location gallery (confusion data)
# =============================================================================

@torch.no_grad()
def evaluate(model, test_ds, device):
    """
    Evaluate drone→satellite retrieval on SUES-200.

    Gallery: all 200 satellite images (80 test + 120 train distractors)
    — matches the standard confusion-data protocol from the literature.
    Queries: all drone images from the 80 test locations.
    """
    model.eval()
    test_tf = get_transforms("test")

    # ── Extract drone query features ──────────────────────────────────────────
    loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)
    drone_feats, drone_labels = [], []
    for b in loader:
        feat = model.extract_embedding(b['drone'].to(device)).cpu()
        drone_feats.append(feat)
        drone_labels.append(b['label'])
    drone_feats  = torch.cat(drone_feats)   # [Q, E]
    drone_labels = torch.cat(drone_labels)  # [Q]

    # ── Build full 200-location satellite gallery ──────────────────────────────
    # Collect all images first, then batch-extract features for speed.
    all_locs       = [f"{l:04d}" for l in range(1, 201)]
    sat_img_list   = []
    sat_label_list = []
    distractor_cnt = 0

    for loc in all_locs:
        sp = os.path.join(test_ds.sat_dir, loc, "0.png")
        if not os.path.exists(sp):
            continue
        sat_img_list.append(test_tf(Image.open(sp).convert('RGB')))
        if loc in test_ds.location_to_idx:
            sat_label_list.append(test_ds.location_to_idx[loc])
        else:
            # unique negative label for train-loc distractors (never matches queries)
            sat_label_list.append(-1000 - distractor_cnt)
            distractor_cnt += 1

    # Batch extract satellite features
    sat_feats  = []
    batch_size = 64
    for i in range(0, len(sat_img_list), batch_size):
        batch = torch.stack(sat_img_list[i : i + batch_size]).to(device)
        feat  = model.extract_embedding(batch).cpu()
        sat_feats.append(feat)
    sat_feats  = torch.cat(sat_feats)
    sat_labels = torch.tensor(sat_label_list)

    print(f"  Gallery: {len(sat_feats)} satellite images "
          f"({len(test_ds.locations)} test + {distractor_cnt} train distractors)")
    print(f"  Queries: {len(drone_feats)} drone images")

    # ── Retrieval metrics ─────────────────────────────────────────────────────
    sim     = drone_feats @ sat_feats.T   # [Q, G]
    _, rank = sim.sort(1, descending=True)
    N       = drone_feats.size(0)
    r1 = r5 = r10 = ap = 0

    for i in range(N):
        matches = torch.where(sat_labels[rank[i]] == drone_labels[i])[0]
        if len(matches) == 0:
            continue
        first = matches[0].item()
        if first < 1:   r1  += 1
        if first < 5:   r5  += 1
        if first < 10:  r10 += 1
        ap += sum((j + 1) / (p.item() + 1)
                  for j, p in enumerate(matches)) / len(matches)

    return {
        'R@1':  r1  / N * 100,
        'R@5':  r5  / N * 100,
        'R@10': r10 / N * 100,
        'mAP':  ap  / N * 100,
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(model, teacher, loader, losses, optimizer, scaler,
                    device, epoch):
    model.eval()   # teacher is always eval; model.train() below
    model.train()
    if teacher is not None:
        teacher.eval()

    infonce, triplet, ce, consist, cross_dist, self_dist, uapa = losses
    total_sum  = 0
    n          = 0
    loss_sums  = defaultdict(float)

    for batch in tqdm(loader, desc=f'Ep{epoch:3d}', leave=False):
        drone  = batch['drone'].to(device)
        sat    = batch['satellite'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            d_out = model(drone, return_parts=True)
            s_out = model(sat,   return_parts=True)

            # 1. Cross-entropy (part-aware + 0.3 × CLS branch, both views)
            l_ce = (ce(d_out['logits'], labels) +
                    ce(s_out['logits'], labels))
            l_ce += 0.3 * (ce(d_out['cls_logits'], labels) +
                           ce(s_out['cls_logits'], labels))

            # 2. SupInfoNCE cross-view contrastive
            l_nce = infonce(d_out['embedding'], s_out['embedding'], labels)

            # 3. Batch-hard triplet (both views)
            l_tri = 0.5 * (triplet(d_out['embedding'], labels) +
                           triplet(s_out['embedding'], labels))

            # 4. Part-distribution consistency
            l_con = consist(d_out['parts']['assignment'],
                            s_out['parts']['assignment'])

            # 5. Cross-distillation: DINOv2-B teacher → student projected feat
            if teacher is not None:
                with torch.no_grad():
                    t_drone = teacher(drone)   # [B, 768]
                    t_sat   = teacher(sat)     # [B, 768]
                l_cross = (cross_dist(d_out['projected_feat'], t_drone) +
                           cross_dist(s_out['projected_feat'], t_sat))
            else:
                l_cross = torch.tensor(0.0, device=device)

            # 6. Self-distillation: part-aware logits → CLS branch logits
            l_self = (self_dist(d_out['cls_logits'], d_out['logits']) +
                      self_dist(s_out['cls_logits'], s_out['logits']))

            # 7. UAPA: satellite teaches drone with adaptive temperature
            l_uapa = uapa(d_out['logits'], s_out['logits'])

            # ── Total loss ────────────────────────────────────────────────────
            loss = (CFG.LAMBDA_CE          * l_ce    +
                    CFG.LAMBDA_INFONCE     * l_nce   +
                    CFG.LAMBDA_TRIPLET     * l_tri   +
                    CFG.LAMBDA_CONSISTENCY * l_con   +
                    CFG.LAMBDA_CROSS_DIST  * l_cross +
                    CFG.LAMBDA_SELF_DIST   * l_self  +
                    CFG.LAMBDA_UAPA        * l_uapa)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_sum += loss.item()
        n         += 1
        loss_sums['ce']    += l_ce.item()
        loss_sums['nce']   += l_nce.item()
        loss_sums['tri']   += l_tri.item()
        loss_sums['con']   += l_con.item()
        loss_sums['cross'] += l_cross.item() if torch.is_tensor(l_cross) else l_cross
        loss_sums['self']  += l_self.item()
        loss_sums['uapa']  += l_uapa.item()

    avg_loss = total_sum / max(n, 1)
    avg_dict = {k: v / max(n, 1) for k, v in loss_sums.items()}
    return avg_loss, avg_dict


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  SPDGeo-D: Semantic Part Discovery + Multi-Level Distillation")
    print(f"  Dataset : SUES-200 | Epochs: {CFG.NUM_EPOCHS} | Device: {DEVICE}")
    print(f"  Parts   : {CFG.N_PARTS} | Img: {CFG.IMG_SIZE} | Embed: {CFG.EMBED_DIM}")
    print(f"  Losses  : CE + SupInfoNCE + Triplet + PartCons "
          f"+ CrossDistill + SelfDistill + UAPA")
    print("=" * 65)

    # ── Data ──────────────────────────────────────────────────────────────────
    print('\nLoading SUES-200 …')
    train_ds = SUES200Dataset(CFG.SUES_ROOT, 'train',
                              transform=get_transforms("train"))
    test_ds  = SUES200Dataset(CFG.SUES_ROOT, 'test',
                              transform=get_transforms("test"))
    train_loader = DataLoader(
        train_ds,
        batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    print('\nBuilding models …')
    model = SPDGeoDistillModel(CFG.NUM_CLASSES).to(DEVICE)

    teacher = None
    try:
        teacher = DINOv2Teacher().to(DEVICE)
        teacher.eval()
    except Exception as e:
        print(f"  [WARN] Could not load DINOv2-B teacher: {e}")
        print("  Training without cross-distillation.")

    # ── Losses ────────────────────────────────────────────────────────────────
    infonce    = SupInfoNCELoss(temp=0.05).to(DEVICE)
    triplet    = TripletLoss(margin=0.3)
    ce         = nn.CrossEntropyLoss(label_smoothing=0.1)
    consist    = PartConsistencyLoss()
    cross_dist = CrossDistillationLoss()
    self_dist  = SelfDistillationLoss(temperature=CFG.DISTILL_TEMP)
    uapa       = UAPALoss(base_temperature=CFG.DISTILL_TEMP)

    losses = (infonce, triplet, ce, consist, cross_dist, self_dist, uapa)

    # ── Optimizer — differential LR ───────────────────────────────────────────
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith('backbone')]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params,      'lr': CFG.BACKBONE_LR},
        {'params': head_params,          'lr': CFG.LR},
        {'params': infonce.parameters(), 'lr': CFG.LR},
    ], weight_decay=CFG.WEIGHT_DECAY)

    scaler = torch.amp.GradScaler('cuda', enabled=CFG.USE_AMP)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_r1     = 0.0
    results_log = []

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        # Cosine LR with linear warm-up
        if epoch <= CFG.WARMUP_EPOCHS:
            lr_scale = epoch / CFG.WARMUP_EPOCHS
        else:
            progress = (epoch - CFG.WARMUP_EPOCHS) / (CFG.NUM_EPOCHS - CFG.WARMUP_EPOCHS)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        lr_scale = max(lr_scale, 0.01)   # floor at 1 % of base LR

        optimizer.param_groups[0]['lr'] = CFG.BACKBONE_LR * lr_scale
        optimizer.param_groups[1]['lr'] = CFG.LR          * lr_scale
        optimizer.param_groups[2]['lr'] = CFG.LR          * lr_scale
        cur_lr = optimizer.param_groups[1]['lr']

        avg_loss, ld = train_one_epoch(
            model, teacher, train_loader, losses,
            optimizer, scaler, DEVICE, epoch)

        print(
            f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | "
            f"Loss {avg_loss:.4f} | "
            f"CE {ld['ce']:.3f}  NCE {ld['nce']:.3f}  "
            f"Tri {ld['tri']:.3f}  Con {ld['con']:.3f}  "
            f"Crs {ld['cross']:.3f}  Slf {ld['self']:.3f}  "
            f"UAPA {ld['uapa']:.3f} | "
            f"LR {cur_lr:.2e}"
        )

        # ── Periodic evaluation ───────────────────────────────────────────────
        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            metrics = evaluate(model, test_ds, DEVICE)
            results_log.append({'epoch': epoch, **metrics})

            print(f"  ► R@1: {metrics['R@1']:.2f}%  R@5: {metrics['R@5']:.2f}%  "
                  f"R@10: {metrics['R@10']:.2f}%  mAP: {metrics['mAP']:.2f}%")

            if metrics['R@1'] > best_r1:
                best_r1 = metrics['R@1']
                torch.save({
                    'epoch':             epoch,
                    'model_state_dict':  model.state_dict(),
                    'metrics':           metrics,
                }, os.path.join(CFG.OUTPUT_DIR, 'spdgeo_d_best.pth'))
                print(f"  ★ New best R@1: {best_r1:.2f}%!")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f'\n{"=" * 65}')
    print(f'  SPDGeo-D TRAINING COMPLETE — SUES-200')
    print(f'{"=" * 65}')
    print(f'  Best R@1: {best_r1:.2f}%  (200-loc confusion gallery)')
    print(f'\n  {"Epoch":>6} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"mAP":>8}')
    print(f'  {"-" * 44}')
    for r in results_log:
        print(f'  {r["epoch"]:6d} {r["R@1"]:8.2f} {r["R@5"]:8.2f} '
              f'{r["R@10"]:8.2f} {r["mAP"]:8.2f}')
    print(f'{"=" * 65}')

    with open(os.path.join(CFG.OUTPUT_DIR, 'spdgeo_d_results.json'), 'w') as f:
        json.dump({
            'results_log': results_log,
            'best_r1':     best_r1,
            'config':      {k: v for k, v in vars(CFG).items()
                            if not k.startswith('_')},
        }, f, indent=2)
    print(f'Results saved to {CFG.OUTPUT_DIR}/spdgeo_d_results.json')


if __name__ == '__main__':
    main()
