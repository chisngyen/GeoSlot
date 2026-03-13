# EXP39: SPDGeo-DualDisc — Dual Discriminative + Altitude Curriculum cho SUES-200
# Mô tả: Bổ sung InfoNCE nội bộ cho từng view (drone / satellite) + curriculum theo độ cao để tăng phân biệt nội-view trước khi align cross-view.

import subprocess
import sys

for _p in ["timm", "tqdm"]:
    try:
        __import__(_p)
    except ImportError:
        subprocess.run(f"pip install -q {_p}", shell=True, capture_output=True)

import copy
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================


class Config:
    SUES_ROOT = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    DRONE_DIR = "drone-view"
    SAT_DIR = "satellite-view"
    OUTPUT_DIR = "/kaggle/working"

    ALTITUDES = ["150", "200", "250", "300"]
    ALT_TO_IDX = {"150": 0, "200": 1, "250": 2, "300": 3}
    NUM_ALTITUDES = 4

    TRAIN_LOCS = list(range(1, 121))
    TEST_LOCS = list(range(121, 201))
    NUM_CLASSES = 120

    IMG_SIZE = 336
    N_PARTS = 8
    PART_DIM = 256
    EMBED_DIM = 512
    TEACHER_DIM = 768
    CLUSTER_TEMP = 0.07
    UNFREEZE_BLOCKS = 4

    NUM_EPOCHS = 120
    P_CLASSES = 16
    K_SAMPLES = 4
    LR = 3e-4
    BACKBONE_LR = 3e-5
    WEIGHT_DECAY = 0.01
    WARMUP_EPOCHS = 5
    USE_AMP = True
    SEED = 42

    # Base losses (DPEA-MAR style)
    LAMBDA_CE = 1.0
    LAMBDA_INFONCE = 1.0
    LAMBDA_CONSISTENCY = 0.1
    LAMBDA_CROSS_DIST = 0.3
    LAMBDA_SELF_DIST = 0.3
    LAMBDA_UAPA = 0.2

    LAMBDA_PROXY = 0.5
    PROXY_MARGIN = 0.1
    PROXY_ALPHA = 32
    LAMBDA_EMA_DIST = 0.2
    EMA_DECAY = 0.996

    LAMBDA_ALT_CONSIST = 0.2
    MASK_RATIO = 0.30
    LAMBDA_MASK_RECON = 0.3
    RECON_WARMUP = 10
    LAMBDA_ALT_PRED = 0.15
    LAMBDA_DIVERSITY = 0.05

    # Dual discriminative heads
    LAMBDA_DRONE_INTRA = 0.3
    LAMBDA_SAT_INTRA = 0.2

    # Altitude difficulty curriculum (per-sample weight)
    ALT_CURR_END_EPOCH = 60
    EASY_ALT = 3    # 300m
    HARD_ALT = 0    # 150m
    EASY_START_W = 1.5
    HARD_START_W = 0.5

    DISTILL_TEMP = 4.0
    EVAL_INTERVAL = 5
    NUM_WORKERS = 2


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


# =============================================================================
# DATASET
# =============================================================================


class SUES200Dataset(Dataset):
    def __init__(self, root, mode: str = "train", altitudes=None, transform=None):
        self.root = root
        self.mode = mode
        self.altitudes = altitudes or CFG.ALTITUDES
        self.transform = transform
        self.drone_dir = os.path.join(root, CFG.DRONE_DIR)
        self.sat_dir = os.path.join(root, CFG.SAT_DIR)

        loc_ids = CFG.TRAIN_LOCS if mode == "train" else CFG.TEST_LOCS
        self.locations = [f"{l:04d}" for l in loc_ids]
        self.location_to_idx = {l: i for i, l in enumerate(self.locations)}

        self.samples = []
        self.drone_by_location = defaultdict(list)

        for loc in self.locations:
            li = self.location_to_idx[loc]
            sp = os.path.join(self.sat_dir, loc, "0.png")
            if not os.path.exists(sp):
                continue
            for alt in self.altitudes:
                ad = os.path.join(self.drone_dir, loc, alt)
                if not os.path.isdir(ad):
                    continue
                for img in sorted(os.listdir(ad)):
                    if img.endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append((os.path.join(ad, img), sp, li, alt))
                        self.drone_by_location[li].append(len(self.samples) - 1)

        print(f"  [{mode}] {len(self.samples)} samples, {len(self.locations)} locations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        dp, sp, li, alt = self.samples[idx]
        try:
            d = Image.open(dp).convert("RGB")
            s = Image.open(sp).convert("RGB")
        except Exception:
            sz = CFG.IMG_SIZE
            d = Image.new("RGB", (sz, sz), (128, 128, 128))
            s = Image.new("RGB", (sz, sz), (128, 128, 128))

        if self.transform:
            d = self.transform(d)
            s = self.transform(s)

        alt_idx = CFG.ALT_TO_IDX.get(alt, 0)
        alt_norm = (int(alt) - 150) / 150.0

        return {
            "drone": d,
            "satellite": s,
            "label": li,
            "altitude": int(alt),
            "alt_idx": alt_idx,
            "alt_norm": alt_norm,
        }


class PKSampler:
    def __init__(self, ds, p: int, k: int):
        self.ds = ds
        self.p = p
        self.k = k
        self.locs = list(ds.drone_by_location.keys())

    def __iter__(self):
        locs = self.locs.copy()
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


def get_transforms(mode: str = "train", img_size: int | None = None):
    sz = img_size or CFG.IMG_SIZE
    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize((sz, sz)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomResizedCrop(sz, scale=(0.8, 1.0)),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.RandomGrayscale(p=0.02),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# =============================================================================
# BACKBONE / TEACHER / MODULES (DPEA-MAR style)
# =============================================================================


class DINOv2Backbone(nn.Module):
    def __init__(self, unfreeze_blocks: int = 4):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        self.patch_size = 14

        for p in self.model.parameters():
            p.requires_grad = False
        for blk in self.model.blocks[-unfreeze_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.model.norm.parameters():
            p.requires_grad = True

        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  DINOv2 ViT-S/14: {frozen/1e6:.1f}M frozen, {trainable/1e6:.1f}M trainable")

    def forward(self, x: torch.Tensor):
        feat = self.model.forward_features(x)
        return feat["x_norm_patchtokens"], feat["x_norm_clstoken"], (
            x.shape[2] // self.patch_size,
            x.shape[3] // self.patch_size,
        )


class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        print("  Loading DINOv2 ViT-B/14 teacher ...")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        for p in self.model.parameters():
            p.requires_grad = False
        print("  DINOv2 ViT-B/14 teacher loaded (all frozen).")

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return self.model.forward_features(x)["x_norm_clstoken"]


class DeepAltitudeFiLM(nn.Module):
    def __init__(self, num_altitudes: int = 4, feat_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(num_altitudes, feat_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )

    def forward(self, x: torch.Tensor, alt_idx: torch.Tensor | None):
        if alt_idx is None:
            return x
        b, n, d = x.shape
        alt = self.embed(alt_idx)
        gamma_beta = self.net(alt).view(b, 2, d)
        gamma, beta = gamma_beta[:, 0], gamma_beta[:, 1]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return x * (1 + gamma) + beta


class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(self, feat_dim: int = 384, n_parts: int = 8, part_dim: int = 256, temperature: float = 0.07, num_altitudes: int = 4):
        super().__init__()
        self.temperature = temperature
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, part_dim),
            nn.LayerNorm(part_dim),
            nn.GELU(),
        )
        self.altitude_film = DeepAltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.LayerNorm(part_dim),
            nn.Linear(part_dim, part_dim * 2),
            nn.GELU(),
            nn.Linear(part_dim * 2, part_dim),
        )
        self.salience_head = nn.Sequential(
            nn.Linear(part_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, patch_features: torch.Tensor, spatial_hw, alt_idx: torch.Tensor | None = None):
        b, n, _ = patch_features.shape
        h, w = spatial_hw

        feat = self.feat_proj(patch_features)
        feat = self.altitude_film(feat, alt_idx)

        feat_norm = F.normalize(feat, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum("bnd,kd->bnk", feat_norm, proto_norm) / self.temperature
        assign = F.softmax(sim, dim=-1)

        assign_t = assign.transpose(1, 2)
        mass = assign_t.sum(-1, keepdim=True).clamp(min=1e-6)
        part_feat = torch.bmm(assign_t, feat) / mass
        part_feat = part_feat + self.refine(part_feat)

        device = feat.device
        gy = torch.arange(h, device=device).float() / max(h - 1, 1)
        gx = torch.arange(w, device=device).float() / max(w - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(b, -1, -1)) / mass

        salience = self.salience_head(part_feat).squeeze(-1)

        return {
            "part_features": part_feat,
            "part_positions": part_pos,
            "assignment": assign,
            "salience": salience,
            "projected_patches": feat,
        }


class PartAwarePooling(nn.Module):
    def __init__(self, part_dim: int = 256, embed_dim: int = 512):
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

    def forward(self, part_features: torch.Tensor, salience: torch.Tensor | None = None):
        aw = self.attn(part_features)
        if salience is not None:
            aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        attn_pool = (aw * part_features).sum(1)
        mean_pool = part_features.mean(1)
        max_pool = part_features.max(1)[0]
        combined = torch.cat([attn_pool, mean_pool, max_pool], dim=-1)
        return F.normalize(self.proj(combined), dim=-1)


class DynamicFusionGate(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb: torch.Tensor, cls_emb: torch.Tensor):
        alpha = torch.sigmoid(self.gate(torch.cat([part_emb, cls_emb], dim=-1)))
        return F.normalize(alpha * part_emb + (1 - alpha) * cls_emb, dim=-1)


class MaskedPartReconstruction(nn.Module):
    def __init__(self, part_dim: int = 256, mask_ratio: float = 0.30):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Linear(part_dim, part_dim * 2),
            nn.GELU(),
            nn.Linear(part_dim * 2, part_dim),
            nn.LayerNorm(part_dim),
        )

    def forward(self, projected_patches: torch.Tensor, part_features: torch.Tensor, assignment: torch.Tensor):
        b, n, d = projected_patches.shape
        num_mask = int(n * self.mask_ratio)
        noise = torch.rand(b, n, device=projected_patches.device)
        mask_indices = noise.argsort(dim=1)[:, :num_mask]

        target = projected_patches.detach()
        mask_expand = mask_indices.unsqueeze(-1).expand(-1, -1, d)
        masked_targets = torch.gather(target, 1, mask_expand)

        k = part_features.shape[1]
        mask_expand_k = mask_indices.unsqueeze(-1).expand(-1, -1, k)
        masked_assign = torch.gather(assignment, 1, mask_expand_k)

        recon = torch.bmm(masked_assign, part_features)
        recon = self.decoder(recon)

        recon_norm = F.normalize(recon, dim=-1)
        target_norm = F.normalize(masked_targets, dim=-1)
        return (1 - (recon_norm * target_norm).sum(dim=-1)).mean()


class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, embedding: torch.Tensor, alt_target: torch.Tensor):
        pred = self.head(embedding).squeeze(-1)
        return F.smooth_l1_loss(pred, alt_target)


class PrototypeDiversityLoss(nn.Module):
    def forward(self, prototypes: torch.Tensor):
        p = F.normalize(prototypes, dim=-1)
        sim = p @ p.T
        k = sim.size(0)
        mask = 1 - torch.eye(k, device=sim.device)
        return (sim * mask).abs().sum() / (k * (k - 1))


# =============================================================================
# NEW: DUAL DISCRIMINATIVE HEADS + ALT CURRICULUM
# =============================================================================


def _info_nce_intra(emb: torch.Tensor, labels: torch.Tensor, temp: float = 0.07):
    """Standard supervised InfoNCE within one view (drone or sat)."""
    emb = F.normalize(emb, dim=-1)
    sim = emb @ emb.T / temp
    labels = labels.view(-1, 1)
    pos_mask = labels.eq(labels.T).float()
    neg_mask = 1.0 - pos_mask
    # Remove self-positives
    pos_mask = pos_mask - torch.eye(pos_mask.size(0), device=pos_mask.device)

    log_prob = sim - torch.logsumexp(sim * (pos_mask + neg_mask), dim=1, keepdim=True)
    pos_log = (log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
    return -(pos_log.mean())


def altitude_curriculum_weights(alt_idx: torch.Tensor, epoch: int) -> torch.Tensor:
    """
    Per-sample curriculum weights over altitude difficulty.
    - At epoch 0: 300m ~1.5, 150m ~0.5 (easy upweighted, hard downweighted)
    - By ALT_CURR_END_EPOCH: all →1.0 (no bias).
    """
    device = alt_idx.device
    if epoch >= CFG.ALT_CURR_END_EPOCH:
        return torch.ones_like(alt_idx, dtype=torch.float32, device=device)

    frac = epoch / max(CFG.ALT_CURR_END_EPOCH, 1)
    # Start weights
    w_start = torch.ones_like(alt_idx, dtype=torch.float32, device=device)
    w_start = torch.where(alt_idx == CFG.EASY_ALT, torch.tensor(CFG.EASY_START_W, device=device), w_start)
    w_start = torch.where(alt_idx == CFG.HARD_ALT, torch.tensor(CFG.HARD_START_W, device=device), w_start)
    # Linear → 1.0
    return w_start + frac * (1.0 - w_start)


class SPDGeoDualDiscModel(nn.Module):
    """
    Same backbone as DPEA-MAR, plus implicit dual-discriminative training
    via intra-view InfoNCE heads (drone & satellite).
    """

    def __init__(self, num_classes: int, cfg: Config = CFG):
        super().__init__()
        self.cfg = cfg
        self.backbone = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = AltitudeAwarePartDiscovery(384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES)
        self.pool = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)

        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
            nn.BatchNorm1d(cfg.EMBED_DIM),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj = nn.Sequential(
            nn.Linear(384, cfg.EMBED_DIM),
            nn.BatchNorm1d(cfg.EMBED_DIM),
            nn.ReLU(inplace=True),
        )
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM), nn.LayerNorm(cfg.TEACHER_DIM))

        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)
        self.alt_pred = AltitudePredictionHead(cfg.EMBED_DIM)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-DualDisc student: {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x: torch.Tensor, alt_idx: torch.Tensor | None = None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts["part_features"], parts["salience"])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)

    def forward(self, x: torch.Tensor, alt_idx: torch.Tensor | None = None, return_parts: bool = False):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)

        emb = self.pool(parts["part_features"], parts["salience"])
        bn = self.bottleneck(emb)
        logits = self.classifier(bn)

        cls_emb_raw = self.cls_proj(cls_tok)
        cls_logits = self.cls_classifier(cls_emb_raw)
        cls_emb_norm = F.normalize(cls_emb_raw, dim=-1)

        fused = self.fusion_gate(emb, cls_emb_norm)
        projected_feat = self.teacher_proj(emb)

        out = {
            "embedding": fused,
            "logits": logits,
            "cls_logits": cls_logits,
            "projected_feat": projected_feat,
            "part_emb": emb,
            "cls_emb": cls_emb_norm,
        }
        if return_parts:
            out["parts"] = parts
        out["parts"] = parts
        return out


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = CFG.EMA_DECAY):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, alt_idx: torch.Tensor | None = None):
        return self.model.extract_embedding(x, alt_idx=alt_idx)


# =============================================================================
# LOSSES
# =============================================================================


class SupInfoNCELoss(nn.Module):
    def __init__(self, temp: float = 0.05):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(temp).log())

    def forward(self, q_emb: torch.Tensor, r_emb: torch.Tensor, labels: torch.Tensor):
        t = self.log_t.exp().clamp(0.01, 1.0)
        sim = q_emb @ r_emb.T / t
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float()
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        loss = -(log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
        return loss.mean()


class PartConsistencyLoss(nn.Module):
    def forward(self, assign_q: torch.Tensor, assign_r: torch.Tensor):
        dist_q = assign_q.mean(dim=1)
        dist_r = assign_r.mean(dim=1)
        kl_qr = F.kl_div((dist_q + 1e-8).log(), dist_r, reduction="batchmean", log_target=False)
        kl_rq = F.kl_div((dist_r + 1e-8).log(), dist_q, reduction="batchmean", log_target=False)
        return 0.5 * (kl_qr + kl_rq)


class CrossDistillationLoss(nn.Module):
    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor):
        s = F.normalize(student_feat, dim=-1)
        t = F.normalize(teacher_feat, dim=-1)
        mse = F.mse_loss(s, t)
        cosine = 1.0 - F.cosine_similarity(s, t).mean()
        return mse + cosine


class SelfDistillationLoss(nn.Module):
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.t = temperature

    def forward(self, weak_logits: torch.Tensor, strong_logits: torch.Tensor):
        p_teacher = F.softmax(strong_logits / self.t, dim=1).detach()
        p_student = F.log_softmax(weak_logits / self.t, dim=1)
        return (self.t**2) * F.kl_div(p_student, p_teacher, reduction="batchmean")


class UAPALoss(nn.Module):
    def __init__(self, base_temperature: float = 4.0):
        super().__init__()
        self.t0 = base_temperature

    @staticmethod
    def _entropy(logits: torch.Tensor):
        probs = F.softmax(logits, dim=1)
        return -(probs * (probs + 1e-8).log()).sum(dim=1).mean()

    def forward(self, drone_logits: torch.Tensor, sat_logits: torch.Tensor):
        delta_u = self._entropy(drone_logits) - self._entropy(sat_logits)
        t = self.t0 * (1 + torch.sigmoid(delta_u))
        p_sat = F.softmax(sat_logits / t, dim=1).detach()
        return (t**2) * F.kl_div(F.log_softmax(drone_logits / t, dim=1), p_sat, reduction="batchmean")


class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int, margin: float = 0.1, alpha: float = 32):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")
        self.margin = margin
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        p = F.normalize(self.proxies, dim=-1)
        sim = embeddings @ p.T
        one_hot = F.one_hot(labels, self.num_classes).float()

        pos_exp = torch.exp(-self.alpha * (sim * one_hot - self.margin)) * one_hot
        p_plus = one_hot.sum(0)
        has_pos = p_plus > 0
        pos_term = torch.log(1 + pos_exp.sum(0))
        pos_loss = pos_term[has_pos].mean() if has_pos.sum() > 0 else torch.tensor(0.0, device=embeddings.device)

        neg_mask = 1 - one_hot
        neg_exp = torch.exp(self.alpha * (sim * neg_mask + self.margin)) * neg_mask
        neg_loss = torch.log(1 + neg_exp.sum(0)).mean()
        return pos_loss + neg_loss


class EMADistillationLoss(nn.Module):
    def forward(self, student_emb: torch.Tensor, ema_emb: torch.Tensor):
        return (1 - F.cosine_similarity(student_emb, ema_emb)).mean()


class AltitudeConsistencyLoss(nn.Module):
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, altitudes: torch.Tensor):
        b = embeddings.size(0)
        if b < 2:
            return torch.tensor(0.0, device=embeddings.device)

        lbl = labels.view(-1, 1)
        alt = altitudes.view(-1, 1)
        same_loc = lbl.eq(lbl.T)
        diff_alt = alt.ne(alt.T)
        mask = same_loc & diff_alt

        if mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        emb_norm = F.normalize(embeddings, dim=-1)
        cos_dist = 1 - emb_norm @ emb_norm.T
        return (cos_dist * mask.float()).sum() / mask.float().sum().clamp(min=1)


# =============================================================================
# EVALUATION
# =============================================================================


@torch.no_grad()
def evaluate(model: SPDGeoDualDiscModel, test_ds: SUES200Dataset, device: torch.device):
    model.eval()
    test_tf = get_transforms("test")

    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    drone_feats, drone_labels, drone_alts = [], [], []
    for b in loader:
        alt_idx = b["alt_idx"].to(device)
        feat = model.extract_embedding(b["drone"].to(device), alt_idx=alt_idx).cpu()
        drone_feats.append(feat)
        drone_labels.append(b["label"])
        drone_alts.append(b["altitude"])

    drone_feats = torch.cat(drone_feats)
    drone_labels = torch.cat(drone_labels)
    drone_alts = torch.cat(drone_alts)

    all_locs = [f"{l:04d}" for l in range(1, 201)]
    sat_img_list, sat_label_list = [], []
    distractor_cnt = 0
    for loc in all_locs:
        sp = os.path.join(test_ds.sat_dir, loc, "0.png")
        if not os.path.exists(sp):
            continue
        sat_img_list.append(test_tf(Image.open(sp).convert("RGB")))
        if loc in test_ds.location_to_idx:
            sat_label_list.append(test_ds.location_to_idx[loc])
        else:
            sat_label_list.append(-1000 - distractor_cnt)
            distractor_cnt += 1

    sat_feats = []
    for i in range(0, len(sat_img_list), 64):
        batch = torch.stack(sat_img_list[i : i + 64]).to(device)
        sat_feats.append(model.extract_embedding(batch, alt_idx=None).cpu())
    sat_feats = torch.cat(sat_feats)
    sat_labels = torch.tensor(sat_label_list)

    print(f"  Gallery: {len(sat_feats)} satellite images | Queries: {len(drone_feats)} drone images")

    sim = drone_feats @ sat_feats.T
    _, rank = sim.sort(1, descending=True)

    n = drone_feats.size(0)
    r1 = r5 = r10 = ap = 0.0
    per_alt = {150: [0, 0, 0, 0], 200: [0, 0, 0, 0], 250: [0, 0, 0, 0], 300: [0, 0, 0, 0]}

    for i in range(n):
        matches = torch.where(sat_labels[rank[i]] == drone_labels[i])[0]
        if len(matches) == 0:
            continue
        first = matches[0].item()
        if first < 1:
            r1 += 1
        if first < 5:
            r5 += 1
        if first < 10:
            r10 += 1
        rel = (sat_labels[rank[i]] == drone_labels[i]).float()
        cumsum = rel.cumsum(0)
        prec = cumsum / torch.arange(1, len(rel) + 1, device=rel.device, dtype=torch.float32)
        if rel.sum() > 0:
            ap += (prec * rel).sum() / rel.sum()

        alt = int(drone_alts[i].item())
        alt_vec = per_alt[alt]
        if first < 1:
            alt_vec[0] += 1
        if first < 5:
            alt_vec[1] += 1
        if first < 10:
            alt_vec[2] += 1
        alt_vec[3] += 1

    r1 = 100.0 * r1 / n
    r5 = 100.0 * r5 / n
    r10 = 100.0 * r10 / n
    ap = 100.0 * float(ap) / n

    metrics = {"R@1": r1, "R@5": r5, "R@10": r10, "mAP": ap}
    return metrics, per_alt


# =============================================================================
# TRAINING
# =============================================================================


def train_one_epoch(
    model: SPDGeoDualDiscModel,
    teacher: nn.Module | None,
    ema: EMAModel,
    loader,
    losses,
    new_losses,
    optimizer,
    scaler,
    device: torch.device,
    epoch: int,
):
    model.train()
    if teacher:
        teacher.eval()

    infonce, ce, consist, cross_dist, self_dist, uapa = losses
    proxy_anchor, ema_dist, alt_consist, diversity_loss = new_losses

    total_sum = 0.0
    n = 0
    loss_sums = defaultdict(float)
    recon_active = epoch >= CFG.RECON_WARMUP

    for batch in tqdm(loader, desc=f"Ep{epoch:3d}", leave=False):
        drone = batch["drone"].to(device)
        sat = batch["satellite"].to(device)
        labels = batch["label"].to(device)
        alt_idx = batch["alt_idx"].to(device)
        alts = batch["altitude"].to(device)
        alt_norm = batch["alt_norm"].to(device).float()

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=CFG.USE_AMP):
            d_out = model(drone, alt_idx=alt_idx, return_parts=True)
            s_out = model(sat, alt_idx=None, return_parts=True)

            w_alt = altitude_curriculum_weights(alt_idx, epoch).view(-1, 1)  # [B,1]

            l_ce_drone = ce(d_out["logits"], labels)
            l_ce_sat = ce(s_out["logits"], labels)
            l_ce = l_ce_drone + l_ce_sat
            l_ce += 0.3 * (ce(d_out["cls_logits"], labels) + ce(s_out["cls_logits"], labels))

            l_nce = infonce(d_out["embedding"], s_out["embedding"], labels)
            l_con = consist(d_out["parts"]["assignment"], s_out["parts"]["assignment"])

            if teacher is not None:
                with torch.no_grad():
                    t_drone = teacher(drone)
                    t_sat = teacher(sat)
                l_cross = cross_dist(d_out["projected_feat"], t_drone) + cross_dist(s_out["projected_feat"], t_sat)
            else:
                l_cross = torch.tensor(0.0, device=device)

            l_self = self_dist(d_out["cls_logits"], d_out["logits"]) + self_dist(s_out["cls_logits"], s_out["logits"])
            l_uapa = uapa(d_out["logits"], s_out["logits"])

            # Dual discriminative intra-view NCE (applied only to drone/sat embeddings, weighted by curriculum)
            l_drone_intra = _info_nce_intra(d_out["embedding"], labels)
            l_sat_intra = _info_nce_intra(s_out["embedding"], labels)

            l_proxy = 0.5 * (proxy_anchor(d_out["embedding"], labels) + proxy_anchor(s_out["embedding"], labels))

            with torch.no_grad():
                ema_drone_emb = ema.forward(drone, alt_idx=alt_idx)
                ema_sat_emb = ema.forward(sat, alt_idx=None)
            l_ema = 0.5 * (ema_dist(d_out["embedding"], ema_drone_emb) + ema_dist(s_out["embedding"], ema_sat_emb))

            l_alt_con = alt_consist(d_out["embedding"], labels, alts)

            if recon_active:
                l_recon_d = model.mask_recon(
                    d_out["parts"]["projected_patches"],
                    d_out["parts"]["part_features"],
                    d_out["parts"]["assignment"],
                )
                l_recon_s = model.mask_recon(
                    s_out["parts"]["projected_patches"],
                    s_out["parts"]["part_features"],
                    s_out["parts"]["assignment"],
                )
                l_recon = 0.5 * (l_recon_d + l_recon_s)
            else:
                l_recon = torch.tensor(0.0, device=device)

            l_alt_pred = model.alt_pred(d_out["embedding"].detach(), alt_norm)
            l_div = diversity_loss(model.part_disc.prototypes)

            # Aggregate with curriculum: CE + intra-NCE scaled by altitude weights (average)
            w_mean = w_alt.mean()

            loss = (
                CFG.LAMBDA_CE * l_ce
                + CFG.LAMBDA_INFONCE * l_nce
                + CFG.LAMBDA_CONSISTENCY * l_con
                + CFG.LAMBDA_CROSS_DIST * l_cross
                + CFG.LAMBDA_SELF_DIST * l_self
                + CFG.LAMBDA_UAPA * l_uapa
                + CFG.LAMBDA_PROXY * l_proxy
                + CFG.LAMBDA_EMA_DIST * l_ema
                + CFG.LAMBDA_ALT_CONSIST * l_alt_con
                + CFG.LAMBDA_MASK_RECON * l_recon
                + CFG.LAMBDA_ALT_PRED * l_alt_pred
                + CFG.LAMBDA_DIVERSITY * l_div
                + CFG.LAMBDA_DRONE_INTRA * w_mean * l_drone_intra
                + CFG.LAMBDA_SAT_INTRA * w_mean * l_sat_intra
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        total_sum += loss.item()
        n += 1
        loss_sums["ce"] += l_ce.item()
        loss_sums["nce"] += l_nce.item()
        loss_sums["con"] += l_con.item()
        loss_sums["cross"] += l_cross.item() if torch.is_tensor(l_cross) else l_cross
        loss_sums["self"] += l_self.item()
        loss_sums["uapa"] += l_uapa.item()
        loss_sums["proxy"] += l_proxy.item()
        loss_sums["ema"] += l_ema.item()
        loss_sums["alt_c"] += l_alt_con.item()
        loss_sums["recon"] += l_recon.item() if torch.is_tensor(l_recon) else l_recon
        loss_sums["alt_p"] += l_alt_pred.item()
        loss_sums["div"] += l_div.item()
        loss_sums["dr_intra"] += l_drone_intra.item()
        loss_sums["sat_intra"] += l_sat_intra.item()

    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}


def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EXP39: SPDGeo-DualDisc — Dual discriminative + altitude curriculum")
    print(f"  Dataset: SUES-200 | Epochs: {CFG.NUM_EPOCHS} | Device: {DEVICE}")
    print("=" * 65)

    print("\nLoading SUES-200 ...")
    train_ds = SUES200Dataset(CFG.SUES_ROOT, "train", transform=get_transforms("train"))
    test_ds = SUES200Dataset(CFG.SUES_ROOT, "test", transform=get_transforms("test"))
    train_loader = DataLoader(
        train_ds,
        batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
    )

    print("\nBuilding models ...")
    model = SPDGeoDualDiscModel(CFG.NUM_CLASSES).to(DEVICE)
    teacher = None
    try:
        teacher = DINOv2Teacher().to(DEVICE)
        teacher.eval()
    except Exception as e:
        print(f"  [WARN] Could not load teacher: {e}")

    ema = EMAModel(model, decay=CFG.EMA_DECAY)
    print(f"  EMA model initialized (decay={CFG.EMA_DECAY})")

    infonce = SupInfoNCELoss(temp=0.05).to(DEVICE)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    consist = PartConsistencyLoss()
    cross_dist = CrossDistillationLoss()
    self_dist = SelfDistillationLoss(temperature=CFG.DISTILL_TEMP)
    uapa_loss = UAPALoss(base_temperature=CFG.DISTILL_TEMP)
    base_losses = (infonce, ce, consist, cross_dist, self_dist, uapa_loss)

    proxy_anchor = ProxyAnchorLoss(CFG.NUM_CLASSES, CFG.EMBED_DIM, margin=CFG.PROXY_MARGIN, alpha=CFG.PROXY_ALPHA).to(DEVICE)
    ema_dist = EMADistillationLoss()
    alt_consist = AltitudeConsistencyLoss()
    diversity_loss = PrototypeDiversityLoss()
    new_losses = (proxy_anchor, ema_dist, alt_consist, diversity_loss)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone")]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": CFG.BACKBONE_LR},
            {"params": head_params, "lr": CFG.LR},
            {"params": infonce.parameters(), "lr": CFG.LR},
            {"params": proxy_anchor.parameters(), "lr": CFG.LR},
        ],
        weight_decay=CFG.WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=CFG.USE_AMP)
    best_r1 = 0.0
    results_log = []

    ckpt_path = os.path.join(CFG.OUTPUT_DIR, "exp39_spdgeo_dualdisc_best.pth")

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        if epoch <= CFG.WARMUP_EPOCHS:
            lr_scale = epoch / CFG.WARMUP_EPOCHS
        else:
            progress = (epoch - CFG.WARMUP_EPOCHS) / (CFG.NUM_EPOCHS - CFG.WARMUP_EPOCHS)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        lr_scale = max(lr_scale, 0.01)

        optimizer.param_groups[0]["lr"] = CFG.BACKBONE_LR * lr_scale
        for i in [1, 2, 3]:
            optimizer.param_groups[i]["lr"] = CFG.LR * lr_scale

        avg_loss, ld = train_one_epoch(
            model, teacher, ema, train_loader, base_losses, new_losses, optimizer, scaler, DEVICE, epoch
        )

        recon_tag = f"Rec {ld['recon']:.3f}" if epoch >= CFG.RECON_WARMUP else "Rec OFF"
        print(
            f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
            f"CE {ld['ce']:.3f} NCE {ld['nce']:.3f} Proxy {ld['proxy']:.3f} EMA {ld['ema']:.3f} | "
            f"AltC {ld['alt_c']:.3f} {recon_tag} AltP {ld['alt_p']:.3f} Div {ld['div']:.3f} | "
            f"DrIntra {ld['dr_intra']:.3f} SatIntra {ld['sat_intra']:.3f}"
        )

        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            metrics, per_alt = evaluate(model, test_ds, DEVICE)
            results_log.append({"epoch": epoch, **metrics})

            print(
                f"  -> R@1: {metrics['R@1']:.2f}% R@5: {metrics['R@5']:.2f}% "
                f"R@10: {metrics['R@10']:.2f}% mAP: {metrics['mAP']:.2f}%"
            )

            if metrics["R@1"] > best_r1:
                best_r1 = metrics["R@1"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "metrics": metrics,
                        "per_alt": per_alt,
                    },
                    ckpt_path,
                )
                print(f"  [BEST] Epoch {epoch} — R@1 {best_r1:.2f}% (checkpoint saved)")

    with open(os.path.join(CFG.OUTPUT_DIR, "exp39_spdgeo_dualdisc_log.json"), "w") as f:
        json.dump(results_log, f, indent=2)

    print("\nTraining finished.")
    print(f"Best R@1: {best_r1:.2f}% (checkpoint: {ckpt_path})")


if __name__ == "__main__":
    main()

