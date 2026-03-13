# EXP38: SPDGeo-CurrMask — Altitude-Adaptive Curriculum Masked Reconstruction cho SUES-200
# Mô tả: MAR được nâng cấp với tỉ lệ mask phụ thuộc độ cao + curriculum theo epoch và ưu tiên mask patch ít quan trọng (saliency thấp) để học vùng phân biệt mạnh hơn.

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

    # Base losses
    LAMBDA_CE = 1.0
    LAMBDA_INFONCE = 1.0
    LAMBDA_CONSISTENCY = 0.1
    LAMBDA_CROSS_DIST = 0.3
    LAMBDA_SELF_DIST = 0.3
    LAMBDA_UAPA = 0.2

    # DPEA + MAR components
    LAMBDA_PROXY = 0.5
    PROXY_MARGIN = 0.1
    PROXY_ALPHA = 32
    LAMBDA_EMA_DIST = 0.2
    EMA_DECAY = 0.996

    LAMBDA_ALT_CONSIST = 0.2
    # Base MAR settings kept but masking is curriculum/altitude-adaptive
    BASE_MASK_RATIO = 0.30
    LAMBDA_MASK_RECON = 0.3
    RECON_WARMUP = 10
    LAMBDA_ALT_PRED = 0.15
    LAMBDA_DIVERSITY = 0.05

    # Altitude-adaptive target mask ratios
    TARGET_MASK_150 = 0.20
    TARGET_MASK_200 = 0.26
    TARGET_MASK_250 = 0.33
    TARGET_MASK_300 = 0.40
    CURR_MASK_WARMUP_EPOCHS = 40

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


class CurriculumMaskedPartReconstruction(nn.Module):
    """
    MAR with:
      - altitude-adaptive target mask ratios
      - epoch curriculum from easy (low ratio) to hard
      - saliency-guided masking (mask least-salient patches first)
    """

    def __init__(self, part_dim: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(part_dim, part_dim * 2),
            nn.GELU(),
            nn.Linear(part_dim * 2, part_dim),
            nn.LayerNorm(part_dim),
        )

    def _alt_target_ratio(self, alt_idx: torch.Tensor) -> torch.Tensor:
        # alt_idx: [B] in {0,1,2,3}
        base = torch.full_like(alt_idx, CFG.TARGET_MASK_250, dtype=torch.float32)
        base = base.to(alt_idx.device)
        base = torch.where(alt_idx == 0, torch.tensor(CFG.TARGET_MASK_150, device=base.device), base)
        base = torch.where(alt_idx == 1, torch.tensor(CFG.TARGET_MASK_200, device=base.device), base)
        base = torch.where(alt_idx == 3, torch.tensor(CFG.TARGET_MASK_300, device=base.device), base)
        return base

    def current_ratio(self, alt_idx: torch.Tensor, epoch: int) -> torch.Tensor:
        target = self._alt_target_ratio(alt_idx)
        if epoch >= CFG.CURR_MASK_WARMUP_EPOCHS:
            return target
        # linear schedule: start 0.15 -> target over warmup epochs
        start = 0.15
        frac = epoch / max(CFG.CURR_MASK_WARMUP_EPOCHS, 1)
        return start + frac * (target - start)

    def forward(self, projected_patches: torch.Tensor, part_features: torch.Tensor, assignment: torch.Tensor, salience: torch.Tensor, alt_idx: torch.Tensor, epoch: int):
        """
        projected_patches: [B, N, D]
        part_features:     [B, K, D]
        assignment:        [B, N, K]
        salience:          [B, K]
        alt_idx:           [B]
        """
        b, n, d = projected_patches.shape
        k = part_features.shape[1]

        # saliency for patches: approximate by expectation of part salience under assignment
        # patch_saliency[b, n] = sum_k assign[b,n,k] * salience[b,k]
        patch_saliency = torch.einsum("bnk,bk->bn", assignment, salience)

        # compute per-sample mask ratio
        ratios = self.current_ratio(alt_idx, epoch).clamp(0.05, 0.6)  # [B]
        # for each sample, mask the lowest-saliency patches according to its ratio
        device = projected_patches.device
        losses = []
        for i in range(b):
            num_mask = max(1, int(n * float(ratios[i].item())))
            # sort ascending (least salient first)
            _, idx_sorted = torch.sort(patch_saliency[i], dim=0, descending=False)
            mask_idx = idx_sorted[:num_mask]  # [M]

            target = projected_patches[i].detach()  # [N,D]
            mask_expand = mask_idx.unsqueeze(-1).expand(-1, d)
            masked_targets = torch.gather(target, 0, mask_expand)

            assign_i = assignment[i]  # [N,K]
            mask_expand_k = mask_idx.unsqueeze(-1).expand(-1, k)
            masked_assign = torch.gather(assign_i, 0, mask_expand_k)  # [M,K]
            recon = masked_assign @ part_features[i]  # [M,D]
            recon = self.decoder(recon)

            recon_norm = F.normalize(recon, dim=-1)
            target_norm = F.normalize(masked_targets, dim=-1)
            losses.append(1 - (recon_norm * target_norm).sum(dim=-1).mean())

        return torch.stack(losses).mean(), ratios.mean().item()


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


class SPDGeoCurrMaskModel(nn.Module):
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

        self.mask_recon = CurriculumMaskedPartReconstruction(cfg.PART_DIM)
        self.alt_pred = AltitudePredictionHead(cfg.EMBED_DIM)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-CurrMask student: {total/1e6:.1f}M trainable parameters")

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


class SupInfoNCELoss(nn.Module):
    def __init__(self, temp: float = 0.05):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(temp).log())

    def forward(self, q_emb: torch.Tensor, r_emb: torch.Tensor, labels: torch.Tensor):
        t = self.log_t.exp().clamp(0.01, 1.0)
        sim = q_emb @ r_emb.t() / t
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


@torch.no_grad()
def evaluate(model: nn.Module, test_ds: SUES200Dataset, device: torch.device):
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


def train_one_epoch(model: SPDGeoCurrMaskModel, teacher: nn.Module | None, ema: EMAModel, loader, losses, new_losses, optimizer, scaler, device, epoch: int):
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

            l_ce = ce(d_out["logits"], labels) + ce(s_out["logits"], labels)
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

            l_proxy = 0.5 * (proxy_anchor(d_out["embedding"], labels) + proxy_anchor(s_out["embedding"], labels))

            with torch.no_grad():
                ema_drone_emb = ema.forward(drone, alt_idx=alt_idx)
                ema_sat_emb = ema.forward(sat, alt_idx=None)
            l_ema = 0.5 * (ema_dist(d_out["embedding"], ema_drone_emb) + ema_dist(s_out["embedding"], ema_sat_emb))

            l_alt_con = alt_consist(d_out["embedding"], labels, alts)

            if recon_active:
                l_recon_d, ratio_d = model.mask_recon(
                    d_out["parts"]["projected_patches"],
                    d_out["parts"]["part_features"],
                    d_out["parts"]["assignment"],
                    d_out["parts"]["salience"],
                    alt_idx,
                    epoch,
                )
                l_recon_s, ratio_s = model.mask_recon(
                    s_out["parts"]["projected_patches"],
                    s_out["parts"]["part_features"],
                    s_out["parts"]["assignment"],
                    s_out["parts"]["salience"],
                    torch.zeros_like(alt_idx),
                    epoch,
                )
                l_recon = 0.5 * (l_recon_d + l_recon_s)
                avg_ratio = 0.5 * (ratio_d + ratio_s)
            else:
                l_recon = torch.tensor(0.0, device=device)
                avg_ratio = 0.0

            l_alt_pred = model.alt_pred(d_out["embedding"].detach(), alt_norm)
            l_div = diversity_loss(model.part_disc.prototypes)

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
        loss_sums["mask_ratio"] += avg_ratio

    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}


def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EXP38: SPDGeo-CurrMask — Altitude-Adaptive Curriculum Masked Reconstruction")
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
    model = SPDGeoCurrMaskModel(CFG.NUM_CLASSES).to(DEVICE)
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

    ckpt_path = os.path.join(CFG.OUTPUT_DIR, "exp38_spdgeo_currmask_best.pth")

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

        recon_tag = f"Rec {ld['recon']:.3f} (mask={ld['mask_ratio']:.3f})" if epoch >= CFG.RECON_WARMUP else "Rec OFF"
        print(
            f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
            f"CE {ld['ce']:.3f} NCE {ld['nce']:.3f} Proxy {ld['proxy']:.3f} EMA {ld['ema']:.3f} | "
            f"AltC {ld['alt_c']:.3f} {recon_tag} AltP {ld['alt_p']:.3f} Div {ld['div']:.3f}"
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

    with open(os.path.join(CFG.OUTPUT_DIR, "exp38_spdgeo_currmask_log.json"), "w") as f:
        json.dump(results_log, f, indent=2)

    print("\nTraining finished.")
    print(f"Best R@1: {best_r1:.2f}% (checkpoint: {ckpt_path})")


if __name__ == "__main__":
    main()

# =============================================================================
# EXP38: SPDGeo-CurrMask — Altitude-Adaptive Curriculum Masking
# =============================================================================
# Base:    SPDGeo-DPEA-MAR (EXP35-FM, 95.08% R@1)
# Novel:   1) CurriculumMaskedPartReconstruction — per-altitude mask ratio:
#             150m=0.20, 200m=0.26, 250m=0.33, 300m=0.40
#          2) EpochCurriculum — all altitudes start at 0.15 mask ratio,
#             linearly reach altitude-specific target by epoch 40
#          3) SaliencyGuidedMasking — masks low-salience patches first,
#             forcing reconstruction of discriminative (high-salience) regions
#          4) AdaptiveReconWeight — λ_mask scales with masking difficulty
#
# Ref:     SinGeo (arxiv 2603.09377) — curriculum learning for robust CVGL
#          Our extension: altitude-conditioned masking + saliency guidance
#
# Total losses: 12 (same as DPEA-MAR, L_mask upgraded to curriculum variant)
# Expected: 95.3–95.8% R@1
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

    NUM_EPOCHS      = 120
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
    LAMBDA_ALT_PRED     = 0.15
    LAMBDA_DIVERSITY    = 0.05

    MASK_RATIOS_BY_ALT  = {0: 0.20, 1: 0.26, 2: 0.33, 3: 0.40}
    MASK_WARMUP_RATIO   = 0.15
    MASK_CURRICULUM_END = 40
    RECON_WARMUP        = 10

    DISTILL_TEMP    = 4.0
    EVAL_INTERVAL   = 5
    NUM_WORKERS     = 2

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
        self.root=root; self.mode=mode; self.altitudes=altitudes or CFG.ALTITUDES
        self.transform=transform
        self.drone_dir=os.path.join(root,CFG.DRONE_DIR); self.sat_dir=os.path.join(root,CFG.SAT_DIR)
        loc_ids=CFG.TRAIN_LOCS if mode=="train" else CFG.TEST_LOCS
        self.locations=[f"{l:04d}" for l in loc_ids]
        self.location_to_idx={l:i for i,l in enumerate(self.locations)}
        self.samples=[]; self.drone_by_location=defaultdict(list)
        for loc in self.locations:
            li=self.location_to_idx[loc]; sp=os.path.join(self.sat_dir,loc,"0.png")
            if not os.path.exists(sp): continue
            for alt in self.altitudes:
                ad=os.path.join(self.drone_dir,loc,alt)
                if not os.path.isdir(ad): continue
                for img in sorted(os.listdir(ad)):
                    if img.endswith(('.png','.jpg','.jpeg')):
                        self.samples.append((os.path.join(ad,img),sp,li,alt))
                        self.drone_by_location[li].append(len(self.samples)-1)
        print(f"  [{mode}] {len(self.samples)} samples, {len(self.locations)} locations")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        dp,sp,li,alt=self.samples[idx]
        try: d=Image.open(dp).convert('RGB'); s=Image.open(sp).convert('RGB')
        except: d=Image.new('RGB',(CFG.IMG_SIZE,)*2,(128,)*3); s=Image.new('RGB',(CFG.IMG_SIZE,)*2,(128,)*3)
        if self.transform: d=self.transform(d); s=self.transform(s)
        return {'drone':d,'satellite':s,'label':li,'altitude':int(alt),
                'alt_idx':CFG.ALT_TO_IDX.get(alt,0),'alt_norm':(int(alt)-150)/150.0}

class PKSampler:
    def __init__(self,ds,p,k): self.ds=ds;self.p=p;self.k=k;self.locs=list(ds.drone_by_location.keys())
    def __iter__(self):
        locs=self.locs.copy();random.shuffle(locs);batch=[]
        for l in locs:
            idx=self.ds.drone_by_location[l]
            if len(idx)<self.k: idx=idx*(self.k//len(idx)+1)
            batch.extend(random.sample(idx,self.k))
            if len(batch)>=self.p*self.k: yield batch[:self.p*self.k]; batch=batch[self.p*self.k:]
    def __len__(self): return len(self.locs)//self.p

def get_transforms(mode="train",img_size=None):
    sz=img_size or CFG.IMG_SIZE
    if mode=="train":
        return transforms.Compose([transforms.Resize((sz,sz)),transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(sz,scale=(0.8,1.0)),transforms.ColorJitter(0.2,0.2,0.2,0.05),
            transforms.RandomGrayscale(p=0.02),transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225]),transforms.RandomErasing(p=0.1,scale=(0.02,0.15))])
    return transforms.Compose([transforms.Resize((sz,sz)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])


# =============================================================================
# BACKBONE / TEACHER
# =============================================================================
class DINOv2Backbone(nn.Module):
    def __init__(self, ub=4):
        super().__init__()
        self.model=torch.hub.load('facebookresearch/dinov2','dinov2_vits14',pretrained=True); self.patch_size=14
        for p in self.model.parameters(): p.requires_grad=False
        for blk in self.model.blocks[-ub:]:
            for p in blk.parameters(): p.requires_grad=True
        for p in self.model.norm.parameters(): p.requires_grad=True
        f=sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        t=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  DINOv2 ViT-S/14: {f/1e6:.1f}M frozen, {t/1e6:.1f}M trainable")
    def forward(self, x):
        ft=self.model.forward_features(x)
        return ft['x_norm_patchtokens'],ft['x_norm_clstoken'],(x.shape[2]//self.patch_size,x.shape[3]//self.patch_size)

class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__(); print("  Loading DINOv2 ViT-B/14 teacher ...")
        self.model=torch.hub.load('facebookresearch/dinov2','dinov2_vitb14',pretrained=True)
        for p in self.model.parameters(): p.requires_grad=False
        print("  Teacher loaded.")
    @torch.no_grad()
    def forward(self,x): return self.model.forward_features(x)['x_norm_clstoken']


# =============================================================================
# MODULES
# =============================================================================
class DeepAltitudeFiLM(nn.Module):
    def __init__(self,na=4,fd=256):
        super().__init__(); self.gamma=nn.Parameter(torch.ones(na,fd)); self.beta=nn.Parameter(torch.zeros(na,fd))
    def forward(self,f,ai=None):
        if ai is None: g=self.gamma.mean(0,True);b=self.beta.mean(0,True); return f*g.unsqueeze(0)+b.unsqueeze(0)
        return f*self.gamma[ai].unsqueeze(1)+self.beta[ai].unsqueeze(1)

class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(self,fd=384,np_=8,pd=256,t=0.07,na=4):
        super().__init__(); self.temperature=t
        self.feat_proj=nn.Sequential(nn.Linear(fd,pd),nn.LayerNorm(pd),nn.GELU())
        self.altitude_film=DeepAltitudeFiLM(na,pd)
        self.prototypes=nn.Parameter(torch.randn(np_,pd)*0.02)
        self.refine=nn.Sequential(nn.LayerNorm(pd),nn.Linear(pd,pd*2),nn.GELU(),nn.Linear(pd*2,pd))
        self.salience_head=nn.Sequential(nn.Linear(pd,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
    def forward(self,pf,hw,alt_idx=None):
        b,n,_=pf.shape; h,w=hw
        ft=self.feat_proj(pf); ft=self.altitude_film(ft,alt_idx)
        fn=F.normalize(ft,-1); pn=F.normalize(self.prototypes,-1)
        sim=torch.einsum('bnd,kd->bnk',fn,pn)/self.temperature; assign=F.softmax(sim,-1)
        at=assign.transpose(1,2); mass=at.sum(-1,True).clamp(1e-6)
        pft=torch.bmm(at,ft)/mass; pft=pft+self.refine(pft)
        dev=ft.device; gy=torch.arange(h,device=dev).float()/max(h-1,1); gx=torch.arange(w,device=dev).float()/max(w-1,1)
        gy2,gx2=torch.meshgrid(gy,gx,indexing='ij'); coords=torch.stack([gx2.flatten(),gy2.flatten()],-1)
        pp=torch.bmm(at,coords.unsqueeze(0).expand(b,-1,-1))/mass
        sal=self.salience_head(pft).squeeze(-1)
        return {"part_features":pft,"part_positions":pp,"assignment":assign,"salience":sal,"projected_patches":ft}

class PartAwarePooling(nn.Module):
    def __init__(self,pd=256,ed=512):
        super().__init__()
        self.attn=nn.Sequential(nn.Linear(pd,pd//2),nn.Tanh(),nn.Linear(pd//2,1))
        self.proj=nn.Sequential(nn.Linear(pd*3,ed),nn.LayerNorm(ed),nn.GELU(),nn.Linear(ed,ed))
    def forward(self,pf,sal=None):
        aw=self.attn(pf)
        if sal is not None: aw=aw+sal.unsqueeze(-1).log().clamp(-10)
        aw=F.softmax(aw,1); ap=(aw*pf).sum(1); mp=pf.mean(1); xp=pf.max(1)[0]
        return F.normalize(self.proj(torch.cat([ap,mp,xp],-1)),-1)

class DynamicFusionGate(nn.Module):
    def __init__(self,ed=512):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(ed*2,ed//2),nn.ReLU(True),nn.Linear(ed//2,1))
        nn.init.constant_(self.gate[-1].bias,0.85)
    def forward(self,pe,ce):
        a=torch.sigmoid(self.gate(torch.cat([pe,ce],-1))); return F.normalize(a*pe+(1-a)*ce,-1)

class CurriculumMaskedPartReconstruction(nn.Module):
    """Altitude-adaptive curriculum masking with saliency guidance."""
    def __init__(self, part_dim=256, alt_ratios=None, warmup_ratio=0.15):
        super().__init__()
        self.alt_ratios = alt_ratios or {0:0.20, 1:0.26, 2:0.33, 3:0.40}
        self.warmup_ratio = warmup_ratio
        self.decoder = nn.Sequential(nn.Linear(part_dim, part_dim*2), nn.GELU(),
                                     nn.Linear(part_dim*2, part_dim), nn.LayerNorm(part_dim))

    def _ratio(self, alt_idx_val, epoch, curriculum_end):
        target = self.alt_ratios.get(alt_idx_val, 0.30)
        progress = min(max(epoch, 0) / max(curriculum_end, 1), 1.0)
        return self.warmup_ratio + (target - self.warmup_ratio) * progress

    def forward(self, projected_patches, part_features, assignment, salience, alt_idx, epoch,
                curriculum_end=40):
        b, n, d = projected_patches.shape
        device = projected_patches.device
        total_loss = torch.tensor(0.0, device=device)

        for si in range(b):
            ai = alt_idx[si].item() if torch.is_tensor(alt_idx[si]) else alt_idx[si]
            ratio = self._ratio(ai, epoch, curriculum_end)
            nm = max(1, int(n * ratio))

            patch_sal = (assignment[si] @ salience[si].unsqueeze(-1)).squeeze(-1)
            noise = torch.rand(n, device=device) * 0.3
            mask_score = -patch_sal + noise
            mi = mask_score.argsort()[:nm]

            tgt = projected_patches[si].detach()[mi]
            ma = assignment[si][mi]
            rec = self.decoder(ma @ part_features[si])
            rn = F.normalize(rec, dim=-1); tn = F.normalize(tgt, dim=-1)
            total_loss = total_loss + (1 - (rn * tn).sum(-1)).mean()

        return total_loss / max(b, 1)

    def avg_ratio(self, epoch, curriculum_end=40):
        return sum(self._ratio(ai, epoch, curriculum_end) for ai in range(4)) / 4.0


class AltitudePredictionHead(nn.Module):
    def __init__(self,ed=512):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(ed,128),nn.ReLU(True),nn.Dropout(0.1),nn.Linear(128,1),nn.Sigmoid())
    def forward(self,emb,at): return F.smooth_l1_loss(self.head(emb).squeeze(-1),at)

class PrototypeDiversityLoss(nn.Module):
    def forward(self,pr):
        p=F.normalize(pr,-1);s=p@p.T;k=s.size(0);return(s*(1-torch.eye(k,device=s.device))).abs().sum()/(k*(k-1))


# =============================================================================
# MODEL
# =============================================================================
class SPDGeoCurrMaskModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone=DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc=AltitudeAwarePartDiscovery(384,cfg.N_PARTS,cfg.PART_DIM,cfg.CLUSTER_TEMP,cfg.NUM_ALTITUDES)
        self.pool=PartAwarePooling(cfg.PART_DIM,cfg.EMBED_DIM)
        self.fusion_gate=DynamicFusionGate(cfg.EMBED_DIM)
        self.bottleneck=nn.Sequential(nn.Linear(cfg.EMBED_DIM,cfg.EMBED_DIM),nn.BatchNorm1d(cfg.EMBED_DIM),nn.ReLU(True))
        self.classifier=nn.Linear(cfg.EMBED_DIM,num_classes)
        self.cls_proj=nn.Sequential(nn.Linear(384,cfg.EMBED_DIM),nn.BatchNorm1d(cfg.EMBED_DIM),nn.ReLU(True))
        self.cls_classifier=nn.Linear(cfg.EMBED_DIM,num_classes)
        self.teacher_proj=nn.Sequential(nn.Linear(cfg.EMBED_DIM,cfg.TEACHER_DIM),nn.LayerNorm(cfg.TEACHER_DIM))
        self.mask_recon=CurriculumMaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIOS_BY_ALT, cfg.MASK_WARMUP_RATIO)
        self.alt_pred=AltitudePredictionHead(cfg.EMBED_DIM)
        t=sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-CurrMask: {t/1e6:.1f}M trainable")

    def extract_embedding(self,x,alt_idx=None):
        pa,cl,hw=self.backbone(x); pts=self.part_disc(pa,hw,alt_idx)
        emb=self.pool(pts["part_features"],pts["salience"])
        ce=F.normalize(self.cls_proj(cl),-1); return self.fusion_gate(emb,ce)

    def forward(self,x,alt_idx=None,return_parts=False):
        pa,cl,hw=self.backbone(x); pts=self.part_disc(pa,hw,alt_idx)
        emb=self.pool(pts["part_features"],pts["salience"])
        bn=self.bottleneck(emb); logits=self.classifier(bn)
        cr=self.cls_proj(cl); cl_logits=self.cls_classifier(cr)
        cn=F.normalize(cr,-1); fused=self.fusion_gate(emb,cn)
        pf=self.teacher_proj(emb)
        out={"embedding":fused,"logits":logits,"cls_logits":cl_logits,"projected_feat":pf,
             "part_emb":emb,"cls_emb":cn}
        if return_parts: out["parts"]=pts
        return out

class EMAModel:
    def __init__(self,m,d=0.996):
        self.decay=d;self.model=copy.deepcopy(m);self.model.eval()
        for p in self.model.parameters(): p.requires_grad=False
    @torch.no_grad()
    def update(self,m):
        for ep,mp in zip(self.model.parameters(),m.parameters()): ep.data.mul_(self.decay).add_(mp.data,alpha=1-self.decay)
    @torch.no_grad()
    def forward(self,x,ai=None): return self.model.extract_embedding(x,ai)


# =============================================================================
# LOSSES
# =============================================================================
class SupInfoNCELoss(nn.Module):
    def __init__(self,t=0.05): super().__init__(); self.log_t=nn.Parameter(torch.tensor(t).log())
    def forward(self,q,r,l):
        t=self.log_t.exp().clamp(0.01,1.0); s=q@r.t()/t; l=l.view(-1,1); p=l.eq(l.T).float()
        lp=s-torch.logsumexp(s,1,True); return(-(lp*p).sum(1)/p.sum(1).clamp(1)).mean()

class PartConsistencyLoss(nn.Module):
    def forward(self,a,b):
        da=a.mean(1);db=b.mean(1)
        return 0.5*(F.kl_div((da+1e-8).log(),db,reduction='batchmean',log_target=False)+F.kl_div((db+1e-8).log(),da,reduction='batchmean',log_target=False))

class CrossDistillationLoss(nn.Module):
    def forward(self,s,t): s=F.normalize(s,-1);t=F.normalize(t,-1); return F.mse_loss(s,t)+(1-F.cosine_similarity(s,t).mean())

class SelfDistillationLoss(nn.Module):
    def __init__(self,t=4.0): super().__init__();self.t=t
    def forward(self,w,s): p=F.softmax(s/self.t,1).detach(); return(self.t**2)*F.kl_div(F.log_softmax(w/self.t,1),p,reduction='batchmean')

class UAPALoss(nn.Module):
    def __init__(self,bt=4.0): super().__init__();self.t0=bt
    @staticmethod
    def _e(l): p=F.softmax(l,1);return-(p*(p+1e-8).log()).sum(1).mean()
    def forward(self,d,s):
        du=self._e(d)-self._e(s);t=self.t0*(1+torch.sigmoid(du))
        p=F.softmax(s/t,1).detach(); return(t**2)*F.kl_div(F.log_softmax(d/t,1),p,reduction='batchmean')

class ProxyAnchorLoss(nn.Module):
    def __init__(self,nc,ed,m=0.1,a=32):
        super().__init__();self.proxies=nn.Parameter(torch.randn(nc,ed)*0.01);nn.init.kaiming_normal_(self.proxies,mode='fan_out')
        self.m=m;self.a=a;self.nc=nc
    def forward(self,emb,l):
        p=F.normalize(self.proxies,-1);s=emb@p.T;oh=F.one_hot(l,self.nc).float()
        pe=torch.exp(-self.a*(s*oh-self.m))*oh;hp=oh.sum(0)>0
        pl=torch.log(1+pe.sum(0))[hp].mean() if hp.sum()>0 else torch.tensor(0.,device=emb.device)
        nm=1-oh;nl=torch.log(1+(torch.exp(self.a*(s*nm+self.m))*nm).sum(0)).mean()
        return pl+nl

class EMADistillationLoss(nn.Module):
    def forward(self,s,e): return(1-F.cosine_similarity(s,e)).mean()

class AltitudeConsistencyLoss(nn.Module):
    def forward(self,emb,l,a):
        b=emb.size(0)
        if b<2: return torch.tensor(0.,device=emb.device)
        ll=l.view(-1,1);aa=a.view(-1,1);mask=ll.eq(ll.T)&aa.ne(aa.T)
        if mask.sum()==0: return torch.tensor(0.,device=emb.device)
        en=F.normalize(emb,-1);return((1-en@en.T)*mask.float()).sum()/mask.float().sum().clamp(1)


# =============================================================================
# EVALUATE
# =============================================================================
@torch.no_grad()
def evaluate(model, test_ds, device):
    model.eval(); ttf=get_transforms("test")
    loader=DataLoader(test_ds,batch_size=64,shuffle=False,num_workers=CFG.NUM_WORKERS,pin_memory=True)
    df,dl,da=[],[],[]
    for b in loader:
        f=model.extract_embedding(b['drone'].to(device),alt_idx=b['alt_idx'].to(device)).cpu()
        df.append(f);dl.append(b['label']);da.append(b['altitude'])
    df=torch.cat(df);dl=torch.cat(dl);da=torch.cat(da)
    als=[f"{l:04d}" for l in range(1,201)];si=[];sl=[];dc=0
    for loc in als:
        sp=os.path.join(test_ds.sat_dir,loc,"0.png")
        if not os.path.exists(sp): continue
        si.append(ttf(Image.open(sp).convert('RGB')))
        sl.append(test_ds.location_to_idx[loc] if loc in test_ds.location_to_idx else -1000-dc)
        if loc not in test_ds.location_to_idx: dc+=1
    sf=[]
    for i in range(0,len(si),64): sf.append(model.extract_embedding(torch.stack(si[i:i+64]).to(device),alt_idx=None).cpu())
    sf=torch.cat(sf);sll=torch.tensor(sl)
    print(f"  Gallery: {len(sf)} | Queries: {len(df)}")
    sim=df@sf.T;_,rank=sim.sort(1,descending=True);n=df.size(0);r1=r5=r10=ap=0
    for i in range(n):
        m=torch.where(sll[rank[i]]==dl[i])[0]
        if len(m)==0: continue
        f=m[0].item()
        if f<1:r1+=1
        if f<5:r5+=1
        if f<10:r10+=1
        ap+=sum((j+1)/(p.item()+1) for j,p in enumerate(m))/len(m)
    overall={"R@1":r1/n*100,"R@5":r5/n*100,"R@10":r10/n*100,"mAP":ap/n*100}
    per_alt={}
    for alt in sorted(da.unique().tolist()):
        msk=da==alt
        if msk.sum()==0: continue
        af=df[msk];al=dl[msk];s=af@sf.T;_,rk=s.sort(1,descending=True)
        k=af.size(0);a1=a5=a10=aap=0
        for i in range(k):
            mm=torch.where(sll[rk[i]]==al[i])[0]
            if len(mm)==0: continue
            ff=mm[0].item()
            if ff<1:a1+=1
            if ff<5:a5+=1
            if ff<10:a10+=1
            aap+=sum((j+1)/(p.item()+1) for j,p in enumerate(mm))/len(mm)
        per_alt[int(alt)]={"R@1":a1/k*100,"R@5":a5/k*100,"R@10":a10/k*100,"mAP":aap/k*100,"n":k}
    return overall,per_alt


# =============================================================================
# TRAIN
# =============================================================================
def train_one_epoch(model, teacher, ema, loader, losses, new_losses, optimizer, scaler, device, epoch):
    model.train()
    if teacher: teacher.eval()
    infonce,ce,consist,cross_dist,self_dist,uapa=losses
    proxy_anchor,ema_dist,alt_consist,div_loss=new_losses
    total_sum=0;n_=0;ls=defaultdict(float)
    recon_active = epoch >= CFG.RECON_WARMUP
    avg_r = model.mask_recon.avg_ratio(epoch, CFG.MASK_CURRICULUM_END)
    lambda_mask = 0.20 + 0.20 * (avg_r / 0.40)

    for batch in tqdm(loader,desc=f"Ep{epoch:3d}",leave=False):
        dr=batch['drone'].to(device);sa=batch['satellite'].to(device)
        lb=batch['label'].to(device);ai=batch['alt_idx'].to(device)
        al=batch['altitude'].to(device);an=batch['alt_norm'].to(device).float()
        optimizer.zero_grad()
        with torch.amp.autocast('cuda',enabled=CFG.USE_AMP):
            do=model(dr,alt_idx=ai,return_parts=True)
            so=model(sa,alt_idx=None,return_parts=True)
            l_ce=ce(do['logits'],lb)+ce(so['logits'],lb)+0.3*(ce(do['cls_logits'],lb)+ce(so['cls_logits'],lb))
            l_nce=infonce(do['embedding'],so['embedding'],lb)
            l_con=consist(do['parts']['assignment'],so['parts']['assignment'])
            if teacher:
                with torch.no_grad(): td=teacher(dr);ts=teacher(sa)
                l_cross=cross_dist(do['projected_feat'],td)+cross_dist(so['projected_feat'],ts)
            else: l_cross=torch.tensor(0.,device=device)
            l_self=self_dist(do['cls_logits'],do['logits'])+self_dist(so['cls_logits'],so['logits'])
            l_uapa=uapa(do['logits'],so['logits'])
            l_proxy=0.5*(proxy_anchor(do['embedding'],lb)+proxy_anchor(so['embedding'],lb))
            with torch.no_grad(): ed=ema.forward(dr,ai);es=ema.forward(sa,None)
            l_ema=0.5*(ema_dist(do['embedding'],ed)+ema_dist(so['embedding'],es))
            l_altc=alt_consist(do['embedding'],lb,al)

            if recon_active:
                l_rec=0.5*(model.mask_recon(do['parts']['projected_patches'],do['parts']['part_features'],
                                            do['parts']['assignment'],do['parts']['salience'],ai,epoch,CFG.MASK_CURRICULUM_END)+
                           model.mask_recon(so['parts']['projected_patches'],so['parts']['part_features'],
                                            so['parts']['assignment'],so['parts']['salience'],
                                            torch.zeros_like(ai),epoch,CFG.MASK_CURRICULUM_END))
            else: l_rec=torch.tensor(0.,device=device)

            l_altp=model.alt_pred(do['embedding'].detach(),an)
            l_div=div_loss(model.part_disc.prototypes)

            loss=(CFG.LAMBDA_CE*l_ce+CFG.LAMBDA_INFONCE*l_nce+CFG.LAMBDA_CONSISTENCY*l_con+
                  CFG.LAMBDA_CROSS_DIST*l_cross+CFG.LAMBDA_SELF_DIST*l_self+CFG.LAMBDA_UAPA*l_uapa+
                  CFG.LAMBDA_PROXY*l_proxy+CFG.LAMBDA_EMA_DIST*l_ema+CFG.LAMBDA_ALT_CONSIST*l_altc+
                  lambda_mask*l_rec+CFG.LAMBDA_ALT_PRED*l_altp+CFG.LAMBDA_DIVERSITY*l_div)

        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
        scaler.step(optimizer);scaler.update();ema.update(model)
        total_sum+=loss.item();n_+=1
        for k2,v2 in [("ce",l_ce),("nce",l_nce),("con",l_con),("cross",l_cross),("self",l_self),
                       ("uapa",l_uapa),("proxy",l_proxy),("ema",l_ema),("alt_c",l_altc),
                       ("recon",l_rec),("alt_p",l_altp),("div",l_div)]:
            ls[k2]+=(v2.item() if torch.is_tensor(v2) else v2)
    return total_sum/max(n_,1),{k:v/max(n_,1) for k,v in ls.items()}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED);os.makedirs(CFG.OUTPUT_DIR,exist_ok=True)
    print("="*65)
    print("  EXP38: SPDGeo-CurrMask — Altitude-Adaptive Curriculum Masking")
    print(f"  Mask ratios: {CFG.MASK_RATIOS_BY_ALT} | Warmup: {CFG.MASK_WARMUP_RATIO}")
    print(f"  Curriculum end: ep{CFG.MASK_CURRICULUM_END} | Epochs: {CFG.NUM_EPOCHS}")
    print("="*65)

    train_ds=SUES200Dataset(CFG.SUES_ROOT,"train",transform=get_transforms("train"))
    test_ds=SUES200Dataset(CFG.SUES_ROOT,"test",transform=get_transforms("test"))
    train_loader=DataLoader(train_ds,batch_sampler=PKSampler(train_ds,CFG.P_CLASSES,CFG.K_SAMPLES),
                            num_workers=CFG.NUM_WORKERS,pin_memory=True)

    model=SPDGeoCurrMaskModel(CFG.NUM_CLASSES).to(DEVICE)
    teacher=None
    try: teacher=DINOv2Teacher().to(DEVICE);teacher.eval()
    except Exception as e: print(f"  [WARN] Teacher: {e}")
    ema=EMAModel(model,CFG.EMA_DECAY)

    infonce=SupInfoNCELoss(0.05).to(DEVICE);ce=nn.CrossEntropyLoss(label_smoothing=0.1)
    consist=PartConsistencyLoss();cross_dist=CrossDistillationLoss()
    self_dist=SelfDistillationLoss(CFG.DISTILL_TEMP);uapa_loss=UAPALoss(CFG.DISTILL_TEMP)
    base_losses=(infonce,ce,consist,cross_dist,self_dist,uapa_loss)

    proxy_anchor=ProxyAnchorLoss(CFG.NUM_CLASSES,CFG.EMBED_DIM,CFG.PROXY_MARGIN,CFG.PROXY_ALPHA).to(DEVICE)
    ema_distl=EMADistillationLoss();alt_consist=AltitudeConsistencyLoss();div_loss=PrototypeDiversityLoss()
    new_losses=(proxy_anchor,ema_distl,alt_consist,div_loss)

    bbp=[p for p in model.backbone.parameters() if p.requires_grad]
    hdp=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('backbone')]
    optimizer=torch.optim.AdamW([{"params":bbp,"lr":CFG.BACKBONE_LR},{"params":hdp,"lr":CFG.LR},
        {"params":infonce.parameters(),"lr":CFG.LR},{"params":proxy_anchor.parameters(),"lr":CFG.LR}],
        weight_decay=CFG.WEIGHT_DECAY)
    scaler=torch.amp.GradScaler('cuda',enabled=CFG.USE_AMP)
    best_r1=0.;results_log=[];ckpt=os.path.join(CFG.OUTPUT_DIR,"exp38_currmask_best.pth")

    for epoch in range(1,CFG.NUM_EPOCHS+1):
        if epoch<=CFG.WARMUP_EPOCHS: lr_s=epoch/CFG.WARMUP_EPOCHS
        else: prog=(epoch-CFG.WARMUP_EPOCHS)/(CFG.NUM_EPOCHS-CFG.WARMUP_EPOCHS); lr_s=0.5*(1+math.cos(math.pi*prog))
        lr_s=max(lr_s,0.01)
        optimizer.param_groups[0]['lr']=CFG.BACKBONE_LR*lr_s
        for i in [1,2,3]: optimizer.param_groups[i]['lr']=CFG.LR*lr_s

        avg_loss,ld=train_one_epoch(model,teacher,ema,train_loader,base_losses,new_losses,optimizer,scaler,DEVICE,epoch)
        ar=model.mask_recon.avg_ratio(epoch,CFG.MASK_CURRICULUM_END)
        rt=f"Rec {ld['recon']:.3f} (r={ar:.2f})" if epoch>=CFG.RECON_WARMUP else "Rec OFF"
        print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | CE {ld['ce']:.3f} NCE {ld['nce']:.3f} | {rt}")

        if epoch%CFG.EVAL_INTERVAL==0 or epoch==CFG.NUM_EPOCHS:
            metrics,per_alt=evaluate(model,test_ds,DEVICE)
            results_log.append({"epoch":epoch,**metrics})
            print(f"  -> R@1: {metrics['R@1']:.2f}% R@5: {metrics['R@5']:.2f}% R@10: {metrics['R@10']:.2f}% mAP: {metrics['mAP']:.2f}%")
            if metrics['R@1']>best_r1:
                best_r1=metrics['R@1']
                torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"metrics":metrics,"per_alt":per_alt},ckpt)
                print(f"  * New best R@1: {best_r1:.2f}%")
            em,_=evaluate(ema.model,test_ds,DEVICE)
            print(f"  -> EMA R@1: {em['R@1']:.2f}%")
            if em['R@1']>best_r1:
                best_r1=em['R@1']
                torch.save({"epoch":epoch,"model_state_dict":ema.model.state_dict(),"metrics":em,"is_ema":True},ckpt)

    print(f"\n{'='*65}\n  EXP38 COMPLETE — Best R@1: {best_r1:.2f}%\n{'='*65}")
    with open(os.path.join(CFG.OUTPUT_DIR,"exp38_currmask_results.json"),"w") as f:
        json.dump({"results_log":results_log,"best_r1":best_r1,
                   "config":{k:v for k,v in vars(CFG).items() if not k.startswith('_')}},f,indent=2)

if __name__=="__main__":
    main()
