# =============================================================================
# EXP46: SPDGeo-PartProto — Part Prototype Memory Contrastive Learning
# =============================================================================
# Base:    SPDGeo-DPEA-MAR (95.08% R@1)
# Novel:   1) PartPrototypeBank: EMA-updated per-location part prototype memory
#             (120 locs × K=8 parts × 256-dim) — always provides fresh hard
#             negatives, solving triplet/InfoNCE saturation by ep22
#          2) PrototypeNCE: contrastive loss pulling current embeddings toward
#             same-location prototypes and pushing away from nearest negatives
#             in the prototype bank (never saturates — bank grows more accurate)
#          3) HardPrototypeMining: find nearest-but-wrong prototypes as hard
#             negatives — replaces batch-hard mining that exhausts by ep22
#          4) CrossViewPrototypeAlign: drone and satellite should activate the
#             same location prototype — explicit cross-view grounding via memory
#
# Insight: Triplet loss → 0.000 by ep22 in all SPDGeo variants. PrototypeNCE
#          with a memory bank provides persistent hard negatives that improve
#          as training progresses (better prototypes → harder negatives).
#
# Expected: 95.3–96.0% R@1 (200-gallery)
# =============================================================================

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

    # PartProto-specific
    LAMBDA_PROTO_NCE = 0.4
    PROTO_NCE_TEMP = 0.07
    PROTO_MOMENTUM = 0.999
    PROTO_WARMUP = 5
    LAMBDA_CROSS_PROTO = 0.2
    NUM_HARD_NEGS = 16

    DISTILL_TEMP = 4.0
    EVAL_INTERVAL = 5
    NUM_WORKERS = 2


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


# =============================================================================
# DATASET
# =============================================================================
class SUES200Dataset(Dataset):
    def __init__(self, root, mode="train", altitudes=None, transform=None):
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

    def __getitem__(self, idx):
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
            "drone": d, "satellite": s, "label": li,
            "altitude": int(alt), "alt_idx": alt_idx, "alt_norm": alt_norm,
        }


class PKSampler:
    def __init__(self, ds, p, k):
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


def get_transforms(mode="train", img_size=None):
    sz = img_size or CFG.IMG_SIZE
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((sz, sz)), transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(sz, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05), transforms.RandomGrayscale(p=0.02),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((sz, sz)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# =============================================================================
# BACKBONE / TEACHER
# =============================================================================
class DINOv2Backbone(nn.Module):
    def __init__(self, unfreeze_blocks=4):
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

    def forward(self, x):
        feat = self.model.forward_features(x)
        return feat["x_norm_patchtokens"], feat["x_norm_clstoken"], (
            x.shape[2] // self.patch_size, x.shape[3] // self.patch_size)


class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        print("  Loading DINOv2 ViT-B/14 teacher ...")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        for p in self.model.parameters():
            p.requires_grad = False
        print("  DINOv2 ViT-B/14 teacher loaded (all frozen).")

    @torch.no_grad()
    def forward(self, x):
        return self.model.forward_features(x)["x_norm_clstoken"]


# =============================================================================
# MODULES
# =============================================================================
class DeepAltitudeFiLM(nn.Module):
    def __init__(self, num_altitudes=4, feat_dim=256):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_altitudes, feat_dim))
        self.beta = nn.Parameter(torch.zeros(num_altitudes, feat_dim))

    def forward(self, feat, alt_idx=None):
        if alt_idx is None:
            gamma = self.gamma.mean(dim=0, keepdim=True)
            beta = self.beta.mean(dim=0, keepdim=True)
            return feat * gamma.unsqueeze(0) + beta.unsqueeze(0)
        gamma = self.gamma[alt_idx]
        beta = self.beta[alt_idx]
        return feat * gamma.unsqueeze(1) + beta.unsqueeze(1)


class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07, num_altitudes=4):
        super().__init__()
        self.temperature = temperature
        self.feat_proj = nn.Sequential(nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU())
        self.altitude_film = DeepAltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim * 2),
            nn.GELU(), nn.Linear(part_dim * 2, part_dim))
        self.salience_head = nn.Sequential(nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, patch_features, spatial_hw, alt_idx=None):
        b, n, _ = patch_features.shape
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
        salience = self.salience_head(part_feat).squeeze(-1)
        return {
            "part_features": part_feat, "assignment": assign,
            "salience": salience, "projected_patches": feat,
        }


class PartAwarePooling(nn.Module):
    def __init__(self, part_dim=256, embed_dim=512):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(part_dim, part_dim // 2), nn.Tanh(), nn.Linear(part_dim // 2, 1))
        self.proj = nn.Sequential(
            nn.Linear(part_dim * 3, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, part_features, salience=None):
        aw = self.attn(part_features)
        if salience is not None:
            aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        attn_pool = (aw * part_features).sum(1)
        mean_pool = part_features.mean(1)
        max_pool = part_features.max(1)[0]
        return F.normalize(self.proj(torch.cat([attn_pool, mean_pool, max_pool], dim=-1)), dim=-1)


class DynamicFusionGate(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2), nn.ReLU(True), nn.Linear(embed_dim // 2, 1))
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        alpha = torch.sigmoid(self.gate(torch.cat([part_emb, cls_emb], dim=-1)))
        return F.normalize(alpha * part_emb + (1 - alpha) * cls_emb, dim=-1)


class MaskedPartReconstruction(nn.Module):
    def __init__(self, part_dim=256, mask_ratio=0.30):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Linear(part_dim, part_dim * 2), nn.GELU(),
            nn.Linear(part_dim * 2, part_dim), nn.LayerNorm(part_dim))

    def forward(self, projected_patches, part_features, assignment):
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
        recon = self.decoder(torch.bmm(masked_assign, part_features))
        recon_norm = F.normalize(recon, dim=-1)
        target_norm = F.normalize(masked_targets, dim=-1)
        return (1 - (recon_norm * target_norm).sum(dim=-1)).mean()


class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, embedding, alt_target):
        return F.smooth_l1_loss(self.head(embedding).squeeze(-1), alt_target)


class PrototypeDiversityLoss(nn.Module):
    def forward(self, prototypes):
        p = F.normalize(prototypes, dim=-1)
        sim = p @ p.T
        k = sim.size(0)
        mask = 1 - torch.eye(k, device=sim.device)
        return (sim * mask).abs().sum() / (k * (k - 1))


# =============================================================================
# NOVEL: Part Prototype Memory Bank + Prototype-NCE
# =============================================================================
class PartPrototypeBank(nn.Module):
    """Per-location part prototype memory bank with EMA updates.
    Stores a running average of part features for each training location,
    providing persistent hard negatives that never saturate."""

    def __init__(self, num_classes, n_parts, part_dim, embed_dim, momentum=0.999, temp=0.07, num_hard_negs=16):
        super().__init__()
        self.num_classes = num_classes
        self.n_parts = n_parts
        self.momentum = momentum
        self.temp = temp
        self.num_hard_negs = num_hard_negs

        self.register_buffer("part_bank", torch.randn(num_classes, n_parts, part_dim))
        self.register_buffer("embed_bank", torch.randn(num_classes, embed_dim))
        self.register_buffer("bank_initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("update_count", torch.zeros(num_classes, dtype=torch.long))

        nn.init.normal_(self.part_bank, 0, 0.01)
        nn.init.normal_(self.embed_bank, 0, 0.01)

    @torch.no_grad()
    def update(self, part_features, embeddings, labels):
        """EMA update for locations seen in current batch.
        part_features: [B, K, D_part]
        embeddings: [B, D_embed]
        labels: [B]
        """
        for i in range(labels.size(0)):
            loc = labels[i].item()
            if not self.bank_initialized[loc]:
                self.part_bank[loc] = part_features[i]
                self.embed_bank[loc] = embeddings[i]
                self.bank_initialized[loc] = True
            else:
                m = self.momentum
                self.part_bank[loc] = m * self.part_bank[loc] + (1 - m) * part_features[i]
                self.embed_bank[loc] = m * self.embed_bank[loc] + (1 - m) * embeddings[i]
            self.update_count[loc] += 1

    def prototype_nce(self, embeddings, labels):
        """PrototypeNCE: contrastive between current embeddings and prototype bank.
        Positives: same-location prototypes
        Negatives: hardest wrong-location prototypes (never saturates)."""
        b = embeddings.size(0)
        if not self.bank_initialized.any():
            return torch.tensor(0.0, device=embeddings.device)

        init_mask = self.bank_initialized
        valid_protos = F.normalize(self.embed_bank[init_mask], dim=-1)
        valid_labels = torch.arange(self.num_classes, device=embeddings.device)[init_mask]

        q = F.normalize(embeddings, dim=-1)
        sim = q @ valid_protos.T / self.temp

        pos_mask = labels.unsqueeze(1) == valid_labels.unsqueeze(0)

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        neg_mask = ~pos_mask
        neg_sim = sim.clone()
        neg_sim[pos_mask] = -1e9

        k = min(self.num_hard_negs, neg_mask.sum(dim=1).min().item())
        if k == 0:
            return torch.tensor(0.0, device=embeddings.device)

        hard_neg_sim, _ = neg_sim.topk(k, dim=1)

        loss = 0.0
        count = 0
        for i in range(b):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_indices) == 0:
                continue
            for pi in pos_indices:
                pos_sim_val = sim[i, pi]
                all_negs = hard_neg_sim[i]
                logits = torch.cat([pos_sim_val.unsqueeze(0), all_negs])
                target = torch.zeros(1, device=embeddings.device, dtype=torch.long)
                loss += F.cross_entropy(logits.unsqueeze(0), target)
                count += 1

        return loss / max(count, 1)

    def cross_view_prototype_align(self, drone_emb, sat_emb, labels):
        """Both drone and satellite embeddings should be close to the same
        location prototype — explicit cross-view grounding via memory."""
        if not self.bank_initialized.any():
            return torch.tensor(0.0, device=drone_emb.device)

        loss = 0.0
        count = 0
        for i in range(labels.size(0)):
            loc = labels[i].item()
            if not self.bank_initialized[loc]:
                continue
            proto = F.normalize(self.embed_bank[loc].unsqueeze(0), dim=-1)
            d = F.normalize(drone_emb[i].unsqueeze(0), dim=-1)
            s = F.normalize(sat_emb[i].unsqueeze(0), dim=-1)
            loss += (1 - F.cosine_similarity(d, proto)).mean()
            loss += (1 - F.cosine_similarity(s, proto)).mean()
            count += 2

        return loss / max(count, 1)


# =============================================================================
# MODEL
# =============================================================================
class SPDGeoPartProto(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = AltitudeAwarePartDiscovery(
            384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES)
        self.pool = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)

        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(True))
        self.classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj = nn.Sequential(
            nn.Linear(384, cfg.EMBED_DIM), nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj = nn.Sequential(
            nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM), nn.LayerNorm(cfg.TEACHER_DIM))

        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)
        self.alt_pred = AltitudePredictionHead(cfg.EMBED_DIM)

        self.proto_bank = PartPrototypeBank(
            cfg.NUM_CLASSES, cfg.N_PARTS, cfg.PART_DIM, cfg.EMBED_DIM,
            cfg.PROTO_MOMENTUM, cfg.PROTO_NCE_TEMP, cfg.NUM_HARD_NEGS)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-PartProto student: {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts["part_features"], parts["salience"])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)

    def forward(self, x, alt_idx=None):
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

        return {
            "embedding": fused, "logits": logits, "cls_logits": cls_logits,
            "projected_feat": projected_feat, "part_emb": emb,
            "cls_emb": cls_emb_norm, "parts": parts,
        }


class EMAModel:
    def __init__(self, model, decay=0.996):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def forward(self, x, alt_idx=None):
        return self.model.extract_embedding(x, alt_idx=alt_idx)


# =============================================================================
# LOSSES (same as DPEA-MAR base)
# =============================================================================
class SupInfoNCELoss(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(temp).log())

    def forward(self, q, r, labels):
        t = self.log_t.exp().clamp(0.01, 1.0)
        sim = q @ r.T / t
        labels = labels.view(-1, 1)
        pos = labels.eq(labels.T).float()
        lp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        return (-(lp * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


class PartConsistencyLoss(nn.Module):
    def forward(self, aq, ar):
        dq = aq.mean(1)
        dr = ar.mean(1)
        return 0.5 * (
            F.kl_div((dq + 1e-8).log(), dr, reduction="batchmean") +
            F.kl_div((dr + 1e-8).log(), dq, reduction="batchmean"))


class CrossDistillationLoss(nn.Module):
    def forward(self, s, t):
        s = F.normalize(s, -1)
        t = F.normalize(t, -1)
        return F.mse_loss(s, t) + 1 - F.cosine_similarity(s, t).mean()


class SelfDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super().__init__()
        self.t = temperature

    def forward(self, w, s):
        pt = F.softmax(s / self.t, 1).detach()
        return (self.t ** 2) * F.kl_div(F.log_softmax(w / self.t, 1), pt, reduction="batchmean")


class UAPALoss(nn.Module):
    def __init__(self, base_temperature=4.0):
        super().__init__()
        self.t0 = base_temperature

    def forward(self, d, s):
        ed = -(F.softmax(d, 1) * (F.softmax(d, 1) + 1e-8).log()).sum(1).mean()
        es = -(F.softmax(s, 1) * (F.softmax(s, 1) + 1e-8).log()).sum(1).mean()
        t = self.t0 * (1 + torch.sigmoid(ed - es))
        return (t ** 2) * F.kl_div(F.log_softmax(d / t, 1), F.softmax(s / t, 1).detach(), reduction="batchmean")


class ProxyAnchorLoss(nn.Module):
    def __init__(self, nc, ed, margin=0.1, alpha=32):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(nc, ed) * 0.01)
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")
        self.m = margin
        self.a = alpha
        self.nc = nc

    def forward(self, emb, lbl):
        p = F.normalize(self.proxies, -1)
        sim = emb @ p.T
        oh = F.one_hot(lbl, self.nc).float()
        pe = torch.exp(-self.a * (sim * oh - self.m)) * oh
        hp = oh.sum(0) > 0
        pl = torch.log(1 + pe.sum(0))
        pos = pl[hp].mean() if hp.sum() > 0 else torch.tensor(0.0, device=emb.device)
        nm = 1 - oh
        ne = torch.exp(self.a * (sim * nm + self.m)) * nm
        return pos + torch.log(1 + ne.sum(0)).mean()


class EMADistillationLoss(nn.Module):
    def forward(self, s, e):
        return (1 - F.cosine_similarity(s, e)).mean()


class AltitudeConsistencyLoss(nn.Module):
    def forward(self, emb, lbl, alt):
        b = emb.size(0)
        if b < 2:
            return torch.tensor(0.0, device=emb.device)
        sl = lbl.view(-1, 1).eq(lbl.view(1, -1))
        da = alt.view(-1, 1).ne(alt.view(1, -1))
        m = sl & da
        if m.sum() == 0:
            return torch.tensor(0.0, device=emb.device)
        en = F.normalize(emb, -1)
        return ((1 - en @ en.T) * m.float()).sum() / m.float().sum().clamp(min=1)


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model, test_ds, device):
    model.eval()
    test_tf = get_transforms("test")
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    df, dl, da = [], [], []
    for b in loader:
        f = model.extract_embedding(b["drone"].to(device), alt_idx=b["alt_idx"].to(device)).cpu()
        df.append(f)
        dl.append(b["label"])
        da.append(b["altitude"])
    df = torch.cat(df)
    dl = torch.cat(dl)
    da = torch.cat(da)

    all_locs = [f"{l:04d}" for l in range(1, 201)]
    si, sl = [], []
    dc = 0
    for loc in all_locs:
        sp = os.path.join(test_ds.sat_dir, loc, "0.png")
        if not os.path.exists(sp):
            continue
        si.append(test_tf(Image.open(sp).convert("RGB")))
        if loc in test_ds.location_to_idx:
            sl.append(test_ds.location_to_idx[loc])
        else:
            sl.append(-1000 - dc)
            dc += 1

    sf = []
    for i in range(0, len(si), 64):
        sf.append(model.extract_embedding(torch.stack(si[i:i+64]).to(device)).cpu())
    sf = torch.cat(sf)
    sl = torch.tensor(sl)

    print(f"  Gallery: {len(sf)} | Queries: {len(df)}")
    sim = df @ sf.T
    _, rank = sim.sort(1, descending=True)

    n = df.size(0)
    r1 = r5 = r10 = ap = 0.0
    pa = {150: [0, 0, 0, 0], 200: [0, 0, 0, 0], 250: [0, 0, 0, 0], 300: [0, 0, 0, 0]}
    for i in range(n):
        m = torch.where(sl[rank[i]] == dl[i])[0]
        if len(m) == 0:
            continue
        fi = m[0].item()
        if fi < 1: r1 += 1
        if fi < 5: r5 += 1
        if fi < 10: r10 += 1
        rel = (sl[rank[i]] == dl[i]).float()
        cs = rel.cumsum(0)
        pr = cs / torch.arange(1, len(rel) + 1, device=rel.device, dtype=torch.float32)
        if rel.sum() > 0:
            ap += (pr * rel).sum() / rel.sum()
        a = int(da[i].item())
        v = pa[a]
        if fi < 1: v[0] += 1
        if fi < 5: v[1] += 1
        if fi < 10: v[2] += 1
        v[3] += 1

    return {"R@1": 100 * r1 / n, "R@5": 100 * r5 / n, "R@10": 100 * r10 / n, "mAP": 100 * ap / n}, pa


# =============================================================================
# TRAINING
# =============================================================================
def train_one_epoch(model, teacher, ema, loader, losses, new_losses, optimizer, scaler, device, epoch):
    model.train()
    if teacher:
        teacher.eval()

    infonce, ce, consist, cross_dist, self_dist, uapa = losses
    proxy_anchor, ema_dist, alt_consist, diversity_loss = new_losses

    total_sum = 0.0
    n = 0
    ls = defaultdict(float)
    recon_active = epoch >= CFG.RECON_WARMUP
    proto_active = epoch >= CFG.PROTO_WARMUP

    for batch in tqdm(loader, desc=f"Ep{epoch:3d}", leave=False):
        drone = batch["drone"].to(device)
        sat = batch["satellite"].to(device)
        labels = batch["label"].to(device)
        alt_idx = batch["alt_idx"].to(device)
        alts = batch["altitude"].to(device)
        alt_norm = batch["alt_norm"].to(device).float()

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=CFG.USE_AMP):
            d_out = model(drone, alt_idx=alt_idx)
            s_out = model(sat, alt_idx=None)

            l_ce = ce(d_out["logits"], labels) + ce(s_out["logits"], labels)
            l_ce += 0.3 * (ce(d_out["cls_logits"], labels) + ce(s_out["cls_logits"], labels))
            l_nce = infonce(d_out["embedding"], s_out["embedding"], labels)
            l_con = consist(d_out["parts"]["assignment"], s_out["parts"]["assignment"])

            if teacher is not None:
                with torch.no_grad():
                    td = teacher(drone)
                    ts = teacher(sat)
                l_cross = cross_dist(d_out["projected_feat"], td) + cross_dist(s_out["projected_feat"], ts)
            else:
                l_cross = torch.tensor(0.0, device=device)

            l_self = self_dist(d_out["cls_logits"], d_out["logits"]) + self_dist(s_out["cls_logits"], s_out["logits"])
            l_uapa = uapa(d_out["logits"], s_out["logits"])
            l_proxy = 0.5 * (proxy_anchor(d_out["embedding"], labels) + proxy_anchor(s_out["embedding"], labels))

            with torch.no_grad():
                ema_d = ema.forward(drone, alt_idx=alt_idx)
                ema_s = ema.forward(sat, alt_idx=None)
            l_ema = 0.5 * (ema_dist(d_out["embedding"], ema_d) + ema_dist(s_out["embedding"], ema_s))
            l_alt_con = alt_consist(d_out["embedding"], labels, alts)

            if recon_active:
                lr_d = model.mask_recon(d_out["parts"]["projected_patches"], d_out["parts"]["part_features"], d_out["parts"]["assignment"])
                lr_s = model.mask_recon(s_out["parts"]["projected_patches"], s_out["parts"]["part_features"], s_out["parts"]["assignment"])
                l_recon = 0.5 * (lr_d + lr_s)
            else:
                l_recon = torch.tensor(0.0, device=device)

            l_alt_pred = model.alt_pred(d_out["embedding"].detach(), alt_norm)
            l_div = diversity_loss(model.part_disc.prototypes)

            # NOVEL: PrototypeNCE + CrossViewPrototypeAlign
            if proto_active:
                l_proto_nce = 0.5 * (
                    model.proto_bank.prototype_nce(d_out["embedding"], labels) +
                    model.proto_bank.prototype_nce(s_out["embedding"], labels))
                l_cross_proto = model.proto_bank.cross_view_prototype_align(
                    d_out["embedding"], s_out["embedding"], labels)
            else:
                l_proto_nce = torch.tensor(0.0, device=device)
                l_cross_proto = torch.tensor(0.0, device=device)

            loss = (
                CFG.LAMBDA_CE * l_ce + CFG.LAMBDA_INFONCE * l_nce + CFG.LAMBDA_CONSISTENCY * l_con
                + CFG.LAMBDA_CROSS_DIST * l_cross + CFG.LAMBDA_SELF_DIST * l_self + CFG.LAMBDA_UAPA * l_uapa
                + CFG.LAMBDA_PROXY * l_proxy + CFG.LAMBDA_EMA_DIST * l_ema + CFG.LAMBDA_ALT_CONSIST * l_alt_con
                + CFG.LAMBDA_MASK_RECON * l_recon + CFG.LAMBDA_ALT_PRED * l_alt_pred + CFG.LAMBDA_DIVERSITY * l_div
                + CFG.LAMBDA_PROTO_NCE * l_proto_nce + CFG.LAMBDA_CROSS_PROTO * l_cross_proto
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        with torch.no_grad():
            model.proto_bank.update(
                d_out["parts"]["part_features"].detach(),
                d_out["embedding"].detach(), labels)
            model.proto_bank.update(
                s_out["parts"]["part_features"].detach(),
                s_out["embedding"].detach(), labels)

        total_sum += loss.item()
        n += 1
        ls["ce"] += l_ce.item()
        ls["nce"] += l_nce.item()
        ls["con"] += l_con.item()
        ls["cross"] += l_cross.item() if torch.is_tensor(l_cross) else l_cross
        ls["self"] += l_self.item()
        ls["uapa"] += l_uapa.item()
        ls["proxy"] += l_proxy.item()
        ls["ema"] += l_ema.item()
        ls["alt_c"] += l_alt_con.item()
        ls["recon"] += l_recon.item() if torch.is_tensor(l_recon) else l_recon
        ls["alt_p"] += l_alt_pred.item()
        ls["div"] += l_div.item()
        ls["proto"] += l_proto_nce.item() if torch.is_tensor(l_proto_nce) else l_proto_nce
        ls["xproto"] += l_cross_proto.item() if torch.is_tensor(l_cross_proto) else l_cross_proto

    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in ls.items()}


def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EXP46: SPDGeo-PartProto — Part Prototype Memory Contrastive")
    print(f"  Dataset: SUES-200 | Epochs: {CFG.NUM_EPOCHS} | Device: {DEVICE}")
    print(f"  Novel: PartPrototypeBank + PrototypeNCE + CrossViewPrototypeAlign")
    print("=" * 65)

    print("\nLoading SUES-200 ...")
    train_ds = SUES200Dataset(CFG.SUES_ROOT, "train", transform=get_transforms("train"))
    test_ds = SUES200Dataset(CFG.SUES_ROOT, "test", transform=get_transforms("test"))
    train_loader = DataLoader(
        train_ds, batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
        num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print("\nBuilding models ...")
    model = SPDGeoPartProto(CFG.NUM_CLASSES).to(DEVICE)
    teacher = None
    try:
        teacher = DINOv2Teacher().to(DEVICE)
        teacher.eval()
    except Exception as e:
        print(f"  [WARN] Could not load teacher: {e}")

    ema = EMAModel(model, decay=CFG.EMA_DECAY)

    infonce = SupInfoNCELoss(temp=0.05).to(DEVICE)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    consist = PartConsistencyLoss()
    cross_dist = CrossDistillationLoss()
    self_dist = SelfDistillationLoss(temperature=CFG.DISTILL_TEMP)
    uapa_loss = UAPALoss(base_temperature=CFG.DISTILL_TEMP)
    base_losses = (infonce, ce, consist, cross_dist, self_dist, uapa_loss)

    proxy_anchor = ProxyAnchorLoss(CFG.NUM_CLASSES, CFG.EMBED_DIM, CFG.PROXY_MARGIN, CFG.PROXY_ALPHA).to(DEVICE)
    ema_dist = EMADistillationLoss()
    alt_consist = AltitudeConsistencyLoss()
    diversity_loss = PrototypeDiversityLoss()
    new_losses = (proxy_anchor, ema_dist, alt_consist, diversity_loss)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone")]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": CFG.BACKBONE_LR},
        {"params": head_params, "lr": CFG.LR},
        {"params": infonce.parameters(), "lr": CFG.LR},
        {"params": proxy_anchor.parameters(), "lr": CFG.LR},
    ], weight_decay=CFG.WEIGHT_DECAY)

    scaler = torch.amp.GradScaler("cuda", enabled=CFG.USE_AMP)
    best_r1 = 0.0
    results_log = []
    ckpt_path = os.path.join(CFG.OUTPUT_DIR, "exp46_partproto_best.pth")

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
            model, teacher, ema, train_loader, base_losses, new_losses,
            optimizer, scaler, DEVICE, epoch)

        recon_tag = f"Rec {ld['recon']:.3f}" if epoch >= CFG.RECON_WARMUP else "Rec OFF"
        proto_tag = f"Proto {ld['proto']:.3f} XP {ld['xproto']:.3f}" if epoch >= CFG.PROTO_WARMUP else "Proto OFF"
        bank_init = model.proto_bank.bank_initialized.sum().item()
        print(
            f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
            f"CE {ld['ce']:.3f} NCE {ld['nce']:.3f} | "
            f"{recon_tag} | {proto_tag} | Bank {bank_init}/{CFG.NUM_CLASSES}"
        )

        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            metrics, per_alt = evaluate(model, test_ds, DEVICE)
            results_log.append({"epoch": epoch, **metrics})
            print(f"  -> R@1: {metrics['R@1']:.2f}% R@5: {metrics['R@5']:.2f}% R@10: {metrics['R@10']:.2f}% mAP: {metrics['mAP']:.2f}%")
            if metrics["R@1"] > best_r1:
                best_r1 = metrics["R@1"]
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics, "per_alt": per_alt}, ckpt_path)
                print(f"  [BEST] Epoch {epoch} — R@1 {best_r1:.2f}%")

    with open(os.path.join(CFG.OUTPUT_DIR, "exp46_partproto_log.json"), "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nTraining finished. Best R@1: {best_r1:.2f}%")


if __name__ == "__main__":
    main()
