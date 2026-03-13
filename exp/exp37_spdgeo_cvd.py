# EXP37: SPDGeo-CVD — Content-Viewpoint Part Disentanglement cho SUES-200 (drone ↔ satellite)
# Mô tả: Tách riêng nhân tố nội dung và góc nhìn ở mức part (K=8), dùng HSIC + head dự đoán độ cao để làm viewpoint thành yếu tố phụ, chỉ dùng CONTENT cho retrieval.

import subprocess, sys

for _p in ["timm", "tqdm"]:
    try:
        __import__(_p)
    except ImportError:
        subprocess.run(f"pip install -q {_p}", shell=True, capture_output=True)

import os
import math
import json
import gc
import random
import copy
from collections import defaultdict

import numpy as np
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
    LAMBDA_TRIPLET = 0.5
    LAMBDA_CONSISTENCY = 0.1
    LAMBDA_CROSS_DIST = 0.3
    LAMBDA_SELF_DIST = 0.3
    LAMBDA_UAPA = 0.2

    # CVD extensions
    PART_CONTENT_DIM = 192
    PART_VIEW_DIM = 64
    LAMBDA_CVD_INDEP = 0.05
    LAMBDA_ALT_VIEW = 0.10

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
            "drone": d,
            "satellite": s,
            "label": li,
            "altitude": int(alt),
            "alt_idx": alt_idx,
            "alt_norm": alt_norm,
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


def get_transforms(mode="train"):
    sz = CFG.IMG_SIZE
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
        return feat["x_norm_patchtokens"], feat["x_norm_clstoken"], (x.shape[2] // self.patch_size, x.shape[3] // self.patch_size)


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
# MODULES: PART DISCOVERY + POOLING
# =============================================================================


class SemanticPartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, part_dim),
            nn.LayerNorm(part_dim),
            nn.GELU(),
        )
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

    def forward(self, patch_features, spatial_hw):
        b, n, _ = patch_features.shape
        h, w = spatial_hw

        feat = self.feat_proj(patch_features)
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
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        alpha = torch.sigmoid(self.gate(torch.cat([part_emb, cls_emb], dim=-1)))
        return F.normalize(alpha * part_emb + (1 - alpha) * cls_emb, dim=-1)


# =============================================================================
# NEW: CVD PART DISENTANGLER
# =============================================================================


class PartContentViewpointDisentangler(nn.Module):
    def __init__(self, part_dim=256, content_dim=192, view_dim=64):
        super().__init__()
        self.content_head = nn.Linear(part_dim, content_dim)
        self.view_head = nn.Linear(part_dim, view_dim)

    def forward(self, part_features):
        """
        part_features: [B, K, D]
        returns:
            content:   [B, K, Dc]
            viewpoint: [B, K, Dv]
        """
        content = self.content_head(part_features)
        view = self.view_head(part_features)
        return content, view


class HSICIndependenceLoss(nn.Module):
    """
    Simple cross-covariance-based independence surrogate between
    content and viewpoint factors.
    """

    def forward(self, content, view):
        # content: [B, K, Dc], view: [B, K, Dv]
        b, k, dc = content.shape
        _, _, dv = view.shape
        x = content.reshape(b * k, dc)
        y = view.reshape(b * k, dv)
        x = x - x.mean(0, keepdim=True)
        y = y - y.mean(0, keepdim=True)
        cov = x.t() @ y / max(x.size(0) - 1, 1)
        return (cov ** 2).mean()


class AltitudeViewpointHead(nn.Module):
    """
    Predict altitude class from viewpoint factor.
    """

    def __init__(self, view_dim=64, num_altitudes=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(view_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_altitudes),
        )

    def forward(self, view_factors, alt_idx):
        # view_factors: [B, K, Dv], alt_idx: [B]
        b, k, d = view_factors.shape
        v = view_factors.reshape(b * k, d)
        logits = self.head(v)
        targets = alt_idx.view(-1, 1).expand(-1, k).reshape(-1)
        return F.cross_entropy(logits, targets)


# =============================================================================
# MODEL
# =============================================================================


class SPDGeoCVDModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.backbone = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = SemanticPartDiscovery(384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP)
        self.cvd = PartContentViewpointDisentangler(cfg.PART_DIM, cfg.PART_CONTENT_DIM, cfg.PART_VIEW_DIM)
        self.pool = PartAwarePooling(cfg.PART_CONTENT_DIM, cfg.EMBED_DIM)
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

        self.teacher_proj = nn.Sequential(
            nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
            nn.LayerNorm(cfg.TEACHER_DIM),
        )

        self.alt_view_head = AltitudeViewpointHead(cfg.PART_VIEW_DIM, cfg.NUM_ALTITUDES)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-CVD student: {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        content, _ = self.cvd(parts["part_features"])
        emb = self.pool(content, parts["salience"])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)

    def forward(self, x, alt_idx=None, return_parts=False):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        content, view = self.cvd(parts["part_features"])
        emb = self.pool(content, parts["salience"])

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
            "parts": parts,
            "content_parts": content,
            "view_parts": view,
        }
        if alt_idx is not None:
            out["alt_idx"] = alt_idx
        if return_parts:
            out["parts"] = parts
        return out


class EMAModel:
    def __init__(self, model, decay=0.999):
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
    def forward(self, x):
        return self.model.extract_embedding(x)


# =============================================================================
# LOSSES
# =============================================================================


class SupInfoNCELoss(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(temp).log())

    def forward(self, q_emb, r_emb, labels):
        t = self.log_t.exp().clamp(0.01, 1.0)
        sim = q_emb @ r_emb.t() / t
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float()
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        loss = -(log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
        return loss.mean()


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.m = margin

    def forward(self, emb, labels):
        d = torch.cdist(emb, emb, p=2)
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float()
        neg_mask = labels.ne(labels.T).float()
        hard_pos = (d * pos_mask).max(1)[0]
        hard_neg = (d * neg_mask + pos_mask * 1e9).min(1)[0]
        return F.relu(hard_pos - hard_neg + self.m).mean()


class PartConsistencyLoss(nn.Module):
    def forward(self, assign_q, assign_r):
        dist_q = assign_q.mean(dim=1)
        dist_r = assign_r.mean(dim=1)
        kl_qr = F.kl_div((dist_q + 1e-8).log(), dist_r, reduction="batchmean", log_target=False)
        kl_rq = F.kl_div((dist_r + 1e-8).log(), dist_q, reduction="batchmean", log_target=False)
        return 0.5 * (kl_qr + kl_rq)


class CrossDistillationLoss(nn.Module):
    def forward(self, student_feat, teacher_feat):
        s = F.normalize(student_feat, dim=-1)
        t = F.normalize(teacher_feat, dim=-1)
        mse = F.mse_loss(s, t)
        cosine = 1.0 - F.cosine_similarity(s, t).mean()
        return mse + cosine


class SelfDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super().__init__()
        self.T = temperature

    def forward(self, weak_logits, strong_logits):
        p_teacher = F.softmax(strong_logits / self.T, dim=1).detach()
        p_student = F.log_softmax(weak_logits / self.T, dim=1)
        return (self.T ** 2) * F.kl_div(p_student, p_teacher, reduction="batchmean")


class UAPALoss(nn.Module):
    def __init__(self, base_temperature=4.0):
        super().__init__()
        self.T0 = base_temperature

    @staticmethod
    def _entropy(logits):
        probs = F.softmax(logits, dim=1)
        return -(probs * (probs + 1e-8).log()).sum(dim=1).mean()

    def forward(self, drone_logits, sat_logits):
        u_drone = self._entropy(drone_logits)
        u_sat = self._entropy(sat_logits)
        delta_u = u_drone - u_sat
        t = self.T0 * (1 + torch.sigmoid(delta_u))
        p_sat = F.softmax(sat_logits / t, dim=1).detach()
        p_drone = F.log_softmax(drone_logits / t, dim=1)
        return (t ** 2) * F.kl_div(p_drone, p_sat, reduction="batchmean")


# =============================================================================
# EVALUATION
# =============================================================================


@torch.no_grad()
def evaluate(model, test_ds, device):
    model.eval()
    test_tf = get_transforms("test")

    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    drone_feats, drone_labels, drone_alts = [], [], []
    for b in loader:
        feat = model.extract_embedding(b["drone"].to(device)).cpu()
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
        sat_feats.append(model.extract_embedding(batch).cpu())
    sat_feats = torch.cat(sat_feats)
    sat_labels = torch.tensor(sat_label_list)

    print(f"  Gallery: {len(sat_feats)} satellite images | Queries: {len(drone_feats)} drone images")

    sim = drone_feats @ sat_feats.T
    _, rank = sim.sort(1, descending=True)

    n = drone_feats.size(0)
    r1 = r5 = r10 = ap = 0
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


def train_one_epoch(model, teacher, ema, loader, losses, new_losses, optimizer, scaler, device, epoch):
    model.train()
    if teacher:
        teacher.eval()

    infonce, ce, triplet, consist, cross_dist, self_dist, uapa = losses
    hsic_loss, alt_view_loss = new_losses

    total_sum = 0
    n = 0
    loss_sums = defaultdict(float)

    for batch in tqdm(loader, desc=f"Ep{epoch:3d}", leave=False):
        drone = batch["drone"].to(device)
        sat = batch["satellite"].to(device)
        labels = batch["label"].to(device)
        alt_idx = batch["alt_idx"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=CFG.USE_AMP):
            d_out = model(drone, alt_idx=alt_idx)
            s_out = model(sat, alt_idx=None)

            l_ce = ce(d_out["logits"], labels) + ce(s_out["logits"], labels)
            l_ce += 0.3 * (ce(d_out["cls_logits"], labels) + ce(s_out["cls_logits"], labels))

            l_nce = infonce(d_out["embedding"], s_out["embedding"], labels)
            l_tri = triplet(d_out["embedding"], labels)
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

            # CVD losses
            l_hsic = hsic_loss(d_out["content_parts"], d_out["view_parts"]) + hsic_loss(
                s_out["content_parts"], s_out["view_parts"]
            )
            l_altv = alt_view_loss(d_out["view_parts"], alt_idx)

            loss = (
                CFG.LAMBDA_CE * l_ce
                + CFG.LAMBDA_INFONCE * l_nce
                + CFG.LAMBDA_TRIPLET * l_tri
                + CFG.LAMBDA_CONSISTENCY * l_con
                + CFG.LAMBDA_CROSS_DIST * l_cross
                + CFG.LAMBDA_SELF_DIST * l_self
                + CFG.LAMBDA_UAPA * l_uapa
                + CFG.LAMBDA_CVD_INDEP * l_hsic
                + CFG.LAMBDA_ALT_VIEW * l_altv
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
        loss_sums["tri"] += l_tri.item()
        loss_sums["con"] += l_con.item()
        loss_sums["cross"] += l_cross.item() if torch.is_tensor(l_cross) else l_cross
        loss_sums["self"] += l_self.item()
        loss_sums["uapa"] += l_uapa.item()
        loss_sums["hsic"] += l_hsic.item()
        loss_sums["altv"] += l_altv.item()

    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}


def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EXP37: SPDGeo-CVD — Content-Viewpoint Disentanglement")
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
    model = SPDGeoCVDModel(CFG.NUM_CLASSES).to(DEVICE)
    teacher = None
    try:
        teacher = DINOv2Teacher().to(DEVICE)
        teacher.eval()
    except Exception as e:
        print(f"  [WARN] Could not load teacher: {e}")

    ema = EMAModel(model)

    infonce = SupInfoNCELoss(temp=0.05).to(DEVICE)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    triplet = TripletLoss(margin=0.3)
    consist = PartConsistencyLoss()
    cross_dist = CrossDistillationLoss()
    self_dist = SelfDistillationLoss(temperature=CFG.DISTILL_TEMP)
    uapa_loss = UAPALoss(base_temperature=CFG.DISTILL_TEMP)
    base_losses = (infonce, ce, triplet, consist, cross_dist, self_dist, uapa_loss)

    hsic_loss = HSICIndependenceLoss()
    alt_view_loss = model.alt_view_head
    new_losses = (hsic_loss, alt_view_loss)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone")]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": CFG.BACKBONE_LR},
            {"params": head_params, "lr": CFG.LR},
            {"params": infonce.parameters(), "lr": CFG.LR},
        ],
        weight_decay=CFG.WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=CFG.USE_AMP)
    best_r1 = 0.0
    results_log = []

    ckpt_path = os.path.join(CFG.OUTPUT_DIR, "exp37_spdgeo_cvd_best.pth")

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        if epoch <= CFG.WARMUP_EPOCHS:
            lr_scale = epoch / CFG.WARMUP_EPOCHS
        else:
            progress = (epoch - CFG.WARMUP_EPOCHS) / (CFG.NUM_EPOCHS - CFG.WARMUP_EPOCHS)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        lr_scale = max(lr_scale, 0.01)

        optimizer.param_groups[0]["lr"] = CFG.BACKBONE_LR * lr_scale
        for i in [1, 2]:
            optimizer.param_groups[i]["lr"] = CFG.LR * lr_scale

        avg_loss, ld = train_one_epoch(model, teacher, ema, train_loader, base_losses, new_losses, optimizer, scaler, DEVICE, epoch)

        print(
            f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
            f"CE {ld['ce']:.3f} NCE {ld['nce']:.3f} Tri {ld['tri']:.3f} "
            f"Con {ld['con']:.3f} Cross {ld['cross']:.3f} Self {ld['self']:.3f} "
            f"UAPA {ld['uapa']:.3f} HSIC {ld['hsic']:.3f} AltV {ld['altv']:.3f}"
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

    with open(os.path.join(CFG.OUTPUT_DIR, "exp37_spdgeo_cvd_log.json"), "w") as f:
        json.dump(results_log, f, indent=2)

    print("\nTraining finished.")
    print(f"Best R@1: {best_r1:.2f}% (checkpoint: {ckpt_path})")


if __name__ == "__main__":
    main()

# =============================================================================
# EXP37: SPDGeo-CVD — Content-Viewpoint Part Disentanglement
# =============================================================================
# Base:    SPDGeo-DPEA-MAR (EXP35-FM, 95.08% R@1)
# Novel:   1) PartContentViewpointSplitter — each of K=8 parts split into
#             content (192-dim) + viewpoint (64-dim) via learned linear heads
#          2) HSICIndependenceLoss — HSIC with RBF kernel minimizes statistical
#             dependence between content and viewpoint factors
#          3) AltitudeViewpointHead — viewpoint factor supervised to predict
#             altitude class (4-way CE), making viewpoint altitude-grounded
#          4) Retrieval uses CONTENT factors ONLY — viewpoint discarded at test
#
# Ref:     CVD (arxiv 2505.11822) — content-viewpoint disentanglement for DVGL
#          Our extension: part-level disentanglement + altitude supervision
#
# Total losses: 14 (12 DPEA-MAR + HSIC + AltViewpointCE)
# Expected: 95.5–96.2% R@1
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
    CONTENT_DIM     = 192
    VIEWPOINT_DIM   = 64
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
    MASK_RATIO          = 0.30
    LAMBDA_MASK_RECON   = 0.3
    RECON_WARMUP        = 10
    LAMBDA_ALT_PRED     = 0.15
    LAMBDA_DIVERSITY    = 0.05

    LAMBDA_HSIC          = 0.05
    LAMBDA_ALT_VIEWPOINT = 0.10

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
            d = Image.new('RGB', (sz, sz), (128,128,128))
            s = Image.new('RGB', (sz, sz), (128,128,128))
        if self.transform: d = self.transform(d); s = self.transform(s)
        alt_idx = CFG.ALT_TO_IDX.get(alt, 0)
        alt_norm = (int(alt) - 150) / 150.0
        return {'drone': d, 'satellite': s, 'label': li, 'altitude': int(alt),
                'alt_idx': alt_idx, 'alt_norm': alt_norm}

class PKSampler:
    def __init__(self, ds, p, k): self.ds=ds; self.p=p; self.k=k; self.locs=list(ds.drone_by_location.keys())
    def __iter__(self):
        locs=self.locs.copy(); random.shuffle(locs); batch=[]
        for l in locs:
            idx=self.ds.drone_by_location[l]
            if len(idx)<self.k: idx=idx*(self.k//len(idx)+1)
            batch.extend(random.sample(idx, self.k))
            if len(batch)>=self.p*self.k: yield batch[:self.p*self.k]; batch=batch[self.p*self.k:]
    def __len__(self): return len(self.locs)//self.p

def get_transforms(mode="train", img_size=None):
    sz = img_size or CFG.IMG_SIZE
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((sz,sz)), transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(sz, scale=(0.8,1.0)),
            transforms.ColorJitter(0.2,0.2,0.2,0.05), transforms.RandomGrayscale(p=0.02),
            transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02,0.15)),
        ])
    return transforms.Compose([transforms.Resize((sz,sz)), transforms.ToTensor(),
                                transforms.Normalize([.485,.456,.406],[.229,.224,.225])])


# =============================================================================
# BACKBONE / TEACHER
# =============================================================================
class DINOv2Backbone(nn.Module):
    def __init__(self, unfreeze_blocks=4):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.patch_size = 14
        for p in self.model.parameters(): p.requires_grad = False
        for blk in self.model.blocks[-unfreeze_blocks:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in self.model.norm.parameters(): p.requires_grad = True
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  DINOv2 ViT-S/14: {frozen/1e6:.1f}M frozen, {trainable/1e6:.1f}M trainable")

    def forward(self, x):
        feat = self.model.forward_features(x)
        return feat['x_norm_patchtokens'], feat['x_norm_clstoken'], \
               (x.shape[2]//self.patch_size, x.shape[3]//self.patch_size)

class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        print("  Loading DINOv2 ViT-B/14 teacher ...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        for p in self.model.parameters(): p.requires_grad = False
        print("  DINOv2 ViT-B/14 teacher loaded (all frozen).")
    @torch.no_grad()
    def forward(self, x): return self.model.forward_features(x)['x_norm_clstoken']


# =============================================================================
# MODULES
# =============================================================================
class DeepAltitudeFiLM(nn.Module):
    def __init__(self, num_altitudes=4, feat_dim=256):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_altitudes, feat_dim))
        self.beta  = nn.Parameter(torch.zeros(num_altitudes, feat_dim))
    def forward(self, feat, alt_idx=None):
        if alt_idx is None:
            g=self.gamma.mean(0,keepdim=True); b=self.beta.mean(0,keepdim=True)
            return feat*g.unsqueeze(0)+b.unsqueeze(0)
        return feat*self.gamma[alt_idx].unsqueeze(1)+self.beta[alt_idx].unsqueeze(1)

class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07, num_altitudes=4):
        super().__init__()
        self.temperature = temperature
        self.feat_proj = nn.Sequential(nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU())
        self.altitude_film = DeepAltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim)*0.02)
        self.refine = nn.Sequential(nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim*2), nn.GELU(), nn.Linear(part_dim*2, part_dim))
        self.salience_head = nn.Sequential(nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, patch_features, spatial_hw, alt_idx=None):
        b, n, _ = patch_features.shape; h, w = spatial_hw
        feat = self.feat_proj(patch_features)
        feat = self.altitude_film(feat, alt_idx)
        feat_norm = F.normalize(feat, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum('bnd,kd->bnk', feat_norm, proto_norm) / self.temperature
        assign = F.softmax(sim, dim=-1)
        assign_t = assign.transpose(1, 2)
        mass = assign_t.sum(-1, keepdim=True).clamp(min=1e-6)
        part_feat = torch.bmm(assign_t, feat) / mass
        part_feat = part_feat + self.refine(part_feat)
        device = feat.device
        gy = torch.arange(h, device=device).float()/max(h-1,1)
        gx = torch.arange(w, device=device).float()/max(w-1,1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(b,-1,-1)) / mass
        salience = self.salience_head(part_feat).squeeze(-1)
        return {"part_features": part_feat, "part_positions": part_pos, "assignment": assign,
                "salience": salience, "projected_patches": feat}


class PartContentViewpointSplitter(nn.Module):
    """Splits each part embedding into content (view-invariant) + viewpoint (view-specific)."""
    def __init__(self, part_dim=256, content_dim=192, viewpoint_dim=64):
        super().__init__()
        self.content_head = nn.Sequential(nn.Linear(part_dim, content_dim), nn.LayerNorm(content_dim))
        self.viewpoint_head = nn.Sequential(nn.Linear(part_dim, viewpoint_dim), nn.LayerNorm(viewpoint_dim))

    def forward(self, part_features):
        return self.content_head(part_features), self.viewpoint_head(part_features)


class PartAwarePooling(nn.Module):
    """Pools over CONTENT part features (content_dim, not part_dim)."""
    def __init__(self, input_dim=192, embed_dim=512):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(input_dim, input_dim//2), nn.Tanh(), nn.Linear(input_dim//2, 1))
        self.proj = nn.Sequential(nn.Linear(input_dim*3, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, part_features, salience=None):
        aw = self.attn(part_features)
        if salience is not None: aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        attn_pool = (aw * part_features).sum(1)
        mean_pool = part_features.mean(1)
        max_pool  = part_features.max(1)[0]
        return F.normalize(self.proj(torch.cat([attn_pool, mean_pool, max_pool], dim=-1)), dim=-1)


class DynamicFusionGate(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(embed_dim*2, embed_dim//2), nn.ReLU(True), nn.Linear(embed_dim//2, 1))
        nn.init.constant_(self.gate[-1].bias, 0.85)
    def forward(self, part_emb, cls_emb):
        a = torch.sigmoid(self.gate(torch.cat([part_emb, cls_emb], dim=-1)))
        return F.normalize(a * part_emb + (1-a) * cls_emb, dim=-1)


class MaskedPartReconstruction(nn.Module):
    """MAR operates on FULL part_features (256-dim), not content-only."""
    def __init__(self, part_dim=256, mask_ratio=0.30):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(nn.Linear(part_dim, part_dim*2), nn.GELU(), nn.Linear(part_dim*2, part_dim), nn.LayerNorm(part_dim))
    def forward(self, projected_patches, part_features, assignment):
        b, n, d = projected_patches.shape
        nm = int(n * self.mask_ratio)
        noise = torch.rand(b, n, device=projected_patches.device)
        mi = noise.argsort(1)[:, :nm]
        tgt = projected_patches.detach()
        me = mi.unsqueeze(-1).expand(-1,-1,d)
        mt = torch.gather(tgt, 1, me)
        k = part_features.shape[1]
        mk = mi.unsqueeze(-1).expand(-1,-1,k)
        ma = torch.gather(assignment, 1, mk)
        rec = self.decoder(torch.bmm(ma, part_features))
        return (1 - (F.normalize(rec,-1)*F.normalize(mt,-1)).sum(-1)).mean()


class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(True), nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, embedding, alt_target):
        return F.smooth_l1_loss(self.head(embedding).squeeze(-1), alt_target)

class PrototypeDiversityLoss(nn.Module):
    def forward(self, protos):
        p = F.normalize(protos, dim=-1); s = p@p.T; k = s.size(0)
        return (s * (1-torch.eye(k, device=s.device))).abs().sum() / (k*(k-1))


class AltitudeViewpointHead(nn.Module):
    """Predicts altitude class from pooled viewpoint factor."""
    def __init__(self, viewpoint_dim=64, num_altitudes=4):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(viewpoint_dim, 32), nn.ReLU(True), nn.Linear(32, num_altitudes))
    def forward(self, viewpoint, alt_idx):
        vp_pooled = viewpoint.mean(dim=1)
        return F.cross_entropy(self.head(vp_pooled), alt_idx)


class HSICIndependenceLoss(nn.Module):
    """HSIC with RBF kernel — minimizes dependence between content and viewpoint."""
    def forward(self, content, viewpoint):
        B, K, _ = content.shape
        c = content.reshape(B*K, -1)
        v = viewpoint.reshape(B*K, -1)
        n = c.size(0)
        if n > 512:
            idx = torch.randperm(n, device=c.device)[:512]
            c = c[idx]; v = v[idx]; n = 512

        def rbf(x):
            d = torch.cdist(x, x, p=2).pow(2)
            sig = d.median().clamp(min=1e-5)
            return torch.exp(-d / (2*sig))

        Kc = rbf(c); Kv = rbf(v)
        H = torch.eye(n, device=c.device) - 1.0/n
        return ((H@Kc@H) * (H@Kv@H)).sum() / (n-1)**2


# =============================================================================
# MODEL
# =============================================================================
class SPDGeoCVDModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = AltitudeAwarePartDiscovery(384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES)
        self.cvd_splitter = PartContentViewpointSplitter(cfg.PART_DIM, cfg.CONTENT_DIM, cfg.VIEWPOINT_DIM)
        self.pool = PartAwarePooling(cfg.CONTENT_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)
        self.bottleneck = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM), nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(True))
        self.classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM), nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM), nn.LayerNorm(cfg.TEACHER_DIM))
        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)
        self.alt_pred = AltitudePredictionHead(cfg.EMBED_DIM)
        self.alt_vp_head = AltitudeViewpointHead(cfg.VIEWPOINT_DIM, cfg.NUM_ALTITUDES)
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-CVD: {total/1e6:.1f}M trainable")

    def _encode(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        content, viewpoint = self.cvd_splitter(parts["part_features"])
        emb = self.pool(content, parts["salience"])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        fused = self.fusion_gate(emb, cls_emb)
        return fused, emb, cls_tok, cls_emb, parts, content, viewpoint

    def extract_embedding(self, x, alt_idx=None):
        fused, *_ = self._encode(x, alt_idx)
        return fused

    def forward(self, x, alt_idx=None, return_parts=False):
        fused, emb, cls_tok, cls_emb, parts, content, viewpoint = self._encode(x, alt_idx)
        bn = self.bottleneck(emb); logits = self.classifier(bn)
        cls_emb_raw = self.cls_proj(cls_tok)
        cls_logits = self.cls_classifier(cls_emb_raw)
        projected_feat = self.teacher_proj(emb)
        out = {"embedding": fused, "logits": logits, "cls_logits": cls_logits,
               "projected_feat": projected_feat, "part_emb": emb,
               "cls_emb": F.normalize(cls_emb_raw, dim=-1),
               "content": content, "viewpoint": viewpoint}
        if return_parts: out["parts"] = parts
        return out


class EMAModel:
    def __init__(self, model, decay=0.996):
        self.decay = decay; self.model = copy.deepcopy(model); self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False
    @torch.no_grad()
    def update(self, model):
        for ep, mp in zip(self.model.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(mp.data, alpha=1-self.decay)
    @torch.no_grad()
    def forward(self, x, alt_idx=None): return self.model.extract_embedding(x, alt_idx)


# =============================================================================
# LOSSES
# =============================================================================
class SupInfoNCELoss(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__(); self.log_t = nn.Parameter(torch.tensor(temp).log())
    def forward(self, q, r, labels):
        t = self.log_t.exp().clamp(0.01,1.0); sim = q@r.t()/t
        l = labels.view(-1,1); pos = l.eq(l.T).float()
        lp = sim - torch.logsumexp(sim, 1, True)
        return (-(lp*pos).sum(1)/pos.sum(1).clamp(min=1)).mean()

class PartConsistencyLoss(nn.Module):
    def forward(self, aq, ar):
        dq=aq.mean(1); dr=ar.mean(1)
        return 0.5*(F.kl_div((dq+1e-8).log(),dr,reduction='batchmean',log_target=False)+
                     F.kl_div((dr+1e-8).log(),dq,reduction='batchmean',log_target=False))

class CrossDistillationLoss(nn.Module):
    def forward(self, s, t):
        s=F.normalize(s,-1); t=F.normalize(t,-1)
        return F.mse_loss(s,t)+(1-F.cosine_similarity(s,t).mean())

class SelfDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0): super().__init__(); self.t=temperature
    def forward(self, w, s):
        p=F.softmax(s/self.t,1).detach()
        return (self.t**2)*F.kl_div(F.log_softmax(w/self.t,1),p,reduction='batchmean')

class UAPALoss(nn.Module):
    def __init__(self, bt=4.0): super().__init__(); self.t0=bt
    @staticmethod
    def _ent(l): p=F.softmax(l,1); return -(p*(p+1e-8).log()).sum(1).mean()
    def forward(self, dl, sl):
        du=self._ent(dl)-self._ent(sl); t=self.t0*(1+torch.sigmoid(du))
        p=F.softmax(sl/t,1).detach()
        return (t**2)*F.kl_div(F.log_softmax(dl/t,1),p,reduction='batchmean')

class ProxyAnchorLoss(nn.Module):
    def __init__(self, nc, ed, margin=0.1, alpha=32):
        super().__init__(); self.proxies=nn.Parameter(torch.randn(nc,ed)*0.01)
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.margin=margin; self.alpha=alpha; self.nc=nc
    def forward(self, emb, labels):
        p=F.normalize(self.proxies,-1); sim=emb@p.T; oh=F.one_hot(labels,self.nc).float()
        pe=torch.exp(-self.alpha*(sim*oh-self.margin))*oh; hp=oh.sum(0)>0
        pl=torch.log(1+pe.sum(0))[hp].mean() if hp.sum()>0 else torch.tensor(0.,device=emb.device)
        nm=1-oh; nl=torch.log(1+(torch.exp(self.alpha*(sim*nm+self.margin))*nm).sum(0)).mean()
        return pl+nl

class EMADistillationLoss(nn.Module):
    def forward(self, s, e): return (1-F.cosine_similarity(s,e)).mean()

class AltitudeConsistencyLoss(nn.Module):
    def forward(self, emb, labels, alts):
        b=emb.size(0)
        if b<2: return torch.tensor(0.,device=emb.device)
        l=labels.view(-1,1); a=alts.view(-1,1); mask=l.eq(l.T)&a.ne(a.T)
        if mask.sum()==0: return torch.tensor(0.,device=emb.device)
        en=F.normalize(emb,-1); return((1-en@en.T)*mask.float()).sum()/mask.float().sum().clamp(1)


# =============================================================================
# EVALUATE
# =============================================================================
@torch.no_grad()
def evaluate(model, test_ds, device):
    model.eval(); test_tf = get_transforms("test")
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    df, dl, da = [], [], []
    for b in loader:
        f = model.extract_embedding(b['drone'].to(device), alt_idx=b['alt_idx'].to(device)).cpu()
        df.append(f); dl.append(b['label']); da.append(b['altitude'])
    df=torch.cat(df); dl=torch.cat(dl); da=torch.cat(da)

    all_locs=[f"{l:04d}" for l in range(1,201)]; si=[]; sl=[]; dc=0
    for loc in all_locs:
        sp=os.path.join(test_ds.sat_dir, loc, "0.png")
        if not os.path.exists(sp): continue
        si.append(test_tf(Image.open(sp).convert('RGB')))
        sl.append(test_ds.location_to_idx[loc] if loc in test_ds.location_to_idx else -1000-dc)
        if loc not in test_ds.location_to_idx: dc+=1
    sf=[]
    for i in range(0,len(si),64):
        sf.append(model.extract_embedding(torch.stack(si[i:i+64]).to(device), alt_idx=None).cpu())
    sf=torch.cat(sf); sll=torch.tensor(sl)

    print(f"  Gallery: {len(sf)} sats | Queries: {len(df)} drones")
    sim=df@sf.T; _,rank=sim.sort(1,descending=True)
    n=df.size(0); r1=r5=r10=ap=0
    for i in range(n):
        m=torch.where(sll[rank[i]]==dl[i])[0]
        if len(m)==0: continue
        f=m[0].item()
        if f<1: r1+=1
        if f<5: r5+=1
        if f<10: r10+=1
        ap+=sum((j+1)/(p.item()+1) for j,p in enumerate(m))/len(m)
    overall={"R@1":r1/n*100,"R@5":r5/n*100,"R@10":r10/n*100,"mAP":ap/n*100}

    per_alt={}
    for alt in sorted(da.unique().tolist()):
        msk=da==alt
        if msk.sum()==0: continue
        af=df[msk]; al=dl[msk]; s=af@sf.T; _,rk=s.sort(1,descending=True)
        k=af.size(0); a1=a5=a10=aap=0
        for i in range(k):
            mm=torch.where(sll[rk[i]]==al[i])[0]
            if len(mm)==0: continue
            ff=mm[0].item()
            if ff<1: a1+=1
            if ff<5: a5+=1
            if ff<10: a10+=1
            aap+=sum((j+1)/(p.item()+1) for j,p in enumerate(mm))/len(mm)
        per_alt[int(alt)]={"R@1":a1/k*100,"R@5":a5/k*100,"R@10":a10/k*100,"mAP":aap/k*100,"n":k}
    return overall, per_alt


# =============================================================================
# TRAIN
# =============================================================================
def train_one_epoch(model, teacher, ema, loader, losses, new_losses, optimizer, scaler, device, epoch):
    model.train()
    if teacher: teacher.eval()
    infonce, ce, consist, cross_dist, self_dist, uapa = losses
    proxy_anchor, ema_dist, alt_consist, diversity_loss, hsic_loss = new_losses

    total_sum=0; n=0; ls=defaultdict(float)
    recon_active = epoch >= CFG.RECON_WARMUP

    for batch in tqdm(loader, desc=f"Ep{epoch:3d}", leave=False):
        drone=batch['drone'].to(device); sat=batch['satellite'].to(device)
        labels=batch['label'].to(device); alt_idx=batch['alt_idx'].to(device)
        alts=batch['altitude'].to(device); alt_norm=batch['alt_norm'].to(device).float()

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            d_out = model(drone, alt_idx=alt_idx, return_parts=True)
            s_out = model(sat, alt_idx=None, return_parts=True)

            l_ce = ce(d_out['logits'], labels)+ce(s_out['logits'], labels)+\
                   0.3*(ce(d_out['cls_logits'], labels)+ce(s_out['cls_logits'], labels))
            l_nce = infonce(d_out['embedding'], s_out['embedding'], labels)
            l_con = consist(d_out['parts']['assignment'], s_out['parts']['assignment'])

            if teacher is not None:
                with torch.no_grad(): td=teacher(drone); ts=teacher(sat)
                l_cross = cross_dist(d_out['projected_feat'], td)+cross_dist(s_out['projected_feat'], ts)
            else: l_cross = torch.tensor(0.,device=device)

            l_self = self_dist(d_out['cls_logits'], d_out['logits'])+self_dist(s_out['cls_logits'], s_out['logits'])
            l_uapa = uapa(d_out['logits'], s_out['logits'])
            l_proxy = 0.5*(proxy_anchor(d_out['embedding'], labels)+proxy_anchor(s_out['embedding'], labels))

            with torch.no_grad():
                ed=ema.forward(drone, alt_idx); es=ema.forward(sat, None)
            l_ema = 0.5*(ema_dist(d_out['embedding'], ed)+ema_dist(s_out['embedding'], es))
            l_alt_c = alt_consist(d_out['embedding'], labels, alts)

            if recon_active:
                l_rec = 0.5*(model.mask_recon(d_out['parts']['projected_patches'], d_out['parts']['part_features'], d_out['parts']['assignment'])+
                             model.mask_recon(s_out['parts']['projected_patches'], s_out['parts']['part_features'], s_out['parts']['assignment']))
            else: l_rec = torch.tensor(0.,device=device)

            l_alt_p = model.alt_pred(d_out['embedding'].detach(), alt_norm)
            l_div = diversity_loss(model.part_disc.prototypes)

            l_hsic = 0.5*(hsic_loss(d_out['content'], d_out['viewpoint'])+
                          hsic_loss(s_out['content'], s_out['viewpoint']))
            l_alt_vp = model.alt_vp_head(d_out['viewpoint'], alt_idx)

            loss = (CFG.LAMBDA_CE*l_ce + CFG.LAMBDA_INFONCE*l_nce + CFG.LAMBDA_CONSISTENCY*l_con +
                    CFG.LAMBDA_CROSS_DIST*l_cross + CFG.LAMBDA_SELF_DIST*l_self + CFG.LAMBDA_UAPA*l_uapa +
                    CFG.LAMBDA_PROXY*l_proxy + CFG.LAMBDA_EMA_DIST*l_ema + CFG.LAMBDA_ALT_CONSIST*l_alt_c +
                    CFG.LAMBDA_MASK_RECON*l_rec + CFG.LAMBDA_ALT_PRED*l_alt_p + CFG.LAMBDA_DIVERSITY*l_div +
                    CFG.LAMBDA_HSIC*l_hsic + CFG.LAMBDA_ALT_VIEWPOINT*l_alt_vp)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()
        ema.update(model)

        total_sum+=loss.item(); n+=1
        for k2,v2 in [("ce",l_ce),("nce",l_nce),("con",l_con),("cross",l_cross),("self",l_self),
                       ("uapa",l_uapa),("proxy",l_proxy),("ema",l_ema),("alt_c",l_alt_c),
                       ("recon",l_rec),("alt_p",l_alt_p),("div",l_div),("hsic",l_hsic),("alt_vp",l_alt_vp)]:
            ls[k2] += (v2.item() if torch.is_tensor(v2) else v2)

    return total_sum/max(n,1), {k:v/max(n,1) for k,v in ls.items()}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED); os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    print("="*65)
    print("  EXP37: SPDGeo-CVD — Content-Viewpoint Part Disentanglement")
    print(f"  Content: {CFG.CONTENT_DIM}d | Viewpoint: {CFG.VIEWPOINT_DIM}d | Parts: {CFG.N_PARTS}")
    print(f"  HSIC λ={CFG.LAMBDA_HSIC} | AltVP λ={CFG.LAMBDA_ALT_VIEWPOINT}")
    print(f"  Epochs: {CFG.NUM_EPOCHS} | Device: {DEVICE}")
    print("="*65)

    train_ds = SUES200Dataset(CFG.SUES_ROOT, "train", transform=get_transforms("train"))
    test_ds  = SUES200Dataset(CFG.SUES_ROOT, "test",  transform=get_transforms("test"))
    train_loader = DataLoader(train_ds, batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    model = SPDGeoCVDModel(CFG.NUM_CLASSES).to(DEVICE)
    teacher = None
    try: teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except Exception as e: print(f"  [WARN] Teacher: {e}")
    ema = EMAModel(model, CFG.EMA_DECAY)

    infonce=SupInfoNCELoss(0.05).to(DEVICE); ce=nn.CrossEntropyLoss(label_smoothing=0.1)
    consist=PartConsistencyLoss(); cross_dist=CrossDistillationLoss()
    self_dist=SelfDistillationLoss(CFG.DISTILL_TEMP); uapa_loss=UAPALoss(CFG.DISTILL_TEMP)
    base_losses=(infonce, ce, consist, cross_dist, self_dist, uapa_loss)

    proxy_anchor=ProxyAnchorLoss(CFG.NUM_CLASSES, CFG.EMBED_DIM, CFG.PROXY_MARGIN, CFG.PROXY_ALPHA).to(DEVICE)
    ema_distl=EMADistillationLoss(); alt_consist=AltitudeConsistencyLoss()
    div_loss=PrototypeDiversityLoss(); hsic=HSICIndependenceLoss()
    new_losses=(proxy_anchor, ema_distl, alt_consist, div_loss, hsic)

    bb_p=[p for p in model.backbone.parameters() if p.requires_grad]
    hd_p=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('backbone')]
    optimizer=torch.optim.AdamW([
        {"params": bb_p, "lr": CFG.BACKBONE_LR},
        {"params": hd_p, "lr": CFG.LR},
        {"params": infonce.parameters(), "lr": CFG.LR},
        {"params": proxy_anchor.parameters(), "lr": CFG.LR},
    ], weight_decay=CFG.WEIGHT_DECAY)

    scaler=torch.amp.GradScaler('cuda', enabled=CFG.USE_AMP)
    best_r1=0.; results_log=[]
    ckpt_path=os.path.join(CFG.OUTPUT_DIR, "exp37_cvd_best.pth")

    for epoch in range(1, CFG.NUM_EPOCHS+1):
        if epoch<=CFG.WARMUP_EPOCHS: lr_s=epoch/CFG.WARMUP_EPOCHS
        else:
            prog=(epoch-CFG.WARMUP_EPOCHS)/(CFG.NUM_EPOCHS-CFG.WARMUP_EPOCHS)
            lr_s=0.5*(1+math.cos(math.pi*prog))
        lr_s=max(lr_s, 0.01)
        optimizer.param_groups[0]['lr']=CFG.BACKBONE_LR*lr_s
        for i in [1,2,3]: optimizer.param_groups[i]['lr']=CFG.LR*lr_s

        avg_loss, ld = train_one_epoch(model, teacher, ema, train_loader, base_losses, new_losses,
                                       optimizer, scaler, DEVICE, epoch)
        rt = f"Rec {ld['recon']:.3f}" if epoch>=CFG.RECON_WARMUP else "Rec OFF"
        print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
              f"CE {ld['ce']:.3f} NCE {ld['nce']:.3f} {rt} | "
              f"HSIC {ld['hsic']:.4f} AltVP {ld['alt_vp']:.3f}")

        if epoch%CFG.EVAL_INTERVAL==0 or epoch==CFG.NUM_EPOCHS:
            metrics, per_alt = evaluate(model, test_ds, DEVICE)
            results_log.append({"epoch": epoch, **metrics})
            print(f"  -> R@1: {metrics['R@1']:.2f}% R@5: {metrics['R@5']:.2f}% "
                  f"R@10: {metrics['R@10']:.2f}% mAP: {metrics['mAP']:.2f}%")
            if metrics['R@1']>best_r1:
                best_r1=metrics['R@1']
                torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),
                            "metrics":metrics,"per_alt":per_alt}, ckpt_path)
                print(f"  * New best R@1: {best_r1:.2f}%")
            ema_m, _=evaluate(ema.model, test_ds, DEVICE)
            print(f"  -> EMA R@1: {ema_m['R@1']:.2f}%")
            if ema_m['R@1']>best_r1:
                best_r1=ema_m['R@1']
                torch.save({"epoch":epoch,"model_state_dict":ema.model.state_dict(),
                            "metrics":ema_m,"is_ema":True}, ckpt_path)

    print(f"\n{'='*65}\n  EXP37 COMPLETE — Best R@1: {best_r1:.2f}%\n{'='*65}")
    with open(os.path.join(CFG.OUTPUT_DIR, "exp37_cvd_results.json"), "w") as f:
        json.dump({"results_log":results_log,"best_r1":best_r1,
                   "config":{k:v for k,v in vars(CFG).items() if not k.startswith('_')}}, f, indent=2)

if __name__ == "__main__":
    main()
