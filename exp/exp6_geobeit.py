"""
EXP6: GeoBEiT — BEiTv2-Large Teacher + RKD + CVRC + Progressive Temperature
=============================================================================
Teacher:  BEiTv2-Large (304M params, 1024-dim) via timm
Student:  ConvNeXt-Tiny (same as baseline)
Novel:    1) Relational Knowledge Distillation (RKD) — preserve pairwise geometry
          2) Cross-View Relational Consistency (CVRC) — teacher guides cross-view similarity
          3) Progressive Temperature Annealing — T: 20→2 cosine decay

Expected: 80-85% R@1 on SUES-200 benchmark
"""
import subprocess, importlib

def pip_install(pkg, extra=""):
    subprocess.run(f"pip install -q {extra} {pkg}", shell=True, capture_output=True, text=True)

print("[1/2] Installing packages...")
for p in ["timm", "tqdm"]:
    try: importlib.import_module(p)
    except ImportError: pip_install(p)
print("[2/2] Setup complete!")

# ============================================================================
# IMPORTS
# ============================================================================
import os, math, random, argparse

import numpy as np
from typing import Dict, Any
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    DATA_ROOT = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    DRONE_DIR = "drone-view"
    SATELLITE_DIR = "satellite-view"
    OUTPUT_DIR = "/kaggle/working"

    NUM_WORKERS = 8
    P = 8; K = 4
    BATCH_SIZE = 256
    NUM_EPOCHS = 120
    LR = 0.001
    WARMUP_EPOCHS = 5

    IMG_SIZE = 224
    NUM_CLASSES = 120
    EMBED_DIM = 768
    DROP_PATH_RATE = 0.1

    # Distillation — progressive temperature
    T_MAX = 20.0   # high T → soft targets early
    T_MIN = 2.0    # low T → hard targets late
    TEMPERATURE = 4.0       # initial (overridden by progressive)
    BASE_TEMPERATURE = 4.0

    # Loss weights
    LAMBDA_TRIPLET = 1.0
    LAMBDA_CSC = 0.5
    LAMBDA_SELF_DIST = 0.5
    LAMBDA_CROSS_DIST = 0.3
    LAMBDA_ALIGN = 0.2
    LAMBDA_RKD = 0.3        # NEW — Relational KD
    LAMBDA_CVRC = 0.2       # NEW — Cross-View Relational Consistency
    MARGIN = 0.3

    ALTITUDES = ["150", "200", "250", "300"]
    TRAIN_LOCS = list(range(1, 121))
    TEST_LOCS  = list(range(121, 201))

    USE_AMP = True
    SEED = 42


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def get_temperature(epoch, total_epochs, t_max=Config.T_MAX, t_min=Config.T_MIN):
    """Progressive temperature: cosine decay from t_max to t_min"""
    progress = epoch / max(total_epochs - 1, 1)
    return t_min + 0.5 * (t_max - t_min) * (1 + math.cos(math.pi * progress))


# ============================================================================
# DATASET
# ============================================================================
class SUES200Dataset(Dataset):
    def __init__(self, root, mode="train", altitudes=None, transform=None,
                 train_locs=None, test_locs=None):
        self.root = root; self.mode = mode
        self.altitudes = altitudes or Config.ALTITUDES
        self.transform = transform
        self.drone_dir = os.path.join(root, Config.DRONE_DIR)
        self.satellite_dir = os.path.join(root, Config.SATELLITE_DIR)
        if train_locs is None: train_locs = Config.TRAIN_LOCS
        if test_locs is None: test_locs = Config.TEST_LOCS
        loc_ids = train_locs if mode == "train" else test_locs
        self.locations = [f"{loc:04d}" for loc in loc_ids]
        self.location_to_idx = {loc: idx for idx, loc in enumerate(self.locations)}
        self.samples = []; self.drone_by_location = defaultdict(list)
        for loc in self.locations:
            loc_idx = self.location_to_idx[loc]
            sat_path = os.path.join(self.satellite_dir, loc, "0.png")
            if not os.path.exists(sat_path): continue
            for alt in self.altitudes:
                alt_dir = os.path.join(self.drone_dir, loc, alt)
                if not os.path.isdir(alt_dir): continue
                for img_name in sorted(os.listdir(alt_dir)):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        drone_path = os.path.join(alt_dir, img_name)
                        self.samples.append((drone_path, sat_path, loc_idx, alt))
                        self.drone_by_location[loc_idx].append(len(self.samples) - 1)
        print(f"[{mode}] Loaded {len(self.samples)} samples from {len(self.locations)} locations")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        drone_path, sat_path, loc_idx, altitude = self.samples[idx]
        drone_img = Image.open(drone_path).convert('RGB')
        sat_img = Image.open(sat_path).convert('RGB')
        if self.transform:
            drone_img = self.transform(drone_img)
            sat_img = self.transform(sat_img)
        return {'drone': drone_img, 'satellite': sat_img, 'label': loc_idx, 'altitude': int(altitude)}


class PKSampler:
    def __init__(self, dataset, p=8, k=4):
        self.dataset = dataset; self.p = p; self.k = k
        self.locations = list(dataset.drone_by_location.keys())
    def __iter__(self):
        locations = self.locations.copy(); random.shuffle(locations)
        batch = []
        for loc in locations:
            indices = self.dataset.drone_by_location[loc]
            if len(indices) < self.k:
                indices = indices * (self.k // len(indices) + 1)
            batch.extend(random.sample(indices, self.k))
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]
                batch = batch[self.p * self.k:]
    def __len__(self): return len(self.locations) // self.p


def get_transforms(mode="train"):
    if mode == "train":
        return T.Compose([
            T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return T.Compose([
        T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ============================================================================
# CONVNEXT-TINY BACKBONE (same as baseline)
# ============================================================================
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps; self.data_format = data_format
        self.normalized_shape = (normalized_shape,)
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True); s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None): super().__init__(); self.drop_prob = drop_prob
    def forward(self, x): return drop_path(x, self.drop_prob, self.training)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path_rate=0., layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
    def forward(self, x):
        shortcut = x; x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1); x = self.norm(x)
        x = self.pwconv1(x); x = self.act(x); x = self.pwconv2(x)
        if self.gamma is not None: x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXtTiny(nn.Module):
    def __init__(self, in_chans=3, depths=[3,3,9,3], dims=[96,192,384,768],
                 drop_path_rate=0., layer_scale_init=1e-6):
        super().__init__()
        self.num_stages = 4; self.dims = dims
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                       nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0; self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[ConvNeXtBlock(dims[i], dp_rates[cur+j], layer_scale_init) for j in range(depths[i])])
            self.stages.append(stage); cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward_features(self, x):
        stage_outputs = []
        for i in range(4):
            x = self.downsample_layers[i](x); x = self.stages[i](x)
            stage_outputs.append(x)
        final_feat = x.mean([-2, -1]); final_feat = self.norm(final_feat)
        return final_feat, stage_outputs
    def forward(self, x): return self.forward_features(x)


def load_convnext_pretrained(model):
    url = "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth"
    try:
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)
        state_dict = {k: v for k, v in checkpoint["model"].items() if not k.startswith('head')}
        model.load_state_dict(state_dict, strict=False)
        print("Loaded ConvNeXt-Tiny pretrained weights from ImageNet-22K")
    except Exception as e: print(f"Could not load pretrained weights: {e}")
    return model


# ============================================================================
# STUDENT MODEL (same as baseline)
# ============================================================================
class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(hidden_dim, num_classes))
    def forward(self, x): return self.fc(self.pool(x).flatten(1))

class GeneralizedMeanPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.ones(1)*p); self.eps = eps
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0/self.p)


class MobileGeoStudent(nn.Module):
    def __init__(self, num_classes, embed_dim=768, drop_path_rate=0.1):
        super().__init__()
        self.backbone = load_convnext_pretrained(ConvNeXtTiny(drop_path_rate=drop_path_rate))
        self.dims = [96, 192, 384, 768]; self.embed_dim = embed_dim
        self.aux_heads = nn.ModuleList([ClassificationHead(dim, num_classes) for dim in self.dims])
        self.bottleneck = nn.Sequential(nn.Linear(768, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.gem = GeneralizedMeanPooling()

    def forward(self, x, return_all=False):
        final_feat, stage_outputs = self.backbone(x)
        stage_logits = [head(feat) for head, feat in zip(self.aux_heads, stage_outputs)]
        embedding = self.bottleneck(final_feat)
        embedding_normed = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)
        if return_all:
            return {'embedding': embedding, 'embedding_normed': embedding_normed,
                    'logits': logits, 'stage_logits': stage_logits,
                    'stage_features': stage_outputs, 'final_feature': final_feat}
        return embedding_normed, logits


# ============================================================================
# BEITV2-LARGE TEACHER (replaces DINOv2)
# ============================================================================
class BEiT3Teacher(nn.Module):
    """BEiTv2-Large Teacher Model (304M params, 1024-dim output)"""

    def __init__(self, num_trainable_blocks=2):
        super().__init__()
        print("Loading BEiTv2-Large teacher model via timm...")
        self.model = timm.create_model('beitv2_large_patch16_224', pretrained=True, num_classes=0)
        self.num_channels = 1024
        self.num_trainable_blocks = num_trainable_blocks

        # Freeze all
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze last N blocks
        if hasattr(self.model, 'blocks'):
            for blk in self.model.blocks[-num_trainable_blocks:]:
                for param in blk.parameters():
                    param.requires_grad = True

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"BEiTv2-Large loaded. {total/1e6:.1f}M params, {trainable/1e6:.1f}M trainable (last {num_trainable_blocks} blocks).")

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)  # [B, 1024]


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3): super().__init__(); self.margin = margin
    def forward(self, embeddings, labels):
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        N = embeddings.size(0); labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.T).float(); mask_neg = labels.ne(labels.T).float()
        hard_pos = (dist_mat * mask_pos).max(dim=1)[0]
        hard_neg = (dist_mat * mask_neg + mask_pos * 1e9).min(dim=1)[0]
        return F.relu(hard_pos - hard_neg + self.margin).mean()


class SymmetricInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07): super().__init__(); self.temperature = temperature
    def forward(self, drone_feats, sat_feats, labels):
        drone_feats = F.normalize(drone_feats, dim=1); sat_feats = F.normalize(sat_feats, dim=1)
        sim_d2s = torch.mm(drone_feats, sat_feats.T) / self.temperature
        sim_s2d = sim_d2s.T
        labels = labels.view(-1, 1); pos_mask = labels.eq(labels.T).float()
        loss_d2s = -torch.log((F.softmax(sim_d2s, dim=1) * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)).mean()
        loss_s2d = -torch.log((F.softmax(sim_s2d, dim=1) * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)).mean()
        return 0.5 * (loss_d2s + loss_s2d)


class SelfDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, weights=[0.1, 0.2, 0.3, 0.4]):
        super().__init__(); self.temperature = temperature; self.weights = weights
    def forward(self, stage_logits, temperature=None):
        T = temperature or self.temperature
        loss = 0.0; final_logits = stage_logits[-1]
        for i in range(len(stage_logits) - 1):
            p_teacher = F.softmax(stage_logits[i] / T, dim=1)
            p_student = F.log_softmax(final_logits / T, dim=1)
            loss += self.weights[i] * (T ** 2) * F.kl_div(p_student, p_teacher, reduction='batchmean')
        return loss


class UAPALoss(nn.Module):
    def __init__(self, base_temperature=4.0): super().__init__(); self.T0 = base_temperature
    def compute_uncertainty(self, logits):
        probs = F.softmax(logits, dim=1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
    def forward(self, drone_logits, sat_logits, base_temperature=None):
        T0 = base_temperature or self.T0
        U_drone = self.compute_uncertainty(drone_logits)
        U_sat = self.compute_uncertainty(sat_logits)
        delta_U = U_drone - U_sat
        T = T0 * (1 + torch.sigmoid(delta_U))
        p_sat = F.softmax(sat_logits / T, dim=1)
        p_drone = F.log_softmax(drone_logits / T, dim=1)
        return (T ** 2) * F.kl_div(p_drone, p_sat, reduction='batchmean')


class CrossDistillationLoss(nn.Module):
    """Cross-Distillation with projection for dim mismatch (student→teacher dim)"""
    def __init__(self, temperature=4.0, student_dim=768, teacher_dim=1024):
        super().__init__()
        self.temperature = temperature
        self.proj = nn.Linear(student_dim, teacher_dim) if student_dim != teacher_dim else nn.Identity()

    def forward(self, student_feat, teacher_feat):
        student_proj = self.proj(student_feat)
        student_proj = F.normalize(student_proj, dim=1)
        teacher_feat = F.normalize(teacher_feat, dim=1)
        mse_loss = F.mse_loss(student_proj, teacher_feat)
        cosine_loss = 1 - F.cosine_similarity(student_proj, teacher_feat).mean()
        return mse_loss + cosine_loss


# ========================== NEW LOSS: RKD ==========================
class RKDLoss(nn.Module):
    """Relational Knowledge Distillation (Park et al., CVPR 2019)
    Transfers pairwise distance and angle relationships from teacher to student.
    """
    def __init__(self, w_dist=1.0, w_angle=2.0):
        super().__init__(); self.w_dist = w_dist; self.w_angle = w_angle

    def pdist(self, e):
        e_sq = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_sq.unsqueeze(1) + e_sq.unsqueeze(0) - 2 * prod).clamp(min=1e-12).sqrt()
        return res

    def forward(self, student, teacher):
        # ---- Distance-wise RKD ----
        with torch.no_grad():
            t_d = self.pdist(teacher)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / (mean_td + 1e-8)

        s_d = self.pdist(student)
        mean_sd = s_d[s_d > 0].mean()
        s_d = s_d / (mean_sd + 1e-8)
        loss_d = F.smooth_l1_loss(s_d, t_d)

        # ---- Angle-wise RKD ----
        with torch.no_grad():
            td = teacher.unsqueeze(0) - teacher.unsqueeze(1)  # [N,N,D]
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = student.unsqueeze(0) - student.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        return self.w_dist * loss_d + self.w_angle * loss_a


# ========================== NEW LOSS: CVRC ==========================
class CVRCLoss(nn.Module):
    """Cross-View Relational Consistency Loss
    Forces student's cross-view similarity structure to match teacher's.
    """
    def __init__(self, temperature=0.1):
        super().__init__(); self.temperature = temperature

    def forward(self, student_drone, student_sat, teacher_drone, teacher_sat):
        with torch.no_grad():
            t_drone = F.normalize(teacher_drone, dim=1)
            t_sat = F.normalize(teacher_sat, dim=1)
            t_sim = t_drone @ t_sat.T / self.temperature
            t_prob = F.softmax(t_sim, dim=1)

        s_drone = F.normalize(student_drone, dim=1)
        s_sat = F.normalize(student_sat, dim=1)
        s_sim = s_drone @ s_sat.T / self.temperature
        s_log_prob = F.log_softmax(s_sim, dim=1)

        return F.kl_div(s_log_prob, t_prob, reduction='batchmean')


# ============================================================================
# COMBINED LOSS
# ============================================================================
class MobileGeoLoss(nn.Module):
    def __init__(self, num_classes, cfg=Config):
        super().__init__()
        self.cfg = cfg
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=cfg.MARGIN)
        self.csc_loss = SymmetricInfoNCELoss()
        self.self_dist_loss = SelfDistillationLoss(temperature=cfg.TEMPERATURE)
        self.uapa_loss = UAPALoss(base_temperature=cfg.BASE_TEMPERATURE)
        self.cross_dist_loss = CrossDistillationLoss(
            temperature=cfg.TEMPERATURE, student_dim=768, teacher_dim=1024)
        self.rkd_loss = RKDLoss()       # NEW
        self.cvrc_loss = CVRCLoss()     # NEW

    def forward(self, drone_out, sat_out, labels,
                teacher_drone_feat=None, teacher_sat_feat=None, temperature=None):
        losses: Dict[str, Any] = {}
        T = temperature or self.cfg.TEMPERATURE

        # CE loss (all stages + final)
        ce_loss = 0.0
        for logits in drone_out['stage_logits']: ce_loss += 0.25 * self.ce_loss(logits, labels)
        ce_loss += self.ce_loss(drone_out['logits'], labels)
        for logits in sat_out['stage_logits']: ce_loss += 0.25 * self.ce_loss(logits, labels)
        ce_loss += self.ce_loss(sat_out['logits'], labels)
        losses['ce'] = ce_loss

        # Triplet
        losses['triplet'] = self.cfg.LAMBDA_TRIPLET * (
            self.triplet_loss(drone_out['embedding_normed'], labels) +
            self.triplet_loss(sat_out['embedding_normed'], labels))

        # Cross-view Symmetric Contrastive
        losses['csc'] = self.cfg.LAMBDA_CSC * self.csc_loss(
            drone_out['embedding_normed'], sat_out['embedding_normed'], labels)

        # Self-distillation (progressive T)
        losses['self_dist'] = self.cfg.LAMBDA_SELF_DIST * (
            self.self_dist_loss(drone_out['stage_logits'], temperature=T) +
            self.self_dist_loss(sat_out['stage_logits'], temperature=T))

        # UAPA (progressive T)
        losses['uapa'] = self.cfg.LAMBDA_ALIGN * self.uapa_loss(
            drone_out['logits'], sat_out['logits'], base_temperature=T)

        # Cross-distillation + RKD + CVRC
        if teacher_drone_feat is not None:
            losses['cross_dist'] = self.cfg.LAMBDA_CROSS_DIST * (
                self.cross_dist_loss(drone_out['final_feature'], teacher_drone_feat) +
                self.cross_dist_loss(sat_out['final_feature'], teacher_sat_feat))

            # RKD — relational knowledge distillation
            losses['rkd'] = self.cfg.LAMBDA_RKD * (
                self.rkd_loss(drone_out['embedding_normed'], F.normalize(teacher_drone_feat, dim=1)) +
                self.rkd_loss(sat_out['embedding_normed'], F.normalize(teacher_sat_feat, dim=1)))

            # CVRC — cross-view relational consistency
            losses['cvrc'] = self.cfg.LAMBDA_CVRC * self.cvrc_loss(
                drone_out['embedding_normed'], sat_out['embedding_normed'],
                teacher_drone_feat, teacher_sat_feat)

        total_loss = sum(losses.values())
        losses['total'] = total_loss
        return total_loss, losses


# ============================================================================
# EVALUATION
# ============================================================================
# ============================================================================
# TRAINING
# ============================================================================

# ============================================================================
# EVALUATION — Per-altitude R@1/R@5/R@10/mAP (paper-grade, standalone)
# ============================================================================
def compute_metrics(query_feats, gallery_feats, query_labels, gallery_labels):
    """Compute Recall@K and mAP."""
    sim_matrix = torch.mm(query_feats, gallery_feats.T)
    _, indices = sim_matrix.sort(dim=1, descending=True)
    N = query_feats.size(0)
    r1 = r5 = r10 = ap_sum = 0
    for i in range(N):
        ql = query_labels[i]
        ranked = gallery_labels[indices[i]]
        correct = torch.where(ranked == ql)[0]
        if len(correct) == 0: continue
        fc = correct[0].item()
        if fc < 1: r1 += 1
        if fc < 5: r5 += 1
        if fc < 10: r10 += 1
        ps = sum((j+1)/(p.item()+1) for j,p in enumerate(correct))
        ap_sum += ps / len(correct)
    return {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100}, ap_sum/N*100


def evaluate(model, test_dataset, device, cfg=Config):
    """Full SUES-200 evaluation: 200-image gallery + per-altitude breakdown."""
    model.eval()
    tl = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                    num_workers=cfg.NUM_WORKERS, pin_memory=True)

    # Extract drone query features + altitudes
    all_feats, all_labels, all_alts = [], [], []
    with torch.no_grad():
        for b in tl:
            f, _ = model(b['drone'].to(device))
            all_feats.append(f.cpu())
            all_labels.append(b['label'])
            all_alts.append(b['altitude'])
    all_feats = torch.cat(all_feats)
    all_labels = torch.cat(all_labels)
    all_alts = torch.cat(all_alts)

    # Build FULL satellite gallery (ALL 200 locations = confusion data)
    tr = T.Compose([T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)), T.ToTensor(),
                     T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    root = test_dataset.root
    sat_dir = os.path.join(root, cfg.SATELLITE_DIR)
    all_loc_ids = cfg.TRAIN_LOCS + cfg.TEST_LOCS
    sf, sl, gn = [], [], []
    for loc in [f"{l:04d}" for l in all_loc_ids]:
        sp = os.path.join(sat_dir, loc, "0.png")
        if os.path.exists(sp):
            with torch.no_grad():
                f, _ = model(tr(Image.open(sp).convert('RGB')).unsqueeze(0).to(device))
            sf.append(f.cpu())
            sl.append(test_dataset.location_to_idx[loc] if loc in test_dataset.location_to_idx else -1-len(gn))
            gn.append(loc)
    sf = torch.cat(sf); sl = torch.tensor(sl)

    # Overall metrics
    overall_r, overall_ap = compute_metrics(all_feats, sf, all_labels, sl)

    # Per-altitude metrics
    altitudes = sorted(all_alts.unique().tolist())
    per_alt = {}
    for alt in altitudes:
        mask = all_alts == alt
        if mask.sum() == 0: continue
        ar, aap = compute_metrics(all_feats[mask], sf, all_labels[mask], sl)
        per_alt[int(alt)] = {'R@1': ar['R@1'], 'R@5': ar['R@5'], 'R@10': ar['R@10'],
                             'mAP': aap, 'n': int(mask.sum())}

    # Print results
    print(f"\n{'='*75}")
    print(f"  Gallery: {len(sf)} satellite images | Queries: {len(all_feats)} drone images")
    print(f"{'='*75}")
    print(f"  {'Altitude':>8s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'#Query':>6s}")
    print(f"  {'-'*50}")
    for alt in altitudes:
        a = per_alt[int(alt)]
        print(f"  {int(alt):>6d}m  {a['R@1']:6.2f}%  {a['R@5']:6.2f}%  {a['R@10']:6.2f}%  {a['mAP']:6.2f}%  {a['n']:>6d}")
    print(f"  {'-'*50}")
    print(f"  {'Overall':>8s}  {overall_r['R@1']:6.2f}%  {overall_r['R@5']:6.2f}%  {overall_r['R@10']:6.2f}%  {overall_ap:6.2f}%  {len(all_feats):>6d}")
    print(f"{'='*75}\n")

    return overall_r, overall_ap, per_alt

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer; self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs; self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups: pg['lr'] = lr
        return lr


def train_one_epoch(model, teacher, train_loader, criterion, optimizer,
                    scaler, device, epoch, cfg=Config):
    model.train()
    if teacher is not None: teacher.eval()
    total_loss = 0; loss_dict_sum = defaultdict(float)
    current_T = get_temperature(epoch, cfg.NUM_EPOCHS)

    for batch_idx, batch in enumerate(train_loader):
        drone_imgs = batch['drone'].to(device)
        sat_imgs = batch['satellite'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=cfg.USE_AMP):
            drone_out = model(drone_imgs, return_all=True)
            sat_out = model(sat_imgs, return_all=True)
            teacher_drone_feat = teacher_sat_feat = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_drone_feat = teacher(drone_imgs)
                    teacher_sat_feat = teacher(sat_imgs)
            loss, loss_dict = criterion(
                drone_out, sat_out, labels,
                teacher_drone_feat, teacher_sat_feat,
                temperature=current_T)

        if cfg.USE_AMP:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_dict_sum[k] += v.item() if torch.is_tensor(v) else v
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, T: {current_T:.1f}")

    if len(train_loader) == 0: return 0.0, {}
    avg_loss = total_loss / len(train_loader)
    return avg_loss, {k: v/len(train_loader) for k, v in loss_dict_sum.items()}


def main():
    parser = argparse.ArgumentParser(description="EXP6: GeoBEiT Training")
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--data_root", type=str, default=Config.DATA_ROOT)
    parser.add_argument("--test", action="store_true")
    args, _ = parser.parse_known_args()
    Config.NUM_EPOCHS = args.epochs; Config.BATCH_SIZE = args.batch_size
    Config.DATA_ROOT = args.data_root
    if args.test:
        print(">>> SMOKE TEST <<<")
        Config.NUM_EPOCHS = 1; Config.NUM_WORKERS = 0; Config.BATCH_SIZE = 8; Config.P = 2
    Config.K = max(2, Config.BATCH_SIZE // Config.P)

    print("=" * 60)
    print("EXP6: GeoBEiT — BEiTv2-Large + RKD + CVRC + ProgTemp")
    print("=" * 60)

    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = SUES200Dataset(Config.DATA_ROOT, mode="train", transform=get_transforms("train"))
    test_dataset = SUES200Dataset(Config.DATA_ROOT, mode="test", transform=get_transforms("test"))
    num_classes = len(Config.TRAIN_LOCS)

    train_sampler = PKSampler(train_dataset, p=Config.P, k=Config.K)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=Config.NUM_WORKERS, pin_memory=True)

    print("\nBuilding models...")
    model = MobileGeoStudent(num_classes=num_classes, embed_dim=Config.EMBED_DIM).to(device)

    try:
        teacher = BEiT3Teacher(num_trainable_blocks=2).to(device)
    except Exception as e:
        print(f"Could not load BEiTv2-Large teacher: {e}")
        print("Training without cross-distillation")
        teacher = None

    criterion = MobileGeoLoss(num_classes=num_classes)
    criterion = criterion.to(device)   # Move projection layers to GPU
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(criterion.parameters()),  # include proj layers
        lr=Config.LR, momentum=0.9, weight_decay=5e-4)
    scheduler = WarmupCosineScheduler(optimizer, Config.WARMUP_EPOCHS, Config.NUM_EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=Config.USE_AMP)

    print("\nStarting training...")
    best_recall = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        lr = scheduler.step(epoch)
        T = get_temperature(epoch, Config.NUM_EPOCHS)
        print(f"\n{'='*40}\nEpoch {epoch+1}/{Config.NUM_EPOCHS}, LR: {lr:.6f}, T: {T:.1f}\n{'='*40}")
        avg_loss, loss_dict = train_one_epoch(model, teacher, train_loader, criterion,
                                              optimizer, scaler, device, epoch)
        print(f"\nEpoch {epoch+1} — Avg Loss: {avg_loss:.4f}")
        for k, v in loss_dict.items(): print(f"  {k}: {v:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == Config.NUM_EPOCHS - 1:
            print("\nEvaluating...")
            recall, ap = evaluate(model, test_dataset, device)
            print(f"  R@1: {recall['R@1']:.2f}%  R@5: {recall['R@5']:.2f}%  R@10: {recall['R@10']:.2f}%  mAP: {ap:.2f}%")
            if recall['R@1'] > best_recall:
                best_recall = recall['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'recall@1': recall['R@1'], 'ap': ap},
                           os.path.join(Config.OUTPUT_DIR, 'best_model_exp6.pth'))
                print("  ★ Saved best model!")

    print(f"\n{'='*60}\nTraining complete! Best R@1: {best_recall:.2f}%\n{'='*60}")


if __name__ == "__main__":
    main()
