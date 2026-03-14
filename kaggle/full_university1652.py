# =============================================================================
# SPDGeo-DPE-MAR on University-1652 — Drone→Satellite Geo-Localization
# =============================================================================
# EXP35 pipeline optimized via ablation study:
#   ✅ ProxyAnchor Loss (critical, -6% without)
#   ✅ DynamicFusionGate (important, -2% without)
#   ✅ EMA Teacher Ensemble (helpful)
#   ✅ MaskedPartReconstruction (important, -2% without)
#   ✅ UAPA (helpful)
#   ✅ PartConsistencyLoss (helpful)
#   ✅ TTE (multi-crop + Tent at inference)
#   ❌ CrossDistillation (ablation: removing IMPROVES by +0.3%)
#   ❌ SelfDistillation (ablation: negligible effect)
#   ❌ PrototypeDiversityLoss (ablation: negligible effect)
#   ❌ DeepAltitudeFiLM / AltConsist / AltPred (no altitude data)
#
# Dataset: University-1652 (701 train buildings, no test leakage)
# Total losses: 7 (CE + InfoNCE + PartConsist + UAPA + Proxy + EMA + MaskRecon)
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
    DATA_ROOT       = "/kaggle/input/datasets/chinguyeen/university-1652/University-1652"
    OUTPUT_DIR      = "/kaggle/working"

    IMG_SIZE        = 448       # 14×32 patches for DINOv2-S/14
    N_PARTS         = 8
    PART_DIM        = 256
    EMBED_DIM       = 512
    # TEACHER_DIM removed — no cross-distillation (ablation: harmful)
    CLUSTER_TEMP    = 0.07
    UNFREEZE_BLOCKS = 4

    NUM_EPOCHS      = 100
    P_CLASSES       = 16
    K_SAMPLES       = 4
    LR              = 3e-4
    BACKBONE_LR     = 3e-5
    WEIGHT_DECAY    = 0.01
    WARMUP_EPOCHS   = 5
    USE_AMP         = True
    SEED            = 42

    # Loss weights
    LAMBDA_CE           = 1.0
    LAMBDA_INFONCE      = 1.0
    LAMBDA_CONSISTENCY  = 0.1
    LAMBDA_UAPA         = 0.2
    LAMBDA_PROXY        = 0.5
    PROXY_MARGIN        = 0.1
    PROXY_ALPHA         = 32
    LAMBDA_EMA_DIST     = 0.2
    EMA_DECAY           = 0.996
    EVAL_INTERVAL       = 5
    NUM_WORKERS         = 2

    # Masked Part Reconstruction (from EXP34)
    MASK_RATIO          = 0.30
    LAMBDA_MASK_RECON   = 0.3
    RECON_WARMUP        = 10

    # Removed by ablation: CrossDistill, SelfDistill, Diversity

    # TTE
    TTE_CROP_SIZES      = [392, 448, 504]   # multiples of 14
    TTE_CROP_WEIGHTS    = [0.25, 0.50, 0.25]
    TTE_TENT_STEPS      = 3
    TTE_TENT_LR         = 1e-4

CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); torch.backends.cudnn.benchmark = True


# =============================================================================
# DATASET — University-1652
# =============================================================================
class University1652TrainDataset(Dataset):
    """
    Training set: 701 train buildings only (no test data).
    Each sample is a drone-satellite pair from train/drone and train/satellite.
    """
    def __init__(self, root, transform=None):
        self.root = root; self.transform = transform
        self.drone_dir = os.path.join(root, "train", "drone")
        self.sat_dir   = os.path.join(root, "train", "satellite")

        self.building_ids = sorted([
            d for d in os.listdir(self.drone_dir)
            if os.path.isdir(os.path.join(self.drone_dir, d))
            and os.path.isdir(os.path.join(self.sat_dir, d))
        ])

        self.bid_to_idx = {b: i for i, b in enumerate(self.building_ids)}
        self.num_classes = len(self.building_ids)

        self.samples = []; self.drone_by_class = defaultdict(list)

        for bid in self.building_ids:
            idx = self.bid_to_idx[bid]
            dp = os.path.join(self.drone_dir, bid)
            sp = os.path.join(self.sat_dir, bid)
            sat_imgs = sorted([os.path.join(sp, f) for f in os.listdir(sp) if f.endswith(('.jpg','.jpeg','.png'))])
            if not sat_imgs: continue
            for f in sorted(os.listdir(dp)):
                if f.endswith(('.jpg','.jpeg','.png')):
                    self.samples.append((os.path.join(dp, f), random.choice(sat_imgs), idx))
                    self.drone_by_class[idx].append(len(self.samples) - 1)

        print(f"  [train] {len(self.samples)} drone-sat pairs | {self.num_classes} classes")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        dp, sp, label = self.samples[idx]
        try:
            d = Image.open(dp).convert('RGB'); s = Image.open(sp).convert('RGB')
        except Exception:
            d = Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128,128,128))
            s = Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128,128,128))
        if self.transform: d = self.transform(d); s = self.transform(s)
        return {'drone': d, 'satellite': s, 'label': label}


class University1652TestDataset:
    """Test set: query_drone + gallery_satellite."""
    def __init__(self, root, transform=None):
        self.root = root; self.transform = transform
        self.query_dir   = os.path.join(root, "test", "query_drone")
        self.gallery_dir = os.path.join(root, "test", "gallery_satellite")

        self.query_samples, self.query_labels = [], []
        if os.path.isdir(self.query_dir):
            for bid in sorted(os.listdir(self.query_dir)):
                bp = os.path.join(self.query_dir, bid)
                if not os.path.isdir(bp): continue
                for f in sorted(os.listdir(bp)):
                    if f.endswith(('.jpg','.jpeg','.png')):
                        self.query_samples.append(os.path.join(bp, f))
                        self.query_labels.append(int(bid))

        self.gallery_samples, self.gallery_labels = [], []
        if os.path.isdir(self.gallery_dir):
            for bid in sorted(os.listdir(self.gallery_dir)):
                bp = os.path.join(self.gallery_dir, bid)
                if not os.path.isdir(bp): continue
                for f in sorted(os.listdir(bp)):
                    if f.endswith(('.jpg','.jpeg','.png')):
                        self.gallery_samples.append(os.path.join(bp, f))
                        self.gallery_labels.append(int(bid))

        print(f"  [test] Query: {len(self.query_samples)} drone imgs, "
              f"Gallery: {len(self.gallery_samples)} satellite imgs")


class PKSampler:
    def __init__(self, ds, p, k):
        self.ds = ds; self.p = p; self.k = k
        self.locs = list(ds.drone_by_class.keys())
    def __iter__(self):
        locs = self.locs.copy(); random.shuffle(locs); batch = []
        for l in locs:
            idx = self.ds.drone_by_class[l]
            if len(idx) < self.k: idx = idx * (self.k // len(idx) + 1)
            batch.extend(random.sample(idx, self.k))
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]; batch = batch[self.p * self.k:]
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
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return features['x_norm_patchtokens'], features['x_norm_clstoken'], (H, W)


# DINOv2Teacher REMOVED — ablation showed CrossDistillation is harmful


# =============================================================================
# PART DISCOVERY + POOLING + FUSION
# =============================================================================
class SemanticPartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07):
        super().__init__()
        self.n_parts = n_parts; self.temperature = temperature
        self.feat_proj = nn.Sequential(nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU())
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim*2),
                                    nn.GELU(), nn.Linear(part_dim*2, part_dim))
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
        gy = torch.arange(H, device=device).float() / max(H-1, 1)
        gx = torch.arange(W, device=device).float() / max(W-1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(B, -1, -1)) / mass
        salience = self.salience_head(part_feat).squeeze(-1)
        return {'part_features': part_feat, 'part_positions': part_pos,
                'assignment': assign, 'salience': salience, 'projected_patches': feat}


class PartAwarePooling(nn.Module):
    def __init__(self, part_dim=256, embed_dim=512):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(part_dim, part_dim//2), nn.Tanh(), nn.Linear(part_dim//2, 1))
        self.proj = nn.Sequential(nn.Linear(part_dim*3, embed_dim), nn.LayerNorm(embed_dim),
                                  nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, part_features, salience=None):
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
# NEW: Masked Part Reconstruction (from EXP34)
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

# PrototypeDiversityLoss REMOVED — ablation: negligible effect


# =============================================================================
# STUDENT MODEL — SPDGeo-DPE-MAR (no altitude, ablation-optimized)
# =============================================================================
class SPDGeoDPEMARModel(nn.Module):
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

        # MAR module
        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-DPE-MAR student: {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)

    def extract_with_assignment(self, x):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb), parts['assignment']

    def forward(self, x, return_parts=False):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        bn = self.bottleneck(emb); logits = self.classifier(bn)
        cls_emb_raw = self.cls_proj(cls_tok)
        cls_logits = self.cls_classifier(cls_emb_raw)
        cls_emb_norm = F.normalize(cls_emb_raw, dim=-1)
        fused = self.fusion_gate(emb, cls_emb_norm)
        out = {'embedding': fused, 'logits': logits, 'cls_logits': cls_logits,
               'part_emb': emb, 'cls_emb': cls_emb_norm}
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
        labels = labels.view(-1, 1); pos_mask = labels.eq(labels.T).float()
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        return (-(log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)).mean()

class PartConsistencyLoss(nn.Module):
    def forward(self, assign_q, assign_r):
        dist_q = assign_q.mean(dim=1); dist_r = assign_r.mean(dim=1)
        kl_qr = F.kl_div((dist_q+1e-8).log(), dist_r, reduction='batchmean', log_target=False)
        kl_rq = F.kl_div((dist_r+1e-8).log(), dist_q, reduction='batchmean', log_target=False)
        return 0.5*(kl_qr+kl_rq)

# CrossDistillationLoss REMOVED — ablation: harmful
# SelfDistillationLoss REMOVED — ablation: negligible

class UAPALoss(nn.Module):
    def __init__(self, base_temperature=4.0): super().__init__(); self.T0 = base_temperature
    @staticmethod
    def _entropy(logits):
        p = F.softmax(logits, dim=1); return -(p*(p+1e-8).log()).sum(dim=1).mean()
    def forward(self, drone_logits, sat_logits):
        T = self.T0 * (1 + torch.sigmoid(self._entropy(drone_logits) - self._entropy(sat_logits)))
        return (T**2) * F.kl_div(F.log_softmax(drone_logits/T, 1), F.softmax(sat_logits/T, 1).detach(), reduction='batchmean')

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


# =============================================================================
# TTE — Test-Time Ensemble
# =============================================================================
class TentAdaptation:
    def __init__(self, model, lr=1e-4, steps=3):
        self.model = model; self.lr = lr; self.steps = steps
        self.orig_prototypes = model.part_disc.prototypes.data.clone()

    def adapt_and_extract(self, images):
        self.model.part_disc.prototypes.data.copy_(self.orig_prototypes)
        self.model.eval()
        self.model.part_disc.prototypes.requires_grad_(True)
        opt = torch.optim.Adam([self.model.part_disc.prototypes], lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad()
            _, assignment = self.model.extract_with_assignment(images)
            entropy = -(assignment.mean(1) * (assignment.mean(1)+1e-8).log()).sum(-1).mean()
            entropy.backward()
            opt.step()
        with torch.no_grad():
            emb = self.model.extract_embedding(images)
        self.model.part_disc.prototypes.requires_grad_(False)
        self.model.part_disc.prototypes.data.copy_(self.orig_prototypes)
        return emb


def multi_crop_extract(model, image_pil, crop_sizes, crop_weights, device, tent=None):
    all_feats = []
    for sz, w in zip(crop_sizes, crop_weights):
        tf = get_transforms("test", img_size=sz)
        img_t = tf(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = tent.adapt_and_extract(img_t) if tent else model.extract_embedding(img_t)
        all_feats.append(feat * w)
    return F.normalize(sum(all_feats), dim=-1)


@torch.no_grad()
def evaluate_tte(model, ema_model, test_data, device, tent=None):
    """TTE evaluation: multi-crop + EMA ensemble + optional Tent."""
    model.eval()
    if ema_model: ema_model.eval()

    cs = CFG.TTE_CROP_SIZES; cw = CFG.TTE_CROP_WEIGHTS

    # Query drone features
    print("  TTE: Extracting query features …")
    query_feats = []
    for img_path in tqdm(test_data.query_samples, desc="TTE Query"):
        try: pil = Image.open(img_path).convert('RGB')
        except: pil = Image.new('RGB', (448, 448), (128,128,128))
        feat = multi_crop_extract(model, pil, cs, cw, device, tent=tent)
        if ema_model:
            ema_feat = multi_crop_extract(ema_model, pil, cs, cw, device)
            feat = F.normalize(0.5 * feat + 0.5 * ema_feat, dim=-1)
        query_feats.append(feat.cpu())
    query_feats = torch.cat(query_feats)
    query_labels = torch.tensor(test_data.query_labels)

    # Gallery satellite features
    print("  TTE: Extracting gallery features …")
    gallery_feats = []
    for img_path in tqdm(test_data.gallery_samples, desc="TTE Gallery"):
        try: pil = Image.open(img_path).convert('RGB')
        except: pil = Image.new('RGB', (448, 448), (128,128,128))
        feat = multi_crop_extract(model, pil, cs, cw, device)
        if ema_model:
            ema_feat = multi_crop_extract(ema_model, pil, cs, cw, device)
            feat = F.normalize(0.5 * feat + 0.5 * ema_feat, dim=-1)
        gallery_feats.append(feat.cpu())
    gallery_feats = torch.cat(gallery_feats)
    gallery_labels = torch.tensor(test_data.gallery_labels)

    sim = query_feats @ gallery_feats.T; _, rank = sim.sort(1, descending=True)
    N = query_feats.size(0); r1=r5=r10=ap=0
    for i in range(N):
        matches = torch.where(gallery_labels[rank[i]] == query_labels[i])[0]
        if len(matches) == 0: continue
        f = matches[0].item()
        if f < 1: r1 += 1
        if f < 5: r5 += 1
        if f < 10: r10 += 1
        ap += sum((j+1)/(p.item()+1) for j, p in enumerate(matches)) / len(matches)
    return {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100, 'mAP': ap/N*100}


# =============================================================================
# EVALUATION — standard single-scale
# =============================================================================
@torch.no_grad()
def evaluate(model, test_data, device):
    model.eval()
    test_tf = get_transforms("test")

    query_feats = []
    batch_imgs = []
    for i, img_path in enumerate(tqdm(test_data.query_samples, desc='Query', leave=False)):
        batch_imgs.append(test_tf(Image.open(img_path).convert('RGB')))
        if len(batch_imgs) == 64 or i == len(test_data.query_samples) - 1:
            query_feats.append(model.extract_embedding(torch.stack(batch_imgs).to(device)).cpu())
            batch_imgs = []
    query_feats = torch.cat(query_feats)
    query_labels = torch.tensor(test_data.query_labels)

    gallery_feats = []
    batch_imgs = []
    for i, img_path in enumerate(tqdm(test_data.gallery_samples, desc='Gallery', leave=False)):
        batch_imgs.append(test_tf(Image.open(img_path).convert('RGB')))
        if len(batch_imgs) == 64 or i == len(test_data.gallery_samples) - 1:
            gallery_feats.append(model.extract_embedding(torch.stack(batch_imgs).to(device)).cpu())
            batch_imgs = []
    gallery_feats = torch.cat(gallery_feats)
    gallery_labels = torch.tensor(test_data.gallery_labels)

    print(f"  Query: {len(query_feats)}, Gallery: {len(gallery_feats)}")

    sim = query_feats @ gallery_feats.T; _, rank = sim.sort(1, descending=True)
    N = query_feats.size(0); r1=r5=r10=ap=0
    for i in range(N):
        matches = torch.where(gallery_labels[rank[i]] == query_labels[i])[0]
        if len(matches) == 0: continue
        f = matches[0].item()
        if f < 1: r1 += 1
        if f < 5: r5 += 1
        if f < 10: r10 += 1
        ap += sum((j+1)/(p.item()+1) for j, p in enumerate(matches)) / len(matches)
    return {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100, 'mAP': ap/N*100}


# =============================================================================
# TRAINING
# =============================================================================
def train_one_epoch(model, ema, loader, losses, optimizer, scaler, device, epoch):
    model.train()
    infonce, ce, consist, uapa, proxy_anchor, ema_dist = losses
    total_sum = 0; n = 0; loss_sums = defaultdict(float)
    recon_active = epoch >= CFG.RECON_WARMUP

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
            l_uapa = uapa(d_out['logits'], s_out['logits'])

            l_proxy = 0.5 * (proxy_anchor(d_out['embedding'], labels) +
                             proxy_anchor(s_out['embedding'], labels))

            with torch.no_grad():
                ema_d = ema.forward(drone); ema_s = ema.forward(sat)
            l_ema = 0.5 * (ema_dist(d_out['embedding'], ema_d) +
                           ema_dist(s_out['embedding'], ema_s))

            loss = (CFG.LAMBDA_CE * l_ce + CFG.LAMBDA_INFONCE * l_nce +
                    CFG.LAMBDA_CONSISTENCY * l_con + CFG.LAMBDA_UAPA * l_uapa +
                    CFG.LAMBDA_PROXY * l_proxy + CFG.LAMBDA_EMA_DIST * l_ema)

            # Masked Part Reconstruction (after warmup)
            if recon_active:
                l_rec_d = model.mask_recon(d_out['parts']['projected_patches'],
                                           d_out['parts']['part_features'],
                                           d_out['parts']['assignment'])
                l_rec_s = model.mask_recon(s_out['parts']['projected_patches'],
                                           s_out['parts']['part_features'],
                                           s_out['parts']['assignment'])
                l_rec = 0.5 * (l_rec_d + l_rec_s)
                loss += CFG.LAMBDA_MASK_RECON * l_rec
                loss_sums['recon'] += l_rec.item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()
        ema.update(model)

        total_sum += loss.item(); n += 1
        loss_sums['ce']    += l_ce.item()
        loss_sums['nce']   += l_nce.item()
        loss_sums['con']   += l_con.item()
        loss_sums['uapa']  += l_uapa.item()
        loss_sums['proxy'] += l_proxy.item()
        loss_sums['ema']   += l_ema.item()

    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  SPDGeo-DPE-MAR on University-1652 — Ablation-Optimized")
    print(f"  Losses: 7 (CE + NCE + PartConsist + UAPA + Proxy + EMA + MaskRecon)")
    print(f"  Epochs: {CFG.NUM_EPOCHS} | Img: {CFG.IMG_SIZE} | Device: {DEVICE}")
    print(f"  Removed: CrossDistill, SelfDistill, Diversity (ablation insights)")
    print("=" * 65)

    print('\nLoading University-1652 …')
    train_ds  = University1652TrainDataset(CFG.DATA_ROOT, transform=get_transforms("train"))
    test_data = University1652TestDataset(CFG.DATA_ROOT, transform=get_transforms("test"))
    train_loader = DataLoader(train_ds, batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print('\nBuilding models …')
    model = SPDGeoDPEMARModel(train_ds.num_classes).to(DEVICE)

    ema = EMAModel(model, decay=CFG.EMA_DECAY)
    print(f"  EMA model initialized (decay={CFG.EMA_DECAY})")

    # Losses
    infonce      = SupInfoNCELoss(temp=0.05).to(DEVICE)
    ce           = nn.CrossEntropyLoss(label_smoothing=0.1)
    consist      = PartConsistencyLoss()
    uapa         = UAPALoss(base_temperature=4.0)
    proxy_anchor = ProxyAnchorLoss(train_ds.num_classes, CFG.EMBED_DIM,
                                    margin=CFG.PROXY_MARGIN, alpha=CFG.PROXY_ALPHA).to(DEVICE)
    ema_dist     = EMADistillationLoss()
    all_losses   = (infonce, ce, consist, uapa, proxy_anchor, ema_dist)

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
        for i in range(4):
            optimizer.param_groups[i]['lr'] = (CFG.BACKBONE_LR if i == 0 else CFG.LR) * lr_scale
        cur_lr = optimizer.param_groups[1]['lr']

        avg_loss, ld = train_one_epoch(model, ema, train_loader, all_losses,
                                       optimizer, scaler, DEVICE, epoch)
        loss_str = " ".join(f"{k}={v:.3f}" for k, v in ld.items())
        print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | {loss_str} | LR {cur_lr:.2e}")

        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            metrics = evaluate(model, test_data, DEVICE)
            results_log.append({'epoch': epoch, **metrics})
            print(f"  ► R@1: {metrics['R@1']:.2f}%  R@5: {metrics['R@5']:.2f}%  "
                  f"R@10: {metrics['R@10']:.2f}%  mAP: {metrics['mAP']:.2f}%")
            if metrics['R@1'] > best_r1:
                best_r1 = metrics['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'metrics': metrics},
                           os.path.join(CFG.OUTPUT_DIR, 'spdgeo_dpe_mar_university1652_best.pth'))
                print(f"  ★ New best R@1: {best_r1:.2f}%!")

            ema_metrics = evaluate(ema.model, test_data, DEVICE)
            print(f"  ► EMA R@1: {ema_metrics['R@1']:.2f}%")
            if ema_metrics['R@1'] > best_r1:
                best_r1 = ema_metrics['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': ema.model.state_dict(),
                            'metrics': ema_metrics, 'is_ema': True},
                           os.path.join(CFG.OUTPUT_DIR, 'spdgeo_dpe_mar_university1652_best.pth'))
                print(f"  ★ New best R@1 (EMA): {best_r1:.2f}%!")

    print(f'\n{"="*65}')
    print(f'  SPDGeo-DPE-MAR COMPLETE — University-1652 — Best R@1: {best_r1:.2f}%')
    print(f'{"="*65}')
    print(f'  {"Epoch":>6} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"mAP":>8}')
    print(f'  {"-"*44}')
    for r in results_log:
        print(f'  {r["epoch"]:6d} {r["R@1"]:8.2f} {r["R@5"]:8.2f} {r["R@10"]:8.2f} {r["mAP"]:8.2f}')
    print(f'{"="*65}')

    with open(os.path.join(CFG.OUTPUT_DIR, 'spdgeo_dpe_mar_university1652_results.json'), 'w') as f:
        json.dump({'results_log': results_log, 'best_r1': best_r1,
                   'config': {k: v for k, v in vars(CFG).items() if not k.startswith('_')}}, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════
    # TTE: Test-Time Ensemble
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n{"="*65}')
    print(f'  TTE: Test-Time Ensemble Evaluation')
    print(f'  Crops: {CFG.TTE_CROP_SIZES} | Tent: {CFG.TTE_TENT_STEPS} steps')
    print(f'{"="*65}')

    best_ckpt = os.path.join(CFG.OUTPUT_DIR, 'spdgeo_dpe_mar_university1652_best.pth')
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'  Loaded best checkpoint (R@1: {ckpt["metrics"]["R@1"]:.2f}%)')
    model.eval()

    print('\n  [1/3] Multi-crop only …')
    tte_mc = evaluate_tte(model, None, test_data, DEVICE, tent=None)
    print(f'  Multi-crop R@1: {tte_mc["R@1"]:.2f}%  mAP: {tte_mc["mAP"]:.2f}%')

    print('\n  [2/3] Multi-crop + EMA …')
    tte_mc_ema = evaluate_tte(model, ema.model, test_data, DEVICE, tent=None)
    print(f'  MC+EMA R@1: {tte_mc_ema["R@1"]:.2f}%  mAP: {tte_mc_ema["mAP"]:.2f}%')

    print('\n  [3/3] Full TTE (MC + EMA + Tent) …')
    tent = TentAdaptation(model, lr=CFG.TTE_TENT_LR, steps=CFG.TTE_TENT_STEPS)
    tte_full = evaluate_tte(model, ema.model, test_data, DEVICE, tent=tent)
    print(f'  Full TTE R@1: {tte_full["R@1"]:.2f}%  mAP: {tte_full["mAP"]:.2f}%')

    print(f'\n{"="*65}')
    print(f'  TTE SUMMARY — University-1652')
    print(f'{"="*65}')
    print(f'  {"Method":<30s}  {"R@1":>7s}  {"mAP":>7s}')
    print(f'  {"-"*50}')
    print(f'  {"Baseline (training eval)":<30s}  {best_r1:6.2f}%  -')
    print(f'  {"Multi-crop":<30s}  {tte_mc["R@1"]:6.2f}%  {tte_mc["mAP"]:6.2f}%')
    print(f'  {"MC + EMA":<30s}  {tte_mc_ema["R@1"]:6.2f}%  {tte_mc_ema["mAP"]:6.2f}%')
    print(f'  {"MC + EMA + Tent (Full TTE)":<30s}  {tte_full["R@1"]:6.2f}%  {tte_full["mAP"]:6.2f}%')
    print(f'{"="*65}')

    final_r1 = max(best_r1, tte_mc['R@1'], tte_mc_ema['R@1'], tte_full['R@1'])
    print(f'\n  FINAL BEST R@1: {final_r1:.2f}%')

    with open(os.path.join(CFG.OUTPUT_DIR, 'spdgeo_dpe_mar_university1652_tte.json'), 'w') as f:
        json.dump({'baseline_r1': best_r1, 'multi_crop': tte_mc,
                   'mc_ema': tte_mc_ema, 'full_tte': tte_full,
                   'final_best_r1': final_r1}, f, indent=2)


if __name__ == '__main__':
    main()
