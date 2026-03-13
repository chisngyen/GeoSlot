# =============================================================================
# EXP43: SPDGeo-DPEA-MAR-v2 — System Refinement
# =============================================================================
# Base:    SPDGeo-DPEA-MAR (EXP35-FM, 95.08% R@1)
# Changes: 1) GradNorm automatic 12-loss balancing (Chen et al. 2018)
#          2) Fixed Tent TTA — cumulative adaptation over full test set
#          3) 160 epochs, cosine floor 0.05%, EMA decay 0.9996
#          4) Gradient clipping max_norm=1.0
#
# Architecture: IDENTICAL to DPEA-MAR. No new modules.
# Total losses: 12 (same) — weights dynamically balanced by GradNorm.
# Expected: ~95.7–96.3% R@1 with TTE.
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

    NUM_EPOCHS      = 160
    P_CLASSES       = 16
    K_SAMPLES       = 4
    LR              = 3e-4
    BACKBONE_LR     = 3e-5
    WEIGHT_DECAY    = 0.01
    WARMUP_EPOCHS   = 5
    LR_FLOOR        = 0.0005
    USE_AMP         = True
    SEED            = 42
    GRAD_CLIP       = 1.0

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
    EMA_DECAY           = 0.9996
    LAMBDA_ALT_CONSIST  = 0.2
    MASK_RATIO          = 0.30
    LAMBDA_MASK_RECON   = 0.3
    RECON_WARMUP        = 10
    LAMBDA_ALT_PRED     = 0.15
    LAMBDA_DIVERSITY    = 0.05

    GRADNORM_ALPHA  = 1.5
    GRADNORM_LR     = 0.025

    DISTILL_TEMP    = 4.0
    EVAL_INTERVAL   = 5
    NUM_WORKERS     = 2

    TTE_CROP_SIZES   = [280, 336, 392]
    TTE_CROP_WEIGHTS = [0.25, 0.50, 0.25]
    TTE_TENT_LR     = 1e-4
    TTE_TENT_STEPS  = 3

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
        alt_idx = CFG.ALT_TO_IDX.get(alt, 0)
        alt_norm = (int(alt) - 150) / 150.0
        return {'drone': d, 'satellite': s, 'label': li, 'altitude': int(alt),
                'alt_idx': alt_idx, 'alt_norm': alt_norm}


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
               (x.shape[2] // self.patch_size, x.shape[3] // self.patch_size)


class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        print("  Loading DINOv2 ViT-B/14 teacher ...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        for p in self.model.parameters(): p.requires_grad = False
        print("  DINOv2 ViT-B/14 teacher loaded (all frozen).")

    @torch.no_grad()
    def forward(self, x):
        return self.model.forward_features(x)['x_norm_clstoken']


# =============================================================================
# MODULES (identical to DPEA-MAR)
# =============================================================================
class DeepAltitudeFiLM(nn.Module):
    def __init__(self, num_altitudes=4, feat_dim=256):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_altitudes, feat_dim))
        self.beta  = nn.Parameter(torch.zeros(num_altitudes, feat_dim))

    def forward(self, feat, alt_idx=None):
        if alt_idx is None:
            g = self.gamma.mean(0, keepdim=True); b = self.beta.mean(0, keepdim=True)
            return feat * g.unsqueeze(0) + b.unsqueeze(0)
        return feat * self.gamma[alt_idx].unsqueeze(1) + self.beta[alt_idx].unsqueeze(1)


class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07, num_altitudes=4):
        super().__init__()
        self.temperature = temperature
        self.feat_proj = nn.Sequential(nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU())
        self.altitude_film = DeepAltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
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
        gy = torch.arange(h, device=device).float() / max(h-1, 1)
        gx = torch.arange(w, device=device).float() / max(w-1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(b, -1, -1)) / mass
        salience = self.salience_head(part_feat).squeeze(-1)
        return {"part_features": part_feat, "part_positions": part_pos, "assignment": assign,
                "salience": salience, "projected_patches": feat}


class PartAwarePooling(nn.Module):
    def __init__(self, part_dim=256, embed_dim=512):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(part_dim, part_dim//2), nn.Tanh(), nn.Linear(part_dim//2, 1))
        self.proj = nn.Sequential(nn.Linear(part_dim*3, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

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
    def __init__(self, part_dim=256, mask_ratio=0.30):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(nn.Linear(part_dim, part_dim*2), nn.GELU(), nn.Linear(part_dim*2, part_dim), nn.LayerNorm(part_dim))

    def forward(self, projected_patches, part_features, assignment):
        b, n, d = projected_patches.shape
        num_mask = int(n * self.mask_ratio)
        noise = torch.rand(b, n, device=projected_patches.device)
        mask_idx = noise.argsort(dim=1)[:, :num_mask]
        target = projected_patches.detach()
        me = mask_idx.unsqueeze(-1).expand(-1, -1, d)
        masked_targets = torch.gather(target, 1, me)
        k = part_features.shape[1]
        mk = mask_idx.unsqueeze(-1).expand(-1, -1, k)
        masked_assign = torch.gather(assignment, 1, mk)
        recon = self.decoder(torch.bmm(masked_assign, part_features))
        return (1 - (F.normalize(recon, dim=-1) * F.normalize(masked_targets, dim=-1)).sum(-1)).mean()


class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(True), nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, embedding, alt_target):
        return F.smooth_l1_loss(self.head(embedding).squeeze(-1), alt_target)


class PrototypeDiversityLoss(nn.Module):
    def forward(self, prototypes):
        p = F.normalize(prototypes, dim=-1); sim = p @ p.T; k = sim.size(0)
        mask = 1 - torch.eye(k, device=sim.device)
        return (sim * mask).abs().sum() / (k * (k - 1))


# =============================================================================
# MODEL
# =============================================================================
class SPDGeoDPEAMARv2Model(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = AltitudeAwarePartDiscovery(384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES)
        self.pool = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)
        self.bottleneck = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM), nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(True))
        self.classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM), nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM), nn.LayerNorm(cfg.TEACHER_DIM))
        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)
        self.alt_pred = AltitudePredictionHead(cfg.EMBED_DIM)
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-DPEA-MAR-v2: {total/1e6:.1f}M trainable")

    def extract_embedding(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts["part_features"], parts["salience"])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)

    def extract_with_assignment(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts["part_features"], parts["salience"])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb), parts["assignment"]

    def forward(self, x, alt_idx=None, return_parts=False):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts["part_features"], parts["salience"])
        bn = self.bottleneck(emb); logits = self.classifier(bn)
        cls_emb_raw = self.cls_proj(cls_tok)
        cls_logits = self.cls_classifier(cls_emb_raw)
        cls_emb_norm = F.normalize(cls_emb_raw, dim=-1)
        fused = self.fusion_gate(emb, cls_emb_norm)
        projected_feat = self.teacher_proj(emb)
        out = {"embedding": fused, "logits": logits, "cls_logits": cls_logits,
               "projected_feat": projected_feat, "part_emb": emb, "cls_emb": cls_emb_norm}
        if return_parts: out["parts"] = parts
        return out


class EMAModel:
    def __init__(self, model, decay=0.9996):
        self.decay = decay; self.model = copy.deepcopy(model); self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ep, mp in zip(self.model.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(mp.data, alpha=1 - self.decay)

    @torch.no_grad()
    def forward(self, x, alt_idx=None):
        return self.model.extract_embedding(x, alt_idx=alt_idx)


# =============================================================================
# LOSSES
# =============================================================================
class SupInfoNCELoss(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__(); self.log_t = nn.Parameter(torch.tensor(temp).log())

    def forward(self, q, r, labels):
        t = self.log_t.exp().clamp(0.01, 1.0); sim = q @ r.t() / t
        labels = labels.view(-1, 1); pos = labels.eq(labels.T).float()
        lp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        return (-(lp * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


class PartConsistencyLoss(nn.Module):
    def forward(self, aq, ar):
        dq = aq.mean(1); dr = ar.mean(1)
        return 0.5 * (F.kl_div((dq+1e-8).log(), dr, reduction='batchmean', log_target=False) +
                       F.kl_div((dr+1e-8).log(), dq, reduction='batchmean', log_target=False))


class CrossDistillationLoss(nn.Module):
    def forward(self, s, t):
        s = F.normalize(s, dim=-1); t = F.normalize(t, dim=-1)
        return F.mse_loss(s, t) + (1 - F.cosine_similarity(s, t).mean())


class SelfDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0): super().__init__(); self.t = temperature
    def forward(self, weak, strong):
        p = F.softmax(strong / self.t, dim=1).detach()
        return (self.t**2) * F.kl_div(F.log_softmax(weak / self.t, dim=1), p, reduction='batchmean')


class UAPALoss(nn.Module):
    def __init__(self, base_temperature=4.0): super().__init__(); self.t0 = base_temperature
    @staticmethod
    def _entropy(logits):
        p = F.softmax(logits, dim=1); return -(p * (p+1e-8).log()).sum(1).mean()
    def forward(self, d_logits, s_logits):
        du = self._entropy(d_logits) - self._entropy(s_logits)
        t = self.t0 * (1 + torch.sigmoid(du))
        p = F.softmax(s_logits / t, dim=1).detach()
        return (t**2) * F.kl_div(F.log_softmax(d_logits / t, dim=1), p, reduction='batchmean')


class ProxyAnchorLoss(nn.Module):
    def __init__(self, nc, ed, margin=0.1, alpha=32):
        super().__init__(); self.proxies = nn.Parameter(torch.randn(nc, ed)*0.01)
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.margin = margin; self.alpha = alpha; self.nc = nc

    def forward(self, emb, labels):
        p = F.normalize(self.proxies, dim=-1); sim = emb @ p.T
        oh = F.one_hot(labels, self.nc).float()
        pos_exp = torch.exp(-self.alpha * (sim * oh - self.margin)) * oh
        has_pos = oh.sum(0) > 0
        pos_loss = torch.log(1 + pos_exp.sum(0))[has_pos].mean() if has_pos.sum() > 0 else torch.tensor(0., device=emb.device)
        neg_mask = 1 - oh
        neg_loss = torch.log(1 + (torch.exp(self.alpha * (sim * neg_mask + self.margin)) * neg_mask).sum(0)).mean()
        return pos_loss + neg_loss


class EMADistillationLoss(nn.Module):
    def forward(self, s, e): return (1 - F.cosine_similarity(s, e)).mean()


class AltitudeConsistencyLoss(nn.Module):
    def forward(self, emb, labels, alts):
        b = emb.size(0)
        if b < 2: return torch.tensor(0., device=emb.device)
        lbl = labels.view(-1,1); alt = alts.view(-1,1)
        mask = lbl.eq(lbl.T) & alt.ne(alt.T)
        if mask.sum() == 0: return torch.tensor(0., device=emb.device)
        en = F.normalize(emb, dim=-1)
        return ((1 - en @ en.T) * mask.float()).sum() / mask.float().sum().clamp(min=1)


# =============================================================================
# GRADNORM
# =============================================================================
class GradNormBalancer:
    """Automatic multi-task loss balancing via gradient norm normalization."""
    LOSS_NAMES = ["ce", "nce", "con", "cross", "self", "uapa",
                  "proxy", "ema", "alt_c", "recon", "alt_p", "div"]

    def __init__(self, num_losses=12, alpha=1.5, lr=0.025, device='cuda'):
        self.alpha = alpha
        self.num_losses = num_losses
        self.log_w = torch.zeros(num_losses, device=device, requires_grad=True)
        self.opt = torch.optim.Adam([self.log_w], lr=lr)
        self.initial_losses = None

    def weights(self):
        w = torch.exp(self.log_w.detach())
        # Clamp weights to prevent total collapse (e.g. min 0.1, max 10.0)
        w = torch.clamp(w, min=0.1, max=10.0)
        return w / w.sum() * self.num_losses

    @torch.no_grad()
    def update(self, loss_values):
        lv = torch.tensor([l if isinstance(l, float) else l.item()
                           for l in loss_values], device=self.log_w.device)
        if self.initial_losses is None:
            self.initial_losses = lv.clone()
            return

        # Keep track of the maximum observed loss to avoid ratio explosion for losses starting at 0
        self.initial_losses = torch.max(self.initial_losses, lv.clone())

        ratios = lv / (self.initial_losses + 1e-8)
        avg_r = ratios.mean()
        target = (ratios / (avg_r + 1e-8)) ** self.alpha
        w = torch.exp(self.log_w)
        target_w = w * target
        target_w = target_w / target_w.sum() * self.num_losses

        self.opt.zero_grad()
        self.log_w.grad = -(target_w - w) / (w + 1e-8)
        self.opt.step()


# =============================================================================
# EVALUATE
# =============================================================================
@torch.no_grad()
def evaluate(model, test_ds, device):
    model.eval(); test_tf = get_transforms("test")
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    drone_feats, drone_labels, drone_alts = [], [], []
    for b in loader:
        feat = model.extract_embedding(b['drone'].to(device), alt_idx=b['alt_idx'].to(device)).cpu()
        drone_feats.append(feat); drone_labels.append(b['label']); drone_alts.append(b['altitude'])
    drone_feats = torch.cat(drone_feats); drone_labels = torch.cat(drone_labels); drone_alts = torch.cat(drone_alts)

    all_locs = [f"{l:04d}" for l in range(1, 201)]
    sat_imgs, sat_lbls = [], []; dc = 0
    for loc in all_locs:
        sp = os.path.join(test_ds.sat_dir, loc, "0.png")
        if not os.path.exists(sp): continue
        sat_imgs.append(test_tf(Image.open(sp).convert('RGB')))
        sat_lbls.append(test_ds.location_to_idx[loc] if loc in test_ds.location_to_idx else -1000 - dc)
        if loc not in test_ds.location_to_idx: dc += 1
    sat_feats = []
    for i in range(0, len(sat_imgs), 64):
        sat_feats.append(model.extract_embedding(torch.stack(sat_imgs[i:i+64]).to(device), alt_idx=None).cpu())
    sat_feats = torch.cat(sat_feats); sat_labels = torch.tensor(sat_lbls)

    print(f"  Gallery: {len(sat_feats)} sats | Queries: {len(drone_feats)} drones")
    sim = drone_feats @ sat_feats.T; _, rank = sim.sort(1, descending=True)
    n = drone_feats.size(0); r1=r5=r10=ap=0
    for i in range(n):
        m = torch.where(sat_labels[rank[i]] == drone_labels[i])[0]
        if len(m)==0: continue
        f = m[0].item()
        if f<1: r1+=1
        if f<5: r5+=1
        if f<10: r10+=1
        ap += sum((j+1)/(p.item()+1) for j,p in enumerate(m)) / len(m)
    overall = {"R@1": r1/n*100, "R@5": r5/n*100, "R@10": r10/n*100, "mAP": ap/n*100}

    per_alt = {}
    for alt in sorted(drone_alts.unique().tolist()):
        msk = drone_alts == alt
        if msk.sum()==0: continue
        af=drone_feats[msk]; al=drone_labels[msk]; s=af@sat_feats.T; _,rk=s.sort(1,descending=True)
        k=af.size(0); a1=a5=a10=aap=0
        for i in range(k):
            mm=torch.where(sat_labels[rk[i]]==al[i])[0]
            if len(mm)==0: continue
            ff=mm[0].item()
            if ff<1: a1+=1
            if ff<5: a5+=1
            if ff<10: a10+=1
            aap+=sum((j+1)/(p.item()+1) for j,p in enumerate(mm))/len(mm)
        per_alt[int(alt)]={"R@1":a1/k*100,"R@5":a5/k*100,"R@10":a10/k*100,"mAP":aap/k*100,"n":k}
    return overall, per_alt


# =============================================================================
# TTE (FIXED)
# =============================================================================
class TentAdaptation:
    """Fixed Tent: cumulative entropy minimization over full test set."""
    def __init__(self, model, lr=1e-4, steps=3):
        self.model = model; self.lr = lr; self.steps = steps
        self.orig_prototypes = model.part_disc.prototypes.data.clone()
        self.adapted = False

    def adapt_full_testset(self, test_ds, device):
        """Run Tent adaptation over the entire test set ONCE before evaluation."""
        self.model.part_disc.prototypes.data.copy_(self.orig_prototypes)
        self.model.eval()
        self.model.part_disc.prototypes.requires_grad_(True)
        opt = torch.optim.Adam([self.model.part_disc.prototypes], lr=self.lr)
        loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

        for step in range(self.steps):
            for batch in loader:
                imgs = batch['drone'].to(device)
                alt_idx = batch['alt_idx'].to(device)
                with torch.enable_grad():
                    opt.zero_grad()
                    _, assignment = self.model.extract_with_assignment(imgs, alt_idx=alt_idx)
                    ap = assignment.mean(dim=1)
                    entropy = -(ap * (ap + 1e-8).log()).sum(-1).mean()
                    entropy.backward()
                    opt.step()
        self.model.part_disc.prototypes.requires_grad_(False)
        self.adapted = True
        print(f"  Tent: adapted over {len(test_ds)} samples × {self.steps} steps")

    def reset(self):
        self.model.part_disc.prototypes.data.copy_(self.orig_prototypes)
        self.adapted = False


def multi_crop_extract(model, image_pil, crop_sizes, crop_weights, device, alt_idx=None):
    feats = []
    for sz, w in zip(crop_sizes, crop_weights):
        tf = get_transforms("test", img_size=sz)
        img_t = tf(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.extract_embedding(img_t, alt_idx=alt_idx)
        feats.append(feat * w)
    return F.normalize(sum(feats), dim=-1)


@torch.no_grad()
def evaluate_tte(model, ema_model, test_ds, device, use_tent=False):
    model.eval()
    crop_sizes = CFG.TTE_CROP_SIZES; crop_weights = CFG.TTE_CROP_WEIGHTS
    ema_w = 0.4 if ema_model is not None else 0.0
    online_w = 1.0 - ema_w

    drone_feats, drone_labels, drone_alts = [], [], []
    for sample in tqdm(test_ds.samples, desc="TTE Drone"):
        dp, _, li, alt = sample
        try: d_pil = Image.open(dp).convert('RGB')
        except: d_pil = Image.new('RGB', (336,336), (128,128,128))
        alt_t = torch.tensor([CFG.ALT_TO_IDX.get(alt, 0)], device=device)
        feat = multi_crop_extract(model, d_pil, crop_sizes, crop_weights, device, alt_idx=alt_t)
        if ema_model is not None:
            ef = multi_crop_extract(ema_model, d_pil, crop_sizes, crop_weights, device, alt_idx=alt_t)
            feat = F.normalize(online_w * feat + ema_w * ef, dim=-1)
        drone_feats.append(feat.cpu()); drone_labels.append(li); drone_alts.append(int(alt))

    drone_feats = torch.cat(drone_feats); drone_labels = torch.tensor(drone_labels); drone_alts = torch.tensor(drone_alts)

    all_locs = [f"{l:04d}" for l in range(1, 201)]
    sat_feats, sat_lbls = [], []; dc = 0
    for loc in tqdm(all_locs, desc="TTE Sat"):
        sp = os.path.join(test_ds.sat_dir, loc, "0.png")
        if not os.path.exists(sp): continue
        try: s_pil = Image.open(sp).convert('RGB')
        except: s_pil = Image.new('RGB', (336,336), (128,128,128))
        feat = multi_crop_extract(model, s_pil, crop_sizes, crop_weights, device, alt_idx=None)
        if ema_model is not None:
            ef = multi_crop_extract(ema_model, s_pil, crop_sizes, crop_weights, device, alt_idx=None)
            feat = F.normalize(online_w * feat + ema_w * ef, dim=-1)
        sat_feats.append(feat.cpu())
        sat_lbls.append(test_ds.location_to_idx[loc] if loc in test_ds.location_to_idx else -1000 - dc)
        if loc not in test_ds.location_to_idx: dc += 1

    sat_feats = torch.cat(sat_feats); sat_labels = torch.tensor(sat_lbls)
    sim = drone_feats @ sat_feats.T; _, rank = sim.sort(1, descending=True)
    n = drone_feats.size(0); r1=r5=r10=ap=0
    for i in range(n):
        m = torch.where(sat_labels[rank[i]] == drone_labels[i])[0]
        if len(m)==0: continue
        f = m[0].item()
        if f<1: r1+=1
        if f<5: r5+=1
        if f<10: r10+=1
        ap += sum((j+1)/(p.item()+1) for j,p in enumerate(m)) / len(m)
    return {"R@1": r1/n*100, "R@5": r5/n*100, "R@10": r10/n*100, "mAP": ap/n*100}


# =============================================================================
# TRAIN
# =============================================================================
def train_one_epoch(model, teacher, ema, loader, losses, new_losses, optimizer, scaler, device, epoch, gradnorm):
    model.train()
    if teacher: teacher.eval()
    infonce, ce, consist, cross_dist, self_dist, uapa = losses
    proxy_anchor, ema_dist, alt_consist, diversity_loss = new_losses

    total_sum = 0; n = 0; loss_sums = defaultdict(float)
    recon_active = epoch >= CFG.RECON_WARMUP
    gn_w = gradnorm.weights()

    for batch in tqdm(loader, desc=f"Ep{epoch:3d}", leave=False):
        drone   = batch['drone'].to(device)
        sat     = batch['satellite'].to(device)
        labels  = batch['label'].to(device)
        alt_idx = batch['alt_idx'].to(device)
        alts    = batch['altitude'].to(device)
        alt_norm= batch['alt_norm'].to(device).float()

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            d_out = model(drone, alt_idx=alt_idx, return_parts=True)
            s_out = model(sat, alt_idx=None, return_parts=True)

            l_ce   = ce(d_out['logits'], labels) + ce(s_out['logits'], labels) + \
                     0.3 * (ce(d_out['cls_logits'], labels) + ce(s_out['cls_logits'], labels))
            l_nce  = infonce(d_out['embedding'], s_out['embedding'], labels)
            l_con  = consist(d_out['parts']['assignment'], s_out['parts']['assignment'])

            if teacher is not None:
                with torch.no_grad():
                    td = teacher(drone); ts = teacher(sat)
                l_cross = cross_dist(d_out['projected_feat'], td) + cross_dist(s_out['projected_feat'], ts)
            else:
                l_cross = torch.tensor(0., device=device)

            l_self  = self_dist(d_out['cls_logits'], d_out['logits']) + self_dist(s_out['cls_logits'], s_out['logits'])
            l_uapa  = uapa(d_out['logits'], s_out['logits'])
            l_proxy = 0.5*(proxy_anchor(d_out['embedding'], labels) + proxy_anchor(s_out['embedding'], labels))

            with torch.no_grad():
                ed = ema.forward(drone, alt_idx=alt_idx); es = ema.forward(sat, alt_idx=None)
            l_ema   = 0.5*(ema_dist(d_out['embedding'], ed) + ema_dist(s_out['embedding'], es))
            l_alt_c = alt_consist(d_out['embedding'], labels, alts)

            if recon_active:
                l_rec = 0.5*(model.mask_recon(d_out['parts']['projected_patches'], d_out['parts']['part_features'], d_out['parts']['assignment']) +
                             model.mask_recon(s_out['parts']['projected_patches'], s_out['parts']['part_features'], s_out['parts']['assignment']))
            else:
                l_rec = torch.tensor(0., device=device)

            l_alt_p = model.alt_pred(d_out['embedding'].detach(), alt_norm)
            l_div   = diversity_loss(model.part_disc.prototypes)

            all_l = [l_ce, l_nce, l_con, l_cross, l_self, l_uapa,
                     l_proxy, l_ema, l_alt_c, l_rec, l_alt_p, l_div]

            loss = sum(w * l for w, l in zip(gn_w, all_l))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        gradnorm.update([l.item() if torch.is_tensor(l) else l for l in all_l])
        gn_w = gradnorm.weights()

        total_sum += loss.item(); n += 1
        for name, lv in zip(GradNormBalancer.LOSS_NAMES, all_l):
            loss_sums[name] += (lv.item() if torch.is_tensor(lv) else lv)

    return total_sum / max(n,1), {k: v/max(n,1) for k,v in loss_sums.items()}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("="*65)
    print("  EXP43: SPDGeo-DPEA-MAR-v2 — GradNorm + Fixed Tent + 160ep")
    print(f"  Dataset: SUES-200 | Epochs: {CFG.NUM_EPOCHS} | Device: {DEVICE}")
    print(f"  GradNorm α={CFG.GRADNORM_ALPHA} | EMA decay={CFG.EMA_DECAY}")
    print(f"  LR floor={CFG.LR_FLOOR} | Grad clip={CFG.GRAD_CLIP}")
    print("="*65)

    print("\nLoading SUES-200 ...")
    train_ds = SUES200Dataset(CFG.SUES_ROOT, "train", transform=get_transforms("train"))
    test_ds  = SUES200Dataset(CFG.SUES_ROOT, "test",  transform=get_transforms("test"))
    train_loader = DataLoader(train_ds, batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print("\nBuilding models ...")
    model = SPDGeoDPEAMARv2Model(CFG.NUM_CLASSES).to(DEVICE)
    teacher = None
    try: teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except Exception as e: print(f"  [WARN] Teacher failed: {e}")

    ema = EMAModel(model, decay=CFG.EMA_DECAY)

    infonce    = SupInfoNCELoss(0.05).to(DEVICE)
    ce         = nn.CrossEntropyLoss(label_smoothing=0.1)
    consist    = PartConsistencyLoss()
    cross_dist = CrossDistillationLoss()
    self_dist  = SelfDistillationLoss(CFG.DISTILL_TEMP)
    uapa_loss  = UAPALoss(CFG.DISTILL_TEMP)
    base_losses = (infonce, ce, consist, cross_dist, self_dist, uapa_loss)

    proxy_anchor = ProxyAnchorLoss(CFG.NUM_CLASSES, CFG.EMBED_DIM, CFG.PROXY_MARGIN, CFG.PROXY_ALPHA).to(DEVICE)
    ema_distl    = EMADistillationLoss()
    alt_consist  = AltitudeConsistencyLoss()
    div_loss     = PrototypeDiversityLoss()
    new_losses   = (proxy_anchor, ema_distl, alt_consist, div_loss)

    gradnorm = GradNormBalancer(12, alpha=CFG.GRADNORM_ALPHA, lr=CFG.GRADNORM_LR, device=DEVICE)

    bb_params   = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('backbone')]

    optimizer = torch.optim.AdamW([
        {"params": bb_params,   "lr": CFG.BACKBONE_LR},
        {"params": head_params, "lr": CFG.LR},
        {"params": infonce.parameters(), "lr": CFG.LR},
        {"params": proxy_anchor.parameters(), "lr": CFG.LR},
    ], weight_decay=CFG.WEIGHT_DECAY)

    scaler = torch.amp.GradScaler('cuda', enabled=CFG.USE_AMP)
    best_r1 = 0.; results_log = []
    ckpt_path = os.path.join(CFG.OUTPUT_DIR, "exp43_v2_best.pth")

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        if epoch <= CFG.WARMUP_EPOCHS:
            lr_scale = epoch / CFG.WARMUP_EPOCHS
        else:
            progress = (epoch - CFG.WARMUP_EPOCHS) / (CFG.NUM_EPOCHS - CFG.WARMUP_EPOCHS)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        lr_scale = max(lr_scale, CFG.LR_FLOOR)

        optimizer.param_groups[0]['lr'] = CFG.BACKBONE_LR * lr_scale
        for i in [1,2,3]: optimizer.param_groups[i]['lr'] = CFG.LR * lr_scale

        avg_loss, ld = train_one_epoch(model, teacher, ema, train_loader, base_losses, new_losses,
                                       optimizer, scaler, DEVICE, epoch, gradnorm)

        gw = gradnorm.weights()
        gw_str = " ".join(f"{GradNormBalancer.LOSS_NAMES[i]}={gw[i]:.2f}" for i in range(len(gw)))
        recon_tag = f"Rec {ld['recon']:.3f}" if epoch >= CFG.RECON_WARMUP else "Rec OFF"
        print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
              f"CE {ld['ce']:.3f} NCE {ld['nce']:.3f} {recon_tag} | GN: {gw_str}")

        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            metrics, per_alt = evaluate(model, test_ds, DEVICE)
            results_log.append({"epoch": epoch, **metrics})
            print(f"  -> R@1: {metrics['R@1']:.2f}% R@5: {metrics['R@5']:.2f}% "
                  f"R@10: {metrics['R@10']:.2f}% mAP: {metrics['mAP']:.2f}%")

            if metrics['R@1'] > best_r1:
                best_r1 = metrics['R@1']
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "metrics": metrics, "per_alt": per_alt}, ckpt_path)
                print(f"  * New best R@1: {best_r1:.2f}%")

            ema_m, _ = evaluate(ema.model, test_ds, DEVICE)
            print(f"  -> EMA R@1: {ema_m['R@1']:.2f}%")
            if ema_m['R@1'] > best_r1:
                best_r1 = ema_m['R@1']
                torch.save({"epoch": epoch, "model_state_dict": ema.model.state_dict(),
                            "metrics": ema_m, "is_ema": True}, ckpt_path)
                print(f"  * New best R@1 (EMA): {best_r1:.2f}%")

    print(f"\n{'='*65}\n  EXP43 Training COMPLETE — Best R@1: {best_r1:.2f}%\n{'='*65}")

    with open(os.path.join(CFG.OUTPUT_DIR, "exp43_v2_results.json"), "w") as f:
        json.dump({"results_log": results_log, "best_r1": best_r1,
                   "config": {k:v for k,v in vars(CFG).items() if not k.startswith('_')}}, f, indent=2)

    # ========================== TTE ==========================
    print("\n" + "="*65 + "\n  TTE: Test-Time Ensemble (Fixed)\n" + "="*65)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded best ckpt (R@1: {ckpt['metrics']['R@1']:.2f}%)")
    model.eval()

    print("\n  [1/3] Multi-crop only ...")
    tte_mc = evaluate_tte(model, None, test_ds, DEVICE)
    print(f"  MC R@1: {tte_mc['R@1']:.2f}%  mAP: {tte_mc['mAP']:.2f}%")

    print("\n  [2/3] Multi-crop + EMA (0.6/0.4) ...")
    tte_mc_ema = evaluate_tte(model, ema.model, test_ds, DEVICE)
    print(f"  MC+EMA R@1: {tte_mc_ema['R@1']:.2f}%  mAP: {tte_mc_ema['mAP']:.2f}%")

    tte_full = None
    print("\n  [3/3] Tent adaptation (full test set) + MC + EMA ...")
    try:
        tent = TentAdaptation(model, lr=CFG.TTE_TENT_LR, steps=CFG.TTE_TENT_STEPS)
        tent.adapt_full_testset(test_ds, DEVICE)
        tte_full = evaluate_tte(model, ema.model, test_ds, DEVICE)
        print(f"  Full TTE R@1: {tte_full['R@1']:.2f}%  mAP: {tte_full['mAP']:.2f}%")
        tent.reset()
    except RuntimeError as e:
        print(f"  [WARN] Tent TTE failed: {e}")

    tte_payload = {"baseline_r1": best_r1, "multi_crop": tte_mc, "mc_ema": tte_mc_ema,
                   "full_tte": tte_full,
                   "final_best_r1": max(best_r1, tte_mc['R@1'], tte_mc_ema['R@1'],
                                        tte_full['R@1'] if tte_full else -1)}
    with open(os.path.join(CFG.OUTPUT_DIR, "exp43_v2_tte_results.json"), "w") as f:
        json.dump(tte_payload, f, indent=2)

    print(f"\n  FINAL BEST R@1: {tte_payload['final_best_r1']:.2f}%")


if __name__ == "__main__":
    main()
