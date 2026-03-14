# =============================================================================
# EVAL: Satellite → Drone  (+ Drone → Satellite for reference)
# =============================================================================
# Loads the best EXP35 checkpoint and evaluates BOTH retrieval directions:
#   1) Drone → Satellite  (query=drone, gallery=satellite)  — same as training
#   2) Satellite → Drone  (query=satellite, gallery=drone)  — NEW
#
# Includes TTE (multi-crop + EMA + Tent) for both directions.
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

    SEED            = 42

    # TTE
    TTE_CROP_SIZES   = [280, 336, 392]
    TTE_CROP_WEIGHTS = [0.25, 0.50, 0.25]
    TTE_TENT_STEPS   = 3
    TTE_TENT_LR      = 1e-4

    # Mask Recon (needed to instantiate model)
    MASK_RATIO       = 0.30

CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


# =============================================================================
# DATASET  (test-only, supports both views)
# =============================================================================
class SUES200TestDataset:
    """Lightweight container for test set — no DataLoader needed."""
    def __init__(self, root, altitudes=None):
        self.root = root
        self.altitudes = altitudes or CFG.ALTITUDES
        self.drone_dir = os.path.join(root, CFG.DRONE_DIR)
        self.sat_dir   = os.path.join(root, CFG.SAT_DIR)
        self.test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]
        self.location_to_idx = {l: i for i, l in enumerate(self.test_locs)}

        # Drone samples: (path, loc_str, alt_str)
        self.drone_samples = []
        for loc in self.test_locs:
            for alt in self.altitudes:
                ad = os.path.join(self.drone_dir, loc, alt)
                if not os.path.isdir(ad): continue
                for img in sorted(os.listdir(ad)):
                    if img.endswith(('.png', '.jpg', '.jpeg')):
                        self.drone_samples.append((os.path.join(ad, img), loc, alt))

        # Satellite samples: all 200 locs (120 train as distractors + 80 test)
        self.sat_samples = []
        all_locs = [f"{l:04d}" for l in range(1, 201)]
        distractor_cnt = 0
        for loc in all_locs:
            sp = os.path.join(self.sat_dir, loc, "0.png")
            if not os.path.exists(sp): continue
            if loc in self.location_to_idx:
                self.sat_samples.append((sp, loc, self.location_to_idx[loc]))
            else:
                self.sat_samples.append((sp, loc, -1000 - distractor_cnt))
                distractor_cnt += 1

        print(f"  Test set: {len(self.drone_samples)} drone imgs, "
              f"{len(self.sat_samples)} sat imgs (incl. {distractor_cnt} distractors)")


# =============================================================================
# TRANSFORMS
# =============================================================================
def get_transforms(mode="test", img_size=None):
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
# MODEL ARCHITECTURE  (identical to full_sues200_80loc.py for checkpoint compat)
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
        patch_tokens = features['x_norm_patchtokens']
        cls_token = features['x_norm_clstoken']
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return patch_tokens, cls_token, (H, W)


class DeepAltitudeFiLM(nn.Module):
    def __init__(self, num_altitudes=4, feat_dim=256):
        super().__init__()
        self.num_altitudes = num_altitudes
        self.gamma = nn.Parameter(torch.ones(num_altitudes, feat_dim))
        self.beta  = nn.Parameter(torch.zeros(num_altitudes, feat_dim))

    def forward(self, feat, alt_idx=None):
        if alt_idx is None:
            gamma = self.gamma.mean(dim=0, keepdim=True)
            beta  = self.beta.mean(dim=0, keepdim=True)
            return feat * gamma.unsqueeze(0) + beta.unsqueeze(0)
        else:
            gamma = self.gamma[alt_idx]
            beta  = self.beta[alt_idx]
            return feat * gamma.unsqueeze(1) + beta.unsqueeze(1)


class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07,
                 num_altitudes=4):
        super().__init__()
        self.n_parts = n_parts; self.temperature = temperature
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU()
        )
        self.altitude_film = DeepAltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim * 2),
            nn.GELU(), nn.Linear(part_dim * 2, part_dim)
        )
        self.salience_head = nn.Sequential(
            nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, patch_features, spatial_hw, alt_idx=None):
        B, N, _ = patch_features.shape; H, W = spatial_hw
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
        gy = torch.arange(H, device=device).float() / max(H - 1, 1)
        gx = torch.arange(W, device=device).float() / max(W - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(B, -1, -1)) / mass
        salience = self.salience_head(part_feat).squeeze(-1)
        return {'part_features': part_feat, 'part_positions': part_pos,
                'assignment': assign, 'salience': salience,
                'projected_patches': feat}


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
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        combined = torch.cat([part_emb, cls_emb], dim=-1)
        alpha = torch.sigmoid(self.gate(combined))
        fused = alpha * part_emb + (1 - alpha) * cls_emb
        return F.normalize(fused, dim=-1)


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
        pass  # not needed for eval


class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, embedding, alt_target):
        pass  # not needed for eval


class SPDGeoDPEAMARModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone  = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = AltitudeAwarePartDiscovery(
            384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES
        )
        self.pool      = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)

        self.bottleneck = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                        nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj   = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
                                            nn.LayerNorm(cfg.TEACHER_DIM))

        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)
        self.alt_pred   = AltitudePredictionHead(cfg.EMBED_DIM)

    def extract_embedding(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)

    def extract_with_assignment(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb), parts['assignment']

    def forward(self, x, alt_idx=None, return_parts=False):
        return self.extract_embedding(x, alt_idx=alt_idx)


# =============================================================================
# EMA Model
# =============================================================================
class EMAModel:
    def __init__(self, model, decay=0.996):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x, alt_idx=None):
        return self.model.extract_embedding(x, alt_idx=alt_idx)


# =============================================================================
# TTE helpers
# =============================================================================
class TentAdaptation:
    def __init__(self, model, lr=1e-4, steps=3):
        self.model = model
        self.lr = lr; self.steps = steps
        self.orig_prototypes = model.part_disc.prototypes.data.clone()

    def adapt_and_extract(self, images, alt_idx=None):
        self.model.part_disc.prototypes.data.copy_(self.orig_prototypes)
        self.model.eval()
        self.model.part_disc.prototypes.requires_grad_(True)
        optimizer = torch.optim.Adam([self.model.part_disc.prototypes], lr=self.lr)
        for _ in range(self.steps):
            optimizer.zero_grad()
            _, assignment = self.model.extract_with_assignment(images, alt_idx=alt_idx)
            assign_probs = assignment.mean(dim=1)
            entropy = -(assign_probs * (assign_probs + 1e-8).log()).sum(dim=-1).mean()
            entropy.backward()
            optimizer.step()
        with torch.no_grad():
            emb = self.model.extract_embedding(images, alt_idx=alt_idx)
        self.model.part_disc.prototypes.requires_grad_(False)
        self.model.part_disc.prototypes.data.copy_(self.orig_prototypes)
        return emb


def multi_crop_extract(model, image_pil, crop_sizes, crop_weights, device,
                       alt_idx=None, tent=None):
    all_feats = []
    for sz, w in zip(crop_sizes, crop_weights):
        tf = get_transforms("test", img_size=sz)
        img_t = tf(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            if tent is not None:
                feat = tent.adapt_and_extract(img_t, alt_idx=alt_idx)
            else:
                feat = model.extract_embedding(img_t, alt_idx=alt_idx)
        all_feats.append(feat * w)
    return F.normalize(sum(all_feats), dim=-1)


# =============================================================================
# METRIC COMPUTATION (shared by both directions)
# =============================================================================
def compute_metrics(query_feats, query_labels, gallery_feats, gallery_labels):
    """Compute R@1, R@5, R@10, mAP given query/gallery features and labels."""
    sim = query_feats @ gallery_feats.T
    _, rank = sim.sort(1, descending=True)
    N = query_feats.size(0)
    r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(gallery_labels[rank[i]] == query_labels[i])[0]
        if len(matches) == 0: continue
        first = matches[0].item()
        if first < 1: r1 += 1
        if first < 5: r5 += 1
        if first < 10: r10 += 1
        ap += sum((j+1)/(p.item()+1) for j, p in enumerate(matches)) / len(matches)
    return {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100, 'mAP': ap/N*100, 'N': N}


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
@torch.no_grad()
def extract_all_features(model, test_data, device, use_tte=False, ema_model=None, tent=None):
    """Extract features for both drone and satellite images."""
    model.eval()
    test_tf = get_transforms("test")
    crop_sizes = CFG.TTE_CROP_SIZES
    crop_weights = CFG.TTE_CROP_WEIGHTS

    # --- Drone features ---
    drone_feats, drone_labels, drone_alts = [], [], []
    desc = "Drone (TTE)" if use_tte else "Drone"

    for dp, loc, alt in tqdm(test_data.drone_samples, desc=desc):
        try: d_pil = Image.open(dp).convert('RGB')
        except: d_pil = Image.new('RGB', (336, 336), (128, 128, 128))

        alt_idx_val = CFG.ALT_TO_IDX.get(alt, 0)
        li = test_data.location_to_idx[loc]

        if use_tte:
            alt_t = torch.tensor([alt_idx_val], device=device)
            feat = multi_crop_extract(model, d_pil, crop_sizes, crop_weights,
                                      device, alt_idx=alt_t, tent=tent)
            if ema_model is not None:
                ema_feat = multi_crop_extract(ema_model, d_pil, crop_sizes, crop_weights,
                                              device, alt_idx=alt_t)
                feat = F.normalize(0.5 * feat + 0.5 * ema_feat, dim=-1)
        else:
            img_t = test_tf(d_pil).unsqueeze(0).to(device)
            alt_t = torch.tensor([alt_idx_val], device=device)
            feat = model.extract_embedding(img_t, alt_idx=alt_t)

        drone_feats.append(feat.cpu())
        drone_labels.append(li)
        drone_alts.append(int(alt))

    drone_feats  = torch.cat(drone_feats)
    drone_labels = torch.tensor(drone_labels)
    drone_alts   = torch.tensor(drone_alts)

    # --- Satellite features ---
    sat_feats, sat_labels = [], []
    desc = "Satellite (TTE)" if use_tte else "Satellite"

    for sp, loc, label in tqdm(test_data.sat_samples, desc=desc):
        try: s_pil = Image.open(sp).convert('RGB')
        except: s_pil = Image.new('RGB', (336, 336), (128, 128, 128))

        if use_tte:
            feat = multi_crop_extract(model, s_pil, crop_sizes, crop_weights,
                                      device, alt_idx=None)
            if ema_model is not None:
                ema_feat = multi_crop_extract(ema_model, s_pil, crop_sizes, crop_weights,
                                              device, alt_idx=None)
                feat = F.normalize(0.5 * feat + 0.5 * ema_feat, dim=-1)
        else:
            img_t = test_tf(s_pil).unsqueeze(0).to(device)
            feat = model.extract_embedding(img_t, alt_idx=None)

        sat_feats.append(feat.cpu())
        sat_labels.append(label)

    sat_feats  = torch.cat(sat_feats)
    sat_labels = torch.tensor(sat_labels)

    return (drone_feats, drone_labels, drone_alts), (sat_feats, sat_labels)


# =============================================================================
# BIDIRECTIONAL EVALUATION
# =============================================================================
def evaluate_bidirectional(drone_data, sat_data, direction_label="Baseline"):
    """Evaluate both Drone→Sat and Sat→Drone retrieval."""
    drone_feats, drone_labels, drone_alts = drone_data
    sat_feats, sat_labels = sat_data

    # ──────────────────────────────────────────────────────────────
    # Direction 1: Drone → Satellite
    # Query = drone, Gallery = satellite (200 locations)
    # ──────────────────────────────────────────────────────────────
    d2s = compute_metrics(drone_feats, drone_labels, sat_feats, sat_labels)

    # Per-altitude breakdown for D→S
    d2s_per_alt = {}
    for alt in sorted(drone_alts.unique().tolist()):
        mask = drone_alts == alt
        if mask.sum() == 0: continue
        d2s_per_alt[int(alt)] = compute_metrics(
            drone_feats[mask], drone_labels[mask], sat_feats, sat_labels)

    # ──────────────────────────────────────────────────────────────
    # Direction 2: Satellite → Drone
    # Query = satellite (80 test locs only), Gallery = drone (all test drone imgs)
    #
    # For S→D: a satellite query is "correct" if any drone image
    # from the same location is retrieved.  This is a 1-to-many match.
    # ──────────────────────────────────────────────────────────────

    # Filter satellite to test locations only (label >= 0)
    test_sat_mask = sat_labels >= 0
    test_sat_feats  = sat_feats[test_sat_mask]
    test_sat_labels = sat_labels[test_sat_mask]

    s2d = compute_metrics(test_sat_feats, test_sat_labels, drone_feats, drone_labels)

    # Per-altitude breakdown for S→D  (gallery filtered by altitude)
    s2d_per_alt = {}
    for alt in sorted(drone_alts.unique().tolist()):
        mask = drone_alts == alt
        if mask.sum() == 0: continue
        s2d_per_alt[int(alt)] = compute_metrics(
            test_sat_feats, test_sat_labels,
            drone_feats[mask], drone_labels[mask])

    # ──────────────────────────────────────────────────────────────
    # Print results
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  {direction_label} — BIDIRECTIONAL RESULTS")
    print(f"{'='*75}")

    # ── Drone → Satellite ──
    print(f"\n  ▶ DRONE → SATELLITE  (Query: {d2s['N']} drone | Gallery: {len(sat_feats)} sat)")
    print(f"  {'Altitude':>8s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'#Query':>6s}")
    print(f"  {'-'*55}")
    for alt in sorted(d2s_per_alt.keys()):
        a = d2s_per_alt[alt]
        print(f"  {alt:>6d}m  {a['R@1']:6.2f}%  {a['R@5']:6.2f}%  {a['R@10']:6.2f}%  {a['mAP']:6.2f}%  {a['N']:>6d}")
    print(f"  {'-'*55}")
    print(f"  {'Overall':>8s}  {d2s['R@1']:6.2f}%  {d2s['R@5']:6.2f}%  {d2s['R@10']:6.2f}%  {d2s['mAP']:6.2f}%  {d2s['N']:>6d}")

    # ── Satellite → Drone ──
    print(f"\n  ▶ SATELLITE → DRONE  (Query: {s2d['N']} sat | Gallery: {len(drone_feats)} drone)")
    print(f"  {'Altitude':>8s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'#Query':>6s}")
    print(f"  {'-'*55}")
    for alt in sorted(s2d_per_alt.keys()):
        a = s2d_per_alt[alt]
        print(f"  {alt:>6d}m  {a['R@1']:6.2f}%  {a['R@5']:6.2f}%  {a['R@10']:6.2f}%  {a['mAP']:6.2f}%  {s2d['N']:>6d}")
    print(f"  {'-'*55}")
    print(f"  {'Overall':>8s}  {s2d['R@1']:6.2f}%  {s2d['R@5']:6.2f}%  {s2d['R@10']:6.2f}%  {s2d['mAP']:6.2f}%  {s2d['N']:>6d}")

    print(f"{'='*75}\n")

    return {
        'drone_to_sat': {'overall': d2s, 'per_alt': d2s_per_alt},
        'sat_to_drone': {'overall': s2d, 'per_alt': s2d_per_alt},
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EVAL: Bidirectional Retrieval — Drone↔Satellite")
    print(f"  Dataset: SUES-200 | Device: {DEVICE}")
    print(f"  Checkpoint: exp35_dpea_ga_best.pth")
    print("=" * 65)

    # Load test data
    test_data = SUES200TestDataset(CFG.SUES_ROOT)

    # Build model & load checkpoint
    print("\nBuilding model …")
    model = SPDGeoDPEAMARModel(CFG.NUM_CLASSES).to(DEVICE)

    best_ckpt_path = "/kaggle/input/datasets/chisnguyen/sues200/exp35_dpea_ga_best.pth"
    if not os.path.exists(best_ckpt_path):
        print(f"  [ERROR] Checkpoint not found: {best_ckpt_path}")
        print("  Run full_sues200_80loc.py first to train the model.")
        return

    ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded checkpoint — training R@1: {ckpt['metrics']['R@1']:.2f}%")
    is_ema = ckpt.get('is_ema', False)
    print(f"  Checkpoint type: {'EMA' if is_ema else 'Student'}")
    model.eval()

    # ═══════════════════════════════════════════════════════════════
    #  1) Baseline evaluation (single-crop, no ensemble)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  [1/4] Baseline (single-crop)")
    print(f"{'='*65}")
    drone_data, sat_data = extract_all_features(model, test_data, DEVICE, use_tte=False)
    results_baseline = evaluate_bidirectional(drone_data, sat_data, "Baseline (single-crop)")

    # ═══════════════════════════════════════════════════════════════
    #  2) TTE: Multi-crop only
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  [2/4] TTE: Multi-crop {CFG.TTE_CROP_SIZES}")
    print(f"{'='*65}")
    drone_data_mc, sat_data_mc = extract_all_features(
        model, test_data, DEVICE, use_tte=True, ema_model=None, tent=None)
    results_mc = evaluate_bidirectional(drone_data_mc, sat_data_mc, "Multi-crop TTE")

    # ═══════════════════════════════════════════════════════════════
    #  3) TTE: Multi-crop + EMA ensemble
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  [3/4] TTE: Multi-crop + EMA ensemble")
    print(f"{'='*65}")
    ema = EMAModel(model, decay=0.996)
    # Load EMA weights if checkpoint has them, otherwise use copy of student
    drone_data_ema, sat_data_ema = extract_all_features(
        model, test_data, DEVICE, use_tte=True, ema_model=ema.model, tent=None)
    results_ema = evaluate_bidirectional(drone_data_ema, sat_data_ema, "Multi-crop + EMA")

    # ═══════════════════════════════════════════════════════════════
    #  4) TTE: Full (Multi-crop + EMA + Tent)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  [4/4] TTE: Full (Multi-crop + EMA + Tent)")
    print(f"{'='*65}")
    tent = TentAdaptation(model, lr=CFG.TTE_TENT_LR, steps=CFG.TTE_TENT_STEPS)
    drone_data_full, sat_data_full = extract_all_features(
        model, test_data, DEVICE, use_tte=True, ema_model=ema.model, tent=tent)
    results_full = evaluate_bidirectional(drone_data_full, sat_data_full, "Full TTE")

    # ═══════════════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════
    all_results = {
        'baseline': results_baseline,
        'multi_crop': results_mc,
        'mc_ema': results_ema,
        'full_tte': results_full,
    }

    print(f"\n{'='*75}")
    print(f"  SUMMARY — ALL METHODS, BOTH DIRECTIONS")
    print(f"{'='*75}")
    print(f"  {'Method':<30s}  {'D→S R@1':>8s}  {'D→S mAP':>8s}  {'S→D R@1':>8s}  {'S→D mAP':>8s}")
    print(f"  {'-'*70}")
    for name, label in [('baseline', 'Baseline (single-crop)'),
                         ('multi_crop', 'Multi-crop'),
                         ('mc_ema', 'MC + EMA'),
                         ('full_tte', 'MC + EMA + Tent')]:
        r = all_results[name]
        d2s = r['drone_to_sat']['overall']
        s2d = r['sat_to_drone']['overall']
        print(f"  {label:<30s}  {d2s['R@1']:7.2f}%  {d2s['mAP']:7.2f}%  "
              f"{s2d['R@1']:7.2f}%  {s2d['mAP']:7.2f}%")
    print(f"{'='*75}")

    # Best results
    best_d2s = max(r['drone_to_sat']['overall']['R@1'] for r in all_results.values())
    best_s2d = max(r['sat_to_drone']['overall']['R@1'] for r in all_results.values())
    print(f"\n  ★ Best Drone→Sat R@1: {best_d2s:.2f}%")
    print(f"  ★ Best Sat→Drone R@1: {best_s2d:.2f}%")

    # Save results
    # Convert per_alt keys to string for JSON
    def jsonify(results):
        out = {}
        for method, data in results.items():
            out[method] = {}
            for direction in ['drone_to_sat', 'sat_to_drone']:
                out[method][direction] = {
                    'overall': data[direction]['overall'],
                    'per_alt': {str(k): v for k, v in data[direction]['per_alt'].items()}
                }
        return out

    output_path = os.path.join(CFG.OUTPUT_DIR, 'exp35_bidirectional_results.json')
    with open(output_path, 'w') as f:
        json.dump(jsonify(all_results), f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()
