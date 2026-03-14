# =============================================================================
# EVAL: Satellite → Drone  (+ Drone → Satellite for reference)
# =============================================================================
# University-1652 — Loads best checkpoint and evaluates BOTH directions:
#   1) Drone → Satellite  (query=drone, gallery=satellite) — same as training
#   2) Satellite → Drone  (query=satellite, gallery=drone) — NEW
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
    DATA_ROOT       = "/kaggle/input/datasets/chinguyeen/university-1652/University-1652"
    OUTPUT_DIR      = "/kaggle/working"
    CHECKPOINT      = "/kaggle/input/datasets/hunhtrungkit/university/spdgeo_dpe_mar_university1652_best.pth"

    IMG_SIZE        = 448
    N_PARTS         = 8
    PART_DIM        = 256
    EMBED_DIM       = 512
    CLUSTER_TEMP    = 0.07
    UNFREEZE_BLOCKS = 4

    SEED            = 42
    MASK_RATIO      = 0.30

    # TTE
    TTE_CROP_SIZES   = [392, 448, 504]
    TTE_CROP_WEIGHTS = [0.25, 0.50, 0.25]
    TTE_TENT_STEPS   = 3
    TTE_TENT_LR      = 1e-4

CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); torch.backends.cudnn.benchmark = True


# =============================================================================
# TRANSFORMS
# =============================================================================
def get_transforms(mode="test", img_size=None):
    sz = img_size or CFG.IMG_SIZE
    return transforms.Compose([
        transforms.Resize((sz, sz)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])


# =============================================================================
# TEST DATASET — scans both query_drone/gallery_satellite AND
#                query_satellite/gallery_drone (reverse direction)
# =============================================================================
class University1652BidirectionalTestData:
    def __init__(self, root):
        self.root = root

        # === Direction 1: Drone → Satellite ===
        # Query: test/query_drone    Gallery: test/gallery_satellite
        self.d2s_query_samples, self.d2s_query_labels = self._scan_dir(
            os.path.join(root, "test", "query_drone"))
        self.d2s_gallery_samples, self.d2s_gallery_labels = self._scan_dir(
            os.path.join(root, "test", "gallery_satellite"))

        print(f"  [D→S] Query: {len(self.d2s_query_samples)} drone, "
              f"Gallery: {len(self.d2s_gallery_samples)} sat")

        # === Direction 2: Satellite → Drone ===
        # Query: test/query_satellite    Gallery: test/gallery_drone
        query_sat_dir = os.path.join(root, "test", "query_satellite")
        gallery_drone_dir = os.path.join(root, "test", "gallery_drone")

        if os.path.isdir(query_sat_dir) and os.path.isdir(gallery_drone_dir):
            self.s2d_query_samples, self.s2d_query_labels = self._scan_dir(query_sat_dir)
            self.s2d_gallery_samples, self.s2d_gallery_labels = self._scan_dir(gallery_drone_dir)
            self.has_s2d = True
            print(f"  [S→D] Query: {len(self.s2d_query_samples)} sat, "
                  f"Gallery: {len(self.s2d_gallery_samples)} drone")
        else:
            # Fallback: reverse the D→S data
            # Query = satellite gallery, Gallery = drone queries
            self.s2d_query_samples  = self.d2s_gallery_samples
            self.s2d_query_labels   = self.d2s_gallery_labels
            self.s2d_gallery_samples = self.d2s_query_samples
            self.s2d_gallery_labels  = self.d2s_query_labels
            self.has_s2d = False
            print(f"  [S→D] No dedicated folders — using reversed D→S data")
            print(f"  [S→D] Query: {len(self.s2d_query_samples)} sat, "
                  f"Gallery: {len(self.s2d_gallery_samples)} drone")

    def _scan_dir(self, base_dir):
        samples, labels = [], []
        if not os.path.isdir(base_dir):
            return samples, labels
        for bid in sorted(os.listdir(base_dir)):
            bp = os.path.join(base_dir, bid)
            if not os.path.isdir(bp): continue
            for f in sorted(os.listdir(bp)):
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    samples.append(os.path.join(bp, f))
                    labels.append(int(bid))
        return samples, labels


# =============================================================================
# MODEL ARCHITECTURE (identical to full_university1652.py for checkpoint compat)
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
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return features['x_norm_patchtokens'], features['x_norm_clstoken'], (H, W)


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

        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)

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
        return self.extract_embedding(x)


# =============================================================================
# EMA Model
# =============================================================================
class EMAModel:
    def __init__(self, model, decay=0.996):
        self.decay = decay
        self.model = copy.deepcopy(model); self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        return self.model.extract_embedding(x)


# =============================================================================
# TTE helpers
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


# =============================================================================
# METRIC COMPUTATION
# =============================================================================
def compute_metrics(query_feats, query_labels, gallery_feats, gallery_labels):
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
def extract_features_batch(model, img_paths, device, batch_size=64,
                           use_tte=False, ema_model=None, tent=None, desc=""):
    """Extract features for a list of image paths."""
    model.eval()
    test_tf = get_transforms("test")
    cs = CFG.TTE_CROP_SIZES; cw = CFG.TTE_CROP_WEIGHTS

    if use_tte:
        # TTE: per-image multi-crop (slower but more accurate)
        feats = []
        for img_path in tqdm(img_paths, desc=f"{desc} (TTE)"):
            try: pil = Image.open(img_path).convert('RGB')
            except: pil = Image.new('RGB', (448, 448), (128, 128, 128))
            feat = multi_crop_extract(model, pil, cs, cw, device, tent=tent)
            if ema_model is not None:
                ema_feat = multi_crop_extract(ema_model, pil, cs, cw, device)
                feat = F.normalize(0.5 * feat + 0.5 * ema_feat, dim=-1)
            feats.append(feat.cpu())
        return torch.cat(feats)
    else:
        # Standard: batched inference
        feats = []
        batch_imgs = []
        for i, img_path in enumerate(tqdm(img_paths, desc=desc, leave=False)):
            try: batch_imgs.append(test_tf(Image.open(img_path).convert('RGB')))
            except: batch_imgs.append(test_tf(Image.new('RGB', (448, 448), (128, 128, 128))))
            if len(batch_imgs) == batch_size or i == len(img_paths) - 1:
                feats.append(model.extract_embedding(torch.stack(batch_imgs).to(device)).cpu())
                batch_imgs = []
        return torch.cat(feats)


# =============================================================================
# BIDIRECTIONAL EVALUATION
# =============================================================================
def evaluate_direction(query_feats, query_labels, gallery_feats, gallery_labels,
                       direction_name, query_desc, gallery_desc):
    """Evaluate and print one retrieval direction."""
    metrics = compute_metrics(query_feats, query_labels, gallery_feats, gallery_labels)
    print(f"\n  ▶ {direction_name}  (Query: {metrics['N']} {query_desc} | "
          f"Gallery: {len(gallery_feats)} {gallery_desc})")
    print(f"    R@1: {metrics['R@1']:6.2f}%  R@5: {metrics['R@5']:6.2f}%  "
          f"R@10: {metrics['R@10']:6.2f}%  mAP: {metrics['mAP']:6.2f}%")
    return metrics


def run_eval(model, test_data, device, label="Baseline",
             use_tte=False, ema_model=None, tent=None):
    """Run bidirectional evaluation."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # === Direction 1: Drone → Satellite ===
    d2s_q_feats = extract_features_batch(
        model, test_data.d2s_query_samples, device,
        use_tte=use_tte, ema_model=ema_model, tent=tent, desc="D→S Query")
    d2s_q_labels = torch.tensor(test_data.d2s_query_labels)

    d2s_g_feats = extract_features_batch(
        model, test_data.d2s_gallery_samples, device,
        use_tte=use_tte, ema_model=ema_model, tent=tent, desc="D→S Gallery")
    d2s_g_labels = torch.tensor(test_data.d2s_gallery_labels)

    d2s = evaluate_direction(d2s_q_feats, d2s_q_labels, d2s_g_feats, d2s_g_labels,
                             "DRONE → SATELLITE", "drone", "satellite")

    # === Direction 2: Satellite → Drone ===
    if test_data.has_s2d:
        # Dedicated S→D folders exist
        s2d_q_feats = extract_features_batch(
            model, test_data.s2d_query_samples, device,
            use_tte=use_tte, ema_model=ema_model, tent=tent, desc="S→D Query")
        s2d_q_labels = torch.tensor(test_data.s2d_query_labels)

        s2d_g_feats = extract_features_batch(
            model, test_data.s2d_gallery_samples, device,
            use_tte=use_tte, ema_model=ema_model, tent=tent, desc="S→D Gallery")
        s2d_g_labels = torch.tensor(test_data.s2d_gallery_labels)
    else:
        # Reuse D→S features in reverse
        s2d_q_feats  = d2s_g_feats;  s2d_q_labels  = d2s_g_labels
        s2d_g_feats  = d2s_q_feats;  s2d_g_labels  = d2s_q_labels

    s2d = evaluate_direction(s2d_q_feats, s2d_q_labels, s2d_g_feats, s2d_g_labels,
                             "SATELLITE → DRONE", "satellite", "drone")

    print(f"{'='*70}")
    return {'drone_to_sat': d2s, 'sat_to_drone': s2d}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EVAL: Bidirectional Retrieval — Drone↔Satellite")
    print(f"  Dataset: University-1652 | Device: {DEVICE}")
    print(f"  Checkpoint: {os.path.basename(CFG.CHECKPOINT)}")
    print("=" * 65)

    # Load test data
    test_data = University1652BidirectionalTestData(CFG.DATA_ROOT)

    # Build model
    print("\nBuilding model …")
    # Determine num_classes from checkpoint
    ckpt = torch.load(CFG.CHECKPOINT, map_location=DEVICE, weights_only=False)
    num_classes = ckpt['model_state_dict']['classifier.weight'].shape[0]
    print(f"  Detected num_classes = {num_classes}")

    model = SPDGeoDPEMARModel(num_classes).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded checkpoint — training R@1: {ckpt['metrics']['R@1']:.2f}%")
    is_ema = ckpt.get('is_ema', False)
    print(f"  Checkpoint type: {'EMA' if is_ema else 'Student'}")
    model.eval()

    # ═══════════════════════════════════════════════════════════════
    #  1) Baseline (single-crop)
    # ═══════════════════════════════════════════════════════════════
    results_baseline = run_eval(model, test_data, DEVICE, label="[1/4] Baseline (single-crop)")

    # ═══════════════════════════════════════════════════════════════
    #  2) TTE: Multi-crop only
    # ═══════════════════════════════════════════════════════════════
    results_mc = run_eval(model, test_data, DEVICE,
                          label=f"[2/4] TTE: Multi-crop {CFG.TTE_CROP_SIZES}",
                          use_tte=True)

    # ═══════════════════════════════════════════════════════════════
    #  3) TTE: Multi-crop + EMA
    # ═══════════════════════════════════════════════════════════════
    ema = EMAModel(model, decay=0.996)
    results_ema = run_eval(model, test_data, DEVICE,
                           label="[3/4] TTE: Multi-crop + EMA",
                           use_tte=True, ema_model=ema.model)

    # ═══════════════════════════════════════════════════════════════
    #  4) TTE: Full (Multi-crop + EMA + Tent)
    # ═══════════════════════════════════════════════════════════════
    tent = TentAdaptation(model, lr=CFG.TTE_TENT_LR, steps=CFG.TTE_TENT_STEPS)
    results_full = run_eval(model, test_data, DEVICE,
                            label="[4/4] TTE: Full (MC + EMA + Tent)",
                            use_tte=True, ema_model=ema.model, tent=tent)

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
    print(f"  SUMMARY — University-1652 — ALL METHODS, BOTH DIRECTIONS")
    print(f"{'='*75}")
    print(f"  {'Method':<30s}  {'D→S R@1':>8s}  {'D→S mAP':>8s}  {'S→D R@1':>8s}  {'S→D mAP':>8s}")
    print(f"  {'-'*70}")
    for name, label in [('baseline', 'Baseline (single-crop)'),
                         ('multi_crop', 'Multi-crop'),
                         ('mc_ema', 'MC + EMA'),
                         ('full_tte', 'MC + EMA + Tent')]:
        r = all_results[name]
        d2s = r['drone_to_sat']
        s2d = r['sat_to_drone']
        print(f"  {label:<30s}  {d2s['R@1']:7.2f}%  {d2s['mAP']:7.2f}%  "
              f"{s2d['R@1']:7.2f}%  {s2d['mAP']:7.2f}%")
    print(f"{'='*75}")

    best_d2s = max(r['drone_to_sat']['R@1'] for r in all_results.values())
    best_s2d = max(r['sat_to_drone']['R@1'] for r in all_results.values())
    print(f"\n  ★ Best Drone→Sat R@1: {best_d2s:.2f}%")
    print(f"  ★ Best Sat→Drone R@1: {best_s2d:.2f}%")

    # Save results
    output_path = os.path.join(CFG.OUTPUT_DIR, 'university1652_bidirectional_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()
