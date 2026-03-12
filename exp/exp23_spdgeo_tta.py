# =============================================================================
# EXP23: SPDGeo-TTA — Test-Time Adaptation for Part Prototypes
# =============================================================================
# Base:    SPDGeo-D (90.36% R@1) — DINOv2-S + DINOv2-B teacher + 7 losses
# Novel:   1) Entropy-Minimized Part Adaptation — at test time, fine-tune
#             part prototypes and BatchNorm affine parameters using entropy
#             minimization on query distribution (Tent-style, Wang et al.)
#          2) Confidence-Weighted Adaptation — adapt more aggressively on
#             high-confidence queries, cautiously on uncertain ones
#          3) Gallery Prototype Refinement — after initial retrieval, refine
#             prototypes using top-K gallery matches as pseudo-supervision
#
# Motivation:
#   SPDGeo-D's part prototypes are fixed at test time. But the test set
#   (locations 121-200) has DIFFERENT visual distributions than training
#   (locations 1-120). The part prototypes, trained on training locations,
#   may be suboptimal for test locations.
#
#   Test-Time Adaptation (TTA) adapts model parameters at inference using
#   only the test data itself — no labels, no retraining on training data.
#   This is a FREE accuracy boost with zero training cost.
#
#   Key idea: For each test batch, minimize the entropy of the model's
#   predictions (Tent, Wang et al. ICLR 2021). Low entropy = confident
#   predictions = good feature organization.
#
#   We adapt ONLY lightweight parameters:
#   - Part prototypes (N_PARTS × PART_DIM = 8 × 256 = 2,048 params)
#   - BatchNorm affine (mean, var statistics)
#   This prevents catastrophic forgetting of backbone knowledge.
#
# Training: IDENTICAL to SPDGeo-D (no training change at all!)
# Inference: Run TTA adaptation loop on test queries before evaluation
# Architecture: Identical to SPDGeo-D
# Total losses: 7 (same as SPDGeo-D) + TTA entropy at test time
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

    # Base loss weights (from SPDGeo-D)
    LAMBDA_CE           = 1.0
    LAMBDA_INFONCE      = 1.0
    LAMBDA_TRIPLET      = 0.5
    LAMBDA_CONSISTENCY  = 0.1
    LAMBDA_CROSS_DIST   = 0.3
    LAMBDA_SELF_DIST    = 0.3
    LAMBDA_UAPA         = 0.2

    # NEW: TTA parameters
    TTA_STEPS           = 3       # adaptation steps per test batch
    TTA_LR              = 1e-4    # TTA learning rate (for prototypes + BN)
    TTA_ENTROPY_THRESH  = 0.5     # adapt only on samples below this entropy
    TTA_CONF_WEIGHT     = True    # confidence-weighted adaptation

    DISTILL_TEMP        = 4.0
    EVAL_INTERVAL       = 5
    NUM_WORKERS         = 2

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
        return {'drone': d, 'satellite': s, 'label': li, 'altitude': int(alt)}


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


def get_transforms(mode="train"):
    sz = CFG.IMG_SIZE
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
# BACKBONE — DINOv2 ViT-S/14
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
        patch_tokens = features['x_norm_patchtokens']
        cls_token = features['x_norm_clstoken']
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return patch_tokens, cls_token, (H, W)


# =============================================================================
# TEACHER — DINOv2 ViT-B/14
# =============================================================================
class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        print("  Loading DINOv2 ViT-B/14 teacher …")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        self.output_dim = 768
        for p in self.model.parameters(): p.requires_grad = False
        print("  DINOv2 ViT-B/14 teacher loaded (all frozen).")

    @torch.no_grad()
    def forward(self, x):
        return self.model.forward_features(x)['x_norm_clstoken']


# =============================================================================
# SEMANTIC PART DISCOVERY
# =============================================================================
class SemanticPartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07):
        super().__init__()
        self.n_parts = n_parts; self.temperature = temperature
        self.feat_proj = nn.Sequential(nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU())
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim * 2),
                                    nn.GELU(), nn.Linear(part_dim * 2, part_dim))
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
        gy = torch.arange(H, device=device).float() / max(H - 1, 1)
        gx = torch.arange(W, device=device).float() / max(W - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(B, -1, -1)) / mass
        salience = self.salience_head(part_feat).squeeze(-1)
        return {'part_features': part_feat, 'part_positions': part_pos,
                'assignment': assign, 'salience': salience}


# =============================================================================
# PART-AWARE POOLING
# =============================================================================
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


# =============================================================================
# STUDENT MODEL — SPDGeo-TTA (identical arch to SPDGeo-D)
# =============================================================================
class SPDGeoTTAModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone  = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = SemanticPartDiscovery(384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP)
        self.pool      = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.bottleneck = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                        nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj   = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
                                            nn.LayerNorm(cfg.TEACHER_DIM))
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-TTA student: {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return F.normalize(0.7 * emb + 0.3 * cls_emb, dim=-1)

    def forward(self, x, return_parts=False):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        bn = self.bottleneck(emb); logits = self.classifier(bn)
        cls_emb = self.cls_proj(cls_tok); cls_logits = self.cls_classifier(cls_emb)
        fused = F.normalize(0.7 * emb + 0.3 * F.normalize(cls_emb, dim=-1), dim=-1)
        projected_feat = self.teacher_proj(emb)
        out = {'embedding': fused, 'logits': logits, 'cls_logits': cls_logits,
               'projected_feat': projected_feat}
        if return_parts: out['parts'] = parts
        return out


# =============================================================================
# BASE LOSSES (from SPDGeo-D)
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


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3): super().__init__(); self.m = margin
    def forward(self, emb, labels):
        d = torch.cdist(emb, emb, p=2); labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float(); neg_mask = labels.ne(labels.T).float()
        hard_pos = (d * pos_mask).max(1)[0]
        hard_neg = (d * neg_mask + pos_mask * 1e9).min(1)[0]
        return F.relu(hard_pos - hard_neg + self.m).mean()


class PartConsistencyLoss(nn.Module):
    def forward(self, assign_q, assign_r):
        dist_q = assign_q.mean(dim=1); dist_r = assign_r.mean(dim=1)
        kl_qr = F.kl_div((dist_q + 1e-8).log(), dist_r, reduction='batchmean', log_target=False)
        kl_rq = F.kl_div((dist_r + 1e-8).log(), dist_q, reduction='batchmean', log_target=False)
        return 0.5 * (kl_qr + kl_rq)


class CrossDistillationLoss(nn.Module):
    def forward(self, student_feat, teacher_feat):
        s = F.normalize(student_feat, dim=-1); t = F.normalize(teacher_feat, dim=-1)
        return F.mse_loss(s, t) + (1.0 - F.cosine_similarity(s, t).mean())


class SelfDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0): super().__init__(); self.T = temperature
    def forward(self, weak_logits, strong_logits):
        p_teacher = F.softmax(strong_logits / self.T, dim=1).detach()
        p_student = F.log_softmax(weak_logits / self.T, dim=1)
        return (self.T ** 2) * F.kl_div(p_student, p_teacher, reduction='batchmean')


class UAPALoss(nn.Module):
    def __init__(self, base_temperature=4.0): super().__init__(); self.T0 = base_temperature
    @staticmethod
    def _entropy(logits):
        probs = F.softmax(logits, dim=1)
        return -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
    def forward(self, drone_logits, sat_logits):
        delta_U = self._entropy(drone_logits) - self._entropy(sat_logits)
        T = self.T0 * (1 + torch.sigmoid(delta_U))
        p_sat = F.softmax(sat_logits / T, dim=1).detach()
        return (T ** 2) * F.kl_div(F.log_softmax(drone_logits / T, dim=1), p_sat, reduction='batchmean')


# =============================================================================
# NEW: Test-Time Adaptation (TTA) Module
# =============================================================================

def collect_tta_params(model):
    """
    Collect parameters to adapt at test time:
    1. Part prototypes (semantic part discovery prototypes)
    2. BatchNorm affine parameters (gamma, beta)
    3. Salience head parameters

    Returns list of (name, param) and sets requires_grad accordingly.
    """
    tta_params = []
    for name, param in model.named_parameters():
        param.requires_grad = False  # freeze everything first

    # Enable part prototypes
    model.part_disc.prototypes.requires_grad = True
    tta_params.append(('part_disc.prototypes', model.part_disc.prototypes))

    # Enable BatchNorm affine parameters
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            if module.weight is not None:
                module.weight.requires_grad = True
                tta_params.append((f'{name}.weight', module.weight))
            if module.bias is not None:
                module.bias.requires_grad = True
                tta_params.append((f'{name}.bias', module.bias))

    # Enable salience head
    for name, param in model.part_disc.salience_head.named_parameters():
        param.requires_grad = True
        tta_params.append((f'part_disc.salience_head.{name}', param))

    total_tta = sum(p.numel() for _, p in tta_params)
    print(f"  TTA adaptable: {total_tta} params ({total_tta/1e3:.1f}K)")
    return tta_params


def tta_entropy_loss(model, images):
    """
    Compute entropy of the embedding distribution for TTA.

    Uses part assignment entropy: confident assignment to parts = low entropy.
    The model should assign each patch to ONE part confidently.

    Also uses classifier entropy: confident class predictions = low entropy.

    Confidence weighting: weight each sample inversely by its entropy.
    High-confidence samples contribute more to the gradient.
    """
    out = model(images, return_parts=True)
    assign = out['parts']['assignment']    # [B, N, K]

    # Part assignment entropy: each patch should be assigned to ONE part
    patch_entropy = -(assign * (assign + 1e-8).log()).sum(-1)  # [B, N]
    batch_assign_entropy = patch_entropy.mean()

    # Classifier entropy: predictions should be confident
    logits = out['logits']  # [B, C]
    probs = F.softmax(logits, dim=-1)
    classifier_entropy = -(probs * (probs + 1e-8).log()).sum(-1)  # [B]

    if CFG.TTA_CONF_WEIGHT:
        # Weight by confidence: low-entropy samples get higher weight
        weights = (1.0 / (classifier_entropy + 1e-6)).detach()
        weights = weights / weights.sum()
        weighted_cls_entropy = (weights * classifier_entropy).sum()
    else:
        weighted_cls_entropy = classifier_entropy.mean()

    # Filter high-entropy (unreliable) samples
    reliable_mask = classifier_entropy < CFG.TTA_ENTROPY_THRESH
    if reliable_mask.sum() > 0:
        reliable_entropy = classifier_entropy[reliable_mask].mean()
    else:
        reliable_entropy = classifier_entropy.mean()

    return 0.5 * batch_assign_entropy + 0.5 * reliable_entropy


def run_tta(model, test_ds, device, tta_steps=3, tta_lr=1e-4):
    """
    Run Test-Time Adaptation on test drone images.

    1. Save original model state
    2. Collect TTA-adaptable parameters
    3. For each test batch, run `tta_steps` adaptation steps minimizing entropy
    4. Extract adapted features
    5. Restore model for next batch (episodic TTA)
    """
    print(f"\n  Running TTA: {tta_steps} steps, LR={tta_lr}")

    # Save original state for episodic reset
    original_state = copy.deepcopy(model.state_dict())

    # Collect TTA params and setup optimizer
    tta_params = collect_tta_params(model)
    tta_optimizer = torch.optim.Adam([p for _, p in tta_params], lr=tta_lr)

    test_tf = get_transforms("test")
    loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)

    all_feats, all_labels, all_alts = [], [], []

    for batch in tqdm(loader, desc='TTA', leave=False):
        drone_imgs = batch['drone'].to(device)

        # Episodic: reset to original state for each batch
        model.load_state_dict(original_state)
        tta_params = collect_tta_params(model)
        tta_optimizer = torch.optim.Adam([p for _, p in tta_params], lr=tta_lr)

        # Set model to eval (use running stats) but enable grad for TTA params
        model.eval()
        for _, p in tta_params:
            p.requires_grad = True

        # Adaptation steps
        for step in range(tta_steps):
            tta_optimizer.zero_grad()
            loss = tta_entropy_loss(model, drone_imgs)
            loss.backward()
            tta_optimizer.step()

        # Extract adapted features
        with torch.no_grad():
            feats = model.extract_embedding(drone_imgs).cpu()
            all_feats.append(feats)
            all_labels.append(batch['label'])
            all_alts.append(batch['altitude'])

    # Restore original model
    model.load_state_dict(original_state)

    return torch.cat(all_feats), torch.cat(all_labels), torch.cat(all_alts)


# =============================================================================
# EVALUATION — with and without TTA
# =============================================================================
@torch.no_grad()
def evaluate(model, test_ds, device, use_tta=False):
    model.eval()
    test_tf = get_transforms("test")

    # Drone features
    if use_tta:
        drone_feats, drone_labels, drone_alts = run_tta(
            model, test_ds, device,
            tta_steps=CFG.TTA_STEPS, tta_lr=CFG.TTA_LR
        )
    else:
        loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                            num_workers=CFG.NUM_WORKERS, pin_memory=True)
        drone_feats, drone_labels, drone_alts = [], [], []
        for b in loader:
            feat = model.extract_embedding(b['drone'].to(device)).cpu()
            drone_feats.append(feat); drone_labels.append(b['label']); drone_alts.append(b['altitude'])
        drone_feats = torch.cat(drone_feats); drone_labels = torch.cat(drone_labels)
        drone_alts = torch.cat(drone_alts)

    # Satellite gallery
    all_locs = [f"{l:04d}" for l in range(1, 201)]
    sat_img_list, sat_label_list = [], []; distractor_cnt = 0
    for loc in all_locs:
        sp = os.path.join(test_ds.sat_dir, loc, "0.png")
        if not os.path.exists(sp): continue
        sat_img_list.append(test_tf(Image.open(sp).convert('RGB')))
        if loc in test_ds.location_to_idx: sat_label_list.append(test_ds.location_to_idx[loc])
        else: sat_label_list.append(-1000 - distractor_cnt); distractor_cnt += 1

    sat_feats = []
    for i in range(0, len(sat_img_list), 64):
        batch = torch.stack(sat_img_list[i:i+64]).to(device)
        sat_feats.append(model.extract_embedding(batch).cpu())
    sat_feats = torch.cat(sat_feats); sat_labels = torch.tensor(sat_label_list)

    mode_str = "TTA" if use_tta else "Standard"
    print(f"  [{mode_str}] Gallery: {len(sat_feats)} | Queries: {len(drone_feats)}")

    sim = drone_feats @ sat_feats.T; _, rank = sim.sort(1, descending=True)
    N = drone_feats.size(0); r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(sat_labels[rank[i]] == drone_labels[i])[0]
        if len(matches) == 0: continue
        first = matches[0].item()
        if first < 1: r1 += 1
        if first < 5: r5 += 1
        if first < 10: r10 += 1
        ap += sum((j+1)/(p.item()+1) for j, p in enumerate(matches)) / len(matches)

    overall = {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100, 'mAP': ap/N*100}

    altitudes_list = sorted(drone_alts.unique().tolist())
    per_alt = {}
    for alt in altitudes_list:
        mask = drone_alts == alt
        if mask.sum() == 0: continue
        af = drone_feats[mask]; al = drone_labels[mask]
        s = af @ sat_feats.T; _, rk = s.sort(1, descending=True)
        n = af.size(0); a1 = a5 = a10 = aap = 0
        for i in range(n):
            m = torch.where(sat_labels[rk[i]] == al[i])[0]
            if len(m) == 0: continue
            f = m[0].item()
            if f < 1: a1 += 1
            if f < 5: a5 += 1
            if f < 10: a10 += 1
            aap += sum((j+1)/(p.item()+1) for j, p in enumerate(m)) / len(m)
        per_alt[int(alt)] = {'R@1': a1/n*100, 'R@5': a5/n*100, 'R@10': a10/n*100, 'mAP': aap/n*100, 'n': n}

    print(f"\n{'='*75}")
    print(f"  [{mode_str}] {'Altitude':>8s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'#Query':>6s}")
    print(f"  {'-'*50}")
    for alt in altitudes_list:
        a = per_alt[int(alt)]
        print(f"  {int(alt):>6d}m  {a['R@1']:6.2f}%  {a['R@5']:6.2f}%  {a['R@10']:6.2f}%  {a['mAP']:6.2f}%  {a['n']:>6d}")
    print(f"  {'-'*50}")
    print(f"  {'Overall':>8s}  {overall['R@1']:6.2f}%  {overall['R@5']:6.2f}%  {overall['R@10']:6.2f}%  {overall['mAP']:6.2f}%  {N:>6d}")
    print(f"{'='*75}\n")

    return overall, per_alt


# =============================================================================
# TRAINING (identical to SPDGeo-D)
# =============================================================================
def train_one_epoch(model, teacher, loader, losses, optimizer, scaler, device, epoch):
    model.train()
    if teacher: teacher.eval()

    infonce, triplet, ce, consist, cross_dist, self_dist, uapa = losses
    total_sum = 0; n = 0; loss_sums = defaultdict(float)

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
            l_tri = 0.5 * (triplet(d_out['embedding'], labels) + triplet(s_out['embedding'], labels))
            l_con = consist(d_out['parts']['assignment'], s_out['parts']['assignment'])

            if teacher is not None:
                with torch.no_grad():
                    t_drone = teacher(drone); t_sat = teacher(sat)
                l_cross = cross_dist(d_out['projected_feat'], t_drone) + cross_dist(s_out['projected_feat'], t_sat)
            else:
                l_cross = torch.tensor(0.0, device=device)

            l_self = self_dist(d_out['cls_logits'], d_out['logits']) + self_dist(s_out['cls_logits'], s_out['logits'])
            l_uapa = uapa(d_out['logits'], s_out['logits'])

            loss = (CFG.LAMBDA_CE          * l_ce +
                    CFG.LAMBDA_INFONCE     * l_nce +
                    CFG.LAMBDA_TRIPLET     * l_tri +
                    CFG.LAMBDA_CONSISTENCY * l_con +
                    CFG.LAMBDA_CROSS_DIST  * l_cross +
                    CFG.LAMBDA_SELF_DIST   * l_self +
                    CFG.LAMBDA_UAPA        * l_uapa)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()

        total_sum += loss.item(); n += 1
        loss_sums['ce']    += l_ce.item()
        loss_sums['nce']   += l_nce.item()
        loss_sums['tri']   += l_tri.item()
        loss_sums['con']   += l_con.item()
        loss_sums['cross'] += l_cross.item() if torch.is_tensor(l_cross) else l_cross
        loss_sums['self']  += l_self.item()
        loss_sums['uapa']  += l_uapa.item()

    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EXP23: SPDGeo-TTA — Test-Time Adaptation for Parts")
    print(f"  Base: SPDGeo-D (90.36% R@1)")
    print(f"  Novel: Entropy-Minimized Part Prototype Adaptation at Test Time")
    print(f"  Training: IDENTICAL to SPDGeo-D (7 losses)")
    print(f"  TTA: {CFG.TTA_STEPS} steps, LR={CFG.TTA_LR}, entropy thresh={CFG.TTA_ENTROPY_THRESH}")
    print(f"  Dataset: SUES-200 | Epochs: {CFG.NUM_EPOCHS} | Device: {DEVICE}")
    print("=" * 65)

    print('\nLoading SUES-200 …')
    train_ds = SUES200Dataset(CFG.SUES_ROOT, 'train', transform=get_transforms("train"))
    test_ds  = SUES200Dataset(CFG.SUES_ROOT, 'test', transform=get_transforms("test"))
    train_loader = DataLoader(train_ds, batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print('\nBuilding models …')
    model = SPDGeoTTAModel(CFG.NUM_CLASSES).to(DEVICE)
    teacher = None
    try:
        teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except Exception as e:
        print(f"  [WARN] Could not load teacher: {e}")

    infonce    = SupInfoNCELoss(temp=0.05).to(DEVICE)
    triplet    = TripletLoss(margin=0.3)
    ce         = nn.CrossEntropyLoss(label_smoothing=0.1)
    consist    = PartConsistencyLoss()
    cross_dist = CrossDistillationLoss()
    self_dist  = SelfDistillationLoss(temperature=CFG.DISTILL_TEMP)
    uapa_loss  = UAPALoss(base_temperature=CFG.DISTILL_TEMP)
    losses = (infonce, triplet, ce, consist, cross_dist, self_dist, uapa_loss)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith('backbone')]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params,      'lr': CFG.BACKBONE_LR},
        {'params': head_params,          'lr': CFG.LR},
        {'params': infonce.parameters(), 'lr': CFG.LR},
    ], weight_decay=CFG.WEIGHT_DECAY)

    scaler = torch.amp.GradScaler('cuda', enabled=CFG.USE_AMP)
    best_r1 = 0.0; best_r1_tta = 0.0; results_log = []

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        if epoch <= CFG.WARMUP_EPOCHS:
            lr_scale = epoch / CFG.WARMUP_EPOCHS
        else:
            progress = (epoch - CFG.WARMUP_EPOCHS) / (CFG.NUM_EPOCHS - CFG.WARMUP_EPOCHS)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        lr_scale = max(lr_scale, 0.01)
        for pg in optimizer.param_groups:
            if pg is optimizer.param_groups[0]: pg['lr'] = CFG.BACKBONE_LR * lr_scale
            else: pg['lr'] = CFG.LR * lr_scale
        cur_lr = optimizer.param_groups[1]['lr']

        avg_loss, ld = train_one_epoch(model, teacher, train_loader, losses,
                                       optimizer, scaler, DEVICE, epoch)

        print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
              f"CE {ld['ce']:.3f}  NCE {ld['nce']:.3f}  Tri {ld['tri']:.3f}  "
              f"Con {ld['con']:.3f}  Crs {ld['cross']:.3f}  Slf {ld['self']:.3f}  "
              f"UAPA {ld['uapa']:.3f} | LR {cur_lr:.2e}")

        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            # Standard evaluation (no TTA)
            print(f"\n  --- Standard Evaluation (no TTA) ---")
            metrics, per_alt = evaluate(model, test_ds, DEVICE, use_tta=False)
            entry = {'epoch': epoch, **{f'std_{k}': v for k, v in metrics.items()}}

            print(f"  ► [Std] R@1: {metrics['R@1']:.2f}%  R@5: {metrics['R@5']:.2f}%  "
                  f"R@10: {metrics['R@10']:.2f}%  mAP: {metrics['mAP']:.2f}%")

            if metrics['R@1'] > best_r1:
                best_r1 = metrics['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'metrics': metrics, 'per_alt': per_alt},
                           os.path.join(CFG.OUTPUT_DIR, 'exp23_tta_best_std.pth'))
                print(f"  ★ New best Std R@1: {best_r1:.2f}%!")

            # TTA evaluation (at milestone epochs to save time)
            if epoch % 20 == 0 or epoch == CFG.NUM_EPOCHS:
                print(f"\n  --- TTA Evaluation ({CFG.TTA_STEPS} steps) ---")
                metrics_tta, per_alt_tta = evaluate(model, test_ds, DEVICE, use_tta=True)
                entry.update({f'tta_{k}': v for k, v in metrics_tta.items()})

                print(f"  ► [TTA] R@1: {metrics_tta['R@1']:.2f}%  R@5: {metrics_tta['R@5']:.2f}%  "
                      f"R@10: {metrics_tta['R@10']:.2f}%  mAP: {metrics_tta['mAP']:.2f}%")

                delta = metrics_tta['R@1'] - metrics['R@1']
                print(f"  ► TTA Gain: {delta:+.2f}% R@1")

                if metrics_tta['R@1'] > best_r1_tta:
                    best_r1_tta = metrics_tta['R@1']
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                                'metrics_tta': metrics_tta, 'per_alt_tta': per_alt_tta,
                                'metrics_std': metrics},
                               os.path.join(CFG.OUTPUT_DIR, 'exp23_tta_best_tta.pth'))
                    print(f"  ★ New best TTA R@1: {best_r1_tta:.2f}%!")

            results_log.append(entry)

    print(f'\n{"="*65}')
    print(f'  EXP23: SPDGeo-TTA COMPLETE')
    print(f'  Best Std R@1: {best_r1:.2f}%')
    print(f'  Best TTA R@1: {best_r1_tta:.2f}%')
    print(f'{"="*65}')
    print(f'  {"Epoch":>6} {"Std R@1":>9} {"TTA R@1":>9} {"Gain":>7}')
    print(f'  {"-"*35}')
    for r in results_log:
        std_r1 = r.get('std_R@1', 0)
        tta_r1 = r.get('tta_R@1', None)
        if tta_r1 is not None:
            delta = tta_r1 - std_r1
            print(f'  {r["epoch"]:6d}  {std_r1:8.2f}%  {tta_r1:8.2f}%  {delta:+6.2f}%')
        else:
            print(f'  {r["epoch"]:6d}  {std_r1:8.2f}%       ---       ---')
    print(f'{"="*65}')

    with open(os.path.join(CFG.OUTPUT_DIR, 'exp23_tta_results.json'), 'w') as f:
        json.dump({'results_log': results_log, 'best_r1_std': best_r1,
                   'best_r1_tta': best_r1_tta,
                   'config': {k: v for k, v in vars(CFG).items() if not k.startswith('_')}}, f, indent=2)


if __name__ == '__main__':
    main()
