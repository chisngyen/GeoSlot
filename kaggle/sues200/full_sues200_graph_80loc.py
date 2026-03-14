# =============================================================================
# EXP36: SPARC-G — Explicit+Implicit Graph Matching
# =============================================================================
# Base:    EXP35 (SPDGeo-DPEA-MAR — FiLM + MAR + AltPred + Diversity)
# NEW:     PartGraphEncoder + CrossViewGraphMatchingLoss
#
# Architecture: SPDGeo-DPE + DeepAltitudeFiLM + MAR + AltPred + DiversityReg
#             + PartGraphEncoder (self-attn + explicit edges)
#             + CrossViewGraphMatchingLoss (Sinkhorn OT + structure warp)
# Train: 120 locations (standard), Eval: 200-loc full gallery
# EMA decay: 0.996 (proven faster convergence)
# Total losses: 14 (12 base + GraphFeatureMatch + GraphStructure)
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
    TRAIN_LOCS      = list(range(1, 121))     # 120 supervised drone+sat locs
    TEST_LOCS       = list(range(121, 201))    # 80 test locations
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

    # Base loss weights
    LAMBDA_CE           = 1.0
    LAMBDA_INFONCE      = 1.0
    LAMBDA_CONSISTENCY  = 0.1
    LAMBDA_CROSS_DIST   = 0.3
    LAMBDA_SELF_DIST    = 0.3
    LAMBDA_UAPA         = 0.2

    # DPE components
    LAMBDA_PROXY        = 0.5
    PROXY_MARGIN        = 0.1
    PROXY_ALPHA         = 32
    LAMBDA_EMA_DIST     = 0.2
    EMA_DECAY           = 0.996

    # Altitude consistency (from EXP27)
    LAMBDA_ALT_CONSIST  = 0.2

    # Masked Part Reconstruction (from EXP34)
    MASK_RATIO          = 0.30
    LAMBDA_MASK_RECON   = 0.3
    RECON_WARMUP        = 10

    # Altitude Prediction (from EXP34)
    LAMBDA_ALT_PRED     = 0.15

    # Prototype Diversity (from EXP34)
    LAMBDA_DIVERSITY    = 0.05

    # Graph Reasoning (NEW in EXP36)
    N_GRAPH_HEADS        = 4
    GRAPH_EDGE_DIM       = 64
    LAMBDA_GRAPH_MATCH   = 0.3
    GRAPH_SINKHORN_ITERS = 5
    GRAPH_SINKHORN_EPS   = 0.05
    GRAPH_STRUCT_WEIGHT  = 0.5

    DISTILL_TEMP        = 4.0
    EVAL_INTERVAL       = 5
    NUM_WORKERS         = 2

    # TTE (Test-Time Ensemble) — inference-only boost
    TTE_CROP_SIZES      = [280, 336, 392]   # must be multiples of 14
    TTE_CROP_WEIGHTS    = [0.25, 0.50, 0.25]
    TTE_TENT_STEPS      = 3
    TTE_TENT_LR         = 1e-4

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
        return {'drone': d, 'satellite': s, 'label': li,
                'altitude': int(alt), 'alt_idx': alt_idx, 'alt_norm': alt_norm}


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
# Deep Altitude FiLM — conditions INSIDE part discovery
# =============================================================================
class DeepAltitudeFiLM(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on altitude.
    Applied to feat_proj output BEFORE prototype similarity, so altitude
    influences the part assignment step itself.
    For satellite images (no altitude), uses mean(γ, β) across all altitudes.
    """
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


# =============================================================================
# SEMANTIC PART DISCOVERY — with Deep Altitude FiLM
# =============================================================================
class AltitudeAwarePartDiscovery(nn.Module):
    """
    SemanticPartDiscovery with DeepAltitudeFiLM injected before prototype sim.
    Pipeline: patch → feat_proj → FiLM(alt) → proto_sim → assign → aggregate
    """
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
# DYNAMIC FUSION GATE
# =============================================================================
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


# =============================================================================
# NEW: Altitude Prediction Head (from EXP34)
# =============================================================================
class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, embedding, alt_target):
        pred = self.head(embedding).squeeze(-1)
        return F.smooth_l1_loss(pred, alt_target)


# =============================================================================
# NEW: Prototype Diversity Loss (from EXP34)
# =============================================================================
class PrototypeDiversityLoss(nn.Module):
    def forward(self, prototypes):
        P = F.normalize(prototypes, dim=-1)
        sim = P @ P.T
        K = sim.size(0)
        mask = 1 - torch.eye(K, device=sim.device)
        return (sim * mask).abs().sum() / (K * (K - 1))


# =============================================================================
# NEW (EXP36): Part Graph Encoder — explicit + implicit graph reasoning
# =============================================================================
class PartGraphEncoder(nn.Module):
    """
    Graph Neural Network for discovered semantic parts.
    Each part = node (with features + 2D position).

    Implicit edges:  Multi-head self-attention between parts
    Explicit edges:  Pairwise edge features from relative differences + positions
    Output:          Graph-enhanced part features  [B, K, D]
                     Structure matrix              [B, K, K]
    """
    def __init__(self, part_dim=256, n_heads=4, edge_dim=64):
        super().__init__()
        self.pos_proj = nn.Linear(2, part_dim)
        self.self_attn = nn.MultiheadAttention(part_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(part_dim)
        self.ffn = nn.Sequential(
            nn.Linear(part_dim, part_dim * 2), nn.GELU(),
            nn.Linear(part_dim * 2, part_dim)
        )
        self.norm2 = nn.LayerNorm(part_dim)
        # Explicit edge feature projection
        self.edge_proj = nn.Sequential(
            nn.Linear(part_dim + 2, edge_dim), nn.GELU(),
            nn.Linear(edge_dim, edge_dim)
        )

    def forward(self, part_features, part_positions):
        B, K, D = part_features.shape
        # Position-aware self-attention (implicit edges)
        pos_emb = self.pos_proj(part_positions)
        x = part_features + pos_emb
        attn_out, attn_weights = self.self_attn(x, x, x)
        x = self.norm1(part_features + attn_out)  # residual from original
        x = self.norm2(x + self.ffn(x))
        # Explicit pairwise edge features
        diff = x.unsqueeze(2) - x.unsqueeze(1)       # [B, K, K, D]
        pos_diff = (part_positions.unsqueeze(2) -
                    part_positions.unsqueeze(1))       # [B, K, K, 2]
        edge_input = torch.cat([diff, pos_diff], dim=-1)  # [B, K, K, D+2]
        edge_features = self.edge_proj(edge_input)    # [B, K, K, edge_dim]
        # Structure matrix: edge strength (L2 norm)
        structure = edge_features.norm(dim=-1)        # [B, K, K]
        return x, structure, attn_weights


# =============================================================================
# NEW (EXP36): Cross-View Graph Matching Loss
# =============================================================================
class CrossViewGraphMatchingLoss(nn.Module):
    """
    Matches drone part-graph to satellite part-graph using:
      Feature cost:    Sinkhorn OT on cosine distance between parts
      Structure cost:  Warp sat structure via transport plan, compare with drone
    """
    def __init__(self, part_dim=256, sinkhorn_iters=5, eps=0.05,
                 struct_weight=0.5):
        super().__init__()
        self.feat_proj = nn.Linear(part_dim, part_dim)
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.struct_weight = struct_weight

    def sinkhorn(self, cost):
        log_alpha = -cost / self.eps
        for _ in range(self.sinkhorn_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        return log_alpha.exp()

    def forward(self, drone_parts, sat_parts, drone_struct, sat_struct):
        B, K, D = drone_parts.shape
        # Feature cost: cosine distance
        d_proj = F.normalize(self.feat_proj(drone_parts), dim=-1)
        s_proj = F.normalize(self.feat_proj(sat_parts), dim=-1)
        C_feat = 1 - torch.bmm(d_proj, s_proj.transpose(1, 2))  # [B, K, K]
        # Sinkhorn optimal transport
        P = self.sinkhorn(C_feat)  # [B, K, K] soft assignment
        # Feature matching loss: optimal transport cost
        feat_loss = (P * C_feat).sum(dim=(1, 2)).mean()
        # Structure consistency: warp sat structure via P, compare with drone
        # P @ sat_struct @ P^T should ≈ drone_struct
        warped_sat = torch.bmm(torch.bmm(P, sat_struct), P.transpose(1, 2))
        struct_loss = F.mse_loss(warped_sat, drone_struct)
        return feat_loss + self.struct_weight * struct_loss


# =============================================================================
# STUDENT MODEL — SPARC-G (SPDGeo-DPEA-MAR + Graph)
# =============================================================================
class SPDGeoDPEAMARModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone  = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = AltitudeAwarePartDiscovery(
            384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES
        )
        # NEW (EXP36): Graph encoder between part discovery and pooling
        self.graph_encoder = PartGraphEncoder(
            cfg.PART_DIM, cfg.N_GRAPH_HEADS, cfg.GRAPH_EDGE_DIM
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

        # EXP34 modules
        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)
        self.alt_pred   = AltitudePredictionHead(cfg.EMBED_DIM)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPARC-G student: {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        graph_parts, _, _ = self.graph_encoder(parts['part_features'], parts['part_positions'])
        emb = self.pool(graph_parts, parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)

    def extract_with_assignment(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        graph_parts, _, _ = self.graph_encoder(parts['part_features'], parts['part_positions'])
        emb = self.pool(graph_parts, parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb), parts['assignment']

    def forward(self, x, alt_idx=None, return_parts=False):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        graph_parts, structure, attn_w = self.graph_encoder(
            parts['part_features'], parts['part_positions'])
        emb = self.pool(graph_parts, parts['salience'])
        bn = self.bottleneck(emb); logits = self.classifier(bn)
        cls_emb_raw = self.cls_proj(cls_tok)
        cls_logits = self.cls_classifier(cls_emb_raw)
        cls_emb_norm = F.normalize(cls_emb_raw, dim=-1)
        fused = self.fusion_gate(emb, cls_emb_norm)
        projected_feat = self.teacher_proj(emb)
        out = {'embedding': fused, 'logits': logits, 'cls_logits': cls_logits,
               'projected_feat': projected_feat, 'part_emb': emb, 'cls_emb': cls_emb_norm,
               'graph_parts': graph_parts, 'structure': structure}
        if return_parts: out['parts'] = parts
        return out


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
    def update(self, model):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def forward(self, x, alt_idx=None):
        return self.model.extract_embedding(x, alt_idx=alt_idx)


# =============================================================================
# BASE LOSSES
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
# DPE LOSSES
# =============================================================================
class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, embed_dim, margin=0.1, alpha=32):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.margin = margin; self.alpha = alpha; self.num_classes = num_classes

    def forward(self, embeddings, labels):
        P = F.normalize(self.proxies, dim=-1)
        sim = embeddings @ P.T
        one_hot = F.one_hot(labels, self.num_classes).float()
        pos_exp = torch.exp(-self.alpha * (sim * one_hot - self.margin)) * one_hot
        P_plus = one_hot.sum(0); has_pos = P_plus > 0
        pos_term = torch.log(1 + pos_exp.sum(0))
        pos_loss = pos_term[has_pos].mean() if has_pos.sum() > 0 else torch.tensor(0.0, device=embeddings.device)
        neg_mask = 1 - one_hot
        neg_exp = torch.exp(self.alpha * (sim * neg_mask + self.margin)) * neg_mask
        neg_loss = torch.log(1 + neg_exp.sum(0)).mean()
        return pos_loss + neg_loss


class EMADistillationLoss(nn.Module):
    def forward(self, student_emb, ema_emb):
        return (1 - F.cosine_similarity(student_emb, ema_emb)).mean()


# =============================================================================
# Altitude Consistency Loss
# =============================================================================
class AltitudeConsistencyLoss(nn.Module):
    """
    Ensures embeddings of same location at different altitudes remain close.
    Prevents DeepFiLM from pushing altitude-specific views apart.
    """
    def forward(self, embeddings, labels, altitudes):
        B = embeddings.size(0)
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)
        lbl = labels.view(-1, 1); alt = altitudes.view(-1, 1)
        same_loc = lbl.eq(lbl.T); diff_alt = alt.ne(alt.T)
        mask = same_loc & diff_alt
        if mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        emb_norm = F.normalize(embeddings, dim=-1)
        cos_sim = emb_norm @ emb_norm.T
        cos_dist = 1 - cos_sim
        loss = (cos_dist * mask.float()).sum() / mask.float().sum().clamp(min=1)
        return loss


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model, test_ds, device):
    model.eval()
    test_tf = get_transforms("test")

    loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)
    drone_feats, drone_labels, drone_alts = [], [], []
    for b in loader:
        alt_idx = b['alt_idx'].to(device)
        feat = model.extract_embedding(b['drone'].to(device), alt_idx=alt_idx).cpu()
        drone_feats.append(feat); drone_labels.append(b['label']); drone_alts.append(b['altitude'])
    drone_feats = torch.cat(drone_feats); drone_labels = torch.cat(drone_labels)
    drone_alts = torch.cat(drone_alts)

    # Satellite gallery: all 200 locations, no altitude → mean FiLM
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
        sat_feats.append(model.extract_embedding(batch, alt_idx=None).cpu())
    sat_feats = torch.cat(sat_feats); sat_labels = torch.tensor(sat_label_list)

    print(f"  Gallery: {len(sat_feats)} sat imgs | Queries: {len(drone_feats)} drone imgs")

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

    altitudes = sorted(drone_alts.unique().tolist())
    per_alt = {}
    for alt in altitudes:
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
    print(f"  Gallery: {len(sat_feats)} satellite images | Queries: {len(drone_feats)} drone images")
    print(f"{'='*75}")
    print(f"  {'Altitude':>8s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'#Query':>6s}")
    print(f"  {'-'*50}")
    for alt in altitudes:
        a = per_alt[int(alt)]
        print(f"  {int(alt):>6d}m  {a['R@1']:6.2f}%  {a['R@5']:6.2f}%  {a['R@10']:6.2f}%  {a['mAP']:6.2f}%  {a['n']:>6d}")
    print(f"  {'-'*50}")
    print(f"  {'Overall':>8s}  {overall['R@1']:6.2f}%  {overall['R@5']:6.2f}%  {overall['R@10']:6.2f}%  {overall['mAP']:6.2f}%  {N:>6d}")
    print(f"{'='*75}\n")

    return overall, per_alt


# =============================================================================
# TEST-TIME ENSEMBLE (TTE)
# =============================================================================

class TentAdaptation:
    """Test-Time Entropy Minimization for Part Prototypes (Tent, Wang et al. 2021)."""
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
    """Extract features at multiple resolutions, weighted average."""
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


def get_transforms_for_size(sz):
    return transforms.Compose([
        transforms.Resize((sz, sz)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])


@torch.no_grad()
def evaluate_tte(model, ema_model, test_ds, device, tent=None):
    """TTE evaluation: multi-crop + EMA ensemble + optional Tent."""
    model.eval()
    if ema_model is not None: ema_model.eval()

    crop_sizes = CFG.TTE_CROP_SIZES
    crop_weights = CFG.TTE_CROP_WEIGHTS

    # Drone features (multi-crop + ensemble)
    print("  TTE: Extracting drone features (multi-crop) …")
    drone_feats, drone_labels, drone_alts = [], [], []
    for i, sample in enumerate(tqdm(test_ds.samples, desc="TTE Drone")):
        dp, sp, li, alt = sample
        try: d_pil = Image.open(dp).convert('RGB')
        except: d_pil = Image.new('RGB', (336, 336), (128, 128, 128))
        alt_idx_val = CFG.ALT_TO_IDX.get(alt, 0)
        alt_t = torch.tensor([alt_idx_val], device=device)

        feat = multi_crop_extract(model, d_pil, crop_sizes, crop_weights,
                                  device, alt_idx=alt_t, tent=tent)
        if ema_model is not None:
            ema_feat = multi_crop_extract(ema_model, d_pil, crop_sizes, crop_weights,
                                          device, alt_idx=alt_t)
            feat = F.normalize(0.5 * feat + 0.5 * ema_feat, dim=-1)

        drone_feats.append(feat.cpu()); drone_labels.append(li); drone_alts.append(int(alt))

    drone_feats = torch.cat(drone_feats)
    drone_labels = torch.tensor(drone_labels)
    drone_alts = torch.tensor(drone_alts)

    # Satellite features (multi-crop, no altitude)
    print("  TTE: Extracting satellite features (multi-crop) …")
    all_locs = [f"{l:04d}" for l in range(1, 201)]
    sat_feats, sat_labels = [], []; distractor_cnt = 0
    for loc in tqdm(all_locs, desc="TTE Satellite"):
        sp = os.path.join(test_ds.sat_dir, loc, "0.png")
        if not os.path.exists(sp): continue
        try: s_pil = Image.open(sp).convert('RGB')
        except: s_pil = Image.new('RGB', (336, 336), (128, 128, 128))

        feat = multi_crop_extract(model, s_pil, crop_sizes, crop_weights,
                                  device, alt_idx=None)
        if ema_model is not None:
            ema_feat = multi_crop_extract(ema_model, s_pil, crop_sizes, crop_weights,
                                          device, alt_idx=None)
            feat = F.normalize(0.5 * feat + 0.5 * ema_feat, dim=-1)

        sat_feats.append(feat.cpu())
        if loc in test_ds.location_to_idx: sat_labels.append(test_ds.location_to_idx[loc])
        else: sat_labels.append(-1000 - distractor_cnt); distractor_cnt += 1

    sat_feats = torch.cat(sat_feats); sat_labels = torch.tensor(sat_labels)

    # Compute metrics
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
    return overall


# =============================================================================
# TRAINING
# =============================================================================
def train_one_epoch(model, teacher, ema, loader, losses, new_losses, optimizer,
                    scaler, device, epoch):
    model.train()
    if teacher: teacher.eval()

    infonce, ce, consist, cross_dist, self_dist, uapa = losses
    proxy_anchor, ema_dist, alt_consist, diversity_loss, graph_match_loss = new_losses

    total_sum = 0; n = 0; loss_sums = defaultdict(float)
    recon_active = epoch >= CFG.RECON_WARMUP

    for batch in tqdm(loader, desc=f'Ep{epoch:3d}', leave=False):
        drone    = batch['drone'].to(device)
        sat      = batch['satellite'].to(device)
        labels   = batch['label'].to(device)
        alt_idx  = batch['alt_idx'].to(device)
        alts     = batch['altitude'].to(device)
        alt_norm = batch['alt_norm'].to(device).float()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            d_out = model(drone, alt_idx=alt_idx, return_parts=True)
            s_out = model(sat, alt_idx=None, return_parts=True)

            # === 6 Base losses ===
            l_ce = (ce(d_out['logits'], labels) + ce(s_out['logits'], labels))
            l_ce += 0.3 * (ce(d_out['cls_logits'], labels) + ce(s_out['cls_logits'], labels))
            l_nce = infonce(d_out['embedding'], s_out['embedding'], labels)
            l_con = consist(d_out['parts']['assignment'], s_out['parts']['assignment'])

            if teacher is not None:
                with torch.no_grad():
                    t_drone = teacher(drone); t_sat = teacher(sat)
                l_cross = cross_dist(d_out['projected_feat'], t_drone) + \
                          cross_dist(s_out['projected_feat'], t_sat)
            else:
                l_cross = torch.tensor(0.0, device=device)

            l_self = self_dist(d_out['cls_logits'], d_out['logits']) + \
                     self_dist(s_out['cls_logits'], s_out['logits'])
            l_uapa = uapa(d_out['logits'], s_out['logits'])

            # === ProxyAnchor + EMA ===
            l_proxy = 0.5 * (proxy_anchor(d_out['embedding'], labels) +
                             proxy_anchor(s_out['embedding'], labels))
            with torch.no_grad():
                ema_drone_emb = ema.forward(drone, alt_idx=alt_idx)
                ema_sat_emb   = ema.forward(sat, alt_idx=None)
            l_ema = 0.5 * (ema_dist(d_out['embedding'], ema_drone_emb) +
                           ema_dist(s_out['embedding'], ema_sat_emb))

            # === Altitude Consistency (EXP27) ===
            l_alt_con = alt_consist(d_out['embedding'], labels, alts)

            # === Masked Part Reconstruction (EXP34) ===
            if recon_active:
                l_recon_d = model.mask_recon(d_out['parts']['projected_patches'],
                                             d_out['parts']['part_features'],
                                             d_out['parts']['assignment'])
                l_recon_s = model.mask_recon(s_out['parts']['projected_patches'],
                                             s_out['parts']['part_features'],
                                             s_out['parts']['assignment'])
                l_recon = 0.5 * (l_recon_d + l_recon_s)
            else:
                l_recon = torch.tensor(0.0, device=device)

            # === Altitude Prediction (EXP34, drone only) ===
            l_alt_pred = model.alt_pred(d_out['embedding'].detach(), alt_norm)

            # === Prototype Diversity (EXP34) ===
            l_div = diversity_loss(model.part_disc.prototypes)

            # === Graph Matching (EXP36) ===
            l_graph = graph_match_loss(
                d_out['graph_parts'], s_out['graph_parts'],
                d_out['structure'], s_out['structure'])

            # === Total loss (14 components) ===
            loss = (CFG.LAMBDA_CE          * l_ce +
                    CFG.LAMBDA_INFONCE     * l_nce +
                    CFG.LAMBDA_CONSISTENCY * l_con +
                    CFG.LAMBDA_CROSS_DIST  * l_cross +
                    CFG.LAMBDA_SELF_DIST   * l_self +
                    CFG.LAMBDA_UAPA        * l_uapa +
                    CFG.LAMBDA_PROXY       * l_proxy +
                    CFG.LAMBDA_EMA_DIST    * l_ema +
                    CFG.LAMBDA_ALT_CONSIST * l_alt_con +
                    CFG.LAMBDA_MASK_RECON  * l_recon +
                    CFG.LAMBDA_ALT_PRED    * l_alt_pred +
                    CFG.LAMBDA_DIVERSITY   * l_div +
                    CFG.LAMBDA_GRAPH_MATCH * l_graph)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()
        ema.update(model)

        total_sum += loss.item(); n += 1
        loss_sums['ce']    += l_ce.item()
        loss_sums['nce']   += l_nce.item()
        loss_sums['con']   += l_con.item()
        loss_sums['cross'] += l_cross.item() if torch.is_tensor(l_cross) else l_cross
        loss_sums['self']  += l_self.item()
        loss_sums['uapa']  += l_uapa.item()
        loss_sums['proxy'] += l_proxy.item()
        loss_sums['ema']   += l_ema.item()
        loss_sums['alt_c'] += l_alt_con.item()
        loss_sums['recon'] += l_recon.item() if torch.is_tensor(l_recon) else l_recon
        loss_sums['alt_p'] += l_alt_pred.item()
        loss_sums['div']   += l_div.item()
        loss_sums['graph'] += l_graph.item()

    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EXP36: SPARC-G — Explicit+Implicit Graph Matching")
    print(f"  Base: EXP35 + PartGraphEncoder + CrossViewGraphMatchingLoss")
    print(f"  Dataset: SUES-200 | Epochs: {CFG.NUM_EPOCHS} | Device: {DEVICE}")
    print(f"  Parts: {CFG.N_PARTS} | Graph heads: {CFG.N_GRAPH_HEADS} | Edge dim: {CFG.GRAPH_EDGE_DIM}")
    print(f"  Losses: 14 total | NUM_CLASSES: {CFG.NUM_CLASSES}")
    print(f"  Recon warmup: {CFG.RECON_WARMUP} | EMA: {CFG.EMA_DECAY} | Sinkhorn: {CFG.GRAPH_SINKHORN_ITERS} iters")
    print("=" * 65)

    print('\nLoading SUES-200 …')
    train_ds = SUES200Dataset(CFG.SUES_ROOT, 'train', transform=get_transforms("train"))
    test_ds  = SUES200Dataset(CFG.SUES_ROOT, 'test', transform=get_transforms("test"))
    train_loader = DataLoader(train_ds, batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print('\nBuilding models …')
    model = SPDGeoDPEAMARModel(CFG.NUM_CLASSES).to(DEVICE)
    teacher = None
    try:
        teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except Exception as e:
        print(f"  [WARN] Could not load teacher: {e}")

    ema = EMAModel(model, decay=CFG.EMA_DECAY)
    print(f"  EMA model initialized (decay={CFG.EMA_DECAY})")

    # Base losses (6)
    infonce    = SupInfoNCELoss(temp=0.05).to(DEVICE)
    ce         = nn.CrossEntropyLoss(label_smoothing=0.1)
    consist    = PartConsistencyLoss()
    cross_dist = CrossDistillationLoss()
    self_dist  = SelfDistillationLoss(temperature=CFG.DISTILL_TEMP)
    uapa_loss  = UAPALoss(base_temperature=CFG.DISTILL_TEMP)
    base_losses = (infonce, ce, consist, cross_dist, self_dist, uapa_loss)

    # Extended losses
    proxy_anchor   = ProxyAnchorLoss(CFG.NUM_CLASSES, CFG.EMBED_DIM,
                                     margin=CFG.PROXY_MARGIN, alpha=CFG.PROXY_ALPHA).to(DEVICE)
    ema_dist       = EMADistillationLoss()
    alt_consist    = AltitudeConsistencyLoss()
    diversity_loss = PrototypeDiversityLoss()
    # NEW (EXP36): Graph matching loss
    graph_match    = CrossViewGraphMatchingLoss(
                         CFG.PART_DIM, CFG.GRAPH_SINKHORN_ITERS,
                         CFG.GRAPH_SINKHORN_EPS, CFG.GRAPH_STRUCT_WEIGHT).to(DEVICE)
    new_losses     = (proxy_anchor, ema_dist, alt_consist, diversity_loss, graph_match)

    # Optimizer
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith('backbone')]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params,            'lr': CFG.BACKBONE_LR},
        {'params': head_params,                'lr': CFG.LR},
        {'params': infonce.parameters(),       'lr': CFG.LR},
        {'params': proxy_anchor.parameters(),  'lr': CFG.LR},
        {'params': graph_match.parameters(),   'lr': CFG.LR},
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
        for i in range(len(optimizer.param_groups)):
            base = CFG.BACKBONE_LR if i == 0 else CFG.LR
            optimizer.param_groups[i]['lr'] = base * lr_scale
        cur_lr = optimizer.param_groups[1]['lr']

        avg_loss, ld = train_one_epoch(model, teacher, ema, train_loader, base_losses,
                                       new_losses, optimizer, scaler, DEVICE, epoch)

        recon_tag = f"Rec {ld['recon']:.3f}" if epoch >= CFG.RECON_WARMUP else "Rec OFF"
        print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
              f"CE {ld['ce']:.3f}  NCE {ld['nce']:.3f}  "
              f"Proxy {ld['proxy']:.3f}  EMA {ld['ema']:.3f} | "
              f"AltC {ld['alt_c']:.3f}  {recon_tag}  AltP {ld['alt_p']:.3f}  "
              f"Div {ld['div']:.3f}  Graph {ld['graph']:.3f} | "
              f"LR {cur_lr:.2e}")

        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            metrics, per_alt = evaluate(model, test_ds, DEVICE)
            results_log.append({'epoch': epoch, **metrics})
            print(f"  ► R@1: {metrics['R@1']:.2f}%  R@5: {metrics['R@5']:.2f}%  "
                  f"R@10: {metrics['R@10']:.2f}%  mAP: {metrics['mAP']:.2f}%")
            if metrics['R@1'] > best_r1:
                best_r1 = metrics['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'metrics': metrics, 'per_alt': per_alt},
                           os.path.join(CFG.OUTPUT_DIR, 'exp36_sparc_g_best.pth'))
                print(f"  ★ New best R@1: {best_r1:.2f}%!")

            ema_metrics, _ = evaluate(ema.model, test_ds, DEVICE)
            print(f"  ► EMA R@1: {ema_metrics['R@1']:.2f}%")
            if ema_metrics['R@1'] > best_r1:
                best_r1 = ema_metrics['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': ema.model.state_dict(),
                            'metrics': ema_metrics, 'is_ema': True},
                           os.path.join(CFG.OUTPUT_DIR, 'exp36_sparc_g_best.pth'))
                print(f"  ★ New best R@1 (EMA): {best_r1:.2f}%!")

    print(f'\n{"="*65}')
    print(f'  EXP36: SPARC-G COMPLETE — Best R@1: {best_r1:.2f}%')
    print(f'{"="*65}')
    print(f'  {"Epoch":>6} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"mAP":>8}')
    print(f'  {"-"*44}')
    for r in results_log:
        print(f'  {r["epoch"]:6d} {r["R@1"]:8.2f} {r["R@5"]:8.2f} {r["R@10"]:8.2f} {r["mAP"]:8.2f}')
    print(f'{"="*65}')

    with open(os.path.join(CFG.OUTPUT_DIR, 'exp36_sparc_g_results.json'), 'w') as f:
        json.dump({'results_log': results_log, 'best_r1': best_r1,
                   'config': {k: v for k, v in vars(CFG).items() if not k.startswith('_')}}, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════
    # TTE: Test-Time Ensemble (inference-only boost)
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n{"="*65}')
    print(f'  TTE: Test-Time Ensemble Evaluation')
    print(f'  Crops: {CFG.TTE_CROP_SIZES} | Tent: {CFG.TTE_TENT_STEPS} steps')
    print(f'{"="*65}')

    # Load best checkpoint
    best_ckpt_path = os.path.join(CFG.OUTPUT_DIR, 'exp36_sparc_g_best.pth')
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'  Loaded best checkpoint (R@1: {ckpt["metrics"]["R@1"]:.2f}%)')
    model.eval()

    # TTE with multi-crop only
    print('\n  [1/3] Multi-crop only …')
    tte_mc = evaluate_tte(model, None, test_ds, DEVICE, tent=None)
    print(f'  Multi-crop R@1: {tte_mc["R@1"]:.2f}%  mAP: {tte_mc["mAP"]:.2f}%')

    # TTE with multi-crop + EMA ensemble
    print('\n  [2/3] Multi-crop + EMA ensemble …')
    tte_mc_ema = evaluate_tte(model, ema.model, test_ds, DEVICE, tent=None)
    print(f'  MC+EMA R@1: {tte_mc_ema["R@1"]:.2f}%  mAP: {tte_mc_ema["mAP"]:.2f}%')

    # TTE with multi-crop + Tent
    print('\n  [3/3] Multi-crop + Tent adaptation …')
    tent = TentAdaptation(model, lr=CFG.TTE_TENT_LR, steps=CFG.TTE_TENT_STEPS)
    tte_full = evaluate_tte(model, ema.model, test_ds, DEVICE, tent=tent)
    print(f'  Full TTE R@1: {tte_full["R@1"]:.2f}%  mAP: {tte_full["mAP"]:.2f}%')

    # Summary
    print(f'\n{"="*65}')
    print(f'  TTE SUMMARY')
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

    with open(os.path.join(CFG.OUTPUT_DIR, 'exp36_sparc_g_tte_results.json'), 'w') as f:
        json.dump({'baseline_r1': best_r1, 'multi_crop': tte_mc,
                   'mc_ema': tte_mc_ema, 'full_tte': tte_full,
                   'final_best_r1': final_r1}, f, indent=2)


if __name__ == '__main__':
    main()
