# =============================================================================
# EXP32: SPDGeo-VCA — View-Conditional LoRA Adaptation
# =============================================================================
# Base:    SPDGeo-DPE (93.59% R@1) — THE CHAMPION
# Novel:   1) ViewConditionalLoRA — rank-4 LoRA adapters injected into the
#             last 4 unfrozen DINOv2 blocks. Drone and satellite views use
#             SEPARATE LoRA weight sets selected by a view flag. This gives
#             each view its own fine-tuned feature space while sharing the
#             frozen backbone core.
#          2) ViewBridgeLoss — aligns intermediate drone-LoRA and sat-LoRA
#             features for same-location pairs, ensuring the two view-spaces
#             remain compatible for cross-view retrieval.
#
# Motivation:
#   DPE uses the SAME backbone for drone (perspective, 150-300m altitude) and
#   satellite (nadir, ortho-rectified). These are structurally different views:
#   drone images have perspective distortion, varying altitude, and oblique
#   angles; satellite images are top-down with consistent scale. Using
#   identical parameters forces the backbone to compromise between both.
#
#   LoRA (Hu et al., ICLR 2022) adds low-rank trainable adapters per layer.
#   View-conditional LoRA gives each view its own small set of adapters
#   (rank=4, ~0.1M extra params total) while sharing the >15M frozen backbone.
#   This is parameter-efficient view specialization.
#
# Architecture: SPDGeo-DPE + ViewConditionalLoRA + ViewBridgeLoss
# Total losses: 9 (8 DPE base + 1 new ViewBridge)
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
    EMA_DECAY           = 0.999
    DISTILL_TEMP        = 4.0

    # NEW: View-Conditional LoRA
    LORA_RANK           = 4      # LoRA rank (low → parameter efficient)
    LORA_ALPHA          = 8      # LoRA scaling factor
    LORA_DROPOUT        = 0.05
    LAMBDA_VIEW_BRIDGE  = 0.2    # ViewBridge loss weight

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
# NEW: LoRA Layer
# =============================================================================
class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) for a linear layer.
    Output = W @ x + (B @ A @ x) * (alpha / rank)
    where A, B are low-rank matrices.
    """
    def __init__(self, in_features, out_features, rank=4, alpha=8, dropout=0.05):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """Returns the LoRA delta: (B @ A @ x) * scaling"""
        return (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class ViewConditionalLoRA(nn.Module):
    """
    Manages TWO sets of LoRA adapters: one for drone view, one for satellite view.
    Each set is applied to the qkv projections of the unfrozen ViT blocks.

    Usage:
        backbone.forward(x, view='drone')  → applies drone LoRA
        backbone.forward(x, view='sat')    → applies satellite LoRA
    """
    def __init__(self, dim=384, rank=4, alpha=8, dropout=0.05, n_blocks=4):
        super().__init__()
        # Each unfrozen block has qkv (dim → 3*dim) and proj (dim → dim)
        # We add LoRA to qkv projection only (most impactful)
        self.drone_loras = nn.ModuleList([
            LoRALayer(dim, dim * 3, rank, alpha, dropout) for _ in range(n_blocks)
        ])
        self.sat_loras = nn.ModuleList([
            LoRALayer(dim, dim * 3, rank, alpha, dropout) for _ in range(n_blocks)
        ])
        self.n_blocks = n_blocks
        total = sum(p.numel() for p in self.parameters())
        print(f"  ViewConditionalLoRA: 2×{n_blocks} adapters, rank={rank}, {total/1e3:.1f}K params")

    def get_lora_set(self, view):
        return self.drone_loras if view == 'drone' else self.sat_loras


# =============================================================================
# BACKBONE with View-Conditional LoRA
# =============================================================================
class DINOv2BackboneVCA(nn.Module):
    def __init__(self, unfreeze_blocks=4, lora_rank=4, lora_alpha=8, lora_dropout=0.05):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.feature_dim = 384; self.patch_size = 14
        self.unfreeze_blocks = unfreeze_blocks

        for p in self.model.parameters(): p.requires_grad = False
        for blk in self.model.blocks[-unfreeze_blocks:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in self.model.norm.parameters(): p.requires_grad = True

        # NEW: View-conditional LoRA adapters
        self.view_lora = ViewConditionalLoRA(384, lora_rank, lora_alpha, lora_dropout, unfreeze_blocks)

        # Store original qkv forward hooks
        self._original_qkv_weights = []
        for blk in self.model.blocks[-unfreeze_blocks:]:
            self._original_qkv_weights.append(blk.attn.qkv)

    def forward(self, x, view='drone'):
        """
        Forward with view-specific LoRA adapters.
        view: 'drone' or 'sat'
        """
        # Get the LoRA set for this view
        lora_set = self.view_lora.get_lora_set(view)

        # Prepare patch embed
        x_prep = self.model.prepare_tokens_with_masks(x)

        # Run through frozen blocks (no LoRA)
        n_frozen = len(self.model.blocks) - self.unfreeze_blocks
        for i in range(n_frozen):
            x_prep = self.model.blocks[i](x_prep)

        # Run through unfrozen blocks with LoRA injection
        intermediate_features = []
        for i, blk in enumerate(self.model.blocks[-self.unfreeze_blocks:]):
            # Compute original attention output
            B, N, C = x_prep.shape
            # Add LoRA delta to qkv
            qkv_orig = blk.attn.qkv(x_prep)           # [B, N, 3*C]
            qkv_lora = lora_set[i](x_prep)             # [B, N, 3*C] LoRA delta
            qkv = qkv_orig + qkv_lora

            # Manual attention computation with LoRA-modified qkv
            num_heads = blk.attn.num_heads
            head_dim = C // num_heads
            qkv_reshaped = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv_reshaped.unbind(0)
            scale = head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attn_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            attn_out = blk.attn.proj(attn_out)
            attn_out = blk.attn.proj_drop(attn_out) if hasattr(blk.attn, 'proj_drop') else attn_out

            # Residual + FFN (using block's ls1/ls2 if available)
            if hasattr(blk, 'ls1'):
                x_prep = x_prep + blk.ls1(attn_out)
                x_prep = x_prep + blk.ls2(blk.mlp(blk.norm2(x_prep)))
            else:
                x_prep = x_prep + attn_out
                x_prep = x_prep + blk.mlp(blk.norm2(x_prep))

            intermediate_features.append(x_prep)

        x_prep = self.model.norm(x_prep)
        patch_tokens = x_prep[:, 1:]    # Remove CLS
        cls_token = x_prep[:, 0]
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return patch_tokens, cls_token, (H, W), intermediate_features


class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        self.output_dim = 768
        for p in self.model.parameters(): p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        return self.model.forward_features(x)['x_norm_clstoken']


# =============================================================================
# SEMANTIC PART DISCOVERY + POOLING + FUSION (same as DPE)
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
            nn.Linear(embed_dim * 2, embed_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        combined = torch.cat([part_emb, cls_emb], dim=-1)
        alpha = torch.sigmoid(self.gate(combined))
        fused = alpha * part_emb + (1 - alpha) * cls_emb
        return F.normalize(fused, dim=-1)


# =============================================================================
# STUDENT MODEL
# =============================================================================
class SPDGeoVCAModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone    = DINOv2BackboneVCA(cfg.UNFREEZE_BLOCKS, cfg.LORA_RANK,
                                              cfg.LORA_ALPHA, cfg.LORA_DROPOUT)
        self.part_disc   = SemanticPartDiscovery(384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP)
        self.pool        = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)
        self.bottleneck     = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj   = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
                                            nn.LayerNorm(cfg.TEACHER_DIM))
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-VCA student: {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x, view='drone'):
        patches, cls_tok, hw, _ = self.backbone(x, view=view)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)

    def forward(self, x, view='drone', return_parts=False):
        patches, cls_tok, hw, intermediates = self.backbone(x, view=view)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        bn = self.bottleneck(emb); logits = self.classifier(bn)
        cls_emb_raw = self.cls_proj(cls_tok); cls_logits = self.cls_classifier(cls_emb_raw)
        cls_emb_norm = F.normalize(cls_emb_raw, dim=-1)
        fused = self.fusion_gate(emb, cls_emb_norm)
        projected_feat = self.teacher_proj(emb)
        out = {'embedding': fused, 'logits': logits, 'cls_logits': cls_logits,
               'projected_feat': projected_feat, 'part_emb': emb, 'cls_emb': cls_emb_norm,
               'intermediates': intermediates}
        if return_parts: out['parts'] = parts
        return out


class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay; self.model = copy.deepcopy(model); self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False
    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
    @torch.no_grad()
    def forward(self, x, view='drone'): return self.model.extract_embedding(x, view=view)


# =============================================================================
# BASE LOSSES
# =============================================================================
class SupInfoNCELoss(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__(); self.log_t = nn.Parameter(torch.tensor(temp).log())
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
# NEW: View Bridge Loss
# =============================================================================
class ViewBridgeLoss(nn.Module):
    """
    Aligns intermediate features from drone-LoRA and sat-LoRA branches.

    For same-location pairs, the intermediate ViT features (from unfrozen blocks)
    should be similar despite using different LoRA adapters. This prevents the
    two view-specific feature spaces from diverging too much, ensuring cross-view
    retrieval compatibility.

    Uses the CLS token from the last intermediate layer.
    """
    def forward(self, drone_intermediates, sat_intermediates):
        """
        drone_intermediates: list of [B, N+1, C] from unfrozen blocks (drone LoRA)
        sat_intermediates:   list of [B, N+1, C] from unfrozen blocks (sat LoRA)
        """
        # Use CLS token from last intermediate layer
        d_cls = F.normalize(drone_intermediates[-1][:, 0], dim=-1)
        s_cls = F.normalize(sat_intermediates[-1][:, 0], dim=-1)
        return (1 - F.cosine_similarity(d_cls, s_cls)).mean()


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
        feat = model.extract_embedding(b['drone'].to(device), view='drone').cpu()
        drone_feats.append(feat); drone_labels.append(b['label']); drone_alts.append(b['altitude'])
    drone_feats = torch.cat(drone_feats); drone_labels = torch.cat(drone_labels)
    drone_alts = torch.cat(drone_alts)

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
        sat_feats.append(model.extract_embedding(batch, view='sat').cpu())
    sat_feats = torch.cat(sat_feats); sat_labels = torch.tensor(sat_label_list)

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

    altitudes_list = sorted(drone_alts.unique().tolist()); per_alt = {}
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
    print(f"  {'Altitude':>8s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'#Q':>6s}")
    print(f"  {'-'*50}")
    for alt in altitudes_list:
        a = per_alt[int(alt)]
        print(f"  {int(alt):>6d}m  {a['R@1']:6.2f}%  {a['R@5']:6.2f}%  {a['R@10']:6.2f}%  {a['mAP']:6.2f}%  {a['n']:>6d}")
    print(f"  {'-'*50}")
    print(f"  {'Overall':>8s}  {overall['R@1']:6.2f}%  {overall['R@5']:6.2f}%  {overall['R@10']:6.2f}%  {overall['mAP']:6.2f}%  {N:>6d}")
    print(f"{'='*75}\n")
    return overall, per_alt


# =============================================================================
# TRAINING
# =============================================================================
def train_one_epoch(model, teacher, ema, loader, losses, new_losses, optimizer,
                    scaler, device, epoch):
    model.train()
    if teacher: teacher.eval()
    infonce, ce, consist, cross_dist, self_dist, uapa = losses
    proxy_anchor, ema_dist, view_bridge = new_losses
    total_sum = 0; n = 0; loss_sums = defaultdict(float)

    for batch in tqdm(loader, desc=f'Ep{epoch:3d}', leave=False):
        drone  = batch['drone'].to(device)
        sat    = batch['satellite'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            # Forward with view-specific LoRA
            d_out = model(drone, view='drone', return_parts=True)
            s_out = model(sat, view='sat', return_parts=True)

            l_ce = (ce(d_out['logits'], labels) + ce(s_out['logits'], labels))
            l_ce += 0.3 * (ce(d_out['cls_logits'], labels) + ce(s_out['cls_logits'], labels))
            l_nce = infonce(d_out['embedding'], s_out['embedding'], labels)
            l_con = consist(d_out['parts']['assignment'], s_out['parts']['assignment'])

            if teacher is not None:
                with torch.no_grad():
                    t_drone = teacher(drone); t_sat = teacher(sat)
                l_cross = cross_dist(d_out['projected_feat'], t_drone) + cross_dist(s_out['projected_feat'], t_sat)
            else:
                l_cross = torch.tensor(0.0, device=device)

            l_self = self_dist(d_out['cls_logits'], d_out['logits']) + self_dist(s_out['cls_logits'], s_out['logits'])
            l_uapa = uapa(d_out['logits'], s_out['logits'])
            l_proxy = 0.5 * (proxy_anchor(d_out['embedding'], labels) + proxy_anchor(s_out['embedding'], labels))

            with torch.no_grad():
                ema_drone = ema.forward(drone, view='drone')
                ema_sat = ema.forward(sat, view='sat')
            l_ema = 0.5 * (ema_dist(d_out['embedding'], ema_drone) + ema_dist(s_out['embedding'], ema_sat))

            # NEW: View Bridge Loss — align intermediate features across views
            l_bridge = view_bridge(d_out['intermediates'], s_out['intermediates'])

            loss = (CFG.LAMBDA_CE * l_ce + CFG.LAMBDA_INFONCE * l_nce +
                    CFG.LAMBDA_CONSISTENCY * l_con + CFG.LAMBDA_CROSS_DIST * l_cross +
                    CFG.LAMBDA_SELF_DIST * l_self + CFG.LAMBDA_UAPA * l_uapa +
                    CFG.LAMBDA_PROXY * l_proxy + CFG.LAMBDA_EMA_DIST * l_ema +
                    CFG.LAMBDA_VIEW_BRIDGE * l_bridge)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()
        ema.update(model)

        total_sum += loss.item(); n += 1
        loss_sums['ce'] += l_ce.item(); loss_sums['nce'] += l_nce.item()
        loss_sums['con'] += l_con.item()
        loss_sums['cross'] += l_cross.item() if torch.is_tensor(l_cross) else l_cross
        loss_sums['self'] += l_self.item(); loss_sums['uapa'] += l_uapa.item()
        loss_sums['proxy'] += l_proxy.item(); loss_sums['ema'] += l_ema.item()
        loss_sums['bridge'] += l_bridge.item()

    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EXP32: SPDGeo-VCA — View-Conditional LoRA Adaptation")
    print(f"  Base: SPDGeo-DPE (93.59% R@1)")
    print(f"  Novel: ViewConditionalLoRA(rank={CFG.LORA_RANK}) + ViewBridgeLoss")
    print(f"  Dataset: SUES-200 | Epochs: {CFG.NUM_EPOCHS} | Device: {DEVICE}")
    print(f"  Losses: 8 DPE + 1 ViewBridge = 9 total")
    print("=" * 65)

    print('\nLoading SUES-200 …')
    train_ds = SUES200Dataset(CFG.SUES_ROOT, 'train', transform=get_transforms("train"))
    test_ds  = SUES200Dataset(CFG.SUES_ROOT, 'test', transform=get_transforms("test"))
    train_loader = DataLoader(train_ds, batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print('\nBuilding models …')
    model = SPDGeoVCAModel(CFG.NUM_CLASSES).to(DEVICE)
    teacher = None
    try: teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except Exception as e: print(f"  [WARN] Could not load teacher: {e}")

    ema = EMAModel(model, decay=CFG.EMA_DECAY)

    infonce    = SupInfoNCELoss(temp=0.05).to(DEVICE)
    ce         = nn.CrossEntropyLoss(label_smoothing=0.1)
    consist    = PartConsistencyLoss()
    cross_dist = CrossDistillationLoss()
    self_dist  = SelfDistillationLoss(temperature=CFG.DISTILL_TEMP)
    uapa_loss  = UAPALoss(base_temperature=CFG.DISTILL_TEMP)
    base_losses = (infonce, ce, consist, cross_dist, self_dist, uapa_loss)

    proxy_anchor = ProxyAnchorLoss(CFG.NUM_CLASSES, CFG.EMBED_DIM,
                                    margin=CFG.PROXY_MARGIN, alpha=CFG.PROXY_ALPHA).to(DEVICE)
    ema_dist     = EMADistillationLoss()
    view_bridge  = ViewBridgeLoss()
    new_losses   = (proxy_anchor, ema_dist, view_bridge)

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
        for i in range(4): optimizer.param_groups[i]['lr'] = (CFG.BACKBONE_LR if i == 0 else CFG.LR) * lr_scale
        cur_lr = optimizer.param_groups[1]['lr']

        avg_loss, ld = train_one_epoch(model, teacher, ema, train_loader, base_losses,
                                       new_losses, optimizer, scaler, DEVICE, epoch)
        print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
              f"CE {ld['ce']:.3f} NCE {ld['nce']:.3f} Con {ld['con']:.3f} "
              f"Crs {ld['cross']:.3f} Slf {ld['self']:.3f} UAPA {ld['uapa']:.3f} | "
              f"Proxy {ld['proxy']:.3f} EMA {ld['ema']:.3f} Bridge {ld['bridge']:.3f} | LR {cur_lr:.2e}")

        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            metrics, per_alt = evaluate(model, test_ds, DEVICE)
            results_log.append({'epoch': epoch, **metrics})
            print(f"  ► R@1: {metrics['R@1']:.2f}% R@5: {metrics['R@5']:.2f}% mAP: {metrics['mAP']:.2f}%")
            if metrics['R@1'] > best_r1:
                best_r1 = metrics['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'metrics': metrics, 'per_alt': per_alt},
                           os.path.join(CFG.OUTPUT_DIR, 'exp32_vca_best.pth'))
                print(f"  ★ New best R@1: {best_r1:.2f}%!")

            ema_metrics, ema_pa = evaluate(ema.model, test_ds, DEVICE)
            print(f"  ► EMA R@1: {ema_metrics['R@1']:.2f}%")
            if ema_metrics['R@1'] > best_r1:
                best_r1 = ema_metrics['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': ema.model.state_dict(),
                            'metrics': ema_metrics, 'per_alt': ema_pa, 'is_ema': True},
                           os.path.join(CFG.OUTPUT_DIR, 'exp32_vca_best.pth'))
                print(f"  ★ New best R@1 (EMA): {best_r1:.2f}%!")

    print(f'\n{"="*65}')
    print(f'  EXP32: SPDGeo-VCA COMPLETE — Best R@1: {best_r1:.2f}%')
    print(f'{"="*65}')
    for r in results_log:
        print(f'  Ep{r["epoch"]:3d} R@1:{r["R@1"]:6.2f} R@5:{r["R@5"]:6.2f} R@10:{r["R@10"]:6.2f} mAP:{r["mAP"]:6.2f}')

    with open(os.path.join(CFG.OUTPUT_DIR, 'exp32_vca_results.json'), 'w') as f:
        json.dump({'results_log': results_log, 'best_r1': best_r1,
                   'config': {k: v for k, v in vars(CFG).items() if not k.startswith('_')}}, f, indent=2)


if __name__ == '__main__':
    main()
