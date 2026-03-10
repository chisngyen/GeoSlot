# =============================================================================
# GeoSlot 2.0 — Full Method on SUES-200 (Drone ↔ Satellite)
# =============================================================================
# BACKBONE: DINOv2 ViT-B/14 (via timm, no mamba_ssm needed)
# Method: Adaptive Mask → Slot Attention → Graph Mamba → UFGW
# 3-Stage Training with Cosine LR
# Images: 512×512 native resolution
#
# Dataset: SUES-200 (200 locations, 4 altitudes, 50 imgs/alt)
#          Train: locations 0001-0120, Test: 0121-0200
#          Evaluates each altitude separately
# =============================================================================

# === SETUP ===
import subprocess, sys, os

def pip_install(pkg, extra=""):
    subprocess.run(f"pip install -q {extra} {pkg}",
                   shell=True, capture_output=True, text=True)

print("[1/2] Installing packages...")
for p in ["timm", "tqdm"]:
    try: __import__(p)
    except ImportError: pip_install(p)

print("[2/2] Setup complete!")

# =============================================================================
# IMPORTS
# =============================================================================
import math, glob, json, time, gc, random
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm


# =============================================================================
# CONFIG
# =============================================================================
# --- Paths (Kaggle) ---
DATA_ROOT      = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
OUTPUT_DIR     = "/kaggle/working"

# --- Training ---
EPOCHS         = 50
BATCH_SIZE     = 128
NUM_WORKERS    = 4
AMP_ENABLED    = True
EVAL_FREQ      = 5

# 3-Stage schedule
STAGE1_END     = 15    # Backbone frozen, InfoNCE only
STAGE2_END     = 35    # + CSM + Dice (slot specialization)
# Stage 3: full fine-tuning until EPOCHS

# Learning rates
LR_HEAD_S1     = 3e-4
LR_BACKBONE_S2 = 3e-5
LR_HEAD_S2     = 1e-4
LR_BACKBONE_S3 = 1e-5
LR_HEAD_S3     = 5e-5
WEIGHT_DECAY   = 0.01
WARMUP_EPOCHS  = 3

# Loss weights
LAMBDA_CSM     = 0.3
LAMBDA_DICE    = 0.1
LAMBDA_BG      = 0.01
CSM_WARMUP     = 5

# --- Model ---
BACKBONE_NAME  = "vit_base_patch14_dinov2"
FEATURE_DIM    = 768    # DINOv2 ViT-B output dim
PATCH_SIZE     = 14
SLOT_DIM       = 256
MAX_SLOTS      = 8
N_REGISTER     = 4
EMBED_DIM_OUT  = 512
N_HEADS        = 4
SA_ITERS       = 3
GM_LAYERS      = 2
FGW_ITERS      = 5
SINKHORN_ITERS = 10

# --- Dataset ---
IMG_SIZE       = 518          # 37*14=518, divisible by DINOv2 patch_size=14
TRAIN_LOCS     = list(range(1, 121))   # 0001-0120
TEST_LOCS      = list(range(121, 201)) # 0121-0200
ALTITUDES      = ["150", "200", "250", "300"]
TEST_ALTITUDE  = "150"                 # Default eval altitude

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# BACKBONE: DINOv2 ViT-B/14
# =============================================================================
class ViTBackbone(nn.Module):
    def __init__(self, model_name=BACKBONE_NAME):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True,
                                       num_classes=0, dynamic_img_size=True)
        self.feature_dim = FEATURE_DIM
        self.patch_size = PATCH_SIZE

    def forward(self, x):
        B, C, H, W = x.shape
        # Get patch tokens (excluding CLS)
        features = self.model.forward_features(x)
        if features.dim() == 3 and features.shape[1] > 1:
            # Remove CLS token if present
            if hasattr(self.model, 'num_prefix_tokens'):
                n_prefix = self.model.num_prefix_tokens
            else:
                n_prefix = 1
            features = features[:, n_prefix:]  # [B, N_patches, D]
        gh = H // self.patch_size
        gw = W // self.patch_size
        return features, (gh, gw)


# =============================================================================
# MODULES: Mask, Slots, GraphMamba, UFGW
# =============================================================================
class AdaptiveGumbelSparsityMask(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.mask_head = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(True),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
        self.gamma_head = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(True),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, features):
        mask = self.mask_head(features)
        gap = features.mean(dim=1)
        gamma = self.gamma_head(gap)
        return features * mask, mask, gamma


class SlotAttentionCore(nn.Module):
    def __init__(self, dim, feature_dim, n_heads=4, iters=3, eps=1e-8):
        super().__init__()
        self.dim = dim; self.n_heads = n_heads; self.iters = iters
        self.eps = eps; self.dh = dim // n_heads; self.scale = self.dh ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(feature_dim, dim, bias=False)
        self.to_v = nn.Linear(feature_dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.norm_in = nn.LayerNorm(feature_dim)
        self.norm_s = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*4),
                                nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self, inputs, slots):
        inputs = self.norm_in(inputs)
        B, N, _ = inputs.shape
        k = self.to_k(inputs).view(B, N, self.n_heads, self.dh)
        v = self.to_v(inputs).view(B, N, self.n_heads, self.dh)
        for _ in range(self.iters):
            B, K, _ = slots.shape
            q = self.to_q(self.norm_s(slots)).view(B, K, self.n_heads, self.dh)
            dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
            attn = dots.flatten(1,2).softmax(dim=1).view(B, K, self.n_heads, -1)
            attn_out = attn.mean(dim=2)
            attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.einsum("bjhd,bihj->bihd", v, attn)
            slots = self.gru(updates.reshape(-1, self.dim),
                             slots.reshape(-1, self.dim)).reshape(B, -1, self.dim)
            slots = slots + self.ff(slots)
        return slots, attn_out


class GumbelSelector(nn.Module):
    def __init__(self, dim, low_bound=1):
        super().__init__()
        self.low_bound = low_bound
        self.net = nn.Sequential(nn.Linear(dim, dim//2), nn.ReLU(True),
                                 nn.Linear(dim//2, 2))

    def forward(self, slots, global_step=None):
        logits = self.net(slots)
        tau = max(0.1, 1.0 - (global_step or 0) / 100000)
        if self.training:
            decision = F.gumbel_softmax(logits, hard=True, tau=tau)[..., 1]
        else:
            decision = (logits.argmax(dim=-1) == 1).float()
        active = (decision != 0).sum(dim=-1)
        for j in (active < self.low_bound).nonzero(as_tuple=True)[0]:
            inactive = (decision[j] == 0).nonzero(as_tuple=True)[0]
            n = min(self.low_bound - int(active[j].item()), len(inactive))
            if n > 0:
                idx = inactive[torch.randperm(len(inactive), device=decision.device)[:n]]
                decision[j, idx] = 1.0
        keep_probs = F.softmax(logits, dim=-1)[..., 1]
        return decision, keep_probs


class SlotSpatialEncoder(nn.Module):
    def __init__(self, dim, pos_dim=64):
        super().__init__()
        self.pos_dim = pos_dim
        self.pos_mlp = nn.Sequential(
            nn.Linear(4 * pos_dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def compute_centroids(self, attn_maps, H, W):
        B, K, N = attn_maps.shape
        device = attn_maps.device
        gy = torch.arange(H, device=device, dtype=attn_maps.dtype) / max(H-1, 1)
        gx = torch.arange(W, device=device, dtype=attn_maps.dtype) / max(W-1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        attn_prob = attn_maps / (attn_maps.sum(dim=-1, keepdim=True) + 1e-8)
        centroids = torch.einsum('bkn,nc->bkc', attn_prob, coords)
        return centroids

    def forward(self, attn_maps, H, W):
        centroids = self.compute_centroids(attn_maps, H, W)
        spreads = torch.ones_like(centroids) * 0.1
        feats = torch.cat([centroids, spreads], dim=-1)
        device = feats.device
        freqs = 1.0 / (10000.0 ** (torch.arange(0, self.pos_dim, 2, device=device).float() / self.pos_dim))
        angles = feats.unsqueeze(-1) * freqs.view(1,1,1,-1) * 3.14159
        enc = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2, -1)
        return self.pos_mlp(enc), centroids


def xy2d_hilbert(n, x, y):
    d = 0; s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1: x = s-1-x; y = s-1-y
            x, y = y, x
        s //= 2
    return d


def hilbert_order(centroids):
    B, K, _ = centroids.shape
    device = centroids.device
    grid_size = 16
    cx = (centroids[:,:,0] * (grid_size-1)).clamp(0, grid_size-1).long()
    cy = (centroids[:,:,1] * (grid_size-1)).clamp(0, grid_size-1).long()
    hilbert_idx = torch.zeros(B, K, device=device)
    for b in range(B):
        for k in range(K):
            hilbert_idx[b, k] = xy2d_hilbert(grid_size, cx[b,k].item(), cy[b,k].item())
    return hilbert_idx.argsort(dim=-1)


class LinearAttentionSSM(nn.Module):
    """Lightweight SSM substitute (no mamba_ssm dependency needed)."""
    def __init__(self, d_model, expand=2):
        super().__init__()
        d_inner = int(d_model * expand)
        self.proj_in = nn.Linear(d_model, d_inner * 2)
        self.proj_out = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        z, gate = self.proj_in(self.norm(x)).chunk(2, dim=-1)
        return x + self.proj_out(self.act(z) * torch.sigmoid(gate))


class GraphMambaLayer(nn.Module):
    def __init__(self, dim, num_layers=2, use_hilbert=True):
        super().__init__()
        self.spatial_encoder = SlotSpatialEncoder(dim)
        self.fwd = nn.ModuleList([LinearAttentionSSM(dim) for _ in range(num_layers)])
        self.bwd = nn.ModuleList([LinearAttentionSSM(dim) for _ in range(num_layers)])
        self.merge = nn.ModuleList([nn.Linear(dim*2, dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*4),
                                   nn.GELU(), nn.Linear(dim*4, dim)) for _ in range(num_layers)])
        self.use_hilbert = use_hilbert

    def forward(self, slots, keep_mask=None, attn_maps=None, spatial_hw=None):
        B, K, D = slots.shape
        centroids = None
        if attn_maps is not None and spatial_hw is not None:
            H, W = spatial_hw
            obj_attn = attn_maps[:, :K, :]
            pos_enc, centroids = self.spatial_encoder(obj_attn, H, W)
            slots = slots + pos_enc

        for i in range(len(self.fwd)):
            residual = slots
            if centroids is not None and self.use_hilbert:
                order = hilbert_order(centroids)
            else:
                order = torch.stack([torch.randperm(K, device=slots.device) for _ in range(B)])
            bi = torch.arange(B, device=slots.device).unsqueeze(1).expand(-1, K)
            ordered = slots[bi, order]
            f = self.fwd[i](ordered)
            b = self.bwd[i](ordered.flip(1)).flip(1)
            merged = self.merge[i](torch.cat([f, b], dim=-1))
            reverse = order.argsort(dim=-1)
            slots = self.norms[i](residual + merged[bi, reverse])
            slots = slots + self.ffns[i](slots)
            if keep_mask is not None:
                slots = slots * keep_mask.unsqueeze(-1)
        return slots, centroids


class FusedGromovWasserstein(nn.Module):
    def __init__(self, dim, lambda_fgw=0.5, tau_kl=0.1,
                 n_outer=5, n_sinkhorn=10, epsilon=0.05):
        super().__init__()
        self.lambda_fgw = lambda_fgw; self.tau_kl = tau_kl
        self.n_outer = n_outer; self.n_sinkhorn = n_sinkhorn
        self.log_eps = nn.Parameter(torch.tensor(math.log(epsilon)))
        self.cost_proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, dim))

    @property
    def epsilon(self): return self.log_eps.exp().clamp(0.01, 0.5)

    def forward(self, slots_q, slots_r, mask_q=None, mask_r=None,
                centroids_q=None, centroids_r=None):
        slots_q = slots_q.float(); slots_r = slots_r.float()
        pq = self.cost_proj(slots_q); pr = self.cost_proj(slots_r)
        diff = pq.unsqueeze(2) - pr.unsqueeze(1)
        C = (diff*diff).sum(-1).clamp(min=1e-6).sqrt()
        B, K, M = C.shape

        if centroids_q is not None and centroids_r is not None:
            cq = centroids_q.float(); cr = centroids_r.float()
            dq = cq.unsqueeze(2) - cq.unsqueeze(1)
            Sq = (dq*dq).sum(-1).clamp(min=1e-6).sqrt()
            dr = cr.unsqueeze(2) - cr.unsqueeze(1)
            Sr = (dr*dr).sum(-1).clamp(min=1e-6).sqrt()
        else:
            Sq = torch.zeros(B, K, K, device=C.device)
            Sr = torch.zeros(B, M, M, device=C.device)

        if mask_q is not None:
            mu = mask_q.float() / (mask_q.float().sum(-1, keepdim=True) + 1e-8)
        else:
            mu = torch.ones(B, K, device=C.device) / K
        if mask_r is not None:
            nu = mask_r.float() / (mask_r.float().sum(-1, keepdim=True) + 1e-8)
        else:
            nu = torch.ones(B, M, device=C.device) / M

        T = mu.unsqueeze(2) * nu.unsqueeze(1)
        eps = self.epsilon
        rho = self.tau_kl / (self.tau_kl + eps)

        for _ in range(self.n_outer):
            Sq2 = Sq * Sq; Sr2 = Sr * Sr
            t1 = torch.bmm(Sq2, T.sum(2, keepdim=True)).squeeze(-1).unsqueeze(2).expand_as(T)
            t2 = torch.bmm(T.sum(1, keepdim=True), Sr2).squeeze(1).unsqueeze(1).expand_as(T)
            t3 = -2.0 * torch.bmm(Sq, torch.bmm(T, Sr))
            L_gw = t1 + t2 + t3
            C_fgw = (1 - self.lambda_fgw) * C + self.lambda_fgw * L_gw

            log_K = -C_fgw / eps
            log_mu = torch.log(mu.clamp(min=1e-8))
            log_nu = torch.log(nu.clamp(min=1e-8))
            log_u = torch.zeros(B, K, device=C.device)
            log_v = torch.zeros(B, M, device=C.device)
            for _ in range(self.n_sinkhorn):
                log_sum_v = torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
                log_u = rho * (log_mu - log_sum_v)
                log_sum_u = torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)
                log_v = rho * (log_nu - log_sum_u)
            T = torch.exp(log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1))

        cost = (T * C_fgw).sum(dim=(-1, -2))
        return {"similarity": torch.sigmoid(-cost), "transport_plan": T,
                "cost_matrix": C, "transport_cost": cost}


# =============================================================================
# FULL PIPELINE
# =============================================================================
class GeoSlotV2(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        # Adaptive mask
        self.bg_mask = AdaptiveGumbelSparsityMask(FEATURE_DIM)

        # Slot Attention
        total_slots = MAX_SLOTS + N_REGISTER
        self.input_proj = nn.Linear(FEATURE_DIM, SLOT_DIM)
        self.slot_mu = nn.Parameter(torch.randn(1, total_slots, SLOT_DIM) * (SLOT_DIM**-0.5))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, total_slots, SLOT_DIM))
        self.slot_attn = SlotAttentionCore(SLOT_DIM, SLOT_DIM, N_HEADS, SA_ITERS)
        self.gumbel = GumbelSelector(SLOT_DIM)

        # Graph Mamba
        self.graph_mamba = GraphMambaLayer(SLOT_DIM, GM_LAYERS, use_hilbert=True)

        # UFGW matcher
        self.ot_matcher = FusedGromovWasserstein(
            SLOT_DIM, lambda_fgw=0.5, tau_kl=0.1, n_outer=FGW_ITERS)

        # Embedding head
        self.embed_head = nn.Sequential(
            nn.LayerNorm(SLOT_DIM), nn.Linear(SLOT_DIM, EMBED_DIM_OUT))

    def encode_view(self, x, global_step=None):
        features, (H, W) = self.backbone(x)
        with torch.amp.autocast('cuda', enabled=False):
            features = features.float()
            features_masked, bg_mask, gamma = self.bg_mask(features)

            proj = self.input_proj(features_masked)
            B = features.shape[0]
            mu = self.slot_mu.expand(B, -1, -1)
            sigma = self.slot_log_sigma.exp().expand(B, -1, -1)
            slots = mu + sigma * torch.randn_like(mu)
            slots, attn_maps = self.slot_attn(proj, slots)

            obj_slots = slots[:, :MAX_SLOTS]
            keep_decision, keep_probs = self.gumbel(obj_slots, global_step)
            obj_slots = obj_slots * keep_decision.unsqueeze(-1)

            obj_slots, centroids = self.graph_mamba(
                obj_slots, keep_decision, attn_maps=attn_maps, spatial_hw=(H, W))

            weights = keep_decision / (keep_decision.sum(dim=-1, keepdim=True) + 1e-8)
            global_slot = (obj_slots * weights.unsqueeze(-1)).sum(dim=1)
            emb = F.normalize(self.embed_head(global_slot), dim=-1)

        return {"embedding": emb, "slots": obj_slots, "keep_mask": keep_decision,
                "bg_mask": bg_mask, "adaptive_gamma": gamma,
                "attn_maps": attn_maps, "keep_probs": keep_probs,
                "centroids": centroids}

    def forward(self, q_img, r_img, global_step=None):
        q = self.encode_view(q_img, global_step)
        r = self.encode_view(r_img, global_step)

        ot_out = self.ot_matcher(
            q["slots"], r["slots"],
            mask_q=q["keep_mask"], mask_r=r["keep_mask"],
            centroids_q=q.get("centroids"), centroids_r=r.get("centroids"))

        return {"query_embedding": q["embedding"], "ref_embedding": r["embedding"],
                "similarity": ot_out["similarity"],
                "transport_plan": ot_out["transport_plan"],
                "transport_cost": ot_out["transport_cost"],
                "query_slots": q["slots"], "ref_slots": r["slots"],
                "query_keep_mask": q["keep_mask"], "ref_keep_mask": r["keep_mask"],
                "query_bg_mask": q["bg_mask"], "ref_bg_mask": r["bg_mask"],
                "query_adaptive_gamma": q["adaptive_gamma"],
                "ref_adaptive_gamma": r["adaptive_gamma"],
                "query_attn_maps": q["attn_maps"], "ref_attn_maps": r["attn_maps"],
                "query_keep_probs": q["keep_probs"], "ref_keep_probs": r["keep_probs"]}

    def extract_embedding(self, x, global_step=None):
        return self.encode_view(x, global_step)["embedding"]


# =============================================================================
# LOSS: 3-Stage with InfoNCE + CSM + Dice + BG
# =============================================================================
class ThreeStageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(0.07).log())
        self.csm_temp = 0.1

    @property
    def temp(self): return self.log_temp.exp().clamp(0.01, 1.0)

    def forward(self, out, epoch=0):
        q_emb = out["query_embedding"]
        r_emb = out["ref_embedding"]
        B = q_emb.shape[0]

        logits = q_emb @ r_emb.t() / self.temp
        labels = torch.arange(B, device=logits.device)
        loss_infonce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        total_loss = loss_infonce
        loss_csm = torch.tensor(0.0, device=total_loss.device)
        loss_dice = torch.tensor(0.0, device=total_loss.device)
        loss_bg = torch.tensor(0.0, device=total_loss.device)

        # Stage 2+: CSM + Dice
        if epoch >= STAGE1_END and out.get("query_slots") is not None:
            ramp = min(1.0, (epoch - STAGE1_END + 1) / CSM_WARMUP)

            q_slots = out["query_slots"]
            r_slots = out["ref_slots"]
            tp = out.get("transport_plan")

            if q_slots is not None and r_slots is not None:
                q_norm = F.normalize(q_slots, dim=-1)
                r_norm = F.normalize(r_slots, dim=-1)
                slot_sim = torch.bmm(q_norm, r_norm.transpose(1, 2)) / self.csm_temp

                if tp is not None:
                    tp_target = tp.detach()
                    tp_target = tp_target / (tp_target.sum(dim=-1, keepdim=True) + 1e-8)
                    csm_per_slot = -(tp_target * F.log_softmax(slot_sim, dim=-1)).sum(dim=-1)
                else:
                    csm_per_slot = F.cross_entropy(
                        slot_sim.view(-1, slot_sim.shape[-1]),
                        torch.arange(slot_sim.shape[1], device=slot_sim.device).repeat(B),
                        reduction='none').view(B, -1)

                q_mask = out.get("query_keep_mask")
                if q_mask is not None:
                    loss_csm = (csm_per_slot * q_mask).sum() / (q_mask.sum() + 1e-8)
                else:
                    loss_csm = csm_per_slot.mean()
                total_loss = total_loss + ramp * LAMBDA_CSM * loss_csm

            q_attn = out.get("query_attn_maps")
            if q_attn is not None:
                K_obj = min(MAX_SLOTS, q_attn.shape[1])
                obj_attn = q_attn[:, :K_obj]
                obj_attn = obj_attn / (obj_attn.sum(dim=-1, keepdim=True) + 1e-8)
                dice_sum = 0.0; dice_count = 0
                for i in range(K_obj):
                    for j in range(i+1, K_obj):
                        overlap = 2 * (obj_attn[:, i] * obj_attn[:, j]).sum(dim=-1)
                        total_mass = obj_attn[:, i].sum(dim=-1) + obj_attn[:, j].sum(dim=-1)
                        dice_sum += (overlap / (total_mass + 0.1)).mean()
                        dice_count += 1
                if dice_count > 0:
                    loss_dice = dice_sum / dice_count
                    total_loss = total_loss + ramp * LAMBDA_DICE * loss_dice

        # BG mask reg (all stages, with gamma prior)
        if out.get("query_bg_mask") is not None:
            mask = out["query_bg_mask"]
            p = mask.squeeze(-1).clamp(1e-6, 1-1e-6)
            entropy = -(p * p.log() + (1-p) * (1-p).log()).mean()
            gamma = out.get("query_adaptive_gamma")
            if gamma is not None:
                coverage = F.relu(gamma - mask.mean(dim=1)).mean()
                gamma_prior = ((gamma - 0.7) ** 2).mean()
                loss_bg = -entropy + coverage + gamma_prior
            else:
                loss_bg = -entropy + (mask.mean() - 0.7) ** 2
            total_loss = total_loss + LAMBDA_BG * loss_bg

        return {"total_loss": total_loss, "accuracy": acc,
                "loss_infonce": loss_infonce.item(),
                "loss_csm": loss_csm.item() if torch.is_tensor(loss_csm) else loss_csm,
                "loss_dice": loss_dice.item() if torch.is_tensor(loss_dice) else loss_dice,
                "loss_bg": loss_bg.item() if torch.is_tensor(loss_bg) else loss_bg}


# =============================================================================
# DATASET: SUES-200
# =============================================================================
class SUES200Dataset(Dataset):
    """
    SUES-200 for GeoSlot: drone (query) ↔ satellite (gallery) pairs.
    Each sample: (drone_image, satellite_image) for the same location.
    """
    def __init__(self, root, split="train", altitude="150",
                 img_size=224, train_locs=None, test_locs=None):
        super().__init__()
        self.root = root
        self.split = split
        self.altitude = altitude

        drone_dir = os.path.join(root, "drone-view")
        satellite_dir = os.path.join(root, "satellite-view")

        if train_locs is None:
            train_locs = TRAIN_LOCS
        if test_locs is None:
            test_locs = TEST_LOCS

        locs = train_locs if split == "train" else test_locs

        # Transforms
        if split == "train":
            self.drone_tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=3),
                transforms.Pad(10, padding_mode='edge'),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
            self.sat_tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=3),
                transforms.Pad(10, padding_mode='edge'),
                transforms.RandomAffine(90),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        else:
            self.drone_tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
            self.sat_tf = self.drone_tf

        # Build pairs: (drone_path, sat_path)
        self.pairs = []
        for loc_id in locs:
            loc_str = f"{loc_id:04d}"
            sat_path = os.path.join(satellite_dir, loc_str, "0.png")
            if not os.path.exists(sat_path):
                continue
            alt_dir = os.path.join(drone_dir, loc_str, altitude)
            if not os.path.isdir(alt_dir):
                continue
            for img_name in sorted(os.listdir(alt_dir)):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    drone_path = os.path.join(alt_dir, img_name)
                    self.pairs.append((drone_path, sat_path))

        print(f"  [SUES-200 {split} alt={altitude}] {len(self.pairs)} pairs "
              f"({len(locs)} locations)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        drone_path, sat_path = self.pairs[idx]
        try:
            drone = Image.open(drone_path).convert("RGB")
            sat = Image.open(sat_path).convert("RGB")
        except Exception:
            drone = Image.new("RGB", (224, 224), (128,128,128))
            sat = Image.new("RGB", (224, 224), (128,128,128))
        return {"query": self.drone_tf(drone), "gallery": self.sat_tf(sat), "idx": idx}


class SUES200GalleryDataset(Dataset):
    """Satellite gallery: 1 image per test location."""
    def __init__(self, root, test_locs=None, img_size=224):
        super().__init__()
        satellite_dir = os.path.join(root, "satellite-view")
        if test_locs is None:
            test_locs = TEST_LOCS
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.images = []
        self.loc_ids = []
        for loc_id in test_locs:
            loc_str = f"{loc_id:04d}"
            sat_path = os.path.join(satellite_dir, loc_str, "0.png")
            if os.path.exists(sat_path):
                self.images.append(sat_path)
                self.loc_ids.append(loc_id)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return {"image": self.tf(img), "loc_id": self.loc_ids[idx]}


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model, data_root, altitude, device, test_locs=None):
    model.eval()

    # Drone queries
    query_ds = SUES200Dataset(data_root, "test", altitude, IMG_SIZE, test_locs=test_locs)
    query_loader = DataLoader(query_ds, batch_size=64, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Satellite gallery
    gallery_ds = SUES200GalleryDataset(data_root, test_locs, IMG_SIZE)
    gallery_loader = DataLoader(gallery_ds, batch_size=64, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

    # Extract gallery embeddings (unique satellites)
    gal_embs = []
    gal_locs = []
    for batch in gallery_loader:
        emb = model.extract_embedding(batch["image"].to(device))
        gal_embs.append(emb.cpu())
        gal_locs.extend(batch["loc_id"].tolist())
    gal_embs = torch.cat(gal_embs, 0)
    gal_locs = np.array(gal_locs)

    # Extract query embeddings
    q_embs = []
    # Build ground-truth: each drone image's location → index in gallery
    q_gt_indices = []
    loc_to_gal_idx = {loc: i for i, loc in enumerate(gal_locs)}

    for batch in query_loader:
        emb = model.extract_embedding(batch["query"].to(device))
        q_embs.append(emb.cpu())

    q_embs = torch.cat(q_embs, 0)

    # Build ground truth mapping
    # Each drone pair → which location → which gallery index
    for drone_path, sat_path in query_ds.pairs:
        # Extract location from sat_path: .../satellite-view/0121/0.png → 121
        loc_str = os.path.basename(os.path.dirname(sat_path))
        loc_id = int(loc_str)
        q_gt_indices.append(loc_to_gal_idx.get(loc_id, -1))

    q_gt_indices = np.array(q_gt_indices)

    # Compute similarity
    sim = q_embs.numpy() @ gal_embs.numpy().T
    ranks = np.argsort(-sim, axis=1)

    N = len(q_embs)
    results = {}
    for k in [1, 5, 10]:
        correct = sum(1 for i in range(N) if q_gt_indices[i] in ranks[i, :k])
        results[f"R@{k}"] = correct / N

    # AP
    ap_sum = 0
    for i in range(N):
        gt = q_gt_indices[i]
        rank_pos = np.where(ranks[i] == gt)[0]
        if len(rank_pos) > 0:
            ap_sum += 1.0 / (rank_pos[0] + 1)
    results["AP"] = ap_sum / N

    return results


# =============================================================================
# LR SCHEDULER
# =============================================================================
def get_cosine_lr(epoch, total_epochs, base_lr, warmup_epochs=3):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def get_stage_lrs(epoch):
    if epoch < STAGE1_END:
        return 0.0, get_cosine_lr(epoch, STAGE1_END, LR_HEAD_S1, WARMUP_EPOCHS)
    elif epoch < STAGE2_END:
        se = epoch - STAGE1_END; sl = STAGE2_END - STAGE1_END
        return get_cosine_lr(se, sl, LR_BACKBONE_S2, 2), get_cosine_lr(se, sl, LR_HEAD_S2, 0)
    else:
        se = epoch - STAGE2_END; sl = EPOCHS - STAGE2_END
        return get_cosine_lr(se, sl, LR_BACKBONE_S3, 0), get_cosine_lr(se, sl, LR_HEAD_S3, 0)


# =============================================================================
# TRAINING
# =============================================================================
def train(model, train_loader, val_fn, device, epochs=EPOCHS):
    criterion = ThreeStageLoss().to(device)
    bb_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    head_params += list(criterion.parameters())

    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": 0.0},
        {"params": head_params, "lr": LR_HEAD_S1},
    ], weight_decay=WEIGHT_DECAY)

    scaler = GradScaler(enabled=AMP_ENABLED and device.type == "cuda")
    global_step = 0
    best_r1 = 0.0
    history = []

    for epoch in range(epochs):
        lr_bb, lr_hd = get_stage_lrs(epoch)
        optimizer.param_groups[0]["lr"] = lr_bb
        optimizer.param_groups[1]["lr"] = lr_hd

        if epoch < STAGE1_END:
            for p in bb_params: p.requires_grad = False
            stage = "S1:frozen"
        else:
            for p in bb_params: p.requires_grad = True
            stage = "S2:+CSM" if epoch < STAGE2_END else "S3:full"

        model.train()
        ep_loss = ep_acc = n = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"  Ep {epoch+1}/{epochs} ({stage})", leave=False)
        for batch in pbar:
            query = batch["query"].to(device)
            gallery = batch["gallery"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=AMP_ENABLED and device.type == "cuda"):
                out = model(query, gallery, global_step)
                loss_dict = criterion(out, epoch)
                loss = loss_dict["total_loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                global_step += 1; continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            ep_loss += loss.item(); ep_acc += loss_dict["accuracy"].item(); n += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{loss_dict['accuracy'].item():.1%}")

        elapsed = time.time() - t0
        ep_loss /= max(n, 1); ep_acc /= max(n, 1)
        entry = {"epoch": epoch+1, "stage": stage, "loss": round(ep_loss, 4),
                 "acc": round(ep_acc, 4), "lr_bb": round(lr_bb, 6),
                 "lr_hd": round(lr_hd, 6), "time": round(elapsed, 1)}

        if (epoch+1) % EVAL_FREQ == 0 or epoch == epochs - 1:
            metrics = val_fn()
            entry.update(metrics)
            r1 = metrics["R@1"]
            if r1 > best_r1:
                best_r1 = r1
                # Save best
                torch.save(model.state_dict(),
                           os.path.join(OUTPUT_DIR, "geoslot_sues200_best.pth"))
            print(f"  Ep {epoch+1} ({stage}) | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | "
                  f"R@1={r1:.2%} | R@5={metrics.get('R@5', 0):.2%} | AP={metrics.get('AP', 0):.2%} | "
                  f"LR={lr_bb:.1e}/{lr_hd:.1e} | {elapsed:.0f}s")
        else:
            print(f"  Ep {epoch+1} ({stage}) | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | "
                  f"LR={lr_bb:.1e}/{lr_hd:.1e} | {elapsed:.0f}s")

        history.append(entry)

    return best_r1, history


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*70)
    print("  GeoSlot 2.0 — Full Method on SUES-200")
    print(f"  Backbone: DINOv2 ViT-B/14")
    print(f"  3-Stage: S1[0-{STAGE1_END}) S2[{STAGE1_END}-{STAGE2_END}) S3[{STAGE2_END}-{EPOCHS})")
    print(f"  Train: locations 0001-0120 | Test: 0121-0200")
    print(f"  Device: {DEVICE}")
    print("="*70)

    # Dataset (train with all 4 altitudes mixed)
    print("\n[DATASET] Loading SUES-200...")
    train_pairs_all = []
    for alt in ALTITUDES:
        ds = SUES200Dataset(DATA_ROOT, "train", alt, IMG_SIZE)
        train_pairs_all.extend(ds.pairs)
    print(f"  Total train pairs (all altitudes): {len(train_pairs_all)}")

    # Create a combined training dataset
    class CombinedTrainDataset(Dataset):
        def __init__(self, pairs, img_size=224):
            self.pairs = pairs
            self.drone_tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=3),
                transforms.Pad(10, padding_mode='edge'),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
            self.sat_tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=3),
                transforms.Pad(10, padding_mode='edge'),
                transforms.RandomAffine(90),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        def __len__(self): return len(self.pairs)
        def __getitem__(self, idx):
            dp, sp = self.pairs[idx]
            try:
                drone = Image.open(dp).convert("RGB")
                sat = Image.open(sp).convert("RGB")
            except Exception:
                drone = Image.new("RGB", (224, 224), (128,128,128))
                sat = Image.new("RGB", (224, 224), (128,128,128))
            return {"query": self.drone_tf(drone), "gallery": self.sat_tf(sat), "idx": idx}

    train_ds = CombinedTrainDataset(train_pairs_all, IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    # Backbone
    print("\n[BACKBONE] Loading DINOv2 ViT-B/14...")
    backbone = ViTBackbone(BACKBONE_NAME)

    # Model
    model = GeoSlotV2(backbone).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Eval function (closure)
    def val_fn():
        return evaluate(model, DATA_ROOT, TEST_ALTITUDE, DEVICE)

    # Train
    print("\n[TRAINING] Starting 3-stage training...")
    best_r1, history = train(model, train_loader, val_fn, DEVICE, EPOCHS)

    # Final: evaluate all altitudes
    print("\n" + "="*70)
    print("  FINAL RESULTS — All Altitudes")
    print("="*70)
    for alt in ALTITUDES:
        metrics = evaluate(model, DATA_ROOT, alt, DEVICE)
        print(f"  Alt={alt}m | R@1={metrics['R@1']:.2%} | R@5={metrics['R@5']:.2%} | "
              f"R@10={metrics['R@10']:.2%} | AP={metrics['AP']:.2%}")
    print(f"\n  Best R@1 (alt={TEST_ALTITUDE}m during training): {best_r1:.2%}")

    # Save results
    results = {
        "method": "GeoSlot_v2_full",
        "backbone": BACKBONE_NAME,
        "dataset": "SUES-200",
        "best_r1": best_r1,
        "history": history,
        "settings": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "stage1_end": STAGE1_END, "stage2_end": STAGE2_END,
            "max_slots": MAX_SLOTS, "n_register": N_REGISTER,
        }
    }
    results_path = os.path.join(OUTPUT_DIR, "geoslot_sues200_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print("="*70)


if __name__ == "__main__":
    main()
