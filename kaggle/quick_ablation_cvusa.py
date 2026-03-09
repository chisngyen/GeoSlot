# =============================================================================
# QUICK ABLATION TEST: GeoSlot 2.0 — Module-by-Module Evaluation on CVUSA (20%)
# =============================================================================
# Purpose: Test EACH module individually to see if it contributes positive R@1
#
# 6 Configs:
#   A1: Backbone + GAP  → Cosine  (Baseline: no GeoSlot modules)
#   A2: + Slot Attention            (Object-Centric decomposition)
#   A3: + Adaptive BG Mask          (γ learned per-image, replaces α=0.7)
#   A4: + Graph Mamba (Hilbert)     (Relational reasoning + Hilbert curve)
#   A5: + FGW OT (balanced)        (Graph-to-Graph matching)
#   A6: + UFGW (unbalanced)        (Full pipeline with occlusion handling)
#
# Each config: 10 epochs, 20% CVUSA data → ~5-8 min per config on H100
# Total: ~30-50 min
# =============================================================================

# === SETUP (Auto-install) ===
import subprocess, sys, re

def run(cmd, verbose=False):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if verbose or result.returncode != 0:
        if result.stdout: print(result.stdout[-500:])
        if result.stderr: print(f"[WARN] {result.stderr[-500:]}")
    return result.returncode == 0

def pip(pkg, extra=""):
    return run(f"pip install -q {extra} {pkg}")

print("[1/6] Detecting CUDA version...")
r = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
cuda_match = re.search(r"CUDA Version:\s*([\d.]+)", r.stdout)
hw_cuda = cuda_match.group(1) if cuda_match else "12.6"
major = int(hw_cuda.split(".")[0])
minor = int(hw_cuda.split(".")[1]) if "." in hw_cuda else 0
if major >= 13 or (major == 12 and minor >= 6):
    cu_tag = "cu126"
elif major == 12 and minor >= 4:
    cu_tag = "cu124"
elif major == 12 and minor >= 1:
    cu_tag = "cu121"
else:
    cu_tag = "cu118"
print(f"  Hardware CUDA: {hw_cuda} -> Using: {cu_tag}")

print("[2/6] Syncing PyTorch...")
run(f"pip install -q -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cu_tag}")

print("[3/6] Installing base packages...")
pip("transformers==4.44.2")
for p in ["timm", "tqdm"]:
    try: __import__(p)
    except ImportError: pip(p)

print("[4/6] Build tools + nvcc PATH...")
pip("packaging ninja wheel setuptools", "--upgrade")
import os
for cuda_path in ["/usr/local/cuda/bin", "/usr/local/cuda-12/bin", "/usr/local/cuda-12.6/bin"]:
    if os.path.isdir(cuda_path):
        os.environ["PATH"] = cuda_path + ":" + os.environ.get("PATH", "")
        os.environ["CUDA_HOME"] = os.path.dirname(cuda_path)
        break

print("[5/6] Building causal-conv1d...")
pip("causal-conv1d>=1.4.0", "--no-build-isolation")

print("[6/6] Building mamba_ssm (5-10 min)...")
ok = pip("mamba_ssm", "--no-build-isolation --no-cache-dir")
if not ok:
    run("pip install -q --no-build-isolation git+https://github.com/state-spaces/mamba.git")

try:
    from mamba_ssm import Mamba
    print("  mamba_ssm OK!")
    MAMBA_AVAILABLE = True
except ImportError:
    print("  mamba_ssm NOT available -> LinearAttention fallback")
    MAMBA_AVAILABLE = False


# =============================================================================
# IMPORTS
# =============================================================================
import os, math, glob, json, time, gc, random
from typing import Optional, Dict
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
from transformers import AutoModel


# =============================================================================
# CONFIG
# =============================================================================
CVUSA_ROOT     = "/kaggle/input/datasets/chinguyeen/cvusa-subdataset/CVUSA"
OUTPUT_DIR     = "/kaggle/working"

# --- Quick Ablation Settings ---
DATA_RATIO     = 0.20         # 20% of CVUSA
EPOCHS         = 10           # Quick: 10 epochs per config
BATCH_SIZE     = 32
NUM_WORKERS    = 4
LR_BACKBONE    = 1e-5
LR_HEAD        = 1e-4
WEIGHT_DECAY   = 0.01
AMP_ENABLED    = True
EVAL_FREQ      = 5            # Evaluate every 5 epochs

# --- Model ---
BACKBONE_NAME  = "nvidia/MambaVision-L-1K"
FEATURE_DIM    = 1568
SLOT_DIM       = 256
MAX_SLOTS      = 8            # Smaller for speed
N_REGISTER     = 4
EMBED_DIM_OUT  = 512
N_HEADS        = 4
SA_ITERS       = 3
GM_LAYERS      = 2
SINKHORN_ITERS = 10
FGW_ITERS      = 5            # Fewer for speed

SAT_SIZE       = 224
PANO_SIZE      = (128, 512)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# BACKBONE
# =============================================================================
class MambaVisionBackbone(nn.Module):
    def __init__(self, model_name="nvidia/MambaVision-L-1K", frozen=False):
        super().__init__()
        old_linspace = torch.linspace
        def patched_linspace(*args, **kwargs):
            kwargs["device"] = "cpu"
            return old_linspace(*args, **kwargs)
        torch.linspace = patched_linspace
        try:
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True, low_cpu_mem_usage=False)
        finally:
            torch.linspace = old_linspace
        if hasattr(self.model, 'head'):
            self.model.head = nn.Identity()
        self.feature_dim = FEATURE_DIM
        if frozen:
            for p in self.model.parameters(): p.requires_grad = False

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            _, features = self.model(x.float())
        feat = features[-1]
        B, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, H * W, C), (H, W)


# =============================================================================
# SSM BLOCK (Mamba or LinearAttention fallback)
# =============================================================================
if MAMBA_AVAILABLE:
    class SSMBlock(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.mamba = Mamba(d_model=d_model, d_state=d_state,
                              d_conv=d_conv, expand=expand)
            self.norm = nn.LayerNorm(d_model)
        def forward(self, x):
            return x + self.mamba(self.norm(x))
else:
    class SSMBlock(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            d_inner = int(d_model * expand)
            self.proj_in = nn.Linear(d_model, d_inner * 2)
            self.proj_out = nn.Linear(d_inner, d_model)
            self.norm = nn.LayerNorm(d_model)
            self.act = nn.SiLU()
        def forward(self, x):
            z, gate = self.proj_in(self.norm(x)).chunk(2, dim=-1)
            return x + self.proj_out(self.act(z) * torch.sigmoid(gate))


# =============================================================================
# MODULE: Adaptive Gumbel-Sparsity Mask
# =============================================================================
class AdaptiveGumbelSparsityMask(nn.Module):
    """
    Adaptive background suppression with learned gamma.
    gamma = sigmoid(MLP(GAP(F))) adapts per-image.
    Desert: gamma -> 1.0 (keep all). Urban: gamma -> 0.4 (suppress 60%).
    """
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
        mask = self.mask_head(features)           # [B, N, 1]
        gap = features.mean(dim=1)                # [B, D]
        gamma = self.gamma_head(gap)              # [B, 1]
        return features * mask, mask, gamma


class StaticBackgroundMask(nn.Module):
    """Original static mask (alpha=0.7) for comparison."""
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(True),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, features):
        mask = self.net(features)
        return features * mask, mask, None  # gamma=None -> static


# =============================================================================
# MODULE: Slot Attention
# =============================================================================
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


# =============================================================================
# MODULE: Spatial Encoder + Graph Mamba
# =============================================================================
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
        spreads = torch.ones_like(centroids) * 0.1  # Simplified
        feats = torch.cat([centroids, spreads], dim=-1)
        device = feats.device
        freqs = 1.0 / (10000.0 ** (torch.arange(0, self.pos_dim, 2, device=device).float() / self.pos_dim))
        angles = feats.unsqueeze(-1) * freqs.view(1,1,1,-1) * 3.14159
        enc = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2, -1)
        return self.pos_mlp(enc), centroids


def xy2d_hilbert(n, x, y):
    """Hilbert curve 2D -> 1D mapping."""
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
    """Sort slot centroids by Hilbert curve index."""
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


class GraphMambaLayer(nn.Module):
    def __init__(self, dim, num_layers=2, spatial_weight=0.3, use_hilbert=True):
        super().__init__()
        self.spatial_encoder = SlotSpatialEncoder(dim)
        self.fwd = nn.ModuleList([SSMBlock(dim) for _ in range(num_layers)])
        self.bwd = nn.ModuleList([SSMBlock(dim) for _ in range(num_layers)])
        self.merge = nn.ModuleList([nn.Linear(dim*2, dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*4),
                                   nn.GELU(), nn.Linear(dim*4, dim)) for _ in range(num_layers)])
        self.spatial_weight = spatial_weight
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
            elif self.training:
                order = torch.stack([torch.randperm(K, device=slots.device) for _ in range(B)])
            else:
                # Degree ordering
                slots_norm = F.normalize(slots, dim=-1)
                sim = torch.bmm(slots_norm, slots_norm.transpose(1,2))
                order = sim.sum(dim=-1).argsort(dim=-1, descending=True)

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


# =============================================================================
# MODULE: OT Matching (Sinkhorn + FGW)
# =============================================================================
class SinkhornOT(nn.Module):
    def __init__(self, dim, num_iters=10, epsilon=0.05, mesh_iters=3):
        super().__init__()
        self.num_iters = num_iters; self.epsilon = epsilon; self.mesh_iters = mesh_iters
        self.cost_proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, dim))

    def forward(self, slots_q, slots_r, mask_q=None, mask_r=None, **kwargs):
        slots_q = slots_q.float(); slots_r = slots_r.float()
        if mask_q is not None: mask_q = mask_q.float()
        if mask_r is not None: mask_r = mask_r.float()
        pq = self.cost_proj(slots_q); pr = self.cost_proj(slots_r)
        diff = pq.unsqueeze(2) - pr.unsqueeze(1)
        C = (diff*diff).sum(-1).clamp(min=1e-6).sqrt()
        B, K, M = C.shape
        log_K = -C / self.epsilon
        if mask_q is not None:
            log_mu = torch.log(mask_q.clamp(min=1e-8))
            log_mu = log_mu - torch.logsumexp(log_mu, dim=-1, keepdim=True)
            log_K = log_K + log_mu.unsqueeze(-1)
        if mask_r is not None:
            log_nu = torch.log(mask_r.clamp(min=1e-8))
            log_nu = log_nu - torch.logsumexp(log_nu, dim=-1, keepdim=True)
            log_K = log_K + log_nu.unsqueeze(-2)
        log_a = torch.zeros(B, K, 1, device=C.device)
        log_b = torch.zeros(B, 1, M, device=C.device)
        for _ in range(self.num_iters):
            log_a = -torch.logsumexp(log_K + log_b, dim=2, keepdim=True)
            log_b = -torch.logsumexp(log_K + log_a, dim=1, keepdim=True)
        T = torch.exp(log_K + log_a + log_b)
        for _ in range(self.mesh_iters):
            T = T ** 2
            T = T / (T.sum(dim=-1, keepdim=True) + 1e-8)
            T = T / (T.sum(dim=-2, keepdim=True) + 1e-8)
        cost = (T * C).sum(dim=(-1, -2))
        return {"similarity": torch.sigmoid(-cost), "transport_plan": T,
                "cost_matrix": C, "transport_cost": cost}


class FusedGromovWasserstein(nn.Module):
    """FGW: matches node features AND graph structure."""
    def __init__(self, dim, lambda_fgw=0.5, tau_kl=0.1,
                 n_outer=5, n_sinkhorn=10, epsilon=0.05):
        super().__init__()
        self.lambda_fgw = lambda_fgw; self.tau_kl = tau_kl
        self.n_outer = n_outer; self.n_sinkhorn = n_sinkhorn
        self.log_eps = nn.Parameter(torch.tensor(math.log(epsilon)))
        self.cost_proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, dim))

    @property
    def epsilon(self): return self.log_eps.exp()

    def forward(self, slots_q, slots_r, mask_q=None, mask_r=None,
                centroids_q=None, centroids_r=None):
        slots_q = slots_q.float(); slots_r = slots_r.float()
        pq = self.cost_proj(slots_q); pr = self.cost_proj(slots_r)
        diff = pq.unsqueeze(2) - pr.unsqueeze(1)
        C = (diff*diff).sum(-1).clamp(min=1e-6).sqrt()
        B, K, M = C.shape

        # Structure costs
        if centroids_q is not None and centroids_r is not None:
            centroids_q = centroids_q.float(); centroids_r = centroids_r.float()
            dq = centroids_q.unsqueeze(2) - centroids_q.unsqueeze(1)
            Sq = (dq*dq).sum(-1).clamp(min=1e-6).sqrt()
            dr = centroids_r.unsqueeze(2) - centroids_r.unsqueeze(1)
            Sr = (dr*dr).sum(-1).clamp(min=1e-6).sqrt()
        else:
            Sq = torch.zeros(B, K, K, device=C.device)
            Sr = torch.zeros(B, M, M, device=C.device)

        # Marginals
        if mask_q is not None:
            mu = mask_q.float() / (mask_q.float().sum(-1, keepdim=True) + 1e-8)
        else:
            mu = torch.ones(B, K, device=C.device) / K
        if mask_r is not None:
            nu = mask_r.float() / (mask_r.float().sum(-1, keepdim=True) + 1e-8)
        else:
            nu = torch.ones(B, M, device=C.device) / M

        # Init transport
        T = mu.unsqueeze(2) * nu.unsqueeze(1)

        eps = self.epsilon
        rho = self.tau_kl / (self.tau_kl + eps)

        for _ in range(self.n_outer):
            # GW cost: |Sq_ik - Sr_jl|^2 T_kl
            Sq2 = Sq * Sq; Sr2 = Sr * Sr
            t1 = torch.bmm(Sq2, T.sum(dim=2, keepdim=True)).squeeze(-1).unsqueeze(2).expand_as(T)
            t2 = torch.bmm(T.sum(dim=1, keepdim=True), Sr2).squeeze(1).unsqueeze(1).expand_as(T)
            t3 = -2.0 * torch.bmm(Sq, torch.bmm(T, Sr))
            L_gw = t1 + t2 + t3
            C_fgw = (1 - self.lambda_fgw) * C + self.lambda_fgw * L_gw

            # Unbalanced Sinkhorn
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
# 6 PIPELINE CONFIGURATIONS
# =============================================================================
class ConfigA1_BaselineGAP(nn.Module):
    """A1: Backbone + GAP -> Cosine. No GeoSlot modules."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.LayerNorm(FEATURE_DIM), nn.Linear(FEATURE_DIM, EMBED_DIM_OUT))

    def encode_view(self, x, global_step=None):
        features, (H, W) = self.backbone(x)
        with torch.cuda.amp.autocast(enabled=False):
            features = features.float()
            emb = F.normalize(self.head(features.mean(dim=1)), dim=-1)
        return {"embedding": emb, "slots": None, "keep_mask": None,
                "bg_mask": None, "adaptive_gamma": None,
                "attn_maps": None, "keep_probs": None, "register_slots": None,
                "centroids": None}

    def forward(self, q_img, r_img, global_step=None):
        q = self.encode_view(q_img, global_step)
        r = self.encode_view(r_img, global_step)
        sim = (q["embedding"] * r["embedding"]).sum(dim=-1)
        return {"query_embedding": q["embedding"], "ref_embedding": r["embedding"],
                "similarity": sim, "transport_plan": None, "transport_cost": torch.zeros(q_img.shape[0], device=q_img.device),
                "query_slots": None, "ref_slots": None,
                "query_keep_mask": None, "ref_keep_mask": None,
                "query_bg_mask": None, "ref_bg_mask": None,
                "query_adaptive_gamma": None, "ref_adaptive_gamma": None,
                "query_attn_maps": None, "ref_attn_maps": None,
                "query_keep_probs": None, "ref_keep_probs": None}

    def extract_embedding(self, x, global_step=None):
        return self.encode_view(x, global_step)["embedding"]


class ConfigA2toA6(nn.Module):
    """A2-A6: Progressively adding modules."""
    def __init__(self, backbone, use_slots=True, use_adaptive_mask=False,
                 use_graph=False, use_hilbert=False,
                 matching='cosine', lambda_fgw=0.5, tau_kl=0.1):
        super().__init__()
        self.backbone = backbone
        self.use_slots = use_slots
        self.use_graph = use_graph
        self.matching_type = matching

        # Background mask
        if use_adaptive_mask:
            self.bg_mask = AdaptiveGumbelSparsityMask(FEATURE_DIM)
        else:
            self.bg_mask = StaticBackgroundMask(FEATURE_DIM)

        if use_slots:
            total = MAX_SLOTS + N_REGISTER
            self.input_proj = nn.Linear(FEATURE_DIM, SLOT_DIM)
            self.slot_mu = nn.Parameter(torch.randn(1, total, SLOT_DIM) * (SLOT_DIM**-0.5))
            self.slot_log_sigma = nn.Parameter(torch.zeros(1, total, SLOT_DIM))
            self.slot_attn = SlotAttentionCore(SLOT_DIM, SLOT_DIM, N_HEADS, SA_ITERS)
            self.gumbel = GumbelSelector(SLOT_DIM)

            if use_graph:
                self.graph_mamba = GraphMambaLayer(SLOT_DIM, GM_LAYERS, use_hilbert=use_hilbert)

            if matching == 'sinkhorn':
                self.ot_matcher = SinkhornOT(SLOT_DIM, SINKHORN_ITERS)
            elif matching == 'fgw':
                self.ot_matcher = FusedGromovWasserstein(SLOT_DIM, lambda_fgw, tau_kl=1e6, n_outer=FGW_ITERS)
            elif matching == 'ufgw':
                self.ot_matcher = FusedGromovWasserstein(SLOT_DIM, lambda_fgw, tau_kl, n_outer=FGW_ITERS)

            embed_input = SLOT_DIM
        else:
            embed_input = FEATURE_DIM

        self.embed_head = nn.Sequential(nn.LayerNorm(embed_input), nn.Linear(embed_input, EMBED_DIM_OUT))

    def encode_view(self, x, global_step=None):
        features, (H, W) = self.backbone(x)
        with torch.cuda.amp.autocast(enabled=False):
            features = features.float()
            features_masked, bg_mask, gamma = self.bg_mask(features)

            if not self.use_slots:
                emb = F.normalize(self.embed_head(features_masked.mean(dim=1)), dim=-1)
                return {"embedding": emb, "slots": None, "keep_mask": None,
                        "bg_mask": bg_mask, "adaptive_gamma": gamma,
                        "attn_maps": None, "keep_probs": None,
                        "register_slots": None, "centroids": None}

            proj = self.input_proj(features_masked)
            B = features.shape[0]
            mu = self.slot_mu.expand(B, -1, -1)
            sigma = self.slot_log_sigma.exp().expand(B, -1, -1)
            slots = mu + sigma * torch.randn_like(mu)
            slots, attn_maps = self.slot_attn(proj, slots)
            obj_slots = slots[:, :MAX_SLOTS]
            reg_slots = slots[:, MAX_SLOTS:]
            keep_decision, keep_probs = self.gumbel(obj_slots, global_step)
            obj_slots = obj_slots * keep_decision.unsqueeze(-1)

            centroids = None
            if self.use_graph:
                obj_slots, centroids = self.graph_mamba(
                    obj_slots, keep_decision, attn_maps=attn_maps, spatial_hw=(H, W))

            weights = keep_decision / (keep_decision.sum(dim=-1, keepdim=True) + 1e-8)
            global_slot = (obj_slots * weights.unsqueeze(-1)).sum(dim=1)
            emb = F.normalize(self.embed_head(global_slot), dim=-1)

        return {"embedding": emb, "slots": obj_slots, "keep_mask": keep_decision,
                "bg_mask": bg_mask, "adaptive_gamma": gamma,
                "attn_maps": attn_maps, "keep_probs": keep_probs,
                "register_slots": reg_slots, "centroids": centroids}

    def forward(self, q_img, r_img, global_step=None):
        q = self.encode_view(q_img, global_step)
        r = self.encode_view(r_img, global_step)

        ot_out = None
        if self.use_slots and self.matching_type in ('sinkhorn', 'fgw', 'ufgw'):
            ot_out = self.ot_matcher(
                q["slots"], r["slots"],
                mask_q=q["keep_mask"], mask_r=r["keep_mask"],
                centroids_q=q.get("centroids"), centroids_r=r.get("centroids"))

        if ot_out is not None:
            sim = ot_out["similarity"]
            tp = ot_out["transport_plan"]
            tc = ot_out["transport_cost"]
        else:
            sim = (q["embedding"] * r["embedding"]).sum(dim=-1)
            tp = None
            tc = torch.zeros(q_img.shape[0], device=q_img.device)

        return {"query_embedding": q["embedding"], "ref_embedding": r["embedding"],
                "similarity": sim, "transport_plan": tp, "transport_cost": tc,
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
# LOSSES (Simplified for ablation — InfoNCE + DWBL only)
# =============================================================================
class AblationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(0.07).log())

    @property
    def temp(self): return self.log_temp.exp().clamp(0.01, 1.0)

    def forward(self, out, epoch=0):
        q_emb = out["query_embedding"]
        r_emb = out["ref_embedding"]
        B = q_emb.shape[0]
        logits = q_emb @ r_emb.t() / self.temp
        labels = torch.arange(B, device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        # BG mask regularization
        loss_bg = torch.tensor(0.0, device=loss.device)
        if out.get("query_bg_mask") is not None:
            mask = out["query_bg_mask"]
            p = mask.squeeze(-1).clamp(1e-6, 1-1e-6)
            entropy = -(p * p.log() + (1-p) * (1-p).log()).mean()
            gamma = out.get("query_adaptive_gamma")
            if gamma is not None:
                coverage = F.relu(gamma - mask.mean(dim=1)).mean()
            else:
                coverage = (mask.mean() - 0.7) ** 2
            loss_bg = -entropy + coverage
            loss = loss + 0.01 * loss_bg

        return {"total_loss": loss, "accuracy": acc, "loss_bg": loss_bg}


# =============================================================================
# DATASET
# =============================================================================
class CVUSADataset(Dataset):
    def __init__(self, root, split="train", sat_size=224, pano_size=(128, 512), ratio=0.2):
        super().__init__()
        self.split = split; self.pairs = []
        self.sat_tf = transforms.Compose([
            transforms.Resize((sat_size, sat_size)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.pano_tf = transforms.Compose([
            transforms.Resize(pano_size), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.pano_tf_aug = transforms.Compose([
            transforms.Resize(pano_size), transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

        subset = os.path.join(root, "CVPR_subset")
        pano_dir = os.path.join(subset, "streetview", "panos")
        sat_dir = os.path.join(subset, "bingmap", "18")
        sf = os.path.join(subset, "splits", "train-19zl.csv" if split == "train" else "val-19zl.csv")

        if os.path.exists(sf):
            with open(sf, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 2: continue
                    pp = None
                    for c in [os.path.join(pano_dir, parts[0].strip()),
                              os.path.join(os.path.dirname(sf), "..", parts[0].strip())]:
                        if os.path.exists(c): pp = c; break
                    sp = None
                    for c in [os.path.join(sat_dir, parts[1].strip()),
                              os.path.join(os.path.dirname(sf), "..", parts[1].strip())]:
                        if os.path.exists(c): sp = c; break
                    if pp and sp: self.pairs.append((pp, sp))
        else:
            # Fallback: match by filename
            if os.path.exists(pano_dir) and os.path.exists(sat_dir):
                panos = sorted(glob.glob(os.path.join(pano_dir, "**", "*.jpg"), recursive=True))
                sat_dict = {os.path.splitext(os.path.basename(s))[0]: s
                            for s in sorted(glob.glob(os.path.join(sat_dir, "**", "*.jpg"), recursive=True))}
                all_pairs = [(p, sat_dict[os.path.splitext(os.path.basename(p))[0]])
                             for p in panos if os.path.splitext(os.path.basename(p))[0] in sat_dict]
                idx = int(len(all_pairs) * 0.8)
                self.pairs = all_pairs[:idx] if split == "train" else all_pairs[idx:]

        # Limit to ratio
        limit = max(32, int(len(self.pairs) * ratio))
        if len(self.pairs) > limit:
            random.seed(42)
            self.pairs = random.sample(self.pairs, limit)
        print(f"  [CVUSA {split}] {len(self.pairs)} pairs")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        pp, sp = self.pairs[idx]
        try:
            pano = Image.open(pp).convert("RGB"); sat = Image.open(sp).convert("RGB")
        except Exception:
            pano = Image.new("RGB", (512, 128), (128,128,128))
            sat = Image.new("RGB", (224, 224), (128,128,128))
        if self.split == "train":
            pano = self.pano_tf_aug(pano)
        else:
            pano = self.pano_tf(pano)
        sat = self.sat_tf(sat)
        return {"query": pano, "gallery": sat, "idx": idx}


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    q_embs, r_embs = [], []
    for batch in val_loader:
        qe = model.extract_embedding(batch["query"].to(device))
        re = model.extract_embedding(batch["gallery"].to(device))
        q_embs.append(qe.cpu()); r_embs.append(re.cpu())
    q_embs = torch.cat(q_embs, 0).numpy()
    r_embs = torch.cat(r_embs, 0).numpy()
    sim = q_embs @ r_embs.T
    ranks = np.argsort(-sim, axis=1)
    N = len(q_embs); gt = np.arange(N)
    results = {}
    for k in [1, 5, 10]:
        correct = sum(1 for i in range(N) if gt[i] in ranks[i, :k])
        results[f"R@{k}"] = correct / N
    return results


# =============================================================================
# TRAIN ONE CONFIG
# =============================================================================
def train_config(config_name, model, train_loader, val_loader, device, epochs=10):
    print(f"\n{'='*70}")
    print(f"  CONFIG: {config_name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*70}")

    model = model.to(device)
    criterion = AblationLoss().to(device)

    bb_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    head_params += list(criterion.parameters())

    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scaler = GradScaler(enabled=AMP_ENABLED and device.type == "cuda")
    global_step = 0
    best_r1 = 0.0
    history = []

    for epoch in range(epochs):
        # Freeze backbone first 2 epochs
        if epoch == 0:
            for p in bb_params: p.requires_grad = False
        elif epoch == 2:
            for p in bb_params: p.requires_grad = True

        model.train()
        ep_loss = ep_acc = n = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"  [{config_name}] Ep {epoch+1}/{epochs}", leave=False)
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

            if AMP_ENABLED and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            ep_loss += loss.item(); ep_acc += loss_dict["accuracy"].item(); n += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{loss_dict['accuracy'].item():.1%}")

        elapsed = time.time() - t0
        ep_loss /= max(n, 1); ep_acc /= max(n, 1)

        entry = {"epoch": epoch+1, "loss": round(ep_loss, 4), "acc": round(ep_acc, 4),
                 "time": round(elapsed, 1)}

        if (epoch+1) % EVAL_FREQ == 0 or epoch == epochs - 1:
            metrics = evaluate(model, val_loader, device)
            entry.update(metrics)
            r1 = metrics["R@1"]
            if r1 > best_r1: best_r1 = r1
            print(f"  [{config_name}] Ep {epoch+1} | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | "
                  f"R@1={r1:.2%} | R@5={metrics['R@5']:.2%} | {elapsed:.0f}s")
        else:
            print(f"  [{config_name}] Ep {epoch+1} | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | {elapsed:.0f}s")

        history.append(entry)

    # Cleanup GPU
    del model, criterion, optimizer, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best_r1, history


# =============================================================================
# MAIN: RUN ALL 6 CONFIGS
# =============================================================================
def main():
    print("\n" + "="*70)
    print("  QUICK ABLATION TEST: GeoSlot 2.0 on CVUSA (20%)")
    print("  6 configs x 10 epochs each")
    print(f"  Device: {DEVICE}")
    print("="*70)

    # === Dataset ===
    print("\n[DATASET] Loading CVUSA...")
    train_ds = CVUSADataset(CVUSA_ROOT, "train", SAT_SIZE, PANO_SIZE, DATA_RATIO)
    val_ds = CVUSADataset(CVUSA_ROOT, "test", SAT_SIZE, PANO_SIZE, DATA_RATIO)

    if len(train_ds) == 0:
        print("[ERROR] No training samples! Check CVUSA_ROOT path.")
        print(f"  CVUSA_ROOT = {CVUSA_ROOT}")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # === Shared Backbone ===
    print("\n[BACKBONE] Loading MambaVision-L...")
    backbone = MambaVisionBackbone(BACKBONE_NAME)

    # === Define 6 Configs ===
    configs = [
        ("A1: Backbone+GAP (Baseline)",
         lambda bb: ConfigA1_BaselineGAP(bb)),
        ("A2: +SlotAttention",
         lambda bb: ConfigA2toA6(bb, use_slots=True, use_adaptive_mask=False,
                                 use_graph=False, matching='cosine')),
        ("A3: +AdaptiveMask",
         lambda bb: ConfigA2toA6(bb, use_slots=True, use_adaptive_mask=True,
                                 use_graph=False, matching='cosine')),
        ("A4: +GraphMamba(Hilbert)",
         lambda bb: ConfigA2toA6(bb, use_slots=True, use_adaptive_mask=True,
                                 use_graph=True, use_hilbert=True, matching='cosine')),
        ("A5: +FGW(balanced)",
         lambda bb: ConfigA2toA6(bb, use_slots=True, use_adaptive_mask=True,
                                 use_graph=True, use_hilbert=True,
                                 matching='fgw', lambda_fgw=0.5, tau_kl=1e6)),
        ("A6: +UFGW(Full)",
         lambda bb: ConfigA2toA6(bb, use_slots=True, use_adaptive_mask=True,
                                 use_graph=True, use_hilbert=True,
                                 matching='ufgw', lambda_fgw=0.5, tau_kl=0.1)),
    ]

    results = {}
    all_history = {}

    for name, model_fn in configs:
        # Re-create backbone (shared pretrained weights, fresh head each time)
        bb = MambaVisionBackbone(BACKBONE_NAME)
        # Copy pretrained weights
        bb.load_state_dict(backbone.state_dict())
        model = model_fn(bb)

        r1, hist = train_config(name, model, train_loader, val_loader, DEVICE, EPOCHS)
        results[name] = r1
        all_history[name] = hist

    # === FINAL RESULTS TABLE ===
    print("\n" + "="*70)
    print("  ABLATION RESULTS SUMMARY")
    print("="*70)
    print(f"  {'Config':<35} {'Best R@1':>10}")
    print(f"  {'-'*35} {'-'*10}")
    prev_r1 = 0.0
    for name, r1 in results.items():
        delta = r1 - prev_r1 if prev_r1 > 0 else 0
        arrow = f" (+{delta:.2%})" if delta > 0 else (f" ({delta:.2%})" if delta < 0 else "")
        print(f"  {name:<35} {r1:>8.2%}{arrow}")
        prev_r1 = r1
    print("="*70)

    # Verdict
    r1_values = list(results.values())
    if len(r1_values) >= 2:
        total_gain = r1_values[-1] - r1_values[0]
        print(f"\n  Total gain (A6 vs A1): +{total_gain:.2%}")
        if total_gain > 0.05:
            print("  VERDICT: PASS -- All modules contribute positively!")
        elif total_gain > 0.02:
            print("  VERDICT: PARTIAL PASS -- Some gain, needs tuning")
        else:
            print("  VERDICT: FAIL -- Modules not contributing enough")

    # Save results
    results_data = {
        "configs": {k: v for k, v in results.items()},
        "history": {k: v for k, v in all_history.items()},
        "settings": {
            "data_ratio": DATA_RATIO, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "max_slots": MAX_SLOTS,
            "mamba_ssm": MAMBA_AVAILABLE,
        }
    }
    results_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print("="*70)


if __name__ == "__main__":
    main()
