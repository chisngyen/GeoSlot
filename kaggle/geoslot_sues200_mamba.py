# =============================================================================
# GeoSlot 2.0 — Full Method on SUES-200 (Drone ↔ Satellite)
# =============================================================================
# BACKBONE: MambaVision-L (core matching backbone of GeoSlot 2.0)
# Method: Adaptive Mask → Slot Attention → Graph Mamba (real Mamba SSM) → UFGW
# 3-Stage Training with Cosine LR
# Images: 512×512 native resolution
#
# Dataset: SUES-200 (200 locations, 4 altitudes, 50 imgs/alt)
#          Train: locations 0001-0120, Test: 0121-0200
# =============================================================================

# === SETUP ===
import subprocess, sys, re, os

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
DATA_ROOT      = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
OUTPUT_DIR     = "/kaggle/working"

EPOCHS         = 50
BATCH_SIZE     = 32           # smaller for MambaVision-L + 512px
NUM_WORKERS    = 4
AMP_ENABLED    = True
EVAL_FREQ      = 5

# 3-Stage
STAGE1_END     = 15
STAGE2_END     = 35

# LR
LR_HEAD_S1     = 3e-4
LR_BACKBONE_S2 = 1e-5
LR_HEAD_S2     = 1e-4
LR_BACKBONE_S3 = 5e-6
LR_HEAD_S3     = 5e-5
WEIGHT_DECAY   = 0.01
WARMUP_EPOCHS  = 3

# Loss
LAMBDA_CSM     = 0.3
LAMBDA_DICE    = 0.1
LAMBDA_BG      = 0.01
CSM_WARMUP     = 5

# Model
BACKBONE_NAME  = "nvidia/MambaVision-L-1K"
FEATURE_DIM    = 1568         # MambaVision-L last stage dim
SLOT_DIM       = 256
MAX_SLOTS      = 8
N_REGISTER     = 4
EMBED_DIM_OUT  = 512
N_HEADS        = 4
SA_ITERS       = 3
GM_LAYERS      = 2
FGW_ITERS      = 5

# Dataset
IMG_SIZE       = 512
TRAIN_LOCS     = list(range(1, 121))
TEST_LOCS      = list(range(121, 201))
ALTITUDES      = ["150", "200", "250", "300"]
TEST_ALTITUDE  = "150"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# BACKBONE: MambaVision-L
# =============================================================================
class MambaVisionBackbone(nn.Module):
    def __init__(self, model_name=BACKBONE_NAME):
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

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            _, features = self.model(x.float())
        feat = features[-1]
        B, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, H * W, C), (H, W)


# =============================================================================
# SSM BLOCK (real Mamba or fallback)
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
# MODULES: Mask, Slots, GraphMamba, UFGW
# =============================================================================
class AdaptiveGumbelSparsityMask(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.mask_head = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(True), nn.Linear(hidden, 1), nn.Sigmoid())
        self.gamma_head = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(True), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, features):
        mask = self.mask_head(features)
        gamma = self.gamma_head(features.mean(dim=1))
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
        self.net = nn.Sequential(nn.Linear(dim, dim//2), nn.ReLU(True), nn.Linear(dim//2, 2))

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
        return decision, F.softmax(logits, dim=-1)[..., 1]


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
        return torch.einsum('bkn,nc->bkc', attn_prob, coords)

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
    hi = torch.zeros(B, K, device=device)
    for b in range(B):
        for k in range(K):
            hi[b, k] = xy2d_hilbert(grid_size, cx[b,k].item(), cy[b,k].item())
    return hi.argsort(dim=-1)


class GraphMambaLayer(nn.Module):
    def __init__(self, dim, num_layers=2, use_hilbert=True):
        super().__init__()
        self.spatial_encoder = SlotSpatialEncoder(dim)
        self.fwd = nn.ModuleList([SSMBlock(dim) for _ in range(num_layers)])
        self.bwd = nn.ModuleList([SSMBlock(dim) for _ in range(num_layers)])
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
            pos_enc, centroids = self.spatial_encoder(attn_maps[:, :K, :], H, W)
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
    def __init__(self, dim, lambda_fgw=0.5, tau_kl=0.1, n_outer=5, n_sinkhorn=10, epsilon=0.05):
        super().__init__()
        self.lambda_fgw = lambda_fgw; self.tau_kl = tau_kl
        self.n_outer = n_outer; self.n_sinkhorn = n_sinkhorn
        self.log_eps = nn.Parameter(torch.tensor(math.log(epsilon)))
        self.cost_proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, dim))

    @property
    def epsilon(self): return self.log_eps.exp().clamp(0.01, 0.5)

    def forward(self, sq, sr, mask_q=None, mask_r=None, centroids_q=None, centroids_r=None):
        sq = sq.float(); sr = sr.float()
        pq = self.cost_proj(sq); pr = self.cost_proj(sr)
        C = (pq.unsqueeze(2) - pr.unsqueeze(1)).pow(2).sum(-1).clamp(min=1e-6).sqrt()
        B, K, M = C.shape
        if centroids_q is not None and centroids_r is not None:
            cq = centroids_q.float(); cr = centroids_r.float()
            Sq = (cq.unsqueeze(2) - cq.unsqueeze(1)).pow(2).sum(-1).clamp(min=1e-6).sqrt()
            Sr = (cr.unsqueeze(2) - cr.unsqueeze(1)).pow(2).sum(-1).clamp(min=1e-6).sqrt()
        else:
            Sq = torch.zeros(B, K, K, device=C.device); Sr = torch.zeros(B, M, M, device=C.device)
        mu = mask_q.float() / (mask_q.float().sum(-1, keepdim=True)+1e-8) if mask_q is not None else torch.ones(B,K,device=C.device)/K
        nu = mask_r.float() / (mask_r.float().sum(-1, keepdim=True)+1e-8) if mask_r is not None else torch.ones(B,M,device=C.device)/M
        T = mu.unsqueeze(2) * nu.unsqueeze(1)
        eps = self.epsilon; rho = self.tau_kl / (self.tau_kl + eps)
        for _ in range(self.n_outer):
            Sq2 = Sq*Sq; Sr2 = Sr*Sr
            t1 = torch.bmm(Sq2, T.sum(2,keepdim=True)).squeeze(-1).unsqueeze(2).expand_as(T)
            t2 = torch.bmm(T.sum(1,keepdim=True), Sr2).squeeze(1).unsqueeze(1).expand_as(T)
            L_gw = t1 + t2 - 2.0 * torch.bmm(Sq, torch.bmm(T, Sr))
            C_fgw = (1-self.lambda_fgw)*C + self.lambda_fgw*L_gw
            log_K = -C_fgw / eps
            log_mu = torch.log(mu.clamp(min=1e-8)); log_nu = torch.log(nu.clamp(min=1e-8))
            log_u = torch.zeros(B,K,device=C.device); log_v = torch.zeros(B,M,device=C.device)
            for _ in range(self.n_sinkhorn):
                log_u = rho * (log_mu - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2))
                log_v = rho * (log_nu - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1))
            T = torch.exp(log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1))
        cost = (T * C_fgw).sum(dim=(-1,-2))
        return {"similarity": torch.sigmoid(-cost), "transport_plan": T,
                "cost_matrix": C, "transport_cost": cost}


# =============================================================================
# FULL PIPELINE
# =============================================================================
class GeoSlotV2(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.bg_mask = AdaptiveGumbelSparsityMask(FEATURE_DIM)
        total = MAX_SLOTS + N_REGISTER
        self.input_proj = nn.Linear(FEATURE_DIM, SLOT_DIM)
        self.slot_mu = nn.Parameter(torch.randn(1, total, SLOT_DIM) * (SLOT_DIM**-0.5))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, total, SLOT_DIM))
        self.slot_attn = SlotAttentionCore(SLOT_DIM, SLOT_DIM, N_HEADS, SA_ITERS)
        self.gumbel = GumbelSelector(SLOT_DIM)
        self.graph_mamba = GraphMambaLayer(SLOT_DIM, GM_LAYERS, use_hilbert=True)
        self.ot_matcher = FusedGromovWasserstein(SLOT_DIM, 0.5, 0.1, n_outer=FGW_ITERS)
        self.embed_head = nn.Sequential(nn.LayerNorm(SLOT_DIM), nn.Linear(SLOT_DIM, EMBED_DIM_OUT))

    def encode_view(self, x, global_step=None):
        features, (H, W) = self.backbone(x)
        with torch.cuda.amp.autocast(enabled=False):
            features = features.float()
            fm, bg_mask, gamma = self.bg_mask(features)
            proj = self.input_proj(fm)
            B = features.shape[0]
            mu = self.slot_mu.expand(B, -1, -1)
            sigma = self.slot_log_sigma.exp().expand(B, -1, -1)
            slots = mu + sigma * torch.randn_like(mu)
            slots, attn_maps = self.slot_attn(proj, slots)
            obj_slots = slots[:, :MAX_SLOTS]
            keep, kp = self.gumbel(obj_slots, global_step)
            obj_slots = obj_slots * keep.unsqueeze(-1)
            obj_slots, centroids = self.graph_mamba(
                obj_slots, keep, attn_maps=attn_maps, spatial_hw=(H, W))
            w = keep / (keep.sum(dim=-1, keepdim=True) + 1e-8)
            emb = F.normalize(self.embed_head((obj_slots * w.unsqueeze(-1)).sum(dim=1)), dim=-1)
        return {"embedding": emb, "slots": obj_slots, "keep_mask": keep,
                "bg_mask": bg_mask, "adaptive_gamma": gamma,
                "attn_maps": attn_maps, "keep_probs": kp, "centroids": centroids}

    def forward(self, q_img, r_img, global_step=None):
        q = self.encode_view(q_img, global_step)
        r = self.encode_view(r_img, global_step)
        ot = self.ot_matcher(q["slots"], r["slots"],
                             mask_q=q["keep_mask"], mask_r=r["keep_mask"],
                             centroids_q=q.get("centroids"), centroids_r=r.get("centroids"))
        return {"query_embedding": q["embedding"], "ref_embedding": r["embedding"],
                "similarity": ot["similarity"], "transport_plan": ot["transport_plan"],
                "transport_cost": ot["transport_cost"],
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
# 3-STAGE LOSS
# =============================================================================
class ThreeStageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(0.07).log())
        self.csm_temp = 0.1

    @property
    def temp(self): return self.log_temp.exp().clamp(0.01, 1.0)

    def forward(self, out, epoch=0):
        q_emb = out["query_embedding"]; r_emb = out["ref_embedding"]
        B = q_emb.shape[0]
        logits = q_emb @ r_emb.t() / self.temp
        labels = torch.arange(B, device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        total = loss
        l_csm = l_dice = l_bg = torch.tensor(0.0, device=loss.device)

        if epoch >= STAGE1_END and out.get("query_slots") is not None:
            ramp = min(1.0, (epoch - STAGE1_END + 1) / CSM_WARMUP)
            qs = out["query_slots"]; rs = out["ref_slots"]; tp = out.get("transport_plan")
            if qs is not None and rs is not None:
                qn = F.normalize(qs, dim=-1); rn = F.normalize(rs, dim=-1)
                ss = torch.bmm(qn, rn.transpose(1,2)) / self.csm_temp
                if tp is not None:
                    tt = tp.detach(); tt = tt / (tt.sum(-1, keepdim=True)+1e-8)
                    cs = -(tt * F.log_softmax(ss, dim=-1)).sum(-1)
                else:
                    cs = F.cross_entropy(ss.view(-1, ss.shape[-1]),
                         torch.arange(ss.shape[1], device=ss.device).repeat(B),
                         reduction='none').view(B, -1)
                qm = out.get("query_keep_mask")
                l_csm = (cs * qm).sum() / (qm.sum()+1e-8) if qm is not None else cs.mean()
                total = total + ramp * LAMBDA_CSM * l_csm
            qa = out.get("query_attn_maps")
            if qa is not None:
                Ko = min(MAX_SLOTS, qa.shape[1])
                oa = qa[:, :Ko]; oa = oa / (oa.sum(-1, keepdim=True)+1e-8)
                ds = 0.0; dc = 0
                for i in range(Ko):
                    for j in range(i+1, Ko):
                        ov = 2*(oa[:,i]*oa[:,j]).sum(-1)
                        tm = oa[:,i].sum(-1)+oa[:,j].sum(-1)
                        ds += (ov/(tm+0.1)).mean(); dc += 1
                if dc > 0:
                    l_dice = ds / dc
                    total = total + ramp * LAMBDA_DICE * l_dice

        if out.get("query_bg_mask") is not None:
            m = out["query_bg_mask"]; p = m.squeeze(-1).clamp(1e-6, 1-1e-6)
            ent = -(p*p.log()+(1-p)*(1-p).log()).mean()
            g = out.get("query_adaptive_gamma")
            if g is not None:
                cov = F.relu(g - m.mean(dim=1)).mean()
                gp = ((g - 0.7)**2).mean()
                l_bg = -ent + cov + gp
            else:
                l_bg = -ent + (m.mean()-0.7)**2
            total = total + LAMBDA_BG * l_bg

        return {"total_loss": total, "accuracy": acc,
                "loss_csm": l_csm.item() if torch.is_tensor(l_csm) else l_csm,
                "loss_dice": l_dice.item() if torch.is_tensor(l_dice) else l_dice,
                "loss_bg": l_bg.item() if torch.is_tensor(l_bg) else l_bg}


# =============================================================================
# DATASET
# =============================================================================
class SUES200Dataset(Dataset):
    def __init__(self, root, split="train", altitude="150", img_size=512,
                 train_locs=None, test_locs=None):
        super().__init__()
        drone_dir = os.path.join(root, "drone-view")
        satellite_dir = os.path.join(root, "satellite-view")
        locs = (train_locs or TRAIN_LOCS) if split == "train" else (test_locs or TEST_LOCS)
        if split == "train":
            self.drone_tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
            self.sat_tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        else:
            tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
            self.drone_tf = self.sat_tf = tf
        self.pairs = []
        for loc_id in locs:
            loc_str = f"{loc_id:04d}"
            sat_path = os.path.join(satellite_dir, loc_str, "0.png")
            if not os.path.exists(sat_path): continue
            alt_dir = os.path.join(drone_dir, loc_str, altitude)
            if not os.path.isdir(alt_dir): continue
            for img in sorted(os.listdir(alt_dir)):
                if img.endswith(('.jpg','.jpeg','.png')):
                    self.pairs.append((os.path.join(alt_dir, img), sat_path))
        print(f"  [SUES-200 {split} alt={altitude}] {len(self.pairs)} pairs ({len(locs)} locs)")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        dp, sp = self.pairs[idx]
        try:
            drone = Image.open(dp).convert("RGB"); sat = Image.open(sp).convert("RGB")
        except Exception:
            drone = Image.new("RGB",(512,512),(128,128,128)); sat = Image.new("RGB",(512,512),(128,128,128))
        return {"query": self.drone_tf(drone), "gallery": self.sat_tf(sat), "idx": idx}


class SUES200Gallery(Dataset):
    def __init__(self, root, test_locs=None, img_size=512):
        sat_dir = os.path.join(root, "satellite-view")
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.images = []; self.loc_ids = []
        for loc_id in (test_locs or TEST_LOCS):
            p = os.path.join(sat_dir, f"{loc_id:04d}", "0.png")
            if os.path.exists(p):
                self.images.append(p); self.loc_ids.append(loc_id)

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        return {"image": self.tf(Image.open(self.images[idx]).convert("RGB")),
                "loc_id": self.loc_ids[idx]}


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model, data_root, altitude, device):
    model.eval()
    q_ds = SUES200Dataset(data_root, "test", altitude, IMG_SIZE)
    q_loader = DataLoader(q_ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    g_ds = SUES200Gallery(data_root, TEST_LOCS, IMG_SIZE)
    g_loader = DataLoader(g_ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    gal_embs, gal_locs = [], []
    for b in g_loader:
        gal_embs.append(model.extract_embedding(b["image"].to(device)).cpu())
        gal_locs.extend(b["loc_id"].tolist())
    gal_embs = torch.cat(gal_embs, 0)
    gal_locs = np.array(gal_locs)
    loc_to_gidx = {l: i for i, l in enumerate(gal_locs)}

    q_embs = []
    for b in q_loader:
        q_embs.append(model.extract_embedding(b["query"].to(device)).cpu())
    q_embs = torch.cat(q_embs, 0)

    q_gt = []
    for dp, sp in q_ds.pairs:
        loc_id = int(os.path.basename(os.path.dirname(sp)))
        q_gt.append(loc_to_gidx.get(loc_id, -1))
    q_gt = np.array(q_gt)

    sim = q_embs.numpy() @ gal_embs.numpy().T
    ranks = np.argsort(-sim, axis=1)
    N = len(q_embs)
    results = {}
    for k in [1, 5, 10]:
        results[f"R@{k}"] = sum(1 for i in range(N) if q_gt[i] in ranks[i, :k]) / N
    ap_sum = sum(1.0/(np.where(ranks[i]==q_gt[i])[0][0]+1)
                 for i in range(N) if len(np.where(ranks[i]==q_gt[i])[0]) > 0)
    results["AP"] = ap_sum / N
    return results


# =============================================================================
# LR SCHEDULER
# =============================================================================
def cosine_lr(ep, total, base, warmup=3):
    if ep < warmup: return base * (ep+1) / warmup
    p = (ep - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))

def get_lrs(epoch):
    if epoch < STAGE1_END:
        return 0.0, cosine_lr(epoch, STAGE1_END, LR_HEAD_S1, WARMUP_EPOCHS)
    elif epoch < STAGE2_END:
        se = epoch-STAGE1_END; sl = STAGE2_END-STAGE1_END
        return cosine_lr(se, sl, LR_BACKBONE_S2, 2), cosine_lr(se, sl, LR_HEAD_S2, 0)
    else:
        se = epoch-STAGE2_END; sl = EPOCHS-STAGE2_END
        return cosine_lr(se, sl, LR_BACKBONE_S3, 0), cosine_lr(se, sl, LR_HEAD_S3, 0)


# =============================================================================
# TRAINING
# =============================================================================
def train(model, train_loader, val_fn, device):
    criterion = ThreeStageLoss().to(device)
    bb_params = list(model.backbone.parameters())
    hd_params = [p for n,p in model.named_parameters() if not n.startswith("backbone")]
    hd_params += list(criterion.parameters())
    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": 0.0},
        {"params": hd_params, "lr": LR_HEAD_S1},
    ], weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=AMP_ENABLED and device.type == "cuda")
    gs = 0; best_r1 = 0.0; history = []

    for epoch in range(EPOCHS):
        lr_bb, lr_hd = get_lrs(epoch)
        optimizer.param_groups[0]["lr"] = lr_bb
        optimizer.param_groups[1]["lr"] = lr_hd
        if epoch < STAGE1_END:
            for p in bb_params: p.requires_grad = False
            stage = "S1:frozen"
        else:
            for p in bb_params: p.requires_grad = True
            stage = "S2:+CSM" if epoch < STAGE2_END else "S3:full"

        model.train()
        el = ea = n = 0; t0 = time.time()
        pbar = tqdm(train_loader, desc=f"  Ep {epoch+1}/{EPOCHS} ({stage})", leave=False)
        for batch in pbar:
            q = batch["query"].to(device); g = batch["gallery"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=AMP_ENABLED and device.type == "cuda"):
                out = model(q, g, gs)
                ld = criterion(out, epoch)
                loss = ld["total_loss"]
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True); gs += 1; continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            el += loss.item(); ea += ld["accuracy"].item(); n += 1; gs += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{ld['accuracy'].item():.1%}")

        elapsed = time.time() - t0; el /= max(n,1); ea /= max(n,1)
        entry = {"epoch": epoch+1, "stage": stage, "loss": round(el,4),
                 "acc": round(ea,4), "time": round(elapsed,1)}

        if (epoch+1) % EVAL_FREQ == 0 or epoch == EPOCHS-1:
            m = val_fn()
            entry.update(m); r1 = m["R@1"]
            if r1 > best_r1:
                best_r1 = r1
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "geoslot_sues200_mamba_best.pth"))
            print(f"  Ep {epoch+1} ({stage}) | Loss={el:.4f} | Acc={ea:.1%} | "
                  f"R@1={r1:.2%} | R@5={m.get('R@5',0):.2%} | AP={m.get('AP',0):.2%} | "
                  f"LR={lr_bb:.1e}/{lr_hd:.1e} | {elapsed:.0f}s")
        else:
            print(f"  Ep {epoch+1} ({stage}) | Loss={el:.4f} | Acc={ea:.1%} | "
                  f"LR={lr_bb:.1e}/{lr_hd:.1e} | {elapsed:.0f}s")
        history.append(entry)

    return best_r1, history


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*70)
    print("  GeoSlot 2.0 — SUES-200 (MambaVision-L backbone)")
    print(f"  3-Stage: S1[0-{STAGE1_END}) S2[{STAGE1_END}-{STAGE2_END}) S3[{STAGE2_END}-{EPOCHS})")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Mamba SSM: {'YES' if MAMBA_AVAILABLE else 'FALLBACK (LinearAttn)'}")
    print(f"  Device: {DEVICE}")
    print("="*70)

    print("\n[DATASET] Loading SUES-200...")
    all_pairs = []
    for alt in ALTITUDES:
        ds = SUES200Dataset(DATA_ROOT, "train", alt, IMG_SIZE)
        all_pairs.extend(ds.pairs)
    print(f"  Total train pairs (all altitudes): {len(all_pairs)}")

    class CombinedDS(Dataset):
        def __init__(self, pairs, img_size=512):
            self.pairs = pairs
            self.dtf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            self.stf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        def __len__(self): return len(self.pairs)
        def __getitem__(self, idx):
            dp, sp = self.pairs[idx]
            try: d = Image.open(dp).convert("RGB"); s = Image.open(sp).convert("RGB")
            except: d = Image.new("RGB",(512,512),(128,128,128)); s = Image.new("RGB",(512,512),(128,128,128))
            return {"query": self.dtf(d), "gallery": self.stf(s), "idx": idx}

    train_ds = CombinedDS(all_pairs, IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    print("\n[BACKBONE] Loading MambaVision-L...")
    backbone = MambaVisionBackbone(BACKBONE_NAME)
    model = GeoSlotV2(backbone).to(DEVICE)
    tp = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {tp:,}")

    def val_fn():
        return evaluate(model, DATA_ROOT, TEST_ALTITUDE, DEVICE)

    print("\n[TRAINING] Starting...")
    best_r1, history = train(model, train_loader, val_fn, DEVICE)

    print("\n" + "="*70)
    print("  FINAL RESULTS — All Altitudes")
    print("="*70)
    for alt in ALTITUDES:
        m = evaluate(model, DATA_ROOT, alt, DEVICE)
        print(f"  Alt={alt}m | R@1={m['R@1']:.2%} | R@5={m['R@5']:.2%} | "
              f"R@10={m['R@10']:.2%} | AP={m['AP']:.2%}")
    print(f"\n  Best R@1 (alt={TEST_ALTITUDE}m): {best_r1:.2%}")

    res = {"method": "GeoSlot_v2_MambaVision", "backbone": BACKBONE_NAME,
           "dataset": "SUES-200", "best_r1": best_r1, "history": history,
           "settings": {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                        "img_size": IMG_SIZE, "mamba_ssm": MAMBA_AVAILABLE}}
    p = os.path.join(OUTPUT_DIR, "geoslot_sues200_mamba_results.json")
    with open(p, "w") as f: json.dump(res, f, indent=2, default=str)
    print(f"  Results saved: {p}")
    print("="*70)


if __name__ == "__main__":
    main()
