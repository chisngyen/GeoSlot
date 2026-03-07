# =============================================================================
# PHASE 1: GeoSlot — Train on CVUSA (Showcase Benchmark)
# Backbone: NVIDIA MambaVision-L (pretrained ImageNet-1K)
# Target: Recall@1 ≥ 99%  |  Hardware: Kaggle H100  |  Self-contained
# =============================================================================

# === SETUP (Auto-install dependencies) ===
import subprocess, sys, re

def run(cmd, verbose=False):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if verbose or result.returncode != 0:
        if result.stdout: print(result.stdout[-500:])
        if result.stderr: print(f"[WARN] {result.stderr[-500:]}")
    return result.returncode == 0

def pip(pkg, extra=""):
    return run(f"pip install -q {extra} {pkg}")

# ----- Step 1: Detect CUDA version từ hardware -----
print("[1/6] Detecting CUDA version...")
r = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
cuda_match = re.search(r"CUDA Version:\s*([\d.]+)", r.stdout)
hw_cuda  = cuda_match.group(1) if cuda_match else "12.6"
major    = int(hw_cuda.split(".")[0])
minor    = int(hw_cuda.split(".")[1]) if "." in hw_cuda else 0
print(f"  Hardware CUDA: {hw_cuda}")

# PyTorch chỉ có wheel đến cu126 (tính đến 2025).
# Nếu hardware CUDA >= 12.6 thì dùng cu126 — tương thích ngược.
# Nếu hardware CUDA 12.x < 12.6 thì dùng đúng version.
if major >= 13 or (major == 12 and minor >= 6):
    cu_tag = "cu126"
elif major == 12 and minor >= 4:
    cu_tag = "cu124"
elif major == 12 and minor >= 1:
    cu_tag = "cu121"
else:
    cu_tag = "cu118"
print(f"  → Using PyTorch index: {cu_tag}  (CUDA is backward-compatible)")

# ----- Step 2: Sync PyTorch với CUDA -----
# Root cause của lỗi "undefined symbol": PyTorch build cu124 chạy trên CUDA 12.6+
# → selective_scan_cuda.so link sai symbol → import mamba_ssm crash
print("[2/6] Syncing PyTorch to match hardware CUDA...")
ok = run(
    f"pip install -q -U torch torchvision torchaudio "
    f"--index-url https://download.pytorch.org/whl/{cu_tag}"
)
if ok:
    print(f"  ✓ PyTorch synced to {cu_tag}")
else:
    print(f"  [WARN] PyTorch sync failed — proceeding anyway")

print("  ⚠  Kaggle/Colab: nếu đây là lần đầu chạy, hãy restart kernel rồi chạy lại!")

# ----- Step 3: Base packages -----
print("[3/6] Installing base packages...")
pip("transformers==4.44.2")  # MUST pin — newer versions break MambaVision (all_tied_weights_keys)
for p in ["timm", "tqdm"]:
    try:
        __import__(p)
    except ImportError:
        pip(p)

# ----- Step 4: Build tools + fix nvcc PATH -----
print("[4/6] Installing build tools & fixing nvcc PATH...")
pip("packaging ninja wheel setuptools", "--upgrade")

# Kaggle: nvcc thường ở /usr/local/cuda/bin nhưng không có trong PATH khi pip build
import os
cuda_bin_paths = [
    "/usr/local/cuda/bin",
    "/usr/local/cuda-12/bin",
    "/usr/local/cuda-12.6/bin",
    "/usr/local/cuda-13/bin",
    "/usr/local/cuda-13.0/bin",
]
existing = [p for p in cuda_bin_paths if os.path.isdir(p)]
if existing:
    os.environ["PATH"] = existing[0] + ":" + os.environ.get("PATH", "")
    os.environ["CUDA_HOME"] = os.path.dirname(existing[0])
    print(f"  ✓ nvcc PATH set: {existing[0]}")
    run(f"nvcc --version", verbose=True)
else:
    print("  [WARN] nvcc not found in common paths — build may fail")
    run("find /usr -name 'nvcc' 2>/dev/null | head -5", verbose=True)

# ----- Step 5: causal-conv1d + mamba_ssm từ source -----
# Phải build từ source vì prebuilt wheel chỉ có đến cu124/cu126
# CUDA >= 12.x đều tương thích — chỉ cần compiler nvcc có sẵn trên Kaggle
print("[5/6] Building causal-conv1d from source...")
ok = pip("causal-conv1d>=1.4.0", "--no-build-isolation")
if not ok:
    # Thử version mới nhất không specify
    ok = pip("causal-conv1d", "--no-build-isolation --no-cache-dir")
if not ok:
    print("  [WARN] causal-conv1d build failed")

print("[6/6] Building mamba_ssm from source (5-10 phút)...")
ok = pip("mamba_ssm", "--no-build-isolation --no-cache-dir")
if not ok:
    # Thử cài thẳng từ GitHub (luôn là bản mới nhất)
    print("  [INFO] pip failed — trying direct GitHub install...")
    ok = run(
        "pip install -q --no-build-isolation "
        "git+https://github.com/state-spaces/mamba.git"
    )
if not ok:
    print("  [WARN] mamba_ssm build failed — will use LinearAttention fallback")
    print("  NOTE: MambaVision backbone cũng sẽ không load được!")
    print("  → Cân nhắc chạy lại sau khi restart kernel (bước quan trọng nhất)")

# ----- Verify -----
print("\n[VERIFY] Checking mamba_ssm...")
try:
    from mamba_ssm import Mamba
    print("  ✓ mamba_ssm loaded successfully!")
    MAMBA_AVAILABLE = True
except ImportError as e:
    print(f"  ✗ mamba_ssm not available: {e}")
    print("  → LinearAttention fallback active")
    MAMBA_AVAILABLE = False

# === IMPORTS ===
import os, math, glob, json, time, gc
from typing import Optional, List, Dict
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

# === CONFIG ===
# ===================== ĐỔI PATH NÀY THEO KAGGLE CỦA BẠN =====================
CVUSA_ROOT = "/kaggle/input/datasets/chinguyeen/cvusa-subdataset/CVUSA"
OUTPUT_DIR = "/kaggle/working"
# ==============================================================================

# --- Model ---
BACKBONE_NAME  = "nvidia/MambaVision-L-1K"
FEATURE_DIM    = 640
SLOT_DIM       = 256
MAX_SLOTS      = 12
N_REGISTER     = 4
EMBED_DIM_OUT  = 512
N_HEADS        = 4
SA_ITERS       = 3
GM_LAYERS      = 2
SINKHORN_ITERS = 15
MESH_ITERS     = 3

# --- Data ---
SAT_SIZE       = 224
PANO_SIZE      = (512, 128)

# --- Training ---
BATCH_SIZE     = 32
NUM_WORKERS    = 4
EPOCHS         = 50
LR_BACKBONE    = 1e-5
LR_HEAD        = 1e-4
WEIGHT_DECAY   = 0.01
WARMUP_EPOCHS  = 3
AMP_ENABLED    = True
FREEZE_BACKBONE_EPOCHS = 5
EVAL_FREQ      = 5
SAVE_FREQ      = 10
STAGE2_EPOCH   = 15
STAGE3_EPOCH   = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("  PHASE 1: GeoSlot — CVUSA Training")
print("  Backbone: NVIDIA MambaVision-L (pretrained ImageNet-1K)")
print("=" * 70)
print(f"  Device:        {DEVICE}")
print(f"  Backbone:      {BACKBONE_NAME}")
print(f"  mamba_ssm:     {'✓ available' if MAMBA_AVAILABLE else '✗ fallback mode'}")
print(f"  Feature dim:   {FEATURE_DIM}")
print(f"  Slots:         {MAX_SLOTS} object + {N_REGISTER} register")
print(f"  Embedding:     {EMBED_DIM_OUT}")
print(f"  Batch size:    {BATCH_SIZE}")
print(f"  Epochs:        {EPOCHS}")
print(f"  LR backbone:   {LR_BACKBONE}")
print(f"  LR head:       {LR_HEAD}")
print(f"  Freeze BB:     {FREEZE_BACKBONE_EPOCHS} epochs")
print(f"  Stage 2 @:     epoch {STAGE2_EPOCH}")
print(f"  Stage 3 @:     epoch {STAGE3_EPOCH}")
print("=" * 70)


# #############################################################################
# PART 1: BACKBONE — MambaVision-L (Pretrained)
# #############################################################################

print("\n[INIT] Loading MambaVision-L pretrained backbone...")

class MambaVisionBackbone(nn.Module):
    """
    NVIDIA MambaVision-L as feature extractor.
    Input:  [B, 3, H, W]
    Output: [B, N, 640]  (dense token features from stage4)
    """
    def __init__(self, model_name="nvidia/MambaVision-L-1K", frozen=False):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "\n\n  MambaVision backbone bắt buộc cần mamba_ssm.\n"
                "  Hãy restart kernel rồi chạy lại script!\n"
                "  Hoặc chạy thủ công: !pip install mamba_ssm --no-build-isolation\n"
            )
        # Fix MambaVision bug: torch.linspace creates meta tensor causing .item() to crash
        old_linspace = torch.linspace
        def patched_linspace(*args, **kwargs):
            kwargs["device"] = "cpu"
            return old_linspace(*args, **kwargs)
        torch.linspace = patched_linspace
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
            )
        finally:
            torch.linspace = old_linspace
        if hasattr(self.model, 'head'):
            self.model.head = nn.Identity()
        self.feature_dim = FEATURE_DIM

        if frozen:
            self.freeze()

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False
        print("  [BACKBONE] Frozen (no gradient)")

    def unfreeze(self):
        for p in self.model.parameters():
            p.requires_grad = True
        print("  [BACKBONE] Unfrozen (fine-tuning)")

    def forward(self, x):
        _, features = self.model(x)
        feat = features[-1]          # [B, 640, H', W']
        B, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, H * W, C)


# #############################################################################
# PART 2: NOVEL MODULES
# #############################################################################

# --- SSM Block: real mamba_ssm if available, else LinearAttention fallback ---
if MAMBA_AVAILABLE:
    from mamba_ssm import Mamba

    class SSMBlock(nn.Module):
        """True Mamba SSM block."""
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.mamba = Mamba(
                d_model=d_model, d_state=d_state,
                d_conv=d_conv, expand=expand,
            )
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            return x + self.mamba(self.norm(x))

else:
    class SSMBlock(nn.Module):
        """LinearAttention fallback (no mamba_ssm)."""
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            d_inner = int(d_model * expand)
            self.proj_in  = nn.Linear(d_model, d_inner * 2)
            self.proj_out = nn.Linear(d_inner, d_model)
            self.norm     = nn.LayerNorm(d_model)
            self.act      = nn.SiLU()

        def forward(self, x):
            z, gate = self.proj_in(self.norm(x)).chunk(2, dim=-1)
            return x + self.proj_out(self.act(z) * torch.sigmoid(gate))


# --- Background Suppression Mask ---
class BackgroundMask(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(True),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, features):
        mask = self.net(features)      # [B, N, 1]
        return features * mask, mask


# --- Slot Attention ---
class SlotAttentionCore(nn.Module):
    def __init__(self, dim, feature_dim, n_heads=4, iters=3, eps=1e-8):
        super().__init__()
        self.dim    = dim
        self.n_heads = n_heads
        self.iters  = iters
        self.eps    = eps
        self.dh     = dim // n_heads
        self.scale  = self.dh ** -0.5

        self.to_q   = nn.Linear(dim, dim, bias=False)
        self.to_k   = nn.Linear(feature_dim, dim, bias=False)
        self.to_v   = nn.Linear(feature_dim, dim, bias=False)
        self.gru    = nn.GRUCell(dim, dim)
        self.norm_in = nn.LayerNorm(feature_dim)
        self.norm_s  = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def step(self, slots, k, v, masks=None):
        B, K, _ = slots.shape
        slots_prev = slots
        q = self.to_q(self.norm_s(slots)).view(B, K, self.n_heads, self.dh)

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            dots.masked_fill_(masks.bool().view(B, K, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1).view(B, K, self.n_heads, -1)
        attn_out = attn.mean(dim=2)
        attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn)
        slots = self.gru(
            updates.reshape(-1, self.dim),
            slots_prev.reshape(-1, self.dim)
        ).reshape(B, -1, self.dim)
        return slots + self.ff(slots), attn_out

    def forward(self, inputs, slots):
        inputs = self.norm_in(inputs)
        B, N, _ = inputs.shape
        k = self.to_k(inputs).view(B, N, self.n_heads, self.dh)
        v = self.to_v(inputs).view(B, N, self.n_heads, self.dh)

        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v)
        return slots, attn


# --- Gumbel Slot Selector ---
class GumbelSelector(nn.Module):
    def __init__(self, dim, low_bound=1):
        super().__init__()
        self.low_bound = low_bound
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(True),
            nn.Linear(dim // 2, 2),
        )

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


# --- Adaptive Slot Attention ---
class AdaptiveSlotAttention(nn.Module):
    def __init__(self, feature_dim, slot_dim, max_slots, n_register,
                 n_heads=4, iters=3, low_bound=1):
        super().__init__()
        self.max_slots  = max_slots
        self.n_register = n_register
        total = max_slots + n_register

        self.bg_mask      = BackgroundMask(feature_dim)
        self.input_proj   = nn.Linear(feature_dim, slot_dim)
        self.slot_mu      = nn.Parameter(torch.randn(1, total, slot_dim) * (slot_dim ** -0.5))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, total, slot_dim))
        self.slot_attn    = SlotAttentionCore(slot_dim, slot_dim, n_heads, iters)
        self.gumbel       = GumbelSelector(slot_dim, low_bound)

    def forward(self, features, global_step=None):
        B = features.shape[0]

        features_masked, bg_mask = self.bg_mask(features)
        features_proj = self.input_proj(features_masked)

        mu    = self.slot_mu.expand(B, -1, -1)
        sigma = self.slot_log_sigma.exp().expand(B, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)

        slots, attn_maps = self.slot_attn(features_proj, slots)

        object_slots   = slots[:, :self.max_slots]
        register_slots = slots[:, self.max_slots:]

        keep_decision, keep_probs = self.gumbel(object_slots, global_step)
        object_slots = object_slots * keep_decision.unsqueeze(-1)

        return {
            "object_slots":   object_slots,
            "register_slots": register_slots,
            "bg_mask":        bg_mask,
            "attn_maps":      attn_maps,
            "keep_decision":  keep_decision,
            "keep_probs":     keep_probs,
        }


# --- Graph Mamba Layer (uses SSMBlock — real or fallback) ---
class GraphMambaLayer(nn.Module):
    def __init__(self, dim, num_layers=2):
        super().__init__()
        self.fwd   = nn.ModuleList([SSMBlock(dim) for _ in range(num_layers)])
        self.bwd   = nn.ModuleList([SSMBlock(dim) for _ in range(num_layers)])
        self.merge = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.ffns  = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4),
                          nn.GELU(), nn.Linear(dim * 4, dim))
            for _ in range(num_layers)
        ])

    def forward(self, slots, keep_mask=None):
        B, K, D = slots.shape
        for i in range(len(self.fwd)):
            residual = slots
            order   = slots.norm(dim=-1).argsort(dim=-1, descending=True)
            bi      = torch.arange(B, device=slots.device).unsqueeze(1).expand(-1, K)
            ordered = slots[bi, order]

            f = self.fwd[i](ordered)
            b = self.bwd[i](ordered.flip(1)).flip(1)
            merged = self.merge[i](torch.cat([f, b], dim=-1))

            reverse = order.argsort(dim=-1)
            slots   = self.norms[i](residual + merged[bi, reverse])
            slots   = slots + self.ffns[i](slots)

            if keep_mask is not None:
                slots = slots * keep_mask.unsqueeze(-1)
        return slots


# --- Sinkhorn OT ---
class SinkhornOT(nn.Module):
    def __init__(self, dim, num_iters=15, epsilon=0.05, mesh_iters=3):
        super().__init__()
        self.num_iters  = num_iters
        self.epsilon    = epsilon
        self.mesh_iters = mesh_iters
        self.cost_proj  = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, dim)
        )

    def forward(self, slots_q, slots_r, mask_q=None, mask_r=None):
        C = torch.cdist(self.cost_proj(slots_q), self.cost_proj(slots_r), p=2.0)
        B, K, M = C.shape

        log_K = -C / self.epsilon
        if mask_q is not None:
            log_K = log_K + torch.log(mask_q.unsqueeze(-1).clamp(min=1e-8))
        if mask_r is not None:
            log_K = log_K + torch.log(mask_r.unsqueeze(-2).clamp(min=1e-8))

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
        return {
            "similarity":      torch.sigmoid(-cost),
            "transport_plan":  T,
            "cost_matrix":     C,
            "transport_cost":  cost,
        }


# #############################################################################
# PART 3: FULL PIPELINE — GeoSlot
# #############################################################################

class GeoSlot(nn.Module):
    def __init__(self, backbone_name, feature_dim, slot_dim, max_slots,
                 n_register, embed_dim_out, frozen_backbone=False):
        super().__init__()
        self.backbone = MambaVisionBackbone(backbone_name, frozen=frozen_backbone)
        self.slot_attention = AdaptiveSlotAttention(
            feature_dim=feature_dim, slot_dim=slot_dim,
            max_slots=max_slots, n_register=n_register,
            n_heads=N_HEADS, iters=SA_ITERS,
        )
        self.graph_mamba = GraphMambaLayer(dim=slot_dim, num_layers=GM_LAYERS)
        self.ot_matcher  = SinkhornOT(
            dim=slot_dim, num_iters=SINKHORN_ITERS, mesh_iters=MESH_ITERS,
        )
        self.embed_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, embed_dim_out),
        )

    def encode_view(self, x, global_step=None):
        features  = self.backbone(x)
        sa_out    = self.slot_attention(features, global_step)
        slots     = sa_out["object_slots"]
        keep_mask = sa_out["keep_decision"]

        slots = self.graph_mamba(slots, keep_mask)

        weights      = keep_mask / (keep_mask.sum(dim=-1, keepdim=True) + 1e-8)
        global_slot  = (slots * weights.unsqueeze(-1)).sum(dim=1)
        embedding    = F.normalize(self.embed_head(global_slot), dim=-1)

        return {
            "slots": slots, "embedding": embedding,
            "keep_mask": keep_mask, "bg_mask": sa_out["bg_mask"],
            "attn_maps": sa_out["attn_maps"], "keep_probs": sa_out["keep_probs"],
            "register_slots": sa_out["register_slots"],
        }

    def forward(self, query_img, ref_img, global_step=None):
        q  = self.encode_view(query_img, global_step)
        r  = self.encode_view(ref_img,   global_step)
        ot = self.ot_matcher(q["slots"], r["slots"], q["keep_mask"], r["keep_mask"])

        return {
            "query_embedding":  q["embedding"],
            "ref_embedding":    r["embedding"],
            "query_slots":      q["slots"],
            "ref_slots":        r["slots"],
            "query_keep_mask":  q["keep_mask"],
            "ref_keep_mask":    r["keep_mask"],
            "similarity":       ot["similarity"],
            "transport_plan":   ot["transport_plan"],
            "transport_cost":   ot["transport_cost"],
            "query_bg_mask":    q["bg_mask"],
            "ref_bg_mask":      r["bg_mask"],
            "query_attn_maps":  q["attn_maps"],
            "ref_attn_maps":    r["attn_maps"],
            "query_keep_probs": q["keep_probs"],
            "ref_keep_probs":   r["keep_probs"],
        }

    def extract_embedding(self, x, global_step=None):
        return self.encode_view(x, global_step)["embedding"]


# #############################################################################
# PART 4: LOSSES
# #############################################################################

class SymmetricInfoNCE(nn.Module):
    def __init__(self, init_temp=0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(init_temp).log())

    @property
    def temp(self):
        return self.log_temp.exp().clamp(0.01, 1.0)

    def forward(self, q, r):
        B = q.shape[0]
        logits = q @ r.t() / self.temp
        labels = torch.arange(B, device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        acc  = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc


class DWBL(nn.Module):
    def __init__(self, temperature=0.1, margin=0.3):
        super().__init__()
        self.t = temperature
        self.m = margin

    def forward(self, q, r):
        B   = q.shape[0]
        sim = q @ r.t()
        pos = sim.diag()
        mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
        neg  = sim[mask].view(B, B - 1)
        weights = F.softmax(neg / self.t, dim=-1)
        wneg    = (weights * torch.exp((neg - self.m) / self.t)).sum(dim=-1)
        pos_exp = torch.exp(pos / self.t)
        return (-torch.log(pos_exp / (pos_exp + wneg + 1e-8))).mean()


class ContrastiveSlotLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature

    def forward(self, out):
        qs = F.normalize(out["query_slots"], dim=-1)
        rs = F.normalize(out["ref_slots"],   dim=-1)
        T  = out["transport_plan"]
        Tn = T / (T.sum(dim=-1, keepdim=True) + 1e-8)
        sim  = torch.bmm(qs, rs.transpose(1, 2)) / self.t
        loss = -(Tn * F.log_softmax(sim, dim=-1)).sum(dim=-1)
        km = out.get("query_keep_mask")
        if km is not None:
            return (loss * km).sum() / (km.sum() + 1e-8)
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.s = smooth

    def forward(self, attn_maps, keep_mask=None):
        B, K, N = attn_maps.shape
        an = attn_maps / (attn_maps.sum(dim=-1, keepdim=True) + 1e-8)
        loss, count = 0.0, 0
        for i in range(K):
            for j in range(i + 1, K):
                inter = (an[:, i] * an[:, j]).sum(dim=-1)
                union = an[:, i].sum(dim=-1) + an[:, j].sum(dim=-1)
                loss += ((2 * inter + self.s) / (union + self.s)).mean()
                count += 1
        return loss / max(count, 1)


class JointLoss(nn.Module):
    def __init__(self, lam_i=1.0, lam_d=1.0, lam_cs=0.5, lam_di=0.3,
                 stage2=15, stage3=30):
        super().__init__()
        self.lam_i  = lam_i
        self.lam_d  = lam_d
        self.lam_cs = lam_cs
        self.lam_di = lam_di
        self.s2 = stage2
        self.s3 = stage3
        self.infonce = SymmetricInfoNCE()
        self.dwbl    = DWBL()
        self.csm     = ContrastiveSlotLoss()
        self.dice    = DiceLoss()

    def forward(self, out, epoch=0):
        loss_i, acc = self.infonce(out["query_embedding"], out["ref_embedding"])
        loss_d = self.dwbl(out["query_embedding"], out["ref_embedding"])
        total  = self.lam_i * loss_i + self.lam_d * loss_d

        loss_cs = loss_di = torch.tensor(0.0, device=total.device)
        if epoch >= self.s2:
            loss_cs = self.csm(out)
            loss_di = self.dice(out["query_attn_maps"], out.get("query_keep_mask"))
            total   = total + self.lam_cs * loss_cs + self.lam_di * loss_di

        stage = 3 if epoch >= self.s3 else (2 if epoch >= self.s2 else 1)
        return {
            "total_loss":   total,
            "loss_infonce": loss_i.detach(),
            "loss_dwbl":    loss_d.detach(),
            "loss_csm":     loss_cs.detach() if loss_cs.requires_grad else loss_cs,
            "loss_dice":    loss_di.detach() if loss_di.requires_grad else loss_di,
            "accuracy":     acc,
            "active_stage": stage,
        }


# #############################################################################
# PART 5: CVUSA DATASET
# #############################################################################

class CVUSADataset(Dataset):
    def __init__(self, root, split="train", sat_size=224, pano_size=(512, 128)):
        super().__init__()
        self.split = split
        self.pairs = []

        self.sat_tf = transforms.Compose([
            transforms.Resize((sat_size, sat_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.sat_tf_aug = transforms.Compose([
            transforms.Resize((int(sat_size * 1.1), int(sat_size * 1.1))),
            transforms.RandomCrop((sat_size, sat_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.pano_tf = transforms.Compose([
            transforms.Resize(pano_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.pano_tf_aug = transforms.Compose([
            transforms.Resize(pano_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        subset   = os.path.join(root, "CVPR_subset")
        pano_dir = os.path.join(subset, "streetview", "panos")
        sat_dir  = os.path.join(subset, "bingmap", "18")

        sf = os.path.join(subset, "splits",
                          "train-19zl.csv" if split == "train" else "val-19zl.csv")
        if os.path.exists(sf):
            self._load_csv(sf, pano_dir, sat_dir)
        else:
            self._load_match(pano_dir, sat_dir, split)

        print(f"  [CVUSA {split}] {len(self.pairs)} pairs loaded")

    def _load_csv(self, csv_path, pano_dir, sat_dir):
        base = os.path.dirname(csv_path)
        with open(csv_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                pano_rel, sat_rel = parts[0].strip(), parts[1].strip()
                for pp in [os.path.join(pano_dir, pano_rel),
                           os.path.join(base, "..", pano_rel), pano_rel]:
                    if os.path.exists(pp): break
                for sp in [os.path.join(sat_dir, sat_rel),
                           os.path.join(base, "..", sat_rel), sat_rel]:
                    if os.path.exists(sp): break
                self.pairs.append((pp, sp))

    def _load_match(self, pano_dir, sat_dir, split):
        if not os.path.exists(pano_dir) or not os.path.exists(sat_dir):
            print(f"  [WARNING] Dirs not found: {pano_dir} / {sat_dir}")
            return
        panos    = sorted(glob.glob(os.path.join(pano_dir, "**", "*.jpg"), recursive=True))
        sat_dict = {}
        for s in sorted(glob.glob(os.path.join(sat_dir, "**", "*.jpg"), recursive=True)):
            sat_dict[os.path.splitext(os.path.basename(s))[0]] = s

        all_pairs = [(p, sat_dict[os.path.splitext(os.path.basename(p))[0]])
                     for p in panos
                     if os.path.splitext(os.path.basename(p))[0] in sat_dict]

        idx = int(len(all_pairs) * 0.8)
        self.pairs = all_pairs[:idx] if split == "train" else all_pairs[idx:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pp, sp = self.pairs[idx]
        try:
            pano = Image.open(pp).convert("RGB")
            sat  = Image.open(sp).convert("RGB")
        except Exception:
            pano = Image.new("RGB", (512, 128), (128, 128, 128))
            sat  = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.split == "train":
            pano = self.pano_tf_aug(pano)
            sat  = self.sat_tf_aug(sat)
        else:
            pano = self.pano_tf(pano)
            sat  = self.sat_tf(sat)

        return {"query": pano, "gallery": sat, "idx": idx}


# #############################################################################
# PART 6: EVALUATION
# #############################################################################

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    q_embs, r_embs = [], []

    for batch in tqdm(val_loader, desc="Evaluating", leave=False):
        qe = model.extract_embedding(batch["query"].to(device))
        re = model.extract_embedding(batch["gallery"].to(device))
        q_embs.append(qe.cpu())
        r_embs.append(re.cpu())

    q_embs = torch.cat(q_embs, 0).numpy()
    r_embs = torch.cat(r_embs, 0).numpy()

    sim   = q_embs @ r_embs.T
    ranks = np.argsort(-sim, axis=1)
    N     = len(q_embs)
    gt    = np.arange(N)

    results = {}
    for k in [1, 5, 10, 100]:
        correct = sum(1 for i in range(N) if gt[i] in ranks[i, :k])
        results[f"R@{k}"] = correct / N
    return results


# #############################################################################
# PART 7: TRAINING LOOP
# #############################################################################

def main():
    # === Dataset ===
    print("\n[1/5] Loading CVUSA dataset...")
    train_ds = CVUSADataset(CVUSA_ROOT, "train", SAT_SIZE, PANO_SIZE)
    val_ds   = CVUSADataset(CVUSA_ROOT, "test",  SAT_SIZE, PANO_SIZE)

    if len(train_ds) == 0:
        print("[ERROR] No training samples! Check CVUSA_ROOT path.")
        print(f"  CVUSA_ROOT = {CVUSA_ROOT}")
        print(f"  Exists: {os.path.exists(CVUSA_ROOT)}")
        if os.path.exists(CVUSA_ROOT):
            for d in os.listdir(CVUSA_ROOT):
                print(f"    {d}")
        return

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # === Model ===
    print("\n[2/5] Building model...")
    model = GeoSlot(
        backbone_name=BACKBONE_NAME,
        feature_dim=FEATURE_DIM,
        slot_dim=SLOT_DIM,
        max_slots=MAX_SLOTS,
        n_register=N_REGISTER,
        embed_dim_out=EMBED_DIM_OUT,
        frozen_backbone=(FREEZE_BACKBONE_EPOCHS > 0),
    ).to(DEVICE)

    bb_params   = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.parameters()) - bb_params
    print(f"  Backbone params:  {bb_params:,} (pretrained)")
    print(f"  Head params:      {head_params:,} (new)")
    print(f"  Total params:     {bb_params + head_params:,}")

    # === Loss ===
    criterion = JointLoss(stage2=STAGE2_EPOCH, stage3=STAGE3_EPOCH).to(DEVICE)

    # === Optimizer ===
    backbone_params  = list(model.backbone.parameters())
    head_params_list = [p for n, p in model.named_parameters()
                        if not n.startswith("backbone")]
    head_params_list += list(criterion.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params,  "lr": LR_BACKBONE},
        {"params": head_params_list, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        return 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP_EPOCHS) /
                                    max(1, EPOCHS - WARMUP_EPOCHS)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda, lr_lambda])
    scaler    = GradScaler(enabled=AMP_ENABLED and DEVICE.type == "cuda")

    log = {
        "config": {
            "backbone": BACKBONE_NAME, "feature_dim": FEATURE_DIM,
            "slot_dim": SLOT_DIM, "max_slots": MAX_SLOTS,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
            "bb_params": bb_params, "head_params": head_params,
            "mamba_ssm": MAMBA_AVAILABLE,
        },
        "history": [],
    }

    # === Train ===
    print(f"\n[3/5] Starting training ({EPOCHS} epochs)...\n")
    best_r1     = 0.0
    global_step = 0

    for epoch in range(EPOCHS):
        if epoch == FREEZE_BACKBONE_EPOCHS and FREEZE_BACKBONE_EPOCHS > 0:
            model.backbone.unfreeze()

        model.train()
        ep_loss = ep_acc = n_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in pbar:
            query   = batch["query"].to(DEVICE)
            gallery = batch["gallery"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=AMP_ENABLED and DEVICE.type == "cuda"):
                out       = model(query, gallery, global_step)
                loss_dict = criterion(out, epoch=epoch)
                loss      = loss_dict["total_loss"]

            if AMP_ENABLED and DEVICE.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            ep_loss  += loss.item()
            ep_acc   += loss_dict["accuracy"].item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{loss_dict['accuracy'].item():.1%}",
                stage=loss_dict["active_stage"],
            )

        scheduler.step()
        ep_loss  /= max(n_batches, 1)
        ep_acc   /= max(n_batches, 1)
        elapsed   = time.time() - t0

        entry = {
            "epoch":   epoch + 1,
            "loss":    round(ep_loss, 4),
            "acc":     round(ep_acc, 4),
            "lr_bb":   optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
            "time":    round(elapsed, 1),
            "stage":   loss_dict["active_stage"],
        }

        # === Evaluation ===
        if (epoch + 1) % EVAL_FREQ == 0 or epoch == EPOCHS - 1:
            print(f"\n  Evaluating @ epoch {epoch+1}...")
            metrics = evaluate(model, val_loader, DEVICE)
            entry.update(metrics)
            r1 = metrics["R@1"]
            print(f"  R@1={r1:.2%}  R@5={metrics['R@5']:.2%}  "
                  f"R@10={metrics['R@10']:.2%}  R@100={metrics['R@100']:.2%}")

            if r1 > best_r1:
                best_r1 = r1
                torch.save({
                    "epoch": epoch + 1, "r1": r1,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(OUTPUT_DIR, "best_model_cvusa.pth"))
                print(f"  ★ New best R@1: {r1:.2%}")

        log["history"].append(entry)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | "
              f"Stage={loss_dict['active_stage']} | {elapsed:.0f}s")

        if (epoch + 1) % SAVE_FREQ == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(OUTPUT_DIR, f"ckpt_epoch{epoch+1}.pth"))

    # === Final results ===
    log["best_r1"] = best_r1
    results_path   = os.path.join(OUTPUT_DIR, "results_cvusa.json")
    with open(results_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Best R@1       = {best_r1:.2%}")
    print(f"  mamba_ssm used = {MAMBA_AVAILABLE}")
    print(f"  Results        : {results_path}")
    print(f"  Best model     : {os.path.join(OUTPUT_DIR, 'best_model_cvusa.pth')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()