# =============================================================================
# PHASE 2: GeoSlot — Train on University-1652 (Main Benchmark #1)
# Target: Drone→Sat R@1 ≥ 97% | SOTA: 96.88% (OG-Sample4Geo, Jan 2025)
# Hardware: Kaggle H100 | Self-contained (không cần file ngoài)
# =============================================================================

# === SETUP ===
# Dùng cùng cách cài với Phase 1 — cài mamba_ssm trước, rồi mới import model
import subprocess, sys, re, os

def run(cmd, verbose=False):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if verbose or result.returncode != 0:
        if result.stdout: print(result.stdout[-500:])
        if result.stderr: print(f"[WARN] {result.stderr[-500:]}")
    return result.returncode == 0

def pip(pkg, extra=""):
    return run(f"pip install -q {extra} {pkg}")

print("[1/5] Installing base packages...")
pip("transformers==4.44.2")  # MUST pin — newer versions break MambaVision
for p in ["timm", "tqdm"]:
    try: __import__(p)
    except ImportError: pip(p)

print("[2/5] Installing build tools...")
pip("packaging ninja wheel setuptools", "--upgrade")

print("[3/5] Installing causal-conv1d...")
ok = pip("causal-conv1d>=1.4.0", "--no-build-isolation")
if not ok:
    pip("causal-conv1d", "--no-build-isolation --no-cache-dir")

print("[4/5] Installing mamba_ssm (5-10 phút, đang build CUDA)...")
ok = pip("mamba_ssm", "--no-build-isolation")
if not ok:
    ok = run("pip install -q --no-build-isolation git+https://github.com/state-spaces/mamba.git")
if not ok:
    print("  [WARN] mamba_ssm build failed — LinearAttention fallback active")

print("[5/5] Verifying...")
try:
    from mamba_ssm import Mamba
    print("  ✓ mamba_ssm ready!")
    MAMBA_AVAILABLE = True
except ImportError:
    print("  ✗ mamba_ssm not available — fallback mode")
    MAMBA_AVAILABLE = False

# === IMPORTS ===
import math, glob, json, time
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
UNI1652_ROOT = "/kaggle/input/datasets/chinguyeen/university-1652/University-1652"
OUTPUT_DIR   = "/kaggle/working"
RESUME_FROM  = None  # Ví dụ: "/kaggle/working/best_model_cvusa.pth"
# ==============================================================================

# --- Model (giữ nguyên với Phase 1) ---
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
IMG_SIZE       = 384   # University-1652 dùng ảnh lớn hơn CVUSA

# --- Training ---
BATCH_SIZE     = 32
NUM_WORKERS    = 4
EPOCHS         = 60
LR_BACKBONE    = 1e-5
LR_HEAD        = 1e-4
WEIGHT_DECAY   = 0.01
WARMUP_EPOCHS  = 3
FREEZE_BB      = 5
EVAL_FREQ      = 5
SAVE_FREQ      = 10
S2_EPOCH       = 20
S3_EPOCH       = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("  PHASE 2: GeoSlot — University-1652")
print("  Target: Drone→Sat R@1 ≥ 97% | SOTA: 96.88%")
print("=" * 70)
print(f"  Device:     {DEVICE}")
print(f"  mamba_ssm:  {'✓' if MAMBA_AVAILABLE else '✗ fallback'}")
print(f"  Image size: {IMG_SIZE}×{IMG_SIZE}")
print(f"  Batch:      {BATCH_SIZE} | Epochs: {EPOCHS}")
print(f"  Resume:     {RESUME_FROM or 'None (train from scratch)'}")
print("=" * 70)


# #############################################################################
# BACKBONE + MODEL (copy từ Phase 1 — giữ nguyên để weights tương thích)
# #############################################################################

print("\n[INIT] Loading MambaVision-L backbone...")

class MambaVisionBackbone(nn.Module):
    def __init__(self, model_name="nvidia/MambaVision-L-1K", frozen=False):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "\n\n  MambaVision backbone bắt buộc cần mamba_ssm.\n"
                "  Restart kernel rồi chạy lại!\n"
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
        if frozen: self.freeze()

    def freeze(self):
        for p in self.model.parameters(): p.requires_grad = False
        print("  [BACKBONE] Frozen")

    def unfreeze(self):
        for p in self.model.parameters(): p.requires_grad = True
        print("  [BACKBONE] Unfrozen (fine-tuning)")

    def forward(self, x):
        _, features = self.model(x)
        feat = features[-1]
        B, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, H * W, C)


if MAMBA_AVAILABLE:
    class SSMBlock(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.norm  = nn.LayerNorm(d_model)
        def forward(self, x):
            return x + self.mamba(self.norm(x))
else:
    class SSMBlock(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            d_inner = int(d_model * expand)
            self.proj_in  = nn.Linear(d_model, d_inner * 2)
            self.proj_out = nn.Linear(d_inner, d_model)
            self.norm = nn.LayerNorm(d_model)
            self.act  = nn.SiLU()
        def forward(self, x):
            z, gate = self.proj_in(self.norm(x)).chunk(2, dim=-1)
            return x + self.proj_out(self.act(z) * torch.sigmoid(gate))


class BackgroundMask(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(True), nn.Linear(hidden, 1), nn.Sigmoid())
    def forward(self, f):
        m = self.net(f); return f * m, m


class SlotAttentionCore(nn.Module):
    def __init__(self, dim, feature_dim, n_heads=4, iters=3, eps=1e-8):
        super().__init__()
        self.dim = dim; self.n_heads = n_heads; self.iters = iters; self.eps = eps
        self.dh = dim // n_heads; self.scale = self.dh ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(feature_dim, dim, bias=False)
        self.to_v = nn.Linear(feature_dim, dim, bias=False)
        self.gru  = nn.GRUCell(dim, dim)
        self.norm_in = nn.LayerNorm(feature_dim); self.norm_s = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def step(self, slots, k, v, masks=None):
        B, K, _ = slots.shape; sp = slots
        q = self.to_q(self.norm_s(slots)).view(B, K, self.n_heads, self.dh)
        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None: dots.masked_fill_(masks.bool().view(B,K,1,1), float("-inf"))
        attn = dots.flatten(1,2).softmax(dim=1).view(B, K, self.n_heads, -1)
        attn_out = attn.mean(dim=2)
        attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)
        updates = torch.einsum("bjhd,bihj->bihd", v, attn)
        slots = self.gru(updates.reshape(-1, self.dim), sp.reshape(-1, self.dim)).reshape(B, -1, self.dim)
        return slots + self.ff(slots), attn_out

    def forward(self, inputs, slots):
        inputs = self.norm_in(inputs); B, N, _ = inputs.shape
        k = self.to_k(inputs).view(B, N, self.n_heads, self.dh)
        v = self.to_v(inputs).view(B, N, self.n_heads, self.dh)
        for _ in range(self.iters): slots, attn = self.step(slots, k, v)
        return slots, attn


class GumbelSelector(nn.Module):
    def __init__(self, dim, low_bound=1):
        super().__init__()
        self.low_bound = low_bound
        self.net = nn.Sequential(nn.Linear(dim, dim//2), nn.ReLU(True), nn.Linear(dim//2, 2))

    def forward(self, slots, global_step=None):
        logits = self.net(slots)
        tau = max(0.1, 1.0 - (global_step or 0) / 100000)
        if self.training: decision = F.gumbel_softmax(logits, hard=True, tau=tau)[..., 1]
        else: decision = (logits.argmax(dim=-1) == 1).float()
        active = (decision != 0).sum(dim=-1)
        for j in (active < self.low_bound).nonzero(as_tuple=True)[0]:
            inactive = (decision[j] == 0).nonzero(as_tuple=True)[0]
            n = min(self.low_bound - int(active[j].item()), len(inactive))
            if n > 0:
                idx = inactive[torch.randperm(len(inactive), device=decision.device)[:n]]
                decision[j, idx] = 1.0
        return decision, F.softmax(logits, dim=-1)[..., 1]


class AdaptiveSlotAttention(nn.Module):
    def __init__(self, feature_dim, slot_dim, max_slots, n_register, n_heads=4, iters=3, low_bound=1):
        super().__init__()
        self.max_slots = max_slots; self.n_register = n_register
        total = max_slots + n_register
        self.bg_mask    = BackgroundMask(feature_dim)
        self.input_proj = nn.Linear(feature_dim, slot_dim)
        self.slot_mu    = nn.Parameter(torch.randn(1, total, slot_dim) * (slot_dim ** -0.5))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, total, slot_dim))
        self.slot_attn  = SlotAttentionCore(slot_dim, slot_dim, n_heads, iters)
        self.gumbel     = GumbelSelector(slot_dim, low_bound)

    def forward(self, features, global_step=None):
        B = features.shape[0]
        fm, bg_mask = self.bg_mask(features)
        fp = self.input_proj(fm)
        mu = self.slot_mu.expand(B,-1,-1); sigma = self.slot_log_sigma.exp().expand(B,-1,-1)
        slots = mu + sigma * torch.randn_like(mu)
        slots, attn_maps = self.slot_attn(fp, slots)
        obj = slots[:, :self.max_slots]; reg = slots[:, self.max_slots:]
        kd, kp = self.gumbel(obj, global_step)
        obj = obj * kd.unsqueeze(-1)
        return {"object_slots": obj, "register_slots": reg, "bg_mask": bg_mask,
                "attn_maps": attn_maps, "keep_decision": kd, "keep_probs": kp}


class GraphMambaLayer(nn.Module):
    def __init__(self, dim, num_layers=2):
        super().__init__()
        self.fwd   = nn.ModuleList([SSMBlock(dim) for _ in range(num_layers)])
        self.bwd   = nn.ModuleList([SSMBlock(dim) for _ in range(num_layers)])
        self.merge = nn.ModuleList([nn.Linear(dim*2, dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.ffns  = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
            for _ in range(num_layers)])

    def forward(self, slots, keep_mask=None):
        B, K, D = slots.shape
        for i in range(len(self.fwd)):
            res = slots
            order = slots.norm(dim=-1).argsort(dim=-1, descending=True)
            bi = torch.arange(B, device=slots.device).unsqueeze(1).expand(-1, K)
            ordered = slots[bi, order]
            f = self.fwd[i](ordered); b = self.bwd[i](ordered.flip(1)).flip(1)
            merged = self.merge[i](torch.cat([f, b], dim=-1))
            reverse = order.argsort(dim=-1)
            slots = self.norms[i](res + merged[bi, reverse])
            slots = slots + self.ffns[i](slots)
            if keep_mask is not None: slots = slots * keep_mask.unsqueeze(-1)
        return slots


class SinkhornOT(nn.Module):
    def __init__(self, dim, num_iters=15, epsilon=0.05, mesh_iters=3):
        super().__init__()
        self.num_iters = num_iters; self.epsilon = epsilon; self.mesh_iters = mesh_iters
        self.cost_proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, dim))

    def forward(self, sq, sr, mq=None, mr=None):
        C = torch.cdist(self.cost_proj(sq), self.cost_proj(sr), p=2.0)
        B, K, M = C.shape
        log_K = -C / self.epsilon
        if mq is not None: log_K = log_K + torch.log(mq.unsqueeze(-1).clamp(min=1e-8))
        if mr is not None: log_K = log_K + torch.log(mr.unsqueeze(-2).clamp(min=1e-8))
        la = torch.zeros(B, K, 1, device=C.device); lb = torch.zeros(B, 1, M, device=C.device)
        for _ in range(self.num_iters):
            la = -torch.logsumexp(log_K + lb, dim=2, keepdim=True)
            lb = -torch.logsumexp(log_K + la, dim=1, keepdim=True)
        T = torch.exp(log_K + la + lb)
        for _ in range(self.mesh_iters):
            T = T ** 2
            T = T / (T.sum(dim=-1, keepdim=True) + 1e-8)
            T = T / (T.sum(dim=-2, keepdim=True) + 1e-8)
        cost = (T * C).sum(dim=(-1,-2))
        return {"similarity": torch.sigmoid(-cost), "transport_plan": T,
                "cost_matrix": C, "transport_cost": cost}


class GeoSlot(nn.Module):
    def __init__(self, backbone_name=BACKBONE_NAME, feature_dim=FEATURE_DIM,
                 slot_dim=SLOT_DIM, max_slots=MAX_SLOTS, n_register=N_REGISTER,
                 embed_dim_out=EMBED_DIM_OUT, frozen_backbone=False):
        super().__init__()
        self.backbone       = MambaVisionBackbone(backbone_name, frozen=frozen_backbone)
        self.slot_attention = AdaptiveSlotAttention(feature_dim, slot_dim, max_slots,
                                                    n_register, N_HEADS, SA_ITERS)
        self.graph_mamba    = GraphMambaLayer(dim=slot_dim, num_layers=GM_LAYERS)
        self.ot_matcher     = SinkhornOT(slot_dim, SINKHORN_ITERS, mesh_iters=MESH_ITERS)
        self.embed_head     = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, embed_dim_out))

    def encode_view(self, x, gs=None):
        features  = self.backbone(x)
        sa        = self.slot_attention(features, gs)
        slots     = self.graph_mamba(sa["object_slots"], sa["keep_decision"])
        w         = sa["keep_decision"] / (sa["keep_decision"].sum(dim=-1, keepdim=True) + 1e-8)
        emb       = F.normalize(self.embed_head((slots * w.unsqueeze(-1)).sum(dim=1)), dim=-1)
        return {"slots": slots, "embedding": emb, "keep_mask": sa["keep_decision"],
                "bg_mask": sa["bg_mask"], "attn_maps": sa["attn_maps"],
                "keep_probs": sa["keep_probs"], "register_slots": sa["register_slots"]}

    def forward(self, q_img, r_img, gs=None):
        q  = self.encode_view(q_img, gs); r = self.encode_view(r_img, gs)
        ot = self.ot_matcher(q["slots"], r["slots"], q["keep_mask"], r["keep_mask"])
        return {"query_embedding": q["embedding"], "ref_embedding": r["embedding"],
                "query_slots": q["slots"], "ref_slots": r["slots"],
                "query_keep_mask": q["keep_mask"], "ref_keep_mask": r["keep_mask"],
                "similarity": ot["similarity"], "transport_plan": ot["transport_plan"],
                "transport_cost": ot["transport_cost"],
                "query_bg_mask": q["bg_mask"], "ref_bg_mask": r["bg_mask"],
                "query_attn_maps": q["attn_maps"], "ref_attn_maps": r["attn_maps"],
                "query_keep_probs": q["keep_probs"], "ref_keep_probs": r["keep_probs"]}

    def extract_embedding(self, x, gs=None):
        return self.encode_view(x, gs)["embedding"]


# #############################################################################
# LOSSES (giữ nguyên với Phase 1)
# #############################################################################

class SymmetricInfoNCE(nn.Module):
    def __init__(self, init_temp=0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(init_temp).log())
    @property
    def temp(self): return self.log_temp.exp().clamp(0.01, 1.0)
    def forward(self, q, r):
        B = q.shape[0]; logits = q @ r.t() / self.temp
        labels = torch.arange(B, device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss, (logits.argmax(dim=-1) == labels).float().mean()

class DWBL(nn.Module):
    def __init__(self, temperature=0.1, margin=0.3):
        super().__init__(); self.t = temperature; self.m = margin
    def forward(self, q, r):
        B = q.shape[0]; sim = q @ r.t(); pos = sim.diag()
        neg = sim[~torch.eye(B, dtype=torch.bool, device=sim.device)].view(B, B-1)
        wneg = (F.softmax(neg/self.t, dim=-1) * torch.exp((neg-self.m)/self.t)).sum(dim=-1)
        return (-torch.log(torch.exp(pos/self.t) / (torch.exp(pos/self.t) + wneg + 1e-8))).mean()

class ContrastiveSlotLoss(nn.Module):
    def __init__(self, temperature=0.1): super().__init__(); self.t = temperature
    def forward(self, out):
        qs = F.normalize(out["query_slots"], dim=-1); rs = F.normalize(out["ref_slots"], dim=-1)
        T = out["transport_plan"]; Tn = T / (T.sum(dim=-1, keepdim=True) + 1e-8)
        loss = -(Tn * F.log_softmax(torch.bmm(qs, rs.transpose(1,2)) / self.t, dim=-1)).sum(dim=-1)
        km = out.get("query_keep_mask")
        return (loss * km).sum() / (km.sum() + 1e-8) if km is not None else loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.s = smooth
    def forward(self, attn_maps, keep_mask=None):
        B, K, N = attn_maps.shape
        an = attn_maps / (attn_maps.sum(dim=-1, keepdim=True) + 1e-8)
        loss, count = 0.0, 0
        for i in range(K):
            for j in range(i+1, K):
                inter = (an[:,i] * an[:,j]).sum(dim=-1); union = an[:,i].sum(-1) + an[:,j].sum(-1)
                loss += ((2*inter + self.s) / (union + self.s)).mean(); count += 1
        return loss / max(count, 1)

class JointLoss(nn.Module):
    def __init__(self, lam_i=1.0, lam_d=1.0, lam_cs=0.5, lam_di=0.3, stage2=20, stage3=40):
        super().__init__()
        self.lam_i=lam_i; self.lam_d=lam_d; self.lam_cs=lam_cs; self.lam_di=lam_di
        self.s2=stage2; self.s3=stage3
        self.infonce=SymmetricInfoNCE(); self.dwbl=DWBL()
        self.csm=ContrastiveSlotLoss(); self.dice=DiceLoss()

    def forward(self, out, epoch=0):
        li, acc = self.infonce(out["query_embedding"], out["ref_embedding"])
        ld = self.dwbl(out["query_embedding"], out["ref_embedding"])
        total = self.lam_i * li + self.lam_d * ld
        lcs = ldi = torch.tensor(0.0, device=total.device)
        if epoch >= self.s2:
            lcs = self.csm(out); ldi = self.dice(out["query_attn_maps"], out.get("query_keep_mask"))
            total = total + self.lam_cs * lcs + self.lam_di * ldi
        stage = 3 if epoch >= self.s3 else (2 if epoch >= self.s2 else 1)
        return {"total_loss": total, "loss_infonce": li.detach(), "loss_dwbl": ld.detach(),
                "loss_csm": lcs.detach() if lcs.requires_grad else lcs,
                "loss_dice": ldi.detach() if ldi.requires_grad else ldi,
                "accuracy": acc, "active_stage": stage}


# #############################################################################
# UNIVERSITY-1652 DATASET
# #############################################################################

class University1652Dataset(Dataset):
    def __init__(self, root, split="train", img_size=384):
        super().__init__()
        self.split = split
        self.pairs = []
        self.query_imgs = []; self.query_labels = []
        self.gallery_imgs = []; self.gallery_labels = []

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.tf_aug = transforms.Compose([
            transforms.Resize((int(img_size*1.1), int(img_size*1.1))),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.15, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

        if split == "train": self._load_train(root)
        else: self._load_test(root)

    def _find_imgs(self, d):
        return sorted(
            glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True) +
            glob.glob(os.path.join(d, "**", "*.jpeg"), recursive=True) +
            glob.glob(os.path.join(d, "**", "*.png"), recursive=True)
        )

    def _load_train(self, root):
        drone_dir = os.path.join(root, "train", "drone")
        sat_dir   = os.path.join(root, "train", "satellite")
        if not os.path.exists(drone_dir):
            print(f"  [WARN] Not found: {drone_dir}"); return
        for cls in sorted(os.listdir(drone_dir)):
            dc = os.path.join(drone_dir, cls)
            sc = os.path.join(sat_dir, cls)
            if not os.path.isdir(dc) or not os.path.isdir(sc): continue
            d_imgs = self._find_imgs(dc); s_imgs = self._find_imgs(sc)
            if not d_imgs or not s_imgs: continue
            for d in d_imgs:
                self.pairs.append((d, s_imgs[0], cls))
        print(f"  [Uni1652 train] {len(self.pairs)} drone-satellite pairs "
              f"from {len(os.listdir(drone_dir))} buildings")

    def _load_test(self, root):
        test = os.path.join(root, "test")
        # Tìm query (drone) folder
        qd = next((os.path.join(test, n) for n in ["query_drone","drone"]
                   if os.path.exists(os.path.join(test, n))), None)
        # Tìm gallery (satellite) folder
        gd = next((os.path.join(test, n) for n in ["gallery_satellite","satellite"]
                   if os.path.exists(os.path.join(test, n))), None)

        for folder, is_query in [(qd, True), (gd, False)]:
            if not folder or not os.path.exists(folder): continue
            for cls in sorted(os.listdir(folder)):
                cd = os.path.join(folder, cls)
                if not os.path.isdir(cd): continue
                for ip in self._find_imgs(cd):
                    if is_query:
                        self.query_imgs.append(ip); self.query_labels.append(cls)
                    else:
                        self.gallery_imgs.append(ip); self.gallery_labels.append(cls)

        print(f"  [Uni1652 test] {len(self.query_imgs)} queries, "
              f"{len(self.gallery_imgs)} gallery")

    def __len__(self):
        return len(self.pairs) if self.split == "train" else len(self.query_imgs)

    def __getitem__(self, idx):
        if self.split == "train":
            dp, sp, cls = self.pairs[idx]
            try:
                d = Image.open(dp).convert("RGB")
                s = Image.open(sp).convert("RGB")
            except:
                d = s = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128,128,128))
            return {"query": self.tf_aug(d), "gallery": self.tf_aug(s), "class_id": cls}
        else:
            try: img = Image.open(self.query_imgs[idx]).convert("RGB")
            except: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128,128,128))
            return {"image": self.tf(img), "label": self.query_labels[idx]}


# #############################################################################
# EVALUATION
# #############################################################################

@torch.no_grad()
def evaluate(model, root, img_size, device):
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    def extract_folder(folder):
        embs, labs = [], []
        if not os.path.exists(folder): return np.array([]), np.array([])
        for cls in sorted(os.listdir(folder)):
            cd = os.path.join(folder, cls)
            if not os.path.isdir(cd): continue
            for ip in sorted(glob.glob(os.path.join(cd,"**","*.jp*g"), recursive=True) +
                             glob.glob(os.path.join(cd,"**","*.png"), recursive=True)):
                try:
                    img = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(device)
                    embs.append(model.extract_embedding(img).cpu().numpy()[0])
                    labs.append(cls)
                except: pass
        return np.array(embs), np.array(labs)

    test = os.path.join(root, "test")
    qd = next((os.path.join(test, n) for n in ["query_drone","drone"]
               if os.path.exists(os.path.join(test, n))), "")
    gd = next((os.path.join(test, n) for n in ["gallery_satellite","satellite"]
               if os.path.exists(os.path.join(test, n))), "")

    print("    Extracting drone queries..."); qe, ql = extract_folder(qd)
    print(f"    Queries:  {len(qe)}")
    print("    Extracting satellite gallery..."); ge, gl = extract_folder(gd)
    print(f"    Gallery:  {len(ge)}")

    if len(qe) == 0 or len(ge) == 0:
        return {"R@1": 0, "R@5": 0, "R@10": 0, "AP": 0}

    sim   = qe @ ge.T
    ranks = np.argsort(-sim, axis=1)
    N     = len(ql)

    results = {}
    for k in [1, 5, 10]:
        results[f"R@{k}"] = sum(1 for i in range(N) if ql[i] in gl[ranks[i, :k]]) / N

    # Mean Average Precision
    ap = 0.0
    for i in range(N):
        rel = (gl[ranks[i]] == ql[i]).astype(float)
        if rel.sum() == 0: continue
        prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        ap += (prec * rel).sum() / rel.sum()
    results["AP"] = ap / N

    return results


# #############################################################################
# TRAINING LOOP
# #############################################################################

def main():
    print("\n[1/5] Loading University-1652 dataset...")
    train_ds = University1652Dataset(UNI1652_ROOT, "train", IMG_SIZE)

    if len(train_ds) == 0:
        print("[ERROR] No training data! Check UNI1652_ROOT:")
        print(f"  Path: {UNI1652_ROOT}")
        print(f"  Exists: {os.path.exists(UNI1652_ROOT)}")
        if os.path.exists(UNI1652_ROOT):
            for d in os.listdir(UNI1652_ROOT): print(f"    {d}")
        return

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )

    print("\n[2/5] Building model...")
    model = GeoSlot(frozen_backbone=(FREEZE_BB > 0)).to(DEVICE)

    # Load weights từ Phase 1 nếu có
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded weights: {RESUME_FROM}")
        if missing: print(f"  Missing keys: {len(missing)}")
        if unexpected: print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print("  Training from scratch (no Phase 1 weights)")

    bb_p   = sum(p.numel() for p in model.backbone.parameters())
    head_p = sum(p.numel() for p in model.parameters()) - bb_p
    print(f"  Backbone: {bb_p:,} params | Head: {head_p:,} params | Total: {bb_p+head_p:,}")

    criterion = JointLoss(stage2=S2_EPOCH, stage3=S3_EPOCH).to(DEVICE)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone")]
                   + list(criterion.parameters()), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS: return (epoch + 1) / WARMUP_EPOCHS
        return 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP_EPOCHS) /
                                    max(1, EPOCHS - WARMUP_EPOCHS)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda, lr_lambda])
    scaler    = GradScaler(enabled=DEVICE.type == "cuda")

    log = {"dataset": "university1652", "history": []}
    best_r1 = 0.0; gs = 0

    print(f"\n[3/5] Training ({EPOCHS} epochs)...\n")

    for epoch in range(EPOCHS):
        if epoch == FREEZE_BB and FREEZE_BB > 0:
            model.backbone.unfreeze()

        model.train(); el = ea = nb = 0; t0 = time.time()

        for batch in tqdm(train_loader, desc=f"E{epoch+1}/{EPOCHS}", leave=False):
            q = batch["query"].to(DEVICE); g = batch["gallery"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=DEVICE.type == "cuda"):
                out = model(q, g, gs); ld = criterion(out, epoch); loss = ld["total_loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            el += loss.item(); ea += ld["accuracy"].item(); nb += 1; gs += 1

        scheduler.step()
        el /= max(nb, 1); ea /= max(nb, 1)
        elapsed = time.time() - t0
        entry = {"epoch": epoch+1, "loss": round(el,4), "acc": round(ea,4),
                 "time": round(elapsed,1), "stage": ld["active_stage"]}

        if (epoch+1) % EVAL_FREQ == 0 or epoch == EPOCHS-1:
            print(f"\n  Evaluating @ epoch {epoch+1}...")
            m = evaluate(model, UNI1652_ROOT, IMG_SIZE, DEVICE)
            entry.update(m); r1 = m.get("R@1", 0)
            print(f"  R@1={r1:.2%}  R@5={m.get('R@5',0):.2%}  "
                  f"R@10={m.get('R@10',0):.2%}  AP={m.get('AP',0):.2%}")
            if r1 > best_r1:
                best_r1 = r1
                torch.save({"epoch": epoch+1, "r1": r1,
                            "model_state_dict": model.state_dict()},
                           os.path.join(OUTPUT_DIR, "best_model_uni1652.pth"))
                print(f"  ★ New best R@1: {r1:.2%}")

        log["history"].append(entry)
        print(f"E{epoch+1}/{EPOCHS} | Loss={el:.4f} | Acc={ea:.1%} | "
              f"Stage={ld['active_stage']} | {elapsed:.0f}s")

        if (epoch+1) % SAVE_FREQ == 0:
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict()},
                       os.path.join(OUTPUT_DIR, f"ckpt_uni1652_ep{epoch+1}.pth"))

    log["best_r1"] = best_r1
    with open(os.path.join(OUTPUT_DIR, "results_uni1652.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Done! Best Drone→Sat R@1 = {best_r1:.2%}")
    print(f"  SOTA target: 96.88% | {'✓ BEAT SOTA!' if best_r1 > 0.9688 else '→ Keep training'}")
    print(f"  Results: {os.path.join(OUTPUT_DIR, 'results_uni1652.json')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()