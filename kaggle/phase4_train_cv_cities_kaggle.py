# =============================================================================
# PHASE 4: GeoSlot — Train on CV-Cities (Generalization Test)
# Target: Establish cross-city baseline (no prior SOTA)
# Train: 12 cities → Test: 4 unseen cities
# Hardware: Kaggle H100 | Self-contained
# =============================================================================

# === SETUP (Auto-install dependencies) ===
import subprocess, sys, re as _re

def _run(cmd, verbose=False):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if verbose or result.returncode != 0:
        if result.stdout: print(result.stdout[-500:])
        if result.stderr: print(f"[WARN] {result.stderr[-500:]}")
    return result.returncode == 0

def _pip(pkg, extra=""):
    return _run(f"pip install -q {extra} {pkg}")

print("[1/6] Detecting CUDA version...")
_r = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
_cm = _re.search(r"CUDA Version:\s*([\d.]+)", _r.stdout)
_hw = _cm.group(1) if _cm else "12.6"
_maj, _min = int(_hw.split(".")[0]), int(_hw.split(".")[1]) if "." in _hw else 0
if _maj >= 13 or (_maj == 12 and _min >= 6): _cu = "cu126"
elif _maj == 12 and _min >= 4: _cu = "cu124"
elif _maj == 12 and _min >= 1: _cu = "cu121"
else: _cu = "cu118"
print(f"  Hardware CUDA: {_hw} → PyTorch index: {_cu}")

print("[2/6] Syncing PyTorch...")
_run(f"pip install -q -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{_cu}")

print("[3/6] Installing base packages...")
_pip("transformers==4.44.2")
for _p in ["timm", "tqdm"]:
    try: __import__(_p)
    except ImportError: _pip(_p)

print("[4/6] Build tools & nvcc PATH...")
_pip("packaging ninja wheel setuptools", "--upgrade")
import os
for _cb in ["/usr/local/cuda/bin", "/usr/local/cuda-12/bin", "/usr/local/cuda-12.6/bin"]:
    if os.path.isdir(_cb):
        os.environ["PATH"] = _cb + ":" + os.environ.get("PATH", "")
        os.environ["CUDA_HOME"] = os.path.dirname(_cb)
        break

print("[5/6] Building causal-conv1d...")
_pip("causal-conv1d>=1.4.0", "--no-build-isolation")

print("[6/6] Building mamba_ssm (5-10 min)...")
if not _pip("mamba_ssm", "--no-build-isolation --no-cache-dir"):
    _run("pip install -q --no-build-isolation git+https://github.com/state-spaces/mamba.git")

try:
    from mamba_ssm import Mamba
    _HAS_MAMBA = True; print("  ✓ mamba_ssm loaded")
except ImportError:
    _HAS_MAMBA = False; print("  ✗ mamba_ssm not available — fallback mode")

import os, math, glob, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel


# #############################################################################
# MODEL (self-contained — same as geoslot_model.py)
# #############################################################################

class LinearAttention(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = int(d_model * expand)
        self.proj_in = nn.Linear(d_model, d_inner * 2)
        self.proj_out = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_inner); self.act = nn.SiLU()
    def forward(self, x):
        z, gate = self.proj_in(x).chunk(2, dim=-1)
        return self.proj_out(self.norm(self.act(z) * torch.sigmoid(gate)))

class MambaVisionBackbone(nn.Module):
    def __init__(self, model_name="nvidia/MambaVision-L-1K", feature_dim=1568, frozen=False):
        super().__init__()
        old_linspace = torch.linspace
        def patched_linspace(*args, **kwargs):
            kwargs["device"] = "cpu"; return old_linspace(*args, **kwargs)
        torch.linspace = patched_linspace
        try:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=False)
        finally:
            torch.linspace = old_linspace
        if hasattr(self.model, 'head'): self.model.head = nn.Identity()
        self.feature_dim = feature_dim
        if frozen: self.freeze()
    def freeze(self):
        for p in self.model.parameters(): p.requires_grad = False
        print("  [BACKBONE] Frozen")
    def unfreeze(self):
        for p in self.model.parameters(): p.requires_grad = True
        print("  [BACKBONE] Unfrozen")
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            _, features = self.model(x.float())
        feat = features[-1]; B, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

class BackgroundMask(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,h), nn.ReLU(True), nn.Linear(h,1), nn.Sigmoid())
    def forward(self, f):
        m = self.net(f); return f * m, m

class SlotAttentionCore(nn.Module):
    def __init__(self, dim, fdim, heads=4, iters=3, eps=1e-8):
        super().__init__()
        self.dim=dim; self.H=heads; self.iters=iters; self.eps=eps
        self.dh=dim//heads; self.scale=self.dh**-0.5
        self.to_q=nn.Linear(dim,dim,False); self.to_k=nn.Linear(fdim,dim,False)
        self.to_v=nn.Linear(fdim,dim,False)
        self.gru=nn.GRUCell(dim,dim); self.ni=nn.LayerNorm(fdim); self.ns=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
    def step(self, s, k, v):
        B,K,_=s.shape; sp=s; q=self.to_q(self.ns(s)).view(B,K,self.H,self.dh)
        d=torch.einsum("bihd,bjhd->bihj",q,k)*self.scale
        a=d.flatten(1,2).softmax(1).view(B,K,self.H,-1); ao=a.mean(2)
        a=(a+self.eps)/(a.sum(-1,True)+self.eps)
        u=torch.einsum("bjhd,bihj->bihd",v,a)
        s=self.gru(u.reshape(-1,self.dim),sp.reshape(-1,self.dim)).reshape(B,-1,self.dim)
        return s+self.ff(s), ao
    def forward(self, inp, slots):
        inp=self.ni(inp); B,N,_=inp.shape
        k=self.to_k(inp).view(B,N,self.H,self.dh); v=self.to_v(inp).view(B,N,self.H,self.dh)
        for _ in range(self.iters): slots,a=self.step(slots,k,v)
        return slots, a

class GumbelSelector(nn.Module):
    def __init__(self, d, lb=1):
        super().__init__()
        self.lb=lb; self.net=nn.Sequential(nn.Linear(d,d//2),nn.ReLU(True),nn.Linear(d//2,2))
    def forward(self, s, gs=None):
        lo=self.net(s); tau=max(0.1,1.0-(gs or 0)/100000)
        dec=F.gumbel_softmax(lo,hard=True,tau=tau)[...,1] if self.training else (lo.argmax(-1)==1).float()
        ac=(dec!=0).sum(-1)
        for j in (ac<self.lb).nonzero(as_tuple=True)[0]:
            ia=(dec[j]==0).nonzero(as_tuple=True)[0]
            n=min(self.lb-int(ac[j].item()),len(ia))
            if n>0: dec[j,ia[torch.randperm(len(ia))[:n]]]=1.0
        return dec, F.softmax(lo,-1)[...,1]

class AdaptiveSlotAttention(nn.Module):
    def __init__(self, fdim, sdim, ms, nr, heads=4, iters=3):
        super().__init__()
        self.ms=ms; self.nr=nr
        self.bgm=BackgroundMask(fdim); self.ip=nn.Linear(fdim,sdim)
        self.mu=nn.Parameter(torch.randn(1,ms+nr,sdim)*(sdim**-0.5))
        self.ls=nn.Parameter(torch.zeros(1,ms+nr,sdim))
        self.sa=SlotAttentionCore(sdim,sdim,heads,iters); self.gs=GumbelSelector(sdim)
    def forward(self, f, step=None):
        B=f.shape[0]; f,bm=self.bgm(f); f=self.ip(f)
        mu=self.mu.expand(B,-1,-1); sl=mu+self.ls.exp().expand(B,-1,-1)*torch.randn_like(mu)
        sl,am=self.sa(f,sl); obj=sl[:,:self.ms]; reg=sl[:,self.ms:]
        kd,kp=self.gs(obj,step)
        return {"object_slots":obj*kd.unsqueeze(-1),"register_slots":reg,"bg_mask":bm,
                "attn_maps":am,"keep_decision":kd,"keep_probs":kp}

class SlotSpatialEncoder(nn.Module):
    def __init__(self, slot_dim, enc_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(enc_dim * 2 + 1, slot_dim), nn.GELU(), nn.Linear(slot_dim, slot_dim))
        self.enc_dim = enc_dim
    def _sinusoidal(self, vals, dim):
        half = dim // 2
        freq = torch.exp(torch.arange(half, device=vals.device, dtype=vals.dtype) * -(math.log(10000.0) / half))
        args = vals.unsqueeze(-1) * freq
        return torch.cat([args.sin(), args.cos()], dim=-1)
    def forward(self, attn_maps, spatial_hw):
        B, K, N = attn_maps.shape; H, W = spatial_hw
        w = attn_maps / (attn_maps.sum(-1, True) + 1e-8)
        gy = torch.arange(H, device=attn_maps.device).float() / max(H - 1, 1)
        gx = torch.arange(W, device=attn_maps.device).float() / max(W - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        grid_y = grid_y.reshape(1, 1, -1).expand(B, K, -1)
        grid_x = grid_x.reshape(1, 1, -1).expand(B, K, -1)
        cy = (w * grid_y).sum(-1); cx = (w * grid_x).sum(-1)
        spread = (w * ((grid_y - cy.unsqueeze(-1))**2 + (grid_x - cx.unsqueeze(-1))**2)).sum(-1).sqrt()
        enc = torch.cat([self._sinusoidal(cy, self.enc_dim), self._sinusoidal(cx, self.enc_dim), spread.unsqueeze(-1)], dim=-1)
        return self.mlp(enc)

class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if _HAS_MAMBA:
            self.core = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.core = LinearAttention(d_model, d_state, d_conv, expand)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        return x + self.core(self.norm(x))

class GraphMambaLayer(nn.Module):
    def __init__(self, d, nl=2, spatial_weight=0.3):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.spatial_enc = SlotSpatialEncoder(d)
        self.fwd=nn.ModuleList([SSMBlock(d) for _ in range(nl)])
        self.bwd=nn.ModuleList([SSMBlock(d) for _ in range(nl)])
        self.mrg=nn.ModuleList([nn.Linear(d*2,d) for _ in range(nl)])
        self.nrm=nn.ModuleList([nn.LayerNorm(d) for _ in range(nl)])
        self.ffn=nn.ModuleList([nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d)) for _ in range(nl)])
    def _spatial_order(self, s, attn_maps, spatial_hw):
        B, K, N = attn_maps.shape; H, W = spatial_hw
        w = attn_maps / (attn_maps.sum(-1, True) + 1e-8)
        gy = torch.arange(H, device=s.device).float() / max(H - 1, 1)
        gx = torch.arange(W, device=s.device).float() / max(W - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        grid_y = grid_y.reshape(1, 1, -1).expand(B, K, -1)
        grid_x = grid_x.reshape(1, 1, -1).expand(B, K, -1)
        cy = (w * grid_y).sum(-1); cx = (w * grid_x).sum(-1)
        return (cy * W + cx).argsort(dim=-1)
    def forward(self, s, km=None, attn_maps=None, spatial_hw=None):
        B,K,D=s.shape
        if attn_maps is not None: attn_maps = attn_maps[:, :K, :]
        if attn_maps is not None and spatial_hw is not None:
            s = s + self.spatial_enc(attn_maps, spatial_hw)
        for i in range(len(self.fwd)):
            r=s
            if attn_maps is not None and spatial_hw is not None:
                o = self._spatial_order(s, attn_maps, spatial_hw)
            else:
                o=s.norm(dim=-1).argsort(dim=-1,descending=True)
            bi=torch.arange(B,device=s.device).unsqueeze(1).expand(-1,K)
            so=s[bi,o]; f=self.fwd[i](so); b=self.bwd[i](so.flip(1)).flip(1)
            rv=o.argsort(dim=-1); m=self.mrg[i](torch.cat([f,b],-1))
            s=self.nrm[i](r+m[bi,rv]); s=s+self.ffn[i](s)
            if km is not None: s=s*km.unsqueeze(-1)
        return s

class SinkhornOT(nn.Module):
    def __init__(self, d, ni=15, eps=0.05, mi=3):
        super().__init__()
        self.ni=ni; self.eps=eps; self.mi=mi
        self.cp=nn.Sequential(nn.Linear(d,d),nn.ReLU(True),nn.Linear(d,d))
    def forward(self, sq, sr, mq=None, mr=None):
        sq=sq.float(); sr=sr.float()
        if mq is not None: mq=mq.float()
        if mr is not None: mr=mr.float()
        pq = self.cp(sq); pr = self.cp(sr)
        # Safe L2: torch.cdist backward has NaN at zero distance (sqrt'(0) = inf)
        diff = pq.unsqueeze(2) - pr.unsqueeze(1)
        C = (diff * diff).sum(-1).clamp(min=1e-6).sqrt()
        B,K,M=C.shape
        lK=-C/self.eps
        if mq is not None:
            mu = mq / (mq.sum(-1, True) + 1e-8)
            lK = lK + torch.log(mu.unsqueeze(-1).clamp(1e-8))
        if mr is not None:
            nu = mr / (mr.sum(-1, True) + 1e-8)
            lK = lK + torch.log(nu.unsqueeze(-2).clamp(1e-8))
        la=torch.zeros(B,K,1,device=C.device); lb=torch.zeros(B,1,M,device=C.device)
        for _ in range(self.ni):
            la=-torch.logsumexp(lK+lb,2,True); lb=-torch.logsumexp(lK+la,1,True)
        T=torch.exp(lK+la+lb)
        for _ in range(self.mi):
            T=T**2; T=T/(T.sum(-1,True)+1e-8); T=T/(T.sum(-2,True)+1e-8)
        c=(T*C).sum(dim=(-1,-2))
        return {"similarity":torch.sigmoid(-c),"transport_plan":T,"cost_matrix":C,"transport_cost":c}

class GeoSlot(nn.Module):
    def __init__(self, backbone_name="nvidia/MambaVision-L-1K", fdim=1568,
                 sdim=256, ms=12, nr=4, edo=512, heads=4, sa_iters=3,
                 gm_layers=2, sk_iters=15, mesh_iters=3, frozen=False,
                 use_slots=True, use_graph=True, use_ot=True):
        super().__init__()
        self.use_slots=use_slots; self.use_graph=use_graph; self.use_ot=use_ot
        self.backbone = MambaVisionBackbone(backbone_name, fdim, frozen)
        if use_slots:
            self.sa = AdaptiveSlotAttention(fdim, sdim, ms, nr, heads, sa_iters)
            if use_graph: self.gm = GraphMambaLayer(sdim, gm_layers)
            if use_ot: self.ot = SinkhornOT(sdim, sk_iters, 0.05, mesh_iters)
            self.eh = nn.Sequential(nn.LayerNorm(sdim), nn.Linear(sdim, edo))
        else:
            self.gap_proj = nn.Sequential(nn.LayerNorm(fdim), nn.Linear(fdim, edo))
    def encode_view(self, x, gs=None):
        f = self.backbone(x)  # already float32
        # Force float32: SlotAttention (GRU/softmax), Gumbel (log noise),
        # GraphMamba (SSM) are NaN-prone in float16 under AMP.
        with torch.cuda.amp.autocast(enabled=False):
            f = f.float()
            if not self.use_slots:
                e = F.normalize(self.gap_proj(f.mean(dim=1)), dim=-1)
                return {"embedding":e,"slots":None,"keep_mask":None,"bg_mask":None,
                        "attn_maps":None,"keep_probs":None,"register_slots":None}
            sa = self.sa(f, gs); s = sa["object_slots"]; km = sa["keep_decision"]
            if self.use_graph:
                N = f.shape[1]; H = W = int(N ** 0.5)
                s = self.gm(s, km, attn_maps=sa['attn_maps'], spatial_hw=(H, W))
            w = km / (km.sum(-1, True) + 1e-8)
            e = F.normalize(self.eh((s * w.unsqueeze(-1)).sum(1)), -1)
        return {"slots":s,"embedding":e,"keep_mask":km,"bg_mask":sa["bg_mask"],
                "attn_maps":sa["attn_maps"],"keep_probs":sa["keep_probs"],
                "register_slots":sa["register_slots"]}
    def forward(self, qi, ri, gs=None):
        q = self.encode_view(qi, gs); r = self.encode_view(ri, gs)
        result = {**{f"query_{k}":v for k,v in q.items()}, **{f"ref_{k}":v for k,v in r.items()}}
        if self.use_ot and self.use_slots and q["slots"] is not None:
            ot = self.ot(q["slots"], r["slots"], q["keep_mask"], r["keep_mask"])
            result.update({"similarity":ot["similarity"],"transport_plan":ot["transport_plan"],
                           "transport_cost":ot["transport_cost"]})
        else:
            result.update({"similarity":(q["embedding"]*r["embedding"]).sum(-1),
                           "transport_plan":None,"transport_cost":None})
        return result
    def extract_embedding(self, x, gs=None):
        return self.encode_view(x, gs)["embedding"]

# === LOSSES ===
class SymmetricInfoNCE(nn.Module):
    def __init__(self, t=0.07):
        super().__init__(); self.log_t = nn.Parameter(torch.tensor(t).log())
    @property
    def t(self): return self.log_t.exp().clamp(0.01, 1.0)
    def forward(self, q, r):
        B=q.shape[0]; lo=q@r.t()/self.t; la=torch.arange(B,device=lo.device)
        return (F.cross_entropy(lo,la)+F.cross_entropy(lo.t(),la))/2, (lo.argmax(-1)==la).float().mean()

class DWBL(nn.Module):
    def __init__(self, t=0.1, m=0.3):
        super().__init__(); self.t=t; self.m=m
    def forward(self, q, r):
        B=q.shape[0]; s=q@r.t(); p=s.diag()
        mk=~torch.eye(B,dtype=torch.bool,device=s.device); n=s[mk].view(B,B-1)
        w=F.softmax(n/self.t,-1); wn=(w*torch.exp(((n-self.m)/self.t).clamp(max=20))).sum(-1)
        pe=torch.exp((p/self.t).clamp(max=20)); return (-torch.log(pe/(pe+wn+1e-8))).mean()

class ContrastiveSlotLoss(nn.Module):
    def __init__(self, t=0.5):
        super().__init__(); self.t=t
    def forward(self, out):
        if out.get("transport_plan") is None: return torch.tensor(0.0)
        qs=F.normalize(out["query_slots"],-1); rs=F.normalize(out["ref_slots"],-1)
        T=out["transport_plan"].detach(); Tn=T/(T.sum(-1,True)+1e-8)
        sim=torch.bmm(qs,rs.transpose(1,2))/self.t
        loss=-(Tn*F.log_softmax(sim,-1)).sum(-1)
        km=out.get("query_keep_mask")
        return (loss*km).sum()/(km.sum()+1e-8) if km is not None else loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, s=0.1):
        super().__init__(); self.s=s
    def forward(self, am, km=None):
        if am is None: return torch.tensor(0.0)
        B,K,N=am.shape; an=am/(am.sum(-1,True)+1e-8)
        l=0.0; c=0
        for i in range(K):
            for j in range(i+1,K):
                inter=(an[:,i]*an[:,j]).sum(-1); union=an[:,i].sum(-1)+an[:,j].sum(-1)
                l+=((2*inter+self.s)/(union+self.s)).mean(); c+=1
        return l/max(c,1)

class JointLoss(nn.Module):
    def __init__(self, s2=20, s3=40, warmup=5):
        super().__init__()
        self.s2=s2; self.s3=s3; self.warmup=warmup
        self.infonce=SymmetricInfoNCE(); self.dwbl=DWBL()
        self.csm=ContrastiveSlotLoss(); self.dice=DiceLoss()
    def forward(self, out, epoch=0):
        li, acc = self.infonce(out["query_embedding"], out["ref_embedding"])
        ld = self.dwbl(out["query_embedding"], out["ref_embedding"])
        total = li + ld
        lc = ldi = torch.tensor(0.0, device=total.device)
        if epoch >= self.s2:
            ramp = min(1.0, (epoch - self.s2 + 1) / self.warmup)
            lc = self.csm(out)
            am = out.get("query_attn_maps")
            ldi = self.dice(am, out.get("query_keep_mask"))
            if lc.requires_grad: total = total + ramp * 0.3 * lc
            if ldi.requires_grad: total = total + ramp * 0.1 * ldi
        qbm = out.get("query_bg_mask")
        if qbm is not None and qbm.requires_grad:
            ent = -(qbm * torch.log(qbm + 1e-8) + (1 - qbm) * torch.log(1 - qbm + 1e-8)).mean()
            cov = (qbm.mean() - 0.7) ** 2
            total = total + 0.01 * (ent + cov)
        st = 3 if epoch >= self.s3 else (2 if epoch >= self.s2 else 1)
        return {"total_loss":total,"loss_infonce":li.detach(),"loss_dwbl":ld.detach(),
                "loss_csm":lc.detach() if isinstance(lc,torch.Tensor) and lc.requires_grad else lc,
                "loss_dice":ldi.detach() if isinstance(ldi,torch.Tensor) and ldi.requires_grad else ldi,
                "accuracy":acc,"active_stage":st}

# #############################################################################
# END MODEL — PHASE 4 SPECIFIC CODE BELOW
# #############################################################################

# === CONFIG ===
CVCITIES_ROOT = "/kaggle/input/datasets/chisboiz/cv-cities"
OUTPUT_DIR    = "/kaggle/working"
RESUME_FROM   = None  # e.g. "/kaggle/working/best_model_vigor.pth"

TRAIN_CITIES = ["barcelona","london","losangeles","melbourne",
                "mexico","newyork","paris","rio","santiago","seattle"]
TEST_CITIES  = ["captown","maynila"]  # Unseen cities (Africa + Asia)

# ★ QUICK TEST MODE
QUICK_TEST = True
QT_RATIO   = 0.20

IMG_SIZE   = 256
BATCH_SIZE = 32
EPOCHS = 40 if not QUICK_TEST else 20
EVAL_FREQ = 5 if not QUICK_TEST else 2
SAVE_FREQ = 10 if not QUICK_TEST else 5
LR_BB = 5e-6;  LR_HEAD = 5e-5
FREEZE_BB = 3
S2_EPOCH = 15 if not QUICK_TEST else 99   # QT: disable Stage 2/3
S3_EPOCH = 30 if not QUICK_TEST else 99
WARMUP_LOSS = 5 if not QUICK_TEST else 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("  PHASE 4: GeoSlot — CV-Cities (Generalization)")
print(f"  Train: {len(TRAIN_CITIES)} cities | Test: {len(TEST_CITIES)} unseen cities")
print(f"  Device: {DEVICE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
if QUICK_TEST:
    print(f"  ★ QUICK TEST:  {int(QT_RATIO*100)}% data")
print("=" * 70)


# === DATASET ===
class CVCitiesDataset(Dataset):
    """CV-Cities: pano ↔ satellite matching — reads img_info.csv (original format)."""
    def __init__(self, root, cities, img_size=256, pad=10, split="train"):
        super().__init__()
        self.split = split; self.pairs = []

        # Eval transforms
        self.sat_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        self.pano_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

        # Satellite train augmentation (RandomAffine(90) for rotation)
        self.sat_aug = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=3),
            transforms.Pad(pad, padding_mode='edge'),
            transforms.RandomAffine(90),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

        # Ground / pano train augmentation (Pad + RandomCrop)
        self.pano_aug = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=3),
            transforms.Pad(pad, padding_mode='edge'),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

        import csv
        for city in cities:
            city_dir = os.path.join(root, city, city)
            if not os.path.isdir(city_dir):
                city_dir = os.path.join(root, city)
            if not os.path.isdir(city_dir):
                print(f"  [WARN] {city} not found"); continue
            info_file = os.path.join(city_dir, 'img_info.csv')
            if not os.path.exists(info_file):
                print(f"  [WARN] {city}: img_info.csv not found"); continue

            # Original CSV: NO header, 6 columns:
            # name, longitude, latitude, city, sat_dir, pano_dir
            with open(info_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 6:
                        continue
                    sat_rel = row[4].strip()
                    pano_rel = row[5].strip()
                    sat_path = os.path.join(root, sat_rel)
                    pano_path = os.path.join(root, pano_rel)
                    # Fallback: try relative to city_dir
                    if not os.path.exists(sat_path):
                        sat_path = os.path.join(city_dir, sat_rel)
                    if not os.path.exists(pano_path):
                        pano_path = os.path.join(city_dir, pano_rel)
                    self.pairs.append((pano_path, sat_path, city))

        print(f"  [CV-Cities {split}] {len(self.pairs)} pairs from {len(cities)} cities")

        # QUICK_TEST: limit data
        if QUICK_TEST:
            import random; random.seed(42)
            limit = max(16, int(len(self.pairs) * QT_RATIO))
            if len(self.pairs) > limit:
                self.pairs = random.sample(self.pairs, limit)
                print(f"  [QUICK_TEST] Limited to {len(self.pairs)} pairs")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        pp, sp, city = self.pairs[idx]
        try: pano=Image.open(pp).convert("RGB"); sat=Image.open(sp).convert("RGB")
        except: pano=Image.new("RGB",(256,256),(128,128,128)); sat=Image.new("RGB",(256,256),(128,128,128))
        if self.split == "train":
            pano=self.pano_aug(pano); sat=self.sat_aug(sat)
        else:
            pano=self.pano_tf(pano); sat=self.sat_tf(sat)
        return {"query": pano, "gallery": sat, "city": city, "idx": idx}


# === EVALUATION ===
@torch.no_grad()
def evaluate_cities(model, root, cities, img_size, device):
    model.eval()
    all_results = {}
    for city in cities:
        ds = CVCitiesDataset(root, [city], img_size, split="test")
        if len(ds) == 0: all_results[city] = {"R@1":0}; continue
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        q_e, r_e = [], []
        for b in tqdm(loader, desc=f"Eval {city}", leave=False):
            q_e.append(model.extract_embedding(b["query"].to(device)).cpu())
            r_e.append(model.extract_embedding(b["gallery"].to(device)).cpu())
        qe = torch.cat(q_e,0).numpy(); re = torch.cat(r_e,0).numpy()
        N = len(qe); sim = qe @ re.T; ranks = np.argsort(-sim, axis=1)
        city_res = {}
        for k in [1,5,10]:
            city_res[f"R@{k}"] = sum(1 for i in range(N) if i in ranks[i,:k]) / N
        all_results[city] = city_res
        print(f"    {city}: R@1={city_res['R@1']:.2%}")
    # Average across cities
    avg = {f"R@{k}": np.mean([r.get(f"R@{k}",0) for r in all_results.values()]) for k in [1,5,10]}
    all_results["average"] = avg
    return all_results


# === TRAINING ===
def main():
    print("\n[1] Loading CV-Cities...")
    train_ds = CVCitiesDataset(CVCITIES_ROOT, TRAIN_CITIES, IMG_SIZE, split="train")
    if len(train_ds) == 0: print("[ERROR] No data!"); return
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)

    print("\n[2] Building model...")
    model = GeoSlot(frozen=FREEZE_BB > 0).to(DEVICE)
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        model.load_state_dict(torch.load(RESUME_FROM,map_location=DEVICE)["model_state_dict"],strict=False)
        print(f"  Loaded: {RESUME_FROM}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = JointLoss(s2=S2_EPOCH, s3=S3_EPOCH, warmup=WARMUP_LOSS).to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BB},
        {"params": [p for n,p in model.named_parameters() if not n.startswith("backbone")]
         + list(criterion.parameters()), "lr": LR_HEAD},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 1e-6)
    scaler = GradScaler(enabled=DEVICE.type == "cuda", init_scale=2048, growth_interval=1000)

    log = {"dataset":"cv_cities","history":[]}; best_r1 = 0; gs = 0
    print(f"\n[3] Training ({EPOCHS} epochs)...\n")

    for epoch in range(EPOCHS):
        if epoch == FREEZE_BB: model.backbone.unfreeze()
        model.train(); el=ea=0; nb=0; t0=time.time()
        for batch in tqdm(train_loader, desc=f"E{epoch+1}/{EPOCHS}", leave=False):
            q,g = batch["query"].to(DEVICE), batch["gallery"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=DEVICE.type=="cuda"):
                out=model(q,g,gs); ld=criterion(out,epoch); loss=ld["total_loss"]
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True); scaler.update(); continue
            scaler.step(optimizer); scaler.update()
            el+=loss.item(); ea+=ld["accuracy"].item(); nb+=1; gs+=1
        scheduler.step(); el/=max(nb,1); ea/=max(nb,1)
        entry={"epoch":epoch+1,"loss":round(el,4),"acc":round(ea,4),"time":round(time.time()-t0,1)}

        if (epoch+1)%EVAL_FREQ==0 or epoch==EPOCHS-1:
            print(f"\n  Eval @ epoch {epoch+1} (cross-city)...")
            m = evaluate_cities(model, CVCITIES_ROOT, TEST_CITIES, IMG_SIZE, DEVICE)
            entry["cross_city"] = m
            avg_r1 = m["average"]["R@1"]
            print(f"  Cross-city avg R@1 = {avg_r1:.2%}")
            if avg_r1 > best_r1:
                best_r1 = avg_r1
                torch.save({"epoch":epoch+1,"r1":avg_r1,"model_state_dict":model.state_dict()},
                           os.path.join(OUTPUT_DIR,"best_model_cv_cities.pth"))
                print(f"  ★ Best cross-city R@1: {avg_r1:.2%}")

        log["history"].append(entry)
        print(f"E{epoch+1} | Loss={el:.4f} | Acc={ea:.1%} | {time.time()-t0:.0f}s")

    log["best_r1"]=best_r1
    with open(os.path.join(OUTPUT_DIR,"results_cv_cities.json"),"w") as f: json.dump(log,f,indent=2)
    print(f"\n{'='*70}\n  Done! Best cross-city R@1 = {best_r1:.2%}\n{'='*70}")

if __name__=="__main__": main()
