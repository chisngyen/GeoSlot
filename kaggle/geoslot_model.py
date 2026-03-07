# =============================================================================
# GeoSlot: Shared Model Code for all Kaggle phases
# Import this at the top of each phase script:
#   exec(open("geoslot_model.py").read())
# =============================================================================
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel

try:
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except ImportError:
    _HAS_MAMBA = False

class LinearAttention(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = int(d_model * expand)
        self.proj_in = nn.Linear(d_model, d_inner * 2)
        self.proj_out = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_inner)
        self.act = nn.SiLU()
    def forward(self, x):
        z, gate = self.proj_in(x).chunk(2, dim=-1)
        return self.proj_out(self.norm(self.act(z) * torch.sigmoid(gate)))

class MambaVisionBackbone(nn.Module):
    def __init__(self, model_name="nvidia/MambaVision-L-1K", feature_dim=1568, frozen=False):
        super().__init__()
        # Fix MambaVision bug: torch.linspace creates meta tensor causing .item() to crash
        old_linspace = torch.linspace
        def patched_linspace(*args, **kwargs):
            kwargs["device"] = "cpu"
            return old_linspace(*args, **kwargs)
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
        _, features = self.model(x)
        feat = features[-1]
        B, C, H, W = feat.shape
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

class SSMBlock(nn.Module):
    """SSM block: uses real Mamba if available, else LinearAttention fallback."""
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
    def __init__(self, d, nl=2):
        super().__init__()
        self.fwd=nn.ModuleList([SSMBlock(d) for _ in range(nl)])
        self.bwd=nn.ModuleList([SSMBlock(d) for _ in range(nl)])
        self.mrg=nn.ModuleList([nn.Linear(d*2,d) for _ in range(nl)])
        self.nrm=nn.ModuleList([nn.LayerNorm(d) for _ in range(nl)])
        self.ffn=nn.ModuleList([nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d)) for _ in range(nl)])
    def forward(self, s, km=None):
        B,K,D=s.shape
        for i in range(len(self.fwd)):
            r=s; o=s.norm(dim=-1).argsort(dim=-1,descending=True)
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
        C=torch.cdist(self.cp(sq),self.cp(sr),p=2.0); B,K,M=C.shape
        lK=-C/self.eps
        if mq is not None: lK=lK+torch.log(mq.unsqueeze(-1).clamp(1e-8))
        if mr is not None: lK=lK+torch.log(mr.unsqueeze(-2).clamp(1e-8))
        la=torch.zeros(B,K,1,device=C.device); lb=torch.zeros(B,1,M,device=C.device)
        for _ in range(self.ni):
            la=-torch.logsumexp(lK+lb,2,True); lb=-torch.logsumexp(lK+la,1,True)
        T=torch.exp(lK+la+lb)
        for _ in range(self.mi):
            T=T**2; T=T/(T.sum(-1,True)+1e-8); T=T/(T.sum(-2,True)+1e-8)
        c=(T*C).sum(dim=(-1,-2))
        return {"similarity":torch.sigmoid(-c),"transport_plan":T,"cost_matrix":C,"transport_cost":c}

class GeoSlot(nn.Module):
    """Full pipeline: MambaVision-L → Slot Attention → Graph Mamba → Sinkhorn OT"""
    def __init__(self, backbone_name="nvidia/MambaVision-L-1K", fdim=1568,
                 sdim=256, ms=12, nr=4, edo=512, heads=4, sa_iters=3,
                 gm_layers=2, sk_iters=15, mesh_iters=3, frozen=False,
                 # Ablation flags
                 use_slots=True, use_graph=True, use_ot=True):
        super().__init__()
        self.use_slots = use_slots
        self.use_graph = use_graph
        self.use_ot = use_ot
        self.backbone = MambaVisionBackbone(backbone_name, fdim, frozen)
        if use_slots:
            self.sa = AdaptiveSlotAttention(fdim, sdim, ms, nr, heads, sa_iters)
            if use_graph:
                self.gm = GraphMambaLayer(sdim, gm_layers)
            if use_ot:
                self.ot = SinkhornOT(sdim, sk_iters, 0.05, mesh_iters)
            self.eh = nn.Sequential(nn.LayerNorm(sdim), nn.Linear(sdim, edo))
        else:
            # Baseline: GAP + projection
            self.gap_proj = nn.Sequential(nn.LayerNorm(fdim), nn.Linear(fdim, edo))

    def encode_view(self, x, gs=None):
        f = self.backbone(x)
        if not self.use_slots:
            e = F.normalize(self.gap_proj(f.mean(dim=1)), dim=-1)
            return {"embedding": e, "slots": None, "keep_mask": None,
                    "bg_mask": None, "attn_maps": None, "keep_probs": None, "register_slots": None}
        sa = self.sa(f, gs)
        s = sa["object_slots"]; km = sa["keep_decision"]
        if self.use_graph:
            s = self.gm(s, km)
        w = km / (km.sum(-1, True) + 1e-8)
        e = F.normalize(self.eh((s * w.unsqueeze(-1)).sum(1)), -1)
        return {"slots": s, "embedding": e, "keep_mask": km, "bg_mask": sa["bg_mask"],
                "attn_maps": sa["attn_maps"], "keep_probs": sa["keep_probs"],
                "register_slots": sa["register_slots"]}

    def forward(self, qi, ri, gs=None):
        q = self.encode_view(qi, gs); r = self.encode_view(ri, gs)
        result = {**{f"query_{k}": v for k, v in q.items()},
                  **{f"ref_{k}": v for k, v in r.items()}}
        if self.use_ot and self.use_slots and q["slots"] is not None:
            ot = self.ot(q["slots"], r["slots"], q["keep_mask"], r["keep_mask"])
            result.update({"similarity": ot["similarity"], "transport_plan": ot["transport_plan"],
                           "transport_cost": ot["transport_cost"]})
        else:
            sim = (q["embedding"] * r["embedding"]).sum(-1)
            result.update({"similarity": sim, "transport_plan": None, "transport_cost": None})
        return result

    def extract_embedding(self, x, gs=None):
        return self.encode_view(x, gs)["embedding"]


# === LOSSES ===
class SymmetricInfoNCE(nn.Module):
    def __init__(self, t=0.07):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(t).log())
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
        w=F.softmax(n/self.t,-1); wn=(w*torch.exp((n-self.m)/self.t)).sum(-1)
        pe=torch.exp(p/self.t); return (-torch.log(pe/(pe+wn+1e-8))).mean()

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
        st = 3 if epoch >= self.s3 else (2 if epoch >= self.s2 else 1)
        return {"total_loss": total, "loss_infonce": li.detach(), "loss_dwbl": ld.detach(),
                "loss_csm": lc.detach() if isinstance(lc, torch.Tensor) and lc.requires_grad else lc,
                "loss_dice": ldi.detach() if isinstance(ldi, torch.Tensor) and ldi.requires_grad else ldi,
                "accuracy": acc, "active_stage": st}

print("[OK] GeoSlot model loaded")
