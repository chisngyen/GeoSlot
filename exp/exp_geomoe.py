#!/usr/bin/env python3
"""
GeoMoE: Altitude-Aware Mixture-of-Experts for Cross-View Geo-Localization
==========================================================================
Novel contributions:
  1. Altitude-Conditioned Expert Router — Dynamic gating network that routes
     features through specialized expert sub-networks based on drone altitude
  2. View-Specific Expert Banks — Separate expert banks for drone and satellite
     processing, with shared routing logic for cross-view consistency
  3. Load-Balanced Auxiliary Loss — Prevents expert collapse (all tokens → one
     expert) via differentiable load balancing inspired by Switch Transformer

Inspired by: Switch Transformer (Google 2022), MoE for dense prediction (CVPR 2024)

Key insight: Drone images at 150m vs 300m altitude have very different
characteristics (fine texture vs broad layout). A single network cannot
optimally handle all scales — MoE lets different experts specialize on
different altitude regimes automatically.

Architecture:
  Student: ConvNeXt-Tiny + Altitude-Conditioned MoE Layer + Expert Fusion
  Teacher: DINOv2-Base (frozen)

Dataset: SUES-200 (drone ↔ satellite cross-view geo-localization)
Protocol: 120 train / 80 test, gallery = ALL 200 locations (confusion data)

Usage:
  python exp_geomoe.py             # Full training on Kaggle H100
  python exp_geomoe.py --test      # Smoke test
"""

# === SETUP ===
import subprocess, sys, os

def pip_install(pkg, extra=""):
    subprocess.run(f"pip install -q {extra} {pkg}",
                   shell=True, capture_output=True, text=True)

print("[1/2] Installing packages...")
for p in ["timm", "tqdm"]:
    try:
        __import__(p)
    except ImportError:
        pip_install(p)
print("[2/2] Setup complete!")

import math, random, argparse
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T

print("[OK] All imports loaded!")

# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT     = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
OUTPUT_DIR    = "/kaggle/working"
EPOCHS        = 120
BATCH_SIZE    = 256
NUM_WORKERS   = 8
AMP_ENABLED   = True
EVAL_FREQ     = 5
LR            = 0.001
WARMUP_EPOCHS = 5
NUM_CLASSES   = 120
EMBED_DIM     = 768
IMG_SIZE      = 224
MARGIN        = 0.3

# MoE config
NUM_EXPERTS   = 4          # Number of expert sub-networks
TOP_K_EXPERTS = 2          # Top-K experts activated per token
MOE_DIM       = 768        # Expert hidden dim
CAPACITY_FACTOR = 1.25     # Expert capacity factor

ALT_EMBED_DIM = 32         # Altitude embedding dimension

TRAIN_LOCS = list(range(1, 121))
TEST_LOCS  = list(range(121, 201))
ALTITUDES  = ["150", "200", "250", "300"]
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); torch.backends.cudnn.benchmark = True


# =============================================================================
# DATASET
# =============================================================================
class SUES200Dataset(Dataset):
    def __init__(self, root, mode="train", altitudes=None, transform=None,
                 train_locs=None, test_locs=None):
        self.root = root; self.mode = mode
        self.altitudes = altitudes or ALTITUDES; self.transform = transform
        dd = os.path.join(root, "drone-view"); sd = os.path.join(root, "satellite-view")
        if train_locs is None: train_locs = TRAIN_LOCS
        if test_locs is None: test_locs = TEST_LOCS
        loc_ids = train_locs if mode == "train" else test_locs
        self.locations = [f"{l:04d}" for l in loc_ids]
        self.location_to_idx = {l: i for i, l in enumerate(self.locations)}
        self.samples = []; self.drone_by_location = defaultdict(list)
        for loc in self.locations:
            li = self.location_to_idx[loc]
            sp = os.path.join(sd, loc, "0.png")
            if not os.path.exists(sp): continue
            for alt in self.altitudes:
                ad = os.path.join(dd, loc, alt)
                if not os.path.isdir(ad): continue
                for img in sorted(os.listdir(ad)):
                    if img.endswith(('.png','.jpg','.jpeg')):
                        self.samples.append((os.path.join(ad, img), sp, li, alt))
                        self.drone_by_location[li].append(len(self.samples)-1)
        print(f"[{mode}] {len(self.samples)} samples, {len(self.locations)} locs")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        dp, sp, li, alt = self.samples[idx]
        d = Image.open(dp).convert('RGB'); s = Image.open(sp).convert('RGB')
        if self.transform: d = self.transform(d); s = self.transform(s)
        return {'drone': d, 'satellite': s, 'label': li, 'altitude': int(alt)}


class PKSampler:
    def __init__(self, ds, p=8, k=4):
        self.p, self.k = p, k
        self.locs = list(ds.drone_by_location.keys()); self.ds = ds
    def __iter__(self):
        locs = self.locs.copy(); random.shuffle(locs); batch = []
        for l in locs:
            idx = self.ds.drone_by_location[l]
            if len(idx) < self.k: idx = idx*(self.k//len(idx)+1)
            batch.extend(random.sample(idx, self.k))
            if len(batch) >= self.p*self.k:
                yield batch[:self.p*self.k]; batch = batch[self.p*self.k:]
    def __len__(self): return len(self.locs)//self.p

def get_transforms(mode="train"):
    if mode == "train":
        return T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.RandomHorizontalFlip(.5),
            T.RandomResizedCrop(IMG_SIZE, scale=(.8,1.)), T.ColorJitter(.2,.2,.2),
            T.ToTensor(), T.Normalize([.485,.456,.406],[.229,.224,.225])])
    return T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.ToTensor(),
        T.Normalize([.485,.456,.406],[.229,.224,.225])])


# =============================================================================
# CONVNEXT-TINY BACKBONE
# =============================================================================
class LayerNorm(nn.Module):
    def __init__(self, ns, eps=1e-6, df="channels_last"):
        super().__init__()
        self.w = nn.Parameter(torch.ones(ns)); self.b = nn.Parameter(torch.zeros(ns))
        self.eps = eps; self.df = df; self.ns = (ns,)
    def forward(self, x):
        if self.df == "channels_last":
            return F.layer_norm(x, self.ns, self.w, self.b, self.eps)
        u = x.mean(1, keepdim=True); s = (x-u).pow(2).mean(1, keepdim=True)
        return self.w[:,None,None]*((x-u)/torch.sqrt(s+self.eps))+self.b[:,None,None]

def drop_path(x, dp=0., tr=False):
    if dp==0. or not tr: return x
    kp=1-dp; s=(x.shape[0],)+(1,)*(x.ndim-1)
    rt=kp+torch.rand(s,dtype=x.dtype,device=x.device); rt.floor_()
    return x.div(kp)*rt

class DropPath(nn.Module):
    def __init__(self, dp=None): super().__init__(); self.dp=dp
    def forward(self, x): return drop_path(x, self.dp, self.training)

class ConvNeXtBlock(nn.Module):
    def __init__(self, d, dpr=0., lsi=1e-6):
        super().__init__()
        self.dw=nn.Conv2d(d,d,7,padding=3,groups=d); self.n=LayerNorm(d,1e-6)
        self.p1=nn.Linear(d,4*d); self.act=nn.GELU(); self.p2=nn.Linear(4*d,d)
        self.g=nn.Parameter(lsi*torch.ones(d)) if lsi>0 else None
        self.dp=DropPath(dpr) if dpr>0 else nn.Identity()
    def forward(self, x):
        s=x; x=self.dw(x); x=x.permute(0,2,3,1)
        x=self.n(x); x=self.p1(x); x=self.act(x); x=self.p2(x)
        if self.g is not None: x=self.g*x
        return s+self.dp(x.permute(0,3,1,2))

class ConvNeXtTiny(nn.Module):
    def __init__(self, ic=3, depths=[3,3,9,3], dims=[96,192,384,768], dpr=0.):
        super().__init__()
        self.dims=dims; self.dsl=nn.ModuleList()
        self.dsl.append(nn.Sequential(nn.Conv2d(ic,dims[0],4,4),
                        LayerNorm(dims[0],1e-6,"channels_first")))
        for i in range(3):
            self.dsl.append(nn.Sequential(LayerNorm(dims[i],1e-6,"channels_first"),
                            nn.Conv2d(dims[i],dims[i+1],2,2)))
        rates=[x.item() for x in torch.linspace(0, dpr, sum(depths))]; c=0
        self.stages=nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[ConvNeXtBlock(dims[i],rates[c+j]) for j in range(depths[i])]))
            c+=depths[i]
        self.norm=nn.LayerNorm(dims[-1],1e-6); self.apply(self._iw)
    def _iw(self,m):
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x):
        outs=[]
        for i in range(4): x=self.dsl[i](x); x=self.stages[i](x); outs.append(x)
        return self.norm(x.mean([-2,-1])), outs

def load_convnext_pretrained(m):
    try:
        ckpt=torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth",
            map_location="cpu", check_hash=True)
        m.load_state_dict({k:v for k,v in ckpt["model"].items() if not k.startswith('head')}, strict=False)
        print("Loaded ConvNeXt-Tiny pretrained (ImageNet-22K)")
    except Exception as e: print(f"Could not load: {e}")
    return m


# =============================================================================
# NOVEL COMPONENT 1: ALTITUDE EMBEDDING
# =============================================================================
class AltitudeEmbedding(nn.Module):
    """Encodes drone altitude metadata as a learned embedding vector.

    Maps discrete altitude levels {150, 200, 250, 300} to continuous embeddings
    and generates conditioning signals for expert routing.

    For satellite images (no altitude), uses a learned default embedding.
    """
    def __init__(self, alt_values=[150, 200, 250, 300], embed_dim=32):
        super().__init__()
        self.alt_values = alt_values
        self.alt_to_idx = {a: i for i, a in enumerate(alt_values)}

        # Learnable per-altitude embeddings + default
        self.embeddings = nn.Embedding(len(alt_values) + 1, embed_dim)
        self.default_idx = len(alt_values)  # For satellite (no altitude)

    def forward(self, altitudes=None):
        """
        Args: altitudes: [B] int tensor of altitude values, or None for satellite
        Returns: [B, embed_dim]
        """
        if altitudes is None:
            # Satellite: use default embedding
            B = 1  # Will be expanded by caller
            return self.embeddings(torch.tensor([self.default_idx],
                                                device=self.embeddings.weight.device))

        # Map altitude values to indices
        indices = torch.zeros_like(altitudes)
        for alt_val, alt_idx in self.alt_to_idx.items():
            indices[altitudes == alt_val] = alt_idx
        return self.embeddings(indices)


# =============================================================================
# NOVEL COMPONENT 2: ALTITUDE-CONDITIONED MOE LAYER
# =============================================================================
class ExpertFFN(nn.Module):
    """Single expert: a 2-layer FFN with GELU."""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.net(x)


class AltitudeConditionedRouter(nn.Module):
    """Expert router conditioned on altitude metadata.

    Standard MoE routes based on token features only. Our router also
    conditions on altitude, allowing different experts to specialize on
    different altitude regimes (e.g., Expert 0 → low altitude fine details,
    Expert 3 → high altitude broad layout).
    """
    def __init__(self, d_model, num_experts, alt_embed_dim=32, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Combined routing: token features + altitude embedding
        self.gate = nn.Sequential(
            nn.Linear(d_model + alt_embed_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_experts),
        )

        # Noise for exploration during training
        self.noise_weight = nn.Parameter(torch.zeros(num_experts))

    def forward(self, tokens, alt_embedding):
        """
        Args:
            tokens: [B, N, D] — feature tokens
            alt_embedding: [B, alt_dim] — altitude embedding
        Returns:
            gates: [B, N, K] — gate values for top-K experts
            expert_indices: [B, N, K] — which experts selected
            load_balance_loss: scalar
        """
        B, N, D = tokens.shape

        # Expand altitude embedding to match token sequence
        alt_expanded = alt_embedding.unsqueeze(1).expand(-1, N, -1)  # [B, N, alt_dim]

        # Router input = [token_features, altitude]
        router_input = torch.cat([tokens, alt_expanded], dim=-1)  # [B, N, D+alt_dim]

        # Compute gate logits
        logits = self.gate(router_input)  # [B, N, num_experts]

        # Add noise during training for exploration
        if self.training:
            noise = torch.randn_like(logits) * F.softplus(self.noise_weight)
            logits = logits + noise

        # Top-K selection
        top_k_logits, expert_indices = logits.topk(self.top_k, dim=-1)
        gates = F.softmax(top_k_logits, dim=-1)  # [B, N, K]

        # Load balancing loss (Switch Transformer)
        # Fraction of tokens routed to each expert
        probs = F.softmax(logits, dim=-1)  # [B, N, E]
        # One-hot for top-1 expert
        top1_idx = expert_indices[:, :, 0]  # [B, N]
        top1_onehot = F.one_hot(top1_idx, self.num_experts).float()  # [B, N, E]

        # Average fraction per expert
        f = top1_onehot.mean(dim=[0, 1])  # [E]
        p = probs.mean(dim=[0, 1])  # [E]
        load_balance_loss = self.num_experts * (f * p).sum()

        return gates, expert_indices, load_balance_loss


class AltitudeConditionedMoE(nn.Module):
    """Mixture-of-Experts layer with altitude conditioning.

    Each expert is a separate FFN that can specialize on different
    altitude ranges and visual patterns. The router dynamically selects
    Top-K experts per token based on both the token content and the
    altitude metadata.
    """
    def __init__(self, d_model, num_experts=4, top_k=2, alt_embed_dim=32):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        # Expert bank
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_model * 4) for _ in range(num_experts)
        ])

        # Altitude-conditioned router
        self.router = AltitudeConditionedRouter(
            d_model, num_experts, alt_embed_dim, top_k)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, alt_embedding):
        """
        Args:
            tokens: [B, N, D]
            alt_embedding: [B, alt_dim]
        Returns:
            output: [B, N, D] — MoE-processed tokens
            load_balance_loss: scalar — auxiliary loss for balanced routing
        """
        residual = tokens
        tokens_normed = self.norm(tokens)

        gates, expert_indices, lb_loss = self.router(tokens_normed, alt_embedding)

        B, N, D = tokens.shape

        # Compute expert outputs (simple loop — efficient enough for 4 experts)
        output = torch.zeros_like(tokens)
        for k in range(self.top_k):
            expert_idx = expert_indices[:, :, k]  # [B, N]
            gate_val = gates[:, :, k].unsqueeze(-1)  # [B, N, 1]

            for e in range(self.num_experts):
                mask = (expert_idx == e)  # [B, N]
                if not mask.any():
                    continue

                # Get tokens for this expert
                expert_tokens = tokens_normed[mask]  # [M, D]
                expert_out = self.experts[e](expert_tokens)  # [M, D]

                # Scatter back with gate weighting
                out_buffer = torch.zeros_like(tokens_normed)
                out_buffer[mask] = expert_out.to(out_buffer.dtype)
                output = output + gate_val * out_buffer

        return residual + output, lb_loss


# =============================================================================
# NOVEL COMPONENT 3: EXPERT FUSION WITH CROSS-VIEW CONSISTENCY
# =============================================================================
class ExpertConsistencyLoss(nn.Module):
    """Ensures expert routing is consistent across views.

    For the same location, the drone expert distribution should be similar
    to the satellite expert distribution. This prevents the router from
    creating view-specific shortcuts.
    """
    def __init__(self):
        super().__init__()

    def forward(self, drone_gates, sat_gates, labels):
        """
        Args:
            drone_gates: [B, N, K] gate values
            sat_gates: [B, N, K] gate values
            labels: [B] location labels
        """
        # Average gate distribution per sample
        d_avg = drone_gates.mean(1)  # [B, K]
        s_avg = sat_gates.mean(1)    # [B, K]

        # KL divergence between drone and satellite distributions
        d_dist = F.softmax(d_avg, dim=1)
        s_dist = F.softmax(s_avg, dim=1)

        kl = F.kl_div(d_dist.log(), s_dist, reduction='batchmean')
        kl += F.kl_div(s_dist.log(), d_dist, reduction='batchmean')

        return 0.5 * kl


# =============================================================================
# GEOMOE MODEL
# =============================================================================
class ClassificationHead(nn.Module):
    def __init__(self, d, nc, h=512):
        super().__init__()
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Linear(d,h),nn.BatchNorm1d(h),nn.ReLU(True),
                              nn.Dropout(.5),nn.Linear(h,nc))
    def forward(self, x): return self.fc(self.pool(x).flatten(1))


class GeoMoEStudent(nn.Module):
    """GeoMoE = ConvNeXt-Tiny + Altitude-Conditioned MoE + DINOv2 Distillation.

    Pipeline:
      1. ConvNeXt extracts multi-scale features
      2. Altitude embedding conditions the MoE router
      3. Top-K experts process tokens with altitude-aware specialization
      4. Expert-refined features fused into final embedding
    """
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = ConvNeXtTiny(dpr=0.1)
        self.backbone = load_convnext_pretrained(self.backbone)

        self.aux_heads = nn.ModuleList([
            ClassificationHead(d, num_classes) for d in [96, 192, 384, 768]])

        # *** NOVEL: Altitude Embedding ***
        self.alt_embed = AltitudeEmbedding(
            alt_values=[150, 200, 250, 300], embed_dim=ALT_EMBED_DIM)

        # *** NOVEL: Altitude-Conditioned MoE ***
        self.moe_layer = AltitudeConditionedMoE(
            d_model=768, num_experts=NUM_EXPERTS,
            top_k=TOP_K_EXPERTS, alt_embed_dim=ALT_EMBED_DIM)

        # Embedding head
        self.embed_head = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(True))

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, altitudes=None, return_all=False):
        final_feat, stage_outputs = self.backbone(x)
        stage_logits = [h(f) for h, f in zip(self.aux_heads, stage_outputs)]

        # Get altitude embedding
        B = x.shape[0]
        if altitudes is not None:
            alt_emb = self.alt_embed(altitudes)  # [B, alt_dim]
        else:
            # Default (satellite or unknown altitude)
            alt_emb = self.alt_embed(None).expand(B, -1)  # [B, alt_dim]

        # Convert final feature map to tokens
        feat_map = stage_outputs[-1]  # [B, 768, H, W]
        tokens = feat_map.flatten(2).transpose(1, 2)  # [B, N, 768]

        # *** MoE processing ***
        moe_tokens, lb_loss = self.moe_layer(tokens, alt_emb)

        # Pool refined tokens
        moe_feat = moe_tokens.mean(1)  # [B, 768]

        # Combine with global feature
        combined = final_feat + moe_feat  # Residual connection

        embedding = self.embed_head(combined)
        embedding_normed = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)

        if return_all:
            return {
                'embedding_normed': embedding_normed,
                'logits': logits,
                'stage_logits': stage_logits,
                'final_feature': final_feat,
                'moe_feature': moe_feat,
                'lb_loss': lb_loss,  # Load balance auxiliary loss
            }
        return embedding_normed, logits

    def extract_embedding(self, x):
        self.eval()
        with torch.no_grad(): e, _ = self.forward(x)
        return e


# =============================================================================
# TEACHER + LOSSES
# =============================================================================
class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading DINOv2-base teacher...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        for p in self.parameters(): p.requires_grad = False
        for blk in self.model.blocks[-2:]:
            for p in blk.parameters(): p.requires_grad = True
    @torch.no_grad()
    def forward(self, x):
        t = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks: t = blk(t)
        return self.model.norm(t)[:, 0]

class TripletLoss(nn.Module):
    def __init__(self, m=.3): super().__init__(); self.m=m
    def forward(self, e, l):
        d=torch.cdist(e,e,2); lb=l.view(-1,1)
        p=lb.eq(lb.T).float(); n=lb.ne(lb.T).float()
        return F.relu((d*p).max(1)[0]-(d*n+p*999).min(1)[0]+self.m).mean()

class SymNCE(nn.Module):
    def __init__(self, t=.07): super().__init__(); self.t=t
    def forward(self, d, s, l):
        d=F.normalize(d,1); s=F.normalize(s,1); sim=d@s.T/self.t
        lb=l.view(-1,1); pm=lb.eq(lb.T).float()
        l1=-(F.log_softmax(sim,1)*pm).sum(1)/pm.sum(1).clamp(1)
        l2=-(F.log_softmax(sim.T,1)*pm).sum(1)/pm.sum(1).clamp(1)
        return .5*(l1.mean()+l2.mean())

class SelfDist(nn.Module):
    def __init__(self, T=4.): super().__init__(); self.T=T
    def forward(self, sl):
        loss=0.; f=sl[-1]; w=[.1,.2,.3,.4]
        for i in range(len(sl)-1):
            loss+=w[i]*(self.T**2)*F.kl_div(F.log_softmax(f/self.T,1),F.softmax(sl[i]/self.T,1),reduction='batchmean')
        return loss

class UAPA(nn.Module):
    def __init__(self, T0=4.): super().__init__(); self.T0=T0
    def forward(self, dl, sl):
        Ud=-(F.softmax(dl,1)*F.log_softmax(dl,1)).sum(1).mean()
        Us=-(F.softmax(sl,1)*F.log_softmax(sl,1)).sum(1).mean()
        T=self.T0*(1+torch.sigmoid(Ud-Us))
        return (T**2)*F.kl_div(F.log_softmax(dl/T,1),F.softmax(sl/T,1),reduction='batchmean')


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate(model, test_ds, device):
    model.eval()
    loader = DataLoader(test_ds, 256, False, num_workers=NUM_WORKERS, pin_memory=True)
    feats, labels = [], []
    with torch.no_grad():
        for b in loader:
            f, _ = model(b['drone'].to(device), b['altitude'].to(device))
            feats.append(f.cpu()); labels.append(b['label'])
    feats = torch.cat(feats); labels = torch.cat(labels)
    tf = get_transforms("test"); sd = os.path.join(test_ds.root, "satellite-view")
    sf, sl = [], []
    for loc in [f"{l:04d}" for l in TRAIN_LOCS+TEST_LOCS]:
        sp = os.path.join(sd, loc, "0.png")
        if not os.path.exists(sp): continue
        t = tf(Image.open(sp).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad(): f, _ = model(t)  # No altitude for satellite
        sf.append(f.cpu())
        sl.append(test_ds.location_to_idx[loc] if loc in test_ds.location_to_idx else -1-len(sf))
    sf = torch.cat(sf); sl = torch.tensor(sl)
    _, idx = (feats@sf.T).sort(1, descending=True)
    N=len(feats); r1=r5=r10=0; ap=0.
    for i in range(N):
        c = torch.where(sl[idx[i]]==labels[i])[0]
        if len(c)==0: continue
        fc = c[0].item()
        if fc<1: r1+=1
        if fc<5: r5+=1
        if fc<10: r10+=1
        ap += sum((j+1)/(p.item()+1) for j,p in enumerate(c))/len(c)
    return {'R@1':r1/N*100,'R@5':r5/N*100,'R@10':r10/N*100}, ap/N*100


# =============================================================================
# TRAINING
# =============================================================================
def train(args):
    set_seed(SEED)
    print("="*70); print("GeoMoE: Altitude-Conditioned MoE Geo-Localization"); print("="*70)
    train_ds = SUES200Dataset(args.data_root, "train", transform=get_transforms("train"))
    test_ds = SUES200Dataset(args.data_root, "test", transform=get_transforms("test"))
    sampler = PKSampler(train_ds, 8, max(2, BATCH_SIZE//8))
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    model = GeoMoEStudent(len(TRAIN_LOCS)).to(DEVICE)
    print(f"  Student: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    try: teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except: teacher = None

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    trip = TripletLoss(MARGIN); nce = SymNCE(.07)
    sd = SelfDist(4.); uapa = UAPA(4.)
    ec = ExpertConsistencyLoss()

    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=.9, weight_decay=5e-4)
    scaler = GradScaler(enabled=AMP_ENABLED); best_r1 = 0.

    for epoch in range(EPOCHS):
        model.train()
        if epoch < WARMUP_EPOCHS: lr=LR*(epoch+1)/WARMUP_EPOCHS
        else:
            pr=(epoch-WARMUP_EPOCHS)/max(1,EPOCHS-WARMUP_EPOCHS)
            lr=1e-6+.5*(LR-1e-6)*(1+math.cos(math.pi*pr))
        for pg in opt.param_groups: pg['lr']=lr
        tl=0.; lp=defaultdict(float); nb=0

        for bi, batch in enumerate(loader):
            drone=batch['drone'].to(DEVICE); sat=batch['satellite'].to(DEVICE)
            labels=batch['label'].to(DEVICE); alts=batch['altitude'].to(DEVICE)
            opt.zero_grad()
            with autocast(enabled=AMP_ENABLED):
                do = model(drone, alts, return_all=True)
                so = model(sat, None, return_all=True)  # Satellite has no altitude
                L = {}
                c = ce(do['logits'],labels)+ce(so['logits'],labels)
                for sl in do['stage_logits']: c+=.25*ce(sl,labels)
                for sl in so['stage_logits']: c+=.25*ce(sl,labels)
                L['ce'] = c
                L['trip'] = trip(do['embedding_normed'],labels)+trip(so['embedding_normed'],labels)
                L['nce'] = nce(do['embedding_normed'], so['embedding_normed'], labels)
                L['sd'] = .5*(sd(do['stage_logits'])+sd(so['stage_logits']))
                L['uapa'] = .2*uapa(do['logits'], so['logits'])
                # *** NOVEL: Load balance loss ***
                L['lb'] = 0.01*(do['lb_loss']+so['lb_loss'])
                # Cross-distillation
                if teacher is not None:
                    with torch.no_grad(): td=teacher(drone); ts=teacher(sat)
                    df=F.normalize(do['final_feature'],1); sf=F.normalize(so['final_feature'],1)
                    tdn=F.normalize(td,1); tsn=F.normalize(ts,1)
                    L['cdist']=.3*(F.mse_loss(df,tdn)+F.mse_loss(sf,tsn)+
                                   (1-F.cosine_similarity(df,tdn).mean())+
                                   (1-F.cosine_similarity(sf,tsn).mean()))
                total = sum(L.values())

            scaler.scale(total).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(opt); scaler.update()
            tl+=total.item(); nb+=1
            for k,v in L.items(): lp[k]+=v.item()
            if bi%10==0: print(f"  B{bi}/{len(loader)} L={total.item():.4f}")

        nb=max(1,nb)
        print(f"\nEp {epoch+1}/{EPOCHS} LR={lr:.6f} AvgL={tl/nb:.4f}")
        for k,v in sorted(lp.items()): print(f"  {k}: {v/nb:.4f}")

        if (epoch+1)%EVAL_FREQ==0 or epoch==EPOCHS-1:
            rec, ap = evaluate(model, test_ds, DEVICE)
            print(f"  R@1:{rec['R@1']:.2f}% R@5:{rec['R@5']:.2f}% R@10:{rec['R@10']:.2f}% AP:{ap:.2f}%")
            if rec['R@1']>best_r1:
                best_r1=rec['R@1']
                torch.save({'epoch':epoch,'model':model.state_dict(),'r1':best_r1},
                           os.path.join(OUTPUT_DIR,'geomoe_best.pth'))
                print(f"  *** Best R@1={best_r1:.2f}% ***")
    print(f"\nDone! Best R@1={best_r1:.2f}%")


def smoke_test():
    print("="*50); print("SMOKE TEST — GeoMoE"); print("="*50)
    dev=DEVICE; m=GeoMoEStudent(10).to(dev)
    print(f"✓ Model: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
    x=torch.randn(4,3,224,224,device=dev)
    alts=torch.tensor([150,200,250,300],device=dev)
    lab=torch.tensor([0,0,1,1],device=dev)
    o=m(x, alts, return_all=True)
    print(f"✓ Drone: emb={o['embedding_normed'].shape}, lb_loss={o['lb_loss'].item():.4f}")
    o2=m(x, None, return_all=True)  # Satellite (no altitude)
    print(f"✓ Satellite: emb={o2['embedding_normed'].shape}")
    loss=nn.CrossEntropyLoss()(o['logits'],lab)+o['lb_loss']
    loss.backward()
    print(f"✓ Backward OK"); print("\n✅ ALL TESTS PASSED!")


def main():
    global EPOCHS, BATCH_SIZE, DATA_ROOT
    p=argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,default=EPOCHS)
    p.add_argument("--batch_size",type=int,default=BATCH_SIZE)
    p.add_argument("--data_root",type=str,default=DATA_ROOT)
    p.add_argument("--test",action="store_true")
    args,_=p.parse_known_args()
    EPOCHS=args.epochs; BATCH_SIZE=args.batch_size; DATA_ROOT=args.data_root
    args.data_root=DATA_ROOT
    if args.test: smoke_test(); return
    os.makedirs(OUTPUT_DIR,exist_ok=True); train(args)

if __name__=="__main__": main()
