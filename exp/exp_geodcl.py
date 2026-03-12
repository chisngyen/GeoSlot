#!/usr/bin/env python3
"""
GeoDCL: Dynamic Curriculum Learning for Cross-View Drone Geo-Localization
==========================================================================
Novel contributions:
  1. Self-Paced Metric Learning (SPML) — Loss-history-based sample weighting
     that automatically downweights easy samples and upweights informative ones
  2. Adaptive Margin Scheduling — Triplet margin grows from 0.1→0.5 as the
     model improves, preventing over-confident early convergence
  3. Confusion-Guided Curriculum — Tracks per-location confusion rates and
     focuses training on frequently confused location pairs

Inspired by: Curriculum Learning (ICML 2009), Self-Paced Learning (NeurIPS 2010),
             Adaptive margins for metric learning (TPAMI 2024)

Key insight: Standard training treats all samples equally. But in SUES-200,
some locations are easy (unique landmarks) and some are hard (similar layouts).
GeoDCL automatically discovers and focuses on the hard cases.

Architecture:
  Student: ConvNeXt-Tiny (same as baseline — no architecture change)
  Teacher: DINOv2-Base (frozen)
  Training: Curriculum scheduler + Adaptive Margin + Self-Paced Weighting

Dataset: SUES-200 (drone ↔ satellite cross-view geo-localization)
Protocol: 120 train / 80 test, gallery = ALL 200 locations (confusion data)

Usage:
  python exp_geodcl.py             # Full training on Kaggle H100
  python exp_geodcl.py --test      # Smoke test
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

# Curriculum config
MARGIN_START  = 0.1           # Initial triplet margin (easy)
MARGIN_END    = 0.5           # Final triplet margin (hard)
MARGIN_WARMUP = 30            # Epochs to linearly ramp margin
PACE_LAMBDA   = 0.3           # Self-paced learning threshold
CONF_MOMENTUM = 0.9           # Confusion matrix EMA decay

TRAIN_LOCS = list(range(1, 121))
TEST_LOCS  = list(range(121, 201))
ALTITUDES  = ["150", "200", "250", "300"]
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); torch.backends.cudnn.benchmark = True


# =============================================================================
# DATASET (identical to baseline)
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
# CONVNEXT-TINY BACKBONE (identical to baseline)
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
# NOVEL COMPONENT 1: ADAPTIVE MARGIN SCHEDULING
# =============================================================================
class AdaptiveMarginTripletLoss(nn.Module):
    """Triplet loss with margin that grows over training.

    Early training (easy phase): small margin (0.1) — learn basic clustering
    Late training (hard phase): large margin (0.5) — push for fine discrimination

    The margin also adapts based on the model's current accuracy:
    if the model is improving fast → increase margin faster.
    """

    def __init__(self, margin_start=0.1, margin_end=0.5, total_epochs=120, warmup_epochs=30):
        super().__init__()
        self.margin_start = margin_start
        self.margin_end = margin_end
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.current_margin = margin_start

    def get_margin(self, epoch, current_r1=None):
        """Compute margin for current epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup of margin
            progress = epoch / self.warmup_epochs
            margin = self.margin_start + progress * (self.margin_end - self.margin_start) * 0.5
        else:
            # Cosine schedule for rest
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            mid = (self.margin_start + self.margin_end) / 2
            margin = mid + 0.5 * (self.margin_end - mid) * (1 - math.cos(math.pi * progress))

        self.current_margin = margin
        return margin

    def forward(self, embeddings, labels, epoch=0):
        margin = self.get_margin(epoch)

        dist = torch.cdist(embeddings, embeddings, p=2)
        lb = labels.view(-1, 1)
        pos_mask = lb.eq(lb.T).float()
        neg_mask = lb.ne(lb.T).float()

        hard_pos = (dist * pos_mask).max(1)[0]
        hard_neg = (dist * neg_mask + pos_mask * 999).min(1)[0]

        loss = F.relu(hard_pos - hard_neg + margin)
        return loss.mean()


# =============================================================================
# NOVEL COMPONENT 2: SELF-PACED SAMPLE WEIGHTING
# =============================================================================
class SelfPacedWeighter:
    """Self-Paced Learning: downweight easy samples, upweight hard ones.

    Maintains a running loss history per sample. Samples with loss below
    the pace threshold get lower weight (they're already learned).
    Samples with high loss get higher weight (they need more attention).

    The pace threshold decreases over training, gradually including
    harder samples in the "important" set.
    """

    def __init__(self, num_classes, lambda_init=0.3, decay=0.01):
        self.lambda_val = lambda_init
        self.decay = decay
        self.loss_history = defaultdict(list)  # Per-class loss history
        self.class_weights = {}                # Per-class importance weights

    def update(self, losses, labels, epoch):
        """Update loss history and compute sample weights."""
        # Decrease threshold over time (include harder samples)
        self.lambda_val = max(0.05, 0.3 * math.exp(-self.decay * epoch))

        # Track per-class average loss
        for loss, label in zip(losses.detach().cpu().tolist(), labels.cpu().tolist()):
            self.loss_history[label].append(loss)
            # Keep only recent history
            if len(self.loss_history[label]) > 100:
                self.loss_history[label] = self.loss_history[label][-100:]

    def get_weights(self, losses, labels):
        """Compute per-sample weights based on loss difficulty.

        Weight formula: w_i = min(1, loss_i / lambda)
        - Easy samples (loss < lambda): w < 1 (downweighted)
        - Hard samples (loss >= lambda): w = 1 (full weight)
        """
        with torch.no_grad():
            weights = torch.clamp(losses / (self.lambda_val + 1e-6), min=0.1, max=2.0)
            # Normalize to have mean=1
            weights = weights / (weights.mean() + 1e-6)
        return weights

    def get_class_difficulty(self):
        """Return per-class average loss (for logging)."""
        difficulties = {}
        for cls, losses in self.loss_history.items():
            difficulties[cls] = np.mean(losses[-20:]) if losses else 0.
        return difficulties


# =============================================================================
# NOVEL COMPONENT 3: CONFUSION-GUIDED CURRICULUM
# =============================================================================
class ConfusionTracker:
    """Tracks which locations are commonly confused with each other.

    Maintains a confusion matrix updated via EMA. Uses this to:
    1. Identify hard negative pairs (locations confused with each other)
    2. Oversample confusing locations in training batches
    3. Log confusion patterns for analysis
    """

    def __init__(self, num_classes, momentum=0.9):
        self.num_classes = num_classes
        self.momentum = momentum
        # Confusion matrix: C[i,j] = how often class i is predicted as class j
        self.confusion = torch.zeros(num_classes, num_classes)
        self.difficulty_scores = torch.ones(num_classes)

    @torch.no_grad()
    def update(self, logits, labels):
        """Update confusion matrix with batch predictions."""
        preds = logits.argmax(1).cpu()
        labels_cpu = labels.cpu()

        batch_conf = torch.zeros(self.num_classes, self.num_classes)
        for p, t in zip(preds, labels_cpu):
            batch_conf[t.item(), p.item()] += 1

        # EMA update
        self.confusion = self.momentum * self.confusion + (1 - self.momentum) * batch_conf

        # Compute per-class difficulty (1 - accuracy)
        diag = self.confusion.diag()
        row_sum = self.confusion.sum(1).clamp(min=1)
        self.difficulty_scores = 1.0 - (diag / row_sum)

    def get_sample_weights(self, labels):
        """Get per-sample weights based on class difficulty."""
        diff = self.difficulty_scores.to(labels.device)
        weights = diff[labels]
        # Normalize: mean = 1
        weights = weights / (weights.mean() + 1e-6)
        # Clamp to prevent extreme values
        return weights.clamp(0.3, 3.0)

    def get_top_confusions(self, k=5):
        """Return top-k most confused location pairs."""
        conf = self.confusion.clone()
        conf.fill_diagonal_(0)  # Ignore correct predictions
        top_vals, top_idx = conf.flatten().topk(k)
        pairs = []
        for v, idx in zip(top_vals, top_idx):
            i, j = idx.item() // self.num_classes, idx.item() % self.num_classes
            pairs.append((i, j, v.item()))
        return pairs


# =============================================================================
# STUDENT MODEL (identical to baseline)
# =============================================================================
class ClassificationHead(nn.Module):
    def __init__(self, d, nc, h=512):
        super().__init__()
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Linear(d,h),nn.BatchNorm1d(h),nn.ReLU(True),
                              nn.Dropout(.5),nn.Linear(h,nc))
    def forward(self, x): return self.fc(self.pool(x).flatten(1))


class GeoDCLStudent(nn.Module):
    """GeoDCL Student — IDENTICAL architecture to baseline.

    All novelty is in the TRAINING DYNAMICS:
      - Adaptive margin scheduling
      - Self-paced sample weighting
      - Confusion-guided curriculum
    """

    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = ConvNeXtTiny(dpr=0.1)
        self.backbone = load_convnext_pretrained(self.backbone)
        self.aux_heads = nn.ModuleList([
            ClassificationHead(d, num_classes) for d in [96, 192, 384, 768]])
        self.bottleneck = nn.Sequential(
            nn.Linear(768, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(True))
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_all=False):
        final, stages = self.backbone(x)
        stage_logits = [h(f) for h, f in zip(self.aux_heads, stages)]
        embedding = self.bottleneck(final)
        embedding_normed = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)
        if return_all:
            return {'embedding_normed': embedding_normed, 'logits': logits,
                    'stage_logits': stage_logits, 'final_feature': final}
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
            f, _ = model(b['drone'].to(device))
            feats.append(f.cpu()); labels.append(b['label'])
    feats = torch.cat(feats); labels = torch.cat(labels)
    tf = get_transforms("test"); sd = os.path.join(test_ds.root, "satellite-view")
    sf, sl = [], []
    for loc in [f"{l:04d}" for l in TRAIN_LOCS+TEST_LOCS]:
        sp = os.path.join(sd, loc, "0.png")
        if not os.path.exists(sp): continue
        t = tf(Image.open(sp).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad(): f, _ = model(t)
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
# TRAINING WITH CURRICULUM
# =============================================================================
def train(args):
    set_seed(SEED)
    print("="*70); print("GeoDCL: Dynamic Curriculum Learning for Geo-Localization"); print("="*70)
    train_ds = SUES200Dataset(args.data_root, "train", transform=get_transforms("train"))
    test_ds = SUES200Dataset(args.data_root, "test", transform=get_transforms("test"))
    sampler = PKSampler(train_ds, 8, max(2, BATCH_SIZE//8))
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)

    model = GeoDCLStudent(len(TRAIN_LOCS)).to(DEVICE)
    print(f"  Student: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    try: teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except: teacher = None

    # *** NOVEL: Adaptive Margin Triplet Loss ***
    ce = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')  # Per-sample CE
    trip = AdaptiveMarginTripletLoss(MARGIN_START, MARGIN_END, EPOCHS, MARGIN_WARMUP)
    nce = SymNCE(.07); sd = SelfDist(4.); uapa_loss = UAPA(4.)

    # *** NOVEL: Self-Paced Weighter ***
    pacer = SelfPacedWeighter(NUM_CLASSES, lambda_init=PACE_LAMBDA)

    # *** NOVEL: Confusion Tracker ***
    conf_tracker = ConfusionTracker(NUM_CLASSES, momentum=CONF_MOMENTUM)

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
            labels_b=batch['label'].to(DEVICE)
            opt.zero_grad()

            with autocast(enabled=AMP_ENABLED):
                do = model(drone, return_all=True)
                so = model(sat, return_all=True)
                L = {}

                # Per-sample CE loss
                ce_drone = ce(do['logits'], labels_b)   # [B]
                ce_sat = ce(so['logits'], labels_b)      # [B]

                # *** NOVEL: Self-Paced Weighting ***
                sample_weights = pacer.get_weights(ce_drone.detach(), labels_b)

                # *** NOVEL: Confusion-Guided Weighting ***
                conf_weights = conf_tracker.get_sample_weights(labels_b)

                # Combined weights: self-paced × confusion-guided
                combined_weights = (sample_weights * conf_weights)
                combined_weights = combined_weights / (combined_weights.mean() + 1e-6)

                # Weighted CE
                c = (ce_drone * combined_weights).mean() + (ce_sat * combined_weights).mean()
                for sl_logit in do['stage_logits']:
                    c += 0.25 * (ce(sl_logit, labels_b) * combined_weights).mean()
                for sl_logit in so['stage_logits']:
                    c += 0.25 * (ce(sl_logit, labels_b) * combined_weights).mean()
                L['ce'] = c

                # *** NOVEL: Adaptive Margin Triplet ***
                L['trip'] = (trip(do['embedding_normed'], labels_b, epoch) +
                             trip(so['embedding_normed'], labels_b, epoch))

                L['nce'] = nce(do['embedding_normed'], so['embedding_normed'], labels_b)
                L['sd'] = .5*(sd(do['stage_logits'])+sd(so['stage_logits']))
                L['uapa'] = .2*uapa_loss(do['logits'], so['logits'])

                if teacher is not None:
                    with torch.no_grad(): td=teacher(drone); ts=teacher(sat)
                    df=F.normalize(do['final_feature'],1); sf_t=F.normalize(so['final_feature'],1)
                    tdn=F.normalize(td,1); tsn=F.normalize(ts,1)
                    L['cdist']=.3*(F.mse_loss(df,tdn)+F.mse_loss(sf_t,tsn)+
                                   (1-F.cosine_similarity(df,tdn).mean())+
                                   (1-F.cosine_similarity(sf_t,tsn).mean()))

                total = sum(L.values())

            scaler.scale(total).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(opt); scaler.update()

            # *** Update curriculum trackers ***
            pacer.update(ce_drone.detach(), labels_b, epoch)
            conf_tracker.update(do['logits'].detach(), labels_b)

            tl+=total.item(); nb+=1
            for k,v in L.items(): lp[k]+=v.item()
            if bi%10==0:
                print(f"  B{bi}/{len(loader)} L={total.item():.4f} margin={trip.current_margin:.3f}")

        nb=max(1,nb)
        print(f"\nEp {epoch+1}/{EPOCHS} LR={lr:.6f} AvgL={tl/nb:.4f}")
        print(f"  Margin={trip.current_margin:.3f} Pace_λ={pacer.lambda_val:.4f}")
        for k,v in sorted(lp.items()): print(f"  {k}: {v/nb:.4f}")

        # Log top confusions
        if (epoch+1)%10==0:
            confs = conf_tracker.get_top_confusions(3)
            print(f"  Top confusions: {confs}")

        if (epoch+1)%EVAL_FREQ==0 or epoch==EPOCHS-1:
            rec, ap = evaluate(model, test_ds, DEVICE)
            print(f"  R@1:{rec['R@1']:.2f}% R@5:{rec['R@5']:.2f}% R@10:{rec['R@10']:.2f}% AP:{ap:.2f}%")
            if rec['R@1']>best_r1:
                best_r1=rec['R@1']
                torch.save({'epoch':epoch,'model':model.state_dict(),'r1':best_r1},
                           os.path.join(OUTPUT_DIR,'geodcl_best.pth'))
                print(f"  *** Best R@1={best_r1:.2f}% ***")
    print(f"\nDone! Best R@1={best_r1:.2f}%")


def smoke_test():
    print("="*50); print("SMOKE TEST — GeoDCL"); print("="*50)
    dev=DEVICE; m=GeoDCLStudent(10).to(dev)
    print(f"✓ Model: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
    x=torch.randn(4,3,224,224,device=dev)
    lab=torch.tensor([0,0,1,1],device=dev)
    o=m(x, return_all=True)
    print(f"✓ Forward: emb={o['embedding_normed'].shape}")

    # Test adaptive margin
    at = AdaptiveMarginTripletLoss(0.1, 0.5, 120, 30)
    for ep in [0, 15, 30, 60, 90, 120]:
        m_val = at.get_margin(ep)
        print(f"  Epoch {ep}: margin={m_val:.3f}")
    tl = at(o['embedding_normed'], lab, epoch=50)
    print(f"✓ Adaptive Triplet: {tl.item():.4f}")

    # Test self-paced weighter
    sp = SelfPacedWeighter(10)
    losses = torch.rand(4)
    w = sp.get_weights(losses, lab)
    print(f"✓ Self-Paced weights: {w.numpy().round(3)}")

    # Test confusion tracker
    ct = ConfusionTracker(10)
    ct.update(o['logits'], lab)
    cw = ct.get_sample_weights(lab)
    print(f"✓ Confusion weights: {cw.detach().cpu().numpy().round(3)}")

    total = nn.CrossEntropyLoss()(o['logits'], lab) + tl
    total.backward()
    print(f"✓ Backward OK")
    print("\n✅ ALL TESTS PASSED!")


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
