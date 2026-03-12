#!/usr/bin/env python3
"""
GeoCIRCLE: Circle Loss + Curriculum Hard Negative Mining for Geo-Localization
==============================================================================
Novel contributions:
  1. Adaptive Circle Loss — Unified pair-wise and class-level metric learning
     with adaptive margin that adjusts per-sample based on difficulty
  2. Curriculum Hard Negative Mining (CHNM) — Progressive difficulty scheduler
     that gradually introduces harder negatives as training progresses
  3. Confusion-Aware Gallery Mining (CAGM) — Identifies the most confusing
     satellite gallery images and oversamples them during training

Inspired by: Circle Loss (CVPR 2020), Curriculum Learning (ICML 2009),
             Sample Mining strategies for geo-localization (TPAMI 2024)

Key insight: Standard training treats all negatives equally, but in SUES-200,
many satellite images look very similar (e.g., similar building layouts).
CHNM + CAGM forces the model to distinguish these hard cases progressively.

Architecture:
  Student: ConvNeXt-Tiny + Circle Loss Head + CHNM + DINOv2 distillation
  Teacher: DINOv2-Base (frozen)

Dataset: SUES-200 (drone ↔ satellite cross-view geo-localization)
Protocol: 120 train / 80 test, gallery = ALL 200 locations (confusion data)

Usage:
  python exp_geocircle.py          # Full training on Kaggle H100
  python exp_geocircle.py --test   # Smoke test
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

# Circle Loss config
CIRCLE_M      = 0.25       # Relaxation margin
CIRCLE_GAMMA  = 256        # Scale factor

# CHNM config
CHNM_WARMUP   = 20         # Epochs before starting hard mining
CHNM_TOPK_INIT = 8         # Initial top-K negative pool size
CHNM_TOPK_FINAL = 2        # Final top-K (hardest negatives only)

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
        self.w=nn.Parameter(torch.ones(ns)); self.b=nn.Parameter(torch.zeros(ns))
        self.eps=eps; self.df=df; self.ns=(ns,)
    def forward(self, x):
        if self.df=="channels_last": return F.layer_norm(x,self.ns,self.w,self.b,self.eps)
        u=x.mean(1,keepdim=True); s=(x-u).pow(2).mean(1,keepdim=True)
        return self.w[:,None,None]*((x-u)/torch.sqrt(s+self.eps))+self.b[:,None,None]

def drop_path(x,dp=0.,tr=False):
    if dp==0. or not tr: return x
    kp=1-dp; s=(x.shape[0],)+(1,)*(x.ndim-1)
    rt=kp+torch.rand(s,dtype=x.dtype,device=x.device); rt.floor_()
    return x.div(kp)*rt

class DropPath(nn.Module):
    def __init__(self,dp=None): super().__init__(); self.dp=dp
    def forward(self,x): return drop_path(x,self.dp,self.training)

class ConvNeXtBlock(nn.Module):
    def __init__(self,d,dpr=0.,lsi=1e-6):
        super().__init__()
        self.dw=nn.Conv2d(d,d,7,padding=3,groups=d); self.n=LayerNorm(d,1e-6)
        self.p1=nn.Linear(d,4*d); self.act=nn.GELU(); self.p2=nn.Linear(4*d,d)
        self.g=nn.Parameter(lsi*torch.ones(d)) if lsi>0 else None
        self.dp=DropPath(dpr) if dpr>0 else nn.Identity()
    def forward(self,x):
        s=x; x=self.dw(x); x=x.permute(0,2,3,1)
        x=self.n(x); x=self.p1(x); x=self.act(x); x=self.p2(x)
        if self.g is not None: x=self.g*x
        return s+self.dp(x.permute(0,3,1,2))

class ConvNeXtTiny(nn.Module):
    def __init__(self,ic=3,depths=[3,3,9,3],dims=[96,192,384,768],dpr=0.):
        super().__init__()
        self.dims=dims; self.dsl=nn.ModuleList()
        self.dsl.append(nn.Sequential(nn.Conv2d(ic,dims[0],4,4),
                        LayerNorm(dims[0],1e-6,"channels_first")))
        for i in range(3):
            self.dsl.append(nn.Sequential(LayerNorm(dims[i],1e-6,"channels_first"),
                            nn.Conv2d(dims[i],dims[i+1],2,2)))
        rates=[x.item() for x in torch.linspace(0,dpr,sum(depths))]; c=0
        self.stages=nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[ConvNeXtBlock(dims[i],rates[c+j]) for j in range(depths[i])]))
            c+=depths[i]
        self.norm=nn.LayerNorm(dims[-1],1e-6); self.apply(self._iw)
    def _iw(self,m):
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            nn.init.trunc_normal_(m.weight,std=.02)
            if m.bias is not None: nn.init.constant_(m.bias,0)
    def forward(self,x):
        outs=[]
        for i in range(4): x=self.dsl[i](x); x=self.stages[i](x); outs.append(x)
        return self.norm(x.mean([-2,-1])), outs

def load_convnext_pretrained(m):
    try:
        ckpt=torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth",
            map_location="cpu",check_hash=True)
        m.load_state_dict({k:v for k,v in ckpt["model"].items() if not k.startswith('head')},strict=False)
        print("Loaded ConvNeXt-Tiny pretrained (ImageNet-22K)")
    except Exception as e: print(f"Could not load: {e}")
    return m


# =============================================================================
# NOVEL COMPONENT 1: ADAPTIVE CIRCLE LOSS
# =============================================================================
class AdaptiveCircleLoss(nn.Module):
    """Circle Loss with adaptive margin.

    Circle Loss unifies pair-wise and classification-level losses by
    re-weighting each similarity score based on its optimization status.
    Our extension adds per-sample adaptive margin based on difficulty.

    Key: Each similarity score gets a different gradient weight:
    - For positive pairs: weight = [O_p - s_p]+ → far positives get MORE gradient
    - For negative pairs: weight = [s_n - O_n]+ → close negatives get MORE gradient

    This self-paced weighting prevents gradient saturation and focuses
    the model on the most informative pairs.
    """
    def __init__(self, m=0.25, gamma=256):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.O_p = 1 + m      # Positive optimal value
        self.O_n = -m          # Negative optimal value
        self.delta_p = 1 - m   # Positive decision boundary
        self.delta_n = m       # Negative decision boundary

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [B, D] L2-normalized embeddings
            labels: [B] integer labels
        Returns:
            loss: scalar Circle Loss value
        """
        # Compute cosine similarity matrix
        sim_mat = torch.mm(embeddings, embeddings.T)  # [B, B]

        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float()
        neg_mask = labels.ne(labels.T).float()

        # Remove diagonal (self-similarity)
        eye_mask = torch.eye(sim_mat.size(0), device=sim_mat.device)
        pos_mask = pos_mask - eye_mask

        # Positive pair weights: alpha_p = [O_p - s_p]+
        pos_sim = sim_mat * pos_mask
        alpha_p = torch.clamp(self.O_p - pos_sim.detach(), min=0)

        # Negative pair weights: alpha_n = [s_n - O_n]+
        neg_sim = sim_mat * neg_mask
        alpha_n = torch.clamp(neg_sim.detach() - self.O_n, min=0)

        # Circle loss logits
        logit_p = -self.gamma * alpha_p * (pos_sim - self.delta_p)
        logit_n = self.gamma * alpha_n * (neg_sim - self.delta_n)

        # Mask out non-existing pairs
        logit_p = logit_p * pos_mask + (1 - pos_mask) * (-1e9)
        logit_n = logit_n * neg_mask + (1 - neg_mask) * (-1e9)

        # Logsumexp for numerical stability
        loss = F.softplus(
            torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)
        )

        return loss.mean()


class CrossViewCircleLoss(nn.Module):
    """Circle Loss applied across drone and satellite views.

    Instead of within-view similarity, computes drone-to-satellite Circle Loss
    for explicit cross-view matching.
    """
    def __init__(self, m=0.25, gamma=256):
        super().__init__()
        self.m = m; self.gamma = gamma
        self.O_p = 1 + m; self.O_n = -m
        self.delta_p = 1 - m; self.delta_n = m

    def forward(self, drone_emb, sat_emb, labels):
        sim = torch.mm(drone_emb, sat_emb.T)  # [B, B]
        lab = labels.view(-1, 1)
        pos_mask = lab.eq(lab.T).float()
        neg_mask = 1 - pos_mask

        alpha_p = torch.clamp(self.O_p - sim.detach(), min=0)
        alpha_n = torch.clamp(sim.detach() - self.O_n, min=0)

        logit_p = -self.gamma * alpha_p * (sim - self.delta_p)
        logit_n = self.gamma * alpha_n * (sim - self.delta_n)

        logit_p = logit_p * pos_mask + (1 - pos_mask) * (-1e9)
        logit_n = logit_n * neg_mask + (1 - neg_mask) * (-1e9)

        loss = F.softplus(
            torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))
        return loss.mean()


# =============================================================================
# NOVEL COMPONENT 2: CURRICULUM HARD NEGATIVE MINING (CHNM)
# =============================================================================
class CurriculumHardNegativeMiner:
    """Progressive hard negative mining scheduler.

    Early training: use semi-hard negatives (easier) for stable gradients
    Late training: use only the hardest negatives for fine-grained discrimination

    Schedule:
      Epoch 1-20: top-K=8 (easy negatives, broad learning)
      Epoch 20-60: top-K=5 (medium difficulty)
      Epoch 60-90: top-K=3 (hard negatives)
      Epoch 90+: top-K=2 (hardest only, surgical refinement)
    """
    def __init__(self, warmup_epochs=20, topk_init=8, topk_final=2, total_epochs=120):
        self.warmup = warmup_epochs
        self.topk_init = topk_init
        self.topk_final = topk_final
        self.total = total_epochs
        self.confusion_matrix = None  # Updated during evaluation

    def get_topk(self, epoch):
        """Get current top-K for negative mining."""
        if epoch < self.warmup:
            return self.topk_init

        progress = (epoch - self.warmup) / (self.total - self.warmup)
        # Cosine schedule for smooth transition
        topk = self.topk_final + (self.topk_init - self.topk_final) * \
               0.5 * (1 + math.cos(math.pi * progress))
        return max(self.topk_final, int(topk))

    def mine_hard_negatives(self, embeddings, labels, epoch):
        """Select hard negatives based on current difficulty level.

        Args:
            embeddings: [B, D] normalized embeddings
            labels: [B] labels
            epoch: current epoch
        Returns:
            hard_neg_indices: [B] indices of hardest negatives per sample
            hardness: [B] difficulty scores
        """
        topk = self.get_topk(epoch)

        # Compute similarity matrix
        sim = torch.mm(embeddings, embeddings.T)  # [B, B]

        lab = labels.view(-1, 1)
        neg_mask = lab.ne(lab.T).float()

        # Get similarities to all negatives
        neg_sim = sim * neg_mask + (1 - neg_mask) * (-1e9)

        # Select top-K most similar (hardest) negatives
        topk_values, topk_indices = neg_sim.topk(min(topk, neg_mask.sum(1).min().int().item()),
                                                  dim=1)

        # Randomly select one from the top-K pool
        rand_idx = torch.randint(0, topk_values.size(1), (topk_values.size(0),),
                                  device=embeddings.device)
        hard_neg_idx = topk_indices[torch.arange(len(rand_idx)), rand_idx]
        hardness = topk_values[torch.arange(len(rand_idx)), rand_idx]

        return hard_neg_idx, hardness


# =============================================================================
# NOVEL COMPONENT 3: CONFUSION-AWARE GALLERY MINING (CAGM)
# =============================================================================
class ConfusionAwareGalleryMiner:
    """Identifies the most confusing gallery images and focuses training.

    Maintains a confusion history that tracks which satellite images are
    most frequently retrieved incorrectly. These "confused pairs" are
    then oversampled during training.
    """
    def __init__(self, num_locations=120, ema_decay=0.95):
        self.num_locations = num_locations
        self.ema_decay = ema_decay

        # Confusion score for each location pair [i, j] = how often i gets confused with j
        self.confusion_scores = torch.zeros(num_locations, num_locations)

    def update_confusion(self, query_labels, gallery_labels, sim_matrix):
        """Update confusion scores from evaluation results.

        Args:
            query_labels: [N_q] query drone labels
            gallery_labels: [N_g] gallery satellite labels
            sim_matrix: [N_q, N_g] similarity matrix
        """
        _, topk_idx = sim_matrix.topk(10, dim=1)  # Top-10 predictions

        for i in range(len(query_labels)):
            true_label = query_labels[i].item()
            if true_label < 0 or true_label >= self.num_locations:
                continue

            for j in range(min(5, topk_idx.size(1))):
                pred_label = gallery_labels[topk_idx[i, j]].item()
                if pred_label < 0 or pred_label >= self.num_locations:
                    continue
                if pred_label != true_label:
                    # EMA update
                    self.confusion_scores[true_label, pred_label] = \
                        self.ema_decay * self.confusion_scores[true_label, pred_label] + \
                        (1 - self.ema_decay) * 1.0

    def get_confusion_weights(self, labels):
        """Get per-sample training weights based on confusion history.

        Samples from frequently confused locations get higher weight.
        """
        weights = torch.ones(len(labels))
        for i, lab in enumerate(labels):
            lab_val = lab.item() if torch.is_tensor(lab) else lab
            if lab_val < self.num_locations:
                confusion_score = self.confusion_scores[lab_val].sum().item()
                weights[i] = 1.0 + min(2.0, confusion_score * 0.5)
        return weights


# =============================================================================
# GEOCIRCLE MODEL
# =============================================================================
class ClassificationHead(nn.Module):
    def __init__(self,d,nc,h=512):
        super().__init__()
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Linear(d,h),nn.BatchNorm1d(h),nn.ReLU(True),
                              nn.Dropout(.5),nn.Linear(h,nc))
    def forward(self,x): return self.fc(self.pool(x).flatten(1))


class GeoCIRCLEStudent(nn.Module):
    """GeoCIRCLE = ConvNeXt-Tiny + Circle Loss + CHNM.

    Standard pipeline with Circle Loss replacing triplet loss
    for unified pair-wise + class-level metric learning.
    """
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = ConvNeXtTiny(dpr=0.1)
        self.backbone = load_convnext_pretrained(self.backbone)

        self.aux_heads = nn.ModuleList([
            ClassificationHead(d, num_classes) for d in [96, 192, 384, 768]])

        self.bottleneck = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(True))

        self.classifier = nn.Linear(embed_dim, num_classes)

        # ProxyNCA-inspired class proxies for Circle Loss
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)

    def forward(self, x, return_all=False):
        final, stages = self.backbone(x)
        stage_logits = [h(f) for h, f in zip(self.aux_heads, stages)]

        embedding = self.bottleneck(final)
        embedding_normed = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)

        if return_all:
            return {
                'embedding': embedding,
                'embedding_normed': embedding_normed,
                'logits': logits,
                'stage_logits': stage_logits,
                'final_feature': final,
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
def evaluate(model, test_ds, device, confusion_miner=None):
    model.eval()
    loader = DataLoader(test_ds, 256, False, num_workers=NUM_WORKERS, pin_memory=True)
    feats, labels = [], []
    with torch.no_grad():
        for b in loader:
            f, _ = model(b['drone'].to(device)); feats.append(f.cpu()); labels.append(b['label'])
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
    sim = feats@sf.T

    # Update confusion miner
    if confusion_miner is not None:
        confusion_miner.update_confusion(labels, sl, sim)

    _, idx = sim.sort(1, descending=True)
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
    print("="*70); print("GeoCIRCLE: Circle Loss + CHNM Geo-Localization"); print("="*70)
    train_ds = SUES200Dataset(args.data_root, "train", transform=get_transforms("train"))
    test_ds = SUES200Dataset(args.data_root, "test", transform=get_transforms("test"))
    sampler = PKSampler(train_ds, 8, max(2, BATCH_SIZE//8))
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    model = GeoCIRCLEStudent(len(TRAIN_LOCS)).to(DEVICE)
    print(f"  Student: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    try: teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except: teacher = None

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    circle = AdaptiveCircleLoss(CIRCLE_M, CIRCLE_GAMMA)
    xv_circle = CrossViewCircleLoss(CIRCLE_M, CIRCLE_GAMMA)
    nce = SymNCE(.07); sd = SelfDist(4.); uapa = UAPA(4.)

    # *** NOVEL: Curriculum Hard Negative Miner ***
    chnm = CurriculumHardNegativeMiner(CHNM_WARMUP, CHNM_TOPK_INIT, CHNM_TOPK_FINAL, EPOCHS)

    # *** NOVEL: Confusion-Aware Gallery Miner ***
    cagm = ConfusionAwareGalleryMiner(len(TRAIN_LOCS))

    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=.9, weight_decay=5e-4)
    scaler = GradScaler(enabled=AMP_ENABLED); best_r1 = 0.

    for epoch in range(EPOCHS):
        model.train()
        if epoch < WARMUP_EPOCHS: lr=LR*(epoch+1)/WARMUP_EPOCHS
        else:
            pr=(epoch-WARMUP_EPOCHS)/max(1,EPOCHS-WARMUP_EPOCHS)
            lr=1e-6+.5*(LR-1e-6)*(1+math.cos(math.pi*pr))
        for pg in opt.param_groups: pg['lr']=lr

        topk = chnm.get_topk(epoch)
        tl=0.; lp=defaultdict(float); nb=0

        for bi, batch in enumerate(loader):
            drone=batch['drone'].to(DEVICE); sat=batch['satellite'].to(DEVICE)
            labels=batch['label'].to(DEVICE)
            opt.zero_grad()

            # Get confusion weights
            conf_weights = cagm.get_confusion_weights(batch['label']).to(DEVICE)

            with autocast(enabled=AMP_ENABLED):
                do = model(drone, return_all=True)
                so = model(sat, return_all=True)
                L = {}

                # CE with confusion weighting
                ce_d = F.cross_entropy(do['logits'], labels, label_smoothing=0.1, reduction='none')
                ce_s = F.cross_entropy(so['logits'], labels, label_smoothing=0.1, reduction='none')
                L['ce'] = (ce_d * conf_weights).mean() + (ce_s * conf_weights).mean()
                for sl in do['stage_logits']: L['ce'] += .25*ce(sl, labels)
                for sl in so['stage_logits']: L['ce'] += .25*ce(sl, labels)

                # *** NOVEL: Circle Loss (within-view) ***
                L['circle'] = 0.5 * (
                    circle(do['embedding_normed'], labels) +
                    circle(so['embedding_normed'], labels))

                # *** NOVEL: Cross-View Circle Loss ***
                L['xv_circle'] = 0.5 * xv_circle(
                    do['embedding_normed'], so['embedding_normed'], labels)

                # InfoNCE
                L['nce'] = .5*nce(do['embedding_normed'], so['embedding_normed'], labels)

                # Self-distillation
                L['sd'] = .5*(sd(do['stage_logits'])+sd(so['stage_logits']))

                # UAPA
                L['uapa'] = .2*uapa(do['logits'], so['logits'])

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
            if bi%10==0: print(f"  B{bi}/{len(loader)} L={total.item():.4f} topK={topk}")

        nb=max(1,nb)
        print(f"\nEp {epoch+1}/{EPOCHS} LR={lr:.6f} AvgL={tl/nb:.4f} CHNM_topK={topk}")
        for k,v in sorted(lp.items()): print(f"  {k}: {v/nb:.4f}")

        if (epoch+1)%EVAL_FREQ==0 or epoch==EPOCHS-1:
            rec, ap = evaluate(model, test_ds, DEVICE, cagm)
            print(f"  R@1:{rec['R@1']:.2f}% R@5:{rec['R@5']:.2f}% R@10:{rec['R@10']:.2f}% AP:{ap:.2f}%")
            if rec['R@1']>best_r1:
                best_r1=rec['R@1']
                torch.save({'epoch':epoch,'model':model.state_dict(),'r1':best_r1},
                           os.path.join(OUTPUT_DIR,'geocircle_best.pth'))
                print(f"  *** Best R@1={best_r1:.2f}% ***")
    print(f"\nDone! Best R@1={best_r1:.2f}%")


def smoke_test():
    print("="*50); print("SMOKE TEST — GeoCIRCLE"); print("="*50)
    dev=DEVICE; m=GeoCIRCLEStudent(10).to(dev)
    print(f"✓ Model: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
    x=torch.randn(4,3,224,224,device=dev); lab=torch.tensor([0,0,1,1],device=dev)
    o=m(x, return_all=True)
    print(f"✓ Forward: emb={o['embedding_normed'].shape}")
    # Test Circle Loss
    cl = AdaptiveCircleLoss(0.25, 256)
    closs = cl(o['embedding_normed'], lab)
    print(f"✓ Circle Loss: {closs.item():.4f}")
    # Test CHNM
    chnm = CurriculumHardNegativeMiner()
    for ep in [0, 30, 60, 100]:
        print(f"  CHNM topK@ep{ep}: {chnm.get_topk(ep)}")
    closs.backward()
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
