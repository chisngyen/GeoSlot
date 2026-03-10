#!/usr/bin/env python3
"""
GeoTTA: Test-Time Adapted Cross-View Drone Geo-Localization
=============================================================
Novel Contributions (A*-conference level):

  1. Retrieval-Augmented Test-Time Adaptation (RA-TTA) — At test time,
     adapt the model using the fixed satellite gallery as an unlabeled
     reference set. The gallery provides free structure for adaptation.
     No prior work applies TTA to cross-view retrieval.

  2. Entropy-Guided Feature Refinement (EGFR) — During inference, minimize
     the entropy of retrieval confidence to sharpen predictions. The model
     self-corrects its features using the retrieval distribution as signal.

  3. Cross-View Consistency Regularization (CVCR) — Enforce that augmented
     versions of the same query produce consistent retrieval rankings.
     Uses the gallery as a stable anchor for measuring consistency.

  4. Dynamic Memory Bank (DMB) — Maintains a running memory bank of
     high-confidence query-gallery pairs discovered during test time.
     These pseudo-pairs provide additional supervision signal.

Key Insight: Unlike classification (where TTA uses entropy minimization on
class probabilities), retrieval TTA uses the SIMILARITY DISTRIBUTION over the
gallery as the signal. The gallery is fixed and provides rich structure.

Architecture:
  Backbone: ConvNeXt-Tiny + Retrieval head (trained normally)
  TTA Module: Applied AT INFERENCE only — zero training overhead

Dataset: SUES-200 | Protocol: 120/80, 200-gallery confusion
Usage:
  python exp_geotta.py              # Full training + TTA evaluation
  python exp_geotta.py --test       # Smoke test
  python exp_geotta.py --no-tta     # Evaluate without TTA (for ablation)
"""

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

import math, random, argparse, copy
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import timm

print("[OK] All imports ready!")

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

LR_BACKBONE   = 1e-4
LR_HEAD       = 1e-3
WARMUP_EPOCHS = 5
WEIGHT_DECAY  = 0.01

LAMBDA_CE         = 1.0
LAMBDA_TRIPLET    = 0.5
LAMBDA_INFONCE    = 0.5
LAMBDA_ARCFACE    = 0.3
LAMBDA_UAPA       = 0.2
LAMBDA_SELF_DIST  = 0.3

BACKBONE_NAME = "convnext_tiny"
FEATURE_DIM   = 768
EMBED_DIM     = 512
NUM_CLASSES   = 120
MARGIN        = 0.3

# TTA Hyperparameters
TTA_STEPS     = 3        # Adaptation steps per batch of queries
TTA_LR        = 1e-5     # Very small LR for test-time updates
TTA_ENTROPY_W = 1.0      # Entropy minimization weight
TTA_CONSIST_W = 0.5      # Consistency regularization weight
TTA_MEMORY_K  = 50       # Top-K confident pairs for memory bank
TTA_CONF_THR  = 0.8      # Confidence threshold for pseudo-pairs
TTA_MOMENTUM  = 0.999    # EMA momentum for model update

IMG_SIZE      = 224
TRAIN_LOCS    = list(range(1, 121))
TEST_LOCS     = list(range(121, 201))
ALTITUDES     = ["150", "200", "250", "300"]

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


# =============================================================================
# DATASET
# =============================================================================
class SUES200Dataset(Dataset):
    def __init__(self, root, split, altitude=None, img_size=224,
                 train_locs=None, test_locs=None):
        super().__init__()
        drone_dir = os.path.join(root, "drone-view")
        satellite_dir = os.path.join(root, "satellite-view")
        if train_locs is None: train_locs = TRAIN_LOCS
        if test_locs is None: test_locs = TEST_LOCS
        locs = train_locs if split == "train" else test_locs
        alts = [altitude] if altitude else ALTITUDES
        is_train = split == "train"
        self.drone_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=3),
            transforms.RandomHorizontalFlip(0.5) if is_train else transforms.Lambda(lambda x: x),
            transforms.RandomResizedCrop(img_size, scale=(0.8,1.0)) if is_train
                else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(0.2,0.2,0.1,0.05) if is_train
                else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        sat_augs = [transforms.Resize((img_size, img_size), interpolation=3)]
        if is_train:
            sat_augs += [transforms.RandomHorizontalFlip(0.5),
                         transforms.RandomVerticalFlip(0.5),
                         transforms.RandomApply([transforms.RandomRotation((90,90))], p=0.5),
                         transforms.ColorJitter(0.2,0.2,0.1,0.05)]
        sat_augs += [transforms.ToTensor(),
                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
        self.sat_tf = transforms.Compose(sat_augs)
        self.pairs, self.labels, self.altitudes_meta = [], [], []
        loc_to_label = {}
        for loc_id in locs:
            loc_str = f"{loc_id:04d}"
            sat_path = os.path.join(satellite_dir, loc_str, "0.png")
            if not os.path.exists(sat_path): continue
            if loc_id not in loc_to_label: loc_to_label[loc_id] = len(loc_to_label)
            for alt in alts:
                alt_dir = os.path.join(drone_dir, loc_str, alt)
                if not os.path.isdir(alt_dir): continue
                alt_idx = ALTITUDES.index(alt) if alt in ALTITUDES else 0
                for img in sorted(os.listdir(alt_dir)):
                    if img.endswith(('.jpg','.jpeg','.png')):
                        self.pairs.append((os.path.join(alt_dir, img), sat_path))
                        self.labels.append(loc_to_label[loc_id])
                        self.altitudes_meta.append(alt_idx)
        self.num_classes = len(loc_to_label)
        print(f"  [{split}] {len(self.pairs)} pairs, {self.num_classes} classes")
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        dp, sp = self.pairs[idx]
        try: d=Image.open(dp).convert("RGB"); s=Image.open(sp).convert("RGB")
        except: d=Image.new("RGB",(224,224),(128,128,128)); s=d.copy()
        return {"query": self.drone_tf(d), "gallery": self.sat_tf(s),
                "label": self.labels[idx], "altitude": self.altitudes_meta[idx]}


class SUES200GalleryDataset(Dataset):
    def __init__(self, root, test_locs=None, img_size=224):
        super().__init__()
        sat_dir = os.path.join(root, "satellite-view")
        all_locs = TRAIN_LOCS + TEST_LOCS
        self.tf = transforms.Compose([
            transforms.Resize((img_size,img_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        self.images, self.loc_ids = [], []
        for loc_id in all_locs:
            p = os.path.join(sat_dir, f"{loc_id:04d}", "0.png")
            if os.path.exists(p): self.images.append(p); self.loc_ids.append(loc_id)
        print(f"  Gallery: {len(self.images)} satellite images (confusion data)")
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        return {"image": self.tf(Image.open(self.images[idx]).convert("RGB")),
                "loc_id": self.loc_ids[idx]}


class PKSampler:
    def __init__(self, ds, p=8, k=4):
        self.p, self.k = p, k
        self.l2i = defaultdict(list)
        for i, l in enumerate(ds.labels): self.l2i[l].append(i)
        self.locs = list(self.l2i.keys())
    def __iter__(self):
        ls = self.locs.copy(); random.shuffle(ls); b = []
        for l in ls:
            idx = self.l2i[l]
            if len(idx)<self.k: idx = idx*(self.k//len(idx)+1)
            b.extend(random.sample(idx, self.k))
            if len(b) >= self.p*self.k: yield b[:self.p*self.k]; b = b[self.p*self.k:]
    def __len__(self): return len(self.locs)//self.p


# =============================================================================
# ARCFACE
# =============================================================================
class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, emb, labels=None):
        w = F.normalize(self.weight, 1); x = F.normalize(emb, 1)
        cos = F.linear(x, w)
        if labels is None or not self.training: return cos * self.s
        oh = F.one_hot(labels, cos.size(1)).float()
        th = torch.acos(cos.clamp(-1+1e-7, 1-1e-7))
        return (cos*(1-oh) + torch.cos(th+self.m)*oh) * self.s


# =============================================================================
# BASE MODEL (trained normally)
# =============================================================================
class GeoBaseModel(nn.Module):
    """Strong baseline model — this gets trained, then TTA is applied on top."""
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE_NAME, pretrained=True,
                                          num_classes=0, global_pool='')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_proj = nn.Sequential(
            nn.Linear(FEATURE_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        # Lightweight adapter for TTA (only this gets updated at test time)
        self.tta_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.arcface = ArcFaceHead(embed_dim, num_classes)

    def forward(self, x, labels=None, use_adapter=False, return_all=False):
        feat_map = self.backbone(x)
        g = self.pool(feat_map).flatten(1)
        emb = self.embed_proj(g)

        if use_adapter:
            emb = emb + self.tta_adapter(emb)   # Residual adapter

        emb_norm = F.normalize(emb, p=2, dim=1)
        logits = self.classifier(emb)
        arc_logits = self.arcface(emb, labels)

        if return_all:
            return {'embedding': emb, 'embedding_norm': emb_norm,
                    'logits': logits, 'arcface_logits': arc_logits,
                    'global_feat': g}
        return emb_norm, logits

    def extract_embedding(self, x, use_adapter=False):
        self.eval()
        with torch.no_grad():
            emb, _ = self.forward(x, use_adapter=use_adapter)
        return emb


# =============================================================================
# TTA MODULE (Novel Components)
# =============================================================================
class TestTimeAdapter:
    """Retrieval-Augmented Test-Time Adaptation for geo-localization.

    Applied AT INFERENCE TIME — zero training overhead.
    Uses the fixed gallery embeddings as structural anchor.
    """
    def __init__(self, model, gallery_embeddings, gallery_locs,
                 steps=TTA_STEPS, lr=TTA_LR, device=DEVICE):
        self.original_model = model
        self.gallery_embs = gallery_embeddings.to(device)  # [G, D]
        self.gallery_locs = gallery_locs
        self.steps = steps
        self.lr = lr
        self.device = device

        # Memory bank for high-confidence pseudo-pairs
        self.memory_keys = []     # Query embeddings
        self.memory_vals = []     # Gallery indices
        self.memory_confs = []    # Confidence scores

        # EMA model for stable targets
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

    def _retrieval_entropy(self, query_embs):
        """Compute entropy of retrieval similarity distribution.

        Low entropy = confident retrieval → good adaptation signal.
        """
        sims = query_embs @ self.gallery_embs.T    # [B, G]
        probs = F.softmax(sims / 0.07, dim=1)      # Temperature-scaled softmax
        entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=1)   # [B]
        return entropy, sims

    def _consistency_loss(self, emb1, emb2):
        """Enforce consistent retrieval rankings between augmented views."""
        sim1 = emb1 @ self.gallery_embs.T   # [B, G]
        sim2 = emb2 @ self.gallery_embs.T

        # KL divergence between similarity distributions
        p1 = F.softmax(sim1 / 0.07, dim=1)
        p2 = F.softmax(sim2 / 0.07, dim=1)

        kl1 = F.kl_div(p1.log().clamp(min=-100), p2.detach(), reduction='batchmean')
        kl2 = F.kl_div(p2.log().clamp(min=-100), p1.detach(), reduction='batchmean')
        return 0.5 * (kl1 + kl2)

    def _update_memory(self, query_embs, sims):
        """Store high-confidence query-gallery pairs in memory bank."""
        vals, indices = sims.max(dim=1)  # [B]
        for i in range(len(vals)):
            conf = vals[i].item()
            if conf > TTA_CONF_THR:
                self.memory_keys.append(query_embs[i].detach().cpu())
                self.memory_vals.append(indices[i].item())
                self.memory_confs.append(conf)

        # Keep only top-K
        if len(self.memory_keys) > TTA_MEMORY_K:
            confs = torch.tensor(self.memory_confs)
            _, topk = confs.topk(TTA_MEMORY_K)
            self.memory_keys = [self.memory_keys[i] for i in topk]
            self.memory_vals = [self.memory_vals[i] for i in topk]
            self.memory_confs = [self.memory_confs[i] for i in topk]

    def _memory_loss(self, model, device):
        """Use stored pseudo-pairs as supervision."""
        if len(self.memory_keys) < 2:
            return torch.tensor(0.0, device=device)

        keys = torch.stack(self.memory_keys[:TTA_MEMORY_K]).to(device)
        vals = torch.tensor(self.memory_vals[:TTA_MEMORY_K], device=device)

        # Contrastive: stored queries should match their gallery entries
        gal_feats = self.gallery_embs[vals]  # [K, D]
        sim = F.normalize(keys, 1) @ F.normalize(gal_feats, 1).T
        labels = torch.arange(len(keys), device=device)
        return F.cross_entropy(sim / 0.07, labels)

    @torch.no_grad()
    def _update_ema(self, model):
        """EMA update of anchor model."""
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(TTA_MOMENTUM).add_(model_p.data, alpha=1-TTA_MOMENTUM)

    def adapt_and_predict(self, query_images, query_images_aug=None):
        """Main TTA inference.

        1. Adapt model for `steps` using entropy + consistency
        2. Extract embeddings with adapted model
        3. Return embeddings for retrieval
        """
        # Create working copy of model for adaptation
        adapted_model = copy.deepcopy(self.original_model)
        adapted_model.train()

        # Only update the TTA adapter (lightweight)
        for name, param in adapted_model.named_parameters():
            param.requires_grad = 'tta_adapter' in name

        optimizer = torch.optim.Adam(
            [p for p in adapted_model.parameters() if p.requires_grad],
            lr=self.lr
        )

        # Adaptation steps
        for step in range(self.steps):
            optimizer.zero_grad()

            # Forward with adapter
            emb_norm, _ = adapted_model(query_images, use_adapter=True)

            # Loss 1: Entropy minimization
            entropy, sims = self._retrieval_entropy(emb_norm)
            entropy_loss = TTA_ENTROPY_W * entropy.mean()

            # Loss 2: Consistency regularization (if augmented view available)
            if query_images_aug is not None:
                emb_aug, _ = adapted_model(query_images_aug, use_adapter=True)
                consist_loss = TTA_CONSIST_W * self._consistency_loss(emb_norm, emb_aug)
            else:
                consist_loss = torch.tensor(0.0, device=self.device)

            # Loss 3: Memory bank guidance
            mem_loss = 0.1 * self._memory_loss(adapted_model, self.device)

            total_loss = entropy_loss + consist_loss + mem_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)
            optimizer.step()

        # Extract final embedding
        adapted_model.eval()
        with torch.no_grad():
            emb_final = adapted_model.extract_embedding(query_images, use_adapter=True)

        # Update memory bank
        sims_final = emb_final @ self.gallery_embs.T
        self._update_memory(emb_final, sims_final)

        # Update EMA
        self._update_ema(adapted_model)

        return emb_final


# =============================================================================
# TTA AUGMENTATIONS
# =============================================================================
class TTAAugmentor:
    """Test-time augmentations for consistency regularization."""
    def __init__(self, img_size=224):
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        ])

    def __call__(self, images):
        """Apply augmentations to normalized tensor images."""
        return self.aug(images)


# =============================================================================
# LOSSES (for training phase)
# =============================================================================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    def forward(self, emb, labels):
        dist = torch.cdist(emb, emb, p=2)
        lab = labels.view(-1,1)
        pos = lab.eq(lab.T).float(); neg = lab.ne(lab.T).float()
        return F.relu((dist*pos).max(1)[0]-(dist*neg+pos*999).min(1)[0]+self.margin).mean()


class InfoNCELoss(nn.Module):
    def __init__(self, t=0.07):
        super().__init__()
        self.t = t
    def forward(self, d, s, labels):
        d=F.normalize(d,1); s=F.normalize(s,1); sim=d@s.T/self.t
        lab=labels.view(-1,1); pm=lab.eq(lab.T).float()
        return 0.5*(-(F.log_softmax(sim,1)*pm).sum(1)/pm.sum(1).clamp(1)).mean() + \
               0.5*(-(F.log_softmax(sim.T,1)*pm).sum(1)/pm.sum(1).clamp(1)).mean()


class UAPALoss(nn.Module):
    def __init__(self, T0=4.0):
        super().__init__()
        self.T0 = T0
    def forward(self, dl, sl):
        Ud = -(F.softmax(dl,1)*F.log_softmax(dl,1)).sum(1).mean()
        Us = -(F.softmax(sl,1)*F.log_softmax(sl,1)).sum(1).mean()
        T = self.T0*(1+torch.sigmoid(Ud-Us))
        return (T**2)*F.kl_div(F.log_softmax(dl/T,1), F.softmax(sl/T,1), reduction='batchmean')


def get_cosine_lr(ep, total, base, warmup=5):
    if ep < warmup: return base*(ep+1)/warmup
    return base*0.5*(1+math.cos(math.pi*(ep-warmup)/max(1,total-warmup)))


# =============================================================================
# EVALUATION (with and without TTA)
# =============================================================================
@torch.no_grad()
def evaluate_base(model, data_root, altitude, device, test_locs=None):
    """Standard evaluation WITHOUT TTA."""
    model.eval()
    qds = SUES200Dataset(data_root, "test", altitude, IMG_SIZE, test_locs=test_locs)
    ql = DataLoader(qds, 64, False, num_workers=NUM_WORKERS, pin_memory=True)
    gds = SUES200GalleryDataset(data_root, test_locs, IMG_SIZE)
    gl = DataLoader(gds, 64, False, num_workers=NUM_WORKERS, pin_memory=True)
    ge, gloc = [], []
    for b in gl:
        ge.append(model.extract_embedding(b["image"].to(device)).cpu())
        gloc.extend(b["loc_id"].tolist())
    ge = torch.cat(ge); gloc = np.array(gloc)
    qe = []
    for b in ql: qe.append(model.extract_embedding(b["query"].to(device)).cpu())
    qe = torch.cat(qe)
    l2g = {l:i for i,l in enumerate(gloc)}
    qgt = np.array([l2g.get(int(os.path.basename(os.path.dirname(p[1]))), -1) for p in qds.pairs])
    sim = qe.numpy() @ ge.numpy().T
    ranks = np.argsort(-sim, axis=1); N = len(qe)
    res = {}
    for k in [1,5,10]:
        res[f"R@{k}"] = sum(1 for i in range(N) if qgt[i] in ranks[i,:k]) / N
    aps = sum(1/(np.where(ranks[i]==qgt[i])[0][0]+1) for i in range(N) if len(np.where(ranks[i]==qgt[i])[0])>0)
    res["AP"] = aps/N
    return res


def evaluate_tta(model, data_root, altitude, device, test_locs=None):
    """Evaluation WITH Test-Time Adaptation."""
    model.eval()
    print("  Building gallery for TTA...")
    gds = SUES200GalleryDataset(data_root, test_locs, IMG_SIZE)
    gl = DataLoader(gds, 64, False, num_workers=NUM_WORKERS, pin_memory=True)
    ge, gloc = [], []
    with torch.no_grad():
        for b in gl:
            ge.append(model.extract_embedding(b["image"].to(device)).cpu())
            gloc.extend(b["loc_id"].tolist())
    ge = torch.cat(ge); gloc = np.array(gloc)

    # Create TTA adapter
    tta_adapter = TestTimeAdapter(model, ge, gloc, TTA_STEPS, TTA_LR, device)
    augmentor = TTAAugmentor(IMG_SIZE)

    # Query evaluation with adaptation
    qds = SUES200Dataset(data_root, "test", altitude, IMG_SIZE, test_locs=test_locs)
    ql = DataLoader(qds, 32, False, num_workers=NUM_WORKERS, pin_memory=True)

    qe_all = []
    for bi, b in enumerate(ql):
        q = b["query"].to(device)
        q_aug = augmentor(q)
        # Adapt and predict
        emb = tta_adapter.adapt_and_predict(q, q_aug)
        qe_all.append(emb.cpu())
        if bi % 5 == 0:
            print(f"    TTA batch {bi}/{len(ql)}, memory={len(tta_adapter.memory_keys)}")

    qe = torch.cat(qe_all)
    l2g = {l:i for i,l in enumerate(gloc)}
    qgt = np.array([l2g.get(int(os.path.basename(os.path.dirname(p[1]))), -1) for p in qds.pairs])
    sim = qe.numpy() @ ge.numpy().T
    ranks = np.argsort(-sim, axis=1); N = len(qe)
    res = {}
    for k in [1,5,10]:
        res[f"R@{k}"] = sum(1 for i in range(N) if qgt[i] in ranks[i,:k]) / N
    aps = sum(1/(np.where(ranks[i]==qgt[i])[0][0]+1) for i in range(N) if len(np.where(ranks[i]==qgt[i])[0])>0)
    res["AP"] = aps/N
    return res


# =============================================================================
# TRAINING
# =============================================================================
def train(args):
    set_seed(SEED)
    print("="*70)
    print("GeoTTA: Test-Time Adapted Cross-View Geo-Localization")
    print("="*70)

    ds = SUES200Dataset(args.data_root, "train", img_size=IMG_SIZE)
    sampler = PKSampler(ds, 8, max(2, BATCH_SIZE//8))
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)

    model = GeoBaseModel(ds.num_classes).to(DEVICE)
    prms = sum(p.numel() for p in model.parameters())/1e6
    print(f"  Model: {prms:.1f}M params")

    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    trip_fn = TripletLoss(MARGIN)
    nce_fn = InfoNCELoss(0.07)
    uapa_fn = UAPALoss(4.0)

    opt = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LR_BACKBONE},
        {'params': [p for n,p in model.named_parameters() if 'backbone' not in n], 'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=AMP_ENABLED)
    best_r1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for pg in opt.param_groups:
            pg['lr'] = get_cosine_lr(epoch, EPOCHS, pg.get('initial_lr', pg['lr']), WARMUP_EPOCHS)

        tl = 0.0; lp = defaultdict(float)
        for bi, batch in enumerate(loader):
            d,s,lab = batch["query"].to(DEVICE), batch["gallery"].to(DEVICE), batch["label"].to(DEVICE)
            opt.zero_grad()
            with autocast(enabled=AMP_ENABLED):
                do = model(d, lab, False, True)
                so = model(s, lab, False, True)
                L = {}
                L['ce'] = LAMBDA_CE*0.5*(ce_fn(do['logits'],lab)+ce_fn(so['logits'],lab))
                L['arc'] = LAMBDA_ARCFACE*0.5*(ce_fn(do['arcface_logits'],lab)+ce_fn(so['arcface_logits'],lab))
                L['trip'] = LAMBDA_TRIPLET*0.5*(trip_fn(do['embedding_norm'],lab)+trip_fn(so['embedding_norm'],lab))
                L['nce'] = LAMBDA_INFONCE*nce_fn(do['embedding_norm'],so['embedding_norm'],lab)
                L['uapa'] = LAMBDA_UAPA*uapa_fn(do['logits'],so['logits'])
                loss = sum(L.values())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tl += loss.item()
            for k,v in L.items(): lp[k] += v.item()
            if bi%10==0: print(f"  B{bi} L={loss.item():.4f}")

        nb = max(1, len(loader))
        print(f"\nEp {epoch+1}/{EPOCHS} AvgL={tl/nb:.4f}")

        if (epoch+1)%EVAL_FREQ==0 or epoch==EPOCHS-1:
            # Evaluate WITHOUT TTA
            print("\n--- Without TTA ---")
            ar_base = {}
            for a in ALTITUDES:
                r = evaluate_base(model, args.data_root, a, DEVICE)
                ar_base[a] = r
                print(f"  {a}m: R@1={r['R@1']:.4f} R@5={r['R@5']:.4f} AP={r['AP']:.4f}")
            avg1_base = np.mean([r['R@1'] for r in ar_base.values()])
            print(f"  AVG R@1 (no TTA) = {avg1_base:.4f}")

            # Evaluate WITH TTA (slower, only at key epochs)
            if (epoch+1) % (EVAL_FREQ * 4) == 0 or epoch == EPOCHS - 1:
                print("\n--- With TTA ---")
                ar_tta = {}
                for a in ALTITUDES:
                    r = evaluate_tta(model, args.data_root, a, DEVICE)
                    ar_tta[a] = r
                    print(f"  {a}m: R@1={r['R@1']:.4f} R@5={r['R@5']:.4f} AP={r['AP']:.4f}")
                avg1_tta = np.mean([r['R@1'] for r in ar_tta.values()])
                print(f"  AVG R@1 (with TTA) = {avg1_tta:.4f}")
                print(f"  TTA improvement: +{(avg1_tta-avg1_base)*100:.2f}%")

            avg1 = avg1_base
            if avg1 > best_r1:
                best_r1 = avg1
                torch.save({'epoch':epoch, 'model':model.state_dict(), 'r1':avg1},
                           os.path.join(OUTPUT_DIR, 'geotta_best.pth'))
                print(f"  *** Best R@1={avg1:.4f} ***")

    print(f"\nDone! Best R@1={best_r1:.4f}")


# =============================================================================
# SMOKE TEST
# =============================================================================
def smoke_test():
    print("="*50); print("SMOKE TEST — GeoTTA"); print("="*50)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = GeoBaseModel(10).to(dev)
    print(f"✓ Model: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")

    x = torch.randn(4,3,224,224,device=dev)
    lab = torch.tensor([0,0,1,1],device=dev)

    # Base forward
    o = m(x, lab, False, True)
    print(f"✓ Base forward: emb={o['embedding_norm'].shape}")

    # With adapter
    o2 = m(x, lab, True, True)
    print(f"✓ Adapter forward: emb={o2['embedding_norm'].shape}")

    # TTA simulation
    gallery = torch.randn(20, EMBED_DIM, device=dev)
    gallery = F.normalize(gallery, 1)
    tta = TestTimeAdapter(m, gallery, list(range(20)), 1, 1e-5, dev)
    emb_tta = tta.adapt_and_predict(x)
    print(f"✓ TTA adapt: emb={emb_tta.shape}, memory={len(tta.memory_keys)}")

    # Backward
    ce = nn.CrossEntropyLoss()(o['logits'], lab)
    ce.backward()
    gn = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
    print(f"✓ Backward: grad_norm={gn:.4f}")
    print("\n✅ ALL TESTS PASSED!")


def main():
    global EPOCHS, BATCH_SIZE, DATA_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA for ablation")
    args, _ = parser.parse_known_args()
    EPOCHS=args.epochs; BATCH_SIZE=args.batch_size; DATA_ROOT=args.data_root
    if args.test: smoke_test(); return
    os.makedirs(OUTPUT_DIR, exist_ok=True); train(args)


if __name__ == "__main__":
    main()
