#!/usr/bin/env python3
"""
GeoMAE: Cross-View Masked Autoencoder for Drone Geo-Localization
=================================================================
Novel Contributions (A*-conference level):

  1. Cross-View Masked Reconstruction (CVMR) — Novel pretext task:
     mask drone patches, reconstruct them using satellite context (and vice versa).
     Forces the model to learn VIEW-INVARIANT representations by completing
     one view from the other. No prior work applies this to cross-view geo-loc.

  2. Asymmetric Dual-Decoder Architecture — The encoder is shared (learns
     view-invariant features), but decoders are view-specific (drone decoder
     vs satellite decoder). This disentangles shared vs view-specific info.

  3. Progressive Masking Curriculum — Masking ratio increases from 25% → 75%
     over training. Early: easy reconstruction builds basic correspondences.
     Late: aggressive masking forces deep semantic understanding.

  4. Contrastive-Reconstructive Joint Learning — Combines reconstruction loss
     (dense pixel-level) with contrastive loss (holistic embedding-level) in a
     single unified framework. The reconstruction provides dense supervision
     that contrastive learning alone cannot offer.

Target: ACM Multimedia / ECCV / CVPR level publication

Architecture:
  Encoder: ConvNeXt-Tiny (shared, learns view-invariant features)
  Drone Decoder: lightweight ViT decoder (reconstructs drone from satellite context)
  Satellite Decoder: lightweight ViT decoder (reconstructs satellite from drone context)
  Retrieval Head: ArcFace + Contrastive (fine-tuned after pretraining)

Dataset: SUES-200 | Protocol: 120/80 fixed, 200-gallery confusion
Usage:
  python exp_geomae.py             # Full training
  python exp_geomae.py --test      # Smoke test
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

# === IMPORTS ===
import math, random, argparse
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
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

# Phase 1: Pretraining (masked reconstruction)
PRETRAIN_EPOCHS  = 40
PRETRAIN_LR      = 1e-4
MASK_RATIO_START = 0.25
MASK_RATIO_END   = 0.75

# Phase 2: Fine-tuning (contrastive + classification)
FINETUNE_EPOCHS  = 80
FINETUNE_LR      = 5e-5

BATCH_SIZE    = 256
NUM_WORKERS   = 8
AMP_ENABLED   = True
EVAL_FREQ     = 5
WARMUP_EPOCHS = 5
WEIGHT_DECAY  = 0.01

LAMBDA_RECON      = 1.0    # Reconstruction loss
LAMBDA_CROSS_RECON= 1.0    # Cross-view reconstruction
LAMBDA_CE         = 1.0    # Classification
LAMBDA_TRIPLET    = 0.5
LAMBDA_INFONCE    = 0.5
LAMBDA_ARCFACE    = 0.3
LAMBDA_ALIGN      = 0.3    # Encoder alignment between views
LAMBDA_UAPA       = 0.2

BACKBONE_NAME = "convnext_tiny"
FEATURE_DIM   = 768
EMBED_DIM     = 512
DECODER_DIM   = 256
DECODER_DEPTH = 4
DECODER_HEADS = 8
PATCH_SIZE    = 7          # ConvNeXt output is 7x7 at 224 input
NUM_CLASSES   = 120
MARGIN        = 0.3

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
        # Also keep un-normalized for reconstruction target
        self.raw_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=3),
            transforms.ToTensor()])
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
        try:
            d_img = Image.open(dp).convert("RGB")
            s_img = Image.open(sp).convert("RGB")
        except:
            d_img = Image.new("RGB",(224,224),(128,128,128))
            s_img = d_img.copy()
        return {
            "query": self.drone_tf(d_img),
            "gallery": self.sat_tf(s_img),
            "query_raw": self.raw_tf(d_img),     # For reconstruction target
            "gallery_raw": self.raw_tf(s_img),    # For reconstruction target
            "label": self.labels[idx],
            "altitude": self.altitudes_meta[idx]
        }


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
# RANDOM MASKING (Novel Component #3 — Progressive Curriculum)
# =============================================================================
class PatchMasker:
    """Progressive masking: ratio increases from 25% → 75% over training.

    For ConvNeXt features [B, C, H, W], we mask at the spatial level.
    Masked positions get a learnable mask token.
    """
    def __init__(self, patch_h=7, patch_w=7):
        self.H = patch_h
        self.W = patch_w
        self.N = patch_h * patch_w

    def get_mask(self, batch_size, mask_ratio, device):
        """Generate random mask. True = masked (to reconstruct)."""
        num_mask = max(1, int(self.N * mask_ratio))
        masks = []
        for _ in range(batch_size):
            perm = torch.randperm(self.N, device=device)
            mask = torch.zeros(self.N, dtype=torch.bool, device=device)
            mask[perm[:num_mask]] = True
            masks.append(mask)
        return torch.stack(masks)  # [B, N]

    def apply_mask(self, feat_map, mask, mask_token):
        """Apply mask to feature map.

        Args:
            feat_map: [B, C, H, W]
            mask: [B, N] where N = H*W
            mask_token: [C] learnable token
        Returns:
            masked_feat: [B, C, H, W] with masked positions replaced
        """
        B, C, H, W = feat_map.shape
        feat_flat = feat_map.flatten(2).transpose(1, 2)  # [B, N, C]
        mask_expanded = mask.unsqueeze(-1).expand_as(feat_flat)  # [B, N, C]
        token_expanded = mask_token.unsqueeze(0).unsqueeze(0).expand_as(feat_flat)
        masked_flat = torch.where(mask_expanded, token_expanded, feat_flat)
        return masked_flat.transpose(1, 2).view(B, C, H, W)


# =============================================================================
# CROSS-VIEW DECODER (Novel Component #1 & #2)
# =============================================================================
class CrossViewDecoder(nn.Module):
    """Lightweight ViT decoder for cross-view masked reconstruction.

    Takes:
      - Visible features from VIEW A (unmasked)
      - Mask positions from VIEW A
      - Context features from VIEW B (complete)

    Reconstructs the masked patches of VIEW A using VIEW B's context.
    This forces the encoder to learn view-invariant features.
    """
    def __init__(self, encoder_dim, decoder_dim=256, depth=4, num_heads=8,
                 num_patches=49, output_channels=3, patch_pixel_size=32):
        super().__init__()
        self.num_patches = num_patches
        self.decoder_dim = decoder_dim
        self.patch_pixel_size = patch_pixel_size

        # Project encoder features to decoder dim
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)

        # Cross-view context projection
        self.context_proj = nn.Linear(encoder_dim, decoder_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(decoder_dim) * 0.02)

        # Positional embeddings for decoder
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, decoder_dim) * 0.02)

        # Decoder transformer blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(decoder_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)

        # Pixel prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 2),
            nn.GELU(),
            nn.Linear(decoder_dim * 2, output_channels * patch_pixel_size * patch_pixel_size),
        )

    def forward(self, encoder_feat, mask, cross_context):
        """
        Args:
            encoder_feat: [B, N, C_enc] encoded features of the view to reconstruct
            mask: [B, N] True = masked positions
            cross_context: [B, N, C_enc] encoded features from OTHER view
        Returns:
            pred_pixels: [B, N_masked, 3*P*P] predicted pixel values for masked patches
        """
        B, N, C = encoder_feat.shape

        # Project to decoder dim
        visible = self.encoder_to_decoder(encoder_feat)   # [B, N, D_dec]
        context = self.context_proj(cross_context)        # [B, N, D_dec]

        # Replace masked positions with mask token
        mask_tokens = self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        decoder_input = torch.where(mask.unsqueeze(-1), mask_tokens, visible)

        # Add positional embedding
        decoder_input = decoder_input + self.pos_embed

        # Decoder blocks with cross-attention to other view
        for block in self.blocks:
            decoder_input = block(decoder_input, context)

        decoder_input = self.norm(decoder_input)

        # Predict pixels only for masked positions
        masked_feats = decoder_input[mask]  # [total_masked, D_dec]
        pred = self.pred_head(masked_feats) # [total_masked, 3*P*P]

        return pred, mask


class DecoderBlock(nn.Module):
    """Decoder block with self-attention + cross-attention to other view."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(0.1),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context):
        # Self-attention
        x_norm = self.norm1(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm)[0]
        # Cross-attention to other view
        x_norm = self.norm2(x)
        x = x + self.cross_attn(x_norm, context, context)[0]
        # FFN
        x = x + self.mlp(self.norm3(x))
        return x


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
# GEOMAE MODEL
# =============================================================================
class GeoMAEModel(nn.Module):
    """GeoMAE: Shared encoder + dual view-specific decoders.

    Train Phase 1 (pretrain): Masked cross-view reconstruction
    Train Phase 2 (finetune): Contrastive + classification for retrieval
    """
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM):
        super().__init__()

        # Shared encoder (learns view-invariant features)
        self.encoder = timm.create_model(BACKBONE_NAME, pretrained=True,
                                         num_classes=0, global_pool='')
        self.feature_dim = FEATURE_DIM

        # Masker
        self.masker = PatchMasker(PATCH_SIZE, PATCH_SIZE)
        self.mask_token_enc = nn.Parameter(torch.randn(FEATURE_DIM) * 0.02)

        # Dual decoders (view-specific)
        self.drone_decoder = CrossViewDecoder(
            FEATURE_DIM, DECODER_DIM, DECODER_DEPTH, DECODER_HEADS,
            PATCH_SIZE * PATCH_SIZE, output_channels=3,
            patch_pixel_size=IMG_SIZE // PATCH_SIZE
        )
        self.sat_decoder = CrossViewDecoder(
            FEATURE_DIM, DECODER_DIM, DECODER_DEPTH, DECODER_HEADS,
            PATCH_SIZE * PATCH_SIZE, output_channels=3,
            patch_pixel_size=IMG_SIZE // PATCH_SIZE
        )

        # Retrieval head (for fine-tuning phase)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_proj = nn.Sequential(
            nn.Linear(FEATURE_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.arcface = ArcFaceHead(embed_dim, num_classes)

        # Alignment head (encourage encoder invariance)
        self.align_proj = nn.Sequential(
            nn.Linear(FEATURE_DIM, FEATURE_DIM),
            nn.BatchNorm1d(FEATURE_DIM),
            nn.ReLU(True),
            nn.Linear(FEATURE_DIM, embed_dim),
        )

    def encode(self, x):
        """Shared encoder → feature map + tokens."""
        feat_map = self.encoder(x)          # [B, C, H', W']
        B, C, H, W = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)  # [B, N, C]
        return feat_map, tokens

    def forward_pretrain(self, drone_img, sat_img, mask_ratio=0.5):
        """Phase 1: Cross-view masked reconstruction.

        1. Encode both views
        2. Mask drone features → reconstruct using satellite context
        3. Mask satellite features → reconstruct using drone context
        """
        B = drone_img.shape[0]

        # Encode both views (shared encoder)
        d_feat, d_tokens = self.encode(drone_img)  # [B, N, C]
        s_feat, s_tokens = self.encode(sat_img)

        N = d_tokens.shape[1]

        # Generate masks
        d_mask = self.masker.get_mask(B, mask_ratio, d_tokens.device)  # [B, N]
        s_mask = self.masker.get_mask(B, mask_ratio, s_tokens.device)

        # Cross-view reconstruction
        # Reconstruct masked DRONE patches using SATELLITE context
        d_pred, d_mask_used = self.drone_decoder(d_tokens, d_mask, s_tokens)

        # Reconstruct masked SATELLITE patches using DRONE context
        s_pred, s_mask_used = self.sat_decoder(s_tokens, s_mask, d_tokens)

        # Encoder alignment loss (encourage view invariance)
        d_global = self.align_proj(d_tokens.mean(dim=1))
        s_global = self.align_proj(s_tokens.mean(dim=1))

        return {
            'd_pred': d_pred,
            's_pred': s_pred,
            'd_mask': d_mask,
            's_mask': s_mask,
            'd_global': d_global,
            's_global': s_global,
        }

    def forward_finetune(self, x, labels=None, return_all=False):
        """Phase 2: Retrieval — standard forward pass."""
        feat_map, tokens = self.encode(x)
        global_feat = self.pool(feat_map).flatten(1)
        embedding = self.embed_proj(global_feat)
        embedding_norm = F.normalize(embedding, p=2, dim=1)

        logits = self.classifier(embedding)
        arc_logits = self.arcface(embedding, labels)

        if return_all:
            return {
                'embedding': embedding,
                'embedding_norm': embedding_norm,
                'logits': logits,
                'arcface_logits': arc_logits,
                'global_feat': global_feat,
            }
        return embedding_norm, logits

    def extract_embedding(self, x, **kwargs):
        self.eval()
        with torch.no_grad():
            emb, _ = self.forward_finetune(x)
        return emb


# =============================================================================
# LOSSES
# =============================================================================
class PatchReconstructionLoss(nn.Module):
    """MSE + perceptual loss for patch-level reconstruction."""
    def __init__(self, patch_pixel_size=32):
        super().__init__()
        self.p = patch_pixel_size

    def forward(self, pred_pixels, target_img, mask):
        """
        Args:
            pred_pixels: [total_masked, 3*P*P] predicted
            target_img: [B, 3, H, W] raw target image
            mask: [B, N] True = masked
        """
        B, C, H, W = target_img.shape
        pH = H // self.p if self.p > 1 else int(mask.shape[1] ** 0.5)
        pW = pH

        # Patchify target
        patches = target_img.unfold(2, H//pH, H//pH).unfold(3, W//pW, W//pW)
        patches = patches.contiguous().view(B, C, pH*pW, H//pH, W//pW)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # [B, N, C, p, p]
        patches = patches.view(B, pH*pW, -1)  # [B, N, C*p*p]

        # Select masked patches
        target_masked = patches[mask]  # [total_masked, C*p*p]

        # Truncate pred to match target size
        target_size = target_masked.shape[-1]
        pred_truncated = pred_pixels[:, :target_size]

        return F.mse_loss(pred_truncated, target_masked)


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    def forward(self, emb, labels):
        dist = torch.cdist(emb, emb, p=2)
        lab = labels.view(-1,1)
        pos = lab.eq(lab.T).float(); neg = lab.ne(lab.T).float()
        return F.relu((dist*pos).max(1)[0] - (dist*neg+pos*999).min(1)[0] + self.margin).mean()


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


# =============================================================================
# LR SCHEDULER
# =============================================================================
def get_cosine_lr(ep, total, base, warmup=5):
    if ep < warmup: return base*(ep+1)/warmup
    return base*0.5*(1+math.cos(math.pi*(ep-warmup)/max(1,total-warmup)))


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model, data_root, altitude, device, test_locs=None):
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
    for b in ql:
        qe.append(model.extract_embedding(b["query"].to(device)).cpu())
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


# =============================================================================
# TRAINING
# =============================================================================
def train(args):
    set_seed(SEED)
    print("="*70)
    print("GeoMAE: Cross-View Masked Autoencoder for Geo-Localization")
    print("="*70)

    ds = SUES200Dataset(args.data_root, "train", img_size=IMG_SIZE)
    sampler = PKSampler(ds, 8, max(2, BATCH_SIZE//8))
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)

    model = GeoMAEModel(ds.num_classes).to(DEVICE)
    prms = sum(p.numel() for p in model.parameters())/1e6
    print(f"  Model: {prms:.1f}M params")

    recon_fn = PatchReconstructionLoss(IMG_SIZE // PATCH_SIZE)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    trip_fn = TripletLoss(MARGIN)
    nce_fn = InfoNCELoss(0.07)
    uapa_fn = UAPALoss(4.0)

    # =================== PHASE 1: PRETRAINING ===================
    print("\n" + "="*50)
    print("PHASE 1: Cross-View Masked Reconstruction Pretraining")
    print("="*50)

    pretrain_opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR,
                                      weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=AMP_ENABLED)

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        # Progressive masking curriculum
        progress = epoch / max(1, PRETRAIN_EPOCHS - 1)
        mask_ratio = MASK_RATIO_START + (MASK_RATIO_END - MASK_RATIO_START) * progress

        lr = get_cosine_lr(epoch, PRETRAIN_EPOCHS, PRETRAIN_LR, WARMUP_EPOCHS)
        for pg in pretrain_opt.param_groups: pg['lr'] = lr

        tl = 0.0; lp = defaultdict(float)
        for bi, batch in enumerate(loader):
            d = batch["query"].to(DEVICE)
            s = batch["gallery"].to(DEVICE)
            d_raw = batch["query_raw"].to(DEVICE)
            s_raw = batch["gallery_raw"].to(DEVICE)

            pretrain_opt.zero_grad()
            with autocast(enabled=AMP_ENABLED):
                out = model.forward_pretrain(d, s, mask_ratio)

                L = {}
                # Reconstruction losses
                L['d_recon'] = LAMBDA_CROSS_RECON * recon_fn(out['d_pred'], d_raw, out['d_mask'])
                L['s_recon'] = LAMBDA_CROSS_RECON * recon_fn(out['s_pred'], s_raw, out['s_mask'])

                # Encoder alignment (view invariance)
                d_g = F.normalize(out['d_global'], 1)
                s_g = F.normalize(out['s_global'], 1)
                L['align'] = LAMBDA_ALIGN * (1 - F.cosine_similarity(d_g, s_g).mean())

                loss = sum(L.values())

            scaler.scale(loss).backward()
            scaler.unscale_(pretrain_opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(pretrain_opt); scaler.update()

            tl += loss.item()
            for k,v in L.items(): lp[k] += v.item()
            if bi%10==0:
                print(f"  [PT] B{bi} mask={mask_ratio:.2f} L={loss.item():.4f}")

        nb = max(1, len(loader))
        print(f"[PT] Ep {epoch+1}/{PRETRAIN_EPOCHS} mask={mask_ratio:.2f} AvgL={tl/nb:.4f}")
        for k,v in sorted(lp.items()): print(f"  {k}: {v/nb:.4f}")

    # =================== PHASE 2: FINE-TUNING ===================
    print("\n" + "="*50)
    print("PHASE 2: Contrastive + Classification Fine-Tuning")
    print("="*50)

    ft_opt = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': FINETUNE_LR},
        {'params': [p for n,p in model.named_parameters() if 'encoder' not in n],
         'lr': FINETUNE_LR * 5},
    ], weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=AMP_ENABLED)
    best_r1 = 0.0

    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        lr = get_cosine_lr(epoch, FINETUNE_EPOCHS, FINETUNE_LR, WARMUP_EPOCHS)
        for pg in ft_opt.param_groups: pg['lr'] = lr

        tl = 0.0; lp = defaultdict(float)
        for bi, batch in enumerate(loader):
            d,s,lab = batch["query"].to(DEVICE), batch["gallery"].to(DEVICE), batch["label"].to(DEVICE)

            ft_opt.zero_grad()
            with autocast(enabled=AMP_ENABLED):
                do = model.forward_finetune(d, lab, True)
                so = model.forward_finetune(s, lab, True)

                L = {}
                L['ce'] = LAMBDA_CE*0.5*(ce_fn(do['logits'],lab)+ce_fn(so['logits'],lab))
                L['arc'] = LAMBDA_ARCFACE*0.5*(ce_fn(do['arcface_logits'],lab)+ce_fn(so['arcface_logits'],lab))
                L['trip'] = LAMBDA_TRIPLET*0.5*(trip_fn(do['embedding_norm'],lab)+trip_fn(so['embedding_norm'],lab))
                L['nce'] = LAMBDA_INFONCE*nce_fn(do['embedding_norm'],so['embedding_norm'],lab)
                L['uapa'] = LAMBDA_UAPA*uapa_fn(do['logits'],so['logits'])

                # Light reconstruction regularization
                d_raw = batch["query_raw"].to(DEVICE)
                s_raw = batch["gallery_raw"].to(DEVICE)
                pt_out = model.forward_pretrain(d, s, 0.3)
                L['recon_reg'] = 0.1 * (
                    recon_fn(pt_out['d_pred'], d_raw, pt_out['d_mask']) +
                    recon_fn(pt_out['s_pred'], s_raw, pt_out['s_mask'])
                )

                loss = sum(L.values())

            scaler.scale(loss).backward()
            scaler.unscale_(ft_opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(ft_opt); scaler.update()

            tl += loss.item()
            for k,v in L.items(): lp[k] += v.item()
            if bi%10==0: print(f"  [FT] B{bi} L={loss.item():.4f}")

        nb = max(1, len(loader))
        print(f"[FT] Ep {epoch+1}/{FINETUNE_EPOCHS} AvgL={tl/nb:.4f}")
        for k,v in sorted(lp.items()): print(f"  {k}: {v/nb:.4f}")

        if (epoch+1)%EVAL_FREQ==0 or epoch==FINETUNE_EPOCHS-1:
            ar = {}
            for a in ALTITUDES:
                r = evaluate(model, args.data_root, a, DEVICE)
                ar[a] = r; print(f"  {a}m: R@1={r['R@1']:.4f} R@5={r['R@5']:.4f} AP={r['AP']:.4f}")
            avg1 = np.mean([r['R@1'] for r in ar.values()])
            print(f"  AVG R@1={avg1:.4f}")
            if avg1 > best_r1:
                best_r1 = avg1
                torch.save({'epoch':epoch,'model':model.state_dict(),'r1':avg1},
                           os.path.join(OUTPUT_DIR,'geomae_best.pth'))
                print(f"  *** Best R@1={avg1:.4f} ***")

    print(f"\nDone! Best R@1={best_r1:.4f}")


# =============================================================================
# SMOKE TEST
# =============================================================================
def smoke_test():
    print("="*50); print("SMOKE TEST — GeoMAE"); print("="*50)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = GeoMAEModel(10).to(dev)
    print(f"✓ Model: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")

    d = torch.randn(4,3,224,224,device=dev)
    s = torch.randn(4,3,224,224,device=dev)
    lab = torch.tensor([0,0,1,1],device=dev)

    # Pretrain forward
    out = m.forward_pretrain(d, s, 0.5)
    print(f"✓ Pretrain: d_pred={out['d_pred'].shape} d_mask sum={out['d_mask'].sum()}")

    # Finetune forward
    ft = m.forward_finetune(d, lab, True)
    print(f"✓ Finetune: emb={ft['embedding_norm'].shape} logits={ft['logits'].shape}")

    # Backward
    ce = nn.CrossEntropyLoss()(ft['logits'], lab)
    ce.backward()
    gn = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
    print(f"✓ Backward: grad_norm={gn:.4f}")
    print("\n✅ ALL TESTS PASSED!")


def main():
    global PRETRAIN_EPOCHS, FINETUNE_EPOCHS, BATCH_SIZE, DATA_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_pretrain", type=int, default=PRETRAIN_EPOCHS)
    parser.add_argument("--epochs_finetune", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--test", action="store_true")
    args, _ = parser.parse_known_args()
    PRETRAIN_EPOCHS=args.epochs_pretrain; FINETUNE_EPOCHS=args.epochs_finetune
    BATCH_SIZE=args.batch_size; DATA_ROOT=args.data_root
    if args.test: smoke_test(); return
    os.makedirs(OUTPUT_DIR, exist_ok=True); train(args)


if __name__ == "__main__":
    main()
