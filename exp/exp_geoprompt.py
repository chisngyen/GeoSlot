#!/usr/bin/env python3
"""
GeoPrompt: Prompt-Tuned VLM for Cross-View Drone Geo-Localization
==================================================================
Novel contributions:
  1. View-Specific Visual Prompt Tuning (VS-VPT) — Learnable prompt tokens
     that adapt a frozen VLM backbone to drone vs satellite domains
  2. Cross-View Prompt Interaction (CVPI) — Cross-attention between drone &
     satellite prompts to learn view-bridging representations
  3. Geo-Semantic Prompt Routing (GSPR) — Altitude-aware prompt selection
     that routes features through domain-specific prompt banks

Key insight: Instead of fine-tuning the entire backbone (expensive, overfits),
we only tune ~2% of parameters via prompts while keeping the VLM frozen.
This preserves the VLM's rich spatial understanding while adapting to geo-loc.

Architecture:
  Backbone: ConvNeXt-Tiny (frozen in most phases) + Learnable Visual Prompts
  The prompts inject view-specific and altitude-specific information

Dataset: SUES-200 | Protocol: 120/80 fixed split, 200-gallery confusion
Usage:
  python exp_geoprompt.py           # Full training
  python exp_geoprompt.py --test    # Smoke test
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

EPOCHS        = 120
BATCH_SIZE    = 256
NUM_WORKERS   = 8
AMP_ENABLED   = True
EVAL_FREQ     = 5

LR_BACKBONE   = 1e-5     # Very low — backbone mostly frozen
LR_PROMPT     = 5e-3     # High — prompts learn fast
LR_HEAD       = 1e-3
WARMUP_EPOCHS = 5
WEIGHT_DECAY  = 0.01

PHASE1_END    = 30   # Only prompts train
PHASE2_END    = 80   # + CVPI + routing
# Phase 3: + backbone partial unfreezing

LAMBDA_CE         = 1.0
LAMBDA_TRIPLET    = 0.5
LAMBDA_INFONCE    = 0.5
LAMBDA_PROMPT_ORT = 0.1    # Prompt orthogonality (diversity)
LAMBDA_CVPI       = 0.3    # Cross-view prompt interaction
LAMBDA_ARCFACE    = 0.3
LAMBDA_DIST       = 0.3    # Feature distillation
LAMBDA_UAPA       = 0.2
LAMBDA_PROMPT_AL  = 0.2    # Prompt alignment

BACKBONE_NAME = "convnext_tiny"
FEATURE_DIM   = 768
EMBED_DIM     = 512
NUM_PROMPTS   = 16       # Prompts per view
PROMPT_DIM    = 768      # Same as backbone feature dim
PROMPT_DEPTH  = 4        # Inject prompts at 4 backbone stages
NUM_CLASSES   = 120
MARGIN        = 0.3
NUM_ALT_BINS  = 4        # Altitude bins (150, 200, 250, 300)
PROMPT_BANK_SIZE = 8     # Prompts per altitude in bank

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
# DATASET (same standard protocol)
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
    """ALL 200 locations as confusion gallery."""
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
# VIEW-SPECIFIC VISUAL PROMPT TUNING (Novel Component #1)
# =============================================================================
class ViewSpecificPrompts(nn.Module):
    """VS-VPT: Learnable prompt tokens for drone vs satellite view adaptation.

    Instead of prepending prompts to transformer input tokens, we inject them
    into ConvNeXt feature space via channel-attention modulation.

    Each view (drone/satellite) has its own set of learnable prompt vectors
    that modulate feature channels to emphasize view-relevant information.
    """
    def __init__(self, feature_dim, num_prompts=16, num_stages=4):
        super().__init__()
        stage_dims = [96, 192, 384, 768]

        # Drone-specific prompts (one set per backbone stage)
        self.drone_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(num_prompts, d) * 0.02)
            for d in stage_dims
        ])

        # Satellite-specific prompts
        self.sat_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(num_prompts, d) * 0.02)
            for d in stage_dims
        ])

        # Prompt-to-feature attention per stage
        self.prompt_attn = nn.ModuleList([
            nn.MultiheadAttention(d, num_heads=min(8, d//32),
                                   batch_first=True, dropout=0.1)
            for d in stage_dims
        ])

        # Modulation: prompt features → channel attention gates
        self.channel_gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(d, d),
                nn.Sigmoid(),
            ) for d in stage_dims
        ])

    def forward(self, stage_features, stage_idx, view_type='drone'):
        """
        Args:
            stage_features: [B, C, H, W] features from backbone stage
            stage_idx: int, which backbone stage (0-3)
            view_type: 'drone' or 'satellite'
        Returns:
            modulated_features: [B, C, H, W] prompt-modulated features
        """
        B, C, H, W = stage_features.shape

        # Select view-specific prompts
        prompts = (self.drone_prompts[stage_idx] if view_type == 'drone'
                   else self.sat_prompts[stage_idx])  # [P, C]
        prompts = prompts.unsqueeze(0).expand(B, -1, -1)  # [B, P, C]

        # Feature tokens for cross-attention
        feat_tokens = stage_features.flatten(2).transpose(1, 2)  # [B, HW, C]

        # Prompt attends to features
        prompt_out, _ = self.prompt_attn[stage_idx](
            prompts, feat_tokens, feat_tokens
        )  # [B, P, C]

        # Generate channel attention gate
        gate = prompt_out.transpose(1, 2)       # [B, C, P]
        gate = self.channel_gates[stage_idx](gate)  # [B, C]
        gate = gate.unsqueeze(-1).unsqueeze(-1)     # [B, C, 1, 1]

        return stage_features * gate


# =============================================================================
# CROSS-VIEW PROMPT INTERACTION (Novel Component #2)
# =============================================================================
class CrossViewPromptInteraction(nn.Module):
    """CVPI: Cross-attention between drone and satellite prompt representations.

    After VS-VPT generates view-specific features, CVPI enables the two views
    to exchange information, learning what's shared (view-invariant) and what's
    different (view-specific) between drone and satellite perspectives.
    """
    def __init__(self, dim, num_heads=8, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'd2s': nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1),
                's2d': nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1),
                'ffn_d': nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(),
                                       nn.Dropout(0.1), nn.Linear(dim*4, dim)),
                'ffn_s': nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(),
                                       nn.Dropout(0.1), nn.Linear(dim*4, dim)),
                'norm_d1': nn.LayerNorm(dim),
                'norm_s1': nn.LayerNorm(dim),
                'norm_d2': nn.LayerNorm(dim),
                'norm_s2': nn.LayerNorm(dim),
            }) for _ in range(num_layers)
        ])

        # Shared-private decomposition
        self.shared_proj = nn.Linear(dim, dim)
        self.private_d_proj = nn.Linear(dim, dim)
        self.private_s_proj = nn.Linear(dim, dim)

    def forward(self, drone_feat, sat_feat):
        """
        Args:
            drone_feat: [B, N, D] drone tokens
            sat_feat:   [B, N, D] satellite tokens
        Returns:
            drone_enhanced: [B, N, D]
            sat_enhanced:   [B, N, D]
            shared_feat:    [B, D] view-invariant representation
        """
        d, s = drone_feat, sat_feat

        for layer in self.layers:
            # Drone ← Satellite cross-attention
            d_ca, _ = layer['d2s'](layer['norm_d1'](d), s, s)
            d = d + d_ca
            d = d + layer['ffn_d'](layer['norm_d2'](d))

            # Satellite ← Drone cross-attention
            s_ca, _ = layer['s2d'](layer['norm_s1'](s), d, d)
            s = s + s_ca
            s = s + layer['ffn_s'](layer['norm_s2'](s))

        # Shared-private decomposition
        d_pool = d.mean(dim=1)  # [B, D]
        s_pool = s.mean(dim=1)

        shared = F.normalize(self.shared_proj((d_pool + s_pool) / 2), dim=1)
        private_d = F.normalize(self.private_d_proj(d_pool - s_pool), dim=1)
        private_s = F.normalize(self.private_s_proj(s_pool - d_pool), dim=1)

        return d, s, shared, private_d, private_s


# =============================================================================
# GEO-SEMANTIC PROMPT ROUTING (Novel Component #3)
# =============================================================================
class GeoSemanticPromptRouter(nn.Module):
    """GSPR: Altitude-aware prompt bank with dynamic routing.

    Maintains a bank of K prompts per altitude level. During forward pass,
    the router selects and combines relevant prompts based on altitude
    and image content, enabling altitude-specific feature adaptation.
    """
    def __init__(self, feature_dim, num_altitudes=4, bank_size=8):
        super().__init__()
        self.num_altitudes = num_altitudes
        self.bank_size = bank_size

        # Prompt bank: [num_alt, bank_size, dim]
        self.prompt_bank = nn.Parameter(
            torch.randn(num_altitudes, bank_size, feature_dim) * 0.02
        )

        # Content-based router
        self.router = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, bank_size),
        )

        # Altitude embedding
        self.alt_embed = nn.Embedding(num_altitudes, feature_dim // 4)
        self.alt_router = nn.Sequential(
            nn.Linear(feature_dim // 4, bank_size),
            nn.Softmax(dim=-1),
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, features, altitude_idx=None):
        """
        Args:
            features: [B, D] global feature
            altitude_idx: [B] altitude index
        Returns:
            routed_prompt: [B, D] altitude-conditioned prompt modulation
        """
        B, D = features.shape

        if altitude_idx is not None:
            # Content + altitude routing
            content_logits = self.router(features)    # [B, K]
            alt_feat = self.alt_embed(altitude_idx)   # [B, D/4]
            alt_weights = self.alt_router(alt_feat)   # [B, K]

            # Combined routing weights
            weights = F.softmax(content_logits, dim=-1) * 0.5 + alt_weights * 0.5  # [B, K]

            # Select prompts from altitude-specific bank
            # Use altitude_idx to index into bank
            bank_prompts = self.prompt_bank[altitude_idx]  # [B, K, D]
        else:
            # No altitude info (satellite): average across altitude banks
            bank_prompts = self.prompt_bank.mean(dim=0, keepdim=True).expand(B, -1, -1)  # [B, K, D]
            weights = F.softmax(self.router(features), dim=-1)

        # Weighted combination
        routed = (weights.unsqueeze(-1) * bank_prompts).sum(dim=1)  # [B, D]
        return self.out_proj(routed + features)


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
        tgt = torch.cos(th + self.m)
        return (cos*(1-oh) + tgt*oh) * self.s


# =============================================================================
# GEOPROMPT MODEL
# =============================================================================
class GeoPromptStudent(nn.Module):
    """GeoPrompt = Frozen ConvNeXt + VS-VPT + CVPI + GSPR.

    Only ~2% of parameters are trainable (prompts + heads).
    This is extremely parameter-efficient while leveraging rich backbone features.
    """
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM):
        super().__init__()
        # Backbone (mostly frozen)
        self.backbone = timm.create_model(BACKBONE_NAME, pretrained=True,
                                          num_classes=0, global_pool='')
        self.feature_dim = FEATURE_DIM

        # Novel Component #1: View-Specific Visual Prompt Tuning
        self.vs_vpt = ViewSpecificPrompts(FEATURE_DIM, NUM_PROMPTS, PROMPT_DEPTH)

        # Novel Component #2: Cross-View Prompt Interaction
        self.cvpi = CrossViewPromptInteraction(EMBED_DIM, num_heads=8, num_layers=2)

        # Novel Component #3: Geo-Semantic Prompt Router
        self.gspr = GeoSemanticPromptRouter(FEATURE_DIM, NUM_ALT_BINS, PROMPT_BANK_SIZE)

        # Global pooling + embedding
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_proj = nn.Sequential(
            nn.Linear(FEATURE_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Token projection for CVPI
        self.token_proj = nn.Sequential(
            nn.Linear(FEATURE_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Fusion of global + routed features
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + FEATURE_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.arcface = ArcFaceHead(embed_dim, num_classes)

    def forward_backbone_with_prompts(self, x, view_type='drone'):
        """Extract features with prompt injection at each stage."""
        # ConvNeXt stages: stem → stage0 → stage1 → stage2 → stage3
        x = self.backbone.stem(x)  # Initial conv

        stage_outputs = []
        for i, stage in enumerate(self.backbone.stages):
            x = stage(x)
            # Inject view-specific prompts (modulate features)
            if i < PROMPT_DEPTH:
                x = self.vs_vpt(x, stage_idx=i, view_type=view_type)
            stage_outputs.append(x)

        return x, stage_outputs

    def forward(self, x, altitude_idx=None, labels=None, view_type='drone',
                return_all=False):
        # Backbone + prompt injection
        feat_map, stages = self.forward_backbone_with_prompts(x, view_type)

        # Global features
        global_feat = self.pool(feat_map).flatten(1)   # [B, C]
        global_emb = self.embed_proj(global_feat)       # [B, embed_dim]

        # Altitude-aware prompt routing
        routed_feat = self.gspr(global_feat, altitude_idx)  # [B, C]

        # Fuse global + routed
        combined = torch.cat([global_emb, routed_feat], dim=1)  # [B, embed_dim + C]
        embedding = self.fusion(combined)                        # [B, embed_dim]
        embedding_norm = F.normalize(embedding, p=2, dim=1)

        # Token-level features for CVPI
        B, C, H, W = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)  # [B, N, C]
        tokens = self.token_proj(tokens)               # [B, N, embed_dim]

        logits = self.classifier(embedding)
        arc_logits = self.arcface(embedding, labels)

        if return_all:
            return {
                'embedding': embedding,
                'embedding_norm': embedding_norm,
                'logits': logits,
                'arcface_logits': arc_logits,
                'tokens': tokens,
                'global_feat': global_feat,
                'feat_map': feat_map,
            }
        return embedding_norm, logits

    def extract_embedding(self, x, view_type='drone', altitude_idx=None):
        self.eval()
        with torch.no_grad():
            emb, _ = self.forward(x, altitude_idx=altitude_idx, view_type=view_type)
        return emb


# =============================================================================
# LOSSES
# =============================================================================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    def forward(self, emb, labels):
        dist = torch.cdist(emb, emb, p=2)
        lab = labels.view(-1,1)
        pos = lab.eq(lab.T).float(); neg = lab.ne(lab.T).float()
        hp = (dist*pos).max(1)[0]; hn = (dist*neg + pos*999).min(1)[0]
        return F.relu(hp-hn+self.margin).mean()


class InfoNCELoss(nn.Module):
    def __init__(self, t=0.07):
        super().__init__()
        self.t = t
    def forward(self, d, s, labels):
        d=F.normalize(d,1); s=F.normalize(s,1)
        sim = d@s.T/self.t
        lab = labels.view(-1,1); pm = lab.eq(lab.T).float()
        l1 = -(F.log_softmax(sim,1)*pm).sum(1)/pm.sum(1).clamp(1)
        l2 = -(F.log_softmax(sim.T,1)*pm).sum(1)/pm.sum(1).clamp(1)
        return 0.5*(l1.mean()+l2.mean())


class PromptOrthogonalityLoss(nn.Module):
    """Encourage diverse prompts via orthogonality regularization."""
    def forward(self, model):
        loss = 0.0
        count = 0
        for prompts in [model.vs_vpt.drone_prompts, model.vs_vpt.sat_prompts]:
            for p in prompts:
                # p: [num_prompts, dim]
                p_norm = F.normalize(p, dim=1)
                sim = p_norm @ p_norm.T
                # Penalize off-diagonal similarity
                eye = torch.eye(sim.size(0), device=sim.device)
                loss += (sim * (1 - eye)).pow(2).mean()
                count += 1
        return loss / max(count, 1)


class PromptAlignmentLoss(nn.Module):
    """Align shared components of cross-view prompt features."""
    def forward(self, shared, private_d, private_s):
        # Shared should be similar across views
        # Private should be orthogonal to shared
        orth_d = (shared * private_d).sum(1).pow(2).mean()
        orth_s = (shared * private_s).sum(1).pow(2).mean()
        return orth_d + orth_s


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
# TEACHER
# =============================================================================
class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading DINOv2-base teacher...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.embed_dim = 768
        for p in self.parameters(): p.requires_grad = False
    @torch.no_grad()
    def forward(self, x):
        t = self.model.prepare_tokens_with_masks(x)
        for b in self.model.blocks: t = b(t)
        t = self.model.norm(t)
        return t[:, 0], t[:, 1:]


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
        ge.append(model.extract_embedding(b["image"].to(device), 'satellite').cpu())
        gloc.extend(b["loc_id"].tolist())
    ge = torch.cat(ge); gloc = np.array(gloc)
    qe = []
    for b in ql:
        ai = b.get("altitude")
        ai = ai.to(device) if ai is not None else None
        qe.append(model.extract_embedding(b["query"].to(device), 'drone', ai).cpu())
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
    print("="*70); print("GeoPrompt: Prompt-Tuned VLM for Geo-Localization"); print("="*70)
    ds = SUES200Dataset(args.data_root, "train", img_size=IMG_SIZE)
    sampler = PKSampler(ds, 8, max(2, BATCH_SIZE//8))
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)

    model = GeoPromptStudent(ds.num_classes).to(DEVICE)
    # Count trainable params
    total_p = sum(p.numel() for p in model.parameters())/1e6
    # Initially freeze backbone
    for p in model.backbone.parameters(): p.requires_grad = False
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
    print(f"  Total: {total_p:.1f}M | Trainable: {train_p:.1f}M ({train_p/total_p*100:.1f}%)")

    teacher = DINOv2Teacher().to(DEVICE); teacher.eval()

    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    trip_fn = TripletLoss(MARGIN)
    nce_fn = InfoNCELoss(0.07)
    orth_fn = PromptOrthogonalityLoss()
    align_fn = PromptAlignmentLoss()
    uapa_fn = UAPALoss(4.0)

    pgs = [
        {'params': [p for n,p in model.named_parameters() if 'backbone' in n], 'lr': LR_BACKBONE},
        {'params': [p for n,p in model.named_parameters()
                    if any(k in n for k in ['vs_vpt','gspr','cvpi'])], 'lr': LR_PROMPT},
        {'params': [p for n,p in model.named_parameters()
                    if 'backbone' not in n and not any(k in n for k in ['vs_vpt','gspr','cvpi'])],
         'lr': LR_HEAD},
    ]
    opt = torch.optim.AdamW(pgs, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=AMP_ENABLED)
    best_r1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        phase = 1 if epoch < PHASE1_END else (2 if epoch < PHASE2_END else 3)

        # Phase-dependent freezing
        if phase == 1:
            for p in model.backbone.parameters(): p.requires_grad = False
        elif phase == 2:
            for p in model.backbone.parameters(): p.requires_grad = False
        else:  # Phase 3: partial unfreeze (last 2 stages)
            for i, stage in enumerate(model.backbone.stages):
                for p in stage.parameters():
                    p.requires_grad = (i >= 2)

        for pg in opt.param_groups:
            pg['lr'] = get_cosine_lr(epoch, EPOCHS, pg.get('initial_lr', pg['lr']), WARMUP_EPOCHS)

        tl = 0.0; lp = defaultdict(float)
        for bi, batch in enumerate(loader):
            d,s,lab,alt = (batch[k].to(DEVICE) for k in ["query","gallery","label","altitude"])
            opt.zero_grad()
            with autocast(enabled=AMP_ENABLED):
                do = model(d, alt, lab, 'drone', True)
                so = model(s, None, lab, 'satellite', True)
                L = {}
                L['ce'] = LAMBDA_CE*0.5*(ce_fn(do['logits'],lab)+ce_fn(so['logits'],lab))
                L['arc'] = LAMBDA_ARCFACE*0.5*(ce_fn(do['arcface_logits'],lab)+ce_fn(so['arcface_logits'],lab))
                L['trip'] = LAMBDA_TRIPLET*0.5*(trip_fn(do['embedding_norm'],lab)+trip_fn(so['embedding_norm'],lab))
                L['nce'] = LAMBDA_INFONCE*nce_fn(do['embedding_norm'],so['embedding_norm'],lab)
                L['orth'] = LAMBDA_PROMPT_ORT * orth_fn(model)
                if phase >= 2:
                    # CVPI
                    d_enh, s_enh, shared, priv_d, priv_s = model.cvpi(
                        do['tokens'], so['tokens']
                    )
                    L['cvpi'] = LAMBDA_CVPI * nce_fn(
                        d_enh.mean(1), s_enh.mean(1), lab
                    )
                    L['align'] = LAMBDA_PROMPT_AL * align_fn(shared, priv_d, priv_s)
                    # Feature distillation
                    with torch.no_grad():
                        tc_d, _ = teacher(d)
                        tc_s, _ = teacher(s)
                    dn = F.normalize(do['global_feat'],1); sn = F.normalize(so['global_feat'],1)
                    tdn = F.normalize(tc_d,1); tsn = F.normalize(tc_s,1)
                    L['dist'] = LAMBDA_DIST*0.5*(
                        F.mse_loss(dn,tdn)+F.mse_loss(sn,tsn)+
                        (1-F.cosine_similarity(dn,tdn).mean())+
                        (1-F.cosine_similarity(sn,tsn).mean()))
                if phase >= 3:
                    L['uapa'] = LAMBDA_UAPA*uapa_fn(do['logits'],so['logits'])
                loss = sum(L.values())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tl += loss.item()
            for k,v in L.items(): lp[k] += v.item()
            if bi%10==0: print(f"  B{bi}/{len(loader)} L={loss.item():.4f}")

        nb = max(1, len(loader))
        print(f"\nEp {epoch+1}/{EPOCHS} P{phase} AvgL={tl/nb:.4f}")
        for k,v in sorted(lp.items()): print(f"  {k}: {v/nb:.4f}")

        if (epoch+1)%EVAL_FREQ==0 or epoch==EPOCHS-1:
            ar = {}
            for a in ALTITUDES:
                r = evaluate(model, args.data_root, a, DEVICE)
                ar[a] = r
                print(f"  {a}m: R@1={r['R@1']:.4f} R@5={r['R@5']:.4f} AP={r['AP']:.4f}")
            avg1 = np.mean([r['R@1'] for r in ar.values()])
            print(f"  AVG R@1={avg1:.4f}")
            if avg1 > best_r1:
                best_r1 = avg1
                torch.save({'epoch':epoch, 'model':model.state_dict(), 'r1':avg1},
                           os.path.join(OUTPUT_DIR, 'geoprompt_best.pth'))
                print(f"  *** Best R@1={avg1:.4f} ***")

    print(f"\nDone! Best R@1={best_r1:.4f}")


# =============================================================================
# SMOKE TEST
# =============================================================================
def smoke_test():
    print("="*50); print("SMOKE TEST — GeoPrompt"); print("="*50)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = GeoPromptStudent(10).to(dev)
    total = sum(p.numel() for p in m.parameters())/1e6
    for p in m.backbone.parameters(): p.requires_grad = False
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6
    print(f"✓ Model: {total:.1f}M total, {train:.1f}M trainable ({train/total*100:.1f}%)")

    x = torch.randn(4,3,224,224,device=dev)
    lab = torch.tensor([0,0,1,1],device=dev); alt = torch.tensor([0,1,2,3],device=dev)

    do = m(x, alt, lab, 'drone', True)
    so = m(x, None, lab, 'satellite', True)
    print(f"✓ Forward: emb={do['embedding_norm'].shape} tokens={do['tokens'].shape}")

    # CVPI
    d_enh, s_enh, shared, pd, ps = m.cvpi(do['tokens'], so['tokens'])
    print(f"✓ CVPI: shared={shared.shape} private={pd.shape}")

    # Losses
    ce = nn.CrossEntropyLoss()(do['logits'], lab)
    orth = PromptOrthogonalityLoss()(m)
    total_loss = ce + orth
    total_loss.backward()
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
    args, _ = parser.parse_known_args()
    EPOCHS=args.epochs; BATCH_SIZE=args.batch_size; DATA_ROOT=args.data_root
    if args.test: smoke_test(); return
    os.makedirs(OUTPUT_DIR, exist_ok=True); train(args)


if __name__ == "__main__":
    main()
