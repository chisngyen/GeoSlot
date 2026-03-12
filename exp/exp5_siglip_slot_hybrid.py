#!/usr/bin/env python3
"""
EXP5: SigLIP + Adaptive Slot Attention Hybrid with Optimal Transport
=====================================================================
Architecture:
  - SigLIP ViT-B/16 backbone (partially frozen, last 4 layers trainable)
  - Adaptive Slot Attention on SigLIP patch tokens (semantic object decomposition)
  - Lightweight Graph Attention (2-layer GAT) for slot reasoning
  - Sinkhorn OT matching between drone and satellite slots

Novelty:
  - VLM-enriched features enable zero-shot semantic understanding of scene components
  - SigLIP's strong vision-language alignment helps bridge drone-satellite domain gap
  - Slot attention decomposes VLM features into interpretable scene parts
  - Combines VLM generalization with GeoSlot structural matching

Target: Best absolute performance, ~85M params (mostly frozen backbone)

Dataset: SUES-200
Usage:
  python exp5_siglip_slot_hybrid.py           # Full training
  python exp5_siglip_slot_hybrid.py --test    # Smoke test
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
try:
    import open_clip
except ImportError:
    pip_install("open-clip-torch")
print("[2/2] Setup complete!")

# =============================================================================
# IMPORTS
# =============================================================================
import math, glob, json, time, gc, random, argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

# === PATCHED: import shared eval_utils for per-altitude evaluation ===
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')
try:
    from eval_utils import evaluate_full, print_paper_results
    HAS_EVAL_UTILS = True
except ImportError:
    HAS_EVAL_UTILS = False
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
import open_clip


# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT      = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
OUTPUT_DIR     = "/kaggle/working"

EPOCHS         = 100
BATCH_SIZE     = 128
NUM_WORKERS    = 8
AMP_ENABLED    = True
EVAL_FREQ      = 5

# 3-Stage schedule
STAGE1_END     = 15    # Backbone fully frozen, train heads only
STAGE2_END     = 50    # Unfreeze last 4 layers + slot specialization
# Stage 3: Full with all losses

# Learning rates
LR_HEAD        = 3e-4
LR_BACKBONE    = 1e-5
WARMUP_EPOCHS  = 3
WEIGHT_DECAY   = 0.01

# Loss weights
LAMBDA_INFONCE = 1.0
LAMBDA_CE      = 0.5
LAMBDA_TRIPLET = 0.3
LAMBDA_CSM     = 0.3     # Contrastive slot matching
LAMBDA_DICE    = 0.1     # Slot diversity
LAMBDA_OT      = 0.5     # OT matching cost

# SigLIP Model
SIGLIP_MODEL   = "ViT-B-16-SigLIP"
SIGLIP_PRETRAINED = "webli"
VLM_DIM        = 768
TRAINABLE_BLOCKS = 4     # Last N blocks trainable

# Slot Attention
SLOT_DIM       = 256
MAX_SLOTS      = 8
N_REGISTER     = 4
SA_ITERS       = 3
N_HEADS        = 4

# GAT
GAT_LAYERS     = 2
GAT_HEADS      = 4

# OT Matching
SINKHORN_ITERS = 10
OT_EPSILON     = 0.05

# Embedding
EMBED_DIM      = 512
NUM_CLASSES    = 160

# Dataset
IMG_SIZE       = 224
TRAIN_LOCS     = list(range(1, 121))
TEST_LOCS      = list(range(121, 201))
ALTITUDES      = ["150", "200", "250", "300"]
TEST_ALTITUDE  = "150"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# SIGLIP BACKBONE (Partially Frozen)
# =============================================================================
class SigLIPBackbone(nn.Module):
    """SigLIP ViT-B/16 with partial freezing — last N blocks trainable."""
    def __init__(self, trainable_blocks=TRAINABLE_BLOCKS):
        super().__init__()
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            SIGLIP_MODEL, pretrained=SIGLIP_PRETRAINED
        )
        self.visual = model.visual
        self.feature_dim = VLM_DIM

        # Freeze everything first
        for p in self.visual.parameters():
            p.requires_grad = False

        # Unfreeze last N blocks
        if hasattr(self.visual, 'trunk'):
            trunk = self.visual.trunk
        else:
            trunk = self.visual

        blocks = trunk.blocks if hasattr(trunk, 'blocks') else (
            trunk.layers if hasattr(trunk, 'layers') else [])
        total_blocks = len(blocks)
        for i in range(max(0, total_blocks - trainable_blocks), total_blocks):
            for p in blocks[i].parameters():
                p.requires_grad = True

        # Unfreeze norm
        if hasattr(trunk, 'norm'):
            for p in trunk.norm.parameters():
                p.requires_grad = True

    def forward(self, x):
        trunk = self.visual.trunk if hasattr(self.visual, 'trunk') else self.visual

        if hasattr(trunk, 'patch_embed'):
            x_tok = trunk.patch_embed(x)
            has_cls = hasattr(trunk, 'cls_token') and trunk.cls_token is not None
            if has_cls:
                cls_token = trunk.cls_token.expand(x_tok.shape[0], -1, -1)
                x_tok = torch.cat([cls_token, x_tok], dim=1)
            if hasattr(trunk, 'pos_embed') and trunk.pos_embed is not None:
                x_tok = x_tok + trunk.pos_embed

            blocks = trunk.blocks if hasattr(trunk, 'blocks') else trunk.layers
            for blk in blocks:
                x_tok = blk(x_tok)
            if hasattr(trunk, 'norm'):
                x_tok = trunk.norm(x_tok)

            if has_cls:
                cls_feat = x_tok[:, 0]
                patch_tokens = x_tok[:, 1:]
            else:
                cls_feat = x_tok.mean(dim=1)
                patch_tokens = x_tok

            # Estimate spatial dims
            N = patch_tokens.shape[1]
            H = W = int(math.sqrt(N))
            if H * W != N:
                H = N; W = 1

            return cls_feat, patch_tokens, (H, W)
        else:
            out = self.visual(x)
            if out.dim() == 2:
                return out, out.unsqueeze(1), (1, 1)
            return out.mean(1), out, (1, out.shape[1])


# =============================================================================
# ADAPTIVE SLOT ATTENTION
# =============================================================================
class AdaptiveSlotAttention(nn.Module):
    """Slot attention module adapted for VLM features."""
    def __init__(self, feat_dim, slot_dim, num_slots, n_register, n_heads, iters):
        super().__init__()
        self.num_slots = num_slots
        self.n_register = n_register
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.iters = iters
        self.head_dim = slot_dim // n_heads
        self.scale = self.head_dim ** -0.5

        total_slots = num_slots + n_register

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, slot_dim),
            nn.LayerNorm(slot_dim),
        )

        # Learnable slot initialization
        self.slot_mu = nn.Parameter(
            torch.randn(1, total_slots, slot_dim) * (slot_dim ** -0.5))
        self.slot_log_sigma = nn.Parameter(
            torch.zeros(1, total_slots, slot_dim))

        # Attention
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_input = nn.LayerNorm(slot_dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim * 4),
            nn.GELU(),
            nn.Linear(slot_dim * 4, slot_dim),
        )

        # Gumbel selector for adaptive slot count
        self.selector = nn.Sequential(
            nn.Linear(slot_dim, slot_dim // 2),
            nn.ReLU(True),
            nn.Linear(slot_dim // 2, 2),
        )

    def forward(self, features):
        """
        Args:
            features: [B, N, feat_dim] patch tokens from backbone
        Returns:
            obj_slots: [B, num_slots, slot_dim]
            keep_mask: [B, num_slots]
            attn_maps: [B, num_slots, N]
        """
        B = features.shape[0]
        inputs = self.input_proj(features)  # [B, N, slot_dim]
        inputs_normed = self.norm_input(inputs)

        # Initialize slots
        mu = self.slot_mu.expand(B, -1, -1)
        sigma = self.slot_log_sigma.exp().expand(B, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)

        # Iterative attention
        k = self.to_k(inputs_normed).view(B, -1, self.n_heads, self.head_dim)
        v = self.to_v(inputs_normed).view(B, -1, self.n_heads, self.head_dim)

        attn_out = None
        for _ in range(self.iters):
            slots_normed = self.norm_slots(slots)
            q = self.to_q(slots_normed).view(B, -1, self.n_heads, self.head_dim)

            dots = torch.einsum('bihd,bjhd->bihj', q, k) * self.scale
            attn = dots.flatten(1, 2).softmax(dim=1)
            attn = attn.view(B, -1, self.n_heads, dots.shape[-1])
            attn_out = attn.mean(dim=2)  # [B, total_slots, N]

            attn_normed = (attn + 1e-8) / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.einsum('bjhd,bihj->bihd', v, attn_normed)
            updates = updates.reshape(-1, self.slot_dim)

            slots = self.gru(updates, slots.reshape(-1, self.slot_dim))
            slots = slots.reshape(B, -1, self.slot_dim)
            slots = slots + self.ffn(slots)

        # Split object slots and registers
        obj_slots = slots[:, :self.num_slots]
        # reg_slots = slots[:, self.num_slots:]

        # Adaptive slot selection via Gumbel
        logits = self.selector(obj_slots)
        if self.training:
            decision = F.gumbel_softmax(logits, hard=True, tau=0.5)[..., 1]
        else:
            decision = (logits.argmax(dim=-1) == 1).float()

        # Ensure minimum 2 active slots
        active = decision.sum(dim=-1)
        for j in (active < 2).nonzero(as_tuple=True)[0]:
            inactive = (decision[j] == 0).nonzero(as_tuple=True)[0]
            n = min(2 - int(active[j].item()), len(inactive))
            if n > 0:
                idx = inactive[torch.randperm(len(inactive), device=decision.device)[:n]]
                decision[j, idx] = 1.0

        obj_slots = obj_slots * decision.unsqueeze(-1)
        attn_maps = attn_out[:, :self.num_slots] if attn_out is not None else None

        return obj_slots, decision, attn_maps


# =============================================================================
# GRAPH ATTENTION NETWORK (GAT)
# =============================================================================
class GATLayer(nn.Module):
    """Graph Attention layer for slot reasoning."""
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.W = nn.Linear(dim, dim, bias=False)
        self.a = nn.Parameter(torch.randn(n_heads, self.head_dim * 2))
        self.leaky = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, mask=None):
        B, K, D = x.shape
        h = self.W(self.norm(x)).view(B, K, self.n_heads, self.head_dim)

        # Compute attention coefficients
        hi = h.unsqueeze(3).expand(-1, -1, -1, K, -1)
        hj = h.unsqueeze(2).expand(-1, -1, K, -1, -1)
        cat_feat = torch.cat([hi, hj], dim=-1)  # [B, K, n_heads, K, 2*head_dim]

        e = (cat_feat * self.a.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(-1)
        e = self.leaky(e)  # [B, K, n_heads, K]

        if mask is not None:
            mask_2d = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(2).unsqueeze(1)
            e = e.masked_fill(mask_2d.unsqueeze(3).expand_as(e) == 0, -1e9)

        alpha = F.softmax(e, dim=-1)
        out = torch.einsum('bkhn,bnhd->bkhd', alpha, h)
        out = out.contiguous().view(B, K, D)

        return x + out + self.ffn(x + out)


# =============================================================================
# SINKHORN OT MATCHING
# =============================================================================
class SinkhornOTMatcher(nn.Module):
    """Sinkhorn optimal transport for slot matching."""
    def __init__(self, dim, n_iters=10, epsilon=0.05):
        super().__init__()
        self.n_iters = n_iters
        self.log_eps = nn.Parameter(torch.tensor(math.log(epsilon)))

        self.cost_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
        )

    @property
    def epsilon(self):
        return self.log_eps.exp().clamp(0.01, 0.5)

    def forward(self, slots_q, slots_r, mask_q=None, mask_r=None):
        slots_q = slots_q.float()
        slots_r = slots_r.float()

        pq = self.cost_proj(slots_q)
        pr = self.cost_proj(slots_r)

        # Cost matrix
        diff = pq.unsqueeze(2) - pr.unsqueeze(1)
        C = (diff * diff).sum(-1).clamp(min=1e-6).sqrt()  # [B, K, K]
        B, K, M = C.shape

        # Marginals
        if mask_q is not None:
            mu = mask_q.float() / (mask_q.float().sum(-1, keepdim=True) + 1e-8)
        else:
            mu = torch.ones(B, K, device=C.device) / K
        if mask_r is not None:
            nu = mask_r.float() / (mask_r.float().sum(-1, keepdim=True) + 1e-8)
        else:
            nu = torch.ones(B, M, device=C.device) / M

        # Log-domain Sinkhorn
        eps = self.epsilon
        log_K = -C / eps
        log_mu = torch.log(mu.clamp(min=1e-8))
        log_nu = torch.log(nu.clamp(min=1e-8))
        log_u = torch.zeros(B, K, device=C.device)
        log_v = torch.zeros(B, M, device=C.device)

        for _ in range(self.n_iters):
            log_sum_v = torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
            log_u = log_mu - log_sum_v
            log_sum_u = torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)
            log_v = log_nu - log_sum_u

        T = torch.exp(log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1))
        cost = (T * C).sum(dim=(-1, -2))

        # Cosine similarity via OT
        cos_q = F.normalize(slots_q, dim=-1)
        cos_r = F.normalize(slots_r, dim=-1)
        cos_sim = torch.bmm(cos_q, cos_r.transpose(1, 2))
        similarity = (T * cos_sim).sum(dim=(-1, -2))

        return {
            'transport_plan': T,
            'cost': cost,
            'similarity': torch.sigmoid(-cost),
            'cos_similarity': similarity,
        }


# =============================================================================
# FULL HYBRID MODEL
# =============================================================================
class SigLIPSlotHybrid(nn.Module):
    """SigLIP + Adaptive Slot Attention + GAT + Sinkhorn OT."""
    def __init__(self):
        super().__init__()
        self.backbone = SigLIPBackbone(TRAINABLE_BLOCKS)

        # Slot Attention
        self.slot_attn = AdaptiveSlotAttention(
            VLM_DIM, SLOT_DIM, MAX_SLOTS, N_REGISTER, N_HEADS, SA_ITERS)

        # GAT for slot reasoning
        self.gat_layers = nn.ModuleList([
            GATLayer(SLOT_DIM, GAT_HEADS) for _ in range(GAT_LAYERS)
        ])

        # OT Matcher
        self.ot_matcher = SinkhornOTMatcher(SLOT_DIM, SINKHORN_ITERS, OT_EPSILON)

        # Embedding head
        self.embed_head = nn.Sequential(
            nn.LayerNorm(SLOT_DIM),
            nn.Linear(SLOT_DIM, EMBED_DIM),
        )

        # Classification head
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def encode_view(self, x):
        """Encode a single view through SigLIP + Slots + GAT."""
        cls_feat, patch_tokens, spatial_hw = self.backbone(x)

        # Slot attention on VLM patch tokens
        with torch.amp.autocast('cuda', enabled=False):
            patch_tokens = patch_tokens.float()
            obj_slots, keep_mask, attn_maps = self.slot_attn(patch_tokens)

            # GAT reasoning
            for gat in self.gat_layers:
                obj_slots = gat(obj_slots, keep_mask)
            obj_slots = obj_slots * keep_mask.unsqueeze(-1)

        # Weighted aggregation for global embedding
        weights = keep_mask / (keep_mask.sum(dim=-1, keepdim=True) + 1e-8)
        global_slot = (obj_slots * weights.unsqueeze(-1)).sum(dim=1)
        embedding = F.normalize(self.embed_head(global_slot), dim=-1)

        return {
            'embedding': embedding,
            'slots': obj_slots,
            'keep_mask': keep_mask,
            'attn_maps': attn_maps,
            'cls_feat': cls_feat,
        }

    def forward(self, q_img, r_img):
        q = self.encode_view(q_img)
        r = self.encode_view(r_img)

        # OT matching between slot sets
        ot_out = self.ot_matcher(
            q['slots'], r['slots'],
            mask_q=q['keep_mask'], mask_r=r['keep_mask'])

        return {
            'query_embedding': q['embedding'],
            'ref_embedding': r['embedding'],
            'transport_plan': ot_out['transport_plan'],
            'transport_cost': ot_out['cost'],
            'ot_similarity': ot_out['similarity'],
            'query_slots': q['slots'], 'ref_slots': r['slots'],
            'query_keep_mask': q['keep_mask'], 'ref_keep_mask': r['keep_mask'],
            'query_attn_maps': q['attn_maps'], 'ref_attn_maps': r['attn_maps'],
        }

    def extract_embedding(self, x):
        return self.encode_view(x)['embedding']


# =============================================================================
# LOSS
# =============================================================================
class HybridLoss(nn.Module):
    """Combined contrastive + slot specialization + OT loss."""
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(0.07).log())
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)
        self.csm_temp = 0.1

    @property
    def temp(self):
        return self.log_temp.exp().clamp(0.01, 1.0)

    def forward(self, model_out, labels=None, epoch=0):
        q_emb = model_out['query_embedding']
        r_emb = model_out['ref_embedding']
        B = q_emb.shape[0]

        # InfoNCE
        logits = q_emb @ r_emb.t() / self.temp
        targets = torch.arange(B, device=logits.device)
        loss_infonce = (F.cross_entropy(logits, targets) +
                        F.cross_entropy(logits.t(), targets)) / 2
        acc = (logits.argmax(dim=-1) == targets).float().mean()

        # CE
        loss_ce = torch.tensor(0.0, device=q_emb.device)
        if labels is not None:
            loss_ce = (F.cross_entropy(self.classifier(q_emb), labels) +
                       F.cross_entropy(self.classifier(r_emb), labels)) / 2

        # Triplet
        loss_triplet = self._triplet_loss(q_emb, r_emb)

        total_loss = (LAMBDA_INFONCE * loss_infonce + LAMBDA_CE * loss_ce +
                      LAMBDA_TRIPLET * loss_triplet)

        # Stage 2+: Slot specialization losses
        loss_csm = loss_dice = loss_ot = torch.tensor(0.0, device=q_emb.device)
        if epoch >= STAGE1_END:
            ramp = min(1.0, (epoch - STAGE1_END + 1) / 5)

            # Contrastive Slot Matching
            q_slots = model_out['query_slots']
            r_slots = model_out['ref_slots']
            tp = model_out.get('transport_plan')

            q_norm = F.normalize(q_slots, dim=-1)
            r_norm = F.normalize(r_slots, dim=-1)
            slot_sim = torch.bmm(q_norm, r_norm.transpose(1, 2)) / self.csm_temp

            if tp is not None:
                tp_target = tp.detach()
                tp_target = tp_target / (tp_target.sum(dim=-1, keepdim=True) + 1e-8)
                csm_per_slot = -(tp_target * F.log_softmax(slot_sim, dim=-1)).sum(dim=-1)
            else:
                K = slot_sim.shape[1]
                csm_per_slot = F.cross_entropy(
                    slot_sim.view(-1, slot_sim.shape[-1]),
                    torch.arange(K, device=slot_sim.device).repeat(B),
                    reduction='none').view(B, -1)

            q_mask = model_out['query_keep_mask']
            loss_csm = (csm_per_slot * q_mask).sum() / (q_mask.sum() + 1e-8)
            total_loss = total_loss + ramp * LAMBDA_CSM * loss_csm

            # Dice loss for slot diversity
            q_attn = model_out.get('query_attn_maps')
            if q_attn is not None:
                K_obj = min(MAX_SLOTS, q_attn.shape[1])
                obj_attn = q_attn[:, :K_obj]
                obj_attn = obj_attn / (obj_attn.sum(dim=-1, keepdim=True) + 1e-8)
                dice_sum = 0.0; dice_count = 0
                for i in range(K_obj):
                    for j in range(i+1, K_obj):
                        overlap = 2 * (obj_attn[:,i] * obj_attn[:,j]).sum(-1)
                        total_m = obj_attn[:,i].sum(-1) + obj_attn[:,j].sum(-1)
                        dice_sum += (overlap / (total_m + 0.1)).mean()
                        dice_count += 1
                if dice_count > 0:
                    loss_dice = dice_sum / dice_count
                    total_loss = total_loss + ramp * LAMBDA_DICE * loss_dice

            # OT cost (positive pairs should have low transport cost)
            loss_ot = model_out['transport_cost'].mean()
            total_loss = total_loss + ramp * LAMBDA_OT * loss_ot

        stage = ("S1:frozen" if epoch < STAGE1_END else
                 "S2:+slots" if epoch < STAGE2_END else "S3:full")

        return {
            'total_loss': total_loss, 'accuracy': acc, 'stage': stage,
            'loss_infonce': loss_infonce.item(),
            'loss_ce': loss_ce.item() if torch.is_tensor(loss_ce) else loss_ce,
            'loss_triplet': loss_triplet.item(),
            'loss_csm': loss_csm.item() if torch.is_tensor(loss_csm) else loss_csm,
            'loss_dice': loss_dice.item() if torch.is_tensor(loss_dice) else loss_dice,
            'loss_ot': loss_ot.item() if torch.is_tensor(loss_ot) else loss_ot,
        }

    def _triplet_loss(self, q_emb, r_emb, margin=0.3):
        dist = 1.0 - torch.mm(q_emb, r_emb.t())
        pos = dist.diag()
        neg_q = dist.clone(); neg_q.fill_diagonal_(float('inf'))
        neg_r = dist.clone().t(); neg_r.fill_diagonal_(float('inf'))
        return (F.relu(pos - neg_q.min(1)[0] + margin).mean() +
                F.relu(pos - neg_r.min(1)[0] + margin).mean()) / 2


# =============================================================================
# DATASET: SUES-200
# =============================================================================
class SUES200Dataset(Dataset):
    def __init__(self, root, split="train", altitude="150",
                 img_size=224, train_locs=None, test_locs=None):
        super().__init__()
        drone_dir = os.path.join(root, "drone-view")
        satellite_dir = os.path.join(root, "satellite-view")
        if train_locs is None: train_locs = TRAIN_LOCS
        if test_locs is None: test_locs = TEST_LOCS
        locs = train_locs if split == "train" else test_locs

        if split == "train":
            self.drone_tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=3),
                transforms.Pad(10, padding_mode='edge'),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
            self.sat_tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=3),
                transforms.Pad(10, padding_mode='edge'),
                transforms.RandomAffine(90),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        else:
            self.drone_tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
            self.sat_tf = self.drone_tf

        self.pairs = []; self.labels = []; loc_to_label = {}
        for loc_id in locs:
            loc_str = f"{loc_id:04d}"
            sat_path = os.path.join(satellite_dir, loc_str, "0.png")
            if not os.path.exists(sat_path): continue
            alt_dir = os.path.join(drone_dir, loc_str, altitude)
            if not os.path.isdir(alt_dir): continue
            if loc_id not in loc_to_label: loc_to_label[loc_id] = len(loc_to_label)
            for img_name in sorted(os.listdir(alt_dir)):
                if img_name.endswith(('.jpg','.jpeg','.png')):
                    self.pairs.append((os.path.join(alt_dir, img_name), sat_path))
                    self.labels.append(loc_to_label[loc_id])
        self.num_classes = len(loc_to_label)
        print(f"  [SUES-200 {split} alt={altitude}] {len(self.pairs)} pairs ({self.num_classes} cls)")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        dp, sp = self.pairs[idx]
        try: drone=Image.open(dp).convert("RGB"); sat=Image.open(sp).convert("RGB")
        except: drone=Image.new("RGB",(224,224),(128,128,128)); sat=Image.new("RGB",(224,224),(128,128,128))
        return {"query":self.drone_tf(drone),"gallery":self.sat_tf(sat),
                "label":self.labels[idx],"idx":idx}


class SUES200GalleryDataset(Dataset):
    """Satellite gallery with ALL 200 locations (confusion data per SUES-200 protocol)."""
    def __init__(self, root, test_locs=None, img_size=224):
        super().__init__()
        satellite_dir = os.path.join(root, "satellite-view")
        # Standard protocol: gallery includes ALL locations as confusion data
        all_locs = TRAIN_LOCS + TEST_LOCS
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.images = []; self.loc_ids = []
        for loc_id in all_locs:
            loc_str = f"{loc_id:04d}"
            sat_path = os.path.join(satellite_dir, loc_str, "0.png")
            if os.path.exists(sat_path):
                self.images.append(sat_path); self.loc_ids.append(loc_id)
        print(f"  Gallery: {len(self.images)} satellite images (confusion data)")
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return {"image": self.tf(img), "loc_id": self.loc_ids[idx]}


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model, data_root, altitude, device, test_locs=None):
    model.eval()
    query_ds = SUES200Dataset(data_root, "test", altitude, IMG_SIZE, test_locs=test_locs)
    query_loader = DataLoader(query_ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    gallery_ds = SUES200GalleryDataset(data_root, test_locs, IMG_SIZE)
    gallery_loader = DataLoader(gallery_ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    gal_embs, gal_locs = [], []
    for batch in gallery_loader:
        emb = model.extract_embedding(batch["image"].to(device))
        gal_embs.append(emb.cpu()); gal_locs.extend(batch["loc_id"].tolist())
    gal_embs = torch.cat(gal_embs, 0); gal_locs = np.array(gal_locs)

    q_embs = []
    for batch in query_loader:
        emb = model.extract_embedding(batch["query"].to(device))
        q_embs.append(emb.cpu())
    q_embs = torch.cat(q_embs, 0)

    loc_to_gal = {loc: i for i, loc in enumerate(gal_locs)}
    q_gt = np.array([loc_to_gal.get(int(os.path.basename(os.path.dirname(sp))), -1)
                      for _, sp in query_ds.pairs])
    sim = q_embs.numpy() @ gal_embs.numpy().T
    ranks = np.argsort(-sim, axis=1); N = len(q_embs)
    results = {}
    for k in [1, 5, 10]:
        results[f"R@{k}"] = sum(1 for i in range(N) if q_gt[i] in ranks[i,:k]) / N
    ap_sum = sum(1.0/(np.where(ranks[i]==q_gt[i])[0][0]+1)
                 for i in range(N) if len(np.where(ranks[i]==q_gt[i])[0])>0)
    results["AP"] = ap_sum / N
    return results


# =============================================================================
# TRAINING
# =============================================================================
def get_cosine_lr(epoch, total, base, warmup=3):
    if epoch < warmup: return base*(epoch+1)/warmup
    p = (epoch-warmup)/max(1, total-warmup)
    return base*0.5*(1+math.cos(math.pi*p))


def get_stage_lrs(epoch):
    if epoch < STAGE1_END:
        return 0.0, get_cosine_lr(epoch, STAGE1_END, LR_HEAD, WARMUP_EPOCHS)
    elif epoch < STAGE2_END:
        se = epoch-STAGE1_END; sl = STAGE2_END-STAGE1_END
        return get_cosine_lr(se, sl, LR_BACKBONE, 2), get_cosine_lr(se, sl, LR_HEAD*0.5, 0)
    else:
        se = epoch-STAGE2_END; sl = EPOCHS-STAGE2_END
        return get_cosine_lr(se, sl, LR_BACKBONE*0.5, 0), get_cosine_lr(se, sl, LR_HEAD*0.3, 0)


def train(model, train_loader, val_fn, device, epochs=EPOCHS):
    criterion = HybridLoss().to(device)

    # Separate backbone (partially frozen) and head parameters
    bb_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    head_params += list(criterion.parameters())

    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": 0.0},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scaler = GradScaler(enabled=AMP_ENABLED and device.type == "cuda")
    best_r1 = 0.0; history = []

    for epoch in range(epochs):
        lr_bb, lr_hd = get_stage_lrs(epoch)
        optimizer.param_groups[0]["lr"] = lr_bb
        optimizer.param_groups[1]["lr"] = lr_hd

        if epoch < STAGE1_END:
            stage = "S1:frozen"
        elif epoch < STAGE2_END:
            stage = "S2:+slots"
        else:
            stage = "S3:full"

        model.train(); ep_loss = ep_acc = n = 0; t0 = time.time()
        pbar = tqdm(train_loader, desc=f"  Ep {epoch+1}/{epochs} ({stage})", leave=False)

        for batch in pbar:
            query = batch["query"].to(device); gallery = batch["gallery"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=AMP_ENABLED and device.type == "cuda"):
                out = model(query, gallery)
                loss_dict = criterion(out, labels=labels, epoch=epoch)
                loss = loss_dict['total_loss']

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True); continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            ep_loss += loss.item(); ep_acc += loss_dict['accuracy'].item(); n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{loss_dict['accuracy'].item():.1%}")

        elapsed = time.time() - t0
        ep_loss /= max(n,1); ep_acc /= max(n,1)
        entry = {"epoch": epoch+1, "stage": stage, "loss": round(ep_loss,4),
                 "acc": round(ep_acc,4), "lr_bb": round(lr_bb,6),
                 "lr_hd": round(lr_hd,6), "time": round(elapsed,1)}

        if (epoch+1) % EVAL_FREQ == 0 or epoch == epochs-1:
            metrics = val_fn()
            entry.update(metrics)
            r1 = metrics["R@1"]
            if r1 > best_r1:
                best_r1 = r1
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "exp5_hybrid_best.pth"))
            print(f"  Ep {epoch+1} ({stage}) | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | "
                  f"R@1={r1:.2%} | R@5={metrics.get('R@5',0):.2%} | AP={metrics.get('AP',0):.2%} | "
                  f"LR={lr_bb:.1e}/{lr_hd:.1e} | {elapsed:.0f}s")
        else:
            print(f"  Ep {epoch+1} ({stage}) | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | "
                  f"LR={lr_bb:.1e}/{lr_hd:.1e} | {elapsed:.0f}s")
        history.append(entry)

    return best_r1, history


# =============================================================================
# SMOKE TEST
# =============================================================================
def run_test():
    print("\n" + "="*60)
    print("  EXP5 SMOKE TEST: SigLIP + Slot Attention Hybrid")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[1/4] Instantiating model...")
    try:
        model = SigLIPSlotHybrid().to(device)
        criterion = HybridLoss().to(device)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ✓ SigLIPSlotHybrid: {total:,} total, {trainable:,} trainable")
        print(f"  ✓ Size: {total*4/(1024*1024):.1f} MB total, {trainable*4/(1024*1024):.1f} MB trainable")
    except Exception as e:
        print(f"  ✗ Failed: {e}"); import traceback; traceback.print_exc(); return False

    print("\n[2/4] Testing forward pass...")
    try:
        dummy_q = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(device)
        dummy_r = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(device)
        out = model(dummy_q, dummy_r)
        print(f"  ✓ Embedding: {out['query_embedding'].shape}")
        print(f"  ✓ Slots: {out['query_slots'].shape}")
        print(f"  ✓ Transport: {out['transport_plan'].shape}")
        print(f"  ✓ Active slots: {out['query_keep_mask'].sum(1).mean():.1f}/{MAX_SLOTS}")
    except Exception as e:
        print(f"  ✗ Failed: {e}"); import traceback; traceback.print_exc(); return False

    print("\n[3/4] Testing loss at all stages...")
    try:
        labels = torch.randint(0, NUM_CLASSES, (2,)).to(device)
        for ep in [0, STAGE1_END, STAGE2_END]:
            ld = criterion(out, labels=labels, epoch=ep)
            assert not torch.isnan(ld['total_loss']) and not torch.isinf(ld['total_loss'])
            print(f"  ✓ {ld['stage']}: loss={ld['total_loss'].item():.4f}")
    except Exception as e:
        print(f"  ✗ Failed: {e}"); import traceback; traceback.print_exc(); return False

    print("\n[4/4] Testing gradient flow...")
    try:
        ld = criterion(out, labels=labels, epoch=STAGE2_END)
        ld['total_loss'].backward()
        trainable_with_grad = sum(1 for p in model.parameters()
                                  if p.requires_grad and p.grad is not None)
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"  ✓ Gradients: {trainable_with_grad}/{total_trainable} trainable params have grads")
    except Exception as e:
        print(f"  ✗ Failed: {e}"); import traceback; traceback.print_exc(); return False

    print("\n" + "="*60 + "\n  ALL TESTS PASSED ✓\n" + "="*60)
    return True


# =============================================================================
# MAIN
# =============================================================================
class PKSampler:
    def __init__(self, labels, p=8, k=4):
        self.p = p
        self.k = k
        self.locations = list(set(labels))
        self.drone_by_location = defaultdict(list)
        for idx, label in enumerate(labels):
            self.drone_by_location[label].append(idx)
            
    def __iter__(self):
        locations = self.locations.copy()
        random.shuffle(locations)
        batch = []
        for loc in locations:
            indices = self.drone_by_location[loc]
            if len(indices) < self.k:
                indices = indices * (self.k // len(indices) + 1)
            sampled = random.sample(indices, self.k)
            batch.extend(sampled)
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]
                batch = batch[self.p * self.k:]
                
    def __len__(self):
        return len(self.locations) // self.p


def main():
    global EPOCHS, BATCH_SIZE, DATA_ROOT
    parser = argparse.ArgumentParser(description="EXP5: SigLIP+Slot Hybrid")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    args, _ = parser.parse_known_args()

    if args.test: sys.exit(0 if run_test() else 1)

    EPOCHS = args.epochs; BATCH_SIZE = args.batch_size; DATA_ROOT = args.data_root

    print("\n" + "="*70)
    print("  EXP5: SigLIP + Adaptive Slot Attention + OT Hybrid")
    print(f"  Backbone: {SIGLIP_MODEL} (last {TRAINABLE_BLOCKS} blocks trainable)")
    print(f"  Slots: {MAX_SLOTS}+{N_REGISTER}reg | GAT: {GAT_LAYERS} layers")
    print(f"  3-Stage: S1[0-{STAGE1_END}) S2[{STAGE1_END}-{STAGE2_END}) S3[{STAGE2_END}-{EPOCHS})")
    print(f"  Device: {DEVICE}")
    print("="*70)

    print("\n[DATASET] Loading SUES-200...")
    all_pairs = []; all_labels = []
    for alt in ALTITUDES:
        ds = SUES200Dataset(DATA_ROOT, "train", alt, IMG_SIZE)
        all_pairs.extend(ds.pairs); all_labels.extend(ds.labels)

    class CombDS(Dataset):
        def __init__(self, pairs, labels, img_size=224):
            self.pairs=pairs; self.labels=labels
            self.drone_tf = transforms.Compose([
                transforms.Resize((img_size,img_size),interpolation=3),
                transforms.Pad(10,padding_mode='edge'),transforms.RandomCrop((img_size,img_size)),
                transforms.RandomHorizontalFlip(),transforms.ColorJitter(0.2,0.2,0.1,0.05),
                transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            self.sat_tf = transforms.Compose([
                transforms.Resize((img_size,img_size),interpolation=3),
                transforms.Pad(10,padding_mode='edge'),transforms.RandomAffine(90),
                transforms.RandomCrop((img_size,img_size)),transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        def __len__(self): return len(self.pairs)
        def __getitem__(self, idx):
            dp,sp=self.pairs[idx]
            try: d=Image.open(dp).convert("RGB"); s=Image.open(sp).convert("RGB")
            except: d=Image.new("RGB",(224,224),(128,128,128)); s=Image.new("RGB",(224,224),(128,128,128))
            return {"query":self.drone_tf(d),"gallery":self.sat_tf(s),"label":self.labels[idx],"idx":idx}

    train_ds = CombDS(all_pairs, all_labels, IMG_SIZE)
    k_samples = max(2, BATCH_SIZE // 8)
    train_sampler = PKSampler(all_labels, p=8, k=k_samples)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = SigLIPSlotHybrid().to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total:,} | Trainable: {trainable:,}")

    def val_fn(): return evaluate(model, DATA_ROOT, TEST_ALTITUDE, DEVICE)

    best_r1, history = train(model, train_loader, val_fn, DEVICE, EPOCHS)

    print("\n" + "="*70 + "\n  FINAL RESULTS\n" + "="*70)
    for alt in ALTITUDES:
        m = evaluate(model, DATA_ROOT, alt, DEVICE)
        print(f"  Alt={alt}m | R@1={m['R@1']:.2%} | R@5={m['R@5']:.2%} | AP={m['AP']:.2%}")

    with open(os.path.join(OUTPUT_DIR, "exp5_results.json"), "w") as f:
        json.dump({"experiment":"EXP5_SigLIP_Slot_Hybrid","best_r1":best_r1,
                   "total_params":total,"trainable_params":trainable,
                   "history":history}, f, indent=2, default=str)



def run_final_evaluation(model, test_dataset, device, exp_name, cfg=Config):
    """Run comprehensive per-altitude evaluation with paper-grade output."""
    if HAS_EVAL_UTILS:
        results = evaluate_full(
            model, test_dataset, device,
            data_root=cfg.DATA_ROOT,
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS,
            img_size=cfg.IMG_SIZE,
            train_locs=cfg.TRAIN_LOCS,
            test_locs=cfg.TEST_LOCS,
        )
        print_paper_results(results, exp_name=exp_name)
        return results
    else:
        print("eval_utils not found, using basic evaluate()")
        r, ap = evaluate(model, test_dataset, device)
        print(f"R@1:{r['R@1']:.2f}% R@5:{r['R@5']:.2f}% R@10:{r['R@10']:.2f}% mAP:{ap:.2f}%")
        return {'overall': {**r, 'mAP': ap}}

if __name__ == "__main__":
    main()
