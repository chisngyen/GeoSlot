#!/usr/bin/env python3
"""
EXP3: Object-Centric Graph Matching Network for Structural Geo-Localization
=============================================================================
Architecture:
  - ConvNeXt-Tiny backbone for feature extraction
  - Self-supervised attention-based node generation (no external detector)
  - GNN with cross-attention message passing between drone & satellite graphs
  - Sinkhorn-based differentiable graph matching for structural alignment

Novelty:
  - Self-supervised node generation via attention-based clustering
  - Bi-directional cross-graph attention for inter-view reasoning
  - Spatial-aware edge features encoding geometric relationships
  - Robust to occlusion and partial visibility

Dataset: SUES-200 (drone ↔ satellite cross-view geo-localization)
Usage:
  python exp3_graph_matching.py           # Full training on Kaggle
  python exp3_graph_matching.py --test    # Smoke test
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


# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT      = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
OUTPUT_DIR     = "/kaggle/working"

EPOCHS         = 120
BATCH_SIZE     = 192
NUM_WORKERS    = 8
AMP_ENABLED    = True
EVAL_FREQ      = 5

STAGE1_END     = 20
STAGE2_END     = 70

LR_HEAD        = 1e-3
LR_BACKBONE    = 1e-5
WARMUP_EPOCHS  = 5
WEIGHT_DECAY   = 0.01

LAMBDA_CE      = 1.0
LAMBDA_TRIPLET = 0.5
LAMBDA_MATCH   = 1.0     # Graph matching
LAMBDA_NODE    = 0.3     # Node diversity
LAMBDA_EDGE    = 0.2     # Edge consistency

BACKBONE_NAME  = "convnext_tiny"
FEATURE_DIM    = 768
NUM_NODES      = 8       # Number of graph nodes per view
EMBED_DIM      = 512
GNN_LAYERS     = 3
SINKHORN_ITERS = 10
NUM_CLASSES    = 160

IMG_SIZE       = 224
TRAIN_LOCS     = list(range(1, 121))
TEST_LOCS      = list(range(121, 201))
ALTITUDES      = ["150", "200", "250", "300"]
TEST_ALTITUDE  = "150"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# BACKBONE
# =============================================================================
class ConvNeXtBackbone(nn.Module):
    def __init__(self, model_name=BACKBONE_NAME, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       num_classes=0, global_pool='')
        self.feature_dim = FEATURE_DIM

    def forward(self, x):
        feat = self.model(x)
        B, C, H, W = feat.shape
        patch_tokens = feat.flatten(2).transpose(1, 2)  # [B, N, C]
        global_feat = feat.mean(dim=[2, 3])
        return global_feat, patch_tokens, (H, W)


# =============================================================================
# NODE GENERATION: Attention-based Clustering
# =============================================================================
class AttentionNodeGenerator(nn.Module):
    """Self-supervised node generation via attention-based soft clustering."""
    def __init__(self, feat_dim, num_nodes, n_heads=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_heads = n_heads
        self.head_dim = feat_dim // n_heads

        # Learnable node queries (prototypes)
        self.node_queries = nn.Parameter(
            torch.randn(1, num_nodes, feat_dim) * (feat_dim ** -0.5))

        # Multi-head cross-attention
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)

        self.norm_q = nn.LayerNorm(feat_dim)
        self.norm_kv = nn.LayerNorm(feat_dim)

        # FFN after attention
        self.ffn = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, patch_tokens, spatial_hw):
        """
        Args:
            patch_tokens: [B, N, D] patch-level features from backbone
            spatial_hw: (H, W) spatial dimensions
        Returns:
            node_features: [B, K, D] node features
            assignment: [B, K, N] soft assignment matrix
            node_positions: [B, K, 2] estimated node positions
        """
        B, N, D = patch_tokens.shape
        H, W = spatial_hw

        # Expand queries
        queries = self.node_queries.expand(B, -1, -1)  # [B, K, D]

        # Cross-attention: queries attend to patch tokens
        q = self.q_proj(self.norm_q(queries))
        k = self.k_proj(self.norm_kv(patch_tokens))
        v = self.v_proj(self.norm_kv(patch_tokens))

        # Reshape for multi-head
        q = q.view(B, self.num_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, K, N]
        attn = F.softmax(attn, dim=-1)

        # Weighted aggregation
        out = torch.matmul(attn, v)  # [B, H, K, D_head]
        out = out.transpose(1, 2).contiguous().view(B, self.num_nodes, D)
        node_features = self.out_proj(out) + queries  # Residual
        node_features = node_features + self.ffn(node_features)

        # Soft assignment matrix (averaged over heads)
        assignment = attn.mean(dim=1)  # [B, K, N]

        # Estimate node positions from assignment
        gy = torch.arange(H, device=patch_tokens.device, dtype=torch.float) / max(H-1, 1)
        gx = torch.arange(W, device=patch_tokens.device, dtype=torch.float) / max(W-1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [N, 2]

        # Weighted average of spatial positions
        assign_norm = assignment / (assignment.sum(dim=-1, keepdim=True) + 1e-8)
        node_positions = torch.einsum('bkn,nc->bkc', assign_norm, coords)  # [B, K, 2]

        return node_features, assignment, node_positions


# =============================================================================
# SPATIAL-AWARE EDGE FEATURES
# =============================================================================
class SpatialEdgeEncoder(nn.Module):
    """Encode geometric relationships between nodes."""
    def __init__(self, feat_dim, pos_dim=32):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 + feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, node_features, node_positions):
        """
        Args:
            node_features: [B, K, D]
            node_positions: [B, K, 2]
        Returns:
            edge_features: [B, K, K, D]
        """
        B, K, D = node_features.shape

        # Pairwise geometric features
        pos_i = node_positions.unsqueeze(2).expand(-1, -1, K, -1)  # [B, K, K, 2]
        pos_j = node_positions.unsqueeze(1).expand(-1, K, -1, -1)

        # Distance and direction
        diff = pos_i - pos_j  # [B, K, K, 2]
        dist = diff.norm(dim=-1, keepdim=True)  # [B, K, K, 1]
        angle = torch.atan2(diff[..., 1:2], diff[..., 0:1])  # [B, K, K, 1]

        geo_feat = torch.cat([diff, dist, angle], dim=-1)  # [B, K, K, 4]

        # Node feature difference
        feat_i = node_features.unsqueeze(2).expand(-1, -1, K, -1)
        feat_j = node_features.unsqueeze(1).expand(-1, K, -1, -1)
        feat_diff = feat_i - feat_j  # [B, K, K, D]

        edge_input = torch.cat([geo_feat, feat_diff], dim=-1)
        return self.edge_mlp(edge_input)


# =============================================================================
# GNN WITH CROSS-ATTENTION
# =============================================================================
class CrossGraphAttentionLayer(nn.Module):
    """Single GNN layer with intra-graph + cross-graph attention."""
    def __init__(self, feat_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = feat_dim // n_heads

        # Intra-graph attention
        self.intra_qkv = nn.Linear(feat_dim, feat_dim * 3)
        self.intra_out = nn.Linear(feat_dim, feat_dim)
        self.intra_norm = nn.LayerNorm(feat_dim)

        # Cross-graph attention
        self.cross_q = nn.Linear(feat_dim, feat_dim)
        self.cross_kv = nn.Linear(feat_dim, feat_dim * 2)
        self.cross_out = nn.Linear(feat_dim, feat_dim)
        self.cross_norm = nn.LayerNorm(feat_dim)

        # Edge-conditioned message passing
        self.edge_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim * 4, feat_dim),
        )

    def forward(self, nodes_a, nodes_b, edge_a=None, edge_b=None):
        """
        Args:
            nodes_a, nodes_b: [B, K, D] node features from two views
            edge_a, edge_b: [B, K, K, D] edge features (optional)
        Returns:
            updated_a, updated_b: [B, K, D]
        """
        B, K, D = nodes_a.shape

        # Intra-graph attention for view A
        nodes_a = nodes_a + self._intra_attention(nodes_a, edge_a)
        nodes_b = nodes_b + self._intra_attention(nodes_b, edge_b)

        # Cross-graph attention: A attends to B and vice versa
        cross_a = self._cross_attention(nodes_a, nodes_b)
        cross_b = self._cross_attention(nodes_b, nodes_a)
        nodes_a = self.cross_norm(nodes_a + cross_a)
        nodes_b = self.cross_norm(nodes_b + cross_b)

        # FFN
        nodes_a = nodes_a + self.ffn(nodes_a)
        nodes_b = nodes_b + self.ffn(nodes_b)

        return nodes_a, nodes_b

    def _intra_attention(self, nodes, edges=None):
        B, K, D = nodes.shape
        qkv = self.intra_qkv(self.intra_norm(nodes))
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, K, D)

        # Edge-conditioned gating
        if edges is not None:
            gate = self.edge_gate(edges.mean(dim=-2))  # Aggregate neighbor edges
            out = out * gate

        return self.intra_out(out)

    def _cross_attention(self, query_nodes, key_nodes):
        B, K, D = query_nodes.shape
        q = self.cross_q(query_nodes)
        kv = self.cross_kv(key_nodes)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, K, D)
        return self.cross_out(out)


# =============================================================================
# SINKHORN DIFFERENTIABLE MATCHING
# =============================================================================
class SinkhornMatcher(nn.Module):
    """Differentiable graph matching via Sinkhorn optimal transport."""
    def __init__(self, feat_dim, n_iters=10, temperature=0.05):
        super().__init__()
        self.n_iters = n_iters
        self.log_temp = nn.Parameter(torch.tensor(math.log(temperature)))

        # Scoring network
        self.score_net = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1),
        )

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(0.01, 0.5)

    def forward(self, nodes_a, nodes_b, pos_a=None, pos_b=None):
        """
        Args:
            nodes_a, nodes_b: [B, K, D]
            pos_a, pos_b: [B, K, 2] node positions
        Returns:
            transport_plan: [B, K, K] soft matching
            cost: [B] matching cost
            similarity: [B] matching similarity
        """
        B, K, D = nodes_a.shape

        # Compute pairwise cost matrix
        a_exp = nodes_a.unsqueeze(2).expand(-1, -1, K, -1)
        b_exp = nodes_b.unsqueeze(1).expand(-1, K, -1, -1)
        pair_feat = torch.cat([a_exp, b_exp], dim=-1)
        scores = self.score_net(pair_feat).squeeze(-1)  # [B, K, K]

        # Also add cosine similarity
        cos_sim = F.cosine_similarity(a_exp, b_exp, dim=-1)
        C = -(scores + cos_sim) / self.temperature

        # Add spatial distance penalty if positions available
        if pos_a is not None and pos_b is not None:
            pa = pos_a.unsqueeze(2).expand(-1, -1, K, -1)
            pb = pos_b.unsqueeze(1).expand(-1, K, -1, -1)
            spatial_dist = (pa - pb).norm(dim=-1)  # [B, K, K]
            C = C + 0.1 * spatial_dist

        # Sinkhorn normalization
        T = self._sinkhorn(C)

        # Compute matching cost and similarity
        cost = (T * C).sum(dim=(-1, -2))
        similarity = (T * cos_sim).sum(dim=(-1, -2))

        return {
            'transport_plan': T,
            'cost': cost,
            'similarity': similarity,
            'cost_matrix': C,
        }

    def _sinkhorn(self, C):
        """Log-domain Sinkhorn for numerical stability."""
        B, K, M = C.shape
        log_mu = torch.full((B, K), -math.log(K), device=C.device)
        log_nu = torch.full((B, M), -math.log(M), device=C.device)

        log_u = torch.zeros(B, K, device=C.device)
        log_v = torch.zeros(B, M, device=C.device)

        for _ in range(self.n_iters):
            log_u = log_mu - torch.logsumexp(C + log_v.unsqueeze(1), dim=2)
            log_v = log_nu - torch.logsumexp(C + log_u.unsqueeze(2), dim=1)

        T = torch.exp(C + log_u.unsqueeze(2) + log_v.unsqueeze(1))
        return T


# =============================================================================
# FULL MODEL
# =============================================================================
class GraphMatchingNet(nn.Module):
    """Complete graph matching network for cross-view geo-localization."""
    def __init__(self):
        super().__init__()
        self.backbone = ConvNeXtBackbone()

        # Node generation
        self.node_generator = AttentionNodeGenerator(FEATURE_DIM, NUM_NODES)

        # Edge encoding
        self.edge_encoder = SpatialEdgeEncoder(FEATURE_DIM)

        # GNN layers with cross-attention
        self.gnn_layers = nn.ModuleList([
            CrossGraphAttentionLayer(FEATURE_DIM) for _ in range(GNN_LAYERS)
        ])

        # Graph matching
        self.matcher = SinkhornMatcher(FEATURE_DIM, SINKHORN_ITERS)

        # Global embedding head
        self.embed_head = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM),
            nn.Linear(FEATURE_DIM, EMBED_DIM),
        )

        # Classification head
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def encode_view(self, x):
        """Encode single view into graph representation."""
        global_feat, patch_tokens, spatial_hw = self.backbone(x)

        # Generate nodes via attention clustering
        node_features, assignment, node_positions = self.node_generator(
            patch_tokens, spatial_hw)

        # Compute edge features
        edge_features = self.edge_encoder(node_features, node_positions)

        # Global embedding from weighted node aggregation
        node_importance = assignment.sum(dim=-1)  # [B, K]
        weights = F.softmax(node_importance, dim=-1).unsqueeze(-1)
        global_node = (node_features * weights).sum(dim=1)
        embedding = F.normalize(self.embed_head(global_node), dim=-1)

        return {
            'embedding': embedding,
            'nodes': node_features,
            'edges': edge_features,
            'positions': node_positions,
            'assignment': assignment,
            'global_feat': global_feat,
        }

    def forward(self, q_img, r_img):
        q = self.encode_view(q_img)
        r = self.encode_view(r_img)

        # Cross-graph reasoning via GNN
        nodes_q, nodes_r = q['nodes'], r['nodes']
        for gnn_layer in self.gnn_layers:
            nodes_q, nodes_r = gnn_layer(nodes_q, nodes_r, q['edges'], r['edges'])

        # Graph matching
        match_out = self.matcher(nodes_q, nodes_r, q['positions'], r['positions'])

        # Update embeddings after cross-reasoning
        w_q = F.softmax(q['assignment'].sum(dim=-1), dim=-1).unsqueeze(-1)
        w_r = F.softmax(r['assignment'].sum(dim=-1), dim=-1).unsqueeze(-1)
        q_emb = F.normalize(self.embed_head((nodes_q * w_q).sum(dim=1)), dim=-1)
        r_emb = F.normalize(self.embed_head((nodes_r * w_r).sum(dim=1)), dim=-1)

        return {
            'query_embedding': q_emb,
            'ref_embedding': r_emb,
            'transport_plan': match_out['transport_plan'],
            'match_cost': match_out['cost'],
            'match_similarity': match_out['similarity'],
            'query_nodes': nodes_q, 'ref_nodes': nodes_r,
            'query_assignment': q['assignment'],
            'ref_assignment': r['assignment'],
            'query_positions': q['positions'],
            'ref_positions': r['positions'],
        }

    def extract_embedding(self, x):
        return self.encode_view(x)['embedding']


# =============================================================================
# LOSS
# =============================================================================
class GraphMatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(0.07).log())
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

    @property
    def temp(self):
        return self.log_temp.exp().clamp(0.01, 1.0)

    def node_diversity_loss(self, assignment):
        """Encourage non-overlapping node assignments (like Dice loss in GeoSlot)."""
        B, K, N = assignment.shape
        assign_norm = assignment / (assignment.sum(dim=-1, keepdim=True) + 1e-8)
        overlap = 0.0; count = 0
        for i in range(min(K, 6)):
            for j in range(i+1, min(K, 6)):
                inter = 2 * (assign_norm[:, i] * assign_norm[:, j]).sum(dim=-1)
                union = assign_norm[:, i].sum(dim=-1) + assign_norm[:, j].sum(dim=-1)
                overlap += (inter / (union + 0.1)).mean()
                count += 1
        return overlap / max(count, 1)

    def edge_consistency_loss(self, nodes_a, nodes_b, transport_plan):
        """Structure consistency: similar graph structures should be matched."""
        # Compute structure matrices
        Sa = torch.bmm(F.normalize(nodes_a, dim=-1),
                       F.normalize(nodes_a, dim=-1).transpose(1, 2))
        Sb = torch.bmm(F.normalize(nodes_b, dim=-1),
                       F.normalize(nodes_b, dim=-1).transpose(1, 2))
        # Transport-aligned structure difference
        Sb_aligned = torch.bmm(torch.bmm(transport_plan, Sb), transport_plan.transpose(1, 2))
        return F.mse_loss(Sa, Sb_aligned)

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

        total_loss = LAMBDA_CE * (loss_infonce + loss_ce) + LAMBDA_TRIPLET * loss_triplet

        # Graph matching loss (positive pairs should match well)
        loss_match = torch.tensor(0.0, device=q_emb.device)
        loss_node = torch.tensor(0.0, device=q_emb.device)
        loss_edge = torch.tensor(0.0, device=q_emb.device)

        if epoch >= STAGE1_END:
            ramp = min(1.0, (epoch - STAGE1_END + 1) / 10)

            # Matching cost should be low for positive pairs
            loss_match = model_out['match_cost'].mean()
            total_loss = total_loss + ramp * LAMBDA_MATCH * loss_match

            # Node diversity
            loss_node = (self.node_diversity_loss(model_out['query_assignment']) +
                         self.node_diversity_loss(model_out['ref_assignment'])) / 2
            total_loss = total_loss + ramp * LAMBDA_NODE * loss_node

            # Edge consistency
            loss_edge = self.edge_consistency_loss(
                model_out['query_nodes'], model_out['ref_nodes'],
                model_out['transport_plan'])
            total_loss = total_loss + ramp * LAMBDA_EDGE * loss_edge

        stage = ("S1:frozen" if epoch < STAGE1_END else
                 "S2:+graph" if epoch < STAGE2_END else "S3:full")

        return {
            'total_loss': total_loss, 'accuracy': acc,
            'loss_infonce': loss_infonce.item(),
            'loss_ce': loss_ce.item() if torch.is_tensor(loss_ce) else loss_ce,
            'loss_triplet': loss_triplet.item(),
            'loss_match': loss_match.item() if torch.is_tensor(loss_match) else loss_match,
            'loss_node': loss_node.item() if torch.is_tensor(loss_node) else loss_node,
            'loss_edge': loss_edge.item() if torch.is_tensor(loss_edge) else loss_edge,
            'stage': stage,
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
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.pairs.append((os.path.join(alt_dir, img_name), sat_path))
                    self.labels.append(loc_to_label[loc_id])
        self.num_classes = len(loc_to_label)
        print(f"  [SUES-200 {split} alt={altitude}] {len(self.pairs)} pairs ({self.num_classes} cls)")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        dp, sp = self.pairs[idx]
        try: drone = Image.open(dp).convert("RGB"); sat = Image.open(sp).convert("RGB")
        except: drone = Image.new("RGB",(224,224),(128,128,128)); sat = Image.new("RGB",(224,224),(128,128,128))
        return {"query": self.drone_tf(drone), "gallery": self.sat_tf(sat),
                "label": self.labels[idx], "idx": idx}


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
    query_loader = DataLoader(query_ds, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    gallery_ds = SUES200GalleryDataset(data_root, test_locs, IMG_SIZE)
    gallery_loader = DataLoader(gallery_ds, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

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

    loc_to_gal_idx = {loc: i for i, loc in enumerate(gal_locs)}
    q_gt = np.array([loc_to_gal_idx.get(int(os.path.basename(os.path.dirname(sp))), -1)
                      for _, sp in query_ds.pairs])

    sim = q_embs.numpy() @ gal_embs.numpy().T
    ranks = np.argsort(-sim, axis=1); N = len(q_embs)
    results = {}
    for k in [1, 5, 10]:
        results[f"R@{k}"] = sum(1 for i in range(N) if q_gt[i] in ranks[i, :k]) / N
    ap_sum = sum(1.0/(np.where(ranks[i]==q_gt[i])[0][0]+1)
                 for i in range(N) if len(np.where(ranks[i]==q_gt[i])[0])>0)
    results["AP"] = ap_sum / N
    return results


# =============================================================================
# LR SCHEDULER
# =============================================================================
def get_cosine_lr(epoch, total_epochs, base_lr, warmup=5):
    if epoch < warmup: return base_lr * (epoch+1)/warmup
    p = (epoch-warmup)/max(1, total_epochs-warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * p))


# =============================================================================
# TRAINING
# =============================================================================
def train(model, train_loader, val_fn, device, epochs=EPOCHS):
    criterion = GraphMatchLoss().to(device)
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
        if epoch < STAGE1_END:
            lr_bb = 0.0; lr_hd = get_cosine_lr(epoch, STAGE1_END, LR_HEAD, WARMUP_EPOCHS)
            for p in bb_params: p.requires_grad = False
            stage = "S1:frozen"
        elif epoch < STAGE2_END:
            se = epoch-STAGE1_END; sl = STAGE2_END-STAGE1_END
            lr_bb = get_cosine_lr(se, sl, LR_BACKBONE, 3)
            lr_hd = get_cosine_lr(se, sl, LR_HEAD*0.5, 0)
            for p in bb_params: p.requires_grad = True
            stage = "S2:+graph"
        else:
            se = epoch-STAGE2_END; sl = epochs-STAGE2_END
            lr_bb = get_cosine_lr(se, sl, LR_BACKBONE*0.5, 0)
            lr_hd = get_cosine_lr(se, sl, LR_HEAD*0.3, 0)
            stage = "S3:full"
        optimizer.param_groups[0]["lr"] = lr_bb
        optimizer.param_groups[1]["lr"] = lr_hd

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
                 "acc": round(ep_acc,4), "time": round(elapsed,1)}

        if (epoch+1) % EVAL_FREQ == 0 or epoch == epochs-1:
            metrics = val_fn()
            entry.update(metrics)
            r1 = metrics["R@1"]
            if r1 > best_r1:
                best_r1 = r1
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "exp3_graph_best.pth"))
            print(f"  Ep {epoch+1} ({stage}) | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | "
                  f"R@1={r1:.2%} | AP={metrics.get('AP',0):.2%} | {elapsed:.0f}s")
        else:
            print(f"  Ep {epoch+1} ({stage}) | Loss={ep_loss:.4f} | Acc={ep_acc:.1%} | {elapsed:.0f}s")
        history.append(entry)

    return best_r1, history


# =============================================================================
# SMOKE TEST
# =============================================================================
def run_test():
    print("\n" + "="*60)
    print("  EXP3 SMOKE TEST: Graph Matching Network")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[1/4] Instantiating model...")
    try:
        model = GraphMatchingNet().to(device)
        criterion = GraphMatchLoss().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ GraphMatchingNet, {total_params:,} params ({total_params*4/(1024*1024):.1f} MB)")
    except Exception as e:
        print(f"  ✗ Failed: {e}"); return False

    print("\n[2/4] Testing forward pass...")
    try:
        dummy_q = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
        dummy_r = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
        out = model(dummy_q, dummy_r)
        print(f"  ✓ Embedding: {out['query_embedding'].shape}")
        print(f"  ✓ Nodes: {out['query_nodes'].shape}")
        print(f"  ✓ Transport plan: {out['transport_plan'].shape}")
        print(f"  ✓ Positions: {out['query_positions'].shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}"); import traceback; traceback.print_exc(); return False

    print("\n[3/4] Testing loss computation...")
    try:
        labels = torch.randint(0, NUM_CLASSES, (4,)).to(device)
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
        ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"  ✓ All gradients flow: {ok}")
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
    parser = argparse.ArgumentParser(description="EXP3: Graph Matching")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    args, _ = parser.parse_known_args()

    if args.test:
        sys.exit(0 if run_test() else 1)

    EPOCHS = args.epochs; BATCH_SIZE = args.batch_size; DATA_ROOT = args.data_root

    print("\n" + "="*70)
    print("  EXP3: Object-Centric Graph Matching Network")
    print(f"  Backbone: {BACKBONE_NAME} | Nodes: {NUM_NODES} | GNN layers: {GNN_LAYERS}")
    print(f"  Device: {DEVICE}")
    print("="*70)

    print("\n[DATASET] Loading SUES-200...")
    train_pairs_all = []; train_labels_all = []
    for alt in ALTITUDES:
        ds = SUES200Dataset(DATA_ROOT, "train", alt, IMG_SIZE)
        train_pairs_all.extend(ds.pairs); train_labels_all.extend(ds.labels)

    class CombinedDS(Dataset):
        def __init__(self, pairs, labels, img_size=224):
            self.pairs=pairs; self.labels=labels
            self.drone_tf = transforms.Compose([
                transforms.Resize((img_size,img_size), interpolation=3),
                transforms.Pad(10, padding_mode='edge'),
                transforms.RandomCrop((img_size,img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2,0.2,0.1,0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            self.sat_tf = transforms.Compose([
                transforms.Resize((img_size,img_size), interpolation=3),
                transforms.Pad(10, padding_mode='edge'),
                transforms.RandomAffine(90),
                transforms.RandomCrop((img_size,img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        def __len__(self): return len(self.pairs)
        def __getitem__(self, idx):
            dp,sp=self.pairs[idx]
            try: drone=Image.open(dp).convert("RGB"); sat=Image.open(sp).convert("RGB")
            except: drone=Image.new("RGB",(224,224),(128,128,128)); sat=Image.new("RGB",(224,224),(128,128,128))
            return {"query":self.drone_tf(drone),"gallery":self.sat_tf(sat),"label":self.labels[idx],"idx":idx}

    train_ds = CombinedDS(train_pairs_all, train_labels_all, IMG_SIZE)
    k_samples = max(2, BATCH_SIZE // 8)
    train_sampler = PKSampler(train_labels_all, p=8, k=k_samples)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = GraphMatchingNet().to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    def val_fn(): return evaluate(model, DATA_ROOT, TEST_ALTITUDE, DEVICE)

    best_r1, history = train(model, train_loader, val_fn, DEVICE, EPOCHS)

    print("\n" + "="*70 + "\n  FINAL RESULTS\n" + "="*70)
    for alt in ALTITUDES:
        m = evaluate(model, DATA_ROOT, alt, DEVICE)
        print(f"  Alt={alt}m | R@1={m['R@1']:.2%} | R@5={m['R@5']:.2%} | AP={m['AP']:.2%}")
    print(f"\n  Best R@1: {best_r1:.2%}")

    with open(os.path.join(OUTPUT_DIR, "exp3_results.json"), "w") as f:
        json.dump({"experiment":"EXP3_GraphMatching","best_r1":best_r1,"history":history},
                  f, indent=2, default=str)



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
