#!/usr/bin/env python3
"""
GeoGraph: Scene Graph Matching for Cross-View Drone Geo-Localization
=====================================================================
Novel contributions:
  1. Spatial Scene Graph Construction — Builds a graph from spatial feature
     tokens where nodes = image regions and edges = spatial relationships
  2. Cross-View Graph Matching Network (CVGMN) — GNN-based differentiable
     graph matching that aligns drone and satellite scene topologies
  3. Topology-Preserving Contrastive Loss — Enforces structural consistency
     between matched graphs beyond simple feature similarity

Inspired by: GNN-based geo-localization (ICCV 2025), AttenGeo cross-view matching

Architecture:
  Student: ConvNeXt-Tiny + Scene Graph Constructor + CVGMN
  Teacher: DINOv2-Base (frozen)

Dataset: SUES-200 (drone ↔ satellite cross-view geo-localization)
Protocol: 120 train / 80 test, gallery = ALL 200 locations (confusion data)

Usage:
  python exp_geograph.py           # Full training on Kaggle H100
  python exp_geograph.py --test    # Smoke test
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

# === IMPORTS ===
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

# Graph config
GRAPH_DIM     = 256       # Node/edge feature dim
GNN_LAYERS    = 3         # GNN message-passing layers
TOP_K_EDGES   = 8         # k-nearest neighbors for edge construction
MATCH_DIM     = 128       # Graph matching projection dim

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
            if len(idx) < self.k: idx = idx * (self.k//len(idx)+1)
            batch.extend(random.sample(idx, self.k))
            if len(batch) >= self.p*self.k:
                yield batch[:self.p*self.k]; batch = batch[self.p*self.k:]
    def __len__(self): return len(self.locs)//self.p


def get_transforms(mode="train"):
    if mode == "train":
        return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(0.5),
            T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)), T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(), T.Normalize([0.485,.456,.406],[0.229,.224,.225])])
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
        T.Normalize([0.485,.456,.406],[0.229,.224,.225])])


# =============================================================================
# CONVNEXT-TINY BACKBONE (from baseline)
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
        return self.w[:,None,None]*((x-u)/torch.sqrt(s+self.eps)) + self.b[:,None,None]

def drop_path(x, dp=0., tr=False):
    if dp == 0. or not tr: return x
    kp = 1-dp; s = (x.shape[0],)+(1,)*(x.ndim-1)
    rt = kp+torch.rand(s, dtype=x.dtype, device=x.device); rt.floor_()
    return x.div(kp)*rt

class DropPath(nn.Module):
    def __init__(self, dp=None): super().__init__(); self.dp = dp
    def forward(self, x): return drop_path(x, self.dp, self.training)

class ConvNeXtBlock(nn.Module):
    def __init__(self, d, dpr=0., lsi=1e-6):
        super().__init__()
        self.dw = nn.Conv2d(d, d, 7, padding=3, groups=d)
        self.n = LayerNorm(d, 1e-6); self.p1 = nn.Linear(d, 4*d)
        self.act = nn.GELU(); self.p2 = nn.Linear(4*d, d)
        self.g = nn.Parameter(lsi*torch.ones(d)) if lsi>0 else None
        self.dp = DropPath(dpr) if dpr>0 else nn.Identity()
    def forward(self, x):
        s = x; x = self.dw(x); x = x.permute(0,2,3,1)
        x = self.n(x); x = self.p1(x); x = self.act(x); x = self.p2(x)
        if self.g is not None: x = self.g*x
        return s + self.dp(x.permute(0,3,1,2))

class ConvNeXtTiny(nn.Module):
    def __init__(self, ic=3, depths=[3,3,9,3], dims=[96,192,384,768], dpr=0.):
        super().__init__()
        self.dims = dims; self.ds_layers = nn.ModuleList()
        self.ds_layers.append(nn.Sequential(nn.Conv2d(ic, dims[0], 4, 4),
                              LayerNorm(dims[0], 1e-6, "channels_first")))
        for i in range(3):
            self.ds_layers.append(nn.Sequential(LayerNorm(dims[i], 1e-6, "channels_first"),
                                  nn.Conv2d(dims[i], dims[i+1], 2, 2)))
        rates = [x.item() for x in torch.linspace(0, dpr, sum(depths))]; c = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[ConvNeXtBlock(dims[i], rates[c+j]) for j in range(depths[i])]))
            c += depths[i]
        self.norm = nn.LayerNorm(dims[-1], 1e-6)
        self.apply(self._iw)
    def _iw(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.ds_layers[i](x); x = self.stages[i](x); outs.append(x)
        return self.norm(x.mean([-2,-1])), outs

def load_convnext_pretrained(m):
    try:
        ckpt = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth",
            map_location="cpu", check_hash=True)
        m.load_state_dict({k:v for k,v in ckpt["model"].items() if not k.startswith('head')}, strict=False)
        print("Loaded ConvNeXt-Tiny pretrained (ImageNet-22K)")
    except Exception as e: print(f"Could not load: {e}")
    return m


# =============================================================================
# NOVEL COMPONENT 1: SPATIAL SCENE GRAPH CONSTRUCTOR
# =============================================================================
class SpatialSceneGraphConstructor(nn.Module):
    """Builds a scene graph from backbone feature tokens.

    Nodes: Each spatial token (from 7×7 = 49 positions) becomes a graph node
    Edges: k-nearest neighbors in feature space + spatial proximity

    Node features encode WHAT is in that region.
    Edge features encode the spatial RELATIONSHIP between regions.
    """
    def __init__(self, d_input, d_graph, top_k=8):
        super().__init__()
        self.top_k = top_k

        # Node feature projection
        self.node_proj = nn.Sequential(
            nn.Linear(d_input, d_graph),
            nn.LayerNorm(d_graph),
            nn.GELU(),
        )

        # Spatial position encoding for edges
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, d_graph),
        )

        # Edge feature: combines feature similarity + spatial relationship
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_graph * 3, d_graph),
            nn.GELU(),
            nn.Linear(d_graph, d_graph),
        )

    def forward(self, feat_map):
        """
        Args: feat_map: [B, C, H, W] from backbone
        Returns:
            nodes: [B, N, D] node features (N = H*W)
            edges: [B, N, K, D] edge features for top-K neighbors
            adj: [B, N, K] neighbor indices
        """
        B, C, H, W = feat_map.shape
        N = H * W

        # Create nodes
        tokens = feat_map.flatten(2).transpose(1, 2)  # [B, N, C]
        nodes = self.node_proj(tokens)  # [B, N, D]

        # Create spatial coordinates [0,1] range
        ys = torch.linspace(0, 1, H, device=feat_map.device)
        xs = torch.linspace(0, 1, W, device=feat_map.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)  # [N, 2]
        coords = coords.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]

        # Find k-nearest neighbors in feature space
        sim = torch.bmm(F.normalize(nodes, dim=-1),
                        F.normalize(nodes, dim=-1).transpose(1, 2))  # [B, N, N]
        # Mask self-connections
        sim = sim - torch.eye(N, device=sim.device).unsqueeze(0) * 1e9
        _, adj = sim.topk(self.top_k, dim=-1)  # [B, N, K]

        # Compute edge features
        # Gather neighbor node features
        adj_flat = adj.reshape(B, -1)  # [B, N*K]
        neighbor_nodes = torch.gather(
            nodes.unsqueeze(2).expand(-1, -1, self.top_k, -1).reshape(B, N * self.top_k, -1),
            1,
            adj_flat.unsqueeze(-1).expand(-1, -1, nodes.shape[-1])
        ).view(B, N, self.top_k, -1)

        # Actually gather properly
        neighbor_nodes = nodes.gather(
            1, adj.reshape(B, -1, 1).expand(-1, -1, nodes.shape[-1])
        ).view(B, N, self.top_k, -1)  # [B, N, K, D]

        # Spatial displacement
        neighbor_coords = coords.gather(
            1, adj.reshape(B, -1, 1).expand(-1, -1, 2)
        ).view(B, N, self.top_k, 2)  # [B, N, K, 2]
        disp = neighbor_coords - coords.unsqueeze(2)  # [B, N, K, 2]
        spatial_feat = self.pos_mlp(disp)  # [B, N, K, D]

        # Edge features = [source, target, spatial]
        source_expanded = nodes.unsqueeze(2).expand(-1, -1, self.top_k, -1)
        edge_input = torch.cat([source_expanded, neighbor_nodes, spatial_feat], dim=-1)
        edges = self.edge_mlp(edge_input)  # [B, N, K, D]

        return nodes, edges, adj


# =============================================================================
# NOVEL COMPONENT 2: CROSS-VIEW GRAPH MATCHING NETWORK (CVGMN)
# =============================================================================
class GraphAttentionLayer(nn.Module):
    """Graph attention network layer with edge-conditioned message passing."""
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.edge_proj = nn.Linear(d_model, num_heads)  # Edge attention bias
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, nodes, edges, adj):
        """
        Args:
            nodes: [B, N, D]
            edges: [B, N, K, D]
            adj: [B, N, K] neighbor indices
        Returns: updated nodes [B, N, D]
        """
        B, N, D = nodes.shape
        K = adj.shape[-1]
        H = self.num_heads

        # Gather neighbor features
        neighbor_nodes = nodes.gather(
            1, adj.reshape(B, -1, 1).expand(-1, -1, D)
        ).view(B, N, K, D)

        # Multi-head attention
        q = self.q_proj(nodes).view(B, N, 1, H, self.d_head)
        k = self.k_proj(neighbor_nodes).view(B, N, K, H, self.d_head)
        v = self.v_proj(neighbor_nodes).view(B, N, K, H, self.d_head)

        # Attention scores + edge bias
        attn = (q * k).sum(-1) / math.sqrt(self.d_head)  # [B, N, K, H]
        edge_bias = self.edge_proj(edges)  # [B, N, K, H]
        attn = attn + edge_bias
        attn = F.softmax(attn, dim=2)  # Softmax over neighbors

        # Aggregate
        out = (attn.unsqueeze(-1) * v).sum(2)  # [B, N, H, d_head]
        out = out.reshape(B, N, D)
        out = self.out_proj(out)

        # Residual + FFN
        nodes = self.norm(nodes + out)
        nodes = self.norm2(nodes + self.ffn(nodes))
        return nodes


class CrossViewGraphMatcher(nn.Module):
    """Cross-View Graph Matching Network.

    Performs graph-level matching between drone and satellite scene graphs
    using differentiable assignment matrices.

    Novel: Uses Sinkhorn-normalized attention to compute soft
    node-to-node correspondences across views.
    """
    def __init__(self, d_model, match_dim=128, sinkhorn_iters=5):
        super().__init__()
        self.match_proj = nn.Linear(d_model, match_dim)
        self.sinkhorn_iters = sinkhorn_iters

    def forward(self, drone_nodes, sat_nodes):
        """
        Args:
            drone_nodes: [B, N_d, D]
            sat_nodes: [B, N_s, D]
        Returns:
            assignment: [B, N_d, N_s] soft correspondence
            match_score: [B] matching score
        """
        # Project to matching space
        d = F.normalize(self.match_proj(drone_nodes), dim=-1)
        s = F.normalize(self.match_proj(sat_nodes), dim=-1)

        # Cost matrix (cosine similarity)
        M = torch.bmm(d, s.transpose(1, 2))  # [B, N_d, N_s]

        # Sinkhorn normalization for doubly-stochastic assignment
        for _ in range(self.sinkhorn_iters):
            M = M - torch.logsumexp(M, dim=2, keepdim=True)
            M = M - torch.logsumexp(M, dim=1, keepdim=True)
        assignment = torch.exp(M)

        # Matching score = trace of assignment × similarity
        match_score = (assignment * torch.bmm(d, s.transpose(1, 2))).sum(dim=[1, 2])
        match_score = match_score / min(d.shape[1], s.shape[1])

        return assignment, match_score


class CVGMN(nn.Module):
    """Full Cross-View Graph Matching Network.

    Pipeline:
      1. Independent GNN processing of drone and satellite graphs
      2. Cross-graph message passing for correspondence learning
      3. Differentiable graph matching for alignment
    """
    def __init__(self, d_model, num_layers=3, num_heads=4, match_dim=128):
        super().__init__()

        # Independent GNN layers
        self.drone_gnn = nn.ModuleList([
            GraphAttentionLayer(d_model, num_heads) for _ in range(num_layers)])
        self.sat_gnn = nn.ModuleList([
            GraphAttentionLayer(d_model, num_heads) for _ in range(num_layers)])

        # Cross-graph attention (every other layer)
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            for _ in range(num_layers // 2 + 1)])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers // 2 + 1)])

        # Graph matcher
        self.matcher = CrossViewGraphMatcher(d_model, match_dim)

    def forward(self, d_nodes, d_edges, d_adj, s_nodes, s_edges, s_adj):
        """
        Returns:
            d_nodes, s_nodes: refined graph node features
            assignment: soft correspondence matrix
            match_score: matching confidence
        """
        cross_idx = 0
        for i in range(len(self.drone_gnn)):
            # Independent GNN
            d_nodes = self.drone_gnn[i](d_nodes, d_edges, d_adj)
            s_nodes = self.sat_gnn[i](s_nodes, s_edges, s_adj)

            # Cross-graph attention at alternate layers
            if i % 2 == 1 and cross_idx < len(self.cross_attn):
                d_cross, _ = self.cross_attn[cross_idx](d_nodes, s_nodes, s_nodes)
                s_cross, _ = self.cross_attn[cross_idx](s_nodes, d_nodes, d_nodes)
                d_nodes = self.cross_norms[cross_idx](d_nodes + d_cross)
                s_nodes = self.cross_norms[cross_idx](s_nodes + s_cross)
                cross_idx += 1

        # Graph matching
        assignment, match_score = self.matcher(d_nodes, s_nodes)

        return d_nodes, s_nodes, assignment, match_score


# =============================================================================
# NOVEL COMPONENT 3: TOPOLOGY-PRESERVING LOSS
# =============================================================================
class TopologyPreservingLoss(nn.Module):
    """Loss that enforces structural consistency in graph matching.

    Beyond feature similarity, ensures that the spatial relationship
    structure is preserved: if node A is adjacent to node B in the
    drone graph, then their matched counterparts in the satellite graph
    should also be adjacent.
    """
    def __init__(self):
        super().__init__()

    def forward(self, d_nodes, s_nodes, assignment, d_adj, s_adj):
        """
        Args:
            d_nodes: [B, N_d, D], s_nodes: [B, N_s, D]
            assignment: [B, N_d, N_s] soft correspondence
            d_adj, s_adj: [B, N, K] adjacency
        """
        B, Nd, D = d_nodes.shape
        Ns = s_nodes.shape[1]
        K = d_adj.shape[-1]

        # Build adjacency matrices from k-NN indices
        d_adj_matrix = torch.zeros(B, Nd, Nd, device=d_nodes.device)
        for k in range(K):
            idx = d_adj[:, :, k]  # [B, Nd]
            d_adj_matrix.scatter_(2, idx.unsqueeze(2), 1.0)

        s_adj_matrix = torch.zeros(B, Ns, Ns, device=s_nodes.device)
        Ks = s_adj.shape[-1]
        for k in range(Ks):
            idx = s_adj[:, :, k]
            s_adj_matrix.scatter_(2, idx.unsqueeze(2), 1.0)

        # Transport adjacency through assignment: P^T * A_d * P should ≈ A_s
        transported = torch.bmm(torch.bmm(assignment.transpose(1, 2), d_adj_matrix), assignment)
        topo_loss = F.mse_loss(transported, s_adj_matrix)

        return topo_loss


# =============================================================================
# GEOGRAPH MODEL
# =============================================================================
class ClassificationHead(nn.Module):
    def __init__(self, d, nc, h=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU(True),
                                nn.Dropout(0.5), nn.Linear(h,nc))
    def forward(self, x): return self.fc(self.pool(x).flatten(1))


class GeoGraphStudent(nn.Module):
    """GeoGraph = ConvNeXt-Tiny + Scene Graph + CVGMN.

    Both a global branch (standard pooling) and a graph branch (topology-aware)
    produce embeddings that are fused for the final descriptor.
    """
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = ConvNeXtTiny(dpr=0.1)
        self.backbone = load_convnext_pretrained(self.backbone)

        self.aux_heads = nn.ModuleList([
            ClassificationHead(d, num_classes) for d in [96, 192, 384, 768]])

        # Scene graph constructor
        self.graph_constructor = SpatialSceneGraphConstructor(768, GRAPH_DIM, TOP_K_EDGES)

        # Cross-view graph matching
        self.cvgmn = CVGMN(GRAPH_DIM, GNN_LAYERS, num_heads=4, match_dim=MATCH_DIM)

        # Graph embedding
        self.graph_embed = nn.Sequential(
            nn.Linear(GRAPH_DIM, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(True))

        # Global embedding
        self.global_embed = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(True))

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim))

        self.classifier = nn.Linear(embed_dim, num_classes)

    def _extract(self, x):
        final, stages = self.backbone(x)
        nodes, edges, adj = self.graph_constructor(stages[-1])
        return final, stages, nodes, edges, adj

    def forward_pair(self, drone_x, sat_x):
        """Training forward with paired cross-view graph matching."""
        d_final, d_stages, d_nodes, d_edges, d_adj = self._extract(drone_x)
        s_final, s_stages, s_nodes, s_edges, s_adj = self._extract(sat_x)

        d_stage_logits = [h(f) for h, f in zip(self.aux_heads, d_stages)]
        s_stage_logits = [h(f) for h, f in zip(self.aux_heads, s_stages)]

        # Cross-view graph matching
        d_nodes_r, s_nodes_r, assignment, match_score = self.cvgmn(
            d_nodes, d_edges, d_adj, s_nodes, s_edges, s_adj)

        # Graph embeddings (pool refined nodes)
        d_graph = self.graph_embed(d_nodes_r.mean(1))
        s_graph = self.graph_embed(s_nodes_r.mean(1))

        # Global embeddings
        d_global = self.global_embed(d_final)
        s_global = self.global_embed(s_final)

        # Fuse
        d_fused = self.fusion(torch.cat([d_global, d_graph], 1))
        s_fused = self.fusion(torch.cat([s_global, s_graph], 1))

        return {
            'drone': {
                'embedding_normed': F.normalize(d_fused, 2, 1),
                'logits': self.classifier(d_fused),
                'stage_logits': d_stage_logits,
                'final_feature': d_final,
                'nodes': d_nodes_r, 'adj': d_adj,
            },
            'sat': {
                'embedding_normed': F.normalize(s_fused, 2, 1),
                'logits': self.classifier(s_fused),
                'stage_logits': s_stage_logits,
                'final_feature': s_final,
                'nodes': s_nodes_r, 'adj': s_adj,
            },
            'assignment': assignment,
            'match_score': match_score,
        }

    def forward(self, x, return_all=False):
        """Single-view inference (no cross-view partner)."""
        final, stages, nodes, edges, adj = self._extract(x)
        stage_logits = [h(f) for h, f in zip(self.aux_heads, stages)]

        # Self-matching for single-view GNN refinement
        nodes_r = nodes
        for layer in self.cvgmn.drone_gnn:
            nodes_r = layer(nodes_r, edges, adj)

        graph_emb = self.graph_embed(nodes_r.mean(1))
        global_emb = self.global_embed(final)
        fused = self.fusion(torch.cat([global_emb, graph_emb], 1))
        fused_norm = F.normalize(fused, 2, 1)
        logits = self.classifier(fused)

        if return_all:
            return {'embedding_normed': fused_norm, 'logits': logits,
                    'stage_logits': stage_logits, 'final_feature': final}
        return fused_norm, logits

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
        print("  DINOv2 loaded!")
    @torch.no_grad()
    def forward(self, x):
        t = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks: t = blk(t)
        return self.model.norm(t)[:, 0]

class TripletLoss(nn.Module):
    def __init__(self, m=0.3): super().__init__(); self.m = m
    def forward(self, e, l):
        d = torch.cdist(e, e, 2); lb = l.view(-1,1)
        p = lb.eq(lb.T).float(); n = lb.ne(lb.T).float()
        return F.relu((d*p).max(1)[0] - (d*n+p*999).min(1)[0] + self.m).mean()

class SymNCE(nn.Module):
    def __init__(self, t=0.07): super().__init__(); self.t = t
    def forward(self, d, s, l):
        d = F.normalize(d,1); s = F.normalize(s,1)
        sim = d@s.T/self.t; lb = l.view(-1,1); pm = lb.eq(lb.T).float()
        l1 = -(F.log_softmax(sim,1)*pm).sum(1)/pm.sum(1).clamp(1)
        l2 = -(F.log_softmax(sim.T,1)*pm).sum(1)/pm.sum(1).clamp(1)
        return 0.5*(l1.mean()+l2.mean())

class SelfDist(nn.Module):
    def __init__(self, T=4.0): super().__init__(); self.T = T
    def forward(self, sl):
        loss = 0.; f = sl[-1]; w = [.1,.2,.3,.4]
        for i in range(len(sl)-1):
            loss += w[i]*(self.T**2)*F.kl_div(F.log_softmax(f/self.T,1), F.softmax(sl[i]/self.T,1), reduction='batchmean')
        return loss

class UAPA(nn.Module):
    def __init__(self, T0=4.0): super().__init__(); self.T0 = T0
    def forward(self, dl, sl):
        Ud = -(F.softmax(dl,1)*F.log_softmax(dl,1)).sum(1).mean()
        Us = -(F.softmax(sl,1)*F.log_softmax(sl,1)).sum(1).mean()
        T = self.T0*(1+torch.sigmoid(Ud-Us))
        return (T**2)*F.kl_div(F.log_softmax(dl/T,1), F.softmax(sl/T,1), reduction='batchmean')


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate(model, test_ds, device):
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
    _, idx = (feats@sf.T).sort(1, descending=True)
    N = len(feats); r1=r5=r10=0; ap=0.
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
    print("="*70); print("GeoGraph: Scene Graph Matching Geo-Localization"); print("="*70)
    train_ds = SUES200Dataset(args.data_root, "train", transform=get_transforms("train"))
    test_ds = SUES200Dataset(args.data_root, "test", transform=get_transforms("test"))
    sampler = PKSampler(train_ds, 8, max(2, BATCH_SIZE//8))
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    model = GeoGraphStudent(len(TRAIN_LOCS)).to(DEVICE)
    print(f"  Student: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    try: teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except: teacher = None
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    trip = TripletLoss(MARGIN); nce = SymNCE(.07)
    sd = SelfDist(4.); uapa = UAPA(4.); topo = TopologyPreservingLoss()
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler(enabled=AMP_ENABLED); best_r1 = 0.

    for epoch in range(EPOCHS):
        model.train()
        if epoch < WARMUP_EPOCHS: lr = LR*(epoch+1)/WARMUP_EPOCHS
        else:
            pr = (epoch-WARMUP_EPOCHS)/max(1, EPOCHS-WARMUP_EPOCHS)
            lr = 1e-6+.5*(LR-1e-6)*(1+math.cos(math.pi*pr))
        for pg in opt.param_groups: pg['lr'] = lr
        tl = 0.; lp = defaultdict(float); nb = 0
        for bi, batch in enumerate(loader):
            drone, sat = batch['drone'].to(DEVICE), batch['satellite'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            opt.zero_grad()
            with autocast(enabled=AMP_ENABLED):
                out = model.forward_pair(drone, sat)
                do, so = out['drone'], out['sat']
                L = {}
                c = ce(do['logits'], labels)+ce(so['logits'], labels)
                for sl in do['stage_logits']: c += .25*ce(sl, labels)
                for sl in so['stage_logits']: c += .25*ce(sl, labels)
                L['ce'] = c
                L['trip'] = trip(do['embedding_normed'], labels)+trip(so['embedding_normed'], labels)
                L['nce'] = nce(do['embedding_normed'], so['embedding_normed'], labels)
                L['sd'] = .5*(sd(do['stage_logits'])+sd(so['stage_logits']))
                L['uapa'] = .2*uapa(do['logits'], so['logits'])
                # *** NOVEL: Topology-preserving loss ***
                L['topo'] = .3*topo(do['nodes'], so['nodes'], out['assignment'], do['adj'], so['adj'])
                # Match score loss (positive pairs should have high match score)
                L['match'] = .2*(1 - out['match_score'].mean())
                if teacher is not None:
                    with torch.no_grad(): td = teacher(drone); ts = teacher(sat)
                    df = F.normalize(do['final_feature'],1); sf = F.normalize(so['final_feature'],1)
                    tdn = F.normalize(td,1); tsn = F.normalize(ts,1)
                    L['cdist'] = .3*(F.mse_loss(df,tdn)+F.mse_loss(sf,tsn)+
                                     (1-F.cosine_similarity(df,tdn).mean())+
                                     (1-F.cosine_similarity(sf,tsn).mean()))
                total = sum(L.values())
            scaler.scale(total).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(opt); scaler.update()
            tl += total.item(); nb += 1
            for k,v in L.items(): lp[k] += v.item()
            if bi%10==0: print(f"  B{bi}/{len(loader)} L={total.item():.4f}")
        nb = max(1, nb)
        print(f"\nEp {epoch+1}/{EPOCHS} LR={lr:.6f} AvgL={tl/nb:.4f}")
        for k,v in sorted(lp.items()): print(f"  {k}: {v/nb:.4f}")
        if (epoch+1)%EVAL_FREQ==0 or epoch==EPOCHS-1:
            rec, ap = evaluate(model, test_ds, DEVICE)
            print(f"  R@1:{rec['R@1']:.2f}% R@5:{rec['R@5']:.2f}% R@10:{rec['R@10']:.2f}% AP:{ap:.2f}%")
            if rec['R@1'] > best_r1:
                best_r1 = rec['R@1']
                torch.save({'epoch':epoch,'model':model.state_dict(),'r1':best_r1},
                           os.path.join(OUTPUT_DIR,'geograph_best.pth'))
                print(f"  *** Best R@1={best_r1:.2f}% ***")
    print(f"\nDone! Best R@1={best_r1:.2f}%")


def smoke_test():
    print("="*50); print("SMOKE TEST — GeoGraph"); print("="*50)
    dev = DEVICE; m = GeoGraphStudent(10).to(dev)
    print(f"✓ Model: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
    d = torch.randn(4,3,224,224,device=dev); s = torch.randn(4,3,224,224,device=dev)
    lab = torch.tensor([0,0,1,1],device=dev)
    out = m.forward_pair(d, s)
    print(f"✓ Pair: emb={out['drone']['embedding_normed'].shape}, "
          f"assign={out['assignment'].shape}, score={out['match_score'].shape}")
    e, l = m(d); print(f"✓ Single: emb={e.shape}")
    loss = nn.CrossEntropyLoss()(out['drone']['logits'], lab); loss.backward()
    print(f"✓ Backward OK"); print("\n✅ ALL TESTS PASSED!")


def main():
    global EPOCHS, BATCH_SIZE, DATA_ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--data_root", type=str, default=DATA_ROOT)
    p.add_argument("--test", action="store_true")
    args, _ = p.parse_known_args()
    EPOCHS=args.epochs; BATCH_SIZE=args.batch_size; DATA_ROOT=args.data_root
    args.data_root = DATA_ROOT
    if args.test: smoke_test(); return
    os.makedirs(OUTPUT_DIR, exist_ok=True); train(args)

if __name__ == "__main__": main()
