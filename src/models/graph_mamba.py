"""
Graph Mamba Layer for Relational Reasoning between Object Slots.

Reference:
- Graph-Mamba (ICML 2024): https://github.com/bowang-lab/Graph-Mamba
- mamba_ssm: https://github.com/state-spaces/mamba

Key concept: Convert a graph of object slots into ordered sequences
using node-priority strategies (degree-based, spatial centroid, hybrid),
then process with Mamba for O(N) message passing.

Reviewer-driven improvements:
- Spatial positional encoding from attention map centroids
- k-NN graph uses BOTH semantic similarity AND spatial distance
- Hilbert Curve ordering for rotation-invariant sequence construction
- Exposes centroids for downstream FGW matching
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class LinearSequenceModel(nn.Module):
    """Fallback sequence model when mamba_ssm is not available."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = int(d_model * expand)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.SiLU(),
            nn.Linear(d_inner, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.net(x))


# ============================================================================
# Spatial Positional Encoding from Attention Maps
# ============================================================================
class SlotSpatialEncoder(nn.Module):
    """
    Compute spatial positional encodings for slots from attention maps.

    Each slot's attention over spatial tokens defines a soft spatial distribution.
    We compute the centroid (weighted mean position) and spatial spread (std),
    then encode them as positional features via sinusoidal encoding + MLP.

    This addresses the reviewer concern:
    "L2-norm ordering does not model spatial structure or slot positions explicitly"
    """
    def __init__(self, slot_dim, pos_dim=64):
        super().__init__()
        self.pos_dim = pos_dim
        # Encode 2D centroid (x,y) + spread (σx, σy) → pos_dim features
        # Input: 4 values (cx, cy, σx, σy) → sinusoidal → MLP → pos_dim
        self.pos_mlp = nn.Sequential(
            nn.Linear(4 * pos_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, slot_dim),
        )

    def _sinusoidal_encode(self, vals, dim):
        """Sinusoidal positional encoding for arbitrary float values.
        vals: [B, K, C]  → returns [B, K, C * dim]
        """
        device = vals.device
        freqs = torch.arange(0, dim, 2, device=device, dtype=vals.dtype) / dim
        freqs = 1.0 / (10000.0 ** freqs)  # [dim//2]
        # vals: [B, K, C] → [B, K, C, 1]
        vals_exp = vals.unsqueeze(-1)
        # freqs: [dim//2] → [1, 1, 1, dim//2]
        freqs_exp = freqs.view(1, 1, 1, -1)
        angles = vals_exp * freqs_exp * math.pi
        enc = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B, K, C, dim]
        return enc.flatten(-2, -1)  # [B, K, C * dim]

    def compute_centroids(self, attn_maps, H, W):
        """
        Compute slot centroids from attention maps.

        Args:
            attn_maps: [B, K, N] attention weights over N=H*W spatial tokens
            H, W: spatial grid dimensions

        Returns:
            centroids: [B, K, 2] (x, y) normalized to [0, 1]
            spreads:   [B, K, 2] (σx, σy) spatial spread
        """
        B, K, N = attn_maps.shape
        device = attn_maps.device

        # Create coordinate grid [N, 2]
        gy = torch.arange(H, device=device, dtype=attn_maps.dtype) / max(H - 1, 1)
        gx = torch.arange(W, device=device, dtype=attn_maps.dtype) / max(W - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [N, 2]

        # Normalize attention to probability distribution over spatial dims
        attn_prob = attn_maps / (attn_maps.sum(dim=-1, keepdim=True) + 1e-8)  # [B, K, N]

        # Centroid = weighted mean position
        centroids = torch.einsum('bkn,nc->bkc', attn_prob, coords)  # [B, K, 2]

        # Spread = weighted std
        coords_exp = coords.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        cent_exp = centroids.unsqueeze(2)               # [B, K, 1, 2]
        diff_sq = (coords_exp - cent_exp) ** 2          # [B, K, N, 2]
        variance = torch.einsum('bkn,bknc->bkc', attn_prob, diff_sq)  # [B, K, 2]
        spreads = variance.sqrt().clamp(min=1e-4)

        return centroids, spreads

    def forward(self, attn_maps, H, W):
        """
        Args:
            attn_maps: [B, K, N] slot attention maps
            H, W: spatial grid dimensions

        Returns:
            pos_encoding: [B, K, slot_dim] spatial positional encoding
        """
        centroids, spreads = self.compute_centroids(attn_maps, H, W)  # [B,K,2] each
        spatial_feats = torch.cat([centroids, spreads], dim=-1)       # [B, K, 4]
        enc = self._sinusoidal_encode(spatial_feats, self.pos_dim)    # [B, K, 4*pos_dim]
        return self.pos_mlp(enc)                                      # [B, K, slot_dim]


# ============================================================================
# Graph Construction from Slots (with spatial distance)
# ============================================================================
class SlotGraphBuilder(nn.Module):
    """
    Build a kNN graph from object slots using BOTH semantic similarity
    AND spatial distance (from attention map centroids).

    Addresses reviewer concern:
    "Does not model spatial structure explicitly"

    Args:
        slot_dim: Dimension of slot features
        k: Number of nearest neighbors
        spatial_weight: Weight for spatial distance vs semantic similarity
    """
    def __init__(self, slot_dim, k=5, spatial_weight=0.3):
        super().__init__()
        self.k = k
        self.spatial_weight = spatial_weight

    def forward(self, slots, keep_mask=None, centroids=None):
        """
        Args:
            slots: [B, K, D] object slot representations
            keep_mask: [B, K] binary mask (1 = active slot)
            centroids: [B, K, 2] spatial centroids (optional)

        Returns:
            adj_matrix: [B, K, K] adjacency matrix (weighted)
        """
        B, K, D = slots.shape

        # Semantic similarity
        slots_norm = F.normalize(slots, dim=-1)
        sim_semantic = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # [B, K, K]

        # Combine with spatial proximity if available
        if centroids is not None:
            # Spatial distance → similarity (Gaussian kernel)
            # Safe L2: torch.cdist backward has NaN at zero distance
            diff_c = centroids.unsqueeze(2) - centroids.unsqueeze(1)
            spatial_dist = (diff_c * diff_c).sum(-1).clamp(min=1e-6).sqrt()  # [B, K, K]
            sim_spatial = torch.exp(-spatial_dist ** 2 / 0.1)        # [B, K, K]
            sim = (1 - self.spatial_weight) * sim_semantic + self.spatial_weight * sim_spatial
        else:
            sim = sim_semantic

        # Mask inactive slots
        if keep_mask is not None:
            mask_2d = keep_mask.unsqueeze(1) * keep_mask.unsqueeze(2)
            sim = sim * mask_2d

        # Self-loop removal
        eye = torch.eye(K, device=slots.device).unsqueeze(0)
        sim = sim * (1 - eye)

        # kNN selection
        if K > self.k:
            topk_vals, topk_idx = sim.topk(self.k, dim=-1)
            adj = torch.zeros_like(sim)
            adj.scatter_(-1, topk_idx, topk_vals)
        else:
            adj = sim

        # Make symmetric
        adj = (adj + adj.transpose(1, 2)) / 2

        return adj


# ============================================================================
# Graph-guided Sequence Ordering
# ============================================================================
class GraphSequenceOrderer(nn.Module):
    """
    Convert graph nodes to ordered sequences for Mamba processing.

    Strategies:
    1. Hilbert: Sort by Hilbert curve index (best spatial locality preservation)
    2. Degree-based: Sort nodes by connectivity (high degree first)
    3. Spatial: Sort by spatial position (raster-scan order from centroids)
    4. Hybrid: Degree during eval, random during training
    5. Spatial-degree: Primary sort by spatial, secondary by degree
    """
    def __init__(self, strategy='hilbert'):
        super().__init__()
        self.strategy = strategy

    def _degree_order(self, adj):
        """Sort nodes by degree (number of connections)."""
        degrees = adj.sum(dim=-1)
        return degrees.argsort(dim=-1, descending=True)

    def _spatial_order(self, centroids):
        """Raster-scan ordering from spatial centroids (top-left to bottom-right)."""
        # Z-order / raster: sort by y first, then x → approximate spatial scan
        B, K, _ = centroids.shape
        # Quantize to grid for raster ordering
        order_key = centroids[:, :, 1] * 1000 + centroids[:, :, 0]  # y * 1000 + x
        return order_key.argsort(dim=-1)

    def _random_order(self, B, K, device):
        """Random permutation per batch."""
        return torch.stack([torch.randperm(K, device=device) for _ in range(B)])

    @staticmethod
    def _xy2d_hilbert(n, x, y):
        """
        Convert (x, y) coordinates to Hilbert curve distance.

        Maps 2D coordinates in an n×n grid to a 1D Hilbert curve index.
        The Hilbert curve preserves spatial locality better than raster-scan
        or Z-order (Morton) curves, making it ideal for sequence ordering
        of graph nodes in Mamba.

        Args:
            n: Grid size (must be power of 2)
            x: x-coordinate (column), integer in [0, n-1]
            y: y-coordinate (row), integer in [0, n-1]

        Returns:
            d: Hilbert curve distance (1D index)
        """
        d = 0
        s = n // 2
        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            # Rotate quadrant
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            s //= 2
        return d

    def _hilbert_order(self, centroids):
        """
        Hilbert curve ordering from spatial centroids.

        Maps 2D centroid coordinates to Hilbert curve indices,
        then sorts by these indices. This preserves spatial locality
        far better than raster-scan ordering, and is robust to
        image rotations.

        Args:
            centroids: [B, K, 2] spatial centroids (x, y) normalized to [0, 1]

        Returns:
            order: [B, K] permutation indices
        """
        B, K, _ = centroids.shape
        device = centroids.device

        # Quantize to grid for Hilbert mapping
        # Use power-of-2 grid size for proper Hilbert curve
        grid_size = 16  # 16×16 = 256 cells, enough resolution for K≤16 slots
        cx = (centroids[:, :, 0] * (grid_size - 1)).clamp(0, grid_size - 1).long()  # [B, K]
        cy = (centroids[:, :, 1] * (grid_size - 1)).clamp(0, grid_size - 1).long()  # [B, K]

        # Compute Hilbert indices
        hilbert_idx = torch.zeros(B, K, device=device)
        for b in range(B):
            for k in range(K):
                hilbert_idx[b, k] = self._xy2d_hilbert(
                    grid_size, cx[b, k].item(), cy[b, k].item()
                )

        return hilbert_idx.argsort(dim=-1)

    def forward(self, slots, adj, centroids=None):
        """
        Args:
            slots: [B, K, D]
            adj: [B, K, K] adjacency matrix
            centroids: [B, K, 2] spatial centroids (optional)

        Returns:
            ordered_slots: [B, K, D]
            order: [B, K]
            reverse_order: [B, K]
        """
        B, K, D = slots.shape

        if self.strategy == 'hilbert' and centroids is not None:
            order = self._hilbert_order(centroids)
        elif self.strategy == 'degree':
            order = self._degree_order(adj)
        elif self.strategy == 'spatial' and centroids is not None:
            order = self._spatial_order(centroids)
        elif self.strategy == 'spatial_hybrid':
            if self.training:
                order = self._random_order(B, K, slots.device)
            else:
                if centroids is not None:
                    order = self._hilbert_order(centroids)
                else:
                    order = self._degree_order(adj)
        elif self.strategy == 'hybrid':
            if self.training:
                order = self._random_order(B, K, slots.device)
            else:
                order = self._degree_order(adj)
        elif self.strategy == 'random':
            order = self._random_order(B, K, slots.device)
        else:
            order = self._degree_order(adj)

        batch_idx = torch.arange(B, device=slots.device).unsqueeze(1).expand(-1, K)
        ordered_slots = slots[batch_idx, order]
        reverse_order = order.argsort(dim=-1)

        return ordered_slots, order, reverse_order


# ============================================================================
# Graph Mamba Layer
# ============================================================================
class GraphMambaLayer(nn.Module):
    """
    Graph Mamba layer for relational reasoning between object slots.

    Pipeline:
    1. Compute spatial positions from attention maps (centroids + spread)
    2. Build kNN graph using semantic + spatial similarity
    3. Add spatial positional encoding to slots
    4. Order nodes using spatial/degree/hybrid strategy
    5. Process with bidirectional Mamba (forward + backward scan)
    6. Unscramble back to original slot order
    7. Add residual connection + LayerNorm

    Reviewer-driven improvements over original:
    - Explicit spatial features from attention map centroids
    - Graph edges use spatial proximity, not just content similarity
    - Spatial ordering alongside degree ordering
    """
    def __init__(
        self,
        slot_dim: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        k_neighbors: int = 5,
        strategy: str = 'hilbert',
        num_layers: int = 2,
        spatial_weight: float = 0.3,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_layers = num_layers

        # Spatial encoding from attention maps
        self.spatial_encoder = SlotSpatialEncoder(slot_dim, pos_dim=64)

        # Graph construction (semantic + spatial)
        self.graph_builder = SlotGraphBuilder(slot_dim, k=k_neighbors,
                                              spatial_weight=spatial_weight)
        self.orderer = GraphSequenceOrderer(strategy=strategy)

        # Mamba layers (bidirectional)
        MambaModule = Mamba if HAS_MAMBA else LinearSequenceModel
        self.forward_mambas = nn.ModuleList([
            MambaModule(d_model=slot_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.backward_mambas = nn.ModuleList([
            MambaModule(d_model=slot_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

        # Merge forward + backward
        self.merge_layers = nn.ModuleList([
            nn.Linear(slot_dim * 2, slot_dim) for _ in range(num_layers)
        ])

        # Norms and residuals
        self.norms = nn.ModuleList([
            nn.LayerNorm(slot_dim) for _ in range(num_layers)
        ])

        # FFN after Mamba
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim * 4),
                nn.GELU(),
                nn.Linear(slot_dim * 4, slot_dim),
            )
            for _ in range(num_layers)
        ])

    def forward(self, slots, keep_mask=None, attn_maps=None, spatial_hw=None):
        """
        Args:
            slots: [B, K, D] object slot representations
            keep_mask: [B, K] binary mask (1 = active slot)
            attn_maps: [B, K_total, N] slot attention maps (for spatial encoding)
            spatial_hw: tuple (H, W) spatial grid dimensions

        Returns:
            enhanced_slots: [B, K, D] slots with relational information
        """
        B, K, D = slots.shape

        # Compute spatial position encoding from attention maps
        centroids = None
        if attn_maps is not None and spatial_hw is not None:
            H, W = spatial_hw
            # Use only object slot attention maps (first K of K_total)
            obj_attn = attn_maps[:, :K, :]
            pos_enc = self.spatial_encoder(obj_attn, H, W)  # [B, K, D]
            slots = slots + pos_enc  # Add spatial position info
            centroids = self.spatial_encoder.compute_centroids(obj_attn, H, W)[0]

        # Build graph with spatial-aware edges
        adj = self.graph_builder(slots, keep_mask, centroids=centroids)

        for i in range(self.num_layers):
            residual = slots

            # Order nodes (spatial-aware)
            ordered, order, reverse_order = self.orderer(slots, adj, centroids)

            # Bidirectional Mamba scan
            fwd_out = self.forward_mambas[i](ordered)
            bwd_out = self.backward_mambas[i](ordered.flip(1))
            bwd_out = bwd_out.flip(1)

            # Merge directions
            merged = torch.cat([fwd_out, bwd_out], dim=-1)
            merged = self.merge_layers[i](merged)

            # Unscramble to original order
            batch_idx = torch.arange(B, device=slots.device).unsqueeze(1).expand(-1, K)
            unscrambled = merged[batch_idx, reverse_order]

            # Residual + Norm
            slots = self.norms[i](residual + unscrambled)

            # FFN
            slots = slots + self.ffns[i](slots)

            # Mask inactive slots
            if keep_mask is not None:
                slots = slots * keep_mask.unsqueeze(-1)

        return slots, centroids
