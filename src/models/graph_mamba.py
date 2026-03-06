"""
Graph Mamba Layer for Relational Reasoning between Object Slots.

Reference:
- Graph-Mamba (ICML 2024): https://github.com/bowang-lab/Graph-Mamba
- mamba_ssm: https://github.com/state-spaces/mamba

Key concept: Convert a graph of object slots into ordered sequences
using node-priority strategies (degree-based, semantic similarity),
then process with Mamba for O(N) message passing.
"""

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
# Graph Construction from Slots
# ============================================================================
class SlotGraphBuilder(nn.Module):
    """
    Build a kNN graph from object slots based on semantic similarity.

    Each slot becomes a node. Edges connect semantically similar slots,
    weighted by cosine similarity. This captures spatial relationships
    like "Building A is near Intersection B".
    """
    def __init__(self, slot_dim, k=5):
        super().__init__()
        self.k = k
        self.edge_proj = nn.Linear(slot_dim * 2, slot_dim)

    def forward(self, slots, keep_mask=None):
        """
        Args:
            slots: [B, K, D] object slot representations
            keep_mask: [B, K] binary mask (1 = active slot)

        Returns:
            adj_matrix: [B, K, K] adjacency matrix (weighted)
            edge_features: [B, K, K, D] edge features for message passing
        """
        B, K, D = slots.shape

        # Compute pairwise cosine similarity
        slots_norm = F.normalize(slots, dim=-1)
        sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # [B, K, K]

        # Mask inactive slots
        if keep_mask is not None:
            mask_2d = keep_mask.unsqueeze(1) * keep_mask.unsqueeze(2)  # [B, K, K]
            sim = sim * mask_2d

        # Self-loop removal
        eye = torch.eye(K, device=slots.device).unsqueeze(0)
        sim = sim * (1 - eye)

        # kNN selection: keep top-k neighbors per node
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

    Strategies (from Graph-Mamba paper):
    1. Degree-based: Sort nodes by connectivity (high degree → processed first)
    2. Random permutation: Stochastic ordering for robustness
    3. Hybrid: Degree sort during eval, random during training

    We use Hybrid strategy for our CVGL task.
    """
    def __init__(self, strategy='hybrid'):
        super().__init__()
        self.strategy = strategy

    def _degree_order(self, adj):
        """Sort nodes by degree (number of connections)."""
        degrees = adj.sum(dim=-1)  # [B, K]
        return degrees.argsort(dim=-1, descending=True)

    def _random_order(self, B, K, device):
        """Random permutation per batch."""
        return torch.stack([torch.randperm(K, device=device) for _ in range(B)])

    def forward(self, slots, adj):
        """
        Args:
            slots: [B, K, D]
            adj: [B, K, K] adjacency matrix

        Returns:
            ordered_slots: [B, K, D] reordered slots
            order: [B, K] permutation indices
            reverse_order: [B, K] inverse permutation
        """
        B, K, D = slots.shape

        if self.strategy == 'degree':
            order = self._degree_order(adj)
        elif self.strategy == 'random':
            order = self._random_order(B, K, slots.device)
        elif self.strategy == 'hybrid':
            if self.training:
                order = self._random_order(B, K, slots.device)
            else:
                order = self._degree_order(adj)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Reorder slots
        batch_idx = torch.arange(B, device=slots.device).unsqueeze(1).expand(-1, K)
        ordered_slots = slots[batch_idx, order]

        # Compute reverse order for unscrambling
        reverse_order = order.argsort(dim=-1)

        return ordered_slots, order, reverse_order


# ============================================================================
# Graph Mamba Layer
# ============================================================================
class GraphMambaLayer(nn.Module):
    """
    Graph Mamba layer for relational reasoning between object slots.

    Pipeline:
    1. Build kNN graph from slots
    2. Order nodes using degree/hybrid strategy
    3. Process with bidirectional Mamba (forward + backward scan)
    4. Unscramble back to original slot order
    5. Add residual connection + LayerNorm

    Args:
        slot_dim: Dimension of slot representations
        d_state: Mamba SSM state dimension
        d_conv: Mamba convolution width
        expand: Mamba expansion factor
        k_neighbors: Number of neighbors in kNN graph
        strategy: Ordering strategy ('degree', 'random', 'hybrid')
        num_layers: Number of stacked Graph Mamba layers
    """
    def __init__(
        self,
        slot_dim: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        k_neighbors: int = 5,
        strategy: str = 'hybrid',
        num_layers: int = 2,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_layers = num_layers

        # Graph construction
        self.graph_builder = SlotGraphBuilder(slot_dim, k=k_neighbors)
        self.orderer = GraphSequenceOrderer(strategy=strategy)

        # Mamba layers (bidirectional: forward + backward)
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
            nn.Linear(slot_dim * 2, slot_dim)
            for _ in range(num_layers)
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

    def forward(self, slots, keep_mask=None):
        """
        Args:
            slots: [B, K, D] object slot representations
            keep_mask: [B, K] binary mask (1 = active slot)

        Returns:
            enhanced_slots: [B, K, D] slots with relational information
        """
        B, K, D = slots.shape

        # Build graph
        adj = self.graph_builder(slots, keep_mask)  # [B, K, K]

        for i in range(self.num_layers):
            residual = slots

            # Order nodes
            ordered, order, reverse_order = self.orderer(slots, adj)

            # Bidirectional Mamba scan
            fwd_out = self.forward_mambas[i](ordered)             # [B, K, D]
            bwd_out = self.backward_mambas[i](ordered.flip(1))    # [B, K, D]
            bwd_out = bwd_out.flip(1)                              # Reverse back

            # Merge directions
            merged = torch.cat([fwd_out, bwd_out], dim=-1)  # [B, K, 2D]
            merged = self.merge_layers[i](merged)            # [B, K, D]

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

        return slots
