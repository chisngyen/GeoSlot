"""
Sinkhorn Optimal Transport + MESH for Cross-View Slot Matching.

Reference:
- Sinkhorn Distances (Cuturi, 2013): https://arxiv.org/abs/1306.0895
- geomloss: https://www.kernel-operations.io/geomloss/
- MESH (Minimize Entropy of Sinkhorn): For hard assignment tie-breaking

Key concept: Instead of simple cosine similarity between global features,
use Optimal Transport to find the minimum-cost bipartite matching between
two sets of object slots from different views.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SinkhornOT(nn.Module):
    """
    Sinkhorn Optimal Transport matcher for cross-view slot matching.

    Computes the optimal transport distance between two sets of slots
    (query view and reference view), providing a similarity score that
    accounts for partial matches and set-level correspondence.

    The MESH mechanism adds entropy minimization to push soft assignments
    toward hard 1-to-1 matching, resolving ambiguity between similar objects.

    Args:
        slot_dim: Dimension of slot representations
        num_iters: Number of Sinkhorn normalization iterations
        epsilon: Entropic regularization strength
            - Smaller ε → harder/sharper matching (but less stable)
            - Larger ε → softer matching (more stable, less precise)
        mesh_iters: Number of MESH entropy minimization steps
        learn_epsilon: If True, make epsilon a learnable parameter
    """
    def __init__(
        self,
        slot_dim: int = 256,
        num_iters: int = 20,
        epsilon: float = 0.05,
        mesh_iters: int = 3,
        learn_epsilon: bool = False,
    ):
        super().__init__()
        self.num_iters = num_iters
        self.mesh_iters = mesh_iters

        if learn_epsilon:
            self.log_epsilon = nn.Parameter(torch.tensor(math.log(epsilon)))
        else:
            self.register_buffer('log_epsilon', torch.tensor(math.log(epsilon)))

        # Learnable cost projection (optional: can refine cost metric)
        self.cost_proj = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, slot_dim),
        )

    @property
    def epsilon(self):
        return self.log_epsilon.exp()

    def _compute_cost_matrix(self, slots_q, slots_r):
        """
        Compute pairwise cost matrix between query and reference slots.

        Args:
            slots_q: [B, K, D] query slots
            slots_r: [B, M, D] reference slots

        Returns:
            C: [B, K, M] cost matrix (lower = more similar)
        """
        # Project for cost computation
        q_proj = self.cost_proj(slots_q)  # [B, K, D]
        r_proj = self.cost_proj(slots_r)  # [B, M, D]

        # L2 distance (squared)
        C = torch.cdist(q_proj, r_proj, p=2.0)  # [B, K, M]
        return C

    def _sinkhorn(self, C, mask_q=None, mask_r=None):
        """
        Sinkhorn-Knopp algorithm for entropy-regularized OT.

        Args:
            C: [B, K, M] cost matrix
            mask_q: [B, K] query slot mask (1 = active)
            mask_r: [B, M] reference slot mask (1 = active)

        Returns:
            T: [B, K, M] transport plan (doubly stochastic matrix)
        """
        B, K, M = C.shape
        eps = self.epsilon

        # Initialize log-space potentials
        log_alpha = torch.zeros(B, K, 1, device=C.device)
        log_beta = torch.zeros(B, 1, M, device=C.device)

        # Kernel matrix in log space
        log_K = -C / eps  # [B, K, M]

        # Mask inactive slots with -inf
        if mask_q is not None:
            log_K = log_K + torch.log(mask_q.unsqueeze(-1).clamp(min=1e-8))
        if mask_r is not None:
            log_K = log_K + torch.log(mask_r.unsqueeze(-2).clamp(min=1e-8))

        # Sinkhorn iterations (log-space for numerical stability)
        for _ in range(self.num_iters):
            # Row normalization
            log_alpha = -torch.logsumexp(log_K + log_beta, dim=2, keepdim=True)
            # Column normalization
            log_beta = -torch.logsumexp(log_K + log_alpha, dim=1, keepdim=True)

        # Transport plan
        T = torch.exp(log_K + log_alpha + log_beta)  # [B, K, M]
        return T

    def _mesh(self, T, C):
        """
        MESH: Minimize Entropy of Sinkhorn for hard assignment.

        Iteratively sharpens the transport plan T by pushing
        soft assignments toward 1-to-1 hard matching.

        Args:
            T: [B, K, M] soft transport plan
            C: [B, K, M] cost matrix

        Returns:
            T_hard: [B, K, M] sharpened transport plan
        """
        for _ in range(self.mesh_iters):
            # Temperature sharpening: raise T to power > 1, then re-normalize
            T_sharp = T ** 2

            # Row normalization
            T_sharp = T_sharp / (T_sharp.sum(dim=-1, keepdim=True) + 1e-8)

            # Column normalization
            T_sharp = T_sharp / (T_sharp.sum(dim=-2, keepdim=True) + 1e-8)

            T = T_sharp

        return T

    def forward(self, slots_q, slots_r, mask_q=None, mask_r=None):
        """
        Compute OT-based similarity between query and reference slot sets.

        Args:
            slots_q: [B, K, D] query view slots
            slots_r: [B, M, D] reference view slots
            mask_q: [B, K] query slot mask (1 = active, from Gumbel selection)
            mask_r: [B, M] reference slot mask

        Returns:
            similarity: [B] similarity scores (higher = more similar)
            transport_plan: [B, K, M] transport plan for visualization
            cost_matrix: [B, K, M] cost matrix
        """
        # Compute cost matrix
        C = self._compute_cost_matrix(slots_q, slots_r)  # [B, K, M]

        # Sinkhorn iterations → soft transport plan
        T = self._sinkhorn(C, mask_q, mask_r)  # [B, K, M]

        # MESH → sharpen to near-hard assignment
        T_hard = self._mesh(T, C)  # [B, K, M]

        # Compute transport cost (= dissimilarity)
        transport_cost = (T_hard * C).sum(dim=(-1, -2))  # [B]

        # Convert to similarity (negate and sigmoid for [0, 1] range)
        similarity = torch.sigmoid(-transport_cost)

        return {
            'similarity': similarity,          # [B]
            'transport_plan': T_hard,          # [B, K, M]
            'cost_matrix': C,                  # [B, K, M]
            'transport_cost': transport_cost,  # [B]
        }



