"""
Fused Gromov-Wasserstein (FGW) Optimal Transport for Cross-View Graph Matching.

Reference:
- Fused Gromov-Wasserstein (Vayer et al., 2020): https://arxiv.org/abs/1811.02834
- Unbalanced OT (Chizat et al., 2018): https://arxiv.org/abs/1607.05816
- POT library: https://pythonot.github.io/

Key concept: Match two sets of object slots using BOTH:
1. Node feature similarity (Wasserstein cost) — "Building A looks like Building B"
2. Graph structure similarity (Gromov-Wasserstein cost) — "A is 10m from intersection C,
   just like B is 10m from intersection D"

This solves the Graph-to-Set Fallacy in the original Sinkhorn OT:
- Sinkhorn only matches node features, discarding all edge/topology information
- FGW preserves the relational structure learned by Graph Mamba

Unbalanced extension (UFGW):
- Standard OT forces total mass conservation (T·1 = μ)
- UFGW relaxes marginals via KL divergence penalty
- Allows occluded/missing slots to be "rejected" gracefully
- Solves the MESH collapse under occlusion (UAV sees 5 buildings, satellite sees 15)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class FusedGromovWasserstein(nn.Module):
    """
    Unbalanced Fused Gromov-Wasserstein matcher for cross-view slot matching.

    Combines:
    - Wasserstein distance for node feature matching (C_ij)
    - Gromov-Wasserstein distance for graph structure matching (S^q, S^r)
    - KL-relaxed marginals for handling occlusion (unbalanced)

    Args:
        slot_dim: Dimension of slot representations
        lambda_fgw: Trade-off between feature (0) and structure (1) cost
        tau_kl: KL divergence penalty for unbalanced relaxation
        num_outer_iters: Number of Block Coordinate Descent iterations for FGW
        num_sinkhorn_iters: Number of Sinkhorn iterations per BCD step
        epsilon: Entropic regularization strength
        learn_epsilon: If True, make epsilon a learnable parameter
    """
    def __init__(
        self,
        slot_dim: int = 256,
        lambda_fgw: float = 0.5,
        tau_kl: float = 0.1,
        num_outer_iters: int = 10,
        num_sinkhorn_iters: int = 20,
        epsilon: float = 0.05,
        learn_epsilon: bool = False,
    ):
        super().__init__()
        self.lambda_fgw = lambda_fgw
        self.tau_kl = tau_kl
        self.num_outer_iters = num_outer_iters
        self.num_sinkhorn_iters = num_sinkhorn_iters

        if learn_epsilon:
            self.log_epsilon = nn.Parameter(torch.tensor(math.log(epsilon)))
        else:
            self.register_buffer('log_epsilon', torch.tensor(math.log(epsilon)))

        # Learnable cost projection for node features
        self.cost_proj = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, slot_dim),
        )

    @property
    def epsilon(self):
        return self.log_epsilon.exp()

    def _compute_feature_cost(self, slots_q, slots_r):
        """
        Compute pairwise feature cost matrix between query and reference slots.

        Args:
            slots_q: [B, K, D] query slots
            slots_r: [B, M, D] reference slots

        Returns:
            C: [B, K, M] feature cost matrix (lower = more similar)
        """
        q_proj = self.cost_proj(slots_q)
        r_proj = self.cost_proj(slots_r)
        # Safe L2: torch.cdist backward has NaN at zero distance
        diff = q_proj.unsqueeze(2) - r_proj.unsqueeze(1)  # [B, K, M, D]
        C = (diff * diff).sum(-1).clamp(min=1e-6).sqrt()  # [B, K, M]
        return C

    def _compute_structure_cost(self, centroids_q, centroids_r):
        """
        Compute intra-graph structure cost matrices.

        These capture the internal spatial structure of each graph:
        S^q_{ik} = ||c^q_i - c^q_k||_2  (distance between slots in query)
        S^r_{jl} = ||c^r_j - c^r_l||_2  (distance between slots in reference)

        Args:
            centroids_q: [B, K, 2] query slot centroids
            centroids_r: [B, M, 2] reference slot centroids

        Returns:
            Sq: [B, K, K] query structure cost
            Sr: [B, M, M] reference structure cost
        """
        # Query intra-distances
        diff_q = centroids_q.unsqueeze(2) - centroids_q.unsqueeze(1)  # [B, K, K, 2]
        Sq = (diff_q * diff_q).sum(-1).clamp(min=1e-6).sqrt()  # [B, K, K]

        # Reference intra-distances
        diff_r = centroids_r.unsqueeze(2) - centroids_r.unsqueeze(1)  # [B, M, M, 2]
        Sr = (diff_r * diff_r).sum(-1).clamp(min=1e-6).sqrt()  # [B, M, M]

        return Sq, Sr

    def _compute_gw_cost(self, Sq, Sr, T):
        """
        Compute the Gromov-Wasserstein structure cost given transport plan T.

        GW cost for each (i,j) pair:
        L_gw(i,j) = sum_{k,l} |S^q_{ik} - S^r_{jl}|^2 * T_{kl}

        This can be computed efficiently as:
        L_gw = Sq^2 @ T @ 1 + 1 @ T @ Sr^2 - 2 * Sq @ T @ Sr

        Args:
            Sq: [B, K, K] query structure cost
            Sr: [B, M, M] reference structure cost
            T: [B, K, M] current transport plan

        Returns:
            L_gw: [B, K, M] structure cost matrix
        """
        # Efficient computation using the quadratic decomposition trick
        # |S^q_{ik} - S^r_{jl}|^2 = S^q_{ik}^2 + S^r_{jl}^2 - 2*S^q_{ik}*S^r_{jl}
        Sq2 = Sq * Sq  # [B, K, K]
        Sr2 = Sr * Sr  # [B, M, M]

        # Term 1: Sq^2 @ T @ 1_M (sum over l)
        term1 = torch.bmm(Sq2, torch.bmm(T, torch.ones(T.shape[0], T.shape[2], 1,
                          device=T.device))).squeeze(-1)  # [B, K]
        term1 = term1.unsqueeze(2).expand_as(T)  # [B, K, M]

        # Term 2: 1_K @ T @ Sr^2 (sum over k)
        term2 = torch.bmm(torch.ones(T.shape[0], 1, T.shape[1],
                          device=T.device), torch.bmm(T, Sr2)).squeeze(1)  # [B, M]
        term2 = term2.unsqueeze(1).expand_as(T)  # [B, K, M]

        # Term 3: -2 * Sq @ T @ Sr
        term3 = -2.0 * torch.bmm(Sq, torch.bmm(T, Sr))  # [B, K, M]

        L_gw = term1 + term2 + term3  # [B, K, M]
        return L_gw

    def _compute_fgw_cost(self, C, Sq, Sr, T):
        """
        Compute the combined Fused Gromov-Wasserstein cost.

        C_fgw = (1 - λ) * C + λ * L_gw(T)

        Args:
            C: [B, K, M] feature cost matrix
            Sq: [B, K, K] query structure cost
            Sr: [B, M, M] reference structure cost
            T: [B, K, M] current transport plan

        Returns:
            C_fgw: [B, K, M] combined cost matrix
        """
        L_gw = self._compute_gw_cost(Sq, Sr, T)
        C_fgw = (1 - self.lambda_fgw) * C + self.lambda_fgw * L_gw
        return C_fgw

    def _unbalanced_sinkhorn(self, C, mu, nu):
        """
        Unbalanced Sinkhorn-Knopp with KL divergence relaxation.

        Instead of hard marginal constraints T·1 = μ and T^T·1 = ν,
        uses KL penalty: min <C,T> + ε·KL(T||K) + τ·KL(T1||μ) + τ·KL(T^T1||ν)

        This allows "mass destruction" — slots that cannot be matched
        are gracefully rejected rather than forced into false matches.

        Args:
            C: [B, K, M] cost matrix
            mu: [B, K] source marginal (query)
            nu: [B, M] target marginal (reference)

        Returns:
            T: [B, K, M] transport plan (possibly unbalanced)
        """
        B, K, M = C.shape
        eps = self.epsilon
        tau = self.tau_kl

        # Kernel in log-domain
        log_K = -C / eps  # [B, K, M]

        # Log marginals
        log_mu = torch.log(mu.clamp(min=1e-8))  # [B, K]
        log_nu = torch.log(nu.clamp(min=1e-8))  # [B, M]

        # Initialize dual variables
        log_u = torch.zeros(B, K, device=C.device)  # [B, K]
        log_v = torch.zeros(B, M, device=C.device)  # [B, M]

        # KL proximal coefficient: τ / (τ + ε)
        rho = tau / (tau + eps)

        for _ in range(self.num_sinkhorn_iters):
            # Update log_u (row scaling) — KL-proximal step
            log_sum_v = torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)  # [B, K]
            log_u = rho * (log_mu - log_sum_v)

            # Update log_v (column scaling) — KL-proximal step
            log_sum_u = torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)  # [B, M]
            log_v = rho * (log_nu - log_sum_u)

        # Recover transport plan
        T = torch.exp(log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1))  # [B, K, M]
        return T

    def _fgw_iterations(self, C, Sq, Sr, mu, nu):
        """
        Block Coordinate Descent for Fused Gromov-Wasserstein.

        Alternates between:
        1. Fix T → compute C_fgw = (1-λ)C + λ·L_gw(T)
        2. Fix C_fgw → solve OT via unbalanced Sinkhorn → update T

        Args:
            C: [B, K, M] feature cost matrix
            Sq: [B, K, K] query structure cost
            Sr: [B, M, M] reference structure cost
            mu: [B, K] source marginal
            nu: [B, M] target marginal

        Returns:
            T: [B, K, M] optimal transport plan
        """
        B, K, M = C.shape

        # Initialize T with uniform plan
        T = mu.unsqueeze(2) * nu.unsqueeze(1)  # [B, K, M]

        for _ in range(self.num_outer_iters):
            # Step 1: Compute combined cost given current T
            C_fgw = self._compute_fgw_cost(C, Sq, Sr, T)

            # Step 2: Solve OT with combined cost
            T = self._unbalanced_sinkhorn(C_fgw, mu, nu)

        return T

    def forward(
        self,
        slots_q: torch.Tensor,
        slots_r: torch.Tensor,
        mask_q: Optional[torch.Tensor] = None,
        mask_r: Optional[torch.Tensor] = None,
        centroids_q: Optional[torch.Tensor] = None,
        centroids_r: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute UFGW-based similarity between query and reference slot sets.

        Args:
            slots_q: [B, K, D] query view slots
            slots_r: [B, M, D] reference view slots
            mask_q: [B, K] query slot mask (1 = active)
            mask_r: [B, M] reference slot mask
            centroids_q: [B, K, 2] query slot spatial centroids
            centroids_r: [B, M, 2] reference slot spatial centroids

        Returns:
            dict with similarity, transport_plan, cost_matrix, transport_cost
        """
        # Force float32 for iterative OT — float16 loses precision
        slots_q = slots_q.float()
        slots_r = slots_r.float()
        if mask_q is not None:
            mask_q = mask_q.float()
        if mask_r is not None:
            mask_r = mask_r.float()

        B, K, D = slots_q.shape
        M = slots_r.shape[1]

        # === Compute feature cost ===
        C = self._compute_feature_cost(slots_q, slots_r)  # [B, K, M]

        # === Compute structure cost (from centroids) ===
        if centroids_q is not None and centroids_r is not None:
            centroids_q = centroids_q.float()
            centroids_r = centroids_r.float()
            Sq, Sr = self._compute_structure_cost(centroids_q, centroids_r)
        else:
            # Fallback: no structure cost → pure Wasserstein
            Sq = torch.zeros(B, K, K, device=C.device)
            Sr = torch.zeros(B, M, M, device=C.device)

        # === Build marginals ===
        if mask_q is not None:
            mu = mask_q / (mask_q.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            mu = torch.ones(B, K, device=C.device) / K

        if mask_r is not None:
            nu = mask_r / (mask_r.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            nu = torch.ones(B, M, device=C.device) / M

        # === FGW iterations ===
        T = self._fgw_iterations(C, Sq, Sr, mu, nu)

        # === Compute transport cost ===
        # Total FGW cost: feature + structure
        C_fgw = self._compute_fgw_cost(C, Sq, Sr, T)
        transport_cost = (T * C_fgw).sum(dim=(-1, -2))  # [B]

        # Convert to similarity (negative cost → high similarity)
        similarity = torch.sigmoid(-transport_cost)

        return {
            'similarity': similarity,
            'transport_plan': T,
            'cost_matrix': C,
            'transport_cost': transport_cost,
        }
