"""
Adaptive Slot Attention with Register Slots and Background Suppression.

Reference:
- AdaSlot (CVPR 2024): https://github.com/amazon-science/AdaSlot
- Slot Attention (NeurIPS 2020): https://arxiv.org/abs/2006.15055
- Register Tokens (ICLR 2024): https://arxiv.org/abs/2309.16588

Key features:
1. SlotAttention: GRU-based iterative routing with competitive softmax
2. Gumbel selection: Adaptive K (number of active slots) via Gumbel-Softmax
3. Register Slots: Dedicated noise-absorbing slots, discarded before matching
4. Background Suppression Mask: Lightweight attention mask to filter transient objects
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Background Suppression Mask
# ============================================================================
class BackgroundSuppressionMask(nn.Module):
    """
    Lightweight module to suppress transient/dynamic objects before Slot Attention.

    Uses a small MLP head to predict a spatial attention mask that
    down-weighs features from dynamic objects (cars, pedestrians, clouds)
    that exist in ground-view but not in satellite imagery.

    Regularization (addresses reviewer concern):
    - Entropy regularization prevents mask collapse (all-0 or all-1)
    - Coverage constraint ensures mask retains sufficient information
    """
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.mask_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        """
        Args:
            features: [B, N, D] dense features from backbone

        Returns:
            masked_features: [B, N, D] features with transient objects suppressed
            mask: [B, N, 1] suppression mask (1 = static/keep, 0 = transient/suppress)
        """
        mask = self.mask_head(features)  # [B, N, 1]
        masked_features = features * mask
        return masked_features, mask

    def entropy_regularization(self, mask):
        """
        Entropy regularization to prevent mask collapse.

        Encourages the mask to be neither all-1 (no suppression)
        nor all-0 (suppress everything). Maximizes entropy of the
        binary Bernoulli distribution at each spatial position.

        Args:
            mask: [B, N, 1] sigmoid mask values

        Returns:
            reg_loss: scalar entropy regularization loss
        """
        p = mask.squeeze(-1).clamp(1e-6, 1 - 1e-6)  # [B, N]
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())  # [B, N]
        # We MAXIMIZE entropy → MINIMIZE negative entropy
        return -entropy.mean()

    def coverage_regularization(self, mask, target_ratio=0.7):
        """
        Coverage constraint: mask should retain ~target_ratio of tokens.

        Prevents the mask from suppressing too much useful signal.

        Args:
            mask: [B, N, 1]
            target_ratio: desired fraction of tokens to keep

        Returns:
            reg_loss: scalar coverage loss
        """
        mean_coverage = mask.mean()
        return (mean_coverage - target_ratio) ** 2


# ============================================================================
# Slot Attention Core
# ============================================================================
class SlotAttention(nn.Module):
    """
    Slot Attention mechanism (Locatello et al., 2020).

    Iteratively routes input features to a set of slot representations
    using competitive attention (softmax over slots, normalize over features).

    Ported from AdaSlot's perceptual_grouping.py with modifications for CVGL.
    """
    def __init__(
        self,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        use_projection_bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps

        self.kvq_dim = kvq_dim or dim
        if self.kvq_dim % self.n_heads != 0:
            raise ValueError("kvq_dim must be divisible by n_heads")
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head ** -0.5

        # Projections
        self.to_q = nn.Linear(dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)

        # GRU for slot update
        self.gru = nn.GRUCell(self.kvq_dim, dim)

        # Norms
        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(dim)

        # FFN after GRU update
        self.ff_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def step(self, slots, k, v, masks=None):
        """
        Single slot attention step.

        Args:
            slots: [B, K, D] current slot representations
            k: [B, N, H, Dh] keys from input features
            v: [B, N, H, Dh] values from input features
            masks: [B, K] boolean mask (True = masked/inactive slot)

        Returns:
            updated_slots: [B, K, D]
            attn: [B, K, N] attention weights
        """
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        # Attention: slots compete for features
        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale

        if masks is not None:
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        # Softmax over slots (competition) → normalize over features
        attn = dots.flatten(1, 2).softmax(dim=1)  # [B, K*H, N]
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # Weighted aggregation
        updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        # GRU update
        slots = self.gru(
            updates.reshape(-1, self.kvq_dim),
            slots_prev.reshape(-1, self.dim)
        )
        slots = slots.reshape(bs, -1, self.dim)

        # FFN
        slots = slots + self.ff_mlp(slots)

        return slots, attn_before_reweighting.mean(dim=2)  # [B, K, N]

    def forward(self, inputs, conditioning, masks=None):
        """
        Args:
            inputs: [B, N, D_feat] input features
            conditioning: [B, K, D_slot] initial slot values
            masks: [B, K] optional slot masks

        Returns:
            slots: [B, K, D_slot] refined slots
            attn: [B, K, N] attention maps
        """
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v, masks)

        return slots, attn


# ============================================================================
# Gumbel Slot Selection (from AdaSlot)
# ============================================================================
class GumbelSlotSelector(nn.Module):
    """
    Gumbel-Softmax selection for adaptive number of active slots.

    During training, uses Gumbel-Softmax with decreasing temperature
    to gradually sharpen the selection. At inference, uses argmax.

    Reference: AdaSlot (CVPR 2024)
    """
    def __init__(self, slot_dim, low_bound=1):
        super().__init__()
        self.low_bound = low_bound
        self.score_net = nn.Sequential(
            nn.Linear(slot_dim, slot_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim // 2, 2),  # [drop_prob, keep_prob]
        )

    def _ensure_minimum_slots(self, decision):
        """Ensure at least `low_bound` slots are kept active."""
        B = decision.shape[0]
        active_counts = (decision != 0).sum(dim=-1)  # [B]
        deficit_mask = active_counts < self.low_bound  # [B]

        if deficit_mask.any():
            for j in deficit_mask.nonzero(as_tuple=True)[0]:
                inactive = (decision[j] == 0).nonzero(as_tuple=True)[0]
                num_needed = self.low_bound - int(active_counts[j].item())
                num_activate = min(num_needed, len(inactive))
                if num_activate > 0:
                    chosen = inactive[torch.randperm(len(inactive), device=decision.device)[:num_activate]]
                    decision[j, chosen] = 1.0

        return decision

    def forward(self, slots, global_step=None):
        """
        Args:
            slots: [B, K, D] slot representations
            global_step: Current training step (for temperature scheduling)

        Returns:
            keep_decision: [B, K] binary mask (1 = keep, 0 = drop)
            keep_probs: [B, K] soft keep probabilities
        """
        logits = self.score_net(slots)  # [B, K, 2]

        # Temperature scheduling: starts warm (τ=1), cools down over training
        if global_step is not None:
            tau = max(0.1, 1.0 - global_step / 100000)
        else:
            tau = 1.0

        if self.training:
            decision = F.gumbel_softmax(logits, hard=True, tau=tau)[..., 1]  # [B, K]
        else:
            decision = (logits.argmax(dim=-1) == 1).float()  # [B, K]

        decision = self._ensure_minimum_slots(decision)
        keep_probs = F.softmax(logits, dim=-1)[..., 1]  # [B, K]

        return decision, keep_probs


# ============================================================================
# Adaptive Slot Attention with Register Slots
# ============================================================================
class AdaptiveSlotAttention(nn.Module):
    """
    Full Adaptive Slot Attention module combining:
    1. Background Suppression Mask (filter transient objects)
    2. Slot Attention (iterative competitive routing)
    3. Gumbel Slot Selection (adaptive K)
    4. Register Slots (noise-absorbing, discarded before matching)

    Args:
        feature_dim: Dimension of input features (from backbone)
        slot_dim: Dimension of slot representations
        max_slots: Maximum number of object slots
        n_register: Number of register (noise-absorbing) slots
        n_heads: Number of attention heads
        iters: Number of SA iterations
        low_bound: Minimum number of active slots
    """
    def __init__(
        self,
        feature_dim: int = 384,
        slot_dim: int = 256,
        max_slots: int = 16,
        n_register: int = 4,
        n_heads: int = 4,
        iters: int = 3,
        low_bound: int = 1,
    ):
        super().__init__()
        self.max_slots = max_slots
        self.n_register = n_register
        self.slot_dim = slot_dim
        total_slots = max_slots + n_register

        # Background suppression
        self.bg_mask = BackgroundSuppressionMask(feature_dim)

        # Slot initialization (learnable)
        self.slot_mu = nn.Parameter(torch.randn(1, total_slots, slot_dim) * (slot_dim ** -0.5))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, total_slots, slot_dim))

        # Feature projection (if feature_dim != slot_dim)
        self.input_proj = nn.Linear(feature_dim, slot_dim) if feature_dim != slot_dim else nn.Identity()

        # Core slot attention
        self.slot_attention = SlotAttention(
            dim=slot_dim,
            feature_dim=slot_dim,
            n_heads=n_heads,
            iters=iters,
        )

        # Gumbel selector (only for object slots, not registers)
        self.gumbel_selector = GumbelSlotSelector(slot_dim, low_bound=low_bound)

    def _init_slots(self, batch_size, device):
        """Initialize slots with learned mean + noise."""
        mu = self.slot_mu.expand(batch_size, -1, -1)
        sigma = self.slot_log_sigma.exp().expand(batch_size, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)
        return slots

    def forward(self, features, global_step=None):
        """
        Args:
            features: [B, N, D] dense features from VimBackbone
            global_step: Current training step (for Gumbel temperature)

        Returns:
            object_slots: [B, K', D_slot] active object slots (K' ≤ max_slots)
            register_slots: [B, n_register, D_slot] register slots (for analysis only)
            bg_mask: [B, N, 1] background suppression mask
            attn_maps: [B, K_total, N] attention maps (for visualization)
            keep_probs: [B, max_slots] slot keep probabilities
        """
        B, N, D = features.shape

        # 1. Background suppression
        features_masked, bg_mask = self.bg_mask(features)

        # 2. Project features to slot dimension
        features_proj = self.input_proj(features_masked)  # [B, N, slot_dim]

        # 3. Initialize all slots (object + register)
        all_slots = self._init_slots(B, features.device)  # [B, max_slots + n_register, slot_dim]

        # 4. Run Slot Attention
        all_slots, attn_maps = self.slot_attention(features_proj, all_slots)

        # 5. Split object slots and register slots
        object_slots = all_slots[:, :self.max_slots, :]         # [B, max_slots, D]
        register_slots = all_slots[:, self.max_slots:, :]       # [B, n_register, D]

        # 6. Gumbel selection on object slots only
        keep_decision, keep_probs = self.gumbel_selector(object_slots, global_step)

        # 7. Mask out dropped slots
        object_slots = object_slots * keep_decision.unsqueeze(-1)  # [B, max_slots, D]

        # 8. Remove zero slots for efficiency (pack active slots)
        # During training: keep all slots (for gradient flow), mask handles it
        # During eval: can optionally pack for efficiency

        return {
            'object_slots': object_slots,           # [B, max_slots, D]
            'register_slots': register_slots,       # [B, n_register, D]
            'bg_mask': bg_mask,                     # [B, N, 1]
            'attn_maps': attn_maps,                 # [B, K_total, N]
            'keep_decision': keep_decision,         # [B, max_slots]
            'keep_probs': keep_probs,               # [B, max_slots]
        }
