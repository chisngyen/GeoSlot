"""
Dice Loss for Slot Attention masks.

Handles scale imbalance between slot masks and full image,
where slots typically cover only a small portion of the image.
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for slot mask quality.

    Computes the Dice coefficient between predicted slot attention maps
    and their reconstruction, encouraging slots to cover distinct regions.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice

    Args:
        smooth: Smoothing factor to avoid division by zero
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, attn_maps, keep_mask=None):
        """
        Args:
            attn_maps: [B, K, N] slot attention maps
            keep_mask: [B, K] active slot mask (optional)

        Returns:
            loss: scalar Dice loss encouraging distinct slot coverage
        """
        B, K, N = attn_maps.shape

        # Each slot's attention should be distinct (low overlap with others)
        # Compute pairwise overlap between slot attention maps
        # attn_maps: [B, K, N] → normalize along N
        attn_norm = attn_maps / (attn_maps.sum(dim=-1, keepdim=True) + 1e-8)

        # Pairwise Dice coefficient between slot pairs
        # For each pair (i, j), compute overlap
        attn_flat = attn_norm.reshape(B * K, N)

        # Ideal: each slot covers a unique region → sum of maps ≈ 1 everywhere
        coverage = attn_norm.sum(dim=1)  # [B, N]

        # Coverage should be uniform → encourage coverage diversity
        # Loss = mean squared deviation from uniform coverage
        target = torch.ones_like(coverage) / K
        coverage_loss = ((coverage - target) ** 2).mean()

        # Dice between each slot's attention and the complement of other slots
        total_loss = 0.0
        count = 0
        for i in range(K):
            for j in range(i + 1, K):
                intersection = (attn_norm[:, i] * attn_norm[:, j]).sum(dim=-1)  # [B]
                union = attn_norm[:, i].sum(dim=-1) + attn_norm[:, j].sum(dim=-1)  # [B]
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                total_loss = total_loss + dice.mean()
                count += 1

        if count > 0:
            overlap_loss = total_loss / count  # Minimize overlap → minimize Dice
        else:
            overlap_loss = torch.tensor(0.0, device=attn_maps.device)

        # Apply keep_mask: only count active slots
        if keep_mask is not None:
            active_ratio = keep_mask.sum() / (B * K)
            overlap_loss = overlap_loss * active_ratio

        return overlap_loss + coverage_loss * 0.1
