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
    def __init__(self, smooth: float = 0.1):
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
        # Normalize along N so each slot's attention sums to 1
        attn_norm = attn_maps / (attn_maps.sum(dim=-1, keepdim=True) + 1e-8)

        # Vectorized pairwise Dice overlap: use matrix multiplication
        # [B, K, N] @ [B, N, K] = [B, K, K] — pairwise dot products
        pairwise_inter = torch.bmm(attn_norm, attn_norm.transpose(1, 2))  # [B, K, K]
        slot_sums = attn_norm.sum(dim=-1)  # [B, K]
        pairwise_union = slot_sums.unsqueeze(2) + slot_sums.unsqueeze(1)  # [B, K, K]

        dice_matrix = (2.0 * pairwise_inter + self.smooth) / (pairwise_union + self.smooth)

        # Take upper triangle only (i < j pairs), excluding diagonal
        mask_triu = torch.triu(torch.ones(K, K, device=attn_maps.device), diagonal=1).bool()
        overlap_loss = dice_matrix[:, mask_triu].mean()

        # Apply keep_mask: only count active slots
        if keep_mask is not None:
            active_ratio = keep_mask.float().mean()
            overlap_loss = overlap_loss * active_ratio

        return overlap_loss
