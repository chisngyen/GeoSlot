"""
Dynamic Weighted Batch-tuple Loss (DWBL) for hard negative mining.

Reference:
- VimGeo (IJCAI 2025): https://github.com/VimGeoTeam/VimGeo
- Weighted Soft-Margin Triplet Loss with dynamic weighting

Key concept: Instead of uniform weighting of negatives in a batch,
DWBL assigns higher weights to "hard negatives" (negative samples
that are close to the anchor in embedding space), accelerating learning
of fine-grained discrimination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWBL(nn.Module):
    """
    Dynamic Weighted Batch-tuple Loss.

    For each anchor-positive pair, computes weighted contributions from
    all negatives in the batch, where weights are proportional to
    the similarity between anchor and negative (hard negative mining).

    Loss = -log(exp(pos_sim/τ) / (exp(pos_sim/τ) + Σ_j w_j * exp(neg_sim_j/τ)))

    where w_j is dynamically computed based on negative difficulty.

    Args:
        temperature: Temperature scaling factor
        margin: Soft margin for triplet-like behavior
        dynamic_weight: If True, use dynamic weighting (harder negatives get more weight)
    """
    def __init__(self, temperature: float = 0.1, margin: float = 0.3, dynamic_weight: bool = True):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.dynamic_weight = dynamic_weight

    def _compute_weights(self, neg_similarities):
        """
        Compute dynamic weights for negatives based on difficulty.

        Hard negatives (high similarity with anchor) get higher weights.

        Args:
            neg_similarities: [B, B-1] similarities with all negatives

        Returns:
            weights: [B, B-1] normalized weights
        """
        # Softmax over negatives → harder negatives get more weight
        weights = F.softmax(neg_similarities / self.temperature, dim=-1)
        return weights

    def forward(self, query_emb, ref_emb):
        """
        Args:
            query_emb: [B, D] L2-normalized query embeddings
            ref_emb: [B, D] L2-normalized reference embeddings

        Returns:
            loss: scalar DWBL loss
        """
        B = query_emb.shape[0]

        # Full similarity matrix
        sim = torch.mm(query_emb, ref_emb.t())  # [B, B]

        # Positive similarities (diagonal)
        pos_sim = sim.diag()  # [B]

        # Create mask for negatives
        mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)  # [B, B]
        neg_sim = sim[mask].view(B, B - 1)  # [B, B-1]

        if self.dynamic_weight:
            # Dynamic weighting: harder negatives contribute more
            weights = self._compute_weights(neg_sim.detach())  # [B, B-1]
            weighted_neg = (weights * torch.exp((neg_sim - self.margin) / self.temperature)).sum(dim=-1)
        else:
            weighted_neg = torch.exp((neg_sim - self.margin) / self.temperature).sum(dim=-1)

        # Loss: soft margin ranking
        pos_exp = torch.exp(pos_sim / self.temperature)
        loss = -torch.log(pos_exp / (pos_exp + weighted_neg + 1e-8))

        return loss.mean()
