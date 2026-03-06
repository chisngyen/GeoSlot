"""
Symmetric InfoNCE Loss for cross-view embedding alignment.

Reference:
- CLIP (Radford et al., 2021)
- Cross-view Geo-localization literature

Aligns query and reference embeddings in a shared space using
symmetric cross-entropy over the similarity matrix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCE(nn.Module):
    """
    Symmetric InfoNCE (Contrastive) Loss.

    For a batch of B (query, reference) pairs:
    - Positive pair: query[i] ↔ ref[i]
    - Negative pairs: query[i] ↔ ref[j] for j ≠ i

    Loss = 0.5 * (CE(sim, labels) + CE(sim.T, labels))

    Args:
        temperature: Softmax temperature (τ). Lower = sharper distribution.
        learn_temperature: If True, make temperature learnable.
    """
    def __init__(self, temperature: float = 0.07, learn_temperature: bool = True):
        super().__init__()
        if learn_temperature:
            self.log_temp = nn.Parameter(torch.tensor(temperature).log())
        else:
            self.register_buffer('log_temp', torch.tensor(temperature).log())

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(min=0.01, max=1.0)

    def forward(self, query_emb, ref_emb):
        """
        Args:
            query_emb: [B, D] L2-normalized query embeddings
            ref_emb: [B, D] L2-normalized reference embeddings

        Returns:
            loss: scalar InfoNCE loss
            accuracy: top-1 retrieval accuracy within batch
        """
        B = query_emb.shape[0]

        # Compute similarity matrix
        logits = torch.mm(query_emb, ref_emb.t()) / self.temperature  # [B, B]

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(B, device=logits.device)

        # Symmetric loss
        loss_q2r = F.cross_entropy(logits, labels)
        loss_r2q = F.cross_entropy(logits.t(), labels)
        loss = (loss_q2r + loss_r2q) / 2

        # Accuracy (for monitoring)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean()

        return loss, accuracy
