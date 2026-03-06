"""
Contrastive Slot Matching Loss with Register Slot Regularization.

Ensures that:
1. Matching object slots (same real-world entity) across views are similar
2. Register slots absorb noise and are maximally different from object slots
3. Slots from different entities are well-separated
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveSlotMatchingLoss(nn.Module):
    """
    Contrastive Slot Matching Loss.

    Uses the OT transport plan to identify slot correspondences,
    then applies contrastive learning to:
    - Pull matched slot pairs together
    - Push non-matched slots apart
    - Regularize register slots to be orthogonal to object slots

    Args:
        temperature: Contrastive temperature
        register_weight: Weight for register slot regularization
    """
    def __init__(self, temperature: float = 0.1, register_weight: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.register_weight = register_weight

    def _slot_contrastive_loss(self, slots_q, slots_r, transport_plan, mask_q=None, mask_r=None):
        """
        Contrastive loss between matched slot pairs.

        Uses the transport plan T[i,j] as soft labels for which slots match.

        Args:
            slots_q: [B, K, D] query slots
            slots_r: [B, M, D] reference slots
            transport_plan: [B, K, M] OT transport plan (soft assignment)
            mask_q: [B, K] query mask
            mask_r: [B, M] reference mask
        """
        B, K, D = slots_q.shape
        M = slots_r.shape[1]

        # Normalize slots
        q_norm = F.normalize(slots_q, dim=-1)  # [B, K, D]
        r_norm = F.normalize(slots_r, dim=-1)  # [B, M, D]

        # Similarity between all slot pairs
        sim = torch.bmm(q_norm, r_norm.transpose(1, 2)) / self.temperature  # [B, K, M]

        # Use transport plan as soft labels (which slots should match)
        # Normalize transport plan to get probability distribution per query slot
        T_norm = transport_plan / (transport_plan.sum(dim=-1, keepdim=True) + 1e-8)  # [B, K, M]

        # Cross-entropy between similarity distribution and transport plan
        log_softmax_sim = F.log_softmax(sim, dim=-1)  # [B, K, M]
        loss = -(T_norm * log_softmax_sim).sum(dim=-1)  # [B, K]

        # Mask inactive slots
        if mask_q is not None:
            loss = loss * mask_q
            loss = loss.sum() / (mask_q.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def _register_regularization(self, object_slots, register_slots, keep_mask=None):
        """
        Regularize register slots to be orthogonal to object slots.

        Register slots should absorb noise/background, so they should
        be maximally different from meaningful object representations.

        Args:
            object_slots: [B, K, D] object slots
            register_slots: [B, R, D] register slots
            keep_mask: [B, K] object slot mask
        """
        obj_norm = F.normalize(object_slots, dim=-1)
        reg_norm = F.normalize(register_slots, dim=-1)

        # Similarity between register and object slots
        sim = torch.bmm(reg_norm, obj_norm.transpose(1, 2))  # [B, R, K]

        if keep_mask is not None:
            sim = sim * keep_mask.unsqueeze(1)  # Only count active object slots

        # Minimize similarity → make registers orthogonal to objects
        loss = sim.abs().mean()

        return loss

    def forward(self, model_output):
        """
        Args:
            model_output: dict from GeoSlot.forward() containing:
                - query_slots, ref_slots: [B, K, D]
                - transport_plan: [B, K, M]
                - query_keep_mask, ref_keep_mask: [B, K]
                - query_register_slots, ref_register_slots (if available)

        Returns:
            loss: scalar contrastive slot matching loss
        """
        # Slot contrastive loss using OT transport plan
        slot_loss = self._slot_contrastive_loss(
            model_output['query_slots'],
            model_output['ref_slots'],
            model_output['transport_plan'],
            model_output.get('query_keep_mask'),
            model_output.get('ref_keep_mask'),
        )

        # Register regularization (if register slots are available)
        reg_loss = torch.tensor(0.0, device=slot_loss.device)
        if 'query_register_slots' in model_output:
            reg_loss_q = self._register_regularization(
                model_output['query_slots'],
                model_output['query_register_slots'],
                model_output.get('query_keep_mask'),
            )
            reg_loss_r = self._register_regularization(
                model_output['ref_slots'],
                model_output['ref_register_slots'],
                model_output.get('ref_keep_mask'),
            )
            reg_loss = (reg_loss_q + reg_loss_r) / 2

        total = slot_loss + self.register_weight * reg_loss
        return total
