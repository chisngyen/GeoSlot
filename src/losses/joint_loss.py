"""
Joint Loss: Multi-Layer Loss orchestrator with stage-wise scheduling.

Combines all loss functions with configurable weights and
automatic stage-wise activation based on training epoch.

Stage-wise training strategy:
- Epoch 1-30:  InfoNCE + DWBL only (learn basic embedding)
- Epoch 30-60: + Contrastive Slot Matching + Dice (refine slots)
- Epoch 60+:   + MESH (via Sinkhorn OT) (fine-tune hard matching)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .infonce import SymmetricInfoNCE
from .dwbl import DWBL
from .dice_loss import DiceLoss
from .contrastive_slot import ContrastiveSlotMatchingLoss


class JointLoss(nn.Module):
    """
    Multi-Layer Joint Loss with stage-wise scheduling.

    Args:
        lambda_infonce: Weight for InfoNCE loss
        lambda_dwbl: Weight for DWBL loss
        lambda_csm: Weight for Contrastive Slot Matching loss
        lambda_dice: Weight for Dice loss
        temperature: Shared temperature parameter
        stage2_epoch: Epoch to activate Slot losses (CSM + Dice)
        stage3_epoch: Epoch to activate full pipeline losses
    """
    def __init__(
        self,
        lambda_infonce: float = 1.0,
        lambda_dwbl: float = 1.0,
        lambda_csm: float = 0.3,
        lambda_dice: float = 0.1,
        temperature: float = 0.07,
        stage2_epoch: int = 30,
        stage3_epoch: int = 60,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.lambda_infonce = lambda_infonce
        self.lambda_dwbl = lambda_dwbl
        self.lambda_csm = lambda_csm
        self.lambda_dice = lambda_dice
        self.stage2_epoch = stage2_epoch
        self.stage3_epoch = stage3_epoch
        self.warmup_epochs = warmup_epochs

        # Loss modules
        self.infonce = SymmetricInfoNCE(temperature=temperature)
        self.dwbl = DWBL(temperature=temperature * 1.5)
        self.csm = ContrastiveSlotMatchingLoss(temperature=max(temperature, 0.5))
        self.dice = DiceLoss()

    def forward(self, model_output: Dict, epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        Compute joint loss with stage-wise activation.

        Args:
            model_output: dict from GeoSlot.forward()
            epoch: Current training epoch

        Returns:
            dict with:
                - total_loss: scalar total loss
                - loss_infonce, loss_dwbl, loss_csm, loss_dice: individual losses
                - accuracy: batch retrieval accuracy
                - active_stage: current active stage (1, 2, or 3)
        """
        losses = {}
        total = torch.tensor(0.0, device=model_output['query_embedding'].device)

        # ===== Stage 1: Always active — Embedding losses =====
        loss_infonce, accuracy = self.infonce(
            model_output['query_embedding'],
            model_output['ref_embedding']
        )
        loss_dwbl = self.dwbl(
            model_output['query_embedding'],
            model_output['ref_embedding']
        )
        total = total + self.lambda_infonce * loss_infonce + self.lambda_dwbl * loss_dwbl
        losses['loss_infonce'] = loss_infonce.detach()
        losses['loss_dwbl'] = loss_dwbl.detach()
        losses['accuracy'] = accuracy

        # ===== Stage 2: Slot quality losses (activated at stage2_epoch) =====
        if epoch >= self.stage2_epoch:
            # Linear warm-up ramp to prevent gradient shock at stage transition
            ramp = min(1.0, (epoch - self.stage2_epoch + 1) / self.warmup_epochs)
            loss_csm = self.csm(model_output)
            loss_dice = self.dice(
                model_output['query_attn_maps'],
                model_output.get('query_keep_mask')
            )
            total = total + ramp * (self.lambda_csm * loss_csm + self.lambda_dice * loss_dice)
            losses['loss_csm'] = loss_csm.detach()
            losses['loss_dice'] = loss_dice.detach()
        else:
            losses['loss_csm'] = torch.tensor(0.0)
            losses['loss_dice'] = torch.tensor(0.0)

        # ===== Determine active stage =====
        if epoch >= self.stage3_epoch:
            active_stage = 3
        elif epoch >= self.stage2_epoch:
            active_stage = 2
        else:
            active_stage = 1

        losses['total_loss'] = total
        losses['active_stage'] = active_stage

        return losses
