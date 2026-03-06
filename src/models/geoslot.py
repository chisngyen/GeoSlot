"""
GeoSlot: Full Pipeline Model for Cross-View Geo-Localization.

Combines all modules into a single end-to-end model:
VimBackbone → Background Suppression → AdaSlot Attention → Graph Mamba → Sinkhorn OT

Shared-weight Siamese architecture for processing query and reference views.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .vim_backbone import VimBackbone, vim_small, vim_tiny
from .slot_attention import AdaptiveSlotAttention
from .graph_mamba import GraphMambaLayer
from .sinkhorn_ot import SinkhornOT


class GeoSlot(nn.Module):
    """
    GeoSlot: Object-Centric State-Space Matching for CVGL.

    Architecture:
        Input Images (Query + Reference)
            ↓ (shared weights)
        Vision Mamba Backbone (SS2D, O(N))
            ↓
        Background Suppression Mask
            ↓
        Adaptive Slot Attention + Register Slots
            ↓
        Graph Mamba (Relational Reasoning)
            ↓
        Sinkhorn OT + MESH (Matching)
            ↓
        Similarity Score

    Args:
        backbone: Backbone variant ('vim_tiny', 'vim_small')
        img_size: Input image size
        embed_dim: Backbone embedding dimension
        slot_dim: Slot representation dimension
        max_slots: Maximum number of object slots per view
        n_register: Number of register (noise-absorbing) slots
        n_heads: Number of attention heads in Slot Attention
        sa_iters: Number of Slot Attention iterations
        gm_layers: Number of Graph Mamba layers
        k_neighbors: kNN neighbors for graph construction
        sinkhorn_iters: Number of Sinkhorn iterations
        epsilon: Sinkhorn entropy regularization
        mesh_iters: Number of MESH sharpening steps
    """
    def __init__(
        self,
        backbone: str = 'vim_tiny',
        img_size: int = 224,
        embed_dim: int = 192,
        slot_dim: int = 256,
        max_slots: int = 16,
        n_register: int = 4,
        n_heads: int = 4,
        sa_iters: int = 3,
        gm_layers: int = 2,
        k_neighbors: int = 5,
        sinkhorn_iters: int = 20,
        epsilon: float = 0.05,
        mesh_iters: int = 3,
        embed_dim_out: int = 512,
    ):
        super().__init__()

        # ===== Backbone (shared weights) =====
        if backbone == 'vim_tiny':
            self.backbone = VimBackbone(
                img_size=img_size, embed_dim=192, depth=12, embed_dim_out=embed_dim_out,
                return_dense=True,
            )
            embed_dim = 192
        elif backbone == 'vim_small':
            self.backbone = VimBackbone(
                img_size=img_size, embed_dim=384, depth=24, embed_dim_out=embed_dim_out,
                return_dense=True,
            )
            embed_dim = 384
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # ===== Slot Attention (shared weights) =====
        self.slot_attention = AdaptiveSlotAttention(
            feature_dim=embed_dim,
            slot_dim=slot_dim,
            max_slots=max_slots,
            n_register=n_register,
            n_heads=n_heads,
            iters=sa_iters,
        )

        # ===== Graph Mamba (shared weights) =====
        self.graph_mamba = GraphMambaLayer(
            slot_dim=slot_dim,
            k_neighbors=k_neighbors,
            num_layers=gm_layers,
        )

        # ===== Sinkhorn OT Matcher =====
        self.ot_matcher = SinkhornOT(
            slot_dim=slot_dim,
            num_iters=sinkhorn_iters,
            epsilon=epsilon,
            mesh_iters=mesh_iters,
        )

        # ===== CGP Head for global embedding (used in InfoNCE/DWBL) =====
        self.embed_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, embed_dim_out),
        )

    def encode_view(self, x, global_step=None):
        """
        Encode a single view (query or reference) through the full pipeline.

        Args:
            x: [B, C, H, W] input image
            global_step: Current training step

        Returns:
            dict with:
                - slots: [B, K, D] graph-enhanced object slots
                - embedding: [B, D_out] global embedding vector
                - keep_mask: [B, K] active slot mask
                - bg_mask: [B, N, 1] background suppression mask
                - attn_maps: [B, K_total, N] slot attention maps
        """
        # 1. Backbone: extract dense features
        features = self.backbone(x)  # [B, N, embed_dim]

        # 2. Slot Attention: decompose into object slots
        sa_out = self.slot_attention(features, global_step)
        object_slots = sa_out['object_slots']     # [B, max_slots, slot_dim]
        keep_mask = sa_out['keep_decision']        # [B, max_slots]

        # 3. Graph Mamba: relational reasoning
        enhanced_slots = self.graph_mamba(object_slots, keep_mask)  # [B, max_slots, slot_dim]

        # 4. Global embedding: weighted average of active slots
        weights = keep_mask / (keep_mask.sum(dim=-1, keepdim=True) + 1e-8)  # [B, K]
        global_slot = (enhanced_slots * weights.unsqueeze(-1)).sum(dim=1)    # [B, slot_dim]
        embedding = self.embed_head(global_slot)  # [B, embed_dim_out]
        embedding = F.normalize(embedding, dim=-1)

        return {
            'slots': enhanced_slots,
            'embedding': embedding,
            'keep_mask': keep_mask,
            'bg_mask': sa_out['bg_mask'],
            'attn_maps': sa_out['attn_maps'],
            'keep_probs': sa_out['keep_probs'],
            'register_slots': sa_out['register_slots'],
        }

    def forward(self, query_img, ref_img, global_step=None):
        """
        Full forward pass: encode both views, compute similarity.

        Args:
            query_img: [B, C, H, W] query view (drone/street/panorama)
            ref_img: [B, C, H, W] reference view (satellite)
            global_step: Current training step

        Returns:
            dict with all outputs needed for loss computation:
                - similarity: [B] OT-based similarity scores
                - query/ref embeddings, slots, masks, etc.
        """
        # Encode both views (shared weights)
        query_out = self.encode_view(query_img, global_step)
        ref_out = self.encode_view(ref_img, global_step)

        # OT-based matching between slot sets
        ot_out = self.ot_matcher(
            query_out['slots'], ref_out['slots'],
            mask_q=query_out['keep_mask'],
            mask_r=ref_out['keep_mask'],
        )

        return {
            # Similarity
            'similarity': ot_out['similarity'],
            'transport_plan': ot_out['transport_plan'],
            'transport_cost': ot_out['transport_cost'],
            # Query branch
            'query_embedding': query_out['embedding'],
            'query_slots': query_out['slots'],
            'query_keep_mask': query_out['keep_mask'],
            'query_bg_mask': query_out['bg_mask'],
            'query_attn_maps': query_out['attn_maps'],
            'query_keep_probs': query_out['keep_probs'],
            # Reference branch
            'ref_embedding': ref_out['embedding'],
            'ref_slots': ref_out['slots'],
            'ref_keep_mask': ref_out['keep_mask'],
            'ref_bg_mask': ref_out['bg_mask'],
            'ref_attn_maps': ref_out['attn_maps'],
            'ref_keep_probs': ref_out['keep_probs'],
        }

    def extract_embedding(self, x, global_step=None):
        """
        Extract only the global embedding for retrieval (inference).

        Args:
            x: [B, C, H, W] input image

        Returns:
            embedding: [B, D_out] L2-normalized embedding
        """
        out = self.encode_view(x, global_step)
        return out['embedding']
