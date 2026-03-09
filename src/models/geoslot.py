"""
GeoSlot 2.0: Full Pipeline Model for Cross-View Geo-Localization.

Combines all modules into a single end-to-end model:
VimBackbone → Adaptive Gumbel-Sparsity Mask → AdaSlot Attention
           → Hilbert Graph Mamba → Unbalanced FGW OT

Shared-weight Siamese architecture for processing query and reference views.

Key improvements over v1:
- Adaptive Gumbel-Sparsity Mask: γ learned per-image (replaces static α=0.7)
- Hilbert Curve ordering in Graph Mamba (rotation-invariant)
- Fused Gromov-Wasserstein: graph-to-graph matching (node + topology)
- Unbalanced FGW: KL-relaxed marginals for occlusion handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .vim_backbone import VimBackbone, vim_small, vim_tiny
from .slot_attention import AdaptiveSlotAttention
from .graph_mamba import GraphMambaLayer
from .fgw_ot import FusedGromovWasserstein
from .sinkhorn_ot import SinkhornOT


class GeoSlot(nn.Module):
    """
    GeoSlot 2.0: Object-Centric State-Space Graph Matching for CVGL.

    Architecture:
        Input Images (Query + Reference)
            ↓ (shared weights)
        Vision Mamba Backbone (SS2D, O(N))
            ↓
        Adaptive Gumbel-Sparsity Mask (γ = σ(MLP(GAP(F))))
            ↓
        Adaptive Slot Attention + Register Slots
            ↓
        Hilbert Graph Mamba (Rotation-Invariant Relational Reasoning)
            ↓
        Unbalanced Fused Gromov-Wasserstein (Graph-to-Graph Matching)
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
        matching: Matching strategy ('fgw' or 'sinkhorn')
        lambda_fgw: FGW trade-off between feature and structure cost
        tau_kl: KL penalty for unbalanced FGW
        fgw_iters: Number of FGW outer iterations
        sinkhorn_iters: Number of Sinkhorn iterations
        epsilon: Entropic regularization
        mesh_iters: Number of MESH sharpening steps (sinkhorn only)
        graph_order: Graph ordering strategy ('hilbert', 'spatial', 'degree')
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
        matching: str = 'fgw',
        lambda_fgw: float = 0.5,
        tau_kl: float = 0.1,
        fgw_iters: int = 10,
        sinkhorn_iters: int = 20,
        epsilon: float = 0.05,
        mesh_iters: int = 3,
        graph_order: str = 'hilbert',
        embed_dim_out: int = 512,
    ):
        super().__init__()
        self.matching_type = matching

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
            strategy=graph_order,
        )

        # ===== Matching Module =====
        if matching == 'fgw':
            self.ot_matcher = FusedGromovWasserstein(
                slot_dim=slot_dim,
                lambda_fgw=lambda_fgw,
                tau_kl=tau_kl,
                num_outer_iters=fgw_iters,
                num_sinkhorn_iters=sinkhorn_iters,
                epsilon=epsilon,
            )
        else:
            # Fallback: original Sinkhorn OT (for ablation)
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
                - centroids: [B, K, 2] slot spatial centroids
                - keep_mask: [B, K] active slot mask
                - bg_mask: [B, N, 1] background suppression mask
                - adaptive_gamma: [B, 1] learned coverage threshold
                - attn_maps: [B, K_total, N] slot attention maps
        """
        # 1. Backbone: extract dense features
        features = self.backbone(x)  # [B, N, embed_dim]

        # Force float32 for all post-backbone ops:
        # SlotAttention (GRU, softmax), Gumbel (log-log noise),
        # GraphMamba (SSM blocks) are unstable in float16 under AMP.
        with torch.cuda.amp.autocast(enabled=False):
            features = features.float()

            # 2. Slot Attention: decompose into object slots
            sa_out = self.slot_attention(features, global_step)
            object_slots = sa_out['object_slots']     # [B, max_slots, slot_dim]
            keep_mask = sa_out['keep_decision']        # [B, max_slots]

            # 3. Graph Mamba: relational reasoning with spatial encoding
            # Infer spatial dimensions from backbone output
            N = features.shape[1]
            H = W = int(N ** 0.5)
            if H * W != N:
                # For non-square feature maps (e.g., panoramic input)
                H = int(round(N ** 0.5))
                W = N // H

            enhanced_slots, centroids = self.graph_mamba(
                object_slots, keep_mask,
                attn_maps=sa_out['attn_maps'],
                spatial_hw=(H, W),
            )

            # 4. Global embedding: weighted average of active slots
            weights = keep_mask / (keep_mask.sum(dim=-1, keepdim=True) + 1e-8)  # [B, K]
            global_slot = (enhanced_slots * weights.unsqueeze(-1)).sum(dim=1)    # [B, slot_dim]
            embedding = self.embed_head(global_slot)  # [B, embed_dim_out]
            embedding = F.normalize(embedding, dim=-1)

        return {
            'slots': enhanced_slots,
            'embedding': embedding,
            'centroids': centroids,
            'keep_mask': keep_mask,
            'bg_mask': sa_out['bg_mask'],
            'adaptive_gamma': sa_out['adaptive_gamma'],
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
                - query/ref embeddings, slots, masks, centroids, etc.
        """
        # Encode both views (shared weights)
        query_out = self.encode_view(query_img, global_step)
        ref_out = self.encode_view(ref_img, global_step)

        # Matching between slot sets (FGW or Sinkhorn)
        if self.matching_type == 'fgw':
            ot_out = self.ot_matcher(
                query_out['slots'], ref_out['slots'],
                mask_q=query_out['keep_mask'],
                mask_r=ref_out['keep_mask'],
                centroids_q=query_out['centroids'],
                centroids_r=ref_out['centroids'],
            )
        else:
            # Sinkhorn fallback (no centroids)
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
            'query_centroids': query_out['centroids'],
            'query_keep_mask': query_out['keep_mask'],
            'query_bg_mask': query_out['bg_mask'],
            'query_adaptive_gamma': query_out['adaptive_gamma'],
            'query_attn_maps': query_out['attn_maps'],
            'query_keep_probs': query_out['keep_probs'],
            # Reference branch
            'ref_embedding': ref_out['embedding'],
            'ref_slots': ref_out['slots'],
            'ref_centroids': ref_out['centroids'],
            'ref_keep_mask': ref_out['keep_mask'],
            'ref_bg_mask': ref_out['bg_mask'],
            'ref_adaptive_gamma': ref_out['adaptive_gamma'],
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
