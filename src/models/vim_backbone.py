"""
Vision Mamba (Vim) Backbone for Cross-View Geo-Localization.

Reference:
- hustvl/Vim: https://github.com/hustvl/Vim
- VimGeo (IJCAI 2025): https://github.com/VimGeoTeam/VimGeo
- mamba_ssm: https://github.com/state-spaces/mamba

This module wraps Vision Mamba with Channel Group Pooling (CGP) head
for efficient O(N) feature extraction from high-resolution images.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("[WARNING] mamba_ssm not installed. Using fallback linear attention.")


# ============================================================================
# Fallback: Linear Attention (when mamba_ssm is not available)
# ============================================================================
class LinearAttention(nn.Module):
    """Fallback linear attention when mamba_ssm is not installed."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = int(d_model * expand)
        self.proj_in = nn.Linear(d_model, d_inner * 2)
        self.proj_out = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_inner)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: [B, L, D]
        z, gate = self.proj_in(x).chunk(2, dim=-1)
        z = self.act(z) * torch.sigmoid(gate)
        return self.proj_out(self.norm(z))


# ============================================================================
# SS2D Block: Bidirectional Selective Scan (4 directions)
# ============================================================================
class SS2DBlock(nn.Module):
    """
    Selective Scan 2D block with multi-directional scanning.
    Processes 2D feature maps by scanning in 4 directions (→←↓↑)
    then merging results for global receptive field.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, num_directions=4):
        super().__init__()
        self.d_model = d_model
        self.num_directions = num_directions

        if HAS_MAMBA:
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(num_directions)
            ])
        else:
            self.mamba_layers = nn.ModuleList([
                LinearAttention(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(num_directions)
            ])

        self.merge = nn.Linear(d_model * num_directions, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _scan_directions(self, x_2d):
        """
        Generate 4 scanning sequences from a 2D feature map.
        x_2d: [B, H, W, D]
        Returns: list of [B, H*W, D] tensors (4 directions)
        """
        B, H, W, D = x_2d.shape

        # Direction 1: Left→Right, Top→Bottom (row-major)
        seq_lr = x_2d.reshape(B, H * W, D)

        # Direction 2: Right→Left, Bottom→Top (reverse)
        seq_rl = x_2d.flip(dims=[1, 2]).reshape(B, H * W, D)

        # Direction 3: Top→Bottom, Left→Right (column-major)
        seq_tb = x_2d.permute(0, 2, 1, 3).reshape(B, H * W, D)

        # Direction 4: Bottom→Top, Right→Left (reverse column-major)
        seq_bt = x_2d.permute(0, 2, 1, 3).flip(dims=[1, 2]).reshape(B, H * W, D)

        return [seq_lr, seq_rl, seq_tb, seq_bt]

    def forward(self, x_2d):
        """
        x_2d: [B, H, W, D]
        Returns: [B, H, W, D]
        """
        B, H, W, D = x_2d.shape
        sequences = self._scan_directions(x_2d)

        outputs = []
        for i, (seq, mamba) in enumerate(zip(sequences, self.mamba_layers)):
            out = mamba(seq)  # [B, H*W, D]
            # Reverse the scanning order to align back
            if i == 1:  # RL was flipped
                out = out.flip(dims=[1])
            elif i == 2:  # TB was permuted
                out = out.reshape(B, W, H, D).permute(0, 2, 1, 3).reshape(B, H * W, D)
            elif i == 3:  # BT was permuted + flipped
                out = out.flip(dims=[1]).reshape(B, W, H, D).permute(0, 2, 1, 3).reshape(B, H * W, D)
            outputs.append(out)

        # Merge all directions
        merged = torch.cat(outputs, dim=-1)  # [B, H*W, 4D]
        merged = self.merge(merged)  # [B, H*W, D]
        merged = self.norm(merged)

        return merged.reshape(B, H, W, D)


# ============================================================================
# Patch Embedding
# ============================================================================
class PatchEmbed(nn.Module):
    """Convert image to patch embeddings."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B = x.shape[0]
        x = self.proj(x)  # [B, D, H/P, W/P]
        H, W = x.shape[2], x.shape[3]
        x = x.permute(0, 2, 3, 1)  # [B, H/P, W/P, D]
        x = self.norm(x)
        return x, H, W


# ============================================================================
# Vision Mamba Layer
# ============================================================================
class VimLayer(nn.Module):
    """A single Vision Mamba layer: SS2D + FFN with residual connections."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ss2d = SS2DBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm2 = nn.LayerNorm(d_model)

        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(drop),
        )

    def forward(self, x):
        """x: [B, H, W, D]"""
        x = x + self.ss2d(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# Channel Group Pooling (CGP) Head — from VimGeo
# ============================================================================
class ChannelGroupPooling(nn.Module):
    """
    Channel Group Pooling (CGP) from VimGeo (IJCAI 2025).

    Instead of Global Average Pooling + FC, CGP:
    1. Splits channels into G groups
    2. Applies spatial pooling per group → preserves local detail
    3. Concatenates group features → final embedding

    This retains spatial local information that GAP destroys.
    """
    def __init__(self, in_dim, out_dim, num_groups=8):
        super().__init__()
        self.num_groups = num_groups
        assert in_dim % num_groups == 0, f"in_dim {in_dim} must be divisible by num_groups {num_groups}"
        self.group_dim = in_dim // num_groups

        # Per-group spatial attention
        self.group_attn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.group_dim, 1),
                nn.Softmax(dim=1)
            ) for _ in range(num_groups)
        ])

        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        x: [B, N, D] where N = num_patches, D = feature_dim
        Returns: [B, out_dim]
        """
        B, N, D = x.shape
        groups = x.reshape(B, N, self.num_groups, self.group_dim)  # [B, N, G, Dg]

        pooled = []
        for g in range(self.num_groups):
            group_feat = groups[:, :, g, :]  # [B, N, Dg]
            attn_weights = self.group_attn[g](group_feat)  # [B, N, 1]
            pooled_feat = (group_feat * attn_weights).sum(dim=1)  # [B, Dg]
            pooled.append(pooled_feat)

        out = torch.cat(pooled, dim=-1)  # [B, D]
        out = self.norm(self.proj(out))  # [B, out_dim]
        return out


# ============================================================================
# Full VimBackbone
# ============================================================================
class VimBackbone(nn.Module):
    """
    Vision Mamba Backbone for Cross-View Geo-Localization.

    Pipeline: Image → PatchEmbed → N × VimLayer (SS2D) → Dense Features

    Args:
        img_size: Input image size (square)
        patch_size: Patch size for tokenization
        in_chans: Number of input channels
        embed_dim: Embedding dimension
        depth: Number of Vim layers
        d_state: SSM state dimension
        d_conv: SSM convolution width
        expand: SSM expansion factor
        mlp_ratio: MLP hidden dim ratio
        drop_rate: Dropout rate
        embed_dim_out: Output embedding dimension for CGP head
        num_groups: Number of groups for CGP
        return_dense: If True, return dense features [B, N, D]; else return pooled [B, D]
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=4.0,
        drop_rate=0.0,
        embed_dim_out=512,
        num_groups=8,
        return_dense=True,
    ):
        super().__init__()
        self.return_dense = return_dense
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches
        self.base_grid = img_size // patch_size  # e.g. 14 for 224/16

        # Positional embedding (learned for base_grid×base_grid, interpolated at runtime)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Vim layers
        self.layers = nn.ModuleList([
            VimLayer(
                d_model=embed_dim, d_state=d_state, d_conv=d_conv,
                expand=expand, mlp_ratio=mlp_ratio, drop=drop_rate
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # CGP head (for producing final embedding vector)
        self.cgp_head = ChannelGroupPooling(embed_dim, embed_dim_out, num_groups=num_groups)

    def interpolate_pos_embed(self, H, W):
        """Interpolate positional embeddings for arbitrary grid sizes.

        Trained on base_grid×base_grid (e.g. 14×14 for 224×224 images),
        interpolated to H×W at runtime (e.g. 32×8 for 512×128 panorama).
        Uses bicubic interpolation (same as ViT/DeiT).

        Args:
            H, W: Target grid height and width (in patches)

        Returns:
            Positional embedding [1, H*W, D]
        """
        N = H * W
        if N == self.pos_embed.shape[1] and H == self.base_grid:
            return self.pos_embed  # No interpolation needed

        # Reshape to 2D grid → interpolate → flatten
        pe = self.pos_embed.reshape(1, self.base_grid, self.base_grid, -1).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(H, W), mode='bicubic', align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, N, -1)

    def forward_features(self, x):
        """Extract dense features from input image.

        Args:
            x: [B, C, H, W] input image (arbitrary size, must be divisible by patch_size)

        Returns:
            dense: [B, N, D] dense feature map
            H, W: grid height and width
        """
        x, H, W = self.patch_embed(x)  # [B, H, W, D]
        B = x.shape[0]

        # Add interpolated positional embedding
        pos = self.interpolate_pos_embed(H, W)
        x_flat = x.reshape(B, H * W, -1) + pos
        x_flat = self.pos_drop(x_flat)
        x = x_flat.reshape(B, H, W, -1)

        # Apply Vim layers
        for layer in self.layers:
            x = layer(x)  # [B, H, W, D]

        x = self.norm(x.reshape(B, H * W, -1))  # [B, N, D]
        return x, H, W

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] input image

        Returns:
            If return_dense=True: [B, N, D] dense feature map
            If return_dense=False: [B, embed_dim_out] pooled embedding
        """
        dense, H, W = self.forward_features(x)

        if self.return_dense:
            return dense  # [B, N, D] → feed to Slot Attention
        else:
            return self.cgp_head(dense)  # [B, embed_dim_out] → direct retrieval


# ============================================================================
# Factory functions
# ============================================================================
def vim_small(img_size=224, **kwargs):
    """Vim-Small: 24 layers, 384 dim (comparable to DeiT-Small)."""
    return VimBackbone(
        img_size=img_size, embed_dim=384, depth=24,
        d_state=16, d_conv=4, expand=2, **kwargs
    )

def vim_tiny(img_size=224, **kwargs):
    """Vim-Tiny: 12 layers, 192 dim (lightweight)."""
    return VimBackbone(
        img_size=img_size, embed_dim=192, depth=12,
        d_state=16, d_conv=4, expand=2, **kwargs
    )

def vim_base(img_size=224, **kwargs):
    """Vim-Base: 24 layers, 768 dim (comparable to ViT-Base)."""
    return VimBackbone(
        img_size=img_size, embed_dim=768, depth=24,
        d_state=16, d_conv=4, expand=2, **kwargs
    )
