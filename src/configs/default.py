"""
Default configuration for GeoSlot training.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    backbone: str = 'vim_tiny'          # 'vim_tiny', 'vim_small'
    img_size: int = 224
    embed_dim: int = 192               # Auto-set based on backbone
    slot_dim: int = 256
    max_slots: int = 16
    n_register: int = 4
    n_heads: int = 4
    sa_iters: int = 3                   # Slot Attention iterations
    gm_layers: int = 2                  # Graph Mamba layers
    k_neighbors: int = 5               # kNN graph neighbors
    sinkhorn_iters: int = 20
    epsilon: float = 0.05              # Entropic regularization
    mesh_iters: int = 3                # MESH sharpening steps (sinkhorn only)
    embed_dim_out: int = 512           # Final embedding dimension
    # GeoSlot 2.0 — FGW & Hilbert parameters
    matching: str = 'fgw'              # 'fgw' (default) or 'sinkhorn' (ablation)
    lambda_fgw: float = 0.5            # FGW trade-off: feature (0) vs structure (1)
    tau_kl: float = 0.1                # KL penalty for unbalanced FGW
    fgw_iters: int = 10                # Number of FGW outer iterations
    graph_order: str = 'hilbert'       # 'hilbert' (default), 'spatial', 'degree'


@dataclass
class LossConfig:
    """Loss function configuration."""
    lambda_infonce: float = 1.0
    lambda_dwbl: float = 1.0
    lambda_csm: float = 0.5            # Contrastive Slot Matching
    lambda_dice: float = 0.3
    temperature: float = 0.07
    stage2_epoch: int = 30             # Enable Slot losses
    stage3_epoch: int = 60             # Enable full pipeline losses


@dataclass
class TrainConfig:
    """Training configuration."""
    # Dataset
    dataset: str = 'university1652'    # 'university1652', 'vigor', 'cvusa', 'cv_cities'
    query_view: str = 'drone'          # For University-1652
    gallery_view: str = 'satellite'    # For University-1652
    vigor_cities: List[str] = field(default_factory=lambda: ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle'])

    # Training params
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Scheduler
    scheduler: str = 'cosine'          # 'cosine', 'step'
    step_size: int = 30
    gamma: float = 0.5

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    amp: bool = True                   # Mixed precision

    # Saving
    save_freq: int = 10                # Save checkpoint every N epochs
    eval_freq: int = 5                 # Evaluate every N epochs
    output_dir: str = './outputs'
    exp_name: str = 'GeoSlot_geo'

    # Resume
    resume: Optional[str] = None       # Path to checkpoint


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    batch_size: int = 64
    top_k: List[int] = field(default_factory=lambda: [1, 5, 10])
    save_embeddings: bool = True       # Save to .npy for analysis


# ============================================================================
# Kaggle-specific paths
# ============================================================================
KAGGLE_PATHS = {
    'university1652': '/kaggle/input/datasets/chinguyeen/university-1652/University-1652',
    'cvusa': '/kaggle/input/datasets/chinguyeen/cvusa-subdataset/CVUSA',
    'cv_cities': '/kaggle/input/datasets/chisboiz/cv-cities',
    'vigor_chicago': '/kaggle/input/datasets/chinguyeen/vigor-chicago',
    'vigor_newyork': '/kaggle/input/datasets/chinguyeen/vigor-newyork',
    'vigor_sanfrancisco': '/kaggle/input/datasets/chinguyeen/vigor-sanfrancisco',
    'vigor_seattle': '/kaggle/input/datasets/chinguyeen/vigor-seattle',
}
