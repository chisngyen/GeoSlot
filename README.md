# GeoSlot: Object-Centric Cross-View Geo-Localization via Slot Transport

> **GeoSlot** — Slot-based object-centric alignment with Graph Mamba reasoning and Sinkhorn Optimal Transport for cross-view geo-localization.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🔬 Key Contributions

1. **Adaptive Slot Attention** with Register Slots and Background Masking for object-centric feature decomposition
2. **Graph Mamba Layer** — bidirectional SSM-based relational reasoning between visual slots
3. **Sinkhorn OT + MESH** — hard 1-to-1 slot matching via Optimal Transport with iterative sharpening
4. **MambaVision-L backbone** — hybrid Mamba + Transformer pretrained on ImageNet-1K

## 📊 Benchmarks & Targets

| Dataset | Metric | SOTA | GeoSlot Target |
|---|---|---|---|
| **CVUSA** | R@1 | 98.68 (Sample4Geo) | ≥98.5 |
| **University-1652** | R@1 / AP | 97.43 / 96.88 | ≥97.5 / ≥97.0 |
| **VIGOR (Same-Area)** | R@1 / Hit@1 | 80.34 / 93.78 (AuxGeo) | ≥82 / ≥94 |
| **VIGOR (Cross-Area)** | R@1 / Hit@1 | ~54 / ~72 (GeoDTR+) | Report |
| **CV-Cities** | R@1 | No baseline | Establish |

## 🏗️ Architecture

```
Input Image → MambaVision-L → Dense Features [B, 49, 640]
    → Background Mask → Foreground Features
    → Adaptive Slot Attention (12 object + 4 register slots)
    → Gumbel Selector (dynamic slot pruning)
    → Graph Mamba (bidirectional relational reasoning)
    → Sinkhorn OT + MESH (1-to-1 slot transport)
    → Embedding [B, 512]
```

## 📁 Project Structure

```
GeoSlot/
├── src/
│   ├── models/
│   │   ├── geoslot.py          # Full model pipeline
│   │   ├── vim_backbone.py     # Vision Mamba backbone
│   │   ├── slot_attention.py   # Adaptive Slot Attention
│   │   ├── graph_mamba.py      # Graph Mamba Layer
│   │   └── sinkhorn_ot.py      # Sinkhorn OT + MESH
│   ├── losses/
│   │   ├── joint_loss.py       # Multi-stage joint loss
│   │   ├── infonce.py          # Symmetric InfoNCE
│   │   ├── dwbl.py             # Distance-Weighted Batch Loss
│   │   └── contrastive_slot.py # Contrastive Slot Matching
│   ├── datasets/
│   │   ├── data_loader.py      # Dataset loaders
│   │   └── test_pipeline.py    # Pipeline validation
│   └── configs/
│       └── default.py          # Default hyperparameters
│
├── kaggle/                     # Self-contained Kaggle scripts
│   ├── geoslot_model.py        # Shared model code
│   ├── phase1_train_cvusa_kaggle.py
│   ├── phase2_train_university1652_kaggle.py
│   ├── phase3_train_vigor_kaggle.py
│   ├── phase4_train_cv_cities_kaggle.py
│   └── ablation_university1652_kaggle.py
│
├── docs/                       # Experiment documentation
│   ├── phase1_cvusa.md
│   ├── phase2_university1652.md
│   ├── phase3_vigor.md
│   ├── phase4_cv_cities.md
│   └── ablation_study.md
│
└── train.py                    # Main training entrypoint
```

## 🚀 Quick Start

### Requirements
```bash
pip install torch torchvision transformers mambavision timm tqdm
```

### Training (Kaggle H100)

```bash
# Phase 1: CVUSA
python kaggle/phase1_train_cvusa_kaggle.py

# Phase 2: University-1652 (Main Benchmark)
python kaggle/phase2_train_university1652_kaggle.py

# Phase 3: VIGOR (Hardest Benchmark)
python kaggle/phase3_train_vigor_kaggle.py

# Ablation Study
python kaggle/ablation_university1652_kaggle.py
```

## 📄 Citation

```bibtex
@article{geoslot2026,
  title={GeoSlot: Object-Centric Cross-View Geo-Localization via Slot Transport},
  author={Tran Chi Nguyen},
  year={2026}
}
```

## 📝 License

This project is licensed under the MIT License.
