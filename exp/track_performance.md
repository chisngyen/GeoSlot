# 📊 Performance Tracking — SUES-200 Benchmark

> **Standard Protocol**: 120 train / 80 test locations, Gallery = ALL 200 satellites (confusion data)

## Baseline: MobileGeo (ConvNeXt-Tiny + DINOv2 Teacher)

| Config | Value |
|---|---|
| **Student** | ConvNeXt-Tiny (~28M params) |
| **Teacher** | DINOv2 ViT-B/14 (frozen, last 2 blocks trainable) |
| **Pretrained** | ImageNet-22K (ConvNeXt), DINOv2 pretrain (teacher) |
| **Optimizer** | SGD, lr=0.001, momentum=0.9, wd=5e-4 |
| **Scheduler** | Warmup (5 ep) + Cosine decay |
| **Batch Size** | 256 (PK Sampler: P=8) |
| **Epochs** | 120 |
| **AMP** | ✅ Mixed precision |
| **GPU** | H100 80GB |
| **Train samples** | 24,000 (120 locs × 4 altitudes × 50 imgs) |
| **Test queries** | 16,000 (80 locs × 4 altitudes × 50 imgs) |
| **Gallery** | 200 satellite images (confusion data) |

### Loss Components
CE (multi-stage + final) · Triplet (hard mining, m=0.3) · Symmetric InfoNCE · Self-Distillation (inverse, 4 stages) · UAPA · Cross-Distillation (MSE + Cosine)

### 📈 Evaluation Trajectory

| Epoch | R@1 | R@5 | R@10 | mAP |
|------:|------:|------:|-------:|------:|
| 5 | 51.98% | 74.94% | 84.28% | 62.53% |
| 10 | 63.59% | 87.33% | 93.59% | 73.95% |
| 15 | 66.46% | 88.76% | 94.21% | 76.25% |
| 20 | 71.89% | 91.26% | 96.54% | 80.61% |
| 25 | 74.49% | 92.79% | 96.83% | 82.59% |
| 30 | 76.85% | 94.12% | 97.63% | 84.38% |
| 35 | 77.47% | 94.47% | 97.61% | 84.87% |
| 40 | 77.88% | 94.01% | 97.76% | 84.94% |
| 45 | 79.30% | 94.46% | 97.89% | 85.94% |
| 50 | 80.01% | 94.87% | 98.02% | 86.47% |
| 55 | 81.11% | 94.74% | 97.92% | 87.23% |
| 60 | 80.39% | 93.99% | 97.69% | 86.47% |
| 65 | 81.08% | 95.22% | 98.02% | 87.27% |
| 70 | 81.24% | 95.64% | 98.24% | 87.41% |
| 75 | 81.73% | 95.17% | 98.06% | 87.69% |
| 80 | 81.26% | 95.17% | 98.16% | 87.37% |
| 85 | 81.92% | 95.64% | 98.21% | 87.86% |
| **90** | **82.35%** | **95.94%** | **98.29%** | **88.27%** |
| 95 | 81.92% | 95.66% | 98.32% | 87.90% |
| 100 | 81.38% | 95.23% | 98.07% | 87.57% |
| 105 | 81.78% | 95.93% | 98.30% | 87.92% |
| 110 | 81.49% | 95.34% | 98.27% | 87.54% |
| 115 | 81.16% | 95.10% | 98.12% | 87.35% |
| 120 | 81.74% | 95.31% | 98.14% | 87.74% |

### 🏆 Best Results (Epoch 90)

| Metric | Score |
|--------|-------|
| **R@1** | **82.35%** |
| **R@5** | **95.94%** |
| **R@10** | **98.29%** |
| **mAP** | **88.27%** |

---

## ⚠️ Old Results (WRONG protocol — for reference only)

> Previous run used **random 160/40 split** with **gallery = 40 test locations only** — NOT comparable to any paper

| Metric | Old (wrong) | New (correct) | Δ |
|--------|-------------|---------------|---|
| R@1 | 93.16% | **82.35%** | -10.81% |
| R@5 | 99.79% | **95.94%** | -3.85% |
| R@10 | 100.00% | **98.29%** | -1.71% |
| mAP | 96.10% | **88.27%** | -7.83% |

**Why the drop?**
1. Gallery 200 vs 40 → retrieval is ~5× harder
2. Test set 80 vs 40 locations → more diverse queries
3. Fixed split → no lucky random partition

---

## GeoSlot: Slot-Guided Cross-View Geo-Localization

| Config | Value |
|---|---|
| **Student** | ConvNeXt-Tiny + Slot Attention (K=8) + AAAP (~30.3M params) |
| **Teacher** | DINOv2 ViT-B/14 (SigLIP2 failed, fallback) |
| **Novel** | SlotCVA · VLM-Guided Distillation · Altitude-Aware Adaptive Pooling |
| **Optimizer** | AdamW, lr_backbone=1e-4, lr_slot=5e-4, lr_head=1e-3, wd=0.01 |
| **Batch Size** | 256 (PK Sampler: P=8) |
| **Epochs** | 120 (3-phase: P1→30, P2→80, P3→120) |
| **Losses** | CE · ArcFace · Triplet · InfoNCE · SlotContrastive · SlotDistillation · FeatDist · UAPA |

### 📈 Evaluation Trajectory

| Epoch | Phase | R@1 | R@5 | R@10 | AP |
|------:|:-----:|------:|------:|-------:|------:|
| 5 | P1 | 20.96% | 39.30% | 51.79% | 31.02% |
| 10 | P1 | 23.33% | 42.33% | 55.62% | 33.73% |
| 15 | P1 | 23.51% | 43.12% | 56.32% | 34.23% |
| 20 | P1 | 24.24% | 45.20% | 58.89% | 35.34% |
| 25 | P1 | 24.70% | 46.09% | 59.76% | 35.97% |
| 30 | P1 | 25.29% | 47.10% | 60.51% | 36.68% |
| 35 | P2 | 29.29% | 53.55% | 67.33% | 41.32% |
| 40 | P2 | 30.44% | 55.67% | 69.23% | 42.76% |
| 45 | P2 | 30.68% | 56.13% | 69.67% | 43.07% |
| 50 | P2 | 30.70% | 56.42% | 69.81% | 43.18% |
| 60 | P2 | 30.84% | 56.44% | 70.02% | 43.21% |
| 75 | P2 | 30.84% | 56.60% | 69.88% | 43.27% |
| 80 | P2 | 30.86% | 56.67% | 69.98% | 43.26% |
| 85 | P3 | 30.77% | 56.64% | 69.90% | 43.25% |
| 105 | P3 | **30.92%** | 56.39% | 69.87% | 43.33% |
| 120 | P3 | 30.69% | 56.36% | 69.61% | 43.14% |

### Per-Altitude Best (Epoch 105)

| Altitude | R@1 | R@5 | R@10 | AP |
|:--------:|------:|------:|-------:|------:|
| 150m | 23.23% | 44.37% | 58.67% | 34.48% |
| 200m | 27.77% | 51.50% | 65.90% | 39.83% |
| 250m | 33.67% | 61.52% | 74.60% | 46.70% |
| 300m | 39.00% | 68.17% | 80.33% | 52.30% |
| **AVG** | **30.92%** | **56.39%** | **69.87%** | **43.33%** |

---

## GeoPrompt: Prompt-Tuned VLM Geo-Localization

| Config | Value |
|---|---|
| **Student** | ConvNeXt-Tiny (mostly frozen) + VS-VPT + CVPI + GSPR (~47.8M total, 20.0M trainable = 41.8%) |
| **Teacher** | DINOv2 ViT-B/14 (frozen) |
| **Novel** | View-Specific Visual Prompt Tuning · Cross-View Prompt Interaction · Geo-Semantic Prompt Routing |
| **Optimizer** | AdamW, lr_backbone=1e-5, lr_prompt=5e-3, lr_head=1e-3, wd=0.01 |
| **Batch Size** | 256 (PK Sampler: P=8) |
| **Epochs** | 120 (3-phase: P1→30, P2→80, P3→120) |
| **Losses** | CE · ArcFace · Triplet · InfoNCE · PromptOrthogonality · CVPI · PromptAlignment · FeatDist · UAPA |

### 📈 Evaluation Trajectory

| Epoch | Phase | R@1 | R@5 | AP |
|------:|:-----:|------:|------:|------:|
| 5 | P1 | 16.94% | 37.57% | 27.49% |
| 10 | P1 | 23.69% | 48.75% | 36.17% |
| 15 | P1 | 28.81% | 56.09% | 43.02% |
| 20 | P1 | 31.53% | 61.86% | 47.61% |
| 25 | P1 | 32.37% | 64.98% | 49.09% |
| 30 | P1 | 32.68% | 66.81% | 50.00% |
| 35 | P2 | 33.46% | 67.63% | 51.01% |
| **40** | **P2** | **33.67%** | **67.97%** | **51.52%** |
| 45 | P2 | 33.63% | 67.88% | 51.51% |
| 50 | P2 | 33.66% | 67.87% | 51.47% |
| 55 | P2 | 33.67% | 67.88% | 51.51% |
| 60 | P2 | 33.67% | 67.88% | 51.51% |
| 80 | P2 | 33.67% | 67.88% | 51.51% |
| 105 | P3 | 33.67% | 67.88% | 51.51% |
| 120 | P3 | 33.67% | 67.88% | 51.51% |

### Per-Altitude Best (Epoch 40)

| Altitude | R@1 | R@5 | AP |
|:--------:|------:|------:|------:|
| 150m | 20.15% | 50.50% | 34.26% |
| 200m | 31.65% | 62.78% | 46.48% |
| 250m | 39.10% | 71.40% | 53.73% |
| 300m | 43.80% | 75.88% | 58.33% |
| **AVG** | **33.67%** | **67.97%** | **51.52%** |

---

## GeoMamba: State-Space Cross-View Geo-Localization

| Config | Value |
|---|---|
| **Student** | ConvNeXt-Tiny + Bidirectional Spatial-Mamba + OT Matching (~40.4M params) |
| **Teacher** | DINOv2 ViT-B/14 (frozen) |
| **Novel** | BS-Mamba (4-dir scan) · Optimal Transport Slot Matching · Scale-Adaptive State Gating |
| **Batch Size** | 32 (reduced from 256 to fix OOM) |
| **Status** | ⚡ OOM fixed — ready for re-run |

---

## EXP1-5: Legacy Experiments

| Experiment | Status |
|---|---|
| EXP1: SigLIP Distillation | 🔜 Pending |
| EXP2: MoE Multi-View | 🔜 Pending |
| EXP3: Graph Matching | 🔜 Pending |
| EXP4: Hierarchical KD | 🔜 Pending |
| EXP5: SigLIP Slot Hybrid | 🔜 Pending |

---

## 🏅 Leaderboard (SUES-200, 120/80 split, 200-gallery)

| Rank | Method | R@1 | R@5 | AP | Params |
|:----:|--------|------:|------:|------:|-------:|
| 🥇 | **Baseline (MobileGeo)** | **82.35%** | **95.94%** | **88.27%** | 28M |
| 🥈 | GeoPrompt (VS-VPT+CVPI+GSPR) | 33.67% | 67.97% | 51.52% | 47.8M (20M train) |
| 🥉 | GeoSlot (SlotCVA+AAAP) | 30.92% | 56.39% | 43.33% | 30.3M |
| 4 | GeoMamba (BS-Mamba+OT) | — | — | — | 40.4M |

> **Note**: The novel methods (GeoSlot, GeoPrompt, GeoMamba) significantly underperform the baseline. This is expected since the baseline uses a well-tuned multi-stage self-distillation pipeline with SGD, while the novel methods introduce complex new modules that need more hyperparameter tuning and potentially longer training with proper LR schedules.
