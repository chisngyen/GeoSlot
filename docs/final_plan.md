# 🏗️ FINAL PLAN: GeoSlot 2.0 — Object-Centric Graph Matching for Cross-View Geo-Localization

> **Tên dự án:** *GeoSlot: Object-Centric State-Space Graph Matching for Cross-View Geo-Localization*
> **Ngày cập nhật:** 2026-03-09 · **Target:** ACM Multimedia 2025 · **Phần cứng:** Kaggle H100

---

## 1. Tổng quan Ý tưởng (GeoSlot 2.0)

Xây dựng hệ thống CVGL **End-to-End** kết hợp suy luận dựa trên đối tượng và đối sánh đồ thị:

| Thành phần | Vai trò | Novelty |
|---|---|---|
| **Vision Mamba (SS2D)** | Backbone trích xuất đặc trưng $\mathcal{O}(N)$ | Thay thế ViT $\mathcal{O}(N^2)$ |
| **Adaptive Gumbel-Sparsity Mask** | Triệt tiêu nền động với γ tự học | Khắc phục Static Coverage Fallacy |
| **Adaptive Slot Attention + Register** | Phân tách Object-Centric + noise absorption | Chưa ai dùng trong CVGL |
| **Spatial 2D Graph Mamba + Hilbert** | Suy luận quan hệ bất biến quay | Hilbert Curve + SSM = novel |
| **Unbalanced Fused Gromov-Wasserstein** | Đối sánh Graph-to-Graph (node + edge) | FGW chưa từng dùng trong CVGL |
| **Multi-Layer Joint Loss** | InfoNCE + DWBL + CSM + Dice + Adaptive coverage | Stage-wise training |

**Paradigm Shift:** Ảnh → *{Tòa nhà A, Ngã tư B, Bãi cỏ C}* → xây đồ thị quan hệ → đối sánh đồ thị-to-đồ thị qua FGW OT.

---

## 2. Kiến trúc Tổng thể (Architecture GeoSlot 2.0)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGES                                 │
│   Query (Drone/Street)              Reference (Satellite)           │
└──────────┬──────────────────────────────────┬───────────────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────┐         ┌──────────────────────┐
│  Vision Mamba (SS2D) │         │  Vision Mamba (SS2D) │  ← Shared Weights
│  + CGP Head          │         │  + CGP Head          │     O(N) linear
└──────────┬───────────┘         └──────────┬───────────┘
           │ Dense Feature Map F             │ Dense Feature Map F
           ▼                                  ▼
┌──────────────────────┐         ┌──────────────────────┐
│ Adaptive Gumbel-     │         │ Adaptive Gumbel-     │  ← γ = σ(MLP(GAP(F)))
│ Sparsity Mask        │         │ Sparsity Mask        │     L_hinge adaptive
└──────────┬───────────┘         └──────────┬───────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────┐         ┌──────────────────────┐
│ Adaptive Slot Attn   │         │ Adaptive Slot Attn   │
│ + Register Slots     │         │ + Register Slots     │
│ (AdaSlot, K dynamic) │         │ (AdaSlot, K dynamic) │
└──────────┬───────────┘         └──────────┬───────────┘
           │ K object slots + centroids      │ M object slots + centroids
           ▼                                  ▼
┌──────────────────────┐         ┌──────────────────────┐
│ Spatial 2D Graph     │         │ Spatial 2D Graph     │
│ Mamba (Hilbert Curve)│         │ Mamba (Hilbert Curve)│  ← Bất biến quay
└──────────┬───────────┘         └──────────┬───────────┘
           │ Enhanced slots + centroids      │ Enhanced slots + centroids
           └──────────┬───────────────────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │  Unbalanced Fused    │
           │  Gromov-Wasserstein  │  ← Node features + Graph topology
           │  (UFGW) Matching    │     KL-relaxed marginals
           │  → Similarity Score │
           └──────────────────────┘
```

### 2.1. Chi tiết từng Module

#### Module A: Vision Mamba Backbone (SS2D)
- **Source:** `hustvl/Vim` hoặc MambaVision-L (ImageNet pre-trained)
- **Kỹ thuật:** Multi-directional scan (4 hướng: ↑↓←→) → global receptive field
- **Head:** Channel Group Pooling (CGP), giữ local detail
- **Output:** Dense features $F \in \mathbb{R}^{N \times d_f}$, $N = (H/P) \times (W/P)$

#### Module B: Adaptive Gumbel-Sparsity Mask (★ MỚI)
- **Thay đổi so với v1:** `target_ratio=0.7` cố định → `γ = σ(MLP(GAP(F)))` tự học
- **Loss:** `L_adaptive = max(0, γ - mean(m))` (Adaptive Hinge Loss)
- **Hiệu quả:**
  - Sa mạc/nông thôn: γ→1.0, giữ 100% features
  - Đô thị đông: γ→0.4, cắt 60% nhiễu động
- **Entropy regularization:** Giữ nguyên (ngăn mask collapse)

#### Module C: Adaptive Slot Attention + Register Slots
- **AdaSlot:** K slots tự điều chỉnh, Gumbel-Softmax pruning
- **Register Slots:** R=4 slots hấp thụ nhiễu, loại bỏ trước matching
- **Iterative Routing:** 3 vòng GRU + softmax cạnh tranh
- **Output bổ sung:** Centroids $(c_y^k, c_x^k)$ cho mỗi slot

#### Module D: Spatial 2D Graph Mamba + Hilbert Curve (★ MỚI)
- **Thay đổi so với v1:** Raster-scan → Hilbert Curve ordering
- **Hilbert mapping:** $\pi_{Hilbert} = \text{argsort}(\mathcal{H}(c^k))$ bảo toàn spatial locality
- **kNN graph:** Dựa trên semantic + spatial distance
- **Bidirectional Mamba scan:** Forward + backward dọc theo Hilbert order
- **Bất biến quay:** Xoay ảnh không phá vỡ thứ tự Hilbert

#### Module E: Unbalanced Fused Gromov-Wasserstein (★ MỚI)
- **Thay đổi so với v1:** Sinkhorn + MESH → UFGW
- **Feature cost:** $C_{ij} = ||\phi(s_i^q) - \phi(s_j^r)||_2$ (node features)
- **Structure cost:** $|S^q_{ik} - S^r_{jl}|^2$ (graph topology)
- **FGW loss:** $(1-\lambda) \cdot \text{Wasserstein} + \lambda \cdot \text{Gromov-Wasserstein}$
- **Unbalanced:** KL divergence nới lỏng marginals → xử lý occlusion
- **Similarity:** $s(x_q, x_r) = -\mathcal{L}_{UFGW}$

---

## 3. Hệ thống Loss Functions

### Tầng 1: Embedding & Metric Learning (Epoch 1-30)
| Loss | Mục đích | Trọng số |
|---|---|---|
| **Symmetric InfoNCE** | Căn chỉnh embedding 2 branches | $\lambda = 1.0$ |
| **DWBL** | Hard negative mining tự động | $\lambda = 1.0$ |

### Tầng 2: Slot Quality (Epoch 30-60)
| Loss | Mục đích | Trọng số |
|---|---|---|
| **Contrastive Slot Matching** | Tối đa MI giữa matched slots | $\lambda = 0.5$ |
| **Dice Loss** | Giải quyết scale imbalance | $\lambda = 0.3$ |

### Background Regularization (Always active)
| Loss | Mục đích | Trọng số |
|---|---|---|
| **Entropy Regularization** | Ngăn mask collapse | $\lambda = 0.01$ |
| **Adaptive Hinge Coverage** | Đảm bảo coverage ≥ γ | $\lambda = 0.01$ |

---

## 4. Chiến thuật Dataset: Train / Test / Benchmark

```
┌─────────────────────────────────────────────────────────────────┐
│                    VAI TRÒ CỦA TỪNG DATASET                    │
├─────────────────┬───────────────────────────────────────────────┤
│ University-1652 │ 🎯 MAIN BENCHMARK #1                        │
│                 │ • Drone→Sat, Sat→Drone                       │
│                 │ • Multi-altitude, scale variation             │
│                 │ • FGW giải quyết geometric homography         │
├─────────────────┼───────────────────────────────────────────────┤
│ VIGOR           │ 🎯 MAIN BENCHMARK #2                        │
│                 │ • Panorama→Sat (Same-Area & Cross-Area)      │
│                 │ • Cross-Area: Slot Attention >> Texture-based │
├─────────────────┼───────────────────────────────────────────────┤
│ CVUSA           │ 📊 SHOWCASE / SATURATION                    │
│                 │ • Mục tiêu ≥ 99% R@1                        │
├─────────────────┼───────────────────────────────────────────────┤
│ CV-Cities       │ 🌍 GENERALIZATION TEST                      │
│                 │ • Cross-city generalization                   │
└─────────────────┴───────────────────────────────────────────────┘
```

### Target SOTA

| Dataset | Split | Metric | Mục tiêu |
|---|---|---|---|
| University-1652 | Drone→Sat | Recall@1, AP | > 85% |
| VIGOR Same-Area | Pano→Sat | Hit Rate@1 | > 95% |
| VIGOR Cross-Area | Pano→Sat | Hit Rate@1 | > 25% |
| CVUSA | Pano→Sat | Recall@1 | ≥ 99% |

---

## 5. Ablation Studies (Bắt buộc cho Paper)

| # | Thí nghiệm | Mục đích |
|---|---|---|
| 1 | Mamba Vision vs ViT | Latency + Peak Memory benchmark |
| 2 | + Slot Attention | Hiệu quả Object-Centric decomposition |
| 3 | + Register + Adaptive BG Mask | So sánh adaptive vs static (α=0.7) |
| 4 | + Graph Mamba (Hilbert) | So sánh Hilbert vs Raster-Scan ordering |
| 5 | + FGW matching | So sánh FGW vs Sinkhorn OT |
| 6 | **Full UFGW** | **Proposed method** |
| 7 | Rotation Robustness | R@1 stability under 0°-360° rotation |
| 8 | Interpretability | Slot Heatmaps + FGW Transport visualization |

---

## 6. Rủi ro & Phương án Dự phòng

| Rủi ro | Xác suất | Phương án |
|---|---|---|
| FGW chậm hơn Sinkhorn | Trung bình | Giảm fgw_iters, dùng approximation |
| Hilbert curve implementation phức tạp | Thấp | Fallback sang Z-order (Morton code) |
| UFGW gradient không ổn | Trung bình | Giảm KL penalty, warm-up từ balanced → unbalanced |
| Slot Attention không converge | Trung bình | Guided Slot (CAM prior) |
| VRAM không đủ | Thấp | Gradient checkpointing, giảm resolution |

---

## 7. Timeline Dự kiến

| Tuần | Công việc | Output |
|---|---|---|
| 1 | Setup + DataLoader + Mamba backbone | Baseline R@1 |
| 2 | Adaptive Mask + Slot Attention + Register | Module B+C |
| 3 | Hilbert Graph Mamba + FGW OT | Full pipeline |
| 4-5 | Train University-1652 & VIGOR | Main results |
| 6 | Ablation studies + CV-Cities + CVUSA | Tables |
| 7 | Viết paper ACM MM + hình kiến trúc | Draft |
