# 🏗️ FINAL PLAN: Cross-View Geo-Localization với Vision Mamba + Slot Attention + Graph Mamba

> **Tên dự án (Working Title):** *GeoSlot: Object-Centric State-Space Matching for Cross-View Geo-Localization*
> **Ngày chốt:** 2026-03-06 · **Phần cứng:** Kaggle H100 · **Backbone:** [hustvl/Vim](https://github.com/hustvl/Vim)

---

## 1. Tổng quan Ý tưởng (Final Idea)

Xây dựng một hệ thống định vị địa lý chéo góc nhìn (CVGL) **End-to-End** bằng cách kết hợp:

| Thành phần | Vai trò | Novelty |
|---|---|---|
| **Vision Mamba (SS2D)** | Backbone trích xuất đặc trưng $\mathcal{O}(N)$ | Thay thế ViT $\mathcal{O}(N^2)$, tiết kiệm bộ nhớ cho ảnh high-res |
| **Adaptive Slot Attention + Register Slots** | Phân tách cảnh thành các đối tượng tĩnh (Object-Centric) | Chưa ai dùng Slot Attention trong CVGL |
| **Graph Mamba** | Lập luận quan hệ không gian giữa các Slots | Relational reasoning tuyến tính thay vì GNN/Graph Transformer |
| **Sinkhorn Optimal Transport (AGOT + MESH)** | So khớp bipartite giữa 2 tập Slots chéo góc nhìn | Hard matching 1-1 thay vì Cosine Similarity |
| **Multi-Layer Joint Loss (4 lớp)** | Hệ thống Loss đa tầng chuyên biệt | Kết hợp CVD + DWBL + Temporal Contrastive + Dice |

**Điểm cốt lõi:** Ảnh không còn là một cục pixel → được tách thành *{Tòa nhà A, Ngã tư B, Bãi cỏ C}* → liên kết quan hệ "A cách B 10m hướng Bắc" → so khớp tối ưu với ảnh vệ tinh qua Optimal Transport.

---

## 2. Kiến trúc Tổng thể (Architecture)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGES                                 │
│   Query (Drone/Street)              Reference (Satellite)           │
└──────────┬──────────────────────────────────┬───────────────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────┐         ┌──────────────────────┐
│  Vision Mamba (SS2D) │         │  Vision Mamba (SS2D) │  ← Shared Weights
│  + Channel Group     │         │  + Channel Group     │
│    Pooling (CGP)     │         │    Pooling (CGP)     │
└──────────┬───────────┘         └──────────┬───────────┘
           │ Dense Feature Map               │ Dense Feature Map
           ▼                                  ▼
┌──────────────────────┐         ┌──────────────────────┐
│ Background Suppress  │         │ Background Suppress  │  ← Auxiliary Loss
│ (Transient Mask)     │         │ (Transient Mask)     │
└──────────┬───────────┘         └──────────┬───────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────┐         ┌──────────────────────┐
│ Adaptive Slot Attn   │         │ Adaptive Slot Attn   │
│ + Register Slots     │         │ + Register Slots     │
│ (AdaSlot, K tự động) │         │ (AdaSlot, K tự động) │
└──────────┬───────────┘         └──────────┬───────────┘
           │ K object slots                   │ M object slots
           ▼                                  ▼
┌──────────────────────┐         ┌──────────────────────┐
│    Graph Mamba        │         │    Graph Mamba        │
│ (Relational Reason.) │         │ (Relational Reason.) │
└──────────┬───────────┘         └──────────┬───────────┘
           │ Graph-enhanced slots             │ Graph-enhanced slots
           └──────────┬───────────────────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │  Sinkhorn OT (AGOT) │
           │  + MESH (Hard Match)│
           │  → Similarity Score │
           └──────────────────────┘
```

### 2.1. Chi tiết từng Module

#### Module A: Vision Mamba Backbone (SS2D)
- **Source:** `hustvl/Vim` (Official, ImageNet pre-trained)
- **Kỹ thuật:** Multi-directional scan (4 hướng: ↑↓←→) → global receptive field
- **Head:** Channel Group Pooling (CGP) thay thế GAP+FC (giảm 1.5% params, giữ local detail)
- **🔧 Có thể thay đổi:** Thử VMamba-S vs VMamba-B (scale khác nhau) tùy theo VRAM budget

#### Module B: Background Suppression Mask
- **Vị trí:** Giữa Mamba output và Slot Attention input
- **Cơ chế:** Lightweight attention map (1 conv layer) học cách mask out transient objects
- **Loss phụ:** Auxiliary Contrastive Loss so sánh mask giữa 2 views
- **🔧 Có thể thay đổi:** Thử thêm Temporal Contrastive Loss (CA-SA) nếu dataset có video sequence

#### Module C: Adaptive Slot Attention + Register Slots
- **AdaSlot:** Số lượng K slots tự điều chỉnh dựa trên scene complexity
- **Register Slots:** Các slots "rác" chuyên hấp thụ nhiễu → loại bỏ trước matching
- **Iterative Routing:** 3-5 vòng lặp GRU + softmax cạnh tranh
- **🔧 Có thể thay đổi:**
  - Số Register Slots: bắt đầu 2-4, tăng nếu observe background leakage
  - Số vòng lặp SA: 3 (nhẹ, nhanh) vs 5 (chính xác hơn nhưng chậm)
  - Thử Dual-Mask (FDSA-Net style) nếu tách nền chưa đủ sạch

#### Module D: Graph Mamba (Relational Reasoning)
- **Xây đồ thị:** Từ K slots → Graph $\mathcal{G}=(V,E)$ với edge weights = semantic similarity + spatial proximity
- **Message passing:** Graph-guided Bidirectional Scan (GBS) thay vì GCN/GAT
- **🔧 Có thể thay đổi:**
  - Dùng kNN graph (k=5) vs Fully connected graph → kNN nhẹ hơn, đủ cho urban scenes
  - Thử thêm positional encoding cho nodes (vị trí tương đối của slots trong feature map)

#### Module E: Sinkhorn OT + MESH (Matching)
- **Ma trận chi phí:** $C_{ij}$ = khoảng cách L2 giữa slot $i$ (query) và slot $j$ (reference)
- **Sinkhorn iterations:** 10-20 vòng → doubly stochastic matrix
- **MESH:** Minimize entropy để ép soft → hard assignment (tie-breaking)
- **🔧 Có thể thay đổi:**
  - Entropy coefficient $\epsilon$: nhỏ = cứng hơn (chính xác nhưng khó hội tụ), lớn = mềm hơn
  - Thử Earth Mover's Distance (EMD) thay thế nếu Sinkhorn không ổn định

---

## 3. Hệ thống Loss Functions (Multi-Layer Joint Loss)

### Tầng 1: Filtering & Slot Quality
| Loss | Mục đích | Trọng số khởi đầu |
|---|---|---|
| **Contrastive Slot Matching** | Tối đa MI giữa object slots và ảnh gốc | $\lambda_1 = 1.0$ |
| **Sinkhorn MESH** | Ép hard assignment 1-1 giữa slots | $\lambda_2 = 0.5$ |
| **Dice Loss** | Giải quyết scale imbalance (slot mask vs image) | $\lambda_3 = 0.3$ |

### Tầng 2: Embedding & Metric Learning
| Loss | Mục đích | Trọng số khởi đầu |
|---|---|---|
| **DWBL** (Dynamic Weighted Batch-tuple) | Hard negative mining tự động trong batch | $\lambda_4 = 1.0$ |
| **Symmetric InfoNCE** | Căn chỉnh không gian nhúng 2 branches | $\lambda_5 = 1.0$ |
| **CVD Loss** (Content-Viewpoint Disentangle) | Tách "nội dung" khỏi "góc nhìn" | $\lambda_6 = 0.5$ |

### Tổng Loss:
$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{CSM} + \lambda_2 \mathcal{L}_{MESH} + \lambda_3 \mathcal{L}_{Dice} + \lambda_4 \mathcal{L}_{DWBL} + \lambda_5 \mathcal{L}_{InfoNCE} + \lambda_6 \mathcal{L}_{CVD}$$

> **🔧 Có thể thay đổi:** Các $\lambda$ nên được điều chỉnh qua ablation study. Bắt đầu với giá trị trên, sau đó grid search trên validation set.
> 
> **⚠️ Lưu ý thực tế:** Nên bật Loss từng tầng (stage-wise training), không bật tất cả cùng lúc từ epoch 1:
> - **Epoch 1-30:** Chỉ bật InfoNCE + DWBL (học embedding cơ bản trước)
> - **Epoch 30-60:** Thêm Contrastive Slot Matching + Dice (refinement Slot quality)
> - **Epoch 60+:** Thêm MESH + CVD (fine-tuning hard matching + disentanglement)

---

## 4. Chiến thuật Dataset: Train / Test / Benchmark

### 4.1. Phân vai Dataset

```
┌─────────────────────────────────────────────────────────────────┐
│                    VAI TRÒ CỦA TỪNG DATASET                    │
├─────────────────┬───────────────────────────────────────────────┤
│ University-1652 │ 🎯 MAIN BENCHMARK #1                        │
│                 │ • Train + Test trực tiếp                     │
│                 │ • Tác vụ: Drone→Sat, Sat→Drone               │
│                 │ • Metric: Recall@1, Recall@5, AP             │
│                 │ • Thử thách: Multi-altitude, scale variation │
├─────────────────┼───────────────────────────────────────────────┤
│ VIGOR           │ 🎯 MAIN BENCHMARK #2                        │
│                 │ • Train + Test (Same-Area & Cross-Area)      │
│                 │ • Tác vụ: Panorama→Sat                       │
│                 │ • Metric: Hit Rate, Recall@1                 │
│                 │ • Thử thách: Decentrality, many-to-many      │
├─────────────────┼───────────────────────────────────────────────┤
│ CVUSA           │ 📊 SHOWCASE / SATURATION                    │
│                 │ • Train + Test                               │
│                 │ • Chỉ dùng để chứng minh method ≥ 99%       │
│                 │ • Không phải main contribution               │
│                 │ • SOTA hiện tại ~99.67%, ta cần ≥ đó        │
├─────────────────┼───────────────────────────────────────────────┤
│ CV-Cities       │ 🌍 GENERALIZATION TEST                      │
│                 │ • Train trên subset → Test cross-city        │
│                 │ • Chứng minh khả năng tổng quát hóa toàn cầu│
│                 │    (châu Á, châu Âu, châu Mỹ...)            │
│                 │ • Metric: Recall@1 cross-city                │
└─────────────────┴───────────────────────────────────────────────┘
```

### 4.2. Chiến thuật Training

#### Giai đoạn 1: Pre-training Backbone (Warm-up)
- **Dữ liệu:** University-1652 (nhỏ gọn, 3 view types, dễ converge)
- **Mục tiêu:** Cho Vision Mamba học cách trích xuất feature cross-view cơ bản
- **Loss:** InfoNCE + DWBL only
- **Epochs:** ~30 epochs, lr=1e-4, cosine decay
- **Batch size:** 32 (H100 đủ VRAM)

#### Giai đoạn 2: Slot Training (Module Refinement)
- **Dữ liệu:** University-1652 (tiếp tục)
- **Mục tiêu:** Train Slot Attention + Register Slots + Background Suppression
- **Loss:** Thêm Contrastive Slot Matching + Dice
- **Epochs:** +30 epochs, lr=5e-5
- **Freeze:** Backbone Mamba (chỉ train Slot head)

#### Giai đoạn 3: Graph + OT (Full Pipeline)
- **Dữ liệu:** University-1652 → chuyển sang VIGOR
- **Mục tiêu:** Train toàn bộ pipeline (Graph Mamba + Sinkhorn OT)
- **Loss:** Full Multi-Layer Joint Loss
- **Epochs:** +40 epochs, lr=2e-5
- **Unfreeze all:** End-to-end fine-tuning

#### Giai đoạn 4: Generalization (Cross-dataset)
- **Dữ liệu:** CV-Cities (16 cities)
- **Mục tiêu:** Fine-tune trên subset cities → test cross-city
- **Chiến thuật:** Leave-one-city-out hoặc train trên 12 cities, test trên 4 cities
- **Expected:** Chứng minh model generalizes tốt qua các đô thị khác nhau

#### CVUSA Run (Showcase)
- **Dữ liệu:** CVUSA
- **Mục tiêu:** Chỉ cần đạt ≥ 99% Recall@1 để report trong paper
- **Chiến thuật:** Fine-tune model từ Giai đoạn 3 trên CVUSA ~10-15 epochs

### 4.3. Testing & Evaluation Protocol

| Dataset | Split | Metrics chính | Mục tiêu SOTA |
|---|---|---|---|
| University-1652 | Drone→Sat | Recall@1, AP | > 85% (hiện tại ~78%) |
| University-1652 | Sat→Drone | Recall@1, AP | > 85% |
| VIGOR Same-Area | Pano→Sat | Hit Rate@1 | > 95% |
| VIGOR Cross-Area | Pano→Sat | Hit Rate@1 | > 25% (hiện tại ~20.72%) |
| CVUSA | Pano→Sat | Recall@1 | ≥ 99% (showcase) |
| CV-Cities | Cross-city | Recall@1 | Report top (no prior baseline) |

---

## 5. Ablation Studies (Bắt buộc cho Paper)

Các thí nghiệm cần chạy để chứng minh đóng góp từng module:

| # | Thí nghiệm | Mục đích |
|---|---|---|
| 1 | Mamba only (no Slot, no Graph) | Baseline backbone |
| 2 | Mamba + Slot Attention (no Graph, no OT) | Hiệu quả của Slot |
| 3 | Mamba + Slot + Register Slots | Hiệu quả của Background Suppression |
| 4 | Mamba + Slot + Graph Mamba (no OT) | Hiệu quả của Relational Reasoning |
| 5 | Full pipeline (with OT + MESH) | **Proposed method** |
| 6 | Full pipeline + CVD Loss | Hiệu quả của Content-Viewpoint Disentanglement |
| 7 | Thay Mamba bằng ViT-B/16 | So sánh backbone efficiency |
| 8 | Thay Graph Mamba bằng GCN | So sánh graph reasoning method |
| 9 | Thay Sinkhorn OT bằng Cosine Similarity | So sánh matching strategy |

---

## 6. Rủi ro & Phương án Dự phòng

| Rủi ro | Xác suất | Phương án |
|---|---|---|
| Slot Attention không converge trên CVGL data | Trung bình | Thử Guided Slot (dùng CAM làm prior) thay vì pure unsupervised |
| Sinkhorn không ổn định (NaN/Inf) | Trung bình | Giảm $\epsilon$, thêm gradient clipping, fallback sang EMD |
| VRAM không đủ cho full pipeline trên H100 | Thấp | Giảm image resolution (384→256), giảm num_slots, dùng gradient checkpointing |
| VIGOR Cross-Area quá khó | Cao | Chấp nhận cải thiện nhỏ (5-10%), highlight rằng đây là unsolved problem |
| Graph Mamba chưa có implementation ổn định | Trung bình | Fallback sang GAT + linear scan, hoặc tự implement dựa trên Mamba codebase |

---

## 7. Timeline Dự kiến

| Tuần | Công việc | Output |
|---|---|---|
| 1 | Setup codebase, DataLoader 4 datasets, test trên Kaggle H100 | Code chạy được |
| 2 | Implement Vision Mamba backbone + CGP head | Baseline Recall@1 |
| 3 | Implement Slot Attention + Register Slots + Background Mask | Module B+C hoạt động |
| 4 | Implement Graph Mamba + Sinkhorn OT | Full pipeline |
| 5-6 | Train Giai đoạn 1-3 trên University-1652 & VIGOR | Main results |
| 7 | Ablation studies + CV-Cities generalization test + CVUSA showcase | Ablation table |
| 8 | Viết paper / báo cáo + hình vẽ kiến trúc | Draft paper |
