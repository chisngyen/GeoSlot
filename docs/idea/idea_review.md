# 🔬 Đánh Giá 4 Đề Xuất Nghiên Cứu vs. GeoSlot Hiện Tại

## Tổng Quan Nhanh

| # | Đề xuất | Trùng GeoSlot? | Khả thi? | Đánh giá tổng |
|---|---------|---------------|----------|---------------|
| 1 | **GW-SSM** | ⚠️ **Trùng ~70%** | ⭐⭐⭐⭐ | Không nên làm riêng |
| 2 | **Causal-MoE** | ✅ Không trùng | ⭐⭐⭐ | Idea hay nhưng rủi ro cao |
| 3 | **EB-INF** | ✅ Không trùng | ⭐⭐ | Quá phức tạp, khó publish |
| 4 | **RDG** | ✅ Không trùng | ⭐⭐⭐ | Đẹp về lý thuyết, khó thực thi |

---

## Đề Xuất 1: GW-SSM — ⚠️ TRÙNG NẶNG VỚI GEOSLOT

> [!CAUTION]
> **Idea này trùng khoảng 70% với GeoSlot 2.0 mà bạn đang làm.** Nếu submit riêng sẽ bị reviewer cho là "incremental" so với chính paper của bạn.

### Phân tích chi tiết sự trùng lặp

| Thành phần GW-SSM | GeoSlot 2.0 hiện tại | Trùng? |
|---|---|---|
| Vision Mamba backbone, O(N) | `MambaVision-L` backbone ([vim_backbone.py](file:///d:/Cross-View-Matching/src/models/vim_backbone.py)) | ✅ **Trùng** |
| Multi-directional selective scan | Bidirectional SSM trong [graph_mamba.py](file:///d:/Cross-View-Matching/src/models/graph_mamba.py) | ✅ **Trùng** |
| Intra-domain metric matrix $C^g, C^s$ | Structure cost `Sq`, `Sr` trong [fgw_ot.py](file:///d:/Cross-View-Matching/src/models/fgw_ot.py) — tính $\|c^q_i - c^q_k\|_2$ | ✅ **Trùng** |
| GW-OT matching | FGW OT = $(1-\lambda) \cdot \text{Wasserstein} + \lambda \cdot \text{Gromov-Wasserstein}$ | ✅ **Trùng** (FGW bao hàm GW) |
| Entropic regularization + Sinkhorn | Unbalanced Sinkhorn trong `_unbalanced_sinkhorn()` | ✅ **Trùng** |
| InfoNCE + $\lambda \cdot GW$ loss | `joint_loss.py` kết hợp InfoNCE + transport cost | ✅ **Trùng** |
| Dense feature matching (pixel-level) | Slot-level matching (object-centric) | ❌ **Khác** — GeoSlot dùng slots |

### Kết luận Đề Xuất 1
GW-SSM về bản chất là **GeoSlot nhưng bỏ Slot Attention**, thay bằng dense patch-level GW matching. GeoSlot 2.0 đã implement Fused Gromov-Wasserstein (bao gồm GW component) + Mamba backbone + structure cost matrices.

**Điểm duy nhất mới:** matching ở mức dense patch thay vì slot → nhưng đây là bước lùi, vì slot-level matching mới là contribution chính của GeoSlot.

> **Khuyến nghị:** ❌ **Không nên làm paper riêng.** Nếu muốn, có thể biến thành 1 ablation variant trong paper GeoSlot (so sánh dense GW vs slot FGW).

---

## Đề Xuất 2: Causal-MoE — 🟡 IDEA HAY, RỦI RO CAO

> [!IMPORTANT]
> Idea **hoàn toàn mới** so với GeoSlot. Không trùng. Nhưng yêu cầu dữ liệu đặc biệt và kỹ thuật huấn luyện phức tạp.

### Điểm mạnh
- **Giải quyết đúng pain point lớn nhất:** Spurious correlations (mô hình học texture thay vì geometry) — đây là vấn đề thực tế khiến cross-area performance sụp giảm
- **Framework toán học đẹp:** Backdoor Adjustment + SCM có nền tảng lý thuyết vững (Judea Pearl)
- **MoE sparse routing** rất phù hợp xu hướng 2025-2026 (MoE đang hot sau Mixtral, DeepSeek)
- **Kết hợp được với GeoSlot:** Có thể dùng GeoSlot backbone + thêm Causal Router layer

### Rủi ro thực tế

| Rủi ro | Mức độ | Chi tiết |
|--------|--------|----------|
| Dữ liệu multi-temporal | 🔴 Cao | Cần ảnh cùng tọa độ, khác thời tiết/mùa. CVUSA/VIGOR không có → phải tự thu thập hoặc augmentation |
| Adversarial training ổn định | 🔴 Cao | GRL + orthogonal loss rất dễ mode collapse |
| Khó đánh giá "nhân quả" | 🟡 Trung bình | Làm sao chứng minh mô hình thực sự học geometry chứ không phải texture? Cần robustness test đặc biệt |
| Tổng thời gian implement | 🔴 Cao | ~3-4 tháng full-time |

### Kết luận Đề Xuất 2
**Paper potential: CVPR/ICCV tier** nếu có kết quả tốt. Cross-view + causal inference là hướng chưa ai làm thực sự tốt. Tuy nhiên, execution risk rất cao.

> **Khuyến nghị:** 🟡 **Có thể làm paper thứ 2 sau GeoSlot**, nhưng cần:
> 1. Giản lược bớt — bỏ MoE, chỉ focus Causal Disentangler + Interventional Training
> 2. Dùng data augmentation thay multi-temporal data
> 3. Kết hợp trên backbone GeoSlot để leverage existing code

---

## Đề Xuất 3: EB-INF — 🔴 QUÁ PHỨC TẠP, KHÓ PUBLISH

> [!WARNING]
> Idea rất đẹp về mặt lý thuyết nhưng **gần như không khả thi** trong timeline 1 paper.

### Vấn đề chính

1. **Inference time không thực tế:**
   - Langevin MCMC cần 50-100 gradient steps **mỗi query** tại runtime
   - Cross-view retrieval trên database 100K images → **bất khả thi** cho real-time
   - Paper sẽ bị reviewer reject vì "not practical"

2. **Training cực kỳ khó:**
   - NeRF feature volume từ ảnh vệ tinh **đơn lẻ** → không đủ multi-view constraint
   - EBM training (Contrastive Divergence) nổi tiếng bất ổn
   - Kết hợp NeRF + EBM = 2 hệ thống bất ổn x nhau

3. **Không có precedent thành công:**
   - Chưa có paper nào combine NeRF + EBM cho retrieval thành công
   - Reviewer sẽ hỏi: "So với NeRF-based localization hiện tại, cái này hơn ở đâu?"

4. **So sánh yếu:**
   - Paper phải beat cả NeRF-based methods (CamNet, OrienterNet) VÀ retrieval methods (Sample4Geo)
   - Rất khó beat cả 2 do trade-off inherent

### Kết luận Đề Xuất 3
> **Khuyến nghị:** ❌ **Không nên làm.** Tỉ lệ risk/reward quá thấp. Cần team 3-4 người, 6+ tháng, và chưa chắc work.

---

## Đề Xuất 4: RDG — 🟡 ĐẸP LÝ THUYẾT, KHÓ THỰC THI

### Điểm mạnh
- **Sub-meter accuracy** nếu hoạt động → đột phá thực sự cho autonomous driving
- **Explainability** qua graph structure → rất hấp dẫn cho reviewer
- **Riemannian + GNN** là hot topic (NeurIPS 2024 có nhiều paper)

### Rủi ro thực tế

| Rủi ro | Mức độ | Chi tiết |
|--------|--------|----------|
| Cascading failure từ semantic parsing | 🔴 **Rất cao** | Toàn bộ pipeline phụ thuộc vào chất lượng segmentation ban đầu |
| Riemannian SDE implementation | 🔴 **Rất cao** | Cần thư viện chuyên biệt (geomstats, geoopt), backward pass trên manifold rất khó debug |
| OSM data alignment | 🟡 Trung bình | Cần liên kết pixel-level với OSM graph → labeling effort lớn |
| Baseline comparison | 🟡 Trung bình | Genre khác hoàn toàn với retrieval → khó so sánh công bằng |

### Kết luận Đề Xuất 4
> **Khuyến nghị:** 🟡 **Idea tốt cho workshop paper hoặc long-term research**, không phù hợp cho 1 paper nhanh. Nếu giản lược (bỏ Riemannian, chỉ dùng Euclidean diffusion trên graph), có thể khả thi hơn.

---

## 🎯 Khuyến Nghị Tổng Hợp

### Ưu tiên 1: Hoàn thành GeoSlot 2.0 (đang làm)
- Bạn đã có codebase hoàn chỉnh với tất cả modules
- Ablation đang chạy trên CVUSA
- **Target: ACM MM 2025 hoặc NeurIPS 2025**
- GeoSlot 2.0 đã bao hàm ~70% nội dung của GW-SSM rồi

### Ưu tiên 2: Causal-MoE (nếu muốn paper thứ 2)
- **Đề xuất giản lược:**
  1. Dùng GeoSlot backbone (reuse code)
  2. Thêm Causal Disentangler vào trước Slot Attention
  3. Bỏ MoE → dùng single network + interventional training
  4. Augmentation-based environment dictionary (không cần multi-temporal data)
  5. Test trên VIGOR Cross-Area (benchmark phù hợp nhất cho robustness)
- **Timeline:** 2-3 tháng sau khi xong GeoSlot
- **Target: CVPR 2026 hoặc ECCV 2026**

### Không khuyến nghị: EB-INF và RDG
- EB-INF: quá phức tạp, inference chậm, training bất ổn
- RDG: đẹp lý thuyết nhưng cần team lớn và thời gian dài

### Tóm tắt quyết định

```
GW-SSM    → ❌ Trùng GeoSlot, bỏ
Causal-MoE → ✅ Paper thứ 2 (giản lược)
EB-INF    → ❌ Bỏ (risk quá cao)
RDG       → ⏸️ Để dành cho future work / workshop
```
