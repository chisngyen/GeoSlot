# Ablation Study — GeoSlot 2.0 (ACM MM 2025)

## Chiến thuật Tổng Quan

```
Round 1: Ablation trên University-1652 (nhanh, ~12h)
    ↓ Nếu mỗi module đều có impact rõ (+1% trở lên)
Round 2: Verify trên VIGOR (hardest, ~8h)
    ↓ Nếu pattern nhất quán
Round 3: Chạy FULL model trên tất cả datasets
    ↓
Paper tables: Full results + Ablation + Efficiency + Robustness + Interpretability
```

---

## Round 1: Component Ablation trên University-1652

| # | Config | Module | Expected ∆R@1 |
|---|---|---|---|
| A1 | MambaVision-L + GAP → Cosine | Baseline backbone | ~90-92% |
| A2 | + Slot Attention | Chỉ thêm Slot | +2-3% |
| A3 | + Register + **Adaptive BG Mask** | Noise absorption + γ tự học | +0.5-1% |
| A4 | + Graph Mamba (**Hilbert** Curve) | Relational reasoning bất biến quay | +1-2% |
| A5 | + **FGW** OT (balanced) | Graph-to-Graph matching | +1-2% |
| **A6 ★** | **Full + UFGW** | **Proposed method** | **≥ 97%** |
| A7 | Full − Graph | Bỏ Graph → thấy Graph quan trọng? | -1-2% |
| A8 | Full − Register | Bỏ Register → background leakage? | -0.5-1% |
| A9 | Full, Sinkhorn OT (thay FGW) | FGW vs Sinkhorn comparison | -1-3% |
| A10 | Full, Static Mask (α=0.7) | Adaptive vs Static Coverage | -0.5-1% |

### Tiêu chí đánh giá
- ✅ **Pass:** A6 > A1 ≥ 5% AND mỗi module đều contribute ≥ 0.5%
- ❌ **Fail:** Nếu Slot/Graph/FGW không có impact → cần redesign

---

## Round 2: Verify trên VIGOR (Chỉ chạy nếu Round 1 Pass)

| # | Config | Mục đích |
|---|---|---|
| A1 | Baseline | Lower bound |
| A2 | + Slots only | Slot contribution trên panorama↔satellite |
| A6 | Full UFGW | Upper bound |
| A9 | Full, Sinkhorn | FGW vs Sinkhorn trên VIGOR |

**Kỳ vọng:** Cross-area improvement (A6 vs A1) phải **lớn hơn** so với Uni-1652 → chứng minh Slot Attention + UFGW giúp generalize tốt hơn.

---

## Round 3: Full Model trên Tất Cả Datasets

| Dataset | Config | Target |
|---|---|---|
| CVUSA | Full (A6) | R@1 ≥ 99% |
| University-1652 | Full (A6) | R@1 ≥ 97% |
| VIGOR | Full (A6) | SA ≥ 95%, CA ≥ 30% |
| CV-Cities | Full (A6) | Establish baseline |

---

## Ablation Mới (Yêu cầu bởi bản Idea ACM MM)

### Study 1: Mamba Vision vs ViT — Efficiency Benchmark

Đo Latency + Peak Memory khi tăng resolution:

| Resolution | ViT (DINOv2-B) Mem | ViT Latency | MambaVision-L Mem | Mamba Latency |
|---|---|---|---|---|
| 224×224 | ~4GB | baseline | ~2GB | baseline |
| 384×384 | ~8GB | +2x | ~3.5GB | +1.3x |
| 512×512 | ~14GB | +4x | ~5GB | +1.6x |
| 512×2048 (panorama) | OOM ❌ | - | ~8GB ✅ | +2.5x |

**Kỳ vọng:** MambaVision O(N) → linear memory growth vs ViT O(N²) → quadratic/OOM.

### Study 2: FGW vs Sinkhorn OT

Trên cùng network, chỉ thay matching module:

| Matching | Uni-1652 R@1 | VIGOR CA HR@1 | Handles Occlusion? |
|---|---|---|---|
| Cosine Similarity | ~92% | ~18% | ❌ |
| Sinkhorn + MESH | ~96% | ~25% | ❌ (false positives) |
| Balanced FGW | ~97% | ~28% | Partial |
| **UFGW** | **~98%** | **~30%** | ✅ (KL relaxation) |

**Kỳ vọng:** UFGW > Sinkhorn +2-3.5% trên datasets có occlusion mạnh.

### Study 3: Hilbert Curve vs Raster-Scan — Rotation Robustness

Test rotation robustness (0°-360°) trên University-1652:

| Rotation | Raster-Scan R@1 | Hilbert R@1 | ΔR@1 |
|---|---|---|---|
| 0° (baseline) | ~97% | ~97% | 0% |
| 45° | ~92% | ~96% | +4% |
| 90° | ~85% | ~96% | +11% |
| 180° | ~78% | ~95% | +17% |
| Random 0-360° | ~82% | ~96% | +14% |

**Kỳ vọng:** Raster-scan suy giảm theo hàm mũ, Hilbert duy trì ổn định.

### Study 4: Interpretability Visualization

Tạo hình minh họa cho paper:
1. **Slot Heatmaps:** Mỗi slot liên kết với object (mái nhà, ngã tư, lùm cây)
2. **FGW Transport Plan:** Đường kẻ nối slots giữa drone ↔ satellite
3. **Adaptive Mask γ:** Visualization γ values cho sa mạc vs đô thị

---

## Paper Table Format

### Table 1: Comparison with SOTA
```
Dataset         | Method          | R@1    | R@5    | AP
University-1652 | Sample4Geo      | 96.88% | ...    | ...
                | GeoSlot 2.0     | 97.x%  | ...    | ...
VIGOR (SA)      | AuxGeo          | ~95%   | ...    | ...
                | GeoSlot 2.0     | 95.x%  | ...    | ...
VIGOR (CA)      | GeoDTR+         | ~25%   | ...    | ...
                | GeoSlot 2.0     | 30.x%  | ...    | ...
CVUSA           | CV-Cities       | 99.19% | ...    | ...
                | GeoSlot 2.0     | 99.x%  | ...    | ...
```

### Table 2: Ablation Study
```
Config | Component           | Uni-1652 R@1 | VIGOR CA HR@1
A1     | Baseline            | 90.x%        | 15.x%
A2     | + Slots             | 93.x%        | 20.x%
A3     | + Adaptive Mask     | 94.x%        | 21.x%
A4     | + Hilbert Graph     | 95.x%        | 24.x%
A5     | + FGW               | 96.x%        | 27.x%
A6 ★   | + UFGW (Full)       | 97.x%        | 30.x%
```

---

## Tổng GPU Budget

| Round | Sessions | Hours | Chạy khi nào |
|---|---|---|---|
| Round 1 (Ablation Uni-1652) | 2-3 | ~15h | **Đầu tiên** |
| Round 2 (Verify VIGOR) | 1-2 | ~10h | Nếu Round 1 pass |
| Round 3 (Full 4 datasets) | 4-5 | ~28h | Nếu Round 1+2 pass |
| Efficiency Study | 1 | ~2h | Cùng Round 1 |
| Robustness Study | 1 | ~3h | Cùng Round 2 |
| **Total** | **10-12** | **~58h** | |
