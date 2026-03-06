# Ablation Study — Plan Chi Tiết

## Chiến thuật Tổng Quan

```
Round 1: Ablation trên University-1652 (nhanh, ~12h)
    ↓ Nếu mỗi module đều có impact rõ (+1% trở lên)
Round 2: Verify trên VIGOR (hardest, ~8h)
    ↓ Nếu pattern nhất quán
Round 3: Chạy FULL model trên tất cả datasets
    ↓
Paper table: Full results trên 4 datasets
```

---

## Round 1: Ablation trên University-1652 (Chạy đầu tiên)

**Tại sao University-1652?**
- Nhỏ gọn (~72K images), train 30 epochs = ~1.5h/config
- 9 configs × 1.5h = ~12h (1-2 Kaggle session)
- Đủ variance để thấy rõ difference giữa configs

### 9 Configs

| # | Config | Tách gì | Expected ∆R@1 |
|---|---|---|---|
| A1 | MambaVision-L + GAP → Cosine | Baseline, không có gì | ~90-92% |
| A2 | + Slot Attention | Chỉ thêm Slot | +2-3% |
| A3 | + Register + BG Mask | Thêm noise absorption | +0.5-1% |
| A4 | + Graph Mamba | Thêm relational reasoning | +1-2% |
| A5 | + Sinkhorn OT (soft) | Thêm OT matching | +0.5-1% |
| **A6 ★** | **Full + MESH** | **Proposed method** | **≥ 97%** |
| A7 | Full − Graph | Bỏ Graph → thấy Graph quan trọng? | -1-2% |
| A8 | Full − Register | Bỏ Register → background leakage? | -0.5-1% |
| A9 | Full, Cosine match | Bỏ OT → Cosine đủ chưa? | -1-3% |

### Tiêu chí đánh giá (Pass/Fail)
- ✅ **Pass:** A6 > A1 ≥ 5% AND mỗi module đều contribute ≥ 0.5%
- ❌ **Fail:** Nếu Slot / Graph / OT không có impact → cần redesign

---

## Round 2: Verify trên VIGOR (Chỉ chạy nếu Round 1 Pass)

**Tại sao VIGOR?**
- Hardest benchmark, cross-area test
- Object-centric approach (Slots) nên có lợi thế lớn nhất ở đây
- Nếu ablation pattern trên VIGOR khớp với Uni-1652 → strong evidence

### Chạy 4 configs chính (không cần full 9)
| # | Config | Mục đích |
|---|---|---|
| A1 | Baseline | Lower bound |
| A2 | + Slots only | Slot contribution trên panorama↔satellite |
| A6 | Full pipeline | Upper bound |
| A9 | Full, Cosine | OT vs Cosine trên VIGOR |

**GPU time:** 4 configs × ~2.5h = ~10h

### Tiêu chí
- ✅ **Pass:** A6 SA ≥ 95% AND cross-area pattern nhất quán
- Cross-area improvement (A6 vs A1) nên **lớn hơn** so với Uni-1652 → chứng minh Slot Attention + OT giúp generalize

---

## Round 3: Full Model trên Tất Cả Datasets (Chỉ chạy nếu Round 1+2 Pass)

| Dataset | Script | Config | Target |
|---|---|---|---|
| CVUSA | `phase1_train_cvusa_kaggle.py` | Full (A6) | R@1 ≥ 99.5% |
| University-1652 | `phase2_train_university1652_kaggle.py` | Full (A6) | R@1 ≥ 97% |
| VIGOR | `phase3_train_vigor_kaggle.py` | Full (A6) | SA ≥ 95%, CA ≥ 30% |
| CV-Cities | `phase4_train_cv_cities_kaggle.py` | Full (A6) | Establish baseline |

**GPU time:** ~28h total (4-5 Kaggle sessions)

---

## Paper Table Format

### Table 1: Comparison with SOTA (Round 3 results)
```
Dataset         | Method          | R@1    | R@5    | AP
University-1652 | Sample4Geo      | 96.88% | ...    | ...
                | GeoSlot   | 97.x%  | ...    | ...
VIGOR (SA)      | AuxGeo          | ~95%   | ...    | ...
                | GeoSlot   | 95.x%  | ...    | ...
VIGOR (CA)      | GeoDTR+         | ~25%   | ...    | ...
                | GeoSlot   | 30.x%  | ...    | ...
CVUSA           | Sample4Geo      | 99.67% | ...    | ...
                | GeoSlot   | 99.x%  | ...    | ...
```

### Table 2: Ablation Study (Round 1 + 2 results)
```
Config | Uni-1652 R@1 | VIGOR SA HR@1 | VIGOR CA HR@1
A1     | 90.x%        | 80.x%         | 15.x%
A2     | 93.x%        | 87.x%         | 20.x%
...
A6 ★   | 97.x%        | 95.x%         | 30.x%
```

---

## Tổng GPU Budget

| Round | Sessions | Hours | Chạy khi nào |
|---|---|---|---|
| Round 1 (Ablation Uni-1652) | 2 | ~12h | **Đầu tiên** |
| Round 2 (Verify VIGOR) | 1-2 | ~10h | Nếu Round 1 pass |
| Round 3 (Full 4 datasets) | 4-5 | ~28h | Nếu Round 1+2 pass |
| **Total** | **7-9** | **~50h** | |
