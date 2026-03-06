# Phase 3: VIGOR — Main Benchmark #2 (Hardest)

## Mục đích
**Hardest benchmark.** Panorama→Satellite, many-to-many matching, decentrality problem. Cross-area evaluation là thách thức lớn nhất.

## Metrics Giải Thích

> **⚠ VIGOR dùng 2 metrics khác nhau:**
> - **R@1** (Recall@1): Top-1 retrieved image đúng class
> - **Hit Rate** (HR): Top-1 satellite image chứa vị trí GPS của query

Hai metrics này KHÁC NHAU. Hit Rate thường cao hơn R@1 vì 1 location có thể nằm trong nhiều satellite tiles.

## Target vs SOTA

### Same-Area

| Method | R@1 | Hit@1 | Year |
|---|---|---|---|
| TransGeo | 61.48 | 73.09 | 2022 |
| FRGeo | 71.26 | 82.41 | 2023 |
| Sample4Geo | 77.86 | 89.82 | 2023 |
| Semantic Ambiguity | 80.10 | 91.50 | 2024 |
| AuxGeo (BIM) | 78.72 | 94.25 | 2024 |
| AuxGeo (BIM+PCM) | 80.34 | 93.78 | 2024 |
| **GeoSlot (Ours)** | **≥82** | **≥94** | **2026** |

### Cross-Area

| Method | R@1 | Hit@1 | Year |
|---|---|---|---|
| TransGeo | ~22 | ~32 | 2022 |
| Sample4Geo | ~40 | ~60 | 2023 |
| GeoDTR+ | ~54 | ~72 | 2024 |
| **GeoSlot (Ours)** | **Report** | **Report** | **2026** |

> **Strong result** = Same-area R@1 ≥ 82% (beat AuxGeo) AND/OR Hit Rate ≥ 94%.
> Object-centric approach (Slot Attention) nên có lợi thế ở cross-area vì focus objects thay vì texture.

## Config
- **Script:** `kaggle/phase3_train_vigor_kaggle.py` + `GeoSlot_model.py`
- **Train cities:** Chicago, NewYork, SanFrancisco
- **Test same-area:** All 4 cities
- **Test cross-area:** Seattle (unseen)
- **Image:** Satellite 224×224, Panorama 512×128
- **Batch:** 32 | **Epochs:** 80
- **LR:** Backbone 5e-6, Head 5e-5

## Dataset Paths
```
/kaggle/input/datasets/chinguyeen/vigor-chicago
/kaggle/input/datasets/chinguyeen/vigor-newyork
/kaggle/input/datasets/chinguyeen/vigor-sanfrancisco
/kaggle/input/datasets/chinguyeen/vigor-seattle
```

## Output
- `best_model_vigor.pth`
- `results_vigor.json` (same-area + cross-area)
