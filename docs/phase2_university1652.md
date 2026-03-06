# Phase 2: University-1652 — Main Benchmark #1

## Mục đích
**Main contribution benchmark.** Drone↔Satellite matching, multi-altitude, scale variation.

## Target vs SOTA

> **⚠ LƯU Ý:** Uni-1652 báo cáo cả **R@1** (Recall@1) và **AP** (Average Precision). Hai metric khác nhau!

### Drone→Satellite (Drone-view Target Localization)

| Method | R@1 | AP | Year |
|---|---|---|---|
| FSRA | 82.25 | 84.82 | 2022 |
| ATRPF | 82.50 | 84.28 | 2023 |
| Sample4Geo | 92.65 | 93.81 | 2023 |
| Cross-view Consistent Attn | 91.57 | 93.31 | 2024 |
| CV-Cities | 97.43 | 95.01 | 2024 |
| OG-Sample4Geo | 96.13 | 96.88 | 2025 |
| **GeoSlot (Ours)** | **≥97.5** | **≥97.0** | **2026** |

### Satellite→Drone (Drone Navigation)

| Method | R@1 | AP | Year |
|---|---|---|---|
| ATRPF | 90.87 | 80.25 | 2023 |
| **GeoSlot (Ours)** | **Report** | **Report** | **2026** |

> **Beat SOTA** = R@1 ≥ 97% (beat CV-Cities 97.43%) hoặc AP ≥ 97% (beat OG-S4G 96.88%)

## Config
- **Script:** `kaggle/phase2_train_university1652_kaggle.py` + `GeoSlot_model.py`
- **Image:** 384×384 (cả drone lẫn satellite — higher res)
- **Batch:** 32 | **Epochs:** 60
- **Transfer learning:** Load từ Phase 1 checkpoint (optional)
- **Eval:** Drone→Satellite R@1, R@5, R@10 + AP

## Dataset Path
```
/kaggle/input/datasets/chinguyeen/university-1652/University-1652
```

## Cách chạy
1. Upload `phase2_train_university1652_kaggle.py` + `GeoSlot_model.py`
2. (Optional) Set `RESUME_FROM = "/kaggle/working/best_model_cvusa.pth"`
3. Attach dataset University-1652, chọn H100, Run

## Output
- `best_model_uni1652.pth`
- `results_uni1652.json`

## Ablation Study
**Ablation chạy trên dataset này** (xem `docs/ablation_study.md`)
