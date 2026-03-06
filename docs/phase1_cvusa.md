# Phase 1: CVUSA — Showcase Benchmark

## Mục đích
Chứng minh method đạt chuẩn trên benchmark gần bão hòa. Đây **không phải main contribution** — chỉ "check the box".

## Target vs SOTA

| Method | R@1 | R@5 | R@10 | R@1% | Year |
|---|---|---|---|---|---|
| GeoDTR | 95.43 | 98.86 | 99.34 | 99.86 | 2023 |
| SAIG-D | 96.34 | 99.10 | 99.50 | 99.86 | 2023 |
| Sample4Geo | 98.68 | 99.68 | 99.78 | 99.87 | 2023 |
| **GeoSlot (Ours)** | **≥98.5** | **–** | **–** | **–** | **2026** |

> **Note:** CVUSA gần bão hòa ở R@1% (~99.9%), nhưng R@1 thực tế vẫn trong khoảng 97-98%. Target ≥ 98.5% R@1 là hợp lý.

## Config
- **Script:** `kaggle/phase1_train_cvusa_kaggle.py` (self-contained)
- **Backbone:** MambaVision-L (pretrained ImageNet-1K, `nvidia/MambaVision-L-1K`)
- **Image:** Satellite 224×224, Panorama 512×128
- **Batch:** 32 | **Epochs:** 50
- **LR:** Backbone 1e-5, Head 1e-4
- **Freeze backbone:** 5 epochs đầu
- **Stage-wise loss:** Stage 1 (InfoNCE+DWBL) → Stage 2 @15 (+CSM+Dice) → Stage 3 @30

## Dataset Path (Kaggle)
```
/kaggle/input/datasets/chinguyeen/cvusa-subdataset/CVUSA
```

## Cách chạy
1. Upload `phase1_train_cvusa_kaggle.py` lên Kaggle notebook
2. Attach dataset CVUSA
3. Chọn GPU: H100
4. Run notebook

## Output
- `best_model_cvusa.pth` — best checkpoint (theo R@1)
- `results_cvusa.json` — training log + metrics
