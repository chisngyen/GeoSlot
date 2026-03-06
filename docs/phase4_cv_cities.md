# Phase 4: CV-Cities — Generalization Test

## Mục đích
Chứng minh model **generalizes globally** qua các thành phố chưa từng thấy (diverse continents). Đây là **contribution phụ** — establish first baseline.

## Target
| Metric | Target |
|---|---|
| Cross-city R@1 | Report (no prior baseline) |
| Cross-city R@5 | Report |
| Cross-city R@10 | Report |

> Không có SOTA → ta establish baseline đầu tiên cho dataset này.

## Config
- **Script:** `kaggle/phase4_train_cv_cities_kaggle.py` + `GeoSlot_model.py`
- **Train:** 12 cities (barcelona, buenosaires, lisbon, london, melbourne, mexicocity, moscow, newyork, sanfrancisco, santiago, saopaulo, toronto)
- **Test:** 4 unseen cities (berlin, osaka, capetown, tokyo) — diverse continents
- **Image:** Satellite 224×224, Panorama 512×128
- **Batch:** 32 | **Epochs:** 40

## Dataset Path
```
/kaggle/input/datasets/chisboiz/cv-cities
```

## Output
- `best_model_cv_cities.pth`
- `results_cv_cities.json` (per-city + average R@K)
