# Bug #03: NaN Loss — torch.cdist Backward NaN at Zero Distance

## Mô tả
`torch.cdist(X, Y, p=2.0)` tính Euclidean distance. Backward pass tính gradient
`d/dx sqrt(x)` = `1 / (2 * sqrt(x))`. Khi distance = 0 (hai vector giống nhau)
→ `1 / (2 * sqrt(0))` = `1 / 0` = **inf** → NaN gradient.

Đây là **known PyTorch issue** — không phải bug của code mà là numerical instability
của `cdist` khi hai input vectors trùng nhau.

## Triệu chứng
- NaN loss xuất hiện **đặc biệt nhiều ở early training** (epoch 1-5)
- Lý do: early training, slot projections chưa phân biệt → nhiều cặp có distance ≈ 0
- NaN xảy ra ở backward pass (forward vẫn ra kết quả đúng = 0), nên loss value
  có thể trông bình thường nhưng gradient NaN → optimizer step NaN → loss NaN ở batch sau

## Root Cause
```python
# Trong SinkhornOT.forward():
C = torch.cdist(self.cost_proj(sq), self.cost_proj(sr), p=2.0)
# Khi cost_proj(sq)[i] ≈ cost_proj(sr)[j] → C[i,j] ≈ 0
# Backward: dC/dx = (x - y) / C[i,j] → division by zero → NaN

# Trong GraphMambaLayer._build_graph():
spatial_dist = torch.cdist(centroids, centroids, p=2.0)
# Khi centroid[i] ≈ centroid[j] (slots attend cùng vùng) → NaN
```

## Fix
Thay `torch.cdist(A, B, p=2.0)` bằng safe L2 distance:
```python
# Safe L2 distance: avoid NaN gradient from torch.cdist at zero distance
diff = A.unsqueeze(2) - B.unsqueeze(1)          # [B, K, M, D]
C = (diff * diff).sum(-1).clamp(min=1e-6).sqrt() # clamp trước sqrt → gradient safe
```

**Tại sao safe:**
- `clamp(min=1e-6)` đảm bảo argument của `sqrt()` luôn > 0
- `d/dx sqrt(x)` = `1 / (2*sqrt(x))` → với x ≥ 1e-6: gradient ≤ `1/(2*sqrt(1e-6))` = 500 — lớn nhưng finite

**So sánh:**
| Method | Forward | Backward tại d=0 |
|--------|---------|-------------------|
| `torch.cdist(p=2.0)` | 0.0 (chính xác) | **NaN/Inf** |
| Safe L2 với `clamp(1e-6)` | 0.001 (xấp xỉ) | 500 (finite) |

## Files affected
- `kaggle/phase1_train_cvusa_kaggle.py` — `SinkhornOT.forward()` + `GraphMambaLayer._build_graph()`
- `kaggle/phase2_train_university1652_kaggle.py` — `SinkhornOT.forward()`
- `kaggle/phase3_train_vigor_kaggle.py` — `SinkhornOT.forward()`
- `kaggle/phase4_train_cv_cities_kaggle.py` — `SinkhornOT.forward()`
- `kaggle/ablation_university1652_kaggle.py` — `SinkhornOT.forward()`
- `kaggle/geoslot_model.py` — `SinkhornOT.forward()` + `GraphMambaLayer._build_graph()`
- `src/models/sinkhorn_ot.py` — `SinkhornOT.forward()`
- `src/models/graph_mamba.py` — `GraphMambaLayer._build_graph()`

## Nguyên tắc tránh lỗi
> **KHÔNG BAO GIỜ dùng `torch.cdist(p=2.0)` khi cần backward pass.**
> Luôn dùng manual L2 với `clamp(min=eps)` trước `sqrt()`.
>
> Pattern an toàn:
> ```python
> diff = A.unsqueeze(2) - B.unsqueeze(1)
> dist = (diff * diff).sum(-1).clamp(min=1e-6).sqrt()
> ```
>
> Tương tự cho cosine distance: `torch.cdist(p=1.0)` cũng có thể NaN ở zero distance.
