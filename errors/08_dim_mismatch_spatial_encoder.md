# Bug #08: Dimension Mismatch — SlotSpatialEncoder MLP Input

## Mô tả
`SlotSpatialEncoder` có 2 variants khác nhau giữa phase1 và phase2+.
MLP input dimension bị hardcode sai → `RuntimeError: mat1 and mat2 shapes cannot be multiplied`.

## Triệu chứng
- Crash ngay khi forward pass qua `GraphMambaLayer` lần đầu tiên
- Error message: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (B*K x 257 and 641 x dim)`
  hoặc tương tự dimension mismatch

## Root Cause

### Phase1 variant (4-value sinusoidal):
```python
# Tính centroids (cx, cy) + spreads (σx, σy) = 4 values
feats = torch.cat([centroids, spreads], dim=-1)  # [B, K, 4]
# Sinusoidal encode 4 values × pos_dim frequencies = 4 * pos_dim
# MLP: nn.Linear(4 * pos_dim, dim) ← ĐÚNG
```

### Phase2+ variant (2-value sinusoidal + spread):
```python
# Tính centroids (cy, cx) = 2 values
# Sinusoidal encode 2 values × enc_dim/2 sin + enc_dim/2 cos = enc_dim per value
# Total = 2 * enc_dim, concat spread = 2 * enc_dim + 1
# MLP input nên là: enc_dim * 2 + 1
```

**Bug:** Phase2/geoslot_model.py có `nn.Linear(enc_dim * 4 + 1, dim)` — sai!
Chỉ có 2 values (cy, cx), không phải 4.

## Fix
```python
# TRƯỚC (sai):
self.pos_mlp = nn.Sequential(
    nn.Linear(enc_dim * 4 + 1, dim),  # 4 * 64 + 1 = 257 → nhưng input chỉ có 129!
    nn.GELU(),
    nn.Linear(dim, dim)
)

# SAU (đúng):
self.pos_mlp = nn.Sequential(
    nn.Linear(enc_dim * 2 + 1, dim),  # 2 * 64 + 1 = 129 → match input
    nn.GELU(),
    nn.Linear(dim, dim)
)
```

## Files affected
- `kaggle/phase2_train_university1652_kaggle.py` — `SlotSpatialEncoder.__init__()`
- `kaggle/geoslot_model.py` — `SlotSpatialEncoder.__init__()`
- (Phase3/4/ablation kế thừa từ geoslot_model.py)

## Nguyên tắc tránh lỗi
> **Khi có 2 variants của cùng một module → document rõ sự khác biệt.**
>
> Luôn verify MLP input dim bằng cách trace qua:
> 1. Input tensor shape tại runtime
> 2. Mỗi transform (concat, encode, reshape) thay đổi dim thế nào
> 3. MLP first layer input dim phải match kết quả cuối
>
> Dùng assertion: `assert x.shape[-1] == self.expected_dim, f"Got {x.shape[-1]}"`.
