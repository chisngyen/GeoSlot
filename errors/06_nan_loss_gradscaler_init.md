# Bug #06: NaN Loss — GradScaler Init Scale Too High

## Mô tả
`torch.cuda.amp.GradScaler` mặc định `init_scale=65536`.
Scale quá cao → scaled gradients overflow float16 range → NaN.
GradScaler sẽ skip step và halve scale, nhưng mất nhiều batch training đầu.

## Triệu chứng
- NaN loss liên tục ở 5-10 batch đầu tiên
- Sau đó tự recover (GradScaler tự giảm scale)
- Training loss "nhảy" ở đầu mỗi epoch

## Root Cause
```python
scaler = GradScaler(enabled=True)
# Default: init_scale=65536, growth_factor=2.0, growth_interval=2000
# Với model lớn (backbone 200M+ params), gradients đã lớn sẵn
# Scale 65536 × large_gradient → overflow
```

## Fix
```python
scaler = GradScaler(
    enabled=AMP_ENABLED and DEVICE.type == "cuda",
    init_scale=2048,         # 32× nhỏ hơn default
    growth_interval=1000,    # Tăng scale chậm hơn
)
```

**Giải thích parameters:**
- `init_scale=2048`: Bắt đầu conservative, tránh overflow ngay batch đầu
- `growth_interval=1000`: Chỉ tăng scale mỗi 1000 steps (default=2000 cũng ok)

## Files affected
- Tất cả training scripts chứa `GradScaler`

## Nguyên tắc tránh lỗi
> **Với model lớn (>100M params) hoặc biết model hay NaN:**
> - Dùng `init_scale=2048` thay vì default 65536
> - Dùng `growth_interval=1000-2000`
>
> Monitor `scaler.get_scale()` — nếu nó liên tục giảm thì model có vấn đề khác.
