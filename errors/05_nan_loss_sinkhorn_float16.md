# Bug #05: NaN Loss — SinkhornOT Float16 Precision Loss

## Mô tả
Sinkhorn-Knopp algorithm là iterative log-space normalization.
Float16 chỉ có ~3.3 decimal digits precision → sau 10-15 iterations,
accumulated error phá hỏng transport plan → NaN.

## Triệu chứng
- NaN xuất hiện ở `transport_cost` hoặc `similarity` output của SinkhornOT
- Xảy ra nhiều hơn khi tăng `num_iters` (>10)
- Transport plan `T` chứa NaN hoặc negative values

## Root Cause
```python
# Sinkhorn iterations trong log-space:
for _ in range(self.num_iters):  # 15 iterations
    log_a = -torch.logsumexp(log_K + log_b, dim=2, keepdim=True)  # float16 logsumexp unstable
    log_b = -torch.logsumexp(log_K + log_a, dim=1, keepdim=True)
# Mỗi iteration tích lũy thêm error → 15 iterations = NaN
```

## Fix
```python
def forward(self, slots_q, slots_r, mask_q=None, mask_r=None):
    # Force float32 for iterative Sinkhorn — float16 loses precision
    slots_q = slots_q.float()
    slots_r = slots_r.float()
    if mask_q is not None: mask_q = mask_q.float()
    if mask_r is not None: mask_r = mask_r.float()
    # ... rest of computation now in float32
```

## Files affected
- Tất cả files chứa class `SinkhornOT`

## Nguyên tắc tránh lỗi
> **Mọi iterative algorithm (Sinkhorn, power iteration, eigenvalue decomp)
> → PHẢI force float32 inputs.**
>
> Float16 precision (~3.3 digits) không đủ cho iterative convergence.
> Nếu algorithm có > 5 iterations → float32 bắt buộc.
