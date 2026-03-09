# Bug #07: NaN Loss — Missing NaN Guard in Training Loop

## Mô tả
Khi NaN loss xảy ra (do bất kỳ nguyên nhân nào ở Bug #01-06),
nếu không có guard → `loss.backward()` propagate NaN gradients →
optimizer update weights với NaN → **toàn bộ model parameters = NaN** →
training crash vĩnh viễn (không recover được).

## Triệu chứng
- Một lần NaN loss → tất cả batch sau đều NaN
- Model weights chứa NaN (check `torch.isnan(p).any()` for p in model.parameters())
- Không có cách recover ngoài load checkpoint cũ

## Root Cause
```python
# Không có guard:
loss.backward()      # NaN loss → NaN gradients
optimizer.step()      # NaN gradients → NaN weights
# Từ đây, mọi forward pass đều NaN (do weights NaN)
```

## Fix
```python
# Trong training loop, SAU khi tính loss:
loss = loss_dict["total_loss"]

# --- NaN / Inf guard ---
if torch.isnan(loss) or torch.isinf(loss):
    print(f"  [WARN] NaN/inf loss at step {global_step}, skipping batch")
    optimizer.zero_grad(set_to_none=True)   # clear stale gradients
    global_step += 1
    continue    # skip backward + optimizer step

# Chỉ backward nếu loss finite:
scaler.scale(loss).backward()
```

**Key points:**
1. Check **trước** `backward()` — sau backward thì đã quá muộn
2. `optimizer.zero_grad(set_to_none=True)` — clear mọi gradient cũ
3. Skip batch hoàn toàn — không backward, không optimizer step
4. Print warning để monitor — nếu skip quá nhiều thì cần fix root cause

## Files affected
- Tất cả training scripts

## Nguyên tắc tránh lỗi
> **MỌI training loop đều PHẢI có NaN/Inf guard trước backward().**
>
> Template chuẩn:
> ```python
> loss = compute_loss(...)
> if torch.isnan(loss) or torch.isinf(loss):
>     optimizer.zero_grad(set_to_none=True)
>     continue
> scaler.scale(loss).backward()
> ```
>
> NaN guard là **phòng thủ cuối cùng** — nó không fix root cause.
> Nếu skip > 5% batches → phải tìm và fix root cause (Bug #01-06).
