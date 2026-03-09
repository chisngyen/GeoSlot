# Bug #01: NaN Loss — Backbone Float16 Instability

## Mô tả
MambaVision-L backbone sử dụng `mamba_ssm` selective-scan CUDA kernel bên trong.
Kernel này **không ổn định khi chạy float16** dưới AMP (`torch.cuda.amp.autocast`).

Kết quả: backbone output chứa NaN/Inf → cascade NaN toàn bộ pipeline.

## Triệu chứng
- `loss = nan` ngay từ epoch đầu hoặc xuất hiện ngẫu nhiên
- NaN xuất hiện ở **tất cả** các loss component (InfoNCE, DWBL, CSM, Dice)
- Không có pattern rõ ràng — xảy ra random mỗi batch

## Root Cause
```python
# AMP autocast chuyển tất cả ops sang float16 để tăng tốc
with autocast(enabled=True):
    out = model(query, gallery)  # backbone chạy float16 → NaN
```

`mamba_ssm.Mamba` internal: selective scan kernel thực hiện nhiều phép tính sequential
(scan, multiply, accumulate) → float16 tích lũy sai số → overflow/NaN.

## Fix
```python
# Trong MambaVisionBackbone.forward():
def forward(self, x):
    # Force float32: mamba_ssm selective-scan CUDA kernel is unstable in float16
    with torch.cuda.amp.autocast(enabled=False):
        _, features = self.model(x.float())
    feat = features[-1]
    B, C, H, W = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
```

**Key points:**
1. `autocast(enabled=False)` — tắt AMP cho backbone
2. `x.float()` — đảm bảo input là float32
3. Chỉ backbone cần fix này, phần còn lại pipeline vẫn dùng AMP bình thường (xem Bug #02)

## Files affected
- `kaggle/phase1_train_cvusa_kaggle.py` — `MambaVisionBackbone.forward()`
- `kaggle/phase2_train_university1652_kaggle.py` — `MambaVisionBackbone.forward()`
- `kaggle/phase3_train_vigor_kaggle.py` — `MambaVisionBackbone.forward()`
- `kaggle/phase4_train_cv_cities_kaggle.py` — `MambaVisionBackbone.forward()`
- `kaggle/ablation_university1652_kaggle.py` — `MambaVisionBackbone.forward()`
- `kaggle/geoslot_model.py` — `MambaVisionBackbone.forward()`
- `src/models/geoslot.py` — `MambaVisionBackbone.forward()`

## Nguyên tắc tránh lỗi
> **Bất kỳ model nào dùng `mamba_ssm` (Mamba, Mamba2) → LUÔN force float32 cho forward pass.**
> Không bao giờ để AMP autocast chạy float16 qua selective-scan kernel.
