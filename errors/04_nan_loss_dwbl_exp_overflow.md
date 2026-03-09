# Bug #04: NaN Loss — DWBL Exponential Overflow

## Mô tả
`DWBL` (Distance Weighted Batch Loss) tính `torch.exp(neg / temperature)`.
Khi `neg` lớn và `temperature` nhỏ (0.1) → exponent quá lớn → **overflow** → Inf → NaN.

## Triệu chứng
- NaN loss xuất hiện khi similarity giữa negative pairs cao (> 2.0)
- Đặc biệt xảy ra khi embedding chưa tốt (early training) hoặc hard negatives
- Chỉ `loss_dwbl` component NaN, các loss khác bình thường

## Root Cause
```python
class DWBL(nn.Module):
    def forward(self, q, r):
        neg = sim[mask].view(B, B-1)
        # neg / 0.1 = neg * 10 → torch.exp(neg*10) dễ overflow
        wneg = (weights * torch.exp(neg / self.t)).sum(dim=-1)
        # float16 max ≈ 65504, float32 max ≈ 3.4e38
        # exp(20) ≈ 4.8e8 (safe), exp(80) ≈ 5.5e34 (float32 gần overflow)
```

## Fix
```python
# Clamp exponent trước khi exp()
wneg = (weights * torch.exp(((neg - self.m) / self.t).clamp(max=20))).sum(dim=-1)
pos_exp = torch.exp((pos / self.t).clamp(max=20))
```

**`.clamp(max=20)`**: `exp(20)` ≈ 4.85 × 10⁸ — an toàn cho cả float16 lẫn float32.

## Files affected
- Tất cả files chứa class `DWBL`

## Nguyên tắc tránh lỗi
> **Bất kỳ `torch.exp(x)` nào → LUÔN clamp x trước, thường `clamp(max=20)`.**
> Đặc biệt khi x = something / temperature và temperature < 1.0.
>
> Tương tự: `torch.log(x)` → clamp `x` với `clamp(min=1e-8)`.
