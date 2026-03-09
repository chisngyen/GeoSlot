# GeoSlot — Bug Tracker & Lessons Learned

> Tài liệu ghi lại tất cả bugs đã gặp và fix trong quá trình phát triển GeoSlot.
> Mục đích: reference nhanh để tránh lặp lại lỗi tương tự.

---

## 📋 Danh sách Bugs

### 🔴 NaN Loss (Critical — Training crashes)

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| [01](01_nan_loss_backbone_float16.md) | Backbone float16 | mamba_ssm kernel unstable in fp16 | `autocast(enabled=False)` in backbone |
| [02](02_nan_loss_post_backbone_float16.md) | Post-backbone float16 | GRU, Gumbel, SSM unstable in fp16 | `autocast(enabled=False)` in encode_view |
| [03](03_nan_loss_cdist_zero_distance.md) | torch.cdist backward NaN | sqrt'(0) = inf at zero distance | Safe L2 with `clamp(min=1e-6).sqrt()` |
| [04](04_nan_loss_dwbl_exp_overflow.md) | DWBL exp overflow | `exp(large/small_temp)` overflow | `.clamp(max=20)` before exp |
| [05](05_nan_loss_sinkhorn_float16.md) | Sinkhorn float16 | Iterative algo loses precision in fp16 | Force `.float()` inputs |
| [06](06_nan_loss_gradscaler_init.md) | GradScaler init too high | Default 65536 overflows gradients | `init_scale=2048` |
| [07](07_nan_loss_missing_guard.md) | No NaN guard | NaN propagates to weights permanently | Check `isnan/isinf` before backward |

### 🟡 Dimension Mismatch (Runtime crash)

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| [08](08_dim_mismatch_spatial_encoder.md) | SlotSpatialEncoder MLP dim | `enc_dim*4+1` vs `enc_dim*2+1` | Match MLP input to actual encoding dim |
| [09](09_dim_mismatch_attn_maps.md) | attn_maps K mismatch | 16 (all) vs 12 (object only) | Slice `attn_maps[:, :K, :]` |

### 🟠 Loss Collapse (Training stalls)

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| [10](10_loss_collapse_stage2.md) | Stage 2 loss stuck | CSM circular grad + weights too high | Detach T, ramp, reduce λ |

### 🔵 Environment / Compatibility

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| [11](11_mambavision_linspace_meta.md) | linspace meta tensor | MambaVision + accelerator config | Monkey-patch + `low_cpu_mem_usage=False` |
| [12](12_transformers_version.md) | transformers version | API change in ≥4.45 breaks MambaVision | Pin `transformers==4.44.2` |

---

## ⚡ Quick Reference — Golden Rules

### AMP / Mixed Precision
1. **mamba_ssm kernel** → always float32 (`autocast(enabled=False)`)
2. **GRU, LSTM, RNN** → float32 (iterative hidden state update)
3. **Gumbel softmax** → float32 (`log(-log(uniform))` with fp16 noise = NaN)
4. **Iterative algorithms** (Sinkhorn, power iter) → float32 inputs
5. **GradScaler** → `init_scale=2048` for large models

### Numerical Stability
6. **`torch.cdist(p=2.0)`** → NEVER use for training. Use safe L2: `(diff²).sum().clamp(1e-6).sqrt()`
7. **`torch.exp(x)`** → always `clamp(max=20)` first
8. **`torch.log(x)`** → always `clamp(min=1e-8)` first
9. **Division** → always `+ 1e-8` denominator

### Multi-Stage Training
10. **Cross-module signals** (transport plan, attn maps) → `.detach()` in auxiliary losses
11. **New loss terms** → linear warm-up ramp, never full weight instantly
12. **Auxiliary λ** → ≤ 0.3 × primary loss weight
13. **Temperature** → start ≥ 0.5, never below 0.1

### Training Loop
14. **NaN guard** → mandatory `isnan(loss)` check before `backward()`
15. **Gradient clipping** → `clip_grad_norm_(model.parameters(), 1.0)`

### Dependencies
16. **`transformers`** → pin version when using `trust_remote_code=True`
17. **mamba_ssm** → build from source, match CUDA version

---

## 🔍 Debugging Checklist (khi gặp NaN)

```
□ Check 1: Loss component nào NaN? (InfoNCE? DWBL? CSM? Dice?)
□ Check 2: Forward output có NaN không? (embedding, slots, transport_plan)
□ Check 3: Model weights có NaN không? (any(isnan(p)) for p in parameters)
□ Check 4: GradScaler scale đang ở mức nào? (scaler.get_scale())
□ Check 5: Có module nào chạy float16 mà không nên không?
□ Check 6: Có torch.cdist, torch.exp, torch.log không clamp không?
□ Check 7: Iterative algo (Sinkhorn) có force float32 inputs không?
```

---

*Last updated: 2026-03-07*
*Project: GeoSlot — Cross-View Geo-Localization with Slot Attention*
