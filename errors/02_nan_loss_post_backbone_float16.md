# Bug #02: NaN Loss — Post-Backbone Modules in Float16

## Mô tả
Sau khi fix backbone (Bug #01), các module post-backbone vẫn chạy float16 dưới AMP.
Nhiều module **không ổn định** ở float16:

1. **`GRUCell`** (trong SlotAttention) — iterative update tích lũy sai số float16
2. **`F.gumbel_softmax`** (trong GumbelSelector) — tính `log(-log(uniform))`, nếu uniform round về 0 ở float16 → `log(0)` = `-inf` → NaN
3. **`Mamba SSMBlock`** (trong GraphMamba) — selective scan kernel không ổn định float16

## Triệu chứng
- Fix backbone rồi nhưng **vẫn NaN loss nhiều batch** (skip 10-30% batches)
- NaN xuất hiện đặc biệt nhiều ở Stage 2+ (khi CSM/Dice loss active)
- `gumbel_softmax` NaN thường xuyên hơn khi `tau` nhỏ (late training)

## Root Cause
```python
# encode_view chạy dưới AMP → tất cả ops sau backbone = float16
def encode_view(self, x, global_step=None):
    features = self.backbone(x)  # ← đã float32 nhờ Bug#01 fix
    # NHƯNG: sa_out, graph_mamba, embed_head vẫn float16!
    sa_out = self.slot_attention(features, global_step)  # ← GRU + Gumbel = NaN
    slots = self.graph_mamba(slots, ...)                  # ← SSMBlock = NaN
```

**Chi tiết `gumbel_softmax` NaN:**
```python
# Bên trong F.gumbel_softmax:
uniform = torch.rand_like(logits)  # float16: min positive ≈ 6e-8
# Nhưng float16 rounding → nhiều giá trị = 0.0 exactly
gumbels = -torch.log(-torch.log(uniform + eps) + eps)
# log(0) = -inf → NaN
```

## Fix
```python
def encode_view(self, x, global_step=None):
    features = self.backbone(x)  # already float32 (Bug#01)

    # Force float32 for ALL post-backbone ops
    with torch.cuda.amp.autocast(enabled=False):
        features = features.float()  # ensure float32
        sa_out = self.slot_attention(features, global_step)
        slots = sa_out["object_slots"]
        keep_mask = sa_out["keep_decision"]
        
        slots = self.graph_mamba(slots, keep_mask, ...)
        
        weights = keep_mask / (keep_mask.sum(dim=-1, keepdim=True) + 1e-8)
        global_slot = (slots * weights.unsqueeze(-1)).sum(dim=1)
        embedding = F.normalize(self.embed_head(global_slot), dim=-1)
    
    return {...}
```

**Key points:**
1. `autocast(enabled=False)` bao trọn **toàn bộ** post-backbone code
2. `features.float()` — force float32 ngay đầu block
3. Kết hợp với Bug#01: backbone cũng có `autocast(enabled=False)` riêng

## Files affected
- `kaggle/phase1_train_cvusa_kaggle.py` — `GeoSlot.encode_view()`
- `kaggle/phase2_train_university1652_kaggle.py` — `GeoSlot.encode_view()`
- `kaggle/phase3_train_vigor_kaggle.py` — `GeoSlot.encode_view()`
- `kaggle/phase4_train_cv_cities_kaggle.py` — `GeoSlot.encode_view()`
- `kaggle/ablation_university1652_kaggle.py` — `GeoSlot.encode_view()`
- `kaggle/geoslot_model.py` — `GeoSlot.encode_view()`
- `src/models/geoslot.py` — `GeoSlot.encode_view()`

## Nguyên tắc tránh lỗi
> **Khi dùng AMP, tất cả module có iterative computation (GRU, RNN, LSTM), 
> stochastic sampling (gumbel, reparameterization), hoặc SSM kernels →
> PHẢI wrap trong `autocast(enabled=False)` và force `.float()`.**
>
> Rule of thumb: Nếu module có vòng lặp internal hoặc random sampling → float32.
