# Bug #09: Dimension Mismatch — attn_maps K vs Total Slots

## Mô tả
`AdaptiveSlotAttention` trả về `attn_maps` cho **tất cả** slots (object + register).
`GraphMambaLayer` chỉ nhận **object slots** nhưng dùng `attn_maps` trực tiếp
→ mismatch: `attn_maps` có K_total (16) hàng, nhưng `slots` chỉ có K_object (12).

## Triệu chứng
- `RuntimeError: shape mismatch` hoặc indexing error trong GraphMambaLayer
- Error: `expected attn_maps[:, :12, :] but got attn_maps[:, :16, :]`

## Root Cause
```python
# AdaptiveSlotAttention.forward():
slots, attn = self.slot_attn(features_proj, slots)  # slots = [B, 16, D], attn = [B, 16, N]
object_slots = slots[:, :self.max_slots]              # [B, 12, D]
# Nhưng return attn_maps = attn  →  [B, 16, N] (bao gồm cả register slots)

# GraphMambaLayer.forward():
B, K, D = slots.shape  # K = 12 (chỉ object)
obj_attn = attn_maps[:, :K, :]  # K = 12, nhưng attn_maps có 16 rows → WRONG indices!
```

## Fix
```python
# Trong GraphMambaLayer.forward():
def forward(self, slots, keep_mask=None, attn_maps=None, spatial_hw=None):
    B, K, D = slots.shape
    if attn_maps is not None and spatial_hw is not None:
        H, W = spatial_hw
        obj_attn = attn_maps[:, :K, :]  # ← Slice chỉ lấy K object slot rows
        pos_enc = self.spatial_encoder(obj_attn, H, W)
        slots = slots + pos_enc
```

**Hoặc** fix trong `AdaptiveSlotAttention.forward()` — chỉ trả về object attn:
```python
return {
    "attn_maps": attn[:, :self.max_slots, :],  # chỉ object, không register
    ...
}
```

Chọn approach 1 (slice trong GraphMamba) vì register attn_maps có thể hữu ích cho visualization.

## Files affected
- `kaggle/phase1_train_cvusa_kaggle.py` — `GraphMambaLayer.forward()`
- `kaggle/geoslot_model.py` — `GraphMambaLayer.forward()`
- `src/models/graph_mamba.py` — `GraphMambaLayer.forward()`
- (Phase2+/ablation cũng đã fix khi inline)

## Nguyên tắc tránh lỗi
> **Khi pipeline có split (object vs register), LUÔN track rõ mỗi tensor đang refer tới subset nào.**
>
> Naming convention giúp:
> - `all_slots` = object + register (16)
> - `object_slots` hoặc `slots` = chỉ object (12)
> - `attn_maps_all` vs `attn_maps_obj`
>
> Assertion: `assert attn_maps.shape[1] >= K, f"attn {attn_maps.shape[1]} < K={K}"`
