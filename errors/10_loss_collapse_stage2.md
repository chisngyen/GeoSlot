# Bug #10: Stage 2 Loss Collapse (Loss Stuck at 4.8590)

## Mô tả
Khi transition từ Stage 1 → Stage 2 (epoch 15), loss đột ngột tăng lên ~4.86
rồi **stuck tại đó vĩnh viễn**, không giảm.
Accuracy cũng không cải thiện (3-12%).

## Triệu chứng
- Stage 1 (epoch 1-14): loss giảm bình thường, accuracy tăng
- Stage 2 (epoch 15+): loss nhảy lên ~4.8590 rồi không thay đổi
- `loss_csm` và `loss_dice` dominate `total_loss`
- Gradient norm rất nhỏ → model không learn

## Root Causes (nhiều vấn đề cùng lúc)

### 1. ContrastiveSlotLoss không detach transport plan
```python
# TRƯỚC (sai):
T = out["transport_plan"]  # T vẫn trong computation graph
# Gradient flow: loss_csm → T → SinkhornOT → slots → backbone
# Circular: CSM cố maximize slot similarity THEO T, nhưng T cũng thay đổi theo slots

# SAU (đúng):
T = out["transport_plan"].detach()  # Break circular gradient
```

### 2. CSM temperature quá thấp
```python
# TRƯỚC: temperature = 0.1 → logits quá sharp → gradient vanish
# SAU:   temperature = 0.5 → smoother distribution → gradient flows
```

### 3. Stage 2 weights quá lớn
```python
# TRƯỚC: lam_cs=1.0, lam_di=0.5 → CSM+Dice dominate InfoNCE
# SAU:   lam_cs=0.3, lam_di=0.1 → InfoNCE vẫn là primary loss
```

### 4. Missing warm-up ramp for Stage 2 losses
```python
# TRƯỚC: Stage 2 losses full weight ngay lập tức → shock
# SAU:   Linear ramp 0→1 over warmup_epochs:
def _ramp(self, epoch, stage_start):
    return min(1.0, (epoch - stage_start + 1) / self.warmup)

total = total + ramp * (self.lam_cs * loss_cs + self.lam_di * loss_di)
```

### 5. DiceLoss smooth quá nhỏ
```python
# TRƯỚC: smooth = 1e-5 → division instability khi overlap ≈ 0
# SAU:   smooth = 0.1  → stable ratio
```

## Fix Summary
| Component | Before | After |
|-----------|--------|-------|
| CSM transport plan | in graph | `.detach()` |
| CSM temperature | 0.1 | 0.5 |
| λ_csm | 1.0 | 0.3 |
| λ_dice | 0.5 | 0.1 |
| Stage transition | instant | linear ramp over warmup |
| DiceLoss smooth | 1e-5 | 0.1 |

## Files affected
- Tất cả files chứa `ContrastiveSlotLoss`, `JointLoss`, `DiceLoss`

## Nguyên tắc tránh lỗi
> **Multi-stage training checklist:**
> 1. **Detach** cross-module supervision signals (transport plan, attention maps)
> 2. **Warm-up ramp** khi thêm loss mới — KHÔNG bao giờ full weight đột ngột
> 3. **Auxiliary loss << Primary loss** — λ_aux ≤ 0.3 × λ_primary
> 4. **Temperature** cho contrastive loss: bắt đầu 0.5, không dưới 0.1
> 5. **Smooth factor** cho ratio-based loss (Dice, IoU): ≥ 0.01
>
> Khi loss stuck → check: gradient norm, individual loss components, 
> xem loss nào dominate total.
