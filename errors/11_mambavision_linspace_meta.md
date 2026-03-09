# Bug #11: MambaVision torch.linspace Meta Tensor Crash

## Mô tả
MambaVision model sử dụng `torch.linspace` internally khi khởi tạo positional embeddings.
Trên một số configurations (đặc biệt khi `low_cpu_mem_usage=True` hoặc trên accelerator),
`torch.linspace` tạo meta tensors → `.item()` crash.

## Triệu chứng
- `RuntimeError: cannot call .item() on meta tensor` khi load MambaVision
- Xảy ra trong `AutoModel.from_pretrained("nvidia/MambaVision-L-1K")`
- Không ảnh hưởng đến inference, chỉ model loading

## Root Cause
```python
# Bên trong MambaVision source code (transformers trust_remote_code):
x = torch.linspace(...)  # Có thể tạo meta tensor nếu device settings sai
x.item()                  # Crash: meta tensor không có data
```

## Fix
```python
class MambaVisionBackbone(nn.Module):
    def __init__(self, model_name, frozen=False):
        super().__init__()
        # Monkey-patch torch.linspace to force CPU device
        old_linspace = torch.linspace
        def patched_linspace(*args, **kwargs):
            kwargs["device"] = "cpu"
            return old_linspace(*args, **kwargs)
        torch.linspace = patched_linspace
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=False,  # QUAN TRỌNG: disable meta tensors
            )
        finally:
            torch.linspace = old_linspace  # Restore original
```

**Key points:**
1. `low_cpu_mem_usage=False` — không dùng meta tensor lazy loading
2. Monkey-patch `torch.linspace` force `device="cpu"` — đảm bảo tensor thật
3. `try/finally` — restore original function dù load thành công hay thất bại

## Files affected
- Tất cả files có `MambaVisionBackbone.__init__()`

## Nguyên tắc tránh lỗi
> **Khi load model với `trust_remote_code=True`:**
> 1. Set `low_cpu_mem_usage=False` trừ khi RAM thật sự không đủ
> 2. Nếu model crash khi load → check meta tensor issues
> 3. Monkey-patch là acceptable workaround cho third-party code
