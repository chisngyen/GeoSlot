# Bug #12: transformers Version Incompatibility

## Mô tả
MambaVision model dùng internal API `all_tied_weights_keys` từ `transformers` library.
Phiên bản transformers mới (≥ 4.45) đổi/xóa API này → crash khi load model.

## Triệu chứng
- `AttributeError: 'MambaVisionModel' object has no attribute 'all_tied_weights_keys'`
- Hoặc: `TypeError` khi khởi tạo model
- Chỉ xảy ra khi `pip install transformers` mà không pin version

## Root Cause
```python
# MambaVision remote code gọi:
self.all_tied_weights_keys  # API bị đổi/deprecate ở transformers >= 4.45
```

## Fix
```python
# Pin version khi install:
pip("transformers==4.44.2")  # MUST pin — newer versions break MambaVision
```

## Nguyên tắc tránh lỗi
> **Khi dùng `trust_remote_code=True`, LUÔN pin `transformers` version.**
>
> Remote code được viết cho một version cụ thể — upgrade transformers có thể break.
>
> Best practice:
> ```
> transformers==4.44.2  # Pinned for MambaVision compatibility
> ```
>
> Trước khi upgrade: test nếu model load thành công với version mới.
