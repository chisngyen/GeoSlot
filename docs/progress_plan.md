# Kế hoạch Nghiên cứu & Theo dõi Tiến độ: CVGL với Vision Mamba và Slot Attention

## 1. Tổng quan Dự án (Project Overview)
**Mục tiêu:** Tích hợp Vision Mamba (Vim), Adaptive Slot Attention, Graph Mamba (Relational Reasoning) và Attention-Guided Optimal Transport (AGOT) vào bài toán Cross-View Geo-Localization (CVGL) để giải quyết nút thắt về độ phức tạp tính toán $\mathcal{O}(N^2)$ của ViT và các vấn đề bất biến không gian/tỷ lệ của học đa góc nhìn.

**Môi trường triển khai dự kiến:** 
- Nền tảng: Kaggle
- Phần cứng: NVIDIA H100
- Dữ liệu: 4 bộ dữ liệu (University-1652, CV-Cities, CVUSA, VIGOR)

## 2. Lộ trình Triển khai (Roadmap)

### Giai đoạn 1: Khởi tạo và Phân tích (Đang thực hiện)
- [x] Phác thảo ý tưởng cốt lõi (idea.txt)
- [x] Lên kế hoạch và tạo file theo dõi tiến độ (`progress_plan.md`)
- [ ] Thực hiện Literature Review về Vision Mamba, Slot Attention, và Graph trong CVGL
- [ ] Đánh giá tính mới (Novelty) và đề xuất hệ thống
- [ ] Chốt kiến trúc tổng thể

### Giai đoạn 2: Chuẩn bị Dữ liệu (Sắp tới)
- [ ] Nhận cấu trúc thư mục dữ liệu từ Kaggle
- [ ] Xây dựng Data Dataloaders/Transforms hỗ trợ nhiều bộ dataset khác nhau
- [ ] Implement hàm đánh giá (Metrics/Evaluation)

### Giai đoạn 3: Triển khai Kiến trúc (Implementation)
- [ ] Xây dựng Backbone Vision Mamba / VimGeo
- [ ] Triển khai Adaptive Slot Attention
- [ ] Triển khai Graph Mamba cho Lập luận Quan hệ Không gian
- [ ] Thiết kế và cài đặt Sinkhorn Optimal Transport (AGOT) module
- [ ] Cài đặt các hàm mất mát (DWBL, Symmetric InfoNCE, Dice Loss)

### Giai đoạn 4: Huấn luyện và Tối ưu (Training & Optimization)
- [ ] Code training loop cho môi trường Kaggle (H100)
- [ ] Huấn luyện trên University-1652 & CV-Cities trước
- [ ] Fine-tuning/Ablation study trên từng module hạn chế nút thắt
- [ ] Đánh giá trên CVUSA và VIGOR (test cross-area)

### Giai đoạn 5: Viết báo cáo khoa học (Paper Writing)
- [ ] Tổng hợp kết quả, so sánh với SOTA
- [ ] Chuẩn bị hình ảnh mô tả kiến trúc
- [ ] Hoàn thiện báo cáo khoa học

---
*Cập nhật lần cuối: Hôm nay.*
