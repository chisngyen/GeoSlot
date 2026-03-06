# Khảo sát Tài liệu (Literature Review)

Đánh giá các công nghệ cốt lõi trong bối cảnh Cross-View Geo-Localization (CVGL) dựa trên các công bố khoa học gần đây nhất (đến năm 2024-2025).

## 1. Vision Mamba (Vim) và State Space Models trong Computer Vision
**Bối cảnh:** Vision Transformers (ViT) đạt nhiều thành tựu trong CVGL nhưng bị giới hạn bởi độ phức tạp tính toán $\mathcal{O}(N^2)$, làm giảm khả năng xử lý ảnh độ phân giải cao trên các thiết bị giới hạn tài nguyên (như Drone/UAV).
**Tình trạng nghiên cứu:**
- **Vision Mamba (Vim):** Các nghiên cứu gần đây đã chứng minh Vim (dựa trên State Space Models) có thể đạt được *global receptive field* tương tự ViT nhưng với tốc độ tuyến tính $\mathcal{O}(N)$ (nhờ cơ chế quét tuyến tính - selective scanning).
- **Ứng dụng trong CVGL:** Việc chuyển đổi từ ViT sang Mamba Backbone (như VimGeo) đang là một hướng đi cực hot. VimGeo sử dụng các kỹ thuật như *Channel Group Pooling* để giữ lại các đặc trưng cục bộ mịn (fine-grained local features), rất phù hợp để phân biệt các tòa nhà tương tự nhau trong ảnh vệ tinh và ảnh drone.

## 2. Slot Attention & Trích xuất Đặc trưng Hướng Đối tượng (Object-Centric Representation)
**Bối cảnh:** So khớp biểu diễn toàn cục (global representation) truyền thống dễ bị sai lệch khi ảnh có nhiều nhiễu động hoặc che khuất cục bộ.
**Tình trạng nghiên cứu:**
- **Ứng dụng hiện tại của Slot Attention:** Slot Attention chủ yếu được dùng trong tác vụ *scene decomposition* (phân tách cảnh) và chỉnh sửa không gian 3D (như mô hình *Slot-TTA* cho multi-view RGB/3D point clouds hay *MVInpainter* cho multi-view 2D inpainting).
- **Khoảng trống Nghiên cứu (Research Gap):** Hiện tại, **chưa có nghiên cứu nào nổi bật** áp dụng trực tiếp Slot Attention vào bài toán nhận dạng vị trí địa lý chéo góc nhìn (CVGL). Việc dùng Slot Attention để trích xuất các "Slot" phân minh (như tòa nhà, nút giao thông, cây cối) độc lập thay vì so sánh toàn mảng pixel là một hướng đi **hoàn toàn mới mẻ và có tính đột phá cao.**

## 3. Optimal Transport (OT) trong So khớp Chéo Góc nhìn
**Bối cảnh:** Khi khung cảnh bị thay đổi góc nhìn mạnh (từ mặt đất sang vệ tinh), các khoảng cách vector truyền thống (Cosine Similarity, L2) thường không hoạt động hiệu quả giữa hai miền dữ liệu dị biệt (domain gap).
**Tình trạng nghiên cứu:**
- **OT trong CVGL:** Kỹ thuật *Cross-View Feature Transport (CVFT)* gần đây đã áp dụng lý thuyết Vận chuyển Tối ưu (Optimal Transport). Nó thiết lập ma trận vận chuyển để gióng hàng (align) các đặc trưng giữa góc nhìn mặt đất và trên không, giúp việc so sánh độ tương đồng trở nên có ý nghĩa về mặt hình học hơn.
- **Sự kết hợp Sinkhorn OT:** OT giải quyết hoàn hảo bài toán so khớp phân phối (bipartite matching) giữa tập hợp các Slots của Ground View và Satellite View mà không yêu cầu chúng phải được căn lề chặt chẽ.

**Kết luận Literature Review:** Ý tưởng kết hợp 3 nhánh này (Mamba cho feature extraction hiệu năng cao, Slot Attention cho object disentanglement, OT/Graph cho geometric matching) hoàn toàn có khả năng tạo ra một chuẩn mực (State-of-The-Art) mới và hội đủ tiêu chuẩn xuất bản tại các hội nghị hàng đầu như CVPR/ECCV.
