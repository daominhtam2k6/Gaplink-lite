# Nhật ký triển khai GapLink-Lite (8 ngày)

## Ngày 1 — 15/11/2026
- **Mục tiêu:**
  1. Khởi tạo thư mục `D:\gaplink`, môi trường ảo Python 3.12 với PyTorch CUDA và các thư viện, khóa `docs/PREREG.md`.
  2. Tải dữ liệu UA-DETRAC vào `D:\gaplink\data\raw`, kiểm tra cấu trúc thật của nhãn và ảnh.
  3. Chuyển đổi dữ liệu sang định dạng MOT nội bộ, thống kê phân bố lớp/độ dài track, gom cụm địa điểm để chia `splits.csv` và trích xuất histogram HSV.
- **Thông tin phần cứng đã xác minh:**
  - GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Driver 621.01, hỗ trợ CUDA 13.5).
  - Môi trường Python: CPython 3.12.16 tại `D:\gaplink\.venv`.
