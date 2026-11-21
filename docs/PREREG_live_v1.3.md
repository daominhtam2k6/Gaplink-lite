# PREREG bổ sung — GapLink live v1.3 (detector-aware), khóa 21/11/2026 trước khi sinh dữ liệu / huấn luyện

## Bối cảnh và những gì đã thấy (minh bạch)
- v1.2 (chỉ đổi ngưỡng τ, họ A) và v1.2b (cổng Gaussian, họ B) **bị bỏ ở giai đoạn val, chưa chạm test**.
  Trên val (một phần video): họ A chỉ đạt Nat-WLR ≤ 5% ở τ ≥ 0,85 với Nat-IRR thấp hơn C-2; họ B làm Nat-IRR giảm mạnh
  vì σ học từ box GT quá hẹp so với nhiễu YOLO. Đây là động cơ của v1.3.
- Kết quả lõi v1.1 (event-level, tracker-level, H1–H3, P1) **không đổi**, vẫn từ `release/final.pt`.

## Can thiệp
- Sinh sự kiện "detector-aware": chạy YOLOv8n + tracker live trên các chuỗi **train** (và val để chọn checkpoint), nhãn
  danh tính từ khớp GT hậu kiểm IoU ≥ 0,45, tracker được dẫn bằng oracle để giữ track sạch (`src/det_events.py`,
  `scripts/gen_det_events.py`; tối đa 3.000 sự kiện/chuỗi train, 2.000/chuỗi val, lấy mẫu ngẫu nhiên seed 20261121). Không dùng chuỗi test.
- Fine-tune từ `release/final.pt` (trọng số tự huấn luyện, khởi tạo ngẫu nhiên ở v1.1 — không pretrained ngoài):
  3.000 bước, batch 64, lr 2e-4 (warmup 100, cosine), 70% sự kiện YOLO-train + 30% pool sự kiện gốc, loss như v1.1,
  1 seed (20261121) do giới hạn thời gian. Checkpoint chọn theo Top-1 trên sự kiện YOLO-**val** (τ=0,5).
  Checkpoint đóng băng: `release/live_v1.3.pt` (< 1 triệu tham số, cùng kiến trúc).

## Tập ứng viên và quy tắc chọn trên val (13 video)
- GapLink: {`final.pt`, `live_v1.3.pt`} × τ ∈ {0,25; 0,35; …; 0,95}. Baseline C-2/C-3/C-4: lưới θ như v1.2.
- Với mỗi phương pháp: **Nat-IRR cao nhất thỏa Nat-WLR ≤ 5%**; nếu không có, Nat-WLR thấp nhất.
  Hòa → WLR thấp hơn → `final.pt` trước `live_v1.3.pt`.
- Lựa chọn ghi vào `release/live_v1.3.yaml`, commit + tag `v1.3-live-selected` trước khi test.

## Test
- Test live v1.3 chạy **một lần** trên 40 video test, kèm parity với v1.1. Báo cáo mọi kết quả, kể cả khi GapLink thua C-2.
- Video test live đã được chấm ở v1.1 (ngưỡng/checkpoint khác) → v1.3 là lần chấm thứ hai (DEVIATIONS D10/D13).

## Giới hạn
- 1 seed; val chỉ 4 địa điểm; nhãn danh tính từ khớp IoU có thể sai ở đám đông dày.
