# PREREG bổ sung — điểm vận hành live v1.2 (khóa 21/11/2026, trước khi chạy biến thể có cổng)

Phạm vi: chỉ chế độ live (YOLOv8n → tracker → GapLink-VMS). Không đổi checkpoint (`release/final.pt`, `844fc391…`),
detector, `a_max`, vùng an ninh, cách chấm. Kết quả v1.1 giữ nguyên trong báo cáo.

## Minh bạch về những gì đã thấy
- Đã xem kết quả val (7/13 video) của họ A bên dưới trước khi viết tài liệu này: với YOLO, GapLink chỉ đạt Nat-WLR ≤ 5%
  ở τ ≥ 0,85 và Nat-IRR thấp hơn C-2. Chưa xem bất kỳ con số nào trên test ở chế độ v1.2.
- Họ B được đề xuất **vì** quan sát trên val đó. Đây là phát triển trên val, không phải trên test.

## Tập ứng viên (cố định)
- Họ A — GapLink, quy tắc v1.1: `τ ∈ {0,25; 0,35; …; 0,95}` (8 cấu hình).
- Họ B — GapLink có cổng Gaussian (`src/gated_assoc.py`): loại ứng viên có
  `d² = ((cx−μx)/σx)² + ((cy−μy)/σy)²` vượt `χ²₂(p)` với `p ∈ {95%; 99%; 99,9%}` dùng μ, σ do chính mô hình dự đoán
  (σ chưa hiệu chuẩn, như live v1.1), `τ` như họ A (24 cấu hình).
- Baseline: C-2, C-3, C-4 quét θ theo lưới trong `scripts/eval_live_v12.py`.

## Quy tắc chọn (PREREG §3 áp cho live)
- Trên 13 video val: với mỗi phương pháp (GapLink = hợp A ∪ B), chọn cấu hình có **Nat-IRR cao nhất thỏa Nat-WLR ≤ 5%**;
  nếu không có, chọn Nat-WLR thấp nhất. Hòa Nat-IRR → Nat-WLR thấp hơn → không cổng trước có cổng → cổng chặt hơn.
- Cấu hình chọn được ghi vào `release/live_v1.2.yaml`, commit + tag `v1.2-live-selected` **trước** khi chạy test.
- Test live v1.2 chạy **một lần**; báo cáo mọi kết quả kể cả khi GapLink thua baseline.

## Giới hạn đã biết
- Val chỉ 4 địa điểm; 32 cấu hình GapLink làm tăng nguy cơ quá khớp val → báo cáo cả Nat-IRR/WLR trên val lẫn test.
- Video test live đã được chấm ở v1.1 (ngưỡng khác). v1.2 là lần chấm thứ hai, cùng checkpoint (DEVIATIONS D10).
