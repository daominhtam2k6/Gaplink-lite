# Tiền đăng ký giả thuyết và tiêu chí quyết định (PREREG v1)

**Ngày khóa:** 15/11/2026 (Ngày 1, trước khi huấn luyện hay đánh giá bất kỳ mô hình nào)

## 1. Giả thuyết nghiên cứu
- **H1 (Event-level Top-1 trên tập khó):** Top-1 accuracy của GapLink-Lite trên tập sự kiện khó (`is_hard == True`) cao hơn `max(C-2, C-3, C-4)` ít nhất **3,0 điểm phần trăm**.
- **H2 (Hiệu chuẩn bất định):** Coverage của vùng elip tin cậy 90% sau hiệu chuẩn theo gap nằm trong khoảng **[85%, 95%]** ở cả 3 nhóm độ dài gap (`[3..15]`, `[16..45]`, `[46..90]`), đo trên nửa B của tập validation (không dùng để khớp tham số hiệu chuẩn).
- **H3 (Tracker-level IRR & WLR):** Tỉ lệ khôi phục đúng ID (IRR) của GapLink-Lite cao hơn cả C-2 và C-3 ở **ít nhất 2/3 nhóm gap**, đồng thời tỉ lệ liên kết nhầm `ΔWLR <= +0.5` điểm phần trăm.
- **P1 (Kiểm tra giao thức C-1):** Khi tắt bộ tái liên kết (C-1), IRR phải `< 5%`. Nếu `>= 5%` thì giao thức tiêm hoặc phép đo có lỗi kỹ thuật và phải sửa trước khi đóng băng.

## 2. Quy mô dữ liệu đánh giá mục tiêu
- **Train pool:** 300.000 sự kiện (tối thiểu 150.000).
- **Validation (event):** >= 6.000 sự kiện, trong đó tập khó >= 300.
- **Test (event):** >= 12.000 sự kiện, trong đó tập khó >= 600 (chỉ chạy đúng 1 lần ở Ngày 7 sau khi đóng băng).
- **Tracker injection (val & test):** >= 600 khoảng ẩn hợp lệ, mỗi nhóm gap >= 150, tập khó >= 300.

## 3. Tiêu chí chọn ngưỡng trên Validation (chốt trước)
- Ngưỡng chấp nhận liên kết (`tau` cho GapLink-Lite, `theta` cho C-2/C-3/C-4) được quét trên cùng lưới ứng viên của tập validation.
- **Quy tắc chọn duy nhất:** Chọn ngưỡng đạt **IRR cao nhất** thỏa mãn ràng buộc **WLR <= 5,0%** (nếu không cấu hình nào đạt WLR <= 5,0%, chọn cấu hình có WLR thấp nhất).

## 4. Quy tắc chọn clip Demo và ca lỗi (tránh chọn lọc thiên vị)
- **`ghost_replay.mp4`:** Chọn xác định 3 khoảng ẩn từ tập khoảng ẩn khó của tập test (mỗi nhóm gap 1 khoảng) bằng `np.random.default_rng(777).choice`, ưu tiên khác chuỗi.
- **`no_match.mp4`:** Chọn 2 sự kiện `nomatch_exit` đầu tiên theo thứ tự tên chuỗi trong tập test.
- **`tracker_full.mp4`:** Chọn chuỗi test có số khoảng ẩn hợp lệ nhiều nhất (nếu hòa, chọn theo thứ tự từ điển của tên chuỗi).
- **`failure_gallery.mp4` & 10 ca lỗi báo cáo:** Lấy 10 sự kiện trên tập test mà GapLink-Lite gán xác suất `P` cho đáp án đúng thấp nhất (chia đều theo 3 nhóm gap: 3-3-4); lấy 3 ca đầu tiên đưa vào video.
- **Cam kết:** Giữ nguyên các clip được chọn theo quy tắc trên kể cả khi GapLink-Lite xử lý sai hoặc thua baseline ở clip đó.
