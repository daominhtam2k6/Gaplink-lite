# GapLink-Lite: Tài liệu đề tài và kế hoạch thực hiện 8 ngày

**Tên đề tài (tiếng Việt):** GapLink-Lite — Mạng nơ-ron nhỏ tự phát triển để tái liên kết danh tính phương tiện qua khoảng trống che khuất, có ước lượng bất định được hiệu chuẩn

**Tên đề tài (tiếng Anh):** GapLink-Lite — A Small Self-Developed Neural Network for Vehicle Identity Re-association Across Occlusion Gaps with Calibrated Uncertainty

**Loại:** Đề tài nghiên cứu sinh viên, thị giác máy tính, theo dõi đa đối tượng (MOT)
**Thời lượng:** 8 ngày × 5 giờ ≈ 42 giờ (Ngày 0 chuẩn bị ≈ 2 giờ, ngoài 8 ngày)
**Môi trường:** laptop Windows, GPU NVIDIA RTX 3050 6GB, ổ `D:\` trống 140GB
**Đối tượng theo dõi:** phương tiện giao thông (không phải người)

### Ba lưu ý về tính trung thực của tài liệu

1. **Chưa có đoạn mã nào trong tài liệu này được chạy thử.** Mã là khung, phải qua unit test trước khi tin.
2. **Mọi chi tiết về UA-DETRAC** (định dạng XML, tên chuỗi, dung lượng, giấy phép) là **từ trí nhớ, chưa xác minh**. Ngày 0 tồn tại chính là để kiểm chứng chúng.
3. **Mọi con số thời gian là ước tính.** Mô hình rất nhỏ (dưới 1 triệu tham số), nên nút cổ chai là **sinh sự kiện trên CPU**, không phải GPU.

---

# PHẦN I. MÔ TẢ BÀI TOÁN

## 1. Bối cảnh và động cơ

Trong video giao thông, một xe chạy sau xe buýt và biến mất khoảng 2 giây. Khi nó hiện lại, bộ theo dõi (tracker) thường làm một trong ba việc sai:

- **Phân mảnh track:** cấp ID mới cho cùng một xe, làm đứt quỹ đạo.
- **Hoán đổi ID (ID switch):** gán ID của xe A cho xe B vì B xuất hiện gần vị trí A biến mất.
- **Giữ "ma" track quá lâu:** tiếp tục dự đoán một xe đã rời cảnh.

Các tracker cổ điển xử lý bằng **Kalman filter** (ngoại suy chuyển động theo vật lý) cộng **khoảng cách Mahalanobis** (cổng thống kê). Cách này yếu ở đúng chỗ khó nhất: nhiều xe giống nhau, chuyển động phi tuyến (rẽ, dừng đột ngột), và khoảng trống dài.

**Động cơ của đề tài:** dữ liệu tracking công khai đã có hàng nghìn quỹ đạo thật với ID chính xác. Ta có thể **tự tạo hàng trăm nghìn sự kiện mất dấu** bằng cách giấu một đoạn quỹ đạo thật, rồi huấn luyện mạng trả lời câu hỏi: *"Trong các phát hiện mới, cái nào là chiếc xe vừa biến mất, hoặc không cái nào cả?"* Nhãn lấy từ ID gốc, nên không cần gán tay.

## 2. Định nghĩa hình thức

### 2.1. Một sự kiện (event)

Một sự kiện tại **thời điểm xác nhận** `t_conf` gồm:

**Tập track mất dấu** `T = {1..N}`, `N ≤ 6`. Với mỗi track `i`:
- Lịch sử `H_i = {b_{i,1}, ..., b_{i,K}}`, `K = 20`, là các box `(cx, cy, w, h)` trong 20 frame cuối trước `t_lost_i`.
- Độ dài gap `g_i = t_conf − t_lost_i`.
- Ngữ cảnh `X_i`: tối đa 8 xe lân cận tại `t_lost_i` và 8 xe tại `t_conf`.

**Tập ứng viên** `C = {1..M}`, `M ≤ 10`. Mỗi ứng viên `j` là một box `b_j` tại `t_conf` chưa được gán cho track đang nhìn thấy, kèm histogram màu HSV `a_j`.

### 2.2. Đầu ra

**(a) Ma trận gán xác suất** `P` kích thước `(N+1) × (M+1)`, hàng `N+1` và cột `M+1` là **dustbin** ("không khớp"). Ràng buộc: tổng mỗi hàng thật = 1, tổng mỗi cột thật = 1, khối lượng dustbin nhận phần còn lại.

**(b) Phân bố dự đoán vị trí** cho mỗi track: một Gaussian đường chéo 4 chiều `N(μ_i, diag(σ_i²))` trên `(cx, cy, log w, log h)` tương đối so với box cuối, và một **elip 90%** vẽ trên tâm sau hiệu chuẩn.

### 2.3. Nhãn

- `label[i] = j` nếu ứng viên `j` có **cùng ID gốc** với track `i`; ngược lại `label[i] = −1` (dustbin).
- `cand_label[j] = i` nếu ứng viên `j` khớp track `i`; ngược lại `−1` (xe mới sinh hoặc dương tính giả).
- Nhãn lấy từ **ID trong dữ liệu gốc**, không từ khoảng cách hình học.

### 2.4. Bốn nguồn thông tin mô hình dùng

| Nguồn | Nội dung | Được học? | Vai trò |
|---|---|---|---|
| Chuyển động | 20 box lịch sử + gap | Có (GRU) | **Chính** |
| Vị trí ứng viên | `m_ij = log p_i(b_j)`, IoU, tỉ lệ kích thước | Có (MLP chấm điểm) | **Chính** |
| Ngữ cảnh | 8 xe lân cận, cho biết ai đang che | Có (DeepSets) | Phụ trợ |
| Ngoại hình | Histogram màu HSV, cosine | **Không** (tính tay) | Yếu |

## 3. Câu hỏi nghiên cứu và giả thuyết

**Câu hỏi:** một mạng nhỏ tự phát triển, học từ quỹ đạo thật, có tái liên kết đúng hơn ngoại suy vận tốc không đổi và Kalman + Mahalanobis không? Và elip bất định nó vẽ ra có đáng tin không?

| Giả thuyết | Phát biểu | Có thể bị bác bỏ bởi |
|---|---|---|
| **H1** | Top-1 trên tập khó cao hơn C-2 và C-3 ít nhất 3 điểm % | Kết quả val Ngày 4 |
| **H2** | Coverage 90% nằm trong 85–95% ở cả 3 nhóm gap, sau hiệu chuẩn, đo trên nửa val không dùng để khớp | Cổng G5 |
| **H3** | IRR cao hơn C-2 và C-3 ở ≥ 2/3 nhóm gap, WLR không cao hơn | Cổng G6 |

**C-1** (tắt GapLink, xe hiện lại luôn nhận ID mới) **không phải giả thuyết** mà là **kiểm tra giao thức**: IRR của C-1 phải gần 0 (< 5%). Nếu cao, giao thức ẩn quá yếu hoặc cách tính IRR có lỗi.

**Tiêu chí "đề tài hoàn chỉnh"** (không phụ thuộc kết quả tốt hay xấu):
1. Có mô hình tự viết, huấn luyện từ đầu, đóng băng bằng hash.
2. Có đánh giá mức sự kiện và mức tracker trên địa điểm chưa từng thấy.
3. Có khoảng tin cậy và phân tích lỗi.
4. Có demo và báo cáo trung thực. Nếu H1–H3 sai, báo cáo nêu rõ và giải thích.

## 4. Phạm vi: những gì bị cắt có chủ đích

| Bị cắt | Lý do | Hệ quả (ghi vào hạn chế) |
|---|---|---|
| CNN ngoại hình học được | Tốn 6–8 giờ, làm chậm huấn luyện | Hai xe cùng màu dễ nhầm |
| Hỗn hợp Gaussian đa đỉnh | Không cần cho 8 ngày | Không mô hình hóa ngã tư rẽ |
| Set Transformer + attention | Sinkhorn đã cho cạnh tranh | Suy luận cấp tập yếu hơn |
| Tracker-in-the-loop | Tốn 5–6 giờ | Còn lệch train/inference |
| Gap tự nhiên | Ít, không ổn định | Chỉ dùng gap nhân tạo |
| HOTA | DetA bị nhiễu bởi detection bị xoá | Dùng IRR/WLR/FragRate + IDF1 |
| So sánh mô hình (ablation rộng) | Theo yêu cầu | Chỉ 3 ablation trên val |

## 5. Hai yêu cầu bắt buộc và khai báo đóng góp

| Yêu cầu | Cách thỏa | Bằng chứng nộp kèm |
|---|---|---|
| **1. Đối tượng không phải người** | Chỉ phương tiện (car, van, bus, others). Bộ lọc lớp trong loader loại mọi lớp người | Biểu đồ phân bố lớp `report/fig_class_hist.png` |
| **2. Mô hình cốt lõi tự phát triển** | GapLink-Lite, loss, bộ sinh sự kiện, hàm `build_event`, tracker, giao thức tiêm đều do sinh viên viết; **khởi tạo ngẫu nhiên, không trọng số tiền huấn luyện** | Git history, log từ bước 0, bảng khai báo đóng góp |

**Bảng khai báo đóng góp (đưa vào báo cáo):**

| Loại | Thành phần |
|---|---|
| **Tự phát triển** | Mạng GapLink-Lite (motion encoder, predictor, context encoder, điểm cặp, Sinkhorn có dustbin), hàm loss, bộ sinh sự kiện + `build_event`, tracker trực tuyến, giao thức tiêm che khuất, hiệu chuẩn theo gap |
| **Thư viện có khai báo** | PyTorch, NumPy, SciPy (Hungarian, L-BFGS), OpenCV, Matplotlib, pytest |
| **Baseline tham chiếu** | C-2 (ngoại suy vận tốc), C-3 (Kalman + Mahalanobis) — do sinh viên viết, dùng làm mốc so sánh, không phải lõi |
| **Không dùng làm lõi** | ByteTrack, DeepSORT, OC-SORT, StrongSORT, hay bất kỳ codebase tracker nào. Chỉ đọc tham khảo |

**Ghi chú trung thực:** GRU, Gaussian NLL, Sinkhorn (kiểu SuperGlue), DeepSets đều là kỹ thuật phổ biến. Đóng góp là **thiết kế riêng cho bài toán**: bất định phụ thuộc gap được hiệu chuẩn, ngữ cảnh vật che, dustbin theo phần tử, và giao thức đánh giá dùng chung `p_partial` giữa train và test. Không tuyên bố phát minh từng khối.

---

# PHẦN II. DỮ LIỆU VÀ BỘ SINH SỰ KIỆN

## 6. Dữ liệu

### 6.1. Lựa chọn

| Bộ | Đối tượng | Ghi chú |
|---|---|---|
| **UA-DETRAC** (chính) | Xe cộ | Camera cố định, 960×540, 25 fps, mật độ cao, nhiều che khuất, có box + ID |
| KITTI Tracking (dự phòng 1) | Car, cyclist | Nhỏ, tải nhanh; **lọc bỏ pedestrian** |
| VisDrone-MOT (dự phòng 2) | Xe (góc nhìn drone) | Khó hơn, chuyển động lớn |

**Cần tải cả ảnh/video lẫn nhãn.** Nếu chỉ có nhãn: bỏ đặc trưng màu, demo vẽ trên nền trắng.

### 6.2. Chia theo địa điểm, không theo chuỗi

Nhiều chuỗi đến từ cùng một góc quay. Chia ngẫu nhiên theo chuỗi → cùng địa điểm nằm ở cả train và test → mô hình học thuộc bố cục đường.

Quy trình:
1. Mỗi chuỗi: lấy ~50 frame trải đều, tính **trung vị theo từng pixel** → ảnh nền không có xe.
2. Thu nhỏ 64×36, chuyển xám, chuẩn hóa.
3. Khoảng cách mọi cặp: `1 − Pearson` hoặc L2 sau chuẩn hóa.
4. Gom phân cấp (`scipy.cluster.hierarchy`), cắt ngưỡng sau khi xem dendrogram.
5. Ghép lưới ảnh nền theo cụm, **kiểm bằng mắt**, sửa tay ca mơ hồ.
6. Ghi `data/splits.csv` (`seq, location, split`), chia **70/15/15 theo địa điểm**.

Test tự động: không chuỗi nào ở hai tập; không địa điểm nào ở hai tập.

### 6.3. Định dạng nội bộ

MOTChallenge: `frame,id,x,y,w,h,conf,class_id,vis` (góc trên trái). Vùng ignored ghi tệp riêng `{seq}_ignore.txt`. `class_id` là **số nguyên**, tên lớp ghi `classes.json`.

Khi đánh giá: loại box GT và box detection nằm trong vùng ignored (kiểm bằng **tâm box**).

## 7. Bộ sinh sự kiện

### 7.1. Ba loại sự kiện

| Loại | Tỉ lệ | Cách tạo |
|---|---|---|
| `occl` — che khuất do vật che | 40% | Xe A bị xe B chồng lên (tỉ lệ giao > 0,3). Ẩn A tới khi hết chồng lấn |
| `drop` — mất detection ngẫu nhiên | 40% | Chọn A ngẫu nhiên, ưu tiên cảnh có xe gần |
| `nomatch_exit` — xe rời cảnh thật | 10% | Lấy từ cuối quỹ đạo, box sát biên ảnh |
| `nomatch_fp` — detector bỏ sót | 10% | Xe còn trong cảnh, xoá ứng viên thật |

### 7.2. Phân bố gap

40% trong `[3, 15]`, 40% trong `[16, 45]`, 20% trong `[46, 90]`. Chỉ chọn track có độ sống ≥ `8 + g + 3` và có ≥ 8 box trước `t_lost`.

### 7.3. Sự kiện nhiều track

Chọn xe chính A. Thêm 0–3 xe phụ đang sống tại `t_conf` và nằm trong `4 × h` quanh vị trí dự kiến của A. Mỗi xe phụ có gap riêng cùng phân bố nhóm. Thêm xe mới sinh và 0–2 dương tính giả. **Tổng `N ≤ 6`, `M ≤ 10`.** Cuối cùng **xáo trộn thứ tự** track và ứng viên, rồi tính lại `label` và `cand_label`.

### 7.4. Nhiễu và `p_partial`

**Ứng viên:** box thật tại `t_conf` + nhiễu tâm `N(0, 0.05·size)`, kích thước ±10%. Với xác suất **`p_partial = 0,4`**, cắt ngẫu nhiên 60–100% diện tích về phía vật che. **Hàm cắt dùng chung cho huấn luyện và giao thức tiêm** — đây là điểm sửa lỗi "lệch nhô dần" của bản v2.

**Lịch sử track:** nhiễu `σ = 0,03 × size`.

**Bỏ sót:** bỏ ngẫu nhiên 5% ứng viên thật.

### 7.5. Sự kiện khó (chỉ dùng hình học)

`is_hard = True` nếu tại `t_conf` tồn tại **ứng viên gây nhiễu** có tâm cách tâm ứng viên đúng `< 1,5 × max(w, h)` của xe đúng. Không dùng kết quả baseline để định nghĩa (tránh vòng lặp).

- Huấn luyện: tỉ lệ sự kiện khó tăng 30% → 60% theo bước.
- Đánh giá: **tập khó cố định phải có ≥ 300 sự kiện.**

### 7.6. Ba lỗi phải tránh

1. Rò thông tin tương lai vào `hist` (chỉ dùng frame ≤ `t_lost`).
2. Dùng khoảng cách thay vì ID gốc để gán nhãn.
3. Quên tính lại `label` sau khi xáo thứ tự.

---

# PHẦN III. MÔ HÌNH VÀ TRACKER

## 8. Kiến trúc GapLink-Lite

Toàn bộ khởi tạo ngẫu nhiên, **dưới 1 triệu tham số**.

### 8.1. Chuẩn hóa đầu vào

- Mỗi bước lịch sử: `[Δcx/h, Δcy/h, log(w/w_last), log(h/h_last), vx/h, vy/h]`, `h` = chiều cao box cuối.
- Vị trí tuyệt đối: `(cx/W, cy/H)`.
- Gap: `log(1+g)` → sin/cos 8 tần số → MLP → 32 chiều.

**In phân phối từng đặc trưng ngay khi viết** — xe nhỏ (h vài pixel) làm `Δ/h` bùng nổ. Đây là lỗi hay gặp nhất.

### 8.2. Các khối

**MotionEncoder.** Linear 6→64, GRU 2 lớp ẩn 128, lấy trạng thái cuối theo độ dài thật (mask), ghép vị trí tuyệt đối + gap embedding → `h_i ∈ R¹²⁸`.

**Predictor.** MLP từ `[h_i, ctx_i]` → `μ` (4) và `log σ²` (4), clamp `log σ² ∈ [−6, 4]`. Loss Gaussian NLL. `m_ij = log p_i(b_j)`.

**ContextEncoder (DeepSets).** Mỗi xe lân cận: `(dx, dy)/h`, tỉ lệ kích thước, vận tốc tương đối, IoU với box cuối, cờ thời điểm (`t_lost` hay `t_conf`) → MLP → mean + max pooling → Linear 128.

**Đặc trưng màu.** Embed `cos(màu_i, màu_j)` vào điểm cặp. Màu track = trung bình 3 histogram cuối. **Khi `partial`: bỏ màu với xác suất 20%** (đặt cosine = 0 và cờ riêng), vì histogram tính từ box đầy đủ không phản ánh đúng box bị cắt.

**Điểm cặp và dustbin.**

```
s_ij = MLP([h_i, ctx_i, box_j_chuẩn_hóa, m_ij, cos(a_i,a_j),
            IoU(μ_i, b_j), log_tỉ_lệ_kích_thước, khoảng_cách/h, gap_emb])
z_i = MLP(h_i, gap_emb, vị_trí_track)          # dustbin theo track
z_j = MLP(đặc_trưng_ứng_viên, vị_trí)          # dustbin theo ứng viên
```

Dustbin theo **vị trí** quan trọng: xe gần biên ảnh nhiều khả năng đã rời cảnh.

**Sinkhorn.** 20 vòng, **fp32** (dù khối khác dùng AMP), mask đầy đủ cho vùng pad, khối lượng biên hàng/cột thật = 1.

### 8.3. Hàm mất mát

`L = L_assign + 0,5 · L_nll`

- `L_assign`: `−log P[i, label[i]]` cho track có nhãn; `−log P[i, M]` cho track dustbin; `−log P[N, j]` cho ứng viên `cand_label[j] = −1`. Trung bình theo số phần tử hợp lệ.
- `L_nll`: Gaussian NLL trên box thật.

### 8.4. Siêu tham số khởi điểm

Batch 256, AdamW `lr 5e-4`, `wd 1e-2`, cosine, warmup 500, clip gradient 1.0, seed 42. Số bước chốt **sau khi đo thông lượng ở Ngày 3** (dự kiến 30–60 nghìn, chưa đo).

## 9. Tracker trực tuyến

```
mỗi frame t:
  dets = detections(t)                    # GT có nhiễu, trừ vùng ignored
  1) khớp IoU + Hungarian (ngưỡng 0.3) với box dự đoán vận tốc không đổi
  2) track không khớp > 1 frame → lost_pool (tối đa A_max)
  3) dets chưa khớp: gán vào lost_pool → build_event
     nếu vượt N>6 / M>10: chia thành phần liên thông theo cổng χ²(9,21)
  4) P = GapLink(event); Hungarian trên P (bỏ dustbin); chấp nhận nếu P_ij > τ
  5) ghi events.jsonl
```

**Thuật toán chia nhóm khi vượt giới hạn:** đồ thị hai phía track–ứng viên, cạnh nếu ứng viên nằm trong cổng `χ²(2, 99%) ≈ 9,21` của dự đoán (dùng C-3 khi GapLink chưa sẵn); mỗi thành phần liên thông là một sự kiện con; nếu vẫn vượt, giữ 6 track và 10 ứng viên gần nhất theo điểm cổng.

**Schema `events.jsonl`** (một dòng mỗi quyết định):

```json
{"seq":"MVI_40171","t":412,"lost_ids":[17],"t_lost":[385],"gap":[27],
 "cand_box":[[x,y,w,h]],"P":[[0.91,0.06,0.02,0.01]],
 "mu":[[cx,cy,logw,logh]],"sigma":[[...]],"pi":[0.7,0.2,0.1],
 "decision":{"kind":"relink","id":17},"injected":true,"gt_id":17}
```

---

# PHẦN IV. ĐÁNH GIÁ

## 10. Thiết kế đánh giá

### 10.1. Mức sự kiện (val/test)

Xuất bảng: top-1 (toàn bộ, theo 3 nhóm gap, tập khó), tỉ lệ liên kết nhầm, dustbin P/R/F1, NLL, coverage. **CI 95% bằng bootstrap theo cụm = chuỗi**, 2.000 lần.

### 10.2. Mức tracker với giao thức tiêm che khuất

`make_hide_plan(tracks, rng, p=0.8, ...)`:
- Khoảng ẩn `[a, b]` dài `g` (phân bố 40/40/20), cách đầu quỹ đạo ≥ 8 frame và cách cuối ≥ 5.
- Cho phép **nhiều khoảng ẩn không chồng** trên cùng một xe.
- Giới hạn mật độ ≤ 20% xe bị ẩn đồng thời tại mỗi frame (duyệt ngẫu nhiên, loại khoảng vượt ngưỡng).
- Seed cố định, lưu `hide_plan_{seq}.json`.
- Hai biến thể: **full** (xoá detection) và **partial** (dùng cùng hàm `p_partial`).

Yêu cầu: ≥ 600 khoảng ẩn, ≥ 150 mỗi nhóm gap, ≥ 300 khoảng ẩn khó.

**Lưu ý quan trọng:** tăng số khoảng ẩn **không thu hẹp CI nhiều**, vì CI phụ thuộc số **chuỗi** (~9), không phải số sự kiện. Phải báo cáo thẳng điều này.

### 10.3. Chỉ số

| Chỉ số | Định nghĩa |
|---|---|
| **IRR** | Giữ đúng ID sau khi hiện lại |
| **WLR** | Mang ID của xe khác |
| **FragRate** | Bị cấp ID mới |
| **Lost** | Không theo lại được trong 10 frame |
| **IDF1** | F1 danh tính trên toàn chuỗi |

IRR **luôn đi kèm WLR**. Một bộ liên kết liều lĩnh có thể IRR cao nhưng nối nhầm nhiều.

### 10.4. Bốn hệ thống được chạy song song

| Hệ thống | Mô tả | Vai trò |
|---|---|---|
| **GapLink-Lite** | Mô hình chính | Đối tượng nghiên cứu |
| **C-1** | Tắt GapLink, xe hiện lại luôn nhận ID mới | **Kiểm tra giao thức** (IRR phải < 5%) |
| **C-2** | Ngoại suy vận tốc không đổi | Baseline |
| **C-3** | Kalman + Mahalanobis + Hungarian | Baseline mạnh nhất |

**Điểm công bằng bắt buộc:** C-2 và C-3 phải **quét ngưỡng trên val giống hệt GapLink** (cùng lưới, cùng tiêu chí chọn, cùng tập ứng viên).

### 10.5. Hiệu chuẩn

`σ' = σ · exp(a + b·log(1+g))`. Khớp `(a, b)` bằng L-BFGS trên NLL của **nửa A val** (tách theo chuỗi), đo coverage trên **nửa B**. Nếu một hệ số tuyến tính không đủ: thêm số hạng `log²`.

**Quyết định đã chốt:** hiệu chuẩn **chỉ áp cho elip hiển thị và cổng lọc**, **không** áp vào `m_ij` của điểm gán (tránh làm P lệch so với lúc học). Ghi rõ vào báo cáo.

### 10.6. Quy tắc kỷ luật

- Mọi ngưỡng chỉ chỉnh trên **val**.
- Test chạy **một lần** ở Ngày 7, sau khi đóng băng bằng SHA256 ở Ngày 6.
- Mọi tiêu chí chọn (clip demo, ca lỗi, ngưỡng) **chốt trước khi nhìn kết quả**.
- Bootstrap **ghép cặp** theo chuỗi cho hiệu GapLink − baseline.

---

# PHẦN V. KẾ HOẠCH TỪNG NGÀY

## Nhịp làm việc chung (áp dụng cả 9 ngày)

- **10 phút đầu:** đọc mục tiêu, ghi 3 việc chính vào `docs/daily_log.md`.
- **20 phút cuối:** `git commit`, ghi **kết quả thật** (không ghi mong muốn), viết một mục báo cáo.
- **Mọi lần chạy dài:** chạy nền, ghi log ra tệp (Phụ lục D).

## Bảng tổng quan

| Ngày | Loại | Trọng tâm | Cổng |
|---|---|---|---|
| **0** | Chuẩn bị | Cài đặt, tải dữ liệu, xem XML thật, đo thông lượng | **G0** |
| **1** | Xây dựng | Chuyển MOT, thống kê, gom địa điểm, `build_event` | **G1** |
| **2** | Xây dựng | DataLoader, tập cố định, C-2, C-3, eval sự kiện | **G2** |
| **3** | Xây dựng | Mạng lõi, Sinkhorn, unit test, overfit 64 | **G3** (chặn cứng) |
| **4** | Xây dựng | Ngữ cảnh, màu, huấn luyện chính, so baseline | **G4** |
| **5** | Xây dựng | Hiệu chuẩn, tracker, giao thức tiêm, chỉ số IRR | **G5** |
| **6** | Xây dựng | Quét ngưỡng, so sánh công bằng, **đóng băng** | **G6** |
| **7** | Sửa lỗi + test | Đệm sửa lỗi, chạy test một lần, phân tích | — |
| **8** | Hoàn thiện | Demo, báo cáo, gói nộp | — |

---

# NGÀY 0 — Chuẩn bị

**Thời lượng:** ≈ 2 giờ làm thực (tải dữ liệu chạy nền)
**Mục tiêu:** máy sẵn sàng, dữ liệu thật nằm trên đĩa, đã nhìn thấy định dạng nhãn thật, biết tốc độ sinh dữ liệu của CPU.

### 0.1. Kiểm tra máy (10 phút)

```powershell
nvidia-smi
wmic cpu get name,NumberOfCores,NumberOfLogicalProcessors
wmic computersystem get TotalPhysicalMemory
python --version
```

Ghi vào `docs/daily_log.md`: phiên bản driver NVIDIA, **số lõi logic**, RAM. Số lõi quyết định `num_workers` ở Ngày 2.

### 0.2. Cài công cụ (30–40 phút)

```powershell
winget install -e --id Python.Python.3.11
winget install -e --id Git.Git
winget install -e --id Microsoft.VisualStudio.Code
winget install -e --id Gyan.FFmpeg
winget install -e --id 7zip.7zip
```

Đóng và mở lại PowerShell. Kiểm tra `python --version`, `git --version`, `ffmpeg -version`. Cấu hình Git:

```powershell
git config --global user.name "Ten ban"
git config --global user.email "email@cua.ban"
```

Trong VS Code cài extension **Python**.

### 0.3. Khung dự án và môi trường (30 phút)

```powershell
cd D:\
mkdir gaplink; cd gaplink
"data","src","scripts","tests","configs","runs","release","demo","docs","report" | ForEach-Object { New-Item -ItemType Directory -Force $_ | Out-Null }
New-Item -ItemType Directory -Force data\raw,data\mot,data\cache | Out-Null
git init
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu `Activate.ps1` bị chặn: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, rồi kích hoạt lại. Đầu dòng phải hiện `(.venv)`.

Tạo `.gitignore`:

```powershell
@"
.venv/
data/
runs/
release/*.pt
__pycache__/
*.pyc
*.npy
*.pkl
*.mp4
*.log
"@ | Out-File -Encoding utf8 .gitignore
```

Cài PyTorch: **vào pytorch.org → Get Started → Windows / Pip / CUDA, dùng đúng lệnh trang đó in ra** (phiên bản thay đổi theo thời gian; đừng chép lệnh trong tài liệu này).

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
pip install numpy scipy pandas opencv-python matplotlib tqdm pyyaml scikit-learn lxml pytest
pip freeze > requirements.txt
```

### 0.4. Khởi động tải dữ liệu (10 phút thao tác, chạy nền)

Tìm trang chính thức UA-DETRAC (từ khóa "UA-DETRAC benchmark"), tải vào `D:\gaplink\data\raw\`. Cần **cả ảnh/video của các chuỗi có nhãn, và tệp nhãn XML**. Đọc giấy phép trước khi dùng.

Nếu sau 60 phút vẫn chưa tải được: chuyển sang **KITTI Tracking** (chỉ lớp Car). Kế hoạch không đổi vì mọi thứ đi qua định dạng MOT.

### 0.5. Kiểm tra môi trường (15 phút)

`scripts/check_env.py`:

```python
import torch, os
print("torch", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name, "| VRAM GB:", round(p.total_memory/1e9, 1))
    x = torch.randn(2000, 2000, device="cuda")
    print("matmul ok:", (x @ x).shape)
print("CPU cores:", os.cpu_count())
```

```powershell
python scripts\check_env.py
```

### 0.6. Đo thông lượng sinh dữ liệu (20 phút) — quan trọng nhất Ngày 0

Đây là phép đo quyết định toàn bộ lịch. `scripts/bench_events.py`:

```python
import time, numpy as np
rng = np.random.default_rng(0)

def fake_one_event():
    N, M, K = rng.integers(1, 5), rng.integers(1, 8), 20
    hist = rng.random((N, K, 4)).astype("float32")
    cand = rng.random((M, 4)).astype("float32")
    ctx  = rng.random((N, 16, 6)).astype("float32")
    ch   = rng.random((M, 128)).astype("float32")
    return hist, cand, ctx, ch

if __name__ == "__main__":
    n = 3000
    t0 = time.time()
    for _ in range(n):
        fake_one_event()
    dt = time.time() - t0
    print(f"{n/dt:.0f} events/s (1 process)")
```

```powershell
python scripts\bench_events.py
```

Nếu một tiến trình đạt ≥ 1.500 sự kiện/giây thì 8 worker đủ cho 100k sự kiện trong vài phút. Đây chỉ là **cận trên** (chưa tính đọc dữ liệu và cắt box).

### 0.7. Xem dữ liệu thật (40 phút, sau khi tải xong)

```powershell
cd D:\gaplink\data\raw
Get-ChildItem -Recurse -Depth 1 | Select-Object -First 40 FullName
```

`scripts/inspect_xml.py`:

```python
import sys, glob, os, collections
from lxml import etree

xmls = sorted(glob.glob(os.path.join(sys.argv[1], "**", "*.xml"), recursive=True))
print("So XML:", len(xmls))
print("10 ten dau:", [os.path.basename(p) for p in xmls[:10]])
p = xmls[0]
with open(p, encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        print(line.rstrip())
        if i >= 40: break
root = etree.parse(p).getroot()
print("\nThe goc:", root.tag, dict(root.attrib))
print("Con cap 1:", collections.Counter(c.tag for c in root))
fr = root.findall("frame")
print("So frame:", len(fr))
if fr:
    print("frame[0]:", dict(fr[0].attrib))
    tl = fr[0].find("target_list")
    if tl is not None and len(tl):
        t = tl[0]
        print("target:", dict(t.attrib))
        for c in t: print("  con:", c.tag, dict(c.attrib))
```

**Ghi lại bằng chữ:** tên thẻ frame, thuộc tính số frame, tên thẻ box và các thuộc tính, có `vehicle_type` không, có `ignored_region` không, ảnh là `.jpg` từng frame hay video.

### 0.8. Đếm sơ bộ và log (15 phút)

Bao nhiêu chuỗi có cả nhãn và ảnh? Bao nhiêu frame mỗi chuỗi? Nhãn tập test chính thức có công khai không? Tổng dung lượng đã dùng trên D:.

Viết mục Ngày 0, `git add . && git commit -m "Ngay 0: moi truong + du lieu"`.

### Cổng G0

- [ ] `check_env.py` in `CUDA available: True` và tên RTX 3050
- [ ] Đã ghi số sự kiện/giây của bench
- [ ] Dữ liệu trong `D:\gaplink\data\raw`, có **cả ảnh và nhãn** (hoặc đã quyết định dùng KITTI)
- [ ] Đã ghi lại cấu trúc XML thật
- [ ] Repo commit đầu tiên

**Nếu G0 chưa đạt:** không sang Ngày 1. Ngày 1 phụ thuộc hoàn toàn vào định dạng dữ liệu thật.

---

# NGÀY 1 — Dữ liệu, địa điểm, bộ sinh sự kiện (300 phút)

**Mục tiêu:** có `data/mot/*.txt`, `splits.csv` chia theo địa điểm, và `build_event` sinh được sự kiện đúng nhãn.

| Phút | Việc | Tệp |
|---|---|---|
| 20 | Đọc log Ngày 0, xác nhận thẻ XML | `docs/daily_log.md` |
| 40 | `convert_detrac.py`: XML → MOT, lọc lớp người, ghi vùng ignored | `scripts/convert_detrac.py` |
| 35 | `stats.py`: phân bố lớp, số xe/chuỗi, độ sống track, gap tự nhiên | `scripts/stats.py` |
| 60 | **Gom địa điểm** (mục 6.2) và `splits.csv` | `scripts/group_locations.py` |
| 90 | `events.py`: `SeqIndex`, `build_event`, 3 loại sự kiện, `p_partial` | `src/events.py` |
| 40 | `viz_events.py`: vẽ 20 sự kiện, kiểm bằng mắt | `scripts/viz_events.py` |
| 15 | Log + mục báo cáo "Dữ liệu" | |

### Việc cần làm

**Chuyển nhãn.** Định dạng đích MOT. Hai điểm dễ sai: box trong vùng ignored phải bị loại (kiểm bằng tâm box); `class_id` phải là số nguyên.

**Thống kê cần biết trước khi thiết kế sự kiện:**
- Phân bố lớp (bằng chứng "không phải người")
- Độ sống track (phân vị 10/50/90). Nếu trung vị < 60 frame thì nhóm gap 46–90 sẽ hiếm → giảm tỉ lệ nhóm dài, ghi vào báo cáo
- Số ID có frame thiếu giữa chừng (chỉ để báo cáo)
- Số khung hình có ≥ 10 xe (ước lượng tải cho tracker)

**Chữ ký `build_event`:**

```python
def build_event(idx: SeqIndex, rng, cfg, t_conf: int, tracks: list[int],
                mode: str | None = None, partial: bool | None = None) -> dict
```

**Kiểm bằng mắt (bước bắt lỗi rẻ nhất của cả dự án).** Vẽ 20 sự kiện vào `report/events_vis/`: frame `t_lost` (lịch sử, box cuối) và frame `t_conf` (ứng viên, đánh dấu đáp án đúng bằng viền dày). Nếu 20 cái đều hợp lý thì nhãn đúng.

### Cổng G1

- [ ] `data/mot/` có ≥ 40 chuỗi, `splits.csv` hợp lệ theo test tự động
- [ ] `stats.json` có phân bố lớp và độ sống track
- [ ] `build_event` sinh 10.000 sự kiện không lỗi, phân bố gap khớp 40/40/20 ±3%
- [ ] 20 sự kiện đã xem bằng mắt
- [ ] Log + commit

**Nếu trễ:** bỏ gom địa điểm tự động, dùng heuristic + kiểm mắt 15 phút. Không được bỏ `build_event`.

---

# NGÀY 2 — DataLoader, tập cố định, hai đối chứng (300 phút)

**Mục tiêu:** có số sàn C-2/C-3 trên val **trước khi viết mạng**. Đây là bước dễ bỏ qua và sẽ hối hận.

| Phút | Việc | Tệp |
|---|---|---|
| 15 | Khởi động | |
| 45 | `extract_hist.py` chạy nền + `dataset.py` | `src/dataset.py` |
| 45 | `collate_fn` pad (6,10) + mask; `worker_init_fn` | `src/dataset.py` |
| 20 | Đo thông lượng thật, quyết định trực tuyến vs sinh trước | `docs/daily_log.md` |
| 35 | Sinh `val_events.pkl`, `test_events.pkl` (seed cố định), đếm tập khó | `scripts/make_fixed_events.py` |
| 75 | **C-2** và **C-3** | `src/baselines.py` |
| 45 | `eval_events.py` + bootstrap theo cụm | `src/eval_events.py` |
| 20 | Log | |

### Việc cần làm

**Histogram màu.** HSV, 8×4×4 = 128 bin, vùng giữa 60% của box (tránh nền), L1-normalize, lưu `float16`/`uint8` vào `data/cache/hist_{split}.npy` + tệp chỉ mục. **Chạy nền ngay từ đầu ngày** vì mất thời gian I/O ảnh.

**DataLoader.** Train: `IterableDataset` sinh trực tuyến, hoặc `Dataset` sinh trước 200k rồi tăng cường (lật ngang, tịnh tiến nhỏ, nhiễu box) — quyết định theo phép đo 20 phút. Val/test: tĩnh, **không tăng cường**.

Windows: `num_workers` = số lõi logic − 2, `persistent_workers=True`, `pin_memory=True`, toàn bộ mã trong `if __name__ == "__main__":`.

**C-2.** Vận tốc = trung bình 5 bước cuối. Vị trí dự kiến = box cuối + `v·g`. Chi phí = khoảng cách tâm chuẩn hóa theo `h` + phạt chênh kích thước. Hungarian.

**C-3.** Kalman 8 chiều `(cx, cy, w, h, vcx, vcy, vw, vh)`. Dự đoán `g` bước không cập nhật → `(μ, S)`. Chi phí = Mahalanobis², cổng `χ²(2, 99%) ≈ 9,21` trên tâm. Hungarian.

**Điểm công bằng bắt buộc:** C-2 và C-3 **quét ngưỡng trên val giống hệt GapLink** (cùng lưới, cùng tiêu chí).

**`eval_events.py`.** Top-1 (toàn bộ, theo gap, tập khó), tỉ lệ liên kết nhầm, dustbin P/R/F1, NLL, coverage. CI 95% bằng bootstrap theo cụm = chuỗi, 2.000 lần. **Hàm bootstrap viết một chỗ, dùng lại ở Ngày 4–7.**

### Cổng G2

- [ ] ≥ 50k sự kiện train sinh được; val ≥ 5.000; **tập khó val ≥ 300**
- [ ] Chạy 1 epoch giả (chỉ DataLoader) để biết sự kiện/giây thật
- [ ] **Có số sàn C-2 và C-3 trên val** (toàn bộ, theo gap, tập khó) đã ghi vào `runs/baselines_val.json`
- [ ] Đếm được tần suất sự kiện vượt N>6 hoặc M>10

**Nếu tập khó < 300:** tăng số sự kiện sinh ở val. **Nếu C-3 quá mạnh:** ghi nhận để xem lại ở Ngày 6.

---

# NGÀY 3 — Mạng lõi, Sinkhorn, unit test, overfit (300 phút)

**Đây là ngày rủi ro nhất.** Sinkhorn có mask và dustbin sai âm thầm: loss vẫn giảm nhưng kết quả vô nghĩa.

| Phút | Việc | Tệp |
|---|---|---|
| 15 | Khởi động | |
| 40 | `features.py`: chuẩn hóa + in phân phối từng đặc trưng | `src/features.py` |
| 50 | `MotionEncoder` + `Predictor` (1 Gaussian đường chéo) | `src/motion.py`, `src/predictor.py` |
| 50 | Điểm cặp + dustbin theo phần tử | `src/assign.py` |
| 60 | `sinkhorn.py` (mask, fp32) + loss gán | `src/sinkhorn.py` |
| 45 | 5 unit test | `tests/` |
| 25 | Overfit 64 sự kiện | `configs/overfit.yaml` |
| 15 | Log + mục "Mô hình" | |

### Năm unit test bắt buộc

1. Tổng mỗi hàng và cột **hợp lệ** của `exp(logP)` ≈ khối lượng biên (tol 1e-3)
2. Đổi nội dung vùng padding → đầu ra **không đổi**
3. Ví dụ tay 2×2 điểm rõ ràng → P đúng như tính bằng giấy
4. Hoán vị thứ tự track/ứng viên → P hoán vị tương ứng
5. Coverage một Gaussian 2 chiều: mẫu giả lập cho ≈ 90% tại `d² ≤ 4,605`

### Cổng G3 (chặn cứng)

- [ ] 5/5 unit test đạt
- [ ] Overfit 64 sự kiện ≥ 99% accuracy
- [ ] Đo được bước/giây (để chốt số bước huấn luyện chính)

**Nếu không đạt: dừng, không sang Ngày 4.** Thứ tự gỡ lỗi: nhãn (vẽ lại 20 sự kiện) → mask padding → Sinkhorn (tổng hàng/cột) → chuẩn hóa (phân phối đặc trưng) → cân bằng dustbin trong batch → tốc độ học.

---

# NGÀY 4 — Ngữ cảnh, màu, huấn luyện chính (300 phút)

| Phút | Việc | Tệp |
|---|---|---|
| 15 | Khởi động | |
| 45 | `ContextEncoder` DeepSets (8 token, mean+max pooling) | `src/context.py` |
| 40 | Đặc trưng màu vào điểm cặp (+ quy tắc bỏ màu khi `partial`) | `src/assign.py` |
| 20 | Chạy lại overfit 64 | |
| 20 | Curriculum 30%→60% khó + **khởi chạy huấn luyện chính** (nền) | `configs/main.yaml` |
| 60 | Mở rộng `eval_events.py`: coverage (HDR/Mahalanobis), NLL theo gap | |
| 60 | Đánh giá val, so sánh C-2/C-3 | `runs/d4_val/` |
| 40 | Log + mục "Ngữ cảnh và ngoại hình" | |

### Cổng G4

**Tiêu chí:** top-1 trên **tập khó** (val) cao hơn `max(C-2, C-3)` ít nhất 3 điểm phần trăm.

- Không đạt → chạy danh sách gỡ lỗi (9 mục) **tối đa nửa ngày**: vẽ lại nhãn, kiểm rò rỉ chỉ số (xáo thứ tự ứng viên), kiểm mask, in phân phối đặc trưng, kiểm tổng hàng/cột của P, kiểm `L_nll` có chiếm hết gradient, thử `lr 1e-3` và `1e-4`, kiểm tỉ lệ dustbin, kiểm seed của các worker
- Vẫn không đạt → **đây là kết quả thật.** Tiếp tục, báo cáo trung thực, Ngày 7 phân tích tại sao. **Quy tắc cứng: không đuổi ngưỡng này quá nửa ngày.**

---

# NGÀY 5 — Hiệu chuẩn, tracker, giao thức tiêm (300 phút)

**Ngày dày nhất của cả kế hoạch.** Nếu cuối Ngày 4 chưa xong G4, kích hoạt thang cắt ở Phần VI ngay.

| Phút | Việc | Tệp |
|---|---|---|
| 15 | Khởi động, quyết định cắt gì | |
| 50 | **Hiệu chuẩn σ theo gap**, khớp nửa A, kiểm nửa B | `src/calib.py` |
| 60 | `tracker.py` + chia nhóm khi vượt N/M | `src/tracker.py` |
| 55 | Giao thức tiêm, hai biến thể full/partial | `src/inject.py` |
| 45 | IRR/WLR/FragRate/Lost + unit test | `src/metrics_tracker.py` |
| 30 | Khởi chạy 3 ablation (nền, tuần tự) | `configs/ablation_*.yaml` |
| 45 | Log + mục "Hiệu chuẩn" + "Tracker" | |

### Cổng G5

- [ ] Coverage trên nửa B của val trong **85–95% ở cả 3 nhóm gap**
- [ ] Tracker chạy hết các chuỗi val, không crash
- [ ] **C-1 có IRR < 5%** (kiểm tra giao thức đủ khó). Nếu cao: lỗi giao thức hoặc lỗi tính IRR, phải sửa trước khi tiếp
- [ ] Unit test `make_hide_plan` (mật độ, khoảng cách đầu/cuối, không chồng) và `irr_wlr_frag` đạt

---

# NGÀY 6 — Quét ngưỡng, so sánh công bằng, đóng băng (300 phút)

| Phút | Việc | Tệp |
|---|---|---|
| 15 | Khởi động; **chốt tiêu chí chọn ngưỡng trước khi nhìn số** | `docs/daily_log.md` |
| 70 | Quét τ và A_max cho GapLink **và** ngưỡng tương ứng cho C-2, C-3 (cùng điều kiện) | `configs/tracker.yaml` |
| 40 | Chạy val đầy đủ, cả hai biến thể tiêm, cả 4 hệ thống | `runs/d6_val/` |
| 20 | **Cổng G6** | |
| 30 | Đóng băng: `final.pt`, SHA256, tag, `calib.json` | `release/` |
| 45 | `scripts/run_test.sh`, thử trên 1 chuỗi val | |
| 60 | Bắt đầu `demo/render.py` (box ma, elip, thanh xác suất) trên val | `demo/render.py` |
| 20 | Log | |

**Tiêu chí chọn ngưỡng (chốt trước):** IRR cao nhất với **WLR ≤ 5%**. Áp **giống hệt** cho cả 4 hệ thống.

### Cổng G6

IRR(GapLink) > IRR(C-2) và > IRR(C-3) ở **≥ 2/3 nhóm gap** trên val, WLR không cao hơn.

- Không đạt → **vẫn đóng băng.** Không huấn luyện lại, không nhìn test để sửa. Ghi nhận, chuyển sang phân tích ở Ngày 7.

### Đóng băng (bắt buộc, đúng thứ tự)

```powershell
Copy-Item runs\main\best.pt release\final.pt
Get-FileHash release\final.pt -Algorithm SHA256 | Out-File release\final.sha256
Copy-Item configs\main.yaml,configs\tracker.yaml,configs\overfit.yaml release\
Copy-Item runs\main\calib.json release\
git add release requirements.txt configs
git commit -m "Dong bang mo hinh v1.0"
git tag v1.0-frozen
```

Chạy lại val bằng `final.pt` để xác nhận **tái lập** (ra đúng số đã ghi). Nếu lệch, có yếu tố ngẫu nhiên chưa cố định — phải tìm ra **trước** khi test.

---

# NGÀY 7 — Đệm sửa lỗi, chạy test một lần, phân tích (300 phút)

**Buổi sáng là đệm.** Nếu các ngày trước trễ, đây là chỗ bù. Chỉ sửa lỗi **kỹ thuật** ở phần không ảnh hưởng mô hình.

| Phút | Việc |
|---|---|
| 15 | Khởi động, kiểm tra danh sách lỗi tồn |
| 75 | **Đệm sửa lỗi**: mã vẽ, mã tính chỉ số, đường dẫn, README, tiến trình crash |
| 20 | Kiểm tra cuối: hash khớp, `run_test.sh` chạy được, val tái lập |
| 60 | **Chạy test đúng một lần**: sự kiện + tracker (full/partial) + C-1/C-2/C-3 |
| 45 | Bootstrap **ghép cặp** theo chuỗi cho hiệu GapLink − C-2, GapLink − C-3 |
| 45 | Hình: bảng kết quả, top-1 và IRR theo gap, IRR–WLR, coverage trước/sau, ablation |
| 40 | Phân tích 10 ca lỗi theo quy tắc đã chốt |

**Nếu phải chạy lại test** (crash, sai mã chỉ số): ghi lý do và thay đổi vào nhật ký, **giữ cả hai kết quả**. Không đổi mô hình, ngưỡng, dữ liệu, tiêu chí.

**Bootstrap ghép cặp:** lấy mẫu lại **cùng một tập chuỗi** cho cả hai hệ thống trong mỗi lần lặp, tính hiệu, rồi lấy phân vị 2,5% và 97,5%. Cách này thu hẹp CI đáng kể và là cách duy nhất để nói "khác biệt có ý nghĩa" khi chỉ có ~9 cụm.

**Chọn 10 ca lỗi theo quy tắc chốt trước:** 10 ca có P của đáp án đúng thấp nhất, chia đều giữa các nhóm gap. Mỗi ca ghi: ảnh trước/sau, P, nhóm lỗi, nguyên nhân.

---

# NGÀY 8 — Demo, báo cáo, gói nộp (300 phút)

| Phút | Việc |
|---|---|
| 10 | Khởi động |
| 90 | Render 4 clip: Ghost Replay (chậm 0,25×), No-Match, video tracker đầy đủ, Failure Gallery |
| 90 | Báo cáo 6–7 trang: gắn bảng/hình test, hạn chế, khai báo đóng góp |
| 30 | README: cài đặt, tải dữ liệu, 1 lệnh train, 1 lệnh eval, kết quả chính |
| 30 | Checklist gói nộp + sao lưu USB |
| 50 | Dự phòng |

### Bốn clip demo

| Clip | Nội dung | Thời lượng |
|---|---|---|
| **Ghost Replay** | Xe chui sau vật che, video chậm 0,25×, box ma + elip phình theo gap, thanh xác suất, nháy xanh "ID recovered" | 60–90 s |
| **No-Match** | Xe rời cảnh, xác suất dồn vào dustbin, "NO MATCH → ID mới" | 30–45 s |
| **Video tracker đầy đủ** | ID có màu, vệt đường đi, bộ đếm `Re-links \| New IDs \| Dustbin \| WLR` | 45–60 s |
| **Failure Gallery** | 2–3 ca lỗi, mỗi ca một dòng nguyên nhân | 30–45 s |

**Nhãn bắt buộc trên mọi clip:** khoảng ẩn là **nhân tạo (tiêm)** hay che khuất tự nhiên; và câu "clip chọn theo quy tắc cố định, không theo chất lượng".

### Cấu trúc báo cáo

| Mục | Trang | Viết vào ngày |
|---|---|---|
| 1. Giới thiệu, câu hỏi, giả thuyết | 0,5 | 1 |
| 2. Dữ liệu, gom địa điểm, chia, biểu đồ lớp | 0,75 | 1 |
| 3. Bộ sinh sự kiện, tập khó | 1 | 2 |
| 4. Mô hình và loss | 1 | 3–4 |
| 5. Hiệu chuẩn, ablation | 0,5 | 5 |
| 6. Tracker, giao thức tiêm, chỉ số | 0,75 | 5–6 |
| 7. Kết quả (bảng, theo gap, CI, IRR–WLR) | 1 | 7 |
| 8. Phân tích lỗi | 0,5 | 7 |
| 9. Hạn chế và mối đe doạ tính hợp lệ | 0,4 | 8 |
| 10. Kết luận, khai báo đóng góp | 0,35 | 8 |

**Mối đe doạ tính hợp lệ phải có:** box GT có nhiễu không phải detector thật; camera cố định; ~9 chuỗi test nên CI rộng; gap nhân tạo; histogram màu tính từ box đầy đủ; không có tracker-in-the-loop nên còn lệch train/infer; ablation một seed; gom địa điểm có thể chưa hoàn hảo.

---

# PHẦN VI. THANG CẮT GIẢM VÀ RỦI RO

## 11. Thang cắt giảm (điểm kích hoạt cụ thể)

| Thời điểm | Nếu chưa xong | Cắt |
|---|---|---|
| Cuối N1 | Chưa có `build_event` | Bỏ gom địa điểm tự động, dùng heuristic + kiểm mắt 15 phút |
| Cuối N2 | Chưa có C-3 | Giữ C-2, đơn giản hoá C-3 (Kalman Q/R mặc định), ghi hạn chế |
| Giữa N3 | Sinkhorn chưa qua test | **Không cắt gì.** Dừng và sửa. Lấy đệm sáng N7 |
| Giữa N4 | Huấn luyện chậm | Bỏ màu (ablation (ii) thành mặc định), giảm batch, giảm số bước |
| Cuối N4 | | Bỏ ablation (iii) "bỏ nhận biết gap" |
| Giữa N5 | | Bỏ biến thể **partial** trong giao thức; bỏ IDF1 |
| Cuối N5 | | Bỏ ablation (i) "bỏ ngữ cảnh" |
| N6 | | Rút lưới quét: τ ∈ {0,4; 0,5; 0,6}, A_max = 60 |
| N8 | | Chỉ 3 clip: Ghost Replay, video tracker, Failure Gallery |

**Không bao giờ cắt:** giao thức tiêm (bản full), C-2 và C-3, bootstrap ghép cặp, hiệu chuẩn theo gap, phân tích lỗi, khai báo đóng góp, kỷ luật test một lần.

## 12. Rủi ro chính

| Rủi ro | Mức | Cách xử lý |
|---|---|---|
| Ngày 3 (Sinkhorn) trượt | Cao | Unit test 5 mục là khoản đầu tư rẻ nhất. Có đệm sáng N7 |
| Ngày 5 quá tải | Cao | Thang cắt có điểm kích hoạt cụ thể; quyết định cắt từ đầu ngày |
| Mô hình không thắng baseline | Trung bình–cao | Camera cố định làm ngoại suy rất mạnh. Chuẩn bị tâm thế từ G4. Kết quả âm có phân tích và CI vẫn là đề tài hoàn chỉnh |
| C-3 quá mạnh, không còn chỗ thắng | Trung bình | Xem lại giao thức ở N6, dịch về phía khó hơn **trước khi đóng băng** |
| Mô hình học thuộc bố cục cảnh | Trung bình | Chia theo địa điểm; thử bỏ vị trí tuyệt đối trên val |
| Không tải được dữ liệu | Thấp | KITTI Car, loader dùng định dạng MOT |
| Chỉ có nhãn, không có ảnh | Thấp | Bỏ màu, demo vẽ trên nền trắng |
| Số khoảng ẩn ít, CI rộng | Chắc chắn | Tăng p, nhiều khoảng ẩn/xe; nói thẳng "CI rộng vì ít cụm" |
| Laptop nóng, tự ngắt | Thấp | Cắm sạc, chế độ hiệu suất cao, tắt tự ngủ, checkpoint thường xuyên |

---

# PHẦN VII. PHỤ LỤC

## Phụ lục A. Hợp đồng dữ liệu

Một sự kiện là `dict` với các khóa sau. Tất cả box lưu **tọa độ tuyệt đối theo pixel**, chuẩn hóa chỉ làm bên trong mô hình.

| Khóa | Shape | Ý nghĩa |
|---|---|---|
| `seq` | str | Tên chuỗi |
| `t_lost` | (N,) int32 | Frame cuối mỗi track được thấy |
| `t_conf` | int | Frame xác nhận |
| `hist` | (N, 20, 4) f32 | 20 box cuối trước `t_lost`, dạng `cx, cy, w, h` |
| `hist_len` | (N,) int32 | Số box thật (≤ 20) |
| `gap` | (N,) int32 | `t_conf − t_lost` |
| `cand` | (M, 4) f32 | Ứng viên `cx, cy, w, h` |
| `cand_hist` | (M, 128) f32 | Histogram HSV, L1-normalized; toàn 0 nếu chưa có |
| `ctx_lost` | (N, 8, 6) f32 | 8 xe lân cận tại `t_lost` |
| `ctx_conf` | (N, 8, 6) f32 | 8 xe lân cận tại `t_conf` |
| `ctx_mask` | (N, 8) bool | True nếu ô có xe thật |
| `label` | (N,) int32 | Chỉ số ứng viên đúng, hoặc `-1` = dustbin |
| `cand_label` | (M,) int32 | Chỉ số track đúng, hoặc `-1` = dustbin |
| `is_hard` | bool | Sự kiện khó theo định nghĩa hình học |
| `gap_bin` | (N,) int8 | 0: 3–15, 1: 16–45, 2: 46–90 |
| `n_src` | str | `occl` / `drop` / `nomatch_exit` / `nomatch_fp` |

`collate_fn` pad N → 6, M → 10, kèm `n_valid`, `m_valid`. **Mọi module dùng chung `build_event`**, không viết bản thứ hai cho tracker.

## Phụ lục B. Cấu hình

```yaml
# configs/main.yaml
seed: 42
data:
  mot_dir: D:/gaplink/data/mot
  cache_dir: D:/gaplink/data/cache
  val_events: data/cache/val_events.pkl
  test_events: data/cache/test_events.pkl
  gap_probs: [0.4, 0.4, 0.2]
  p_partial: 0.4
  hard_ratio: {start: 0.3, end: 0.6, ramp_steps: 40000}
model:
  d: 128
  hist_len: 20
  n_ctx: 8
  n_cand_max: 10
  n_trk_max: 6
train:
  batch: 256
  steps: 40000          # chỉ chốt sau khi đo ở Ngày 3
  lr: 5.0e-4
  wd: 1.0e-2
  warmup: 500
  clip: 1.0
  amp: true             # Sinkhorn luôn fp32
  eval_every: 2000
  patience: 5
loss: {assign: 1.0, nll: 0.5}
num_workers: 6
```

```yaml
# configs/tracker.yaml  (chỉ sửa trên val, đóng băng ở N6)
tau: 0.5
a_max: 60
iou_gate: 0.3
chi2_gate: 9.21
```

## Phụ lục C. Cấu trúc repo

```
D:\gaplink\
├── README.md   requirements.txt   .gitignore
├── configs/    main.yaml  tracker.yaml  overfit.yaml  ablation_*.yaml
├── data/       raw/  mot/  cache/  splits.csv
├── src/        events.py  dataset.py  features.py
│               motion.py  predictor.py  context.py  assign.py  sinkhorn.py
│               baselines.py  calib.py  tracker.py  inject.py
│               eval_events.py  metrics_tracker.py
├── scripts/    check_env.py  bench_events.py  inspect_xml.py
│               convert_detrac.py  stats.py  group_locations.py  viz_events.py
│               extract_hist.py  make_fixed_events.py  bootstrap_paired.py
│               run_test.sh
├── tests/      test_sinkhorn.py  test_events.py  test_inject.py  test_metrics.py
├── train.py
├── runs/       main/  d4_val/  d6_val/  test_final/  baselines_val.json
├── release/    final.pt  final.sha256  main.yaml  tracker.yaml  calib.json
├── demo/       render.py  *.mp4
├── docs/       daily_log.md
└── report/     main.tex (hoặc .docx)  figures/  events_vis/
```

## Phụ lục D. Chạy nền trên Windows

```powershell
Start-Process -NoNewWindow -FilePath "python" `
  -ArgumentList "train.py --config configs/main.yaml" `
  -RedirectStandardOutput "runs\main\train.log" `
  -RedirectStandardError  "runs\main\train.err.log"
```

Theo dõi:
```powershell
Get-Content runs\main\train.log -Wait -Tail 30
```

Checkpoint mỗi 2.000 bước, giữ "tốt nhất" và "gần nhất", hỗ trợ `--resume`.

## Phụ lục E. Danh sách kiểm tra trước khi nộp

**Dữ liệu và quy trình**
- [ ] Chia theo địa điểm, test tự động không trùng chuỗi/địa điểm
- [ ] Biểu đồ phân bố lớp (không có người)
- [ ] Test chạy một lần (nếu lặp, ghi lý do kỹ thuật vào nhật ký)
- [ ] Mọi ngưỡng chỉnh trên val. Tiêu chí chọn chốt trước khi nhìn kết quả

**Mô hình**
- [ ] Khởi tạo ngẫu nhiên, không trọng số tiền huấn luyện
- [ ] `final.pt`, `final.sha256`, tag Git, config, seed
- [ ] Unit test Sinkhorn, giao thức tiêm, chỉ số đều đạt

**Kết quả**
- [ ] Bảng sự kiện: top-1, WLR, dustbin P/R/F1, NLL, coverage, theo gap và tập khó, có CI
- [ ] Bảng tracker: IRR, WLR, FragRate theo gap, hai biến thể, có CI ghép cặp
- [ ] C-1 (kiểm tra giao thức), C-2, C-3 trên cùng ứng viên
- [ ] Số khoảng ẩn ≥ 600, tập khó ≥ 300 (hoặc ghi con số thật)
- [ ] 10 ca lỗi có phân tích. Ablation val kèm ghi chú một seed

**Sản phẩm**
- [ ] Repo Git, README, lệnh train và eval một dòng
- [ ] 3–4 clip demo, có nhãn "nhân tạo" và quy tắc chọn clip
- [ ] Báo cáo có bảng khai báo đóng góp và mục hạn chế
- [ ] Bản sao USB

---