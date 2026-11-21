# Sai lệch so với PREREG v1 và quy trình đóng băng (ghi nhận 21/11/2026)

Tài liệu này ghi lại mọi thay đổi xảy ra **sau** mốc đóng băng `v1.0-frozen` (ef23f27, 15/11/2026 22:37)
và mọi chỗ báo cáo diễn đạt khác `docs/PREREG.md`. PREREG v1 **không bị sửa**; mọi đánh giá giả thuyết
chính thức dùng đúng câu chữ PREREG. Bằng chứng: `git log`, tag `v1.0-frozen`, `final-v2`, `final-v3`.

## D1. Test bị chạy lại sau khi đã xem kết quả (checkpoint v1.0 → v1.1)
- `final-v2` (ce8acdf, 15/11 22:49): chạy test với checkpoint `589b4154…` (Top-1 83,59%).
- Phát hiện lỗi chuyển đổi dữ liệu: box vùng *ignored* bị gộp thành track giả ID −1 mỗi chuỗi.
  Sửa loader → tạo lại sự kiện → huấn luyện lại 5 seed → đóng băng lại `final-v3` (cd2d1b8, 16/11 12:12),
  checkpoint `844fc391…`, test chạy lại (`runs/test_final/test_results.json`, 16/11 12:08 +07).
- Cùng lúc thay đổi các tham số vận hành: `a_max 60→95`, `gaplink_tau 0,20→0,25`, `c2_theta 2,5→3,0`,
  phương pháp hiệu chuẩn (thêm `bin_scales`), thêm ngưỡng `*_theta_wlr5` cho baseline.
- **Hệ quả:** tập test 12.000 sự kiện đã được nhìn ít nhất 2 lần. Lý do sửa (lỗi dữ liệu, `a_max=60` < gap tối đa 90)
  là lỗi kỹ thuật chính đáng, nhưng **không thể chứng minh** các lựa chọn mới chỉ dựa trên val:
  `runs/d4_val/val_summary.json` bị ghi lại cùng phút với `test_results.json` (16/11 05:08 UTC).
  Báo cáo phải nêu rõ đây là lần chạy test thứ hai.

## D2. H1 trong báo cáo LaTeX khác câu chữ PREREG
- PREREG: Top-1 tập khó > max(C-2, C-3, C-4) **≥ 3,0 điểm %**.
- `chapter01*.tex` viết lại thành "≥ +4,0 pp so với C-3 và ≥ +2,0 pp so với C-4, CI không chứa 0".
- Kết luận theo PREREG không đổi (ĐẠT: +7,33 pp vs C-4, CI95 [6,33; 8,38], bootstrap theo chuỗi — xem D5),
  nhưng phải dùng lại đúng câu chữ PREREG.

## D3. H2 — đổi tập đo và đổi "mô hình chính" sau khi xem test
- PREREG: coverage **sau hiệu chuẩn**, đo trên **nửa B của validation**, cả 3 nhóm gap ∈ [85%, 95%].
- Hiệu chuẩn v1.1 trên val (`val_summary.json:calib_val.after`): 87,71% / 87,64% / **95,14%** → nhóm dài vượt 95%.
  Trên test (`calib_test.after`): 87,87% / 90,26% / **95,86%**.
- **H2 theo PREREG: KHÔNG ĐẠT (sát biên, nhóm gap dài).**
- Báo cáo chuyển sang "mô hình chưa hiệu chuẩn là mô hình chính" (commit c8a1a3c, 17/11) **sau khi thấy test**.
  Đây là phân tích hậu kiểm (exploratory), không được dùng để tuyên bố H2 đạt.
  Lưu ý thêm: `calib_val` (n=5466) không ghi rõ là nửa B hay toàn val → cần kiểm lại trong `scripts/calibrate_and_freeze.py`.

## D4. So sánh baseline "cùng WLR trên đường cong test"
- Các con số C-2 58,61% / C-3 24,82% / C-4 60,70% ở ~3% WLR được chọn **trên đường cong test** → chỉnh ngưỡng trên test.
  Chỉ được trình bày như phân tích độ nhạy, không phải kết quả chính. Kết quả chính dùng ngưỡng chọn trên val
  (`C-*_wlr5`) — lưu ý các ngưỡng này **không** giữ được WLR ≤ 5% trên test (C-3_wlr5: 9,13%; C-4_wlr5: 6,70%;
  tracker partial C-3 (WLR≤5%): 7,70%) vì val chỉ có 4 cụm địa điểm.

## D5. Bootstrap theo chuỗi, không theo cụm địa điểm
- `scripts/run_test.py` bootstrap trên **40 chuỗi test**, trong khi tập test chỉ có **15 địa điểm** (`data/splits.csv`).
  Các chuỗi cùng địa điểm tương quan → CI hiện tại hẹp hơn thực tế. Cần tính lại theo `location`
  (chỉ hậu xử lý trên `by_seq`, không chạy lại mô hình, không đổi ngưỡng).

## D6. Rò rỉ địa điểm train/val ↔ test chính thức
- 7 cụm camera xuất hiện ở cả train/val và test chính thức. Đã kiểm toán: Top-1 cụm trùng 93,39% vs cụm mới 93,44%
  (`chapter03_vi.tex`). Giữ nguyên, nhưng phải nêu là giới hạn của split UA-DETRAC.

## D7. Số latency mâu thuẫn
- README: 12,54 ms/frame (79,7 FPS; GapLink-VMS 1,10 ms).
- `runs/test_final/live_security_results.json` và LaTeX: **28,95 ms/frame (34,5 FPS)** ở ngã tư đông
  (YOLO 12,53 + HSV/NMS 3,26 + GapLink-VMS 13,16 ms). Số 79,7 FPS không có file nguồn trong `runs/` → không dùng làm số chính.

## D8. Clip demo VMS không thuộc quy tắc chọn của PREREG §4
- `demo/out/scenario_*` được render lại nhiều lần với chuỗi khác nhau (18–19/11). PREREG §4 chỉ khóa
  ghost_replay / no_match / tracker_full / failure_gallery. Clip VMS là minh họa định tính, không dùng làm bằng chứng định lượng.

## D9. Hai bản thống kê 5 seed lệch nhau rất nhỏ
- `test_results.json:event_level.GapLink-Lite_5seed`: Top-1 93,6416%, WLR 2,9735%, gap dài 80,679%.
- `audit_diagnostics.json:item6.Full` (bản LaTeX dùng): 93,6401%, 2,9823%, 80,666%.
- Chênh ≤ 0,01 pp, không đổi kết luận. Chỉ seed 1 (`final.pt`) có đầy đủ per-seed trong `test_results.json`;
  số 5 seed chỉ dùng mô tả độ ổn định, không dùng cho kiểm định giả thuyết.

## D10. Điểm vận hành live v1.2 (bổ sung sau v1.1)
- Với YOLOv8n, điểm v1.1 (`tau=0,25`, chọn trên sự kiện box GT) cho Nat-WLR 10,20% trên test — vi phạm tinh thần WLR ≤ 5%.
- v1.2 áp lại đúng quy tắc PREREG §3 cho chế độ live: với mọi phương pháp, chọn ngưỡng có Nat-IRR cao nhất thỏa Nat-WLR ≤ 5%
  trên **13 video val** (lưới cố định trong `scripts/eval_live_v12.py`). Ngưỡng được ghi vào `release/live_v1.2.yaml`
  và commit/tag `v1.2-live-selected` **trước** khi chạy test; test live v1.2 chạy một lần.
- Lưu ý: bộ YOLO test (`data/cache/yolo_test`) đã được dùng ở v1.1, nên v1.2 là lần đánh giá live thứ hai trên cùng video test
  (khác ngưỡng, cùng checkpoint). Kết quả v1.1 giữ nguyên trong báo cáo bên cạnh v1.2.
- Bước chạy kèm kiểm tra parity: điểm v1.1 chạy lại bằng scorer mới phải trùng khít `live_security_results.json`.

## D11. Tiêu chí chọn τ trên val dùng Top-1 sự kiện thay vì IRR tracker
- PREREG §3: "IRR cao nhất thỏa WLR ≤ 5%". `scripts/calibrate_and_freeze.py` chọn τ theo **Top-1 cấp sự kiện** thỏa WLR ≤ 5%.
- Trên val, Top-1 gần như phẳng theo τ (93,16–93,17% với τ ∈ [0,05; 0,30]) nên ảnh hưởng thực tế nhỏ, nhưng vẫn là sai lệch tiêu chí.
  Phiên bản live v1.2 dùng đúng IRR (Nat-IRR) theo PREREG.

## D12. Mở rộng tập ứng viên live v1.2 sau khi xem val (trước mọi lần chạy test v1.2)
- Kết quả val giữa chừng (họ A, chỉ ngưỡng τ) cho thấy GapLink chỉ đạt Nat-WLR ≤ 5% khi τ ≥ 0,85 và Nat-IRR thấp hơn C-2.
- Quét val họ A bị dừng tay (Ctrl+C) ở 10/13 video, chưa ghi `release/live_v1.2.yaml`, chưa commit/tag lựa chọn nào; bước test bị chặn thêm bằng `release/live_v1.2.HOLD`.
- Thêm họ B (GapLink có cổng Gaussian từ chính μ, σ của mô hình), tập ứng viên + quy tắc ghi ở `docs/PREREG_live_v1.2.md`,
  commit/tag `v1.2b-prereg` trước khi đánh giá họ B. Không retrain, không đổi checkpoint. Test v1.2 vẫn chạy đúng một lần.

## D13. Bỏ v1.2/v1.2b trước test, chuyển sang live v1.3 (checkpoint riêng cho chế độ live)
- v1.2b dừng tay khi quét val; không có test nào chạy ở v1.2/v1.2b. Họ B (cổng Gaussian) là kết quả âm trên val (một phần video).
- v1.3 thêm checkpoint `release/live_v1.3.pt` fine-tune trên sự kiện YOLO của tập train; chỉ dùng cho chế độ live.
  Kế hoạch khóa ở `docs/PREREG_live_v1.3.md` (tag `v1.3-prereg`) trước khi sinh dữ liệu và huấn luyện.

## D14. Bộ chấm dwell bị đổi sau lần chạy live v1.1
- `runs/test_final/live_security_results.json` tạo ngày 18/11 (code 0690d37). Commit f31a7c9 (20/11) thêm bộ lọc xe đứng yên
  vào `src/security_analytics.py` nhưng không cập nhật định nghĩa GT dwell trong `scripts/eval_live_security.py`.
- Chạy lại điểm v1.1 bằng code hiện tại: Nat-IRR, Nat-WLR và mọi số đếm tuyến cấm **trùng khít**; số đếm dwell khác
  (GapLink TP 429 → 191, C-2 372 → 179). Cột dwell trong bảng live v1.1 chỉ đúng với code 18/11, không dùng để so sánh với v1.3.
- Bảng live v1.3 trong báo cáo không có cột dwell.

## D15. Kết quả live v1.3 (một lần test, 21/11/2026)
- Chọn trên val: GapLink-v13 τ=0,65 (77,55% / 4,21%), C-2 θ=1,2 (81,56% / 4,87%), C-3 θ=1,0 (không đạt ≤5%), C-4 θ=1,2.
- Test: GapLink-v13 84,41% Nat-IRR / 6,09% Nat-WLR; C-2 84,58% / 12,92%. ΔWLR −6,82 pp, CI theo địa điểm [−11,04; −3,18].
- Thứ hạng Nat-IRR giữa val và test khác nhau (val C-2 dẫn 4 pp) → val 4 địa điểm yếu; không phương pháp nào giữ WLR ≤ 5% trên test.

## D16. `release/main.yaml` không khớp checkpoint đóng băng
- `release/main.yaml` ghi 6.000 bước, lr 5e-4; checkpoint `final.pt` lưu `cfg.steps = 2400`, `step = 2400` và log
  `runs/main/train_log.json` (đánh giá mỗi 600 bước) cũng là 2.400 bước. Learning rate không được lưu trong checkpoint.
- Báo cáo ghi 2.400 bước theo checkpoint và không nêu giá trị lr không kiểm chứng được. Script huấn luyện sinh ra log này
  (có `loss_assign_ema`) không còn trong repo → hạn chế về khả năng tái lập huấn luyện (không ảnh hưởng tới đánh giá).

## D17. Ca lỗi báo cáo trước đây không theo quy tắc PREREG §4
- `report/failure_cases/` (tạo bởi `scripts/run_test.py`) là 10 lỗi **đầu tiên theo thứ tự sự kiện**, không phải 10 sự kiện có
  P(đáp án đúng) thấp nhất chia 3-3-4 theo nhóm gap như PREREG §4.
- `scripts/failure_analysis.py` (21/11) chọn lại đúng quy tắc từ dự đoán đóng băng (kiểm tra trùng khít số đóng băng trước khi chọn);
  báo cáo dùng `report/figures/fig_failure_cases.png` mới. Không đổi ngưỡng, checkpoint hay tập test.
