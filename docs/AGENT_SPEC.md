# AGENT_SPEC — GapLink-Lite v2 (8 ngày) — đặc tả cho agent triển khai

> **Đọc theo thứ tự:** §0 → §1 → §2 → ngày hiện tại ở §5. Quay lại §3–§4 khi viết code, §6 khi làm demo.
> **Thứ tự ưu tiên khi mâu thuẫn:** `AGENT_SPEC.md` (v2, tài liệu này) > `GapLink-Lite.md` (v1, người dùng cung cấp).
> v1 được chép vào repo thành `docs/GapLink-Lite_v1.md`. Các mục ghi **[INHERIT v1 §x]** nghĩa là làm đúng như v1 mục đó.
> Mọi chi tiết về UA-DETRAC (XML, tên chuỗi, dung lượng) là **từ trí nhớ/tìm kiếm, chưa xác minh** → ngày 1 phải kiểm chứng và ghi vào `docs/daily_log.md`.

---

## 0. Quy tắc vận hành bắt buộc

| # | Quy tắc |
|---|---|
| R1 | Thư mục làm việc `D:\gaplink` (là workspace; sandbox chỉ ghi được trong workspace). Mọi đường dẫn trong code lấy từ `configs/paths.yaml`, không hardcode. |
| R2 | **Lệnh cần mạng** (tải dữ liệu, `uv pip install`, `hf download`) chạy ngoài sandbox và cần người dùng duyệt. Gộp thành **một đợt duyệt** ở đầu ngày 1 (§2.2). Mọi lệnh khác chạy trong sandbox. |
| R3 | Lệnh chạy dài (train, sinh dữ liệu, trích histogram, render) chạy nền, log ra `runs/<name>/*.log`, có `--resume`. Không `sleep`, không poll liên tục; làm việc khác rồi quay lại khi hệ thống báo xong. |
| R4 | **Cấm xem kết quả test trước ngày 7.** `data/cache/test_events.pkl` được sinh ngày 2 và chỉ được đọc bởi `scripts/run_test.py`. Test tracker chỉ chạy ở ngày 7 (một lần). |
| R5 | **Tiêu chí chọn** (ngưỡng, clip demo, ca lỗi) ghi vào `docs/PREREG.md` **ngày 1**, commit tag `prereg-v1`. Sau đó không sửa. |
| R6 | **Mô hình lõi tự viết, khởi tạo ngẫu nhiên.** Cấm `torch.hub`, `from_pretrained`, tải trọng số, và cấm chép/import mã tracker bên thứ ba (ByteTrack, DeepSORT, OC-SORT, SuperGlue…). Sinkhorn, loss, tracker, `build_event`, giao thức tiêm đều viết từ đầu. Thư viện cho phép: PyTorch, NumPy, SciPy, OpenCV, pandas, Matplotlib, scikit-learn, lxml, pytest, imageio-ffmpeg, tqdm, pyyaml, (tùy chọn) motmetrics cho IDF1. |
| R7 | Chỉ phương tiện. Loader **loại mọi lớp người** (UA-DETRAC không có lớp người, nhưng vẫn viết bộ lọc và test). |
| R8 | Cuối mỗi ngày: `git commit`, ghi **kết quả thật** (không ghi mong muốn) vào `docs/daily_log.md`, cập nhật mục báo cáo tương ứng trong `report/`. |
| R9 | Mỗi lần chạy sinh một thư mục `runs/<name>/` chứa `config.yaml` (bản sao), `git_commit.txt`, `seed.txt`, `metrics.json`, log. |
| R10 | Seed cố định 42. Worker DataLoader/Pool dùng `np.random.default_rng([seed, worker_id])`. |
| R11 | Windows: mọi script chạy đa tiến trình phải có `if __name__ == "__main__":`; `multiprocessing` dùng `spawn`. |
| R12 | Khi một cổng (gate) không đạt: áp dụng **đúng** quy tắc ở §8. Không tự đổi mục tiêu. Nếu cần quyết định ngoài §8, dừng và hỏi người dùng. |

---

## 1. Quyết định đã chốt (không bàn lại)

### 1.1. Phạm vi và đối tượng
| ID | Quyết định |
|---|---|
| D-01 | Dữ liệu chính **UA-DETRAC**. `train` chính thức (60 chuỗi) → tách `train`/`val` theo địa điểm. `test` chính thức (40 chuỗi, địa điểm khác train theo tài liệu gốc, **kiểm chứng bằng ảnh nền**) → `test`. Nếu **không có nhãn test công khai**: tách 60 chuỗi theo địa điểm 60/20/20. |
| D-02 | Lớp: `car, van, bus, others` → `class_id` 1..4 (`classes.json`). |
| D-03 | Luồng detection cho giao thức = **GT + nhiễu + bỏ sót 5%** (không có detector học). Ghi vào hạn chế. Đã chốt dùng `p_partial = 0,4` dùng chung train và tiêm. |
| D-04 | Mô hình, loss, siêu tham số: **[INHERIT v1 §8]** với các làm rõ ở §4. Tham số < 1.000.000 (có test). |
| D-05 | Sự kiện huấn luyện **sinh trước** thành pool (không dùng IterableDataset). Hard/easy là hai pool; curriculum = tỉ lệ lấy mẫu giữa hai pool. **Không lật ngang** ảnh/tọa độ (lật đổi phía lái xe và bố cục làn). |
| D-06 | Thêm loại nhiễu mới so với v1: **distractor nhìn thấy** (0–2 xe đang nhìn thấy, gần vị trí dự kiến, `cand_label = −1`) để khớp với ứng viên thật của tracker (xe nhìn thấy nhưng IoU kém). |
| D-07 | Hệ thống so sánh: **C-1** (kiểm tra giao thức), **C-2** (vận tốc không đổi), **C-3** (Kalman + Mahalanobis), **C-4** (C-3 + kích thước + màu, trọng số chỉnh trên val; kiểm soát "đóng góp của đặc trưng so với đóng góp của học"). C-4 cắt sau cùng (§8). |
| D-08 | Ablation (một seed, ghi chú): (i) bỏ ngữ cảnh, (ii) bỏ màu, (iii) bỏ gap-aware. Chạy nền đêm ngày 4. |
| D-09 | Hiệu chuẩn σ theo gap chỉ áp cho elip hiển thị và cổng lọc, **không** áp vào `m_ij` **[INHERIT v1 §10.5]**. Báo thêm coverage của elip Kalman (thô và sau cùng hàm hiệu chuẩn). |
| D-10 | `a_max = 100` frame (**không phải 60**: gap tối đa 90 frame, `a_max = 60` làm nhóm gap 46–90 không thể khôi phục). Không tuyên bố `a_max` là siêu tham số quét. |
| D-11 | Demo: video **so sánh song song C-3 và GapLink** trên cùng chuỗi, cùng khoảng ẩn, với **hộp che khuất ảo nhìn thấy được** (§6). Bộ render **chỉ đọc log** (không chạy mô hình). |

### 1.2. Giả thuyết và ngưỡng tiền đăng ký (ghi vào `PREREG.md` ngày 1)
| ID | Phát biểu | Đo ở | Quy tắc kết luận |
|---|---|---|---|
| H1 | Top-1 trên **tập khó** của GapLink ≥ `max(C-2, C-3, C-4)` + **3 điểm %** | val (cổng G4) rồi test | Báo ΔTop1 + CI bootstrap ghép cặp theo chuỗi. "Có ý nghĩa" khi cận dưới CI > 0. |
| H2 | Coverage elip 90% (sau hiệu chuẩn) ∈ [85%, 95%] ở cả 3 nhóm gap, đo trên nửa B của val | G5 | Đạt/không đạt, báo thật. |
| H3 | IRR(GapLink) > IRR(C-2) và > IRR(C-3) ở ≥ 2/3 nhóm gap **và** ΔWLR ≤ +0,5 điểm % | G6 rồi test | Báo kèm CI. |
| P1 | C-1 có IRR < 5% | G5 | Không đạt ⇒ lỗi giao thức/metric, phải sửa. |

**Kỷ luật kết quả:** H1–H3 sai vẫn là đề tài hoàn chỉnh. Báo cáo viết đúng kết quả kèm phân tích. Không đổi ngưỡng sau khi thấy số.

### 1.3. Quy mô dữ liệu đánh giá
| Tập | Sự kiện (mức event) | Khoảng ẩn (mức tracker) |
|---|---|---|
| train pool | 300.000 (tối thiểu 150.000) | — |
| val | 6.000, **khó ≥ 300** | ≥ 600, mỗi nhóm gap ≥ 150, khó ≥ 300 |
| test | 12.000, **khó ≥ 600** | ≥ 600 (mục tiêu ≥ 1.500), mỗi nhóm gap ≥ 150, khó ≥ 300 |
Mỗi chuỗi có hạn ngạch sự kiện bằng nhau (`ceil(n / số_chuỗi)`) để không chuỗi nào chi phối.

---

## 2. Môi trường và dữ liệu (chạy ngày 1)

### 2.1. Hiện trạng máy (đã kiểm tra)
- GPU RTX 3050 6GB. Python hệ thống: `C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\python.exe`, `uv` có ở `C:\Users\LENOVO\.local\bin\uv.exe`, `git` có. **Chưa có** torch, cv2, ffmpeg. `D:\Anaconda\python.exe` bị lỗi shim ("Failed to find real location") → **không dùng Anaconda**.
- Sandbox không có mạng. Chưa đo được dung lượng trống ổ D (người dùng báo ~140 GB).

### 2.2. Đợt duyệt mạng duy nhất (làm ngay đầu ngày 1)
```powershell
# (1) môi trường — ngoài sandbox
cd D:\gaplink
uv venv .venv --python 3.12            # nếu không tải được 3.12: uv venv .venv --python "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\python.exe"
# Chọn index CUDA theo `nvidia-smi` (cột "CUDA Version") và pytorch.org/get-started; ví dụ cu126:
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe numpy scipy pandas opencv-python-headless matplotlib tqdm pyyaml scikit-learn lxml pytest imageio imageio-ffmpeg huggingface_hub motmetrics
# (2) dữ liệu — ngoài sandbox, chạy nền
.venv\Scripts\python.exe scripts\fetch_data.py   # xem 2.3
```
Kiểm tra: `.venv\Scripts\python.exe scripts\check_env.py` phải in `CUDA available: True` và tên RTX 3050 (matmul 2000×2000 trên GPU chạy được).

### 2.3. `scripts/fetch_data.py` (ưu tiên theo thứ tự, dừng ở nguồn đầu tiên thành công)
1. **Mirror Hugging Face** `abhineet123/ua_detrac` (dataset): `huggingface_hub.list_repo_files(repo, repo_type="dataset")` → in danh sách và kích thước, rồi tải. Tìm kiếm cho thấy có `ua_detrac_training_set.zip` ≈ 5,6 GB và `ua_detrac_test_set.zip` ≈ 4,3 GB (**chưa xác minh**; nhãn có thể nằm trong hoặc ngoài hai zip này). Nếu nhãn XML không có trong mirror → bước 2 chỉ cho nhãn.
2. **Trang chính thức** UA-DETRAC (tìm "UA-DETRAC benchmark"): cần `DETRAC-train-data`, `DETRAC-Train-Annotations-XML`, `DETRAC-test-data`, `DETRAC-Test-Annotations-XML`. Nếu chỉ có Google Drive: dùng `gdown`. Đọc giấy phép (nghiên cứu phi thương mại) và ghi vào `docs/daily_log.md`.
3. **Nếu cả hai thất bại trong 90 phút:** **dừng và hỏi người dùng** (xem §8, F-DATA). Không tự chuyển sang bộ khác.

Giải nén bằng Python `zipfile` (không phụ thuộc 7-zip) vào `data/raw/`. Ghi SHA-256 và kích thước vào `data/raw/MANIFEST.json`.

### 2.4. Kiểm chứng định dạng thật (T-04, bắt buộc trước khi viết bộ chuyển đổi)
`scripts/inspect_xml.py` in: số XML, 10 tên đầu, thẻ gốc, thẻ con, thuộc tính `frame`, cấu trúc `target`, `box`, `attribute`, `ignored_region`. Ghi vào `docs/daily_log.md` **bằng chữ**: ảnh là JPG từng frame hay video, tên tệp ảnh, số frame mỗi chuỗi, FPS, độ phân giải, có nhãn test không. **Nếu khác mô tả trong §3 thì sửa §3 trong repo (`docs/SPEC_local.md`) rồi mới code.**

---

## 3. Hợp đồng dữ liệu

### 3.1. Cây thư mục
```
D:\gaplink\
├─ configs\ paths.yaml main.yaml tracker.yaml overfit.yaml ablation_{ctx,color,gap}.yaml
├─ data\ raw\  mot\{seq}.txt {seq}_ignore.txt  classes.json  splits.csv  stats.json
│        cache\ hist\{seq}.npy  pool\{train}\*.npy  val_events.pkl  test_events.pkl  *.sha256
├─ src\ io_mot.py events.py dataset.py features.py
│       model\ motion.py predictor.py context.py assign.py sinkhorn.py gaplink.py
│       baselines.py calib.py tracker.py inject.py eval_events.py metrics_tracker.py stats.py
├─ scripts\ check_env.py fetch_data.py inspect_xml.py convert_detrac.py stats.py group_locations.py
│           extract_hist.py viz_events.py make_fixed_events.py gen_pool.py run_val.py run_test.py
│           make_figures.py bootstrap_paired.py
├─ demo\ render.py overlays.py select_clips.py  out\*.mp4
├─ tests\  test_io.py test_splits.py test_events.py test_sinkhorn.py test_model.py test_inject.py
│          test_metrics.py test_tracker.py test_parity.py test_calib.py
├─ train.py   runs\   release\   docs\ GapLink-Lite_v1.md PREREG.md daily_log.md   report\
```

### 3.2. Định dạng nội bộ
- `data/mot/{seq}.txt`: `frame,id,x,y,w,h,conf,class_id,vis` (góc trên-trái, pixel, `frame` bắt đầu từ 1). `vis` điền 1 nếu nguồn không có.
- `{seq}_ignore.txt`: mỗi dòng `x,y,w,h` của vùng ignored. Box GT hoặc detection có **tâm** trong vùng ignored bị loại.
- `data/splits.csv`: `seq,source_split(train|test),location,split(train|val|test)`.
- `data/cache/hist/{seq}.npy`: `uint8 (R,128)`, hàng *k* ứng với dòng *k* của `{seq}.txt` đã sắp xếp theo `(frame, id)`. HSV 8×4×4, vùng giữa 60% của box, L1-normalize rồi nhân 255 làm tròn (giải nén chia cho tổng).

### 3.3. Sự kiện (event) — **[INHERIT v1 Phụ lục A]** cộng:
| Khóa mới | Shape | Ý nghĩa |
|---|---|---|
| `cand_kind` | (M,) int8 | 0 thật, 1 xe mới sinh, 2 dương tính giả, 3 distractor nhìn thấy |
| `side` | (N,) int8 | hướng cắt `partial` (0 không cắt, 1 trên, 2 dưới, 3 trái, 4 phải) |
| `mode` | str | `occl` / `drop` / `nomatch_exit` / `nomatch_miss` |
Tên `nomatch_fp` của v1 được đổi thành **`nomatch_miss`** (detector bỏ sót xe còn trong cảnh). Chữ "Ba loại" ở v1 §7.1 thực ra là **bốn** loại.

### 3.4. Kế hoạch ẩn `hide_plan_{seq}.json`
```json
{"seq":"MVI_x","variant":"full|partial","seed":123,"p_partial":0.4,
 "intervals":[{"gt_id":17,"a":385,"b":411,"g":27,"gap_bin":1,"is_hard":true,"side":3,"crop_frac":0.8}]}
```
`[a, b]` là các frame bị ẩn (gồm cả hai đầu). Định nghĩa: `t_lost = a−1`, `t_conf = b+1`, `g = t_conf − t_lost = b − a + 2`.

### 3.5. Log tracker
- `runs/<run>/<seq>/<system>_tracks.csv`: `frame,track_id,x,y,w,h,state(active|lost_ghost),det_idx,gt_id`. `gt_id` chỉ ghi để chấm điểm/ render, **tracker không đọc**.
- `runs/<run>/<seq>/<system>_events.jsonl`: schema v1 §9 cộng `P_top3` cho mỗi ứng viên, `mu_cal`, `sigma_cal`, `elip90` (tâm, hai bán trục, góc), `decision`, `correct` (true/false/null).
- `runs/<run>/<seq>/dets.csv`: detection sau tiêm (`frame,det_idx,x,y,w,h,gt_id,cand_kind`).

### 3.6. Hợp đồng chạy
| Lệnh | Việc |
|---|---|
| `python scripts\convert_detrac.py` | XML → MOT + ignore |
| `python scripts\group_locations.py` | ảnh nền + gom cụm + `splits.csv` |
| `python scripts\extract_hist.py --workers 6` | histogram màu (chạy nền) |
| `python scripts\gen_pool.py --split train --n 300000 --workers 6` | pool train |
| `python scripts\make_fixed_events.py` | `val_events.pkl`, `test_events.pkl` |
| `python train.py --config configs\main.yaml [--resume]` | huấn luyện |
| `python scripts\run_val.py --release-candidate runs\main\best.pt` | đánh giá val (event + tracker) |
| `python scripts\run_test.py --ckpt release\final.pt` | **test một lần** (ngày 7) |
| `python demo\render.py --run runs\test_final --out demo\out` | render video |

---

## 4. Đặc tả mô-đun (làm rõ những chỗ v1 còn mơ hồ)

### 4.1. `src/io_mot.py`
```python
@dataclass
class SeqData:
    name: str; W: int; H: int; n_frames: int; img_dir: Path
    rows: np.ndarray            # (R,9) sắp theo (frame,id)
    by_frame: dict[int, slice]  # frame -> slice của rows
    tracks: dict[int, dict]     # id -> {"frames": int32[], "boxes": float32[,4] (cx,cy,w,h), "row_idx": int32[]}
    ignore: np.ndarray          # (K,4)
def load_seq(name: str) -> SeqData
def center_in_ignored(boxes_xywh, ignore) -> np.ndarray[bool]
```
Test: số dòng và ID khớp XML gốc; không có lớp người; không box nào có tâm trong vùng ignored.

### 4.2. `src/events.py`
```python
@dataclass
class EventCfg: ...           # mặc định theo v1 §7 + D-06; đọc từ configs/main.yaml
def partial_crop(box_xywh, side: int, frac: float) -> np.ndarray      # DÙNG CHUNG bởi events.py và inject.py
def assemble_event(hist_boxes, hist_len, gap, cand_boxes, cand_hist, ctx_lost, ctx_conf, ctx_mask, W, H) -> dict
    # HÀM THUẦN: chỉ nhận dữ liệu QUAN SÁT ĐƯỢC. Dùng cho sampler VÀ tracker. Không có nhãn.
def sample_event(idx: SeqIndex, rng, cfg, want_hard: bool | None = None) -> dict | None   # None = bị loại
def event_is_hard(cand_boxes, cand_label, label, factor=1.5) -> bool
```
Quy tắc cụ thể:
- `t_lost = a−1`, `t_conf = b+1`; mô hình **không** nhìn frame > `t_lost` cho lịch sử, và chỉ nhìn detection tại `t_conf` cho ứng viên.
- **occl:** chọn xe A và xe B có `IoU(A,B) > 0,3` tại frame `a`; gap `g` lấy theo phân bố 40/40/20 (nhóm gap 3–15, 16–45, 46–90). Ẩn A trong `[a, a+g−2]`. Hướng cắt `partial` quay về phía B.
- **drop:** A ngẫu nhiên, `g` theo phân bố; hướng cắt ngẫu nhiên.
- **nomatch_exit:** xe A có frame cuối cách biên ảnh ≤ `0,5·h`; `t_lost` = frame cuối, `t_conf = t_lost + g`; không có ứng viên thật.
- **nomatch_miss:** như `drop` nhưng xóa ứng viên thật khỏi tập ứng viên (A vẫn trong cảnh).
- Ràng buộc: `hist_len ≥ 8`; xe A sống ít nhất `g + 3` frame sau `t_lost` (trừ `nomatch_exit`).
- Xe phụ: 0–3 xe còn sống tại `t_conf` và có tâm nằm trong `4·h_A` quanh vị trí dự kiến của A; gap riêng. Thêm 0–2 xe mới sinh, 0–2 dương tính giả, 0–2 distractor nhìn thấy. Giới hạn `N ≤ 6`, `M ≤ 10` (cắt theo gần nhất). Sau cùng **xáo trộn**, rồi **tính lại** `label`, `cand_label`.
- Nhiễu: ứng viên `N(0, 0,05·size)` ở tâm, kích thước ±10%; lịch sử `σ = 0,03·size`; bỏ ngẫu nhiên 5% ứng viên thật.
- `is_hard`: tồn tại ứng viên gây nhiễu có tâm cách tâm ứng viên đúng `< 1,5·max(w,h)` của xe đúng (chỉ hình học).
- Ngữ cảnh mỗi xe lân cận (6 chiều, **chốt**): `[dx/h, dy/h, log(w_n/w_last), log(h_n/h_last), IoU với box cuối, cờ(0=t_lost,1=t_conf)]`; `h`, `w_last` lấy từ box cuối của track. Không dùng vận tốc lân cận.
- Test bắt buộc: `test_events.py` (hist không chứa frame > `t_lost`; nhãn khớp ID gốc sau khi xáo; phân bố gap ±3%; `N≤6,M≤10`), `test_parity.py` (dựng lại sự kiện từ nguyên liệu quan sát bằng `assemble_event` cho ra mảng **y hệt** `sample_event` khi bỏ nhãn).

### 4.3. Pool và DataLoader (`scripts/gen_pool.py`, `src/dataset.py`)
- `multiprocessing.Pool(workers)`; mỗi worker nhận (seq-list, quota, seed) và ghi shard `.npy` cố định chiều (pad N→6, M→10) kèm `n_valid`, `m_valid`.
- Gộp thành `memmap` theo từng khóa; ghi `pool_meta.json` (số sự kiện, hash, số hard/easy).
- `PoolDataset` lấy mẫu `hard_ratio(step)` tuyến tính 0,3→0,6 trong 40.000 bước. `collate` chuyển thẳng sang tensor; không tăng cường ngoài nhiễu box nhẹ trên GPU.

### 4.4. Mô hình (`src/model/*`)
**[INHERIT v1 §8]** với các điểm chốt:
- Mục tiêu của đầu dự đoán: `((cx−cx_last)/h_last, (cy−cy_last)/h_last, log(w/w_last), log(h/h_last))`. Elip pixel = nhân ngược `h_last`.
- Cờ cấu hình: `use_ctx`, `use_color`, `gap_aware` (ablation).
- `forward(batch) -> {"logP": (B,N+1,M+1), "mu": (B,N,4), "logvar": (B,N,4), "s": (B,N,M)}`.
- Khi `partial` và `use_color`: đặt cosine = 0 và cờ riêng với xác suất 0,2 (chỉ lúc train).
- Test: `sum(p.numel()) < 1_000_000`.

**Sinkhorn (viết đúng như sau, log-domain, fp32, NEG hữu hạn):**
```python
def log_sinkhorn(S, z_t, z_c, tm, cm, iters=20, NEG=-1e4):
    # S (B,N,M); z_t (B,N); z_c (B,M); tm (B,N) bool; cm (B,M) bool
    B, N, M = S.shape
    A = S.new_full((B, N+1, M+1), NEG)
    A[:, :N, :M] = S; A[:, :N, M] = z_t; A[:, N, :M] = z_c; A[:, N, M] = 0.0
    vr = torch.cat([tm, tm.new_ones(B, 1)], 1); vc = torch.cat([cm, cm.new_ones(B, 1)], 1)
    A = A.masked_fill(~(vr[:, :, None] & vc[:, None, :]), NEG)
    n = tm.sum(1).float(); m = cm.sum(1).float()
    log_r = torch.cat([torch.zeros_like(S[:, :, 0]), m.clamp(min=1).log()[:, None]], 1).masked_fill(~vr, NEG)
    log_c = torch.cat([torch.zeros_like(S[:, 0, :]), n.clamp(min=1).log()[:, None]], 1).masked_fill(~vc, NEG)
    u = A.new_zeros(B, N+1); v = A.new_zeros(B, M+1)
    for _ in range(iters):
        u = log_r - torch.logsumexp(A + v[:, None, :], 2)
        v = log_c - torch.logsumexp(A + u[:, :, None], 1)
    return A + u[:, :, None] + v[:, None, :]       # exp = P; hàng thật tổng ≈ 1, cột thật tổng ≈ 1
```
Hàng dustbin có khối lượng `m`, cột dustbin có khối lượng `n`. Gọi trong `torch.autocast(enabled=False)`.

**Loss:** `L = L_assign + 0,5·L_nll` **[INHERIT v1 §8.3]**; trung bình theo số phần tử hợp lệ, không theo kích thước pad.

**Năm unit test Sinkhorn/model [INHERIT v1 Ngày 3]** cộng `test_param_count`.

### 4.5. Baseline (`src/baselines.py`)
Giao diện chung: `Associator.decide(event: dict) -> np.ndarray[int] (N,)` (chỉ số ứng viên hoặc −1) và `.cost(event) -> (N,M)`.
- **C-2:** vận tốc = trung bình 5 bước cuối; vị trí dự kiến `box_last + v·g`; chi phí = khoảng cách tâm/`h` + 0,5·|log tỉ lệ kích thước|; Hungarian; chấp nhận nếu chi phí ≤ `θ`.
- **C-3:** Kalman 8 chiều `(cx,cy,w,h,vcx,vcy,vw,vh)`, mô hình vận tốc hằng, `Q, R` lấy `σ_a = 0,5 px/frame²`, `σ_obs = 0,03·size` làm điểm xuất phát; dự đoán `g` bước không cập nhật; chi phí Mahalanobis² trên tâm; cổng `χ²(2,0,99)=9,21` **và** ngưỡng chấp nhận `θ`. Hungarian.
- **C-4:** C-3 + `λ_s·|log tỉ lệ kích thước| + λ_c·(1−cos màu)`; quét `λ_s, λ_c ∈ {0, 0,5, 1, 2, 4}` trên val.
- **Quét ngưỡng công bằng:** cả GapLink (`τ`), C-2, C-3, C-4 dùng **cùng lưới tiêu chí** (§1.2) và cùng tập ứng viên.

### 4.6. `src/eval_events.py`
```python
def evaluate(associator, events, *, by="all|gap|hard") -> dict
# top1  = tỉ lệ track (N hợp lệ) có quyết định == label (dustbin = −1)
# wrong_link = tỉ lệ track được gán vào ứng viên sai; dustbin_P/R/F1; nll (chỉ GapLink); coverage90 (xem calib)
def bootstrap_cluster(values_by_seq: dict[str, np.ndarray], n=2000, seed=0) -> (lo, hi)
def bootstrap_paired(a_by_seq, b_by_seq, n=2000, seed=0) -> (delta, lo, hi)   # lấy lại CÙNG tập chuỗi cho cả hai
```
**Hàm bootstrap viết một nơi, dùng lại mọi chỗ.**

### 4.7. `src/calib.py`
`σ' = σ·exp(a + b·log(1+g))`; khớp `(a,b)` bằng `scipy.optimize.minimize(method="L-BFGS-B")` trên NLL của tâm (2 chiều, đường chéo) ở **nửa A val** (chia theo chuỗi, cố định bằng hash tên chuỗi); đo coverage ở nửa B. Nếu coverage ngoài [85,95]% thì thêm `c·log²(1+g)` và khớp lại. Làm tương tự để báo coverage của elip Kalman (thô và sau hàm hiệu chuẩn). Test: Gaussian giả lập cho coverage ≈ 90% tại `d² ≤ 4,605`.

### 4.8. `src/inject.py` và `src/tracker.py`
**`make_hide_plan(seq: SeqData, rng, variant, p=0.8) -> dict`**: khoảng ẩn dài `g` theo 40/40/20; cách đầu quỹ đạo ≥ 8 frame, cách cuối ≥ 5; cho phép nhiều khoảng không chồng trên một xe; duyệt ngẫu nhiên và loại khoảng làm vượt **20% xe bị ẩn đồng thời**; seed cố định; ghi `hide_plan_{seq}.json`. Biến thể `partial` dùng **`partial_crop` chung** với `events.py`.

**`make_detections(seq, plan, rng) -> per-frame arrays`**: GT − khoảng ẩn − bỏ sót 5% + nhiễu (đã nêu §4.2) − vùng ignored; mỗi detection kèm `gt_id` (**ẩn khỏi tracker**).

**Tracker (mỗi frame t)**
1. Khớp IoU + Hungarian (ngưỡng IoU 0,3) giữa track đang hoạt động và detection, dùng box dự đoán vận tốc hằng.
2. Track không khớp > 1 frame → `lost_pool` (tối đa `a_max = 100` frame; quá hạn bị xóa).
3. Detection chưa khớp (conf ≥ ngưỡng) + `lost_pool` → `assemble_event` (nếu vượt N>6 hoặc M>10 thì chia theo thành phần liên thông trên cổng `χ²` như v1 §9).
4. `Associator.decide(event)` (GapLink: Hungarian trên P bỏ dustbin, chấp nhận khi `P_ij > τ`). Ứng viên không được gán → track mới.
5. Ghi `events.jsonl`, `tracks.csv`.
**Bộ đệm quan sát:** giữ `frame_dets[t]` cho 100 frame cuối để dựng ngữ cảnh tại `t_lost`.
**Hệ thống:** `C-1` (không gọi associator, mọi ứng viên là track mới), `C-2`, `C-3`, `C-4`, `GapLink`. Cùng `Tracker`, chỉ khác `Associator`.

### 4.9. `src/metrics_tracker.py` — định nghĩa chính xác
Với mỗi khoảng ẩn `(gt_id v, a, b)`:
- `orig = ID tracker gán cho v tại frame a−1` (bằng cách tra detection của v). Không có ⇒ **loại** khoảng này (đếm `n_excluded`).
- `r = b+1`. `new = ID tracker gán cho detection của v` tại frame đầu tiên trong `[r, r+4]` có detection của v (không có ⇒ `Lost`).
- **IRR:** `new == orig` và giữ nguyên tại `r+4`. **WLR:** `new` là `orig` của một xe khác (tại frame `a−1`). **FragRate:** `new` chưa từng tồn tại tại frame `a−1`. **Lost:** không có detection của v trong `[r, r+9]`.
- Báo theo nhóm gap, khó/dễ, biến thể full/partial; CI bootstrap theo chuỗi; **ΔIRR ghép cặp**. IDF1: dùng `motmetrics` nếu còn thời gian (cắt sớm, §8).
- Test: chuỗi dựng tay (đã biết đáp án) cho đúng IRR/WLR/Frag/Lost.

---

## 5. Kế hoạch từng ngày (mỗi ngày ≈ 5 giờ làm việc; việc chạy nền chồng lên việc viết mã)

> Mỗi việc có mã **T-xx**. "Xong khi" là kiểm tra chạy được bằng lệnh. **Không sang ngày kế khi cổng chưa đạt, trừ khi §8 cho phép.**

### NGÀY 1 — Môi trường, dữ liệu, chia tập, tiền đăng ký
| Việc | Chi tiết | Xong khi |
|---|---|---|
| T-01 (15') | Chép `GapLink-Lite.txt` thành `docs/GapLink-Lite_v1.md`; tạo cây thư mục §3.1; `git init`; `.gitignore` (venv, data, runs, `*.pt`, `*.mp4`, `*.npy`, `*.pkl`) | `git status` sạch |
| T-02 (25') | **Đợt duyệt mạng §2.2**, `scripts/check_env.py` | `CUDA available: True` |
| T-03 (10') | Khởi động `fetch_data.py` (nền) | log tải chạy |
| T-04 (30') | Trong lúc chờ: viết `inspect_xml.py`, `convert_detrac.py` dựa trên mô tả §2.4 và tạo **XML giả nhỏ** để unit test; khi XML thật về thì chạy `inspect_xml.py` và đối chiếu | Ghi định dạng thật vào `daily_log.md` |
| T-05 (40') | `convert_detrac.py` → `data/mot`, `*_ignore.txt`, `classes.json`; bộ lọc lớp người | `pytest tests/test_io.py` đạt |
| T-06 (30') | `scripts/stats.py` → `stats.json`, `report/fig_class_hist.png` (phân bố lớp, độ sống track P10/50/90, số xe/frame, số khung ≥ 10 xe) | Có hình, `stats.json` |
| T-07 (45') | `group_locations.py`: ảnh nền (trung vị 50 frame, 64×36 xám, chuẩn hóa), khoảng cách `1−Pearson`, `scipy.cluster.hierarchy.linkage` (average), cắt ngưỡng theo dendrogram; **ghép lưới ảnh nền theo cụm và xem bằng mắt**; kiểm chéo test↔train (chuỗi test có khoảng cách tới train/val < ngưỡng ⇒ **loại khỏi test**, ghi log); tách train/val theo địa điểm ≈ 80/20 | `splits.csv`; `pytest tests/test_splits.py` (không chuỗi/địa điểm ở hai tập) |
| T-08 (30') | `extract_hist.py` (nền, `--workers`), kiểm tra đồng bộ hàng với `{seq}.txt` | `hist/{seq}.npy` đủ, số hàng khớp |
| T-09 (30') | `src/io_mot.py` + `SeqIndex` | test §4.1 đạt |
| T-10 (30') | Viết `docs/PREREG.md` (H1–H3, P1, tiêu chí quét ngưỡng "IRR cao nhất với WLR ≤ 5%", quy tắc chọn clip §6.4, quy tắc chọn 10 ca lỗi, số lượng mục tiêu §1.3); `git tag prereg-v1` | Tag tồn tại |
| T-11 (15') | Log + mục báo cáo "Dữ liệu" | commit |

**Cổng G1:** [ ] có CUDA; [ ] `data/mot` ≥ 40 chuỗi (mục tiêu 100); [ ] `splits.csv` hợp lệ theo test tự động; [ ] `stats.json` + biểu đồ lớp; [ ] đã xem ảnh nền ghép lưới; [ ] `PREREG.md` đã gắn tag.
**Nếu trễ:** bỏ gom tự động, dùng heuristic theo tên chuỗi + xem tay (15'); histogram có thể tiếp tục chạy nền sang ngày 2. Tuyệt đối không bỏ T-04, T-10.

### NGÀY 2 — Bộ sinh sự kiện, tập cố định, baseline, đánh giá, renderer v0
| Việc | Chi tiết | Xong khi |
|---|---|---|
| T-20 (90') | `events.py`: `partial_crop`, `assemble_event`, `sample_event`, `event_is_hard` (§4.2) | `tests/test_events.py`, `tests/test_parity.py` đạt |
| T-21 (25') | `scripts/viz_events.py`: vẽ **20 sự kiện** (frame `t_lost` + `t_conf`, đáp án viền dày) vào `report/events_vis/`; **xem bằng mắt** | 20 ảnh hợp lý |
| T-22 (35') | `gen_pool.py` (train) + `make_fixed_events.py` (val/test, seed cố định, hạn ngạch theo chuỗi); đo **sự kiện/giây thật**; đếm hard | `pool_meta.json`, `val_events.pkl` (≥ 6.000, khó ≥ 300), `test_events.pkl` (≥ 12.000, khó ≥ 600), hash |
| T-23 (55') | `baselines.py`: C-2, C-3 (+ quét ngưỡng trên val) | `runs/baselines_val.json` |
| T-24 (50') | `eval_events.py` + bootstrap (§4.6) | Số sàn C-2/C-3 theo gap, khó, toàn bộ, kèm CI |
| T-25 (35') | `demo/render.py` v0 + `overlays.py`: đọc `data/mot` + ảnh, vẽ box có màu theo ID và **hộp che ảo** lên một khoảng ẩn mẫu, ghi MP4 H.264 (`imageio-ffmpeg`) | `demo/out/v0_check.mp4` mở được, trích 3 frame kiểm bằng mắt |
| T-26 (10') | Log + mục "Bộ sinh sự kiện" | commit |
**Cổng G2:** [ ] 300.000 sự kiện train (≥ 150.000); [ ] val ≥ 6.000 (khó ≥ 300); [ ] test ≥ 12.000 (khó ≥ 600) **chỉ ghi hash, không đánh giá**; [ ] **số sàn C-2/C-3 trên val**; [ ] video v0 xem được.
**Nếu tập khó < ngưỡng:** tăng `n` và hạ `1,5` xuống `1,25`×`max(w,h)` (ghi trong log, làm trước khi thấy kết quả mô hình).

### NGÀY 3 — Mô hình lõi, Sinkhorn, kiểm thử, overfit, khởi chạy huấn luyện chính
| Việc | Chi tiết | Xong khi |
|---|---|---|
| T-30 (30') | `features.py`: chuẩn hóa (v1 §8.1) + in phân phối (min/P1/P50/P99/max) từng đặc trưng; **cắt/clip** các đặc trưng chia cho `h` | Báo cáo phân phối, không NaN/Inf |
| T-31 (40') | `motion.py` (MotionEncoder GRU 2 lớp, mask độ dài thật), `predictor.py` | forward/backward chạy |
| T-32 (40') | `context.py` (DeepSets), đặc trưng màu trong `assign.py` (quy tắc bỏ màu) | `use_ctx`, `use_color` bật/tắt được |
| T-33 (35') | `assign.py`: điểm cặp `s_ij`, dustbin theo track/ứng viên (dùng vị trí) | shape `(B,N,M)` đúng |
| T-34 (45') | `sinkhorn.py` (§4.4) + loss | — |
| T-35 (35') | 5 unit test + `test_param_count` | `pytest tests/test_sinkhorn.py tests/test_model.py` đạt |
| T-36 (20') | `train.py` (AdamW, cosine, warmup 500, AMP, clip 1,0, checkpoint mỗi 2.000 bước, `--resume`, đánh giá val mỗi 2.000 bước bằng `eval_events.evaluate`); overfit 64 sự kiện (`overfit.yaml`) | **Overfit ≥ 99% accuracy** |
| T-37 (30') | Đo bước/giây; **chốt số bước**: `steps = clamp(5400·steps_per_sec, 20000, 60000)`; **khởi chạy `main.yaml` ở nền** | `runs/main/train.log` có loss giảm |
**Cổng G3 (chặn cứng):** 5/5 test đạt; overfit ≥ 99%; huấn luyện chính đã chạy. **Không đạt ⇒ dừng.** Thứ tự gỡ lỗi: vẽ lại 20 sự kiện → mask padding → tổng hàng/cột của P → phân phối đặc trưng → cân bằng dustbin → lr.

### NGÀY 4 — Đánh giá val, hiệu chuẩn, C-4, khởi chạy ablation, lõi tracker
| Việc | Chi tiết | Xong khi |
|---|---|---|
| T-40 (40') | Khi `main` xong: đánh giá val GapLink so với C-2/C-3 (`runs/d4_val/`) | **Cổng G4** |
| T-41 (45') | `calib.py` + coverage (GapLink, Kalman thô, Kalman sau hiệu chuẩn) theo 3 nhóm gap | `calib.json`; `test_calib.py` |
| T-42 (30') | C-4 + quét `λ_s, λ_c` trên val | số sàn C-4 |
| T-43 (15') | Khởi chạy 3 ablation tuần tự ở nền (`ablation_ctx/color/gap.yaml`): nếu bước/giây ≥ 8 thì đủ số bước, ngược lại **12.000 bước** cho cả 3 và 1 lần chạy đối chứng cùng ngân sách của mô hình đầy đủ | log chạy |
| T-44 (100') | `tracker.py`: bước 1–5 (§4.8), bộ đệm quan sát, `assemble_event` dùng chung | `tests/test_tracker.py` (2 xe, 1 xe ẩn rồi hiện: C-2 nối đúng) |
| T-45 (35') | Log + mục "Mô hình", "Hiệu chuẩn" | commit |
**Cổng G4 (H1 trên val):** Top1(hard) − max(C-2,C-3,C-4) ≥ +3 điểm. **Không đạt:** gỡ lỗi **tối đa 2 giờ** (danh sách 9 mục v1 §Ngày 4), rồi **tiếp tục**; đây là kết quả thật. Không dùng test để chẩn đoán.

### NGÀY 5 — Giao thức tiêm, chỉ số tracker, tích hợp, renderer v1
| Việc | Chi tiết | Xong khi |
|---|---|---|
| T-50 (55') | `inject.py`: `make_hide_plan`, `make_detections`, hai biến thể | `tests/test_inject.py` (mật độ ≤ 20%, cách đầu/cuối, không chồng, seed tái lập) |
| T-51 (50') | `metrics_tracker.py` (§4.9) | `tests/test_metrics.py` đạt |
| T-52 (40') | Tích hợp 5 hệ thống vào `Tracker`; `scripts/run_val.py` | chạy hết chuỗi val không crash |
| T-53 (30') | Chạy val biến thể **full**, 5 hệ thống | `runs/d5_val/` |
| T-54 (75') | `demo/render.py` v1: **side-by-side C-3 | GapLink** từ log val (§6) | `demo/out/v1_val.mp4` |
| T-55 (30') | Log + mục "Tracker và giao thức" | commit |
**Cổng G5:** [ ] H2 (coverage 85–95% ở cả 3 nhóm gap, nửa B); [ ] tracker chạy hết val; [ ] **P1: C-1 IRR < 5%** (nếu cao ⇒ lỗi giao thức/metric, sửa trước khi đi tiếp); [ ] test inject/metrics đạt; [ ] video v1 có đủ phần tử §6.2.

### NGÀY 6 — Quét ngưỡng, so sánh công bằng, đóng băng, diễn tập demo
| Việc | Chi tiết | Xong khi |
|---|---|---|
| T-60 (15') | Nhắc lại tiêu chí trong `PREREG.md` (không sửa) | — |
| T-61 (60') | Quét `τ ∈ {0,3…0,8}` (GapLink) và ngưỡng `θ`/`λ` tương ứng (C-2/C-3/C-4) trên val, cùng tiêu chí "IRR cao nhất với WLR ≤ 5%"; ghi vào `configs/tracker.yaml` | bảng quét |
| T-62 (40') | Chạy val đầy đủ: full + partial, 5 hệ thống, IRR/WLR/Frag/Lost theo gap | `runs/d6_val/` |
| T-63 (15') | **Cổng G6** (H3 trên val) | — |
| T-64 (45') | **Đóng băng** (§5.1) và chạy lại val để xác nhận tái lập | `release/final.sha256`; số val trùng |
| T-65 (25') | `scripts/run_test.py` (một lệnh, đọc `release/*`); thử trên **1 chuỗi val** | chạy ra đủ log |
| T-66 (60') | Render 4 clip **từ val** bằng `select_clips.py` + `render.py` (tập dượt) | 4 mp4, kiểm 3 frame/clip |
| T-67 (20') | `make_figures.py` khung (đọc `metrics.json`, vẽ bảng/biểu đồ) | hình mẫu từ val |
**5.1. Đóng băng (đúng thứ tự):**
```powershell
Copy-Item runs\main\best.pt release\final.pt
Get-FileHash release\final.pt -Algorithm SHA256 | Out-File release\final.sha256
Copy-Item configs\main.yaml,configs\tracker.yaml release\; Copy-Item runs\main\calib.json release\
git add release configs requirements.txt docs; git commit -m "Dong bang v1.0"; git tag v1.0-frozen
```
**Cổng G6:** IRR(GapLink) > C-2 và > C-3 ở ≥ 2/3 nhóm gap và ΔWLR ≤ +0,5. **Không đạt ⇒ vẫn đóng băng.** Không huấn luyện lại, không nhìn test.

### NGÀY 7 — Test một lần, thống kê, phân tích, render video cuối
| Việc | Chi tiết | Xong khi |
|---|---|---|
| T-70 (15') | Kiểm tra cuối: hash `final.pt` khớp, `run_test.py` chạy được, val tái lập | ✔ |
| T-71 (60') | **`python scripts\run_test.py --ckpt release\final.pt`**: event-level + tracker (full + partial) + 5 hệ thống, **đúng một lần** | `runs/test_final/` |
| T-72 (30') | `bootstrap_paired.py`: ΔTop1, ΔIRR, ΔWLR (GapLink − C-2/C-3/C-4), CI 95% theo chuỗi, 2.000 lần | `runs/test_final/paired.json` |
| T-73 (45') | `make_figures.py`: bảng kết quả, Top-1 và IRR theo gap, IRR–WLR, coverage trước/sau, ablation (val), đường loss | `report/figures/*.png` |
| T-74 (40') | Phân tích **10 ca lỗi** theo quy tắc chốt trước (10 ca P(đáp án đúng) thấp nhất, chia đều theo gap): ảnh trước/sau, P, nhóm lỗi, nguyên nhân | `report/failure_cases/` |
| T-75 (80') | Render **video cuối từ log test** (§6) | `demo/out/*.mp4` |
| T-76 (10') | Log | commit |
**Nếu phải chạy lại test (crash, lỗi mã metric):** ghi lý do vào nhật ký, giữ **cả hai** kết quả, không đổi mô hình/ngưỡng/dữ liệu/tiêu chí.

### NGÀY 8 — Ghép demo, báo cáo, đóng gói, dự phòng
| Việc | Chi tiết | Xong khi |
|---|---|---|
| T-80 (60') | `demo/assemble.py`: thẻ tiêu đề, phụ đề, nối thành `demo_full.mp4` (≤ 4 phút) + 4 clip lẻ; kiểm bằng `cv2` (kích thước, FPS, số frame) và **xem 3 frame/clip** | video đạt §6.5 |
| T-81 (100') | Báo cáo `report/report.md` (+ định dạng cuối theo mẫu người dùng nếu có): v1 §Cấu trúc báo cáo; **bảng khai báo đóng góp**; **hạn chế và mối đe dọa tính hợp lệ** | đủ mục, mọi số khớp `metrics.json` |
| T-82 (50') | `README.md`: cài đặt, tải dữ liệu, một lệnh train, một lệnh eval, một lệnh render, kết quả chính; `requirements.txt`; kiểm tra `git log` thể hiện lịch sử phát triển | chạy theo README trên bản sao sạch (smoke: `pytest -q`) |
| T-83 (60') | **Dự phòng**: sửa lỗi, render lại nếu cần | — |
| T-84 (30') | Checklist cuối §9, gói `release/`, `git tag final-v2` | ✔ |

### 5.2. Thang cắt giảm (kích hoạt theo điểm cụ thể)
| Điểm | Nếu chưa xong | Cắt (theo thứ tự) |
|---|---|---|
| Giữa N1 | Chưa có nhãn/ảnh | Hỏi người dùng (F-DATA). Không bỏ T-04, T-10 |
| Cuối N2 | Chưa có C-3 | Giữ C-2, C-3 đơn giản (Kalman Q/R mặc định), ghi hạn chế |
| Giữa N3 | Sinkhorn chưa qua test | **Không cắt.** Dừng và sửa; mượn thời gian của T-42/T-43 |
| Giữa N4 | Huấn luyện chậm | Bỏ ablation (iii), giảm bước ablation còn 12.000 |
| Cuối N4 | — | Bỏ ablation (i) |
| Giữa N5 | Trễ | Bỏ biến thể **partial**, bỏ IDF1; renderer chỉ giữ clip tracker và Ghost Replay |
| Cuối N5 | — | Bỏ C-4 **sau cùng** (khi bỏ, ghi rõ: đóng góp học và đặc trưng chưa tách được) |
| N6 | — | Lưới quét τ ∈ {0,4; 0,5; 0,6} |
| N8 | — | Chỉ 3 clip: Ghost Replay, video tracker, Failure Gallery |
**Không bao giờ cắt:** giao thức tiêm (bản full), C-2, C-3, bootstrap ghép cặp, hiệu chuẩn theo gap, phân tích lỗi, khai báo đóng góp, kỷ luật test một lần, **video demo so sánh song song**.

---

## 6. Đặc tả demo video (yêu cầu bắt buộc)

### 6.1. Nguyên tắc
- Bộ render **chỉ đọc**: ảnh frame, `data/mot` (để vẽ hộp che ảo), `hide_plan`, `*_tracks.csv`, `*_events.jsonl`. Không gọi mô hình. Cho phép render lại bất kỳ lúc nào.
- Mọi video ghi nhãn **"Khoảng ẩn do tiêm (mô phỏng)"** và câu **"Clip chọn theo quy tắc cố định, không theo chất lượng"**.

### 6.2. Bố cục khung 1920×720 (H.264, yuv420p, 25 fps)
| Vùng | Nội dung |
|---|---|
| Header 1920×60 | Tên chuỗi, frame, nhãn nhóm gap, nhãn "SIMULATED OCCLUSION" |
| Panel trái 960×540 | **C-3 (Kalman + Mahalanobis)** |
| Panel phải 960×540 | **GapLink-Lite (đề tài)** |
| Footer 1920×120 | Bộ đếm mỗi bên: `Re-links ✓ | Wrong ✗ | New IDs | gap hiện tại`; một dòng chú thích |

**Phần tử vẽ trong mỗi panel**
1. Box theo ID, màu từ bảng 20 màu cố định, nhãn `#id`.
2. **Hộp che ảo:** hình chữ nhật xám đậm có vạch chéo (alpha 0,85) phủ lên xe bị ẩn (box GT + 6 px), có chữ "ẨN"; bám theo xe trong suốt khoảng ẩn. Người xem **thấy xe biến mất**, còn tracker thì không có detection.
3. **Hộp ma** (nét đứt) tại `μ` dự đoán của track mất dấu + **elip 90%** bán trong suốt, to dần theo gap. C-3: từ Kalman. GapLink: `mu_cal`, `sigma_cal`.
4. Khi xe hiện lại: ứng viên có viền vàng và nhãn `P=0,91` (top-3); xác suất dustbin hiện ở footer.
5. Kết quả (giữ 1,5 giây): viền xanh + "ID RECOVERED #17" (đúng); viền đỏ + "ID SWITCH #17→#23" (liên kết nhầm); viền cam + "NEW ID" (phân mảnh).

### 6.3. Bốn clip + video tổng
| Clip | Nội dung | Thời lượng |
|---|---|---|
| `ghost_replay.mp4` | 3 khoảng ẩn (mỗi nhóm gap 1), **chậm 0,25×** quanh lúc hiện lại (nhân bản frame ×4) | 60–90 s |
| `no_match.mp4` | 2 ca xe rời cảnh: xác suất dồn vào dustbin | 30–45 s |
| `tracker_full.mp4` | Một chuỗi test đầy đủ 45–60 s, bộ đếm chạy | 45–60 s |
| `failure_gallery.mp4` | 3 ca lỗi trong 10 ca đã phân tích, mỗi ca một dòng nguyên nhân | 30–45 s |
| `demo_full.mp4` | Thẻ tiêu đề → 4 clip → thẻ kết quả (bảng IRR/WLR test) ≤ 4 phút | ≤ 4 phút |

### 6.4. Quy tắc chọn clip (ghi vào `PREREG.md` ngày 1)
Chuỗi/khoảng ẩn chọn **xác định**, không theo kết quả: `ghost_replay` = từ tập khoảng ẩn **khó** của test, lấy 3 khoảng bằng `np.random.default_rng(777).choice` (không trùng chuỗi nếu có thể); `no_match` = 2 sự kiện `nomatch_exit` đầu tiên theo thứ tự tên chuỗi; `tracker_full` = chuỗi test có **nhiều khoảng ẩn nhất** (hòa thì theo tên); `failure_gallery` = 3 ca trong 10 ca lỗi đầu tiên (đã chia theo gap). Nếu ở các clip này GapLink **thua**, vẫn giữ.

### 6.5. Kiểm tra video (tự động + người)
`demo/check_video.py`: mở bằng `cv2.VideoCapture`; kiểm kích thước 1920×720, FPS 25, số frame > 0, thời lượng đúng khoảng; trích frame ở 10/50/90% lưu `demo/out/check/*.png` rồi **agent xem ảnh** xác nhận có hộp che, hộp ma, elip, bộ đếm. Người dùng xem `demo_full.mp4` ở bước nghiệm thu.

---

## 7. Báo cáo và gói nộp
- **Báo cáo** (6–7 trang): v1 §"Cấu trúc báo cáo" + bảng khai báo đóng góp (v1 §5) + mục "Hạn chế và mối đe dọa tính hợp lệ" (GT có nhiễu không phải detector thật; camera cố định; gap nhân tạo; histogram màu từ box đầy đủ; không tracker-in-the-loop; ablation một seed; gom địa điểm có thể chưa hoàn hảo; **CI phụ thuộc số chuỗi chứ không phải số sự kiện**). Mọi con số lấy từ `metrics.json`/`paired.json`, không gõ tay. Nếu người dùng cung cấp mẫu LaTeX/Word thì dùng mẫu đó; mặc định Markdown.
- **Gói nộp:** repo Git (có lịch sử), `release/{final.pt, final.sha256, configs, calib.json}`, `demo/out/*.mp4`, `report/`, README, `requirements.txt`, `docs/daily_log.md`.

---

## 8. Quy tắc quyết định khi sự cố

| Mã | Tình huống | Hành động |
|---|---|---|
| F-DATA | Không tải được UA-DETRAC trong 90' | Dừng, hỏi người dùng. Phương án B (cần duyệt): **VisDrone-MOT**, chỉ lớp xe 4–10 (car, van, truck, bus, motor…), bỏ gom địa điểm theo ảnh nền, chia theo chuỗi; **camera di chuyển** nên bỏ vị trí tuyệt đối khỏi đặc trưng và ghi vào hạn chế. **KITTI Tracking không phải phương án thay thế "kế hoạch không đổi"** (camera di chuyển theo xe, chỉ ~21 chuỗi) và chỉ dùng như phương án sau cùng với cảnh báo rõ. |
| F-LABEL | Chỉ có nhãn, không có ảnh | Bỏ đặc trưng màu và nền ảnh; demo vẽ trên nền trắng (vẫn đủ yêu cầu video nhưng kém trực quan; hỏi người dùng) |
| F-TESTLABEL | Không có nhãn test công khai | Tách 60 chuỗi train theo địa điểm 60/20/20 |
| F-G3 | G3 không đạt | Dừng, gỡ lỗi (§5 N3). Mượn thời gian T-42/T-43, sau đó ablation |
| F-G4 | G4 không đạt | Gỡ lỗi ≤ 2 giờ, **tiếp tục** |
| F-P1 | C-1 IRR ≥ 5% | Sửa giao thức/metric trước G5 (lỗi hay gặp: IoU stage 1 đã nối lại track; `a_max` quá nhỏ; tra `orig` sai) |
| F-REPRO | Val không tái lập sau đóng băng | Tìm yếu tố ngẫu nhiên chưa cố định (seed worker, `cudnn`, thứ tự dict) **trước** test |
| F-CRASH | `run_test.py` crash | Sửa lỗi kỹ thuật, chạy lại, giữ cả hai log |
| F-OOM | GPU hết bộ nhớ | Giảm batch 256 → 128, AMP bật, Sinkhorn fp32 chỉ ở khối cuối |
| F-HOT | Máy nóng/ngắt | Checkpoint 2.000 bước, `--resume` |

## 9. Danh sách kiểm tra cuối (agent tự tick trước khi báo xong)
**Dữ liệu và quy trình:** [ ] chia theo địa điểm, test tự động; [ ] biểu đồ lớp (không có người); [ ] test chạy một lần (nếu lặp có lý do); [ ] mọi ngưỡng chỉnh trên val; [ ] `prereg-v1` có trước mọi kết quả.
**Mô hình:** [ ] khởi tạo ngẫu nhiên, không tải trọng số; [ ] `final.pt`, `final.sha256`, tag; [ ] test Sinkhorn/inject/metrics/parity đạt; [ ] < 1M tham số.
**Kết quả:** [ ] bảng event (Top-1, wrong-link, dustbin P/R/F1, NLL, coverage; theo gap, khó; CI); [ ] bảng tracker (IRR, WLR, Frag, Lost theo gap, full/partial; ΔIRR ghép cặp); [ ] C-1/C-2/C-3/C-4 trên cùng ứng viên; [ ] số khoảng ẩn và tập khó đạt §1.3 (hoặc ghi số thật); [ ] 10 ca lỗi; [ ] ablation val có ghi chú một seed; [ ] nêu rõ H1–H3 đạt hay không.
**Sản phẩm:** [ ] `demo_full.mp4` + 4 clip đạt §6.5; [ ] README chạy được; [ ] báo cáo có bảng khai báo đóng góp và hạn chế.
