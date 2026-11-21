# GapLink-Lite v1.1 — Kết quả đóng băng

- Checkpoint: `release/final.pt`, SHA256 `844fc391deafe951b429e3e4f2b9f1983647ce7ff1dbfa048eadb6f4c0741bfd` (327.979 tham số)
- Test events: `data/cache/test_events.pkl`, SHA256 `fca7fa99…8351` (12.000 sự kiện, 40 chuỗi, 15 địa điểm)
- `tau = 0,25`, `a_max = 95` (`release/tracker.yaml`)
- Nguồn duy nhất cho mọi con số: `release/results_v1.1/*.json` (bản sao có hash của `runs/test_final/`)
- Kiểm tra: `D:\gaplink\.venv\Scripts\python.exe D:\gaplink\scripts\freeze_results.py` → `[OK]`
- Sai lệch quy trình: xem `docs/DEVIATIONS.md` (đặc biệt D1: đây là lần chạy test thứ hai).
- CI 95%: bootstrap ghép cặp theo **15 địa điểm camera**, 10.000 lần (`release/results_v1.1/location_bootstrap.json`, D5). Dự đoán tính lại từ `final.pt` trùng khít số đóng băng (`reproduction_check` = all True).

## Giả thuyết PREREG (đúng câu chữ PREREG v1)
| | Tiêu chí | Kết quả (seed 1 = final.pt) | Nguồn (`test_results.json`) | Trạng thái |
|---|---|---|---|---|
| H1 | Top-1 hard ≥ max(C-2,C-3,C-4) + 3,0 pp | 93,18% vs C-4 85,85% → Δ +7,33 pp, CI95 [6,13; 8,77] (cận dưới > 3 pp) | `location_bootstrap.event_level.H1_prereg` | **ĐẠT** |
| H2 | Coverage 90% sau hiệu chuẩn ∈ [85,95] ở 3 nhóm gap, val nửa B | val: 87,71 / 87,64 / **95,14**; test: 87,87 / 90,26 / **95,86** | `val_summary.calib_val.after`, `calib_test.after` | **KHÔNG ĐẠT** (nhóm dài, sát biên) |
| H3 | IRR > C-2 và C-3 ở ≥ 2/3 nhóm gap, ΔWLR ≤ +0,5 pp | full: 3/3 nhóm; WLR 0,21% (C-2 4,32%, C-3 10,25%) | `tracker_level.full` | **ĐẠT** |
| P1 | C-1 IRR < 5% | 0,00% | `tracker_level.full.C-1` | **ĐẠT** |

Phân tích hậu kiểm (không thay H2): đầu ra **chưa** hiệu chuẩn trên test: 86,42 / 85,46 / 88,02% (`calib_test.before`).

## Event-level (12.000 sự kiện test)
| Phương pháp | Top-1 | WLR | Top-1 gap dài | Dustbin F1 |
|---|---|---|---|---|
| C-2 | 76,58% | 21,14% | 49,10% | 0,448 |
| C-3 | 80,29% | 19,24% | 54,39% | 0,523 |
| C-4 | 83,69% | 15,88% | 64,24% | 0,512 |
| **GapLink-Lite (seed 1)** | **93,41%** (CI [91,97; 94,90]) | **3,01%** | **79,58%** | **0,840** |
| GapLink-Lite 5 seed¹ | 93,64% ± 0,14% | 2,98% ± 0,11% | 80,67% ± 0,82% | 0,846 ± 0,004 |

¹ Nguồn: `audit_diagnostics.json:item6.Full` (khớp LaTeX). `test_results.json:GapLink-Lite_5seed` cho giá trị lệch ≤0,01 pp (93,6416 / 2,9735) — hai lần tính 5-seed khác nhau, xem D9. Số chính luôn là seed 1 = `final.pt`.

Δ Top-1: vs C-3 +13,12 pp [12,29; 14,19]; vs C-4 +9,72 pp [8,68; 10,94]; vs C-4 (val WLR≤5%) +17,38 pp [14,42; 21,07] (`location_bootstrap.event_level.paired`).

## Tracker online (1.922 khoảng che khuất tiêm, a_max = 95)
| | IRR | IRR CI95 | WLR | IDF1 | IDSW |
|---|---|---|---|---|---|
| C-3 full | 84,96% | | 10,25% | 77,60% | |
| GapLink full | **96,20%** | [94,84; 97,56] | **0,21%** | **99,11%** | 98 |
| GapLink partial | **95,99%** | [94,86; 97,11] | **0,36%** | **99,02%** | 103 |

Δ IRR (full) vs C-3 +11,24 pp [9,10; 14,35]; vs C-2 +11,03 pp [7,84; 15,91]. Partial vs C-3 +14,83 pp [13,00; 17,17].

## Live YOLOv8n + GapLink-VMS (40 video test, 3.103 khoảng mất dấu tự nhiên) — `live_security_results.json`
| | Nat-IRR | Nat-WLR | FAR lộ trình cấm | F1 dwell khi khuất |
|---|---|---|---|---|
| C-3 | 77,35% | 37,78% | 49,22% | 35,39% |
| GapLink-Lite | **87,30%** | **10,20%** | **12,88%** | **53,26%** |

Cột dwell của bảng live v1.1 dùng bộ chấm ngày 18/11, không tái tạo được với code cuối (D14).

Latency (ngã tư đông, RTX 3050 6GB, batch 1): **28,95 ms/frame = 34,5 FPS**
(YOLO 12,53 + HSV/NMS 3,26 + GapLink-VMS 13,16 ms). Không dùng số 79,7 FPS (D7).

## Live v1.3 — checkpoint theo detector (đóng băng riêng: `release/FREEZE_MANIFEST_v1.3.json`)
- Checkpoint `release/live_v1.3.pt` SHA256 `da2a3b14…637f`, fine-tune từ `final.pt` (bước 1.500/3.000, seed 20261121).
- Chọn trên val theo `docs/PREREG_live_v1.3.md` (tag `v1.3-prereg` → `v1.3-live-selected`), test chạy 1 lần.
- Kiểm tra: `python scripts/freeze_results.py --set v1.3` → `[OK]`.

| (chọn trên val) | Val Nat-IRR | Val Nat-WLR | Test Nat-IRR | Test Nat-WLR | Route recall (che) | Route FAR |
|---|---|---|---|---|---|---|
| C-1 | 6,05% | 0,00% | 8,93% | 0,00% | 33,33% | 3,45% |
| C-2 θ=1,2 | 81,56% | 4,87% | 84,58% | 12,92% | 41,98% | 11,32% |
| C-3 θ=1,0 † | 32,23% | 16,37% | 57,63% | 24,70% | 47,53% | 32,56% |
| C-4 θ=1,2 | 32,08% | 3,81% | 61,85% | 9,94% | 49,38% | 14,04% |
| GapLink v1.1 τ=0,25 (tham chiếu) | 81,69% | 13,33% | 87,30% | 10,20% | 45,68% | 12,88% |
| **GapLink v1.3 τ=0,65** | 77,55% | 4,21% | **84,41%** | **6,09%** | 42,59% | **9,27%** |

† không θ nào đạt Nat-WLR ≤ 5% trên val.
CI 95% theo 15 địa điểm (`release/results_v1.3/location_bootstrap_live.json`): v1.3 − C-2: ΔNat-IRR −0,17 pp [−1,33; 1,08], ΔNat-WLR −6,82 pp [−11,04; −3,18];
v1.3 − v1.1: ΔNat-IRR −2,89 [−3,47; −1,77], ΔNat-WLR −4,11 [−5,98; −1,57], ΔFAR −3,61 [−7,24; −0,74].
Dwell không báo cáo cho v1.3 (D14).
