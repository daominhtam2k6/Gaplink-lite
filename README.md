# GapLink-Lite & GapLink-VMS: Uncertainty-Aware Vehicle Re-Association & Real-Time Security Analytics

A compact, self-developed neural network (**327,979 parameters**, `1.3 MB`, randomly initialized) for re-associating vehicle trajectories across multi-second occlusion gaps ($3$ to $90$ frames) using gap-conditioned 2D Gaussian uncertainty and differentiable Sinkhorn Optimal Transport with a learnable dustbin.

Packaged end-to-end with **Live YOLOv8n + Online 128-bin HSV Extraction** into **GapLink-VMS** — an OCR-free real-time security camera analytics console solving three concrete CCTV operational tasks on raw video streams (`.mp4` / RTSP):
1. **Forbidden Zone-to-Zone Route & Origin-Destination (OD) Flow Enforcement (`Z_A → Z_B`)**
2. **Restricted-Zone Anti-Loitering / Dwell-Time Alerts (`Dwell > T_max` across passing occluders)**
3. **Click-to-Lock Suspect Target Pursuit (`2σ` Gaussian Search Zone & Dustbin Safe-Hold)**

## Key Results on Locked UA-DETRAC Test Split (40 sequences, 15 camera locations)
Frozen checkpoint `release/final.pt` (SHA256 `844fc391…`). All numbers: `release/FROZEN_RESULTS.md`, sources `release/results_v1.1/*.json`; verify with `python scripts/freeze_results.py`. Protocol deviations (incl. a second test run after a data bug fix): `docs/DEVIATIONS.md`.

- **Event level (12,000 events):** Top-1 **93.41%**, WLR **3.01%** (5-seed mean 93.64% ± 0.14%) vs. C-3 Kalman+Mahalanobis 80.29% / 19.24%. CIs: paired bootstrap over camera locations (`release/results_v1.1/location_bootstrap.json`).
- **Online tracker (1,922 injected occlusions):** IRR **96.20%** / IDF1 **99.11%** (full), **95.99%** / **99.02%** (partial); IRR gain over C-3 +11.24 pp, location-bootstrap 95% CI [9.10, 14.35].
- **Pre-registered hypotheses:** H1 supported, **H2 not supported** (calibrated long-gap coverage 95.14% val / 95.86% test > 95%), H3 supported, P1 passed.
- **Live YOLOv8n + GapLink-VMS (40 raw test videos):** v1.1 point (τ = 0.25): Nat-IRR 87.30%, Nat-WLR 10.20%. **v1.3** detector-aware checkpoint (`release/live_v1.3.pt`) with validation-selected operating points: Nat-IRR **84.41%** vs 84.58% for C-2 (Δ −0.17 pp, CI [−1.33, 1.08]) at **half the wrong-link rate** (6.09% vs 12.92%, Δ −6.82 pp, CI [−11.04, −3.18]). Details: `release/FROZEN_RESULTS.md`.
- **Speed (RTX 3050 Laptop 6GB, batch 1, dense intersection):** **28.95 ms/frame (34.5 FPS)** = 12.53 ms YOLOv8n + 3.26 ms HSV/NMS + 13.16 ms GapLink-Lite + VMS.

## Quick Start & Reproduce Commands
```powershell
# 1. Run full unit test suite (19 tests: core model, tracker, IDF1 toy, and VMS security analytics)
D:\gaplink\.venv\Scripts\pytest.exe -c D:\gaplink\pytest.ini D:\gaplink\tests -s -q

# 2. Launch Interactive Local Web VMS Security Dashboard (http://127.0.0.1:8000)
D:\gaplink\.venv\Scripts\python.exe D:\gaplink\demo\security_vms_app.py --port 8000

# 3. Reproduce Live YOLOv8n end-to-end security evaluation across all 40 raw test videos
D:\gaplink\.venv\Scripts\python.exe D:\gaplink\scripts\eval_live_security.py

# 4. Render side-by-side VMS security MP4 demo videos
D:\gaplink\.venv\Scripts\python.exe D:\gaplink\demo\render_security_demo.py

# 5. Reproduce controlled 12,000-event & tracker benchmark from frozen release checkpoint
D:\gaplink\.venv\Scripts\python.exe D:\gaplink\scripts\run_test.py --ckpt release/final.pt
```

## Key Deliverables
- **Frozen Model & Configs:** `release/final.pt`, `release/final.sha256`, `release/calib.json`, `release/tracker.yaml`
- **Live Security Camera Pipeline & VMS App:** `src/live_detector.py`, `src/security_analytics.py`, `demo/security_vms_app.py`, `demo/render_security_demo.py`
- **LaTeX Report (SIC Template):** `report/latex/main.tex` (synced to `D:\Capstone-Project_latex\Capstone-Project_latex`)
