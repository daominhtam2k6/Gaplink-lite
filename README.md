# GapLink-Lite & GapLink-VMS: Uncertainty-Aware Vehicle Re-Association & Real-Time Security Analytics

A compact, self-developed neural network (**327,979 parameters**, `1.3 MB`, randomly initialized) for re-associating vehicle trajectories across multi-second occlusion gaps ($3$ to $90$ frames) using gap-conditioned 2D Gaussian uncertainty and differentiable Sinkhorn Optimal Transport with a learnable dustbin.

Packaged end-to-end with **Live YOLOv8n + Online 128-bin HSV Extraction** into **GapLink-VMS** — an OCR-free real-time security camera analytics console solving three concrete CCTV operational tasks on raw video streams (`.mp4` / RTSP):
1. **Forbidden Zone-to-Zone Route & Origin-Destination (OD) Flow Enforcement (`Z_A → Z_B`)**
2. **Restricted-Zone Anti-Loitering / Dwell-Time Alerts (`Dwell > T_max` across passing occluders)**
3. **Click-to-Lock Suspect Target Pursuit (`2σ` Gaussian Search Zone & Dustbin Safe-Hold)**

## Key Results on Locked UA-DETRAC Test Split (40 Sequences)
- **5-Seed Event-Level Accuracy (`12,000` locked test events):** **`93.64% ± 0.14%`** Top-1 (`93.41%` Seed 1) vs. `80.29%` for Kalman+Mahalanobis (`C-3`), cutting Wrong-Link Rate (WLR) by **$6.5\times$** (`19.24% → 2.98%`).
- **Online Tracker Identity Recovery (`1,922` controlled occlusions):** **`96.20%` IRR / `99.11%` IDF1** (full occlusion) and **`95.99%` IRR / `99.02%` IDF1** (partial occlusion).
- **Live YOLOv8n End-to-End Speed (Laptop RTX 3050 6GB):** **`12.54 ms/frame` (`79.7 FPS`)** single-frame streaming latency (`9.99 ms` YOLOv8n + `1.45 ms` online HSV + `1.10 ms` GapLink-Lite VMS).

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
