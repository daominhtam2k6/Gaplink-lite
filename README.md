# GapLink-Lite: Vehicle Identity Re-Association Across Occlusion Gaps

A 100% self-developed, compact neural network (`327,979` parameters, randomly initialized) for re-associating vehicle trajectories across occlusion gaps ($3$ to $90$ frames) with calibrated 2D Gaussian uncertainty on the **UA-DETRAC** benchmark.

## Key Results on Locked Test Split (40 Sequences, 12,000 Events)
- **Event-Level Top-1 Accuracy:** **83.59%** vs `68.13%` for Kalman+Mahalanobis (`C-3`) and `72.42%` for Kalman+Size+Color (`C-4`).
- **Wrong-Link Rate (WLR):** Reduced $6\times$ from `20.61%` (`C-3`) down to **4.30%**.
- **Calibrated 90% Ellipse Coverage:** **87.70%** overall (`86.46%` short, `90.34%` medium, `84.72%` long).
- **Online Tracker Identity Recovery Rate (IRR):** **96.15%** (full occlusion) and **96.08%** (partial occlusion) vs `90.06%` / `88.89%` for `C-3`.

## Quick Reproduce Commands
```powershell
# 1. Run all 16 unit tests
D:\gaplink\.venv\Scripts\pytest.exe -c D:\gaplink\pytest.ini D:\gaplink\tests -s -q

# 2. Verify frozen release checkpoint and reproduce test evaluation
D:\gaplink\.venv\Scripts\python.exe D:\gaplink\scripts\run_test.py --ckpt release/final.pt

# 3. Regenerate all report figures and side-by-side MP4 demo clips
D:\gaplink\.venv\Scripts\python.exe D:\gaplink\scripts\make_figures.py
D:\gaplink\.venv\Scripts\python.exe -m demo.select_clips
```

## Key Deliverables
- **Frozen Model & Configs:** `release/final.pt`, `release/final.sha256`, `release/calib.json`, `release/tracker.yaml`
- **Visual Demo Videos (MP4):** `demo/out/demo_full.mp4` and 4 individual sequence clips in `demo/out/clip_01_*.mp4` .. `clip_04_*.mp4`
- **LaTeX Report (SIC Template):** `report/latex/main.tex` (also synced to `D:\Capstone-Project_latex\Capstone-Project_latex`)
- **Markdown Report:** `report/report.md`
