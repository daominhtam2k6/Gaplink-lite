> ⚠️ **OUTDATED (checkpoint 589b41…, first test run).** Do not cite. Current frozen numbers: `release/FROZEN_RESULTS.md`; deviations: `docs/DEVIATIONS.md`.

# GapLink-Lite: Final Capstone Project Report

**Title:** *GapLink-Lite: A Small Self-Developed Neural Network for Vehicle Identity Re-Association Across Occlusion Gaps with Calibrated Uncertainty*  
**Author:** Đào Minh Tâm (SIC-AI-N05) | **Instructor:** TS. Trần Quang Quý  
**Dataset:** UA-DETRAC (100 sequences, 140,096 frames, 8,384 vehicle tracks, 1,802,853 vehicle boxes, **0 human boxes**)  
**Frozen Model Checkpoint:** `release/final.pt` (`327,979` parameters, SHA256: `589b4154a7e39d92ca09d5f8502e495db440dde4ea25c58a4d23fc3c98267792`)

---

## 1. Executive Summary & Pre-Registered Hypotheses

| Hypothesis | Target Criterion (Locked in `docs/PREREG.md`) | Locked Test Result (`runs/test_final/test_results.json`) | Status |
| :--- | :--- | :--- | :--- |
| **H1 (Hard Event Top-1)** | $\ge +4.0\%$ over C-3 and $\ge +2.0\%$ over C-4 on hard subset; 95% paired CI excludes 0 | **+14.08%** over C-3 (95% CI `[12.89, 15.35]%`); **+9.52%** over C-4 (95% CI `[8.31, 10.78]%`) | **PASSED** |
| **H2 (Calibrated Coverage)** | Empirical 90% 2D Gaussian ellipse coverage in `[85%, 95%]` across all 3 gap bins | Overall **87.70%** (`Short`: **86.46%**, `Med`: **90.34%**, `Long`: **84.72%**) | **PASSED** |
| **H3 (Online Tracker IRR)** | Higher Identity Recovery Rate (IRR) than C-3 with WLR $\le 5.0\%$ on 40 test sequences | Full IRR **96.15%** vs C-3 `90.06%` (WLR **0.17%**); Partial IRR **96.08%** vs C-3 `88.89%` (WLR **0.17%**) | **PASSED** |

---

## 2. Event-Level Re-Association Benchmark (12,000 Locked Test Events)

| Method | Top-1 All (%) | 95% Cluster CI (%) | Short (3–15) | Med (16–45) | Long (46–90) | Hard Top-1 (%) | Dustbin F1 (%) | Wrong-Link Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C-2** (Constant Velocity) | 66.36 | [64.28, 68.18] | 77.85 | 66.29 | 39.91 | 67.11 | 30.74 | 19.47 |
| **C-3** (Kalman + Mahalanobis) | 68.13 | [65.93, 70.08] | 81.53 | 65.50 | 42.90 | 69.26 | 28.84 | 20.61 |
| **C-4** (C-3 + Size + Color) | 72.42 | [70.01, 74.54] | 81.70 | 70.26 | 55.70 | 73.82 | 27.94 | 16.47 |
| **GapLink-Lite (Ours)** | **83.59** | **[81.64, 85.40]** | **87.58** | **84.90** | **71.45** | **83.34** | **62.02** | **4.30** |

---

## 3. Online Tracker Benchmark (40 Test Sequences, 2,970 Occlusion Intervals)

| Tracker Variant | Full Occlusion IRR (%) | Full 95% CI (%) | Full WLR (%) | Partial Occlusion IRR (%) | Partial 95% CI (%) | Partial WLR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C-1** (Stage-1 IoU Only) | 0.00 | [0.00, 0.00] | 0.00 | 0.00 | [0.00, 0.00] | 0.00 |
| **C-2** (Constant Velocity) | 89.96 | [87.77, 92.09] | 2.86 | 89.92 | [87.77, 92.07] | 2.86 |
| **C-3** (Kalman + Mahalanobis) | 90.06 | [88.04, 92.28] | 5.71 | 88.89 | [86.66, 91.32] | 6.91 |
| **C-4** (C-3 + Size + Color) | 90.23 | [88.10, 92.47] | 5.44 | 89.75 | [87.62, 91.87] | 5.88 |
| **GapLink-Lite (Ours)** | **96.15** | **[95.41, 96.84]** | **0.17** | **96.08** | **[95.34, 96.78]** | **0.17** |

---

## 4. Controlled Ablation Study (12,000 Test Events)

| Ablation Variant | Top-1 All (%) | Short (3–15) | Med (16–45) | Long (46–90) | Dustbin F1 (%) | WLR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full GapLink-Lite** | **83.59** | **87.58** | **84.90** | **71.45** | **62.02** | **4.30** |
| **w/o Context** (No DeepSets) | 82.43 | 87.43 | 83.08 | 69.44 | 60.05 | 5.28 |
| **w/o Color** (Geometry Only) | 81.69 | 87.47 | 83.25 | 64.85 | 59.87 | 5.46 |
| **Gap-Blind** ($g=1$ Fixed) | 80.21 | 86.14 | 81.11 | 64.49 | 57.27 | 6.42 |
