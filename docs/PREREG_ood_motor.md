# PREREG — Out-of-domain exploratory test: motorcycles (VisDrone2019-MOT)

Written 2026-11-21, **before** any VisDrone data was downloaded or converted. Not edited after results.
Status: **exploratory**. It does not change H1/H2/H3/P1 or any frozen result in `release/`.

## Question
UA-DETRAC has no motorcycle labels, so GapLink-Lite never saw motorcycle trajectories.
Does the frozen model still re-link motorcycles after occlusion better than the classical baselines?

## Data
- VisDrone2019-MOT **valset + test-dev** (both have public GT). Trainset is not used.
- All sequences are used. A group (below) skips a sequence only if it has < 2 tracks of that group.
- Two groups built from the same videos:
  - `motor` (VisDrone category 10) — **primary**
  - `car` (categories 4 car, 5 van, 6 truck, 9 bus) — **control**: separates the drone-viewpoint domain shift from the motorcycle effect.
- Category 0 boxes → ignored regions. Boxes with score 0 or w/h < 2 px are dropped. Occlusion 0/1/2 → visibility 1.0/0.6/0.3.
- Camera motion per sequence: median phase-correlation shift between frames t and t+5, in native px/frame. **static** if < 1.0 px/frame. Used only for the stratified report below; no sequence is excluded on it.

## Protocol (fixed)
- Model: `release/final.pt` (SHA in `release/final.sha256`). No training, fine-tuning or recalibration. Calibration: `release/calib.json`.
- Operating points: Phase-1 values in `release/tracker.yaml` (`gaplink_tau`, `c2_theta`, `c3_theta`, `c4_*`). Not re-selected.
- Events: `scripts/make_fixed_events.generate_split_events` with the default `EventCfg`, 3000 events per group, seeds motor = 9100 and car = 9200. Events are written once; SHA-256 recorded.
- Tracker-level: the same GT-injection protocol as Phase 1 (`make_hide_plan` seed 7000+i, `make_detections` seed 8000+i, modes full and partial).
- `scripts/eval_ood_motor.py --eval` runs **once**. It refuses to run when `runs/ood_visdrone/results.json` exists.
- CIs: cluster bootstrap over sequences (2000 resamples, seed 0). VisDrone has no location metadata, so sequence = cluster.

## Hypotheses
- **O1 (control):** on `car`, the paired Top-1 difference GapLink − C-3 has a 95% CI lower bound > 0.
- **O2 (primary):** on `motor`, the paired Top-1 difference GapLink − C-3 has a 95% CI lower bound > 0.
- **O3:** on `motor`, GapLink WLR ≤ C-3 WLR (point estimate).
- Secondary (descriptive only): vs C-2 and C-4; Top-1 by gap bin; static vs moving strata; tracker-level IRR/WLR with CIs.

## Interpretation rules
- Every outcome is reported as is, including failures. No thresholds are changed and no sequences are dropped after seeing results.
- O1 fails → the drone domain shift dominates; motorcycle results are not interpretable as a motorcycle effect.
- O1 passes and O2 fails → the limitation is specific to motorcycles. Report it in Limitations and the Roadmap (motorcycle tracking data).
- If the run crashes before `results.json` is written, it may be restarted with the same code and data. This is logged in `docs/DEVIATIONS.md`.

## Known differences from the training domain
Drone viewpoint (some sequences have a moving camera), different resolutions, mostly Chinese cities, much denser scenes than UA-DETRAC. These differences are why the test is exploratory.
