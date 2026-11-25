"""Location-cluster bootstrap (D5) for the frozen v1.1 test results.

No threshold, calibration or checkpoint is changed. Event-level predictions are recomputed
deterministically from release/final.pt (CPU is fine) and MUST reproduce the frozen
aggregate numbers in release/results_v1.1/test_results.json; otherwise the script aborts.
Tracker-level CIs are re-aggregated from the stored per-sequence records (no inference).

    .venv\\Scripts\\python.exe -m scripts.rebootstrap_location
Output: runs/test_final_locboot/location_bootstrap.json (+ copy in release/results_v1.1/)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_events import bootstrap_cluster, bootstrap_paired, evaluate_predictions  # noqa: E402
from src.location_bootstrap import counts_to_arrays, load_seq_to_location, regroup_by_location  # noqa: E402

N_BOOT = 10000
SEED = 0


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolve_root(cli: str | None) -> Path:
    if cli:
        return Path(cli)
    cfg = ROOT / "configs" / "paths.yaml"
    r = ROOT / yaml.safe_load(cfg.read_text(encoding="utf-8"))["root_dir"]
    return r if r.exists() else ROOT


def ci(by_loc):
    lo, hi = bootstrap_cluster(by_loc, n_boot=N_BOOT, seed=SEED)
    return [lo, hi]


def paired(a, b):
    r = bootstrap_paired(a, b, n_boot=N_BOOT, seed=SEED)
    r["cluster_unit"] = "location"
    return r


def event_level(root: Path, frozen: dict, s2l: dict) -> dict:
    import torch
    from scripts.run_test import decide_from_comps, decide_with_tau, infer_all_events, precompute_kalman_test
    from src.baselines import ConstantVelocityBaseline
    from src.model.gaplink import GapLinkLite

    tr = yaml.safe_load((root / "release/tracker.yaml").read_text())
    bv = json.loads((root / "runs/baselines_val.json").read_text())
    events = pickle.loads((root / "data/cache/test_events.pkl").read_bytes())

    ckpt = torch.load(root / "release/final.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = GapLinkLite(use_ctx=bool(cfg.get("use_ctx", True)), use_color=bool(cfg.get("use_color", True)),
                        gap_aware=bool(cfg.get("gap_aware", True)))
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds = {}
    preds["C-2"] = [ConstantVelocityBaseline(theta=float(tr["c2_theta"])).decide(e) for e in events]
    preds["C-2_wlr5"] = [ConstantVelocityBaseline(theta=float(bv["C-2_wlr5"]["theta"])).decide(e) for e in events]
    comps, _ = precompute_kalman_test(events)
    preds["C-3"] = decide_from_comps(comps, theta=float(tr["c3_theta"]))
    preds["C-3_wlr5"] = decide_from_comps(comps, theta=float(bv["C-3_wlr5"]["theta"]))
    preds["C-4"] = decide_from_comps(comps, theta=float(tr["c4_theta"]), ls=float(tr["c4_lambda_s"]), lc=float(tr["c4_lambda_c"]))
    c4w = bv["C-4_wlr5"]
    preds["C-4_wlr5"] = decide_from_comps(comps, theta=float(c4w["theta"]), ls=float(c4w["lambda_s"]), lc=float(c4w["lambda_c"]))
    P, _ = infer_all_events(model, events, torch.device("cpu"))
    preds["GapLink-Lite"] = decide_with_tau(events, P, tau=float(tr["gaplink_tau"]))

    hard_idx = [i for i, e in enumerate(events) if e["is_hard"]]
    hard_events = [events[i] for i in hard_idx]
    res, by_loc, by_loc_hard, repro = {}, {}, {}, {}
    for name, pr in preds.items():
        r, by_seq = evaluate_predictions(events, pr)
        _, by_seq_h = evaluate_predictions(hard_events, [pr[i] for i in hard_idx])
        by_loc[name] = regroup_by_location(by_seq, s2l)
        by_loc_hard[name] = regroup_by_location(by_seq_h, s2l)
        f = frozen["event_level"].get(name, {})
        repro[name] = {k: {"frozen": f.get(k), "recomputed": r[k], "match": f.get(k) is not None and abs(f[k] - r[k]) < 1e-9}
                       for k in ("top1", "top1_hard", "wrong_link_rate")}
        res[name] = {"top1": r["top1"], "top1_ci95_loc": ci(by_loc[name]),
                     "top1_hard": r["top1_hard"], "top1_hard_ci95_loc": ci(by_loc_hard[name]),
                     "wrong_link_rate": r["wrong_link_rate"]}
    bad = [f"{n}.{k}" for n, d in repro.items() for k, v in d.items() if not v["match"]]
    if bad:
        print(json.dumps(repro, indent=1))
        raise SystemExit(f"recomputed predictions do not reproduce frozen numbers: {bad}")
    pairs = {}
    for b in ("C-2", "C-3", "C-4", "C-2_wlr5", "C-3_wlr5", "C-4_wlr5"):
        pairs[f"GapLink_vs_{b}_all"] = paired(by_loc["GapLink-Lite"], by_loc[b])
    for b in ("C-2", "C-3", "C-4"):
        pairs[f"GapLink_vs_{b}_hard"] = paired(by_loc_hard["GapLink-Lite"], by_loc_hard[b])
    # PREREG H1: hard Top-1 vs max(C-2, C-3, C-4)
    best = max(("C-2", "C-3", "C-4"), key=lambda k: res[k]["top1_hard"])
    h1 = dict(pairs[f"GapLink_vs_{best}_hard"])
    h1.update({"strongest_baseline": best, "threshold_pp": 3.0, "passed": h1["delta"] >= 0.03,
               "ci_lo_above_threshold": h1["ci95_lo"] >= 0.03})
    return {"metrics": res, "paired": pairs, "H1_prereg": h1, "reproduction_check": repro}


def tracker_level(frozen: dict, s2l: dict) -> dict:
    out = {}
    for mode, per_seq in frozen["per_seq_tracker"].items():
        methods = sorted({m for rec in per_seq.values() for m in rec})
        by_loc = {}
        for m in methods:
            recs = {s: r[m] for s, r in per_seq.items() if m in r}
            by_loc[m] = regroup_by_location(counts_to_arrays(recs, "IRR", "n_intervals"), s2l)
        mres = {}
        for m in methods:
            agg = frozen["tracker_level"][mode][m]["IRR"]
            point = sum(v.sum() for v in by_loc[m].values()) / sum(len(v) for v in by_loc[m].values())
            if abs(point - agg) > 1e-9:
                raise SystemExit(f"tracker {mode}/{m}: per-seq IRR {point} != frozen {agg}")
            mres[m] = {"IRR": point, "IRR_ci95_loc": ci(by_loc[m])}
        pairs = {f"GapLink_vs_{b}": paired(by_loc["GapLink-Lite"], by_loc[b])
                 for b in methods if b != "GapLink-Lite"}
        out[mode] = {"metrics": mres, "paired": pairs}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--skip-events", action="store_true", help="tracker-level only (no torch needed)")
    a = ap.parse_args()
    root = resolve_root(a.root)

    # integrity: frozen artifacts must be untouched
    man = json.loads((root / "release/FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    for rel in ("release/final.pt", "data/cache/test_events.pkl", "release/tracker.yaml", "data/splits.csv",
                "runs/test_final/test_results.json", "runs/baselines_val.json"):
        assert sha256_file(root / rel) == man["files"][rel]["sha256"], f"frozen artifact changed: {rel}"

    frozen = json.loads((root / "runs/test_final/test_results.json").read_text(encoding="utf-8"))
    s2l = load_seq_to_location(root / "data/splits.csv", split="test")
    result = {
        "deviation": "D5 - CI recomputed with camera location (15 clusters) as resampling unit",
        "n_boot": N_BOOT, "seed": SEED,
        "n_locations": len(set(s2l.values())), "n_sequences": len(s2l),
        "final_sha256": man["final_sha256"],
        "git_commit": subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "tracker_level": tracker_level(frozen, s2l),
    }
    if not a.skip_events:
        result["event_level"] = event_level(root, frozen, s2l)
    out_dir = root / "runs/test_final_locboot"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "location_bootstrap.json"
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copy2(p, root / "release/results_v1.1/location_bootstrap.json")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
