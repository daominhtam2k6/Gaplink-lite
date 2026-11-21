"""GapLink-VMS v1.2: live operating points chosen on VALIDATION, then ONE locked test run.

Why: with YOLOv8n input the v1.1 point (tau=0.25, chosen on GT-box events) gives Nat-WLR 10.2% on test,
violating the WLR<=5% spirit of PREREG §3. v1.2 re-applies the PREREG §3 rule to the live setting:
  for every method, pick the acceptance threshold with the highest Nat-IRR s.t. Nat-WLR <= 5% on the
  13 validation videos (4 locations); if none qualifies, the lowest Nat-WLR.
Nothing else changes: same checkpoint (release/final.pt), same detector, a_max, zones and scoring.

Steps (Windows):
  python -m scripts.eval_live_v12 --select            # val sweep   -> runs/live_v12_val/, release/live_v1.2.yaml
  python -m scripts.eval_live_v12 --test              # single test -> runs/live_v12_test/live_v12_results.json
Both resume from per-sequence checkpoints (--select / --test can be re-launched after interruption).
--test refuses to run without release/live_v1.2.yaml, and refuses to overwrite a finished test result.
--test also re-runs the v1.1 operating points and asserts they reproduce
runs/test_final/live_security_results.json exactly (parity of the refactored scorer).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TAU_GRID = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
THETA_GRID = {
    "C-2": [0.35, 0.5, 0.65, 0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0],
    "C-3": [0.5, 1.0, 1.8, 2.8, 4.61, 6.0, 9.21, 13.8, 18.0, 25.0],
    "C-4": [1.2, 2.0, 3.2, 4.8, 7.0, 10.0, 15.0, 22.0, 29.0],
}
WLR_MAX = 0.05
GATE_GRID = ["95", "99", "99.9"]  # chi2_2D levels, see docs/PREREG_live_v1.2.md


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def paths():
    cfg = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text(encoding="utf-8"))
    root = Path(cfg["root_dir"])
    return root if root.exists() else ROOT


def make_assoc(kind, thr, tcfg, model, device):
    from src.baselines import ConstantVelocityBaseline, KalmanMahalanobisBaseline
    from src.model.gaplink import GapLinkAssociator
    if kind == "C-1":
        return None
    if kind == "C-2":
        return ConstantVelocityBaseline(theta=float(thr))
    if kind == "C-3":
        return KalmanMahalanobisBaseline(theta=float(thr), name="C-3")
    if kind == "C-4":
        return KalmanMahalanobisBaseline(theta=float(thr), lambda_s=float(tcfg["c4_lambda_s"]),
                                         lambda_c=float(tcfg["c4_lambda_c"]), name="C-4")
    if kind == "GapLink-Lite":
        return GapLinkAssociator(model, device=device, tau=float(thr), calib=None)
    if kind == "GapLink-Gated":
        from src.gated_assoc import CHI2_2D, make_gated_associator
        gate, tau = thr
        return make_gated_associator(model, device, tau=float(tau), chi2_gate=CHI2_2D[gate])
    raise ValueError(kind)


def run_split(root: Path, seqs: list[str], configs: list[tuple[str, str, float]], out_dir: Path, cache_name: str):
    """configs: (label, kind, threshold). Returns {label: {seq: counts}}; per-seq results cached in out_dir."""
    import torch
    import scripts.eval_live_security as els
    from src.io_mot import load_seq
    from src.live_detector import LiveVehicleDetector
    from src.live_eval import run_method_on_seq
    from src.model.gaplink import GapLinkLite

    els.CACHE_DIR = root / "data" / "cache" / cache_name
    tcfg = yaml.safe_load((root / "release" / "tracker.yaml").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(root / "release" / "final.pt", map_location=device, weights_only=False)
    model = GapLinkLite(use_ctx=True, use_color=True, gap_aware=True).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    detector = LiveVehicleDetector(imgsz=640, conf_thr=0.28, nms_iou=0.52)

    seq_dir = out_dir / "per_seq"
    seq_dir.mkdir(parents=True, exist_ok=True)
    out = {lbl: {} for lbl, _, _ in configs}
    t0 = time.time()
    for i, s_name in enumerate(seqs, 1):
        cp = seq_dir / f"{s_name}.json"
        done = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
        todo = [c for c in configs if c[0] not in done]
        if todo:
            seq = load_seq(root / "data" / "mot", s_name)
            dets = els.extract_or_load_yolo_seq(seq, detector, batch_size=32)
            det_gt_map, nat_gaps = els.match_dets_to_gt_posthoc(seq, dets, iou_match=0.45)
            zones, _, _, _ = els.build_adaptive_security_zones(seq)
            gt_sets = els.compute_yolo_observable_security_gt(seq, zones, det_gt_map, dets)
            for lbl, kind, thr in todo:
                assoc = make_assoc(kind, thr, tcfg, model, device)
                done[lbl] = run_method_on_seq(seq, dets, det_gt_map, nat_gaps, zones, gt_sets, assoc, tcfg)
                cp.write_text(json.dumps(done), encoding="utf-8")
        for lbl, _, _ in configs:
            out[lbl][s_name] = done[lbl]
        print(f"[{i:02d}/{len(seqs)}] {s_name} done ({time.time() - t0:.0f}s)", flush=True)
    return out


def aggregate(per_seq: dict) -> dict:
    from src.live_eval import add_counts, empty_counts, summarize
    tot = empty_counts()
    for c in per_seq.values():
        tot = add_counts(tot, c)
    return summarize(tot)


def git_commit(root):
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def cmd_select(root: Path):
    import csv
    from src.live_eval import select_operating_point
    splits = list(csv.DictReader(open(root / "data" / "splits.csv", encoding="utf-8")))
    val_seqs = [r["seq"] for r in splits if r["split"] == "val"]
    configs = [("C-1", "C-1", 0.0)]
    configs += [(f"GapLink-Lite@{t}", "GapLink-Lite", t) for t in TAU_GRID]
    configs += [(f"GapLink-Gated@{g}@{t}", "GapLink-Gated", (g, t)) for g in GATE_GRID for t in TAU_GRID]
    for k, grid in THETA_GRID.items():
        configs += [(f"{k}@{t}", k, t) for t in grid]
    out_dir = root / "runs" / "live_v12_val"
    per = run_split(root, val_seqs, configs, out_dir, cache_name="yolo_val")
    sweep = {}
    for lbl, kind, thr in configs:
        s = aggregate(per[lbl])
        sweep[lbl] = {"kind": kind, "thr": thr, **s["natural_occlusion"], "route": s["forbidden_route"], "dwell": s["dwell_alert"]}
    chosen = {}
    families = {"GapLink-Lite": ("GapLink-Lite", "GapLink-Gated"), **{k: (k,) for k in THETA_GRID}}
    complexity = {"GapLink-Lite": 0, "GapLink-Gated": 1}
    for fam, kinds in families.items():
        rows = []
        for l, v in sweep.items():
            if v["kind"] in kinds:
                gate_rank = GATE_GRID.index(v["thr"][0]) if v["kind"] == "GapLink-Gated" else -1
                rows.append({"label": l, "kind": v["kind"], "thr": v["thr"], "nat_irr": v["nat_irr"],
                             "nat_wlr": v["nat_wlr"], "_tie": (complexity.get(v["kind"], 0), gate_rank)})
        # PREREG tie-break: higher IRR, lower WLR, ungated before gated, tighter gate first
        rows.sort(key=lambda r: r["_tie"])
        best = select_operating_point(rows, wlr_max=WLR_MAX)
        best.pop("_tie", None)
        chosen[fam] = best
    res = {"split": "val", "n_seqs": len(val_seqs), "wlr_max": WLR_MAX, "git_commit": git_commit(root),
           "final_sha256": sha256_file(root / "release" / "final.pt"), "sweep": sweep, "chosen": chosen}
    (out_dir / "select.json").write_text(json.dumps(res, indent=2, default=list), encoding="utf-8")
    op = {"version": "1.2-live", "selected_on": "val (13 videos, 4 locations), YOLOv8n imgsz640 conf0.28",
          "rule": f"PREREG §3: max Nat-IRR s.t. Nat-WLR <= {WLR_MAX}; else min Nat-WLR",
          "final_sha256": res["final_sha256"], "val_select_sha256": sha256_file(out_dir / "select.json"),
          "gaplink_kind_live": chosen["GapLink-Lite"]["kind"],
          "gaplink_thr_live": list(chosen["GapLink-Lite"]["thr"]) if isinstance(chosen["GapLink-Lite"]["thr"], (tuple, list)) else chosen["GapLink-Lite"]["thr"],
          "c2_theta_live": chosen["C-2"]["thr"], "c3_theta_live": chosen["C-3"]["thr"], "c4_theta_live": chosen["C-4"]["thr"]}
    (root / "release" / "live_v1.2.yaml").write_text(yaml.safe_dump(op, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps(chosen, indent=2))
    print("wrote release/live_v1.2.yaml  -> commit it BEFORE running --test")


def cmd_test(root: Path):
    import csv
    hold = root / "release" / "live_v1.2.HOLD"
    if hold.exists():
        raise SystemExit(f"test run blocked by {hold}: candidate set still being developed on val")
    op_p = root / "release" / "live_v1.2.yaml"
    if not op_p.exists():
        raise SystemExit("release/live_v1.2.yaml missing: run --select first")
    op = yaml.safe_load(op_p.read_text(encoding="utf-8"))
    assert sha256_file(root / "release" / "final.pt") == op["final_sha256"], "checkpoint changed"
    assert sha256_file(root / "runs" / "live_v12_val" / "select.json") == op["val_select_sha256"], "val selection changed"
    out_dir = root / "runs" / "live_v12_test"
    final = out_dir / "live_v12_results.json"
    if final.exists():
        raise SystemExit(f"{final} exists: the v1.2 test run is single-shot")
    tcfg = yaml.safe_load((root / "release" / "tracker.yaml").read_text(encoding="utf-8"))
    splits = list(csv.DictReader(open(root / "data" / "splits.csv", encoding="utf-8")))
    test_seqs = [r["seq"] for r in splits if r["split"] == "test"]
    configs = [
        ("C-1", "C-1", 0.0),
        ("v1.1/C-2", "C-2", tcfg["c2_theta"]), ("v1.1/C-3", "C-3", tcfg["c3_theta"]),
        ("v1.1/C-4", "C-4", tcfg["c4_theta"]), ("v1.1/GapLink-Lite", "GapLink-Lite", tcfg["gaplink_tau"]),
        ("v1.2/C-2", "C-2", op["c2_theta_live"]), ("v1.2/C-3", "C-3", op["c3_theta_live"]),
        ("v1.2/C-4", "C-4", op["c4_theta_live"]), ("v1.2/GapLink-Lite", op["gaplink_kind_live"], tuple(op["gaplink_thr_live"]) if isinstance(op["gaplink_thr_live"], list) else op["gaplink_thr_live"]),
    ]
    per = run_split(root, test_seqs, configs, out_dir, cache_name="yolo_test")
    methods = {lbl: aggregate(per[lbl]) for lbl, _, _ in configs}
    # parity: refactored scorer must reproduce frozen v1.1 counts exactly
    frozen = json.loads((root / "runs" / "test_final" / "live_security_results.json").read_text(encoding="utf-8"))["methods"]
    parity = {}
    for lbl, fk in [("C-1", "C-1"), ("v1.1/C-2", "C-2"), ("v1.1/C-3", "C-3"), ("v1.1/C-4", "C-4"), ("v1.1/GapLink-Lite", "GapLink-Lite")]:
        parity[lbl] = methods[lbl]["raw_counts"] == frozen[fk]["raw_counts"]
    res = {"split": "test", "n_seqs": len(test_seqs), "git_commit": git_commit(root), "operating_points": op,
           "parity_with_v1.1": parity, "methods": methods, "per_seq_counts": per}
    final.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(parity, indent=2))
    for lbl in methods:
        print(lbl, methods[lbl]["natural_occlusion"], methods[lbl]["forbidden_route"], methods[lbl]["dwell_alert"])
    if not all(parity.values()):
        print("WARNING: parity failed - do not report v1.2 until the scorer difference is explained")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select", action="store_true")
    g.add_argument("--test", action="store_true")
    a = ap.parse_args()
    root = paths()
    cmd_select(root) if a.select else cmd_test(root)


if __name__ == "__main__":
    main()
