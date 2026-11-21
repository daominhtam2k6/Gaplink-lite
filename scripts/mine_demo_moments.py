"""Find the demo moments where GapLink-Lite (frozen live v1.3) clearly outperforms the baseline, on the 40 test videos.

Every candidate window is re-run from a fresh tracker state exactly as it will be played in the demo, so the
counts saved here are the counts the audience will see. Nothing is tuned: operating points come from
release/live_v1.3.yaml through demo/vms_profile.py.

  route : forbidden Z_A -> Z_B route; baseline false alarms / missed occluded routes vs GapLink
  dwell : stationary vehicles in Z_DWELL hidden by passing traffic; baseline false dwell alarms vs GapLink
  lock  : a MOVING vehicle locked just before a natural occlusion (>= 15 frames); who re-acquires the same vehicle

    python -m scripts.mine_demo_moments [--baseline C-3]
Output: runs/demo_mining/candidates.json (ranked) + summary of lock trials over all test videos.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FPS = 25.0
ROUTE_PRE, ROUTE_POST = 300, 75
LOCK_PRE, LOCK_POST = 60, 60
MAX_LOCK_TRIALS_PER_SEQ = 12


def run_window(seq, dets, det_gt_map, zones, assoc_factory, f0, f1, lock=None):
    """Fresh tracker + VMS on frames [f0, f1]; optional lock=(frame, gt_tid). Returns summary dict."""
    from src.security_analytics import SecurityVMSEngine
    from src.tracker import OnlineTracker
    trk = OnlineTracker(associator=assoc_factory(), iou_thr=0.3, a_max=95, W=seq.W, H=seq.H, seq_name=seq.name, min_hits=3)
    vms = SecurityVMSEngine(trk, zones=zones, forbidden_routes=[("Z_A", "Z_B")], max_dwell_sec=5.0, fps=FPS)
    locked = False
    for f in range(f0, f1 + 1):
        res = vms.step(f, dets[f]["boxes"], dets[f]["hists"], det_gt_map[f])
        if lock is not None and not locked and f >= lock[0]:
            for o in res["tracks"]:
                if int(o.get("gt_tid", -1)) == lock[1]:
                    locked = vms.target_lock.lock_by_track_id(int(o["track_id"]), res["tracks"], frame_id=f)
                    break
    rm, dm, tl = vms.route_mon, vms.dwell_mon, vms.target_lock
    return {
        "route_true": len(rm.true_alerts), "route_fp": len(rm.false_alarms),
        "route_true_gt": sorted({int(a["dest_gt_tid"]) for a in rm.true_alerts}),
        "route_true_occl": sum(1 for a in rm.true_alerts if a.get("occlusions_bridged", 0) > 0),
        "dwell_true": len(dm.true_alerts), "dwell_fp": len(dm.false_alarms),
        "alerts": [{"kind": "route", "frame": int(a["frame"]), "fp": bool(a.get("is_false_alarm")), "track_id": int(a["track_id"]),
                    "box": [float(v) for v in a["box"]] if "box" in a else None} for a in rm.alerts]
                  + [{"kind": "dwell", "frame": int(a["frame"]), "fp": bool(a.get("is_false_alarm")), "track_id": int(a["track_id"]),
                      "box": [float(v) for v in a["box"]] if "box" in a else None} for a in dm.alerts],
        "locked": bool(locked), "reacquired": int(tl.reacquire_count), "hijacked": int(tl.hijack_count),
        "lock_state": tl.state, "reassoc_wrong": sum(1 for r in trk.reassoc_log if r["gt_tid_before"] != -1
                                                       and r["gt_tid_after"] != -1 and r["gt_tid_before"] != r["gt_tid_after"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="C-3")
    ap.add_argument("--seqs", nargs="*", default=None)
    a = ap.parse_args()

    import torch
    import scripts.eval_live_security as els
    from demo.vms_profile import load_profile, make_baseline, make_gaplink
    from src.io_mot import load_seq
    from src.live_detector import LiveVehicleDetector

    prof = load_profile(ROOT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gl = make_gaplink(prof, device)  # stateless associator: safe to reuse
    factories = {"gaplink": lambda: gl, "baseline": lambda: make_baseline(a.baseline, prof)}
    els.CACHE_DIR = ROOT / "data" / "cache" / "yolo_test"
    detector = LiveVehicleDetector(imgsz=640, conf_thr=0.28, nms_iou=0.52)
    rows = list(csv.DictReader(open(ROOT / "data" / "splits.csv", encoding="utf-8")))
    seqs = a.seqs or [r["seq"] for r in rows if r["split"] == "test"]

    out = {"profile": prof.version, "gaplink": prof.gaplink_label, "baseline": prof.baseline_label(a.baseline),
           "route": [], "dwell": [], "lock": [], "lock_trials": []}
    t0 = time.time()
    for si, s in enumerate(seqs, 1):
        seq = load_seq(ROOT / "data" / "mot", s)
        dets = els.extract_or_load_yolo_seq(seq, detector, batch_size=32)
        det_gt_map, nat_gaps = els.match_dets_to_gt_posthoc(seq, dets, iou_match=0.45)
        zones = els.build_adaptive_security_zones(seq)[0]
        N = seq.n_frames
        full = {k: run_window(seq, dets, det_gt_map, zones, f, 1, N) for k, f in factories.items()}

        # ---- route / dwell windows around baseline false alarms and GapLink-only true alerts
        for kind in ("route", "dwell"):
            anchors = [x["frame"] for x in full["baseline"]["alerts"] if x["kind"] == kind and x["fp"]]
            gl_true = [x["frame"] for x in full["gaplink"]["alerts"] if x["kind"] == kind and not x["fp"]]
            anchors += gl_true
            seen = set()
            tried = 0
            for fa in sorted(anchors):
                if tried >= 6:  # cap re-runs per sequence and task
                    break
                f0, f1 = max(1, fa - ROUTE_PRE), min(N, fa + ROUTE_POST)
                if any(abs(f0 - g) < 100 for g in seen):
                    continue
                seen.add(f0)
                tried += 1
                w = {k: run_window(seq, dets, det_gt_map, zones, f, f0, f1) for k, f in factories.items()}
                g, b = w["gaplink"], w["baseline"]
                tp, fp = f"{kind}_true", f"{kind}_fp"
                if g[fp] > 0 or g[tp] < b[tp] or g[tp] == 0:
                    continue  # demo rule: GapLink clean and at least as many true alerts
                score = 2 * (b[fp] - g[fp]) + (g[tp] - b[tp])
                if score <= 0:
                    continue
                out[kind].append({"seq": s, "f0": f0, "f1": f1, "score": score,
                                  "gaplink": {k: g[k] for k in (tp, fp)}, "baseline": {k: b[k] for k in (tp, fp)},
                                  "key_frames": sorted({x["frame"] for x in b["alerts"] if x["kind"] == kind and x["fp"]}
                                                       | {x["frame"] for x in g["alerts"] if x["kind"] == kind and not x["fp"]}),
                                  "alerts_baseline": [x for x in b["alerts"] if x["kind"] == kind],
                                  "alerts_gaplink": [x for x in g["alerts"] if x["kind"] == kind]})

        # ---- lock trials on moving vehicles across natural occlusions
        trials = []
        for gp in nat_gaps:
            if not (15 <= gp["gap"] <= 95):
                continue
            tr = seq.tracks.get(gp["gt_tid"])
            if tr is None:
                continue
            fr = list(tr["frames"])
            try:
                b0 = tr["boxes"][fr.index(gp["t_lost"])]
                b1 = tr["boxes"][fr.index(gp["t_reappear"])]
            except ValueError:
                continue
            disp = float(np.hypot(b1[0] - b0[0], b1[1] - b0[1]))
            if disp < 1.0 * float(b0[3]):
                continue  # must be a moving vehicle
            trials.append((gp["gap"], gp, disp))
        trials.sort(key=lambda x: -x[0])
        for gap, gp, disp in trials[:MAX_LOCK_TRIALS_PER_SEQ]:
            f0, f1 = max(1, gp["t_lost"] - LOCK_PRE), min(N, gp["t_reappear"] + LOCK_POST)
            lk = (max(f0 + 10, gp["t_lost"] - 8), gp["gt_tid"])
            w = {k: run_window(seq, dets, det_gt_map, zones, f, f0, f1, lock=lk) for k, f in factories.items()}
            g, b = w["gaplink"], w["baseline"]
            if not (g["locked"] and b["locked"]):
                continue
            outcome = lambda r: "reacquired" if (r["reacquired"] > 0 and r["hijacked"] == 0) else ("hijacked" if r["hijacked"] > 0 else "lost")
            t = {"seq": s, "f0": f0, "f1": f1, "lock_frame": lk[0], "gt_tid": gp["gt_tid"], "gap": gap,
                 "t_lost": gp["t_lost"], "t_reappear": gp["t_reappear"], "displacement_px": round(disp, 1),
                 "gaplink": outcome(g), "baseline": outcome(b)}
            out["lock_trials"].append(t)
            if t["gaplink"] == "reacquired" and t["baseline"] != "reacquired":
                t2 = dict(t, score=(2 if t["baseline"] == "hijacked" else 1) * gap, key_frames=[gp["t_reappear"]])
                out["lock"].append(t2)
        print(f"[{si:02d}/{len(seqs)}] {s}: route {len(out['route'])} dwell {len(out['dwell'])} lock {len(out['lock'])} "
              f"({time.time() - t0:.0f}s)", flush=True)

    for k in ("route", "dwell", "lock"):
        out[k].sort(key=lambda r: -r["score"])
    lt = out["lock_trials"]
    out["lock_summary"] = {m: {o: sum(1 for t in lt if t[m] == o) for o in ("reacquired", "hijacked", "lost")}
                           for m in ("gaplink", "baseline")} | {"n_trials": len(lt)}
    d = ROOT / "runs" / "demo_mining"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"candidates_{a.baseline}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["lock_summary"], indent=1))
    for k in ("route", "dwell", "lock"):
        print(k, [(r["seq"], r["f0"], r["score"]) for r in out[k][:5]])


if __name__ == "__main__":
    main()
