"""Find test-set windows that show identity recovery clearly: GapLink re-links occluded vehicles correctly
while the baseline links them to the wrong vehicle (or gives them a new ID).

Runs both live trackers (cached YOLO detections, same settings as the VMS) over the 40 test videos and
scores every re-link with demo.relink_viz.RelinkRecorder (verdicts judged against GT, occlusions >= MIN_GAP).
Demo selection only: nothing here feeds a reported number.

  .venv\\Scripts\\python.exe -m scripts.mine_relink_moments --baseline C-3
Output: runs/demo_mining/relink_candidates_<BL>.json
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

WIN, STEP = 250, 25          # 10 s windows
MAX_PER_SEQ = 2


class _CachedOnly:
    imgsz = 640

    def __getattr__(self, name):
        raise RuntimeError(f"YOLO cache miss: detector.{name} needed")


def run_seq(seq, dets, det_gt_map, assoc, N):
    from demo.relink_viz import ExplainedAssociator, RelinkRecorder
    from src.tracker import OnlineTracker
    ex = ExplainedAssociator(assoc)
    trk = OnlineTracker(associator=ex, iou_thr=0.3, a_max=95, W=seq.W, H=seq.H, seq_name=seq.name, min_hits=3)
    rec = RelinkRecorder(ex, min_hits=3)
    trk.event_hook = rec.on_event
    verdicts = []
    for f in range(1, N + 1):
        d = dets.get(f)
        if d is None:
            continue
        outs = trk.step(f, d["boxes"], d["hists"], det_gt_map.get(f))
        before = {k: v["frame"] for k, v in rec.flash.items()}
        rec.update(trk, {"tracks": outs}, f)
        for tid, fl in rec.flash.items():
            if fl["frame"] == f and before.get(tid) != f:
                b = next((o["box"] for o in outs if int(o["track_id"]) == tid), None)
                verdicts.append({"frame": f, "tid": tid, "verdict": fl["verdict"], "gap": fl["gap"],
                                 "box": None if b is None else [round(float(v), 1) for v in b]})
    return verdicts, dict(rec.counts)


def score_window(gl, bl):
    c = lambda L, v: sum(1 for x in L if x["verdict"] == v)
    gok, gw, gm = c(gl, "ok"), c(gl, "wrong"), c(gl, "missed")
    bok, bw, bm = c(bl, "ok"), c(bl, "wrong"), c(bl, "missed")
    ok_rule = gw == 0 and gok >= bok and (bw >= 1 or bm > gm)
    s = 3 * bw + 2 * (gok - bok) + (bm - gm) - 3 * gw
    return ok_rule, s, {"gaplink": {"ok": gok, "wrong": gw, "missed": gm}, "baseline": {"ok": bok, "wrong": bw, "missed": bm}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="C-3")
    ap.add_argument("--seqs", nargs="*", default=None)
    a = ap.parse_args()

    import torch
    import scripts.eval_live_security as els
    from demo.vms_profile import load_profile, make_baseline, make_gaplink
    from src.io_mot import load_seq

    prof = load_profile(ROOT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gl = make_gaplink(prof, device)
    els.CACHE_DIR = ROOT / "data" / "cache" / "yolo_test"
    rows = list(csv.DictReader(open(ROOT / "data" / "splits.csv", encoding="utf-8")))
    seqs = a.seqs or [r["seq"] for r in rows if r["split"] == "test"]

    out = {"profile": prof.version, "gaplink": prof.gaplink_label, "baseline": prof.baseline_label(a.baseline),
           "note": "demo selection on the test videos; not a reported result", "totals": {}, "relink": []}
    tot = {"gaplink": {"ok": 0, "wrong": 0, "missed": 0, "unk": 0}, "baseline": {"ok": 0, "wrong": 0, "missed": 0, "unk": 0}}
    t0 = time.time()
    for si, s in enumerate(seqs, 1):
        seq = load_seq(ROOT / "data" / "mot", s)
        dets = els.extract_or_load_yolo_seq(seq, _CachedOnly())
        det_gt_map, _ = els.match_dets_to_gt_posthoc(seq, dets, iou_match=0.45)
        N = seq.n_frames
        v_gl, c_gl = run_seq(seq, dets, det_gt_map, gl, N)
        v_bl, c_bl = run_seq(seq, dets, det_gt_map, make_baseline(a.baseline, prof), N)
        for k in tot["gaplink"]:
            tot["gaplink"][k] += c_gl[k]
            tot["baseline"][k] += c_bl[k]
        cands = []
        for f0 in range(1, max(2, N - WIN + 2), STEP):
            f1 = min(N, f0 + WIN - 1)
            g = [x for x in v_gl if f0 + 15 <= x["frame"] <= f1 - 15]
            b = [x for x in v_bl if f0 + 15 <= x["frame"] <= f1 - 15]
            ok, sc, cnt = score_window(g, b)
            if ok and sc > 0:
                hs = [x["box"][3] for x in g + b if x["box"]]
                cands.append({"seq": s, "f0": f0, "f1": f1, "score": sc, "mean_box_h": round(float(np.mean(hs)), 1) if hs else 0.0,
                              **cnt, "events_gaplink": g, "events_baseline": b})
        cands.sort(key=lambda c: (c["score"], c["mean_box_h"]), reverse=True)
        picked = []
        for c in cands:
            if all(c["f1"] < p["f0"] or c["f0"] > p["f1"] for p in picked):
                picked.append(c)
            if len(picked) >= MAX_PER_SEQ:
                break
        out["relink"].extend(picked)
        print(f"[{si:02d}/{len(seqs)}] {s}: GL {c_gl}  BL {c_bl}  windows {len(picked)} ({time.time() - t0:.0f}s)", flush=True)
    out["relink"].sort(key=lambda c: (c["score"], c["mean_box_h"]), reverse=True)
    out["totals"] = tot
    od = ROOT / "runs" / "demo_mining"
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"relink_candidates_{a.baseline}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("totals (gap >= MIN_GAP, test videos):", json.dumps(tot))
    for i, c in enumerate(out["relink"][:8]):
        print(i, c["seq"], c["f0"], c["f1"], "score", c["score"], "h", c["mean_box_h"], c["gaplink"], c["baseline"])
    print("->", p)


if __name__ == "__main__":
    main()
