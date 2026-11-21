"""One pass over the 40 test videos to pick the clips of the 3-video demo (demo selection only, not a result).

For each video, baseline and GapLink run with the live settings (cached YOLO, min_hits 3) inside the VMS engine,
with the adaptive Z_A / Z_B zones that export_demo_clip writes into the clip. Every event is judged against GT
exactly as the lean VMS does (demo/vms_lite_core.py, demo/relink_viz.py):

  relink : windows where GapLink makes 0 wrong links and the baseline makes wrong links / new IDs
  route  : windows where GapLink's violations all have the correct ID and the baseline has wrong-ID / false ones
  lock   : single vehicles (box height >= 40 px) occluded >= 15 frames that GapLink re-links correctly while
           the baseline gives them to another vehicle or a new ID

  .venv\\Scripts\\python.exe -m scripts.mine_demo3 --baseline C-3
Output: runs/demo_mining/demo3_candidates_<BL>.json  (consumed by scripts/export_demo_clip.py --candidates ...)
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

WIN, STEP = 250, 25
MAX_PER_SEQ = 2


class _CachedOnly:
    imgsz = 640

    def __getattr__(self, name):
        raise RuntimeError(f"YOLO cache miss: detector.{name} needed")


def gt_meta_from_seq(seq) -> dict:
    by_f = {}
    for f, sl in seq.by_frame.items():
        r = seq.rows[sl]
        b = np.column_stack([r[:, 2] + r[:, 4] / 2, r[:, 3] + r[:, 5] / 2, r[:, 4], r[:, 5]]).astype(np.float32)
        by_f[int(f)] = (b, r[:, 1].astype(np.int32))
    return {"by_frame": by_f}


def run_method(seq, dets, det_gt_map, assoc, zones, gt_meta):
    from demo.relink_viz import RelinkRecorder
    from demo.vms_lite_core import RouteJudge
    from src.security_analytics import SecurityVMSEngine
    from src.tracker import OnlineTracker
    vms = SecurityVMSEngine(OnlineTracker(associator=assoc, iou_thr=0.3, a_max=95, W=seq.W, H=seq.H, seq_name=seq.name,
                                          min_hits=3), zones=zones, forbidden_routes=[("Z_A", "Z_B")], max_dwell_sec=1e9, fps=25.0)
    rec = RelinkRecorder(None, min_hits=3)
    judge = RouteJudge(gt_meta, seq.W, seq.H)
    za = next((z for z in zones if z.zone_id == "Z_A"), None)
    relink, route = [], []
    for f in range(1, seq.n_frames + 1):
        d = dets.get(f)
        if d is None:
            continue
        res = vms.step(f, d["boxes"], d["hists"], det_gt_map.get(f))
        outs = res["tracks"]
        seen = {k: v["frame"] for k, v in rec.flash.items()}
        rec.update(vms.tracker, res, f)
        for tid, fl in rec.flash.items():
            if fl["frame"] == f and seen.get(tid) != f:
                o = next((o for o in outs if int(o["track_id"]) == tid), None)
                relink.append({"frame": f, "tid": tid, "verdict": fl["verdict"], "gap": fl["gap"],
                               "gt": -1 if o is None else int(o.get("gt_tid", -1)),
                               "box": None if o is None else [round(float(v), 1) for v in o["box"]]})
        for a in res["route_alerts"]:
            v = judge.judge(a, za)
            route.append({"kind": "route", "frame": int(a["frame"]), "track_id": int(a["track_id"]), "verdict": v,
                          "fp": v != "ok", "box": [round(float(x), 1) for x in a["box"]]})
    return relink, route, dict(rec.counts)


def cnt(L, key="verdict"):
    out = {}
    for x in L:
        out[x[key]] = out.get(x[key], 0) + 1
    return out


def windows(N):
    return [(f0, min(N, f0 + WIN - 1)) for f0 in range(1, max(2, N - WIN + 2), STEP)]


def pick(cands):
    cands.sort(key=lambda c: (c["score"], c.get("mean_box_h", 0)), reverse=True)
    out = []
    for c in cands:
        if all(c["f1"] < p["f0"] or c["f0"] > p["f1"] for p in out):
            out.append(c)
        if len(out) >= MAX_PER_SEQ:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="C-3")
    ap.add_argument("--seqs", nargs="*", default=None)
    a = ap.parse_args()

    import torch
    import scripts.eval_live_security as els
    from demo.vms_profile import load_profile, make_baseline, make_gaplink
    from src.io_mot import load_seq

    torch.set_num_threads(1)
    prof = load_profile(ROOT)
    gl = make_gaplink(prof, "cpu")
    els.CACHE_DIR = ROOT / "data" / "cache" / "yolo_test"
    rows = list(csv.DictReader(open(ROOT / "data" / "splits.csv", encoding="utf-8")))
    seqs = a.seqs or [r["seq"] for r in rows if r["split"] == "test"]
    out = {"profile": prof.version, "gaplink": prof.gaplink_label, "baseline": prof.baseline_label(a.baseline),
           "note": "demo selection on the test videos; not a reported result", "relink": [], "route": [], "lock": [],
           "totals": {"gaplink": {}, "baseline": {}}}
    t0 = time.time()
    for si, s in enumerate(seqs, 1):
        seq = load_seq(ROOT / "data" / "mot", s)
        dets = els.extract_or_load_yolo_seq(seq, _CachedOnly())
        det_gt_map, _ = els.match_dets_to_gt_posthoc(seq, dets, iou_match=0.45)
        zones = [z for z in els.build_adaptive_security_zones(seq)[0] if z.zone_id in ("Z_A", "Z_B")]
        gtm = gt_meta_from_seq(seq)
        rg, ug, cg = run_method(seq, dets, det_gt_map, gl, zones, gtm)
        rb, ub, cb = run_method(seq, dets, det_gt_map, make_baseline(a.baseline, prof), zones, gtm)
        for k, v in cg.items():
            out["totals"]["gaplink"][k] = out["totals"]["gaplink"].get(k, 0) + v
        for k, v in cb.items():
            out["totals"]["baseline"][k] = out["totals"]["baseline"].get(k, 0) + v
        N = seq.n_frames
        rel, rou = [], []
        for f0, f1 in windows(N):
            inn = lambda L: [x for x in L if f0 + 15 <= x["frame"] <= f1 - 15]
            g, b = cnt(inn(rg)), cnt(inn(rb))
            gw, bw = g.get("wrong", 0), b.get("wrong", 0)
            if gw == 0 and g.get("ok", 0) >= b.get("ok", 0) and (bw >= 1 or b.get("missed", 0) > g.get("missed", 0)):
                sc = 3 * bw + 2 * (g.get("ok", 0) - b.get("ok", 0)) + (b.get("missed", 0) - g.get("missed", 0))
                hs = [x["box"][3] for x in inn(rg) + inn(rb) if x["box"]]
                if sc > 0:
                    rel.append({"seq": s, "f0": f0, "f1": f1, "score": sc, "mean_box_h": round(float(np.mean(hs)), 1) if hs else 0.0,
                                "gaplink": {k: g.get(k, 0) for k in ("ok", "wrong", "missed")},
                                "baseline": {k: b.get(k, 0) for k in ("ok", "wrong", "missed")},
                                "events_gaplink": inn(rg), "events_baseline": inn(rb)})
            G, B = cnt(inn(ug)), cnt(inn(ub))
            gbad, bbad = G.get("wrong_id", 0) + G.get("false", 0), B.get("wrong_id", 0) + B.get("false", 0)
            if G.get("ok", 0) >= 1 and gbad == 0 and bbad >= 1:
                rou.append({"seq": s, "f0": f0, "f1": f1, "score": 3 * bbad + G.get("ok", 0) - B.get("ok", 0),
                            "gaplink": {k: G.get(k, 0) for k in ("ok", "wrong_id", "false")},
                            "baseline": {k: B.get(k, 0) for k in ("ok", "wrong_id", "false")},
                            "alerts_gaplink": inn(ug), "alerts_baseline": inn(ub)})
        out["relink"] += pick(rel)
        out["route"] += pick(rou)
        # lock: GapLink re-links a vehicle correctly; the baseline mishandles the same GT vehicle near that time
        for e in rg:
            if e["verdict"] != "ok" or e["gap"] < 15 or e["gt"] < 0 or not e["box"] or e["box"][3] < 40:
                continue
            bad = [x for x in rb if x["gt"] == e["gt"] and abs(x["frame"] - e["frame"]) <= 30 and x["verdict"] in ("wrong", "missed")]
            if not bad:
                continue
            t_re = e["frame"]
            t_lost = t_re - e["gap"]
            lock_f = max(2, t_lost - 20)
            out["lock"].append({"seq": s, "f0": max(1, lock_f - 40), "f1": min(N, t_re + 60), "lock_frame": lock_f,
                                "gt_tid": e["gt"], "t_reappear": t_re, "gap": e["gap"], "box_h": e["box"][3],
                                "score": e["gap"] + e["box"][3], "gaplink": "reacquired", "baseline": bad[0]["verdict"]})
        print(f"[{si:02d}/{len(seqs)}] {s}: relink {len(rel)}  route {len(rou)}  lock {len(out['lock'])}  ({time.time() - t0:.0f}s)", flush=True)
    for k in ("relink", "route", "lock"):
        out[k].sort(key=lambda c: (c["score"], c.get("mean_box_h", c.get("box_h", 0))), reverse=True)
    od = ROOT / "runs" / "demo_mining"
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"demo3_candidates_{a.baseline}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("totals (re-link verdicts, test videos):", json.dumps(out["totals"]))
    for k in ("relink", "route", "lock"):
        print(f"--- {k}")
        for i, c in enumerate(out[k][:6]):
            extra = {kk: c[kk] for kk in ("gaplink", "baseline", "gap", "box_h", "mean_box_h") if kk in c}
            print(f"  {i} {c['seq']} [{c['f0']}-{c['f1']}] score {c['score']} {extra}")
    print("->", p)


if __name__ == "__main__":
    main()
