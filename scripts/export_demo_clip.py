"""Export a mined demo moment as a self-contained showcase clip in demo/sample_videos/.

    python -m scripts.export_demo_clip --scenario route --rank 0 --name showcase_1_route
writes <name>.mp4, <name>_dets.npz, <name>_gt.npz, <name>_meta.json (zones, key moments, lock schedule, expected counts).
The clip is cut from the raw test sequence images; detections are the cached live YOLOv8n boxes (no re-detection).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TITLES = {"route": "Forbidden route Z_A -> Z_B through occlusion", "dwell": "Stopped vehicle hidden by passing traffic",
          "lock": "Locked moving vehicle re-acquired after occlusion",
          "relink": "Identity recovery after occlusion"}



class _CachedOnly:
    """Stand-in detector: only the cache key (imgsz) is needed; a cache miss raises."""
    imgsz = 640

    def __getattr__(self, name):
        raise RuntimeError(f"YOLO cache miss: detector.{name} needed - run the live eval/mining first")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=None,
                    help="default: runs/demo_mining/relink_candidates_C-3.json for relink, candidates_C-3.json otherwise")
    ap.add_argument("--scenario", choices=["route", "dwell", "lock", "relink"], required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    if a.candidates is None:
        a.candidates = ("runs/demo_mining/relink_candidates_C-3.json" if a.scenario == "relink"
                        else "runs/demo_mining/candidates_C-3.json")

    import scripts.eval_live_security as els
    from src.io_mot import load_seq

    cand = json.loads((ROOT / a.candidates).read_text(encoding="utf-8"))
    c = cand[a.scenario][a.rank]
    seq = load_seq(ROOT / "data" / "mot", c["seq"])
    f0, f1 = int(c["f0"]), int(c["f1"])
    W, H = seq.W, seq.H
    els.CACHE_DIR = ROOT / "data" / "cache" / "yolo_test"
    dets = els.extract_or_load_yolo_seq(seq, detector=_CachedOnly())  # cache hit required
    out_dir = ROOT / "demo" / "sample_videos"
    stem = out_dir / a.name

    # 1. video from raw frames
    tmp = stem.with_name(stem.name + "_tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (W, H))
    for f in range(f0, f1 + 1):
        p = seq.img_dir / f"img{f:05d}.jpg"
        if not p.exists():
            p = seq.img_dir / f"image{f:06d}.jpg"
        im = cv2.imread(str(p))
        vw.write(im if im is not None else np.zeros((H, W, 3), np.uint8))
    vw.release()
    tmp.replace(stem.with_suffix(".mp4"))

    # 2. detections (re-indexed to clip frames 1..n)
    frames, offs, boxes, hists = [], [0], [], []
    for i, f in enumerate(range(f0, f1 + 1), 1):
        d = dets[f]
        frames.append(i); boxes.append(d["boxes"]); hists.append(d["hists"]); offs.append(offs[-1] + len(d["boxes"]))
    np.savez_compressed(stem.with_name(stem.name + "_dets.npz"), frames=np.array(frames, np.int32), offsets=np.array(offs, np.int32),
                        boxes=np.concatenate(boxes).astype(np.float32), hists=np.concatenate(hists).astype(np.float32))

    # 3. ground truth (post-hoc scoring only), boxes as (cx, cy, w, h)
    g_frames, g_offs, g_boxes, g_tids = [], [0], [], []
    for i, f in enumerate(range(f0, f1 + 1), 1):
        sl = seq.by_frame.get(f)
        rows = seq.rows[sl] if sl is not None else np.zeros((0, 9), np.float32)
        b = np.column_stack([rows[:, 2] + rows[:, 4] / 2, rows[:, 3] + rows[:, 5] / 2, rows[:, 4], rows[:, 5]]) if len(rows) else np.zeros((0, 4))
        g_frames.append(i); g_boxes.append(b.astype(np.float32)); g_tids.append(rows[:, 1].astype(np.int32) if len(rows) else np.zeros(0, np.int32))
        g_offs.append(g_offs[-1] + len(b))
    np.savez_compressed(stem.with_name(stem.name + "_gt.npz"), frames=np.array(g_frames, np.int32), offsets=np.array(g_offs, np.int32),
                        boxes=np.concatenate(g_boxes), tids=np.concatenate(g_tids), ignore=np.asarray(seq.ignore, np.float32),
                        orig_seq=np.array(c["seq"]), f_start=np.array(f0))

    # 4. meta: zones (same adaptive zones used when mining), key moments, lock schedule
    zones = els.build_adaptive_security_zones(seq)[0]
    zj = [{"zone_id": z.zone_id, "name": z.name, "role": z.role,
           "points": [[round(float(x) / W, 4), round(float(y) / H, 4)] for x, y in z.to_pixel_array(W, H)]} for z in zones]

    def centre_of_track_gt(gt_tid, f):
        tr = seq.tracks.get(int(gt_tid))
        if tr is None:
            return None
        fr = list(tr["frames"])
        if f not in fr:
            return None
        b = tr["boxes"][fr.index(f)]
        return float(b[0]), float(b[1])

    kms, lock = [], []
    if a.scenario == "relink":
        pool = ([dict(x, who="GapLink re-link") for x in c["events_gaplink"] if x["verdict"] == "ok"]
                + [dict(x, who="baseline " + x["verdict"]) for x in c["events_baseline"] if x["verdict"] in ("wrong", "missed")])
        for x in sorted(pool, key=lambda x: x["frame"]):
            cx, cy = (x["box"][0], x["box"][1]) if x.get("box") else (W / 2, H / 2)
            kms.append({"frame": int(x["frame"]) - f0 + 1, "cx": round(cx / W, 4), "cy": round(cy / H, 4), "label": x["who"]})
    elif a.scenario in ("route", "dwell"):
        pool = [x for x in c["alerts_baseline"] if x["fp"]] + [x for x in c["alerts_gaplink"] if not x["fp"]]
        for x in sorted(pool, key=lambda x: x["frame"]):
            cx, cy = (x["box"][0], x["box"][1]) if x.get("box") else (W / 2, H / 2)
            label = "baseline false alarm" if x in c["alerts_baseline"] else "GapLink true alert"
            kms.append({"frame": int(x["frame"]) - f0 + 1, "cx": round(cx / W, 4), "cy": round(cy / H, 4), "label": label})
    else:
        ctr = centre_of_track_gt(c["gt_tid"], c["t_reappear"]) or (W / 2, H / 2)
        kms.append({"frame": int(c["t_reappear"]) - f0 + 1, "cx": round(ctr[0] / W, 4), "cy": round(ctr[1] / H, 4),
                    "label": "target re-appears"})
        lc = centre_of_track_gt(c["gt_tid"], c["lock_frame"])
        if lc:
            lock.append([int(c["lock_frame"]) - f0 + 1, round(lc[0], 1), round(lc[1], 1)])
    # merge key moments closer than 2 s
    merged = []
    for k in kms:
        if merged and k["frame"] - merged[-1]["frame"] < 50:
            continue
        merged.append(k)
    meta = {"scenario": a.scenario, "title": TITLES[a.scenario], "zones": zj, "forbidden_routes": [["Z_A", "Z_B"]],
            "max_dwell_sec": 5.0, "key_moments": merged, "lock": lock,
            "source": {"seq": c["seq"], "f0": f0, "f1": f1, "candidates": a.candidates, "rank": a.rank},
            "expected": {k: c[k] for k in ("gaplink", "baseline") if k in c}}
    stem.with_name(stem.name + "_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"exported {stem}.mp4 ({f1 - f0 + 1} frames) from {c['seq']} [{f0}-{f1}], key moments {[k['frame'] for k in merged]}")


if __name__ == "__main__":
    main()
