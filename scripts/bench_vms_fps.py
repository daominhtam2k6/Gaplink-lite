"""Measure VMS per-frame cost on this machine (cached YOLO dets, so detector time is excluded).

  .venv\\Scripts\\python.exe -m scripts.bench_vms_fps            # all showcase clips, cuda and cpu
Writes runs/bench_vms_fps/results.json and prints ms/frame per stage and device.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from demo.clip_meta import load_clip_meta, zones_from_meta
from demo.render_security_demo import _zoom_at, draw_vms_overlay
from demo.vms_profile import load_profile, make_baseline, make_gaplink, resolve_root
from src.security_analytics import SecurityVMSEngine
from src.tracker import OnlineTracker


def bench_clip(vp: Path, gl_assoc, bl_assoc, n_max: int = 300) -> dict:
    import demo.security_vms_app as app
    meta = load_clip_meta(vp)
    dets = app.load_clip_dets_cache(vp)
    gtm = app.load_clip_gt_meta(vp)
    cap = cv2.VideoCapture(str(vp))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def eng(assoc):
        z, fr, md = zones_from_meta(meta)
        return SecurityVMSEngine(OnlineTracker(associator=assoc, W=W, H=H, min_hits=3), zones=z,
                                 forbidden_routes=fr, max_dwell_sec=md, fps=25)

    a, b = eng(bl_assoc), eng(gl_assoc)
    mode = meta.get("scenario", "all")
    T = {k: 0.0 for k in ("read", "baseline_step", "gaplink_step", "draw", "zoom", "jpeg")}
    n = 0
    for f in range(1, n_max + 1):
        t = time.perf_counter(); ok, frm = cap.read(); T["read"] += time.perf_counter() - t
        if not ok or f not in dets:
            break
        d = dets[f]
        g = app.match_frame_dets_to_gt(d["boxes"], gtm, f)
        t = time.perf_counter(); ra = a.step(f, d["boxes"], d["hists"], det_gt_tids=g, frame_bgr=frm); T["baseline_step"] += time.perf_counter() - t
        t = time.perf_counter(); rb = b.step(f, d["boxes"], d["hists"], det_gt_tids=g, frame_bgr=frm)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        T["gaplink_step"] += time.perf_counter() - t
        t = time.perf_counter()
        vl = draw_vms_overlay(frm, a, ra, "L", scenario_mode=mode)
        vr = draw_vms_overlay(frm, b, rb, "R", scenario_mode=mode)
        T["draw"] += time.perf_counter() - t
        t = time.perf_counter(); vl = _zoom_at(vl, 0.5, 0.5, 1.9); vr = _zoom_at(vr, 0.5, 0.5, 1.9); T["zoom"] += time.perf_counter() - t
        t = time.perf_counter()
        out = np.concatenate([cv2.resize(vl, (960, 540)), cv2.resize(vr, (960, 540))], axis=1)
        cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        T["jpeg"] += time.perf_counter() - t
        n += 1
    res = {k: round(v / max(1, n) * 1000.0, 2) for k, v in T.items()}
    res["total_ms"] = round(sum(res.values()), 2)
    res["fps"] = round(1000.0 / max(1e-6, res["total_ms"]), 1)
    res["frames"] = n
    return res


def main():
    root = resolve_root()
    prof = load_profile(root)
    clips = sorted((root / "demo" / "sample_videos").glob("showcase_*.mp4"))
    devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu:1", "cpu:2", "cpu:4"]
    out = {}
    for dev in devices:
        if dev.startswith("cpu"):
            torch.set_num_threads(int(dev.split(":")[1]))
        gl = make_gaplink(prof, dev.split(":")[0])
        for vp in clips:
            r = bench_clip(vp, gl, make_baseline("C-3", prof))
            out[f"{vp.stem}|{dev}"] = r
            print(f"{vp.stem:20s} GapLink on {dev:6s}: " + "  ".join(f"{k}={v}" for k, v in r.items()), flush=True)
    od = root / "runs" / "bench_vms_fps"
    od.mkdir(parents=True, exist_ok=True)
    (od / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
