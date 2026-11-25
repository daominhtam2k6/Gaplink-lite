"""Prepare any video for the GapLink-VMS console: trim, resize, run YOLOv8n once and cache the detections.

    .venv\\Scripts\\python.exe -m scripts.prepare_video demo\\uploads\\my_clip.mp4 --end 8.5
    .venv\\Scripts\\python.exe -m scripts.prepare_video demo\\uploads\\taxi.mp4 --name taxi --scenario lock

writes demo/user_videos/<name>.mp4, <name>_dets.npz and <name>_meta.json, plus runs/prepare_video_<name>/info.json.
The GapLink live console (python -m demo.gaplink_live) does the same when a video is uploaded.

YOLOv8n is only the input of the trackers, with the same settings as the live evaluation (imgsz 640, conf 0.28,
NMS 0.52, four-wheel vehicle classes). Nothing of GapLink-Lite or of the baselines is changed. A user video has no
ground truth, so the console shows its events as unverified.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "demo" / "user_videos"
SCENARIOS = ("relink", "route", "lock")
DETECTOR_CFG = {"weights": "release/yolov8n.pt", "imgsz": 640, "conf_thr": 0.28, "nms_iou": 0.52}


# ------------------------------------------------------------------------------------------------ pure helpers
def safe_name(s: str) -> str:
    n = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_").lower()
    return n[:48] or "video"


def target_size(w: int, h: int, width: int) -> tuple[int, int]:
    """Resize to `width` (never upscale), keep the aspect ratio, even dimensions for the video codec."""
    if w <= width:
        return w - w % 2, h - h % 2
    nh = int(round(h * width / w))
    return width - width % 2, nh - nh % 2


def frame_range(n_frames: int, fps: float, start: float | None, end: float | None) -> tuple[int, int]:
    """0-based [first, last) frame indices of the trimmed clip."""
    a = 0 if start is None else max(0, int(round(start * fps)))
    b = n_frames if end is None else min(n_frames, int(round(end * fps)))
    if b - a < 2:
        raise ValueError(f"empty clip: frames {a}..{b} of {n_frames}")
    return a, b


def default_zones() -> list[dict]:
    """Two vertical bands for the forbidden-route mode; they are meant to be redrawn in the console."""
    return [{"zone_id": "Z_A", "name": "Restricted origin A", "role": "entry",
             "points": [[0.02, 0.05], [0.30, 0.05], [0.30, 0.95], [0.02, 0.95]]},
            {"zone_id": "Z_B", "name": "Restricted zone B", "role": "restricted",
             "points": [[0.70, 0.05], [0.98, 0.05], [0.98, 0.95], [0.70, 0.95]]}]


def build_meta(name: str, scenario: str, title: str | None, src: Path, fps: float, first: int, last: int,
               orig_wh: tuple[int, int], out_wh: tuple[int, int]) -> dict:
    return {"scenario": scenario, "title": title or name, "no_gt": True, "zones": default_zones(),
            "forbidden_routes": [["Z_A", "Z_B"]], "max_dwell_sec": 5.0, "key_moments": [], "lock": [],
            "source": {"file": src.name, "fps": round(float(fps), 3), "first_frame": int(first), "last_frame": int(last - 1),
                       "orig_size": list(orig_wh), "size": list(out_wh)},
            "detector": DETECTOR_CFG}


def pack_dets(per_frame: list[dict]) -> dict[str, np.ndarray]:
    """Per-frame detections (clip frames 1..n) -> the ragged arrays read by demo.render_security_demo.load_clip_dets_cache."""
    offs, boxes, hists = [0], [], []
    for d in per_frame:
        b = np.asarray(d["boxes"], np.float32).reshape(-1, 4)
        h = np.asarray(d["hists"], np.float32).reshape(-1, 128)
        assert len(b) == len(h)
        boxes.append(b)
        hists.append(h)
        offs.append(offs[-1] + len(b))
    return {"frames": np.arange(1, len(per_frame) + 1, dtype=np.int32), "offsets": np.asarray(offs, np.int32),
            "boxes": np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
            "hists": np.concatenate(hists) if hists else np.zeros((0, 128), np.float32)}


def det_stats(per_frame: list[dict]) -> dict:
    n = np.array([len(d["boxes"]) for d in per_frame])
    return {"frames": int(len(n)), "boxes_total": int(n.sum()), "boxes_per_frame_mean": round(float(n.mean()), 3) if len(n) else 0.0,
            "frames_without_detection": int((n == 0).sum())}


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------------------------------------ main
def resample_keep(n_src: int, fps_src: float, fps_out: float | None) -> list[bool]:
    """Which source frames to keep to play at fps_out (None or a higher target: keep all)."""
    if not fps_out or fps_src <= fps_out + 0.5:
        return [True] * n_src
    keep, nxt = [], 0.0
    for i in range(n_src):
        t = i / fps_src
        if t + 1e-9 >= nxt:
            keep.append(True)
            nxt += 1.0 / fps_out
        else:
            keep.append(False)
    return keep


def prepare(src: Path, name: str | None = None, start: float | None = None, end: float | None = None,
            scenario: str = "relink", title: str | None = None, width: int = 960, batch_size: int = 16,
            target_fps: float | None = 25.0, out_dir: Path | None = None, progress=None, detector=None) -> dict:
    """Trim, resample to target_fps (UA-DETRAC is 25 fps), resize, run YOLOv8n once; progress(done, total) is called."""
    name = safe_name(name or src.stem)
    cap = cv2.VideoCapture(str(src))
    fps_src = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    fps_src = fps_src if 5 <= fps_src <= 120 else 25.0
    n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n_all <= 0 or w0 <= 0:
        cap.release()
        raise ValueError("cannot read the video (unsupported format?)")
    first, last = frame_range(n_all, fps_src, start, end)
    keep = resample_keep(last, fps_src, target_fps)
    fps = float(target_fps) if target_fps and fps_src > target_fps + 0.5 else fps_src
    W, H = target_size(w0, h0, width)
    if detector is None:
        from src.live_detector import LiveVehicleDetector
        detector = LiveVehicleDetector(weights=ROOT / DETECTOR_CFG["weights"], imgsz=DETECTOR_CFG["imgsz"],
                                       conf_thr=DETECTOR_CFG["conf_thr"], nms_iou=DETECTOR_CFG["nms_iou"])
    od = OUT_DIR if out_dir is None else Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    stem = od / name
    tmp = stem.with_name(name + "_tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    t0 = time.time()
    per_frame: list[dict] = []
    batch: list[np.ndarray] = []
    total = sum(keep[first:last])

    def flush():
        for r in detector.detect_batch(batch):
            per_frame.append({"boxes": r["boxes"], "hists": r["hists"]})
        batch.clear()
        if progress:
            progress(len(per_frame), total)

    for i in range(last):                  # read from the start: seeking in mp4 files is not frame-exact
        ok, frm = cap.read()
        if not ok or frm is None:
            break
        if i < first or not keep[i]:
            continue
        if frm.shape[1] != W or frm.shape[0] != H:
            frm = cv2.resize(frm, (W, H), interpolation=cv2.INTER_AREA)
        vw.write(frm)
        batch.append(frm)
        if len(batch) >= batch_size:
            flush()
    if batch:
        flush()
    cap.release()
    vw.release()
    if not per_frame:
        tmp.unlink(missing_ok=True)
        raise ValueError("no frame could be read")
    np.savez_compressed(stem.with_name(name + "_dets.npz"), **pack_dets(per_frame))
    meta = build_meta(name, scenario, title, src, fps, first, first + len(per_frame), (w0, h0), (W, H))
    meta["source"]["fps_original"] = round(fps_src, 3)
    stem.with_name(name + "_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(stem.with_suffix(".mp4"))
    stats = det_stats(per_frame)
    run_dir = ROOT / "runs" / f"prepare_video_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    info = {"name": name, "path": str(stem.with_suffix(".mp4")), "frames": len(per_frame), "fps": fps, "size": [W, H],
            "args": {"src": str(src), "start": start, "end": end, "scenario": scenario, "width": width,
                     "target_fps": target_fps}, "detector": DETECTOR_CFG,
            "git_commit": _git_commit(), "seconds": round(time.time() - t0, 1), "meta": meta, "detections": stats}
    (run_dir / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--name", default=None, help="clip name (default: file name)")
    ap.add_argument("--start", type=float, default=None, help="start time in seconds")
    ap.add_argument("--end", type=float, default=None, help="end time in seconds")
    ap.add_argument("--scenario", choices=SCENARIOS, default="relink", help="mode shown first (can be changed in the console)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--width", type=int, default=960, help="output width in pixels (UA-DETRAC is 960x540)")
    ap.add_argument("--fps", type=float, default=25.0, help="resample faster videos to this frame rate (0: keep)")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args(argv)
    src = a.video if a.video.is_absolute() else (Path.cwd() / a.video)
    if not src.exists():
        raise SystemExit(f"video not found: {src}")
    try:
        info = prepare(src, a.name, a.start, a.end, a.scenario, a.title, a.width, a.batch, a.fps or None,
                       progress=lambda d, t: print(f"  {d}/{t} frames", flush=True) if d % 64 < 16 else None)
    except ValueError as e:
        raise SystemExit(str(e))
    st = info["detections"]
    print(f"wrote {info['path']}  {info['frames']} frames {info['size'][0]}x{info['size'][1]} @ {info['fps']:g} fps  "
          f"{st['boxes_per_frame_mean']} boxes/frame, {st['frames_without_detection']} frames without detection  "
          f"({info['seconds']} s)")
    print("open the console with:  .venv\\Scripts\\python.exe -m demo.gaplink_live")


if __name__ == "__main__":
    main()
