"""GapLink live console: upload any video, process it on the spot, play it with GapLink-Lite only.

    .venv\\Scripts\\python.exe -m demo.gaplink_live            # http://127.0.0.1:8001, lists videos processed before
    .venv\\Scripts\\python.exe -m demo.gaplink_live --fresh    # empty list: only what is uploaded in this session

Upload (drag & drop) -> the file is saved in demo/uploads/ and hashed (SHA-256) -> scripts/prepare_video.prepare()
resamples it to 25 fps, resizes it to 960 px wide and runs YOLOv8n once (same detector settings as the evaluation)
-> demo/live_videos/<name>.mp4 + _dets.npz. Everything on screen (IDs, trails, hidden boxes, target path, ellipse,
zones, alerts) is drawn by the console while the video plays; key O hides all of it to show the untouched video.
No ground truth and no baseline: nothing is scored, the viewer checks the identities by eye.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from demo.gaplink_live_core import MODES, LiveRunner
from demo.vms_lite_core import ROOT, save_zones_to_meta

UPLOAD_DIR = ROOT / "demo" / "uploads"
LIVE_DIR = ROOT / "demo" / "live_videos"
UI_PATH = Path(__file__).with_name("gaplink_live_ui.html")
SPEEDS = (1.0, 0.5, 0.25)
MAX_UPLOAD_MB = 800
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}


def make_gaplink():
    import os
    import torch
    from demo.vms_profile import load_profile, make_gaplink as mk
    prof = load_profile(ROOT)
    dev = os.environ.get("GAPLINK_GL_DEVICE", "cpu")
    if dev == "cpu":
        torch.set_num_threads(int(os.environ.get("GAPLINK_GL_THREADS", "1")))
    return mk(prof, dev), f"GapLink-Lite {prof.version}"


def list_videos(d: Path) -> list[Path]:
    return sorted([p for p in d.glob("*.mp4") if p.with_name(p.stem + "_dets.npz").exists() and not p.stem.endswith("_tmp")],
                  key=lambda p: p.stat().st_mtime)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def probe(p: Path) -> dict:
    cap = cv2.VideoCapture(str(p))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": round(fps, 2), "frames": n, "w": w, "h": h, "seconds": round(n / fps, 1) if fps > 0 else None}


class Session:
    def __init__(self, fresh: bool = False):
        self.lock = threading.Lock()
        self.gl, self.label = make_gaplink()
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        self.videos: list[dict] = [] if fresh else [self._entry(p) for p in list_videos(LIVE_DIR)]
        self.cur: dict | None = None
        self.runner: LiveRunner | None = None
        self.cap = None
        self.frame_idx, self.frame = 0, None
        self.paused, self.speed = True, 1.0
        self.opts = {"ids": True, "trails": True, "ellipse": False, "overlay": True}
        self.mode = "relink"
        self.lock_req: tuple[int, float, float] | None = None
        self.zones: dict[str, list[dict] | None] = {}
        self.ema_ms = None
        self.hist_ver = 0
        self.jobs: list[dict] = []
        self._detector = None
        self._det_lock = threading.Lock()
        if self.videos:
            self.load(self.videos[-1]["id"])

    @staticmethod
    def _entry(p: Path) -> dict:
        info = {}
        mp = p.with_name(p.stem + "_meta.json")
        if mp.exists():
            info = json.loads(mp.read_text(encoding="utf-8"))
        up = info.get("upload", {})
        return {"id": p.stem, "path": p, "file": up.get("file") or info.get("source", {}).get("file", p.name),
                "sha256": up.get("sha256", ""), "seconds": up.get("seconds")}

    # ------------------------------------------------------------ playback
    def load(self, vid: str):
        v = next((v for v in self.videos if v["id"] == vid), None)
        if v is None:
            return
        self.cur, self.lock_req, self.paused = v, None, True
        self._rebuild()

    def _rebuild(self):
        if self.cap is not None:
            self.cap.release()
        self.runner = LiveRunner(self.cur["path"], self.mode, self.gl, self.label,
                                 zones_override=self.zones.get(self.cur["id"]))
        self.cap = cv2.VideoCapture(str(self.cur["path"]))
        self.frame_idx, self.frame = 0, None
        self.hist_ver += 1
        self._advance()

    def _advance(self) -> bool:
        ok, frm = self.cap.read()
        if not ok or frm is None:
            self.paused = True
            return False
        self.frame_idx += 1
        self.frame = frm
        n0 = len(self.runner.links) + len(self.runner.alerts) + len(self.runner.pane.history)
        self.runner.step(self.frame_idx, frm)
        if self.lock_req is not None and self.lock_req[0] == self.frame_idx:
            self.runner.lock_click(self.lock_req[1], self.lock_req[2], self.frame_idx, frm)
            self.hist_ver += 1
        if len(self.runner.links) + len(self.runner.alerts) + len(self.runner.pane.history) != n0:
            self.hist_ver += 1
        return True

    def seek(self, target: int):
        target = int(np.clip(target, 1, max(1, self.runner.n)))
        if target <= self.frame_idx:
            self._rebuild()
        while self.frame_idx < target and self._advance():
            pass

    def tick(self) -> dict:
        with self.lock:
            base = {"jobs": [{k: v for k, v in j.items() if k != "path"} for j in self.jobs[-6:]],
                    "videos": [{"id": v["id"], "file": v["file"], "sha256": v["sha256"], "seconds": v["seconds"]}
                               for v in self.videos], "label": self.label}
            if self.runner is None:
                return {**base, "img": None}
            t0 = time.perf_counter()
            if not self.paused:
                self._advance()
            img = self.runner.draw(self.frame, self.frame_idx, self.opts["ids"], self.opts["trails"],
                                   self.opts["ellipse"], self.opts["overlay"])
            ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            ms = (time.perf_counter() - t0) * 1000.0
            if not self.paused:
                self.ema_ms = ms if self.ema_ms is None else 0.85 * self.ema_ms + 0.15 * ms
            return {**base, "img": base64.b64encode(jpg.tobytes()).decode("ascii"), "frame": self.frame_idx,
                    "total": self.runner.n, "paused": self.paused, "speed": self.speed, "opts": self.opts,
                    "mode": self.mode, "fps": round(1000.0 / self.ema_ms, 1) if self.ema_ms else None,
                    "clip_fps": round(self.runner.fps, 3), "W": self.runner.W, "H": self.runner.H,
                    "cur": {"id": self.cur["id"], "file": self.cur["file"], "sha256": self.cur["sha256"]},
                    "lock_frame": self.lock_req[0] if self.lock_req else None, "stats": self.runner.stats(),
                    "events": self.runner.events(), "hist_ver": self.hist_ver}

    # ------------------------------------------------------------ uploads
    def detector(self):
        with self._det_lock:
            if self._detector is None:
                from scripts.prepare_video import DETECTOR_CFG
                from src.live_detector import LiveVehicleDetector
                self._detector = LiveVehicleDetector(weights=ROOT / DETECTOR_CFG["weights"], imgsz=DETECTOR_CFG["imgsz"],
                                                     conf_thr=DETECTOR_CFG["conf_thr"], nms_iou=DETECTOR_CFG["nms_iou"])
            return self._detector

    def start_job(self, path: Path, orig_name: str, sha: str, size: int) -> dict:
        info = probe(path)
        job = {"id": f"j{len(self.jobs) + 1}", "file": orig_name, "sha256": sha, "mb": round(size / 1e6, 1),
               "src": info, "state": "queued", "done": 0, "total": 0, "error": None, "video": None, "path": str(path)}
        self.jobs.append(job)
        threading.Thread(target=self._run_job, args=(job, path), daemon=True).start()
        return job

    def _run_job(self, job: dict, path: Path):
        from scripts.prepare_video import prepare, safe_name
        try:
            job["state"] = "processing"
            t0 = time.time()
            name = safe_name(Path(job["file"]).stem) + time.strftime("_%H%M%S")

            def prog(d, t):
                job["done"], job["total"] = d, t
            info = prepare(path, name=name, out_dir=LIVE_DIR, progress=prog, detector=self.detector())
            mp = LIVE_DIR / f"{name}_meta.json"
            meta = json.loads(mp.read_text(encoding="utf-8"))
            meta["upload"] = {"file": job["file"], "sha256": job["sha256"], "bytes": int(job["mb"] * 1e6),
                              "seconds": job["src"]["seconds"], "processed_s": round(time.time() - t0, 1)}
            mp.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
            with self.lock:                     # list and load the video first, then report the job as done
                self.videos.append(self._entry(LIVE_DIR / f"{name}.mp4"))
                self.load(name)
            job.update(state="done", video=name, seconds=round(time.time() - t0, 1), frames=info["frames"])
        except Exception as e:          # shown in the UI; the console keeps running
            job.update(state="error", error=str(e)[:300])


from fastapi import Request  # noqa: E402  (module level: annotations are strings here)
from pydantic import BaseModel  # noqa: E402


class Act(BaseModel):
    action: str
    vid: Optional[str] = None
    value: Optional[float] = None
    flag: Optional[bool] = None
    key: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    zone_id: Optional[str] = None
    points: Optional[List[List[float]]] = None
    mode: Optional[str] = None


SESSION: Session | None = None
FRESH = False


def sess() -> Session:
    global SESSION
    if SESSION is None:
        SESSION = Session(fresh=FRESH)
    return SESSION


def build_app():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    app = FastAPI(title="GapLink live")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return UI_PATH.read_text(encoding="utf-8")

    @app.post("/api/upload")
    async def upload(request: Request, name: str):
        """Raw body upload (no multipart dependency): the file name comes in the query string."""
        s = sess()
        ext = Path(name).suffix.lower()
        if ext not in VIDEO_EXT:
            return JSONResponse({"ok": False, "error": f"không hỗ trợ định dạng {ext or '(không có đuôi)'}"}, 400)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        from scripts.prepare_video import safe_name
        dst = UPLOAD_DIR / f"{safe_name(Path(name).stem)}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
        h, size = hashlib.sha256(), 0
        with open(dst, "wb") as fh:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1e6:
                    fh.close()
                    dst.unlink(missing_ok=True)
                    return JSONResponse({"ok": False, "error": f"file lớn hơn {MAX_UPLOAD_MB} MB"}, 400)
                h.update(chunk)
                fh.write(chunk)
        if size == 0:
            dst.unlink(missing_ok=True)
            return JSONResponse({"ok": False, "error": "file rỗng"}, 400)
        job = s.start_job(dst, Path(name).name, h.hexdigest(), size)
        return {"ok": True, "job": {k: v for k, v in job.items() if k != "path"}}

    @app.get("/api/history")
    def history():
        s = sess()
        with s.lock:
            return {"ver": s.hist_ver, "items": s.runner.history() if s.runner else []}

    @app.post("/api/act")
    def act(a: Act):
        s = sess()
        with s.lock:
            if a.action == "load" and a.vid:
                s.load(a.vid)
                return {"ok": True}
            if s.runner is None:
                return {"ok": False}
            if a.action == "play":
                s.paused = not s.paused if a.flag is None else not a.flag
            elif a.action == "restart":
                s._rebuild()
                s.paused = False
            elif a.action == "step" and a.value is not None:
                s.paused = True
                s.seek(s.frame_idx + int(a.value))
            elif a.action == "seek" and a.value is not None:
                s.seek(int(a.value))
            elif a.action == "speed" and a.value in SPEEDS:
                s.speed = float(a.value)
            elif a.action == "opt" and a.key in s.opts and a.flag is not None:
                s.opts[a.key] = bool(a.flag)
            elif a.action == "mode" and a.mode in MODES:
                s.mode = a.mode
                at = s.frame_idx
                s._rebuild()
                s.seek(at)
            elif a.action == "lock" and a.x is not None and a.y is not None:
                px, py = a.x * s.runner.W, a.y * s.runner.H
                tid = s.runner.lock_click(px, py, s.frame_idx, s.frame)
                s.lock_req = (s.frame_idx, px, py) if tid is not None else None
                s.hist_ver += 1
                return {"ok": True, "tid": tid}
            elif a.action == "unlock":
                s.runner.unlock()
                s.lock_req = None
                s.hist_ver += 1
            elif a.action == "zone" and a.zone_id in ("Z_A", "Z_B") and a.points and len(a.points) >= 3:
                cur = [{"zone_id": z.zone_id, "points": [list(p) for p in z.points]} for z in s.runner.zones]
                s.zones[s.cur["id"]] = [z for z in cur if z["zone_id"] != a.zone_id] + [{"zone_id": a.zone_id,
                                                                                       "points": a.points}]
                at = s.frame_idx
                s._rebuild()
                s.seek(at)
            elif a.action == "save_zones" and s.zones.get(s.cur["id"]):
                save_zones_to_meta(s.cur["path"], s.zones[s.cur["id"]])
            elif a.action == "reset_zones":
                s.zones.pop(s.cur["id"], None)
                s._rebuild()
            return {"ok": True}

    @app.get("/api/stream")
    async def stream():
        s = sess()

        async def gen():
            while True:
                t = time.perf_counter()
                pkt = await asyncio.to_thread(s.tick)
                yield f"data: {json.dumps(pkt)}\n\n"
                fps = s.runner.fps if s.runner else 25.0
                period = 1.0 / fps / max(0.05, s.speed)
                idle = s.paused or s.runner is None
                await asyncio.sleep(max(0.004, (0.15 if idle else period) - (time.perf_counter() - t)))

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def main():
    global FRESH
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--fresh", action="store_true", help="start with an empty video list")
    a = ap.parse_args()
    FRESH = a.fresh
    import uvicorn
    app = build_app()
    sess()
    print(f"GapLink live: http://127.0.0.1:{a.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
