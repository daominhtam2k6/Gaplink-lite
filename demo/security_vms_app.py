"""GapLink-VMS: Interactive Local Web Security Camera Console (FastAPI + SSE + HTML5 Canvas).

Run locally:
    D:\\gaplink\\.venv\\Scripts\\python.exe D:\\gaplink\\demo\\security_vms_app.py --port 8000
Then open http://127.0.0.1:8000 in any browser.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import sys
import threading
import time
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
import numpy as np
from pydantic import BaseModel
import torch
import uvicorn
import yaml

ROOT = Path(r"D:\gaplink")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.render_security_demo import ZONE_DEFAULT_COLORS, default_preset_zones, draw_vms_overlay
from src.baselines import KalmanMahalanobisBaseline
from src.live_detector import LiveVehicleDetector
from src.model.gaplink import GapLinkAssociator, GapLinkLite
from src.security_analytics import PolygonZone, SecurityVMSEngine
from src.tracker import OnlineTracker

SAMPLE_DIR = ROOT / "demo" / "sample_videos"
UPLOAD_DIR = ROOT / "demo" / "uploads"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PRESET_CATALOG = [
    {
        "id": "detrac_MVI_40855_intersection.mp4",
        "label": "[Ngã tư 4 hướng] MVI_40855 — Xe buýt cắt ngang giao lộ (Raw Video)",
        "key": "40855",
    },
    {
        "id": "detrac_MVI_40864_rain_buses.mp4",
        "label": "[Đại lộ Trời mưa] MVI_40864 — Xe buýt 2 tầng che khuất liên tục (Raw Video)",
        "key": "40864",
    },
    {
        "id": "detrac_MVI_39311_gore_zone.mp4",
        "label": "[Đảo phân làn cấm đỗ] MVI_39311 — Dừng đỗ trái phép & Nhập làn (Raw Video)",
        "key": "39311",
    },
    {
        "id": "ext_security_parking_gate.mp4",
        "label": "[Camera An ninh Cổng & Bãi xe — Ngoài tập Train/Test] Zero-Shot CCTV",
        "key": "parking",
    },
    {
        "id": "ext_parking_aisle_cctv.mp4",
        "label": "[Camera Làn Nội bộ Bãi xe — Ngoài tập Train/Test] Zero-Shot CCTV",
        "key": "parking",
    },
]


class LiveVMSSession:
    """Thread-safe state manager running Live YOLOv8n + (C-3 vs GapLink-Lite) on any raw .mp4 camera feed."""

    def __init__(self):
        self.lock = threading.Lock()
        rel_dir = ROOT / "release"
        self.tcfg = yaml.safe_load((rel_dir / "tracker.yaml").read_text(encoding="utf-8"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(rel_dir / "final.pt", map_location=self.device, weights_only=False)
        self.model = GapLinkLite(use_ctx=True, use_color=True, gap_aware=True).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.gl_assoc = GapLinkAssociator(
            self.model, device=self.device, tau=float(self.tcfg["gaplink_tau"]), calib=None
        )
        self.c3_theta = float(self.tcfg["c3_theta"])
        self.detector_4w = LiveVehicleDetector(imgsz=640, include_motorcycles=False)
        self.detector_all = LiveVehicleDetector(imgsz=640, include_motorcycles=True)

        self.include_motorcycles = False
        self.view_mode = "side_by_side"  # "side_by_side" or "single"
        self.paused = False
        self.max_dwell_sec = 5.0
        self.custom_zones_spec: list[dict] | None = None

        self.video_path: Path | None = None
        self.preset_key: str = "40855"
        self.cap: cv2.VideoCapture | None = None
        self.W = 960
        self.H = 540
        self.fps = 25.0
        self.total_frames = 0
        self.frame_idx = 0

        self.vms_c3: SecurityVMSEngine | None = None
        self.vms_gl: SecurityVMSEngine | None = None
        self.last_res_c3: dict | None = None
        self.last_res_gl: dict | None = None
        self.last_frame_bgr: np.ndarray | None = None
        self.alert_feed: list[dict] = []

        # Load first available preset automatically
        for p in PRESET_CATALOG:
            vp = SAMPLE_DIR / p["id"]
            if vp.exists():
                self.load_video(vp, preset_key=p["key"], reset_zones=True)
                break

    def _build_zones(self) -> tuple[list[PolygonZone], list[tuple[str, str]]]:
        if self.custom_zones_spec:
            zones = []
            for zs in self.custom_zones_spec:
                zid = zs["zone_id"]
                role = zs.get("role", "both")
                col = ZONE_DEFAULT_COLORS.get(zid, (0, 200, 255))
                pts = [(float(p[0]), float(p[1])) for p in zs["points"]]
                zones.append(PolygonZone(zid, zs.get("name", zid), pts, role=role, normalized=True, color_bgr=col))
            return zones, [("Z_A", "Z_B")]
        zones, forb, def_dw = default_preset_zones(self.preset_key, self.W, self.H)
        return zones, forb

    def _rebuild_engines(self):
        zones_l, forb_l = self._build_zones()
        zones_r, forb_r = self._build_zones()
        c3_assoc = KalmanMahalanobisBaseline(theta=self.c3_theta, name="C-3")
        self.vms_c3 = SecurityVMSEngine(
            OnlineTracker(associator=c3_assoc, W=self.W, H=self.H, min_hits=3),
            zones=zones_l,
            forbidden_routes=forb_l,
            max_dwell_sec=self.max_dwell_sec,
            fps=self.fps,
        )
        self.vms_gl = SecurityVMSEngine(
            OnlineTracker(associator=self.gl_assoc, W=self.W, H=self.H, min_hits=3),
            zones=zones_r,
            forbidden_routes=forb_r,
            max_dwell_sec=self.max_dwell_sec,
            fps=self.fps,
        )
        self.last_res_c3 = None
        self.last_res_gl = None
        self.alert_feed = []

    def load_video(self, path: Path, preset_key: str = "generic", reset_zones: bool = True):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            self.video_path = path
            self.preset_key = preset_key
            if reset_zones:
                self.custom_zones_spec = None
                _, _, def_dw = default_preset_zones(preset_key, 960, 540)
                self.max_dwell_sec = def_dw

            self.cap = cv2.VideoCapture(str(path))
            self.W = max(64, int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            self.H = max(64, int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps_v = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
            self.fps = fps_v if 5.0 <= fps_v <= 60.0 else 25.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.frame_idx = 0
            self.paused = False
            self._rebuild_engines()

    def restart(self):
        if self.video_path is not None:
            self.load_video(self.video_path, preset_key=self.preset_key, reset_zones=False)

    def lock_at_normalized(self, norm_x: float, norm_y: float) -> dict:
        with self.lock:
            if self.last_frame_bgr is None or self.vms_gl is None or self.vms_c3 is None:
                return {"ok": False}
            px = float(np.clip(norm_x, 0.0, 1.0)) * float(self.W)
            py = float(np.clip(norm_y, 0.0, 1.0)) * float(self.H)
            tid_gl = None
            tid_c3 = None
            if self.last_res_gl is not None:
                tid_gl = self.vms_gl.target_lock.lock_by_click(
                    px, py, self.last_res_gl["tracks"], frame_id=self.frame_idx, frame_bgr=self.last_frame_bgr
                )
            if self.last_res_c3 is not None:
                tid_c3 = self.vms_c3.target_lock.lock_by_click(
                    px, py, self.last_res_c3["tracks"], frame_id=self.frame_idx, frame_bgr=self.last_frame_bgr
                )
            return {"ok": tid_gl is not None or tid_c3 is not None, "gl_tid": tid_gl, "c3_tid": tid_c3}

    def clear_lock(self):
        with self.lock:
            if self.vms_gl:
                self.vms_gl.target_lock.reset()
            if self.vms_c3:
                self.vms_c3.target_lock.reset()

    def set_custom_zone(self, zone_id: str, points_norm: list[list[float]]):
        with self.lock:
            base_zones, _ = self._build_zones()
            specs = []
            replaced = False
            role_map = {"Z_A": "entry", "Z_C": "entry", "Z_B": "restricted", "Z_DWELL": "dwell"}
            name_map = {
                "Z_A": "Z_A: Vùng Nguồn Giám Sát",
                "Z_C": "Z_C: Vùng Nguồn Hợp Lệ",
                "Z_B": "Z_B: Vùng Hạn Chế",
                "Z_DWELL": "Vùng Cấm Dừng Đỗ Lâu",
            }
            for z in base_zones:
                pts_n = z.points if z.normalized else [(p[0] / self.W, p[1] / self.H) for p in z.points]
                if z.zone_id == zone_id:
                    specs.append({"zone_id": zone_id, "name": name_map.get(zone_id, zone_id), "role": role_map.get(zone_id, "both"), "points": points_norm})
                    replaced = True
                else:
                    specs.append({"zone_id": z.zone_id, "name": z.name, "role": z.role, "points": pts_n})
            if not replaced:
                specs.append({"zone_id": zone_id, "name": name_map.get(zone_id, zone_id), "role": role_map.get(zone_id, "both"), "points": points_norm})
            self.custom_zones_spec = specs
        self.restart()

    def reset_default_zones(self):
        with self.lock:
            self.custom_zones_spec = None
        self.restart()

    def step_frame(self) -> dict | None:
        with self.lock:
            if self.cap is None or self.vms_gl is None or self.vms_c3 is None:
                return None

            t0 = time.perf_counter()
            if not self.paused or self.last_frame_bgr is None:
                ok, frm = self.cap.read()
                if not ok or frm is None:
                    # Loop seamlessly when reaching end of clip
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_idx = 0
                    self._rebuild_engines()
                    ok, frm = self.cap.read()
                    if not ok or frm is None:
                        return None
                self.frame_idx += 1
                self.last_frame_bgr = frm

                det_engine = self.detector_all if self.include_motorcycles else self.detector_4w
                d = det_engine.detect_frame(frm)

                prev_gl_ev_cnt = len(self.vms_gl.target_lock.events)
                res_c3 = self.vms_c3.step(self.frame_idx, d["boxes"], d["hists"], frame_bgr=frm)
                res_gl = self.vms_gl.step(self.frame_idx, d["boxes"], d["hists"], frame_bgr=frm)
                self.last_res_c3 = res_c3
                self.last_res_gl = res_gl

                for a in res_gl["route_alerts"]:
                    self.alert_feed.insert(0, {
                        "badge": "XÂM NHẬP LỘ TRÌNH CẤM",
                        "color": "#ef4444",
                        "title": f"Xe #{a['track_id']}: {a['origin_zone']} → {a['dest_zone']}",
                        "detail": f"Thời điểm {a['time_sec']}s • Vượt {a['occlusions_bridged']} lần che khuất",
                        "crop1": a["entry_crop"],
                        "crop2": a["alert_crop"],
                    })
                for a in res_gl["dwell_alerts"]:
                    self.alert_feed.insert(0, {
                        "badge": "ĐỖ / RÌNH RẬP QUÁ GIỜ",
                        "color": "#a855f7",
                        "title": f"Xe #{a['track_id']} dừng {a['dwell_sec']}s (> {a['max_dwell_sec']}s)",
                        "detail": f"Tại {a['zone_id']} • Duy trì qua {a['occlusions_bridged']} lần bị xe lớn che",
                        "crop1": a["entry_crop"],
                        "crop2": a["alert_crop"],
                    })
                if len(self.vms_gl.target_lock.events) > prev_gl_ev_cnt:
                    ev = self.vms_gl.target_lock.events[-1]
                    self.alert_feed.insert(0, {
                        "badge": "BẮT LẠI MỤC TIÊU KHÓA",
                        "color": "#10b981",
                        "title": f"Mục tiêu #{ev['track_id']} nối lại sau {ev['gap_sec']}s khuất ({ev['gap_frames']} frames)",
                        "detail": f"Khung hình #{ev['frame']} • Khớp xác suất Optimal Transport",
                        "crop1": self.vms_gl.target_lock.lock_crop_b64,
                        "crop2": ev["crop_b64"],
                    })
                self.alert_feed = self.alert_feed[:25]

            frm = self.last_frame_bgr
            res_c3 = self.last_res_c3
            res_gl = self.last_res_gl
            if frm is None or res_c3 is None or res_gl is None:
                return None

            if self.view_mode == "side_by_side":
                vis_l = draw_vms_overlay(frm, self.vms_c3, res_c3, "YOLOv8n + C-3 (Kalman)")
                vis_r = draw_vms_overlay(frm, self.vms_gl, res_gl, "YOLOv8n + GapLink-Lite (Ours)")
                pw, ph = 608, 342
                out_img = np.concatenate([cv2.resize(vis_l, (pw, ph)), cv2.resize(vis_r, (pw, ph))], axis=1)
            else:
                vis_r = draw_vms_overlay(frm, self.vms_gl, res_gl, "GapLink-VMS Live Security Console")
                out_img = cv2.resize(vis_r, (960, 540))

            dt_ms = (time.perf_counter() - t0) * 1000.0
            live_fps = round(1000.0 / max(1.0, dt_ms), 1)
            _, jpg = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64_frame = base64.b64encode(jpg.tobytes()).decode("ascii")

            od_gl = {f"{k[0]}->{k[1]}": v for k, v in self.vms_gl.route_mon.od_counts.items()}
            od_c3 = {f"{k[0]}->{k[1]}": v for k, v in self.vms_c3.route_mon.od_counts.items()}

            return {
                "frame_b64": b64_frame,
                "frame": self.frame_idx,
                "total_frames": self.total_frames,
                "fps": live_fps,
                "paused": self.paused,
                "view_mode": self.view_mode,
                "gl": {
                    "active": len(res_gl["tracks"]),
                    "route_alerts": len(self.vms_gl.route_mon.alerts),
                    "dwell_alerts": len(self.vms_gl.dwell_mon.alerts),
                    "unverified": len(self.vms_gl.route_mon.unverified_entries.get("Z_B", set())),
                    "od": od_gl,
                    "lock_state": res_gl["lock"].get("state", "IDLE"),
                    "lock_tid": res_gl["lock"].get("track_id"),
                    "reacquired": self.vms_gl.target_lock.reacquire_count,
                    "dustbin_holds": self.vms_gl.target_lock.dustbin_holds,
                },
                "c3": {
                    "active": len(res_c3["tracks"]),
                    "route_alerts": len(self.vms_c3.route_mon.alerts),
                    "dwell_alerts": len(self.vms_c3.dwell_mon.alerts),
                    "unverified": len(self.vms_c3.route_mon.unverified_entries.get("Z_B", set())),
                    "od": od_c3,
                    "lock_state": res_c3["lock"].get("state", "IDLE"),
                    "lock_tid": res_c3["lock"].get("track_id"),
                    "reacquired": self.vms_c3.target_lock.reacquire_count,
                },
                "alerts": self.alert_feed[:12],
            }


app = FastAPI(title="GapLink-VMS Smart Security Console")
SESSION: LiveVMSSession | None = None


def get_session() -> LiveVMSSession:
    global SESSION
    if SESSION is None:
        SESSION = LiveVMSSession()
    return SESSION


class ActionReq(BaseModel):
    action: str
    preset_id: str | None = None
    view_mode: str | None = None
    norm_x: float | None = None
    norm_y: float | None = None
    max_dwell_sec: float | None = None
    include_motorcycles: bool | None = None
    zone_id: str | None = None
    points: list[list[float]] | None = None


@app.get("/api/presets")
def api_presets():
    avail = []
    for p in PRESET_CATALOG:
        if (SAMPLE_DIR / p["id"]).exists():
            avail.append(p)
    return {"presets": avail}


@app.get("/api/benchmark")
def api_benchmark():
    jp = ROOT / "runs" / "test_final" / "live_security_results.json"
    if jp.exists():
        return JSONResponse(json.loads(jp.read_text(encoding="utf-8")))
    return JSONResponse({"pending": True})


@app.get("/api/benchmark_fig")
def api_benchmark_fig():
    fp = ROOT / "report" / "figures" / "fig_live_security_eval.png"
    if fp.exists():
        return FileResponse(str(fp), media_type="image/png")
    return JSONResponse({"error": "not_ready"}, status_code=404)


@app.post("/api/upload_video")
async def api_upload_video(file: UploadFile = File(...)):
    sess = get_session()
    safe_name = "".join(ch for ch in (file.filename or "upload.mp4") if ch.isalnum() or ch in ("_", "-", "."))
    dst = UPLOAD_DIR / f"user_{int(time.time())}_{safe_name}"
    content = await file.read()
    dst.write_bytes(content)
    sess.load_video(dst, preset_key="generic", reset_zones=True)
    return {"ok": True, "filename": dst.name, "W": sess.W, "H": sess.H}


@app.post("/api/action")
def api_action(req: ActionReq):
    sess = get_session()
    act = req.action
    if act == "select_preset" and req.preset_id:
        for p in PRESET_CATALOG:
            if p["id"] == req.preset_id and (SAMPLE_DIR / p["id"]).exists():
                sess.load_video(SAMPLE_DIR / p["id"], preset_key=p["key"], reset_zones=True)
                return {"ok": True, "max_dwell_sec": sess.max_dwell_sec}
    elif act == "toggle_pause":
        sess.paused = not sess.paused
        return {"ok": True, "paused": sess.paused}
    elif act == "restart":
        sess.restart()
        return {"ok": True}
    elif act == "set_view" and req.view_mode in ("side_by_side", "single"):
        sess.view_mode = req.view_mode
        return {"ok": True, "view_mode": sess.view_mode}
    elif act == "lock_click" and req.norm_x is not None and req.norm_y is not None:
        return sess.lock_at_normalized(req.norm_x, req.norm_y)
    elif act == "clear_lock":
        sess.clear_lock()
        return {"ok": True}
    elif act == "set_params":
        need_restart = False
        if req.max_dwell_sec is not None and abs(req.max_dwell_sec - sess.max_dwell_sec) > 0.05:
            sess.max_dwell_sec = float(req.max_dwell_sec)
            need_restart = True
        if req.include_motorcycles is not None:
            sess.include_motorcycles = bool(req.include_motorcycles)
        if need_restart:
            sess.restart()
        return {"ok": True}
    elif act == "set_zone" and req.zone_id and req.points and len(req.points) >= 3:
        sess.set_custom_zone(req.zone_id, req.points)
        return {"ok": True}
    elif act == "reset_zones":
        sess.reset_default_zones()
        return {"ok": True}
    return {"ok": False}


@app.get("/api/stream")
async def api_stream():
    sess = get_session()

    async def event_gen():
        while True:
            pkt = await asyncio.to_thread(sess.step_frame)
            if pkt is not None:
                yield f"data: {json.dumps(pkt)}\n\n"
            await asyncio.sleep(0.035 if not sess.paused else 0.18)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>GapLink-VMS | Hệ Thống Giám Sát An Ninh Phương Tiện</title>
<style>
  :root {
    --bg: #090d16; --panel: #111827; --panel2: #1e293b; --border: #334155;
    --text: #f1f5f9; --muted: #94a3b8; --emerald: #10b981; --amber: #f59e0b;
    --crimson: #ef4444; --blue: #3b82f6; --purple: #a855f7;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
  body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
  header {
    background: #0f172a; border-bottom: 1px solid var(--border); padding: 10px 20px;
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .dot-live { width: 10px; height: 10px; border-radius: 50%; background: var(--emerald); box-shadow: 0 0 10px var(--emerald); }
  .brand h1 { font-size: 16px; font-weight: 700; letter-spacing: 0.4px; }
  .sub-badge { font-size: 11.5px; color: var(--emerald); background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.35); padding: 3px 9px; border-radius: 999px; }
  .tabs { display: flex; gap: 8px; }
  .tab-btn {
    background: var(--panel2); color: var(--muted); border: 1px solid var(--border);
    padding: 7px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600;
  }
  .tab-btn.active { background: var(--blue); color: #fff; border-color: var(--blue); }
  .main-grid {
    display: grid; grid-template-columns: 290px 1fr 350px; gap: 12px; padding: 12px; flex: 1;
  }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; gap: 10px; }
  .card-title { font-size: 12px; text-transform: uppercase; letter-spacing: 0.7px; color: var(--muted); font-weight: 700; border-bottom: 1px solid #1f2937; padding-bottom: 6px; }
  select, input[type="file"], button {
    width: 100%; background: var(--panel2); color: var(--text); border: 1px solid var(--border);
    padding: 7px 10px; border-radius: 6px; font-size: 12.5px; cursor: pointer;
  }
  button:hover { border-color: #64748b; background: #273549; }
  button.primary { background: #1d4ed8; border-color: #2563eb; font-weight: 600; }
  button.emerald { background: #047857; border-color: var(--emerald); font-weight: 600; }
  button.danger { background: #991b1b; border-color: var(--crimson); }
  .btn-row { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .tool-btn.active { outline: 2px solid var(--amber); background: rgba(245,158,11,0.18); color: #fde68a; }
  .canvas-wrap { position: relative; background: #020617; border: 1px solid var(--border); border-radius: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; overflow: hidden; }
  canvas { width: 100%; height: auto; display: block; cursor: crosshair; }
  .status-strip {
    width: 100%; background: #0f172a; border-top: 1px solid var(--border); padding: 8px 14px;
    display: flex; justify-content: space-between; align-items: center; font-size: 12.5px; flex-wrap: wrap; gap: 8px;
  }
  .cmp-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .cmp-table th, .cmp-table td { padding: 6px 6px; border-bottom: 1px solid #1f2937; text-align: center; }
  .cmp-table th:first-child, .cmp-table td:first-child { text-align: left; }
  .val-gl { color: #34d399; font-weight: 700; }
  .val-c3 { color: #fb923c; font-weight: 600; }
  .alert-list { display: flex; flex-direction: column; gap: 8px; overflow-y: auto; max-height: 430px; padding-right: 4px; }
  .alert-item { background: #0f172a; border-left: 4px solid var(--crimson); border-radius: 6px; padding: 8px; font-size: 12px; display: flex; flex-direction: column; gap: 5px; }
  .alert-badge { font-size: 10px; font-weight: 800; padding: 2px 6px; border-radius: 4px; display: inline-block; width: fit-content; color: #fff; }
  .crops-row { display: flex; gap: 8px; align-items: center; margin-top: 3px; }
  .crops-row img { height: 54px; border-radius: 4px; border: 1px solid var(--border); object-fit: cover; }
  .hint { font-size: 11.5px; color: var(--muted); line-height: 1.4; }
  #tab-bench { display: none; padding: 18px; max-width: 1280px; margin: 0 auto; width: 100%; }
  .bench-card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 18px; margin-bottom: 16px; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="dot-live"></div>
    <h1>GAPLINK-VMS // TRẠM GIÁM SÁT AN NINH PHƯƠNG TIỆN THỜI GIAN THỰC</h1>
    <span class="sub-badge">Live YOLOv8n + Online HSV + GapLink-Lite (328K params) • Không dùng nhãn GT</span>
  </div>
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('live', this)">🎥 Trạm Camera Trực Tiếp</button>
    <button class="tab-btn" onclick="switchTab('bench', this)">📊 Báo Cáo Định Lượng 40 Video Test</button>
  </div>
</header>

<div id="tab-live" class="main-grid">
  <!-- Left Controls -->
  <div class="card">
    <div class="card-title">1. Nguồn Camera Giám Sát</div>
    <select id="presetSelect" onchange="onSelectPreset()"></select>
    <label class="hint">Hoặc tải lên file camera <code>.mp4</code> bất kỳ:</label>
    <input type="file" id="videoUpload" accept="video/mp4,video/avi,video/quicktime" onchange="uploadVideo()"/>

    <div class="card-title" style="margin-top:4px;">2. Chế Độ Màn Hình & Điều Khiển</div>
    <div class="btn-row">
      <button id="btnSide" class="primary" onclick="setView('side_by_side')">🔀 Đối chiếu C-3 / Ours</button>
      <button id="btnSingle" onclick="setView('single')">🖥️ Màn hình Đơn VMS</button>
    </div>
    <div class="btn-row">
      <button onclick="postAct({action:'toggle_pause'})" id="btnPause">⏸ Tạm dừng / Chạy</button>
      <button onclick="postAct({action:'restart'})">⏮ Chạy lại từ đầu</button>
    </div>

    <div class="card-title" style="margin-top:4px;">3. Tương Tác Chuột Trên Camera</div>
    <button id="tool_lock" class="tool-btn active" onclick="setMouseTool('lock')">🎯 Click chuột vào xe để KHÓA MỤC TIÊU</button>
    <div class="btn-row">
      <button id="tool_ZA" class="tool-btn" onclick="setMouseTool('Z_A')">✏️ Vẽ Vùng Z_A</button>
      <button id="tool_ZB" class="tool-btn" onclick="setMouseTool('Z_B')">✏️ Vẽ Vùng Cấm Z_B</button>
    </div>
    <div class="btn-row">
      <button id="tool_ZDWELL" class="tool-btn" onclick="setMouseTool('Z_DWELL')">✏️ Vẽ Vùng Cấm Đỗ</button>
      <button onclick="finishPoly()" class="emerald">✔ Lưu vùng vừa vẽ</button>
    </div>
    <div class="btn-row">
      <button onclick="postAct({action:'clear_lock'})">✖ Bỏ khóa xe</button>
      <button onclick="resetZones()">↺ Khôi phục vùng gốc</button>
    </div>

    <div class="card-title" style="margin-top:4px;">4. Cấu Hình Nghiệp Vụ</div>
    <label class="hint">Ngưỡng Báo Động Đỗ/Dừng Lâu (T_max): <b id="lblDwell">5.0s</b></label>
    <input type="range" id="rngDwell" min="2.0" max="12.0" step="0.5" value="5.0" onchange="updateParams()" oninput="document.getElementById('lblDwell').innerText=this.value+'s'"/>
    <label class="hint" style="display:flex;align-items:center;gap:8px;cursor:pointer;">
      <input type="checkbox" id="chkMoto" onchange="updateParams()"/> Bao gồm cả Xe máy (Motorcycles)
    </label>
    <p class="hint">💡 <b>Mẹo Demo:</b> Bấm trực tiếp vào một xe sắp đi qua sau xe buýt để so sánh khả năng giữ khóa mục tiêu và vùng elip dự báo 2σ của <b>GapLink-Lite</b> so với <b>C-3 Kalman</b>.</p>
  </div>

  <!-- Center Video Canvas -->
  <div class="canvas-wrap">
    <canvas id="vmsCanvas" width="1216" height="342"></canvas>
    <div class="status-strip">
      <div>Khung hình: <b id="stFrame">0 / 0</b> • Tốc độ xử lý: <b id="stFps" style="color:#34d399;">-- FPS</b></div>
      <div>Khóa C-3: <b id="stLockC3" class="val-c3">IDLE</b></div>
      <div>Khóa GapLink-Lite: <b id="stLockGL" class="val-gl">IDLE</b></div>
    </div>
  </div>

  <!-- Right Analytics & Evidence Feed -->
  <div class="card">
    <div class="card-title">Đối Chiếu Nghiệp Vụ Trực Tiếp</div>
    <table class="cmp-table">
      <thead>
        <tr><th>Chỉ số An ninh</th><th>C-3 (Kalman)</th><th>GapLink-Lite</th></tr>
      </thead>
      <tbody>
        <tr><td>🚨 Xâm nhập Z_A → Z_B</td><td id="m_c3_rt" class="val-c3">0</td><td id="m_gl_rt" class="val-gl">0</td></tr>
        <tr><td>⏱️ Đỗ / Nán lại quá giờ</td><td id="m_c3_dw" class="val-c3">0</td><td id="m_gl_dw" class="val-gl">0</td></tr>
        <tr><td>🎯 Bắt lại Mục tiêu Khóa</td><td id="m_c3_re" class="val-c3">0</td><td id="m_gl_re" class="val-gl">0</td></tr>
        <tr><td>🛡️ Dustbin Chặn nhảy nhầm</td><td>0</td><td id="m_gl_db" class="val-gl">0</td></tr>
        <tr><td>❓ Vào Z_B chưa rõ nguồn</td><td id="m_c3_uv">0</td><td id="m_gl_uv">0</td></tr>
      </tbody>
    </table>

    <div class="card-title">Ma Trận Lưu Lượng OD (GapLink-Lite)</div>
    <div id="odBox" class="hint" style="background:#0f172a;padding:8px;border-radius:6px;font-family:monospace;">Chưa ghi nhận lượt chuyển vùng...</div>

    <div class="card-title">Nhật Ký Bằng Chứng Thực Tế (Tự động cắt ảnh)</div>
    <div id="alertList" class="alert-list">
      <div class="hint">Đang giám sát luồng camera... Các cảnh báo xâm nhập lộ trình cấm, đỗ quá giờ và bắt lại mục tiêu sẽ tự động trích xuất ảnh phương tiện tại đây.</div>
    </div>
  </div>
</div>

<div id="tab-bench">
  <div class="bench-card">
    <h2 style="font-size:18px;margin-bottom:8px;">Thực Nghiệm Đầu-Cuối với Live YOLOv8n trên 40 Chuỗi Video Gốc Tập Test UA-DETRAC</h2>
    <p class="hint" style="margin-bottom:14px;">Hoàn toàn không dùng nhãn Ground-Truth ở đầu vào và không dùng hộp đen che khuất giả lập (<code>make_hide_plan</code>). Toàn bộ hiện tượng che khuất, rớt frame và nhiễu hộp giới hạn đều sinh ra tự nhiên từ YOLOv8n chạy trực tiếp trên khung hình camera.</p>
    <div id="benchContent" class="hint">Đang tải kết quả định lượng...</div>
    <div style="margin-top:16px;text-align:center;">
      <img id="benchImg" src="/api/benchmark_fig" style="max-width:100%;border-radius:8px;border:1px solid #334155;" onerror="this.style.display='none'"/>
    </div>
  </div>
</div>

<script>
let currentView = 'side_by_side';
let mouseTool = 'lock';
let drawPts = [];
const canvas = document.getElementById('vmsCanvas');
const ctx = canvas.getContext('2d');
const frameImg = new Image();

async function postAct(payload) {
  const r = await fetch('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  return await r.json();
}

async function initPresets() {
  const r = await fetch('/api/presets');
  const d = await r.json();
  const sel = document.getElementById('presetSelect');
  sel.innerHTML = '';
  d.presets.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.id; opt.textContent = p.label;
    sel.appendChild(opt);
  });
}

async function onSelectPreset() {
  const pid = document.getElementById('presetSelect').value;
  drawPts = [];
  const res = await postAct({action:'select_preset', preset_id: pid});
  if (res.max_dwell_sec) {
    document.getElementById('rngDwell').value = res.max_dwell_sec;
    document.getElementById('lblDwell').innerText = res.max_dwell_sec + 's';
  }
}

async function uploadVideo() {
  const inp = document.getElementById('videoUpload');
  if (!inp.files.length) return;
  const fd = new FormData();
  fd.append('file', inp.files[0]);
  await fetch('/api/upload_video', {method:'POST', body: fd});
}

function setView(v) {
  currentView = v;
  document.getElementById('btnSide').className = (v === 'side_by_side') ? 'primary' : '';
  document.getElementById('btnSingle').className = (v === 'single') ? 'primary' : '';
  postAct({action:'set_view', view_mode: v});
}

function setMouseTool(t) {
  mouseTool = t;
  drawPts = [];
  ['lock','ZA','ZB','ZDWELL'].forEach(k => document.getElementById('tool_'+k).classList.remove('active'));
  const mapId = {'lock':'tool_lock','Z_A':'tool_ZA','Z_B':'tool_ZB','Z_DWELL':'tool_ZDWELL'};
  if (mapId[t]) document.getElementById(mapId[t]).classList.add('active');
}

canvas.addEventListener('click', async (ev) => {
  const rect = canvas.getBoundingClientRect();
  let nx = (ev.clientX - rect.left) / rect.width;
  let ny = (ev.clientY - rect.top) / rect.height;
  if (currentView === 'side_by_side') {
    nx = (nx < 0.5) ? (nx * 2.0) : ((nx - 0.5) * 2.0);
  }
  if (mouseTool === 'lock') {
    await postAct({action:'lock_click', norm_x: nx, norm_y: ny});
  } else {
    drawPts.push([nx, ny]);
  }
});

async function finishPoly() {
  if (mouseTool === 'lock' || drawPts.length < 3) return;
  await postAct({action:'set_zone', zone_id: mouseTool, points: drawPts});
  drawPts = [];
  setMouseTool('lock');
}

async function resetZones() {
  drawPts = [];
  await postAct({action:'reset_zones'});
}

async function updateParams() {
  const dw = parseFloat(document.getElementById('rngDwell').value);
  const mc = document.getElementById('chkMoto').checked;
  await postAct({action:'set_params', max_dwell_sec: dw, include_motorcycles: mc});
}

function drawPolygonDraft() {
  if (drawPts.length === 0) return;
  ctx.save();
  ctx.strokeStyle = '#fde047';
  ctx.fillStyle = 'rgba(253, 224, 71, 0.25)';
  ctx.lineWidth = 2;
  const drawOnPane = (xOff, wPane) => {
    ctx.beginPath();
    drawPts.forEach((p, idx) => {
      const px = xOff + p[0] * wPane;
      const py = p[1] * canvas.height;
      if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      ctx.fillRect(px - 3, py - 3, 6, 6);
    });
    ctx.closePath();
    ctx.fill(); ctx.stroke();
  };
  if (currentView === 'side_by_side') {
    drawOnPane(0, canvas.width / 2);
    drawOnPane(canvas.width / 2, canvas.width / 2);
  } else {
    drawOnPane(0, canvas.width);
  }
  ctx.restore();
}

const es = new EventSource('/api/stream');
es.onmessage = (ev) => {
  const d = JSON.parse(ev.data);
  frameImg.onload = () => {
    if (canvas.width !== frameImg.width || canvas.height !== frameImg.height) {
      canvas.width = frameImg.width; canvas.height = frameImg.height;
    }
    ctx.drawImage(frameImg, 0, 0);
    drawPolygonDraft();
  };
  frameImg.src = 'data:image/jpeg;base64,' + d.frame_b64;

  document.getElementById('stFrame').innerText = `${d.frame} / ${d.total_frames}`;
  document.getElementById('stFps').innerText = `${d.fps} FPS`;
  document.getElementById('stLockC3').innerText = d.c3.lock_tid ? `#${d.c3.lock_tid} (${d.c3.lock_state})` : 'Chưa khóa (Click vào xe)';
  document.getElementById('stLockGL').innerText = d.gl.lock_tid ? `#${d.gl.lock_tid} (${d.gl.lock_state})` : 'Chưa khóa (Click vào xe)';

  document.getElementById('m_c3_rt').innerText = d.c3.route_alerts;
  document.getElementById('m_gl_rt').innerText = d.gl.route_alerts;
  document.getElementById('m_c3_dw').innerText = d.c3.dwell_alerts;
  document.getElementById('m_gl_dw').innerText = d.gl.dwell_alerts;
  document.getElementById('m_c3_re').innerText = d.c3.reacquired;
  document.getElementById('m_gl_re').innerText = d.gl.reacquired;
  document.getElementById('m_gl_db').innerText = d.gl.dustbin_holds;
  document.getElementById('m_c3_uv').innerText = d.c3.unverified;
  document.getElementById('m_gl_uv').innerText = d.gl.unverified;

  const odEntries = Object.entries(d.gl.od || {});
  document.getElementById('odBox').innerHTML = odEntries.length
    ? odEntries.map(([k,v]) => `<div>${k}: <b style="color:#38bdf8;">${v} xe</b></div>`).join('')
    : 'Chưa ghi nhận lượt chuyển vùng...';

  if (d.alerts && d.alerts.length) {
    document.getElementById('alertList').innerHTML = d.alerts.map(a => `
      <div class="alert-item" style="border-left-color:${a.color}">
        <span class="alert-badge" style="background:${a.color}">${a.badge}</span>
        <div style="font-weight:700;">${a.title}</div>
        <div class="hint">${a.detail}</div>
        <div class="crops-row">
          ${a.crop1 ? `<img src="${a.crop1}" title="Ảnh ban đầu"/>` : ''}
          ${a.crop1 && a.crop2 ? '<span style="color:#94a3b8;">➔</span>' : ''}
          ${a.crop2 ? `<img src="${a.crop2}" title="Ảnh thời điểm cảnh báo"/>` : ''}
        </div>
      </div>
    `).join('');
  }
};

async function switchTab(name, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-live').style.display = (name === 'live') ? 'grid' : 'none';
  document.getElementById('tab-bench').style.display = (name === 'bench') ? 'block' : 'none';
  if (name === 'bench') {
    const r = await fetch('/api/benchmark');
    const j = await r.json();
    if (j.methods) {
      let rows = '';
      for (const [m, v] of Object.entries(j.methods)) {
        const bold = (m === 'GapLink-Lite') ? 'style="color:#34d399;font-weight:700;"' : '';
        rows += `<tr ${bold}>
          <td>${m}</td>
          <td>${(v.natural_occlusion.nat_irr*100).toFixed(2)}%</td>
          <td>${(v.natural_occlusion.nat_wlr*100).toFixed(2)}%</td>
          <td>${(v.forbidden_route.recall_occl*100).toFixed(2)}%</td>
          <td>${(v.forbidden_route.false_alarm_rate*100).toFixed(2)}%</td>
          <td>${(v.dwell_alert.f1_occl*100).toFixed(2)}%</td>
        </tr>`;
      }
      const lat = j.latency_rtx3050 || {};
      document.getElementById('benchContent').innerHTML = `
        <div style="margin-bottom:12px;color:#e2e8f0;">
          <b>Tốc độ thực thi đơn khung hình (RTX 3050 6GB):</b> YOLOv8n = ${lat.yolo_ms} ms | Trích xuất HSV = ${lat.hsv_post_ms} ms | GapLink-Lite + VMS = ${lat.gaplink_vms_ms} ms ➔ <b>Tổng = ${lat.total_ms} ms (${lat.fps} FPS)</b>
        </div>
        <table class="cmp-table" style="font-size:13px;">
          <thead>
            <tr>
              <th>Phương pháp (sau Live YOLOv8n)</th>
              <th>Nat-IRR (%) ↑</th>
              <th>Nat-WLR (%) ↓</th>
              <th>Bắt Lộ Trình Cấm Khi Khuất (%) ↑</th>
              <th>Tỷ Lệ Báo Giả Lộ Trình (%) ↓</th>
              <th>Cảnh Báo Đỗ Quá Giờ F1 (%) ↑</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>`;
      document.getElementById('benchImg').style.display = 'inline-block';
      document.getElementById('benchImg').src = '/api/benchmark_fig?t=' + Date.now();
    }
  }
}

initPresets();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTMLResponse(DASHBOARD_HTML)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    get_session()
    print(f"GapLink-VMS Dashboard ready at http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
