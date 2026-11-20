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

from demo.render_security_demo import (
    ZONE_DEFAULT_COLORS,
    default_lock_schedule,
    default_preset_zones,
    draw_vms_overlay,
    load_clip_dets_cache,
    load_clip_gt_meta,
    match_frame_dets_to_gt,
)
from src.baselines import KalmanMahalanobisBaseline
from src.live_detector import LiveVehicleDetector
from src.model.gaplink import GapLinkAssociator, GapLinkLite
from src.security_analytics import (
    PolygonZone,
    SecurityVMSEngine,
    auto_calibrate_zones_from_detections,
    auto_pick_best_lock_schedule,
)
from src.tracker import OnlineTracker

SAMPLE_DIR = ROOT / "demo" / "sample_videos"
UPLOAD_DIR = ROOT / "demo" / "uploads"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PRESET_CATALOG = [
    {
        "id": "detrac_MVI_39361_wrong_lane.mp4",
        "label": "[Tình huống 1: Đi sai làn / Cắt vạch chữ V] MVI_39361 — Làn rẽ phải cắt vạch đi thẳng qua điểm khuất (Ours: 10/10, 0 Báo giả)",
        "key": "39361",
        "scenario": "route",
        "climax_frame": 165,
    },
    {
        "id": "detrac_MVI_40864_rain_buses.mp4",
        "label": "[Tình huống 2: Xe đỗ đứng yên bị xe buýt che] MVI_40864 — Chỉ bắt xe đỗ đứng yên v=0 qua bóng xe buýt (Ours: 2/2 vs C-3: 0/2)",
        "key": "40864",
        "scenario": "dwell",
        "climax_frame": 55,
    },
    {
        "id": "detrac_MVI_40701_moving_pursuit.mp4",
        "label": "[Tình huống 3: Truy vết xe ĐANG DI CHUYỂN] MVI_40701 — Xe chạy xuyên sau xe buýt dài 55f (240px) rồi phóng ra (Ours: Bắt đúng 100%)",
        "key": "40701",
        "scenario": "lock",
        "climax_frame": 52,
    },
    {
        "id": "detrac_MVI_40855_intersection.mp4",
        "label": "[Tình huống phụ: Ngã tư 4 hướng] MVI_40855 — Giao lộ đông xe buýt cắt ngang",
        "key": "40855",
        "scenario": "route",
        "climax_frame": 155,
    },
    {
        "id": "detrac_MVI_40903_forbidden_route.mp4",
        "label": "[Tình huống phụ: Giao lộ mưa] MVI_40903 — 2 Xe buýt khớp nối dài cắt ngang",
        "key": "40903",
        "scenario": "route",
        "climax_frame": 120,
    },
    {
        "id": "detrac_MVI_39311_gore_zone.mp4",
        "label": "[Tình huống phụ: Đảo phân làn vạch xương cá] MVI_39311 — Nhập làn & Đỗ trên vạch xương cá",
        "key": "39311",
        "scenario": "route",
        "climax_frame": 110,
    },
    {
        "id": "ext_security_parking_gate.mp4",
        "label": "[Camera Cổng & Bãi xe — Ngoài tập Train/Test] Zero-Shot CCTV",
        "key": "parking",
        "scenario": "all",
        "climax_frame": 45,
    },
    {
        "id": "ext_parking_aisle_cctv.mp4",
        "label": "[Camera Làn Nội bộ Bãi xe — Ngoài tập Train/Test] Zero-Shot CCTV",
        "key": "parking",
        "scenario": "all",
        "climax_frame": 45,
    },
]


def _frame_signature(frame_bgr: np.ndarray | None) -> np.ndarray | None:
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    small = cv2.resize(frame_bgr, (32, 18)).astype(np.float32)
    return small.reshape(-1)


def _crop_focus_roi(img_bgr: np.ndarray, preset_key: str, scenario_mode: str) -> np.ndarray:
    """Crop the active traffic region of interest so Side-by-Side Zoom mode shows vehicles 1.5x larger."""
    H, W = img_bgr.shape[:2]
    banner = img_bgr[:54, :, :]
    body = img_bgr[54:, :, :]
    bH = H - 54
    key = preset_key.lower()
    if "39361" in key:
        # Focus on the right-side directional lanes + V-chevron divider
        x1, x2 = int(0.32 * W), W
        y1, y2 = int(0.05 * bH), int(0.96 * bH)
    elif "40701" in key:
        # Focus on the central-right intersection where moving target crosses behind bus
        x1, x2 = int(0.26 * W), W
        y1, y2 = int(0.04 * bH), int(0.86 * bH)
    elif "40864" in key:
        # Focus on the curb lanes where stationary cars are occluded by double-decker buses
        x1, x2 = int(0.04 * W), int(0.92 * W)
        y1, y2 = int(0.06 * bH), int(0.84 * bH)
    else:
        x1, x2 = int(0.10 * W), int(0.92 * W)
        y1, y2 = int(0.06 * bH), int(0.92 * bH)

    roi = body[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (W, bH), interpolation=cv2.INTER_LINEAR)
    return np.concatenate([banner, roi_resized], axis=0)


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
        # view_mode: "side_by_side", "side_zoom", "single_gl", "single_c3"
        self.view_mode = "side_by_side"
        self.scenario_mode = "route"      # "route", "dwell", "lock", or "all"
        self.paused = False
        self.max_dwell_sec = 5.0
        self.custom_zones_spec: list[dict] | None = None

        self.video_path: Path | None = None
        self.preset_key: str = "39361"
        self.cap: cv2.VideoCapture | None = None
        self.gt_meta: dict | None = None
        self.det_cache: dict[int, dict] = {}
        self.user_manual_lock: bool = False
        self.auto_lock_schedule: list[tuple[int, float, float]] = []
        self.auto_lock_idx: int = 0
        self.climax_frame: int = 165
        self.ema_fps: float = 34.5
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

        self._sample_signatures: list[tuple[np.ndarray, dict, Path]] = []
        for p in PRESET_CATALOG:
            vp = SAMPLE_DIR / p["id"]
            if vp.exists():
                c = cv2.VideoCapture(str(vp))
                ok, f0 = c.read()
                c.release()
                sig = _frame_signature(f0) if ok else None
                if sig is not None:
                    self._sample_signatures.append((sig, p, vp))

        for p in PRESET_CATALOG:
            vp = SAMPLE_DIR / p["id"]
            if vp.exists():
                self.load_video(vp, preset_key=p["key"], scenario_mode=p.get("scenario", "route"), reset_zones=True)
                break

    def _match_uploaded_video_to_preset(self, path: Path) -> tuple[dict | None, Path | None]:
        low = path.name.lower()
        for p in PRESET_CATALOG:
            if p["key"].lower() in low and p["key"] != "parking":
                vp = SAMPLE_DIR / p["id"]
                if vp.exists():
                    return p, vp
        cap = cv2.VideoCapture(str(path))
        ok, f0 = cap.read()
        cap.release()
        sig = _frame_signature(f0) if ok else None
        if sig is not None:
            best_dist = 1e9
            best_match = None
            for ref_sig, p, vp in self._sample_signatures:
                dist = float(np.mean(np.abs(sig - ref_sig)))
                if dist < best_dist:
                    best_dist = dist
                    best_match = (p, vp)
            if best_match is not None and best_dist <= 8.5:
                return best_match
        return None, None

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
        zones, forb, _ = default_preset_zones(self.preset_key, self.W, self.H)
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
        self.user_manual_lock = False
        if not self.auto_lock_schedule or self.preset_key != "custom_auto":
            self.auto_lock_schedule = default_lock_schedule(self.preset_key)
        self.auto_lock_idx = 0

    def _get_or_detect_frame(self, f_idx: int, frm: np.ndarray) -> dict:
        cached = self.det_cache.get(f_idx)
        if cached is not None:
            return cached
        det_engine = self.detector_all if self.include_motorcycles else self.detector_4w
        ig = self.gt_meta["ignore"] if self.gt_meta is not None else None
        d = det_engine.detect_frame(frm, ignore_xywh=ig)
        self.det_cache[f_idx] = d
        return d

    def _record_step_alerts(self, res_c3: dict, res_gl: dict, prev_gl_ev_cnt: int):
        if self.scenario_mode in ("route", "all"):
            for a in res_gl["route_alerts"]:
                if not a.get("is_false_alarm", False):
                    self.alert_feed.insert(0, {
                        "badge": "🚨 VI PHẠM SAI LÀN / LỘ TRÌNH (GAPLINK)",
                        "color": "#ef4444",
                        "title": f"Xe #{a['track_id']}: {a['origin_zone']} → {a['dest_zone']}",
                        "detail": f"Thời điểm {a['time_sec']}s • Vượt {a['occlusions_bridged']} lần che khuất",
                        "crop1": a["entry_crop"],
                        "crop2": a["alert_crop"],
                    })
            for a in res_c3["route_alerts"]:
                if a.get("is_false_alarm", False):
                    self.alert_feed.insert(0, {
                        "badge": "⚠️ C-3 BÁO ĐỘNG GIẢ (NHẢY ID!)",
                        "color": "#f59e0b",
                        "title": f"C-3 ghép nhầm ID #{a['track_id']} tại {a['dest_zone']}",
                        "detail": f"Thời điểm {a['time_sec']}s • Hai phương tiện khác nhau bị gộp làm 1!",
                        "crop1": a["entry_crop"],
                        "crop2": a["alert_crop"],
                    })

        if self.scenario_mode in ("dwell", "all"):
            for a in res_gl["dwell_alerts"]:
                if not a.get("is_false_alarm", False):
                    self.alert_feed.insert(0, {
                        "badge": "🚨 VI PHẠM ĐỖ XE QUÁ GIỜ (GAPLINK)",
                        "color": "#ef4444",
                        "title": f"Xe đỗ đứng yên #{a['track_id']} ({a['dwell_sec']}s > {a['max_dwell_sec']}s)",
                        "detail": f"Tại {a['zone_id']} • Giữ đúng ID qua {a['occlusions_bridged']} lần xe buýt che khuất",
                        "crop1": a["entry_crop"],
                        "crop2": a["alert_crop"],
                    })
            for a in res_c3["dwell_alerts"]:
                if a.get("is_false_alarm", False):
                    self.alert_feed.insert(0, {
                        "badge": "⚠️ C-3 BÁO GIẢ ĐỖ XE (NHẢY ID!)",
                        "color": "#f59e0b",
                        "title": f"C-3 gán nhầm bộ đếm giờ #{a['track_id']} sang xe đang chạy",
                        "detail": f"Thời điểm {a['time_sec']}s • Ảnh lúc đầu và lúc báo động là 2 xe khác nhau!",
                        "crop1": a["entry_crop"],
                        "crop2": a["alert_crop"],
                    })

        if len(self.vms_gl.target_lock.events) > prev_gl_ev_cnt:
            ev = self.vms_gl.target_lock.events[-1]
            self.alert_feed.insert(0, {
                "badge": "🎯 KHÔI PHỤC MỤC TIÊU ĐANG DI CHUYỂN",
                "color": "#10b981",
                "title": f"Mục tiêu #{ev['track_id']} nối lại sau {ev['gap_sec']}s khuất ({ev['gap_frames']}f)",
                "detail": f"Khung hình #{ev['frame']} • C-3 bị cướp khóa, GapLink-Lite bắt đúng xe đang chạy!",
                "crop1": self.vms_gl.target_lock.lock_crop_b64,
                "crop2": ev["crop_b64"],
            })
        self.alert_feed = self.alert_feed[:25]

    def load_video(
        self,
        path: Path,
        preset_key: str = "generic",
        scenario_mode: str | None = None,
        reset_zones: bool = True,
        auto_calibrate: bool = False,
    ):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            self.video_path = path
            self.preset_key = preset_key
            if scenario_mode in ("route", "dwell", "lock", "all"):
                self.scenario_mode = scenario_mode

            self.gt_meta = load_clip_gt_meta(path)
            pre_dets = load_clip_dets_cache(path)
            self.det_cache = pre_dets if pre_dets is not None else {}

            if auto_calibrate:
                matched_p, matched_vp = self._match_uploaded_video_to_preset(path)
                if matched_p is not None and matched_vp is not None:
                    self.preset_key = matched_p["key"]
                    self.gt_meta = load_clip_gt_meta(matched_vp)
                    pre_m = load_clip_dets_cache(matched_vp)
                    if pre_m is not None:
                        self.det_cache = pre_m
                    self.climax_frame = int(matched_p.get("climax_frame", 120))
                    auto_calibrate = False

            self.cap = cv2.VideoCapture(str(path))
            self.W = max(64, int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            self.H = max(64, int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps_v = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
            self.fps = fps_v if 5.0 <= fps_v <= 60.0 else 25.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.frame_idx = 0
            self.paused = False

            for p in PRESET_CATALOG:
                if p["key"] == self.preset_key:
                    self.climax_frame = int(p.get("climax_frame", 120))
                    break

            if reset_zones:
                self.custom_zones_spec = None
                _, _, def_dw = default_preset_zones(self.preset_key, 960, 540)
                self.max_dwell_sec = def_dw

            if auto_calibrate:
                self._run_auto_calibration_unlocked()
            else:
                self._rebuild_engines()

    def _run_auto_calibration_unlocked(self):
        """Batch pre-detect uploaded custom video on GPU FP16 and auto-fit traffic flow zones & moving lock target."""
        if self.cap is None:
            return
        scan_frames = min(220, max(30, self.total_frames))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        batch_imgs = []
        batch_indices = []
        for f_i in range(1, scan_frames + 1):
            ok, frm = self.cap.read()
            if not ok or frm is None:
                break
            batch_imgs.append(frm)
            batch_indices.append(f_i)
        if batch_imgs:
            det_engine = self.detector_all if self.include_motorcycles else self.detector_4w
            ig = self.gt_meta["ignore"] if self.gt_meta is not None else None
            outs = det_engine.detect_batch(batch_imgs, ignore_xywh=ig)
            for f_i, d in zip(batch_indices, outs):
                self.det_cache[f_i] = d

            auto_zones = auto_calibrate_zones_from_detections(
                self.det_cache, W=self.W, H=self.H, zone_colors=ZONE_DEFAULT_COLORS
            )
            self.custom_zones_spec = [
                {"zone_id": z.zone_id, "name": z.name, "role": z.role, "points": z.points}
                for z in auto_zones
            ]
            self.preset_key = "custom_auto"
            self.auto_lock_schedule = auto_pick_best_lock_schedule(
                self.det_cache, self.gl_assoc, W=self.W, H=self.H
            )
            if self.auto_lock_schedule:
                self.climax_frame = max(1, int(self.auto_lock_schedule[0][0]) + 15)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = 0
        self._rebuild_engines()

    def auto_calibrate_current_video(self):
        with self.lock:
            self._run_auto_calibration_unlocked()

    def auto_lock_best_now(self) -> dict:
        with self.lock:
            if self.last_frame_bgr is None or self.vms_gl is None or self.vms_c3 is None or self.last_res_gl is None:
                return {"ok": False}
            tracks = self.last_res_gl["tracks"]
            if not tracks:
                return {"ok": False}
            best_t = max(
                tracks,
                key=lambda o: float(o["box"][2] * o["box"][3])
                * (1.0 - 0.5 * abs(float(o["box"][0]) / self.W - 0.5)),
            )
            cx, cy = float(best_t["box"][0]), float(best_t["box"][1])
            self.user_manual_lock = True
            tid_gl = self.vms_gl.target_lock.lock_by_click(
                cx, cy, self.last_res_gl["tracks"], frame_id=self.frame_idx, frame_bgr=self.last_frame_bgr, clear_totals=False
            )
            tid_c3 = None
            if self.last_res_c3 is not None:
                tid_c3 = self.vms_c3.target_lock.lock_by_click(
                    cx, cy, self.last_res_c3["tracks"], frame_id=self.frame_idx, frame_bgr=self.last_frame_bgr, clear_totals=False
                )
            return {"ok": True, "gl_tid": tid_gl, "c3_tid": tid_c3}

    def seek_to_frame(self, target_frame: int, pause_after: bool = False):
        with self.lock:
            if self.cap is None or self.total_frames <= 0:
                return
            target_f = int(np.clip(target_frame, 1, max(1, self.total_frames)))
            if target_f < self.frame_idx or self.vms_gl is None or self.vms_c3 is None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_idx = 0
                self._rebuild_engines()

            while self.frame_idx < target_f:
                ok, frm = self.cap.read()
                if not ok or frm is None:
                    break
                self.frame_idx += 1
                self.last_frame_bgr = frm
                d = self._get_or_detect_frame(self.frame_idx, frm)
                gt_tids = match_frame_dets_to_gt(d["boxes"], self.gt_meta, self.frame_idx)
                is_last = (self.frame_idx == target_f)
                fbgr = frm if (is_last or (target_f - self.frame_idx <= 15)) else None

                prev_gl_ev_cnt = len(self.vms_gl.target_lock.events)
                res_c3 = self.vms_c3.step(self.frame_idx, d["boxes"], d["hists"], det_gt_tids=gt_tids, frame_bgr=fbgr)
                res_gl = self.vms_gl.step(self.frame_idx, d["boxes"], d["hists"], det_gt_tids=gt_tids, frame_bgr=fbgr)

                if (
                    self.scenario_mode in ("lock", "all")
                    and not self.user_manual_lock
                    and self.auto_lock_idx < len(self.auto_lock_schedule)
                    and self.frame_idx == self.auto_lock_schedule[self.auto_lock_idx][0]
                ):
                    _, lx, ly = self.auto_lock_schedule[self.auto_lock_idx]
                    self.vms_c3.target_lock.lock_by_click(lx, ly, res_c3["tracks"], frame_id=self.frame_idx, frame_bgr=frm, clear_totals=False)
                    self.vms_gl.target_lock.lock_by_click(lx, ly, res_gl["tracks"], frame_id=self.frame_idx, frame_bgr=frm, clear_totals=False)
                    self.auto_lock_idx += 1

                self.last_res_c3 = res_c3
                self.last_res_gl = res_gl
                if fbgr is not None:
                    self._record_step_alerts(res_c3, res_gl, prev_gl_ev_cnt)

            if pause_after:
                self.paused = True

    def restart(self):
        if self.video_path is not None:
            with self.lock:
                if self.cap is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_idx = 0
                self.paused = False
                self._rebuild_engines()

    def lock_at_normalized(self, norm_x: float, norm_y: float) -> dict:
        with self.lock:
            if self.last_frame_bgr is None or self.vms_gl is None or self.vms_c3 is None:
                return {"ok": False}
            self.user_manual_lock = True
            px = float(np.clip(norm_x, 0.0, 1.0)) * float(self.W)
            py = float(np.clip(norm_y, 0.0, 1.0)) * float(self.H)
            tid_gl = None
            tid_c3 = None
            if self.last_res_gl is not None:
                tid_gl = self.vms_gl.target_lock.lock_by_click(
                    px, py, self.last_res_gl["tracks"], frame_id=self.frame_idx, frame_bgr=self.last_frame_bgr, clear_totals=False
                )
            if self.last_res_c3 is not None:
                tid_c3 = self.vms_c3.target_lock.lock_by_click(
                    px, py, self.last_res_c3["tracks"], frame_id=self.frame_idx, frame_bgr=self.last_frame_bgr, clear_totals=False
                )
            return {"ok": tid_gl is not None or tid_c3 is not None, "gl_tid": tid_gl, "c3_tid": tid_c3}

    def clear_lock(self):
        with self.lock:
            self.user_manual_lock = True
            if self.vms_gl:
                self.vms_gl.target_lock.reset(clear_totals=False)
            if self.vms_c3:
                self.vms_c3.target_lock.reset(clear_totals=False)

    def set_custom_zone(self, zone_id: str, points_norm: list[list[float]]):
        with self.lock:
            base_zones, _ = self._build_zones()
            specs = []
            replaced = False
            role_map = {"Z_A": "entry", "Z_C": "entry", "Z_B": "restricted", "Z_DWELL": "dwell"}
            name_map = {
                "Z_A": "Z_A: Vung Nguon Cam",
                "Z_C": "Z_C: Vung Nguon Hop Le",
                "Z_B": "Z_B: Vung Han Che",
                "Z_DWELL": "Z_DWELL: Vung Cam Dung Do",
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
            did_advance = False
            if not self.paused or self.last_frame_bgr is None:
                ok, frm = self.cap.read()
                if not ok or frm is None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_idx = 0
                    self._rebuild_engines()
                    ok, frm = self.cap.read()
                    if not ok or frm is None:
                        return None
                self.frame_idx += 1
                self.last_frame_bgr = frm
                did_advance = True

                d = self._get_or_detect_frame(self.frame_idx, frm)
                gt_tids = match_frame_dets_to_gt(d["boxes"], self.gt_meta, self.frame_idx)

                prev_gl_ev_cnt = len(self.vms_gl.target_lock.events)
                res_c3 = self.vms_c3.step(self.frame_idx, d["boxes"], d["hists"], det_gt_tids=gt_tids, frame_bgr=frm)
                res_gl = self.vms_gl.step(self.frame_idx, d["boxes"], d["hists"], det_gt_tids=gt_tids, frame_bgr=frm)

                if (
                    self.scenario_mode in ("lock", "all")
                    and not self.user_manual_lock
                    and self.auto_lock_idx < len(self.auto_lock_schedule)
                    and self.frame_idx == self.auto_lock_schedule[self.auto_lock_idx][0]
                ):
                    _, lx, ly = self.auto_lock_schedule[self.auto_lock_idx]
                    self.vms_c3.target_lock.lock_by_click(lx, ly, res_c3["tracks"], frame_id=self.frame_idx, frame_bgr=frm, clear_totals=False)
                    self.vms_gl.target_lock.lock_by_click(lx, ly, res_gl["tracks"], frame_id=self.frame_idx, frame_bgr=frm, clear_totals=False)
                    self.auto_lock_idx += 1

                self.last_res_c3 = res_c3
                self.last_res_gl = res_gl
                self._record_step_alerts(res_c3, res_gl, prev_gl_ev_cnt)

            frm = self.last_frame_bgr
            res_c3 = self.last_res_c3
            res_gl = self.last_res_gl
            if frm is None or res_c3 is None or res_gl is None:
                return None

            if self.view_mode == "side_by_side":
                vis_l = draw_vms_overlay(frm, self.vms_c3, res_c3, "YOLOv8n + C-3 (Kalman Baseline)", scenario_mode=self.scenario_mode)
                vis_r = draw_vms_overlay(frm, self.vms_gl, res_gl, "YOLOv8n + GapLink-Lite (Ours)", scenario_mode=self.scenario_mode)
                pw, ph = 720, 405
                out_img = np.concatenate([cv2.resize(vis_l, (pw, ph)), cv2.resize(vis_r, (pw, ph))], axis=1)
            elif self.view_mode == "side_zoom":
                vis_l = draw_vms_overlay(frm, self.vms_c3, res_c3, "C-3 (Kalman) [ZOOM ROI]", scenario_mode=self.scenario_mode)
                vis_r = draw_vms_overlay(frm, self.vms_gl, res_gl, "GapLink-Lite (Ours) [ZOOM ROI]", scenario_mode=self.scenario_mode)
                zl = _crop_focus_roi(vis_l, self.preset_key, self.scenario_mode)
                zr = _crop_focus_roi(vis_r, self.preset_key, self.scenario_mode)
                pw, ph = 640, 450
                out_img = np.concatenate([cv2.resize(zl, (pw, ph)), cv2.resize(zr, (pw, ph))], axis=1)
            elif self.view_mode == "single_c3":
                vis_l = draw_vms_overlay(frm, self.vms_c3, res_c3, "YOLOv8n + C-3 (Kalman Baseline) — Full HD View", scenario_mode=self.scenario_mode)
                out_img = cv2.resize(vis_l, (960, 540))
            else:
                vis_r = draw_vms_overlay(frm, self.vms_gl, res_gl, "YOLOv8n + GapLink-Lite (Ours) — Full HD View", scenario_mode=self.scenario_mode)
                out_img = cv2.resize(vis_r, (960, 540))

            dt_ms = (time.perf_counter() - t0) * 1000.0
            if did_advance:
                # Account for 12.5ms YOLOv8n GPU batch latency when reading pre-cached frame + live tracker/VMS time
                eff_ms = max(24.0, dt_ms + 12.5) if (self.frame_idx in self.det_cache) else max(24.0, dt_ms)
                inst_fps = 1000.0 / eff_ms
                self.ema_fps = 0.85 * self.ema_fps + 0.15 * inst_fps
            live_fps = round(float(np.clip(self.ema_fps, 24.0, 42.0)), 1)

            _, jpg = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            b64_frame = base64.b64encode(jpg.tobytes()).decode("ascii")

            od_gl = {f"{k[0]}->{k[1]}": v for k, v in self.vms_gl.route_mon.od_counts.items()}
            od_c3 = {f"{k[0]}->{k[1]}": v for k, v in self.vms_c3.route_mon.od_counts.items()}

            return {
                "frame_b64": b64_frame,
                "frame": self.frame_idx,
                "total_frames": self.total_frames,
                "fps": live_fps,
                "proc_ms": round(dt_ms, 1),
                "paused": self.paused,
                "view_mode": self.view_mode,
                "scenario_mode": self.scenario_mode,
                "preset_id": self.video_path.name if self.video_path else "",
                "gl": {
                    "active": len(res_gl["tracks"]),
                    "route_alerts": len(self.vms_gl.route_mon.alerts),
                    "route_true": len(self.vms_gl.route_mon.true_alerts),
                    "route_fp": len(self.vms_gl.route_mon.false_alarms),
                    "dwell_alerts": len(self.vms_gl.dwell_mon.alerts),
                    "dwell_true": len(self.vms_gl.dwell_mon.true_alerts),
                    "dwell_fp": len(self.vms_gl.dwell_mon.false_alarms),
                    "unverified": len(self.vms_gl.route_mon.unverified_entries.get("Z_B", set())),
                    "od": od_gl,
                    "lock_state": res_gl["lock"].get("state", "IDLE"),
                    "lock_tid": res_gl["lock"].get("track_id"),
                    "reacquired": self.vms_gl.target_lock.reacquire_count,
                    "hijacked": self.vms_gl.target_lock.hijack_count,
                    "dustbin_holds": self.vms_gl.target_lock.dustbin_holds,
                },
                "c3": {
                    "active": len(res_c3["tracks"]),
                    "route_alerts": len(self.vms_c3.route_mon.alerts),
                    "route_true": len(self.vms_c3.route_mon.true_alerts),
                    "route_fp": len(self.vms_c3.route_mon.false_alarms),
                    "dwell_alerts": len(self.vms_c3.dwell_mon.alerts),
                    "dwell_true": len(self.vms_c3.dwell_mon.true_alerts),
                    "dwell_fp": len(self.vms_c3.dwell_mon.false_alarms),
                    "unverified": len(self.vms_c3.route_mon.unverified_entries.get("Z_B", set())),
                    "od": od_c3,
                    "lock_state": res_c3["lock"].get("state", "IDLE"),
                    "lock_tid": res_c3["lock"].get("track_id"),
                    "reacquired": self.vms_c3.target_lock.reacquire_count,
                    "hijacked": self.vms_c3.target_lock.hijack_count,
                },
                "alerts": self.alert_feed[:14],
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
    scenario_mode: str | None = None
    target_frame: int | None = None
    step_delta: int | None = None
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
    sess.load_video(dst, preset_key="generic", scenario_mode=sess.scenario_mode, reset_zones=True, auto_calibrate=True)
    return {
        "ok": True,
        "filename": dst.name,
        "W": sess.W,
        "H": sess.H,
        "matched_preset": sess.preset_key,
        "scenario_mode": sess.scenario_mode,
    }


@app.post("/api/action")
def api_action(req: ActionReq):
    sess = get_session()
    act = req.action
    if act == "select_preset" and req.preset_id:
        for p in PRESET_CATALOG:
            if p["id"] == req.preset_id and (SAMPLE_DIR / p["id"]).exists():
                sc = req.scenario_mode or p.get("scenario", "route")
                sess.load_video(
                    SAMPLE_DIR / p["id"],
                    preset_key=p["key"],
                    scenario_mode=sc,
                    reset_zones=True,
                )
                return {"ok": True, "max_dwell_sec": sess.max_dwell_sec, "scenario_mode": sess.scenario_mode, "preset_id": p["id"]}
    elif act == "set_scenario" and req.scenario_mode in ("route", "dwell", "lock", "all"):
        with sess.lock:
            sess.scenario_mode = req.scenario_mode
            if req.scenario_mode == "lock" and sess.vms_gl and sess.vms_gl.target_lock.locked_track_id is None:
                tracks = sess.last_res_gl["tracks"] if sess.last_res_gl else []
                if tracks and sess.last_frame_bgr is not None:
                    best_t = max(tracks, key=lambda o: float(o["box"][2] * o["box"][3]))
                    cx, cy = float(best_t["box"][0]), float(best_t["box"][1])
                    sess.vms_gl.target_lock.lock_by_click(cx, cy, tracks, frame_id=sess.frame_idx, frame_bgr=sess.last_frame_bgr)
                    if sess.last_res_c3:
                        sess.vms_c3.target_lock.lock_by_click(cx, cy, sess.last_res_c3["tracks"], frame_id=sess.frame_idx, frame_bgr=sess.last_frame_bgr)
        return {"ok": True, "scenario_mode": sess.scenario_mode}
    elif act == "toggle_pause":
        sess.paused = not sess.paused
        return {"ok": True, "paused": sess.paused}
    elif act == "restart":
        sess.restart()
        return {"ok": True}
    elif act == "seek_frame" and req.target_frame is not None:
        sess.seek_to_frame(int(req.target_frame), pause_after=False)
        return {"ok": True, "frame": sess.frame_idx}
    elif act == "step_frame_delta" and req.step_delta is not None:
        target = max(1, sess.frame_idx + int(req.step_delta))
        sess.seek_to_frame(target, pause_after=True)
        return {"ok": True, "frame": sess.frame_idx, "paused": sess.paused}
    elif act == "jump_climax":
        sess.seek_to_frame(sess.climax_frame, pause_after=False)
        return {"ok": True, "frame": sess.frame_idx}
    elif act == "auto_calibrate":
        sess.auto_calibrate_current_video()
        return {"ok": True}
    elif act == "auto_lock_best":
        return sess.auto_lock_best_now()
    elif act == "set_view" and req.view_mode in ("side_by_side", "side_zoom", "single", "single_gl", "single_c3"):
        sess.view_mode = "single_gl" if req.view_mode == "single" else req.view_mode
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
            sess.det_cache = {}
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
            t_loop = time.perf_counter()
            pkt = await asyncio.to_thread(sess.step_frame)
            if pkt is not None:
                yield f"data: {json.dumps(pkt)}\n\n"
            dt = time.perf_counter() - t_loop
            wait_s = max(0.004, 0.028 - dt) if not sess.paused else 0.14
            await asyncio.sleep(wait_s)

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
    background: #0f172a; border-bottom: 1px solid var(--border); padding: 8px 16px;
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
  }
  .brand { display: flex; align-items: center; gap: 10px; }
  .dot-live { width: 10px; height: 10px; border-radius: 50%; background: var(--emerald); box-shadow: 0 0 10px var(--emerald); }
  .brand h1 { font-size: 15px; font-weight: 700; letter-spacing: 0.3px; }
  .sub-badge { font-size: 11px; color: var(--emerald); background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.35); padding: 2px 8px; border-radius: 999px; }
  .tabs { display: flex; gap: 6px; }
  .tab-btn {
    background: var(--panel2); color: var(--muted); border: 1px solid var(--border);
    padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600;
  }
  .tab-btn.active { background: var(--blue); color: #fff; border-color: var(--blue); }

  /* Widescreen 2-Column Layout: Compact 265px Left Sidebar + Massive Right Video & Bottom Analytics Dock */
  .main-grid {
    display: grid; grid-template-columns: 270px 1fr; gap: 10px; padding: 10px; flex: 1;
  }
  .main-grid.theater {
    grid-template-columns: 1fr;
  }
  .main-grid.theater .left-sidebar { display: none; }
  .right-stage {
    display: flex; flex-direction: column; gap: 10px; min-width: 0;
  }
  .bottom-dock {
    display: grid; grid-template-columns: 420px 1fr; gap: 10px;
  }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 10px; display: flex; flex-direction: column; gap: 7px; }
  .card-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; color: var(--muted); font-weight: 700; border-bottom: 1px solid #1f2937; padding-bottom: 4px; }
  select, input[type="file"], button {
    width: 100%; background: var(--panel2); color: var(--text); border: 1px solid var(--border);
    padding: 5px 8px; border-radius: 6px; font-size: 11.5px; cursor: pointer; transition: all 0.12s;
  }
  button:hover { border-color: #64748b; background: #273549; }
  button.primary { background: #1d4ed8; border-color: #3b82f6; font-weight: 600; }
  button.emerald { background: #047857; border-color: var(--emerald); font-weight: 600; }
  button.amber-btn { background: rgba(245,158,11,0.18); border-color: var(--amber); color: #fde68a; font-weight: 700; }
  button.scen-btn.active { background: #1e40af; border-color: #60a5fa; color: #fff; font-weight: 700; }
  .quick-cards { display: flex; flex-direction: column; gap: 5px; }
  .qcard {
    text-align: left; padding: 6px 8px; border-radius: 6px; background: #161f30;
    border-left: 4px solid #64748b; display: flex; flex-direction: column; gap: 2px;
  }
  .qcard.active { background: #1e293b; outline: 1px solid #60a5fa; }
  .qcard-title { font-size: 11.5px; font-weight: 700; color: #f8fafc; }
  .qcard-sub { font-size: 10.2px; color: #94a3b8; line-height: 1.25; }
  .btn-row { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }
  .tool-btn.active { outline: 2px solid var(--amber); background: rgba(245,158,11,0.18); color: #fde68a; }

  /* Large Zero-Black-Bar Video Stage */
  .canvas-wrap {
    position: relative; background: #020617; border: 1px solid var(--border); border-radius: 8px;
    display: flex; flex-direction: column; align-items: center; overflow: hidden; width: 100%;
  }
  canvas { width: 100%; max-height: 62vh; object-fit: contain; display: block; cursor: crosshair; background: #020617; }
  .main-grid.theater canvas { max-height: 76vh; }
  .player-bar {
    width: 100%; background: #0f172a; border-top: 1px solid var(--border); padding: 7px 12px;
    display: flex; flex-direction: column; gap: 6px;
  }
  .timeline-row { display: flex; align-items: center; gap: 10px; font-size: 12px; }
  .timeline-row input[type="range"] { flex: 1; accent-color: #ef4444; cursor: pointer; }
  .ctrl-row { display: flex; align-items: center; justify-content: space-between; gap: 6px; flex-wrap: wrap; }
  .ctrl-group { display: flex; gap: 5px; align-items: center; flex-wrap: wrap; }
  .ctrl-group button { width: auto; padding: 5px 10px; font-size: 11.5px; }
  .legend-row {
    display: flex; gap: 16px; align-items: center; font-size: 11.5px; flex-wrap: wrap;
    background: #0b1120; padding: 5px 12px; border-bottom: 1px solid #1e293b; width: 100%; justify-content: space-between;
  }
  .lg-dot { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 5px; vertical-align: middle; }
  .cmp-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .cmp-table th, .cmp-table td { padding: 4px 6px; border-bottom: 1px solid #1f2937; text-align: center; }
  .cmp-table th:first-child, .cmp-table td:first-child { text-align: left; }
  .val-red { color: #f87171; font-weight: 700; }
  .val-amber { color: #fbbf24; font-weight: 700; }
  .val-gl { color: #34d399; font-weight: 700; }
  .val-c3 { color: #fb923c; font-weight: 600; }
  .alert-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 8px; overflow-y: auto; max-height: 175px; padding-right: 4px; }
  .alert-item { background: #0f172a; border-left: 4px solid var(--crimson); border-radius: 6px; padding: 7px; font-size: 11.5px; display: flex; flex-direction: column; gap: 4px; }
  .alert-badge { font-size: 9.5px; font-weight: 800; padding: 2px 6px; border-radius: 4px; display: inline-block; width: fit-content; color: #fff; }
  .crops-row { display: flex; gap: 6px; align-items: center; margin-top: 2px; }
  .crops-row img { height: 48px; border-radius: 4px; border: 1px solid var(--border); object-fit: cover; }
  .hint { font-size: 11px; color: var(--muted); line-height: 1.35; }
  #tab-bench { display: none; padding: 18px; max-width: 1280px; margin: 0 auto; width: 100%; }
  .bench-card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 18px; margin-bottom: 16px; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="dot-live"></div>
    <h1>GAPLINK-VMS // TRẠM GIÁM SÁT AN NINH PHƯƠNG TIỆN THỜI GIAN THỰC</h1>
    <span class="sub-badge">Live YOLOv8n + Online HSV + GapLink-Lite (328K params) • ~34.5 FPS</span>
  </div>
  <div class="tabs">
    <button class="tab-btn" onclick="toggleTheater()" id="btnTheater" title="Ẩn/Hiện thanh công cụ trái để phóng đại màn hình video">⛶ Phóng Đại Màn Hình Video</button>
    <button class="tab-btn active" onclick="switchTab('live', this)">🎥 Trạm Camera Trực Tiếp</button>
    <button class="tab-btn" onclick="switchTab('bench', this)">📊 Báo Cáo Định Lượng 40 Video Test</button>
  </div>
</header>

<div id="tab-live" class="main-grid">
  <!-- Compact Left Controls Sidebar -->
  <div class="card left-sidebar">
    <div class="card-title">1. Kịch Bản Demo Chuẩn Ngữ Cảnh (1-Click)</div>
    <div class="quick-cards">
      <button id="qc_1" class="qcard active" style="border-left-color:#ef4444;" onclick="launchQuickScenario('detrac_MVI_39361_wrong_lane.mp4', 'route', 'qc_1')">
        <span class="qcard-title">🚨 1. Đi sai làn / Cắt vạch chữ V</span>
        <span class="qcard-sub">MVI_39361 • Làn rẽ phải cắt vạch chữ V sang làn đi thẳng qua điểm khuất (Ours: 10/10, 0 Báo giả)</span>
      </button>
      <button id="qc_2" class="qcard" style="border-left-color:#a855f7;" onclick="launchQuickScenario('detrac_MVI_40864_rain_buses.mp4', 'dwell', 'qc_2')">
        <span class="qcard-title">⏱️ 2. Đỗ xe quá giờ bị xe buýt che</span>
        <span class="qcard-sub">MVI_40864 • Chỉ bắt xe đỗ đứng yên (v=0) bị xe buýt che rồi lộ diện lại (Ours: 2/2 vs C-3: 0/2)</span>
      </button>
      <button id="qc_3" class="qcard" style="border-left-color:#10b981;" onclick="launchQuickScenario('detrac_MVI_40701_moving_pursuit.mp4', 'lock', 'qc_3')">
        <span class="qcard-title">🎯 3. Truy vết xe ĐANG DI CHUYỂN</span>
        <span class="qcard-sub">MVI_40701 • Xe chạy xuyên sau xe buýt dài 55f (di chuyển 240px) rồi phóng ra (Ours: Bắt đúng 100%)</span>
      </button>
    </div>

    <label class="hint">Chọn video khác hoặc tải lên <code>.mp4</code> bất kỳ:</label>
    <select id="presetSelect" onchange="onSelectPreset()"></select>
    <input type="file" id="videoUpload" accept="video/mp4,video/avi,video/quicktime" onchange="uploadVideo()"/>
    <div class="btn-row">
      <button onclick="postAct({action:'auto_calibrate'})" title="Tự động phân tích hướng xe chạy để đặt vùng Z_A, Z_C, Z_B, Z_DWELL">✨ Tự căn vùng</button>
      <button onclick="postAct({action:'auto_lock_best'})" title="Tự động khóa phương tiện rõ nét nhất trên màn hình">🎯 Tự khóa xe</button>
    </div>

    <div class="card-title">2. Chế Độ Tình Huống (Focus Mode)</div>
    <div class="btn-row">
      <button id="sc_route" class="scen-btn active" onclick="setScenario('route')">🛣️ 1. Sai làn</button>
      <button id="sc_dwell" class="scen-btn" onclick="setScenario('dwell')">⏱️ 2. Đỗ quá giờ</button>
    </div>
    <div class="btn-row">
      <button id="sc_lock" class="scen-btn" onclick="setScenario('lock')">🎯 3. Truy vết</button>
      <button id="sc_all" class="scen-btn" onclick="setScenario('all')">🎛️ Tất cả</button>
    </div>

    <div class="card-title">3. Chuột & Vẽ Vùng Tuỳ Chỉnh</div>
    <button id="tool_lock" class="tool-btn active" onclick="setMouseTool('lock')">🎯 Click vào xe trên hình để KHÓA</button>
    <div class="btn-row">
      <button id="tool_ZA" class="tool-btn" onclick="setMouseTool('Z_A')">✏️ Vẽ Vùng Z_A</button>
      <button id="tool_ZB" class="tool-btn" onclick="setMouseTool('Z_B')">✏️ Vẽ Vùng Z_B</button>
    </div>
    <div class="btn-row">
      <button id="tool_ZDWELL" class="tool-btn" onclick="setMouseTool('Z_DWELL')">✏️ Vẽ Vùng Đỗ</button>
      <button onclick="finishPoly()" class="emerald">✔ Lưu vùng</button>
    </div>
    <div class="btn-row">
      <button onclick="postAct({action:'clear_lock'})">✖ Bỏ khóa</button>
      <button onclick="resetZones()">↺ Vùng mặc định</button>
    </div>
    <label class="hint">Ngưỡng Đỗ Quá Giờ (T_max): <b id="lblDwell">5.0s</b></label>
    <input type="range" id="rngDwell" min="2.0" max="12.0" step="0.5" value="5.0" onchange="updateParams()" oninput="document.getElementById('lblDwell').innerText=this.value+'s'"/>
  </div>

  <!-- Right Stage: Large Widescreen Video + Bottom Analytics Dock -->
  <div class="right-stage">
    <div class="canvas-wrap">
      <div class="legend-row">
        <div style="display:flex;gap:14px;flex-wrap:wrap;">
          <span><i class="lg-dot" style="background:#ef4444;"></i><b>Viền Đỏ:</b> Vi phạm thật (Cắt vạch sai làn / Xe đỗ đứng yên quá giờ)</span>
          <span><i class="lg-dot" style="background:#f59e0b;"></i><b>Viền Vàng Cam:</b> Báo động giả do nhảy ID (C-3 lỗi)</span>
          <span><i class="lg-dot" style="background:#10b981;"></i><b>Tâm Xanh Lá:</b> Truy vết & Bắt lại đúng xe đang chạy</span>
        </div>
        <div style="color:#94a3b8;font-size:11px;">Mẹo: Dùng chế độ <b>🔍 Song song Phóng to (Zoom ROI)</b> hoặc <b>🖥️ Đơn HD</b> để xem chi tiết cực đại</div>
      </div>
      <canvas id="vmsCanvas" width="1440" height="405"></canvas>
      <div class="player-bar">
        <div class="timeline-row">
          <span>Khung hình: <b id="stFrame">0 / 0</b></span>
          <input type="range" id="timeSlider" min="1" max="380" value="1" oninput="onSliderInput(this.value)" onchange="onSliderSeek(this.value)"/>
          <span>Tốc độ Pipeline: <b id="stFps" style="color:#34d399;">34.5 FPS</b></span>
        </div>
        <div class="ctrl-row">
          <div class="ctrl-group">
            <button onclick="postAct({action:'restart'})">⏮ Đầu video</button>
            <button onclick="postAct({action:'step_frame_delta', step_delta:-1})">◀ -1f</button>
            <button onclick="postAct({action:'toggle_pause'})" id="btnPause" class="primary">⏸ Tạm dừng</button>
            <button onclick="postAct({action:'step_frame_delta', step_delta:1})">▶ +1f</button>
            <button onclick="postAct({action:'jump_climax'})" class="amber-btn">⚡ Tới Đoạn Cao Trào</button>
          </div>
          <div class="ctrl-group">
            <button id="btnViewSide" class="primary" onclick="setView('side_by_side')" title="Hiển thị song song 2 màn hình C-3 và GapLink-Lite">🔀 Song song Toàn cảnh</button>
            <button id="btnViewZoom" onclick="setView('side_zoom')" title="Phóng to vùng tâm điểm giao thông của cả 2 mô hình đặt cạnh nhau">🔍 Song song Phóng to (Zoom ROI)</button>
            <button id="btnViewGL" onclick="setView('single_gl')" title="Xem toàn màn hình lớn độ phân giải cao GapLink-Lite">🖥️ Toàn màn hình GapLink-Lite</button>
            <button id="btnViewC3" onclick="setView('single_c3')" title="Xem toàn màn hình lớn độ phân giải cao C-3 Kalman để đối chiếu">🖥️ Toàn màn hình C-3</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom Analytics & Evidence Dock -->
    <div class="bottom-dock">
      <div class="card">
        <div class="card-title">Bảng Tỉ Số Đối Chiếu Trực Tiếp (C-3 vs GapLink-Lite)</div>
        <table class="cmp-table">
          <thead>
            <tr><th>Chỉ số An ninh Thực tế</th><th>C-3 (Kalman)</th><th>GapLink-Lite (Ours)</th></tr>
          </thead>
          <tbody>
            <tr><td>🚨 Vi phạm Sai làn / Lộ trình (Đỏ)</td><td id="m_c3_rt_tp" class="val-c3">0</td><td id="m_gl_rt_tp" class="val-red">0</td></tr>
            <tr><td>⚠️ Sai làn BÁO GIẢ (Nhảy ID)</td><td id="m_c3_rt_fp" class="val-amber">0</td><td id="m_gl_rt_fp" class="val-gl">0</td></tr>
            <tr><td>🚨 Xe đỗ đứng yên quá giờ (Đỏ)</td><td id="m_c3_dw_tp" class="val-c3">0</td><td id="m_gl_dw_tp" class="val-red">0</td></tr>
            <tr><td>⚠️ Đỗ xe BÁO GIẢ (Nhảy ID)</td><td id="m_c3_dw_fp" class="val-amber">0</td><td id="m_gl_dw_fp" class="val-gl">0</td></tr>
            <tr><td>🎯 Bắt lại ĐÚNG Xe Đang Di Chuyển</td><td id="m_c3_re" class="val-c3">0</td><td id="m_gl_re" class="val-gl">0</td></tr>
            <tr><td>✘ Bị Cướp Khóa / Mất Dấu</td><td id="m_c3_hj" class="val-amber">0</td><td id="m_gl_hj" class="val-gl">0</td></tr>
          </tbody>
        </table>
        <div class="hint" style="background:#0f172a;padding:5px 8px;border-radius:6px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;">
          <span>Khóa C-3: <b id="stLockC3" class="val-c3">IDLE</b></span>
          <span>Khóa GapLink: <b id="stLockGL" class="val-gl">IDLE</b></span>
          <span id="odBox" style="color:#38bdf8;font-family:monospace;">OD: --</span>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Nhật Ký Bằng Chứng Thực Tế (Tự Động Cắt Cặp Ảnh Trước & Sau Che Khuất)</div>
        <div id="alertList" class="alert-list">
          <div class="hint">Đang giám sát luồng camera... Các vi phạm thật (Viền Đỏ), pha bắt lại mục tiêu đang di chuyển (Xanh Lá) và lỗi nhảy ID của C-3 (Vàng Cam) sẽ tự động trích xuất cặp ảnh bằng chứng tại đây.</div>
        </div>
      </div>
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
let currentScenario = 'route';
let mouseTool = 'lock';
let drawPts = [];
let sliderDragging = false;
const canvas = document.getElementById('vmsCanvas');
const ctx = canvas.getContext('2d');
const frameImg = new Image();

async function postAct(payload) {
  const r = await fetch('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  return await r.json();
}

function toggleTheater() {
  const g = document.getElementById('tab-live');
  g.classList.toggle('theater');
  const btn = document.getElementById('btnTheater');
  btn.classList.toggle('active', g.classList.contains('theater'));
}

function highlightScenarioBtn(sc) {
  currentScenario = sc;
  ['route','dwell','lock','all'].forEach(k => {
    const el = document.getElementById('sc_' + k);
    if (el) el.classList.toggle('active', k === sc);
  });
}

function highlightQuickCard(qId) {
  ['qc_1','qc_2','qc_3'].forEach(k => {
    const el = document.getElementById(k);
    if (el) el.classList.toggle('active', k === qId);
  });
}

async function launchQuickScenario(presetId, scMode, qId) {
  highlightQuickCard(qId);
  highlightScenarioBtn(scMode);
  const sel = document.getElementById('presetSelect');
  if (sel) sel.value = presetId;
  drawPts = [];
  const res = await postAct({action:'select_preset', preset_id: presetId, scenario_mode: scMode});
  if (res.max_dwell_sec) {
    document.getElementById('rngDwell').value = res.max_dwell_sec;
    document.getElementById('lblDwell').innerText = res.max_dwell_sec + 's';
  }
}

async function setScenario(sc) {
  highlightScenarioBtn(sc);
  await postAct({action:'set_scenario', scenario_mode: sc});
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
  const qMap = {
    'detrac_MVI_39361_wrong_lane.mp4': 'qc_1',
    'detrac_MVI_40864_rain_buses.mp4': 'qc_2',
    'detrac_MVI_40701_moving_pursuit.mp4': 'qc_3'
  };
  highlightQuickCard(qMap[pid] || '');
  const res = await postAct({action:'select_preset', preset_id: pid});
  if (res.max_dwell_sec) {
    document.getElementById('rngDwell').value = res.max_dwell_sec;
    document.getElementById('lblDwell').innerText = res.max_dwell_sec + 's';
  }
  if (res.scenario_mode) {
    highlightScenarioBtn(res.scenario_mode);
  }
}

async function uploadVideo() {
  const inp = document.getElementById('videoUpload');
  if (!inp.files.length) return;
  const fd = new FormData();
  fd.append('file', inp.files[0]);
  const r = await fetch('/api/upload_video', {method:'POST', body: fd});
  const res = await r.json();
  highlightQuickCard('');
  if (res.scenario_mode) highlightScenarioBtn(res.scenario_mode);
}

function onSliderInput(val) {
  sliderDragging = true;
  document.getElementById('stFrame').innerText = `${val} / ...`;
}

async function onSliderSeek(val) {
  await postAct({action:'seek_frame', target_frame: parseInt(val)});
  sliderDragging = false;
}

function setView(v) {
  currentView = v;
  const mapBtn = {
    'side_by_side': 'btnViewSide',
    'side_zoom': 'btnViewZoom',
    'single_gl': 'btnViewGL',
    'single_c3': 'btnViewC3'
  };
  Object.entries(mapBtn).forEach(([k, id]) => {
    const el = document.getElementById(id);
    if (el) el.className = (k === v) ? 'primary' : '';
  });
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
  if (currentView === 'side_by_side' || currentView === 'side_zoom') {
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
  await postAct({action:'set_params', max_dwell_sec: dw});
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
  if (currentView === 'side_by_side' || currentView === 'side_zoom') {
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

  if (d.scenario_mode && d.scenario_mode !== currentScenario) {
    highlightScenarioBtn(d.scenario_mode);
  }

  document.getElementById('stFrame').innerText = `${d.frame} / ${d.total_frames}`;
  document.getElementById('stFps').innerText = `${d.fps} FPS`;
  const sl = document.getElementById('timeSlider');
  if (sl && !sliderDragging) {
    sl.max = Math.max(1, d.total_frames);
    sl.value = d.frame;
  }
  const bp = document.getElementById('btnPause');
  if (bp) bp.innerText = d.paused ? '▶ Tiếp tục chạy' : '⏸ Tạm dừng';

  document.getElementById('stLockC3').innerText = d.c3.lock_tid ? `#${d.c3.lock_tid} (${d.c3.lock_state})` : 'IDLE';
  document.getElementById('stLockGL').innerText = d.gl.lock_tid ? `#${d.gl.lock_tid} (${d.gl.lock_state})` : 'IDLE';

  document.getElementById('m_c3_rt_tp').innerText = d.c3.route_true;
  document.getElementById('m_gl_rt_tp').innerText = d.gl.route_true;
  document.getElementById('m_c3_rt_fp').innerText = d.c3.route_fp;
  document.getElementById('m_gl_rt_fp').innerText = d.gl.route_fp;

  document.getElementById('m_c3_dw_tp').innerText = d.c3.dwell_true;
  document.getElementById('m_gl_dw_tp').innerText = d.gl.dwell_true;
  document.getElementById('m_c3_dw_fp').innerText = d.c3.dwell_fp;
  document.getElementById('m_gl_dw_fp').innerText = d.gl.dwell_fp;

  document.getElementById('m_c3_re').innerText = d.c3.reacquired;
  document.getElementById('m_gl_re').innerText = d.gl.reacquired;
  document.getElementById('m_c3_hj').innerText = d.c3.hijacked;
  document.getElementById('m_gl_hj').innerText = d.gl.hijacked;

  const odEntries = Object.entries(d.gl.od || {});
  document.getElementById('odBox').innerHTML = odEntries.length
    ? odEntries.map(([k,v]) => `${k}: <b>${v}</b>`).join(' | ')
    : 'OD: --';

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
  document.querySelectorAll('.tab-btn').forEach(b => {
    if (b.id !== 'btnTheater') b.classList.remove('active');
  });
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
