"""VMS frame overlay painter and side-by-side MP4 renderer on raw camera videos using Live YOLOv8n + GapLink-Lite.

Supports separated single-scenario presentation modes so each security capability is crystal clear:
  - scenario_mode="route": Scenario 1 — Forbidden Route (Z_A -> Z_B) & False-Alarm Prevention
  - scenario_mode="dwell": Scenario 2 — Anti-Loitering Dwell-Time Alert (Z_DWELL) under Passing Occluders
  - scenario_mode="lock":  Scenario 3 — Click-to-Lock Target Pursuit with 2-Sigma Zone & Dustbin Hold
  - scenario_mode="all":   Combined multi-layer view
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import yaml

ROOT = Path(r"D:\gaplink")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import KalmanMahalanobisBaseline
from src.events import box_iou_cxcywh
from src.live_detector import LiveVehicleDetector
from src.model.gaplink import GapLinkAssociator, GapLinkLite
from src.security_analytics import PolygonZone, SecurityVMSEngine
from src.tracker import OnlineTracker


ZONE_DEFAULT_COLORS = {
    "Z_A": (0, 165, 255),       # Amber / Orange (Restricted Origin Z_A)
    "Z_C": (220, 195, 40),      # Cyan / Teal (Permitted Origin Z_C)
    "Z_B": (55, 55, 240),       # Crimson Red (Restricted Destination Z_B)
    "Z_DWELL": (215, 85, 195),  # Magenta / Purple (Anti-Loitering Dwell Zone)
}


def default_preset_zones(preset_key: str, W: int = 960, H: int = 540) -> tuple[list[PolygonZone], list[tuple[str, str]], float]:
    """Return calibrated security zones, forbidden routes, and max_dwell_sec for each camera preset."""
    key = preset_key.lower()
    if "39361" in key:
        # Scenario 1: Road with painted directional arrows & solid V-chevron gore divider.
        # Right lane (Z_A) must merge/turn right; cutting across the solid V-chevron into straight lanes (Z_B) is a lane violation!
        zones = [
            PolygonZone("Z_A", "Z_A: Lan Phai (Cam Cat Vach Chu V)", [(0.55, 0.45), (0.98, 0.45), (0.98, 0.95), (0.52, 0.95)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_A"]),
            PolygonZone("Z_C", "Z_C: Lan Di Thang Hop Le", [(0.45, 0.45), (0.54, 0.45), (0.51, 0.95), (0.43, 0.95)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_C"]),
            PolygonZone("Z_B", "Z_B: Tuyen Di Thang (Cam Cat Vach Tu Z_A)", [(0.15, 0.12), (0.86, 0.12), (0.86, 0.38), (0.15, 0.38)], role="restricted", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_B"]),
            PolygonZone("Z_DWELL", "Z_DWELL: Vung Cam Dung Do (>5.0s)", [(0.10, 0.20), (0.90, 0.20), (0.90, 0.88), (0.10, 0.88)], role="dwell", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_DWELL"]),
        ]
        return zones, [("Z_A", "Z_B")], 5.0
    if "40701" in key:
        # Scenario 3: Arterial intersection with dump truck & articulated bus crossing moving vehicles
        zones = [
            PolygonZone("Z_A", "Z_A: Nhanh Cam", [(0.55, 0.12), (0.96, 0.12), (0.96, 0.45), (0.52, 0.45)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_A"]),
            PolygonZone("Z_C", "Z_C: Nhanh Hop Le", [(0.10, 0.48), (0.65, 0.48), (0.72, 0.95), (0.08, 0.95)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_C"]),
            PolygonZone("Z_B", "Z_B: Vung Kiem Soat", [(0.02, 0.15), (0.46, 0.15), (0.46, 0.46), (0.02, 0.46)], role="restricted", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_B"]),
            PolygonZone("Z_DWELL", "Z_DWELL: Vung Cam Dung Do (>5.0s)", [(0.15, 0.22), (0.85, 0.22), (0.85, 0.82), (0.15, 0.82)], role="dwell", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_DWELL"]),
        ]
        return zones, [("Z_A", "Z_B")], 5.0
    if "40903" in key:
        # Rainy arterial intersection with two articulated buses cutting across traffic flow
        zones = [
            PolygonZone("Z_A", "Z_A: Nhanh Cam", [(0.910, 0.413), (0.654, 0.479), (0.632, 0.204), (0.888, 0.138)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_A"]),
            PolygonZone("Z_C", "Z_C: Nhanh Hop Le", [(0.888, 0.138), (0.632, 0.204), (0.614, 0.010), (0.870, 0.010)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_C"]),
            PolygonZone("Z_B", "Z_B: Vung Cam Xam Nhap", [(0.409, 0.543), (0.053, 0.635), (0.013, 0.145), (0.369, 0.053)], role="restricted", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_B"]),
            PolygonZone("Z_DWELL", "Z_DWELL: Vung Cam Dung Do (>5.0s)", [(0.10, 0.20), (0.90, 0.20), (0.90, 0.88), (0.10, 0.88)], role="dwell", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_DWELL"]),
        ]
        return zones, [("Z_A", "Z_B")], 5.0
    if "40854" in key:
        # 6-lane arterial intersection under pedestrian overpass
        zones = [
            PolygonZone("Z_A", "Z_A: Nhanh Cam", [(0.626, 0.990), (0.355, 0.990), (0.544, 0.431), (0.814, 0.670)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_A"]),
            PolygonZone("Z_C", "Z_C: Nhanh Hop Le", [(0.814, 0.670), (0.544, 0.431), (0.647, 0.061), (0.918, 0.301)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_C"]),
            PolygonZone("Z_B", "Z_B: Vung Cam Xam Nhap", [(0.162, 0.932), (0.010, 0.779), (0.281, 0.010), (0.454, 0.010)], role="restricted", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_B"]),
            PolygonZone("Z_DWELL", "Z_DWELL: Vung Cam Dung Do Giao Lo (>5.0s)", [(0.10, 0.20), (0.90, 0.20), (0.90, 0.88), (0.10, 0.88)], role="dwell", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_DWELL"]),
        ]
        return zones, [("Z_A", "Z_B")], 5.0
    if "40855" in key:
        # Dense 4-way urban intersection
        zones = [
            PolygonZone("Z_A", "Z_A: Nhanh Cam", [(0.995, 0.703), (0.858, 0.914), (0.583, 0.299), (0.731, 0.089)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_A"]),
            PolygonZone("Z_C", "Z_C: Nhanh Hop Le", [(0.731, 0.089), (0.583, 0.299), (0.402, 0.010), (0.550, 0.010)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_C"]),
            PolygonZone("Z_B", "Z_B: Vung Cam Xam Nhap", [(0.746, 0.990), (0.483, 0.990), (0.026, 0.429), (0.290, 0.056)], role="restricted", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_B"]),
            PolygonZone("Z_DWELL", "Z_DWELL: Vung Cam Dung Do Giao Lo (>5.0s)", [(0.10, 0.20), (0.90, 0.20), (0.90, 0.88), (0.10, 0.88)], role="dwell", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_DWELL"]),
        ]
        return zones, [("Z_A", "Z_B")], 5.0
    if "40864" in key:
        # Scenario 2: Rainy avenue where stationary parked vehicles (v=0) are occluded by passing double-decker & articulated buses
        zones = [
            PolygonZone("Z_A", "Z_A: Nhanh Cam", [(0.102, 0.010), (0.321, 0.010), (0.308, 0.220), (0.089, 0.182)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_A"]),
            PolygonZone("Z_C", "Z_C: Nhanh Hop Le", [(0.089, 0.182), (0.308, 0.220), (0.287, 0.619), (0.067, 0.582)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_C"]),
            PolygonZone("Z_B", "Z_B: Cong Kiem Soat", [(0.594, 0.030), (0.990, 0.098), (0.956, 0.734), (0.559, 0.666)], role="restricted", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_B"]),
            PolygonZone("Z_DWELL", "Z_DWELL: Lan Cam Dung / Do Xe (>3.5s)", [(0.06, 0.22), (0.88, 0.22), (0.88, 0.62), (0.06, 0.62)], role="dwell", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_DWELL"]),
        ]
        return zones, [("Z_A", "Z_B")], 3.5
    if "39311" in key:
        zones = [
            PolygonZone("Z_A", "Z_A: Tuyen Chinh", [(0.34, 0.26), (0.68, 0.26), (0.66, 0.52), (0.28, 0.52)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_A"]),
            PolygonZone("Z_C", "Z_C: Nhanh Phai", [(0.70, 0.32), (0.98, 0.32), (0.98, 0.62), (0.70, 0.62)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_C"]),
            PolygonZone("Z_B", "Z_B: Cua Ra Kiem Soat", [(0.16, 0.62), (0.74, 0.62), (0.74, 0.96), (0.12, 0.96)], role="restricted", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_B"]),
            PolygonZone("Z_DWELL", "Z_DWELL: Dao Phan Lan Cam Do (>5.0s)", [(0.10, 0.20), (0.90, 0.20), (0.90, 0.88), (0.10, 0.88)], role="dwell", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_DWELL"]),
        ]
        return zones, [("Z_A", "Z_B")], 5.0
    if "parking" in key:
        zones = [
            PolygonZone("Z_A", "Z_A: Cong Vao", [(0.04, 0.15), (0.48, 0.15), (0.48, 0.52), (0.04, 0.52)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_A"]),
            PolygonZone("Z_B", "Z_B: Lan Noi Bo Han Che", [(0.45, 0.48), (0.96, 0.48), (0.96, 0.95), (0.45, 0.95)], role="restricted", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_B"]),
            PolygonZone("Z_DWELL", "Z_DWELL: Vung Cam Dung Do (>4.0s)", [(0.12, 0.22), (0.88, 0.22), (0.88, 0.88), (0.12, 0.88)], role="dwell", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_DWELL"]),
        ]
        return zones, [("Z_A", "Z_B")], 4.0

    zones = [
        PolygonZone("Z_A", "Z_A: Vung Nguon A", [(0.06, 0.18), (0.48, 0.18), (0.48, 0.46), (0.06, 0.46)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_A"]),
        PolygonZone("Z_C", "Z_C: Vung Nguon C", [(0.52, 0.18), (0.94, 0.18), (0.94, 0.46), (0.52, 0.46)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_C"]),
        PolygonZone("Z_B", "Z_B: Vung Han Che B", [(0.12, 0.60), (0.88, 0.60), (0.88, 0.94), (0.12, 0.94)], role="restricted", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_B"]),
        PolygonZone("Z_DWELL", "Z_DWELL: Vung Giam Sat Dung Do", [(0.15, 0.28), (0.85, 0.28), (0.85, 0.82), (0.15, 0.82)], role="dwell", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_DWELL"]),
    ]
    return zones, [("Z_A", "Z_B")], 5.0


def load_clip_dets_cache(video_path: Path | None) -> dict[int, dict] | None:
    """Load pre-cached live YOLOv8n detections (<video_stem>_dets.npz) if available for instant 35+ FPS playback."""
    if video_path is None:
        return None
    dp = video_path.with_name(f"{video_path.stem}_dets.npz")
    if not dp.exists():
        return None
    try:
        data = np.load(dp)
        frames = data["frames"]
        offsets = data["offsets"]
        boxes = data["boxes"]
        hists = data["hists"]
        out = {}
        for i, f in enumerate(frames):
            s, e = int(offsets[i]), int(offsets[i + 1])
            out[int(f)] = {"boxes": boxes[s:e], "hists": hists[s:e]}
        data.close()
        return out
    except Exception:
        return None


def load_clip_gt_meta(video_path: Path | None) -> dict | None:
    """Load optional post-hoc ground-truth metadata (<video_stem>_gt.npz) if available for a sample video."""
    if video_path is None:
        return None
    gt_p = video_path.with_name(f"{video_path.stem}_gt.npz")
    if not gt_p.exists():
        return None
    try:
        data = np.load(gt_p)
        frames = data["frames"]
        offsets = data["offsets"]
        boxes = data["boxes"]
        tids = data["tids"]
        ignore = data["ignore"] if "ignore" in data else np.zeros((0, 4), dtype=np.float32)
        by_f = {}
        for i, f in enumerate(frames):
            s, e = int(offsets[i]), int(offsets[i + 1])
            by_f[int(f)] = (boxes[s:e], tids[s:e])
        return {"by_frame": by_f, "ignore": ignore}
    except Exception:
        return None


def match_frame_dets_to_gt(det_boxes: np.ndarray, gt_meta: dict | None, frame_idx: int, iou_match: float = 0.45) -> np.ndarray | None:
    """Strictly post-hoc matching between live YOLOv8 boxes and GT boxes so true vs false-alarm alerts are labeled."""
    if gt_meta is None:
        return None
    K = len(det_boxes)
    out_gt = np.full((K,), -1, dtype=np.int32)
    pair = gt_meta["by_frame"].get(int(frame_idx))
    if K == 0 or pair is None:
        return out_gt
    gt_b, gt_t = pair
    if len(gt_b) == 0:
        return out_gt
    ious = box_iou_cxcywh(det_boxes, gt_b)
    r_ind, c_ind = linear_sum_assignment(-ious)
    for r, c in zip(r_ind, c_ind):
        if ious[r, c] >= iou_match:
            out_gt[int(r)] = int(gt_t[c])
    return out_gt


def draw_vms_overlay(
    frame_bgr: np.ndarray,
    vms: SecurityVMSEngine,
    step_out: dict,
    title: str,
    scenario_mode: str = "all",
    show_zones: bool = True,
) -> np.ndarray:
    """Render clean, scenario-focused security camera annotations onto a copy of frame_bgr.

    scenario_mode:
      - "route": Only Scenario 1 (Z_A, Z_C, Z_B polygons + True Route Violation vs False Alarm ID-Hijack badges)
      - "dwell": Only Scenario 2 (Z_DWELL polygon + Dwell Timer progress bars + True Dwell vs False Alarm badges)
      - "lock":  Only Scenario 3 (No zone polygons! Dimmed background + Target Reticle + 2sigma Hold + PiP Reference Card)
      - "all":   All layers combined
    """
    mode = (scenario_mode or "all").lower()
    canvas = frame_bgr.copy()
    H, W = canvas.shape[:2]
    overlay = canvas.copy()

    # 1. Draw Polygon Security Zones filtered by scenario_mode
    if show_zones and mode != "lock":
        drew_any_zone = False
        for z in vms.zones:
            if mode == "route" and z.role == "dwell":
                continue
            if mode == "dwell" and z.role != "dwell":
                continue
            pts = np.round(z.to_pixel_array(W, H)).astype(np.int32)
            if len(pts) < 3:
                continue
            drew_any_zone = True
            col = z.color_bgr
            if z.role == "dwell":
                cv2.fillPoly(overlay, [pts], col)
                cv2.polylines(canvas, [pts], True, col, 2, cv2.LINE_AA)
            else:
                cv2.fillPoly(overlay, [pts], col)
                cv2.polylines(canvas, [pts], True, col, 2, cv2.LINE_AA)
            # Place zone title near top-most vertex (below top banner)
            top_idx = int(np.argmin(pts[:, 1]))
            lx = int(np.clip(pts[top_idx, 0] - 20, 10, W - 220))
            ly = int(np.clip(pts[top_idx, 1] + 22, 72, H - 14))
            lbl_z = z.name if mode == "dwell" else z.zone_id
            (zw, zh), _ = cv2.getTextSize(lbl_z, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            cv2.rectangle(canvas, (lx - 3, ly - zh - 4), (lx + zw + 5, ly + 4), (18, 22, 30), -1)
            cv2.putText(canvas, lbl_z, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2, cv2.LINE_AA)
        if drew_any_zone:
            alpha = 0.12 if mode == "dwell" else 0.18
            cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    # Index true vs false-alarm alerts by track_id and alert frame (only show large text badge for 55 frames after firing!)
    cur_f = int(step_out.get("frame", 0))
    badge_ttl = 55
    route_true_tids = {a["track_id"]: int(a["frame"]) for a in vms.route_mon.true_alerts}
    route_fp_tids = {a["track_id"]: int(a["frame"]) for a in vms.route_mon.false_alarms}
    dwell_true_tids = {a["track_id"]: int(a["frame"]) for a in vms.dwell_mon.true_alerts}
    dwell_fp_tids = {a["track_id"]: int(a["frame"]) for a in vms.dwell_mon.false_alarms}

    dwell_timers = step_out.get("dwell_timers", {})
    lock = step_out.get("lock", {"state": "IDLE", "track_id": None})
    locked_tid = lock.get("track_id")
    max_dw = max(1.0, float(vms.dwell_mon.max_dwell_sec))

    # 2. Draw Active Vehicle Tracks according to scenario_mode
    for o in step_out["tracks"]:
        tid = int(o["track_id"])
        cx, cy, bw, bh = [float(v) for v in o["box"]]
        x1, y1 = int(round(cx - 0.5 * bw)), int(round(cy - 0.5 * bh))
        x2, y2 = int(round(cx + 0.5 * bw)), int(round(cy + 0.5 * bh))

        is_locked = (locked_tid is not None and tid == int(locked_tid)) and (mode in ("lock", "all"))

        if mode == "lock":
            if not is_locked:
                # Subtle background box so all attention stays on the locked target
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (110, 118, 128), 1, cv2.LINE_AA)
                continue

        if is_locked:
            st_lk = lock.get("state", "LOCKED")
            if st_lk == "HIJACKED_WRONG":
                col = (30, 30, 255)  # Bright Red error for hijacked target!
                cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
                cv2.line(canvas, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
                cv2.line(canvas, (x1, y2), (x2, y1), col, 2, cv2.LINE_AA)
                lbl = f"X HIJACKED! (WRONG CAR #{tid})"
                bg_col = (20, 20, 180)
            else:
                col = (50, 255, 110) if st_lk == "REACQUIRED" else (0, 235, 255)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
                clen = max(7, int(min(bw, bh) * 0.30))
                for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                    cv2.line(canvas, (px, py), (px + dx * clen, py), (0, 255, 100), 3, cv2.LINE_AA)
                    cv2.line(canvas, (px, py), (px, py + dy * clen), (0, 255, 100), 3, cv2.LINE_AA)
                lbl = f"RE-ACQUIRED TARGET #{tid}!" if st_lk == "REACQUIRED" else f"LOCKED TARGET #{tid}"
                bg_col = (18, 85, 35) if st_lk == "REACQUIRED" else (20, 30, 40)

            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            ty = max(th + 64, y1 - 6)
            lx = int(np.clip(x1, 4, W - tw - 10))
            cv2.rectangle(canvas, (lx, ty - th - 6), (lx + tw + 8, ty + 4), bg_col, -1)
            cv2.putText(canvas, lbl, (lx + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
            continue

        # Scenario 1 ("route") rendering — Red for TRUE Violations, Amber/Yellow for FALSE ALARMS (ID Switch)
        if mode == "route":
            orig_info = vms.route_mon.track_origin.get(tid)
            orig_z = orig_info["zone_id"] if orig_info else ""
            if tid in route_fp_tids:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 3, cv2.LINE_AA)
                if cur_f - route_fp_tids[tid] > badge_ttl:
                    continue
                lbl = f"! FALSE ALARM (ID SWITCH) #{tid}"
                bg_col = (0, 185, 245)
                txt_col = (15, 15, 20)
                font_sc, thick = 0.48, 2
            elif tid in route_true_tids:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 30, 250), 3, cv2.LINE_AA)
                if cur_f - route_true_tids[tid] > badge_ttl:
                    continue
                lbl = f"VIOLATION Z_A->Z_B #{tid}"
                bg_col = (20, 20, 195)
                txt_col = (255, 255, 255)
                font_sc, thick = 0.48, 2
            elif orig_z == "Z_A":
                col = (0, 175, 255)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
                lbl = f"#{tid} <Z_A"
                bg_col = (20, 24, 30)
                txt_col = col
                font_sc, thick = 0.40, 1
            elif orig_z == "Z_C":
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (195, 175, 50), 1, cv2.LINE_AA)
                continue
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (125, 125, 125), 1, cv2.LINE_AA)
                continue

            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_sc, thick)
            ty = max(th + 64, y1 - 4)
            lx = int(np.clip(x1, 4, W - tw - 10))
            cv2.rectangle(canvas, (lx, ty - th - 5), (lx + tw + 8, ty + 3), bg_col, -1)
            cv2.putText(canvas, lbl, (lx + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, font_sc, txt_col, thick, cv2.LINE_AA)
            continue

        # Scenario 2 ("dwell") rendering — Red for TRUE Dwell Violations, Amber/Yellow for FALSE ALARMS (ID Switch)
        if mode == "dwell":
            dw_s = float(dwell_timers.get(tid, 0.0))
            if tid in dwell_fp_tids:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 3, cv2.LINE_AA)
                if cur_f - dwell_fp_tids[tid] > badge_ttl:
                    continue
                lbl = f"! FALSE ALARM (ID SWITCH) #{tid}"
                bg_col = (0, 185, 245)
                txt_col = (15, 15, 20)
                font_sc, thick = 0.46, 2
            elif tid in dwell_true_tids:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 30, 250), 3, cv2.LINE_AA)
                if cur_f - dwell_true_tids[tid] > badge_ttl:
                    continue
                lbl = f"DWELL VIOLATION #{tid} ({dw_s:.1f}s)"
                bg_col = (20, 20, 195)
                txt_col = (255, 255, 255)
                font_sc, thick = 0.46, 2
            elif dw_s >= 1.8:
                # Draw live timer progress bar above vehicle
                ratio = float(np.clip(dw_s / max_dw, 0.0, 1.0))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (230, 140, 230), 2, cv2.LINE_AA)
                bar_w = max(36, int(bw))
                bx1 = int(np.clip(x1, 4, W - bar_w - 4))
                by1 = max(62, y1 - 4)
                cv2.rectangle(canvas, (bx1, by1 - 5), (bx1 + bar_w, by1), (30, 35, 45), -1)
                cv2.rectangle(canvas, (bx1, by1 - 5), (bx1 + int(bar_w * ratio), by1), (255, 180, 0), -1)
                continue
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (120, 120, 120), 1, cv2.LINE_AA)
                continue

            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_sc, thick)
            ty = max(th + 64, y1 - 10)
            lx = int(np.clip(x1, 4, W - tw - 10))
            cv2.rectangle(canvas, (lx, ty - th - 5), (lx + tw + 8, ty + 3), bg_col, -1)
            cv2.putText(canvas, lbl, (lx + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, font_sc, txt_col, thick, cv2.LINE_AA)
            continue

        # Combined ("all") mode — Red for TRUE Violations, Amber/Yellow for FALSE ALARMS
        is_fp = (tid in route_fp_tids) or (tid in dwell_fp_tids)
        is_tp = (tid in route_true_tids) or (tid in dwell_true_tids)
        orig_info = vms.route_mon.track_origin.get(tid)
        orig_tag = f"<{orig_info['zone_id']}" if orig_info else ""
        dw_s = dwell_timers.get(tid, 0.0)
        dw_tag = f" {dw_s:.1f}s" if dw_s >= 1.5 else ""
        if is_fp:
            col = (0, 200, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
            lbl = f"! FALSE ALARM #{tid}{orig_tag}"
            bg_c, tx_c = (0, 185, 245), (15, 15, 20)
        elif is_tp:
            col = (30, 30, 250)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
            lbl = f"VIOLATION #{tid}{orig_tag}{dw_tag}"
            bg_c, tx_c = (20, 20, 195), (255, 255, 255)
        else:
            col = (195, 225, 75) if orig_info else (150, 150, 150)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
            if not orig_tag and not dw_tag:
                continue
            lbl = f"#{tid}{orig_tag}{dw_tag}"
            bg_c, tx_c = (20, 24, 30), col
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty = max(th + 64, y1 - 4)
        cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), bg_c, -1)
        cv2.putText(canvas, lbl, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, tx_c, 1, cv2.LINE_AA)

    # 3. Draw Target Lock Trajectory Trail, 2-Sigma Search Zone & PiP Reference Card
    if mode in ("lock", "all") and locked_tid is not None:
        trail = lock.get("trail") or []
        if len(trail) >= 2:
            pts_tr = np.round(np.array(trail, dtype=np.float32)).astype(np.int32)
            tr_col = (30, 40, 255) if lock.get("state") == "HIJACKED_WRONG" else (0, 235, 255)
            cv2.polylines(canvas, [pts_tr], False, tr_col, 2, cv2.LINE_AA)

        if lock.get("state") in ("OCCLUDED_SEARCHING", "AMBIGUOUS_HOLD") and lock.get("pred_box") is not None:
            pb = lock["pred_box"]
            sig = lock["sigma_xy"]
            pcx, pcy, pw, ph = [int(round(float(v))) for v in pb]
            rx = int(np.clip(round(2.0 * float(sig[0])), 14, W // 3))
            ry = int(np.clip(round(2.0 * float(sig[1])), 10, H // 3))
            el_col = (0, 180, 255)
            el_ov = canvas.copy()
            cv2.ellipse(el_ov, (pcx, pcy), (rx, ry), 0, 0, 360, el_col, -1, cv2.LINE_AA)
            cv2.addWeighted(el_ov, 0.28, canvas, 0.72, 0, canvas)
            cv2.ellipse(canvas, (pcx, pcy), (rx, ry), 0, 0, 360, el_col, 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (pcx - pw // 2, pcy - ph // 2), (pcx + pw // 2, pcy + ph // 2), el_col, 2, cv2.LINE_AA)
            has_dustbin = hasattr(vms.tracker.associator, "infer_event")
            tag = (
                f"HOLD #{locked_tid} (Dustbin Safe, Occluded {lock['gap_frames']}f)"
                if has_dustbin
                else f"SEARCHING #{locked_tid} (Kalman Extrap, {lock['gap_frames']}f)"
            )
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
            tx = int(np.clip(pcx - tw // 2, 8, W - 180 - tw))
            ty = int(np.clip(pcy - ry - 10, 74, H - 12))
            cv2.rectangle(canvas, (tx - 4, ty - th - 5), (tx + tw + 6, ty + 4), (20, 28, 40), -1)
            cv2.putText(canvas, tag, (tx, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.46, el_col, 2, cv2.LINE_AA)

        # Draw Picture-in-Picture (PiP) Locked Target Reference Card in top-right corner
        ref_img = vms.target_lock.lock_crop_img
        if ref_img is not None and ref_img.size > 0:
            card_w, card_h = 156, 108
            cx1, cy1 = W - card_w - 10, 62
            cx2, cy2 = W - 10, 62 + card_w
            st_lk = lock.get("state", "LOCKED")
            border_col = (30, 30, 255) if st_lk in ("HIJACKED_WRONG", "LOST_EXPIRED") else (50, 235, 110)
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy1 + card_h), (16, 20, 28), -1)
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy1 + card_h), border_col, 2)
            cv2.putText(canvas, f"LOCKED REF #{locked_tid}", (cx1 + 8, cy1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 230, 240), 1, cv2.LINE_AA)
            thumb = cv2.resize(ref_img, (86, 56))
            canvas[cy1 + 22 : cy1 + 78, cx1 + 35 : cx1 + 121] = thumb
            status_lbl = "STATUS: HIJACKED!" if st_lk == "HIJACKED_WRONG" else ("STATUS: LOST!" if st_lk == "LOST_EXPIRED" else f"STATUS: {st_lk[:10]}")
            cv2.putText(canvas, status_lbl, (cx1 + 8, cy1 + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.40, border_col, 2, cv2.LINE_AA)

    # 4. Top 2-Row Scoreboard HUD Banner (Crystal Clear Metrics)
    cv2.rectangle(canvas, (0, 0), (W, 54), (14, 18, 26), -1)
    cv2.line(canvas, (0, 54), (W, 54), (55, 65, 85), 1)
    hdr = f"{title}   |   Frame: {step_out['frame']:04d}"
    cv2.putText(canvas, hdr, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 242, 250), 1, cv2.LINE_AA)

    if mode == "route":
        n_tp = len(vms.route_mon.true_alerts)
        n_fp = len(vms.route_mon.false_alarms)
        b1_txt = f"VI PHAM THAT (True Violations): {n_tp}"
        b2_txt = f"BAO DONG GIA (ID Switch FP): {n_fp}" + (" (CLEAN!)" if n_fp == 0 else "")
        b1_bg, b1_bd = (22, 22, 165), (50, 50, 255)
        b2_bg = (10, 120, 195) if n_fp > 0 else (22, 85, 45)
        b2_bd = (0, 210, 255) if n_fp > 0 else (55, 210, 105)
    elif mode == "dwell":
        n_tp = len(vms.dwell_mon.true_alerts)
        n_fp = len(vms.dwell_mon.false_alarms)
        b1_txt = f"VI PHAM DO QUA GIO (True Dwell): {n_tp}"
        b2_txt = f"BAO DONG GIA (ID Switch FP): {n_fp}" + (" (CLEAN!)" if n_fp == 0 else "")
        b1_bg, b1_bd = (22, 22, 165), (50, 50, 255)
        b2_bg = (10, 120, 195) if n_fp > 0 else (22, 85, 45)
        b2_bd = (0, 210, 255) if n_fp > 0 else (55, 210, 105)
    elif mode == "lock":
        n_ok = vms.target_lock.reacquire_count
        n_bad = vms.target_lock.hijack_count
        b1_txt = f"KHOI PHUC DUNG MUC TIEU (Re-Acquired): {n_ok}"
        b2_txt = f"BI CUOP KHOA / MAT DAU (Hijacked/Lost): {n_bad}" + (" (CLEAN!)" if n_bad == 0 else "")
        b1_bg, b1_bd = (24, 92, 48), (60, 215, 110)
        b2_bg = (22, 24, 165) if n_bad > 0 else (22, 85, 45)
        b2_bd = (50, 65, 255) if n_bad > 0 else (55, 210, 105)
    else:
        n_tp = len(vms.route_mon.true_alerts) + len(vms.dwell_mon.true_alerts)
        n_fp = len(vms.route_mon.false_alarms) + len(vms.dwell_mon.false_alarms)
        b1_txt = f"VI PHAM THAT (True Violations): {n_tp}"
        b2_txt = f"BAO DONG GIA (ID Switch FP): {n_fp}" + (" (CLEAN!)" if n_fp == 0 else "")
        b1_bg, b1_bd = (22, 22, 165), (50, 50, 255)
        b2_bg = (10, 120, 195) if n_fp > 0 else (22, 85, 45)
        b2_bd = (0, 210, 255) if n_fp > 0 else (55, 210, 105)

    # Left Pill Badge (True Violations in Red / Re-acquired Targets in Green)
    cv2.rectangle(canvas, (12, 27), (390, 49), b1_bg, -1)
    cv2.rectangle(canvas, (12, 27), (390, 49), b1_bd, 1)
    cv2.putText(canvas, b1_txt, (20, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (245, 250, 255), 1, cv2.LINE_AA)

    # Right Pill Badge (False Alarms in Amber / Clean in Green)
    cv2.rectangle(canvas, (402, 27), (835, 49), b2_bg, -1)
    cv2.rectangle(canvas, (402, 27), (835, 49), b2_bd, 1)
    cv2.putText(canvas, b2_txt, (410, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 250, 240), 1, cv2.LINE_AA)

    return canvas


def default_lock_schedule(preset_key: str) -> list[tuple[int, float, float]]:
    """Return pre-calibrated (frame_idx, click_x, click_y) target lock schedule for Scenario 3 showcase."""
    key = preset_key.lower()
    if "40701" in key:
        # Scenario 3: Fast-moving minivan (GT=22) traveling across intersection, occluded 55 frames behind
        # an articulated bus while moving 240 pixels from (689, 152) to (940, 235)!
        # GapLink-Lite tracks the moving trajectory and re-acquires 100%; C-3 gets hijacked by the bus!
        return [
            (55, 688.9, 152.0),
        ]
    if "40864" in key:
        return [
            (10, 757.7, 265.1),
            (171, 178.2, 79.0),
        ]
    return [(15, 480.0, 270.0)]


def render_security_comparison_mp4(
    video_path: Path,
    preset_key: str,
    scenario_mode: str,
    out_mp4: Path,
    detector: LiveVehicleDetector,
    gl_assoc: GapLinkAssociator,
    c3_theta: float = 25.0,
    max_frames: int = 385,
    snapshot_frame: int = 220,
    snapshot_path: Path | None = None,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_bgr = []
    while len(frames_bgr) < max_frames:
        ok, frm = cap.read()
        if not ok:
            break
        frames_bgr.append(frm)
    cap.release()

    gt_meta = load_clip_gt_meta(video_path)
    cached_dets = load_clip_dets_cache(video_path)
    if cached_dets is not None and len(cached_dets) >= len(frames_bgr):
        seq_dets = cached_dets
    else:
        ig = gt_meta["ignore"] if gt_meta is not None else None
        dets_list = detector.detect_batch(frames_bgr, ignore_xywh=ig)
        seq_dets = {i + 1: dets_list[i] for i in range(len(dets_list))}

    zones_l, forb_l, max_dw = default_preset_zones(preset_key, W, H)
    zones_r, forb_r, _ = default_preset_zones(preset_key, W, H)

    c3_assoc = KalmanMahalanobisBaseline(theta=c3_theta, name="C-3")
    vms_l = SecurityVMSEngine(
        OnlineTracker(associator=c3_assoc, W=W, H=H, min_hits=3),
        zones=zones_l,
        forbidden_routes=forb_l,
        max_dwell_sec=max_dw,
        fps=fps,
    )
    vms_r = SecurityVMSEngine(
        OnlineTracker(associator=gl_assoc, W=W, H=H, min_hits=3),
        zones=zones_r,
        forbidden_routes=forb_r,
        max_dwell_sec=max_dw,
        fps=fps,
    )

    lock_sched = default_lock_schedule(preset_key) if scenario_mode == "lock" else []
    sched_idx = 0

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_mp4), fps=min(25, int(round(fps))), codec="libx264", quality=8, macro_block_size=None)
    pane_w, pane_h = 640, 360
    snap_saved = False

    for idx, frm in enumerate(frames_bgr):
        f = idx + 1
        d = seq_dets[f]
        gt_tids = match_frame_dets_to_gt(d["boxes"], gt_meta, f)
        res_l = vms_l.step(f, d["boxes"], d["hists"], det_gt_tids=gt_tids, frame_bgr=frm)
        res_r = vms_r.step(f, d["boxes"], d["hists"], det_gt_tids=gt_tids, frame_bgr=frm)

        if sched_idx < len(lock_sched) and f == lock_sched[sched_idx][0]:
            _, lx, ly = lock_sched[sched_idx]
            vms_l.target_lock.lock_by_click(lx, ly, res_l["tracks"], frame_id=f, frame_bgr=frm, clear_totals=False)
            vms_r.target_lock.lock_by_click(lx, ly, res_r["tracks"], frame_id=f, frame_bgr=frm, clear_totals=False)
            sched_idx += 1

        vis_l = draw_vms_overlay(frm, vms_l, res_l, "LIVE YOLOv8n + C-3 (Kalman Baseline)", scenario_mode=scenario_mode)
        vis_r = draw_vms_overlay(frm, vms_r, res_r, "LIVE YOLOv8n + GapLink-Lite (Ours)", scenario_mode=scenario_mode)

        comb = np.concatenate([cv2.resize(vis_l, (pane_w, pane_h)), cv2.resize(vis_r, (pane_w, pane_h))], axis=1)
        writer.append_data(cv2.cvtColor(comb, cv2.COLOR_BGR2RGB))

        if snapshot_path is not None and not snap_saved and f == min(snapshot_frame, len(frames_bgr)):
            cv2.imwrite(str(snapshot_path), comb)
            snap_saved = True

    writer.close()
    return {
        "video": out_mp4.name,
        "scenario_mode": scenario_mode,
        "frames": len(frames_bgr),
        "gl_route_true": len(vms_r.route_mon.true_alerts),
        "gl_route_fp": len(vms_r.route_mon.false_alarms),
        "c3_route_true": len(vms_l.route_mon.true_alerts),
        "c3_route_fp": len(vms_l.route_mon.false_alarms),
        "gl_dwell_true": len(vms_r.dwell_mon.true_alerts),
        "gl_dwell_fp": len(vms_r.dwell_mon.false_alarms),
        "c3_dwell_true": len(vms_l.dwell_mon.true_alerts),
        "c3_dwell_fp": len(vms_l.dwell_mon.false_alarms),
        "gl_reacquired": vms_r.target_lock.reacquire_count,
        "gl_hijacked": vms_r.target_lock.hijack_count,
        "c3_reacquired": vms_l.target_lock.reacquire_count,
        "c3_hijacked": vms_l.target_lock.hijack_count,
    }


def main():
    rel_dir = ROOT / "release"
    tcfg = yaml.safe_load((rel_dir / "tracker.yaml").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(rel_dir / "final.pt", map_location=device, weights_only=False)
    model = GapLinkLite(use_ctx=True, use_color=True, gap_aware=True).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    gl_assoc = GapLinkAssociator(model, device=device, tau=float(tcfg["gaplink_tau"]), calib=None)
    detector = LiveVehicleDetector(imgsz=640)

    sample_dir = ROOT / "demo" / "sample_videos"
    out_dir = ROOT / "demo" / "out"
    fig_dir = ROOT / "report" / "figures"
    latex_fig = ROOT / "report" / "latex" / "figures"
    cap_fig = Path(r"D:\Capstone-Project_latex\Capstone-Project_latex\figures")

    # 3 Semantically-Verified Scenarios on 3 Distinct Real Traffic Videos:
    # 1. Scenario 1 (Wrong-Lane / V-Chevron Gore Crossing Z_A -> Z_B): MVI_39361 (Clear road arrows & V-chevron markings)
    # 2. Scenario 2 (Illegal Parking / Stationary Stopping under Bus Occlusion Z_DWELL): MVI_40864 (Stationary v=0 cars occluded by passing buses)
    # 3. Scenario 3 (Moving Target Trajectory Pursuit across Bus Occlusion): MVI_40701 (Fast-moving vehicle occluded 55f while traveling 240px behind bus)
    jobs = [
        ("detrac_MVI_39361_wrong_lane.mp4", "39361", "route", "security_demo_01_intersection.mp4", 240, "fig_vms_demo_intersection.png"),
        ("detrac_MVI_40864_rain_buses.mp4", "40864", "dwell", "security_demo_02_rain_dwell.mp4", 120, "fig_vms_demo_rain.png"),
        ("detrac_MVI_40701_moving_pursuit.mp4", "40701", "lock", "security_demo_03_gore_zone.mp4", 118, "fig_vms_demo_facility.png"),
    ]
    manifest = []
    for src_fn, pkey, smode, out_fn, snap_f, snap_fn in jobs:
        src_p = sample_dir / src_fn
        if not src_p.exists():
            continue
        snap_p = fig_dir / snap_fn
        info = render_security_comparison_mp4(
            src_p,
            pkey,
            smode,
            out_dir / out_fn,
            detector,
            gl_assoc,
            c3_theta=float(tcfg["c3_theta"]),
            max_frames=385,
            snapshot_frame=snap_f,
            snapshot_path=snap_p,
        )
        manifest.append(info)
        print("Rendered Separated Scenario Demo:", info, flush=True)
        if snap_p.exists():
            if latex_fig.exists():
                shutil.copy2(snap_p, latex_fig / snap_fn)
            if cap_fig.exists():
                shutil.copy2(snap_p, cap_fig / snap_fn)

    # Also copy to descriptive scenario filenames so user can open them directly by scenario name
    shutil.copy2(out_dir / "security_demo_01_intersection.mp4", out_dir / "scenario_01_wrong_lane_MVI_39361.mp4")
    shutil.copy2(out_dir / "security_demo_02_rain_dwell.mp4", out_dir / "scenario_02_illegal_parking_dwell_MVI_40864.mp4")
    shutil.copy2(out_dir / "security_demo_03_gore_zone.mp4", out_dir / "scenario_03_moving_target_pursuit_MVI_40701.mp4")

    (out_dir / "security_demo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
