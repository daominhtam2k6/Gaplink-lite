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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.events import box_iou_cxcywh
from src.live_detector import LiveVehicleDetector
from src.model.gaplink import GapLinkAssociator
from src.security_analytics import PolygonZone, SecurityVMSEngine
from src.tracker import OnlineTracker
from demo.clip_meta import load_clip_meta, slow_windows, zones_from_meta
from demo.relink_viz import MIN_GAP as RELINK_MIN_GAP, attach_relink, draw_relink_layer, relink_summary


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



ID_STABLE = 3    # consecutive frames before a GT identity is trusted
ID_NEW_TTL = 75  # frames a "(was #x)" tag stays visible after a vehicle gets a new tracker ID


def _id_states(vms, step_out: dict, cur_f: int) -> dict:
    """Per-track identity status, judged against ground truth when the clip has it.

    Returns tid -> (kind, text): kind in {"ok", "switch", "newid"}.
      switch : this tracker ID now sits on a different real vehicle than the one it started on
      newid  : this real vehicle was shown under another tracker ID before (identity lost)
    State lives on the engine and resets if the frame index goes backwards (replay / seek).
    """
    st = getattr(vms, "_idviz", None)
    if st is None or cur_f < st["last_f"]:
        st = {"first_gt": {}, "gt_tid": {}, "newid": {}, "cand": {}, "stable": {}, "last_f": cur_f}
        vms._idviz = st
    st["last_f"] = cur_f
    out = {}
    active = {int(o["track_id"]) for o in step_out.get("tracks", [])}
    for o in step_out.get("tracks", []):
        tid = int(o["track_id"])
        g_raw = int(o.get("gt_tid", -1))
        # debounce: a GT identity counts only after ID_STABLE consecutive frames (IoU matching can flicker)
        cg, cn = st["cand"].get(tid, (-1, 0))
        cn = cn + 1 if g_raw == cg else 1
        st["cand"][tid] = (g_raw, cn)
        if g_raw >= 0 and cn >= ID_STABLE:
            st["stable"][tid] = g_raw
        g = st["stable"].get(tid, -1)
        if g >= 0 and cn >= ID_STABLE and g_raw == g:
            st["first_gt"].setdefault(tid, g)
            prev = st["gt_tid"].get(g)
            if prev is not None and prev != tid and prev not in active:
                st["newid"][tid] = (prev, cur_f)
            st["gt_tid"][g] = tid
        g0 = st["first_gt"].get(tid, -1)
        if g >= 0 and g0 >= 0 and g != g0:
            out[tid] = ("switch", f"#{tid} SWITCH")
        elif tid in st["newid"] and cur_f - st["newid"][tid][1] <= ID_NEW_TTL:
            out[tid] = ("newid", f"#{tid} (was #{st['newid'][tid][0]})")
        else:
            out[tid] = ("ok", f"#{tid}")
    return out


def _draw_id_tags(canvas: np.ndarray, step_out: dict, states: dict) -> None:
    """Small tracker-ID tag under every box, so every identity change on screen is visible."""
    H, W = canvas.shape[:2]
    style = {"ok": ((28, 30, 36), (235, 235, 235)),
             "switch": ((0, 185, 245), (15, 15, 20)),
             "newid": ((150, 40, 200), (255, 255, 255))}
    for o in step_out.get("tracks", []):
        tid = int(o["track_id"])
        kind, txt = states.get(tid, ("ok", f"#{tid}"))
        cx, cy, bw, bh = [float(v) for v in o["box"]]
        x1, y2 = int(round(cx - 0.5 * bw)), int(round(cy + 0.5 * bh))
        fs = 0.46 if kind == "ok" else 0.52
        th_ = 1
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th_)
        lx = int(np.clip(x1, 2, W - tw - 6))
        ty = int(np.clip(y2 + th + 4, th + 4, H - 4))
        bg, fg = style[kind]
        cv2.rectangle(canvas, (lx, ty - th - 3), (lx + tw + 5, ty + 2), bg, -1)
        cv2.putText(canvas, txt, (lx + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, fg, th_, cv2.LINE_AA)
        if kind == "switch":
            x2, y1 = int(round(cx + 0.5 * bw)), int(round(cy - 0.5 * bh))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 185, 245), 2, cv2.LINE_AA)


def draw_vms_overlay(
    frame_bgr: np.ndarray,
    vms: SecurityVMSEngine,
    step_out: dict,
    title: str,
    scenario_mode: str = "all",
    show_zones: bool = True,
    show_ids: bool = True,
    show_cands: bool = False,
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
    if show_zones and mode not in ("lock", "relink"):
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

        if mode == "relink":
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (175, 175, 175), 1, cv2.LINE_AA)
            continue

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

    # 2b. Tracker-ID tag on every vehicle (identity changes judged against GT when available)
    _id_st = _id_states(vms, step_out, cur_f)  # always updated so toggling on mid-video stays consistent
    if show_ids:
        # in identity-recovery mode the verdicts are drawn by the relink layer: plain ID tags only
        _draw_id_tags(canvas, step_out, {} if mode == "relink" else _id_st)

    # 2c. Identity recovery layer: ghosts, predicted position / uncertainty, re-link verdicts
    _rec = getattr(vms, "relink", None)
    if _rec is not None:
        _rec.update(vms.tracker, step_out, cur_f)
        if mode == "relink":
            draw_relink_layer(canvas, _rec, step_out, cur_f,
                              is_gaplink=(getattr(_rec.assoc, "kind", "") == "gaplink"), show_cands=show_cands)

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
                else f"SEARCHING #{locked_tid} ({getattr(vms.tracker.associator, 'name', 'baseline')} extrap, {lock['gap_frames']}f)"
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
    elif mode == "relink":
        c = relink_summary(_rec)
        b1_txt = f"GHEP LAI DUNG (Re-linked OK): {c['ok']}"
        b2_txt = f"GHEP NHAM: {c['wrong']}   |   BO LO, CAP ID MOI: {c['missed']}"
        b1_bg, b1_bd = (24, 92, 48), (60, 215, 110)
        bad = c["wrong"] > 0
        b2_bg = (22, 24, 165) if bad else ((10, 120, 195) if c["missed"] > 0 else (22, 85, 45))
        b2_bd = (50, 65, 255) if bad else ((0, 210, 255) if c["missed"] > 0 else (55, 210, 105))
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
        # Lock point fixed when the clip was chosen (before v1.3); outcome is whatever the trackers do.
        return [
            (55, 688.9, 152.0),
        ]
    if "40864" in key:
        return [
            (10, 757.7, 265.1),
            (171, 178.2, 79.0),
        ]
    return [(15, 480.0, 270.0)]


OUT_W, OUT_H = 1920, 1080
PANE_W, PANE_H = 960, 540
SLOW_FACTOR = 4          # 0.25x playback inside key-moment windows
ZOOM_MAX = 1.9           # crop zoom reached in the middle of a key-moment window
EASE = 10                # frames to ease the zoom in / out


def _zoom_at(img: np.ndarray, cx: float, cy: float, z: float) -> np.ndarray:
    """Crop around normalised centre (cx, cy) by factor z and resize back to the original size."""
    if z <= 1.001:
        return img
    H, W = img.shape[:2]
    cw, ch = int(round(W / z)), int(round(H / z))
    x0 = int(np.clip(round(cx * W - cw / 2), 0, W - cw))
    y0 = int(np.clip(round(cy * H - ch / 2), 0, H - ch))
    # crop + linear resize: same picture as the former affine warp, ~4x cheaper
    return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)


def _window_state(f: int, wins: list) -> tuple[bool, float, float, float]:
    """(in_slow_window, zoom, cx, cy) for frame f."""
    for a, b, cx, cy in wins:
        if a <= f <= b:
            ramp = min(f - a, b - f) / float(EASE)
            ease = 0.5 - 0.5 * np.cos(np.pi * float(np.clip(ramp, 0.0, 1.0)))
            return True, 1.0 + (ZOOM_MAX - 1.0) * ease, cx, cy
    return False, 1.0, 0.5, 0.5


def _compose(vis_l: np.ndarray, vis_r: np.ndarray, slow: bool, scenario_title: str,
             left_label: str = "Baseline", right_label: str = "GapLink-Lite", mode: str = "") -> np.ndarray:
    """1920x1080 frame: two 960x540 panes centred vertically, minimal header, slow-motion badge."""
    canvas = np.full((OUT_H, OUT_W, 3), (18, 14, 10), dtype=np.uint8)
    y0 = (OUT_H - PANE_H) // 2
    canvas[y0:y0 + PANE_H, 0:PANE_W] = cv2.resize(vis_l, (PANE_W, PANE_H), interpolation=cv2.INTER_AREA)
    canvas[y0:y0 + PANE_H, PANE_W:OUT_W] = cv2.resize(vis_r, (PANE_W, PANE_H), interpolation=cv2.INTER_AREA)
    cv2.line(canvas, (PANE_W, y0), (PANE_W, y0 + PANE_H), (60, 60, 60), 2)
    cv2.putText(canvas, "GapLink-VMS", (40, 120), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, scenario_title, (40, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (190, 190, 190), 2, cv2.LINE_AA)
    if slow:
        txt = "SLOW MOTION  x0.25"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
        x, y = OUT_W - tw - 60, 140
        cv2.rectangle(canvas, (x - 20, y - th - 18), (x + tw + 20, y + 16), (40, 170, 255), -1)
        cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.2, (15, 15, 15), 2, cv2.LINE_AA)
    # pane names stay visible even when the panes are zoomed
    yl = y0 + PANE_H + 52
    def _short(t: str) -> str:  # cv2 fonts are ASCII-only: keep the name before the first " ("
        return t.split(" (")[0].encode("ascii", "ignore").decode().strip()
    for x0, lab, col in ((0, _short(left_label), (90, 90, 235)), (PANE_W, _short(right_label), (90, 210, 90))):
        (tw, _), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        cv2.putText(canvas, lab, (x0 + (PANE_W - tw) // 2, yl), cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2, cv2.LINE_AA)
    # ID legend
    yg = yl + 70
    if mode == "relink":
        items = [("#12 lost", (45, 45, 45), (230, 230, 230), "hidden (dashed) + predicted position"),
                 ("RE-LINKED", (70, 200, 70), (255, 255, 255), "same vehicle, same ID"),
                 ("WRONG LINK", (40, 40, 235), (255, 255, 255), "ID given to another vehicle"),
                 ("MISSED", (200, 60, 200), (255, 255, 255), "vehicle got a new ID")]
    else:
        items = [("#12", (28, 30, 36), (235, 235, 235), "tracker ID"),
                 ("#12 SWITCH", (0, 185, 245), (15, 15, 20), "ID moved onto another vehicle"),
                 ("#31 (was #12)", (150, 40, 200), (255, 255, 255), "same vehicle, new ID")]
    x = 40
    for tag, bg, fg, desc in items:
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(canvas, (x, yg - th - 8), (x + tw + 12, yg + 8), bg, -1)
        cv2.putText(canvas, tag, (x + 6, yg), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fg, 2, cv2.LINE_AA)
        x += tw + 24
        cv2.putText(canvas, desc, (x, yg), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        (dw, _), _ = cv2.getTextSize(desc, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        x += dw + (40 if mode == "relink" else 60)
    note = (f"re-links after gaps >= {RELINK_MIN_GAP} frames, judged against UA-DETRAC ground truth" if mode == "relink"
            else "identities checked against UA-DETRAC ground truth")
    cv2.putText(canvas, note, (40 if mode == "relink" else x, yg + (50 if mode == "relink" else 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1, cv2.LINE_AA)
    return canvas


def render_security_comparison_mp4(
    video_path: Path,
    preset_key: str,
    scenario_mode: str,
    out_mp4: Path,
    detector: LiveVehicleDetector,
    gl_assoc: GapLinkAssociator,
    bl_assoc=None,
    bl_label: str = "Baseline",
    gl_label: str = "GapLink-Lite",
    max_frames: int = 385,
    snapshot_frame: int = 220,
    snapshot_path: Path | None = None,
    extra_writer=None,
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

    meta = load_clip_meta(video_path)
    if meta and meta.get("zones"):
        zones_l, forb_l, max_dw = zones_from_meta(meta)
        zones_r, forb_r, _ = zones_from_meta(meta)
    else:
        zones_l, forb_l, max_dw = default_preset_zones(preset_key, W, H)
        zones_r, forb_r, _ = default_preset_zones(preset_key, W, H)
    wins = slow_windows(meta)

    vms_l = SecurityVMSEngine(
        OnlineTracker(associator=bl_assoc, W=W, H=H, min_hits=3),
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

    attach_relink(vms_l)
    attach_relink(vms_r)

    if scenario_mode == "lock":
        lock_sched = [tuple(x) for x in meta["lock"]] if meta and meta.get("lock") else default_lock_schedule(preset_key)
    else:
        lock_sched = []
    sched_idx = 0

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_mp4), fps=min(25, int(round(fps))), codec="libx264", quality=None, macro_block_size=None,
                                pixelformat="yuv420p", ffmpeg_params=["-crf", "18", "-preset", "veryfast"])
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

        vis_l = draw_vms_overlay(frm, vms_l, res_l, f"LIVE YOLOv8n + {bl_label}", scenario_mode=scenario_mode)
        vis_r = draw_vms_overlay(frm, vms_r, res_r, f"LIVE YOLOv8n + {gl_label}", scenario_mode=scenario_mode)

        slow, z, zcx, zcy = _window_state(f, wins)
        comb = _compose(_zoom_at(vis_l, zcx, zcy, z), _zoom_at(vis_r, zcx, zcy, z), slow,
                        (meta or {}).get("title", scenario_mode), left_label=bl_label, right_label=gl_label,
                        mode=scenario_mode)
        rgb = cv2.cvtColor(comb, cv2.COLOR_BGR2RGB)
        for _ in range(SLOW_FACTOR if slow else 1):
            writer.append_data(rgb)
            if extra_writer is not None:
                extra_writer.append_data(rgb)

        if snapshot_path is not None and not snap_saved and f == min(snapshot_frame, len(frames_bgr)):
            cv2.imwrite(str(snapshot_path), np.concatenate([cv2.resize(vis_l, (pane_w, pane_h)), cv2.resize(vis_r, (pane_w, pane_h))], axis=1))
            snap_saved = True

    writer.close()
    return {
        "video": out_mp4.name,
        "scenario_mode": scenario_mode,
        "frames": len(frames_bgr),
        "gl_route_true": len(vms_r.route_mon.true_alerts),
        "gl_route_fp": len(vms_r.route_mon.false_alarms),
        "baseline": bl_label,
        "gaplink": gl_label,
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
        "gl_relink": relink_summary(getattr(vms_r, "relink", None)),
        "c3_relink": relink_summary(getattr(vms_l, "relink", None)),
    }


def main():
    import argparse
    from demo.vms_profile import BASELINE_INFO, DEFAULT_BASELINE, load_profile, make_baseline, make_gaplink, resolve_root

    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=sorted(BASELINE_INFO), default=DEFAULT_BASELINE)
    ap.add_argument("--profile", choices=["v1.3", "v1.1"], default="v1.3")
    ap.add_argument("--update-report-figs", action="store_true", help="also overwrite report/latex figure snapshots")
    ap.add_argument("--showcase", action="store_true", help="render the mined showcase clips (demo/sample_videos/showcase_*)")
    a = ap.parse_args()
    root = resolve_root()
    prof = load_profile(root, version=a.profile)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gl_assoc = make_gaplink(prof, device)
    detector = LiveVehicleDetector(imgsz=640)

    sample_dir = root / "demo" / "sample_videos"
    out_dir = root / "demo" / "out" / f"{prof.version}_vs_{a.baseline}"
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = root / "report" / "figures"
    latex_fig = root / "report" / "latex" / "figures"
    paths_cfg = yaml.safe_load((root / "configs" / "paths.yaml").read_text(encoding="utf-8"))
    cap_fig = Path(paths_cfg.get("latex_template_dir", "")) / "figures"

    if a.showcase:
        render_showcase(root, prof, a.baseline, gl_assoc, detector)
        return

    # Same three clips as before (chosen 18-20/11/2026, before v1.3 existed); results reported as they come out.
    jobs = [
        ("detrac_MVI_39361_wrong_lane.mp4", "39361", "route", "scenario_01_wrong_lane_MVI_39361.mp4", 240, "fig_vms_demo_intersection.png"),
        ("detrac_MVI_40864_rain_buses.mp4", "40864", "dwell", "scenario_02_illegal_parking_dwell_MVI_40864.mp4", 120, "fig_vms_demo_rain.png"),
        ("detrac_MVI_40701_moving_pursuit.mp4", "40701", "lock", "scenario_03_moving_target_pursuit_MVI_40701.mp4", 118, "fig_vms_demo_facility.png"),
    ]
    manifest = {"profile": prof.version, "gaplink": prof.gaplink_label, "gaplink_ckpt": str(prof.ckpt.name),
                "baseline": prof.baseline_label(a.baseline), "clips": []}
    for src_fn, pkey, smode, out_fn, snap_f, snap_fn in jobs:
        src_p = sample_dir / src_fn
        if not src_p.exists():
            continue
        snap_p = snap_dir / snap_fn
        info = render_security_comparison_mp4(
            src_p, pkey, smode, out_dir / out_fn, detector, gl_assoc,
            bl_assoc=make_baseline(a.baseline, prof), bl_label=prof.baseline_label(a.baseline), gl_label=prof.gaplink_label,
            max_frames=385, snapshot_frame=snap_f, snapshot_path=snap_p,
        )
        manifest["clips"].append(info)
        print("Rendered:", info, flush=True)
        if a.update_report_figs and snap_p.exists():
            for d in (fig_dir, latex_fig, cap_fig):
                if d.exists():
                    shutil.copy2(snap_p, d / snap_fn)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_dir / 'manifest.json'}")


def render_showcase(root: Path, prof, baseline: str, gl_assoc, detector):
    """Render every demo/sample_videos/showcase_*.mp4 (with _meta.json) and one combined reel."""
    from demo.vms_profile import make_baseline
    clips = sorted((root / "demo" / "sample_videos").glob("showcase_*.mp4"))
    clips = [c for c in clips if load_clip_meta(c)]
    out_dir = root / "demo" / "out" / f"showcase_{prof.version}_vs_{baseline}"
    out_dir.mkdir(parents=True, exist_ok=True)
    reel = imageio.get_writer(str(out_dir / "showcase_reel.mp4"), fps=25, codec="libx264", quality=None, macro_block_size=None,
                              pixelformat="yuv420p", ffmpeg_params=["-crf", "18", "-preset", "veryfast"])
    manifest = {"profile": prof.version, "gaplink": prof.gaplink_label, "baseline": prof.baseline_label(baseline), "clips": []}
    for c in clips:
        meta = load_clip_meta(c)
        info = render_security_comparison_mp4(
            c, "showcase", meta["scenario"], out_dir / f"{c.stem}.mp4", detector, gl_assoc,
            bl_assoc=make_baseline(baseline, prof), bl_label=prof.baseline_label(baseline), gl_label=prof.gaplink_label,
            max_frames=10_000, snapshot_frame=(meta.get("key_moments") or [{"frame": 1}])[0]["frame"],
            snapshot_path=out_dir / f"{c.stem}_key.png", extra_writer=reel)
        info["expected"] = meta.get("expected")
        manifest["clips"].append(info)
        print("Rendered showcase:", info, flush=True)
    reel.close()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
