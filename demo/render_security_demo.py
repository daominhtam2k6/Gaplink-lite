"""VMS frame overlay painter and side-by-side MP4 renderer on raw camera videos using Live YOLOv8n + GapLink-Lite."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

ROOT = Path(r"D:\gaplink")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import KalmanMahalanobisBaseline
from src.live_detector import LiveVehicleDetector
from src.model.gaplink import GapLinkAssociator, GapLinkLite
from src.security_analytics import PolygonZone, SecurityVMSEngine
from src.tracker import OnlineTracker


ZONE_DEFAULT_COLORS = {
    "Z_A": (0, 175, 255),       # Amber / Orange (Restricted Origin)
    "Z_C": (210, 190, 40),      # Cyan / Teal (Permitted Origin)
    "Z_B": (55, 55, 235),       # Crimson Red (Restricted Destination)
    "Z_DWELL": (220, 90, 190),  # Magenta / Purple (Anti-Loitering Dwell Zone)
}


def default_preset_zones(preset_key: str, W: int = 960, H: int = 540) -> tuple[list[PolygonZone], list[tuple[str, str]], float]:
    """Return clean, visually intuitive security zones, forbidden routes, and max_dwell_sec for camera presets."""
    key = preset_key.lower()
    if "40855" in key:
        # 4-way urban intersection: Top-right approach (Z_A) cuts behind central bus into Left exit (Z_B); bottom approach (Z_C)
        zones = [
            PolygonZone("Z_A", "Z_A: Nhánh Cấm Cắt Giao Lộ", [(0.54, 0.16), (0.96, 0.16), (0.96, 0.46), (0.50, 0.46)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_A"]),
            PolygonZone("Z_C", "Z_C: Tuyến Đi Thẳng Hợp Lệ", [(0.18, 0.66), (0.72, 0.66), (0.78, 0.96), (0.14, 0.96)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_C"]),
            PolygonZone("Z_B", "Z_B: Vùng Hạn Chế Cánh Trái", [(0.02, 0.28), (0.44, 0.28), (0.44, 0.62), (0.02, 0.62)], role="restricted", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_B"]),
            PolygonZone("Z_DWELL", "Vùng Cấm Dừng Giao Lộ", [(0.20, 0.24), (0.78, 0.24), (0.80, 0.64), (0.18, 0.64)], role="dwell", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_DWELL"]),
        ]
        return zones, [("Z_A", "Z_B")], 5.0
    if "40864" in key:
        # Rainy avenue with double-decker buses sweeping across inner-lane slow/lingering traffic
        zones = [
            PolygonZone("Z_A", "Z_A: Làn Ngoài", [(0.04, 0.32), (0.36, 0.22), (0.44, 0.48), (0.06, 0.58)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_A"]),
            PolygonZone("Z_C", "Z_C: Làn Trong", [(0.37, 0.14), (0.68, 0.10), (0.72, 0.32), (0.39, 0.36)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_C"]),
            PolygonZone("Z_B", "Z_B: Cổng Kiểm Soát Hạ Nguồn", [(0.45, 0.52), (0.96, 0.42), (0.98, 0.94), (0.42, 0.96)], role="restricted", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_B"]),
            PolygonZone("Z_DWELL", "Vùng Cấm Đỗ / Dừng Lâu", [(0.22, 0.22), (0.82, 0.18), (0.88, 0.68), (0.20, 0.72)], role="dwell", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_DWELL"]),
        ]
        return zones, [("Z_A", "Z_B")], 4.5
    if "39311" in key:
        # Ramp merge + striped zebra gore island (no-stopping & forbidden cut-across)
        zones = [
            PolygonZone("Z_A", "Z_A: Tuyến Chính", [(0.34, 0.26), (0.68, 0.26), (0.66, 0.52), (0.28, 0.52)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_A"]),
            PolygonZone("Z_C", "Z_C: Nhánh Phải", [(0.70, 0.32), (0.98, 0.32), (0.98, 0.62), (0.70, 0.62)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_C"]),
            PolygonZone("Z_B", "Z_B: Cửa Ra Kiểm Soát", [(0.16, 0.62), (0.74, 0.62), (0.74, 0.96), (0.12, 0.96)], role="restricted", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_B"]),
            PolygonZone("Z_DWELL", "Đảo Phân Làn Cấm Đỗ", [(0.66, 0.42), (0.98, 0.42), (0.98, 0.76), (0.68, 0.76)], role="dwell", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_DWELL"]),
        ]
        return zones, [("Z_A", "Z_B")], 4.0
    if "parking" in key:
        # External facility parking / entrance gate camera
        zones = [
            PolygonZone("Z_A", "Z_A: Cổng Vào Khuôn Viên", [(0.04, 0.15), (0.48, 0.15), (0.48, 0.52), (0.04, 0.52)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_A"]),
            PolygonZone("Z_B", "Z_B: Làn Nội Bộ Hạn Chế", [(0.45, 0.48), (0.96, 0.48), (0.96, 0.95), (0.45, 0.95)], role="restricted", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_B"]),
            PolygonZone("Z_DWELL", "Vùng Cấm Dừng Đỗ Lâu", [(0.12, 0.22), (0.88, 0.22), (0.88, 0.88), (0.12, 0.88)], role="dwell", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_DWELL"]),
        ]
        return zones, [("Z_A", "Z_B")], 4.0

    # Generic default for any uploaded raw camera feed
    zones = [
        PolygonZone("Z_A", "Z_A: Vùng Nguồn A", [(0.06, 0.18), (0.48, 0.18), (0.48, 0.46), (0.06, 0.46)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_A"]),
        PolygonZone("Z_C", "Z_C: Vùng Nguồn C", [(0.52, 0.18), (0.94, 0.18), (0.94, 0.46), (0.52, 0.46)], role="entry", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_C"]),
        PolygonZone("Z_B", "Z_B: Vùng Hạn Chế B", [(0.12, 0.60), (0.88, 0.60), (0.88, 0.94), (0.12, 0.94)], role="restricted", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_B"]),
        PolygonZone("Z_DWELL", "Vùng Giám Sát Dừng Đỗ", [(0.15, 0.28), (0.85, 0.28), (0.85, 0.82), (0.15, 0.82)], role="dwell", normalized=True, color_bgr=ZONE_DEFAULT_COLORS["Z_DWELL"]),
    ]
    return zones, [("Z_A", "Z_B")], 5.0


def draw_vms_overlay(
    frame_bgr: np.ndarray,
    vms: SecurityVMSEngine,
    step_out: dict,
    title: str,
    show_zones: bool = True,
) -> np.ndarray:
    """Render professional dark-HUD security camera annotations onto a copy of frame_bgr."""
    canvas = frame_bgr.copy()
    H, W = canvas.shape[:2]
    overlay = canvas.copy()

    # 1. Draw Polygon Security Zones
    if show_zones:
        for z in vms.zones:
            pts = np.round(z.to_pixel_array(W, H)).astype(np.int32)
            if len(pts) < 3:
                continue
            col = z.color_bgr
            if z.role == "dwell":
                cv2.polylines(canvas, [pts], True, col, 1, cv2.LINE_AA)
            else:
                cv2.fillPoly(overlay, [pts], col)
                cv2.polylines(canvas, [pts], True, col, 2, cv2.LINE_AA)
            lx, ly = int(pts[0, 0]) + 5, max(44, int(pts[0, 1]) + 16)
            cv2.putText(canvas, z.zone_id, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, z.zone_id, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0, canvas)

    alerted_route_tids = {a["track_id"] for a in vms.route_mon.alerts}
    alerted_dwell_tids = {a["track_id"] for a in vms.dwell_mon.alerts}
    dwell_timers = step_out.get("dwell_timers", {})
    lock = step_out.get("lock", {"state": "IDLE", "track_id": None})
    locked_tid = lock.get("track_id")

    # 2. Draw Active Vehicle Tracks
    for o in step_out["tracks"]:
        tid = int(o["track_id"])
        cx, cy, bw, bh = [float(v) for v in o["box"]]
        x1, y1 = int(round(cx - 0.5 * bw)), int(round(cy - 0.5 * bh))
        x2, y2 = int(round(cx + 0.5 * bw)), int(round(cy + 0.5 * bh))

        is_locked = (locked_tid is not None and tid == int(locked_tid))
        is_alert = (tid in alerted_route_tids) or (tid in alerted_dwell_tids)

        orig_info = vms.route_mon.track_origin.get(tid)
        orig_tag = f"<{orig_info['zone_id']}" if orig_info else ""
        dw_s = dwell_timers.get(tid, 0.0)
        dw_tag = f" {dw_s:.1f}s" if dw_s >= 1.2 else ""

        if is_locked:
            col = (50, 255, 120) if lock["state"] == "REACQUIRED" else (0, 230, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
            # Corner reticle ticks
            clen = max(6, int(min(bw, bh) * 0.28))
            for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                cv2.line(canvas, (px, py), (px + dx * clen, py), (0, 0, 255), 3, cv2.LINE_AA)
                cv2.line(canvas, (px, py), (px, py + dy * clen), (0, 0, 255), 3, cv2.LINE_AA)
            lbl = f"LOCKED #{tid}" + (" [RE-ACQUIRED!]" if lock["state"] == "REACQUIRED" else "")
        elif is_alert:
            col = (40, 40, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
            lbl = f"! ALERT #{tid}{orig_tag}{dw_tag}"
        else:
            col = (195, 225, 75) if orig_info else (180, 180, 180)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
            lbl = f"#{tid}{orig_tag}{dw_tag}"

        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty = max(th + 36, y1 - 4)
        cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (20, 24, 30), -1)
        cv2.putText(canvas, lbl, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    # 3. Draw Target Lock Trajectory Trail & 2-Sigma Occlusion Search Zone
    trail = lock.get("trail") or []
    if len(trail) >= 2:
        pts_tr = np.round(np.array(trail, dtype=np.float32)).astype(np.int32)
        cv2.polylines(canvas, [pts_tr], False, (0, 230, 255), 2, cv2.LINE_AA)

    if lock.get("state") in ("OCCLUDED_SEARCHING", "AMBIGUOUS_HOLD") and lock.get("pred_box") is not None:
        pb = lock["pred_box"]
        sig = lock["sigma_xy"]
        pcx, pcy, pw, ph = [int(round(float(v))) for v in pb]
        rx = int(np.clip(round(2.0 * float(sig[0])), 10, W // 3))
        ry = int(np.clip(round(2.0 * float(sig[1])), 8, H // 3))
        el_col = (0, 140, 255) if lock["state"] == "AMBIGUOUS_HOLD" else (255, 190, 0)
        el_ov = canvas.copy()
        cv2.ellipse(el_ov, (pcx, pcy), (rx, ry), 0, 0, 360, el_col, -1, cv2.LINE_AA)
        cv2.addWeighted(el_ov, 0.24, canvas, 0.76, 0, canvas)
        cv2.ellipse(canvas, (pcx, pcy), (rx, ry), 0, 0, 360, el_col, 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (pcx - pw // 2, pcy - ph // 2), (pcx + pw // 2, pcy + ph // 2), el_col, 1, cv2.LINE_AA)
        tag = (
            f"HOLD #{locked_tid} (Dustbin Safe, gap={lock['gap_frames']}f)"
            if lock["state"] == "AMBIGUOUS_HOLD"
            else f"SEARCHING #{locked_tid} (2sigma, gap={lock['gap_frames']}f)"
        )
        cv2.putText(canvas, tag, (max(8, pcx - 90), max(48, pcy - ry - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, el_col, 2, cv2.LINE_AA)

    # 4. Top Dark VMS Banner
    cv2.rectangle(canvas, (0, 0), (W, 34), (16, 20, 28), -1)
    n_rt = len(vms.route_mon.alerts)
    n_dw = len(vms.dwell_mon.alerts)
    lk_str = f" | Lock: {lock['state']}" if locked_tid is not None else ""
    hud = f"{title} | F:{step_out['frame']:04d} | RouteAlerts:{n_rt} | DwellAlerts:{n_dw}{lk_str}"
    cv2.putText(canvas, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 242, 250), 1, cv2.LINE_AA)
    return canvas


def pick_interesting_target_track(seq_dets: dict[int, dict], gl_assoc: GapLinkAssociator, W: int, H: int) -> tuple[int, tuple[float, float]] | None:
    """Find a vehicle visible early (frames 15..90) that later experiences a real natural YOLOv8 occlusion and re-emerges."""
    trk = OnlineTracker(associator=gl_assoc, W=W, H=H, min_hits=3)
    first_seen_pos: dict[int, tuple[int, float, float]] = {}
    for f in sorted(seq_dets.keys())[:280]:
        outs = trk.step(f, seq_dets[f]["boxes"], seq_dets[f]["hists"])
        for o in outs:
            tid = int(o["track_id"])
            if tid not in first_seen_pos and 15 <= f <= 95:
                first_seen_pos[tid] = (f, float(o["box"][0]), float(o["box"][1]))
    # Pick track that had a good Stage-2 re-association gap (>= 5 frames)
    best = None
    best_gap = -1
    for r in trk.reassoc_log:
        tid = int(r["track_id"])
        if tid in first_seen_pos and r["gap"] >= 5 and r["gap"] > best_gap:
            best_gap = int(r["gap"])
            f0, cx0, cy0 = first_seen_pos[tid]
            best = (f0 + 2, (cx0, cy0))
    return best


def render_security_comparison_mp4(
    video_path: Path,
    preset_key: str,
    out_mp4: Path,
    detector: LiveVehicleDetector,
    gl_assoc: GapLinkAssociator,
    c3_theta: float = 25.0,
    max_frames: int = 340,
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

    dets_list = detector.detect_batch(frames_bgr)
    seq_dets = {i + 1: dets_list[i] for i in range(len(dets_list))}

    lock_plan = pick_interesting_target_track(seq_dets, gl_assoc, W, H)

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

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_mp4), fps=min(25, int(round(fps))), codec="libx264", quality=7, macro_block_size=None)
    pane_w, pane_h = 640, 360
    snap_saved = False

    for idx, frm in enumerate(frames_bgr):
        f = idx + 1
        d = seq_dets[f]
        res_l = vms_l.step(f, d["boxes"], d["hists"], frame_bgr=frm)
        res_r = vms_r.step(f, d["boxes"], d["hists"], frame_bgr=frm)

        if lock_plan is not None and f == lock_plan[0]:
            lx, ly = lock_plan[1]
            vms_l.target_lock.lock_by_click(lx, ly, res_l["tracks"], frame_id=f, frame_bgr=frm)
            vms_r.target_lock.lock_by_click(lx, ly, res_r["tracks"], frame_id=f, frame_bgr=frm)

        vis_l = draw_vms_overlay(frm, vms_l, res_l, "LIVE YOLOv8n + C-3 (Kalman)")
        vis_r = draw_vms_overlay(frm, vms_r, res_r, "LIVE YOLOv8n + GapLink-Lite (VMS)")

        comb = np.concatenate([cv2.resize(vis_l, (pane_w, pane_h)), cv2.resize(vis_r, (pane_w, pane_h))], axis=1)
        writer.append_data(cv2.cvtColor(comb, cv2.COLOR_BGR2RGB))

        if snapshot_path is not None and not snap_saved:
            # Capture snapshot right when GapLink-Lite re-acquires target or fires a route/dwell alert late in clip
            if (res_r["lock"].get("state") == "REACQUIRED" and f > 60) or (f == int(len(frames_bgr) * 0.82)):
                cv2.imwrite(str(snapshot_path), comb)
                snap_saved = True

    writer.close()
    return {
        "video": out_mp4.name,
        "frames": len(frames_bgr),
        "c3_route_alerts": len(vms_l.route_mon.alerts),
        "gl_route_alerts": len(vms_r.route_mon.alerts),
        "c3_dwell_alerts": len(vms_l.dwell_mon.alerts),
        "gl_dwell_alerts": len(vms_r.dwell_mon.alerts),
        "c3_reacquired": vms_l.target_lock.reacquire_count,
        "gl_reacquired": vms_r.target_lock.reacquire_count,
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

    jobs = [
        ("detrac_MVI_40855_intersection.mp4", "40855", "security_demo_01_intersection.mp4", "fig_vms_demo_intersection.png"),
        ("detrac_MVI_40864_rain_buses.mp4", "40864", "security_demo_02_rain_dwell.mp4", "fig_vms_demo_rain.png"),
        ("detrac_MVI_39311_gore_zone.mp4", "39311", "security_demo_03_gore_zone.mp4", "fig_vms_demo_facility.png"),
    ]
    manifest = []
    for src_fn, pkey, out_fn, snap_fn in jobs:
        src_p = sample_dir / src_fn
        if not src_p.exists():
            continue
        snap_p = fig_dir / snap_fn
        info = render_security_comparison_mp4(
            src_p, pkey, out_dir / out_fn, detector, gl_assoc, c3_theta=float(tcfg["c3_theta"]), max_frames=340, snapshot_path=snap_p
        )
        manifest.append(info)
        print("Rendered VMS security demo:", info, flush=True)
        if snap_p.exists():
            if latex_fig.exists():
                shutil.copy2(snap_p, latex_fig / snap_fn)
            if cap_fig.exists():
                shutil.copy2(snap_p, cap_fig / snap_fn)

    (out_dir / "security_demo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
