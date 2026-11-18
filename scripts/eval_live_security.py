"""End-to-end evaluation of GapLink-VMS downstream of LIVE YOLOv8n detections on all 40 raw UA-DETRAC test videos.

Zero ground-truth boxes and zero synthetic occlusion masks (`make_hide_plan`) enter the pipeline.
Ground-truth trajectories are used strictly post-hoc to verify:
  1. Natural YOLOv8 Occlusion Recovery Rate (Nat-IRR) & Wrong-Link Rate (Nat-WLR) across real vehicle occlusions.
  2. Zone-to-Zone Forbidden-Route Security Alerts (Recall & False-Alarm Rate when vehicles cross between zones).
  3. Restricted-Zone Anti-Loitering / Dwell-Time Alerts (Recall, Precision, F1 across passing occluders).
  4. Real-time single-frame end-to-end latency & throughput (FPS) on RTX 3050 6GB GPU.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import shutil
import sys
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import yaml

ROOT = Path(r"D:\gaplink")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import ConstantVelocityBaseline, KalmanMahalanobisBaseline
from src.events import box_iou_cxcywh
from src.io_mot import SeqData, load_seq
from src.live_detector import LiveVehicleDetector, extract_online_hsv_hists
from src.model.gaplink import GapLinkAssociator, GapLinkLite
from src.security_analytics import PolygonZone, SecurityVMSEngine
from src.tracker import OnlineTracker


CACHE_DIR = ROOT / "data" / "cache" / "yolo_test"
OUT_JSON = ROOT / "runs" / "test_final" / "live_security_results.json"
FIG_DIR = ROOT / "report" / "figures"
LATEX_FIG_DIR = ROOT / "report" / "latex" / "figures"
CAP_FIG_DIR = Path(r"D:\Capstone-Project_latex\Capstone-Project_latex\figures")


def extract_or_load_yolo_seq(seq: SeqData, detector: LiveVehicleDetector, batch_size: int = 48) -> dict[int, dict]:
    """Run live YOLOv8n + online 128-bin HSV extraction on raw sequence frames (cached on disk for fast sweeps)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp = CACHE_DIR / f"{seq.name}_imgsz{detector.imgsz}.npz"
    if cp.exists():
        try:
            data = np.load(cp)
            frames_arr = data["frames"]
            offsets = data["offsets"]
            all_boxes = data["boxes"]
            all_hists = data["hists"]
            if len(frames_arr) == seq.n_frames:
                dets = {}
                for idx, f in enumerate(frames_arr):
                    s, e = int(offsets[idx]), int(offsets[idx + 1])
                    dets[int(f)] = {"boxes": all_boxes[s:e], "hists": all_hists[s:e]}
                return dets
        except Exception:
            cp.unlink(missing_ok=True)

    dets = {}

    def _read_one(f_num: int):
        p = seq.img_dir / f"img{f_num:05d}.jpg"
        if not p.exists():
            p = seq.img_dir / f"image{f_num:06d}.jpg"
        if not p.exists():
            return f_num, None
        return f_num, cv2.imread(str(p))

    all_f = list(range(1, seq.n_frames + 1))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start_i in range(0, len(all_f), batch_size):
            chunk_f = all_f[start_i : start_i + batch_size]
            loaded = list(pool.map(_read_one, chunk_f))
            valid_f, valid_im = [], []
            for f_num, im in loaded:
                if im is None:
                    dets[f_num] = {
                        "boxes": np.zeros((0, 4), dtype=np.float32),
                        "hists": np.zeros((0, 128), dtype=np.float32),
                    }
                else:
                    valid_f.append(f_num)
                    valid_im.append(im)
            if valid_im:
                outs = detector.detect_batch(valid_im, ignore_xywh=seq.ignore)
                for f_num, o in zip(valid_f, outs):
                    dets[int(f_num)] = {"boxes": o["boxes"], "hists": o["hists"]}

    frames_sorted = sorted(dets.keys())
    offsets = [0]
    boxes_cat = []
    hists_cat = []
    for f in frames_sorted:
        b = dets[f]["boxes"]
        h = dets[f]["hists"]
        offsets.append(offsets[-1] + len(b))
        if len(b) > 0:
            boxes_cat.append(b)
            hists_cat.append(h)

    all_b = np.concatenate(boxes_cat, axis=0) if boxes_cat else np.zeros((0, 4), dtype=np.float32)
    all_h = np.concatenate(hists_cat, axis=0) if hists_cat else np.zeros((0, 128), dtype=np.float32)
    tmp_cp = CACHE_DIR / f"{seq.name}_imgsz{detector.imgsz}_tmp.npz"
    np.savez_compressed(
        tmp_cp,
        frames=np.array(frames_sorted, dtype=np.int32),
        offsets=np.array(offsets, dtype=np.int32),
        boxes=all_b.astype(np.float32),
        hists=all_h.astype(np.float32),
    )
    tmp_cp.replace(cp)
    return dets


def match_dets_to_gt_posthoc(seq: SeqData, dets: dict[int, dict], iou_match: float = 0.45) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Strictly post-hoc 1-to-1 matching between live YOLO boxes and GT boxes to score real-world accuracy.
    Also discovers all NATURAL YOLO detection gaps (`3 <= gap <= 95` frames) on real vehicles!
    """
    det_gt_map: dict[int, np.ndarray] = {}
    seen_frames_by_tid: dict[int, list[int]] = {}

    for f in range(1, seq.n_frames + 1):
        d_boxes = dets[f]["boxes"]
        K = len(d_boxes)
        gt_ids_for_det = np.full((K,), -1, dtype=np.int32)
        sl = seq.by_frame.get(f)
        if K > 0 and sl is not None and sl.stop > sl.start:
            rows = seq.rows[sl]
            gt_b = np.column_stack([
                rows[:, 2] + 0.5 * rows[:, 4],
                rows[:, 3] + 0.5 * rows[:, 5],
                rows[:, 4],
                rows[:, 5],
            ]).astype(np.float32)
            gt_tids = rows[:, 1].astype(np.int32)
            ious = box_iou_cxcywh(d_boxes, gt_b)
            r_ind, c_ind = linear_sum_assignment(-ious)
            for r, c in zip(r_ind, c_ind):
                if ious[r, c] >= iou_match:
                    tid = int(gt_tids[c])
                    gt_ids_for_det[int(r)] = tid
                    seen_frames_by_tid.setdefault(tid, []).append(f)
        det_gt_map[f] = gt_ids_for_det

    # Identify natural YOLOv8 occlusion/dropout gaps on real vehicles (min 5 visible detections before, 3 after)
    natural_gaps = []
    for tid, f_list in seen_frames_by_tid.items():
        if len(f_list) < 10:
            continue
        f_arr = np.array(sorted(f_list), dtype=np.int32)
        diffs = np.diff(f_arr)
        gap_indices = np.nonzero((diffs >= 3) & (diffs <= 95))[0]
        for idx in gap_indices:
            # Require at least 4 YOLO hits in the 10 frames preceding f_arr[idx] and 3 hits after f_arr[idx+1]
            t_lost = int(f_arr[idx])
            t_reap = int(f_arr[idx + 1])
            vis_before = int(np.sum((f_arr <= t_lost) & (f_arr >= t_lost - 12)))
            vis_after = int(np.sum((f_arr >= t_reap) & (f_arr <= t_reap + 10)))
            if vis_before >= 4 and vis_after >= 3:
                natural_gaps.append({
                    "gt_tid": int(tid),
                    "t_lost": t_lost,
                    "t_reappear": t_reap,
                    "gap": int(t_reap - t_lost),
                })
    return det_gt_map, natural_gaps


def build_adaptive_security_zones(seq: SeqData) -> tuple[list[PolygonZone], set[int], set[int], set[int], set[int]]:
    """Construct scene-adaptive Entry Zones (Z_A monitored vs Z_C adjacent), Exit Zone (Z_B), and Dwell Zone (Z_DWELL).
    Returns:
      zones, gt_route_all, gt_route_occl_candidates, gt_dwell_all, gt_legal_zc_zb
    """
    W, H = float(seq.W), float(seq.H)
    disps = []
    starts = []
    ends = []
    all_pts = []
    for tid, tr in seq.tracks.items():
        if len(tr["frames"]) < 20:
            continue
        b0, b1 = tr["boxes"][0], tr["boxes"][-1]
        dx, dy = float(b1[0] - b0[0]), float(b1[1] - b0[1])
        if np.hypot(dx, dy) >= 70.0:
            disps.append([dx, dy])
            starts.append([float(b0[0]), float(b0[1])])
            ends.append([float(b1[0]), float(b1[1])])
            all_pts.append(tr["boxes"][:, :2])

    if len(disps) >= 4:
        disp_arr = np.array(disps, dtype=np.float32)
        med_v = np.median(disp_arr, axis=0)
        norm = float(np.hypot(med_v[0], med_v[1]))
        if norm < 1e-3:
            u_flow = np.array([0.0, 1.0], dtype=np.float32)
        else:
            u_flow = med_v / norm
    else:
        u_flow = np.array([0.0, 1.0], dtype=np.float32)
    u_perp = np.array([-u_flow[1], u_flow[0]], dtype=np.float32)

    pts_cat = np.concatenate(all_pts, axis=0) if all_pts else np.array([[0.5 * W, 0.5 * H]], dtype=np.float32)
    center = np.median(pts_cat, axis=0)
    rel = pts_cat - center[None, :]
    s_proj = rel @ u_flow
    p_proj = rel @ u_perp

    s_10, s_38, s_62, s_94 = np.percentile(s_proj, [8, 36, 64, 95])
    p_02, p_med, p_98 = np.percentile(p_proj, [2, 50, 98])
    pad_p = 45.0

    def rect_poly(s_lo, s_hi, p_lo, p_hi) -> list[tuple[float, float]]:
        corners = [
            center + s_lo * u_flow + p_lo * u_perp,
            center + s_hi * u_flow + p_lo * u_perp,
            center + s_hi * u_flow + p_hi * u_perp,
            center + s_lo * u_flow + p_hi * u_perp,
        ]
        return [(float(np.clip(c[0], 0, W)), float(np.clip(c[1], 0, H))) for c in corners]

    z_a = PolygonZone("Z_A", "Restricted Origin A", rect_poly(s_10 - 25, s_38, p_02 - pad_p, p_med), role="entry")
    z_c = PolygonZone("Z_C", "Permitted Origin C", rect_poly(s_10 - 25, s_38, p_med, p_98 + pad_p), role="entry")
    z_b = PolygonZone("Z_B", "Checkpoint Zone B", rect_poly(s_62, s_94 + 35, p_02 - pad_p, p_98 + pad_p), role="restricted")
    z_dwell = PolygonZone(
        "Z_DWELL",
        "No-Stopping Security Zone",
        [(0.10 * W, 0.20 * H), (0.90 * W, 0.20 * H), (0.90 * W, 0.88 * H), (0.10 * W, 0.88 * H)],
        role="dwell",
    )
    zones = [z_a, z_c, z_b, z_dwell]

    # Compute true GT reference events
    gt_za_zb = set()
    gt_zc_zb = set()
    gt_dwell = set()
    for tid, tr in seq.tracks.items():
        frames = tr["frames"]
        boxes = tr["boxes"]
        first_entry = None
        reached_zb = False
        dwell_first = None
        for f, b in zip(frames, boxes):
            if first_entry is None:
                if z_a.contains_box(b, seq.W, seq.H):
                    first_entry = ("Z_A", int(f))
                elif z_c.contains_box(b, seq.W, seq.H):
                    first_entry = ("Z_C", int(f))
            elif int(f) > first_entry[1] and z_b.contains_box(b, seq.W, seq.H):
                reached_zb = True

            if z_dwell.contains_box(b, seq.W, seq.H):
                if dwell_first is None:
                    dwell_first = int(f)
                elif (int(f) - dwell_first) / 25.0 >= 5.0:
                    gt_dwell.add(int(tid))
            else:
                dwell_first = None

        if first_entry is not None and reached_zb:
            if first_entry[0] == "Z_A":
                gt_za_zb.add(int(tid))
            else:
                gt_zc_zb.add(int(tid))

    return zones, gt_za_zb, gt_zc_zb, gt_dwell


def compute_yolo_observable_security_gt(
    seq: SeqData,
    zones: list[PolygonZone],
    det_gt_map: dict[int, np.ndarray],
    dets: dict[int, dict],
) -> tuple[set[int], set[int], set[int], set[int]]:
    """Determine ground-truth security events among vehicles actually observable by YOLOv8:
      - gt_za_zb_all: GT vehicles truly traveling Z_A -> Z_B with >=3 YOLO detections in Z_A and >=3 in Z_B.
      - gt_za_zb_occl: subset of gt_za_zb_all that suffered >=1 natural YOLO gap (>=3 frames) between Z_A and Z_B.
      - gt_dwell_all: GT vehicles truly dwelling >=5.0s in Z_DWELL with >=6 YOLO detections in Z_DWELL.
      - gt_dwell_occl: subset of gt_dwell_all whose longest uninterrupted YOLO visibility streak in Z_DWELL is < 5.0s
        (i.e. passing traffic occluded the vehicle mid-dwell, resetting any naive tracker's timer!).
    """
    z_map = {z.zone_id: z for z in zones}
    z_a, z_c, z_b, z_dw = z_map["Z_A"], z_map["Z_C"], z_map["Z_B"], z_map["Z_DWELL"]

    # Collect per-tid frames where YOLOv8 detected the vehicle inside each zone
    yolo_frames_in: dict[str, dict[int, list[int]]] = {"Z_A": {}, "Z_C": {}, "Z_B": {}, "Z_DWELL": {}}
    yolo_all_frames: dict[int, list[int]] = {}
    for f in range(1, seq.n_frames + 1):
        boxes = dets[f]["boxes"]
        gids = det_gt_map[f]
        for k in range(len(boxes)):
            gid = int(gids[k])
            if gid == -1:
                continue
            yolo_all_frames.setdefault(gid, []).append(f)
            b = boxes[k]
            for zid, zobj in z_map.items():
                if zobj.contains_box(b, seq.W, seq.H):
                    yolo_frames_in[zid].setdefault(gid, []).append(f)

    gt_za_zb_all = set()
    gt_za_zb_occl = set()
    gt_dwell_all = set()
    gt_dwell_occl = set()

    for tid, tr in seq.tracks.items():
        frames = tr["frames"]
        boxes = tr["boxes"]
        first_entry = None
        reached_zb_frame = None
        dwell_first = None
        true_dweller = False
        for f, b in zip(frames, boxes):
            if first_entry is None:
                if z_a.contains_box(b, seq.W, seq.H):
                    first_entry = ("Z_A", int(f))
                elif z_c.contains_box(b, seq.W, seq.H):
                    first_entry = ("Z_C", int(f))
            elif int(f) > first_entry[1] and z_b.contains_box(b, seq.W, seq.H):
                if reached_zb_frame is None:
                    reached_zb_frame = int(f)

            if z_dw.contains_box(b, seq.W, seq.H):
                if dwell_first is None:
                    dwell_first = int(f)
                elif (int(f) - dwell_first) / 25.0 >= 5.0:
                    true_dweller = True
            else:
                dwell_first = None

        if first_entry is not None and first_entry[0] == "Z_A" and reached_zb_frame is not None:
            fa = yolo_frames_in["Z_A"].get(int(tid), [])
            fb = yolo_frames_in["Z_B"].get(int(tid), [])
            if len(fa) >= 3 and len(fb) >= 3 and min(fa) < max(fb):
                gt_za_zb_all.add(int(tid))
                all_f = np.array(sorted(yolo_all_frames.get(int(tid), [])), dtype=np.int32)
                mid_f = all_f[(all_f >= min(fa)) & (all_f <= max(fb))]
                if len(mid_f) >= 2 and np.any(np.diff(mid_f) >= 3):
                    gt_za_zb_occl.add(int(tid))

        if true_dweller:
            fdw = sorted(yolo_frames_in["Z_DWELL"].get(int(tid), []))
            if len(fdw) >= 8 and (fdw[-1] - fdw[0]) / 25.0 >= 5.0:
                gt_dwell_all.add(int(tid))
                # Compute max uninterrupted streak (gaps <= 2 frames) inside Z_DWELL
                max_streak_span = 0
                seg_start = fdw[0]
                prev_f = fdw[0]
                for cf in fdw[1:]:
                    if cf - prev_f >= 3:
                        max_streak_span = max(max_streak_span, prev_f - seg_start)
                        seg_start = cf
                    prev_f = cf
                max_streak_span = max(max_streak_span, prev_f - seg_start)
                if (max_streak_span / 25.0) < 5.0:
                    gt_dwell_occl.add(int(tid))

    return gt_za_zb_all, gt_za_zb_occl, gt_dwell_all, gt_dwell_occl


def benchmark_single_frame_latency(seq: SeqData, detector: LiveVehicleDetector, gl_assoc: GapLinkAssociator) -> dict:
    """Measure genuine frame-by-frame online latency (batch=1) across 250 consecutive frames."""
    trk = OnlineTracker(associator=gl_assoc, W=seq.W, H=seq.H, seq_name=seq.name, min_hits=3)
    zones, _, _, _ = build_adaptive_security_zones(seq)
    vms = SecurityVMSEngine(trk, zones=zones, forbidden_routes=[("Z_A", "Z_B")], max_dwell_sec=5.0, fps=25.0)

    imgs = []
    for f in range(1, min(260, seq.n_frames) + 1):
        p = seq.img_dir / f"img{f:05d}.jpg"
        if p.exists():
            im = cv2.imread(str(p))
            if im is not None:
                imgs.append(im)
    # Warmup
    for im in imgs[:10]:
        detector.detect_frame(im, ignore_xywh=seq.ignore)

    t_yolo, t_hsv, t_vms = [], [], []
    for idx, im in enumerate(imgs[10:], start=1):
        t0 = time.perf_counter()
        res = detector.model.predict(
            source=im,
            conf=detector.conf_thr,
            iou=detector.nms_iou,
            imgsz=detector.imgsz,
            classes=detector.classes,
            agnostic_nms=True,
            device=detector.device,
            verbose=False,
        )[0]
        t1 = time.perf_counter()
        det_out = detector._postprocess_one(res, im, ignore_xywh=seq.ignore)
        t2 = time.perf_counter()
        vms.step(idx, det_out["boxes"], det_out["hists"])
        t3 = time.perf_counter()
        t_yolo.append((t1 - t0) * 1000.0)
        t_hsv.append((t2 - t1) * 1000.0)
        t_vms.append((t3 - t2) * 1000.0)

    tot_ms = float(np.mean(t_yolo) + np.mean(t_hsv) + np.mean(t_vms))
    return {
        "yolo_ms": round(float(np.mean(t_yolo)), 2),
        "hsv_post_ms": round(float(np.mean(t_hsv)), 2),
        "gaplink_vms_ms": round(float(np.mean(t_vms)), 2),
        "total_ms": round(tot_ms, 2),
        "fps": round(1000.0 / max(1e-3, tot_ms), 1),
    }


def make_live_security_figure(summary: dict):
    methods = ["C-1", "C-2", "C-3", "C-4", "C-3 (WLR<=5%)", "GapLink-Lite"]
    labels = ["C-1\n(IoU Only)", "C-2\n(Const-Vel)", "C-3\n(Kalman)", "C-4\n(Kal.+Color)", "C-3\n(Strict Gate)", "GapLink-Lite\n(Ours)"]
    colors = ["#8c969e", "#6c8ebf", "#d97724", "#b85450", "#9673a6", "#1a7f5a"]

    nat_irr = [summary["methods"][m]["natural_occlusion"]["nat_irr"] * 100.0 for m in methods]
    nat_wlr = [summary["methods"][m]["natural_occlusion"]["nat_wlr"] * 100.0 for m in methods]
    route_rec = [summary["methods"][m]["forbidden_route"]["recall_occl"] * 100.0 for m in methods]
    route_far = [summary["methods"][m]["forbidden_route"]["false_alarm_rate"] * 100.0 for m in methods]
    dwell_f1 = [summary["methods"][m]["dwell_alert"]["f1_occl"] * 100.0 for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), dpi=220)
    x = np.arange(len(methods))
    w = 0.38

    # Panel A: Natural Live-YOLOv8 Occlusion Recovery vs Wrong-Link Rate
    ax = axes[0]
    b1 = ax.bar(x - w / 2, nat_irr, width=w, color="#1a7f5a", alpha=0.88, label="Nat-IRR (%) ↑")
    b2 = ax.bar(x + w / 2, nat_wlr, width=w, color="#c93b2b", alpha=0.88, label="Nat-WLR (%) ↓")
    ax.set_title("(a) Natural YOLOv8 Occlusion Recovery", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylim(0, 114)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="upper left")
    for rect in b1:
        h = rect.get_height()
        if h > 1.0:
            ax.text(rect.get_x() + rect.get_width() / 2, h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=7)
    for rect in b2:
        h = rect.get_height()
        if h > 0.3:
            ax.text(rect.get_x() + rect.get_width() / 2, h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=7, color="#901a11")

    # Panel B: Forbidden Route (Z_A -> Z_B) Recall (Occluded) vs False-Alarm Rate
    ax = axes[1]
    b3 = ax.bar(x - w / 2, route_rec, width=w, color="#2563eb", alpha=0.88, label="Occl. Route Recall (%) ↑")
    b4 = ax.bar(x + w / 2, route_far, width=w, color="#dc2626", alpha=0.88, label="False-Alarm Rate (%) ↓")
    ax.set_title("(b) Forbidden Route Alert (Z_A → Z_B)", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylim(0, 114)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="upper left")
    for rect in b3:
        h = rect.get_height()
        if h > 1.0:
            ax.text(rect.get_x() + rect.get_width() / 2, h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=7)
    for rect in b4:
        h = rect.get_height()
        if h > 0.2:
            ax.text(rect.get_x() + rect.get_width() / 2, h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=7, color="#901a11")

    # Panel C: Restricted-Zone Dwell-Time Alert F1 (Occluded Dwellers)
    ax = axes[2]
    b5 = ax.bar(x, dwell_f1, width=0.52, color=colors, alpha=0.92)
    ax.set_title("(c) Anti-Loitering Dwell Alert F1 (Occl.)", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylim(0, 114)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for rect in b5:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_fig = FIG_DIR / "fig_live_security_eval.png"
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)
    if LATEX_FIG_DIR.exists():
        shutil.copy2(out_fig, LATEX_FIG_DIR / out_fig.name)
    if CAP_FIG_DIR.exists():
        shutil.copy2(out_fig, CAP_FIG_DIR / out_fig.name)
    print(f"Saved figure -> {out_fig}", flush=True)


def main():
    splits = list(csv.DictReader(open(ROOT / "data" / "splits.csv", encoding="utf-8")))
    test_seqs = [r["seq"] for r in splits if r["split"] == "test"]
    print(f"Running Live YOLOv8n + GapLink-VMS evaluation across {len(test_seqs)} raw test videos...", flush=True)

    rel_dir = ROOT / "release"
    tcfg = yaml.safe_load((rel_dir / "tracker.yaml").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(rel_dir / "final.pt", map_location=device, weights_only=False)
    model = GapLinkLite(use_ctx=True, use_color=True, gap_aware=True).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    detector = LiveVehicleDetector(imgsz=640, conf_thr=0.28, nms_iou=0.52)

    def build_associators():
        return {
            "C-1": None,
            "C-2": ConstantVelocityBaseline(theta=float(tcfg["c2_theta"])),
            "C-3": KalmanMahalanobisBaseline(theta=float(tcfg["c3_theta"]), name="C-3"),
            "C-4": KalmanMahalanobisBaseline(
                theta=float(tcfg["c4_theta"]),
                lambda_s=float(tcfg["c4_lambda_s"]),
                lambda_c=float(tcfg["c4_lambda_c"]),
                name="C-4",
            ),
            "C-3 (WLR<=5%)": KalmanMahalanobisBaseline(theta=float(tcfg["c3_theta_wlr5"]), name="C-3 (WLR<=5%)"),
            "GapLink-Lite": GapLinkAssociator(model, device=device, tau=float(tcfg["gaplink_tau"]), calib=None),
        }

    method_names = ["C-1", "C-2", "C-3", "C-4", "C-3 (WLR<=5%)", "GapLink-Lite"]
    agg = {
        m: {
            "nat_gaps_total": 0,
            "nat_gaps_recovered": 0,
            "reassoc_total_valid_gt": 0,
            "reassoc_wrong": 0,
            "route_gt_all": 0,
            "route_gt_occl": 0,
            "route_tp_all": 0,
            "route_tp_occl": 0,
            "route_fp": 0,
            "route_unverified": 0,
            "dwell_gt_all": 0,
            "dwell_gt_occl": 0,
            "dwell_tp_all": 0,
            "dwell_tp_occl": 0,
            "dwell_fp": 0,
        }
        for m in method_names
    }

    per_seq_summary = {}
    latency_info = None

    for idx_s, s_name in enumerate(test_seqs, start=1):
        seq = load_seq(ROOT / "data" / "mot", s_name)
        if idx_s == 1:
            gl_tmp = GapLinkAssociator(model, device=device, tau=float(tcfg["gaplink_tau"]), calib=None)
            latency_info = benchmark_single_frame_latency(seq, detector, gl_tmp)
            print(f"Single-frame Live Latency: {latency_info}", flush=True)

        dets = extract_or_load_yolo_seq(seq, detector, batch_size=32)
        det_gt_map, nat_gaps = match_dets_to_gt_posthoc(seq, dets, iou_match=0.45)
        zones, _, _, _ = build_adaptive_security_zones(seq)
        gt_za_zb, gt_za_zb_occl, gt_dwell, gt_dwell_occl = compute_yolo_observable_security_gt(
            seq, zones, det_gt_map, dets
        )

        assocs = build_associators()
        per_seq_summary[s_name] = {
            "n_nat_gaps": len(nat_gaps),
            "gt_route": len(gt_za_zb),
            "gt_route_occl": len(gt_za_zb_occl),
            "gt_dwell": len(gt_dwell),
            "gt_dwell_occl": len(gt_dwell_occl),
        }

        for m_name, assoc in assocs.items():
            trk = OnlineTracker(
                associator=assoc,
                iou_thr=float(tcfg["iou_thr"]),
                a_max=int(tcfg["a_max"]),
                W=seq.W,
                H=seq.H,
                seq_name=s_name,
                min_hits=3,
            )
            vms = SecurityVMSEngine(
                trk,
                zones=zones,
                forbidden_routes=[("Z_A", "Z_B")],
                max_dwell_sec=5.0,
                fps=25.0,
            )

            # Note: det_gt_tids is ONLY stored as passive metadata for post-hoc evaluation;
            # OnlineTracker and Associators NEVER read gt_tids when making matching decisions!
            pred_tid_at: dict[tuple[int, int], int] = {}
            for f in range(1, seq.n_frames + 1):
                res = vms.step(f, dets[f]["boxes"], dets[f]["hists"], det_gt_map[f])
                for o in res["tracks"]:
                    if o["gt_tid"] != -1:
                        pred_tid_at[(f, int(o["gt_tid"]))] = int(o["track_id"])

            st = agg[m_name]
            # 1. Natural YOLO gap recovery (fair denominator: check confirmed track before t_lost and within [t_reappear..t_reappear+3])
            for g in nat_gaps:
                gt_id = g["gt_tid"]
                id_pre = None
                for fp in (g["t_lost"], g["t_lost"] - 1, g["t_lost"] - 2):
                    if (fp, gt_id) in pred_tid_at:
                        id_pre = pred_tid_at[(fp, gt_id)]
                        break
                id_post = None
                for fq in range(g["t_reappear"], g["t_reappear"] + 4):
                    if (fq, gt_id) in pred_tid_at:
                        id_post = pred_tid_at[(fq, gt_id)]
                        break
                if id_pre is not None and id_post is not None:
                    st["nat_gaps_total"] += 1
                    if id_pre == id_post:
                        st["nat_gaps_recovered"] += 1

            for rlog in trk.reassoc_log:
                gb, ga = rlog["gt_tid_before"], rlog["gt_tid_after"]
                if gb != -1 and ga != -1:
                    st["reassoc_total_valid_gt"] += 1
                    if gb != ga:
                        st["reassoc_wrong"] += 1

            # 2. Forbidden Route (Z_A -> Z_B) evaluation
            st["route_gt_all"] += len(gt_za_zb)
            st["route_gt_occl"] += len(gt_za_zb_occl)
            st["route_unverified"] += len(vms.route_mon.unverified_entries.get("Z_B", set()))

            caught_gt_route = set()
            for al in vms.route_mon.alerts:
                g_orig = al["origin_gt_tid"]
                g_dest = al["dest_gt_tid"]
                if g_orig != -1 and g_orig == g_dest and (g_dest in gt_za_zb):
                    caught_gt_route.add(g_dest)
                elif g_orig != -1 and g_dest != -1 and (g_orig != g_dest or g_dest not in gt_za_zb):
                    st["route_fp"] += 1
            st["route_tp_all"] += len(caught_gt_route)
            st["route_tp_occl"] += len(caught_gt_route & gt_za_zb_occl)

            # 3. Restricted-Zone Dwell-Time Alert evaluation
            st["dwell_gt_all"] += len(gt_dwell)
            st["dwell_gt_occl"] += len(gt_dwell_occl)
            caught_gt_dwell = set()
            for al in vms.dwell_mon.alerts:
                g_orig = al["origin_gt_tid"]
                g_now = al["now_gt_tid"]
                if g_orig != -1 and g_orig == g_now and (g_now in gt_dwell):
                    caught_gt_dwell.add(g_now)
                elif g_orig != -1 and g_now != -1 and (g_orig != g_now):
                    st["dwell_fp"] += 1
            st["dwell_tp_all"] += len(caught_gt_dwell)
            st["dwell_tp_occl"] += len(caught_gt_dwell & gt_dwell_occl)

            if m_name == "GapLink-Lite":
                per_seq_summary[s_name]["gaplink_route_tp"] = len(caught_gt_route)
                per_seq_summary[s_name]["gaplink_dwell_tp"] = len(caught_gt_dwell)

        if idx_s % 5 == 0 or idx_s == len(test_seqs):
            gl_s = agg["GapLink-Lite"]
            c3_s = agg["C-3"]
            print(
                f"  [{idx_s:02d}/{len(test_seqs)}] Nat-IRR: GapLink={gl_s['nat_gaps_recovered']}/{max(1, gl_s['nat_gaps_total'])} "
                f"({100*gl_s['nat_gaps_recovered']/max(1, gl_s['nat_gaps_total']):.1f}%) vs "
                f"C-3={100*c3_s['nat_gaps_recovered']/max(1, c3_s['nat_gaps_total']):.1f}%",
                flush=True,
            )

    methods_out = {}
    for m_name in method_names:
        s = agg[m_name]
        nat_irr = s["nat_gaps_recovered"] / max(1, s["nat_gaps_total"])
        nat_wlr = s["reassoc_wrong"] / max(1, s["reassoc_total_valid_gt"])

        r_rec_all = s["route_tp_all"] / max(1, s["route_gt_all"])
        r_rec_occl = s["route_tp_occl"] / max(1, s["route_gt_occl"])
        r_tot_fired = s["route_tp_all"] + s["route_fp"]
        r_far = s["route_fp"] / max(1, r_tot_fired)

        d_rec_all = s["dwell_tp_all"] / max(1, s["dwell_gt_all"])
        d_rec_occl = s["dwell_tp_occl"] / max(1, s["dwell_gt_occl"])
        d_prec = s["dwell_tp_all"] / max(1, s["dwell_tp_all"] + s["dwell_fp"])
        d_f1_occl = (2 * d_prec * d_rec_occl / max(1e-6, d_prec + d_rec_occl)) if (d_prec + d_rec_occl) > 0 else 0.0

        methods_out[m_name] = {
            "raw_counts": s,
            "natural_occlusion": {
                "n_gaps": s["nat_gaps_total"],
                "nat_irr": round(nat_irr, 4),
                "nat_wlr": round(nat_wlr, 4),
            },
            "forbidden_route": {
                "gt_all": s["route_gt_all"],
                "gt_occl": s["route_gt_occl"],
                "tp_all": s["route_tp_all"],
                "tp_occl": s["route_tp_occl"],
                "fp_wrong_id": s["route_fp"],
                "unverified_entries": s["route_unverified"],
                "recall_all": round(r_rec_all, 4),
                "recall_occl": round(r_rec_occl, 4),
                "false_alarm_rate": round(r_far, 4),
            },
            "dwell_alert": {
                "gt_all": s["dwell_gt_all"],
                "gt_occl": s["dwell_gt_occl"],
                "tp_all": s["dwell_tp_all"],
                "tp_occl": s["dwell_tp_occl"],
                "fp_wrong_id": s["dwell_fp"],
                "recall_all": round(d_rec_all, 4),
                "recall_occl": round(d_rec_occl, 4),
                "precision": round(d_prec, 4),
                "f1_occl": round(d_f1_occl, 4),
            },
        }

    summary = {
        "n_test_sequences": len(test_seqs),
        "detector": "YOLOv8n (imgsz=640, conf=0.28, agnostic_nms=0.52, min_hits=3)",
        "latency_rtx3050": latency_info,
        "methods": methods_out,
        "per_seq": per_seq_summary,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved Live Security Evaluation JSON -> {OUT_JSON}", flush=True)
    make_live_security_figure(summary)


if __name__ == "__main__":
    main()
