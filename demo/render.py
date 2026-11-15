"""Render side-by-side MP4 demo clips comparing C-3 (Kalman+Mahalanobis) vs GapLink-Lite."""
import argparse
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np

from src.io_mot import load_seq
from src.inject import make_hide_plan, make_detections
from src.baselines import KalmanMahalanobisBaseline, kalman_predict_1track
from src.tracker import OnlineTracker
from demo.overlays import draw_occlusion_zone, draw_track_box, draw_ghost_and_ellipse, draw_hud_banner


def find_frame_image(seq_name: str, frame_id: int, raw_root: Path = Path(r"D:\gaplink\data\raw\images")) -> Path | None:
    fname = f"img{frame_id:05d}.jpg"
    for sub in ("train", "test"):
        base = raw_root / sub
        if not base.exists():
            continue
        for d in base.glob(f"*_{seq_name}"):
            p = d / fname
            if p.exists():
                return p
    return None


def render_side_by_side_clip(
    seq_name: str,
    out_mp4: Path,
    start_frame: int = 1,
    end_frame: int = 180,
    associator_right=None,
    theta_c3: float = 25.0,
    seed: int = 2026,
    fps: int = 20,
):
    mot_dir = Path(r"D:\gaplink\data\mot")
    hist_dir = Path(r"D:\gaplink\data\cache\hist")
    seq = load_seq(mot_dir, seq_name)
    hp = hist_dir / f"{seq_name}.npy"
    hc = np.load(hp) if hp.exists() else None

    plan = make_hide_plan(seq, seed=seed, max_density=0.18)
    dets = make_detections(seq, plan, hc, mode="full", seed=seed)

    c3_assoc = KalmanMahalanobisBaseline(theta=theta_c3, name="C-3")
    if associator_right is None:
        associator_right = KalmanMahalanobisBaseline(theta=theta_c3, lambda_s=0.5, lambda_c=2.0, name="C-4 (v0 check)")

    trk_left = OnlineTracker(associator=c3_assoc, W=seq.W, H=seq.H, seq_name=seq_name)
    trk_right = OnlineTracker(associator=associator_right, W=seq.W, H=seq.H, seq_name=seq_name)

    hidden_at_frame: dict[int, list[dict]] = {}
    for item in plan:
        for f in range(item["t_start"], item["t_end"] + 1):
            hidden_at_frame.setdefault(f, []).append(item)

    seq_img_dir = seq.img_dir if (seq.img_dir and seq.img_dir.exists()) else None

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_mp4), fps=fps, codec="libx264", quality=7, macro_block_size=None)

    stats_l = {"rec": 0, "tot": 0, "wrong": 0, "frag": 0}
    stats_r = {"rec": 0, "tot": 0, "wrong": 0, "frag": 0}
    pre_id_l = {}
    pre_id_r = {}
    highlight_l = {}
    highlight_r = {}

    end_f = min(seq.n_frames, end_frame)
    for f in range(1, end_f + 1):
        d = dets[f]
        out_l = trk_left.step(f, d["boxes"], d["hists"], d["gt_tids"])
        out_r = trk_right.step(f, d["boxes"], d["hists"], d["gt_tids"])

        map_l = {o["gt_tid"]: o["track_id"] for o in out_l if o["gt_tid"] != -1}
        map_r = {o["gt_tid"]: o["track_id"] for o in out_r if o["gt_tid"] != -1}

        for item in plan:
            if item["t_lost"] == f:
                if item["tid"] in map_l:
                    pre_id_l[item["tid"]] = map_l[item["tid"]]
                if item["tid"] in map_r:
                    pre_id_r[item["tid"]] = map_r[item["tid"]]
            if item["t_reappear"] == f:
                tid = item["tid"]
                if tid in pre_id_l and tid in map_l:
                    stats_l["tot"] += 1
                    if map_l[tid] == pre_id_l[tid]:
                        stats_l["rec"] += 1
                        highlight_l[map_l[tid]] = (f + 18, True, False)
                    else:
                        r_match = [r for r in trk_left.reassoc_log if r["frame"] == f and r["gt_tid_after"] == tid]
                        if r_match and not r_match[0]["correct"]:
                            stats_l["wrong"] += 1
                        else:
                            stats_l["frag"] += 1
                        highlight_l[map_l[tid]] = (f + 18, False, True)
                if tid in pre_id_r and tid in map_r:
                    stats_r["tot"] += 1
                    if map_r[tid] == pre_id_r[tid]:
                        stats_r["rec"] += 1
                        highlight_r[map_r[tid]] = (f + 18, True, False)
                    else:
                        r_match = [r for r in trk_right.reassoc_log if r["frame"] == f and r["gt_tid_after"] == tid]
                        if r_match and not r_match[0]["correct"]:
                            stats_r["wrong"] += 1
                        else:
                            stats_r["frag"] += 1
                        highlight_r[map_r[tid]] = (f + 18, False, True)

        if f < start_frame:
            continue

        img_p = (seq_img_dir / f"image{f:06d}.jpg") if (seq_img_dir and (seq_img_dir / f"image{f:06d}.jpg").exists()) else ((seq_img_dir / f"img{f:05d}.jpg") if (seq_img_dir and (seq_img_dir / f"img{f:05d}.jpg").exists()) else None)
        if img_p is not None:
            base_bgr = cv2.imread(str(img_p))
        else:
            base_bgr = np.full((seq.H, seq.W, 3), 30, dtype=np.uint8)

        pane_w, pane_h = 640, 360
        sx_scale = pane_w / float(seq.W)
        sy_scale = pane_h / float(seq.H)

        def scale_box(b):
            return np.array([b[0] * sx_scale, b[1] * sy_scale, b[2] * sx_scale, b[3] * sy_scale], dtype=np.float32)

        pane_l = cv2.resize(base_bgr, (pane_w, pane_h))
        pane_r = pane_l.copy()

        # Draw occlusion zones using ground-truth track positions at frame f
        for item in hidden_at_frame.get(f, []):
            tr_gt = seq.tracks.get(item["tid"], None)
            if tr_gt is not None:
                pos = int(np.searchsorted(tr_gt["frames"], f))
                if pos < len(tr_gt["frames"]) and tr_gt["frames"][pos] == f:
                    gt_b = scale_box(tr_gt["boxes"][pos])
                    draw_occlusion_zone(pane_l, gt_b)
                    draw_occlusion_zone(pane_r, gt_b)

        planned_tids = {item["tid"] for item in hidden_at_frame.get(f, [])}
        for lt in trk_left.lost_tracks:
            if lt.initial_gt_tid in planned_tids and 1 <= lt.age <= 50:
                h_pad, h_len = lt.get_padded_hist()
                mu, var = kalman_predict_1track(h_pad, h_len, lt.age)
                sig = np.sqrt(np.maximum(1.0, var[:2])) * np.array([sx_scale, sy_scale])
                draw_ghost_and_ellipse(pane_l, scale_box(mu), sig, lt.track_id, lt.age)

        if hasattr(associator_right, "infer_event"):
            lost_vis = [lt for lt in trk_right.lost_tracks if lt.initial_gt_tid in planned_tids and 1 <= lt.age <= 50][:6]
            if lost_vis:
                ev_tmp = trk_right.build_event_from_subpools(
                    f, lost_vis, np.zeros((1, 4), dtype=np.float32), np.zeros((1, 128), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
                )
                inf = associator_right.infer_event(ev_tmp)
                for idx_l, lt in enumerate(lost_vis):
                    pb = scale_box(inf["pred_box"][idx_l])
                    sc = inf["sigma_cal"][idx_l, :2] * np.array([sx_scale, sy_scale])
                    draw_ghost_and_ellipse(pane_r, pb, sc, lt.track_id, lt.age)
        else:
            for lt in trk_right.lost_tracks:
                if lt.initial_gt_tid in planned_tids and 1 <= lt.age <= 50:
                    h_pad, h_len = lt.get_padded_hist()
                    mu, var = kalman_predict_1track(h_pad, h_len, lt.age)
                    sig = np.sqrt(np.maximum(1.0, var[:2])) * np.array([sx_scale, sy_scale])
                    draw_ghost_and_ellipse(pane_r, scale_box(mu), sig, lt.track_id, lt.age)

        for o in out_l:
            tid_k = o["track_id"]
            hl = highlight_l.get(tid_k, (0, False, False))
            draw_track_box(pane_l, scale_box(o["box"]), tid_k, is_recovered=(f <= hl[0] and hl[1]), is_wrong=(f <= hl[0] and hl[2]))

        for o in out_r:
            tid_k = o["track_id"]
            hl = highlight_r.get(tid_k, (0, False, False))
            draw_track_box(pane_r, scale_box(o["box"]), tid_k, is_recovered=(f <= hl[0] and hl[1]), is_wrong=(f <= hl[0] and hl[2]))

        draw_hud_banner(pane_l, "C-3: Kalman + Mahalanobis", f, stats_l)
        draw_hud_banner(pane_r, getattr(associator_right, "name", "GapLink-Lite"), f, stats_r)

        combined_bgr = np.concatenate([pane_l, pane_r], axis=1)
        combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(combined_rgb)

    writer.close()
    return {"out_mp4": str(out_mp4), "left": stats_l, "right": stats_r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=str, default="MVI_20011")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=150)
    ap.add_argument("--out", type=str, default=r"D:\gaplink\demo\out\v0_check.mp4")
    args = ap.parse_args()
    info = render_side_by_side_clip(args.seq, Path(args.out), start_frame=args.start, end_frame=args.end)
    print("Rendered demo clip:", info)


if __name__ == "__main__":
    main()
