"""Synthetic occlusion injection protocol (`full` and `partial` variants) for online tracker evaluation.
Uses exact gap bins [3,15], [16,45], [46,90] matching EventCfg and extracts partial-crop histograms from visible sub-boxes only.
"""
import numpy as np
from src.io_mot import SeqData
from src.events import partial_crop


def make_hide_plan(seq: SeqData, seed: int, max_density: float = 0.18, min_vis_before: int = 8, min_vis_after: int = 5) -> list[dict]:
    """Deterministically plan non-overlapping occlusion intervals (tid, t_start, t_end, gap, gap_bin)
    such that at most `max_density` of active boxes in any frame are occluded and each track has >=min_vis_before visible frames prior.
    Gap durations span [3, 15] (short, 40%), [16, 45] (medium, 40%), and [46, 90] (long, 20%).
    """
    rng = np.random.default_rng(seed)
    T = seq.n_frames
    active_per_frame = np.zeros((T + 2,), dtype=np.int32)
    hidden_per_frame = np.zeros((T + 2,), dtype=np.int32)
    for f, sl in seq.by_frame.items():
        if 1 <= f <= T:
            active_per_frame[f] = sl.stop - sl.start

    tids = sorted(seq.tracks.keys())
    rng.shuffle(tids)

    intervals = []
    for tid in tids:
        tr = seq.tracks[tid]
        frames = tr["frames"]
        L = len(frames)
        if L < min_vis_before + min_vis_after + 6:
            continue
        diffs = np.diff(frames)
        if np.any(diffs != 1):
            continue

        n_try = 2 if L >= 110 else 1
        used_mask = np.zeros((L,), dtype=bool)

        for _ in range(n_try):
            u = rng.random()
            if u < 0.40:
                gb, g_lo, g_hi = 0, 3, 15
            elif u < 0.80:
                gb, g_lo, g_hi = 1, 16, 45
            else:
                gb, g_lo, g_hi = 2, 46, 90
            # gap = (t_end + 1) - t_lost = num_hidden_frames + 1
            n_hide = int(rng.integers(max(2, g_lo - 1), g_hi))
            max_start_idx = L - min_vis_after - n_hide
            if max_start_idx <= min_vis_before:
                continue
            start_idx = int(rng.integers(min_vis_before, max_start_idx + 1))
            end_idx = start_idx + n_hide - 1

            lo_chk = max(0, start_idx - min_vis_before)
            hi_chk = min(L, end_idx + min_vis_after + 1)
            if np.any(used_mask[lo_chk:hi_chk]):
                continue

            t_start = int(frames[start_idx])
            t_end = int(frames[end_idx])

            can_hide = True
            for f in range(t_start, t_end + 1):
                act = max(1, int(active_per_frame[f]))
                if (hidden_per_frame[f] + 1) / act > max_density and hidden_per_frame[f] >= 1:
                    can_hide = False
                    break
            if not can_hide:
                continue

            used_mask[start_idx : end_idx + 1] = True
            hidden_per_frame[t_start : t_end + 1] += 1
            gap_actual = int(frames[end_idx + 1] - frames[start_idx - 1])
            gb_actual = 0 if gap_actual <= 15 else (1 if gap_actual <= 45 else 2)
            intervals.append({
                "tid": int(tid),
                "t_lost": int(frames[start_idx - 1]),
                "t_start": t_start,
                "t_end": t_end,
                "t_reappear": int(frames[end_idx + 1]),
                "gap": gap_actual,
                "gap_bin": gb_actual,
                "last_box": tr["boxes"][start_idx - 1].copy(),
                "reappear_box": tr["boxes"][end_idx + 1].copy(),
            })

    intervals.sort(key=lambda x: (x["t_start"], x["tid"]))
    return intervals


def make_detections(
    seq: SeqData,
    hide_plan: list[dict],
    hist_cache: np.ndarray | None,
    mode: str = "full",
    seed: int = 42,
) -> dict[int, dict]:
    """Produce per-frame detection dictionary for online tracker evaluation.
    - `mode == "full"`: drop detections of planned tracks during [t_start, t_end].
    - `mode == "partial"`: drop during [t_start, t_end], and on [t_reappear, t_reappear+2] apply `partial_crop`
      and read the HSV histogram strictly from the visible cropped sub-box (`hist_cache[row_idx, side]`).
    """
    rng = np.random.default_rng(seed)
    hide_lookup = {}
    reappear_side = {}
    for item in hide_plan:
        for f in range(item["t_start"], item["t_end"] + 1):
            hide_lookup[(f, item["tid"])] = item
        side = int(rng.integers(1, 5))
        frac = float(rng.uniform(0.62, 0.80))
        for f_r in range(item["t_reappear"], min(seq.n_frames + 1, item["t_reappear"] + 3)):
            reappear_side[(f_r, item["tid"])] = (side, frac)

    dets_by_frame = {}
    for f in range(1, seq.n_frames + 1):
        sl = seq.by_frame.get(f, None)
        if sl is None or sl.stop <= sl.start:
            dets_by_frame[f] = {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "hists": np.zeros((0, 128), dtype=np.float32),
                "gt_tids": np.zeros((0,), dtype=np.int32),
            }
            continue

        sub = seq.rows[sl]
        boxes_out = []
        hists_out = []
        tids_out = []

        for local_i, row_idx in enumerate(range(sl.start, sl.stop)):
            r = sub[local_i]
            tid_i = int(r[1])
            if (f, tid_i) in hide_lookup:
                continue

            x, y, w, h = float(r[2]), float(r[3]), float(r[4]), float(r[5])
            box = np.array([x + 0.5 * w, y + 0.5 * h, w, h], dtype=np.float32)

            view_idx = 0
            if mode == "partial" and (f, tid_i) in reappear_side:
                side, frac = reappear_side[(f, tid_i)]
                box = partial_crop(box, side, frac)
                view_idx = side

            if hist_cache is not None and row_idx < len(hist_cache):
                raw_h = hist_cache[row_idx, view_idx] if hist_cache.ndim == 3 else hist_cache[row_idx]
                h_vec = raw_h.astype(np.float32)
                s = float(h_vec.sum())
                h_vec = h_vec / s if s > 1e-6 else np.zeros((128,), dtype=np.float32)
            else:
                h_vec = np.zeros((128,), dtype=np.float32)

            boxes_out.append(box)
            hists_out.append(h_vec)
            tids_out.append(tid_i)

        if boxes_out:
            dets_by_frame[f] = {
                "boxes": np.stack(boxes_out, axis=0),
                "hists": np.stack(hists_out, axis=0),
                "gt_tids": np.array(tids_out, dtype=np.int32),
            }
        else:
            dets_by_frame[f] = {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "hists": np.zeros((0, 128), dtype=np.float32),
                "gt_tids": np.zeros((0,), dtype=np.int32),
            }
    return dets_by_frame
