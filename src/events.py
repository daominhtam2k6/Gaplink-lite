"""Event assembly and synthetic occlusion/drop/nomatch sampler (shared between training and tracker)."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from src.io_mot import SeqData

@dataclass
class EventCfg:
    hist_len: int = 20
    min_hist: int = 8
    n_ctx: int = 8
    n_trk_max: int = 6
    n_cand_max: int = 10
    gap_bins: tuple = ((3, 15), (16, 45), (46, 90))
    gap_probs: tuple = (0.4, 0.4, 0.2)
    mode_probs: tuple = (0.4, 0.4, 0.1, 0.1)  # occl, drop, nomatch_exit, nomatch_miss
    p_partial: float = 0.4
    p_drop_true_cand: float = 0.05
    cand_pos_sigma: float = 0.05
    cand_size_jitter: float = 0.10
    hist_pos_sigma: float = 0.03
    hard_dist_factor: float = 1.15

def box_iou_cxcywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between a (N,4) and b (M,4) in (cx,cy,w,h) format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1 = a[:, 0:1] - 0.5 * a[:, 2:3]
    ay1 = a[:, 1:2] - 0.5 * a[:, 3:4]
    ax2 = a[:, 0:1] + 0.5 * a[:, 2:3]
    ay2 = a[:, 1:2] + 0.5 * a[:, 3:4]

    bx1 = b[:, 0][None, :] - 0.5 * b[:, 2][None, :]
    by1 = b[:, 1][None, :] - 0.5 * b[:, 3][None, :]
    bx2 = b[:, 0][None, :] + 0.5 * b[:, 2][None, :]
    by2 = b[:, 1][None, :] + 0.5 * b[:, 3][None, :]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = np.maximum(1e-3, a[:, 2:3] * a[:, 3:4])
    area_b = np.maximum(1e-3, b[:, 2][None, :] * b[:, 3][None, :])
    return (inter / (area_a + area_b - inter + 1e-6)).astype(np.float32)

def partial_crop(box_cxcywh: np.ndarray, side: int, frac: float) -> np.ndarray:
    """Shared partial occlusion crop function (used by both sampler and tracker injection).
    side: 0=none, 1=keep_top, 2=keep_bottom, 3=keep_left, 4=keep_right
    frac: fraction of width/height retained in [0.6, 1.0].
    """
    cx, cy, w, h = [float(v) for v in box_cxcywh]
    frac = float(np.clip(frac, 0.6, 1.0))
    if side == 0 or frac >= 0.999:
        return np.array([cx, cy, w, h], dtype=np.float32)
    x1, y1 = cx - 0.5 * w, cy - 0.5 * h
    x2, y2 = cx + 0.5 * w, cy + 0.5 * h
    if side == 1:    # keep top
        y2 = y1 + frac * h
    elif side == 2:  # keep bottom
        y1 = y2 - frac * h
    elif side == 3:  # keep left
        x2 = x1 + frac * w
    elif side == 4:  # keep right
        x1 = x2 - frac * w
    nw = max(2.0, x2 - x1)
    nh = max(2.0, y2 - y1)
    return np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), nw, nh], dtype=np.float32)

def build_neighbor_context(last_box: np.ndarray, neigh_boxes: np.ndarray, flag_time: float, n_ctx: int = 8):
    """Build (n_ctx, 6) context tensor and (n_ctx,) bool mask for one track at t_lost (flag=0) or t_conf (flag=1).
    Features per neighbor: [dx/h, dy/h, log(w_n/w_last), log(h_n/h_last), iou_with_last, time_flag]
    """
    ctx = np.zeros((n_ctx, 6), dtype=np.float32)
    mask = np.zeros((n_ctx,), dtype=bool)
    if neigh_boxes is None or len(neigh_boxes) == 0:
        return ctx, mask
    cx, cy, w_l, h_l = [float(v) for v in last_box]
    h_safe = max(8.0, h_l)
    w_safe = max(8.0, w_l)
    dists = np.hypot(neigh_boxes[:, 0] - cx, neigh_boxes[:, 1] - cy)
    # Exclude exact self if distance < 1e-2
    valid = np.nonzero(dists > 0.5)[0]
    if len(valid) == 0:
        return ctx, mask
    nb = neigh_boxes[valid]
    d_sub = dists[valid]
    order = np.argsort(d_sub)[:n_ctx]
    nb = nb[order]
    K = len(nb)
    ious = box_iou_cxcywh(last_box[None, :], nb)[0]
    ctx[:K, 0] = np.clip((nb[:, 0] - cx) / h_safe, -10.0, 10.0)
    ctx[:K, 1] = np.clip((nb[:, 1] - cy) / h_safe, -10.0, 10.0)
    ctx[:K, 2] = np.clip(np.log(np.maximum(2.0, nb[:, 2]) / w_safe), -3.0, 3.0)
    ctx[:K, 3] = np.clip(np.log(np.maximum(2.0, nb[:, 3]) / h_safe), -3.0, 3.0)
    ctx[:K, 4] = ious
    ctx[:K, 5] = float(flag_time)
    mask[:K] = True
    return ctx, mask

def assemble_event(
    seq_name: str,
    t_lost: np.ndarray,
    t_conf: int,
    hist: np.ndarray,
    hist_len: np.ndarray,
    track_color: np.ndarray,
    cand: np.ndarray,
    cand_hist: np.ndarray,
    ctx_lost: np.ndarray,
    ctx_conf: np.ndarray,
    ctx_mask_lost: np.ndarray,
    ctx_mask_conf: np.ndarray,
    W: int = 960,
    H: int = 540,
) -> dict:
    """Pure function: assemble observable event dictionary (used identically by sampler and online tracker)."""
    N = len(t_lost)
    M = len(cand)
    gap = (int(t_conf) - t_lost.astype(np.int32)).astype(np.int32)
    gap_bin = np.zeros((N,), dtype=np.int8)
    gap_bin[(gap >= 16) & (gap <= 45)] = 1
    gap_bin[gap >= 46] = 2
    return {
        "seq": str(seq_name),
        "W": int(W),
        "H": int(H),
        "t_lost": t_lost.astype(np.int32),
        "t_conf": int(t_conf),
        "hist": hist.astype(np.float32),               # (N, K, 4) right-aligned: valid boxes at end [K-L : K]
        "hist_len": hist_len.astype(np.int32),         # (N,)
        "track_color": track_color.astype(np.float32), # (N, 128) L1-normalized
        "gap": gap,                                    # (N,) int32
        "gap_bin": gap_bin,                            # (N,) int8
        "cand": cand.astype(np.float32),               # (M, 4)
        "cand_hist": cand_hist.astype(np.float32),     # (M, 128) L1-normalized
        "ctx_lost": ctx_lost.astype(np.float32),       # (N, 8, 6)
        "ctx_conf": ctx_conf.astype(np.float32),       # (N, 8, 6)
        "ctx_mask_lost": ctx_mask_lost.astype(bool),   # (N, 8)
        "ctx_mask_conf": ctx_mask_conf.astype(bool),   # (N, 8)
    }

def event_is_hard(cand_boxes: np.ndarray, label: np.ndarray, factor: float = 1.5) -> bool:
    """Geometric hardness check: True if any matched track has another candidate within factor * max(w, h)."""
    if len(cand_boxes) < 2:
        return False
    for i, c_idx in enumerate(label):
        if c_idx < 0 or c_idx >= len(cand_boxes):
            continue
        tb = cand_boxes[c_idx]
        rad = factor * max(float(tb[2]), float(tb[3]))
        dists = np.hypot(cand_boxes[:, 0] - tb[0], cand_boxes[:, 1] - tb[1])
        dists[c_idx] = 1e9
        if float(dists.min()) < rad:
            return True
    return False


class SeqIndex:
    """Fast per-sequence lookup for event sampling."""
    def __init__(self, seq: SeqData, hist_cache: np.ndarray | None = None):
        self.seq = seq
        self.name = seq.name
        self.W = seq.W
        self.H = seq.H
        if hist_cache is not None and len(hist_cache) == len(seq.rows):
            s = hist_cache.astype(np.float32)
            if s.ndim == 2:
                s = np.repeat(s[:, None, :], 5, axis=1)
            sums = s.sum(axis=-1, keepdims=True)
            self.hist_views = np.where(sums > 0, s / np.maximum(sums, 1e-6), 0.0).astype(np.float32)
            self.hist_norm = self.hist_views[:, 0, :]
        else:
            self.hist_views = np.zeros((len(seq.rows), 5, 128), dtype=np.float32)
            self.hist_norm = self.hist_views[:, 0, :]

        # Precompute track frame -> local index map and valid anchors
        self.track_ids = sorted(seq.tracks.keys())
        self.frame_boxes = {}
        self.frame_ids = {}
        self.frame_ridx = {}
        for f, sl in seq.by_frame.items():
            sub = seq.rows[sl]
            x, y, w, h = sub[:, 2], sub[:, 3], sub[:, 4], sub[:, 5]
            self.frame_boxes[f] = np.stack([x + 0.5 * w, y + 0.5 * h, w, h], axis=1).astype(np.float32)
            self.frame_ids[f] = sub[:, 1].astype(np.int32)
            self.frame_ridx[f] = np.arange(sl.start, sl.stop, dtype=np.int32)

    def get_track_box_at(self, tid: int, frame: int) -> tuple[np.ndarray | None, int]:
        tinfo = self.seq.tracks.get(tid)
        if tinfo is None:
            return None, -1
        f_arr = tinfo["frames"]
        pos = int(np.searchsorted(f_arr, frame))
        if pos < len(f_arr) and f_arr[pos] == frame:
            return tinfo["boxes"][pos], int(tinfo["row_idx"][pos])
        return None, -1

    def get_track_history_up_to(self, tid: int, t_lost: int, K: int = 20, noise_sigma: float = 0.0, rng=None):
        """Strictly uses frames <= t_lost (no future leakage)."""
        tinfo = self.seq.tracks[tid]
        f_arr = tinfo["frames"]
        pos = int(np.searchsorted(f_arr, t_lost, side="right"))
        # All indices < pos have frame <= t_lost
        start = max(0, pos - K)
        sub_boxes = tinfo["boxes"][start:pos].copy()
        sub_ridx = tinfo["row_idx"][start:pos]
        L = len(sub_boxes)
        if L == 0:
            return np.zeros((K, 4), dtype=np.float32), 0, np.zeros((128,), dtype=np.float32), np.zeros((4,), dtype=np.float32)
        last_clean = sub_boxes[-1].copy()
        if noise_sigma > 0.0 and rng is not None:
            sz = np.maximum(4.0, sub_boxes[:, 2:4])
            sub_boxes[:, 0:2] += rng.normal(0.0, noise_sigma, size=(L, 2)).astype(np.float32) * sz
        out = np.zeros((K, 4), dtype=np.float32)
        out[K - L : K] = sub_boxes
        # Average last up to 3 histograms
        c_slice = self.hist_norm[sub_ridx[max(0, L - 3) : L]]
        c_mean = c_slice.mean(axis=0)
        cs = float(c_mean.sum())
        if cs > 0:
            c_mean = c_mean / cs
        return out, L, c_mean.astype(np.float32), last_clean

def sample_gap(rng, cfg: EventCfg, bin_idx: int | None = None) -> int:
    if bin_idx is None:
        bin_idx = int(rng.choice(len(cfg.gap_bins), p=cfg.gap_probs))
    lo, hi = cfg.gap_bins[bin_idx]
    return int(rng.integers(lo, hi + 1))

def sample_event(idx: SeqIndex, rng, cfg: EventCfg, want_hard: bool | None = None) -> dict | None:
    if len(idx.track_ids) < 2:
        return None
    modes = ("occl", "drop", "nomatch_exit", "nomatch_miss")

    for _attempt in range(25):
        mode = str(rng.choice(modes, p=cfg.mode_probs))
        g_main = sample_gap(rng, cfg)

        # Filter eligible primary tracks
        eligible = []
        for tid in idx.track_ids:
            f_arr = idx.seq.tracks[tid]["frames"]
            if len(f_arr) >= cfg.min_hist + g_main + 3:
                eligible.append(tid)
        if not eligible:
            continue

        tid_a = int(rng.choice(eligible))
        tinfo_a = idx.seq.tracks[tid_a]
        f_a = tinfo_a["frames"]
        b_a = tinfo_a["boxes"]

        if mode == "nomatch_exit":
            t_lost_a = int(f_a[-1])
            t_conf = t_lost_a + g_main
            if len(f_a) < cfg.min_hist:
                continue
        else:
            max_i = len(f_a) - g_main - 3
            if max_i <= cfg.min_hist:
                continue
            pick_i = int(rng.integers(cfg.min_hist - 1, max_i))
            t_lost_a = int(f_a[pick_i])
            t_conf = t_lost_a + g_main
            if mode != "nomatch_miss":
                box_at_conf, _ = idx.get_track_box_at(tid_a, t_conf)
                if box_at_conf is None:
                    continue

        # Build track pool: primary track A + 0..3 auxiliary tracks alive near A
        tracks_list = [(tid_a, t_lost_a, mode)]
        last_box_a = b_a[int(np.searchsorted(f_a, t_lost_a, side="right")) - 1]
        h_a = max(12.0, float(last_box_a[3]))

        n_aux = int(rng.integers(0, min(4, cfg.n_trk_max)))
        if n_aux > 0 and t_conf in idx.frame_ids:
            cands_tids = [int(t) for t in idx.frame_ids[t_conf] if int(t) != tid_a]
            rng.shuffle(cands_tids)
            for tid_b in cands_tids:
                if len(tracks_list) >= 1 + n_aux:
                    break
                f_b = idx.seq.tracks[tid_b]["frames"]
                g_b = sample_gap(rng, cfg)
                t_lost_b = t_conf - g_b
                pos_b = int(np.searchsorted(f_b, t_lost_b, side="right"))
                if pos_b >= cfg.min_hist:
                    lb_b = idx.seq.tracks[tid_b]["boxes"][pos_b - 1]
                    if np.hypot(lb_b[0] - last_box_a[0], lb_b[1] - last_box_a[1]) <= 6.0 * h_a:
                        tracks_list.append((tid_b, t_lost_b, "drop"))

        N = len(tracks_list)
        K = cfg.hist_len
        hist_arr = np.zeros((N, K, 4), dtype=np.float32)
        hist_len_arr = np.zeros((N,), dtype=np.int32)
        track_color_arr = np.zeros((N, 128), dtype=np.float32)
        t_lost_arr = np.zeros((N,), dtype=np.int32)
        ctx_lost_arr = np.zeros((N, cfg.n_ctx, 6), dtype=np.float32)
        ctx_conf_arr = np.zeros((N, cfg.n_ctx, 6), dtype=np.float32)
        ctx_mask_lost = np.zeros((N, cfg.n_ctx), dtype=bool)
        ctx_mask_conf = np.zeros((N, cfg.n_ctx), dtype=bool)
        side_arr = np.zeros((N,), dtype=np.int8)
        true_box_arr = np.zeros((N, 4), dtype=np.float32)
        has_true_target = np.zeros((N,), dtype=bool)
        track_gt_ids = np.zeros((N,), dtype=np.int32)

        cand_boxes_list = []
        cand_hists_list = []
        cand_gt_ids_list = []
        cand_kind_list = []

        neigh_at_conf = idx.frame_boxes.get(t_conf, np.zeros((0, 4), dtype=np.float32))

        for i, (tid, tl, m_i) in enumerate(tracks_list):
            track_gt_ids[i] = tid
            t_lost_arr[i] = tl
            h_i, l_i, col_i, last_clean = idx.get_track_history_up_to(
                tid, tl, K=K, noise_sigma=cfg.hist_pos_sigma, rng=rng
            )
            hist_arr[i] = h_i
            hist_len_arr[i] = l_i
            track_color_arr[i] = col_i

            neigh_tl = idx.frame_boxes.get(tl, np.zeros((0, 4), dtype=np.float32))
            ctx_lost_arr[i], ctx_mask_lost[i] = build_neighbor_context(last_clean, neigh_tl, 0.0, cfg.n_ctx)
            ctx_conf_arr[i], ctx_mask_conf[i] = build_neighbor_context(last_clean, neigh_at_conf, 1.0, cfg.n_ctx)

            tb, ridx = idx.get_track_box_at(tid, t_conf)
            if tb is not None:
                true_box_arr[i] = tb
                has_true_target[i] = True
            else:
                true_box_arr[i] = last_clean
                has_true_target[i] = False

            # Add true candidate unless mode removes it or random 5% detector miss
            if m_i not in ("nomatch_exit", "nomatch_miss") and tb is not None:
                if rng.random() >= cfg.p_drop_true_cand:
                    cb = tb.copy()
                    # Add detector position and size jitter
                    cb[0:2] += rng.normal(0.0, cfg.cand_pos_sigma, size=2).astype(np.float32) * np.maximum(4.0, cb[2:4])
                    cb[2:4] *= rng.uniform(1.0 - cfg.cand_size_jitter, 1.0 + cfg.cand_size_jitter, size=2).astype(np.float32)
                    s_i = 0
                    if rng.random() < cfg.p_partial:
                        s_i = int(rng.integers(1, 5))
                        frac_i = float(rng.uniform(0.6, 0.90))
                        cb = partial_crop(cb, s_i, frac_i)
                        side_arr[i] = s_i
                    cand_boxes_list.append(cb)
                    cand_hists_list.append(idx.hist_views[ridx, s_i] if ridx >= 0 else np.zeros((128,), dtype=np.float32))
                    cand_gt_ids_list.append(tid)
                    cand_kind_list.append(0)

        # Add visible distractors / new vehicles from frame t_conf (real objects with other IDs)
        used_tids = set(track_gt_ids.tolist())
        if t_conf in idx.frame_ids:
            f_ids = idx.frame_ids[t_conf]
            f_boxes = idx.frame_boxes[t_conf]
            f_ridx = idx.frame_ridx[t_conf]
            other_idx = [k for k, tid_k in enumerate(f_ids) if int(tid_k) not in used_tids]
            if other_idx:
                # Sort by proximity to primary track A
                d_other = [np.hypot(f_boxes[k, 0] - true_box_arr[0, 0], f_boxes[k, 1] - true_box_arr[0, 1]) for k in other_idx]
                sorted_other = [other_idx[k] for k in np.argsort(d_other)]
                for k in sorted_other[: min(len(sorted_other), cfg.n_cand_max - len(cand_boxes_list))]:
                    cb = f_boxes[k].copy()
                    cb[0:2] += rng.normal(0.0, cfg.cand_pos_sigma, size=2).astype(np.float32) * np.maximum(4.0, cb[2:4])
                    cand_boxes_list.append(cb)
                    cand_hists_list.append(idx.hist_norm[f_ridx[k]])
                    cand_gt_ids_list.append(-1)  # distractor / newborn -> dustbin
                    cand_kind_list.append(3)

        # Add 0..2 false positive or hard close distractor boxes so M >= 1
        n_fp = int(rng.integers(1 if len(cand_boxes_list) == 0 else 0, 3))
        if want_hard and len(cand_boxes_list) > 0 and len(cand_boxes_list) < cfg.n_cand_max:
            n_fp = max(n_fp, 1)
        for _ in range(n_fp):
            if len(cand_boxes_list) >= cfg.n_cand_max:
                break
            ref = true_box_arr[0]
            rad = max(float(ref[2]), float(ref[3]))
            offset = rng.uniform(-1.1 * rad, 1.1 * rad, size=2).astype(np.float32) if want_hard else rng.uniform(-3.0 * rad, 3.0 * rad, size=2).astype(np.float32)
            fp_box = np.array([
                np.clip(ref[0] + offset[0], 10.0, idx.W - 10.0),
                np.clip(ref[1] + offset[1], 10.0, idx.H - 10.0),
                ref[2] * float(rng.uniform(0.85, 1.15)),
                ref[3] * float(rng.uniform(0.85, 1.15)),
            ], dtype=np.float32)
            cand_boxes_list.append(fp_box)
            # Random color histogram from sequence
            rand_h = idx.hist_norm[int(rng.integers(0, len(idx.hist_norm)))] if len(idx.hist_norm) > 0 else np.zeros((128,), dtype=np.float32)
            cand_hists_list.append(rand_h)
            cand_gt_ids_list.append(-1)
            cand_kind_list.append(2)

        M = len(cand_boxes_list)
        if M == 0 or M > cfg.n_cand_max:
            continue

        # Shuffle tracks and candidates independently, then recompute ground-truth labels
        perm_t = rng.permutation(N)
        perm_c = rng.permutation(M)

        t_lost_arr = t_lost_arr[perm_t]
        hist_arr = hist_arr[perm_t]
        hist_len_arr = hist_len_arr[perm_t]
        track_color_arr = track_color_arr[perm_t]
        ctx_lost_arr = ctx_lost_arr[perm_t]
        ctx_conf_arr = ctx_conf_arr[perm_t]
        ctx_mask_lost = ctx_mask_lost[perm_t]
        ctx_mask_conf = ctx_mask_conf[perm_t]
        side_arr = side_arr[perm_t]
        true_box_arr = true_box_arr[perm_t]
        has_true_target = has_true_target[perm_t]
        track_gt_ids = track_gt_ids[perm_t]

        cand_arr = np.stack(cand_boxes_list, axis=0)[perm_c]
        cand_hist_arr = np.stack(cand_hists_list, axis=0)[perm_c]
        cand_gt_ids = np.array(cand_gt_ids_list, dtype=np.int32)[perm_c]
        cand_kind_arr = np.array(cand_kind_list, dtype=np.int8)[perm_c]

        label = np.full((N,), -1, dtype=np.int32)
        cand_label = np.full((M,), -1, dtype=np.int32)
        for i in range(N):
            matches = np.nonzero(cand_gt_ids == track_gt_ids[i])[0]
            if len(matches) > 0:
                j = int(matches[0])
                label[i] = j
                cand_label[j] = i

        hard_flag = event_is_hard(cand_arr, label, factor=cfg.hard_dist_factor)
        if want_hard is True and not hard_flag:
            continue
        if want_hard is False and hard_flag:
            continue

        ev = assemble_event(
            seq_name=idx.name,
            t_lost=t_lost_arr,
            t_conf=t_conf,
            hist=hist_arr,
            hist_len=hist_len_arr,
            track_color=track_color_arr,
            cand=cand_arr,
            cand_hist=cand_hist_arr,
            ctx_lost=ctx_lost_arr,
            ctx_conf=ctx_conf_arr,
            ctx_mask_lost=ctx_mask_lost,
            ctx_mask_conf=ctx_mask_conf,
            W=idx.W,
            H=idx.H,
        )
        ev["label"] = label
        ev["cand_label"] = cand_label
        ev["true_box"] = true_box_arr
        ev["has_true_target"] = has_true_target
        ev["is_hard"] = bool(hard_flag)
        ev["mode"] = mode
        ev["side"] = side_arr
        ev["cand_kind"] = cand_kind_arr
        return ev

    return None
