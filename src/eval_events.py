"""Event-level evaluation and cluster/paired bootstrap by sequence."""
import math
import numpy as np

CHI2_2D_90 = 4.605170185988092

def evaluate_predictions(events: list[dict], preds: list[np.ndarray], gaussians: list[tuple[np.ndarray, np.ndarray]] | None = None) -> dict:
    """Compute event-level metrics over a list of events and their predicted track decisions.
    Each pred in `preds` is an int32 array of shape (N,) with candidate indices in 0..M-1 or -1 (dustbin).
    If `gaussians` is provided, each element is (pred_box_px (N,4), sigma_px (N,4)).
    """
    n_tracks = 0
    n_correct = 0
    n_wrong_link = 0
    tp_dust = 0
    fp_dust = 0
    fn_dust = 0

    by_gap_total = [0, 0, 0]
    by_gap_corr = [0, 0, 0]

    hard_total = 0
    hard_corr = 0

    cov_total = 0
    cov_hit = 0
    cov_by_gap_total = [0, 0, 0]
    cov_by_gap_hit = [0, 0, 0]
    nll_list = []

    seq_track_corr = {}

    for idx_e, (ev, dec) in enumerate(zip(events, preds)):
        seq = ev["seq"]
        lbl = ev["label"]
        gb = ev["gap_bin"]
        is_hard = bool(ev.get("is_hard", False))
        N = len(lbl)

        corr_mask = (dec == lbl)
        seq_track_corr.setdefault(seq, []).extend(corr_mask.astype(np.float32).tolist())

        for i in range(N):
            n_tracks += 1
            gt_j = int(lbl[i])
            pr_j = int(dec[i])
            b = int(gb[i])
            by_gap_total[b] += 1
            if is_hard:
                hard_total += 1

            if pr_j == gt_j:
                n_correct += 1
                by_gap_corr[b] += 1
                if is_hard:
                    hard_corr += 1
            else:
                if pr_j >= 0:
                    n_wrong_link += 1

            if gt_j == -1 and pr_j == -1:
                tp_dust += 1
            elif gt_j >= 0 and pr_j == -1:
                fp_dust += 1
            elif gt_j == -1 and pr_j >= 0:
                fn_dust += 1

        if gaussians is not None and "true_box" in ev and "has_true_target" in ev:
            pred_b, sig_b = gaussians[idx_e]
            tb = ev["true_box"]
            has_t = ev["has_true_target"]
            for i in range(N):
                if not has_t[i]:
                    continue
                b = int(gb[i])
                sx = max(1.0, float(sig_b[i, 0]))
                sy = max(1.0, float(sig_b[i, 1]))
                dx = float(tb[i, 0] - pred_b[i, 0])
                dy = float(tb[i, 1] - pred_b[i, 1])
                d2 = (dx / sx) ** 2 + (dy / sy) ** 2
                nll_2d = 0.5 * (2.0 * math.log(sx) + 2.0 * math.log(sy) + d2 + 2.0 * math.log(2.0 * math.pi))
                nll_list.append(nll_2d)
                cov_total += 1
                cov_by_gap_total[b] += 1
                if d2 <= CHI2_2D_90:
                    cov_hit += 1
                    cov_by_gap_hit[b] += 1

    top1 = n_correct / max(1, n_tracks)
    wlr = n_wrong_link / max(1, n_tracks)
    dust_p = tp_dust / max(1, tp_dust + fp_dust)
    dust_r = tp_dust / max(1, tp_dust + fn_dust)
    dust_f1 = 2 * dust_p * dust_r / max(1e-8, dust_p + dust_r)

    res = {
        "n_events": len(events),
        "n_tracks": n_tracks,
        "top1": float(top1),
        "top1_gap_short": float(by_gap_corr[0] / max(1, by_gap_total[0])),
        "top1_gap_med": float(by_gap_corr[1] / max(1, by_gap_total[1])),
        "top1_gap_long": float(by_gap_corr[2] / max(1, by_gap_total[2])),
        "top1_hard": float(hard_corr / max(1, hard_total)),
        "n_hard_tracks": int(hard_total),
        "wrong_link_rate": float(wlr),
        "dustbin_P": float(dust_p),
        "dustbin_R": float(dust_r),
        "dustbin_F1": float(dust_f1),
    }
    if cov_total > 0:
        res["coverage90"] = float(cov_hit / cov_total)
        res["coverage90_gap_short"] = float(cov_by_gap_hit[0] / max(1, cov_by_gap_total[0]))
        res["coverage90_gap_med"] = float(cov_by_gap_hit[1] / max(1, cov_by_gap_total[1]))
        res["coverage90_gap_long"] = float(cov_by_gap_hit[2] / max(1, cov_by_gap_total[2]))
        res["nll_2d"] = float(np.mean(nll_list))

    seq_arrays = {k: np.array(v, dtype=np.float32) for k, v in seq_track_corr.items()}
    return res, seq_arrays

def bootstrap_cluster(values_by_seq: dict[str, np.ndarray], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Cluster bootstrap 95% CI by resampling sequences with replacement."""
    seqs = sorted(values_by_seq.keys())
    if len(seqs) <= 1:
        val = float(np.concatenate(list(values_by_seq.values())).mean()) if seqs else 0.0
        return val, val
    sums = np.array([values_by_seq[s].sum() for s in seqs], dtype=np.float64)
    cnts = np.array([len(values_by_seq[s]) for s in seqs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    K = len(seqs)
    idx = rng.integers(0, K, size=(n_boot, K))
    boot_means = sums[idx].sum(axis=1) / np.maximum(1.0, cnts[idx].sum(axis=1))
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def bootstrap_paired(a_by_seq: dict[str, np.ndarray], b_by_seq: dict[str, np.ndarray], n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired cluster bootstrap for difference (A - B) resampling the exact same sequences in each draw."""
    seqs = sorted(set(a_by_seq.keys()) & set(b_by_seq.keys()))
    a_sums = np.array([a_by_seq[s].sum() for s in seqs], dtype=np.float64)
    a_cnts = np.array([len(a_by_seq[s]) for s in seqs], dtype=np.float64)
    b_sums = np.array([b_by_seq[s].sum() for s in seqs], dtype=np.float64)
    b_cnts = np.array([len(b_by_seq[s]) for s in seqs], dtype=np.float64)

    mean_a = float(a_sums.sum() / np.maximum(1.0, a_cnts.sum()))
    mean_b = float(b_sums.sum() / np.maximum(1.0, b_cnts.sum()))
    delta = mean_a - mean_b

    rng = np.random.default_rng(seed)
    K = len(seqs)
    idx = rng.integers(0, K, size=(n_boot, K))
    boot_a = a_sums[idx].sum(axis=1) / np.maximum(1.0, a_cnts[idx].sum(axis=1))
    boot_b = b_sums[idx].sum(axis=1) / np.maximum(1.0, b_cnts[idx].sum(axis=1))
    diff = boot_a - boot_b
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta": delta,
        "ci95_lo": float(np.percentile(diff, 2.5)),
        "ci95_hi": float(np.percentile(diff, 97.5)),
        "n_clusters": K,
    }
