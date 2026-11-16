"""Tracker-level metrics on injected occlusion intervals (IRR, WLR, FragRate, Lost)
PLUS standard MOT metrics (IDF1, MOTA, IDSW) computed across all visible vehicle detections.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.io_mot import SeqData
from src.tracker import OnlineTracker


def evaluate_tracker_on_seq(
    seq: SeqData,
    hide_plan: list[dict],
    dets_by_frame: dict[int, dict],
    associator=None,
    a_max: int = 95,
) -> dict:
    """Run OnlineTracker on `dets_by_frame` and compute both interval recovery metrics (IRR, WLR, FragRate)
    and standard MOTChallenge identity/tracking metrics (IDF1, MOTA, IDSW).
    """
    tracker = OnlineTracker(
        associator=associator,
        iou_thr=0.30,
        a_max=a_max,
        W=seq.W,
        H=seq.H,
        seq_name=seq.name,
    )
    for f in range(1, seq.n_frames + 1):
        d = dets_by_frame[f]
        tracker.step(f, d["boxes"], d["hists"], d["gt_tids"])

    assigned_id = {}
    gt_to_pred_cooccur = {}
    gt_totals = {}
    pred_totals = {}
    last_pred_for_gt = {}
    idsw = 0
    total_gt_dets = 0
    total_pred_dets = 0
    matched_dets = 0

    for f in range(1, seq.n_frames + 1):
        d = dets_by_frame[f]
        outs = tracker.frame_outputs.get(f, [])
        total_gt_dets += len(d["gt_tids"])
        total_pred_dets += len(outs)
        for o in outs:
            pid = int(o["track_id"])
            gid = int(o["gt_tid"])
            pred_totals[pid] = pred_totals.get(pid, 0) + 1
            if gid != -1:
                assigned_id[(f, gid)] = pid
                gt_totals[gid] = gt_totals.get(gid, 0) + 1
                gt_to_pred_cooccur[(gid, pid)] = gt_to_pred_cooccur.get((gid, pid), 0) + 1
                matched_dets += 1
                if gid in last_pred_for_gt and last_pred_for_gt[gid] != pid:
                    idsw += 1
                last_pred_for_gt[gid] = pid

    # Compute exact global IDF1 via maximum bipartite overlap between GT tracks and predicted tracks
    g_list = sorted(gt_totals.keys())
    p_list = sorted(pred_totals.keys())
    if g_list and p_list:
        g_idx = {g: i for i, g in enumerate(g_list)}
        p_idx = {p: j for j, p in enumerate(p_list)}
        overlap = np.zeros((len(g_list), len(p_list)), dtype=np.int32)
        for (g, p), cnt in gt_to_pred_cooccur.items():
            overlap[g_idx[g], p_idx[p]] = cnt
        r_ind, c_ind = linear_sum_assignment(-overlap)
        idtp = int(overlap[r_ind, c_ind].sum())
        idf1 = float(2.0 * idtp / max(1, total_gt_dets + total_pred_dets))
    else:
        idtp = 0
        idf1 = 0.0

    fn = max(0, total_gt_dets - matched_dets)
    fp = max(0, total_pred_dets - matched_dets)
    mota = float(1.0 - (fn + fp + idsw) / max(1, total_gt_dets))

    n_intervals = 0
    n_recovered = 0
    n_wrong_link = 0
    n_frag = 0
    n_lost = 0

    by_gap_tot = [0, 0, 0]
    by_gap_rec = [0, 0, 0]
    interval_outcomes = []

    reassoc_by_frame_gt = {}
    for r in tracker.reassoc_log:
        reassoc_by_frame_gt[(r["frame"], r["gt_tid_after"])] = r

    for item in hide_plan:
        tid = int(item["tid"])
        t_lost = int(item["t_lost"])
        t_reap = int(item["t_reappear"])
        gb = int(item["gap_bin"])

        id_before = assigned_id.get((t_lost, tid), None)
        if id_before is None:
            continue

        n_intervals += 1
        by_gap_tot[gb] += 1

        id_after = assigned_id.get((t_reap, tid), None)
        r_log = reassoc_by_frame_gt.get((t_reap, tid), None)

        if id_after is None:
            n_lost += 1
            interval_outcomes.append(0.0)
        elif id_after == id_before:
            n_recovered += 1
            by_gap_rec[gb] += 1
            interval_outcomes.append(1.0)
        else:
            if r_log is not None and not r_log["correct"]:
                n_wrong_link += 1
            else:
                n_frag += 1
            interval_outcomes.append(0.0)

    denom = max(1, n_intervals)
    return {
        "seq": seq.name,
        "n_intervals": n_intervals,
        "by_gap_tot": by_gap_tot,
        "by_gap_rec": by_gap_rec,
        "IRR": float(n_recovered / denom),
        "WLR": float(n_wrong_link / denom),
        "FragRate": float(n_frag / denom),
        "LostRate": float(n_lost / denom),
        "IRR_gap_short": float(by_gap_rec[0] / max(1, by_gap_tot[0])),
        "IRR_gap_med": float(by_gap_rec[1] / max(1, by_gap_tot[1])),
        "IRR_gap_long": float(by_gap_rec[2] / max(1, by_gap_tot[2])),
        "IDF1": idf1,
        "MOTA": mota,
        "IDSW": int(idsw),
        "IDTP": int(idtp),
        "total_gt_dets": int(total_gt_dets),
        " total_pred_dets": int(total_pred_dets),
        "outcomes": np.array(interval_outcomes, dtype=np.float32),
        "reassoc_total": len(tracker.reassoc_log),
        "reassoc_wrong": sum(1 for r in tracker.reassoc_log if not r["correct"]),
    }
