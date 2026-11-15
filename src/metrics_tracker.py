"""Tracker-level metrics on injected occlusion intervals: IRR, WLR, FragRate, Lost."""
import numpy as np
from src.io_mot import SeqData
from src.tracker import OnlineTracker


def evaluate_tracker_on_seq(
    seq: SeqData,
    hide_plan: list[dict],
    dets_by_frame: dict[int, dict],
    associator=None,
    a_max: int = 60,
) -> dict:
    """Run OnlineTracker on `dets_by_frame` and score recovery across `hide_plan` intervals."""
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
    for f, outs in tracker.frame_outputs.items():
        for o in outs:
            if o["gt_tid"] != -1:
                assigned_id[(f, int(o["gt_tid"]))] = int(o["track_id"])

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
        "IRR": float(n_recovered / denom),
        "WLR": float(n_wrong_link / denom),
        "FragRate": float(n_frag / denom),
        "LostRate": float(n_lost / denom),
        "IRR_gap_short": float(by_gap_rec[0] / max(1, by_gap_tot[0])),
        "IRR_gap_med": float(by_gap_rec[1] / max(1, by_gap_tot[1])),
        "IRR_gap_long": float(by_gap_rec[2] / max(1, by_gap_tot[2])),
        "outcomes": np.array(interval_outcomes, dtype=np.float32),
        "reassoc_total": len(tracker.reassoc_log),
        "reassoc_wrong": sum(1 for r in tracker.reassoc_log if not r["correct"]),
    }
