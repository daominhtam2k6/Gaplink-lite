"""Per-sequence scoring for the live YOLOv8n + VMS evaluation (factored out of scripts/eval_live_security.py).

Logic is copied 1:1 from eval_live_security.main(); scripts/eval_live_v12.py --parity checks that
it reproduces the frozen v1.1 numbers exactly before it is used for anything new.
"""
from __future__ import annotations

COUNT_KEYS = (
    "nat_gaps_total", "nat_gaps_recovered", "reassoc_total_valid_gt", "reassoc_wrong",
    "route_gt_all", "route_gt_occl", "route_tp_all", "route_tp_occl", "route_fp", "route_unverified",
    "dwell_gt_all", "dwell_gt_occl", "dwell_tp_all", "dwell_tp_occl", "dwell_fp",
)


def empty_counts() -> dict:
    return {k: 0 for k in COUNT_KEYS}


def add_counts(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in COUNT_KEYS}


def run_method_on_seq(seq, dets, det_gt_map, nat_gaps, zones, gt_sets, assoc, tcfg, a_max=None) -> dict:
    from src.security_analytics import SecurityVMSEngine
    from src.tracker import OnlineTracker

    gt_za_zb, gt_za_zb_occl, gt_dwell, gt_dwell_occl = gt_sets
    trk = OnlineTracker(associator=assoc, iou_thr=float(tcfg["iou_thr"]),
                        a_max=int(a_max if a_max is not None else tcfg["a_max"]),
                        W=seq.W, H=seq.H, seq_name=seq.name, min_hits=3)
    vms = SecurityVMSEngine(trk, zones=zones, forbidden_routes=[("Z_A", "Z_B")], max_dwell_sec=5.0, fps=25.0)
    pred_tid_at: dict[tuple[int, int], int] = {}
    for f in range(1, seq.n_frames + 1):
        res = vms.step(f, dets[f]["boxes"], dets[f]["hists"], det_gt_map[f])
        for o in res["tracks"]:
            if o["gt_tid"] != -1:
                pred_tid_at[(f, int(o["gt_tid"]))] = int(o["track_id"])

    st = empty_counts()
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

    st["route_gt_all"] += len(gt_za_zb)
    st["route_gt_occl"] += len(gt_za_zb_occl)
    st["route_unverified"] += len(vms.route_mon.unverified_entries.get("Z_B", set()))
    caught = set()
    for al in vms.route_mon.alerts:
        g_orig, g_dest = al["origin_gt_tid"], al["dest_gt_tid"]
        if g_orig != -1 and g_orig == g_dest and (g_dest in gt_za_zb):
            caught.add(g_dest)
        elif g_orig != -1 and g_dest != -1 and (g_orig != g_dest or g_dest not in gt_za_zb):
            st["route_fp"] += 1
    st["route_tp_all"] += len(caught)
    st["route_tp_occl"] += len(caught & gt_za_zb_occl)

    st["dwell_gt_all"] += len(gt_dwell)
    st["dwell_gt_occl"] += len(gt_dwell_occl)
    caught_d = set()
    for al in vms.dwell_mon.alerts:
        g_orig, g_now = al["origin_gt_tid"], al["now_gt_tid"]
        if g_orig != -1 and g_orig == g_now and (g_now in gt_dwell):
            caught_d.add(g_now)
        elif g_orig != -1 and g_now != -1 and (g_orig != g_now):
            st["dwell_fp"] += 1
    st["dwell_tp_all"] += len(caught_d)
    st["dwell_tp_occl"] += len(caught_d & gt_dwell_occl)
    return st


def summarize(s: dict) -> dict:
    nat_irr = s["nat_gaps_recovered"] / max(1, s["nat_gaps_total"])
    nat_wlr = s["reassoc_wrong"] / max(1, s["reassoc_total_valid_gt"])
    r_rec_occl = s["route_tp_occl"] / max(1, s["route_gt_occl"])
    r_far = s["route_fp"] / max(1, s["route_tp_all"] + s["route_fp"])
    d_rec_occl = s["dwell_tp_occl"] / max(1, s["dwell_gt_occl"])
    d_prec = s["dwell_tp_all"] / max(1, s["dwell_tp_all"] + s["dwell_fp"])
    d_f1 = (2 * d_prec * d_rec_occl / max(1e-6, d_prec + d_rec_occl)) if (d_prec + d_rec_occl) > 0 else 0.0
    return {
        "raw_counts": dict(s),
        "natural_occlusion": {"n_gaps": s["nat_gaps_total"], "nat_irr": round(nat_irr, 4), "nat_wlr": round(nat_wlr, 4)},
        "forbidden_route": {"recall_all": round(s["route_tp_all"] / max(1, s["route_gt_all"]), 4),
                            "recall_occl": round(r_rec_occl, 4), "false_alarm_rate": round(r_far, 4)},
        "dwell_alert": {"recall_all": round(s["dwell_tp_all"] / max(1, s["dwell_gt_all"]), 4),
                        "recall_occl": round(d_rec_occl, 4), "precision": round(d_prec, 4), "f1_occl": round(d_f1, 4)},
    }


def select_operating_point(rows: list[dict], wlr_key: str = "nat_wlr", irr_key: str = "nat_irr", wlr_max: float = 0.05) -> dict:
    """PREREG §3 rule: highest IRR with WLR <= wlr_max; if none qualifies, lowest WLR.

    Ties on IRR are broken by lower WLR, then by the smaller (more conservative) threshold index order given.
    """
    if not rows:
        raise ValueError("empty sweep")
    ok = [r for r in rows if r[wlr_key] <= wlr_max]
    if ok:
        best = max(ok, key=lambda r: (r[irr_key], -r[wlr_key]))
        return {**best, "rule": f"max {irr_key} s.t. {wlr_key}<={wlr_max}"}
    best = min(rows, key=lambda r: (r[wlr_key], -r[irr_key]))
    return {**best, "rule": f"no config with {wlr_key}<={wlr_max}: min {wlr_key}"}
