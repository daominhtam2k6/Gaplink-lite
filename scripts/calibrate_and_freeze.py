"""Day 4-6: Fit post-hoc calibration on val half-A, verify coverage on val half-B,
sweep acceptance thresholds on val under the pre-registered WLR <= 5.0% rule,
and freeze release/final.pt + release/final.sha256 + release/calib.json + release/tracker.yaml.
"""
import csv
import hashlib
import json
import pickle
import shutil
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import yaml

from src.dataset import EventListDataset, collate_padded_events
from src.eval_events import evaluate_predictions, bootstrap_cluster, bootstrap_paired
from src.calib import fit_gap_calibration, compute_calib_stats
from src.io_mot import load_seq
from src.inject import make_hide_plan, make_detections
from src.baselines import ConstantVelocityBaseline, KalmanMahalanobisBaseline
from src.metrics_tracker import evaluate_tracker_on_seq
from src.model.gaplink import GapLinkLite, GapLinkAssociator

ROOT = Path(r"D:\gaplink")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def infer_all_events(model: GapLinkLite, events: list[dict], device: torch.device):
    ds = EventListDataset(events)
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate_padded_events)
    P_list = []
    gaussians = []
    idx = 0
    model.eval()
    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch_dev)
        P_b = out["P"].cpu().numpy()
        pb_b = out["pred_box"].cpu().numpy()
        sb_b = out["sigma_px"].cpu().numpy()
        B = P_b.shape[0]
        N_pad = P_b.shape[1] - 1
        M_pad = P_b.shape[2] - 1
        for b in range(B):
            ev = events[idx]
            idx += 1
            N = min(len(ev["t_lost"]), N_pad)
            M = min(len(ev["cand"]), M_pad)
            P_full = P_b[b]
            row_idx = list(range(N)) + [N_pad]
            col_idx = list(range(M)) + [M_pad]
            P_sub = P_full[row_idx][:, col_idx]
            P_list.append(P_sub)

            pb = np.zeros((len(ev["t_lost"]), 4), dtype=np.float32)
            sb = np.ones((len(ev["t_lost"]), 4), dtype=np.float32) * 10.0
            pb[:N] = pb_b[b, :N]
            sb[:N] = sb_b[b, :N]
            gaussians.append((pb, sb))
    return P_list, gaussians


def decide_with_tau(events: list[dict], P_list: list[np.ndarray], tau: float) -> list[np.ndarray]:
    preds = []
    for ev, P in zip(events, P_list):
        N = len(ev["t_lost"])
        M = len(ev["cand"])
        out_arr = np.full((N,), -1, dtype=np.int32)
        if N > 0 and M > 0:
            P_real = P[:N, :M]
            P_dust = P[:N, M]
            r_ind, c_ind = linear_sum_assignment(-P_real)
            for r, c in zip(r_ind, c_ind):
                if P_real[r, c] >= tau and P_real[r, c] > P_dust[r]:
                    out_arr[r] = int(c)
        preds.append(out_arr)
    return preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(ROOT / "data" / "cache" / "val_events.pkl", "rb") as f:
        val_events = pickle.load(f)

    ckpt_path = ROOT / "runs" / "main" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    model = GapLinkLite(
        use_ctx=bool(cfg.get("use_ctx", True)),
        use_color=bool(cfg.get("use_color", True)),
        gap_aware=bool(cfg.get("gap_aware", True)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Running batch inference on 6,000 val events...")
    P_list, raw_gaussians = infer_all_events(model, val_events, device)

    # 1. Fit post-hoc calibration on val half-A (first 3,000 events) and evaluate on val half-B (last 3,000 events)
    half = len(val_events) // 2
    ev_A, g_A = val_events[:half], raw_gaussians[:half]
    ev_B, g_B = val_events[half:], raw_gaussians[half:]

    zero_params = {"a": 0.0, "b": 0.0, "c": 0.0}
    stats_before_B = compute_calib_stats(ev_B, g_B, zero_params)
    calib_params = fit_gap_calibration(ev_A, g_A, target_cov=0.90)
    stats_after_B = compute_calib_stats(ev_B, g_B, calib_params)
    stats_after_all = compute_calib_stats(val_events, raw_gaussians, calib_params)

    calib_out = {
        "params": calib_params,
        "val_halfB_before": stats_before_B,
        "val_halfB_after": stats_after_B,
        "val_all_after": stats_after_all,
    }
    (ROOT / "runs" / "main" / "calib.json").write_text(json.dumps(calib_out, indent=2), encoding="utf-8")
    print("Calibration fitted:", json.dumps(calib_out, indent=2))

    # 2. Sweep tau on val_events under pre-registered rule: highest top1 subject to WLR <= 5.0%
    best_tau = 0.25
    best_ev_res = None
    best_ev_seq_arr = None
    tau_grid = [round(0.05 + 0.05 * k, 2) for k in range(18)]
    tau_sweep_rows = []
    for tau in tau_grid:
        preds = decide_with_tau(val_events, P_list, tau)
        res, seq_arr = evaluate_predictions(val_events, preds)
        tau_sweep_rows.append({"tau": tau, "top1": res["top1"], "wlr": res["wrong_link_rate"], "dust_F1": res["dustbin_F1"]})
        if res["wrong_link_rate"] <= 0.050:
            if best_ev_res is None or res["top1"] > best_ev_res["top1"]:
                best_tau = tau
                best_ev_res = res
                best_ev_seq_arr = seq_arr

    lo, hi = bootstrap_cluster(best_ev_seq_arr)
    best_ev_res["top1_ci95"] = [lo, hi]
    best_ev_res["coverage90"] = stats_after_all["coverage90"]
    best_ev_res["coverage90_gap_short"] = stats_after_all["coverage90_gap_short"]
    best_ev_res["coverage90_gap_med"] = stats_after_all["coverage90_gap_med"]
    best_ev_res["coverage90_gap_long"] = stats_after_all["coverage90_gap_long"]
    best_ev_res["nll_2d"] = stats_after_all["nll_2d"]

    print(f"Selected tau={best_tau} on val_events: top1={best_ev_res['top1']*100:.2f}% | WLR={best_ev_res['wrong_link_rate']*100:.2f}%")

    # 3. Tracker-level evaluation on all 12 val sequences (`full` and `partial` occlusion injection)
    with open(ROOT / "data" / "splits.csv", encoding="utf-8") as f:
        val_seqs = [r["seq"] for r in csv.DictReader(f) if r["split"] == "val"]

    baselines_val = json.loads((ROOT / "runs" / "baselines_val.json").read_text(encoding="utf-8"))
    th_c2 = float(baselines_val["C-2"]["theta"])
    th_c3 = float(baselines_val["C-3"]["theta"])
    th_c4 = float(baselines_val["C-4"]["theta"])
    ls_c4 = float(baselines_val["C-4"]["lambda_s"])
    lc_c4 = float(baselines_val["C-4"]["lambda_c"])

    assoc_map = {
        "C-1": None,
        "C-2": ConstantVelocityBaseline(theta=th_c2),
        "C-3": KalmanMahalanobisBaseline(theta=th_c3, name="C-3"),
        "C-4": KalmanMahalanobisBaseline(theta=th_c4, lambda_s=ls_c4, lambda_c=lc_c4, name="C-4"),
        "GapLink-Lite": GapLinkAssociator(model, device=str(device), tau=best_tau, calib=calib_params),
    }

    mot_dir = ROOT / "data" / "mot"
    hist_dir = ROOT / "data" / "cache" / "hist"

    tracker_val_summary = {}
    for mode in ("full", "partial"):
        mode_res = {k: {"n_intervals": 0, "rec": 0, "wrong": 0, "frag": 0, "lost": 0, "by_seq": {}} for k in assoc_map}
        for i_s, s_name in enumerate(val_seqs):
            seq = load_seq(mot_dir, s_name)
            hp = hist_dir / f"{s_name}.npy"
            hc = np.load(hp) if hp.exists() else None
            plan = make_hide_plan(seq, seed=5000 + i_s, max_density=0.18)
            dets = make_detections(seq, plan, hc, mode=mode, seed=6000 + i_s)
            for m_name, assoc in assoc_map.items():
                r_seq = evaluate_tracker_on_seq(seq, plan, dets, associator=assoc)
                n_int = r_seq["n_intervals"]
                mode_res[m_name]["n_intervals"] += n_int
                mode_res[m_name]["rec"] += int(round(r_seq["IRR"] * n_int))
                mode_res[m_name]["wrong"] += int(round(r_seq["WLR"] * n_int))
                mode_res[m_name]["frag"] += int(round(r_seq["FragRate"] * n_int))
                mode_res[m_name]["lost"] += int(round(r_seq["LostRate"] * n_int))
                mode_res[m_name]["by_seq"][s_name] = r_seq["outcomes"]

        tracker_val_summary[mode] = {}
        for m_name, d_m in mode_res.items():
            tot = max(1, d_m["n_intervals"])
            lo_i, hi_i = bootstrap_cluster(d_m["by_seq"])
            tracker_val_summary[mode][m_name] = {
                "n_intervals": d_m["n_intervals"],
                "IRR": d_m["rec"] / tot,
                "IRR_ci95": [lo_i, hi_i],
                "WLR": d_m["wrong"] / tot,
                "FragRate": d_m["frag"] / tot,
                "LostRate": d_m["lost"] / tot,
            }
        print(f"Val Tracker ({mode}): GapLink-Lite IRR={tracker_val_summary[mode]['GapLink-Lite']['IRR']*100:.2f}% vs C-3 IRR={tracker_val_summary[mode]['C-3']['IRR']*100:.2f}%")

    # Save d4_val summary
    d4_dir = ROOT / "runs" / "d4_val"
    d4_dir.mkdir(parents=True, exist_ok=True)
    (d4_dir / "val_summary.json").write_text(
        json.dumps(
            {
                "selected_tau": best_tau,
                "tau_sweep": tau_sweep_rows,
                "gaplink_val_events": best_ev_res,
                "calib": calib_out,
                "tracker_val": tracker_val_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 4. Freeze release/final.pt and configs
    rel_dir = ROOT / "release"
    rel_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_path, rel_dir / "final.pt")
    final_sha = sha256_file(rel_dir / "final.pt")
    (rel_dir / "final.sha256").write_text(final_sha + chr(10), encoding="utf-8")
    (rel_dir / "calib.json").write_text(json.dumps(calib_params, indent=2), encoding="utf-8")
    shutil.copy2(ROOT / "configs" / "main.yaml", rel_dir / "main.yaml")
    tracker_cfg = {
        "iou_thr": 0.30,
        "a_max": 60,
        "gaplink_tau": best_tau,
        "c2_theta": th_c2,
        "c3_theta": th_c3,
        "c4_theta": th_c4,
        "c4_lambda_s": ls_c4,
        "c4_lambda_c": lc_c4,
        "final_sha256": final_sha,
    }
    (rel_dir / "tracker.yaml").write_text(yaml.safe_dump(tracker_cfg, sort_keys=False), encoding="utf-8")
    print(f"FROZEN release/final.pt (SHA256={final_sha})")


if __name__ == "__main__":
    main()
