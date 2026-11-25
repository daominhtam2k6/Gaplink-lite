"""Day 7 SINGLE TEST EVALUATION:
Verify SHA256 of release/final.pt and data/cache/test_events.pkl,
evaluate C-2, C-3, C-4, and GapLink-Lite on the locked 12,000 test events + 40 test sequences (full & partial injection),
compute paired cluster bootstrap 95% CIs across the 40 test sequences, and extract 10 failure cases.
"""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.dataset import EventListDataset, collate_padded_events
from src.eval_events import evaluate_predictions, bootstrap_cluster, bootstrap_paired
from src.calib import compute_calib_stats, apply_calibration
from src.io_mot import load_seq
from src.inject import make_hide_plan, make_detections
from src.baselines import ConstantVelocityBaseline, KalmanMahalanobisBaseline, kalman_predict_1track
from src.metrics_tracker import evaluate_tracker_on_seq
from src.model.gaplink import GapLinkLite, GapLinkAssociator

ROOT = Path(__file__).resolve().parents[1]


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


def precompute_kalman_test(events: list[dict]):
    comps = []
    gauss_list = []
    for ev in events:
        N = len(ev["t_lost"])
        M = len(ev["cand"])
        cand = ev["cand"]
        mus = np.zeros((N, 4), dtype=np.float32)
        vars_ = np.zeros((N, 4), dtype=np.float32)
        d2 = np.zeros((N, M), dtype=np.float32)
        sz = np.zeros((N, M), dtype=np.float32)
        col = np.zeros((N, M), dtype=np.float32)
        ch = ev.get("cand_hist", None)
        cn = np.linalg.norm(ch, axis=1) if ch is not None else None
        for i in range(N):
            mu, var = kalman_predict_1track(ev["hist"][i], int(ev["hist_len"][i]), int(ev["gap"][i]))
            mus[i] = mu
            vars_[i] = var
            d2[i] = ((cand[:, 0] - mu[0]) ** 2) / var[0] + ((cand[:, 1] - mu[1]) ** 2) / var[1]
            w_i = max(8.0, float(ev["hist"][i, -1, 2]))
            h_i = max(8.0, float(ev["hist"][i, -1, 3]))
            sz[i] = np.abs(np.log(np.maximum(2.0, cand[:, 2]) / w_i)) + np.abs(np.log(np.maximum(2.0, cand[:, 3]) / h_i))
            if ch is not None and "track_color" in ev:
                tc = ev["track_color"][i]
                tn = float(np.linalg.norm(tc))
                cos = np.where((tn > 1e-6) & (cn > 1e-6), (ch @ tc) / np.maximum(tn * cn, 1e-6), 0.0)
                col[i] = 1.0 - np.clip(cos, 0.0, 1.0)
        comps.append((d2, sz, col))
        gauss_list.append((mus, np.sqrt(np.maximum(1.0, vars_))))
    return comps, gauss_list


def decide_from_comps(comps, theta: float, ls: float = 0.0, lc: float = 0.0) -> list[np.ndarray]:
    preds = []
    for d2, sz, col in comps:
        N, M = d2.shape
        out = np.full((N,), -1, dtype=np.int32)
        if N > 0 and M > 0:
            C = d2 + ls * sz + lc * col
            r_ind, c_ind = linear_sum_assignment(C)
            for r, c in zip(r_ind, c_ind):
                if C[r, c] <= theta:
                    out[r] = int(c)
        preds.append(out)
    return preds


def save_failure_cases(test_events: list[dict], preds_gl: list[np.ndarray], preds_c3: list[np.ndarray], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for idx_e, (ev, dec_gl, dec_c3) in enumerate(zip(test_events, preds_gl, preds_c3)):
        lbl = ev["label"]
        if np.any(dec_gl != lbl):
            # Categorize failure type
            wrongs = [i for i in range(len(lbl)) if dec_gl[i] != lbl[i]]
            i0 = wrongs[0]
            gt_j = int(lbl[i0])
            pr_j = int(dec_gl[i0])
            if gt_j >= 0 and pr_j == -1:
                ftype = "false_dustbin_rejection"
            elif gt_j == -1 and pr_j >= 0:
                ftype = "missed_exit_false_link"
            else:
                ftype = "distractor_id_switch"
            cases.append({
                "event_idx": idx_e,
                "seq": ev["seq"],
                "t_conf": int(ev["t_conf"]),
                "gap": int(ev["gap"][i0]),
                "failure_type": ftype,
                "gt_cand": gt_j,
                "pred_cand": pr_j,
                "c3_cand": int(dec_c3[i0]),
                "is_hard": bool(ev.get("is_hard", False)),
            })
            if len(cases) <= 10:
                fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)
                ax.set_xlim(0, ev["W"])
                ax.set_ylim(ev["H"], 0)
                ax.set_facecolor("#1A202C")
                L = int(ev["hist_len"][i0])
                hb = ev["hist"][i0, -L:]
                ax.plot(hb[:, 0], hb[:, 1], "-o", color="#63B3ED", markersize=3, label=f"Lost Trk #{i0} (g={ev['gap'][i0]})")
                for j in range(len(ev["cand"])):
                    cb = ev["cand"][j]
                    col = "#68D391" if j == gt_j else ("#FC8181" if j == pr_j else "#A0AEC0")
                    lw = 2.6 if j in (gt_j, pr_j) else 1.0
                    rect = patches.Rectangle((cb[0] - 0.5 * cb[2], cb[1] - 0.5 * cb[3]), cb[2], cb[3], linewidth=lw, edgecolor=col, facecolor="none")
                    ax.add_patch(rect)
                    ax.text(cb[0], cb[1] - 0.5 * cb[3] - 4, f"C{j}", color=col, fontsize=7, ha="center")
                ax.set_title(f"Failure #{len(cases)}: {ftype} | seq={ev['seq']} | g={ev['gap'][i0]} | GT={gt_j}, Pred={pr_j}", fontsize=8.5)
                ax.legend(loc="upper right", fontsize=7)
                fig.tight_layout()
                fig.savefig(out_dir / f"failure_{len(cases):02d}.png")
                plt.close(fig)
        if len(cases) >= 10:
            break
    (out_dir / "failure_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="release/final.pt")
    args = ap.parse_args()

    rel_dir = ROOT / "release"
    ckpt_path = ROOT / args.ckpt
    expected_sha = (rel_dir / "final.sha256").read_text(encoding="utf-8").strip()
    actual_sha = sha256_file(ckpt_path)
    assert actual_sha == expected_sha, f"SHA256 mismatch on {ckpt_path}: {actual_sha} != {expected_sha}"

    test_events_path = ROOT / "data" / "cache" / "test_events.pkl"
    expected_test_sha = (ROOT / "data" / "cache" / "test_events.sha256").read_text(encoding="utf-8").strip()
    actual_test_sha = sha256_file(test_events_path)
    assert actual_test_sha == expected_test_sha, "SHA256 mismatch on test_events.pkl!"

    print(f"Verified SHA256 for {ckpt_path.name} and test_events.pkl. Running SINGLE test evaluation...")
    with open(test_events_path, "rb") as f:
        test_events = pickle.load(f)

    tracker_cfg = yaml.safe_load((rel_dir / "tracker.yaml").read_text(encoding="utf-8"))
    calib_params = json.loads((rel_dir / "calib.json").read_text(encoding="utf-8"))
    tau = float(tracker_cfg["gaplink_tau"])
    th_c2 = float(tracker_cfg["c2_theta"])
    th_c3 = float(tracker_cfg["c3_theta"])
    th_c4 = float(tracker_cfg["c4_theta"])
    ls_c4 = float(tracker_cfg["c4_lambda_s"])
    lc_c4 = float(tracker_cfg["c4_lambda_c"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = GapLinkLite(
        use_ctx=bool(cfg.get("use_ctx", True)),
        use_color=bool(cfg.get("use_color", True)),
        gap_aware=bool(cfg.get("gap_aware", True)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 1. Event-level evaluation on 12,000 test events
    print("Evaluating C-2, C-3, C-4, and GapLink-Lite on 12,000 test events...")
    c2_model = ConstantVelocityBaseline(theta=th_c2)
    preds_c2 = [c2_model.decide(e) for e in test_events]
    res_c2, seq_c2 = evaluate_predictions(test_events, preds_c2)
    res_c2["top1_ci95"] = list(bootstrap_cluster(seq_c2))

    comps_k, gauss_k = precompute_kalman_test(test_events)
    preds_c3 = decide_from_comps(comps_k, theta=th_c3, ls=0.0, lc=0.0)
    res_c3, seq_c3 = evaluate_predictions(test_events, preds_c3, gaussians=gauss_k)
    res_c3["top1_ci95"] = list(bootstrap_cluster(seq_c3))

    preds_c4 = decide_from_comps(comps_k, theta=th_c4, ls=ls_c4, lc=lc_c4)
    res_c4, seq_c4 = evaluate_predictions(test_events, preds_c4, gaussians=gauss_k)
    res_c4["top1_ci95"] = list(bootstrap_cluster(seq_c4))

    P_gl, raw_gauss_gl = infer_all_events(model, test_events, device)
    cal_gauss_gl = [(pb, apply_calibration(sb, ev["gap"], calib_params)) for ev, (pb, sb) in zip(test_events, raw_gauss_gl)]
    preds_gl = decide_with_tau(test_events, P_gl, tau=tau)
    res_gl, seq_gl = evaluate_predictions(test_events, preds_gl, gaussians=cal_gauss_gl)
    res_gl["top1_ci95"] = list(bootstrap_cluster(seq_gl))

    calib_test_before = compute_calib_stats(test_events, raw_gauss_gl, {"a": 0.0, "b": 0.0, "c": 0.0})
    calib_test_after = compute_calib_stats(test_events, raw_gauss_gl, calib_params)

    # Also evaluate hard-subset paired cluster bootstrap (GapLink-Lite vs C-3 and C-4)
    hard_events = [e for e in test_events if e["is_hard"]]
    hard_idx = [i for i, e in enumerate(test_events) if e["is_hard"]]
    _, seq_c3_hard = evaluate_predictions(hard_events, [preds_c3[i] for i in hard_idx])
    _, seq_c4_hard = evaluate_predictions(hard_events, [preds_c4[i] for i in hard_idx])
    _, seq_gl_hard = evaluate_predictions(hard_events, [preds_gl[i] for i in hard_idx])

    paired_events = {
        "GapLink_vs_C2_all": bootstrap_paired(seq_gl, seq_c2),
        "GapLink_vs_C3_all": bootstrap_paired(seq_gl, seq_c3),
        "GapLink_vs_C4_all": bootstrap_paired(seq_gl, seq_c4),
        "GapLink_vs_C3_hard": bootstrap_paired(seq_gl_hard, seq_c3_hard),
        "GapLink_vs_C4_hard": bootstrap_paired(seq_gl_hard, seq_c4_hard),
    }

    # 2. Tracker-level evaluation across all 40 test sequences (`full` and `partial`)
    with open(ROOT / "data" / "splits.csv", encoding="utf-8") as f:
        test_seqs = [r["seq"] for r in csv.DictReader(f) if r["split"] == "test"]

    assoc_map = {
        "C-1": None,
        "C-2": ConstantVelocityBaseline(theta=th_c2),
        "C-3": KalmanMahalanobisBaseline(theta=th_c3, name="C-3"),
        "C-4": KalmanMahalanobisBaseline(theta=th_c4, lambda_s=ls_c4, lambda_c=lc_c4, name="C-4"),
        "GapLink-Lite": GapLinkAssociator(model, device=str(device), tau=tau, calib=calib_params),
    }

    mot_dir = ROOT / "data" / "mot"
    hist_dir = ROOT / "data" / "cache" / "hist"

    tracker_test = {}
    paired_tracker = {}
    per_seq_records = {"full": {}, "partial": {}}

    for mode in ("full", "partial"):
        print(f"Running online tracker on 40 test sequences (mode={mode})...")
        mode_res = {
            k: {
                "n_intervals": 0,
                "rec": 0,
                "wrong": 0,
                "frag": 0,
                "lost": 0,
                "by_seq": {},
            }
            for k in assoc_map
        }
        for i_s, s_name in enumerate(test_seqs):
            seq = load_seq(mot_dir, s_name)
            hp = hist_dir / f"{s_name}.npy"
            hc = np.load(hp) if hp.exists() else None
            plan = make_hide_plan(seq, seed=7000 + i_s, max_density=0.18)
            dets = make_detections(seq, plan, hc, mode=mode, seed=8000 + i_s)
            per_seq_records[mode][s_name] = {}
            for m_name, assoc in assoc_map.items():
                r_seq = evaluate_tracker_on_seq(seq, plan, dets, associator=assoc)
                n_int = r_seq["n_intervals"]
                mode_res[m_name]["n_intervals"] += n_int
                mode_res[m_name]["rec"] += int(round(r_seq["IRR"] * n_int))
                mode_res[m_name]["wrong"] += int(round(r_seq["WLR"] * n_int))
                mode_res[m_name]["frag"] += int(round(r_seq["FragRate"] * n_int))
                mode_res[m_name]["lost"] += int(round(r_seq["LostRate"] * n_int))
                mode_res[m_name]["by_seq"][s_name] = r_seq["outcomes"]
                per_seq_records[mode][s_name][m_name] = {
                    "n_intervals": n_int,
                    "IRR": r_seq["IRR"],
                    "WLR": r_seq["WLR"],
                    "FragRate": r_seq["FragRate"],
                }

        tracker_test[mode] = {}
        for m_name, d_m in mode_res.items():
            tot = max(1, d_m["n_intervals"])
            lo_i, hi_i = bootstrap_cluster(d_m["by_seq"])
            tracker_test[mode][m_name] = {
                "n_intervals": d_m["n_intervals"],
                "IRR": d_m["rec"] / tot,
                "IRR_ci95": [lo_i, hi_i],
                "WLR": d_m["wrong"] / tot,
                "FragRate": d_m["frag"] / tot,
                "LostRate": d_m["lost"] / tot,
            }
        paired_tracker[mode] = {
            "GapLink_vs_C1": bootstrap_paired(mode_res["GapLink-Lite"]["by_seq"], mode_res["C-1"]["by_seq"]),
            "GapLink_vs_C2": bootstrap_paired(mode_res["GapLink-Lite"]["by_seq"], mode_res["C-2"]["by_seq"]),
            "GapLink_vs_C3": bootstrap_paired(mode_res["GapLink-Lite"]["by_seq"], mode_res["C-3"]["by_seq"]),
            "GapLink_vs_C4": bootstrap_paired(mode_res["GapLink-Lite"]["by_seq"], mode_res["C-4"]["by_seq"]),
        }

    # 3. Evaluate the 3 ablations on test_events for the ablation table
    ablations_test = {}
    for abl_name, abl_dir in [
        ("No-Context", "runs/ablation_ctx"),
        ("No-Color", "runs/ablation_color"),
        ("Gap-Blind", "runs/ablation_gap"),
    ]:
        abl_ckpt_p = ROOT / abl_dir / "best.pt"
        if abl_ckpt_p.exists():
            ack = torch.load(abl_ckpt_p, map_location=device, weights_only=False)
            acfg = ack["cfg"]
            am = GapLinkLite(
                use_ctx=bool(acfg.get("use_ctx", True)),
                use_color=bool(acfg.get("use_color", True)),
                gap_aware=bool(acfg.get("gap_aware", True)),
            ).to(device)
            am.load_state_dict(ack["model"])
            am.eval()
            P_a, g_a = infer_all_events(am, test_events, device)
            preds_a = decide_with_tau(test_events, P_a, tau=tau)
            res_a, seq_a = evaluate_predictions(test_events, preds_a, gaussians=g_a)
            res_a["top1_ci95"] = list(bootstrap_cluster(seq_a))
            ablations_test[abl_name] = res_a

    # 4. Save failure cases and final JSON
    failures = save_failure_cases(test_events, preds_gl, preds_c3, ROOT / "report" / "failure_cases")

    out_dir = ROOT / "runs" / "test_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_results = {
        "final_sha256": actual_sha,
        "test_events_sha256": actual_test_sha,
        "tau": tau,
        "event_level": {
            "C-2": res_c2,
            "C-3": res_c3,
            "C-4": res_c4,
            "GapLink-Lite": res_gl,
        },
        "calib_test": {
            "before": calib_test_before,
            "after": calib_test_after,
            "params": calib_params,
        },
        "paired_bootstrap_events": paired_events,
        "tracker_level": tracker_test,
        "paired_bootstrap_tracker": paired_tracker,
        "ablations_test": ablations_test,
        "per_seq_tracker": per_seq_records,
        "failure_cases": failures,
    }
    (out_dir / "test_results.json").write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    print("SINGLE TEST EVALUATION COMPLETE! Saved to runs/test_final/test_results.json")
    print(f"Event Top-1: C-2={res_c2['top1']*100:.2f}% | C-3={res_c3['top1']*100:.2f}% | C-4={res_c4['top1']*100:.2f}% | GapLink-Lite={res_gl['top1']*100:.2f}%")
    print(f"Hard Event Top-1: C-3={res_c3['top1_hard']*100:.2f}% | C-4={res_c4['top1_hard']*100:.2f}% | GapLink-Lite={res_gl['top1_hard']*100:.2f}%")
    print(f"Tracker Full IRR: C-1={tracker_test['full']['C-1']['IRR']*100:.2f}% | C-3={tracker_test['full']['C-3']['IRR']*100:.2f}% | C-4={tracker_test['full']['C-4']['IRR']*100:.2f}% | GapLink-Lite={tracker_test['full']['GapLink-Lite']['IRR']*100:.2f}%")
    print(f"Tracker Partial IRR: C-1={tracker_test['partial']['C-1']['IRR']*100:.2f}% | C-3={tracker_test['partial']['C-3']['IRR']*100:.2f}% | C-4={tracker_test['partial']['C-4']['IRR']*100:.2f}% | GapLink-Lite={tracker_test['partial']['GapLink-Lite']['IRR']*100:.2f}%")


if __name__ == "__main__":
    main()
