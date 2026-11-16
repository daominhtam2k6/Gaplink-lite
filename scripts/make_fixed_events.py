"""Generate fixed val_events.pkl (6,000 events) and test_events.pkl (12,000 events), record SHA256,
and evaluate C-2, C-3, C-4 on val both at unconstrained Top-1 and at matched WLR <= 5.0% ( plus full Top-1 vs WLR curves ).
"""
import csv
import json
import pickle
import hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.io_mot import load_seq
from src.events import EventCfg, SeqIndex, sample_event
from src.baselines import ConstantVelocityBaseline, KalmanMahalanobisBaseline, kalman_predict_1track
from src.eval_events import evaluate_predictions, bootstrap_cluster


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_split_events(seqs: list[str], total_n: int, seed: int, mot_dir: Path, hist_dir: Path) -> list[dict]:
    rng = np.random.default_rng(seed)
    cfg = EventCfg()
    quota = int(np.ceil(total_n / max(1, len(seqs))))
    events = []
    for s in seqs:
        seq = load_seq(mot_dir, s)
        if len(seq.tracks) < 2:
            continue
        hp = hist_dir / f"{s}.npy"
        hc = np.load(hp) if hp.exists() else None
        idx = SeqIndex(seq, hc)
        # Natural sampling (want_hard=None) so hard events (~40-55%) reflect true geometric ambiguity
        cnt_s = 0
        for _ in range(quota * 8):
            if cnt_s >= quota:
                break
            ev = sample_event(idx, rng, cfg, want_hard=None)
            if ev is not None:
                events.append(ev)
                cnt_s += 1
    return events[:total_n]


def precompute_kalman_components(events: list[dict]):
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


def decide_from_cost_list(comps, theta: float, ls: float = 0.0, lc: float = 0.0) -> list[np.ndarray]:
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


def compute_split_diagnostics(events: list[dict]) -> dict:
    gaps = np.concatenate([e["gap"] for e in events])
    n_cands = np.array([len(e["cand"]) for e in events], dtype=np.float32)
    n_trks = np.array([len(e["t_lost"]) for e in events], dtype=np.float32)
    hard_flags = np.array([1.0 if e["is_hard"] else 0.0 for e in events], dtype=np.float32)
    return {
        "n_events": len(events),
        "n_tracks": int(len(gaps)),
        "hard_event_frac": float(hard_flags.mean()),
        "gap_mean": float(gaps.mean()),
        "gap_p50": float(np.percentile(gaps, 50)),
        "gap_p90": float(np.percentile(gaps, 90)),
        "cand_mean": float(n_cands.mean()),
        "trk_mean": float(n_trks.mean()),
    }


def main():
    mot_dir = Path(r"D:\gaplink\data\mot")
    hist_dir = Path(r"D:\gaplink\data\cache\hist")
    cache_dir = Path(r"D:\gaplink\data\cache")
    runs_dir = Path(r"D:\gaplink\runs")
    splits_csv = Path(r"D:\gaplink\data\splits.csv")
    cache_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    val_seqs = [r["seq"] for r in rows if r["split"] == "val"]
    test_seqs = [r["seq"] for r in rows if r["split"] == "test"]

    val_path = cache_dir / "val_events.pkl"
    test_path = cache_dir / "test_events.pkl"

    print(f"Generating fresh val_events.pkl across {len(val_seqs)} val sequences...", flush=True)
    val_events = generate_split_events(val_seqs, total_n=6000, seed=202601, mot_dir=mot_dir, hist_dir=hist_dir)
    with open(val_path, "wb") as f:
        pickle.dump(val_events, f, protocol=pickle.HIGHEST_PROTOCOL)
    val_sha = sha256_file(val_path)
    (cache_dir / "val_events.sha256").write_text(val_sha + chr(10), encoding="utf-8")

    print(f"Generating fresh test_events.pkl across {len(test_seqs)} test sequences...", flush=True)
    test_events = generate_split_events(test_seqs, total_n=12000, seed=202602, mot_dir=mot_dir, hist_dir=hist_dir)
    with open(test_path, "wb") as f:
        pickle.dump(test_events, f, protocol=pickle.HIGHEST_PROTOCOL)
    test_sha = sha256_file(test_path)
    (cache_dir / "test_events.sha256").write_text(test_sha + chr(10), encoding="utf-8")

    val_diag = compute_split_diagnostics(val_events)
    test_diag = compute_split_diagnostics(test_events)
    print("Val diagnostics:", val_diag)
    print("Test diagnostics:", test_diag)

    # Sweep C-2 across wide theta grid to get both unconstrained best Top-1 and WLR<=5% operating point
    c2_base = ConstantVelocityBaseline(theta=1.0)
    c2_costs = [c2_base.cost(e) for e in val_events]
    c2_curve = []
    best_c2 = None
    best_c2_wlr5 = None
    for theta in (0.35, 0.50, 0.65, 0.80, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0):
        preds = []
        for C in c2_costs:
            N, M = C.shape
            out = np.full((N,), -1, dtype=np.int32)
            if N > 0 and M > 0:
                r_ind, c_ind = linear_sum_assignment(C)
                for r, c in zip(r_ind, c_ind):
                    if C[r, c] <= theta:
                        out[r] = int(c)
            preds.append(out)
        res, seq_arr = evaluate_predictions(val_events, preds)
        c2_curve.append({"theta": theta, "top1": res["top1"], "wlr": res["wrong_link_rate"], "dust_F1": res["dustbin_F1"]})
        if best_c2 is None or res["top1"] > best_c2["metrics"]["top1"]:
            lo, hi = bootstrap_cluster(seq_arr)
            res["top1_ci95"] = [lo, hi]
            best_c2 = {"theta": theta, "metrics": res}
        if res["wrong_link_rate"] <= 0.050:
            if best_c2_wlr5 is None or res["top1"] > best_c2_wlr5["metrics"]["top1"]:
                lo, hi = bootstrap_cluster(seq_arr)
                res_cp = dict(res, top1_ci95=[lo, hi])
                best_c2_wlr5 = {"theta": theta, "metrics": res_cp}

    print("Precomputing Kalman components on val_events...", flush=True)
    comps, gauss = precompute_kalman_components(val_events)

    c3_curve = []
    best_c3 = None
    best_c3_wlr5 = None
    for theta in (0.5, 1.0, 1.8, 2.8, 4.61, 6.0, 9.21, 13.8, 18.0, 25.0):
        preds = decide_from_cost_list(comps, theta=theta, ls=0.0, lc=0.0)
        res, seq_arr = evaluate_predictions(val_events, preds, gaussians=gauss)
        c3_curve.append({"theta": theta, "top1": res["top1"], "wlr": res["wrong_link_rate"], "dust_F1": res["dustbin_F1"]})
        if best_c3 is None or res["top1"] > best_c3["metrics"]["top1"]:
            lo, hi = bootstrap_cluster(seq_arr)
            res["top1_ci95"] = [lo, hi]
            best_c3 = {"theta": theta, "metrics": res}
        if res["wrong_link_rate"] <= 0.050:
            if best_c3_wlr5 is None or res["top1"] > best_c3_wlr5["metrics"]["top1"]:
                lo, hi = bootstrap_cluster(seq_arr)
                res_cp = dict(res, top1_ci95=[lo, hi])
                best_c3_wlr5 = {"theta": theta, "metrics": res_cp}

    best_c4 = None
    best_c4_wlr5 = None
    c4_curve = []
    for ls in (0.5, 1.0):
        for lc in (1.0, 2.0):
            for th in (1.2, 2.0, 3.2, 4.8, 7.0, 10.0, 15.0, 22.0, 29.0):
                preds = decide_from_cost_list(comps, theta=th, ls=ls, lc=lc)
                res, seq_arr = evaluate_predictions(val_events, preds)
                if ls == 0.5 and lc == 2.0:
                    c4_curve.append({"theta": th, "top1": res["top1"], "wlr": res["wrong_link_rate"], "dust_F1": res["dustbin_F1"]})
                if best_c4 is None or res["top1"] > best_c4["metrics"]["top1"]:
                    lo, hi = bootstrap_cluster(seq_arr)
                    res["top1_ci95"] = [lo, hi]
                    best_c4 = {"theta": th, "lambda_s": ls, "lambda_c": lc, "metrics": res}
                if res["wrong_link_rate"] <= 0.050:
                    if best_c4_wlr5 is None or res["top1"] > best_c4_wlr5["metrics"]["top1"]:
                        lo, hi = bootstrap_cluster(seq_arr)
                        res_cp = dict(res, top1_ci95=[lo, hi])
                        best_c4_wlr5 = {"theta": th, "lambda_s": ls, "lambda_c": lc, "metrics": res_cp}

    summary = {
        "val_diagnostics": val_diag,
        "test_diagnostics": test_diag,
        "val_sha256": val_sha,
        "test_sha256": test_sha,
        "C-2": best_c2,
        "C-2_wlr5": best_c2_wlr5,
        "C-2_curve": c2_curve,
        "C-3": best_c3,
        "C-3_wlr5": best_c3_wlr5,
        "C-3_curve": c3_curve,
        "C-4": best_c4,
        "C-4_wlr5": best_c4_wlr5,
        "C-4_curve": c4_curve,
    }
    (runs_dir / "baselines_val.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Val Event Top-1 (unconstrained): C-2={best_c2['metrics']['top1']*100:.2f}% | C-3={best_c3['metrics']['top1']*100:.2f}% | C-4={best_c4['metrics']['top1']*100:.2f}%")
    print(f"Val Event Top-1 (WLR<=5%): C-2={best_c2_wlr5['metrics']['top1']*100:.2f}% | C-3={best_c3_wlr5['metrics']['top1']*100:.2f}% | C-4={best_c4_wlr5['metrics']['top1']*100:.2f}%")


if __name__ == "__main__":
    main()
