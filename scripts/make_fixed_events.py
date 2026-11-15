"""Generate fixed val_events.pkl (>=6,000, hard>=1,200) and test_events.pkl (>=12,000, hard>=2,400), record SHA256, and evaluate C-2/C-3/C-4 on val."""
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
        n_hard_target = int(np.ceil(0.25 * quota))
        cnt_s = 0
        hard_s = 0
        for _ in range(quota * 8):
            if cnt_s >= quota:
                break
            want_h = True if hard_s < n_hard_target else None
            ev = sample_event(idx, rng, cfg, want_hard=want_h)
            if ev is not None:
                events.append(ev)
                cnt_s += 1
                if ev["is_hard"]:
                    hard_s += 1
    return events[:total_n]


def precompute_kalman_components(events: list[dict]):
    """Precompute d2 (Mahalanobis^2), size_diff, color_dist, and (mus, sigmas_px) once per event."""
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

    if val_path.exists() and (cache_dir / "val_events.sha256").exists():
        print("Loading existing val_events.pkl...")
        with open(val_path, "rb") as f:
            val_events = pickle.load(f)
        val_sha = sha256_file(val_path)
    else:
        print(f"Generating val_events.pkl across {len(val_seqs)} val sequences...", flush=True)
        val_events = generate_split_events(val_seqs, total_n=6000, seed=202601, mot_dir=mot_dir, hist_dir=hist_dir)
        with open(val_path, "wb") as f:
            pickle.dump(val_events, f, protocol=pickle.HIGHEST_PROTOCOL)
        val_sha = sha256_file(val_path)
        (cache_dir / "val_events.sha256").write_text(val_sha + "\n", encoding="utf-8")

    if test_path.exists() and (cache_dir / "test_events.sha256").exists():
        print("Loading existing test_events.pkl...")
        with open(test_path, "rb") as f:
            test_events = pickle.load(f)
        test_sha = sha256_file(test_path)
    else:
        print(f"Generating test_events.pkl across {len(test_seqs)} test sequences (LOCKED until Day 7)...", flush=True)
        test_events = generate_split_events(test_seqs, total_n=12000, seed=202602, mot_dir=mot_dir, hist_dir=hist_dir)
        with open(test_path, "wb") as f:
            pickle.dump(test_events, f, protocol=pickle.HIGHEST_PROTOCOL)
        test_sha = sha256_file(test_path)
        (cache_dir / "test_events.sha256").write_text(test_sha + "\n", encoding="utf-8")

    n_val_hard = sum(1 for e in val_events if e["is_hard"])
    n_test_hard = sum(1 for e in test_events if e["is_hard"])
    print(f"Val events: {len(val_events)} (hard: {n_val_hard}) | Test events: {len(test_events)} (hard: {n_test_hard})", flush=True)

    # Precompute C-2 costs once
    c2_base = ConstantVelocityBaseline(theta=1.0)
    c2_costs = [c2_base.cost(e) for e in val_events]
    best_c2 = None
    for theta in (1.2, 1.6, 2.0, 2.5, 3.0, 4.0):
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
        if best_c2 is None or res["top1"] > best_c2["metrics"]["top1"]:
            lo, hi = bootstrap_cluster(seq_arr)
            res["top1_ci95"] = [lo, hi]
            best_c2 = {"theta": theta, "metrics": res}

    print("Precomputing Kalman components on val_events...", flush=True)
    comps, gauss = precompute_kalman_components(val_events)

    best_c3 = None
    for theta in (4.61, 6.0, 9.21, 13.8, 18.0, 25.0):
        preds = decide_from_cost_list(comps, theta=theta, ls=0.0, lc=0.0)
        res, seq_arr = evaluate_predictions(val_events, preds, gaussians=gauss)
        if best_c3 is None or res["top1"] > best_c3["metrics"]["top1"]:
            lo, hi = bootstrap_cluster(seq_arr)
            res["top1_ci95"] = [lo, hi]
            best_c3 = {"theta": theta, "metrics": res}

    best_c4 = None
    c3_theta = best_c3["theta"]
    for ls in (0.5, 1.0, 2.0):
        for lc in (0.5, 1.0, 2.0):
            for th in (c3_theta, c3_theta + 2.0, c3_theta + 4.0):
                preds = decide_from_cost_list(comps, theta=th, ls=ls, lc=lc)
                res, seq_arr = evaluate_predictions(val_events, preds)
                if best_c4 is None or res["top1"] > best_c4["metrics"]["top1"]:
                    lo, hi = bootstrap_cluster(seq_arr)
                    res["top1_ci95"] = [lo, hi]
                    best_c4 = {"theta": th, "lambda_s": ls, "lambda_c": lc, "metrics": res}

    summary = {
        "val_count": len(val_events),
        "val_hard_count": n_val_hard,
        "val_sha256": val_sha,
        "test_count": len(test_events),
        "test_hard_count": n_test_hard,
        "test_sha256": test_sha,
        "C-2": best_c2,
        "C-3": best_c3,
        "C-4": best_c4,
    }
    (runs_dir / "baselines_val.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Baselines on val saved to runs/baselines_val.json", flush=True)
    print(f"C-2 top1={best_c2['metrics']['top1']*100:.2f}% | C-3 top1={best_c3['metrics']['top1']*100:.2f}% | C-4 top1={best_c4['metrics']['top1']*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
