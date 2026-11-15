"""Generate pre-sampled training event pool (easy + hard) using multiprocessing."""
import argparse
import csv
import json
import pickle
import hashlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from src.io_mot import load_seq
from src.events import EventCfg, SeqIndex, sample_event

def _worker_sample(args):
    seq_name, n_easy, n_hard, seed = args
    mot_dir = Path(r"D:\gaplink\data\mot")
    hist_dir = Path(r"D:\gaplink\data\cache\hist")
    seq = load_seq(mot_dir, seq_name)
    if len(seq.tracks) < 2:
        return [], []
    hp = hist_dir / f"{seq_name}.npy"
    hc = np.load(hp) if hp.exists() else None
    idx = SeqIndex(seq, hc)
    rng = np.random.default_rng(seed)
    cfg = EventCfg()

    easy_list = []
    hard_list = []
    for _ in range(n_hard * 10):
        if len(hard_list) >= n_hard:
            break
        ev = sample_event(idx, rng, cfg, want_hard=True)
        if ev is not None:
            hard_list.append(ev)

    for _ in range(n_easy * 6):
        if len(easy_list) >= n_easy:
            break
        ev = sample_event(idx, rng, cfg, want_hard=False)
        if ev is not None:
            easy_list.append(ev)

    return easy_list, hard_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=160000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    splits_csv = Path(r"D:\gaplink\data\splits.csv")
    pool_dir = Path(r"D:\gaplink\data\cache\pool\train")
    pool_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_csv, encoding="utf-8") as f:
        train_seqs = [r["seq"] for r in csv.DictReader(f) if r["split"] == "train"]

    per_seq = int(np.ceil(args.n / max(1, len(train_seqs))))
    n_hard_seq = int(round(0.45 * per_seq))
    n_easy_seq = per_seq - n_hard_seq

    tasks = [(s, n_easy_seq, n_hard_seq, 42000 + i) for i, s in enumerate(train_seqs)]
    easy_all = []
    hard_all = []
    print(f"Generating training pool (~{args.n} events) across {len(train_seqs)} train sequences with {args.workers} workers...")
    with Pool(args.workers) as pool:
        for e_part, h_part in pool.imap_unordered(_worker_sample, tasks):
            easy_all.extend(e_part)
            hard_all.extend(h_part)

    easy_file = pool_dir / "easy_pool.pkl"
    hard_file = pool_dir / "hard_pool.pkl"
    with open(easy_file, "wb") as f:
        pickle.dump(easy_all, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(hard_file, "wb") as f:
        pickle.dump(hard_all, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "n_easy": len(easy_all),
        "n_hard": len(hard_all),
        "total": len(easy_all) + len(hard_all),
        "n_train_seqs": len(train_seqs),
    }
    (pool_dir / "pool_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Pool generated:", json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
