"""Re-group per-sequence statistics into camera-location clusters for bootstrap CIs.

Sequences from the same location are correlated, so the resampling unit must be the
location (data/splits.csv:location), not the sequence. These helpers only re-aggregate
existing per-sequence arrays; they never touch predictions or thresholds.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_seq_to_location(splits_csv: str | Path, split: str | None = None) -> dict[str, str]:
    with open(splits_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["seq"]: r["location"] for r in rows if split is None or r["split"] == split}


def regroup_by_location(values_by_seq: dict[str, np.ndarray], seq2loc: dict[str, str]) -> dict[str, np.ndarray]:
    """Concatenate per-sequence 0/1 arrays into per-location arrays (sorted, deterministic)."""
    missing = sorted(set(values_by_seq) - set(seq2loc))
    if missing:
        raise KeyError(f"sequences without location: {missing[:5]}")
    out: dict[str, list[np.ndarray]] = {}
    for seq in sorted(values_by_seq):
        out.setdefault(seq2loc[seq], []).append(np.asarray(values_by_seq[seq], dtype=np.float64))
    return {loc: np.concatenate(v) for loc, v in sorted(out.items())}


def counts_to_arrays(per_seq: dict[str, dict], num_key: str, den_key: str) -> dict[str, np.ndarray]:
    """Turn {seq: {rate, n}} records (e.g. IRR, n_intervals) back into 0/1 arrays.

    The count is recovered as round(rate * n); raises if the rate is not an exact k/n.
    """
    arrs = {}
    for seq, rec in per_seq.items():
        n = int(rec[den_key])
        k = int(round(float(rec[num_key]) * n))
        if n > 0 and abs(k / n - float(rec[num_key])) > 1e-9:
            raise ValueError(f"{seq}: {num_key}={rec[num_key]} is not k/{n}")
        a = np.zeros(n, dtype=np.float64)
        a[:k] = 1.0
        arrs[seq] = a
    return arrs
