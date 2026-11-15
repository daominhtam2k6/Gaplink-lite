import numpy as np
from src.eval_events import bootstrap_cluster, bootstrap_paired

def test_bootstrap_paired_zero_diff():
    a = {"s1": np.array([1.0, 0.0, 1.0]), "s2": np.array([0.0, 1.0]), "s3": np.array([1.0, 1.0])}
    b = {"s1": np.array([0.0, 0.0, 1.0]), "s2": np.array([0.0, 0.0]), "s3": np.array([1.0, 0.0])}
    res = bootstrap_paired(a, b, n_boot=500, seed=42)
    assert res["delta"] > 0.0
    assert res["ci95_lo"] <= res["delta"] <= res["ci95_hi"]
    lo, hi = bootstrap_cluster(a, n_boot=500, seed=42)
    assert lo <= res["mean_a"] <= hi
