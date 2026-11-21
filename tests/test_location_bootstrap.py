import numpy as np
import pytest

from src.eval_events import bootstrap_cluster, bootstrap_paired
from src.location_bootstrap import counts_to_arrays, regroup_by_location


def test_regroup_preserves_totals():
    by_seq = {"s1": np.array([1, 0, 1]), "s2": np.array([1, 1]), "s3": np.array([0])}
    s2l = {"s1": "L1", "s2": "L1", "s3": "L2"}
    by_loc = regroup_by_location(by_seq, s2l)
    assert set(by_loc) == {"L1", "L2"}
    assert by_loc["L1"].sum() == 4 and len(by_loc["L1"]) == 5
    assert sum(v.sum() for v in by_loc.values()) == sum(v.sum() for v in by_seq.values())


def test_regroup_missing_location_raises():
    with pytest.raises(KeyError):
        regroup_by_location({"s9": np.array([1])}, {"s1": "L1"})


def test_counts_to_arrays_exact():
    arr = counts_to_arrays({"a": {"IRR": 0.75, "n": 4}}, "IRR", "n")
    assert arr["a"].tolist() == [1, 1, 1, 0]
    with pytest.raises(ValueError):
        counts_to_arrays({"a": {"IRR": 0.7, "n": 4}}, "IRR", "n")


def test_location_ci_wider_when_sequences_correlated():
    rng = np.random.default_rng(0)
    by_seq, s2l = {}, {}
    for loc in range(6):
        p = 0.5 + 0.08 * (loc - 2.5)  # location effect shared by its sequences
        for s in range(5):
            k = f"L{loc}_s{s}"
            by_seq[k] = (rng.random(200) < p).astype(float)
            s2l[k] = f"L{loc}"
    lo_s, hi_s = bootstrap_cluster(by_seq, n_boot=4000)
    lo_l, hi_l = bootstrap_cluster(regroup_by_location(by_seq, s2l), n_boot=4000)
    assert (hi_l - lo_l) > (hi_s - lo_s)


def test_paired_on_locations_point_estimate_unchanged():
    by_seq_a = {"s1": np.array([1, 1, 0]), "s2": np.array([1, 0])}
    by_seq_b = {"s1": np.array([0, 1, 0]), "s2": np.array([0, 0])}
    s2l = {"s1": "L1", "s2": "L2"}
    r_seq = bootstrap_paired(by_seq_a, by_seq_b)
    r_loc = bootstrap_paired(regroup_by_location(by_seq_a, s2l), regroup_by_location(by_seq_b, s2l))
    assert r_seq["delta"] == pytest.approx(r_loc["delta"])
