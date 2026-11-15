import numpy as np
from src.calib import fit_gap_calibration, compute_calib_stats

def test_calib_improves_coverage():
    rng = np.random.default_rng(123)
    events = []
    gaussians = []
    # Create 300 synthetic events where raw sigma is 2x too small (underconfident/overconfident)
    for _ in range(300):
        N = 3
        gb = rng.integers(0, 3, size=N, dtype=np.int8)
        gaps = np.where(gb == 0, 3, np.where(gb == 1, 10, 22)).astype(np.int32)
        pred_box = np.zeros((N, 4), dtype=np.float32)
        true_stddev = 4.0 + 0.8 * gaps.astype(np.float32)
        err_x = rng.normal(0.0, true_stddev).astype(np.float32)
        err_y = rng.normal(0.0, true_stddev).astype(np.float32)
        true_box = np.zeros((N, 4), dtype=np.float32)
        true_box[:, 0] = err_x
        true_box[:, 1] = err_y
        # Deliberately underestimate raw sigma by factor 0.45
        raw_sigma = np.ones((N, 4), dtype=np.float32) * (0.45 * true_stddev[:, None])
        events.append({
            "gap": gaps,
            "gap_bin": gb,
            "true_box": true_box,
            "has_true_target": np.ones((N,), dtype=bool),
        })
        gaussians.append((pred_box, raw_sigma))

    before = compute_calib_stats(events[150:], gaussians[150:], {"a": 0.0, "b": 0.0, "c": 0.0})
    params = fit_gap_calibration(events[:150], gaussians[:150], target_cov=0.90)
    after = compute_calib_stats(events[150:], gaussians[150:], params)
    assert before["coverage90"] < 0.65
    assert 0.83 <= after["coverage90"] <= 0.96
    assert after["nll_2d"] < before["nll_2d"]
