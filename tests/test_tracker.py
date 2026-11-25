from pathlib import Path

import pytest
from src.io_mot import load_seq
from src.inject import make_hide_plan, make_detections
from src.baselines import KalmanMahalanobisBaseline
from src.metrics_tracker import evaluate_tracker_on_seq

ROOT = Path(__file__).resolve().parents[1]  # project root, wherever the repo is unpacked

def test_c1_vs_c3_on_injected_seq():
    mot_dir = ROOT / "data" / "mot"
    if not (mot_dir / "MVI_20011.txt").exists():
        pytest.skip("needs UA-DETRAC in data/mot (scripts.fetch_data + scripts.convert_detrac)")
    seq = load_seq(mot_dir, "MVI_20011")
    plan = make_hide_plan(seq, seed=99)
    dets = make_detections(seq, plan, None, mode="full", seed=99)

    res_c1 = evaluate_tracker_on_seq(seq, plan, dets, associator=None)
    res_c3 = evaluate_tracker_on_seq(seq, plan, dets, associator=KalmanMahalanobisBaseline(theta=25.0))
    assert res_c1["IRR"] < 0.05
    assert res_c3["IRR"] > res_c1["IRR"] + 0.30
