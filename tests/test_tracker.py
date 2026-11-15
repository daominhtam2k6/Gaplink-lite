from pathlib import Path
from src.io_mot import load_seq
from src.inject import make_hide_plan, make_detections
from src.baselines import KalmanMahalanobisBaseline
from src.metrics_tracker import evaluate_tracker_on_seq

def test_c1_vs_c3_on_injected_seq():
    mot_dir = Path(r"D:\gaplink\data\mot")
    seq = load_seq(mot_dir, "MVI_20011")
    plan = make_hide_plan(seq, seed=99)
    dets = make_detections(seq, plan, None, mode="full", seed=99)

    res_c1 = evaluate_tracker_on_seq(seq, plan, dets, associator=None)
    res_c3 = evaluate_tracker_on_seq(seq, plan, dets, associator=KalmanMahalanobisBaseline(theta=25.0))
    assert res_c1["IRR"] < 0.05
    assert res_c3["IRR"] > res_c1["IRR"] + 0.30
