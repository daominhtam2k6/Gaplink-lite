import numpy as np
import pytest

from src.det_events import OracleAssociator, label_from_gt


def test_label_basic():
    lab, cl = label_from_gt(np.array([5, 7, 9]), np.array([7, -1, 5]))
    assert lab.tolist() == [2, 0, -1]
    assert cl.tolist() == [1, -1, 0]


def test_label_unknown_track_and_duplicates():
    lab, cl = label_from_gt(np.array([-1, 3]), np.array([3, 3]))
    assert lab.tolist() == [-1, 0]
    assert cl.tolist() == [1, -1]


def test_oracle_returns_copy():
    ev = {"t_lost": np.zeros(2), "_oracle_label": np.array([1, -1])}
    d = OracleAssociator().decide(ev)
    d[0] = 9
    assert ev["_oracle_label"][0] == 1


def test_oracle_without_label_is_dustbin():
    assert OracleAssociator().decide({"t_lost": np.zeros(3)}).tolist() == [-1, -1, -1]


def test_logging_tracker_records_labelled_flicker_event():
    from src.det_events import LoggingTracker
    trk = LoggingTracker(associator=OracleAssociator(), iou_thr=0.3, a_max=95, W=960, H=540, seq_name="syn", min_hits=3)
    hist = np.full((128,), 1.0 / 128, dtype=np.float32)
    for f in range(1, 21):
        boxes, gts = [], []
        # vehicle 1 moves right 6 px/frame, missing (detector flicker) in frames 9..11
        if not (9 <= f <= 11):
            boxes.append([100 + 6 * f, 200, 40, 30]); gts.append(1)
        # vehicle 2 static far away
        boxes.append([700, 400, 50, 40]); gts.append(2)
        b = np.array(boxes, dtype=np.float32)
        trk.step(f, b, np.tile(hist, (len(b), 1)), np.array(gts, dtype=np.int32))
    assert trk.logged, "no event logged"
    ev = trk.logged[0]
    assert ev["t_conf"] == 12 and int(ev["gap"][0]) == 4
    j = int(ev["label"][0])
    assert j >= 0 and np.allclose(ev["cand"][j][:2], [100 + 6 * 12, 200])
    assert ev["has_true_target"][0] and np.allclose(ev["true_box"][0], ev["cand"][j])
    # oracle relinked: the vehicle keeps a single identity
    assert len(trk.reassoc_log) == 1 and trk.reassoc_log[0]["correct"]
