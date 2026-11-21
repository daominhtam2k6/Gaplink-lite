import numpy as np
import pytest

from demo.vms_lite_core import RouteJudge, find_clips, zones_for_clip
from src.security_analytics import PolygonZone

ZA = PolygonZone("Z_A", "Z_A", [(0, 0), (100, 0), (100, 100), (0, 100)], role="entry")


def _gt():
    # vehicle 1 drives through Z_A, vehicle 2 never does
    by_f = {1: (np.array([[50, 50, 20, 10], [500, 300, 20, 10]], np.float32), np.array([1, 2])),
            2: (np.array([[300, 300, 20, 10], [520, 300, 20, 10]], np.float32), np.array([1, 2]))}
    return {"by_frame": by_f}


def test_route_judge_three_classes():
    j = RouteJudge(_gt(), 960, 540)
    assert j.judge({"origin_gt_tid": 1, "dest_gt_tid": 1, "frame": 2}, ZA) == "ok"
    assert j.judge({"origin_gt_tid": 2, "dest_gt_tid": 1, "frame": 2}, ZA) == "wrong_id"
    assert j.judge({"origin_gt_tid": 1, "dest_gt_tid": 2, "frame": 2}, ZA) == "false"
    assert j.judge({"origin_gt_tid": 1, "dest_gt_tid": -1, "frame": 2}, ZA) == "unk"


def test_zones_only_a_and_b():
    meta = {"zones": [{"zone_id": z, "points": [[0, 0], [1, 0], [1, 1]]} for z in ("Z_A", "Z_B", "Z_C", "Z_DWELL")]}
    assert [z.zone_id for z in zones_for_clip(meta, 960, 540)] == ["Z_A", "Z_B"]


def test_clip_runner_end_to_end_with_baselines():
    clips = find_clips()
    if not clips:
        pytest.skip("no demo clips exported")
    import cv2
    from demo.vms_lite_core import ClipRunner
    from src.baselines import KalmanMahalanobisBaseline
    c = clips[0]
    r = ClipRunner(c["path"], c["scenario"], KalmanMahalanobisBaseline(theta=1.0), KalmanMahalanobisBaseline(theta=1.0), "L", "R")
    cap = cv2.VideoCapture(str(c["path"]))
    for f in range(1, 41):
        ok, frm = cap.read()
        if not ok:
            break
        r.step(f, frm)
        img = r.draw(frm, f, show_ids=(f % 2 == 0))
    assert img.shape[1] == 2 * r.W
    m = r.metrics()
    assert set(m) == {"bl", "gl"} and m["bl"] == m["gl"]      # identical trackers -> identical metrics
