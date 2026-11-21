import numpy as np

from scripts.convert_visdrone import GROUPS, convert_annotation
from scripts.eval_ood_motor import boot_ratio, boot_ratio_diff


def _ann():
    # frame,id,x,y,w,h,score,cat,trunc,occ
    return np.array([
        [1, 1, 10, 10, 20, 30, 1, 10, 0, 0],   # motor, visible
        [1, 2, 50, 50, 40, 30, 1, 4, 0, 1],    # car, partial occl
        [1, 3, 0, 0, 100, 100, 0, 0, 0, 0],    # ignored region
        [2, 1, 12, 11, 20, 30, 1, 10, 0, 2],   # motor, heavy occl
        [2, 1, 12, 11, 20, 30, 1, 10, 0, 2],   # duplicate (frame,id)
        [2, 4, 5, 5, 1, 1, 1, 10, 0, 0],       # too small
        [2, 5, 80, 80, 30, 30, 0, 10, 0, 0],   # score 0 -> not a target
        [2, 6, 90, 90, 60, 40, 1, 9, 0, 0],    # bus
    ], dtype=np.float32)


def test_motor_group():
    rows, ign = convert_annotation(_ann(), GROUPS["motor"])
    assert rows.shape == (2, 9)
    assert set(rows[:, 1].astype(int)) == {1}
    assert np.allclose(rows[:, 8], [1.0, 0.3])
    assert ign.shape == (1, 4)


def test_car_group():
    rows, _ = convert_annotation(_ann(), GROUPS["car"])
    assert sorted(rows[:, 1].astype(int).tolist()) == [2, 6]
    assert np.isclose(rows[rows[:, 1] == 2][0, 8], 0.6)


def test_empty():
    rows, ign = convert_annotation(np.zeros((0, 10), np.float32), GROUPS["motor"])
    assert rows.shape == (0, 9) and ign.shape == (0, 4)


def test_boot():
    num = {"a": 8, "b": 9, "c": 7}
    den = {"a": 10, "b": 10, "c": 10}
    lo, hi = boot_ratio(num, den)
    assert 0.7 <= lo <= 0.8 <= hi <= 0.9
    d = boot_ratio_diff(num, {"a": 5, "b": 5, "c": 5}, den)
    assert np.isclose(d["delta"], 0.3)
    assert d["ci95"][0] > 0
