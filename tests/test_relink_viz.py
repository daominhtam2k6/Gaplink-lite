from types import SimpleNamespace

import numpy as np

from demo.relink_viz import GHOST_MIN_AGE, MIN_GAP, ExplainedAssociator, RelinkRecorder, draw_relink_layer, relink_summary


class _FakeGL:
    tau = 0.5

    def infer_event(self, ev):
        P = np.array([[0.7, 0.1, 0.2], [0.2, 0.3, 0.5], [0.0, 0.0, 0.0]], np.float32)  # rows: 2 lost + pad
        return {"P": P, "pred_box": np.array([[10, 10, 5, 5], [20, 20, 5, 5]], np.float32),
                "sigma_px": np.ones((2, 4), np.float32) * 3}


def _ev(N=2, M=2):
    return {"t_lost": np.zeros(N), "cand": np.zeros((M, 4), np.float32), "gap": np.full(N, 10)}


def test_explained_gaplink_same_rule():
    ex = ExplainedAssociator(_FakeGL())
    dec = ex.decide(_ev())
    assert dec.tolist() == [0, -1]          # row 2: 0.3 < tau -> wait (dustbin)
    assert ex.last["kind"] == "gaplink" and ex.last["mu"].shape == (2, 2)


def _trk(tid, box, last_frame, gt, hits=5):
    return SimpleNamespace(track_id=tid, boxes=[np.array(box, np.float32)], last_frame=last_frame, last_gt_tid=gt, hits=hits)


def _tracker(lost=(), log=()):
    return SimpleNamespace(lost_tracks=list(lost), reassoc_log=list(log), W=960, H=540)


def _out(*items):
    return {"tracks": [{"track_id": t, "gt_tid": g, "box": np.array([400, 300, 40, 30], np.float32)} for t, g in items]}


def test_relink_ok_and_wrong():
    rec = RelinkRecorder(None)
    rec.update(_tracker(), _out((1, 10), (2, 20)), 1)
    rec.update(_tracker(lost=[_trk(1, [400, 300, 40, 30], 1, 10), _trk(2, [500, 300, 40, 30], 1, 20)]), _out(), 5)
    log = [{"track_id": 1, "gap": MIN_GAP + 2, "gt_tid_before": 10, "gt_tid_after": 10},
           {"track_id": 2, "gap": MIN_GAP + 2, "gt_tid_before": 20, "gt_tid_after": 99}]
    rec.update(_tracker(log=log), _out((1, 10), (2, 99)), 1 + MIN_GAP + 2)
    c = relink_summary(rec)
    assert c["ok"] == 1 and c["wrong"] == 1
    assert rec.flash[1]["verdict"] == "ok" and rec.flash[2]["verdict"] == "wrong"


def test_short_gap_not_judged():
    rec = RelinkRecorder(None)
    rec.update(_tracker(), _out((1, 10)), 1)
    rec.update(_tracker(log=[{"track_id": 1, "gap": 2, "gt_tid_before": 10, "gt_tid_after": 11}]), _out((1, 11)), 3)
    assert relink_summary(rec)["wrong"] == 0


def test_missed_new_id():
    rec = RelinkRecorder(None)
    rec.update(_tracker(), _out((1, 10)), 1)
    lost = [_trk(1, [400, 300, 40, 30], 1, 10)]
    rec.update(_tracker(lost=lost), _out(), 2)
    rec.update(_tracker(lost=lost), _out((7, 10)), 1 + MIN_GAP + 1)
    assert relink_summary(rec)["missed"] == 1
    assert rec.flash[7]["was"] == 1


def test_draw_runs():
    rec = RelinkRecorder(None)
    f = 1 + GHOST_MIN_AGE
    rec.update(_tracker(), _out((1, 10)), 1)
    rec.update(_tracker(lost=[_trk(1, [400, 300, 40, 30], 1, 10)]), _out(), f)
    rec.pred[1] = (f, np.array([430, 300], np.float32), np.array([8, 6], np.float32))
    img = np.zeros((540, 960, 3), np.uint8)
    draw_relink_layer(img, rec, _out(), f, is_gaplink=True)
    assert img.sum() > 0
