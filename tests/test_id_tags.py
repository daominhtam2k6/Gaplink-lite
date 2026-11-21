from types import SimpleNamespace

import numpy as np

from demo.render_security_demo import _draw_id_tags, _id_states


def _step(f, items):
    return {"frame": f, "tracks": [{"track_id": t, "gt_tid": g, "box": [100.0 + 50 * i, 100.0, 40.0, 30.0]}
                                   for i, (t, g) in enumerate(items)]}


def _run(seq):
    vms = SimpleNamespace()
    out = None
    for f, items in seq:
        out = _id_states(vms, _step(f, items), f)
    return out


def test_stable_id_is_ok():
    out = _run([(f, [(1, 10)]) for f in range(1, 6)])
    assert out[1] == ("ok", "#1")


def test_switch_detected_after_debounce():
    seq = [(f, [(1, 10)]) for f in range(1, 6)] + [(f, [(1, 20)]) for f in range(6, 9)]
    assert _run(seq)[1][0] == "switch"


def test_single_frame_flicker_ignored():
    seq = [(f, [(1, 10)]) for f in range(1, 6)] + [(6, [(1, 20)]), (7, [(1, 10)])]
    assert _run(seq)[1][0] == "ok"


def test_new_id_after_loss():
    seq = [(f, [(1, 10)]) for f in range(1, 6)] + [(f, []) for f in range(6, 20)] + [(f, [(7, 10)]) for f in range(20, 24)]
    assert _run(seq)[7] == ("newid", "#7 (was #1)")


def test_reset_on_replay():
    vms = SimpleNamespace()
    for f in range(1, 6):
        _id_states(vms, _step(f, [(1, 10)]), f)
    for f in range(6, 9):
        _id_states(vms, _step(f, [(1, 20)]), f)
    out = _id_states(vms, _step(2, [(1, 20)]), 2)
    assert out[1][0] == "ok"


def test_draw_tags_runs():
    img = np.zeros((540, 960, 3), np.uint8)
    st = _step(1, [(1, 10), (2, 11)])
    _draw_id_tags(img, st, {1: ("switch", "#1 SWITCH"), 2: ("ok", "#2")})
    assert img.sum() > 0
