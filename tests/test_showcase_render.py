import numpy as np

from demo.render_security_demo import EASE, ZOOM_MAX, _compose, _window_state, _zoom_at


def test_window_state_eases_in_and_out():
    wins = [(100, 170, 0.3, 0.6)]
    assert _window_state(99, wins) == (False, 1.0, 0.5, 0.5)
    slow, z, cx, cy = _window_state(100, wins)
    assert slow and z == 1.0 and (cx, cy) == (0.3, 0.6)
    assert _window_state(100 + EASE, wins)[1] == ZOOM_MAX
    assert _window_state(135, wins)[1] == ZOOM_MAX


def test_zoom_keeps_size_and_identity_at_one():
    img = np.random.randint(0, 255, (54, 96, 3), np.uint8)
    assert _zoom_at(img, 0.5, 0.5, 1.0) is img
    z = _zoom_at(img, 0.9, 0.1, 2.0)
    assert z.shape == img.shape


def test_compose_is_1080p():
    a = np.zeros((540, 960, 3), np.uint8)
    out = _compose(a, a, True, "route")
    assert out.shape == (1080, 1920, 3)
