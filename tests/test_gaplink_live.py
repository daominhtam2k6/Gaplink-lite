"""GapLink-only live console: drawing layer, 25 fps resampling, upload -> processing -> playback (no ground truth)."""
import hashlib
import json
import time

import cv2
import numpy as np
import pytest

from scripts.prepare_video import pack_dets, prepare, resample_keep

W, H = 320, 180


def _toy(d, n=60, gap=(25, 37), fps=25.0, name="toy"):
    """A car (h=50 px) driving left to right, missing for frames gap[0]..gap[1]-1, plus a parked car."""
    vw = cv2.VideoWriter(str(d / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    rng = np.random.default_rng(0)
    hc, hp = rng.dirichlet(np.ones(128)).astype(np.float32), rng.dirichlet(np.ones(128)).astype(np.float32)
    per = []
    for f in range(1, n + 1):
        img = np.full((H, W, 3), 90, np.uint8)
        boxes, hists = [[260, 130, 40, 30]], [hp]
        if not gap[0] <= f < gap[1]:
            boxes.append([50 + 3 * f, 80, 60, 50])
            hists.append(hc)
            cv2.rectangle(img, (25 + 3 * f, 60), (75 + 3 * f, 100), (40, 40, 200), -1)
        vw.write(img)
        per.append({"boxes": np.array(boxes, np.float32), "hists": np.array(hists, np.float32)})
    vw.release()
    np.savez_compressed(d / f"{name}_dets.npz", **pack_dets(per))
    (d / f"{name}_meta.json").write_text(json.dumps({"scenario": "relink", "no_gt": True}), encoding="utf-8")
    return d / f"{name}.mp4", per


def _assoc():
    from src.baselines import KalmanMahalanobisBaseline
    return KalmanMahalanobisBaseline(theta=9.21)        # stands in for GapLink-Lite: same Stage-2 interface


def test_resample_keep():
    assert sum(resample_keep(100, 50.0, 25.0)) == 50
    assert all(resample_keep(100, 25.0, 25.0))
    assert abs(sum(resample_keep(300, 30.0, 25.0)) - 250) <= 1
    assert all(resample_keep(10, 60.0, None))


def test_live_runner_keeps_id_and_draws_bridge(tmp_path):
    from demo.gaplink_live_core import LiveRunner, id_colour
    p, _ = _toy(tmp_path)
    r = LiveRunner(p, "relink", _assoc(), "test")
    cap = cv2.VideoCapture(str(p))
    hidden_seen, img_on = False, None
    for f in range(1, 61):
        ok, frm = cap.read()
        r.step(f, frm)
        if 30 <= f < 37:
            hidden_seen |= any(h["gap"] >= 3 for h in r.hidden)
        if f == 40:
            off = r.draw(frm, f, overlay=False)
            assert np.array_equal(off, frm)                  # key O: the untouched frame
            img_on = r.draw(frm, f)
    assert hidden_seen
    assert len(r.links) == 1 and r.links[0]["gap"] >= 8
    tid = r.links[0]["tid"]
    assert any(p[3] == "bridge" for p in r.trails[tid])
    assert not np.array_equal(img_on, off)
    st = r.stats()
    assert st["links"] == 1 and st["longest"] == r.links[0]["gap"]
    assert [e["kind"] for e in r.events()] == ["link"] and r.history()[0]["kind"] == "link"
    assert id_colour(3) == id_colour(3) and id_colour(3) != id_colour(4)


def test_live_runner_lock_any_mode(tmp_path):
    from demo.gaplink_live_core import LiveRunner
    p, _ = _toy(tmp_path)
    r = LiveRunner(p, "route", _assoc(), "test")
    cap = cv2.VideoCapture(str(p))
    for f in range(1, 61):
        ok, frm = cap.read()
        r.step(f, frm)
        if f == 15:
            assert r.lock_click(50 + 45, 80, f, frm) is not None
    lm = r.stats()["lock"]
    assert lm["reacquired"] == 1 and lm["hidden_frames"] >= 8


class _FakeDet:
    def detect_batch(self, frames):
        return [{"boxes": np.array([[100, 90, 50, 40]], np.float32), "hists": np.full((1, 128), 1 / 128, np.float32)}
                for _ in frames]


def test_prepare_resamples_to_25fps(tmp_path):
    p, _ = _toy(tmp_path, n=100, fps=50.0, name="fast")
    calls = []
    info = prepare(p, "fast_out", out_dir=tmp_path / "out", detector=_FakeDet(), progress=lambda d, t: calls.append((d, t)))
    assert info["frames"] == 50 and info["fps"] == 25.0
    cap = cv2.VideoCapture(str(tmp_path / "out" / "fast_out.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 50 and abs(cap.get(cv2.CAP_PROP_FPS) - 25.0) < 0.01
    assert calls[-1] == (50, 50)
    meta = json.loads((tmp_path / "out" / "fast_out_meta.json").read_text())
    assert meta["source"]["fps_original"] == 50.0


def test_upload_process_play(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import demo.gaplink_live as gl
    monkeypatch.setattr(gl, "UPLOAD_DIR", tmp_path / "up")
    monkeypatch.setattr(gl, "LIVE_DIR", tmp_path / "live")
    monkeypatch.setattr(gl, "make_gaplink", lambda: (_assoc(), "test-model"))
    monkeypatch.setattr(gl.Session, "detector", lambda self: _FakeDet())
    monkeypatch.setattr(gl, "SESSION", None)
    monkeypatch.setattr(gl, "FRESH", True)
    p, _ = _toy(tmp_path, n=40, name="clip")
    data = p.read_bytes()
    c = TestClient(gl.build_app())
    assert c.post("/api/upload?name=notes.txt", content=b"x").status_code == 400
    r = c.post("/api/upload?name=my clip.mp4", content=data)
    assert r.status_code == 200 and r.json()["job"]["sha256"] == hashlib.sha256(data).hexdigest()
    s = gl.sess()
    for _ in range(100):
        if s.jobs[-1]["state"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert s.jobs[-1]["state"] == "done", s.jobs[-1]
    assert len(s.videos) == 1 and s.cur["sha256"] == hashlib.sha256(data).hexdigest() and s.cur["file"] == "my clip.mp4"
    pkt = s.tick()
    assert pkt["img"] and pkt["cur"]["file"] == "my clip.mp4" and pkt["total"] == 40
    c.post("/api/act", json={"action": "seek", "value": 10})          # tracks are confirmed after 3 hits
    assert c.post("/api/act", json={"action": "lock", "x": 100 / W, "y": 90 / H}).json()["tid"] is not None
    assert c.post("/api/act", json={"action": "opt", "key": "overlay", "flag": False}).json()["ok"]
    assert s.opts["overlay"] is False
    assert c.post("/api/act", json={"action": "mode", "mode": "route"}).json()["ok"] and s.runner.mode == "route"
