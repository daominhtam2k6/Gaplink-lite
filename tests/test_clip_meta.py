import json

from demo.clip_meta import load_clip_meta, slow_windows, zones_from_meta


def test_roundtrip_and_windows(tmp_path):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"")
    meta = {"scenario": "route", "zones": [{"zone_id": "Z_A", "role": "entry", "points": [[0, 0], [0.5, 0], [0.5, 0.5]]}],
            "key_moments": [{"frame": 100, "cx": 0.3, "cy": 0.6}], "lock": [[10, 50.0, 60.0]]}
    (tmp_path / "clip_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    m = load_clip_meta(v)
    assert m["scenario"] == "route"
    zones, forb, dw = zones_from_meta(m)
    assert zones[0].zone_id == "Z_A" and zones[0].normalized and forb == [("Z_A", "Z_B")] and dw == 5.0
    assert slow_windows(m) == [(70, 140, 0.3, 0.6)]


def test_missing_meta(tmp_path):
    assert load_clip_meta(tmp_path / "none.mp4") is None and slow_windows(None) == []
