from pathlib import Path

import pytest
from src.io_mot import load_seq
from src.inject import make_hide_plan, make_detections

ROOT = Path(__file__).resolve().parents[1]  # project root, wherever the repo is unpacked

def test_inject_density_and_determinism():
    mot_dir = ROOT / "data" / "mot"
    if not (mot_dir / "MVI_20011.txt").exists():
        pytest.skip("needs UA-DETRAC in data/mot (scripts.fetch_data + scripts.convert_detrac)")
    seq = load_seq(mot_dir, "MVI_20011")
    plan1 = make_hide_plan(seq, seed=77, max_density=0.18)
    plan2 = make_hide_plan(seq, seed=77, max_density=0.18)
    assert len(plan1) > 0
    assert [p["tid"] for p in plan1] == [p["tid"] for p in plan2]
    assert [p["t_start"] for p in plan1] == [p["t_start"] for p in plan2]

    dets_full = make_detections(seq, plan1, None, mode="full", seed=77)
    dets_part = make_detections(seq, plan1, None, mode="partial", seed=77)
    assert len(dets_full) == seq.n_frames
    assert len(dets_part) == seq.n_frames
    first = plan1[0]
    for f in range(first["t_start"], first["t_end"] + 1):
        assert first["tid"] not in dets_full[f]["gt_tids"]
    assert first["tid"] in dets_full[first["t_reappear"]]["gt_tids"]
