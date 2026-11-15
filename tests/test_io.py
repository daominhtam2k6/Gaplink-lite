import numpy as np
from pathlib import Path
from src.io_mot import center_in_ignored, load_seq

def test_center_in_ignored():
    ig = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)
    boxes = np.array([
        [15.0, 15.0, 4.0, 4.0],   # center (17, 17) -> inside
        [50.0, 50.0, 10.0, 10.0], # center (55, 55) -> outside
    ], dtype=np.float32)
    mask = center_in_ignored(boxes, ig)
    assert mask.tolist() == [True, False]

def test_load_seq_roundtrip(tmp_path: Path):
    mot_file = tmp_path / "SEQ_TEST.txt"
    ig_file = tmp_path / "SEQ_TEST_ignore.txt"
    ig_file.write_text("0,0,10,10\n", encoding="utf-8")
    # frame,id,x,y,w,h,conf,class_id,vis
    mot_file.write_text(
        "2,1,100,200,40,30,1,1,1.0\n"
        "1,1,90,200,40,30,1,1,1.0\n"
        "1,2,300,150,50,40,1,2,0.8\n",
        encoding="utf-8",
    )
    seq = load_seq(tmp_path, "SEQ_TEST")
    assert seq.n_frames == 2
    assert len(seq.rows) == 3
    # Sorted by (frame, id)
    assert seq.rows[:, 0].tolist() == [1.0, 1.0, 2.0]
    assert seq.rows[:, 1].tolist() == [1.0, 2.0, 1.0]
    assert 1 in seq.tracks and 2 in seq.tracks
    assert seq.tracks[1]["frames"].tolist() == [1, 2]
    # Center conversion check: x=90, w=40 -> cx=110
    assert np.isclose(seq.tracks[1]["boxes"][0, 0], 110.0)
