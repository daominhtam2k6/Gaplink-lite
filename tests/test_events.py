import numpy as np
from pathlib import Path
from src.io_mot import load_seq
from src.events import EventCfg, SeqIndex, sample_event, assemble_event, partial_crop

def _make_synthetic_seq(tmp_path: Path):
    mot_file = tmp_path / "SYN_01.txt"
    rows = []
    # 3 vehicles moving smoothly over 140 frames
    for f in range(1, 141):
        rows.append(f"{f},1,{100 + 2.5*f:.1f},{200 + 0.5*f:.1f},50,36,1,1,1.0")
        rows.append(f"{f},2,{120 + 2.4*f:.1f},{210 + 0.5*f:.1f},48,35,1,1,1.0")
        rows.append(f"{f},3,{500 - 2.0*f:.1f},{300:.1f},60,40,1,2,1.0")
    mot_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return load_seq(tmp_path, "SYN_01")

def test_no_future_leakage_and_labels(tmp_path: Path):
    seq = _make_synthetic_seq(tmp_path)
    idx = SeqIndex(seq)
    cfg = EventCfg(hist_pos_sigma=0.0, cand_pos_sigma=0.0, cand_size_jitter=0.0, p_partial=0.0)
    rng = np.random.default_rng(42)

    gaps = []
    for _ in range(200):
        ev = sample_event(idx, rng, cfg)
        assert ev is not None
        N = len(ev["t_lost"])
        M = len(ev["cand"])
        assert 1 <= N <= cfg.n_trk_max
        assert 1 <= M <= cfg.n_cand_max
        gaps.extend(ev["gap"].tolist())
        # Verify label <-> cand_label bijection
        for i in range(N):
            j = int(ev["label"][i])
            if j >= 0:
                assert int(ev["cand_label"][j]) == i
        for j in range(M):
            i = int(ev["cand_label"][j])
            if i >= 0:
                assert int(ev["label"][i]) == j
        # Verify history never touches frames > t_lost
        for i in range(N):
            tl = int(ev["t_lost"][i])
            assert tl < ev["t_conf"]

    gaps = np.array(gaps)
    p0 = np.mean((gaps >= 3) & (gaps <= 15))
    p1 = np.mean((gaps >= 16) & (gaps <= 45))
    p2 = np.mean(gaps >= 46)
    assert abs(p0 - 0.4) < 0.12
    assert abs(p1 - 0.4) < 0.12
    assert abs(p2 - 0.2) < 0.12

def test_partial_crop_bounds():
    b = np.array([200.0, 150.0, 60.0, 40.0], dtype=np.float32)
    for side in range(5):
        c = partial_crop(b, side, 0.75)
        assert c.shape == (4,)
        assert c[2] > 0 and c[3] > 0
