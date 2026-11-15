import numpy as np
from pathlib import Path
from src.io_mot import load_seq
from src.events import EventCfg, SeqIndex, sample_event, assemble_event

def test_assemble_event_parity(tmp_path: Path):
    mot_file = tmp_path / "SYN_02.txt"
    rows = [f"{f},1,{80 + 2*f},{180 + f},44,32,1,1,1.0\n{f},2,{200 + f},{220},50,36,1,1,1.0" for f in range(1, 120)]
    mot_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    seq = load_seq(tmp_path, "SYN_02")
    idx = SeqIndex(seq)
    rng = np.random.default_rng(7)
    ev = sample_event(idx, rng, EventCfg())
    assert ev is not None

    rebuilt = assemble_event(
        seq_name=ev["seq"],
        t_lost=ev["t_lost"],
        t_conf=ev["t_conf"],
        hist=ev["hist"],
        hist_len=ev["hist_len"],
        track_color=ev["track_color"],
        cand=ev["cand"],
        cand_hist=ev["cand_hist"],
        ctx_lost=ev["ctx_lost"],
        ctx_conf=ev["ctx_conf"],
        ctx_mask_lost=ev["ctx_mask_lost"],
        ctx_mask_conf=ev["ctx_mask_conf"],
        W=ev["W"],
        H=ev["H"],
    )
    for k, v in rebuilt.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, ev[k]), f"Mismatch on key {k}"
        else:
            assert v == ev[k]
