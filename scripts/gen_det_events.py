"""v1.3: build detector-aware events from YOLOv8n boxes on TRAIN and VAL sequences (never test).

    python -m scripts.gen_det_events            # -> data/cache/det_events_{train,val}.pkl (+ _meta.json)
Resumable: per-sequence pickles in data/cache/det_events_parts/<split>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAP_PER_SEQ = {"train": 3000, "val": 2000}
SEED = 20261121


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text(encoding="utf-8"))
    root = Path(cfg["root_dir"]) if Path(cfg["root_dir"]).exists() else ROOT

    import scripts.eval_live_security as els
    from src.det_events import LoggingTracker, OracleAssociator
    from src.io_mot import load_seq
    from src.live_detector import LiveVehicleDetector

    tcfg = yaml.safe_load((root / "release" / "tracker.yaml").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(root / "data" / "splits.csv", encoding="utf-8")))
    detector = LiveVehicleDetector(imgsz=640, conf_thr=0.28, nms_iou=0.52)
    for split in a.splits:
        assert split in ("train", "val"), "detector-aware events are never built on test"
        seqs = [r["seq"] for r in rows if r["split"] == split]
        els.CACHE_DIR = root / "data" / "cache" / f"yolo_{split}"
        part_dir = root / "data" / "cache" / "det_events_parts" / split
        part_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, s in enumerate(seqs, 1):
            pp = part_dir / f"{s}.pkl"
            if pp.exists():
                continue
            seq = load_seq(root / "data" / "mot", s)
            dets = els.extract_or_load_yolo_seq(seq, detector, batch_size=32)
            det_gt_map, _ = els.match_dets_to_gt_posthoc(seq, dets, iou_match=0.45)
            trk = LoggingTracker(associator=OracleAssociator(), iou_thr=float(tcfg["iou_thr"]), a_max=int(tcfg["a_max"]),
                                 W=seq.W, H=seq.H, seq_name=s, min_hits=3)
            for f in range(1, seq.n_frames + 1):
                trk.step(f, dets[f]["boxes"], dets[f]["hists"], det_gt_map[f])
            evs = trk.logged
            if len(evs) > CAP_PER_SEQ[split]:
                rng = np.random.default_rng([SEED, int.from_bytes(s.encode(), "little") % (2**32)])  # per-seq, resume-safe
                keep = np.sort(rng.choice(len(evs), CAP_PER_SEQ[split], replace=False))
                evs = [evs[k] for k in keep]
            pp.write_bytes(pickle.dumps(evs))
            print(f"[{split} {i:02d}/{len(seqs)}] {s}: {len(trk.logged)} events (kept {len(evs)}) {time.time()-t0:.0f}s", flush=True)
        allev = []
        for s in seqs:
            allev += pickle.loads((part_dir / f"{s}.pkl").read_bytes())
        gaps = np.concatenate([e["gap"] for e in allev]) if allev else np.zeros(0)
        labs = np.concatenate([e["label"] for e in allev]) if allev else np.zeros(0)
        meta = {"split": split, "n_seqs": len(seqs), "n_events": len(allev), "n_tracks": int(len(labs)),
                "frac_dustbin": float(np.mean(labs < 0)) if len(labs) else 0.0,
                "gap_p50": float(np.median(gaps)) if len(gaps) else 0.0,
                "frac_gap_le5": float(np.mean(gaps <= 5)) if len(gaps) else 0.0,
                "frac_hard": float(np.mean([e["is_hard"] for e in allev])) if allev else 0.0,
                "cap_per_seq": CAP_PER_SEQ[split], "seed": SEED}
        out = root / "data" / "cache" / f"det_events_{split}.pkl"
        out.write_bytes(pickle.dumps(allev))
        (root / "data" / "cache" / f"det_events_{split}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
