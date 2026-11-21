"""5-view HSV histograms for the VisDrone OOD sequences (same features as scripts/extract_hist.py)."""
import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np

from scripts.convert_visdrone import load_root


def process_one(seq_name: str, mot_dir: str, hist_dir: str) -> tuple[str, int]:
    import cv2
    from scripts.extract_hist import compute_box_hsv_hist_from_hsv
    from src.events import partial_crop
    from src.io_mot import load_seq

    out_file = Path(hist_dir) / f"{seq_name}.npy"
    seq = load_seq(Path(mot_dir), seq_name)
    R = len(seq.rows)
    if out_file.exists():
        try:
            if np.load(out_file, mmap_mode="r").shape == (R, 5, 128):
                return seq_name, R
        except Exception:
            pass
    out = np.zeros((R, 5, 128), dtype=np.uint8)
    jpgs = {}
    for p in seq.img_dir.glob("*.jpg"):
        d = "".join(ch for ch in p.stem if ch.isdigit())
        if d:
            jpgs[int(d)] = p
    for fnum, sl in seq.by_frame.items():
        p = jpgs.get(fnum)
        if p is None:
            continue
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        for r in range(sl.start, sl.stop):
            x, y, w, h = (float(v) for v in seq.rows[r, 2:6])
            full = np.array([x + 0.5 * w, y + 0.5 * h, w, h], dtype=np.float32)
            out[r, 0] = compute_box_hsv_hist_from_hsv(hsv, *full)
            for side in (1, 2, 3, 4):
                cb = partial_crop(full, side, 0.72)
                out[r, side] = compute_box_hsv_hist_from_hsv(hsv, float(cb[0]), float(cb[1]), float(cb[2]), float(cb[3]))
    tmp = Path(hist_dir) / f"{seq_name}_tmp.npy"
    np.save(tmp, out)
    tmp.replace(out_file)
    return seq_name, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["motor", "car"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    base = load_root() / "data" / "ood_visdrone" / args.group
    mot_dir, hist_dir = base / "mot", base / "hist"
    hist_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in mot_dir.glob("*.txt") if not p.name.endswith("_ignore.txt"))
    fn = partial(process_one, mot_dir=str(mot_dir), hist_dir=str(hist_dir))
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (s, R) in enumerate(pool.imap_unordered(fn, names), 1):
            print(f"[{args.group}] {i}/{len(names)} {s} rows={R}", flush=True)


if __name__ == "__main__":
    main()
