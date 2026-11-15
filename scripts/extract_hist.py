"""Extract 128-bin HSV histograms (8x4x4, inner 60% crop, uint8 L1-normalized) aligned with MOT rows."""
import argparse
import json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import cv2
from src.io_mot import load_seq

def compute_box_hsv_hist(bgr_img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    H_im, W_im = bgr_img.shape[:2]
    # Inner 60% box to suppress road background
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    iw = max(2.0, 0.6 * w)
    ih = max(2.0, 0.6 * h)
    x1 = int(np.clip(round(cx - 0.5 * iw), 0, W_im - 1))
    y1 = int(np.clip(round(cy - 0.5 * ih), 0, H_im - 1))
    x2 = int(np.clip(round(cx + 0.5 * iw), x1 + 1, W_im))
    y2 = int(np.clip(round(cy + 0.5 * ih), y1 + 1, H_im))
    crop = bgr_img[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((128,), dtype=np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).reshape(-1)
    s = float(hist.sum())
    if s <= 0:
        return np.zeros((128,), dtype=np.uint8)
    q = np.clip(np.round((hist / s) * 255.0), 0, 255).astype(np.uint8)
    return q

def process_one_seq(seq_name: str) -> tuple[str, int]:
    mot_dir = Path(r"D:\gaplink\data\mot")
    hist_dir = Path(r"D:\gaplink\data\cache\hist")
    hist_dir.mkdir(parents=True, exist_ok=True)
    out_file = hist_dir / f"{seq_name}.npy"

    seq = load_seq(mot_dir, seq_name)
    R = len(seq.rows)
    if out_file.exists():
        arr = np.load(out_file, mmap_mode="r")
        if len(arr) == R:
            return seq_name, R

    out = np.zeros((R, 128), dtype=np.uint8)
    if R == 0 or not seq.img_dir.exists():
        np.save(out_file, out)
        return seq_name, R

    # Build frame -> image path lookup
    jpgs = {}
    for p in seq.img_dir.glob("*.jpg"):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits:
            jpgs[int(digits)] = p

    for fnum, sl in seq.by_frame.items():
        img_p = jpgs.get(fnum)
        if img_p is None:
            continue
        bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        for r_idx in range(sl.start, sl.stop):
            row = seq.rows[r_idx]
            out[r_idx] = compute_box_hsv_hist(bgr, row[2], row[3], row[4], row[5])

    np.save(out_file, out)
    return seq_name, R

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    mot_dir = Path(r"D:\gaplink\data\mot")
    seq_names = sorted([p.stem for p in mot_dir.glob("*.txt") if not p.name.endswith("_ignore.txt")])
    print(f"Extracting HSV histograms for {len(seq_names)} sequences with {args.workers} workers...")
    with Pool(args.workers) as pool:
        for seq_name, n_rows in pool.imap_unordered(process_one_seq, seq_names):
            pass
    print("Finished HSV histogram extraction.")

if __name__ == "__main__":
    main()
