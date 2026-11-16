"""Fast 5-view 128-bin HSV histogram extraction: converts each image to HSV once per frame, then slices 5 crops per box."""
import argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import cv2
from src.io_mot import load_seq
from src.events import partial_crop


def compute_box_hsv_hist_from_hsv(hsv_img: np.ndarray, cx: float, cy: float, w: float, h: float) -> np.ndarray:
    H_im, W_im = hsv_img.shape[:2]
    iw = max(2.0, 0.6 * w)
    ih = max(2.0, 0.6 * h)
    x1 = int(np.clip(round(cx - 0.5 * iw), 0, W_im - 1))
    y1 = int(np.clip(round(cy - 0.5 * ih), 0, H_im - 1))
    x2 = int(np.clip(round(cx + 0.5 * iw), x1 + 1, W_im))
    y2 = int(np.clip(round(cy + 0.5 * ih), y1 + 1, H_im))
    crop = hsv_img[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((128,), dtype=np.uint8)
    hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).reshape(-1)
    s = float(hist.sum())
    if s <= 0:
        return np.zeros((128,), dtype=np.uint8)
    return np.clip(np.round((hist / s) * 255.0), 0, 255).astype(np.uint8)


def process_one_seq(seq_name: str) -> tuple[str, int]:
    mot_dir = Path(r"D:\gaplink\data\mot")
    hist_dir = Path(r"D:\gaplink\data\cache\hist")
    hist_dir.mkdir(parents=True, exist_ok=True)
    out_file = hist_dir / f"{seq_name}.npy"

    seq = load_seq(mot_dir, seq_name)
    R = len(seq.rows)
    if out_file.exists():
        try:
            arr = np.load(out_file)
            if arr.shape == (R, 5, 128):
                return seq_name, R
        except Exception:
            pass
        out_file.unlink(missing_ok=True)

    out = np.zeros((R, 5, 128), dtype=np.uint8)
    if R == 0 or not seq.img_dir.exists():
        np.save(out_file, out)
        return seq_name, R

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
        hsv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        for r_idx in range(sl.start, sl.stop):
            row = seq.rows[r_idx]
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            full_cxcywh = np.array([x + 0.5 * w, y + 0.5 * h, w, h], dtype=np.float32)
            out[r_idx, 0] = compute_box_hsv_hist_from_hsv(hsv_img, full_cxcywh[0], full_cxcywh[1], full_cxcywh[2], full_cxcywh[3])
            for side in (1, 2, 3, 4):
                cb = partial_crop(full_cxcywh, side, 0.72)
                out[r_idx, side] = compute_box_hsv_hist_from_hsv(hsv_img, float(cb[0]), float(cb[1]), float(cb[2]), float(cb[3]))

    tmp_file = hist_dir / f"{seq_name}_tmp.npy"
    np.save(tmp_file, out)
    tmp_file.replace(out_file)
    return seq_name, R


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    mot_dir = Path(r"D:\gaplink\data\mot")
    seq_names = sorted([p.stem for p in mot_dir.glob("*.txt") if not p.name.endswith("_ignore.txt")])
    print(f"Extracting fast 5-view HSV histograms for {len(seq_names)} sequences with {args.workers} workers...")
    with Pool(args.workers) as pool:
        done = 0
        for seq_name, _ in pool.imap_unordered(process_one_seq, seq_names):
            done += 1
            if done % 20 == 0:
                print(f"  done {done}/{len(seq_names)} sequences...", flush=True)
    print("Finished fast 5-view HSV histogram extraction.")


if __name__ == "__main__":
    main()
