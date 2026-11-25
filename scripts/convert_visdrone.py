"""Convert VisDrone2019-MOT (val + test-dev) into GapLink MOT format for the out-of-domain
exploratory evaluation (docs/PREREG_ood_motor.md). Nothing here is used for training.

Input  : <raw>/VisDrone2019-MOT-val/{sequences,annotations}, <raw>/VisDrone2019-MOT-test-dev/{...}
Output : data/ood_visdrone/<group>/mot/<seq>.txt|_ignore.txt|_meta.json  and  data/ood_visdrone/seqs.csv

VisDrone annotation columns: frame,id,x,y,w,h,score,category,truncation,occlusion
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

VISDRONE_CATS = {0: "ignored", 1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
                 6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others"}
GROUPS = {"motor": (10,), "car": (4, 5, 6, 9)}
OCC_TO_VIS = {0: 1.0, 1: 0.6, 2: 0.3}
SPLITS = ("VisDrone2019-MOT-val", "VisDrone2019-MOT-test-dev")
STATIC_THR_PX = 1.0  # pre-registered: median global motion < 1.0 px/frame (native resolution) => static


def load_root() -> Path:
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "paths.yaml").read_text(encoding="utf-8"))
    return Path(__file__).resolve().parents[1] / cfg["root_dir"]


def read_annotation(p: Path) -> np.ndarray:
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 10), dtype=np.float32)
    a = np.loadtxt(p, delimiter=",", dtype=np.float32, usecols=range(10))
    return a[None, :] if a.ndim == 1 else a


def convert_annotation(ann: np.ndarray, cats: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Return (rows (R,9) frame,id,x,y,w,h,conf,class,vis ; ignore (K,4) x,y,w,h)."""
    if len(ann) == 0:
        return np.zeros((0, 9), np.float32), np.zeros((0, 4), np.float32)
    cat = ann[:, 7].astype(int)
    score = ann[:, 6]
    ign = ann[cat == 0][:, 2:6]
    ign = np.unique(ign, axis=0) if len(ign) else np.zeros((0, 4), np.float32)
    keep = np.isin(cat, cats) & (score > 0) & (ann[:, 4] >= 2) & (ann[:, 5] >= 2)
    sub = ann[keep]
    vis = np.array([OCC_TO_VIS.get(int(o), 0.3) for o in sub[:, 9]], dtype=np.float32)
    rows = np.column_stack([sub[:, 0], sub[:, 1], sub[:, 2:6], np.ones(len(sub)), sub[:, 7], vis]).astype(np.float32)
    # one row per (frame, id): keep first occurrence
    if len(rows):
        _, first = np.unique(rows[:, :2], axis=0, return_index=True)
        rows = rows[np.sort(first)]
        rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    return rows, ign.astype(np.float32)


def camera_motion_px(img_dir: Path, step: int = 5, max_pairs: int = 60) -> float:
    """Median global translation (px/frame, native resolution) via phase correlation on downscaled frames."""
    import cv2
    imgs = sorted(img_dir.glob("*.jpg"))
    if len(imgs) <= step:
        return float("nan")
    idx = np.linspace(0, len(imgs) - 1 - step, num=min(max_pairs, len(imgs) - step)).astype(int)
    mags = []
    for i in idx:
        a = cv2.imread(str(imgs[i]), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(imgs[i + step]), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            continue
        s = 480.0 / a.shape[1]
        a = cv2.resize(a, None, fx=s, fy=s).astype(np.float32)
        b = cv2.resize(b, None, fx=s, fy=s).astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(a, b)
        mags.append(np.hypot(dx, dy) / s / step)
    return float(np.median(mags)) if mags else float("nan")


def image_size(img_dir: Path) -> tuple[int, int, int]:
    import cv2
    imgs = sorted(img_dir.glob("*.jpg"))
    im = cv2.imread(str(imgs[0]), cv2.IMREAD_COLOR)
    return int(im.shape[1]), int(im.shape[0]), len(imgs)


def write_seq(out_mot: Path, name: str, rows: np.ndarray, ign: np.ndarray, meta: dict) -> None:
    out_mot.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_mot / f"{name}.txt", rows, delimiter=",", fmt=["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f", "%d", "%.3f"])
    np.savetxt(out_mot / f"{name}_ignore.txt", ign, delimiter=",", fmt="%.2f")
    (out_mot / f"{name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default=None, help="folder containing the extracted VisDrone2019-MOT-* folders")
    ap.add_argument("--no-motion", action="store_true")
    args = ap.parse_args()
    root = load_root()
    raw = Path(args.raw) if args.raw else root / "data" / "raw" / "visdrone"
    out_root = root / "data" / "ood_visdrone"
    table = []
    for split in SPLITS:
        sd = raw / split
        if not sd.exists():
            raise SystemExit(f"Missing {sd} - extract the VisDrone zip there first.")
        for ann_p in sorted((sd / "annotations").glob("*.txt")):
            name = ann_p.stem
            img_dir = sd / "sequences" / name
            W, H, n_img = image_size(img_dir)
            ann = read_annotation(ann_p)
            motion = float("nan") if args.no_motion else camera_motion_px(img_dir)
            rec = {"seq": name, "split": split, "W": W, "H": H, "n_frames": n_img,
                   "cam_motion_px": round(motion, 3),
                   "static": bool(motion < STATIC_THR_PX) if motion == motion else ""}
            for g, cats in GROUPS.items():
                rows, ign = convert_annotation(ann, cats)
                n_tr = int(len(np.unique(rows[:, 1]))) if len(rows) else 0
                meta = {"name": name, "W": W, "H": H, "n_frames": n_img, "img_dir": str(img_dir),
                        "source": f"VisDrone/{split}", "group": g, "categories": [VISDRONE_CATS[c] for c in cats],
                        "n_rows": int(len(rows)), "n_tracks": n_tr}
                write_seq(out_root / g / "mot", name, rows, ign, meta)
                rec[f"n_tracks_{g}"] = n_tr
            table.append(rec)
            print(f"{name}: {W}x{H} frames={n_img} motion={rec['cam_motion_px']} "
                  f"motor={rec['n_tracks_motor']} car={rec['n_tracks_car']}", flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "seqs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"Wrote {len(table)} sequences -> {out_root}")


if __name__ == "__main__":
    main()
