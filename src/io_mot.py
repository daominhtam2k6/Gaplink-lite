from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

@dataclass
class SeqData:
    name: str
    W: int
    H: int
    n_frames: int
    img_dir: Path
    rows: np.ndarray            # (R, 9) sorted by (frame, id): frame,id,x,y,w,h,conf,class_id,vis
    by_frame: dict              # frame (int) -> slice
    tracks: dict                # id (int) -> {"frames": int32[], "boxes": float32[,4] (cx,cy,w,h), "row_idx": int32[]}
    ignore: np.ndarray          # (K, 4) float32 x,y,w,h

def center_in_ignored(boxes_xywh: np.ndarray, ignore: np.ndarray) -> np.ndarray:
    """Return boolean mask (N,) True if box center falls inside any ignored region."""
    if ignore is None or len(ignore) == 0 or len(boxes_xywh) == 0:
        return np.zeros((len(boxes_xywh),), dtype=bool)
    cx = boxes_xywh[:, 0] + 0.5 * boxes_xywh[:, 2]
    cy = boxes_xywh[:, 1] + 0.5 * boxes_xywh[:, 3]
    ix1 = ignore[:, 0][None, :]
    iy1 = ignore[:, 1][None, :]
    ix2 = (ignore[:, 0] + ignore[:, 2])[None, :]
    iy2 = (ignore[:, 1] + ignore[:, 3])[None, :]
    inside = (cx[:, None] >= ix1) & (cx[:, None] <= ix2) & (cy[:, None] >= iy1) & (cy[:, None] <= iy2)
    return inside.any(axis=1)

def load_seq(mot_dir: Path | str, name: str, img_root: Path | str | None = None, W: int = 960, H: int = 540) -> SeqData:
    mot_dir = Path(mot_dir)
    mot_file = mot_dir / f"{name}.txt"
    ig_file = mot_dir / f"{name}_ignore.txt"
    meta_file = mot_dir / f"{name}_meta.json"

    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        W = int(meta.get("W", W))
        H = int(meta.get("H", H))
        img_dir = Path(meta.get("img_dir", "")) if meta.get("img_dir") else (Path(img_root) / name if img_root else Path())
        n_frames_meta = int(meta.get("n_frames", 0))
    else:
        img_dir = Path(img_root) / name if img_root else Path()
        n_frames_meta = 0

    if ig_file.exists() and ig_file.stat().st_size > 0:
        ignore = np.loadtxt(ig_file, delimiter=",", dtype=np.float32)
        if ignore.ndim == 1:
            ignore = ignore[None, :]
    else:
        ignore = np.zeros((0, 4), dtype=np.float32)

    if mot_file.exists() and mot_file.stat().st_size > 0:
        rows = np.loadtxt(mot_file, delimiter=",", dtype=np.float32)
        if rows.ndim == 1:
            rows = rows[None, :]
        order = np.lexsort((rows[:, 1], rows[:, 0]))
        rows = rows[order]
    else:
        rows = np.zeros((0, 9), dtype=np.float32)

    by_frame = {}
    tracks = {}
    if len(rows) > 0:
        frames_int = rows[:, 0].astype(np.int32)
        unique_f, start_idx, counts = np.unique(frames_int, return_index=True, return_counts=True)
        for f, s, c in zip(unique_f, start_idx, counts):
            by_frame[int(f)] = slice(int(s), int(s + c))
        n_frames = max(int(frames_int.max()), n_frames_meta)

        ids_int = rows[:, 1].astype(np.int32)
        for tid in np.unique(ids_int):
            r_idx = np.nonzero(ids_int == tid)[0].astype(np.int32)
            sub = rows[r_idx]
            f_arr = sub[:, 0].astype(np.int32)
            # convert top-left (x,y,w,h) -> center (cx,cy,w,h)
            x, y, w, h = sub[:, 2], sub[:, 3], sub[:, 4], sub[:, 5]
            boxes = np.stack([x + 0.5 * w, y + 0.5 * h, w, h], axis=1).astype(np.float32)
            tracks[int(tid)] = {"frames": f_arr, "boxes": boxes, "row_idx": r_idx}
    else:
        n_frames = n_frames_meta

    return SeqData(
        name=name,
        W=W,
        H=H,
        n_frames=n_frames,
        img_dir=img_dir,
        rows=rows,
        by_frame=by_frame,
        tracks=tracks,
        ignore=ignore,
    )
