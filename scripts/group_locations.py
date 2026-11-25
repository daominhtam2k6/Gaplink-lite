"""Compute median background per sequence, cluster camera locations (~25 locations at thresh=0.65),
preserve official 40-sequence UA-DETRAC test split, and split 60 source-train sequences by location into train/val.
"""
import csv
import json
from pathlib import Path
import numpy as np
import cv2
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]  # project root, wherever the repo is unpacked


def compute_median_bg(img_dir: Path, n_samples: int = 35) -> np.ndarray:
    jpgs = sorted(img_dir.glob("*.jpg"))
    if not jpgs:
        return np.zeros((36, 64), dtype=np.float32)
    idx = np.linspace(0, len(jpgs) - 1, min(n_samples, len(jpgs)), dtype=int)
    frames = []
    for i in idx:
        im = cv2.imread(str(jpgs[i]), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            small = cv2.resize(im, (64, 36), interpolation=cv2.INTER_AREA)
            frames.append(small)
    if not frames:
        return np.zeros((36, 64), dtype=np.float32)
    med = np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)
    return med


def corr_dist_matrix(bgs: np.ndarray) -> np.ndarray:
    N = len(bgs)
    flat = bgs.reshape(N, -1).astype(np.float64)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    normed = flat / norms
    sim = np.clip(normed @ normed.T, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return 0.5 * (dist + dist.T)


def main():
    mot_dir = ROOT / "data" / "mot"
    splits_csv = ROOT / "data" / "splits.csv"
    fig_dir = ROOT / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metas = []
    for mf in sorted(mot_dir.glob("*_meta.json")):
        metas.append(json.loads(mf.read_text(encoding="utf-8")))

    assert len(metas) > 0, "No *_meta.json found in data/mot"
    print(f"Computing median background for {len(metas)} sequences...")
    bgs = []
    for m in metas:
        bgs.append(compute_median_bg(Path(m["img_dir"])))
    bgs = np.stack(bgs, axis=0)

    dist = corr_dist_matrix(bgs)
    condensed = squareform(dist, checks=False)
    Z = sch.linkage(condensed, method="average")

    # Threshold 0.65 on (1 - Pearson r) yields 25 camera location clusters matching UA-DETRAC's ~24 locations
    thresh = 0.65
    raw_clusters = sch.fcluster(Z, t=thresh, criterion="distance")

    train_idx = [i for i, m in enumerate(metas) if m["source_split"] == "train"]
    test_idx = [i for i, m in enumerate(metas) if m["source_split"] == "test"]

    # Group the 60 source-train sequences by location cluster and split 80% train / 20% val by location
    loc_to_train_seqs = {}
    for i in train_idx:
        loc_id = f"LOC_{int(raw_clusters[i]):02d}"
        loc_to_train_seqs.setdefault(loc_id, []).append(i)

    sorted_locs = sorted(loc_to_train_seqs.keys(), key=lambda k: (-len(loc_to_train_seqs[k]), k))
    val_locs = set()
    val_seq_count = 0
    target_val_seqs = 12
    for idx_l, loc_id in enumerate(sorted_locs):
        if idx_l % 4 == 1 and val_seq_count + len(loc_to_train_seqs[loc_id]) <= target_val_seqs + 2:
            val_locs.add(loc_id)
            val_seq_count += len(loc_to_train_seqs[loc_id])

    rows_out = []
    for i, m in enumerate(metas):
        if m["source_split"] == "test":
            loc_id = f"LOC_T{int(raw_clusters[i]):02d}"
            sp = "test"
        else:
            loc_id = f"LOC_{int(raw_clusters[i]):02d}"
            sp = "val" if loc_id in val_locs else "train"
        rows_out.append({
            "seq": m["name"],
            "source_split": m["source_split"],
            "location": loc_id,
            "split": sp,
            "n_frames": m["n_frames"],
            "n_tracks": m["n_tracks"],
        })

    with open(splits_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seq", "source_split", "location", "split", "n_frames", "n_tracks"])
        writer.writeheader()
        writer.writerows(rows_out)

    counts = {sp: sum(1 for r in rows_out if r["split"] == sp) for sp in ("train", "val", "test")}
    locs = {sp: len({r["location"] for r in rows_out if r["split"] == sp}) for sp in ("train", "val", "test")}
    total_raw_locs = len(set(raw_clusters))
    print(f"Total camera clusters (t={thresh}): {total_raw_locs} | Splits seq counts: {counts} | unique locations per split: {locs}")

    n_show = min(40, len(bgs))
    cols = 8
    rows_g = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows_g, cols, figsize=(14, 1.6 * rows_g), dpi=130)
    axes = np.array(axes).reshape(-1)
    for k in range(len(axes)):
        axes[k].axis("off")
        if k < n_show:
            axes[k].imshow(bgs[k], cmap="gray")
            r = rows_out[k]
            axes[k].set_title(f"{r['seq']}\n{r['location']} ({r['split']})", fontsize=7)
    fig.suptitle(f"Median Background & Camera Location Clustering ({total_raw_locs} Viewpoint Clusters)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_location_clusters.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
