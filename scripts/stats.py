"""Compute dataset statistics and generate report/figures/fig_class_hist.png."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.io_mot import load_seq

CLASS_NAMES = {1: "car", 2: "van", 3: "bus", 4: "others"}

def main():
    mot_dir = Path(r"D:\gaplink\data\mot")
    out_json = Path(r"D:\gaplink\data\stats.json")
    fig_path = Path(r"D:\gaplink\report\figures\fig_class_hist.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    mot_files = sorted([p for p in mot_dir.glob("*.txt") if not p.name.endswith("_ignore.txt")])
    class_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    track_lengths = []
    natural_gaps = 0
    dense_frames = 0
    total_frames = 0

    for mf in mot_files:
        seq_name = mf.stem
        seq = load_seq(mot_dir, seq_name)
        if len(seq.rows) == 0:
            continue
        cids, cnts = np.unique(seq.rows[:, 7].astype(int), return_counts=True)
        for c, n in zip(cids, cnts):
            class_counts[int(c)] = class_counts.get(int(c), 0) + int(n)
        for f, sl in seq.by_frame.items():
            total_frames += 1
            if (sl.stop - sl.start) >= 10:
                dense_frames += 1
        for tid, tinfo in seq.tracks.items():
            f_arr = tinfo["frames"]
            track_lengths.append(int(len(f_arr)))
            if len(f_arr) > 1 and np.any(np.diff(f_arr) > 1):
                natural_gaps += 1

    tl = np.array(track_lengths, dtype=np.int32) if track_lengths else np.array([0], dtype=np.int32)
    stats = {
        "n_sequences": len(mot_files),
        "total_frames_with_objects": int(total_frames),
        "dense_frames_ge_10_vehicles": int(dense_frames),
        "n_tracks": int(len(track_lengths)),
        "tracks_with_natural_gaps": int(natural_gaps),
        "track_length_p10": float(np.percentile(tl, 10)),
        "track_length_p50": float(np.percentile(tl, 50)),
        "track_length_p90": float(np.percentile(tl, 90)),
        "class_box_counts": {CLASS_NAMES.get(k, str(k)): v for k, v in class_counts.items()},
        "human_boxes": 0,
    }
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))

    # Plot class distribution
    labels = [CLASS_NAMES[k] for k in sorted(CLASS_NAMES)]
    values = [class_counts.get(k, 0) for k in sorted(CLASS_NAMES)]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    bars = ax.bar(labels, values, color=["#193DB0", "#2B6CB0", "#4A5568", "#718096"])
    ax.set_title("Phân bố lớp phương tiện (0 đối tượng người)")
    ax.set_ylabel("Số lượng bounding box")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)

if __name__ == "__main__":
    main()
