"""Compute dataset statistics on cleaned UA-DETRAC MOT files and generate report/figures/fig_class_hist.png."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.io_mot import load_seq


def main():
    mot_dir = Path(r"D:\gaplink\data\mot")
    out_json = Path(r"D:\gaplink\data\stats.json")
    fig_path = Path(r"D:\gaplink\report\figures\fig_class_hist.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    mot_files = sorted([p for p in mot_dir.glob("*.txt") if not p.name.endswith("_ignore.txt")])
    track_lengths = []
    boxes_per_frame = []
    natural_gaps = 0
    dense_frames = 0
    total_frames = 0
    total_boxes = 0
    total_ignore_regions = 0

    for mf in mot_files:
        seq_name = mf.stem
        seq = load_seq(mot_dir, seq_name)
        total_ignore_regions += len(seq.ignore)
        if len(seq.rows) == 0:
            continue
        total_boxes += len(seq.rows)
        for f, sl in seq.by_frame.items():
            total_frames += 1
            cnt = sl.stop - sl.start
            boxes_per_frame.append(cnt)
            if cnt >= 10:
                dense_frames += 1
        for tid, tinfo in seq.tracks.items():
            f_arr = tinfo["frames"]
            track_lengths.append(int(len(f_arr)))
            if len(f_arr) > 1 and np.any(np.diff(f_arr) > 1):
                natural_gaps += 1

    tl = np.array(track_lengths, dtype=np.int32)
    bpf = np.array(boxes_per_frame, dtype=np.int32)
    stats = {
        "n_sequences": len(mot_files),
        "total_frames_with_objects": int(total_frames),
        "dense_frames_ge_10_vehicles": int(dense_frames),
        "n_tracks": int(len(track_lengths)),
        "total_vehicle_boxes": int(total_boxes),
        "total_ignore_regions": int(total_ignore_regions),
        "tracks_with_natural_gaps": int(natural_gaps),
        "track_length_p10": float(np.percentile(tl, 10)),
        "track_length_p50": float(np.percentile(tl, 50)),
        "track_length_p90": float(np.percentile(tl, 90)),
        "boxes_per_frame_mean": float(bpf.mean()),
        "boxes_per_frame_p90": float(np.percentile(bpf, 90)),
        "human_boxes": 0,
    }
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))

    # Plot 2-panel EDA: Track Length Distribution + Vehicles per Frame Distribution (100% English labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=150)
    ax1.hist(np.clip(tl, 1, 600), bins=35, color="#2B6CB0", edgecolor="white", alpha=0.9)
    ax1.axvline(float(np.percentile(tl, 50)), color="#E53E3E", linestyle="--", linewidth=1.5, label=f"Median = {np.percentile(tl,50):.0f} frames")
    ax1.set_title(f"Vehicle Trajectory Length Distribution (N = {len(tl):,} Tracks)")
    ax1.set_xlabel("Trajectory Length (Frames, Clipped at 600)")
    ax1.set_ylabel("Number of Vehicle Tracks")
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    ax2.hist(bpf, bins=30, color="#2F855A", edgecolor="white", alpha=0.9)
    ax2.axvline(float(bpf.mean()), color="#D69E2E", linestyle="--", linewidth=1.5, label=f"Mean = {bpf.mean():.1f} veh/frame")
    ax2.set_title(f"Simultaneous Vehicles per Frame ({total_boxes:,} Total Boxes)")
    ax2.set_xlabel("Active Vehicles per Frame")
    ax2.set_ylabel("Number of Frames")
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
