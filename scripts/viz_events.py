"""Render 20 sampled events visually into report/events_vis/ for human/agent inspection."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.io_mot import load_seq
from src.events import EventCfg, SeqIndex, sample_event

def main():
    mot_dir = Path(r"D:\gaplink\data\mot")
    hist_dir = Path(r"D:\gaplink\data\cache\hist")
    splits_csv = Path(r"D:\gaplink\data\splits.csv")
    out_dir = Path(r"D:\gaplink\report\events_vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_csv, encoding="utf-8") as f:
        train_seqs = [r["seq"] for r in csv.DictReader(f) if r["split"] == "train"]

    rng = np.random.default_rng(101)
    cfg = EventCfg()
    indices = []
    for s in train_seqs[:12]:
        seq = load_seq(mot_dir, s)
        hp = hist_dir / f"{s}.npy"
        hc = np.load(hp) if hp.exists() else None
        if len(seq.tracks) >= 2:
            indices.append(SeqIndex(seq, hc))

    saved = 0
    for k in range(100):
        if saved >= 20:
            break
        idx = indices[k % len(indices)]
        ev = sample_event(idx, rng, cfg, want_hard=(saved % 2 == 0))
        if ev is None:
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
        ax.set_xlim(0, ev["W"])
        ax.set_ylim(ev["H"], 0)
        ax.set_facecolor("#1A202C")
        N = len(ev["t_lost"])
        M = len(ev["cand"])

        # Plot track history trails and last box before occlusion
        colors = ["#63B3ED", "#F6AD55", "#68D391", "#FC8181", "#B794F4", "#F687B3"]
        for i in range(N):
            L = int(ev["hist_len"][i])
            h_boxes = ev["hist"][i, -L:]
            c = colors[i % len(colors)]
            ax.plot(h_boxes[:, 0], h_boxes[:, 1], "-o", color=c, markersize=2.5, linewidth=1.2, label=f"Trk {i} (g={ev['gap'][i]})")
            lb = h_boxes[-1]
            rect = patches.Rectangle((lb[0] - 0.5*lb[2], lb[1] - 0.5*lb[3]), lb[2], lb[3],
                                     linewidth=1.5, edgecolor=c, facecolor="none", linestyle="--")
            ax.add_patch(rect)

        # Plot candidates at t_conf (thick green border if matched to a track, gray if distractor)
        for j in range(M):
            cb = ev["cand"][j]
            matched_trk = int(ev["cand_label"][j])
            if matched_trk >= 0:
                ec = colors[matched_trk % len(colors)]
                lw = 2.8
                tag = f"C{j}=T{matched_trk}*"
            else:
                ec = "#A0AEC0"
                lw = 1.0
                tag = f"C{j} (dust)"
            rect = patches.Rectangle((cb[0] - 0.5*cb[2], cb[1] - 0.5*cb[3]), cb[2], cb[3],
                                     linewidth=lw, edgecolor=ec, facecolor="none")
            ax.add_patch(rect)
            ax.text(cb[0], cb[1] - 0.5*cb[3] - 4, tag, color=ec, fontsize=7, ha="center")

        ax.set_title(f"Event #{saved+1} | seq={ev['seq']} | mode={ev['mode']} | hard={ev['is_hard']} | N={N}, M={M}", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"event_{saved+1:02d}.png")
        plt.close(fig)
        saved += 1

    print(f"Saved {saved} event visualizations to {out_dir}")

if __name__ == "__main__":
    main()
