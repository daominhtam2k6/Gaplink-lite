"""Generate all publication-quality figures for the GapLink-Lite SIC report."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"D:\gaplink")
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})


def main():
    test_res = json.loads((ROOT / "runs" / "test_final" / "test_results.json").read_text(encoding="utf-8"))
    val_sum = json.loads((ROOT / "runs" / "d4_val" / "val_summary.json").read_text(encoding="utf-8"))
    train_log = json.loads((ROOT / "runs" / "main" / "train_log.json").read_text(encoding="utf-8"))

    # 1. Training curves (loss + val top-1 accuracy)
    hist = train_log["history"]
    steps = [r["step"] for r in hist]
    losses = [r["loss_assign"] for r in hist]
    val_top1 = [r["val_top1"] * 100.0 for r in hist]
    val_hard = [r["val_top1_hard"] * 100.0 for r in hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(steps, losses, "-o", color="#2B6CB0", linewidth=2, markersize=4)
    ax1.set_title("Sinkhorn Assignment Loss vs Training Step")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Assignment Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, val_top1, "-o", color="#2F855A", linewidth=2, markersize=4, label="Val All Events")
    ax2.plot(steps, val_hard, "--s", color="#D69E2E", linewidth=2, markersize=4, label="Val Hard Events")
    ax2.set_title("Validation Top-1 Accuracy (%)")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Top-1 Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_training_curves.png")
    plt.close(fig)

    # 2. Event-level Top-1 Accuracy across gap bins (Short, Medium, Long, Hard, All)
    ev = test_res["event_level"]
    methods = ["C-2", "C-3", "C-4", "GapLink-Lite"]
    colors = ["#718096", "#3182CE", "#DD6B20", "#38A169"]
    cats = ["Short Gap (3-15)", "Medium Gap (16-45)", "Long Gap (46-90)", "Hard Subset", "Overall (All)"]
    keys = ["top1_gap_short", "top1_gap_med", "top1_gap_long", "top1_hard", "top1"]

    x = np.arange(len(cats))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, m in enumerate(methods):
        vals = [ev[m][k] * 100.0 for k in keys]
        rects = ax.bar(x + (i - 1.5) * width, vals, width, label=m, color=colors[i])
        for r in rects:
            h = r.get_height()
            ax.annotate(f"{h:.1f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylabel("Event-Level Top-1 Accuracy (%)")
    ax.set_title("Test Event-Level Re-Association Top-1 Accuracy by Occlusion Gap Bin (N = 12,000 Events)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_event_gap_bins.png")
    plt.close(fig)

    # 3. Calibration Coverage Before vs After across Gap Bins on Test Split
    cal_b = test_res["calib_test"]["before"]
    cal_a = test_res["calib_test"]["after"]
    cal_c3 = ev["C-3"]
    bins_labels = ["Short (3-15)", "Medium (16-45)", "Long (46-90)", "All Gaps"]
    b_keys = ["coverage90_gap_short", "coverage90_gap_med", "coverage90_gap_long", "coverage90"]

    v_c3 = [cal_c3[k] * 100.0 for k in b_keys]
    v_bef = [cal_b[k] * 100.0 for k in b_keys]
    v_aft = [cal_a[k] * 100.0 for k in b_keys]

    x = np.arange(len(bins_labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - w, v_c3, w, label="C-3 (Kalman Uncalibrated)", color="#3182CE")
    ax.bar(x, v_bef, w, label="GapLink-Lite (Before Calib)", color="#E53E3E")
    ax.bar(x + w, v_aft, w, label="GapLink-Lite (After Calib)", color="#38A169")
    ax.axhline(90.0, color="#1A202C", linestyle="--", linewidth=1.5, label="Nominal 90% Target")
    ax.axhspan(85.0, 95.0, color="#38A169", alpha=0.12, label="Target Band [85%, 95%]")
    ax.set_xticks(x)
    ax.set_xticklabels(bins_labels)
    ax.set_ylim(40, 102)
    ax.set_ylabel("Empirical 90% Ellipse Coverage (%)")
    ax.set_title("Uncertainty Calibration on Test Split: Empirical 90% Coverage Across Gap Bins")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_calibration_coverage.png")
    plt.close(fig)

    # 4. Online Tracker IRR & WLR under Full and Partial Occlusion Injection
    trk_full = test_res["tracker_level"]["full"]
    trk_part = test_res["tracker_level"]["partial"]
    t_methods = ["C-1", "C-2", "C-3", "C-4", "GapLink-Lite"]
    t_colors = ["#A0AEC0", "#718096", "#3182CE", "#DD6B20", "#38A169"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    x = np.arange(len(t_methods))
    irr_f = [trk_full[m]["IRR"] * 100.0 for m in t_methods]
    wlr_f = [trk_full[m]["WLR"] * 100.0 for m in t_methods]
    irr_p = [trk_part[m]["IRR"] * 100.0 for m in t_methods]
    wlr_p = [trk_part[m]["WLR"] * 100.0 for m in t_methods]

    b1 = ax1.bar(x - 0.18, irr_f, 0.35, label="IRR (%) [Higher is Better]", color="#2F855A")
    b2 = ax1.bar(x + 0.18, wlr_f, 0.35, label="WLR (%) [Lower is Better]", color="#C53030")
    ax1.set_xticks(x)
    ax1.set_xticklabels(t_methods, rotation=15)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Online Tracker: Full Occlusion Injection (40 Test Seqs)")
    ax1.legend(fontsize=8.5)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x - 0.18, irr_p, 0.35, label="IRR (%) [Higher is Better]", color="#2B6CB0")
    ax2.bar(x + 0.18, wlr_p, 0.35, label="WLR (%) [Lower is Better]", color="#C53030")
    ax2.set_xticks(x)
    ax2.set_xticklabels(t_methods, rotation=15)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Online Tracker: Partial Occlusion Injection (40 Test Seqs)")
    ax2.legend(fontsize=8.5)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_tracker_irr_wlr.png")
    plt.close(fig)

    # 5. Ablation study chart
    abl = test_res.get("ablations_test", {})
    if abl:
        a_names = ["Full GapLink-Lite", "w/o Context (DeepSets)", "w/o Color Histogram", "Gap-Blind (g=1)"]
        a_entries = [
            ev["GapLink-Lite"],
            abl.get("No-Context", ev["GapLink-Lite"]),
            abl.get("No-Color", ev["GapLink-Lite"]),
            abl.get("Gap-Blind", ev["GapLink-Lite"]),
        ]
        top1_all = [e["top1"] * 100.0 for e in a_entries]
        top1_long = [e["top1_gap_long"] * 100.0 for e in a_entries]

        x = np.arange(len(a_names))
        fig, ax = plt.subplots(figsize=(8.2, 4.3))
        ax.bar(x - 0.18, top1_all, 0.35, label="Overall Top-1 (%)", color="#2F855A")
        ax.bar(x + 0.18, top1_long, 0.35, label="Long Gap (46-90) Top-1 (%)", color="#D69E2E")
        ax.set_xticks(x)
        ax.set_xticklabels(a_names, rotation=10)
        ax.set_ylim(40, 100)
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title("Ablation Study on Test Events (Contribution of Each Self-Developed Module)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_ablation_study.png")
        plt.close(fig)

    print("Saved all report figures to", FIG_DIR)


if __name__ == "__main__":
    main()
