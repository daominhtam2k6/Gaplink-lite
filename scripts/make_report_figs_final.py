"""Regenerate the report figures that had layout or labelling problems, plus the 3-video demo figure.

  fig_class_hist.png     : trajectory length / vehicles per frame (from data/mot, all 100 sequences)
  fig_event_gap_bins.png : Phase-1 event Top-1 by gap bin, frozen seed-1 checkpoint (release/results_v1.1/test_results.json)
  fig_demo3_results.png  : the three demo clips (runs/vms_lite_check/metrics.json, written by `python -m demo.vms_lite --check`)

  .venv\\Scripts\\python.exe -m scripts.make_report_figs_final
Figures are written to report/figures and report/latex/figures.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = [ROOT / "report" / "figures", ROOT / "report" / "latex" / "figures"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"GapLink-Lite": "#2a78d6", "C-2": "#eb6834", "C-3": "#1baf7a", "C-4": "#eda100", "C-1": "#8a8984"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout(rect=(0, 0.06, 1, 1) if name == "fig_demo3_results.png" else (0, 0, 1, 1))
    p = OUT[0] / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    for o in OUT[1:]:
        o.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, o / name)
    plt.close(fig)
    print("wrote", p)


def fig_class_hist():
    lens, per_frame = [], []
    files = sorted(p for p in (ROOT / "data" / "mot").glob("MVI_*.txt") if not p.stem.endswith("_ignore"))
    for p in files:
        a = np.loadtxt(p, delimiter=",", usecols=(0, 1), dtype=np.int64, ndmin=2)
        if len(a) == 0:
            continue
        _, c = np.unique(a[:, 1], return_counts=True)
        lens.extend(c.tolist())
        _, cf = np.unique(a[:, 0], return_counts=True)
        per_frame.extend(cf.tolist())
    lens, per_frame = np.array(lens), np.array(per_frame)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
    a1.hist(np.clip(lens, 0, 600), bins=40, color=COL["GapLink-Lite"], edgecolor=SURF)
    a1.axvline(np.median(lens), color=COL["C-2"], ls="--", lw=1.6, label=f"median = {int(np.median(lens))} frames")
    a1.set_title(f"Trajectory length ({len(lens):,} tracks)")
    a1.set_xlabel("Frames per trajectory (clipped at 600)")
    a1.set_ylabel("Number of trajectories")
    a1.legend(frameon=False)
    a2.hist(per_frame, bins=np.arange(0, per_frame.max() + 2) - 0.5, color=COL["C-3"], edgecolor=SURF)
    a2.axvline(per_frame.mean(), color=COL["C-2"], ls="--", lw=1.6, label=f"mean = {per_frame.mean():.1f} vehicles")
    a2.set_title(f"Vehicles per frame ({int(per_frame.sum()):,} boxes, {len(per_frame):,} frames)")
    a2.set_xlabel("Vehicles visible in the frame")
    a2.set_ylabel("Number of frames")
    a2.set_xlim(0, 45)
    a2.legend(frameon=False)
    save(fig, "fig_class_hist.png")
    return {"tracks": int(len(lens)), "median_len": float(np.median(lens)), "boxes": int(per_frame.sum()),
            "frames": int(len(per_frame)), "mean_per_frame": float(per_frame.mean()),
            "frames_ge10": int((per_frame >= 10).sum())}


def fig_event_gap_bins():
    ev = json.loads((ROOT / "release" / "results_v1.1" / "test_results.json").read_text(encoding="utf-8"))["event_level"]
    methods = ["C-2", "C-3", "C-4", "GapLink-Lite"]
    keys = [("top1_gap_short", "Short gap\n(3-15)"), ("top1_gap_med", "Medium gap\n(16-45)"),
            ("top1_gap_long", "Long gap\n(46-90)"), ("top1_hard", "Hard subset\n(PREREG H1)"), ("top1", "All events")]
    x = np.arange(len(keys))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    for i, m in enumerate(methods):
        vals = [100 * ev[m][k] for k, _ in keys]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, color=COL[m], label=m if m != "GapLink-Lite" else "GapLink-Lite (frozen seed 1)")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in keys])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Event Top-1 accuracy (%)")
    ax.set_title("Phase 1 test: Top-1 by occlusion length (12,000 events, validation-selected thresholds)")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    save(fig, "fig_event_gap_bins.png")


def fig_demo3():
    p = ROOT / "runs" / "vms_lite_check" / "metrics.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    r1 = next(v for k, v in d.items() if not k.startswith("_") and v["scenario"] == "relink")
    r2 = next(v for k, v in d.items() if not k.startswith("_") and v["scenario"] == "route")
    r3 = next(v for k, v in d.items() if not k.startswith("_") and v["scenario"] == "lock")
    panels = [("Video 1: identity recovery", ["Correct\nre-link", "Wrong\nlink", "New ID"],
               [r1["bl"]["ok"], r1["bl"]["wrong"], r1["bl"]["missed"]], [r1["gl"]["ok"], r1["gl"]["wrong"], r1["gl"]["missed"]],
               f"occluded vehicles: C-3 {r1['bl']['total']}, GapLink {r1['gl']['total']}"),
              ("Video 2: forbidden route", ["Correct\nID", "Wrong\nID", "False\nviolation"],
               [r2["bl"]["ok"], r2["bl"]["wrong_id"], r2["bl"]["false"]], [r2["gl"]["ok"], r2["gl"]["wrong_id"], r2["gl"]["false"]], ""),
              ("Video 3: target following", ["Re-acquired", "Wrong car\n/ lost"],
               [r3["bl"]["reacquired"], r3["bl"]["wrong"]], [r3["gl"]["reacquired"], r3["gl"]["wrong"]], "")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), gridspec_kw={"width_ratios": [3, 3, 2]})
    for ax, (title, labels, bl, gl, note) in zip(axes, panels):
        x = np.arange(len(labels))
        for off, vals, m, lab in ((-0.19, bl, "C-3", "C-3 Kalman"), (0.19, gl, "GapLink-Lite", "GapLink-Lite v1.3")):
            bars = ax.bar(x + off, vals, 0.36, color=COL[m], label=lab)
            for b, v in zip(bars, vals):
                ax.annotate(str(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylim(0, max(bl + gl) * 1.22 + 1)
        ax.set_title(title + (f"\n({note})" if note else "\n "), fontsize=10)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_ylabel("Count on the clip")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, frameon=False, loc="lower center")
    fig.subplots_adjust(bottom=0.2)
    save(fig, "fig_demo3_results.png")


def main():
    s = fig_class_hist()
    print("dataset:", s)
    fig_event_gap_bins()
    fig_demo3()


if __name__ == "__main__":
    main()
