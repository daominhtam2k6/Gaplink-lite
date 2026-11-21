"""Report figures redrawn from FROZEN results only (release/results_v1.1, release/results_v1.3).

    python scripts/make_report_figs_v13.py
Writes report/figures/*.png and copies them to report/latex/figures/.
Palette: validated categorical order (dataviz skill); identity also carried by marker shape + direct labels.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R11 = ROOT / "release" / "results_v1.1"
R13 = ROOT / "release" / "results_v1.3"
OUT = ROOT / "report" / "figures"
LTX = ROOT / "report" / "latex" / "figures"

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"GapLink-Lite": "#2a78d6", "C-2": "#eb6834", "C-3": "#1baf7a", "C-4": "#eda100", "C-1": "#8a8984", "ref": "#8a8984"}
MRK = {"GapLink-Lite": "o", "C-2": "s", "C-3": "^", "C-4": "D", "C-1": "v", "ref": "o"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def _grid(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    if LTX.exists():
        shutil.copy2(p, LTX / name)
    print("wrote", p)


def fig_event_curve():
    d = json.loads((R11 / "test_results.json").read_text(encoding="utf-8"))
    oc, ev = d["operating_curves_test"], d["event_level"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for k in ["C-2", "C-3", "C-4", "GapLink-Lite"]:
        pts = sorted(oc[k], key=lambda r: r["wlr"])
        ax.plot([100 * r["wlr"] for r in pts], [100 * r["top1"] for r in pts], color=COL[k], lw=2, zorder=2)
        sel = ev[k]  # operating point chosen on validation
        ax.scatter(100 * sel["wrong_link_rate"], 100 * sel["top1"], s=70, marker=MRK[k], color=COL[k],
                   edgecolor=SURF, linewidth=2, zorder=3)
        ax.annotate(f"{k}", (100 * sel["wrong_link_rate"], 100 * sel["top1"]), xytext=(6, 6),
                    textcoords="offset points", color=INK, fontsize=9, fontweight="bold" if k == "GapLink-Lite" else None)
    ax.set_xlabel("Wrong-link rate, WLR (%)  — lower is better")
    ax.set_ylabel("Event Top-1 accuracy (%)  — higher is better")
    ax.set_xlim(0, 25)
    ax.set_ylim(20, 100)
    _grid(ax)
    ax.set_title("Accuracy vs. wrong links across all thresholds (test, 12,000 events)\nmarkers = operating points selected on validation",
                 fontsize=10, color=INK, loc="left")
    save(fig, "fig_top1_vs_wlr_curve.png")


def fig_calibration():
    d = json.loads((R11 / "test_results.json").read_text(encoding="utf-8"))
    v = json.loads((R11 / "val_summary.json").read_text(encoding="utf-8"))["calib_val"]["after"]
    ct = d["calib_test"]
    bins = [("coverage90_gap_short", "Short (3–15)"), ("coverage90_gap_med", "Medium (16–45)"), ("coverage90_gap_long", "Long (46–90)")]
    series = [("Post-calibration, validation (PREREG H2)", v, COL["GapLink-Lite"], None),
              ("Post-calibration, test", ct["after"], COL["GapLink-Lite"], "//"),
              ("Uncalibrated, test (post hoc)", ct["before"], COL["C-2"], None),
              ("C-3 Kalman, test", ct["c3"], COL["C-3"], None)]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    w = 0.19
    ax.axhspan(85, 95, color=GRID, alpha=0.7, zorder=0, lw=0)
    ax.axhline(90, color=INK2, lw=1, ls="--", zorder=1)
    for i, (lbl, src, c, hatch) in enumerate(series):
        xs = [b + (i - 1.5) * (w + 0.02) for b in range(3)]
        ax.bar(xs, [100 * src[k] for k, _ in bins], width=w, color=c if hatch is None else SURF, edgecolor=c,
               hatch=hatch, linewidth=1.5, label=lbl, zorder=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for _, n in bins])
    ax.set_ylim(75, 100)
    ax.set_ylabel("Empirical 90% ellipse coverage (%)")
    ax.text(-0.45, 94.2, "H2 band [85, 95]%", color=INK2, fontsize=8, ha="left")
    _grid(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, "fig_calibration_coverage.png")


def fig_live_scatter():
    res = json.loads((R13 / "live_v13_results.json").read_text(encoding="utf-8"))["methods"]
    op = json.loads((R13 / "live_v13_results.json").read_text(encoding="utf-8"))["operating_points"]
    rows = [("C-1", "C-1", "C-1 (IoU only)"), ("C-2", "v1.3/C-2", f"C-2 (θ={op['c2_theta_live']})"),
            ("C-3", "v1.3/C-3", f"C-3 Kalman (θ={op['c3_theta_live']})"), ("C-4", "v1.3/C-4", f"C-4 (θ={op['c4_theta_live']})"),
            ("GapLink-Lite", "v1.3/GapLink-Lite", f"GapLink-Lite v1.3 (τ={op['gaplink_tau_live']})")]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axvspan(0, 5, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(0.2, 95, "target WLR ≤ 5%", color=INK2, fontsize=8)
    offs = {"C-1": (10, 0), "C-2": (10, -14), "C-3": (-10, -16), "C-4": (10, -4), "GapLink-Lite": (-6, 16)}
    for k, key, lbl in rows:
        m = res[key]["natural_occlusion"]
        x, y = 100 * m["nat_wlr"], 100 * m["nat_irr"]
        ax.scatter(x, y, s=95, marker=MRK[k], color=COL[k], edgecolor=SURF, linewidth=2, zorder=3)
        ax.annotate(lbl, (x, y), xytext=offs[k], textcoords="offset points", color=INK, fontsize=9,
                    fontweight="bold" if k == "GapLink-Lite" else None, va="center",
                    ha="right" if k == "C-3" else "left")
    ax.set_xlabel("Nat-WLR (%)  — lower is better")
    ax.set_ylabel("Nat-IRR (%)  — higher is better")
    ax.set_xlim(-1, 30)
    ax.set_ylim(0, 100)
    _grid(ax)
    ax.set_title("Live YOLOv8n, 40 test videos — operating points selected on validation", fontsize=10, color=INK, loc="left")
    save(fig, "fig_live_v13_scatter.png")


if __name__ == "__main__":
    fig_event_curve()
    fig_calibration()
    fig_live_scatter()
