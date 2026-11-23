"""Additional report figures, all drawn from the project data and frozen result files (no hand-made drawings).

    python -m scripts.make_report_figures            # writes report/latex/figures/fig_*.png (EN) and fig_*_vi.png (VI)

Figures
  fig_problem      schematic of the re-association problem (vehicle #7 before, during and after an occlusion by a
                   bus, and the candidates at the decision frame), drawn from coordinates in this script.
  fig_live_curve   Phase-2 validation sweep of every method (Nat-IRR vs Nat-WLR over its acceptance thresholds,
                   runs/live_v13_val/select.json), with the selected operating points and their test results.
  fig_dataset      six UA-DETRAC frames with ground-truth vehicle boxes and ignore regions; one sequence per camera
                   location, chosen to span the range of image brightness (day to night).
  fig_split        median background of every training and validation camera location (data/splits.csv).
  fig_calibration  90% coverage per occlusion-length bin before/after calibration
                   (runs/d4_val/val_summary.json -> calib_val, runs/test_final/test_results.json -> calib_test).
  fig_live         Phase-2 test results of the operating points selected on validation
                   (runs/live_v13_test/live_v13_results.json).
  fig_failures     two of the pre-registered failure cases (runs/test_final_errors/failure_analysis.json) drawn on
                   the real frame at the decision time.
"""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.io_mot import load_seq

ROOT = Path(__file__).resolve().parents[1]
MOT = ROOT / "data" / "mot"
OUT = ROOT / "report" / "latex" / "figures"
LANGS = ("en", "vi")

C_TRUE, C_LAST, C_HIST, C_CAND = "#48d17a", "#63b3ed", "#ffd84d", "#f6ad55"
C_GL, C_C3 = "#e53e3e", "#b794f4"
METHOD_COLORS = {"C-2": "#a0aec0", "C-3": "#dd6b20", "C-4": "#805ad5", "GapLink-Lite": "#2b6cb0"}


def _save(fig, stem, lang):
    name = f"{stem}.png" if lang == "en" else f"{stem}_vi.png"
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", OUT / name)


def _meta(seq):
    return json.loads((MOT / f"{seq}_meta.json").read_text(encoding="utf-8"))


def _frame(seq, t):
    m = _meta(seq)
    d = ROOT / "data" / "raw" / "images" / m.get("source_split", "train") / m["folder_name"]
    p = d / f"image{t:06d}.jpg"
    if not p.exists():
        p = d / f"img{t:05d}.jpg"
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _rect(ax, b, ox=0.0, oy=0.0, s=1.0, **kw):
    cx, cy, w, h = b
    ax.add_patch(patches.Rectangle(((cx - w / 2 - ox) * s, (cy - h / 2 - oy) * s), w * s, h * s, fill=False, **kw))


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def _frames_of(npz):
    """frame -> rows of a ragged (frames, offsets, boxes) npz."""
    fr, off = npz["frames"], npz["offsets"]
    return {int(f): (int(off[k]), int(off[k + 1])) for k, f in enumerate(fr)}


# ----------------------------------------------------------------------------------------------- problem
def _shade(color, f):
    """Lighten (f > 0) or darken (f < 0) a colour."""
    c = np.array(matplotlib.colors.to_rgb(color))
    return tuple(c + (1 - c) * f) if f > 0 else tuple(c * (1 + f))


def _badge(ax, x, y, text, color, z=9, r=0.24, fs=9.5):
    ax.add_patch(patches.Circle((x, y), r, facecolor="white", edgecolor=color, linewidth=2.0, zorder=z))
    ax.text(x, y - 0.01, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=color, zorder=z + 0.1)


def _veh(ax, x, yb, color, L=1.3, H=0.42, D=0.22, s=1.0, kind="car", ghost=False, z=4):
    """Vehicle seen by an elevated roadside camera: side face, roof (depth D) and wheels; x = centre, yb = ground line.
    s scales far vehicles down. Drawn with patches from the coordinates given here."""
    L, H, D = L * s, H * s, D * s
    x0 = x - L / 2
    side = [(x0, yb), (x0 + L, yb), (x0 + L, yb + H), (x0, yb + H)]
    roof = [(x0, yb + H), (x0 + L, yb + H), (x0 + L + D * 0.6, yb + H + D), (x0 + D * 0.6, yb + H + D)]
    if ghost:
        for poly in (side, roof):
            ax.add_patch(patches.Polygon(poly, closed=True, facecolor=(1, 1, 1, 0.15), edgecolor="white",
                                         linewidth=1.8, linestyle=(0, (4, 3)), zorder=z))
        return
    ax.add_patch(patches.Ellipse((x + D * 0.3, yb + 0.02), L * 1.05, 0.16 * s, facecolor="black", alpha=0.3,
                                 edgecolor="none", zorder=z - 0.3))
    ax.add_patch(patches.Polygon(roof, closed=True, facecolor=_shade(color, 0.25), edgecolor=_shade(color, -0.35),
                                 linewidth=0.9, zorder=z))
    ax.add_patch(patches.Polygon(side, closed=True, facecolor=color, edgecolor=_shade(color, -0.35), linewidth=0.9,
                                 zorder=z + 0.1))
    glass = "#2d3748"
    if kind == "bus":
        n = 6
        for k in range(n):
            wx = x0 + 0.12 * s + k * (L - 0.24 * s) / n
            ax.add_patch(patches.Rectangle((wx + 0.03 * s, yb + H * 0.5), (L - 0.24 * s) / n - 0.06 * s, H * 0.33,
                                           facecolor=glass, alpha=0.85, edgecolor="none", zorder=z + 0.2))
        ax.add_patch(patches.Rectangle((x0 + 0.1 * s, yb + H * 0.18), L - 0.2 * s, H * 0.08, facecolor="#2b6cb0",
                                       edgecolor="none", zorder=z + 0.2))
        wheels = (x0 + L * 0.15, x0 + L * 0.8)
    else:
        ax.add_patch(patches.Polygon([(x0 + L * 0.22, yb + H * 0.55), (x0 + L * 0.78, yb + H * 0.55),
                                      (x0 + L * 0.68, yb + H * 0.95), (x0 + L * 0.32, yb + H * 0.95)], closed=True,
                                     facecolor=glass, alpha=0.85, edgecolor="none", zorder=z + 0.2))
        ax.add_patch(patches.Rectangle((x0 + L - 0.07 * s, yb + H * 0.3), 0.07 * s, H * 0.14, facecolor="#fefcbf",
                                       edgecolor="none", zorder=z + 0.3))
        wheels = (x0 + L * 0.2, x0 + L * 0.8)
    for wx in wheels:
        ax.add_patch(patches.Circle((wx, yb), 0.11 * s, facecolor="#1a202c", edgecolor="#4a5568", linewidth=0.8,
                                    zorder=z + 0.3))


def fig_problem():
    """Schematic of the re-association problem as seen by an elevated roadside camera: a bus in the near lane passes
    in front of vehicle #7 in the far lane. Drawn from coordinates defined here; no dataset image is used."""
    T = {"en": {"t1": "Before the occlusion", "t2": "During the occlusion", "t3": "At the decision frame",
                "s1": "the camera tracks vehicle #7", "s2": "a bus passes in front of #7", "s3": "which candidate is #7?",
                "hist": "history of #7", "pred": "predicted position", "hidden": "hidden",
                "q": "#7 = A, B, C — or none of them?", "near": "near lane", "far": "far lane",
                "nA": "#7 re-appears", "nB": "look-alike", "nC": "new vehicle", "key": "How to read the figure",
                "k1": "vehicle #7, followed by the tracker", "k2": "where the model predicts #7 (Gaussian, 90% ellipse)",
                "k3": "candidates detected at the decision frame",
                "k4": "Answer: one candidate, or \u201cnone\u201d if #7 is still hidden"},
         "vi": {"t1": "Trước khi bị che", "t2": "Trong khi bị che", "t3": "Tại khung quyết định",
                "s1": "camera đang theo dõi xe #7", "s2": "xe buýt đi qua phía trước #7", "s3": "ứng viên nào là #7?",
                "hist": "lịch sử của #7", "pred": "vị trí dự đoán", "hidden": "bị che",
                "q": "#7 là A, B, C — hay không phải xe nào?", "near": "làn gần", "far": "làn xa",
                "nA": "#7 xuất hiện lại", "nB": "xe trông giống", "nC": "xe mới", "key": "Cách đọc hình",
                "k1": "xe #7, đang được tracker theo dõi", "k2": "nơi mô hình dự đoán #7 (Gaussian, elip 90%)",
                "k3": "ứng viên được phát hiện tại khung quyết định",
                "k4": "Đáp án: một ứng viên, hoặc \u201ckhông xe nào\u201d nếu #7 vẫn bị che"}}
    blue, red, acc = "#3182ce", "#e05252", "#2b6cb0"
    y_near, y_far, y_far2 = 0.75, 1.95, 2.55   # ground lines; the far lanes are higher in the image
    s_far, s_far2 = 0.85, 0.75                 # far vehicles look smaller

    def scene(ax):
        ax.add_patch(patches.Rectangle((0, 3.0), 10, 1.2, facecolor="#e6f0f8", zorder=0))          # background
        ax.add_patch(patches.Rectangle((0, 2.85), 10, 0.15, facecolor="#9ae6b4", zorder=0.2))      # far verge
        ax.add_patch(patches.Polygon([(0, 0.3), (10, 0.3), (10, 2.85), (0, 2.85)], facecolor="#4a5568", zorder=0.3))
        ax.add_patch(patches.Rectangle((0, 0.0), 10, 0.3, facecolor="#cbd5e0", zorder=0.4))        # near kerb
        for yl in (1.45, 2.3):
            for x0 in np.arange(0.2, 10, 0.9):
                ax.add_patch(patches.Rectangle((x0, yl - 0.022), 0.5, 0.044, facecolor="white", alpha=0.8,
                                               zorder=0.5))
        ax.set_xlim(1.0, 8.6)
        ax.set_ylim(0, 4.2)
        ax.set_aspect("equal")
        ax.axis("off")

    def title(ax, n, t, sub):
        ax.text(0.0, 1.17, f"{n}", transform=ax.transAxes, fontsize=11, fontweight="bold", color="white",
                ha="left", va="center", bbox=dict(boxstyle="circle,pad=0.25", facecolor=acc, edgecolor="none"))
        ax.text(0.07, 1.17, t, transform=ax.transAxes, fontsize=14, fontweight="bold", color="#1a202c",
                ha="left", va="center")
        ax.text(0.07, 1.04, sub, transform=ax.transAxes, fontsize=12, color="#4a5568", ha="left", va="center")

    def trail(ax, alpha=1.0):
        xs = np.linspace(1.2, 3.35, 8)
        for k, x in enumerate(xs):
            ax.add_patch(patches.Circle((x, y_far + 0.2), 0.05 + 0.012 * k, facecolor="#f6e05e",
                                        alpha=alpha * (0.35 + 0.07 * k), edgecolor="none", zorder=7))

    def lane_tags(ax, L):
        ax.text(8.5, y_near - 0.3, L["near"], fontsize=10, color="#e2e8f0", ha="right", va="center", zorder=1)
        ax.text(8.5, y_far - 0.22, L["far"], fontsize=10, color="#e2e8f0", ha="right", va="center", zorder=1)

    for lang in LANGS:
        L = T[lang]
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), dpi=200)
        # (1) before: #7 visible in the far lane, the bus is behind it in the near lane
        ax = axes[0]
        scene(ax)
        lane_tags(ax, L)
        trail(ax)
        _veh(ax, 7.2, y_far2, red, s=s_far2, z=3)
        _veh(ax, 4.0, y_far, blue, s=s_far, z=4)
        _badge(ax, 4.0, y_far + 0.85, "#7", acc, r=0.34, fs=12)
        ax.text(2.3, y_far + 0.5, L["hist"], fontsize=10.5, color="#1a202c", ha="center", va="bottom", zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#fefcbf", alpha=0.85, edgecolor="none"))
        _veh(ax, 1.6, y_near, "#f6c343", L=3.0, H=1.25, D=0.3, kind="bus", z=6)
        title(ax, 1, L["t1"], L["s1"])
        # (2) during: the bus covers #7; only the predicted position is left
        ax = axes[1]
        scene(ax)
        lane_tags(ax, L)
        trail(ax, alpha=0.55)
        _veh(ax, 7.9, y_far2, red, s=s_far2, z=3)
        _veh(ax, 5.2, y_far, blue, s=s_far, z=4)
        gx, gy = np.meshgrid(np.linspace(3.3, 7.6, 240), np.linspace(1.4, 3.0, 100))
        dens = np.exp(-0.5 * (((gx - 5.5) / 0.7) ** 2 + ((gy - (y_far + 0.2)) / 0.26) ** 2))
        rgba = np.zeros(dens.shape + (4,))
        rgba[..., :3] = matplotlib.colors.to_rgb("#68d391")
        rgba[..., 3] = 0.5 * dens
        _veh(ax, 5.0, y_near, "#f6c343", L=3.4, H=1.3, D=0.3, kind="bus", z=6)
        ax.imshow(rgba, extent=(3.3, 7.6, 1.4, 3.0), origin="lower", zorder=7, interpolation="bilinear")
        ax.add_patch(patches.Ellipse((5.5, y_far + 0.2), 2 * 0.7 * 2.146, 2 * 0.26 * 2.146, facecolor="none",
                                     edgecolor="#9ae6b4", linestyle=(0, (4, 3)), linewidth=1.3, zorder=7.5))
        _veh(ax, 5.2, y_far, blue, s=s_far, ghost=True, z=8)
        _badge(ax, 5.2, y_far + 0.85, "#7", acc, r=0.34, fs=12)
        ax.text(5.2, y_far + 0.22, L["hidden"], fontsize=11, color="white", ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a202c", alpha=0.75, edgecolor="none"))
        ax.annotate(L["pred"], xy=(6.9, y_far + 0.35), xytext=(7.3, 3.65), fontsize=11, color="#276749",
                    ha="center", va="center", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0fff4", edgecolor="#9ae6b4"),
                    arrowprops=dict(arrowstyle="-", color="#276749", lw=0.9))
        title(ax, 2, L["t2"], L["s2"])
        # (3) decision: the bus has moved on; A = #7, B = look-alike, C = new vehicle
        ax = axes[2]
        scene(ax)
        lane_tags(ax, L)
        cands = ((3.0, y_far2, red, s_far2, "C", L["nC"]), (7.7, y_far2, blue, s_far2, "B", L["nB"]),
                 (5.3, y_far, blue, s_far, "A", L["nA"]))
        for x, yb, col, sc, k, note in cands:
            _veh(ax, x, yb, col, s=sc, z=4 if yb == y_far else 3)
            _badge(ax, x, yb + 0.85 * sc / s_far, k, "#dd6b20", r=0.3, fs=12)
            below = k in ("A", "B")
            ax.text(x if below else x + 0.36, yb - 0.32 * sc / s_far if below else yb + 0.85 * sc / s_far, note, fontsize=10.5,
                    color="white", ha="center" if below else "left", va="center",
                    zorder=9, bbox=dict(boxstyle="round,pad=0.15", facecolor="#1a202c", alpha=0.65, edgecolor="none"))
        _veh(ax, 8.9, y_near, "#f6c343", L=3.4, H=1.3, D=0.3, kind="bus", z=6)
        ax.text(4.8, 0.15, L["q"], fontsize=12, fontweight="bold", color="#c05621", ha="center", va="center", zorder=9)
        title(ax, 3, L["t3"], L["s3"])
        fig.subplots_adjust(left=0.005, right=0.995, top=0.82, bottom=0.01, wspace=0.035)
        _save(fig, "fig_problem", lang)


# ----------------------------------------------------------------------------------------------- dataset
def fig_dataset():
    T = {"en": {"title": "{seq} · {loc} · {n} vehicles", "box": "annotated vehicle box", "ign": "ignore region"},
         "vi": {"title": "{seq} · {loc} · {n} xe", "box": "box phương tiện được gán nhãn", "ign": "vùng bỏ qua"}}
    with open(ROOT / "data" / "splits.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    first = {}
    for r in rows:  # first sequence of every camera location
        first.setdefault(r["location"], r)
    cands = []
    for loc, r in first.items():
        seq = r["seq"]
        t = max(1, int(r["n_frames"]) // 2)
        img = _frame(seq, t)
        cands.append((float(img.mean()), seq, loc, t))
    cands.sort()
    pick = [cands[int(round(k * (len(cands) - 1) / 5))] for k in range(6)][::-1]
    print("dataset:", [(c[1], c[2], round(c[0], 1)) for c in pick])
    panels = []
    for _, seq, loc, t in pick:
        sd = load_seq(MOT, seq)
        boxes = [tr["boxes"][np.nonzero(tr["frames"] == t)[0][0]] for tr in sd.tracks.values() if t in tr["frames"]]
        ign_p = MOT / f"{seq}_ignore.txt"
        ign = np.loadtxt(ign_p, delimiter=",", ndmin=2) if ign_p.exists() and ign_p.stat().st_size else np.zeros((0, 4))
        panels.append((seq, loc, _frame(seq, t), boxes, ign))
    for lang in LANGS:
        L = T[lang]
        fig, axes = plt.subplots(2, 3, figsize=(15, 5.9), dpi=150)
        for ax, (seq, loc, img, boxes, ign) in zip(axes.ravel(), panels):
            ax.imshow(img)
            for x, y, w, h in ign:
                ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="#000000", alpha=0.45, edgecolor="white",
                                               linewidth=0.6, hatch="//"))
            for b in boxes:
                _rect(ax, b, edgecolor=C_TRUE, linewidth=1.2)
            ax.set_title(L["title"].format(seq=seq, loc=loc, n=len(boxes)), fontsize=9.5)
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            ax.set_xticks([])
            ax.set_yticks([])
        handles = [patches.Patch(fill=False, edgecolor=C_TRUE, linewidth=1.5, label=L["box"]),
                   patches.Patch(facecolor="#000000", alpha=0.45, hatch="//", edgecolor="white", label=L["ign"])]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _save(fig, "fig_dataset", lang)


# ----------------------------------------------------------------------------------------------- split
def _median_bg(seq, n_frames, n=25):
    ts = np.linspace(1, n_frames, n).astype(int)
    frames = [cv2.resize(_frame(seq, int(t)), (320, 180), interpolation=cv2.INTER_AREA) for t in ts]
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def fig_split():
    T = {"en": {"train": "training", "val": "validation", "title": "{loc} ({n} video{pl})"},
         "vi": {"train": "huấn luyện", "val": "validation", "title": "{loc} ({n} video)"}}
    col = {"train": "#2b6cb0", "val": "#dd6b20"}
    with open(ROOT / "data" / "splits.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in ("train", "val")]
    locs = {}
    for r in rows:
        locs.setdefault((r["split"], r["location"]), []).append(r)
    keys = sorted(locs, key=lambda k: (k[0] != "train", k[1]))
    tiles = [(sp, loc, len(locs[(sp, loc)]), _median_bg(locs[(sp, loc)][0]["seq"], int(locs[(sp, loc)][0]["n_frames"])))
             for sp, loc in keys]
    ncol = 6
    nrow = int(np.ceil(len(tiles) / ncol))
    for lang in LANGS:
        L = T[lang]
        fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.0 * nrow + 0.4), dpi=150)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (sp, loc, n, bg) in zip(axes.ravel(), tiles):
            ax.axis("on")
            ax.imshow(bg)
            for sp_ in ax.spines.values():
                sp_.set_edgecolor(col[sp])
                sp_.set_linewidth(3.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(L["title"].format(loc=loc, n=n, pl="s" if n > 1 else ""), fontsize=8.5, color=col[sp])
        handles = [patches.Patch(facecolor="white", edgecolor=col[k], linewidth=3, label=L[k]) for k in ("train", "val")]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5, frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        _save(fig, "fig_split", lang)


# ----------------------------------------------------------------------------------------------- calibration
def fig_calibration():
    T = {"en": {"val": "Validation half B (pre-registered check of H2)", "test": "Test (reported after the test run)",
                "before": "before calibration", "after": "after calibration", "c3": "Kalman baseline C-3",
                "band": "target band 85–95%", "bins": ["short\n3–15 frames", "medium\n16–45 frames", "long\n46–90 frames"],
                "y": "90% ellipse coverage (%)"},
         "vi": {"val": "Validation nửa B (kiểm định H2 đã đăng ký trước)", "test": "Test (báo cáo sau lần chạy test)",
                "before": "trước hiệu chuẩn", "after": "sau hiệu chuẩn", "c3": "baseline Kalman C-3",
                "band": "dải mục tiêu 85–95%", "bins": ["ngắn\n3–15 khung", "trung bình\n16–45 khung", "dài\n46–90 khung"],
                "y": "Độ phủ elip 90% (%)"}}
    val = json.loads((ROOT / "runs" / "d4_val" / "val_summary.json").read_text())["calib_val"]
    test = json.loads((ROOT / "runs" / "test_final" / "test_results.json").read_text())["calib_test"]
    keys = ["coverage90_gap_short", "coverage90_gap_med", "coverage90_gap_long"]
    for lang in LANGS:
        L = T[lang]
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.7), dpi=170, sharey=True)
        for ax, (title, d, series) in zip(axes, ((L["val"], val, ("before", "after")),
                                                 (L["test"], test, ("before", "after", "c3")))):
            ax.axhspan(85, 95, color="#c6f6d5", alpha=0.6, zorder=0, label=L["band"])
            x = np.arange(3)
            w = 0.8 / len(series)
            colors = {"before": "#a0aec0", "after": "#2b6cb0", "c3": "#4a5568"}
            for k, sname in enumerate(series):
                v = [100 * d[sname][key] for key in keys]
                bars = ax.bar(x + (k - (len(series) - 1) / 2) * w, v, w * 0.92, color=colors[sname],
                              label=L[sname], zorder=2)
                for bar, val_ in zip(bars, v):
                    ax.text(bar.get_x() + bar.get_width() / 2, val_ + 0.4, f"{val_:.2f}", ha="center", va="bottom",
                            fontsize=7.5)
            ax.set_xticks(x)
            ax.set_xticklabels(L["bins"], fontsize=9)
            ax.set_ylim(75, 101)
            ax.set_title(title, fontsize=10)
            ax.grid(axis="y", alpha=0.3, zorder=1)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0].set_ylabel(L["y"])
        h, lab = axes[1].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=4, fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        _save(fig, "fig_calibration", lang)


# ----------------------------------------------------------------------------------------------- phase 2
def fig_live():
    T = {"en": {"irr": "Nat-IRR: identity kept after a natural gap (%)\nhigher is better",
                "wlr": "Nat-WLR: re-links joining two vehicles (%)\nlower is better",
                "far": "Route FAR: false forbidden-route alerts (%)\nlower is better", "target": "5% target",
                "gl": "GapLink-Lite (tuned)"},
         "vi": {"irr": "Nat-IRR: giữ được ID sau khoảng trống tự nhiên (%)\ncàng cao càng tốt",
                "wlr": "Nat-WLR: tái liên kết nối hai xe khác nhau (%)\ncàng thấp càng tốt",
                "far": "Route FAR: cảnh báo lộ trình cấm giả (%)\ncàng thấp càng tốt", "target": "mục tiêu 5%",
                "gl": "GapLink-Lite (đã tinh chỉnh)"}}
    res = json.loads((ROOT / "runs" / "live_v13_test" / "live_v13_results.json").read_text())["methods"]
    names = ["C-2", "C-3", "C-4", "GapLink-Lite"]
    vals = {n: res[f"v1.3/{n}"] for n in names}
    metrics = [("irr", lambda m: m["natural_occlusion"]["nat_irr"]), ("wlr", lambda m: m["natural_occlusion"]["nat_wlr"]),
               ("far", lambda m: m["forbidden_route"]["false_alarm_rate"])]
    for lang in LANGS:
        L = T[lang]
        labels = [n if n != "GapLink-Lite" else L["gl"] for n in names]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), dpi=170)
        for ax, (key, fn) in zip(axes, metrics):
            v = [100 * fn(vals[n]) for n in names]
            bars = ax.bar(range(4), v, color=[METHOD_COLORS[n] for n in names], width=0.65, zorder=2)
            for bar, x in zip(bars, v):
                ax.text(bar.get_x() + bar.get_width() / 2, x + max(v) * 0.015, f"{x:.2f}", ha="center", va="bottom",
                        fontsize=8.5)
            if key == "wlr":
                ax.axhline(5, color=C_GL, ls="--", lw=1.2, zorder=3)
                ax.text(-0.4, 5.4, L["target"], color=C_GL, fontsize=8, ha="left", va="bottom")
            ax.set_xticks(range(4))
            ax.set_xticklabels(labels, fontsize=8.5, rotation=12)
            ax.set_ylim(0, max(v) * 1.15)
            ax.set_title(L[key], fontsize=9.5)
            ax.grid(axis="y", alpha=0.3, zorder=1)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.tight_layout()
        _save(fig, "fig_live", lang)


# ----------------------------------------------------------------------------------------------- phase 2 curve
def fig_live_curve():
    T = {"en": {"x": "Nat-WLR — wrong re-links (%), lower is better", "y": "Nat-IRR — identity kept (%), higher is better",
                "target": "5% wrong-link limit", "val": "validation sweep (13 videos)", "sel": "selected on validation",
                "test": "test result at the selected point (40 videos)", "gl": "GapLink-Lite (tuned)",
                "c1": "C-1 (no re-association)"},
         "vi": {"x": "Nat-WLR — tái liên kết sai (%), càng thấp càng tốt",
                "y": "Nat-IRR — giữ được ID (%), càng cao càng tốt", "target": "giới hạn 5% liên kết sai",
                "val": "quét ngưỡng trên validation (13 video)", "sel": "điểm chọn trên validation",
                "test": "kết quả test tại điểm đã chọn (40 video)", "gl": "GapLink-Lite (đã tinh chỉnh)",
                "c1": "C-1 (không tái liên kết)"}}
    sel = json.loads((ROOT / "runs" / "live_v13_val" / "select.json").read_text())
    test = json.loads((ROOT / "runs" / "live_v13_test" / "live_v13_results.json").read_text())["methods"]
    sweep = sel["sweep"]
    kinds = {"C-2": "C-2", "C-3": "C-3", "C-4": "C-4", "GapLink-Lite": "GapLink-v13"}
    for lang in LANGS:
        L = T[lang]
        fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=170)
        ax.axvline(5, color=C_GL, ls="--", lw=1.1, zorder=1)
        ax.text(5.3, 3, L["target"], color=C_GL, fontsize=8, rotation=90, va="bottom")
        for name, kind in kinds.items():
            pts = sorted((v["thr"], 100 * v["nat_wlr"], 100 * v["nat_irr"]) for k, v in sweep.items()
                         if k.split("@")[0] == kind)
            col = METHOD_COLORS[name]
            lab = L["gl"] if name == "GapLink-Lite" else name
            ax.plot([p[1] for p in pts], [p[2] for p in pts], "-o", color=col, ms=3.5, lw=1.6, label=lab, zorder=2)
            ch = sel["chosen"][name]
            ax.plot(100 * ch["nat_wlr"], 100 * ch["nat_irr"], marker="*", ms=14, color=col, mec="black", mew=0.6,
                    zorder=4)
            t = test[f"v1.3/{name}"]["natural_occlusion"]
            tx, ty = 100 * t["nat_wlr"], 100 * t["nat_irr"]
            ax.plot(tx, ty, marker="D", ms=7.5, mfc="white", mec=col, mew=2.0, zorder=5)
            ax.annotate("", xy=(tx, ty), xytext=(100 * ch["nat_wlr"], 100 * ch["nat_irr"]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.9, alpha=0.7), zorder=3)
        c1 = sweep["C-1"]
        ax.plot(100 * c1["nat_wlr"], 100 * c1["nat_irr"], "s", color="#2d3748", ms=6, label=L["c1"], zorder=4)
        h, lab = ax.get_legend_handles_labels()
        h += [plt.Line2D([], [], color="#4a5568", lw=1.6, marker="o", ms=3.5),
              plt.Line2D([], [], ls="", marker="*", ms=12, color="#a0aec0", mec="black", mew=0.6),
              plt.Line2D([], [], ls="", marker="D", ms=7, mfc="white", mec="#4a5568", mew=2.0)]
        lab += [L["val"], L["sel"], L["test"]]
        ax.legend(h, lab, fontsize=7.8, loc="lower right", frameon=True, framealpha=0.95)
        ax.set_xlabel(L["x"], fontsize=9)
        ax.set_ylabel(L["y"], fontsize=9)
        ax.set_xlim(-0.5, None)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        _save(fig, "fig_live_curve", lang)


# ----------------------------------------------------------------------------------------------- failures
def fig_failures():
    T = {"en": {"wrong_candidate": "(a) Both methods link to a wrong vehicle", "missed_relink": "(b) GapLink-Lite declines, C-3 links wrongly",
                "gap": "hidden for {g} frames", "hist": "history", "last": "last seen", "true": "true vehicle",
                "gl": "GapLink-Lite choice", "c3": "C-3 Kalman choice", "cand": "other candidates",
                "abst": "GapLink-Lite: no match"},
         "vi": {"wrong_candidate": "(a) Cả hai phương pháp liên kết nhầm xe", "missed_relink": "(b) GapLink-Lite từ chối, C-3 liên kết nhầm",
                "gap": "bị che {g} khung", "hist": "lịch sử", "last": "vị trí cuối", "true": "xe thật",
                "gl": "lựa chọn của GapLink-Lite", "c3": "lựa chọn của C-3 Kalman", "cand": "ứng viên khác",
                "abst": "GapLink-Lite: không ghép"}}
    fa = json.loads((ROOT / "runs" / "test_final_errors" / "failure_analysis.json").read_text())["prereg_cases"]
    events = pickle.load(open(ROOT / "data" / "cache" / "test_events.pkl", "rb"))
    chosen = []
    for kind, need_distinct in (("wrong_candidate", True), ("missed_relink", False)):
        for c in fa:
            ev = events[c["event_idx"]]
            M = len(ev["cand"])
            ok = c["type"] == kind and 0 <= c["gt_cand"] < M and 0 <= c["c3_pred"] < M
            if kind == "wrong_candidate":
                ok = ok and 0 <= c["gaplink_pred"] < M and (not need_distinct or c["gaplink_pred"] != c["c3_pred"])
            if ok:
                chosen.append((kind, c, ev))
                break
    print("failures:", [(k, c["event_idx"], c["seq"], c["gap"]) for k, c, _ in chosen])
    for lang in LANGS:
        L = T[lang]
        fig, axes = plt.subplots(1, len(chosen), figsize=(10.5, 3.6), dpi=170)
        for ax, (kind, c, ev) in zip(np.atleast_1d(axes), chosen):
            i = c["track"]
            img = _frame(ev["seq"], int(ev["t_conf"]))
            hist = ev["hist"][i, -int(ev["hist_len"][i]):]
            last = hist[-1]
            key = [ev["cand"][c["gt_cand"]], ev["cand"][c["c3_pred"]]]
            if c["gaplink_pred"] >= 0:
                key.append(ev["cand"][c["gaplink_pred"]])
            pts = np.concatenate([hist[:, :2], np.array(key)[:, :2]])
            size = max(last[2], last[3])
            x0, y0 = pts.min(0) - 1.8 * size
            x1, y1 = pts.max(0) + 1.8 * size
            H, W = img.shape[:2]
            cw = min(max(x1 - x0, (y1 - y0) * 16 / 9, 280.0), W, H * 16 / 9)
            ch = cw * 9 / 16
            ox = int(np.clip((x0 + x1) / 2 - cw / 2, 0, W - cw))
            oy = int(np.clip((y0 + y1) / 2 - ch / 2, 0, H - ch))
            cw, ch = int(cw), int(ch)
            s = 640 / cw
            crop = cv2.resize(img[oy:oy + ch, ox:ox + cw], (640, int(round(ch * s))))
            ax.imshow(crop)
            special = {c["gt_cand"], c["c3_pred"], c["gaplink_pred"]}
            for j, b in enumerate(ev["cand"]):
                if j not in special and ox - b[2] < b[0] < ox + cw + b[2] and oy - b[3] < b[1] < oy + ch + b[3]:
                    _rect(ax, b, ox, oy, s, edgecolor=C_CAND, linewidth=1.0, alpha=0.75)
            ax.plot((hist[:, 0] - ox) * s, (hist[:, 1] - oy) * s, "-o", color=C_HIST, ms=2.3, lw=1.3)
            _rect(ax, last, ox, oy, s, edgecolor=C_LAST, linewidth=2.0, linestyle="--")
            _rect(ax, ev["cand"][c["gt_cand"]], ox, oy, s, edgecolor=C_TRUE, linewidth=2.8)
            _rect(ax, ev["cand"][c["c3_pred"]], ox, oy, s, edgecolor=C_C3, linewidth=2.2, linestyle=(0, (4, 2)))
            if c["gaplink_pred"] >= 0:
                _rect(ax, ev["cand"][c["gaplink_pred"]], ox, oy, s, edgecolor=C_GL, linewidth=2.2)
            else:
                ax.text(8, 22, L["abst"], color="white", fontsize=9, bbox=dict(facecolor=C_GL, alpha=0.85, pad=2))
            ax.set_xlim(0, crop.shape[1])
            ax.set_ylim(crop.shape[0], 0)
            ax.set_title(L[kind] + " — " + L["gap"].format(g=c["gap"]), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        handles = [plt.Line2D([], [], color="#e0b400", marker="o", ms=4, lw=1.3, label=L["hist"]),
                   patches.Patch(fill=False, edgecolor=C_LAST, linestyle="--", linewidth=2, label=L["last"]),
                   patches.Patch(fill=False, edgecolor=C_TRUE, linewidth=2.6, label=L["true"]),
                   patches.Patch(fill=False, edgecolor=C_GL, linewidth=2.2, label=L["gl"]),
                   patches.Patch(fill=False, edgecolor=C_C3, linestyle="--", linewidth=2.2, label=L["c3"]),
                   patches.Patch(fill=False, edgecolor=C_CAND, linewidth=1.2, label=L["cand"])]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        _save(fig, "fig_failures", lang)


def main(which=None):
    funcs = {"problem": fig_problem, "dataset": fig_dataset, "split": fig_split, "calibration": fig_calibration,
             "live": fig_live, "live_curve": fig_live_curve, "failures": fig_failures}
    for k, fn in funcs.items():
        if which is None or k in which:
            fn()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or None)
