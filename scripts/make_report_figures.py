"""Additional report figures, all drawn from the project data and frozen result files (no hand-made drawings).

    python -m scripts.make_report_figures            # writes report/latex/figures/fig_*.png (EN) and fig_*_vi.png (VI)

Figures
  fig_problem      three frames of demo clip 3 (MVI_40864): a van before, during and after an occlusion by a bus.
                   The van is the ground-truth track under the operator's lock point; "hidden" frames are those in which
                   no cached YOLOv8n detection overlaps its ground-truth box (IoU < 0.3).
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
METHOD_COLORS = {"C-2": "#a0aec0", "C-3": "#718096", "C-4": "#4a5568", "GapLink-Lite": "#2b6cb0"}


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
def fig_problem():
    T = {"en": {"before": "Frame {f}: the van is visible", "during": "Frame {f}: hidden behind the bus",
                "after": "Frame {f}: it re-appears", "box": "the van (ground truth)",
                "hidden": "hidden position (ground truth, not seen by the tracker)", "path": "path so far"},
         "vi": {"before": "Khung {f}: xe van đang nhìn thấy", "during": "Khung {f}: bị xe buýt che",
                "after": "Khung {f}: xe xuất hiện lại", "box": "xe van (ground truth)",
                "hidden": "vị trí khi bị che (ground truth, tracker không thấy)", "path": "đường đi đến lúc đó"}}
    vd = ROOT / "demo" / "sample_videos"
    meta = json.loads((vd / "demo3_lock_meta.json").read_text(encoding="utf-8"))
    gt, det = np.load(vd / "demo3_lock_gt.npz", allow_pickle=True), np.load(vd / "demo3_lock_dets.npz")
    seq, f0 = str(meta["source"]["seq"]), int(meta["source"]["f0"])
    lock_f, lx, ly = meta["lock"][0]
    gidx, didx = _frames_of(gt), _frames_of(det)
    a, b = gidx[int(lock_f)]
    d = np.hypot(gt["boxes"][a:b, 0] - lx, gt["boxes"][a:b, 1] - ly)
    tid = int(gt["tids"][a + int(np.argmin(d))])
    track, visible = {}, {}
    for f, (a, b) in gidx.items():
        rows = np.nonzero(gt["tids"][a:b] == tid)[0]
        if len(rows):
            box = gt["boxes"][a + rows[0]]
            track[f] = box
            da, db = didx.get(f, (0, 0))
            visible[f] = any(_iou(box, dbox) >= 0.3 for dbox in det["boxes"][da:db])
    fs = sorted(track)
    # longest run of hidden frames after the lock
    runs, cur = [], []
    for f in fs:
        if f >= lock_f and not visible[f]:
            cur.append(f)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    hid = max(runs, key=len)
    f_before, f_mid = hid[0] - 1, hid[len(hid) // 2]
    f_after = next(f for f in fs if f > hid[-1] and visible[f])
    pts = np.array([track[f][:2] for f in fs if f <= f_after])
    size = max(track[f_before][2], track[f_before][3])
    x0, y0 = pts.min(0) - 2.5 * size
    x1, y1 = pts.max(0) + 2.5 * size
    img0 = _frame(seq, f_before - 1 + f0)
    H, W = img0.shape[:2]
    cw = min(max(x1 - x0, (y1 - y0) * 16 / 9), W, H * 16 / 9)
    ch = cw * 9 / 16
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ox, oy = int(np.clip(cx - cw / 2, 0, W - cw)), int(np.clip(cy - ch / 2, 0, H - ch))
    cw, ch = int(cw), int(ch)
    s = 640 / cw
    print("problem: track", tid, "visible until", f_before, "hidden", hid[0], "-", hid[-1], "re-appears", f_after)
    for lang in LANGS:
        L = T[lang]
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.55), dpi=170)
        for ax, f, key in zip(axes, (f_before, f_mid, f_after), ("before", "during", "after")):
            crop = _frame(seq, f - 1 + f0)[oy:oy + ch, ox:ox + cw]
            crop = cv2.resize(crop, (640, int(round(ch * s))))
            ax.imshow(crop)
            if key == "during":
                _rect(ax, track[f], ox, oy, s, edgecolor="white", linewidth=2.0, linestyle="--")
            else:
                _rect(ax, track[f], ox, oy, s, edgecolor=C_TRUE, linewidth=2.6)
            ax.set_xlim(0, crop.shape[1])
            ax.set_ylim(crop.shape[0], 0)
            ax.set_title(L[key].format(f=f), fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        handles = [patches.Patch(fill=False, edgecolor=C_TRUE, linewidth=2.6, label=L["box"]),
                   patches.Patch(fill=False, edgecolor="#555555", linestyle="--", linewidth=2, label=L["hidden"])]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
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
             "live": fig_live, "failures": fig_failures}
    for k, fn in funcs.items():
        if which is None or k in which:
            fn()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or None)
