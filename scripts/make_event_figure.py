"""Figure for the report: three training events drawn on the real video frame.

Each panel shows one lost vehicle: its (noisy) history, the box where it was last seen, the candidates at the
decision frame and, if it exists, the true continuation. Events are sampled from TRAIN sequences with the same
sampler used for training (src.events.sample_event), so the figure shows exactly what the model is trained on.

    python -m scripts.make_event_figure            # writes report/latex/figures/fig_events.png and fig_events_vi.png
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.events import EventCfg, SeqIndex, sample_event
from src.io_mot import load_seq

ROOT = Path(__file__).resolve().parents[1]
MOT = ROOT / "data" / "mot"
HIST = ROOT / "data" / "cache" / "hist"
OUT = ROOT / "report" / "latex" / "figures"

TEXT = {
    "en": {"a": "(a) Hard case: a look-alike vehicle nearby",
           "b": "(b) Partial occlusion at the decision frame",
           "c": "(c) No match: the vehicle is still hidden",
           "hist": "history (with box noise)", "last": "last seen", "true": "true vehicle",
           "cand": "other candidates", "gap": "hidden for {g} frames"},
    "vi": {"a": "(a) Ca khó: có xe trông giống ở gần",
           "b": "(b) Bị che một phần tại khung quyết định",
           "c": "(c) Không ghép: xe vẫn còn bị che",
           "hist": "lịch sử (có nhiễu box)", "last": "vị trí cuối", "true": "xe thật",
           "cand": "ứng viên khác", "gap": "bị che {g} khung"},
}


def _img_path(seq_name: str, t: int) -> Path:
    meta = json.loads((MOT / f"{seq_name}_meta.json").read_text(encoding="utf-8"))
    split = meta.get("source_split", "train")
    return ROOT / "data" / "raw" / "images" / split / meta["folder_name"] / f"image{t:06d}.jpg"


def _rect(ax, b, ox, oy, s, **kw):
    cx, cy, w, h = b
    ax.add_patch(patches.Rectangle(((cx - w / 2 - ox) * s, (cy - h / 2 - oy) * s), w * s, h * s, fill=False, **kw))


def _pick(events, kind):
    for ev in events:
        for i in range(len(ev["t_lost"])):
            tgt = np.nonzero(ev["cand_label"] == i)[0]
            if kind == "a" and ev["mode"] == "occl" and ev["is_hard"] and len(tgt) == 1 and ev["side"][i] == 0 \
                    and ev["gap"][i] >= 20:
                c = ev["cand"][tgt[0]]
                d = np.linalg.norm(ev["cand"][:, :2] - c[:2], axis=1)
                d[tgt[0]] = 1e9
                if d.min() < 1.3 * max(c[2], c[3]):
                    return ev, i
            if kind == "b" and ev["mode"] == "occl" and len(tgt) == 1 and ev["side"][i] != 0 and ev["gap"][i] >= 15:
                return ev, i
            if kind == "c" and ev["mode"] == "nomatch_miss" and len(tgt) == 0 and ev["gap"][i] >= 15:
                return ev, i
    return None, None


def draw_panel(ax, ev, i, title, T):
    img = cv2.cvtColor(cv2.imread(str(_img_path(ev["seq"], int(ev["t_conf"])))), cv2.COLOR_BGR2RGB)
    L = int(ev["hist_len"][i])
    hist = ev["hist"][i, -L:]
    last = hist[-1]
    tgt = np.nonzero(ev["cand_label"] == i)[0]
    pts = [hist[:, :2]]
    if len(tgt):
        pts.append(ev["cand"][tgt][:, :2])
    pts = np.concatenate(pts)
    size = max(last[2], last[3])
    x0, y0 = pts.min(0) - 2.2 * size
    x1, y1 = pts.max(0) + 2.2 * size
    W, H = img.shape[1], img.shape[0]
    # 16:9 crop that contains the history and the true vehicle, kept inside the image
    cw = max(x1 - x0, (y1 - y0) * 16 / 9, 280.0)
    cw = min(cw, W, H * 16 / 9)
    ch = cw * 9 / 16
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = int(np.clip(cx - cw / 2, 0, W - cw))
    y0 = int(np.clip(cy - ch / 2, 0, H - ch))
    cw, ch = int(cw), int(ch)
    crop = img[y0:y0 + ch, x0:x0 + cw]
    s = 640 / cw
    crop = cv2.resize(crop, (640, int(round(ch * s))), interpolation=cv2.INTER_LINEAR)
    ax.imshow(crop)
    for j, c in enumerate(ev["cand"]):
        if j in tgt:
            continue
        if x0 - c[2] < c[0] < x0 + cw + c[2] and y0 - c[3] < c[1] < y0 + ch + c[3]:
            _rect(ax, c, x0, y0, s, edgecolor="#f6ad55", linewidth=1.4)
    ax.plot((hist[:, 0] - x0) * s, (hist[:, 1] - y0) * s, "-o", color="#ffd84d", ms=2.5, lw=1.4, alpha=0.95)
    _rect(ax, last, x0, y0, s, edgecolor="#63b3ed", linewidth=2.0, linestyle="--")
    for j in tgt:
        _rect(ax, ev["cand"][j], x0, y0, s, edgecolor="#48d17a", linewidth=2.6)
    ax.set_xlim(0, crop.shape[1])
    ax.set_ylim(crop.shape[0], 0)
    ax.set_title(title + " — " + T["gap"].format(g=int(ev["gap"][i])), fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    with open(ROOT / "data" / "splits.csv", encoding="utf-8") as f:
        seqs = [r["seq"] for r in csv.DictReader(f) if r["split"] == "train"]
    rng = np.random.default_rng(2026)
    cfg = EventCfg()
    idxs = []
    for s in seqs[:20]:
        seq = load_seq(MOT, s)
        hp = HIST / f"{s}.npy"
        if len(seq.tracks) >= 2:
            idxs.append(SeqIndex(seq, np.load(hp) if hp.exists() else None))
    events = []
    for k in range(3000):
        ev = sample_event(idxs[k % len(idxs)], rng, cfg, want_hard=(k % 2 == 0))
        if ev is not None:
            events.append(ev)
    chosen = {k: _pick(events, k) for k in "abc"}
    for lang, T in TEXT.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), dpi=170)
        for ax, k in zip(axes, "abc"):
            ev, i = chosen[k]
            if ev is None:
                ax.axis("off")
                continue
            draw_panel(ax, ev, i, T[k], T)
        handles = [plt.Line2D([], [], color="#e0b400", marker="o", ms=4, lw=1.4, label=T["hist"]),
                   patches.Patch(fill=False, edgecolor="#63b3ed", linestyle="--", linewidth=2, label=T["last"]),
                   patches.Patch(fill=False, edgecolor="#48d17a", linewidth=2.6, label=T["true"]),
                   patches.Patch(fill=False, edgecolor="#f6ad55", linewidth=1.4, label=T["cand"])]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        name = "fig_events.png" if lang == "en" else "fig_events_vi.png"
        fig.savefig(OUT / name)
        plt.close(fig)
        print("wrote", OUT / name)
    for k, (ev, i) in chosen.items():
        if ev is not None:
            print(k, ev["seq"], int(ev["t_conf"]), ev["mode"], int(ev["gap"][i]), bool(ev["is_hard"]), int(ev["side"][i]))


if __name__ == "__main__":
    main()
