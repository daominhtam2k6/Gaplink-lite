"""Error analysis on the frozen v1.1 test predictions (no threshold or checkpoint change).

1. Re-computes GapLink-Lite / C-3 / C-4 decisions from release/final.pt and asserts they reproduce the frozen
   aggregates in release/results_v1.1/test_results.json.
2. Failure taxonomy over ALL 27,120 lost tracks, per gap bin:
     correct | missed relink (dustbin while target visible) | wrong candidate | false link (target had left)
3. PREREG §4 failure cases: the 10 events where GapLink-Lite gives the LOWEST probability to the correct answer,
   split 3-3-4 over the gap bins (short, medium, long); one panel each.
4. Per-location event Top-1 (15 test camera locations) for the report.

    python -m scripts.failure_analysis
Outputs: runs/test_final_errors/failure_analysis.json, report/figures/fig_failure_cases.png (+ latex/figures copy)
"""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BIN_NAMES = ["short (3-15)", "medium (16-45)", "long (46-90)"]
PER_BIN = [3, 3, 4]  # PREREG §4


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def outcome(gt: int, pr: int) -> str:
    if pr == gt:
        return "correct"
    if gt >= 0 and pr == -1:
        return "missed_relink"
    if gt >= 0 and pr >= 0:
        return "wrong_candidate"
    return "false_link_after_exit"  # gt == -1, pr >= 0


def taxonomy(events, preds):
    cats = ["correct", "missed_relink", "wrong_candidate", "false_link_after_exit"]
    t = {b: {c: 0 for c in cats} for b in range(3)}
    for ev, pr in zip(events, preds):
        for i, gt in enumerate(ev["label"]):
            t[int(ev["gap_bin"][i])][outcome(int(gt), int(pr[i]))] += 1
    return {BIN_NAMES[b]: v for b, v in t.items()}


def p_correct(ev, P):
    N, M = len(ev["t_lost"]), len(ev["cand"])
    out = []
    for i in range(N):
        gt = int(ev["label"][i])
        out.append(float(P[i, gt] if 0 <= gt < M else P[i, M]))
    return out


def plot_cases(cases, events, preds_gl, preds_c3, P_list, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    INK, GT, GL, C3, OTH = "#0b0b0b", "#1baf7a", "#e34948", "#2a78d6", "#a3a29c"
    fig, axes = plt.subplots(5, 2, figsize=(10, 15.5))
    for ax, c in zip(axes.ravel(), cases):
        ev = events[c["event_idx"]]; i = c["track"]
        L = int(ev["hist_len"][i]); hb = ev["hist"][i, -L:]
        gt, pg, pc = c["gt_cand"], c["gaplink_pred"], c["c3_pred"]
        xs = list(hb[:, 0]) + list(ev["cand"][:, 0]); ys = list(hb[:, 1]) + list(ev["cand"][:, 1])
        pad = 60
        ax.set_xlim(max(0, min(xs) - pad), min(ev["W"], max(xs) + pad))
        ax.set_ylim(min(ev["H"], max(ys) + pad), max(0, min(ys) - pad))
        ax.plot(hb[:, 0], hb[:, 1], "-o", color=INK, ms=2.5, lw=1.2)
        for j, cb in enumerate(ev["cand"]):
            col = GT if j == gt else (GL if j == pg else OTH)
            ax.add_patch(patches.Rectangle((cb[0] - cb[2] / 2, cb[1] - cb[3] / 2), cb[2], cb[3], fill=False,
                                           edgecolor=col, lw=2.2 if col != OTH else 0.8))
            if j == pc:
                ax.plot(cb[0], cb[1], "x", color=C3, ms=9, mew=2)
        lab = {-1: "dustbin"}
        ax.set_title(f"{c['seq']} | gap {c['gap']} | P(correct)={c['p_correct']:.2f}\n"
                     f"GT={lab.get(gt, gt)}  GapLink={lab.get(pg, pg)}  C-3={lab.get(pc, pc)}  [{c['type']}]", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Lowest-confidence test cases (PREREG §4, 3-3-4 per gap bin). Black: lost-track history; "
                 "green box: true candidate; red box: GapLink choice; blue x: C-3 choice", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=150); plt.close(fig)


def main():
    import torch
    from scripts.run_test import decide_from_comps, decide_with_tau, infer_all_events, precompute_kalman_test
    from src.eval_events import evaluate_predictions
    from src.location_bootstrap import load_seq_to_location
    from src.model.gaplink import GapLinkLite

    root = ROOT
    man = json.loads((root / "release/FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    for rel in ("release/final.pt", "data/cache/test_events.pkl"):
        assert sha256_file(root / rel) == man["files"][rel]["sha256"], f"frozen artifact changed: {rel}"
    frozen = json.loads((root / "release/results_v1.1/test_results.json").read_text(encoding="utf-8"))["event_level"]
    tr = yaml.safe_load((root / "release/tracker.yaml").read_text(encoding="utf-8"))
    events = pickle.loads((root / "data/cache/test_events.pkl").read_bytes())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(root / "release/final.pt", map_location=device, weights_only=False)
    model = GapLinkLite(use_ctx=True, use_color=True, gap_aware=True).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    P_list, _ = infer_all_events(model, events, device)
    pred = {"GapLink-Lite": decide_with_tau(events, P_list, float(tr["gaplink_tau"]))}
    comps, _ = precompute_kalman_test(events)
    pred["C-3"] = decide_from_comps(comps, theta=float(tr["c3_theta"]))
    pred["C-4"] = decide_from_comps(comps, theta=float(tr["c4_theta"]), ls=float(tr["c4_lambda_s"]), lc=float(tr["c4_lambda_c"]))
    by_seq = {}
    for k, p in pred.items():
        r, by_seq[k] = evaluate_predictions(events, p)
        assert abs(r["top1"] - frozen[k]["top1"]) < 1e-9, f"{k}: recomputed top1 {r['top1']} != frozen {frozen[k]['top1']}"

    # 2. taxonomy
    tax = {k: taxonomy(events, p) for k, p in pred.items()}

    # 3. PREREG cases
    rows = []
    for e, (ev, P) in enumerate(zip(events, P_list)):
        for i, pc in enumerate(p_correct(ev, P)):
            rows.append((pc, e, i, int(ev["gap_bin"][i])))
    rows.sort()
    cases, used = [], set()
    for b, k in zip(range(3), PER_BIN):
        n = 0
        for pc, e, i, gb in rows:
            if gb != b or e in used:
                continue
            ev = events[e]
            gt, pg, p3 = int(ev["label"][i]), int(pred["GapLink-Lite"][e][i]), int(pred["C-3"][e][i])
            cases.append({"event_idx": e, "track": i, "seq": ev["seq"], "gap": int(ev["gap"][i]), "gap_bin": BIN_NAMES[b],
                          "p_correct": pc, "gt_cand": gt, "gaplink_pred": pg, "c3_pred": p3, "type": outcome(gt, pg),
                          "c3_correct": p3 == gt, "is_hard": bool(ev["is_hard"])})
            used.add(e); n += 1
            if n == k:
                break

    # 4. per-location Top-1
    s2l = load_seq_to_location(root / "data/splits.csv", split="test")
    per_loc = {}
    for k in ("GapLink-Lite", "C-3", "C-4"):
        for s, arr in by_seq[k].items():
            d = per_loc.setdefault(s2l[s], {}).setdefault(k, [0.0, 0])
            d[0] += float(np.sum(arr)); d[1] += int(len(arr))
    per_loc = {L: {k: v[0] / v[1] for k, v in d.items()} | {"n_tracks": d["GapLink-Lite"][1],
                                                            "n_seqs": sum(1 for s in s2l if s2l[s] == L)}
               for L, d in sorted(per_loc.items())}

    out_dir = root / "runs/test_final_errors"; out_dir.mkdir(parents=True, exist_ok=True)
    fig_p = root / "report/figures/fig_failure_cases.png"
    plot_cases(cases, events, pred["GapLink-Lite"], pred["C-3"], P_list, fig_p)
    lt = root / "report/latex/figures"
    if lt.exists():
        shutil.copy2(fig_p, lt / fig_p.name)
    res = {"source": "release/final.pt + data/cache/test_events.pkl (frozen); tau/theta from release/tracker.yaml",
           "taxonomy": tax, "prereg_cases": cases, "per_location_top1": per_loc}
    (out_dir / "failure_analysis.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    rel_copy = root / "release/results_v1.1_extra"; rel_copy.mkdir(exist_ok=True)
    shutil.copy2(out_dir / "failure_analysis.json", rel_copy / "failure_analysis.json")
    print(json.dumps(tax, indent=1)); print(f"{len(cases)} cases ->", fig_p)


if __name__ == "__main__":
    main()
