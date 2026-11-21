"""Out-of-domain exploratory evaluation on VisDrone2019-MOT (docs/PREREG_ood_motor.md).

  --make-events : build fixed events per group (motor / car) from GT tracks, write pkl + sha256
  --eval        : SINGLE run. Frozen release/final.pt + Phase-1 operating points (release/tracker.yaml).
                  Refuses to run twice (runs/ood_visdrone/results.json exists).
No training, no threshold change, no sequence filtering beyond what the PREREG states.
"""
import argparse
import csv
import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.convert_visdrone import load_root

GROUPS = ("car", "motor")          # car = control, motor = primary
N_EVENTS = 3000
SEEDS = {"car": 9200, "motor": 9100}


def sha256_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_seqs(root: Path) -> list[dict]:
    with open(root / "data" / "ood_visdrone" / "seqs.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_events(root: Path) -> None:
    from scripts.make_fixed_events import generate_split_events
    seqs = read_seqs(root)
    for g in GROUPS:
        base = root / "data" / "ood_visdrone" / g
        names = [r["seq"] for r in seqs if int(r[f"n_tracks_{g}"]) >= 2]
        ev = generate_split_events(names, N_EVENTS, SEEDS[g], base / "mot", base / "hist")
        p = base / "events.pkl"
        if p.exists():
            raise SystemExit(f"{p} already exists - events are fixed once; delete only if PREREG allows.")
        with open(p, "wb") as f:
            pickle.dump(ev, f)
        (base / "events.sha256").write_text(sha256_file(p), encoding="utf-8")
        print(f"[{g}] {len(ev)} events from {len(names)} sequences -> {p}")


def boot_ratio(num: dict, den: dict, n_boot: int = 2000, seed: int = 0) -> list[float]:
    keys = sorted(den)
    a = np.array([num[k] for k in keys], float)
    b = np.array([den[k] for k in keys], float)
    if len(keys) < 2:
        v = float(a.sum() / max(1, b.sum()))
        return [v, v]
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(n_boot, len(keys)))
    r = a[idx].sum(1) / np.maximum(1.0, b[idx].sum(1))
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def boot_ratio_diff(numA, numB, den, n_boot=2000, seed=0) -> dict:
    keys = sorted(den)
    a, c, b = (np.array([d[k] for k in keys], float) for d in (numA, numB, den))
    delta = float((a.sum() - c.sum()) / max(1, b.sum()))
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(n_boot, len(keys)))
    d = (a[idx].sum(1) - c[idx].sum(1)) / np.maximum(1.0, b[idx].sum(1))
    return {"delta": delta, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def run_eval(root: Path, skip_tracker: bool) -> None:
    import torch
    import yaml
    from scripts.run_test import infer_all_events, decide_with_tau, precompute_kalman_test, decide_from_comps
    from src.baselines import ConstantVelocityBaseline, KalmanMahalanobisBaseline
    from src.calib import apply_calibration
    from src.eval_events import evaluate_predictions, bootstrap_cluster, bootstrap_paired
    from src.inject import make_hide_plan, make_detections
    from src.io_mot import load_seq
    from src.metrics_tracker import evaluate_tracker_on_seq
    from src.model.gaplink import GapLinkLite, GapLinkAssociator

    out_dir = root / "runs" / "ood_visdrone"
    if (out_dir / "results.json").exists():
        raise SystemExit("runs/ood_visdrone/results.json exists - the OOD test is single-shot.")
    out_dir.mkdir(parents=True, exist_ok=True)
    rel = root / "release"
    ckpt_path = rel / "final.pt"
    sha = sha256_file(ckpt_path)
    assert sha == (rel / "final.sha256").read_text(encoding="utf-8").strip(), "final.pt SHA mismatch"
    tcfg = yaml.safe_load((rel / "tracker.yaml").read_text(encoding="utf-8"))
    calib = json.loads((rel / "calib.json").read_text(encoding="utf-8"))
    tau, th2, th3, th4 = (float(tcfg[k]) for k in ("gaplink_tau", "c2_theta", "c3_theta", "c4_theta"))
    ls4, lc4 = float(tcfg["c4_lambda_s"]), float(tcfg["c4_lambda_c"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck["cfg"]
    model = GapLinkLite(use_ctx=bool(c.get("use_ctx", True)), use_color=bool(c.get("use_color", True)),
                        gap_aware=bool(c.get("gap_aware", True))).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    seqs = read_seqs(root)
    static = {r["seq"]: (r["static"] == "True") for r in seqs}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        commit = "unknown"
    results = {"prereg": "docs/PREREG_ood_motor.md", "final_sha256": sha, "git_commit": commit,
               "operating_points": {"gaplink_tau": tau, "c2_theta": th2, "c3_theta": th3, "c4_theta": th4,
                                    "c4_lambda_s": ls4, "c4_lambda_c": lc4},
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "groups": {}}

    for g in GROUPS:
        base = root / "data" / "ood_visdrone" / g
        ev_p = base / "events.pkl"
        assert sha256_file(ev_p) == (base / "events.sha256").read_text(encoding="utf-8").strip(), f"{g} events SHA mismatch"
        with open(ev_p, "rb") as f:
            events = pickle.load(f)
        print(f"[{g}] event-level on {len(events)} events", flush=True)
        preds = {"C-2": [ConstantVelocityBaseline(theta=th2).decide(e) for e in events]}
        comps, gk = precompute_kalman_test(events)
        preds["C-3"] = decide_from_comps(comps, theta=th3)
        preds["C-4"] = decide_from_comps(comps, theta=th4, ls=ls4, lc=lc4)
        P, raw_g = infer_all_events(model, events, device)
        preds["GapLink-Lite"] = decide_with_tau(events, P, tau=tau)
        gl_g = [(pb, apply_calibration(sb, e["gap"], calib)) for e, (pb, sb) in zip(events, raw_g)]

        gres, seqarr = {}, {}
        for m, pr in preds.items():
            r, sa = evaluate_predictions(events, pr, gaussians=gl_g if m == "GapLink-Lite" else gk)
            r["top1_ci95"] = list(bootstrap_cluster(sa))
            for stratum, flag in (("static", True), ("moving", False)):
                ids = [i for i, e in enumerate(events) if static.get(e["seq"], False) == flag]
                if ids:
                    rs, _ = evaluate_predictions([events[i] for i in ids], [pr[i] for i in ids])
                    r[f"top1_{stratum}"], r[f"n_tracks_{stratum}"] = rs["top1"], rs["n_tracks"]
            gres[m], seqarr[m] = r, sa
        paired = {f"GapLink_vs_{b}": bootstrap_paired(seqarr["GapLink-Lite"], seqarr[b]) for b in ("C-2", "C-3", "C-4")}
        grp = {"n_events": len(events), "n_seqs": len(seqarr["C-3"]), "event_level": gres, "paired_top1": paired}

        if not skip_tracker:
            assoc = {"C-1": None, "C-2": ConstantVelocityBaseline(theta=th2),
                     "C-3": KalmanMahalanobisBaseline(theta=th3, name="C-3"),
                     "C-4": KalmanMahalanobisBaseline(theta=th4, lambda_s=ls4, lambda_c=lc4, name="C-4"),
                     "GapLink-Lite": GapLinkAssociator(model, device=str(device), tau=tau, calib=calib)}
            names = [r["seq"] for r in seqs if int(r[f"n_tracks_{g}"]) >= 2]
            trk = {}
            for mode in ("full", "partial"):
                cnt = {m: {"n": {}, "rec": {}, "wrong": {}} for m in assoc}
                for i_s, s in enumerate(names):
                    seq = load_seq(base / "mot", s)
                    hp = base / "hist" / f"{s}.npy"
                    hc = np.load(hp) if hp.exists() else None
                    plan = make_hide_plan(seq, seed=7000 + i_s, max_density=0.18)
                    dets = make_detections(seq, plan, hc, mode=mode, seed=8000 + i_s)
                    for m, a in assoc.items():
                        rs = evaluate_tracker_on_seq(seq, plan, dets, associator=a)
                        n = rs["n_intervals"]
                        cnt[m]["n"][s] = n
                        cnt[m]["rec"][s] = int(round(rs["IRR"] * n))
                        cnt[m]["wrong"][s] = int(round(rs["WLR"] * n))
                    print(f"[{g}/{mode}] {i_s + 1}/{len(names)} {s}", flush=True)
                mres = {}
                for m, d in cnt.items():
                    n = sum(d["n"].values())
                    mres[m] = {"n_intervals": n, "IRR": sum(d["rec"].values()) / max(1, n),
                               "WLR": sum(d["wrong"].values()) / max(1, n),
                               "IRR_ci95": boot_ratio(d["rec"], d["n"]), "WLR_ci95": boot_ratio(d["wrong"], d["n"])}
                for b in ("C-2", "C-3"):
                    mres[f"IRR_GapLink_minus_{b}"] = boot_ratio_diff(cnt["GapLink-Lite"]["rec"], cnt[b]["rec"], cnt[b]["n"])
                    mres[f"WLR_GapLink_minus_{b}"] = boot_ratio_diff(cnt["GapLink-Lite"]["wrong"], cnt[b]["wrong"], cnt[b]["n"])
                trk[mode] = mres
            grp["tracker_level"] = trk
        results["groups"][g] = grp

    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# OOD VisDrone results (single run)", "", f"final.pt SHA `{sha[:12]}`, commit `{commit[:8]}`", ""]
    for g, grp in results["groups"].items():
        lines += [f"## {g} ({grp['n_events']} events, {grp['n_seqs']} seqs)", "",
                  "| Method | Top-1 | 95% CI | WLR | Top-1 static | Top-1 moving |", "|---|---|---|---|---|---|"]
        for m, r in grp["event_level"].items():
            lo, hi = r["top1_ci95"]
            lines.append(f"| {m} | {r['top1']*100:.2f}% | [{lo*100:.2f}, {hi*100:.2f}] | {r['wrong_link_rate']*100:.2f}% | "
                         f"{r.get('top1_static', float('nan'))*100:.2f}% | {r.get('top1_moving', float('nan'))*100:.2f}% |")
        for k, v in grp["paired_top1"].items():
            lines.append(f"- {k}: {json.dumps(v)}")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-events", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--skip-tracker", action="store_true")
    args = ap.parse_args()
    root = load_root()
    if args.make_events:
        make_events(root)
    if args.eval:
        run_eval(root, args.skip_tracker)


if __name__ == "__main__":
    main()
