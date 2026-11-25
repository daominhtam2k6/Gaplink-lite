"""v1.3: detector-aware fine-tuning of GapLink-Lite for the live (YOLOv8n) setting.

Starts from release/final.pt (our own randomly-initialised v1.1 weights; no external pretrained model),
trains on a mixture of detector-aware TRAIN events (data/cache/det_events_train.pkl) and the original
synthetic train pool (to keep GT-box skills). Checkpoint is selected on detector-aware VAL events only.

    python -m scripts.finetune_live --seed 20261121
Output: runs/live_v13_s<seed>/{best.pt,last.pt,train_log.json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import collate_padded_events, pad_event_to_fixed  # noqa: E402
from src.eval_events import evaluate_predictions  # noqa: E402
from src.model.gaplink import GapLinkLite  # noqa: E402

CFG = {"steps": 3000, "batch_size": 64, "lr": 2e-4, "weight_decay": 1e-4, "warmup_steps": 100, "grad_clip": 1.0,
       "w_assign": 1.0, "w_nll": 0.5, "p_det": 0.7, "eval_every": 250, "eval_tau": 0.5}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@torch.no_grad()
def eval_events(model, padded, events, device, tau):
    from scripts.run_test import decide_with_tau
    model.eval()
    P_list = []
    for s in range(0, len(padded), 256):
        chunk = padded[s:s + 256]
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in collate_padded_events(chunk).items()}
        P = model(b)["P"].cpu().numpy()
        for k, ev in enumerate(events[s:s + 256]):
            N, M = min(len(ev["t_lost"]), 6), min(len(ev["cand"]), 10)
            P_list.append(P[k][list(range(N)) + [6]][:, list(range(M)) + [10]])
    evs = [{**e, "t_lost": e["t_lost"][:6], "cand": e["cand"][:10],
            "label": np.where(e["label"][:6] >= 10, -1, e["label"][:6]), "gap_bin": e["gap_bin"][:6]} for e in events]
    preds = decide_with_tau(evs, P_list, tau)
    res, _ = evaluate_predictions(evs, preds)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--steps", type=int, default=CFG["steps"])
    a = ap.parse_args()
    cfg = {**CFG, "steps": a.steps, "seed": a.seed}
    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text(encoding="utf-8"))
    root = ROOT / paths["root_dir"] if (ROOT / paths["root_dir"]).exists() else ROOT
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init = root / "release" / "final.pt"
    ck = torch.load(init, map_location=device, weights_only=False)
    model = GapLinkLite(use_ctx=True, use_color=True, gap_aware=True).to(device)
    model.load_state_dict(ck["model"])
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 1_000_000

    det_tr = pickle.loads((root / "data/cache/det_events_train.pkl").read_bytes())
    det_va = pickle.loads((root / "data/cache/det_events_val.pkl").read_bytes())
    pool_dir = root / "data/cache/pool/train"
    pool = pickle.loads((pool_dir / "easy_pool.pkl").read_bytes()) + pickle.loads((pool_dir / "hard_pool.pkl").read_bytes())
    print(f"det train {len(det_tr)} | det val {len(det_va)} | pool {len(pool)} | params {n_params:,}", flush=True)
    det_va_p = [pad_event_to_fixed(e) for e in det_va]

    run_dir = root / "runs" / f"live_v13_s{a.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    log = []
    r0 = eval_events(model, det_va_p, det_va, device, cfg["eval_tau"])
    log.append({"step": 0, "val_top1": r0["top1"], "val_wlr": r0["wrong_link_rate"], "val_dust_F1": r0["dustbin_F1"]})
    print(f"[step 0 = final.pt] det-val top1={r0['top1']*100:.2f}% WLR={r0['wrong_link_rate']*100:.2f}%", flush=True)
    best = (r0["top1"], 0)
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": 0, "val": r0}, run_dir / "best.pt")
    t0 = time.time()
    B = cfg["batch_size"]
    for step in range(1, cfg["steps"] + 1):
        lr = cfg["lr"] * (step / cfg["warmup_steps"] if step <= cfg["warmup_steps"] else
                          0.5 * (1 + math.cos(math.pi * (step - cfg["warmup_steps"]) / (cfg["steps"] - cfg["warmup_steps"]))))
        for g in opt.param_groups:
            g["lr"] = lr
        n_det = int(rng.binomial(B, cfg["p_det"]))
        items = [pad_event_to_fixed(det_tr[i]) for i in rng.integers(0, len(det_tr), n_det)]
        items += [pad_event_to_fixed(pool[i]) for i in rng.integers(0, len(pool), B - n_det)]
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in collate_padded_events(items).items()}
        model.train()
        opt.zero_grad(set_to_none=True)
        L = model.compute_loss(b, w_assign=cfg["w_assign"], w_nll=cfg["w_nll"])
        L["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            r = eval_events(model, det_va_p, det_va, device, cfg["eval_tau"])
            log.append({"step": step, "loss": float(L["loss"]), "val_top1": r["top1"], "val_wlr": r["wrong_link_rate"],
                        "val_dust_F1": r["dustbin_F1"]})
            print(f"[step {step}] loss={float(L['loss']):.4f} det-val top1={r['top1']*100:.2f}% "
                  f"WLR={r['wrong_link_rate']*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)
            ckpt = {"model": model.state_dict(), "cfg": cfg, "step": step, "val": r}
            torch.save(ckpt, run_dir / "last.pt")
            if r["top1"] > best[0]:
                best = (r["top1"], step)
                torch.save(ckpt, run_dir / "best.pt")
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (run_dir / "train_log.json").write_text(json.dumps(
        {"cfg": cfg, "init": str(init), "init_sha256": sha256_file(init), "git_commit": commit, "n_params": n_params,
         "best_step": best[1], "best_val_top1": best[0], "history": log}, indent=2), encoding="utf-8")
    print(f"best step {best[1]} det-val top1={best[0]*100:.2f}% -> {run_dir/'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
