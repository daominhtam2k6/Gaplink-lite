import argparse
import json
import math
import pathlib
import pickle
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from scipy.optimize import linear_sum_assignment

from src.dataset import EventListDataset, CurriculumPoolDataset, collate_padded_events
from src.eval_events import evaluate_predictions, bootstrap_cluster
from src.model.gaplink import GapLinkLite

ROOT = pathlib.Path(__file__).resolve().parent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def batch_predict_val(model: GapLinkLite, val_events: list[dict], val_ds: EventListDataset, device: torch.device, tau: float = 0.25):
    model.eval()
    loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_padded_events)
    preds = []
    gaussians = []
    idx_global = 0
    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch_dev)
        P_batch = out["P"].cpu().numpy()
        pred_box_batch = out["pred_box"].cpu().numpy()
        sigma_px_batch = out["sigma_px"].cpu().numpy()
        B = P_batch.shape[0]
        N_pad = P_batch.shape[1] - 1
        M_pad = P_batch.shape[2] - 1
        for b in range(B):
            ev = val_events[idx_global]
            idx_global += 1
            N = min(len(ev["t_lost"]), N_pad)
            M = min(len(ev["cand"]), M_pad)
            dec = np.full((len(ev["t_lost"]),), -1, dtype=np.int32)
            if N > 0 and M > 0:
                P_full = P_batch[b]
                P_real = P_full[:N, :M]
                P_dust = P_full[:N, M_pad]
                r_ind, c_ind = linear_sum_assignment(-P_real)
                for r, c in zip(r_ind, c_ind):
                    if P_real[r, c] >= tau and P_real[r, c] > P_dust[r]:
                        dec[r] = int(c)
            preds.append(dec)
            pb = np.zeros((len(ev["t_lost"]), 4), dtype=np.float32)
            sb = np.ones((len(ev["t_lost"]), 4), dtype=np.float32) * 10.0
            pb[:N] = pred_box_batch[b, :N]
            sb[:N] = sigma_px_batch[b, :N]
            gaussians.append((pb, sb))
    res, seq_arr = evaluate_predictions(val_events, preds, gaussians=gaussians)
    return res, seq_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg_path = ROOT / args.config if not pathlib.Path(args.config).is_absolute() else pathlib.Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 20260417))
    set_seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = ROOT / cfg["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)

    model = GapLinkLite(
        use_ctx=bool(cfg.get("use_ctx", True)),
        use_color=bool(cfg.get("use_color", True)),
        gap_aware=bool(cfg.get("gap_aware", True)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GapLinkLite initialized on {device} with {n_params:,} parameters.")

    val_events_path = ROOT / "data" / "cache" / "val_events.pkl"
    with open(val_events_path, "rb") as f:
        val_events = pickle.load(f)

    w_assign = float(cfg.get("w_assign", 1.0))
    w_nll = float(cfg.get("w_nll", 0.5))

    if "overfit_n" in cfg:
        n_ov = int(cfg["overfit_n"])
        sub_events = val_events[:n_ov]
        sub_ds = EventListDataset(sub_events)
        loader = DataLoader(sub_ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=collate_padded_events)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg.get("epochs", 260)), eta_min=1e-5)
        epochs = int(cfg.get("epochs", 200))
        history = []
        t0 = time.time()
        acc = 0.0
        for ep in range(1, epochs + 1):
            model.train()
            ep_loss = 0.0
            steps = 0
            for batch in loader:
                batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                losses = model.compute_loss(batch_dev, w_assign=w_assign, w_nll=0.15 if 'overfit_n' in cfg else w_nll)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
                opt.step()
                ep_loss += float(losses["loss"].item())
                steps += 1
            sched.step()
            res, _ = batch_predict_val(model, sub_events, sub_ds, device, tau=0.05)
            acc = res["top1"]
            history.append({"epoch": ep, "loss": ep_loss / max(steps, 1), "top1": acc})
            if ep % 20 == 0 or acc >= 0.995:
                print(f"[Overfit] Epoch {ep:03d} | loss={ep_loss/max(steps,1):.4f} | top1_acc={acc*100:.2f}%")
            if acc >= 0.995 and ep >= 35:
                break
        dt = time.time() - t0
        torch.save({"model": model.state_dict(), "cfg": cfg, "top1": acc}, run_dir / "overfit.pt")
        (run_dir / "overfit_metrics.json").write_text(
            json.dumps({"final_top1": acc, "epochs_ran": ep, "elapsed_sec": dt, "n_params": n_params, "history": history}, indent=2),
            encoding="utf-8",
        )
        print(f"Overfit finished in {dt:.1f}s: final_top1={acc*100:.2f}%")
        return

    # Full training mode
    pool_dir = ROOT / "data" / "cache" / "pool" / "train"
    with open(pool_dir / "easy_pool.pkl", "rb") as f:
        easy_events = pickle.load(f)
    with open(pool_dir / "hard_pool.pkl", "rb") as f:
        hard_events = pickle.load(f)

    print(f"Padding training pool ({len(easy_events)} easy, {len(hard_events)} hard)...")
    pool_ds = CurriculumPoolDataset(easy_events, hard_events)
    val_ds = EventListDataset(val_events)

    total_steps = int(cfg.get("steps", 6000))
    batch_size = int(cfg.get("batch_size", 64))
    warmup_steps = int(cfg.get("warmup_steps", 300))
    eval_every = int(cfg.get("eval_every", 500))
    base_lr = float(cfg.get("lr", 5e-4))
    hp_start = float(cfg.get("hard_prob_start", 0.4))
    hp_end = float(cfg.get("hard_prob_end", 0.8))

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=float(cfg.get("weight_decay", 1e-4)))

    best_top1 = -1.0
    log_rows = []
    t0 = time.time()

    for step in range(1, total_steps + 1):
        frac = step / float(total_steps)
        hard_prob = hp_start + (hp_end - hp_start) * min(1.0, frac / 0.6)

        if step <= warmup_steps:
            lr = base_lr * (step / float(max(warmup_steps, 1)))
        else:
            prog = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        batch = pool_ds.sample_batch(batch_size, hard_prob=hard_prob, rng=rng)
        batch_dev = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        model.train()
        opt.zero_grad(set_to_none=True)
        losses = model.compute_loss(batch_dev, w_assign=w_assign, w_nll=0.15 if 'overfit_n' in cfg else w_nll)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
        opt.step()

        if step % 200 == 0 or step == 1:
            print(
                f"[{cfg['run_dir']}] step {step:04d}/{total_steps} | "
                f"loss={losses['loss'].item():.4f} (assign={losses['loss_assign'].item():.4f}, nll={losses['loss_nll'].item():.4f}) | "
                f"lr={lr:.6f} | hard_p={hard_prob:.2f} | elapsed={time.time()-t0:.1f}s"
            )

        if step % eval_every == 0 or step == total_steps:
            res, seq_arr = batch_predict_val(model, val_events, val_ds, device, tau=0.25)
            lo, hi = bootstrap_cluster(seq_arr)
            res["top1_ci95"] = [lo, hi]
            top1 = res["top1"]
            row = {
                "step": step,
                "loss": float(losses["loss"].item()),
                "loss_assign": float(losses["loss_assign"].item()),
                "loss_nll": float(losses["loss_nll"].item()),
                "val_top1": top1,
                "val_top1_hard": res["top1_hard"],
                "val_dustbin_F1": res["dustbin_F1"],
                "val_coverage90": res.get("coverage90", 0.0),
                "val_metrics": res,
            }
            log_rows.append(row)
            print(
                f"  --> [VAL step {step}] top1={top1*100:.2f}% (hard={res['top1_hard']*100:.2f}%) | "
                f"dust_F1={res['dustbin_F1']*100:.2f}% | WLR={res['wrong_link_rate']*100:.2f}% | cov90={res.get('coverage90',0)*100:.1f}%"
            )
            ckpt = {"model": model.state_dict(), "cfg": cfg, "step": step, "val_metrics": res, "n_params": n_params}
            torch.save(ckpt, run_dir / "last.pt")
            if top1 >= best_top1:
                best_top1 = top1
                torch.save(ckpt, run_dir / "best.pt")

    best_ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    (run_dir / "train_log.json").write_text(
        json.dumps({"cfg": cfg, "n_params": n_params, "best_val": best_ckpt["val_metrics"], "history": log_rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Finished {cfg['run_dir']}! Best val top1={best_ckpt['val_metrics']['top1']*100:.2f}% (step {best_ckpt['step']})")


if __name__ == "__main__":
    main()
