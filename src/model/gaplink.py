import numpy as np
"""Complete GapLink-Lite network (< 1M parameters, randomly initialized)."""
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from src.model.motion import MotionEncoder
from src.model.context import ContextEncoder
from src.model.predictor import GaussianPredictor, gaussian_nll_loss, decode_predicted_boxes
from src.model.assign import PairwiseAndDustbinScorer
from src.model.sinkhorn import log_sinkhorn, assignment_loss
from src.dataset import pad_event_to_fixed, collate_padded_events

class GapLinkLite(nn.Module):
    def __init__(
        self,
        d: int = 128,
        use_ctx: bool = True,
        use_color: bool = True,
        gap_aware: bool = True,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.d = d
        self.use_ctx = use_ctx
        self.use_color = use_color
        self.gap_aware = gap_aware
        self.sinkhorn_iters = sinkhorn_iters

        self.motion_enc = MotionEncoder(d=d)
        self.ctx_enc = ContextEncoder(d=d)
        self.predictor = GaussianPredictor(d=d)
        self.scorer = PairwiseAndDustbinScorer(d=d)

    def forward(self, batch: dict) -> dict:
        hist = batch["hist"]
        hist_len = batch["hist_len"]
        gap = batch["gap"]
        W = batch["W"]
        H = batch["H"]
        tm = batch["trk_mask"]
        cm = batch["cand_mask"]

        h_i, g_emb, last_box = self.motion_enc(
            hist, hist_len, gap, W, H, gap_aware=self.gap_aware
        )
        ctx_i = self.ctx_enc(
            batch["ctx_lost"],
            batch["ctx_conf"],
            batch["ctx_mask_lost"],
            batch["ctx_mask_conf"],
            use_ctx=self.use_ctx,
        )
        mu, logvar = self.predictor(
            h_i, ctx_i, hist, hist_len, gap, gap_aware=self.gap_aware
        )
        S, z_t, z_c = self.scorer(
            h_i=h_i,
            ctx_i=ctx_i,
            g_emb=g_emb,
            last_box=last_box,
            mu=mu,
            logvar=logvar,
            track_color=batch["track_color"],
            cand=batch["cand"],
            cand_hist=batch["cand_hist"],
            side=batch.get("side", torch.zeros_like(gap)),
            W=W,
            H=H,
            use_color=self.use_color,
            training=self.training,
        )
        logP = log_sinkhorn(S, z_t, z_c, tm, cm, iters=self.sinkhorn_iters)
        pred_box, sigma_px = decode_predicted_boxes(last_box, mu, logvar)
        return {
            "logP": logP,
            "P": torch.exp(logP),
            "mu": mu,
            "logvar": logvar,
            "pred_box": pred_box,
            "sigma_px": sigma_px,
            "last_box": last_box,
            "S": S,
        }

    def compute_loss(self, batch: dict, out: dict | None = None, w_assign: float = 1.0, w_nll: float = 0.5) -> dict:
        if out is None:
            out = self.forward(batch)
        l_assign = assignment_loss(
            out["logP"],
            batch["label"],
            batch["cand_label"],
            batch["trk_mask"],
            batch["cand_mask"],
        )
        valid_nll = batch["trk_mask"] & batch["has_true_target"]
        l_nll = gaussian_nll_loss(
            out["mu"],
            out["logvar"],
            out["last_box"],
            batch["true_box"],
            valid_nll,
        )
        total = w_assign * l_assign + w_nll * l_nll
        return {"loss": total, "loss_assign": l_assign, "loss_nll": l_nll}

class GapLinkAssociator:
    """Inference wrapper matching the Associator interface (.decide(ev) and .infer_event(ev))."""
    name = "GapLink-Lite"

    def __init__(self, model: GapLinkLite, device: str = "cuda", tau: float = 0.5, calib: dict | None = None):
        self.model = model.to(device).eval()
        self.device = device
        self.tau = float(tau)
        self.calib = calib or {"a": 0.0, "b": 0.0, "c": 0.0}

    @torch.no_grad()
    def infer_event(self, ev: dict) -> dict:
        N = len(ev["t_lost"])
        M = len(ev["cand"])
        padded = pad_event_to_fixed(ev, n_max=max(6, N), m_max=max(10, M))
        batch = collate_padded_events([padded])
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = self.model(batch)
        P_full = out["P"][0].cpu().numpy()
        # Extract (N+1, M+1) submatrix: rows 0..N-1 + row N_pad, cols 0..M-1 + col M_pad
        N_pad = P_full.shape[0] - 1
        M_pad = P_full.shape[1] - 1
        row_idx = list(range(N)) + [N_pad]
        col_idx = list(range(M)) + [M_pad]
        P = P_full[row_idx][:, col_idx]

        pred_box = out["pred_box"][0, :N].cpu().numpy()
        sigma_px = out["sigma_px"][0, :N].cpu().numpy()
        mu_norm = out["mu"][0, :N].cpu().numpy()
        logvar_norm = out["logvar"][0, :N].cpu().numpy()

        # Apply gap-dependent calibration factor to display/gate sigma
        g = ev["gap"].astype(np.float32)
        lg = np.log1p(np.maximum(1.0, g))
        scale = np.exp(self.calib.get("a", 0.0) + self.calib.get("b", 0.0) * lg + self.calib.get("c", 0.0) * (lg ** 2))
        sigma_cal = sigma_px.copy()
        sigma_cal[:, 0] *= scale
        sigma_cal[:, 1] *= scale

        return {
            "P": P,
            "pred_box": pred_box,
            "sigma_px": sigma_px,
            "sigma_cal": sigma_cal,
            "mu_norm": mu_norm,
            "logvar_norm": logvar_norm,
        }

    def decide(self, ev: dict) -> np.ndarray:
        N = len(ev["t_lost"])
        M = len(ev["cand"])
        out_arr = np.full((N,), -1, dtype=np.int32)
        if N == 0 or M == 0:
            return out_arr
        res = self.infer_event(ev)
        P_real = res["P"][:N, :M]
        P_dust = res["P"][:N, M]
        r_ind, c_ind = linear_sum_assignment(-P_real)
        for r, c in zip(r_ind, c_ind):
            if P_real[r, c] >= self.tau and P_real[r, c] > P_dust[r]:
                out_arr[r] = int(c)
        return out_arr
