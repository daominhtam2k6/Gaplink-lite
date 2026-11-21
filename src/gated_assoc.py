"""Gaussian-gated GapLink decision rule (v1.2 live candidate).

GapLink-Lite already predicts a gap-conditioned 2D Gaussian N(mu, diag(sigma^2)) for each lost track.
v1.1 used it only as a feature for the Sinkhorn scorer; a link was accepted when P >= tau and P > P_dustbin.
The gated rule additionally vetoes a link whose candidate centre falls outside the model's own
chi-square ellipse:  d2 = ((cx - mu_x)/sigma_x)^2 + ((cy - mu_y)/sigma_y)^2 <= chi2_gate.
No retraining, same checkpoint; only the decision rule changes. Gate levels are fixed in
docs/PREREG_live_v1.2.md before evaluation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

CHI2_2D = {"95": 5.991464547107979, "99": 9.21034037197618, "99.9": 13.815510557964274}


def mahalanobis2_centre(pred_box: np.ndarray, sigma_px: np.ndarray, cand: np.ndarray, min_sigma: float = 1.0) -> np.ndarray:
    """(N, M) squared normalised centre distance between N predicted boxes and M candidate boxes (cx, cy, w, h)."""
    sx = np.maximum(min_sigma, sigma_px[:, 0])[:, None]
    sy = np.maximum(min_sigma, sigma_px[:, 1])[:, None]
    dx = (cand[None, :, 0] - pred_box[:, None, 0]) / sx
    dy = (cand[None, :, 1] - pred_box[:, None, 1]) / sy
    return dx * dx + dy * dy


def gated_decision(P: np.ndarray, d2: np.ndarray, tau: float, chi2_gate: float | None) -> np.ndarray:
    """P: (N+1, M+1) with dustbin row/col last. Returns (N,) candidate index or -1."""
    N, M = d2.shape
    out = np.full((N,), -1, dtype=np.int32)
    if N == 0 or M == 0:
        return out
    P_real = P[:N, :M].copy()
    P_dust = P[:N, M]
    if chi2_gate is not None:
        P_real[d2 > chi2_gate] = 0.0  # outside the model's own ellipse: not assignable
    r_ind, c_ind = linear_sum_assignment(-P_real)
    for r, c in zip(r_ind, c_ind):
        if P_real[r, c] >= tau and P_real[r, c] > P_dust[r]:
            out[r] = int(c)
    return out


def make_gated_associator(model, device: str, tau: float, chi2_gate: float | None):
    from src.model.gaplink import GapLinkAssociator

    class GatedGapLinkAssociator(GapLinkAssociator):
        name = "GapLink-Lite-gated"

        def __init__(self):
            super().__init__(model, device=device, tau=tau, calib=None)
            self.chi2_gate = chi2_gate

        def decide(self, ev: dict) -> np.ndarray:
            N, M = len(ev["t_lost"]), len(ev["cand"])
            if N == 0 or M == 0:
                return np.full((N,), -1, dtype=np.int32)
            res = self.infer_event(ev)
            d2 = mahalanobis2_centre(res["pred_box"], res["sigma_px"], np.asarray(ev["cand"], dtype=np.float32))
            return gated_decision(res["P"], d2, self.tau, self.chi2_gate)

    return GatedGapLinkAssociator()
