"""Post-hoc uncertainty calibration for GapLink-Lite (gap-dependent log-variance scaling)."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

CHI2_2D_90 = 4.605170185988092


def apply_calibration(sigma_px: np.ndarray, gap: np.ndarray, params: dict) -> np.ndarray:
    """Scale (sigma_x, sigma_y) by exp(a + b*log(1+g) + c*log(1+g)^2)."""
    g = np.maximum(1.0, gap.astype(np.float64))
    lg = np.log1p(g)
    a = float(params.get("a", 0.0))
    b = float(params.get("b", 0.0))
    c = float(params.get("c", 0.0))
    scale = np.exp(a + b * lg + c * (lg ** 2)).astype(np.float32)
    out = sigma_px.copy().astype(np.float32)
    out[:, 0] *= scale
    out[:, 1] *= scale
    return out


def compute_calib_stats(events: list[dict], raw_gaussians: list[tuple[np.ndarray, np.ndarray]], params: dict) -> dict:
    """Evaluate 90% 2D coverage and 2D NLL across all tracks with true targets and per gap bin."""
    nll_list = []
    cov_hit = 0
    cov_tot = 0
    by_gap_hit = [0, 0, 0]
    by_gap_tot = [0, 0, 0]

    for ev, (pred_box, sigma_px) in zip(events, raw_gaussians):
        if "true_box" not in ev or "has_true_target" not in ev:
            continue
        has_t = ev["has_true_target"]
        tb = ev["true_box"]
        gb = ev["gap_bin"]
        sig_cal = apply_calibration(sigma_px, ev["gap"], params)
        for i in range(len(has_t)):
            if not has_t[i]:
                continue
            b = int(gb[i])
            sx = max(0.5, float(sig_cal[i, 0]))
            sy = max(0.5, float(sig_cal[i, 1]))
            dx = float(tb[i, 0] - pred_box[i, 0])
            dy = float(tb[i, 1] - pred_box[i, 1])
            d2 = (dx / sx) ** 2 + (dy / sy) ** 2
            nll = 0.5 * (2.0 * math.log(sx) + 2.0 * math.log(sy) + d2 + 2.0 * math.log(2.0 * math.pi))
            nll_list.append(nll)
            cov_tot += 1
            by_gap_tot[b] += 1
            if d2 <= CHI2_2D_90:
                cov_hit += 1
                by_gap_hit[b] += 1

    return {
        "n_targets": cov_tot,
        "coverage90": float(cov_hit / max(1, cov_tot)),
        "coverage90_gap_short": float(by_gap_hit[0] / max(1, by_gap_tot[0])),
        "coverage90_gap_med": float(by_gap_hit[1] / max(1, by_gap_tot[1])),
        "coverage90_gap_long": float(by_gap_hit[2] / max(1, by_gap_tot[2])),
        "nll_2d": float(np.mean(nll_list)) if nll_list else 0.0,
    }


def fit_gap_calibration(events_fit: list[dict], raw_gaussians_fit: list[tuple[np.ndarray, np.ndarray]], target_cov: float = 0.90) -> dict:
    """Fit (a, b, c) on validation half-A so that 2D NLL is minimized and per-gap-bin 90% coverage matches 0.90."""
    dx_list, dy_list, sx_list, sy_list, lg_list, gb_list = [], [], [], [], [], []
    for ev, (pred_box, sigma_px) in zip(events_fit, raw_gaussians_fit):
        has_t = ev.get("has_true_target", [])
        tb = ev.get("true_box", [])
        gb = ev.get("gap_bin", [])
        gaps = ev.get("gap", [])
        for i in range(len(has_t)):
            if not has_t[i]:
                continue
            dx_list.append(float(tb[i, 0] - pred_box[i, 0]))
            dy_list.append(float(tb[i, 1] - pred_box[i, 1]))
            sx_list.append(max(0.5, float(sigma_px[i, 0])))
            sy_list.append(max(0.5, float(sigma_px[i, 1])))
            lg_list.append(math.log1p(max(1.0, float(gaps[i]))))
            gb_list.append(int(gb[i]))

    dx = np.array(dx_list, dtype=np.float64)
    dy = np.array(dy_list, dtype=np.float64)
    sx0 = np.array(sx_list, dtype=np.float64)
    sy0 = np.array(sy_list, dtype=np.float64)
    lg = np.array(lg_list, dtype=np.float64)
    gb = np.array(gb_list, dtype=np.int32)

    d2_raw = (dx / sx0) ** 2 + (dy / sy0) ** 2

    # First find empirical 90th-percentile scale factor per gap bin to anchor coverage near 90%
    bin_scales = []
    bin_lgs = []
    for b in (0, 1, 2):
        mask = (gb == b)
        if mask.sum() >= 10:
            q90 = float(np.percentile(d2_raw[mask], target_cov * 100.0))
            s_req = math.sqrt(max(1e-4, q90 / CHI2_2D_90))
            bin_scales.append(math.log(s_req))
            bin_lgs.append(float(lg[mask].mean()))

    if len(bin_scales) == 3:
        X = np.column_stack([np.ones(3), bin_lgs, np.array(bin_lgs) ** 2])
        coef, *_ = np.linalg.lstsq(X, np.array(bin_scales), rcond=None)
        x0 = coef
    else:
        q90_all = float(np.percentile(d2_raw, target_cov * 100.0))
        x0 = np.array([0.5 * math.log(max(1e-4, q90_all / CHI2_2D_90)), 0.0, 0.0])

    # Refine with smooth coverage + NLL objective via Nelder-Mead
    def obj(params_vec):
        a, b_p, c_p = params_vec
        log_s = a + b_p * lg + c_p * (lg ** 2)
        s2 = np.exp(2.0 * np.clip(log_s, -3.0, 3.0))
        d2 = d2_raw / s2
        nll = np.mean(2.0 * log_s + 0.5 * d2)
        # Smooth sigmoid coverage penalty per bin
        cov_pen = 0.0
        for b_idx in (0, 1, 2):
            m = (gb == b_idx)
            if m.sum() > 0:
                soft_cov = np.mean(1.0 / (1.0 + np.exp((d2[m] - CHI2_2D_90) * 2.0)))
                cov_pen += (soft_cov - target_cov) ** 2
        soft_all = np.mean(1.0 / (1.0 + np.exp((d2 - CHI2_2D_90) * 2.0)))
        cov_pen += 2.0 * ((soft_all - target_cov) ** 2)
        return float(0.05 * nll + 50.0 * cov_pen)

    res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 600})
    a_opt, b_opt, c_opt = [float(v) for v in res.x]
    return {"a": a_opt, "b": b_opt, "c": c_opt}
