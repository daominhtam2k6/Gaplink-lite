"""Gaussian trajectory predictor head and NLL / Mahalanobis log-density computation."""
import math
import torch
import torch.nn as nn

class GaussianPredictor(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, 8),
        )

    def forward(
        self,
        h_i: torch.Tensor,
        ctx_i: torch.Tensor,
        hist: torch.Tensor,
        hist_len: torch.Tensor,
        gap: torch.Tensor,
        gap_aware: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict normalized offset mu (B,N,4) and logvar (B,N,4) relative to last_box.
        Target coordinates:
          target_0 = (cx_true - cx_last) / h_last
          target_1 = (cy_true - cy_last) / h_last
          target_2 = log(w_true / w_last)
          target_3 = log(h_true / h_last)
        Uses constant-velocity residual anchor for fast, stable convergence.
        """
        raw = self.mlp(torch.cat([h_i, ctx_i], dim=-1))
        d_mu = raw[..., :4]
        logvar = raw[..., 4:].clamp(-6.0, 4.0)

        # Constant-velocity anchor over last up to 4 steps
        last_box = hist[:, :, -1, :]
        h_l = last_box[..., 3].clamp(min=8.0)
        prev_box = hist[:, :, -4, :]
        has_4 = (hist_len >= 4).float()
        vx = ((last_box[..., 0] - prev_box[..., 0]) / 3.0) * has_4 / h_l
        vy = ((last_box[..., 1] - prev_box[..., 1]) / 3.0) * has_4 / h_l
        g = gap.float().clamp(min=1.0) if gap_aware else torch.full_like(gap.float(), 20.0)
        cv_dx = (vx * g).clamp(-15.0, 15.0)
        cv_dy = (vy * g).clamp(-15.0, 15.0)

        mu = torch.stack([
            cv_dx + d_mu[..., 0],
            cv_dy + d_mu[..., 1],
            d_mu[..., 2].clamp(-2.0, 2.0),
            d_mu[..., 3].clamp(-2.0, 2.0),
        ], dim=-1)
        return mu, logvar

def target_from_boxes(last_box: torch.Tensor, target_box: torch.Tensor) -> torch.Tensor:
    """Compute normalized 4D target from last_box (..., 4) and target_box (..., 4)."""
    w_l = last_box[..., 2].clamp(min=8.0)
    h_l = last_box[..., 3].clamp(min=8.0)
    t0 = ((target_box[..., 0] - last_box[..., 0]) / h_l).clamp(-20.0, 20.0)
    t1 = ((target_box[..., 1] - last_box[..., 1]) / h_l).clamp(-20.0, 20.0)
    t2 = torch.log(target_box[..., 2].clamp(min=2.0) / w_l).clamp(-3.0, 3.0)
    t3 = torch.log(target_box[..., 3].clamp(min=2.0) / h_l).clamp(-3.0, 3.0)
    return torch.stack([t0, t1, t2, t3], dim=-1)

def decode_predicted_boxes(last_box: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode normalized (mu, logvar) back to pixel-space (cx, cy, w, h) mean and pixel center std (sigma_x, sigma_y)."""
    w_l = last_box[..., 2].clamp(min=8.0)
    h_l = last_box[..., 3].clamp(min=8.0)
    cx = last_box[..., 0] + mu[..., 0] * h_l
    cy = last_box[..., 1] + mu[..., 1] * h_l
    w = w_l * torch.exp(mu[..., 2].clamp(-2.0, 2.0))
    h = h_l * torch.exp(mu[..., 3].clamp(-2.0, 2.0))
    pred_box = torch.stack([cx, cy, w, h], dim=-1)
    std_norm = torch.exp(0.5 * logvar)
    sigma_px = torch.stack([std_norm[..., 0] * h_l, std_norm[..., 1] * h_l, std_norm[..., 2], std_norm[..., 3]], dim=-1)
    return pred_box, sigma_px

def gaussian_nll_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    last_box: torch.Tensor,
    true_box: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    tgt = target_from_boxes(last_box, true_box)
    var = torch.exp(logvar)
    nll = 0.5 * (logvar + ((tgt - mu) ** 2) / var + math.log(2.0 * math.pi)).sum(dim=-1)  # (B, N)
    m = valid_mask.float()
    return (nll * m).sum() / m.sum().clamp(min=1.0)
