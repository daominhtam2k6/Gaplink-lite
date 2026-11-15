"""Feature extraction and normalization inside GapLink-Lite (with safe clipping)."""
import math
import torch
import torch.nn as nn

def build_motion_history_features(hist: torch.Tensor, hist_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw history (B, N, K, 4) in (cx,cy,w,h) into 6D normalized step features.
    Returns:
      step_feats: (B, N, K, 6) = [dx/h_last, dy/h_last, log(w/w_last), log(h/h_last), vx/h_last, vy/h_last]
      last_box:   (B, N, 4)    = hist[:, :, -1, :]
    """
    B, N, K, _ = hist.shape
    last_box = hist[:, :, -1, :]  # right-aligned history: last valid box is at index K-1
    cx_l = last_box[..., 0:1]
    cy_l = last_box[..., 1:2]
    w_l = last_box[..., 2:3].clamp(min=8.0)
    h_l = last_box[..., 3:4].clamp(min=8.0)

    # Mask valid positions: right-aligned [K - L : K]
    idx = torch.arange(K, device=hist.device).view(1, 1, K)
    valid_step = idx >= (K - hist_len.unsqueeze(-1).clamp(min=1))

    cx = hist[..., 0]
    cy = hist[..., 1]
    w = hist[..., 2].clamp(min=2.0)
    h = hist[..., 3].clamp(min=2.0)

    dx = ((cx - cx_l) / h_l).clamp(-15.0, 15.0)
    dy = ((cy - cy_l) / h_l).clamp(-15.0, 15.0)
    lw = torch.log(w / w_l).clamp(-3.0, 3.0)
    lh = torch.log(h / h_l).clamp(-3.0, 3.0)

    vx = torch.zeros_like(cx)
    vy = torch.zeros_like(cy)
    vx[..., 1:] = ((cx[..., 1:] - cx[..., :-1]) / h_l).clamp(-5.0, 5.0)
    vy[..., 1:] = ((cy[..., 1:] - cy[..., :-1]) / h_l).clamp(-5.0, 5.0)
    # Zero velocity at first valid step
    first_valid = idx == (K - hist_len.unsqueeze(-1).clamp(min=1))
    vx = vx.masked_fill(first_valid | (~valid_step), 0.0)
    vy = vy.masked_fill(first_valid | (~valid_step), 0.0)

    feats = torch.stack([dx, dy, lw, lh, vx, vy], dim=-1)
    feats = feats * valid_step.unsqueeze(-1).float()
    return feats, last_box

class GapEmbedding(nn.Module):
    """Fourier encoding of log(1 + g) -> MLP -> 32D."""
    def __init__(self, out_dim: int = 32, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs
        freqs = 2.0 ** torch.linspace(0.0, n_freqs - 1, n_freqs)
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(1 + 2 * n_freqs, 32),
            nn.SiLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, gap: torch.Tensor, gap_aware: bool = True) -> torch.Tensor:
        # gap: (B, N)
        if not gap_aware:
            g = torch.full_like(gap.float(), 20.0)
        else:
            g = gap.float().clamp(min=1.0)
        lg = torch.log1p(g).unsqueeze(-1)  # (B, N, 1)
        angles = lg * self.freqs * (math.pi / 4.0)
        x = torch.cat([lg, torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(x)
