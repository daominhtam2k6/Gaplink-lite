"""MotionEncoder: 6->64 Linear + 2-layer GRU(128) + absolute position + gap embedding -> 128D."""
import torch
import torch.nn as nn
from src.features import build_motion_history_features, GapEmbedding

class MotionEncoder(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        self.gru = nn.GRU(input_size=64, hidden_size=d, num_layers=2, batch_first=True)
        self.gap_enc = GapEmbedding(out_dim=32)
        self.out_proj = nn.Sequential(
            nn.Linear(d + 4 + 32, d),
            nn.LayerNorm(d),
            nn.SiLU(),
        )

    def forward(
        self,
        hist: torch.Tensor,
        hist_len: torch.Tensor,
        gap: torch.Tensor,
        W: torch.Tensor,
        H: torch.Tensor,
        gap_aware: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, K, _ = hist.shape
        step_feats, last_box = build_motion_history_features(hist, hist_len)
        x = self.in_proj(step_feats.reshape(B * N, K, 6))
        out_seq, _ = self.gru(x)
        # Right-aligned history -> final valid timestep is at index K - 1
        h_gru = out_seq[:, -1, :].reshape(B, N, -1)

        W_b = W.view(B, 1, 1).clamp(min=100.0)
        H_b = H.view(B, 1, 1).clamp(min=100.0)
        abs_pos = torch.cat([
            last_box[..., 0:1] / W_b,
            last_box[..., 1:2] / H_b,
            last_box[..., 2:3] / W_b,
            last_box[..., 3:4] / H_b,
        ], dim=-1)

        g_emb = self.gap_enc(gap, gap_aware=gap_aware)
        h_i = self.out_proj(torch.cat([h_gru, abs_pos, g_emb], dim=-1))
        return h_i, g_emb, last_box
