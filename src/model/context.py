"""DeepSets ContextEncoder over up to 8 neighbor vehicles at t_lost and 8 at t_conf."""
import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(128, d),
            nn.LayerNorm(d),
            nn.SiLU(),
        )

    def forward(
        self,
        ctx_lost: torch.Tensor,
        ctx_conf: torch.Tensor,
        mask_lost: torch.Tensor,
        mask_conf: torch.Tensor,
        use_ctx: bool = True,
    ) -> torch.Tensor:
        B, N, K, _ = ctx_lost.shape
        if not use_ctx:
            return ctx_lost.new_zeros((B, N, 128))
        tokens = torch.cat([ctx_lost, ctx_conf], dim=2)         # (B, N, 2K, 6)
        mask = torch.cat([mask_lost, mask_conf], dim=2)         # (B, N, 2K)
        emb = self.token_mlp(tokens)                            # (B, N, 2K, 64)
        m_f = mask.unsqueeze(-1).float()
        emb_masked = emb * m_f

        # Mean pooling over valid tokens
        denom = m_f.sum(dim=2).clamp(min=1.0)
        mean_pool = emb_masked.sum(dim=2) / denom

        # Max pooling over valid tokens
        neg_inf = emb.masked_fill(~mask.unsqueeze(-1), -1e4)
        max_pool = neg_inf.max(dim=2).values
        has_any = mask.any(dim=2, keepdim=True).float()
        max_pool = max_pool * has_any

        pooled = torch.cat([mean_pool, max_pool], dim=-1)       # (B, N, 128)
        return self.out_proj(pooled) * has_any
