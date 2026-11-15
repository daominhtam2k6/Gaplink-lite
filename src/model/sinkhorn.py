"""Log-domain Sinkhorn with per-element dustbins and full padding masks (always fp32)."""
import torch

def log_sinkhorn(
    S: torch.Tensor,
    z_t: torch.Tensor,
    z_c: torch.Tensor,
    tm: torch.Tensor,
    cm: torch.Tensor,
    iters: int = 20,
    NEG: float = -1e4,
) -> torch.Tensor:
    """Compute log assignment matrix logP of shape (B, N+1, M+1) in float32.
    Row N is candidate dustbin; Col M is track dustbin.
    Valid row i sums to 1; valid col j sums to 1.
    """
    with torch.amp.autocast("cuda", enabled=False):
        S = S.float()
        z_t = z_t.float()
        z_c = z_c.float()
        B, N, M = S.shape

        A = S.new_full((B, N + 1, M + 1), NEG)
        A[:, :N, :M] = S
        A[:, :N, M] = z_t
        A[:, N, :M] = z_c
        A[:, N, M] = 0.0

        vr = torch.cat([tm, tm.new_ones(B, 1)], dim=1)  # (B, N+1)
        vc = torch.cat([cm, cm.new_ones(B, 1)], dim=1)  # (B, M+1)
        valid_cell = vr[:, :, None] & vc[:, None, :]
        A = A.masked_fill(~valid_cell, NEG)

        n = tm.sum(dim=1).float()  # (B,)
        m = cm.sum(dim=1).float()  # (B,)

        # Real rows have target marginal 1 (log=0), dustbin row N has marginal m (so total sum = n + m)
        log_r = torch.cat([torch.zeros_like(S[:, :, 0]), m.clamp(min=1.0).log()[:, None]], dim=1)
        log_c = torch.cat([torch.zeros_like(S[:, 0, :]), n.clamp(min=1.0).log()[:, None]], dim=1)
        log_r = log_r.masked_fill(~vr, NEG)
        log_c = log_c.masked_fill(~vc, NEG)

        u = A.new_zeros((B, N + 1))
        v = A.new_zeros((B, M + 1))
        for _ in range(iters):
            u = log_r - torch.logsumexp(A + v[:, None, :], dim=2)
            u = u.masked_fill(~vr, 0.0)
            v = log_c - torch.logsumexp(A + u[:, :, None], dim=1)
            v = v.masked_fill(~vc, 0.0)

        logP = A + u[:, :, None] + v[:, None, :]
        logP = logP.masked_fill(~valid_cell, NEG)
        return logP

def assignment_loss(
    logP: torch.Tensor,
    label: torch.Tensor,
    cand_label: torch.Tensor,
    tm: torch.Tensor,
    cm: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood of ground-truth assignment (tracks + unmatched candidates)."""
    B, N1, M1 = logP.shape
    N = N1 - 1
    M = M1 - 1

    # Track targets: if label[b,i] >= 0 -> col = label[b,i], else col = M (dustbin)
    t_target = torch.where(label >= 0, label, torch.full_like(label, M)).clamp(0, M)
    row_logp = logP[:, :N, :].gather(dim=2, index=t_target.unsqueeze(-1)).squeeze(-1)  # (B, N)
    loss_t = -(row_logp * tm.float()).sum()
    n_t = tm.float().sum().clamp(min=1.0)

    # Candidate dustbin targets: candidates with cand_label == -1 should map to row N
    unmatched_c = cm & (cand_label < 0)
    col_dust_logp = logP[:, N, :M]  # (B, M)
    loss_c = -(col_dust_logp * unmatched_c.float()).sum()
    n_c = unmatched_c.float().sum()

    return (loss_t + loss_c) / (n_t + n_c)
