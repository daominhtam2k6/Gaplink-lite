"""Pairwise scoring s_ij and per-element track/candidate dustbins z_i, z_j."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.predictor import target_from_boxes, decode_predicted_boxes

def pairwise_iou_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (B, N, 4), b: (B, M, 4) in (cx, cy, w, h) -> returns (B, N, M)."""
    ax1 = a[..., 0:1] - 0.5 * a[..., 2:3]
    ay1 = a[..., 1:2] - 0.5 * a[..., 3:4]
    ax2 = a[..., 0:1] + 0.5 * a[..., 2:3]
    ay2 = a[..., 1:2] + 0.5 * a[..., 3:4]

    bx1 = b[:, None, :, 0] - 0.5 * b[:, None, :, 2]
    by1 = b[:, None, :, 1] - 0.5 * b[:, None, :, 3]
    bx2 = b[:, None, :, 0] + 0.5 * b[:, None, :, 2]
    by2 = b[:, None, :, 1] + 0.5 * b[:, None, :, 3]

    ix1 = torch.maximum(ax1, bx1)
    iy1 = torch.maximum(ay1, by1)
    ix2 = torch.minimum(ax2, bx2)
    iy2 = torch.minimum(ay2, by2)
    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)
    area_a = (a[..., 2:3] * a[..., 3:4]).clamp(min=1.0)
    area_b = (b[:, None, :, 2] * b[:, None, :, 3]).clamp(min=1.0)
    return inter / (area_a + area_b - inter + 1e-6)

class PairwiseAndDustbinScorer(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        # Pairwise inputs:
        # h_i (d) + ctx_i (d) + g_emb (32) + rel_cand (4) + pair_geom (9) = 2*d + 45
        pair_in = 2 * d + 32 + 4 + 9
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # Track dustbin: h_i (d) + g_emb (32) + border_pos (5)
        self.trk_dust_mlp = nn.Sequential(
            nn.Linear(d + 32 + 9 + 128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # Candidate dustbin: cand_norm (4) + border_dist (1)
        self.cand_dust_mlp = nn.Sequential(
            nn.Linear(7 + 128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        h_i: torch.Tensor,
        ctx_i: torch.Tensor,
        g_emb: torch.Tensor,
        last_box: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        track_color: torch.Tensor,
        cand: torch.Tensor,
        cand_hist: torch.Tensor,
        side: torch.Tensor,
        W: torch.Tensor,
        H: torch.Tensor,
        use_color: bool = True,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = h_i.shape
        M = cand.shape[1]

        pred_box, _ = decode_predicted_boxes(last_box, mu, logvar)  # (B, N, 4)

        # Normalized target coordinates of each candidate j relative to track i's last_box: (B, N, M, 4)
        lb_exp = last_box[:, :, None, :].expand(B, N, M, 4)
        c_exp = cand[:, None, :, :].expand(B, N, M, 4)
        rel_cand = target_from_boxes(lb_exp, c_exp)  # (B, N, M, 4)

        # Gaussian log-probability m_ij (clamped for numerical stability)
        mu_exp = mu[:, :, None, :]
        lv_exp = logvar[:, :, None, :]
        var_exp = torch.exp(lv_exp)
        m_ij = -0.5 * (lv_exp + ((rel_cand - mu_exp) ** 2) / var_exp).sum(dim=-1)  # (B, N, M)
        m_ij = (m_ij / 10.0).clamp(-10.0, 2.0)

        # Center Mahalanobis distance squared and Euclidean distance in units of h_last
        d2_center = (((rel_cand[..., 0] - mu_exp[..., 0]) ** 2) / var_exp[..., 0] +
                     ((rel_cand[..., 1] - mu_exp[..., 1]) ** 2) / var_exp[..., 1]).clamp(0.0, 100.0) / 10.0
        euclid_pred = torch.hypot(rel_cand[..., 0] - mu_exp[..., 0], rel_cand[..., 1] - mu_exp[..., 1]).clamp(0.0, 20.0)
        euclid_last = torch.hypot(rel_cand[..., 0], rel_cand[..., 1]).clamp(0.0, 20.0)

        # Size ratio log differences
        dw_log = (rel_cand[..., 2] - mu_exp[..., 2]).abs().clamp(0.0, 3.0)
        dh_log = (rel_cand[..., 3] - mu_exp[..., 3]).abs().clamp(0.0, 3.0)

        # IoU between predicted box and candidate box
        iou_pred = pairwise_iou_torch(pred_box, cand)  # (B, N, M)

        # Color cosine similarity + drop rule when partial
        if use_color:
            tc_norm = F.normalize(track_color, p=2, dim=-1)  # (B, N, 128)
            ch_norm = F.normalize(cand_hist, p=2, dim=-1)    # (B, M, 128)
            cos_sim = torch.bmm(tc_norm, ch_norm.transpose(1, 2)).clamp(0.0, 1.0)
            has_color = (track_color.sum(dim=-1, keepdim=True) > 1e-4) & (cand_hist.sum(dim=-1).unsqueeze(1) > 1e-4)
            color_valid = has_color.float()
            if training:
                is_part = (side > 0).unsqueeze(-1)  # (B, N, 1)
                drop_mask = (torch.rand_like(cos_sim) < 0.20) & is_part
                cos_sim = cos_sim.masked_fill(drop_mask, 0.0)
                color_valid = color_valid.masked_fill(drop_mask, 0.0)
            cos_sim = cos_sim * color_valid
        else:
            cos_sim = h_i.new_zeros((B, N, M))
            color_valid = h_i.new_zeros((B, N, M))

        pair_geom = torch.stack([
            m_ij,
            d2_center,
            euclid_pred,
            euclid_last,
            dw_log,
            dh_log,
            iou_pred,
            cos_sim,
            color_valid,
        ], dim=-1)  # (B, N, M, 9)

        h_exp = h_i[:, :, None, :].expand(B, N, M, -1)
        ctx_exp = ctx_i[:, :, None, :].expand(B, N, M, -1)
        g_exp = g_emb[:, :, None, :].expand(B, N, M, -1)

        pair_feat = torch.cat([h_exp, ctx_exp, g_exp, rel_cand, pair_geom], dim=-1)
        S = self.pair_mlp(pair_feat).squeeze(-1) + m_ij  # residual log-density anchor

        # Track dustbin z_t (uses distance to image borders -> vehicles near border likely exited)
        W_b = W.view(B, 1).clamp(min=100.0)
        H_b = H.view(B, 1).clamp(min=100.0)
        px = pred_box[..., 0] / W_b
        py = pred_box[..., 1] / H_b
        pw = pred_box[..., 2] / W_b
        ph = pred_box[..., 3] / H_b
        border_d = torch.minimum(
            torch.minimum(px.clamp(0.0, 1.0), (1.0 - px).clamp(0.0, 1.0)),
            torch.minimum(py.clamp(0.0, 1.0), (1.0 - py).clamp(0.0, 1.0)),
        )
        max_s_t = S.max(dim=-1).values if M > 0 else h_i.new_zeros((B, N))
        max_iou_t = iou_pred.max(dim=-1).values if M > 0 else h_i.new_zeros((B, N))
        max_cos_t = cos_sim.max(dim=-1).values if M > 0 else h_i.new_zeros((B, N))
        min_d_t = euclid_pred.min(dim=-1).values if M > 0 else h_i.new_full((B, N), 20.0)
        t_pos = torch.stack([px.clamp(-0.5, 1.5), py.clamp(-0.5, 1.5), pw, ph, border_d, max_s_t, max_iou_t, max_cos_t, min_d_t], dim=-1)
        tc_in = track_color if use_color else h_i.new_zeros((B, N, 128))
        z_t = self.trk_dust_mlp(torch.cat([h_i, g_emb, t_pos, tc_in], dim=-1)).squeeze(-1)

        # Candidate dustbin z_c
        max_s_c = S.max(dim=1).values if N > 0 else h_i.new_zeros((B, M))
        max_iou_c = iou_pred.max(dim=1).values if N > 0 else h_i.new_zeros((B, M))
        cx_n = (cand[..., 0] / W_b).clamp(0.0, 1.0)
        cy_n = (cand[..., 1] / H_b).clamp(0.0, 1.0)
        cw_n = (cand[..., 2] / W_b).clamp(0.0, 1.0)
        ch_n = (cand[..., 3] / H_b).clamp(0.0, 1.0)
        cb_d = torch.minimum(torch.minimum(cx_n, 1.0 - cx_n), torch.minimum(cy_n, 1.0 - cy_n))
        c_pos = torch.stack([cx_n, cy_n, cw_n, ch_n, cb_d, max_s_c, max_iou_c], dim=-1)
        ch_in = cand_hist if use_color else h_i.new_zeros((B, M, 128))
        z_c = self.cand_dust_mlp(torch.cat([c_pos, ch_in], dim=-1)).squeeze(-1)

        return S, z_t, z_c
