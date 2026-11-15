"""Mandatory Day 3 Gate G3 unit tests for Sinkhorn, padding invariance, permutation equivariance, and Gaussian coverage."""
import numpy as np
import torch
from src.model.sinkhorn import log_sinkhorn
from src.model.gaplink import GapLinkLite
from src.dataset import pad_event_to_fixed, collate_padded_events

def test_1_sinkhorn_marginals():
    torch.manual_seed(0)
    B, N, M = 4, 6, 10
    S = torch.randn(B, N, M)
    z_t = torch.randn(B, N)
    z_c = torch.randn(B, M)
    tm = torch.zeros(B, N, dtype=torch.bool)
    cm = torch.zeros(B, M, dtype=torch.bool)
    for b in range(B):
        tm[b, : b + 1] = True
        cm[b, : b + 2] = True

    logP = log_sinkhorn(S, z_t, z_c, tm, cm, iters=25)
    P = torch.exp(logP)
    for b in range(B):
        n_v = b + 1
        m_v = b + 2
        # Valid row sums == 1
        row_sums = P[b, :n_v, :].sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
        # Valid col sums == 1
        col_sums = P[b, :, :m_v].sum(dim=0)
        assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3)
        # Padded rows/cols have ~0 probability
        assert P[b, n_v:N, :].max().item() < 1e-3
        assert P[b, :, m_v:M].max().item() < 1e-3

def test_2_padding_content_invariance():
    torch.manual_seed(1)
    B, N, M = 2, 6, 10
    S1 = torch.randn(B, N, M)
    z_t1 = torch.randn(B, N)
    z_c1 = torch.randn(B, M)
    tm = torch.tensor([[True, True, False, False, False, False],
                       [True, True, True, False, False, False]])
    cm = torch.tensor([[True, True, True, False, False, False, False, False, False, False],
                       [True, True, False, False, False, False, False, False, False, False]])

    S2 = S1.clone()
    z_t2 = z_t1.clone()
    z_c2 = z_c1.clone()
    # Corrupt padded positions with huge random numbers
    S2[:, 3:, :] = 999.0
    S2[:, :, 4:] = -888.0
    z_t2[:, 3:] = 777.0
    z_c2[:, 4:] = -666.0

    P1 = torch.exp(log_sinkhorn(S1, z_t1, z_c1, tm, cm))
    P2 = torch.exp(log_sinkhorn(S2, z_t2, z_c2, tm, cm))
    assert torch.allclose(P1[:, :2, :2], P2[:, :2, :2], atol=1e-5)

def test_3_hand_2x2_clear_assignment():
    S = torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]])
    z_t = torch.tensor([[-5.0, -5.0]])
    z_c = torch.tensor([[-5.0, -5.0]])
    tm = torch.ones(1, 2, dtype=torch.bool)
    cm = torch.ones(1, 2, dtype=torch.bool)
    P = torch.exp(log_sinkhorn(S, z_t, z_c, tm, cm, iters=20))[0]
    assert P[0, 0].item() > 0.99 and P[1, 1].item() > 0.99
    assert P[0, 1].item() < 0.01 and P[1, 0].item() < 0.01

def test_4_permutation_equivariance():
    torch.manual_seed(42)
    model = GapLinkLite(d=128).eval()
    N, M, K = 4, 5, 20
    ev1 = {
        "W": 960, "H": 540,
        "t_lost": np.array([10, 12, 15, 18], dtype=np.int32),
        "t_conf": 30,
        "hist": np.random.uniform(50, 400, size=(N, K, 4)).astype(np.float32),
        "hist_len": np.array([12, 15, 20, 10], dtype=np.int32),
        "track_color": np.random.uniform(0, 1, size=(N, 128)).astype(np.float32),
        "gap": np.array([20, 18, 15, 12], dtype=np.int32),
        "gap_bin": np.array([1, 1, 0, 0], dtype=np.int8),
        "cand": np.random.uniform(50, 400, size=(M, 4)).astype(np.float32),
        "cand_hist": np.random.uniform(0, 1, size=(M, 128)).astype(np.float32),
        "ctx_lost": np.random.randn(N, 8, 6).astype(np.float32),
        "ctx_conf": np.random.randn(N, 8, 6).astype(np.float32),
        "ctx_mask_lost": np.ones((N, 8), dtype=bool),
        "ctx_mask_conf": np.ones((N, 8), dtype=bool),
    }
    perm_t = np.array([2, 0, 3, 1])
    perm_c = np.array([3, 1, 4, 0, 2])
    ev2 = {
        "W": 960, "H": 540,
        "t_lost": ev1["t_lost"][perm_t],
        "t_conf": 30,
        "hist": ev1["hist"][perm_t],
        "hist_len": ev1["hist_len"][perm_t],
        "track_color": ev1["track_color"][perm_t],
        "gap": ev1["gap"][perm_t],
        "gap_bin": ev1["gap_bin"][perm_t],
        "cand": ev1["cand"][perm_c],
        "cand_hist": ev1["cand_hist"][perm_c],
        "ctx_lost": ev1["ctx_lost"][perm_t],
        "ctx_conf": ev1["ctx_conf"][perm_t],
        "ctx_mask_lost": ev1["ctx_mask_lost"][perm_t],
        "ctx_mask_conf": ev1["ctx_mask_conf"][perm_t],
    }
    b1 = collate_padded_events([pad_event_to_fixed(ev1)])
    b2 = collate_padded_events([pad_event_to_fixed(ev2)])
    with torch.no_grad():
        P1 = model(b1)["P"][0, :N, :M].numpy()
        P2 = model(b2)["P"][0, :N, :M].numpy()
    P1_perm = P1[perm_t][:, perm_c]
    assert np.allclose(P1_perm, P2, atol=1e-4)

def test_5_gaussian_2d_coverage_chi2():
    rng = np.random.default_rng(123)
    mu = np.array([300.0, 200.0])
    sigma = np.array([15.0, 10.0])
    samples = rng.normal(loc=mu, scale=sigma, size=(20000, 2))
    d2 = ((samples[:, 0] - mu[0]) / sigma[0]) ** 2 + ((samples[:, 1] - mu[1]) / sigma[1]) ** 2
    cov90 = float(np.mean(d2 <= 4.60517))
    assert 0.885 <= cov90 <= 0.915

def test_param_count_under_1m():
    model = GapLinkLite(d=128)
    n_params = sum(p.numel() for p in model.parameters())
    print("GapLinkLite params:", n_params)
    assert n_params < 1_000_000
