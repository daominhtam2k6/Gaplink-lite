import numpy as np

from src.gated_assoc import CHI2_2D, gated_decision, mahalanobis2_centre


def test_mahalanobis_basic():
    pred = np.array([[100.0, 50.0, 20, 10]])
    sig = np.array([[2.0, 4.0, 0, 0]])
    cand = np.array([[104.0, 50.0, 20, 10], [100.0, 58.0, 20, 10]])
    d2 = mahalanobis2_centre(pred, sig, cand)
    assert np.allclose(d2, [[4.0, 4.0]])


def test_min_sigma_floor():
    d2 = mahalanobis2_centre(np.zeros((1, 4)), np.zeros((1, 4)), np.array([[3.0, 4.0, 1, 1]]))
    assert np.isclose(d2[0, 0], 25.0)


def _P(p_link, p_dust):
    # one track, two candidates, dustbin last
    return np.array([[p_link[0], p_link[1], p_dust], [0.0, 0.0, 0.0]])


def test_no_gate_equals_v11_rule():
    P = _P([0.7, 0.1], 0.2)
    d2 = np.array([[100.0, 0.0]])
    assert gated_decision(P, d2, tau=0.25, chi2_gate=None).tolist() == [0]


def test_gate_vetoes_far_candidate_and_falls_back():
    P = _P([0.7, 0.3], 0.2)
    d2 = np.array([[100.0, 1.0]])
    # best candidate outside ellipse -> vetoed; second inside and above tau/dustbin -> chosen
    assert gated_decision(P, d2, tau=0.25, chi2_gate=CHI2_2D["99"]).tolist() == [1]


def test_gate_can_send_to_dustbin():
    P = _P([0.7, 0.1], 0.2)
    d2 = np.array([[100.0, 1.0]])
    assert gated_decision(P, d2, tau=0.25, chi2_gate=CHI2_2D["99"]).tolist() == [-1]
