import numpy as np

from scripts.failure_analysis import outcome, p_correct, taxonomy


def test_outcome_categories():
    assert outcome(2, 2) == "correct" and outcome(-1, -1) == "correct"
    assert outcome(2, -1) == "missed_relink"
    assert outcome(2, 0) == "wrong_candidate"
    assert outcome(-1, 1) == "false_link_after_exit"


def test_taxonomy_counts_every_track_once():
    evs = [{"label": np.array([0, -1]), "gap_bin": np.array([0, 2])},
           {"label": np.array([1]), "gap_bin": np.array([1])}]
    preds = [np.array([0, 3]), np.array([-1])]
    t = taxonomy(evs, preds)
    assert sum(sum(v.values()) for v in t.values()) == 3
    assert t["short (3-15)"]["correct"] == 1
    assert t["long (46-90)"]["false_link_after_exit"] == 1
    assert t["medium (16-45)"]["missed_relink"] == 1


def test_p_correct_uses_dustbin_for_no_match():
    ev = {"t_lost": np.zeros(2), "cand": np.zeros((2, 4)), "label": np.array([1, -1])}
    P = np.array([[0.1, 0.7, 0.2], [0.3, 0.1, 0.6], [0.0, 0.0, 0.0]])
    assert p_correct(ev, P) == [0.7, 0.6]
