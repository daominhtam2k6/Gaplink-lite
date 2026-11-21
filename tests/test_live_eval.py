import pytest

from src.live_eval import add_counts, empty_counts, select_operating_point, summarize


def test_select_prefers_highest_irr_under_constraint():
    rows = [{"tau": 0.25, "nat_irr": 0.90, "nat_wlr": 0.10},
            {"tau": 0.5, "nat_irr": 0.86, "nat_wlr": 0.045},
            {"tau": 0.7, "nat_irr": 0.80, "nat_wlr": 0.02}]
    assert select_operating_point(rows)["tau"] == 0.5


def test_select_falls_back_to_lowest_wlr():
    rows = [{"tau": 0.25, "nat_irr": 0.9, "nat_wlr": 0.2}, {"tau": 0.9, "nat_irr": 0.5, "nat_wlr": 0.08}]
    r = select_operating_point(rows)
    assert r["tau"] == 0.9 and "no config" in r["rule"]


def test_select_empty_raises():
    with pytest.raises(ValueError):
        select_operating_point([])


def test_counts_and_summary():
    c = add_counts(empty_counts(), {**empty_counts(), "nat_gaps_total": 4, "nat_gaps_recovered": 3,
                                    "reassoc_total_valid_gt": 10, "reassoc_wrong": 1})
    s = summarize(c)
    assert s["natural_occlusion"]["nat_irr"] == 0.75
    assert s["natural_occlusion"]["nat_wlr"] == 0.1
