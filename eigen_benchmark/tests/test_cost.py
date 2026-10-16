import math

import pytest

from eigen_benchmark import cost


def test_timed_records_positive_duration():
    with cost.timed("x") as t:
        sum(range(100000))
    assert t["seconds"] > 0.0
    assert t["label"] == "x"


def test_line_items_runs_all_and_records_each():
    out = cost.line_items({"a": lambda: sum(range(1000)),
                           "b": lambda: sum(range(2000))})
    assert set(out) == {"a", "b"}
    assert all(v >= 0.0 for v in out.values())


def test_expected_time_to_success_charges_failures():
    assert cost.expected_time_to_success(10.0, 0.5) == pytest.approx(20.0)
    assert cost.expected_time_to_success(10.0, 1.0) == pytest.approx(10.0)
    assert math.isinf(cost.expected_time_to_success(10.0, 0.0))


def test_time_to_accuracy_finds_first_crossing():
    hist = [(0.5, {"e_lam": 1.0, "mac": 0.1}),
            (1.0, {"e_lam": 0.05, "mac": 0.5}),
            (1.5, {"e_lam": 0.005, "mac": 0.97})]
    levels = {"elam_1e-2": ("e_lam", "<=", 1e-2), "mac_0.95": ("mac", ">=", 0.95)}
    got = cost.time_to_accuracy(hist, levels)
    assert got["elam_1e-2"] == pytest.approx(1.5)
    assert got["mac_0.95"] == pytest.approx(1.5)


def test_time_to_accuracy_reports_none_when_never_reached():
    hist = [(0.5, {"e_lam": 1.0})]
    got = cost.time_to_accuracy(hist, {"elam_1e-3": ("e_lam", "<=", 1e-3)})
    assert got["elam_1e-3"] is None


def test_default_levels_match_the_spec():
    # §3.5의 사전고정 수준 네 가지
    assert set(cost.DEFAULT_LEVELS) == {"elam_1e-2", "elam_1e-3",
                                        "mac_0.95", "resid_1e-3"}
