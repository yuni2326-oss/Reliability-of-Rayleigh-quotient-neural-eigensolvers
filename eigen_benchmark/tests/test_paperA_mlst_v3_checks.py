"""논문 A MLST v3 Word 메모(2026-10-15) 대응 검사 — 타이밍 산포, fp64 기준의 4배 정밀도 대조, Cholesky 경계, 커리큘럼 목표 비용.

산술과 배선을 본다. 마지막 시험만 커밋된 자료를 읽는다.
"""
import math
import os

import numpy as np
import pytest

from eigen_benchmark.drivers import paperA_mlst_v3_checks as vc

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "_generated", "data", "paper2")


def test_timing_span_is_the_decimal_log_of_max_over_min():
    rows = [{"key": "a", "seconds_min": 2.0, "seconds_max": 20.0},
            {"key": "b", "seconds_min": 1.0, "seconds_max": 1.25}]
    out = vc.timing_spans(rows, key=lambda r: r["key"])
    assert out["rows"]["a"]["orders"] == pytest.approx(1.0)
    assert out["rows"]["b"]["ratio"] == pytest.approx(1.25)
    assert out["max_orders"] == pytest.approx(1.0)
    assert out["max_key"] == "a"


def test_quad_remainder_matches_the_exact_answer_on_a_coordinate_span():
    w = np.ones(4)
    P = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    f = np.array([3.0, 4.0, 5.0, 6.0])
    r = vc.quad_remainder(P, f, w)
    assert r.dtype == np.longdouble
    np.testing.assert_allclose(r.astype(float), [0.0, 0.0, 5.0, 6.0], atol=1e-30)


def test_quad_remainder_is_orthogonal_to_a_nearly_dependent_span():
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 1.5, 40)
    P = rng.standard_normal((3, 40))
    P[2] = P[0] + 1e-9 * P[2]
    f = rng.standard_normal(40)
    r = vc.quad_remainder(P, f, w)
    ip = (w.astype(np.longdouble) * r) @ P.astype(np.longdouble).T
    assert float(np.max(np.abs(ip))) < 1e-24


def _raw_sets(n_modes=3, n_seeds=2, n=32, seed=1):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    base = np.stack([x ** (k + 2) for k in range(n_modes)])
    raw = base[:, None, :] + 0.05 * rng.standard_normal((n_modes, n_seeds, n))
    return raw.astype(np.float32).astype(np.float64), np.full(n, 1.0 / n)


def test_reference_check_agrees_for_the_fp64_svd_reference():
    raw, w = _raw_sets()
    out = vc.reference_check(raw, w)
    assert set(out["by_stage"]) == {2, 3}
    assert out["max_rel_diff"] < 1e-12
    assert out["n_pairs"] == 4


def test_reference_check_detects_a_perturbed_reference():
    raw, w = _raw_sets()

    def bad(A, b):
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        return c * (1.0 + 1e-6)

    out = vc.reference_check(raw, w, solver=bad)
    assert out["max_rel_diff"] > 1e-9


def test_cholesky_boundary_reports_each_triangle_separately():
    # 아래 삼각형은 양정치 [[1, .5], [.5, 1]], 위 삼각형은 부정치 [[1, 2], [2, 1]]
    M = np.array([[1.0, 2.0], [0.5, 1.0]])
    K = np.eye(2)
    out = vc.cholesky_boundary(K, M)
    assert out["potrf_lower_info"] == 0
    assert out["potrf_upper_info"] == 2
    assert out["asymmetry_rel"] == pytest.approx(1.5 / 2.0)
    assert set(out["eigh"]) == {"gvd", "gv", "gvx"}


def test_cholesky_boundary_on_a_definite_pair_accepts_every_route():
    M = np.array([[2.0, 0.3], [0.3, 1.0]])
    K = np.diag([1.0, 3.0])
    out = vc.cholesky_boundary(K, M)
    assert out["potrf_lower_info"] == out["potrf_upper_info"] == 0
    assert out["scipy_cholesky_lower"]["ok"] and out["scipy_cholesky_upper"]["ok"]
    for d in ("gvd", "gv", "gvx"):
        assert out["eigh"][d]["ok"]
        assert out["eigh"][d]["backward_error"] < 1e-14


def test_coarse_target_seconds_times_the_committed_target_function():
    pytest.importorskip("torch")
    out = vc.coarse_target_seconds(reps=1, modes=(1,))
    row = out["rows"][0]
    assert row["mode"] == 1 and row["reps"] == 1
    assert row["seconds_median"] > 0
    assert row["lambda"] == pytest.approx(12.362, rel=1e-2)


def test_committed_timing_rows_cover_the_table16_configurations():
    out = vc.table16_timing_spans(DATA)
    keys = set(out["neural"]["rows"]) | set(out["classical"]["rows"])
    assert "c_curriculum/1" in keys
    assert {"p4_neural_2d[exact projection]/1", "p4_neural_2d[exact projection]/3"} <= keys
    assert "P1/monomial_raw/8" in keys
    assert out["neural"]["max_orders"] < 0.5
    assert math.isfinite(out["classical"]["max_orders"])
