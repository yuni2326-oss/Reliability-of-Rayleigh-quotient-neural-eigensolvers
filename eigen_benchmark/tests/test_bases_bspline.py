import numpy as np
import pytest

from eigen_benchmark.bases.bspline import BSplineC0Basis
from eigen_benchmark.problems import p1_beam as p1


def test_clamped_boundary_conditions_hold():
    b = BSplineC0Basis(n_elem=8, degree=3)
    z = np.array([0.0])
    assert np.allclose(b.eval(z), 0.0, atol=1e-13)
    assert np.allclose(b.d1(z), 0.0, atol=1e-13)


def test_converges_to_analytic_beam_eigenvalues():
    # n_q는 구간당 점수. 구적은 basis.breaks(노트)에 자동 정렬된다.
    ref = p1.beta_roots(3) ** 4
    r = p1.solve(BSplineC0Basis(n_elem=16, degree=3), n_q=8, n_modes=3)
    assert r["cholesky_ok"]
    assert np.all(r["Lam"][:3] >= ref * (1 - 1e-12))      # 적합 Galerkin = 상한
    assert abs(r["Lam"][0] - ref[0]) / ref[0] < 1e-6
    assert abs(r["Lam"][2] - ref[2]) / ref[2] < 1e-4


def test_bspline_eigenvalue_convergence_rate():
    # 3차 스플라인 고유값은 O(h⁴) → 요소수 배증마다 오차가 최소 8배 감소해야 한다.
    ref = p1.beta_roots(1)[0] ** 4
    errs = []
    for nel in (8, 16, 32):
        lam = p1.solve(BSplineC0Basis(nel, degree=3), n_q=8, n_modes=1)["Lam"][0]
        errs.append(abs(lam - ref) / ref)
    assert errs[0] / errs[1] > 8.0
    assert errs[1] / errs[2] > 8.0


def test_misaligned_quadrature_breaks_the_upper_bound_property():
    """구적을 노트에 정렬하지 않으면 조각다항을 매끄러운 것으로 오적분해 강성을
    과소평가하고 Ritz 값이 정확해 **아래로** 떨어진다 — §3.6 구적분리의 존재 이유."""
    ref = p1.beta_roots(1)[0] ** 4
    good = p1.solve(BSplineC0Basis(16, degree=3), n_q=8, n_modes=1)["Lam"][0]
    bad = p1.solve(BSplineC0Basis(16, degree=3), n_q=128, n_modes=1,
                   breaks=(0.0, 1.0))["Lam"][0]
    assert good >= ref * (1 - 1e-12)
    assert bad < ref                       # 하한 위반 = 비적합
    assert abs(bad - ref) / ref > abs(good - ref) / ref


def test_c0_knot_produces_nonzero_slope_jump():
    # x_c에 중복도 p를 주면 C⁰ → 기울기 점프를 표현할 수 있다.
    b = BSplineC0Basis(n_elem=10, degree=3, xc=0.2, c0=True)
    j = b.d1_jump(0.2)
    assert np.max(np.abs(j)) > 1e-3


def test_without_c0_knot_slope_jump_vanishes():
    b = BSplineC0Basis(n_elem=10, degree=3, xc=0.2, c0=False)
    assert np.max(np.abs(b.d1_jump(0.2))) < 1e-9


def test_second_derivative_matches_finite_difference():
    b = BSplineC0Basis(n_elem=8, degree=4)
    x0, e = 0.53, 1e-5
    f = lambda t: b.eval(np.array([t]))[:, 0]
    fd = (f(x0 + e) - 2 * f(x0) + f(x0 - e)) / e ** 2
    assert np.allclose(b.d2(np.array([x0]))[:, 0], fd, rtol=1e-4, atol=1e-4)


def test_xc_is_a_knot_so_quadrature_can_split_there():
    b = BSplineC0Basis(n_elem=10, degree=3, xc=0.2, c0=True)
    assert 0.2 in set(np.round(b.knots, 12))
