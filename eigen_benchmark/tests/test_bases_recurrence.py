import numpy as np
import pytest

from eigen_benchmark.bases.recurrence import (ChebyshevBasis,
                                              IntegratedLegendreBasis,
                                              ShiftedLegendreBasis)
from eigen_benchmark.problems import p1_beam as p1


@pytest.mark.parametrize("cls", [ShiftedLegendreBasis, ChebyshevBasis,
                                 IntegratedLegendreBasis])
def test_satisfies_clamped_boundary_conditions(cls):
    b = cls(6)
    z = np.array([0.0])
    assert np.allclose(b.eval(z), 0.0, atol=1e-12)
    assert np.allclose(b.d1(z), 0.0, atol=1e-12)


@pytest.mark.parametrize("cls", [ShiftedLegendreBasis, ChebyshevBasis,
                                 IntegratedLegendreBasis])
def test_second_derivative_matches_finite_difference(cls):
    # 2계 편차분은 절단 O(ε²f⁗)과 상쇄 O(eps_machine/ε²)로 ~1e-5까지만 정확하다.
    # 도함수가 틀리면 O(1) 차이가 나므로 이 허용오차로도 충분히 잡는다.
    b = cls(5)
    x0, e = 0.37, 1e-5
    f = lambda t: b.eval(np.array([t]))[:, 0]
    fd = (f(x0 + e) - 2 * f(x0) + f(x0 - e)) / e ** 2
    assert np.allclose(b.d2(np.array([x0]))[:, 0], fd, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("cls", [ShiftedLegendreBasis, ChebyshevBasis,
                                 IntegratedLegendreBasis])
def test_converges_to_analytic_beam_eigenvalues(cls):
    ref = p1.beta_roots(3) ** 4
    r = p1.solve(cls(14), n_q=400, n_modes=3)
    assert r["cholesky_ok"]
    assert abs(r["Lam"][0] - ref[0]) / ref[0] < 1e-9
    assert abs(r["Lam"][2] - ref[2]) / ref[2] < 1e-7


@pytest.mark.parametrize("cls", [ShiftedLegendreBasis, ChebyshevBasis,
                                 IntegratedLegendreBasis])
def test_c1_bases_have_no_slope_jump(cls):
    b = cls(5)
    assert np.allclose(b.d1_jump(0.2), np.zeros(5))


def test_integrated_legendre_bubbles_vanish_at_both_ends():
    # 버블(자유단 Hermite 2개 이후)은 양 끝에서 값·기울기가 0이어야 한다.
    b = IntegratedLegendreBasis(6)
    one = np.array([1.0])
    v, s = b.eval(one)[:, 0], b.d1(one)[:, 0]
    assert np.allclose(v[2:], 0.0, atol=1e-11)
    assert np.allclose(s[2:], 0.0, atol=1e-11)


def test_integrated_legendre_bubbles_are_not_identically_zero():
    """L_0·L_1로 버블을 만들면 3차 Hermite 보정을 빼는 순간 항등적으로 0이 된다.
    그러면 기저에 영함수가 들어가 K·M이 특이해지고 λ≈0의 허위 고유값이 나온다.
    차수 오프셋 +2(L_2부터)가 지켜지는지 확인한다."""
    b = IntegratedLegendreBasis(6)
    x = np.linspace(0.05, 0.95, 21)
    for i in range(2, b.n_dof):
        assert np.max(np.abs(b.eval(x)[i])) > 1e-6, f"버블 {i}가 영함수"


def test_integrated_legendre_stiffness_is_nearly_banded():
    # 이중적분 Legendre의 이점: K = ∫ψ_i″ψ_j″가 거의 대각(버블 블록에서).
    from eigen_benchmark.quadrature import piecewise_gauss
    x, w = piecewise_gauss([0.0, 1.0], 400)
    K, _ = p1.assemble(IntegratedLegendreBasis(10), x, w)
    Kb = K[2:, 2:]
    diag = np.abs(np.diag(Kb)).sum()
    off = np.abs(Kb - np.diag(np.diag(Kb))).sum()
    assert off < diag
