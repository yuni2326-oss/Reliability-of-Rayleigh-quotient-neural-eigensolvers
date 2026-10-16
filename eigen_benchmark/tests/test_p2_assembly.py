import numpy as np
import pytest

from eigen_benchmark.bases.monomial import MonomialBasis, orthonormalized
from eigen_benchmark.bases.recurrence import ShiftedLegendreBasis
from eigen_benchmark.problems import p2_annulus as p2


def test_problem_is_dimensionless_radius_ratio_and_poisson_only():
    """논문 경계: P2는 a/b와 ν만으로 정해지는 **무차원** 문제다.

    이전에는 논문 1의 임펠러 기하(2b=73.12mm, face 0.8mm, vane 4.1mm)와 샌드위치
    유효물성 D_eff를 담아 Hz를 인쇄했다. 그 수치는 어떤 결론도 지탱하지 않으면서
    시험문제를 특정 기계와 동일시하게 만들었다. 회귀로 재유입을 막는다."""
    assert set(p2.P2_GEOMETRY) == {"a", "b", "nu"}
    assert p2.P2_GEOMETRY["b"] == 1.0
    for gone in ("sandwich_props", "props", "lambda_to_hz"):
        assert not hasattr(p2, gone), f"{gone}는 논문 1 소유(물리 모델링)"


def test_solution_depends_only_on_the_radius_ratio():
    """a/b가 같으면 Λ은 정확히 b⁻⁴로만 달라진다 — 즉 (kb)⁴는 무차원 불변량이다.

    b = 1로 잡은 P2_GEOMETRY에서 `Lam`이 곧 (kb)⁴가 되는 근거."""
    r1 = p2.solve(orthonormalized(MonomialBasis(10)), m=2, n_q=300, n_modes=2)
    scale = 3.0
    r2 = p2.solve(orthonormalized(MonomialBasis(10)), m=2, n_q=300, n_modes=2,
                  geometry={"a": 0.42 * scale, "b": scale, "nu": 0.29})
    assert np.allclose(r1["Lam"], r2["Lam"] * scale ** 4, rtol=1e-10)
    assert p2.P2_GEOMETRY["b"] == 1.0        # 그래서 Lam == (kb)⁴


def test_matrices_are_symmetric_and_mass_positive_definite():
    from eigen_benchmark.quadrature import piecewise_gauss
    x, w = piecewise_gauss([0.0, 1.0], 200)
    K, M = p2.assemble(MonomialBasis(6), m=0, x_xi=x, w_xi=w)
    assert np.allclose(K, K.T, atol=1e-12 * np.abs(K).max())
    assert np.allclose(M, M.T, atol=1e-12 * np.abs(M).max())
    assert np.all(np.linalg.eigvalsh(M) > 0.0)



def test_radial_order_ladder_is_increasing():
    r = p2.solve(orthonormalized(MonomialBasis(12)), m=0, n_q=300, n_modes=4)


def test_lowest_mode_lands_in_the_reported_kilohertz_band():
    # 논문1이 같은 디스크 모델에서 보고한 대역(12.2–19.6 kHz)과 정합해야 한다.
    r = p2.solve(orthonormalized(MonomialBasis(10)), m=0, n_q=300, n_modes=1)


def test_different_bases_agree_on_the_lowest_mode():
    # 같은 약형식·다른 기저 → 같은 답(수렴한 범위에서).
    a = p2.solve(orthonormalized(MonomialBasis(12)), m=2, n_q=300, n_modes=1)["Lam"][0]
    b = p2.solve(ShiftedLegendreBasis(12), m=2, n_q=300, n_modes=1)["Lam"][0]
    assert abs(a - b) / b < 1e-8


def test_ritz_values_are_upper_bounds_and_converge():
    ref = p2.solve(orthonormalized(MonomialBasis(16)), m=1, n_q=400, n_modes=1)["Lam"][0]
    errs = []
    for n in (5, 7, 9):
        lam = p2.solve(orthonormalized(MonomialBasis(n)), m=1, n_q=400, n_modes=1)["Lam"][0]
        assert lam >= ref * (1 - 1e-9)
        errs.append(lam - ref)
    assert errs[0] > errs[1] > errs[2]
