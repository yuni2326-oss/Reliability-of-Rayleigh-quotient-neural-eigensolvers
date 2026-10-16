import numpy as np
import pytest

from eigen_benchmark.bases.radial_fem import solve_radial_fem
from eigen_benchmark.reference import bessel_annulus as ba


def test_fem_matches_bessel_reference():
    for m in (0, 2, 4):
        k = ba.annulus_k_roots(m, n_modes=1)[0]
        lam = solve_radial_fem(40, m, n_modes=1)["Lam"][0]
        assert abs(lam - k ** 4) / k ** 4 < 1e-6


def test_fem_converges_at_fourth_order():
    k = ba.annulus_k_roots(1, n_modes=1)[0]
    ref = k ** 4
    errs = [abs(solve_radial_fem(n, 1, n_modes=1)["Lam"][0] - ref) / ref
            for n in (5, 10, 20)]
    assert errs[0] / errs[1] > 8.0
    assert errs[1] / errs[2] > 8.0


def test_fem_eigenvalues_are_upper_bounds():
    k = ba.annulus_k_roots(0, n_modes=3)
    lam = solve_radial_fem(30, 0, n_modes=3)["Lam"]
    assert np.all(lam >= k ** 4 * (1 - 1e-9))


def test_three_way_agreement_bessel_ritz_fem():
    """Bessel(정확) · 다항 Ritz(에너지) · Hermite FEM(에너지+조각다항) 세 구현이
    같은 값을 준다 = 기준해가 구현 우연이 아님을 보이는 삼중 확인."""
    from eigen_benchmark.bases.monomial import MonomialBasis, orthonormalized
    from eigen_benchmark.problems import p2_annulus as p2
    m = 3
    k4 = ba.annulus_k_roots(m, n_modes=1)[0] ** 4
    ritz = p2.solve(orthonormalized(MonomialBasis(14)), m=m, n_q=400, n_modes=1)["Lam"][0]
    fem = solve_radial_fem(40, m, n_modes=1)["Lam"][0]
    assert abs(ritz - k4) / k4 < 1e-6
    assert abs(fem - k4) / k4 < 1e-6
