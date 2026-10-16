import numpy as np
import pytest

from eigen_benchmark.bases.monomial import MonomialBasis
from eigen_benchmark.problems import p1_beam as p1



def test_ritz_eigenvalues_are_upper_bounds_and_converge():
    # Rayleigh–Ritz는 위에서 수렴. DOF를 늘리면 오차가 단조 감소해야 한다.
    # N은 fp64 Cholesky가 깨지는 차수(≈12) 아래로 잡는다 — 붕괴는 별도 테스트에서 다룬다.
    ref = p1.beta_roots(3) ** 4
    errs = []
    for n in (4, 8, 10):
        r = p1.solve(MonomialBasis(n), n_q=300, n_modes=3)
        assert r["cholesky_ok"]
        assert np.all(r["Lam"][:3] >= ref * (1 - 1e-9))     # 상한
        errs.append(abs(r["Lam"][2] - ref[2]) / ref[2])
    assert errs[0] > errs[1] > errs[2]


def test_solve_records_cholesky_failure_instead_of_raising():
    # 원시 단항식 고차에서 질량행렬이 수치적 양정부호를 잃는다(파일럿 소견).
    # 예외로 터지면 기저연구 드라이버가 멈추므로 NaN + 플래그로 남겨야 한다.
    r = p1.solve(MonomialBasis(20), n_q=400, n_modes=3)
    assert r["cholesky_ok"] is False
    assert np.all(np.isnan(r["Lam"]))
    assert np.linalg.cond(r["M"]) > 1e14


def test_assembled_matrices_are_symmetric_and_mass_is_positive_definite():
    from eigen_benchmark.quadrature import piecewise_gauss
    x, w = piecewise_gauss([0.0, 1.0], 200)
    K, M = p1.assemble(MonomialBasis(6), x, w)
    assert np.allclose(K, K.T, atol=1e-14 * np.abs(K).max())
    assert np.allclose(M, M.T, atol=1e-14 * np.abs(M).max())
    assert np.all(np.linalg.eigvalsh(M) > 0.0)


def test_mass_matrix_matches_closed_form():
    # M_ij = ∫₀¹ x^{i+2} x^{j+2} dx = 1/(i+j+5)
    from eigen_benchmark.quadrature import piecewise_gauss
    x, w = piecewise_gauss([0.0, 1.0], 40)
    _, M = p1.assemble(MonomialBasis(4), x, w)
    exact = np.array([[1.0 / (i + j + 5) for j in range(4)] for i in range(4)])
    assert np.allclose(M, exact, rtol=1e-13)
