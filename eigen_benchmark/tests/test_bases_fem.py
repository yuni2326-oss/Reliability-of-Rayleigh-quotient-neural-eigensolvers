import numpy as np
import pytest
from scipy.linalg import eigh

from eigen_benchmark.bases.fem import element_matrices, hermite_beam_matrices
from eigen_benchmark.problems import p1_beam as p1


def test_element_matrices_are_symmetric_and_scale_correctly():
    Ke, Me = element_matrices(0.25)
    assert np.allclose(Ke, Ke.T)
    assert np.allclose(Me, Me.T)
    # 강성은 h⁻³, 질량은 h에 비례
    Ke2, Me2 = element_matrices(0.5)
    assert Ke[0, 0] / Ke2[0, 0] == pytest.approx(8.0, rel=1e-12)
    assert Me[0, 0] / Me2[0, 0] == pytest.approx(0.5, rel=1e-12)


def test_fem_recovers_analytic_cantilever_eigenvalues():
    ref = p1.beta_roots(3) ** 4
    K, M, _ = hermite_beam_matrices(24)
    Lam = eigh(K, M, eigvals_only=True)
    assert np.all(Lam[:3] >= ref * (1 - 1e-12))       # 적합 Galerkin = 상한
    # n_elem=24, O(h⁴) 실측: 모드1 5e-8, 모드3 1.6e-5(고차모드는 βh가 커서 느리게 수렴)
    assert abs(Lam[0] - ref[0]) / ref[0] < 1e-7
    assert abs(Lam[2] - ref[2]) / ref[2] < 1e-4


def test_fem_converges_at_expected_rate():
    ref = p1.beta_roots(1)[0] ** 4
    errs = []
    for n in (4, 8, 16):
        K, M, _ = hermite_beam_matrices(n)
        errs.append(abs(eigh(K, M, eigvals_only=True)[0] - ref) / ref)
    # 3차 Hermite 고유값 수렴은 O(h⁴) → 배증마다 최소 8배 감소
    assert errs[0] / errs[1] > 8.0
    assert errs[1] / errs[2] > 8.0


def test_rotation_split_adds_one_dof_and_unit_jump_vector():
    K0, M0, j0 = hermite_beam_matrices(10, xc=0.2, split_rotation=False)
    K1, M1, j1 = hermite_beam_matrices(10, xc=0.2, split_rotation=True)
    assert K1.shape[0] == K0.shape[0] + 1
    assert np.allclose(j0, 0.0)
    assert np.sum(np.abs(j1)) == pytest.approx(2.0)     # +1과 −1
    assert set(np.round(j1[np.abs(j1) > 0], 12)) == {1.0, -1.0}


def test_split_rotation_without_spring_has_a_hinge_mechanism():
    # 스프링을 안 걸면 두 회전이 자유 → 강성행렬이 특이(고유값 ≈0 존재).
    K, M, _ = hermite_beam_matrices(10, xc=0.2, split_rotation=True)
    Lam = eigh(K, M, eigvals_only=True)
    assert abs(Lam[0]) < 1e-6 * p1.beta_roots(1)[0] ** 4


def test_adding_the_spring_term_removes_the_mechanism():
    # k̂⟦u′⟧⟦v′⟧를 더하면 힌지가 구속되어 최저 고유값이 물리적 값으로 올라온다.
    K, M, j = hermite_beam_matrices(10, xc=0.2, split_rotation=True)
    Ks = K + 1e6 * np.outer(j, j)                       # 매우 강한 스프링 = 균일보 극한
    Lam = eigh(Ks, M, eigvals_only=True)
    ref = p1.beta_roots(1)[0] ** 4
    assert abs(Lam[0] - ref) / ref < 1e-5


def test_xc_must_be_a_node():
    with pytest.raises(ValueError):
        hermite_beam_matrices(10, xc=0.23, split_rotation=True)
