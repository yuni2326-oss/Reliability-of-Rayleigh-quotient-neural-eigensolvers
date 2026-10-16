import numpy as np
import pytest
from scipy.linalg import hilbert

from eigen_benchmark.conditioning import (condition_report,
                                          generalized_backward_error,
                                          highprec_eigenvalues)


def test_highprec_matches_scipy_on_well_conditioned_pair():
    K = np.array([[4.0, 1.0], [1.0, 3.0]])
    M = np.eye(2)
    ref = np.linalg.eigvalsh(K)
    hp = highprec_eigenvalues(K, M, dps=40, n_eig=2)
    assert np.allclose(hp, ref, rtol=1e-12)


def test_highprec_falls_back_when_fp64_assembly_destroyed_definiteness():
    """M=Hilbert(14)는 **fp64로 반올림되는 순간** 양정부호를 잃는다(λ_min≈1e-21 ≪ 1e-17).
    그래서 mpmath Cholesky가 옳게 거부하고, 일반 pencil 경로로 넘어가 fp64가 건네준
    그 행렬쌍의 참 고유값을 보고한다 — 고정밀 풀이가 조립 손실을 복원하지 못한다는 증거."""
    n = 14
    M = hilbert(n)
    K = np.diag(np.arange(1.0, n + 1))
    hp, info = highprec_eigenvalues(K, M, dps=60, n_eig=2, return_info=True)
    assert info["path"] == "general_pencil"
    assert all(np.isfinite(hp))
    # 정확 Hilbert 펜슬이라면 전부 양수여야 하나, fp64 반올림 탓에 음수가 나타난다
    assert hp[0] < 0.0


def test_highprec_uses_cholesky_path_when_matrix_is_sound():
    hp, info = highprec_eigenvalues(np.diag([1.0, 2.0, 3.0]), np.eye(3),
                                    dps=40, n_eig=3, return_info=True)
    assert info["path"] == "cholesky"
    assert np.allclose(hp, [1.0, 2.0, 3.0], rtol=1e-14)


def test_backward_error_is_tiny_for_exact_pair():
    K = np.array([[2.0, 0.0], [0.0, 5.0]])
    M = np.eye(2)
    eta = generalized_backward_error(K, M, 2.0, np.array([1.0, 0.0]))
    assert eta < 1e-15


def test_backward_error_grows_for_wrong_pair():
    K = np.array([[2.0, 0.0], [0.0, 5.0]])
    M = np.eye(2)
    assert generalized_backward_error(K, M, 3.0, np.array([1.0, 0.0])) > 1e-2


def test_condition_report_keys_and_massnorm_improvement():
    n = 10
    x = np.linspace(0.0, 1.0, 500)
    w = np.full(500, 1.0 / 500)
    e = np.arange(2, n + 2, dtype=float)
    P = x[None, :] ** e[:, None]
    D2 = e[:, None] * (e[:, None] - 1) * x[None, :] ** (e[:, None] - 2)
    M = (P * w) @ P.T
    K = (D2 * w) @ D2.T
    rep = condition_report(K, M, dps=50, n_eig=3)
    for key in ("kappa_M_raw", "kappa_K_raw", "kappa_M_massnorm",
                "kappa_K_equilibrated", "kappa_A_transformed", "cholesky_ok",
                "Lam_fp64", "Lam_highprec", "Lam_absdiff_rel", "backward_error"):
        assert key in rep
    assert rep["kappa_M_massnorm"] < rep["kappa_M_raw"]


def test_condition_report_records_cholesky_failure_without_raising():
    # 특이 질량행렬 → Cholesky 실패를 예외 없이 기록해야 한다.
    M = np.ones((4, 4))
    K = np.eye(4)
    rep = condition_report(K, M, dps=30, n_eig=1)
    assert rep["cholesky_ok"] is False
    assert np.isnan(rep["kappa_A_transformed"])
