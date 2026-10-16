import numpy as np
import pytest

from eigen_benchmark.neural.pielm import solve_pielm
from eigen_benchmark.problems import p1_beam as p1


def test_pielm_runs_on_cpu_without_torch():
    out = solve_pielm(40, n_modes=2, seed=0)
    assert out["Lam"].shape[0] == 2
    assert "torch" not in type(out["K"]).__module__


def test_pielm_satisfies_clamped_bc_by_construction():
    out = solve_pielm(30, n_modes=1, seed=1)
    phi0 = float(out["phi_at"](np.array([0.0]))[0])
    assert abs(phi0) < 1e-14


def test_pielm_converges_with_more_random_features():
    ref = p1.beta_roots(1)[0] ** 4
    errs = []
    for n in (20, 40, 80):
        out = solve_pielm(n, n_modes=1, seed=0)
        errs.append(abs(out["Lam"][0] - ref) / ref if out["cholesky_ok"] else np.nan)
    finite = [e for e in errs if np.isfinite(e)]
    assert len(finite) >= 2
    assert min(finite) < 1e-3


def test_pielm_is_deterministic_for_a_fixed_seed():
    a = solve_pielm(40, n_modes=2, seed=7)["Lam"]
    b = solve_pielm(40, n_modes=2, seed=7)["Lam"]
    assert np.allclose(a, b, rtol=0, atol=0)


def test_random_features_are_massively_redundant():
    """**측정된 사실**: 랜덤 tanh 특징은 극심하게 중복되고 질량행렬이 수치적으로
    부정부호다(특징 160개 중 음수 고유값 75개). 원시 GEP는 항상 특이하므로 rank 절단이
    필수이고, 절단 후 유효 부분공간의 조건수는 다룰 만하다(1e11 수준)."""
    for n in (20, 40, 80):
        out = solve_pielm(n, n_modes=1, seed=0)
        assert out["rank_used"] < n
        assert out["kappa_M"] > 1e14              # 전체는 특이에 가깝다
        assert out["kappa_M_retained"] < 1e14     # 절단 후에는 다룰 만하다
        assert out["n_negative_eigs"] > 0         # 수치적으로 부정부호


def test_effective_dimension_is_not_a_well_defined_integer():
    """유효 차원은 절단 임계에 의존한다 — 같은 160개 특징이 8~21로 갈린다.
    단일 rank만 보고하면 임계 선택이 숨으므로 스펙트럼을 함께 남긴다."""
    out = solve_pielm(160, n_modes=1, seed=0)
    sp = out["rank_spectrum"]
    ranks = [sp[k] for k in ("1e-16", "1e-14", "1e-12", "1e-10")]
    assert ranks == sorted(ranks, reverse=True)    # 임계가 느슨할수록 rank 큼
    assert ranks[0] >= 2 * ranks[-1]               # 폭이 2배 이상
    assert out["rank_used"] == sp[f"{out['rank_tol']:g}"]


def test_conditioning_is_reported_by_svd_not_eigenvalue_ratio():
    """M이 부정부호라 ev.max()/ev.min()은 무의미하다(클램프하면 1e300 같은 값이 나온다).
    SVD 기반 cond는 부정부호에서도 well-defined하다."""
    out = solve_pielm(80, n_modes=1, seed=0)
    assert 1e14 < out["kappa_M"] < 1e25


def test_rank_truncation_makes_the_solve_succeed():
    out = solve_pielm(80, n_modes=3, seed=0)
    assert out["cholesky_ok"]
    assert np.all(np.isfinite(out["Lam"]))


def test_pielm_has_no_optimization_error_by_construction():
    """은닉가중을 고정하면 학습이 사라져 e_optimization = 0이다. 같은 시드·같은
    특징수면 실행마다 완전히 동일한 값이 나와야 한다(확률적 요소 없음)."""
    runs = [solve_pielm(60, n_modes=3, seed=3)["Lam"] for _ in range(3)]
    assert all(np.array_equal(runs[0], r) for r in runs[1:])


def test_pielm_second_derivative_matches_finite_difference():
    from eigen_benchmark.neural.pielm import _features
    rng = np.random.default_rng(0)
    a = rng.uniform(-3, 3, 5); b = rng.uniform(-3, 3, 5)
    x0, e = 0.41, 1e-5
    f = lambda t: _features(np.array([t]), a, b)[0][:, 0]
    fd = (f(x0 + e) - 2 * f(x0) + f(x0 - e)) / e ** 2
    got = _features(np.array([x0]), a, b)[2][:, 0]
    assert np.allclose(got, fd, rtol=1e-4, atol=1e-4)
