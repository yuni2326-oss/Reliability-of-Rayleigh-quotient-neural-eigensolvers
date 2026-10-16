import numpy as np

from eigen_benchmark.bases.monomial import MonomialBasis
from eigen_benchmark.problems import p1_beam as p1
from eigen_benchmark.quadrature_study import doubling_table


def test_doubling_table_changes_only_at_roundoff_once_the_rule_is_exact():
    # n_q는 기저 자유도 이상이어야 질량행렬이 full-rank다(아래 테스트 참조).
    rows = doubling_table(lambda: MonomialBasis(8),
                          lambda b, nq: p1.solve(b, n_q=nq, n_modes=1)["Lam"][0],
                          [9, 10, 20, 40])
    assert [r["n_q"] for r in rows] == [9, 10, 20, 40]
    assert rows[0]["abs_change"] is None
    assert all(np.isfinite(r["value"]) for r in rows)
    # 8항 단항 기저(x² … x⁹)의 질량 피적분은 18차, 강성은 14차이고 n_q점 Gauss는 2n_q − 1차까지 정확하다.
    # n_q ≥ 10이면 두 행렬이 정확하므로 배증은 반올림만 바꾼다. 예전 판은 반올림 잡음끼리의 대소(배증할수록
    # 준다)를 검사해 라이브러리 판에 따라 갈렸다 — NumPy 2.1.0/SciPy 1.17.1에서 8.2e-12 → 3.0e-11.
    lam = rows[-1]["value"]
    assert max(r["abs_change"] for r in rows[1:]) <= 1e-10 * abs(lam)


def test_doubling_table_separates_quadrature_from_basis_error():
    # 같은 기저(8항)를 두 구적차수로 풀면 기저오차는 같고 구적오차만 다르다.
    rows = doubling_table(lambda: MonomialBasis(8),
                          lambda b, nq: p1.solve(b, n_q=nq, n_modes=1)["Lam"][0],
                          [30, 60])
    assert abs(rows[1]["value"] - rows[0]["value"]) < 1e-8


def test_under_integration_below_n_dof_makes_mass_matrix_rank_deficient():
    """Gauss 점수가 기저 자유도보다 적으면 M = Ψ diag(w) Ψᵀ의 rank가 n_q로 제한되어
    구성상 특이해진다 — 구적차수를 기저차수와 독립으로 두되 **하한은 n_dof**라는 조건.
    §5.3에서 구적 부족을 기저 부족으로 오독하지 않도록 이 경계를 명시한다."""
    r = p1.solve(MonomialBasis(8), n_q=5, n_modes=1)
    assert r["cholesky_ok"] is False
    assert np.linalg.matrix_rank(r["M"]) <= 5
