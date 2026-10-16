import numpy as np
import pytest

from eigen_benchmark.bases.monomial import MonomialBasis, orthonormalized
from eigen_benchmark.problems import p1_beam as p1


def test_orthonormalized_gram_is_identity():
    b = orthonormalized(MonomialBasis(8))
    r = p1.solve(b, n_q=400, n_modes=3)
    assert np.allclose(r["M"], np.eye(8), atol=1e-10)


def test_orthonormalization_preserves_the_function_space():
    # 같은 span → 같은 Ritz 고유값(좌표변환은 스펙트럼 불변).
    # 원시 좌표가 이미 여러 자리를 잃으므로 허용오차는 1e-6로 둔다.
    n = 8
    a = p1.solve(MonomialBasis(n), n_q=400, n_modes=4)
    b = p1.solve(orthonormalized(MonomialBasis(n)), n_q=400, n_modes=4)
    assert np.allclose(a["Lam"][:4], b["Lam"][:4], rtol=1e-6)


def test_coordinate_choice_alone_produces_algebraic_error():
    """같은 함수공간의 두 좌표계가 서로 다른 값을 준다 = 순수 e_algebraic.

    span이 동일하므로 e_approx는 같다. 따라서 차이는 전부 좌표(조건수) 탓이며,
    해석기준에 더 가까운 쪽이 정규직교 좌표라는 것이 §5.3의 (i)/(ii) 대조 논거다."""
    n = 8
    ref = p1.beta_roots(4)[3] ** 4
    raw = p1.solve(MonomialBasis(n), n_q=400, n_modes=4)
    ort = p1.solve(orthonormalized(MonomialBasis(n)), n_q=400, n_modes=4)
    # 두 좌표계의 불일치가 반올림 수준을 넘어 실재한다
    assert abs(raw["Lam"][3] - ort["Lam"][3]) / ort["Lam"][3] > 1e-10
    # 그리고 정규직교 좌표가 해석기준에 더 가깝다(또는 최소한 나쁘지 않다)
    assert (abs(ort["Lam"][3] - ref) / ref) <= (abs(raw["Lam"][3] - ref) / ref)


def test_orthonormalization_survives_where_raw_monomial_breaks_down():
    # N=20: 원시 단항식은 fp64 Cholesky가 실패하지만 정규직교 좌표는 정상 동작.
    n = 20
    raw = p1.solve(MonomialBasis(n), n_q=400, n_modes=3)
    ort = p1.solve(orthonormalized(MonomialBasis(n)), n_q=400, n_modes=3)
    assert raw["cholesky_ok"] is False
    assert ort["cholesky_ok"] is True
    assert np.linalg.cond(raw["M"]) > 1e14
    assert np.linalg.cond(ort["M"]) < 1e3
    # 정규직교 좌표는 해석 기본고유값을 회복한다. 단 완전하지는 않다 —
    # 부모기저가 fp64에서 이미 수치적 rank 결손이라 QR도 잃은 정보는 복원 못 하고
    # 상대 ~2.5e-10의 e_algebraic이 남는다(원시 좌표의 완전 실패와 대비되는 실측값).
    ref1 = p1.beta_roots(1)[0] ** 4
    assert abs(ort["Lam"][0] - ref1) / ref1 < 1e-8
