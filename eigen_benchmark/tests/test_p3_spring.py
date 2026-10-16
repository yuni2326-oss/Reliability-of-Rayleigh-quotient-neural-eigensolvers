import numpy as np
import pytest

from eigen_benchmark.bases.bspline import BSplineC0Basis
from eigen_benchmark.bases.enriched import EnrichedBasis
from eigen_benchmark.bases.monomial import MonomialBasis, orthonormalized
from eigen_benchmark.problems import p1_beam as p1
from eigen_benchmark.problems import p3_spring as p3


def test_kappa_is_the_reciprocal_of_the_spring_stiffness():
    from eigen_benchmark.reference.transfer_matrix import kappa_from_k_hat
    assert kappa_from_k_hat(10.0) == pytest.approx(0.1, rel=1e-15)
    with pytest.raises(ValueError):
        kappa_from_k_hat(0.0)


def test_config_declares_a_decade_sweep_from_near_hinge_to_near_rigid():
    ks = p3.P3_CONFIG["k_hats"]
    assert ks == (1.0, 10.0, 100.0, 1000.0)
    assert p3.P3_CONFIG["k_hat_central"] in ks


def test_reference_betas_are_below_uniform_beam():
    ref = p3.reference_betas(10.0, n_modes=4)
    assert np.all(ref < p1.beta_roots(4))


def test_c1_global_basis_saturates_at_the_uniform_beam_limit():
    """**핵심 소견**: C¹ 전역기저는 ⟦u′⟧≡0이라 k_θ 항이 소거된다.
    따라서 DOF를 아무리 늘려도 스프링보가 아니라 **균일보** 고유값으로 수렴한다 —
    세분으로 제거되지 않는 e_approx 하한."""
    # 좌표는 정규직교화해 조건수 붕괴(e_algebraic)를 배제한다 — 포화는 공간의 성질이다.
    uniform = p1.beta_roots(1)[0] ** 4
    spring = p3.reference_betas(10.0, n_modes=1)[0] ** 4
    lam = [p3.solve_basis(orthonormalized(MonomialBasis(n)), 10.0,
                          n_q=400, n_modes=1)["Lam"][0]
           for n in (8, 12, 16)]
    # 균일보 극한으로 수렴
    assert abs(lam[-1] - uniform) / uniform < 1e-6
    # 스프링 기준과는 유한한 간극이 남는다(포화)
    assert abs(lam[-1] - spring) / spring > 0.01
    # 게다가 그 간극이 DOF 증가로 줄지 않는다
    gaps = [abs(v - spring) / spring for v in lam]
    assert gaps[-1] > 0.5 * gaps[0]


def test_enriched_basis_converges_to_the_spring_reference():
    spring = p3.reference_betas(10.0, n_modes=3) ** 4
    r = p3.solve_basis(EnrichedBasis(10, xc=0.2, n_enrich=3), 10.0,
                       n_q=400, n_modes=3)
    assert abs(r["Lam"][0] - spring[0]) / spring[0] < 1e-6
    assert abs(r["Lam"][2] - spring[2]) / spring[2] < 1e-4


def test_enriched_failure_at_high_dof_is_coordinates_not_space():
    """강화공간은 옳다 — 고차에서 깨지는 건 단항식 **좌표**다.
    같은 공간을 정규직교 좌표로 쓰면 Cholesky가 살아난다. 다만 부모기저 평가 자체가
    fp64에서 정밀도를 잃으므로 DOF를 늘릴수록 정확도는 오히려 나빠진다 —
    공간이 이미 해를 담고 있으면 DOF 추가는 조건수 손해만 남긴다."""
    spring = p3.reference_betas(10.0, n_modes=1)[0] ** 4
    raw = p3.solve_basis(EnrichedBasis(22, xc=0.2, n_enrich=2), 10.0,
                         n_q=60, n_modes=1)
    ort = p3.solve_basis(orthonormalized(EnrichedBasis(22, xc=0.2, n_enrich=2)),
                         10.0, n_q=60, n_modes=1)
    assert raw["cholesky_ok"] is False
    assert ort["cholesky_ok"] is True
    assert abs(ort["Lam"][0] - spring) / spring < 1e-5

    small = p3.solve_basis(orthonormalized(EnrichedBasis(10, xc=0.2, n_enrich=2)),
                           10.0, n_q=32, n_modes=1)
    assert abs(small["Lam"][0] - spring) / spring < 1e-10      # N=12에서 이미 거의 정확
    assert (abs(small["Lam"][0] - spring) < abs(ort["Lam"][0] - spring))


def test_c0_bspline_converges_to_the_spring_reference():
    spring = p3.reference_betas(10.0, n_modes=3) ** 4
    r = p3.solve_basis(BSplineC0Basis(20, degree=3, xc=0.2, c0=True), 10.0,
                       n_q=12, n_modes=3)
    assert abs(r["Lam"][0] - spring[0]) / spring[0] < 1e-5


def test_fem_with_split_rotation_converges_to_the_spring_reference():
    spring = p3.reference_betas(10.0, n_modes=3) ** 4
    r = p3.solve_fem(20, 10.0, n_modes=3)
    assert abs(r["Lam"][0] - spring[0]) / spring[0] < 1e-5
    assert abs(r["Lam"][2] - spring[2]) / spring[2] < 1e-3


def test_stiffer_spring_approaches_uniform_beam():
    # ā가 작을수록(스프링 강할수록) 균일보에 가까워야 한다.
    lam_small = p3.solve_fem(20, 1000.0, n_modes=1)["Lam"][0]
    lam_large = p3.solve_fem(20, 1.0, n_modes=1)["Lam"][0]
    uniform = p1.beta_roots(1)[0] ** 4
    assert abs(lam_small - uniform) < abs(lam_large - uniform)


def test_misaligned_quadrature_degrades_the_enriched_basis():
    # 구적을 조각경계에 정렬하지 않으면(단일 [0,1] 규칙) 강화항의 불연속을
    # 매끄러운 것으로 오적분해 정확기준에서 멀어진다.
    spring = p3.reference_betas(10.0, n_modes=1)[0] ** 4
    good = p3.solve_basis(EnrichedBasis(10, xc=0.2, n_enrich=2), 10.0,
                          n_q=200, n_modes=1, split_at_xc=True)["Lam"][0]
    bad = p3.solve_basis(EnrichedBasis(10, xc=0.2, n_enrich=2), 10.0,
                         n_q=200, n_modes=1, split_at_xc=False)["Lam"][0]
    assert abs(good - spring) / spring < abs(bad - spring) / spring


def test_stiffer_spring_gives_higher_reference_frequency():
    # k̂가 클수록 계면이 단단해져 균일보에 접근한다 → βL 단조 증가.
    b = [p3.reference_betas(k, n_modes=1)[0] for k in p3.P3_CONFIG["k_hats"]]
    assert b[0] < b[1] < b[2] < b[3]
    assert b[-1] < p1.beta_roots(1)[0]                    # 유한 강성이면 균일보 미만


def test_near_rigid_spring_approaches_the_uniform_beam():
    b = p3.reference_betas(1000.0, n_modes=1)[0]
    ref = p1.beta_roots(1)[0]
    assert abs(b - ref) / ref < 5e-3
