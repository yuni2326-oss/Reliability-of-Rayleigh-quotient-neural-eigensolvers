import numpy as np
import pytest

from eigen_benchmark.bases.enriched import EnrichedBasis
from eigen_benchmark.problems import p1_beam as p1


def test_clamped_boundary_conditions_hold():
    b = EnrichedBasis(n_global=6, xc=0.2, n_enrich=2)
    z = np.array([0.0])
    assert np.allclose(b.eval(z), 0.0)
    assert np.allclose(b.d1(z), 0.0)


def test_enrichment_is_zero_left_of_xc():
    b = EnrichedBasis(n_global=4, xc=0.2, n_enrich=2)
    x = np.array([0.1])
    assert np.allclose(b.eval(x)[4:], 0.0)
    assert np.allclose(b.d1(x)[4:], 0.0)
    assert np.allclose(b.d2(x)[4:], 0.0)


def test_first_enrichment_gives_unit_slope_jump():
    b = EnrichedBasis(n_global=4, xc=0.2, n_enrich=2)
    j = b.d1_jump(0.2)
    assert np.allclose(j[:4], 0.0)          # 전역 다항은 점프 없음
    assert j[4] == pytest.approx(1.0)       # 램프 (x−x_c)H
    assert j[5] == pytest.approx(0.0)       # (x−x_c)²H는 기울기 연속


def test_enrichment_creates_a_zero_energy_hinge_in_the_spring_free_problem():
    """램프 E_1은 broken form에서 굽힘에너지가 정확히 0이다(E_1″ ≡ 0, 조각별).
    따라서 스프링 항이 없는 P1에 이 기저를 쓰면 x_c에 **자유 힌지 기구**가 생겨
    K가 특이해지고 λ≈0이 나온다 — 강화기저는 P1에 비적합이며 k̂⟦u′⟧⟦v′⟧가 힌지를
    구속하는 P3에서만 유효하다. 이 성질을 명시적으로 고정해 오용을 막는다."""
    r = p1.solve(EnrichedBasis(n_global=8, xc=0.2, n_enrich=2), n_q=300, n_modes=1)
    ref1 = p1.beta_roots(1)[0] ** 4
    assert r["Lam"][0] < 1e-6 * ref1


def test_ramp_enrichment_has_zero_bending_energy():
    # 위 힌지 기구의 직접 증거: 램프의 2차도함수가 x_c 양쪽에서 0.
    b = EnrichedBasis(n_global=3, xc=0.2, n_enrich=1)
    x = np.array([0.05, 0.15, 0.25, 0.9])
    assert np.allclose(b.d2(x)[3], 0.0)


def test_second_derivative_is_piecewise_correct():
    b = EnrichedBasis(n_global=2, xc=0.2, n_enrich=2)
    x = np.array([0.5])
    # (x−xc)H → d2 = 0,  (x−xc)²H → d2 = 2
    assert b.d2(x)[2, 0] == pytest.approx(0.0)
    assert b.d2(x)[3, 0] == pytest.approx(2.0)


def test_breaks_include_xc_for_quadrature_alignment():
    b = EnrichedBasis(n_global=4, xc=0.2, n_enrich=1)
    assert 0.2 in set(np.round(np.asarray(b.breaks), 12))
