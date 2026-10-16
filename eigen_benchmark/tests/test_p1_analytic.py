import math

import numpy as np
import pytest
from mpmath import mp

from eigen_benchmark.problems import p1_beam as p1


def test_first_four_betas_match_textbook_table():
    b = p1.beta_roots(4)
    expected = [1.8751041, 4.6940911, 7.8547574, 10.9955407]
    assert np.allclose(b, expected, rtol=1e-7)


def test_fourteen_betas_are_increasing_and_asymptotic():
    b = p1.beta_roots(14)
    assert len(b) == 14
    assert np.all(np.diff(b) > 0)
    # 고차 근은 (2n-1)π/2에 접근
    assert b[13] == pytest.approx((2 * 14 - 1) * math.pi / 2, rel=1e-9)


def test_betas_match_mpmath_to_1e14():
    # fp64 근찾기가 고정밀 근과 상대 1e-14 이내여야 한다(sech 형태 사용의 근거).
    mp.dps = 50
    b = p1.beta_roots(10)
    for k, val in enumerate(b, start=1):
        ref = mp.findroot(lambda z: mp.cos(z) + mp.sech(z), mp.mpf(val))
        assert abs(val - float(ref)) / float(ref) < 1e-14



def test_analytic_mode_satisfies_clamped_boundary_conditions():
    x = np.array([0.0])
    for n in (1, 2, 5):
        assert abs(p1.analytic_mode(x, n)[0]) < 1e-12
    # φ'(0)=0을 편차분으로 확인. φ(0)=φ'(0)=0이면 잔차는 ½φ''(0)ε뿐이므로
    # 허용오차를 곡률로 스케일한다(φ'(0)≠0이면 d가 O(1)이 되어 반드시 실패).
    eps = 1e-7
    for n in (1, 2, 5):
        d = (p1.analytic_mode(np.array([eps]), n)[0]
             - p1.analytic_mode(np.array([0.0]), n)[0]) / eps
        curv = abs(p1.analytic_mode_d2(np.array([0.0]), n)[0])
        assert abs(d) < 0.6 * curv * eps


def test_analytic_mode_rayleigh_quotient_equals_beta_fourth():
    # ∫(φ'')²/∫φ² = β⁴ — 이후 신경망 arm의 정합성 기준(계획 3에서 재사용).
    from eigen_benchmark.quadrature import gauss_legendre
    x, w = gauss_legendre(400, 0.0, 1.0)
    for n in (1, 2, 3):
        num = np.sum(w * p1.analytic_mode_d2(x, n) ** 2)
        den = np.sum(w * p1.analytic_mode(x, n) ** 2)
        assert num / den == pytest.approx(p1.beta_roots(3)[n - 1] ** 4, rel=1e-8)
