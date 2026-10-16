import numpy as np
import pytest

from eigen_benchmark.bases.monomial import MonomialBasis


def test_monomial_shapes_and_clamped_conditions():
    b = MonomialBasis(5)
    x = np.linspace(0.0, 1.0, 7)
    assert b.n_dof == 5
    assert b.eval(x).shape == (5, 7)
    assert b.d1(x).shape == (5, 7)
    assert b.d2(x).shape == (5, 7)
    # 지수 ≥2 → φ(0)=φ'(0)=0
    assert np.allclose(b.eval(np.array([0.0])), 0.0)
    assert np.allclose(b.d1(np.array([0.0])), 0.0)


def test_monomial_values_and_derivatives_are_exact():
    b = MonomialBasis(3)          # x², x³, x⁴
    x = np.array([0.5])
    assert np.allclose(b.eval(x).ravel(), [0.25, 0.125, 0.0625])
    assert np.allclose(b.d1(x).ravel(), [1.0, 0.75, 0.5])
    assert np.allclose(b.d2(x).ravel(), [2.0, 3.0, 3.0])


def test_c1_basis_has_zero_slope_jump():
    # 전역 다항은 C^∞ → ⟦ψ'⟧ = 0. P3에서 k_θ 항이 소거되는 근거.
    b = MonomialBasis(4)
    assert np.allclose(b.d1_jump(0.2), np.zeros(4))
