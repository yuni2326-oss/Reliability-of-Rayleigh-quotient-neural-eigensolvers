import numpy as np
import pytest

from eigen_benchmark.quadrature import gauss_legendre, piecewise_gauss


def test_gauss_integrates_polynomial_of_degree_2n_minus_1_exactly():
    # n점 Gauss는 차수 2n-1까지 정확. ∫₀¹ x⁹ dx = 1/10.
    x, w = gauss_legendre(5, 0.0, 1.0)
    assert np.sum(w * x ** 9) == pytest.approx(1.0 / 10.0, rel=1e-14)


def test_gauss_on_shifted_interval():
    # ∫₂⁵ x² dx = (125-8)/3 = 39
    x, w = gauss_legendre(4, 2.0, 5.0)
    assert np.sum(w * x ** 2) == pytest.approx(39.0, rel=1e-14)


def test_piecewise_gauss_handles_kink_exactly():
    # |x - 0.2|는 x=0.2에서 미분불가. 조각분할 없으면 오차가 남고, 분할하면 정확.
    xc = 0.2
    exact = 0.5 * xc ** 2 + 0.5 * (1 - xc) ** 2
    x, w = piecewise_gauss([0.0, xc, 1.0], 6)
    assert np.sum(w * np.abs(x - xc)) == pytest.approx(exact, rel=1e-14)


def test_piecewise_gauss_total_weight_is_interval_length():
    x, w = piecewise_gauss([0.0, 0.2, 1.0], 8)
    assert np.sum(w) == pytest.approx(1.0, rel=1e-15)
    assert len(x) == 16


def test_piecewise_gauss_rejects_unsorted_breaks():
    with pytest.raises(ValueError):
        piecewise_gauss([0.0, 1.0, 0.2], 4)
