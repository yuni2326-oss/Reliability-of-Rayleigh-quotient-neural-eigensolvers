"""(g) 시공간 PINN — negative result의 구조를 검증한다(성능이 아니라 구조)."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from eigen_benchmark.neural import spacetime as st       # noqa: E402
from eigen_benchmark.problems import p1_beam as p1       # noqa: E402


def test_bin_spacing_is_inverse_in_the_number_of_periods():
    """빈 간격 Δω/ω = 1/n_periods, Λ은 그 두 배."""
    a, b = st.bin_spacing(8), st.bin_spacing(32)
    assert a["rel_omega_bin"] == pytest.approx(1 / 8)
    assert a["rel_lambda_bin"] == pytest.approx(2 / 8)
    assert b["rel_lambda_bin"] == pytest.approx(a["rel_lambda_bin"] / 4)


def test_bin_spacing_uses_the_nondimensional_first_mode_frequency():
    f = st.bin_spacing(8)
    b1 = float(p1.beta_roots(1)[0])
    assert f["omega_nondim"] == pytest.approx(b1 ** 2)
    assert f["T_nondim"] == pytest.approx(8 * 2 * math.pi / b1 ** 2)


@pytest.mark.parametrize("n", [4, 8, 16, 32, 137])
def test_the_reference_frequency_always_lands_exactly_on_a_bin(n):
    """빈 간격은 **오차 하한이 아니다** — 창을 정수 주기로 잡으면 참조 주파수가 항상
    정확히 빈 번호 n_periods에 떨어진다. 이걸 하한으로 읽으면 짧은 창에서 오차가
    작아 보이는 것을 '좋은 결과'로 오독한다(실제로 4주기에서 e_λ가 빈 간격의 1/100)."""
    f = st.bin_spacing(n)
    d_omega = 2 * math.pi / f["T_nondim"]
    assert f["omega_nondim"] / d_omega == pytest.approx(n, rel=1e-12)


def test_network_satisfies_the_clamped_condition_exactly():
    """w = x²·N이므로 x=0에서 w와 w_x가 **비트 단위로** 0이어야 한다."""
    net = st.SpaceTimeNet(width=8, depth=2).double()
    t = torch.rand(5, 1, dtype=torch.float64)
    x = torch.zeros(5, 1, dtype=torch.float64, requires_grad=True)
    w = net(x, t)
    assert torch.all(w == 0)
    wx = torch.autograd.grad(w, x, torch.ones_like(w), create_graph=True)[0]
    assert torch.all(wx == 0)


def test_peak_frequency_recovers_a_known_sinusoid():
    """FFT 첨두 판독이 맞는지 — 알려진 정현파로."""
    om = 3.5160152
    T = 16 * 2 * math.pi / om
    n = 2048
    t = np.linspace(0.0, T, n)
    sig = np.cos(om * t)
    hat, d = st._peak_frequency(sig, t[1] - t[0])
    assert abs(hat - om) / om < 1 / 16          # 격자 하한 안
    assert d > 0


def test_solver_reports_the_error_the_bin_width_and_the_residual():
    """negative result의 논지는 잔차와 고유값 오차의 **역상관**이므로 둘 다 있어야 한다."""
    o = st.solve_spacetime(n_periods=4, iters=5, n_col=128, n_probe=128,
                           device="cpu")
    for k in ("e_lam", "rel_lambda_bin", "seconds", "n_params", "loss_final"):
        assert k in o
    assert o["rel_lambda_bin"] == pytest.approx(0.5)
    assert o["Lam_ref"] == pytest.approx(float(p1.beta_roots(1)[0]) ** 4)


def test_the_time_domain_route_is_far_more_expensive_than_the_classical_one():
    """이 arm의 존재 이유 — 같은 고유값 하나를 얻는 비용이 자리수로 다르다.

    고전 6 DOF 정규직교 단항식은 0.14 ms에 e_λ = 1.6e-10이다(p1_basis_study.csv).
    시공간 PINN은 5반복만 해도 그보다 오래 걸린다."""
    import time as _t
    t0 = _t.perf_counter()
    st.solve_spacetime(n_periods=4, iters=5, n_col=128, n_probe=128, device="cpu")
    assert _t.perf_counter() - t0 > 1e-3 * 10       # 고전 1회의 10배 이상
