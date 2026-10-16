import numpy as np
import pytest

from eigen_benchmark.bases.monomial import MonomialBasis, orthonormalized
from eigen_benchmark.problems import p2_annulus as p2
from eigen_benchmark.reference import bessel_annulus as ba


def test_mpmath_derivative_recurrence_matches_scipy():
    """고정밀 경로의 Bessel 도함수는 차수 이동 점화식의 이항전개로 구한다.
    그 구현이 옳은지 scipy의 jvp/yvp/ivp/kvp와 fp64 수준에서 대조한다."""
    from mpmath import mp
    from scipy.special import ivp, jvp, kvp, yvp
    mp.dps = 40
    k, r, m = 300.0, 0.02, 2
    got = ba._derivs_mp(mp.mpf(k), mp.mpf(r), m)
    z = k * r
    ref = [[jvp(m, z, n) * k ** n for n in range(4)],
           [yvp(m, z, n) * k ** n for n in range(4)],
           [ivp(m, z, n) * k ** n for n in range(4)],
           [kvp(m, z, n) * k ** n for n in range(4)]]
    for i in range(4):
        for n in range(4):
            assert float(got[i][n]) == pytest.approx(ref[i][n], rel=1e-10)


def test_highprec_char_det_matches_fp64_sign_structure():
    k = ba.annulus_k_roots(1, n_modes=1, dps=0)[0]
    for delta in (-0.01, 0.01):
        kk = k * (1 + delta)
        assert np.sign(ba.char_det(kk, 1)) == np.sign(float(ba.char_det(kk, 1, dps=40)))


def test_roots_are_positive_and_increasing():
    k = ba.annulus_k_roots(m=0, n_modes=4)
    assert len(k) == 4
    assert np.all(k > 0)
    assert np.all(np.diff(k) > 0)


def test_char_det_changes_sign_around_each_root():
    for m in (0, 2):
        for k in ba.annulus_k_roots(m, n_modes=2):
            lo, hi = k * 0.995, k * 1.005
            assert ba.char_det(lo, m) * ba.char_det(hi, m) < 0.0


def test_mode_shape_satisfies_clamped_inner_edge():
    g = p2.P2_GEOMETRY
    k = ba.annulus_k_roots(1, n_modes=1)[0]
    a, L = g["a"], g["b"] - g["a"]
    W0 = ba.mode_shape(np.array([a]), k, 1)[0]
    eps = 1e-7 * L
    dW = (ba.mode_shape(np.array([a + eps]), k, 1)[0] - W0) / eps
    assert abs(W0) < 1e-8            # 전역 최댓값이 1로 정규화돼 있음
    assert abs(dW) * L < 1e-4


def test_mode_shape_normalization_is_independent_of_query_points():
    """정규화 기준이 질의점이면 한 점만 물었을 때 항상 1이 나와 합성이 깨진다.
    고정 격자 기준이어야 부분집합을 물어도 같은 값이 나온다."""
    k = ba.annulus_k_roots(2, n_modes=1)[0]
    g = p2.P2_GEOMETRY
    dense = np.linspace(g["a"], g["b"], 200)
    full = ba.mode_shape(dense, k, 2)
    single = ba.mode_shape(np.array([dense[137]]), k, 2)[0]
    assert single == pytest.approx(full[137], rel=1e-12)
    assert np.max(np.abs(full)) == pytest.approx(1.0, rel=1e-3)


def test_bessel_reference_agrees_with_energy_ritz():
    """**핵심 교차검증**: 독립 두 방법이 같은 값을 줘야 한다.
    Bessel 정확해(미분방정식 + 자유단 BC 직접 부과)와 에너지형 Ritz(자연경계조건이
    변분원리에서 자동 만족)는 유도 경로가 완전히 달라, 일치하면 자유단 BC 유도가 옳다."""
    for m in (0, 1, 2, 3, 4):
        k_exact = ba.annulus_k_roots(m, n_modes=1)[0]
        lam_ritz = p2.solve(orthonormalized(MonomialBasis(14)), m=m,
                            n_q=400, n_modes=1)["Lam"][0]
        assert abs(lam_ritz - k_exact ** 4) / k_exact ** 4 < 1e-6


def test_higher_radial_orders_also_agree_with_ritz():
    m = 0
    k = ba.annulus_k_roots(m, n_modes=3)
    lam = p2.solve(orthonormalized(MonomialBasis(16)), m=m, n_q=500, n_modes=3)["Lam"]
    for i in range(3):
        assert abs(lam[i] - k[i] ** 4) / k[i] ** 4 < 1e-4



def test_root_cache_returns_slices_of_longer_results():
    """근 4개를 구해둔 뒤 1개를 물으면 재계산 없이 잘라 써야 한다.
    mpmath 근찾기가 근 하나에 수백 ms라 드라이버에서 이 재사용이 분 단위 차이를 만든다."""
    import time
    ba._ROOT_CACHE.clear()
    four = ba.annulus_k_roots(3, n_modes=4)
    t0 = time.perf_counter()
    one = ba.annulus_k_roots(3, n_modes=1)
    elapsed = time.perf_counter() - t0
    assert one[0] == four[0]
    assert elapsed < 0.05


def test_root_cache_returns_copies_so_callers_cannot_corrupt_it():
    ba._ROOT_CACHE.clear()
    a = ba.annulus_k_roots(1, n_modes=2)
    a[0] = -999.0
    b = ba.annulus_k_roots(1, n_modes=2)
    assert b[0] > 0


def test_highprec_and_fp64_roots_agree():
    hp = ba.annulus_k_roots(2, n_modes=3, dps=50)
    fp = ba.annulus_k_roots(2, n_modes=3, dps=0)     # dps=0 → fp64만
    assert np.allclose(hp, fp, rtol=1e-9)
