"""P2 정확 기준해 — 환형 Kirchhoff 판(클램프 내경 / 자유 외경)의 Bessel 해.

∇⁴w = k⁴w, k⁴ = ρhω²/D. w = W(r)cos(mθ)일 때
    W(r) = A J_m(kr) + B Y_m(kr) + C I_m(kr) + E K_m(kr)
경계조건 4개로 4×4 동차계 → 행렬식 = 0의 근이 고유값.

  r = a (클램프):  W = 0,  W′ = 0
  r = b (자유):    W″ + ν(W′/r − m²W/r²) = 0                       (M_r = 0)
                   (∇²W)′ − (1−ν)(m²/r²)(W′ − W/r) = 0             (V_r = 0)
      ∇²W  = W″ + W′/r − m²W/r²
      (∇²W)′ = W‴ + W″/r − W′/r² − m²W′/r² + 2m²W/r³

**열 스케일 정규화가 필수**다. I_m은 e^{kr}로 폭증, K_m은 e^{−kr}로 소멸해
kb ≈ 26에서 열 크기가 10¹¹배 벌어진다. 각 열을 최대절댓값으로 나눈 뒤 행렬식을
취하면 근 위치는 그대로이면서 상쇄가 사라진다.

**고정밀 경로**는 mpmath Bessel + 도함수 점화식으로 별도 구현한다. fp64 함수를
mpmath findroot로 정밀화하려 하면 함수 자체의 정밀도가 fp64라 수렴하지 않는다
(잔차가 1e-50 부근에서 정체). 점화식(차수 이동 연산자 S의 이항전개):
    J,Y:  f⁽ⁿ⁾_m = 2⁻ⁿ Σ_j C(n,j)(−1)^j f_{m+2j−n}
    I:    f⁽ⁿ⁾_m = 2⁻ⁿ Σ_j C(n,j)     f_{m+2j−n}
    K:    f⁽ⁿ⁾_m = (−1)ⁿ 2⁻ⁿ Σ_j C(n,j) f_{m+2j−n}
r에 대한 도함수는 여기에 kⁿ을 곱한다.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ivp, jvp, kvp, yvp

from ..problems.p2_annulus import P2_GEOMETRY


def _derivs_fp64(k: float, r: float, m: int):
    """네 기본해의 (W, W′, W″, W‴)을 (4,4) 배열로. 행=기본해, 열=미분차수."""
    z = k * r
    out = np.empty((4, 4))
    for n in range(4):
        out[0, n] = jvp(m, z, n) * k ** n
        out[1, n] = yvp(m, z, n) * k ** n
        out[2, n] = ivp(m, z, n) * k ** n
        out[3, n] = kvp(m, z, n) * k ** n
    return out


def _derivs_mp(k, r, m: int):
    """고정밀 판(mpmath). 도함수는 차수 이동 점화식의 이항전개로 취한다."""
    from mpmath import besseli, besselj, besselk, bessely, binomial, mp

    z = k * r
    fams = ((besselj, -1, False), (bessely, -1, False),
            (besseli, +1, False), (besselk, +1, True))
    out = []
    for fn, sign, k_flip in fams:
        row = []
        for n in range(4):
            s = mp.mpf(0)
            for j in range(n + 1):
                order = m + 2 * j - n
                coef = binomial(n, j) * ((-1) ** j if sign < 0 else 1)
                s += coef * fn(order, z)
            s = s / (2 ** n)
            if k_flip:
                s = s * ((-1) ** n)
            row.append(s * k ** n)
        out.append(row)
    return out


def _bc_rows(Da, Db, b, nu, m):
    """경계조건 4행. Da/Db는 r=a, r=b에서의 (4해 × 4차수) 도함수 표.

    fp64(numpy)와 mpmath 양쪽에서 같은 대수를 쓰도록 인덱싱만으로 작성한다."""
    def col(D, n):
        return [D[i][n] for i in range(4)]

    W0a, W1a = col(Da, 0), col(Da, 1)
    W0b, W1b, W2b, W3b = col(Db, 0), col(Db, 1), col(Db, 2), col(Db, 3)

    moment, shear = [], []
    for i in range(4):
        moment.append(W2b[i] + nu * (W1b[i] / b - (m ** 2) * W0b[i] / b ** 2))
        lap_p = (W3b[i] + W2b[i] / b - W1b[i] / b ** 2
                 - (m ** 2) * W1b[i] / b ** 2 + 2 * (m ** 2) * W0b[i] / b ** 3)
        shear.append(lap_p - (1.0 - nu) * (m ** 2 / b ** 2) * (W1b[i] - W0b[i] / b))
    return [W0a, W1a, moment, shear]


def char_det(k: float, m: int, geometry=None, dps: int | None = None):
    """4×4 경계조건 행렬식(열 스케일 정규화). 근 위치는 스케일에 불변.

    dps를 주면 mpmath 고정밀로 계산한다."""
    g = P2_GEOMETRY if geometry is None else geometry
    a, b, nu = g["a"], g["b"], g["nu"]
    if dps is None:
        Da = _derivs_fp64(float(k), a, int(m))
        Db = _derivs_fp64(float(k), b, int(m))
        A = np.array(_bc_rows(Da, Db, b, nu, int(m)), dtype=float)
        scale = np.max(np.abs(A), axis=0)
        scale[scale == 0.0] = 1.0
        return float(np.linalg.det(A / scale))

    from mpmath import det, matrix, mp
    old = mp.dps
    mp.dps = dps
    try:
        kk = mp.mpf(k)
        Da = _derivs_mp(kk, mp.mpf(a), int(m))
        Db = _derivs_mp(kk, mp.mpf(b), int(m))
        rows = _bc_rows(Da, Db, mp.mpf(b), mp.mpf(nu), int(m))
        scale = [max(abs(rows[i][j]) for i in range(4)) for j in range(4)]
        scale = [s if s != 0 else mp.mpf(1) for s in scale]
        return det(matrix([[rows[i][j] / scale[j] for j in range(4)]
                           for i in range(4)]))
    finally:
        mp.dps = old


def _scan(m, geometry, k_max, n_scan):
    grid = np.linspace(1.0, k_max, n_scan)
    vals = np.array([char_det(k, m, geometry) for k in grid])
    out = []
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) \
                and vals[i] * vals[i + 1] < 0.0:
            out.append((float(grid[i]), float(grid[i + 1])))
    return out


_ROOT_CACHE: dict = {}


def annulus_k_roots(m: int, n_modes: int = 4, geometry=None, dps: int = 50,
                    k_max: float | None = None, n_scan: int = 4000) -> np.ndarray:
    """파수 k[1/m] 오름차순 n_modes개. dps>0이면 근을 고정밀로 정밀화.

    **결과를 캐시한다.** 순수함수인데 mpmath dps=50 근찾기가 근 하나에 수백 ms라
    (Bessel 4계열 × 도함수 4차 × 이항항 = 근당 수백 회 고정밀 평가) 드라이버가
    같은 값을 반복 계산하면 전체가 분 단위로 늘어난다."""
    g = P2_GEOMETRY if geometry is None else geometry
    key = (int(m), int(dps), k_max, int(n_scan), tuple(sorted(g.items())))
    hit = _ROOT_CACHE.get(key)
    if hit is not None and len(hit) >= n_modes:
        return hit[:n_modes].copy()      # 더 많이 계산해둔 결과를 잘라 쓴다
    L = g["b"] - g["a"]
    k_max = k_max or (n_modes + m + 3) * math.pi / L
    brackets = _scan(m, geometry, k_max, n_scan)
    if len(brackets) < n_modes:
        raise RuntimeError(f"m={m}: 근 {n_modes}개를 못 찾음(브래킷 {len(brackets)}개)")
    roots = [brentq(char_det, lo, hi, args=(m, geometry), xtol=1e-12, rtol=8.9e-16)
             for lo, hi in brackets[:n_modes]]
    if dps and dps > 0:
        from mpmath import findroot, mp
        old = mp.dps
        mp.dps = dps
        try:
            roots = [float(findroot(lambda z: char_det(z, m, geometry, dps=dps),
                                    mp.mpf(r0))) for r0 in roots]
        finally:
            mp.dps = old
    out = np.array(roots)
    if hit is None or len(out) > len(hit):
        _ROOT_CACHE[key] = out
    return out.copy()



def _null_coefficients(k: float, m: int, geometry=None):
    """4×4 경계조건계의 영공간 벡터(=모드형 계수). 열 스케일 되돌림 포함."""
    g = P2_GEOMETRY if geometry is None else geometry
    a, b, nu = g["a"], g["b"], g["nu"]
    Da = _derivs_fp64(float(k), a, int(m))
    Db = _derivs_fp64(float(k), b, int(m))
    A = np.array(_bc_rows(Da, Db, b, nu, int(m)), dtype=float)
    scale = np.max(np.abs(A), axis=0)
    scale[scale == 0.0] = 1.0
    _, _, Vt = np.linalg.svd(A / scale)
    return Vt[-1] / scale


def _combine(c, k: float, m: int, r):
    from scipy.special import iv, jv, kv, yv
    z = k * np.asarray(r, float)
    return c[0] * jv(m, z) + c[1] * yv(m, z) + c[2] * iv(m, z) + c[3] * kv(m, z)


def mode_shape(r, k: float, m: int, geometry=None, n_norm: int = 512) -> np.ndarray:
    """W(r) — [a,b] 전역 최댓값이 1이 되도록 정규화.

    **정규화 기준은 질의점이 아니라 고정 격자다.** 넘겨받은 점들로 정규화하면
    한 점만 물었을 때 항상 1이 나오고 스케일이 질의마다 달라져 합성이 깨진다."""
    g = P2_GEOMETRY if geometry is None else geometry
    c = _null_coefficients(k, m, geometry)
    grid = np.linspace(g["a"], g["b"], n_norm)
    peak = float(np.max(np.abs(_combine(c, k, m, grid))))
    W = _combine(c, k, m, r)
    return W / peak if peak > 0 else W
