"""(ix) 전달행렬 / 동적강성 정확기준 — P3의 기준해이자 P1의 교차확인.

회전스프링 캔틸레버의 정확 특성방정식(8×8). 두 구간 각각
    φ_i(x̃) = A cosh(sx̃) + B sinh(sx̃) + C cos(sx̃) + D sin(sx̃),  s = βL
조건: φ(0)=φ′(0)=0, x_c에서 변위·모멘트·전단 연속 + 기울기 점프 Δφ′ = κ φ″,
자유단에서 φ″(1)=φ‴(1)=0.

**파라미터는 무차원 스프링강성 k̂ = k_θL/(EI) 하나뿐이고 κ = 1/k̂ 이다.**
특정 손상 기전(균열 깊이 등)으로 k̂를 유도하지 않는다 — 이 벤치마크의 관심사는
"영폭 계면의 기울기 점프를 기저가 표현할 수 있는가"라는 이산화 문제이지 손상
모델링이 아니다(손상 지문은 별도 논문의 영역).

**정밀도 전략**: 부호변화 스캔은 fp64(빠름), 근 정밀화는 mpmath(dps=50).
두 결과의 차를 그대로 보고해 e_algebraic의 증거로 쓴다. cosh(30)≈5.3e12이라
고차모드에서 fp64 행렬식은 상쇄가 심해진다 — 그 사실 자체가 데이터다.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


def kappa_from_k_hat(k_hat: float) -> float:
    """κ = 1/k̂. k̂ = k_θL/(EI)가 크면(강한 스프링) κ→0으로 균일보에 접근한다."""
    if k_hat <= 0:
        raise ValueError("k_hat은 양수여야 합니다")
    return 1.0 / float(k_hat)


def _rows(s, xc, kap, fns):
    cosh, sinh, cos, sin, zero = fns

    def f0(t): return [cosh(t), sinh(t), cos(t), sin(t)]
    def f1(t): return [sinh(t), cosh(t), -sin(t), cos(t)]
    def f2(t): return [cosh(t), sinh(t), -cos(t), -sin(t)]
    def f3(t): return [sinh(t), cosh(t), sin(t), -cos(t)]

    z = [zero] * 4
    tc, tL = s * xc, s
    one, nil = 1.0 * (zero + 1), zero

    def neg(v): return [-e for e in v]

    return [
        [one, nil, one, nil] + z,                          # φ(0) = 0
        [nil, one, nil, one] + z,                          # φ′(0) = 0
        z + f2(tL),                                        # φ″(1) = 0
        z + f3(tL),                                        # φ‴(1) = 0
        f0(tc) + neg(f0(tc)),                              # 변위 연속
        f2(tc) + neg(f2(tc)),                              # 모멘트 연속
        f3(tc) + neg(f3(tc)),                              # 전단 연속
        [-(a + kap * s * b) for a, b in zip(f1(tc), f2(tc))] + f1(tc),   # 기울기 점프
    ]


def char_det(s: float, xc: float, kappa: float, dps: int | None = None):
    """8×8 특성행렬식. dps를 주면 mpmath 고정밀로 계산."""
    if dps is None:
        fns = (math.cosh, math.sinh, math.cos, math.sin, 0.0)
        rows = _rows(float(s), float(xc), float(kappa), fns)
        return float(np.linalg.det(np.array(rows, dtype=float)))
    from mpmath import det, matrix, mp
    old = mp.dps
    mp.dps = dps
    try:
        fns = (mp.cosh, mp.sinh, mp.cos, mp.sin, mp.mpf(0))
        rows = _rows(mp.mpf(s), mp.mpf(xc), mp.mpf(kappa), fns)
        return det(matrix(rows))
    finally:
        mp.dps = old


def _scan_brackets(xc, kappa, n_modes, s_max, n_scan):
    grid = np.linspace(0.3, s_max, n_scan)
    vals = np.array([char_det(s, xc, kappa) for s in grid])
    out = []
    for i in range(len(grid) - 1):
        if vals[i] * vals[i + 1] < 0.0:
            out.append((float(grid[i]), float(grid[i + 1])))
        if len(out) >= n_modes:
            break
    return out


def spring_beam_betas(xc: float, kappa: float, n_modes: int = 6,
                      dps: int = 50, s_max: float | None = None,
                      n_scan: int = 6000) -> np.ndarray:
    """회전스프링 캔틸레버의 βL 근(오름차순). fp64 스캔 + mpmath 정밀화."""
    from mpmath import findroot, mp
    s_max = s_max or ((2 * n_modes + 1) * math.pi / 2 + 2.0)
    brackets = _scan_brackets(xc, kappa, n_modes, s_max, n_scan)
    if len(brackets) < n_modes:
        raise RuntimeError(f"근 {n_modes}개를 못 찾음(브래킷 {len(brackets)}개)")
    roots = []
    old = mp.dps
    mp.dps = dps
    try:
        for lo, hi in brackets:
            g = brentq(char_det, lo, hi, args=(xc, kappa), xtol=1e-13)
            r = findroot(lambda s: char_det(s, xc, kappa, dps=dps), mp.mpf(g))
            roots.append(float(r))
    finally:
        mp.dps = old
    return np.array(roots)


def fp64_vs_highprec_betas(xc: float, kappa: float, n_modes: int = 6,
                           dps: int = 50) -> dict:
    """같은 근을 fp64와 고정밀로 각각 구해 상대차를 남긴다(e_algebraic 증거)."""
    s_max = (2 * n_modes + 1) * math.pi / 2 + 2.0
    brackets = _scan_brackets(xc, kappa, n_modes, s_max, 6000)
    f64 = [brentq(char_det, lo, hi, args=(xc, kappa), xtol=1e-13)
           for lo, hi in brackets[:n_modes]]
    hp = spring_beam_betas(xc, kappa, n_modes, dps=dps).tolist()
    return {"fp64": f64, "highprec": hp,
            "rel_diff": [abs(a - b) / abs(b) for a, b in zip(f64, hp)]}
