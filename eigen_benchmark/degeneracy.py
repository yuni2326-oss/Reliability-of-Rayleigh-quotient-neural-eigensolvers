"""축퇴 고유공간 — m > 0 절점직경 쌍을 고유공간으로 비교하기 위한 배선.

m > 0에서 W(r)cos(mθ)와 W(r)sin(mθ)는 **같은 고유값**을 갖는다. 솔버가 이 2차원
공간 안의 임의로 회전된 기저를 내놓아도 틀린 답이 아니다. 그래서 모드별 MAC이 아니라
부분공간 MAC·주각으로 판정해야 한다(§3.5, Babuška–Osborn의 다중고유값 처리).

여기서는 (r,θ) 격자에서 정확 쌍과 회전된 쌍을 만들어 그 사실을 데이터로 보인다.
질량가중은 면적요소 r dr dθ (판의 ρh가 균일하므로 상수배는 무관).
"""
from __future__ import annotations

import numpy as np

from .problems.p2_annulus import P2_GEOMETRY
from .reference.bessel_annulus import mode_shape


def polar_grid(n_r: int = 40, n_theta: int = 64, geometry=None):
    """환형 (r,θ) 격자와 면적가중 r·Δr·Δθ. 반환 (r_flat, theta_flat, weights)."""
    g = P2_GEOMETRY if geometry is None else geometry
    a, b = g["a"], g["b"]
    dr = (b - a) / n_r
    dth = 2.0 * np.pi / n_theta
    r_c = a + dr * (np.arange(n_r) + 0.5)              # 중점법
    th_c = dth * (np.arange(n_theta) + 0.5)
    R, TH = np.meshgrid(r_c, th_c, indexing="ij")
    Wt = R * dr * dth
    return R.ravel(), TH.ravel(), Wt.ravel()


def degenerate_pair(k: float, m: int, grid, geometry=None) -> np.ndarray:
    """(n_points, 2) — 열이 W(r)cos(mθ), W(r)sin(mθ). m=0이면 둘째 열은 0."""
    r, th, _ = grid
    W = mode_shape(r, k, m, geometry)
    return np.column_stack([W * np.cos(m * th), W * np.sin(m * th)])


def rotated_pair(k: float, m: int, grid, alpha: float, geometry=None) -> np.ndarray:
    """같은 고유공간의 다른 정규직교 기저 — θ를 α만큼 회전한 쌍."""
    P = degenerate_pair(k, m, grid, geometry)
    c, s = np.cos(m * alpha), np.sin(m * alpha)
    return P @ np.array([[c, -s], [s, c]])
