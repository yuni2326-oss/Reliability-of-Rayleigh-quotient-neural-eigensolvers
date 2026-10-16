"""P2의 독립 이산화 — 반경방향 3차 Hermite C¹ 요소.

계획 1의 `bases/fem.py`는 등간격 보요소라 닫힌형 행렬을 썼지만, 환형판 에너지는
계수 1/r, 1/r²가 r에 의존해 닫힌형이 없다. 따라서 요소마다 Gauss 구적으로
    K_ij = ∫ [A_iA_j − (1−ν)(κ_rr,i κ_θθ,j + κ_θθ,i κ_rr,j) + 2(1−ν)κ_rθ,i κ_rθ,j] r dr
    M_ij = ∫ W_i W_j r dr
를 적분한다. 클램프 내경의 (W, W′) 두 DOF를 제거해 BC를 부과한다.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError, eigh

from ..problems.p2_annulus import P2_GEOMETRY
from ..quadrature import gauss_legendre


def _hermite(s, h):
    """국소좌표 s∈[0,1], 요소길이 h에서 3차 Hermite 형상함수의 값·1차·2차도함수.
    DOF 순서 (W_left, W′_left, W_right, W′_right). 각 반환은 (4, len(s))."""
    s = np.asarray(s, float)
    N = np.stack([1 - 3 * s ** 2 + 2 * s ** 3,
                  h * (s - 2 * s ** 2 + s ** 3),
                  3 * s ** 2 - 2 * s ** 3,
                  h * (-s ** 2 + s ** 3)])
    dN = np.stack([(-6 * s + 6 * s ** 2) / h,
                   (1 - 4 * s + 3 * s ** 2),
                   (6 * s - 6 * s ** 2) / h,
                   (-2 * s + 3 * s ** 2)])
    d2N = np.stack([(-6 + 12 * s) / h ** 2,
                    (-4 + 6 * s) / h,
                    (6 - 12 * s) / h ** 2,
                    (-2 + 6 * s) / h])
    return N, dN, d2N


def radial_annulus_matrices(n_elem: int, m: int, geometry=None, n_q: int = 6):
    """클램프 내경 DOF가 제거된 (K, M)."""
    if n_elem < 1:
        raise ValueError("n_elem은 1 이상이어야 합니다")
    g = P2_GEOMETRY if geometry is None else geometry
    a, b, nu = g["a"], g["b"], g["nu"]
    nodes = np.linspace(a, b, n_elem + 1)
    n_dof = 2 * (n_elem + 1)
    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))
    s, w = gauss_legendre(n_q, 0.0, 1.0)
    for e in range(n_elem):
        r0, r1 = nodes[e], nodes[e + 1]
        h = r1 - r0
        r = r0 + h * s
        wr = w * h * r                                  # ∫ … r dr
        N, dN, d2N = _hermite(s, h)
        krr = d2N
        kth = dN / r - (m ** 2) * N / r ** 2
        krt = m * (dN / r - N / r ** 2)
        A = krr + kth
        Ke = (A * wr) @ A.T
        cross = (krr * wr) @ kth.T
        Ke = Ke - (1 - nu) * (cross + cross.T) + 2 * (1 - nu) * ((krt * wr) @ krt.T)
        Me = (N * wr) @ N.T
        idx = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
        K[np.ix_(idx, idx)] += Ke
        M[np.ix_(idx, idx)] += Me
    keep = np.arange(2, n_dof)                          # 클램프 내경 DOF 제거
    return K[np.ix_(keep, keep)], M[np.ix_(keep, keep)]


def solve_radial_fem(n_elem: int, m: int, n_modes: int = 4, geometry=None) -> dict:
    K, M = radial_annulus_matrices(n_elem, m, geometry)
    K = 0.5 * (K + K.T)
    M = 0.5 * (M + M.T)
    n = K.shape[0]
    try:
        Lam, V = eigh(K, M)
        ok = True
    except (LinAlgError, np.linalg.LinAlgError):
        Lam, V, ok = np.full(n, np.nan), np.full((n, n), np.nan), False
    k = min(n_modes, n)
    return {"Lam": Lam[:k], "vec": V[:, :k],
            "K": K, "M": M, "cholesky_ok": ok,
            "basis": f"radial_hermite_fem(n_elem={n_elem})", "m": int(m)}
