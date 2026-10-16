"""P3 — 영폭 회전스프링, **모든 솔버가 같은 약형식**을 이산화한다.

무차원화(x̃ = x/L, Λ = ω²ρAL⁴/EI):
    a(u,v) = ∫₀^{x̃c} u″v″ + ∫_{x̃c}^{1} u″v″ + k̂⟦u′⟧⟦v′⟧,   b(u,v) = ∫₀¹ uv
    **k̂ = k_θL/(EI) 가 유일한 파라미터**, κ = 1/k̂
V = {u : 각 구간에서 H², ⟦u⟧ = 0}.

k̂를 특정 손상 기전(균열 깊이 등)으로 유도하지 않는다. 이 문제의 관심사는 영폭
계면의 기울기 점프를 **기저가 표현할 수 있는가**이지 손상 모델링이 아니다.
k̂ ∈ {1, 10, 100, 1000}의 십진 스윕이 near-hinge(k̂=1)부터 near-rigid(k̂=1000)까지
전 영역을 덮는다.

v1/v2의 유한폭 대체모델이 e_model을 끌고 들어왔던 것을 제거한 정본 정식화다.
**C¹ 기저는 ⟦u′⟧ ≡ 0이라 k̂ 항이 사라져 균일보 극한으로 수렴한다** — 세분으로
없어지지 않는 e_approx 하한이며, 이것이 §5.3의 핵심 대조축이다.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError, eigh

from ..quadrature import piecewise_gauss

P3_CONFIG = {"xc_over_L": 0.2,
             "k_hats": (1.0, 10.0, 100.0, 1000.0),
             "k_hat_central": 10.0}


def assemble(basis, k_hat_value: float, xc: float, n_q: int,
             split_at_xc: bool = True):
    """P3 약형식의 이산 (K, M). 점프항은 ⟦ψ′⟧의 외적으로 들어간다.

    구적은 기저의 조각경계와 x_c를 합집합한 분할에 정렬한다.
    **split_at_xc=False는 정렬을 통째로 버리고 [0,1] 단일구간을 쓴다** — 조각다항
    피적분함수를 매끄러운 것으로 오적분하는 효과를 재기 위한 대조군(§3.6)."""
    if split_at_xc:
        base = list(getattr(basis, "breaks", (0.0, 1.0)))
        breaks = sorted(set(np.round(base + [xc], 12)))
    else:
        breaks = [0.0, 1.0]
    x, w = piecewise_gauss(breaks, n_q)
    P, D2 = basis.eval(x), basis.d2(x)
    K = (D2 * w) @ D2.T
    M = (P * w) @ P.T
    j = basis.d1_jump(xc)
    K = K + k_hat_value * np.outer(j, j)
    return 0.5 * (K + K.T), 0.5 * (M + M.T)


def _eig(K, M, n_modes, extra):
    n = K.shape[0]
    try:
        Lam, V = eigh(K, M)
        ok = True
    except (LinAlgError, np.linalg.LinAlgError):
        Lam, V, ok = np.full(n, np.nan), np.full((n, n), np.nan), False
    k = min(n_modes, n)
    out = {"Lam": Lam[:k], "vec": V[:, :k], "K": K, "M": M,
           "cholesky_ok": ok}
    out.update(extra)
    return out


def solve_basis(basis, k_hat: float, n_q: int = 200, n_modes: int = 6,
                split_at_xc: bool = True) -> dict:
    """기저 하나로 P3를 풀어 Λ=(βL)⁴(무차원)·고유벡터·(K,M) 반환. k_hat = k_θL/(EI)."""
    xc = P3_CONFIG["xc_over_L"]
    K, M = assemble(basis, float(k_hat), xc, n_q, split_at_xc)
    return _eig(K, M, n_modes,
                {"basis": basis.name, "k_hat": float(k_hat)})


def solve_fem(n_elem: int, k_hat: float, n_modes: int = 6) -> dict:
    """(viii) Hermite FEM + 회전DOF 이중화 + k̂ 결합으로 P3를 푼다.

    n_elem은 x_c/L = 0.2를 절점으로 갖도록 5의 배수여야 한다(아니면 ValueError)."""
    from ..bases.fem import hermite_beam_matrices
    xc = P3_CONFIG["xc_over_L"]
    K, M, j = hermite_beam_matrices(n_elem, xc=xc, split_rotation=True)
    K = 0.5 * (K + K.T) + float(k_hat) * np.outer(j, j)
    return _eig(0.5 * (K + K.T), 0.5 * (M + M.T), n_modes,
                {"basis": f"hermite_fem(n_elem={n_elem})", "k_hat": float(k_hat)})


def reference_betas(k_hat: float, n_modes: int = 6, dps: int = 50) -> np.ndarray:
    """전달행렬 정확기준 βL(오름차순). κ = 1/k̂."""
    from ..reference.transfer_matrix import kappa_from_k_hat, spring_beam_betas
    return spring_beam_betas(P3_CONFIG["xc_over_L"], kappa_from_k_hat(k_hat),
                             n_modes, dps=dps)
