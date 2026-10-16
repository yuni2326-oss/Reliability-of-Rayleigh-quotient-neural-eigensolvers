"""P2 — 축퇴가 있는 환형 Kirchhoff 판(클램프 내경 / 자유 외경).

w(r,θ) = W(r)cos(mθ)로 분리하면 절점직경 m마다 1D 반경 문제가 된다.
    κ_rr = W″,  κ_θθ = W′/r − m²W/r²,  κ_rθ = m(W′/r − W/r²)
    e = (κ_rr+κ_θθ)² − 2(1−ν)(κ_rr κ_θθ − κ_rθ²)
    Λ ≡ (kb)⁴ = ∫ e·r dr / ∫ W² r dr   (b = 1 로 무차원화)

치환 ξ = (r−a)/L, L = b−a 로 정의역을 [0,1]에 맞추면 **클램프 내경 W(a)=W′(a)=0이
계획 1 기저의 φ(0)=φ′(0)=0과 정확히 같아져** 기저를 그대로 재사용할 수 있다.
자유 외경은 변분원리의 자연경계조건이므로 강제하지 않는다.

Λ = (kb)⁴는 **무차원**이고 a/b와 ν에만 의존한다. 그래서 b = 1로 잡고 절대 치수·재료·
유효물성(D, ρh)을 일절 쓰지 않는다 — 논문 2의 결론은 전부 무차원량(Λ, e_λ, MAC,
주각, 조건수)이므로 잃는 것이 없고, 시험문제가 특정 기계와 동일시되지 않는다.
샌드위치 유효물성은 물리 모델링이므로 논문 1 소유다(docs/paper2-cs/README.md 경계표).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import LinAlgError, eigh

# **무차원 문제다.** Λ ≡ (kb)⁴는 반경비 a/b와 ν에만 의존하므로 외경을 b = 1로 잡는다.
# 이전에는 논문 1의 임펠러 기하(2a=30.8mm, 2b=73.12mm, face 0.8mm, vane 4.1mm)를 그대로
# 담아 D_eff·ρh로 Hz를 인쇄했다. 그 수치는 이 논문의 어떤 결론도 지탱하지 않으면서
# (Λ·e_lam·MAC·조건수는 전부 무차원) 시험문제를 특정 기계와 동일시하게 만들었다.
# 샌드위치 유효물성은 물리 모델링 선택이므로 논문 1 소유다 — 여기서는 쓰지 않는다.
# (논문 경계 정리, 2026-08-05)
# 반경비 a/b = 0.42 (둥근 값), ν = 0.29. 이 둘이 문제를 완전히 정한다.
P2_GEOMETRY = {"a": 0.42, "b": 1.0, "nu": 0.29}





def assemble(basis, m: int, x_xi, w_xi, geometry=None):
    """ξ 격자에서 P2의 이산 (K, M). K는 굽힘에너지, M은 ∫W²r dr."""
    g = P2_GEOMETRY if geometry is None else geometry
    a, b, nu = g["a"], g["b"], g["nu"]
    L = b - a
    xi = np.asarray(x_xi, float)
    r = a + L * xi
    w = np.asarray(w_xi, float) * L               # dr = L dξ

    W = basis.eval(xi)
    W1 = basis.d1(xi) / L                          # dW/dr
    W2 = basis.d2(xi) / L ** 2                     # d²W/dr²

    krr = W2
    kth = W1 / r - (m ** 2) * W / r ** 2
    krt = m * (W1 / r - W / r ** 2)
    A = krr + kth

    # e의 이중선형형: A_i A_j − (1−ν)(κ_rr,i κ_θθ,j + κ_θθ,i κ_rr,j) + 2(1−ν) κ_rθ,i κ_rθ,j
    rw = r * w
    K = (A * rw) @ A.T
    cross = (krr * rw) @ kth.T
    K -= (1.0 - nu) * (cross + cross.T)
    K += 2.0 * (1.0 - nu) * ((krt * rw) @ krt.T)
    M = (W * rw) @ W.T
    return 0.5 * (K + K.T), 0.5 * (M + M.T)


def solve(basis, m: int, n_q: int = 200, n_modes: int = 4, geometry=None) -> dict:
    """일반화 고유문제 Kc = ΛMc. 계획 1과 같이 Cholesky 실패는 기록만 한다."""
    from ..quadrature import piecewise_gauss

    br = list(getattr(basis, "breaks", (0.0, 1.0)))
    x, w = piecewise_gauss(br, n_q)
    K, M = assemble(basis, m, x, w, geometry)
    n = K.shape[0]
    try:
        Lam, V = eigh(K, M)
        ok = True
    except (LinAlgError, np.linalg.LinAlgError):
        Lam, V, ok = np.full(n, np.nan), np.full((n, n), np.nan), False
    k = min(n_modes, n)
    return {"Lam": Lam[:k], "vec": V[:, :k],
            "K": K, "M": M, "cholesky_ok": ok, "m": int(m), "basis": basis.name}
