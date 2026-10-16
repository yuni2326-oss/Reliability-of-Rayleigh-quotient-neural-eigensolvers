"""(f) Eig-PIELM — 랜덤특징 극한학습 고유치솔버. **역전파가 없다.**

은닉가중 (a_j, b_j)를 무작위로 뽑아 **고정**하면 시행함수
    φ_j(x) = x²·tanh(a_j x + b_j)     (x² 인자로 클램프 BC를 하드 만족)
가 정해지고, 출력층 계수만 남아 문제가 계획 1과 똑같은 일반화 고유치문제
Kc = ΛMc로 환원된다. 학습이 사라지므로 **e_optimization이 0**이고, 남는 것은
e_approx(랜덤특징 공간)와 e_algebraic(랜덤 기저의 조건수)뿐이다 —
gradient-trained arm과의 대조가 바로 그 지점이다.

v3의 [DECIDE](Eig-PIELM을 비교 arm으로 넣을 것인가)를 **포함으로 해소**한다.
numpy만 쓰므로 CPU에서 돌고 GPU 예산을 쓰지 않는다.

**랜덤특징은 극심하게 중복되고, 질량행렬이 수치적으로 부정부호다.** 측정치(특징 80개):
음수 고유값 32개(최소 −3.0e−15), κ₂(M) ≈ 8.9e18. 원시 GEP는 항상 실패하므로
**rank 절단이 필수**다.

**유효 차원은 잘 정의된 정수가 아니다.** 같은 160개 특징이 절단 임계에 따라
tol=1e−16에서 21, 1e−14에서 12, 1e−12에서 10, 1e−10에서 8이 된다. 그래서 단일
`rank_used`만 보고하면 임계 선택이 숨는다 — `rank_spectrum`으로 여러 임계의 rank를
함께 남기고, 조건수는 SVD 기반(부정부호에서도 well-defined)으로 보고한다.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.linalg import LinAlgError, eigh

from ..quadrature import gauss_legendre


def _features(x, a, b):
    """φ, φ′, φ″ — φ = x²·tanh(z), z = a x + b.

    t = tanh(z), t′ = 1 − t², t″ = −2t(1−t²)
    φ′  = 2x t + x² a t′
    φ″  = 2 t + 4x a t′ + x² a² t″
    """
    x = np.asarray(x, float)[None, :]
    aa = a[:, None]
    z = aa * x + b[:, None]
    t = np.tanh(z)
    tp = 1.0 - t ** 2
    tpp = -2.0 * t * tp
    phi = x ** 2 * t
    d1 = 2 * x * t + x ** 2 * aa * tp
    d2 = 2 * t + 4 * x * aa * tp + x ** 2 * (aa ** 2) * tpp
    return phi, d1, d2


def solve_pielm(n_features: int, n_modes: int = 3, n_q: int = 512, seed: int = 0,
                scale: float = 3.0, rank_tol: float = 1e-12) -> dict:
    """랜덤특징 GEP를 rank 절단으로 풀어 최저 n_modes개 Λ = (βL)⁴를 반환.

    반환의 `rank_used`가 이 arm의 실질 자유도다(특징수가 아니라)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    a = rng.uniform(-scale, scale, n_features)
    b = rng.uniform(-scale, scale, n_features)
    x, w = gauss_legendre(n_q, 0.0, 1.0)
    phi, _, d2 = _features(x, a, b)
    K = 0.5 * ((d2 * w) @ d2.T + ((d2 * w) @ d2.T).T)
    M = 0.5 * ((phi * w) @ phi.T + ((phi * w) @ phi.T).T)

    # M의 근-영공간을 버리고 남은 부분공간에서 표준 대칭 고유문제로 환원한다.
    ev, U = np.linalg.eigh(M)
    keep = ev > max(float(ev.max()), 0.0) * rank_tol
    rank = int(keep.sum())
    # 조건수는 SVD 기반으로 — M이 수치적으로 부정부호라 ev.min()을 쓰면 무의미해진다
    kappa_full = float(np.linalg.cond(M))
    rank_spectrum = {f"{t:g}": int((ev > max(float(ev.max()), 0.0) * t).sum())
                     for t in (1e-16, 1e-14, 1e-12, 1e-10)}
    n_neg = int((ev < 0).sum())
    if rank < 1:
        nan = np.full(n_modes, np.nan)
        return {"Lam": nan, "vec": np.full((n_features, n_modes), np.nan), "K": K, "M": M, "cholesky_ok": False,
                "phi_at": lambda xq: np.full(len(np.atleast_1d(xq)), np.nan),
                "rank_used": 0, "n_features": int(n_features), "seed": int(seed),
                "kappa_M": kappa_full, "kappa_M_retained": float("nan"),
                "rank_spectrum": rank_spectrum, "n_negative_eigs": n_neg,
                "rank_tol": float(rank_tol), "seconds": time.perf_counter() - t0,
                "arm": "f_eig_pielm(random features, no backprop)"}

    T = U[:, keep] / np.sqrt(ev[keep])            # Tᵀ M T = I
    A = T.T @ K @ T
    try:
        Lam, Y = eigh(0.5 * (A + A.T))
        ok = True
    except (LinAlgError, np.linalg.LinAlgError):
        Lam = np.full(rank, np.nan)
        Y = np.full((rank, rank), np.nan)
        ok = False
    V = T @ Y
    k = min(n_modes, rank)
    Lam_out = np.full(n_modes, np.nan)
    Lam_out[:k] = Lam[:k]

    def phi_at(xq):
        p, _, _ = _features(np.atleast_1d(np.asarray(xq, float)), a, b)
        return (V[:, 0] @ p) if ok else np.full(p.shape[1], np.nan)

    return {"Lam": Lam_out, "vec": V[:, :k],
            "K": K, "M": M, "cholesky_ok": ok, "phi_at": phi_at,
            "rank_used": rank, "n_features": int(n_features), "seed": int(seed),
            "kappa_M": kappa_full,
            "kappa_M_retained": float(ev[keep].max() / ev[keep].min()),
            "rank_spectrum": rank_spectrum, "n_negative_eigs": n_neg,
            "rank_tol": float(rank_tol),
            "seconds": time.perf_counter() - t0,
            "arm": "f_eig_pielm(random features, no backprop)"}
