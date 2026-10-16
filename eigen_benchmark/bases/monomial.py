"""(i) 원시 단항식 Ritz 기저 — 의도적으로 순진한 기준점.

ψ_j(x) = x^{j+2} (j = 0..N−1). 지수 ≥ 2라 클램프 BC를 자동 만족한다.
질량행렬이 Hilbert형(M_ij = 1/(i+j+5))이라 조건수가 지수적으로 커진다 —
파일럿에서 차수 ≈10을 넘자 fp64 Cholesky가 실패한 그 현상의 원인.
"""
from __future__ import annotations

import numpy as np

from .base import Basis


class MonomialBasis(Basis):
    def __init__(self, n_dof: int):
        if n_dof < 1:
            raise ValueError("n_dof는 1 이상이어야 합니다")
        self.n_dof = int(n_dof)
        self.name = f"monomial_raw(N={n_dof})"
        self.exp = np.arange(2, n_dof + 2, dtype=float)      # 2..N+1

    def eval(self, x):
        x = np.asarray(x, dtype=float)
        return x[None, :] ** self.exp[:, None]

    def d1(self, x):
        x = np.asarray(x, dtype=float)
        e = self.exp[:, None]
        return e * x[None, :] ** (e - 1.0)

    def d2(self, x):
        x = np.asarray(x, dtype=float)
        e = self.exp[:, None]
        return e * (e - 1.0) * x[None, :] ** (e - 2.0)


def orthonormalized(parent, n_ref: int = 400, breaks=(0.0, 1.0)):
    """(ii) parent와 **같은 span**을 갖되 질량내적에서 정규직교인 좌표계.

    Householder QR로 얻으므로 Gram의 Cholesky가 깨지는 차수에서도 안정하다.
    B = ψ·√w (n×n_q)라 하면 Bᵀ = QR → 새 기저 = Qᵀ = R⁻ᵀB, Gram = I.
    R은 고정 기준구적(n_ref)에서 한 번만 계산해 좌표변환 C = R⁻ᵀ로 굳힌다."""
    from scipy.linalg import qr, solve_triangular

    from ..quadrature import piecewise_gauss
    from .base import TransformedBasis

    x, w = piecewise_gauss(list(breaks), n_ref)
    B = parent.eval(x) * np.sqrt(w)
    _, R = qr(B.T, mode="economic")
    C = solve_triangular(R, np.eye(parent.n_dof), lower=False).T
    return TransformedBasis(parent, C, name=f"orthonormalized[{parent.name}]")
