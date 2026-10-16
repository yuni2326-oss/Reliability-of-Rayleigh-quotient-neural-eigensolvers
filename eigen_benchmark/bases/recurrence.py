"""(iii)~(v) 직교다항 기저 — 재귀식으로 직접 평가(거듭제곱 계수 변환 금지).

거듭제곱 기저로 변환하면 그 변환의 조건수가 결과를 오염시켜 "Legendre가 나쁘다"는
잘못된 결론이 나온다. 따라서 legval/chebval(Clenshaw)로 평가하고, 도함수도
legder/chebder 계수공간에서 취한다.

BC 재조합: ψ_k(x) = f_k(x) − f_k(0) − f_k′(0)·x.
클램프 조건 두 개를 만족시키되 **직교성을 깨뜨린다** — 명세가 경고한 그 효과를
측정하기 위해 일부러 이 표준 방식을 쓴다.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as C
from numpy.polynomial import legendre as L

from .base import Basis


class _RecombinedOrthoBasis(Basis):
    """t = 2x−1 위의 직교다항 f_k를 클램프 BC에 맞게 재조합한 기저."""

    _val = None      # (t, coef) -> value
    _der = None      # (coef, m) -> derivative coef
    _family = "?"

    def __init__(self, n_dof: int, degree_offset: int = 2):
        if n_dof < 1:
            raise ValueError("n_dof는 1 이상이어야 합니다")
        self.n_dof = int(n_dof)
        self.name = f"{self._family}_bc_recombined(N={n_dof})"
        self._deg = [degree_offset + k for k in range(self.n_dof)]
        self._c0, self._s0 = [], []
        for i in range(self.n_dof):
            c = self._coef(i)
            self._c0.append(float(self._val(np.array([-1.0]), c)[0]))
            self._s0.append(2.0 * float(self._val(np.array([-1.0]),
                                                 self._der(c, 1))[0]))

    def _coef(self, i):
        d = self._deg[i]
        c = np.zeros(d + 1)
        c[d] = 1.0
        return c

    def eval(self, x):
        x = np.asarray(x, float); t = 2.0 * x - 1.0
        out = np.empty((self.n_dof, x.size))
        for i in range(self.n_dof):
            out[i] = self._val(t, self._coef(i)) - self._c0[i] - self._s0[i] * x
        return out

    def d1(self, x):
        x = np.asarray(x, float); t = 2.0 * x - 1.0
        out = np.empty((self.n_dof, x.size))
        for i in range(self.n_dof):
            out[i] = 2.0 * self._val(t, self._der(self._coef(i), 1)) - self._s0[i]
        return out

    def d2(self, x):
        x = np.asarray(x, float); t = 2.0 * x - 1.0
        out = np.empty((self.n_dof, x.size))
        for i in range(self.n_dof):
            out[i] = 4.0 * self._val(t, self._der(self._coef(i), 2))
        return out


class ShiftedLegendreBasis(_RecombinedOrthoBasis):
    _val = staticmethod(lambda t, c: L.legval(t, c))
    _der = staticmethod(lambda c, m: L.legder(c, m))
    _family = "shifted_legendre"


class ChebyshevBasis(_RecombinedOrthoBasis):
    _val = staticmethod(lambda t, c: C.chebval(t, c))
    _der = staticmethod(lambda c, m: C.chebder(c, m))
    _family = "chebyshev"


_H = {                                   # 3차 Hermite 기저(거듭제곱 계수, 저차라 안전)
    "h00": np.array([1.0, 0.0, -3.0, 2.0]),      # 값@0
    "h10": np.array([0.0, 1.0, -2.0, 1.0]),      # 기울기@0
    "h01": np.array([0.0, 0.0, 3.0, -2.0]),      # 값@1
    "h11": np.array([0.0, 0.0, -1.0, 1.0]),      # 기울기@1
}


def _poly(c, x, m=0):
    p = np.polynomial.Polynomial(c)
    for _ in range(m):
        p = p.deriv()
    return p(np.asarray(x, float))


class IntegratedLegendreBasis(Basis):
    """(iv) 적분 Legendre / Babuška–Shen 계열.

    구성: 자유단(x=1) Hermite 2개(값·기울기) + 이중적분 Legendre 버블.
    버블 B_j = (1/4)∫∫L_{j+2}(t) − (양끝 값·기울기를 지우는 3차 Hermite 보정)
    → B_j(0)=B_j′(0)=B_j(1)=B_j′(1)=0이고 B_j″ = L_{j+2} + (선형) 이라 강성행렬이
    버블 블록에서 거의 대각이 된다(hp 표준의 이점).

    **차수 오프셋 +2가 필수**: L_0·L_1의 이중적분은 2·3차라서 3차 Hermite 보정을 빼면
    항등적으로 0이 된다(양끝 값·기울기 4조건이 3차 공간을 {0}으로 만든다). 그대로 쓰면
    기저에 영함수가 들어가 K·M이 특이해지고 λ≈0의 허위 고유값이 생긴다. 4차 문제의
    진짜 버블은 L_2(이중적분 4차)부터다."""

    def __init__(self, n_dof: int):
        if n_dof < 2:
            raise ValueError("n_dof는 2 이상이어야 합니다(자유단 Hermite 2개 포함)")
        self.n_dof = int(n_dof)
        self.name = f"integrated_legendre(N={n_dof})"
        self._n_bub = self.n_dof - 2
        self._bub = []
        for j in range(self._n_bub):
            deg = j + 2                       # L_{j+2} — 오프셋 +2 필수(위 설명)
            e = np.zeros(deg + 1)
            e[deg] = 1.0
            c2 = L.legint(L.legint(e, 1), 1)          # t에 대한 이중적분
            g = lambda x, c2=c2: 0.25 * L.legval(2 * np.asarray(x, float) - 1, c2)
            g1 = lambda x, c2=c2: 0.5 * L.legval(2 * np.asarray(x, float) - 1,
                                                 L.legder(c2, 1))
            g2 = lambda x, c2=c2: L.legval(2 * np.asarray(x, float) - 1,
                                           L.legder(c2, 2))
            g0, gL = g(np.array([0.0]))[0], g(np.array([1.0]))[0]
            s0, sL = g1(np.array([0.0]))[0], g1(np.array([1.0]))[0]
            corr = (g0 * _H["h00"] + s0 * _H["h10"]
                    + gL * _H["h01"] + sL * _H["h11"])
            self._bub.append((g, g1, g2, corr))

    def _stack(self, x, m):
        x = np.asarray(x, float)
        rows = [_poly(_H["h01"], x, m), _poly(_H["h11"], x, m)]   # 자유단 Hermite
        for g, g1, g2, corr in self._bub:
            base = (g, g1, g2)[m](x)
            rows.append(base - _poly(corr, x, m))
        return np.stack(rows)

    def eval(self, x):
        return self._stack(x, 0)

    def d1(self, x):
        return self._stack(x, 1)

    def d2(self, x):
        return self._stack(x, 2)
