"""(vi) 연속성 제어 B-spline 기저.

차수 p 노트의 중복도 m에서 연속성은 C^{p−m}이다. 따라서 x_c에 **중복도 p**를 주면
C⁰가 되어 기울기 점프를 표현할 수 있다(p=3 → 중복도 3, 명세 §3.3의 그 설정).
매끄러운 유한폭 케이스에는 중복도 없이 단순 세분만 쓴다.

한쪽 극한은 PPoly 조각다항 계수로 **정확히** 취한다(δ 오프셋 근사 아님).
클램프 BC: x=0에서 값·기울기가 0이 아닌 기저함수는 앞의 2개뿐이라 그것만 제거한다.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PPoly

from .base import Basis


class BSplineC0Basis(Basis):
    def __init__(self, n_elem: int, degree: int = 3, xc: float | None = None,
                 c0: bool = True):
        p = int(degree)
        if p < 2:
            raise ValueError("degree는 2 이상이어야 합니다(H² 문제)")
        if n_elem < 2:
            raise ValueError("n_elem은 2 이상이어야 합니다")
        interior = list(np.linspace(0.0, 1.0, n_elem + 1)[1:-1])
        if xc is not None:
            interior = [v for v in interior if abs(v - xc) > 1e-12]
            interior += [float(xc)] * (p if c0 else 1)
        interior.sort()
        self.knots = np.array([0.0] * (p + 1) + interior + [1.0] * (p + 1))
        self.degree = p
        n_all = len(self.knots) - p - 1
        self._keep = np.arange(2, n_all)                 # 앞 2개 제거 = 클램프 BC
        self.n_dof = len(self._keep)
        self.name = (f"bspline_p{p}_nel{n_elem}"
                     + (f"_C0@{xc}" if (xc is not None and c0) else ""))
        self.breaks = tuple(np.unique(self.knots))       # 구적 정렬용 조각경계

        self._pp = []
        for i in self._keep:
            c = np.zeros(n_all)
            c[i] = 1.0
            self._pp.append(PPoly.from_spline((self.knots, c, p)))
        self._pp1 = [q.derivative(1) for q in self._pp]
        self._pp2 = [q.derivative(2) for q in self._pp]

    @staticmethod
    def _ev(polys, x):
        x = np.asarray(x, float)
        return np.stack([np.nan_to_num(q(x, extrapolate=False)) for q in polys])

    def eval(self, x):
        return self._ev(self._pp, x)

    def d1(self, x):
        return self._ev(self._pp1, x)

    def d2(self, x):
        return self._ev(self._pp2, x)

    def d1_jump(self, xc: float):
        """조각다항 구간에서 좌·우 한쪽 극한을 정확히 취해 ⟦ψ′⟧을 계산."""
        out = np.zeros(self.n_dof)
        for i, q in enumerate(self._pp1):
            br = q.x
            hit = np.where(np.abs(br - xc) < 1e-12)[0]
            if len(hit) == 0:
                continue                       # x_c가 노트가 아니면 점프 없음
            k = int(hit[0])
            if k <= 0 or k >= len(br) - 1:
                continue                       # 정의역 끝
            left = float(np.polyval(q.c[:, k - 1], xc - br[k - 1]))
            right = float(np.polyval(q.c[:, k], 0.0))
            out[i] = right - left
        return out
