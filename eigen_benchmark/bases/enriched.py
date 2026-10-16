"""(vii) 계면강화 / 분할영역 기저 — broken form 안에서 ⟦u⟧=0만 만족.

전역 다항 span에 강화함수 E_j(x) = (x − x_c)^j·H(x − x_c) (j = 1..n_enrich)를 더한다.
- E_1 = 램프: ⟦E_1′⟧ = 1 → **기울기 점프를 담당**하는 유일한 함수
- E_j (j ≥ 2): 우측 영역의 매끄러운 보정, 기울기 연속
전체 공간은 C⁰ 조각다항이므로 분할영역(split-domain) Ritz와 동일한 span이며,
⟦u⟧ = 0이 구성상 만족되어 라그랑주 승수가 필요 없다.
`breaks`에 x_c가 들어가므로 구적이 자동 정렬된다(정렬하지 않으면 강화항의 2차도함수
불연속을 매끄러운 것으로 오적분한다).
"""
from __future__ import annotations

import numpy as np

from .base import Basis
from .monomial import MonomialBasis


class EnrichedBasis(Basis):
    def __init__(self, n_global: int, xc: float, n_enrich: int = 1, parent=None):
        if n_enrich < 1:
            raise ValueError("n_enrich는 1 이상이어야 합니다")
        self.parent = MonomialBasis(n_global) if parent is None else parent
        self.xc = float(xc)
        self.n_enrich = int(n_enrich)
        self.n_dof = self.parent.n_dof + self.n_enrich
        self.name = f"enriched[{self.parent.name}]+{n_enrich}@{xc}"
        pb = list(getattr(self.parent, "breaks", (0.0, 1.0)))
        self.breaks = tuple(np.unique(np.array(pb + [self.xc])))

    def _enrich(self, x, m: int):
        x = np.asarray(x, float)
        s = x - self.xc
        H = (s >= 0.0).astype(float)
        rows = []
        for j in range(1, self.n_enrich + 1):
            if m == 0:
                rows.append(H * s ** j)
            elif m == 1:
                rows.append(H * (j * s ** (j - 1)))
            else:
                rows.append(H * (j * (j - 1) * s ** (j - 2)) if j >= 2
                            else np.zeros_like(s))
        return np.stack(rows)

    def eval(self, x):
        return np.vstack([self.parent.eval(x), self._enrich(x, 0)])

    def d1(self, x):
        return np.vstack([self.parent.d1(x), self._enrich(x, 1)])

    def d2(self, x):
        return np.vstack([self.parent.d2(x), self._enrich(x, 2)])

    def d1_jump(self, xc: float):
        out = np.zeros(self.n_dof)
        if abs(xc - self.xc) > 1e-12:
            return out                       # 강화 위치가 아니면 점프 없음
        out[:self.parent.n_dof] = self.parent.d1_jump(xc)
        out[self.parent.n_dof] = 1.0         # E_1 = 램프만 단위 점프
        return out
