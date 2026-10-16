"""기저 프로토콜 — 문제를 모르는 시행함수 집합.

계약: 기저는 정의역 [0,1]에서 값·1차·2차도함수를 (n_dof, n_points) 배열로 준다.
클램프 경계 φ(0)=φ′(0)=0은 **기저가 스스로** 만족시킨다(약형식에 벌점 없음).
`d1_jump(xc)`는 x_c에서의 1차도함수 우측한계 − 좌측한계로, C¹ 기저는 0벡터다
(→ P3의 k_θ⟦u′⟧⟦v′⟧ 항이 소거되어 균일보 극한으로 수렴하는 e_approx 하한이 생긴다).
"""
from __future__ import annotations

import numpy as np


class Basis:
    """시행함수 집합의 최소 인터페이스.

    `breaks`는 이 기저가 조각다항인 경계점 목록이다. 구적이 여기에 정렬되지 않으면
    매끄럽지 않은 피적분함수를 매끄러운 것으로 오적분해 **강성을 과소평가**하고,
    Ritz 값이 정확해보다 낮게 나오는 비적합 현상이 생긴다(§3.6이 잡으려는 오류).
    전역 매끄러운 기저는 (0,1)이다."""

    name: str = "basis"
    n_dof: int = 0
    breaks: tuple = (0.0, 1.0)

    def eval(self, x):
        raise NotImplementedError

    def d1(self, x):
        raise NotImplementedError

    def d2(self, x):
        raise NotImplementedError

    def d1_jump(self, xc: float):
        """⟦ψ′⟧(x_c) = ψ′(x_c⁺) − ψ′(x_c⁻). C¹ 기저는 0."""
        return np.zeros(self.n_dof)


class TransformedBasis(Basis):
    """좌표변환 래퍼: new_i = Σ_j C_ij old_j. **함수공간은 그대로, 좌표만 바꾼다.**

    (i) 원시 단항식과 (ii) 정규직교화가 같은 span을 갖되 조건수만 다르다는 것을
    구조적으로 보장하기 위한 장치 — 두 셀의 차이는 전부 e_algebraic이다."""

    def __init__(self, parent: Basis, C, name: str):
        self.parent = parent
        self.C = np.asarray(C, dtype=float)
        if self.C.shape[1] != parent.n_dof:
            raise ValueError("C의 열수가 parent.n_dof와 달라요")
        self.name = name
        self.n_dof = self.C.shape[0]
        self.breaks = getattr(parent, "breaks", (0.0, 1.0))

    def eval(self, x):
        return self.C @ self.parent.eval(x)

    def d1(self, x):
        return self.C @ self.parent.d1(x)

    def d2(self, x):
        return self.C @ self.parent.d2(x)

    def d1_jump(self, xc: float):
        return self.C @ self.parent.d1_jump(xc)
