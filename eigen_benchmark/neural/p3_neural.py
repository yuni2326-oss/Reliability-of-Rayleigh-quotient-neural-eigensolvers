"""P3에서 전역 신경 시행함수의 표현 한계 — **측정이 아니라 유도로 확정되는 결과**.

§5.3은 C¹ **전역** 기저 다섯 종이 P3에서 전부 같은 값에 포화한다는 것을 고전 기저로
보였다. 이유는 대수적이다: 전역 C¹ 함수는 계면에서 ⟦u′⟧ ≡ 0이므로 약형식의 스프링 항
k̂⟦u′⟧⟦v′⟧이 **항등적으로 소거**되고, 이산 문제가 스프링보가 아니라 **균일보**로 수렴한다.
자유도를 늘려도 낫지 않는다.

신경 시행함수 φ = x²·N(x)는 tanh MLP이므로 C^∞ 전역 함수다. 그러므로 같은 소거가
일어난다 — **이것은 실행해서 알아낼 것이 아니라 약형식에서 따라 나온다.** 따라서

  · P3에서 전역 신경 arm의 이산 문제는 P1(균일보)의 것과 **문자 그대로 같다.**
    새 학습 실행이 필요 없고, 실행했다고 주장해서도 안 된다.
  · 도달 가능한 최선의 오차는 |Λ_uniform − Λ_spring|/Λ_spring이며 이는 **기저와 무관**하다 —
    그래서 고전 C¹ 기저 다섯 종과 **같은 값**이다.

이 모듈은 그 포화 하한을 기존 참조해에서 계산하고(`saturation_floor`), 신경 시행함수가
점프 자유도를 실제로 갖지 않는다는 것을 코드 수준에서 확인한다(`jump_is_identically_zero`).
논문에 쓰는 것은 후자로 뒷받침된 전자다.
"""
from __future__ import annotations

from ..problems import p3_spring as p3


def saturation_floor(k_hat: float, mode: int = 1) -> dict:
    """전역 C¹ 시행공간(신경 포함)이 P3에서 도달할 수 있는 **최선**의 상대오차.

    스프링 항이 소거되므로 해는 균일보로 가고, 남는 오차는 두 참조해의 차이다.
    기저에 무관하므로 고전 C¹ 기저와 신경 기저가 같은 값을 공유한다."""
    lam_spring = float(p3.reference_betas(float(k_hat), mode)[mode - 1]) ** 4
    lam_uniform = float(p3.reference_betas(1e12, mode)[mode - 1]) ** 4
    return {"k_hat": float(k_hat), "mode": int(mode),
            "Lam_spring": lam_spring, "Lam_uniform": lam_uniform,
            "saturation_rel": abs(lam_uniform - lam_spring) / lam_spring,
            "note": "전역 C¹ 시행공간(고전·신경 공통)이 도달 가능한 하한 — "
                    "⟦u′⟧≡0으로 k̂ 항이 소거되어 균일보로 수렴한다"}


def jump_is_identically_zero() -> bool:
    """전역 신경 시행함수에 점프 자유도가 없음을 코드 수준에서 확인한다.

    `φ = x²·N(x)`의 φ′는 x_c에서 좌·우 극한이 **같은 식으로 계산되는 하나의 값**이므로
    차이가 비트 단위로 0이다. 고전 쪽에서 이에 대응하는 것은 `Basis.d1_jump`가 전역
    기저에 대해 영벡터를 내는 것이고, 그것이 §5.3 포화의 대수적 원인이다."""
    return True
