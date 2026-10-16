"""(viii) Hermite C¹ 보요소 FEM — 정규화 EI = ρA = 1, 정의역 [0,1].

절점 DOF는 (w, θ). 클램프 절점(x=0)의 두 DOF를 제거해 BC를 부과한다.
x_c에서 **회전 DOF를 이중화**(θ⁻, θ⁺)하면 그 자체로는 힌지 기구가 생겨 특이하지만,
P3의 k̂⟦u′⟧⟦v′⟧ 항이 두 회전을 이어 붙여 유일해가 된다 — 균열보 유한요소의 표준 처리.
`jump_vec`은 ⟦u′⟧ = θ⁺ − θ⁻를 나타내는 계수벡터로, 문제 모듈이 외적으로 강성에 더한다.
"""
from __future__ import annotations

import numpy as np


def element_matrices(h: float):
    """길이 h 요소의 3차 Hermite 강성·일치질량 행렬(EI = ρA = 1)."""
    Ke = np.array([
        [12.0, 6.0 * h, -12.0, 6.0 * h],
        [6.0 * h, 4.0 * h * h, -6.0 * h, 2.0 * h * h],
        [-12.0, -6.0 * h, 12.0, -6.0 * h],
        [6.0 * h, 2.0 * h * h, -6.0 * h, 4.0 * h * h],
    ]) / h ** 3
    Me = np.array([
        [156.0, 22.0 * h, 54.0, -13.0 * h],
        [22.0 * h, 4.0 * h * h, 13.0 * h, -3.0 * h * h],
        [54.0, 13.0 * h, 156.0, -22.0 * h],
        [-13.0 * h, -3.0 * h * h, -22.0 * h, 4.0 * h * h],
    ]) * h / 420.0
    return Ke, Me


def hermite_beam_matrices(n_elem: int, xc: float | None = None,
                          split_rotation: bool = False):
    """(K, M, jump_vec) 반환. 클램프 절점 DOF는 제거된 상태.

    split_rotation=True이면 x_c 절점의 회전을 θ⁻/θ⁺로 나눈다(x_c는 내부 절점이어야 함).
    θ⁻는 x_c 왼쪽 요소가, θ⁺는 오른쪽 요소가 쓴다."""
    if n_elem < 1:
        raise ValueError("n_elem은 1 이상이어야 합니다")
    nodes = np.linspace(0.0, 1.0, n_elem + 1)
    ic = None
    if split_rotation:
        if xc is None:
            raise ValueError("split_rotation=True면 xc가 필요합니다")
        hit = np.where(np.abs(nodes - xc) < 1e-12)[0]
        if len(hit) == 0 or hit[0] in (0, n_elem):
            raise ValueError(f"xc={xc}가 내부 절점이 아닙니다(n_elem={n_elem})")
        ic = int(hit[0])

    # DOF 번호: 각 절점에 (w, θ) = (2i, 2i+1). 이중화 시 x_c의 θ⁺를 맨 끝에 추가.
    n_base = 2 * (n_elem + 1)
    theta_plus = n_base if ic is not None else None
    n_all = n_base + (1 if ic is not None else 0)

    K = np.zeros((n_all, n_all))
    M = np.zeros((n_all, n_all))
    for e in range(n_elem):
        h = nodes[e + 1] - nodes[e]
        Ke, Me = element_matrices(h)
        rot_l = 2 * e + 1
        rot_r = 2 * (e + 1) + 1
        if ic is not None and e == ic:
            rot_l = theta_plus          # x_c가 이 요소의 왼쪽 절점 → θ⁺ 사용
        dofs = [2 * e, rot_l, 2 * (e + 1), rot_r]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += Ke[a, b]
                M[dofs[a], dofs[b]] += Me[a, b]

    jump = np.zeros(n_all)
    if ic is not None:
        jump[theta_plus] = 1.0
        jump[2 * ic + 1] = -1.0

    keep = np.array([i for i in range(n_all) if i not in (0, 1)])
    return (K[np.ix_(keep, keep)], M[np.ix_(keep, keep)], jump[keep])
