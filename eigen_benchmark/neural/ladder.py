"""초기화 사다리 I0–I5 — "얼마나 적은 사전정보로 되는가"를 재는 축.

  I0 랜덤 · I1 BC적합 일반다항 · I2 목표 고유값 구간만(형상정보 없음)
  I3 조립 FEM/Ritz 모드(4–6항) — **계산가능한 비-oracle** warm start
  I4 커리큘럼 고정-λ 단계화(arm (c)가 소비) · I5 해석 고유함수 H² 사전학습 = **oracle 상한**

"초기화가 답을 전제한다"는 진술은 **I5에만** 적용된다. 무게중심은 I2–I4다.

I5는 값만 맞추면 안 된다 — 스파이크에서 값 MSE만 최소화하니 4.9e-3으로 느슨했다.
2차도함수는 적합오차를 증폭하므로 곡률까지 맞춰야 축소행렬 K = ∫(φ″)²가 정확해진다.
"""
from __future__ import annotations

import torch
from torch.func import vmap

from ..bases.monomial import MonomialBasis, orthonormalized
from ..problems import p1_beam as p1
from . import core

LEVELS = ("I0", "I1", "I2", "I3", "I4", "I5")


def fit_target(ens, target_val, target_d2, xs, wq, iters: int = 4000,
               lr: float = 2e-3):
    """앙상블 전 시드를 목표 (값, 곡률)에 H² 적합. 두 항은 각자 스케일로 정규화."""
    sv = float(torch.sqrt((wq * target_val ** 2).sum()).clamp_min(1e-30))
    tv = target_val / sv
    tc = None if target_d2 is None else target_d2 / sv
    sc = 1.0 if tc is None else float(torch.sqrt((wq * tc ** 2).sum()).clamp_min(1e-30))

    opt = torch.optim.Adam(list(ens.params.values()), lr=lr)

    def one(p):
        ph, d2 = core.eval_phi_one(p, ens.base, xs)
        loss = (wq * (ph - tv) ** 2).sum()
        if tc is not None:
            loss = loss + (wq * (d2 - tc) ** 2).sum() / sc ** 2
        return loss

    for _ in range(iters):
        vmap(one)(ens.params).sum().backward()
        opt.step()
        opt.zero_grad()


def _analytic_targets(mode: int, xs):
    x = xs.cpu().numpy()
    return (torch.tensor(p1.analytic_mode(x, mode), device=xs.device),
            torch.tensor(p1.analytic_mode_d2(x, mode), device=xs.device))


def _ritz_target(mode: int, n_terms: int, xs):
    """조립 Ritz 모드형 — 계산가능한 warm start(고유함수 공식을 쓰지 않는다)."""
    basis = orthonormalized(MonomialBasis(n_terms))
    r = p1.solve(basis, n_q=400, n_modes=max(mode, 1))
    if not r["cholesky_ok"]:
        raise RuntimeError("조립 Ritz가 수치적으로 실패")
    x = xs.cpu().numpy()
    c = r["vec"][:, mode - 1]
    return (torch.tensor(c @ basis.eval(x), device=xs.device),
            torch.tensor(c @ basis.d2(x), device=xs.device))


def make_ladder(level: str, mode: int = 1, iters: int = 4000, n_terms: int = 5,
                lam_lo: float = 0.0, lam_hi: float = float("inf"),
                barrier_weight: float = 1.0, seed: int = 0):
    """준위별 초기화기(또는 I2의 창 배리어)를 만든다. I0·I4는 None."""
    if level not in LEVELS:
        raise ValueError(f"알 수 없는 사다리 준위: {level} (가능: {LEVELS})")

    if level in ("I0", "I4"):
        return None            # I0=랜덤 그대로, I4=arm (c)가 자체 단계화로 소비

    if level == "I2":
        def barrier(R):
            """목표 고유값 구간만 아는 상태 — 형상정보는 주지 않는다."""
            R = torch.as_tensor(R)
            lo = torch.clamp(R.new_tensor(lam_lo) - R, min=0.0)
            hi = torch.clamp(R - R.new_tensor(lam_hi), min=0.0)
            return barrier_weight * (lo ** 2 + hi ** 2)
        return barrier

    def init(ens, k, xs, wq):
        m = mode if mode else k + 1
        if level == "I1":
            g = torch.Generator(); g.manual_seed(seed + k)
            co = torch.randn(4, generator=g).tolist()
            v = sum(c * xs ** (j + 2) for j, c in enumerate(co))
            fit_target(ens, v, None, xs, wq, iters=max(iters // 4, 200))
        elif level == "I3":
            v, c = _ritz_target(m, n_terms, xs)
            fit_target(ens, v, c, xs, wq, iters=iters)
        elif level == "I5":
            v, c = _analytic_targets(m, xs)
            fit_target(ens, v, c, xs, wq, iters=iters)

    init.__doc__ = f"사다리 {level} 초기화(mode={mode})"
    return init
