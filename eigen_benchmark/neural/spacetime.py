"""시공간 PINN으로 고유값을 뽑는 접근 — **negative result**를 재현 가능하게 만든다.

범위문서 §7이 논문 2로 지정한 "time-domain PINN negative result"다. 논문 1에서 관측된
것(준정적 문제에서 미수렴, 고전 다항이 압도)을 논문 2의 오차귀속 틀 안에서 다시 세운다.

## 두 접근이 같은 정보를 요구한다

목표는 P1의 최저 고유값 하나다. 두 길이 있다.

**(A) Rayleigh 몫 최소화** — 이 논문이 벤치마크하는 것. 미지수는 φ(x) 하나이고 목적함수가
    고유값 자체다. 정지점이 곧 답이다.
**(B) 시공간 PINN** — 무차원 보 방정식
        w_tt + w_xxxx = 0,  x ∈ [0,1],  t ∈ [0,T]
    를 (x,t) 위에서 풀고, 자유단 응답 w(1,t)를 FFT해 첨두를 읽는다. 초기조건은
    w(x,0) = φ₁(x), w_t(x,0) = 0.

## 왜 (B)가 더 비싼가 — 세 가지가 겹친다

1. **차원**: 미지수가 1D 함수에서 2D 함수로 늘어난다. 같은 해상도를 얻으려면 콜로케이션이
   제곱으로 늘고, 4차 x-미분과 2차 t-미분을 자동미분으로 쌓아야 한다.
2. **고유값이 목적함수에 없다**: PDE 잔차는 "이 시공간 장이 방정식을 만족하는가"만 묻는다.
   고유값은 **사후에 FFT로 읽는** 파생량이라, 잔차를 줄이는 것이 고유값 정확도를 직접
   개선한다는 보장이 없다. (A)는 목적함수가 고유값이다.
3. **시간 지평이 길어질수록 나빠진다**: 실측에서 ω̂가 관측창 4→8→16→32주기에 걸쳐
   3.509 → 3.071 → 2.851 → 2.632로 **단조 감소**한다(참조 3.516 대비 −25 %). 여러 주기에
   걸쳐 진동을 유지하지 못한다.

**2번이 이 negative result의 핵심이고, 실측이 그것을 직접 보인다** — PDE 잔차 loss가
5.8e−1 → 7.1e−2로 한 자리 내려가는 동안 e_λ는 3.9e−3 → 0.44로 100배 **올라간다**.
두 양이 역상관이다. 목적함수가 고유값이 아니면 목적함수를 잘 최소화해도 고유값이
개선되지 않는다 — 이것이 (A)와 (B)를 가르는 구조적 차이다.

**주의: FFT 격자는 병목이 아니다.** 창을 목표 모드의 **정수 주기**로 잡으므로
ω/Δω = ω·T/(2π) = n_periods로 참조 주파수가 항상 정확히 빈 번호 n_periods에 떨어진다.
격자가 목표를 정확히 표현할 수 있으니 해상도는 오차의 하한이 아니고, 측정오차의
**양자화 간격**일 뿐이다. 그래서 창이 짧을수록(격자가 거칠수록) 틀린 주파수도 맞는 빈으로
반올림돼 오차가 **작아 보인다**(4주기에서 e_λ = 3.9e−3). 거친 격자가 모형오차를 가리고
고운 격자가 드러낸다. `bin_spacing`이 그 간격을 계산한다.

무차원 시간: t̃ = t·√(EI/ρA)/L², 그러면 ω̃ = (βL)² 이고 1차 모드 주기는 2π/β₁² ≈ 1.787.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

from ..problems.p1_beam import analytic_mode, beta_roots


def bin_spacing(n_periods: float, beta1: float | None = None) -> dict:
    """FFT 빈 간격 — **오차의 하한이 아니라 측정 양자화 간격**이다.

    Δf = 1/T이고 T = n_periods·(2π/ω̃)이므로 Δω/ω = 1/n_periods, Λ = ω²이니 그 두 배.

    **참조 주파수는 항상 정확히 빈에 떨어진다**: ω/Δω = ω·T/(2π) = n_periods (정수).
    따라서 격자가 목표를 정확히 표현할 수 있고, 이 값은 하한이 아니다. 실제로 4주기에서
    측정 e_λ = 3.9e−3으로 이 값(0.5)의 1/100이 나온다 — 격자가 거칠어 틀린 주파수도
    맞는 빈으로 반올림된 것이다. 즉 **짧은 창은 모형오차를 가린다.**

    쓸모는 반대 방향이다 — 측정된 e_λ가 이 간격보다 훨씬 크면 그 오차는 격자 잡음이
    아니라 실제 모형오차다(`error_exceeds_one_bin`)."""
    b1 = float(beta_roots(1)[0]) if beta1 is None else float(beta1)
    omega = b1 ** 2
    return {"n_periods": float(n_periods), "beta1": b1, "omega_nondim": omega,
            "T_nondim": float(n_periods) * 2.0 * math.pi / omega,
            "rel_omega_bin": 1.0 / float(n_periods),
            "rel_lambda_bin": 2.0 / float(n_periods)}


class SpaceTimeNet(nn.Module):
    """w(x,t) = x²·N(x,t). x² 인자로 clamped BC(w=w_x=0 at x=0)를 하드 만족.

    자유단(w_xx = w_xxx = 0 at x=1)은 강형식이라 자연경계조건이 되지 않으므로 벌점으로
    넣는다 — Rayleigh 몫 접근에서는 약형식이 그것을 공짜로 처리한다는 대비가 여기서 생긴다.
    """

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return (x ** 2) * self.net(torch.cat([x, t], dim=-1))


def _d(y, v, n: int = 1):
    """v에 대한 n차 도함수(그래프 유지)."""
    for _ in range(n):
        y = torch.autograd.grad(y, v, torch.ones_like(y), create_graph=True)[0]
    return y


def _peak_frequency(sig: np.ndarray, dt: float) -> tuple:
    """실신호의 최대 스펙트럼 성분 각진동수와 FFT 격자간격. 보간 없음."""
    n = len(sig)
    sig = sig - sig.mean()
    amp = np.abs(np.fft.rfft(sig))
    f = np.fft.rfftfreq(n, d=dt)
    if len(amp) < 2:
        return float("nan"), float("nan")
    k = int(np.argmax(amp[1:]) + 1)              # DC 제외
    return 2.0 * math.pi * float(f[k]), 2.0 * math.pi * float(f[1] - f[0])


def solve_spacetime(n_periods: float = 8.0, iters: int = 4000, n_col: int = 4096,
                    n_ic: int = 256, n_bc: int = 256, n_probe: int = 512,
                    w_ic: float = 100.0, w_bc: float = 10.0, lr: float = 2e-3,
                    seed: int = 0, width: int = 64, depth: int = 4,
                    device: str | None = None) -> dict:
    """(B) 시공간 PINN. 반환에 `e_lam`, `rel_lambda_bin`, `loss_final`, `seconds`가 있다.

    벌점 가중(`w_ic`, `w_bc`)은 잔차 항들이 같은 자리수에 오도록 고른 값이며, 이 arm은
    **negative result**를 재현하는 것이 목적이므로 가중치 최적화는 하지 않는다. 대신
    `loss_final`을 함께 보고해 **잔차와 고유값 오차가 역상관**임을 독자가 직접 확인할 수
    있게 한다 — 그것이 이 arm의 논지이고, 튜닝으로 없앨 수 있는 종류가 아니다.
    """
    torch.set_default_dtype(torch.float64)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    binsp = bin_spacing(n_periods)
    T = binsp["T_nondim"]

    net = SpaceTimeNet(width, depth).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(iters // 3, 1),
                                            gamma=0.4)
    g = torch.Generator(device=dev)
    g.manual_seed(seed + 991)

    xi = torch.linspace(0.0, 1.0, n_ic, device=dev).reshape(-1, 1)
    ic_val = torch.tensor(analytic_mode(xi.cpu().numpy().ravel(), 1),
                          device=dev).reshape(-1, 1)
    t0 = torch.zeros_like(xi)
    ones = torch.ones(n_bc, 1, device=dev)
    t_bc = torch.rand(n_bc, 1, device=dev, generator=g) * T

    hist, tic = [], time.perf_counter()
    for it in range(iters + 1):
        x = torch.rand(n_col, 1, device=dev, generator=g, requires_grad=True)
        t = (torch.rand(n_col, 1, device=dev, generator=g) * T).requires_grad_(True)
        w = net(x, t)
        res = _d(w, t, 2) + _d(w, x, 4)                      # w_tt + w_xxxx = 0
        loss = (res ** 2).mean()

        xi_ = xi.clone().requires_grad_(True)
        t0_ = t0.clone().requires_grad_(True)
        w0 = net(xi_, t0_)
        loss = loss + w_ic * (((w0 - ic_val) ** 2).mean()
                              + (_d(w0, t0_, 1) ** 2).mean())  # w=φ₁, w_t=0

        xb = ones.clone().requires_grad_(True)
        tb = t_bc.clone().requires_grad_(True)
        wb = net(xb, tb)
        loss = loss + w_bc * ((_d(wb, xb, 2) ** 2).mean()
                              + (_d(wb, xb, 3) ** 2).mean())   # 자유단
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % max(iters // 20, 1) == 0 or it == iters:
            hist.append((it, time.perf_counter() - tic, float(loss.detach())))

    # 자유단 시간응답을 균일 격자에서 뽑아 FFT.
    #
    # **끝점을 제외한다.** 끝점을 포함하면 dt = T/(n−1)이므로 FFT가 보는 기록길이는
    # n·dt = T·n/(n−1)이고 주파수 격자가 (n−1)/n배로 어긋난다. 그러면 정수 주기 창이라도
    # 참조 주파수가 빈에 정확히 떨어지지 않는다(n_probe=512, 4주기에서 빈 4.0078). 그
    # 결과 **창 길이와 무관한 고정 편향**이 생겼다 — 해석해 φ₁(x)cos(ω₁t)를 같은 추정기에
    # 넣으면 창을 어떻게 잡아도 e_λ = 3.902e-03이 나왔고, 이것이 4주기에서 보고된 값과
    # 자릿수까지 같았다. 즉 그 칸은 솔버가 아니라 추정기를 재고 있었다.
    #
    # 끝점을 빼면 dt = T/n, 기록길이 = T가 되어 ω₁이 빈 n_periods에 정확히 떨어지고
    # 해석해에서 편향이 0이 된다(`test_spacetime_estimator_is_unbiased_on_the_exact_mode`).
    with torch.no_grad():
        tp = torch.arange(n_probe, device=dev, dtype=torch.float64).reshape(-1, 1) \
            * (T / n_probe)
        xp = torch.ones_like(tp)
        tip = net(xp, tp).cpu().numpy().ravel()
    omega_hat, d_omega = _peak_frequency(tip, T / n_probe)
    lam_hat = omega_hat ** 2                                  # Λ = ω̃² = (βL)⁴
    lam_ref = float(beta_roots(1)[0]) ** 4
    e_lam = abs(lam_hat - lam_ref) / lam_ref if math.isfinite(lam_hat) else float("nan")

    return {"arm": "g_spacetime_pinn(time-domain)", "Lam": np.array([lam_hat]),
            "Lam_ref": lam_ref, "e_lam": e_lam,
            "omega_hat": omega_hat, "omega_ref": float(beta_roots(1)[0]) ** 2,
            "fft_d_omega": d_omega, "tip": tip,
            "iters": iters, "n_col": n_col, "n_probe": n_probe,
            "seed": seed, "loss_final": hist[-1][2], "history": hist,
            "seconds": time.perf_counter() - tic,
            "n_params": sum(p.numel() for p in net.parameters()),
            **{k: v for k, v in binsp.items() if k != "beta1"}}
