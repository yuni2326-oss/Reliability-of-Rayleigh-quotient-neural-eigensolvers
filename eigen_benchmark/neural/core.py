"""vmap 시드앙상블 Rayleigh 몫 엔진.

**기본 구적은 고정 Gauss**다. 파일럿의 몬테카를로 콜로케이션은 구적오차를
e_algebraic로 흘려보내 모드1 cold 정확도를 1.4 %로 만들었고(고정 Gauss에서는 1.45e-6),
비용도 8.2배 비싸다(스파이크). MC는 구적 대조군으로만 남긴다.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.func import functional_call, grad, vmap


def gauss_nodes(n_q: int = 256, device: str = "cuda", dtype=torch.float64):
    """[0,1]의 고정 Gauss–Legendre 노드·가중치.

    노드는 **항상 fp64로 만들고** 마지막에 요청 dtype으로 내린다 — leggauss를 낮은
    정밀도로 흉내내면 구적 자체가 정밀도 실험의 교란요인이 된다."""
    x, w = np.polynomial.legendre.leggauss(n_q)
    return (torch.tensor(0.5 * x + 0.5, device=device, dtype=torch.float64).to(dtype),
            torch.tensor(0.5 * w, device=device, dtype=torch.float64).to(dtype))


def mc_nodes(n: int, generator, device: str = "cuda", dtype=torch.float64):
    """구적 대조군 — 매 반복 새로 뽑는 몬테카를로 콜로케이션."""
    x = torch.rand(n, device=device, generator=generator, dtype=dtype)
    return x, torch.full((n,), 1.0 / n, device=device, dtype=dtype)


def _phi_scalar(p, base, x):
    return functional_call(base, (p, {}), (x.reshape(1, 1),)).reshape(())


_D2 = grad(grad(_phi_scalar, argnums=2), argnums=2)


def eval_phi_one(p, base, xs):
    """한 시드의 (φ, φ″) — 각 (n_q,). vmap 안에서 호출한다."""
    return (vmap(lambda t: _phi_scalar(p, base, t))(xs),
            vmap(lambda t: _D2(p, base, t))(xs))


def eval_phi(ens, xs):
    """앙상블 전체의 (φ, φ″) — 각 (n_seeds, n_q). 학습 밖 평가용."""
    return vmap(lambda p: eval_phi_one(p, ens.base, xs))(ens.params)


def rayleigh(phi, d2, wq):
    """R = Σw φ″² / Σw φ². 마지막 축이 구적점."""
    return (wq * d2 ** 2).sum(-1) / (wq * phi ** 2).sum(-1)


def train(ens, loss_fn, iters: int, lr: float = 2e-3, snapshot_every: int = 25,
          step_gamma: float = 0.4, extra_batched=(), on_iter=None):
    """vmap 배치 학습. history = [(iter, seconds, R(n_seeds,))]로 t(ε)를 산출한다.

    `extra_batched`는 **시드 축을 함께 배치할** 추가 인자(예: 수렴한 하위모드의
    파라미터 dict). 클로저로 잡으면 vmap이 배치하지 않아 (n_seeds, out, in) 가중치가
    그대로 Linear에 들어가 3D 오류가 난다 — 반드시 이 경로로 넘긴다.

    `on_iter(it)`은 매 반복 **vmap 밖에서** 호출된다. MC 구적처럼 난수를 새로 뽑아야
    하는 경우에 쓴다 — vmap 안에서 `torch.rand`를 부르면 randomness error mode로 막힌다.
    (배치되지 않는 평범한 텐서를 클로저로 읽는 것은 문제없다.)"""
    opt = torch.optim.Adam(list(ens.params.values()), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(iters // 3, 1),
                                            gamma=step_gamma)
    history, t0 = [], time.perf_counter()
    for it in range(iters + 1):
        if on_iter is not None:
            on_iter(it)
        r = vmap(loss_fn)(ens.params, *extra_batched)
        if it % snapshot_every == 0 or it == iters:
            # numpy에 bfloat16이 없으므로 기록 시점에 fp64로 올린다(값 손실 없음).
            history.append((it, time.perf_counter() - t0,
                            r.detach().double().cpu().numpy().copy()))
        if it == iters:
            break
        r.sum().backward()
        opt.step()
        sched.step()
        opt.zero_grad()
    return {"history": history, "params": ens.params,
            "seconds": time.perf_counter() - t0}
