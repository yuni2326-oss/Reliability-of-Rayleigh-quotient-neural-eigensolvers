"""(c) 커리큘럼 arm — 고정-λ 2단 학습 (v3 [5] Kim et al. 대응).

1단: 목표 고유값 λ*를 고정하고 (R − λ*)²/λ*² 를 최소화. **모드형은 주지 않고 고유값만**
     주므로 사다리 I4에 해당한다. λ*는 조립 Ritz(5항)에서 얻어 해석해를 쓰지 않는다.
2단: 그 항을 끄고 deflation 하에서 R을 직접 최소화.

고차모드를 직접 노리는 전략이므로, 순차 deflation이 하위모드로 되돌아가는 실패를
"목표 근방에 먼저 데려다 놓는" 방식으로 회피할 수 있는지가 쟁점이다.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.func import vmap

from ..bases.monomial import MonomialBasis, orthonormalized
from ..problems import p1_beam as p1
from . import core
from .net import EnsembleMLP
from .sequential import _converged, _prev_stack


def coarse_ritz_lambda(mode: int, n_terms: int = 5) -> float:
    """조립 Ritz(n_terms항)의 mode번째 고유값 — 계산가능한 비-oracle 목표."""
    r = p1.solve(orthonormalized(MonomialBasis(n_terms)), n_q=400, n_modes=mode)
    if not r["cholesky_ok"]:
        raise RuntimeError("조립 Ritz가 수치적으로 실패")
    return float(r["Lam"][mode - 1])


def solve_curriculum(n_modes: int, deflation, lam_targets=None, n_seeds: int = 50,
                     iters: int = 4000, stage1_frac: float = 0.5, n_q: int = 256,
                     seed: int = 0, lr: float = 2e-3, snapshot_every: int = 25,
                     device: str | None = None) -> dict:
    torch.set_default_dtype(torch.float64)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    xs, wq = core.gauss_nodes(n_q, dev)
    n1 = int(round(iters * stage1_frac))

    lam, shapes, hists, prevs, stage_secs = [], [], [], [], []
    total = 0.0
    for k in range(n_modes):
        tgt = (float(lam_targets[k]) if lam_targets is not None
               else coarse_ritz_lambda(k + 1))
        ens = EnsembleMLP(n_seeds, seed=seed * 1000 + 500 + k, device=dev)

        def make_loss(stage1: bool):
            def loss_one(p, *pv):
                ph, d2 = core.eval_phi_one(p, ens.base, xs)
                if pv:
                    PH = torch.stack([core.eval_phi_one(q, ens.base, xs)[0].detach()
                                      for q in pv])
                    PH2 = torch.stack([core.eval_phi_one(q, ens.base, xs)[1].detach()
                                       for q in pv])
                    ph, d2, pen = deflation.apply(ph, d2, PH, PH2, wq)
                else:
                    pen = ph.new_zeros(())
                r = (wq * d2 ** 2).sum() / (wq * ph ** 2).sum()
                if stage1:
                    return ((r - tgt) / tgt) ** 2 + pen
                return r + pen
            return loss_one

        hist, stage_outs = [], []
        if n1 > 0:
            o1 = core.train(ens, make_loss(True), n1, lr, snapshot_every,
                            extra_batched=tuple(prevs))
            total += o1["seconds"]; stage_outs.append(o1)
            hist += o1["history"]
        if iters - n1 > 0:
            o2 = core.train(ens, make_loss(False), iters - n1, lr, snapshot_every,
                            extra_batched=tuple(prevs))
            total += o2["seconds"]; stage_outs.append(o2)
            hist += [(n1 + it, o1["seconds"] + s if n1 > 0 else s, r)
                     for it, s, r in o2["history"]]
        stage_secs.append(sum(o["seconds"] for o in stage_outs))
        hists.append(hist)

        with torch.no_grad():
            ph, d2 = core.eval_phi(ens, xs)
            if prevs:
                PH, PH2 = _prev_stack(prevs, ens.base, xs)
                ph, d2, _ = vmap(lambda a, b, c, e:
                                 deflation.apply(a, b, c, e, wq))(ph, d2, PH, PH2)
        r = core.rayleigh(ph, d2, wq).clamp_min(0).cpu().numpy()
        lam.append(r ** 0.25)
        shapes.append(ph.cpu().numpy())
        prevs.append(ens.clone_params())

    return {"lam": np.array(lam), "shapes": np.array(shapes), "history": hists,
            "seconds": total, "xs": xs.cpu().numpy(), "wq": wq.cpu().numpy(),
            "converged": np.array([_converged(h) for h in hists]),
            "deflation": deflation.name, "nodes": "gauss", "n_q": n_q,
            "iters": iters, "n_seeds": n_seeds, "stage1_frac": float(stage1_frac),
            "stage_seconds": stage_secs,
            "arm": f"c_curriculum(stage1={stage1_frac:g})"}
