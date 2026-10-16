"""(d) 신경기저 Galerkin · (e) 동시 다출력 부분공간 — 순차 오차누적이 없는 두 접근.

공통 목적함수는 **부분공간 trace**다. K차원 시행공간 Φ에 대해
    tr((ΦᵀWΦ)⁻¹ ΦᵀW″Φ)
는 그 공간의 K개 Ritz값의 합이고, 최소화하면 span이 최저 K개 고유공간으로 끌려간다.
개별 고유쌍은 마지막에 축소 GEP를 풀어 뽑는다 — 순차 deflation처럼 앞 모드의 오차가
뒤로 전파되지 않는다.

두 arm의 차이는 **표현 구조 하나**뿐이다.
    (d) 분리된 M개 망을 기저로 학습        (e) 공유 trunk 다출력 망(K출력)
그래야 "부분공간 접근이 좋은가"와 "가중치 공유가 좋은가"가 분리된다.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, stack_module_state, vmap

from . import core
from .net import EnsembleMLP


def subspace_trace(Phi, Phi2, wq):
    """tr((ΦᵀWΦ)⁻¹ΦᵀW″Φ) — K개 Ritz값의 합. Phi/Phi2는 (K, n_q)."""
    A = torch.einsum("ki,i,li->kl", Phi2, wq, Phi2)
    B = torch.einsum("ki,i,li->kl", Phi, wq, Phi)
    return torch.linalg.solve(0.5 * (B + B.T), 0.5 * (A + A.T)).diagonal().sum()


def subspace_logtrace(Phi, Phi2, wq):
    """Σ log λ_k = logdet(A) − logdet(B) — **상대 가중** 목적함수.

    절대 trace Σλ_k는 ∂/∂λ_k = 1이라 모든 모드가 **절대량으로** 등가다. P1에서 λ₁ = 12.4,
    λ₆ = 89140이므로 모드 1을 7 % 희생해 얻는 0.87을 모드 6의 0.1 % 개선(89)이 100배로
    갚는다 — 목적함수가 최저 모드를 버리라고 적극적으로 보상한다.

    로그를 씌우면 ∂/∂λ_k = 1/λ_k가 되어 **상대** 변화가 등가가 되고 그 비대칭이 사라진다.
    최소점은 같다: Cauchy 인터레이싱으로 모든 k에서 λ_k(Φ) ≥ λ_k이므로 Σλ와 Σlog λ가
    동시에 같은 불변부분공간에서 하한을 얻는다. 즉 **같은 답을 다른 가중으로 찾는다.**

    logdet는 고유분해 없이 미분 가능해 학습 루프에서 trace만큼 싸다."""
    A = torch.einsum("ki,i,li->kl", Phi2, wq, Phi2)
    B = torch.einsum("ki,i,li->kl", Phi, wq, Phi)
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    sa, la = torch.linalg.slogdet(A)
    sb, lb = torch.linalg.slogdet(B)
    # 부호가 음수면(수치적 특이) 발산하지 않게 큰 벌점으로 되돌린다
    bad = (sa <= 0) | (sb <= 0)
    return torch.where(bad, la.new_tensor(1e6), la - lb)


OBJECTIVES = {"trace": subspace_trace, "logtrace": subspace_logtrace}


def gram_offdiagonal_penalty(Phi, wq):
    """상관 정규화한 그람행렬의 비대각 제곱합 — 기저함수 간 중복에 벌점.

    trace 목적함수는 **span만** 보므로 개별 기저함수가 서로 겹쳐도 벌점이 없다.
    게다가 거의 겹친 두 함수의 차 (φᵢ−φⱼ)는 질량도 강성도 동시에 작아 Ritz 기여가
    0/0 형태로 유한하게 남아, 떼어놓을 기울기 압력이 거의 생기지 않는다.
    이 항이 그 없던 압력을 공급한다(귀속 확인용 ablation — 새 방법 제안이 아니다)."""
    B = torch.einsum("ki,i,li->kl", Phi, wq, Phi)
    d = torch.sqrt(torch.clamp(torch.diagonal(B), min=1e-300))
    C = B / torch.outer(d, d)
    off = C - torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
    return (off ** 2).sum()


def galerkin_solve(Phi, Phi2, wq, rank_tol: float = 1e-12):
    """축소 GEP A c = Λ B c. **rank 절단** 후 푼다. 반환 (Λ 오름차순, 계수, rank).

    학습된 신경기저는 수치적으로 rank 결손이 되기 쉽다 — trace 목적함수는 span만
    보므로 개별 기저함수가 서로 겹쳐도 벌점이 없다. M=9·50시드에서 실제로 B가
    양정부호를 잃고 Cholesky가 실패했다. 계획 1의 원시 단항식(Hilbert형)과 PIELM
    랜덤특징(80개→rank 21)에 이은 **세 번째 사례**이며, 처리도 같다: B의 대칭
    고유분해에서 상대 tol 이상인 방향만 남긴다. 유효 rank는 보고값이다."""
    A = torch.einsum("ki,i,li->kl", Phi2, wq, Phi2)
    B = torch.einsum("ki,i,li->kl", Phi, wq, Phi)
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    ev, U = torch.linalg.eigh(B)
    keep = ev > torch.clamp(ev.max(), min=0.0) * rank_tol
    r = int(keep.sum())
    if r < 1:
        empty = torch.zeros(0, dtype=A.dtype, device=A.device)
        return empty, torch.zeros((B.shape[0], 0), dtype=A.dtype, device=A.device), 0
    T = U[:, keep] / torch.sqrt(ev[keep])          # TᵀBT = I
    C = T.T @ A @ T
    w, Y = torch.linalg.eigh(0.5 * (C + C.T))
    return w, T @ Y, r


class _MultiMLP(nn.Module):
    def __init__(self, width=64, depth=4, n_out=4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return (x ** 2) * self.net(x)


class MultiOutputMLP:
    """(e)용 공유 trunk 다출력 앙상블. φ(x) = x²·N(x) ∈ R^K."""

    def __init__(self, n_seeds: int, n_out: int, width: int = 64, depth: int = 4,
                 seed: int = 0, device: str | None = None):
        torch.set_default_dtype(torch.float64)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        models = [_MultiMLP(width, depth, n_out).to(dev) for _ in range(n_seeds)]
        self.params, _ = stack_module_state(models)
        self.base = _MultiMLP(width, depth, n_out).to(dev).to("meta")
        self.n_seeds, self.n_out, self.device = n_seeds, n_out, dev


def _multi_scalar(p, base, x, j):
    return functional_call(base, (p, {}), (x.reshape(1, 1),)).reshape(-1)[j]


_MD2 = grad(grad(_multi_scalar, argnums=2), argnums=2)


def eval_multi_one(p, base, xs, n_out):
    """한 시드의 (φ, φ″) — 각 (n_out, n_q)."""
    ph, d2 = [], []
    for j in range(n_out):
        ph.append(vmap(lambda t, j=j: _multi_scalar(p, base, t, j))(xs))
        d2.append(vmap(lambda t, j=j: _MD2(p, base, t, j))(xs))
    return torch.stack(ph), torch.stack(d2)


def eval_multi(net, xs):
    return vmap(lambda p: eval_multi_one(p, net.base, xs, net.n_out))(net.params)


def _extract(PH, PH2, wq, n_seeds, n_modes):
    """시드별 축소 GEP → (lam4(n_modes,n_seeds), shapes(n_modes,n_seeds,n_q), ranks).

    rank가 n_modes보다 작으면 부족한 모드는 **NaN으로 남긴다** — 0이나 마지막 값으로
    채우면 표가 거짓말을 한다. NaN은 사전등록 규칙에서 non_converged로 분류된다."""
    n_q = PH.shape[-1]
    lam4 = np.full((n_seeds, n_modes), np.nan)
    shapes = np.full((n_seeds, n_modes, n_q), np.nan)
    curv = np.full((n_seeds, n_modes, n_q), np.nan)      # φ″ — H² 오차 측정용
    ranks = []
    for s in range(n_seeds):
        w, V, r = galerkin_solve(PH[s], PH2[s], wq)
        ranks.append(r)
        k = min(n_modes, r)
        if k > 0:
            lam4[s, :k] = np.clip(w[:k].cpu().numpy(), 0, None)
            shapes[s, :k] = (V[:, :k].T @ PH[s]).cpu().numpy()
            curv[s, :k] = (V[:, :k].T @ PH2[s]).cpu().numpy()
    return (lam4.T, np.transpose(shapes, (1, 0, 2)),
            np.array(ranks), np.transpose(curv, (1, 0, 2)))


def _mode_snapshot(PH, PH2, wq, n_seeds, n_modes):
    """스냅샷마다 축소 GEP를 풀어 **모드별 Ritz값**을 낸다 (n_modes, n_seeds).

    수렴판정을 trace로 하면 안 된다 — trace는 K개 고유값의 합이라 최대 모드가 지배하고
    (λ₁/λ₆ ≈ 1.4e-4) 최소 모드의 수렴은 보이지도 않는다. 순차 arm과 같은 기준으로
    비교하려면 모드별 Ritz값의 상대변화를 봐야 한다. 9×9 eigh는 비용이 무시가능하다."""
    lam4, _, _, _ = _extract(PH, PH2, wq, n_seeds, n_modes)
    return lam4


def _pack(lam4, shapes, hist, xs, wq, secs, arm, n_seeds, iters, n_q):
    """hist는 [(it, sec, (n_modes, n_seeds))] — 모드별 히스토리로 쪼개 수렴판정한다."""
    from .sequential import _converged
    n_modes = np.asarray(lam4).shape[0]
    per_mode = [[(it, sc, np.asarray(v)[m]) for it, sc, v in hist]
                for m in range(n_modes)]
    return {"lam": np.array(lam4) ** 0.25, "shapes": np.array(shapes),
            "history": per_mode, "seconds": secs, "xs": xs.cpu().numpy(),
            "wq": wq.cpu().numpy(),
            "converged": np.array([_converged(h) for h in per_mode]),
            "deflation": "subspace", "nodes": "gauss", "n_q": n_q,
            "iters": iters, "n_seeds": n_seeds, "arm": arm}


def solve_simultaneous(n_modes: int, n_seeds: int = 50, iters: int = 4000,
                       n_q: int = 256, seed: int = 0, lr: float = 2e-3,
                       snapshot_every: int = 25, objective: str = "trace",
                       device: str | None = None) -> dict:
    """(e) 공유 trunk 다출력 망으로 부분공간 목적함수 최소화 후 대각화.

    `objective="logtrace"`는 절대 가중을 상대 가중으로 바꾼다 — 최저 모드가 목적함수에서
    사실상 무시되는(λ₁이 trace의 0.0084 %) 문제를 겨냥한 대조군이다."""
    if objective not in OBJECTIVES:
        raise ValueError(f"알 수 없는 목적함수: {objective} (가능: {sorted(OBJECTIVES)})")
    obj = OBJECTIVES[objective]
    torch.set_default_dtype(torch.float64)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    xs, wq = core.gauss_nodes(n_q, dev)
    net = MultiOutputMLP(n_seeds, n_modes, seed=seed + 700, device=dev)
    opt = torch.optim.Adam(list(net.params.values()), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(iters // 3, 1),
                                            gamma=0.4)

    def loss_one(p):
        ph, d2 = eval_multi_one(p, net.base, xs, n_modes)
        return obj(ph, d2, wq)

    hist, t0 = [], time.perf_counter()
    for it in range(iters + 1):
        r = vmap(loss_one)(net.params)
        if it % snapshot_every == 0 or it == iters:
            with torch.no_grad():
                PHs, PH2s = eval_multi(net, xs)
                snap = _mode_snapshot(PHs, PH2s, wq, n_seeds, n_modes)
            hist.append((it, time.perf_counter() - t0, snap))
        if it == iters:
            break
        r.sum().backward()
        opt.step()
        sched.step()
        opt.zero_grad()

    with torch.no_grad():
        PH, PH2 = eval_multi(net, xs)
        lam4, shapes, ranks, curv = _extract(PH, PH2, wq, n_seeds, n_modes)
    out = _pack(lam4, shapes, hist, xs, wq,
                time.perf_counter() - t0,
                f"e_simultaneous_subspace(obj={objective})", n_seeds, iters, n_q)
    # 진단용 **원시 기저**(Galerkin 추출 전) — 추출된 모드는 구성상 B-직교라
    # 공선성을 잴 수 없다.
    out["basis_shapes"] = PH.cpu().numpy()
    out["curvatures"] = curv
    out["rank_median"] = float(np.median(ranks))
    out["rank_min"] = int(np.min(ranks))
    return out


def solve_neural_basis_galerkin(n_modes: int, n_basis: int | None = None,
                                n_seeds: int = 50, iters: int = 4000,
                                n_q: int = 256, seed: int = 0, lr: float = 2e-3,
                                snapshot_every: int = 25, w_orth: float = 0.0,
                                device: str | None = None) -> dict:
    """(d) 분리된 M개 망을 기저로 trace 학습 → 축소 GEP.

    시드 축과 기저 축을 (n_seeds × M)으로 평탄화해 한 번에 vmap하고, 평가 시 되돌린다.

    `w_orth > 0`이면 그람 비대각 벌점을 더한다 — **귀속 확인용 ablation**이다.
    기본 격자에서 유효 rank가 4로 붕괴했고 예산 5배로는 5까지만 올랐는데, rank가
    충족된 모드조차 정확도가 0이었다. 원인이 "직교성 압력 부재"라면 이 항으로 회복되고,
    아니면 원인은 다른 곳(표현력 자체)이다."""
    torch.set_default_dtype(torch.float64)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    M = n_basis or (n_modes + 3)
    xs, wq = core.gauss_nodes(n_q, dev)
    ens = EnsembleMLP(n_seeds * M, seed=seed + 900, device=dev)
    opt = torch.optim.Adam(list(ens.params.values()), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(iters // 3, 1),
                                            gamma=0.4)

    def flat_eval(p):
        return core.eval_phi_one(p, ens.base, xs)

    hist, t0 = [], time.perf_counter()
    for it in range(iters + 1):
        ph, d2 = vmap(flat_eval)(ens.params)           # (S*M, n_q)
        ph = ph.reshape(n_seeds, M, -1)
        d2 = d2.reshape(n_seeds, M, -1)
        if w_orth > 0.0:
            r = vmap(lambda a, b: subspace_trace(a, b, wq)
                     + w_orth * gram_offdiagonal_penalty(a, wq))(ph, d2)
        else:
            r = vmap(lambda a, b: subspace_trace(a, b, wq))(ph, d2)
        if it % snapshot_every == 0 or it == iters:
            with torch.no_grad():
                snap = _mode_snapshot(ph.detach(), d2.detach(), wq, n_seeds, n_modes)
            hist.append((it, time.perf_counter() - t0, snap))
        if it == iters:
            break
        r.sum().backward()
        opt.step()
        sched.step()
        opt.zero_grad()

    with torch.no_grad():
        ph, d2 = vmap(flat_eval)(ens.params)
        ph = ph.reshape(n_seeds, M, -1)
        d2 = d2.reshape(n_seeds, M, -1)
        lam4, shapes, ranks, curv = _extract(ph, d2, wq, n_seeds, n_modes)
        raw_basis = ph.cpu().numpy()      # 진단용 원시 기저(추출 전)
    out = _pack(lam4, shapes, hist, xs, wq,
                time.perf_counter() - t0, f"d_neural_basis_galerkin(M={M})",
                n_seeds, iters, n_q)
    out["n_basis"] = M
    out["basis_shapes"] = raw_basis
    out["curvatures"] = curv
    out["w_orth"] = float(w_orth)
    out["arm"] = (f"d_neural_basis_galerkin(M={M}"
                  + (f",w_orth={w_orth:g})" if w_orth > 0 else ")"))
    out["rank_median"] = float(np.median(ranks))
    out["rank_min"] = int(np.min(ranks))
    return out
