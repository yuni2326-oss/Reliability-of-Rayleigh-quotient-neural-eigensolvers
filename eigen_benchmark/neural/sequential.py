"""순차 deflation solver — 모드 1..n을 차례로 잡고 사전등록 규칙으로 분류한다.

수렴한 하위모드는 **파라미터로 보관하고 현재 노드에서 재평가**한다. 고정 Gauss면
매번 같은 값이지만 MC 대조군에서는 노드가 바뀌므로 재평가가 필수이고, 그 비용까지가
MC의 실제 비용이다(스파이크: Gauss 402 s vs MC 3279 s).

**평가도 사영된 φ̃로 한다.** raw φ로 재면 하위모드 성분이 남아 λ가 β₂⁴ 아래로
내려가는 물리적으로 불가능한 값이 나온다(스파이크 1차 실행의 실제 오류).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.func import vmap

from .. import metrics as mt
from ..problems import p1_beam as p1
from . import core
from . import diagnostics as dgn
from .net import EnsembleMLP


# 저장 규약(MLST 재투고 계산, 사전규정 2026-10-14-mlst-prespecification.md). "raw"는 커밋된 경로 — 수렴한 망을
# 들고 매 반복 다시 잰다. "orthogonalized"는 저장 시 직교화(Rowan식) — 그 단계가 내놓은 사영된 상태를 고정 절점의
# 배열로 저장한다. 대각 규칙과 함께 쓰면 고전 Gram–Schmidt이고, 정확 사영과 같은 연산자가 된다.
STORAGE = ("raw", "orthogonalized")


def orthogonalize_pair(arrays, wq):
    """저장 시 직교화 — û_k = u_k − Σ_{j<k} (⟨u_k, û_j⟩_W / ⟨û_j, û_j⟩_W) û_j (고전 GS), 같은 계수를 φ″에도.

    `arrays`는 모드별 (φ, φ″) 목록이고 각 배열의 마지막 축이 절점이다(앞 축은 시드). 대각 규칙으로 사영한 그
    단계의 출력과 같은 식이다 — 앞 저장 모드가 서로 직교하면 대각 규칙이 정확 사영이기 때문이다."""
    out = []
    for ph, d2 in arrays:
        u, u2 = ph, d2
        for qh, q2 in out:
            c = (wq * ph * qh).sum(-1) / (wq * qh * qh).sum(-1)
            u = u - c[..., None] * qh
            u2 = u2 - c[..., None] * q2
        out.append((u, u2))
    return out


def mix_basis(Q, alpha: float):
    """같은 부분공간의 다른 기저 — v₁ = q₁, v₂ = (q₂ + αq₁)/√(1 + α²) (기저 불변 시험).

    q₁·q₂가 질량 정규직교면 ⟨v₁, v₂⟩_W = r = α/√(1 + α²)이고 두 벡터 모두 단위 노름이다. 앞 축이 기저
    번호이며 뒤 축은 무엇이든 된다(φ와 φ″에 같은 결합을 쓴다)."""
    a = float(alpha)
    return torch.stack([Q[0], (Q[1] + a * Q[0]) / float(np.sqrt(1.0 + a * a))])


def diagonal_operator_gap(V, wq) -> float:
    """‖P_G − A_D‖_W = ‖I − C‖₂ — 대각 규칙과 정확 사영의 질량 내적 연산자 거리(C는 질량 정규화 그람)."""
    G = torch.einsum("ki,i,li->kl", V, wq, V)
    d = torch.sqrt(torch.diagonal(G))
    C = G / (d[:, None] * d[None, :])
    return float((torch.linalg.eigvalsh(0.5 * (C + C.T)) - 1.0).abs().max())


def _converged(hist, tol: float = 1e-3, tail: float = 0.1) -> np.ndarray:
    """마지막 tail 구간의 R 상대변화가 tol 이하면 수렴(사전등록 규칙)."""
    if len(hist) < 3:
        return np.zeros_like(hist[-1][2], dtype=bool)
    k = max(int(len(hist) * tail), 2)
    a, b = hist[-k][2], hist[-1][2]
    rel = np.abs(b - a) / np.maximum(np.abs(b), 1e-300)
    return np.isfinite(b) & (rel <= tol)


def _prev_stack(prevs, base, xs):
    """이전 모드들을 현재 노드에서 재평가 — (n_seeds, k, n_q) 두 개."""
    PH, PH2 = [], []
    for pr in prevs:
        a, b = vmap(lambda q: core.eval_phi_one(q, base, xs))(pr)
        PH.append(a)
        PH2.append(b)
    return torch.stack(PH, 1), torch.stack(PH2, 1)


def solve_sequential(n_modes: int, deflation, n_seeds: int = 50, iters=4000,
                     nodes: str = "gauss", n_q: int = 256, init=None,
                     seed: int = 0, lr: float = 2e-3, snapshot_every: int = 25,
                     n_mc: int = 2000, barrier=None, dtype=None,
                     n_diag_seeds: int | None = None,
                     check_n_q: int | None = None,
                     device: str | None = None, storage: str = "raw",
                     stored_arrays=None) -> dict:
    """`iters`는 정수이거나 **모드별 예산 시퀀스**다.

    `barrier`는 사다리 I2용이다 — 초기화가 아니라 **목적함수에 붙는 항**으로, 목표
    고유값 구간만 아는 상태를 모형화한다(형상정보는 주지 않는다). 그래서 `init`과
    배타적으로 쓰이고, 이것이 없으면 I2는 사다리에서 조용히 빠진다(실제로 빠져 있었다).

    장기변주에서 이미 100 %인 저차모드까지 5배로 돌리는 것은 낭비이므로,
    실패가 관측된 모드에만 큰 예산을 준다. 예: [4000]*8 + [20000]*2.

    `check_n_q`는 **학습에 쓰지 않은 두 번째 구적차수**로 같은 망을 다시 재는 계기다.
    학습되는 기저는 자신을 재는 격자에 맞춰 변형하므로, 참조 모드로 사전 인증한 격자가
    최적화가 도달한 함수까지 분해한다는 보장이 없다(P5에서 실측). 켜면 `lam_check`와
    `quad_drift`가 실린다. 기본은 꺼짐 — 커밋된 실행의 결과를 바꾸지 않는다.

    `storage`·`stored_arrays`(MLST 재투고 계산)는 **배열 저장**을 연다. 고정 가우스 절점에서는 저장 모드를 망
    대신 그 절점 위 (φ, φ″) 배열로 들고 있어도 같은 deflation이다. `stored_arrays`는 앞 모드들의 배열이며 그
    단계는 학습하지 않는다 — 같은 원시 쌍에서 모드 3만 푸는 고정 접두부, 같은 부분공간의 기저만 바꾸는 시험이
    이 경로다. 학습하는 단계의 초기화 시드는 순차 풀이와 같다(seed·1000 + k). `storage="orthogonalized"`는 그
    단계의 사영된 상태를 저장한다(Rowan식 저장 시 직교화). 둘 다 기본값이면 커밋된 경로 그대로다."""
    if storage not in STORAGE:
        raise ValueError(f"storage는 {STORAGE} 중 하나여야 합니다: {storage!r}")
    n_stored = 0 if stored_arrays is None else len(stored_arrays)
    arrays_mode = storage == "orthogonalized" or n_stored > 0
    if arrays_mode and nodes != "gauss":
        raise ValueError("배열 저장은 고정 가우스 절점에서만 같은 deflation이다(MC 절점은 매 반복 바뀐다)")
    if arrays_mode and check_n_q:
        raise ValueError("검사 격자 재평가는 망을 다시 재야 한다 — 배열 저장과 함께 쓰지 않는다")
    if n_stored >= n_modes:
        raise ValueError(f"저장 단계 {n_stored}개로는 풀 단계가 남지 않는다(n_modes={n_modes})")
    dtype = torch.float64 if dtype is None else dtype
    try:
        torch.set_default_dtype(dtype)
    except TypeError:
        torch.set_default_dtype(torch.float32)
    budget = ([int(iters)] * n_modes if np.isscalar(iters)
              else [int(v) for v in iters])
    if len(budget) != n_modes:
        raise ValueError(f"iters 시퀀스 길이({len(budget)})가 n_modes({n_modes})와 다릅니다")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(20260802 + seed)
    xs, wq = core.gauss_nodes(n_q, dev, dtype)

    # `shapes`는 **사영된** 함수이고 `prevs`(→ 다음 단계 deflation의 Φ)는 **원시 망 출력**이다.
    # 이 구분이 (a′)와 (b)를 가르는 지점이므로 원시 출력도 반환한다 — 진단이 사영된 쪽을
    # 재면 그람이 구성상 대각이 되어 "두 arm이 동일하다"는 잘못된 결론이 나온다.
    lam, shapes, raw_shapes, hists, prevs, stage_secs = [], [], [], [], [], []
    init_secs, span_diag = [], []
    raw_d2 = []          # 사영 전 원시 φ″ — 저장 쌍을 배열로 다시 쓰려면 곡률도 필요하다
    lam_check, quad_drift = [], []
    xs_c, wq_c = (core.gauss_nodes(check_n_q, dev, dtype) if check_n_q
                  else (None, None))
    total = 0.0
    for k in range(n_modes):
        if k < n_stored:
            # 저장된 단계 — 학습하지 않는다. 주어진 배열이 곧 그 단계의 저장 모드다.
            ph_s, d2_s = (torch.as_tensor(a).to(device=dev, dtype=dtype) for a in stored_arrays[k])
            raw_shapes.append(ph_s.double().cpu().numpy())
            raw_d2.append(d2_s.double().cpu().numpy())
            shapes.append(ph_s.double().cpu().numpy())
            lam.append(core.rayleigh(ph_s, d2_s, wq).clamp_min(0).double().cpu().numpy() ** 0.25)
            hists.append([])
            stage_secs.append(0.0)
            init_secs.append(0.0)
            span_diag.append(None)
            prevs.append((ph_s, d2_s))
            continue
        ens = EnsembleMLP(n_seeds, seed=seed * 1000 + k, device=dev, dtype=dtype)
        # **사전적합 비용을 따로 센다.** I3·I5는 목표에 H² 적합을 먼저 돌리므로 공짜가
        # 아니다. 이걸 청구하지 않으면 "warm-start는 이득이 없다"가 "이득도 없고 비용도
        # 없다"로 읽혀, 실제 결론("이득이 없는데 비용은 든다 = 열등")을 놓친다.
        _t_init = time.perf_counter()
        if init is not None:
            init(ens, k, xs, wq)                     # 사다리 I1–I5
        init_secs.append(time.perf_counter() - _t_init)

        # MC 노드는 **vmap 밖에서** 반복마다 새로 뽑는다(vmap 안 난수는 금지된다).
        node_state = {"x": xs, "w": wq}

        def redraw(_it):
            if nodes == "mc":
                node_state["x"], node_state["w"] = core.mc_nodes(n_mc, gen, dev, dtype)

        def loss_one(p, *pv):
            x, w = node_state["x"], node_state["w"]
            ph, d2 = core.eval_phi_one(p, ens.base, x)
            if arrays_mode and pv:
                # 배열 저장 — 시드축이 벗겨진 (k, n_q) 두 개가 들어온다.
                ph, d2, pen = deflation.apply(ph, d2, pv[0], pv[1], w)
            elif pv:
                PH = torch.stack([core.eval_phi_one(q, ens.base, x)[0].detach()
                                  for q in pv])
                PH2 = torch.stack([core.eval_phi_one(q, ens.base, x)[1].detach()
                                   for q in pv])
                ph, d2, pen = deflation.apply(ph, d2, PH, PH2, w)
            else:
                pen = ph.new_zeros(())
            R = (w * d2 ** 2).sum() / (w * ph ** 2).sum()
            if barrier is not None:
                pen = pen + barrier(R)       # I2: 고유값 창 배리어
            return R + pen

        if arrays_mode:
            batched = ((torch.stack([a for a, _ in prevs], 1), torch.stack([b for _, b in prevs], 1))
                       if prevs else ())
        else:
            batched = tuple(prevs)
        out = core.train(ens, loss_one, budget[k], lr, snapshot_every,
                         extra_batched=batched, on_iter=redraw)
        total += out["seconds"]
        stage_secs.append(out["seconds"])
        hists.append(out["history"])

        with torch.no_grad():
            ph, d2 = core.eval_phi(ens, xs)
            raw_shapes.append(ph.double().cpu().numpy())  # 사영 전 원시 출력
            raw_d2.append(d2.double().cpu().numpy())
            ph_raw, d2_raw = ph, d2
            if prevs:
                PH, PH2 = batched if arrays_mode else _prev_stack(prevs, ens.base, xs)
                # **진단은 사영 전에.** 이 스테이지가 실제로 넘겨받은 스팬(원시 저장 모드)과
                # 그 스팬에 사영될 원시 후보 φ가 여기서만 동시에 존재한다.
                span_diag.append(_summarize_span(PH, PH2, wq, ph, d2, n_diag_seeds))
                ph, d2, _ = vmap(lambda a, b, c, e:
                                 deflation.apply(a, b, c, e, wq))(ph, d2, PH, PH2)
            else:
                span_diag.append(None)
        r = core.rayleigh(ph, d2, wq).clamp_min(0).double().cpu().numpy()
        # **검사격자 재평가는 prevs에 이번 단계를 넣기 전에.** 넣은 뒤에 하면 자기 자신을
        # 스팬에 포함시켜 사영하게 되어 0이 나온다.
        if check_n_q:
            with torch.no_grad():
                ph_c, d2_c = core.eval_phi(ens, xs_c)
                if prevs:
                    PH_c, PH2_c = _prev_stack(prevs, ens.base, xs_c)
                    ph_c, d2_c, _ = vmap(lambda a, b, c, e:
                                         deflation.apply(a, b, c, e, wq_c))(
                        ph_c, d2_c, PH_c, PH2_c)
                r_c = (core.rayleigh(ph_c, d2_c, wq_c).clamp_min(0)
                       .double().cpu().numpy())
            lam_check.append(r_c ** 0.25)
            with np.errstate(divide="ignore", invalid="ignore"):
                quad_drift.append(np.abs(r_c - r) / np.where(r > 0, r, np.nan))
        lam.append(r ** 0.25)
        shapes.append(ph.double().cpu().numpy())
        if not arrays_mode:
            prevs.append(ens.clone_params())
        elif storage == "orthogonalized":
            prevs.append((ph.detach(), d2.detach()))        # 사영된 상태 = Rowan식 저장
        else:
            prevs.append((ph_raw.detach(), d2_raw.detach()))

    return {"lam": np.array(lam), "shapes": np.array(shapes),
            "raw_shapes": np.array(raw_shapes), "raw_d2": np.array(raw_d2),
            "history": hists, "storage": storage, "n_stored": n_stored,
            "stored": ([(a.double().cpu().numpy(), b.double().cpu().numpy()) for a, b in prevs]
                       if arrays_mode else None),
            "seconds": total, "xs": xs.double().cpu().numpy(),
            "wq": wq.double().cpu().numpy(),
            "converged": np.array([_converged(h) if h else np.zeros(n_seeds, dtype=bool)
                                   for h in hists]),
            "deflation": deflation.name, "nodes": nodes, "n_q": n_q,
            "iters": budget, "n_seeds": n_seeds, "stage_seconds": stage_secs,
            "init_seconds": init_secs, "span_diagnostics": span_diag,
            "dtype": str(dtype).replace("torch.", ""),
            "arm": f"sequential[{deflation.name}]",
            "check_n_q": check_n_q,
            **({"lam_check": np.array(lam_check),
                "quad_drift": np.array(quad_drift)} if check_n_q else {})}


def solve_fixed_span(target: int, deflation, n_ref_modes: int | None = None,
                     n_seeds: int = 50, iters: int = 4000, n_q: int = 256,
                     seed: int = 0, lr: float = 2e-3, snapshot_every: int = 25,
                     dtype=None, device: str | None = None) -> dict:
    """목표 모드 하나를, **해석 모드 1..target-1이 만드는 스팬**에 사영해 푼다.

    순차 solver에서 깊은 모드가 실패할 때 원인은 둘로 갈린다: 그 단계의 풀이가 어려운
    것인가, 아니면 사영 대상인 저장 집합이 이미 틀린 것인가. 앞 단계들을 정확한 해로
    바꿔치면 두 번째가 제거되므로, 회복 여부가 그대로 답이 된다.

    저장 집합이 파라미터가 아니라 고정 배열이므로 앞 모드들을 매 반복 재평가할 필요가
    없다 — 같은 목표를 순차 경로로 푸는 것보다 오히려 싸다.
    """
    dtype = torch.float64 if dtype is None else dtype
    try:
        torch.set_default_dtype(dtype)
    except TypeError:
        torch.set_default_dtype(torch.float32)
    k = int(target) - 1 if n_ref_modes is None else int(n_ref_modes)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    xs, wq = core.gauss_nodes(n_q, dev, dtype)
    xn = xs.double().cpu().numpy()
    PHI = torch.tensor(np.stack([p1.analytic_mode(xn, n) for n in range(1, k + 1)]),
                       device=dev).to(dtype) if k else None
    PHI2 = torch.tensor(np.stack([p1.analytic_mode_d2(xn, n) for n in range(1, k + 1)]),
                        device=dev).to(dtype) if k else None

    ens = EnsembleMLP(n_seeds, seed=seed * 1000 + int(target) - 1, device=dev, dtype=dtype)

    def loss_one(p):
        ph, d2 = core.eval_phi_one(p, ens.base, xs)
        if PHI is not None:
            ph, d2, pen = deflation.apply(ph, d2, PHI, PHI2, wq)
        else:
            pen = ph.new_zeros(())
        return (wq * d2 ** 2).sum() / (wq * ph ** 2).sum() + pen

    out = core.train(ens, loss_one, int(iters), lr, snapshot_every)

    with torch.no_grad():
        ph, d2 = core.eval_phi(ens, xs)
        if PHI is not None:
            ph, d2 = vmap(lambda a, b: deflation.apply(a, b, PHI, PHI2, wq)[:2])(ph, d2)
        r = core.rayleigh(ph, d2, wq).clamp_min(0).double().cpu().numpy()

    return {"target": int(target), "lam": r ** 0.25,
            "shapes": ph.double().cpu().numpy(),
            "xs": xn, "wq": wq.double().cpu().numpy(),
            "history": out["history"], "seconds": out["seconds"],
            "converged": _converged(out["history"]),
            "n_seeds": n_seeds, "iters": int(iters), "n_q": n_q,
            "n_ref_modes": k, "deflation": deflation.name,
            "dtype": str(dtype).replace("torch.", ""),
            "span_source": ("analytic clamped-free modes 1..%d, exact to root-finding "
                            "tolerance; no stored network output enters the deflation" % k),
            "arm": f"fixed_span[{deflation.name}]"}


def _summarize_span(PH, PH2, wq, ph, d2, n_diag_seeds=None) -> dict:
    """스팬 진단의 시드 중앙값. **기본은 셀의 전 시드다.**

    처음에는 앞 8개 시드만 썼다. 근거는 "전 시드 eigvalsh가 비싸다"였는데 실측해 보니
    진단은 스테이지당 한 번이고 행렬이 k×k(k ≤ 20)라 학습 4000회에 비하면 무시할 만하다.
    그 절충의 대가는 컸다 — 같은 arm의 같은 κ₂(G)가 8시드 표와 50시드 표에서 최대 3배
    달라, 논문 안에서 한 물리량이 두 값을 가졌다. 요약 시드 수는 계속 기록하되 기본을
    전 시드로 둔다."""
    ns = int(PH.shape[0]) if n_diag_seeds is None else min(int(n_diag_seeds),
                                                           int(PH.shape[0]))
    per = [dgn.span_diagnostics(PH[i], PH2[i], wq, phi=ph[i], d2=d2[i])
           for i in range(ns)]
    out = {"n_seeds_summarized": ns}
    for key in per[0]:
        vals = [d[key] for d in per]
        if isinstance(vals[0], bool):
            out[key + "_frac"] = float(sum(vals)) / len(vals)
        elif isinstance(vals[0], str):
            out[key] = vals[0]
        elif isinstance(vals[0], int):
            out[key] = int(np.median(vals))
        else:
            out[key] = float(np.median(vals))
    return out


def classify_stage(shapes, lam, target: int, xs, wq, n_ref: int = 14,
                   converged=None):
    """사전등록 4분기 판정(계획 1 `metrics.classify` 위임). 기준 모드형은 해석해."""
    REF = np.stack([p1.analytic_mode(xs, n) for n in range(1, n_ref + 1)], axis=1)
    ref_lam = p1.beta_roots(n_ref) ** 4
    W = np.diag(wq)
    out = []
    for i in range(shapes.shape[0]):
        ok = True if converged is None else bool(converged[i])
        out.append(mt.classify(shapes[i], float(lam[i]) ** 4, REF, ref_lam,
                               target=target, converged=ok, M=W))
    return out
