"""P4(L형 평면탄성)에 2차원 신경 arm을 올린다 — **mesh-free 주장이 실제로 시험되는 곳**.

여기까지 신경 arm은 1차원 문제(P1)에서만 돌았고, 재진입 코너 정의역에서는 돌지 않았다.
논문이 "mesh-free 주장이 진짜 걸려 있는 무대"라고 부른 자리가 비어 있던 것이다. 이 모듈이
그것을 채운다.

## 하드 경계조건 — 비볼록 정의역에서 곱셈 다항식은 못 쓴다

Ω = [0,2]×[0,1] ∪ [0,1]×[1,2]이고 **∂Ω 전체가 클램프**다. 1차원에서 쓴 `x²·N(x)` 같은
곱셈 인자를 여기서 만들려고 변마다 선형함수를 곱하면, 재진입 변의 **연장선이 정의역
내부를 지난다**(예: x=1은 0<y<1 구간에서 Ω의 내부다). 그런 인자는 내부 직선 위에서 해를
0으로 강제해 버린다.

그래서 Sukumar & Srivastava [3]의 **정규화 근사거리(ADF)와 R-등가**를 쓴다. 변 i마다

    φ_i = √(d_i² + ((√(t_i² + d_i⁴) − t_i)/2)²)

로 선분에서만 0이 되는 매끄러운 함수를 만들고(d_i는 변을 담은 직선까지의 부호거리,
t_i는 선분 밖을 잘라내는 접선항), 이를 R-등가

    φ_Ω = (Σ_i φ_i^{−m})^{−1/m},   m = 2

로 합친다. φ_Ω는 ∂Ω에서 정확히 0이고 내부에서 양수이며 꼭짓점을 뺀 모든 곳에서 매끄럽다.
시행함수는

    u(x,y) = φ_Ω(x,y) · N(x,y),    N: R² → R² (tanh MLP)

이므로 클램프 조건이 **구조적으로** 만족되고 경계 벌점이 없다 — P1에서와 같은 계약이다.

## 약형식과 무차원화

평면응력, L = E = ρ = 1, ν = 0.29. 변형률 ε = (u_x, v_y, u_y+v_x)에 대해

    Ω² = ∫ εᵀCε dΩ / ∫ (u² + v²) dΩ,   C = 1/(1−ν²)·[[1,ν,0],[ν,1,0],[0,0,(1−ν)/2]]

이고 이것이 `p4_lshape.solve`가 내는 것과 같은 무차원 고유값이다. H¹ 문제라 1차 도함수만
필요하고, P1의 H² 문제보다 자동미분이 싸다.

## 구적

정의역을 세 단위블록 A=[0,1]², B=[1,2]×[0,1], C=[0,1]×[1,2]로 쪼개 각 블록에 텐서
Gauss–Legendre를 깐다. 블록 경계가 곧 재진입 코너를 지나는 선이므로 적분이 특이점을
가로지르지 않는다. **코너의 응력특이성(u ~ r^λ, λ<1)은 시행공간이 매끄러워 표현되지
않는다** — 그것이 이 문제에서 재는 e_approx이고, 고전 등급메시와의 대비가 §5.4의 논점이다.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, stack_module_state, vmap

from ..problems.p4_lshape import P4_GEOMETRY
from ..quadrature import gauss_legendre

# ∂Ω의 여섯 변 (반시계). 재진입 코너는 (1,1).
EDGES = (((0.0, 0.0), (2.0, 0.0)),      # 아래
         ((2.0, 0.0), (2.0, 1.0)),      # 오른쪽
         ((2.0, 1.0), (1.0, 1.0)),      # B의 위 = 재진입
         ((1.0, 1.0), (1.0, 2.0)),      # C의 오른쪽 = 재진입
         ((1.0, 2.0), (0.0, 2.0)),      # 위
         ((0.0, 2.0), (0.0, 0.0)))      # 왼쪽
BLOCKS = (((0.0, 1.0), (0.0, 1.0)),
          ((1.0, 2.0), (0.0, 1.0)),
          ((0.0, 1.0), (1.0, 2.0)))


def quadrature(n_per_block: int = 20, device: str = "cpu"):
    """세 단위블록의 텐서 Gauss. 반환 (pts(n,2), w(n,))."""
    P, W = [], []
    for (x0, x1), (y0, y1) in BLOCKS:
        xs, wx = gauss_legendre(n_per_block, x0, x1)
        ys, wy = gauss_legendre(n_per_block, y0, y1)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        P.append(np.stack([X.ravel(), Y.ravel()], 1))
        W.append(np.outer(wx, wy).ravel())
    p = torch.tensor(np.concatenate(P), dtype=torch.float64, device=device)
    w = torch.tensor(np.concatenate(W), dtype=torch.float64, device=device)
    return p, w


def _graded_cuts(a: float, b: float, corner: float, layers: int, ratio: float):
    """[a, b]를 corner(a나 b) 쪽으로 기하 등급을 주어 자르는 절점들 — 코너에서 L·ratio^j(j = 1..layers)."""
    L = b - a
    cuts = [corner + (L * ratio ** j if corner == a else -L * ratio ** j) for j in range(1, layers + 1)]
    return sorted({a, b, *cuts})


def quadrature_graded(n_per_cell: int = 20, layers: int = 0, device: str = "cpu",
                      ratio: float = 0.5):
    """재진입 코너 (1, 1) 쪽으로 등급을 준 합성 텐서 가우스(P4-ADF, MLST 재투고 계산).

    블록마다 두 축을 코너 쪽으로 L·ratio^j(j = 1..layers)에서 잘라 (layers + 1)²개 셀로 나누고, 셀마다
    `n_per_cell`² 가우스를 깐다. `layers = 0`이면 훈련 구적 `quadrature(n_per_cell)`과 같은 점이다."""
    P, W = [], []
    for (x0, x1), (y0, y1) in BLOCKS:
        cx = x1 if x1 == 1.0 else x0                   # 코너의 x(블록의 끝 중 1인 쪽)
        cy = y1 if y1 == 1.0 else y0
        xc = _graded_cuts(x0, x1, cx, layers, ratio)
        yc = _graded_cuts(y0, y1, cy, layers, ratio)
        for xa, xb in zip(xc[:-1], xc[1:]):
            xs, wx = gauss_legendre(n_per_cell, xa, xb)
            for ya, yb in zip(yc[:-1], yc[1:]):
                ys, wy = gauss_legendre(n_per_cell, ya, yb)
                X, Y = np.meshgrid(xs, ys, indexing="ij")
                P.append(np.stack([X.ravel(), Y.ravel()], 1))
                W.append(np.outer(wx, wy).ravel())
    return (torch.tensor(np.concatenate(P), dtype=torch.float64, device=device),
            torch.tensor(np.concatenate(W), dtype=torch.float64, device=device))


def adf_corner_profile(radii=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6), n_angles: int = 13) -> list:
    """재진입 코너 (1, 1) 둘레에서 φ_Ω와 |∇φ_Ω|의 반지름별 범위(P4-ADF). 정의역은 코너에서 θ ∈ (π/2, 2π)다."""
    th = np.pi / 2 + 1.5 * np.pi * (np.arange(1, n_angles + 1) / (n_angles + 1))
    rows = []
    for r in radii:
        xy = torch.tensor(np.stack([1.0 + r * np.cos(th), 1.0 + r * np.sin(th)], 1),
                          dtype=torch.float64, requires_grad=True)
        phi = boundary_adf(xy)
        (g,) = torch.autograd.grad(phi.sum(), xy)
        gn = torch.linalg.vector_norm(g, dim=1)
        rows.append({"r": float(r), "n_angles": int(n_angles),
                     "phi_min": float(phi.min()), "phi_max": float(phi.max()),
                     "phi_over_r_max": float(phi.max() / r),
                     "grad_min": float(gn.min()), "grad_max": float(gn.max())})
    return rows


def _edge_adf(xy, a, b):
    """변 하나의 정규화 근사거리 [3]. 선분 위에서 0, 그 밖에서 양수·매끄럽다."""
    ax, ay = a
    bx, by = b
    L = float(np.hypot(bx - ax, by - ay))
    x, y = xy[..., 0], xy[..., 1]
    d = ((x - ax) * (by - ay) - (y - ay) * (bx - ax)) / L        # 직선까지 부호거리
    cx, cy = 0.5 * (ax + bx), 0.5 * (ay + by)
    t = ((0.5 * L) ** 2 - ((x - cx) ** 2 + (y - cy) ** 2)) / L   # 선분 밖 절단항
    inner = torch.sqrt(t ** 2 + d ** 4)
    return torch.sqrt(d ** 2 + (0.5 * (inner - t)) ** 2)


def boundary_adf(xy, m: int = 2, eps: float = 1e-300):
    """φ_Ω = (Σ φ_i^{−m})^{−1/m}. ∂Ω에서 정확히 0, 내부에서 양수."""
    acc = None
    for a, b in EDGES:
        phi = _edge_adf(xy, a, b).clamp_min(eps)
        term = phi ** (-m)
        acc = term if acc is None else acc + term
    return acc ** (-1.0 / m)


class _VecMLP(nn.Module):
    """N: R² → R². 시행함수는 u = φ_Ω·N이므로 여기서 BC를 신경쓰지 않는다."""

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


class EnsembleVecMLP:
    """시드별 파라미터를 배치 텐서로 쌓아 vmap 한 번에 전 시드를 학습한다(P1과 같은 방식)."""

    def __init__(self, n_seeds: int, width: int = 64, depth: int = 4,
                 seed: int = 0, device: str | None = None):
        torch.set_default_dtype(torch.float64)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        models = [_VecMLP(width, depth).to(dev) for _ in range(n_seeds)]
        self.params, _ = stack_module_state(models)
        self.base = _VecMLP(width, depth).to(dev).to("meta")
        self.n_seeds, self.width, self.depth, self.device = n_seeds, width, depth, dev

    def disclosure(self) -> dict:
        n = sum(v[0].numel() for v in self.params.values())
        return {"width": self.width, "depth": self.depth, "activation": "tanh",
                "n_params_per_seed": int(n), "n_seeds": self.n_seeds,
                "ansatz": "u = phi_Omega(x,y) * N(x,y), phi_Omega = R-equivalence ADF"}


def _u_comp(p, base, pt, j: int):
    """한 점에서 u_j — grad로 미분하기 위한 스칼라 진입점."""
    n = functional_call(base, (p, {}), (pt.reshape(1, 2),)).reshape(-1)
    return boundary_adf(pt.reshape(1, 2)).reshape(()) * n[j]


def _state_one(p, base, pts):
    """한 시드의 상태 (6, n_q) = (u, v, u_x, u_y, v_x, v_y).

    정확 질량-직교 사영은 장(場)뿐 아니라 **변형률까지** 같은 계수로 빼야 한다 —
    사영된 시행함수의 Rayleigh 몫을 재려면 그 함수의 도함수가 필요하다. 그래서 여섯
    성분을 한 텐서로 들고 다닌다."""
    g0 = vmap(lambda t: grad(_u_comp, argnums=2)(p, base, t, 0))(pts)   # (n_q,2)
    g1 = vmap(lambda t: grad(_u_comp, argnums=2)(p, base, t, 1))(pts)
    u = vmap(lambda t: _u_comp(p, base, t, 0))(pts)
    v = vmap(lambda t: _u_comp(p, base, t, 1))(pts)
    return torch.stack([u, v, g0[:, 0], g0[:, 1], g1[:, 0], g1[:, 1]], 0)


def _rq_from_state(S, w, nu: float):
    """상태에서 Ω² = 변형에너지/질량."""
    u, v, ux, uy, vx, vy = S
    c = 1.0 / (1.0 - nu ** 2)
    gxy = uy + vx
    num = (w * c * (ux ** 2 + vy ** 2 + 2.0 * nu * ux * vy
                    + 0.5 * (1.0 - nu) * gxy ** 2)).sum()
    den = (w * (u ** 2 + v ** 2)).sum()
    return num / den


def _project_exact(S, P, w):
    """정확 질량-직교 사영. S:(6,n_q), P:(k,6,n_q) → 사영된 상태 (6,n_q).

    질량 내적 ⟨a,b⟩ = ∫(a_u b_u + a_v b_v)는 성분 0·1만 쓰고, 얻은 계수를 **여섯 성분
    전부에** 적용한다 — 선형연산이므로 도함수도 같은 계수로 따라간다."""
    G = torch.einsum("aij,j,bij->ab", P[:, :2], w, P[:, :2])
    r = torch.einsum("aij,j,ij->a", P[:, :2], w, S[:2])
    c = torch.linalg.solve(
        G + 1e-14 * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), r)
    return S - torch.einsum("a,anj->nj", c, P)


def _project_diagonal(S, P, w):
    """arm (a′)의 2차원 대응 — 그람의 대각만 쓴다. 저장 모드가 서로 질량직교일 때만
    정확 사영과 같고, 아니면 **사영이 아닌 연산자**다."""
    num = torch.einsum("aij,j,ij->a", P[:, :2], w, S[:2])
    den = torch.einsum("aij,j,aij->a", P[:, :2], w, P[:, :2])
    return S - torch.einsum("a,anj->nj", num / den, P)


def _project_qr(S, P, w, drop_rtol: float | None = None):
    """arm (b′)의 2차원 대응 — 그람을 만들지 않는다.

    질량 내적은 성분 0·1만 쓰지만 직교화 연산은 **여섯 성분 전부에** 같은 계수로
    적용한다. 그래야 사영된 상태의 변형률이 그 함수의 변형률이 된다."""
    eps = torch.finfo(S.dtype).eps
    tiny = torch.finfo(S.dtype).tiny
    rtol = eps ** 0.5 if drop_rtol is None else float(drop_rtol)
    Q = []
    for a in range(P.shape[0]):
        v = P[a]
        n0 = torch.sqrt(torch.clamp((w * (v[0] ** 2 + v[1] ** 2)).sum(), min=0.0))
        for q in Q:
            r = (w * (q[0] * v[0] + q[1] * v[1])).sum()
            v = v - r * q
        n = torch.sqrt(torch.clamp((w * (v[0] ** 2 + v[1] ** 2)).sum(), min=0.0))
        ok = n > rtol * torch.clamp(n0, min=tiny)
        Q.append(v * torch.where(ok, 1.0 / torch.clamp(n, min=tiny),
                                 torch.zeros_like(n)))
    out = S
    for q in Q:
        out = out - (w * (q[0] * S[0] + q[1] * S[1])).sum() * q
    return out


PROJECTIONS = {"exact": _project_exact, "diagonal": _project_diagonal,
               "qr": _project_qr}

# 저장 규약(MLST 재투고 계산). "projected"는 커밋된 경로 — 학습한 단계의 사영된 상태를 저장하므로 저장 집합이
# 구성상 직교이고 세 규칙이 같은 사영이다. "raw"는 P1처럼 사영 전 원시 상태를 저장한다.
STORAGE = ("projected", "raw")
SPAN_KEYS = ("k", "kappa_gram", "kappa_gram_times_eps", "dev_identity", "max_offdiag",
             "lam_min_normalized", "qr_orth_defect", "n_dropped", "rank_1e10", "rank_1e12",
             "rank_1e16", "proj_agreement_rel", "gram_solve_finite")


def _span_rows(stage_mode: int, P, w, S_raw, S_proj) -> list:
    """시드마다 한 행 — 그 단계가 넘겨받은 저장 집합의 진단과, 적용된 사영의 실제 누설
    ‖ΦᵀW φ̃ / √diag G‖ / ‖φ̃‖_W(P1 `mode3_leak_rel`과 같은 정의, 두 성분 질량 내적)."""
    from . import diagnostics as dgn
    w2 = torch.cat([w, w], dim=-1)

    def flat(A):
        return torch.cat([A[..., 0, :], A[..., 1, :]], dim=-1)

    rows = []
    for s_ in range(P.shape[0]):
        f = flat(P[s_])
        d = dgn.span_diagnostics(f, f, w2, phi=flat(S_raw[s_]), d2=flat(S_raw[s_]))
        G = torch.einsum("ai,i,bi->ab", f, w2, f)
        sp = flat(S_proj[s_])
        leak = torch.einsum("ai,i,i->a", f, w2, sp) / torch.sqrt(torch.diagonal(G))
        nrm = torch.sqrt((w2 * sp * sp).sum())
        rows.append({"stage_mode": int(stage_mode), "seed": int(s_),
                     **{k_: d[k_] for k_ in SPAN_KEYS},
                     "leak_rel": float(torch.linalg.vector_norm(leak) / torch.clamp(nrm, min=1e-300))})
    return rows


def project(S, P, w, kind: str = "exact"):
    """사영 전략 하나를 고른다. 이름은 P1의 arm과 같은 규약이다
    (exact = (b), diagonal = (a′), qr = (b′))."""
    try:
        return PROJECTIONS[kind](S, P, w)
    except KeyError:
        raise ValueError(f"알 수 없는 사영: {kind!r} — {sorted(PROJECTIONS)}") from None


ARM_NAME = {"exact": "p4_neural_2d[exact projection]",
            "diagonal": "p4_neural_2d[diagonal projection]",
            "qr": "p4_neural_2d[qr projection]"}


def span_diagnostics_2d(P, w, probe=None, n_diag_seeds=None):
    """P4 저장 스팬의 진단 — P1과 **같은 지표**를 두 성분 질량 내적에서 잰다.

    ⟨a,b⟩ = ∫(a_u b_u + a_v b_v)는 두 성분을 이어붙이고 가중을 두 번 쓴 1차원 내적과
    같다. 그래서 `neural.diagnostics.span_diagnostics`를 그대로 재사용한다 — 지표 정의가
    두 문제에서 갈리면 비교가 성립하지 않는다.

    `P`는 (k, 6, n_q)이거나 시드축이 앞에 붙은 (n_seeds, k, 6, n_q)다."""
    from . import diagnostics as dgn

    w2 = torch.cat([w, w], dim=-1)

    def _flat(A):
        return torch.cat([A[..., 0, :], A[..., 1, :]], dim=-1)

    def one(Pk, pr=None):
        flat = _flat(Pk)                                        # (k, 2 n_q)
        if pr is None:
            return dgn.span_diagnostics(flat, flat, w2)
        f = _flat(pr)
        return dgn.span_diagnostics(flat, flat, w2, phi=f, d2=f)

    if P.dim() == 3:
        return one(P, None if probe is None else probe)
    # **기본은 셀의 전 시드다.** 앞 몇 시드만 요약하면 κ₂(G)처럼 꼬리가 두꺼운 지표가
    # 표마다 어긋나고, 채점 결과는 멀쩡하므로 두 표를 대조하기 전에는 보이지 않는다.
    # P1 경로(`sequential._summarize_span`)와 같은 규칙이어야 두 문제가 비교 가능하다.
    ns = (int(P.shape[0]) if n_diag_seeds is None
          else min(int(n_diag_seeds), int(P.shape[0])))
    per = [one(P[i], None if probe is None else probe[i]) for i in range(ns)]
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


def solve_p4_neural(n_modes: int = 3, n_seeds: int = 50, iters: int = 4000,
                    n_per_block: int = 20, seed: int = 0, lr: float = 2e-3,
                    width: int = 64, depth: int = 4, snapshot_every: int = 25,
                    geometry=None, projection: str = "exact",
                    check_n_per_block: int | None = None,
                    device: str | None = None, storage: str = "projected",
                    return_params: bool = False, per_seed_span: bool = False) -> dict:
    """L형 평면탄성의 최저 n_modes개 Ω²를 순차 Rayleigh 최소화로 푼다.

    `projection`은 P1의 arm과 같은 세 규칙이다: `"exact"` = (b) 정확 질량-직교 사영(기본),
    `"diagonal"` = (a′) 파일럿의 대각근사, `"qr"` = (b′) W-내적 QR. (b)가 P1에서 모드 10까지
    벽 없이 갔으므로 2차원 비교의 기준선이고, (a′)를 같은 정의역에 올리면 lower-mode basin이
    **차원에 무관한 기전**인지가 시험된다.

    **jitter는 (b) 경로에만 있다.** `_project_exact`는 그람에 1e-14·I를 더한다(원래 구현).
    (a′)·(b′)에는 없다 — 대각 규칙은 역행렬을 풀지 않고 QR은 애초에 그람을 만들지 않기
    때문이다. 이 비대칭은 표 각주와 부록 B에 적는다.

    `check_n_per_block`은 **학습에 쓰지 않은 두 번째 구적차수**로 같은 망을 다시 재는
    계기다(P1 `solve_sequential(check_n_q=)`와 같은 규약). Λ가 Λ₁ 위에 있으면서
    과소적분된 셀은 하한 스캔에 걸리지 않으므로, 그 빈틈은 두 차수로만 닫힌다.
    기본은 꺼짐 — 커밋된 실행의 결과를 바꾸지 않는다.

    `storage`(MLST 재투고 계산): 기본 "projected"는 커밋된 경로로, 저장 집합이 구성상 직교라 세 규칙이 같은
    사영이다. "raw"는 P1처럼 사영 전 원시 상태를 저장한다 — 모드 1·2는 두 규약에서 같고(모드 2는 저장 모드
    하나에 사영한다) 모드 3부터 갈린다. `return_params`는 단계마다 망 파라미터를, `per_seed_span`은 시드마다
    저장 집합의 진단과 적용 사영의 누설(`span_per_seed`)을 싣는다. 셋 다 기본값이면 커밋된 경로 그대로다.
    """
    if storage not in STORAGE:
        raise ValueError(f"storage는 {STORAGE} 중 하나여야 합니다: {storage!r}")
    torch.set_default_dtype(torch.float64)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = P4_GEOMETRY if geometry is None else geometry
    nu = float(g["nu"])
    pts, w = quadrature(n_per_block, dev)
    # 검사격자 재평가에는 저장값이 아니라 **학습된 망**이 필요하다 — 저장값은 pts에서만
    # 유효하다. 그래서 앙상블을 들고 있는다(모드 3개 × 소형 MLP라 메모리 부담이 없다).
    ensembles = []

    lam, shapes, prevs, stage_secs, hists = [], [], [], [], []
    span_diag, raw_shapes, span_rows = [], [], []
    for k in range(n_modes):
        ens = EnsembleVecMLP(n_seeds, width, depth, seed=seed * 1000 + k, device=dev)
        ensembles.append(ens)
        opt = torch.optim.Adam(list(ens.params.values()), lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=max(iters // 3, 1), gamma=0.4)

        proj = PROJECTIONS[projection]

        def loss_one(p, *prev):
            S = _state_one(p, ens.base, pts)
            if prev:
                S = proj(S, torch.stack(prev, 0), w)
            return _rq_from_state(S, w, nu)

        def snap_one(p, *prev):
            S = _state_one(p, ens.base, pts)
            if prev:
                S = proj(S, torch.stack(prev, 0), w)
            return _rq_from_state(S, w, nu)

        t0, hist = time.perf_counter(), []
        prev_stack = tuple(prevs)          # 각 원소가 (n_seeds, 6, n_q) — vmap이 시드축을 맵
        for it in range(iters + 1):
            r = vmap(loss_one)(ens.params, *prev_stack)
            if it % snapshot_every == 0 or it == iters:
                with torch.no_grad():
                    R = vmap(snap_one)(ens.params, *prev_stack)
                hist.append((it, time.perf_counter() - t0, R.cpu().numpy()))
            if it == iters:
                break
            r.sum().backward()
            opt.step()
            sched.step()
            opt.zero_grad()
        stage_secs.append(time.perf_counter() - t0)
        hists.append(hist)

        with torch.no_grad():
            St = vmap(lambda p: _state_one(p, ens.base, pts))(ens.params)  # (S,6,n_q)
            St_raw = St
            # **진단은 사영 전에.** 넘겨받은 스팬과 그 스팬에 사영될 원시 후보가 여기서만
            # 동시에 존재한다 — P1의 `solve_sequential`과 같은 규약이다.
            span_diag.append(span_diagnostics_2d(torch.stack(prevs, 1), w, probe=St)
                             if prevs else None)
            if prev_stack:
                St = vmap(lambda a, *b: proj(a, torch.stack(b, 0), w))(
                    St, *prev_stack)
            if per_seed_span and prevs:
                span_rows += _span_rows(k + 1, torch.stack(prevs, 1), w, St_raw, St)
            R = vmap(lambda a: _rq_from_state(a, w, nu))(St)
        lam.append(R.clamp_min(0).cpu().numpy())
        shapes.append(St[:, :2].cpu().numpy())
        raw_shapes.append(St_raw[:, :2].cpu().numpy())
        prevs.append(St_raw if storage == "raw" else St)

    # ── 구적분리: 같은 신경망을 두 번째 구적차수에서 다시 잰다 ──────────────
    # **P4는 prevs에 사영된 상태를 저장한다**(P1·P5는 원시). 검사격자에서도 같은 규약을
    # 써야 같은 스팬에 사영된다 — 원시를 쌓으면 다른 deflation을 재는 셈이 된다.
    lam_check = quad_drift = None
    if check_n_per_block:
        pts_c, w_c = quadrature(check_n_per_block, dev)
        proj_c = PROJECTIONS[projection]
        with torch.no_grad():
            prev_c, chk = [], []
            for e in ensembles:
                Sc = vmap(lambda p, e_=e: _state_one(p, e_.base, pts_c))(e.params)
                Sc_raw = Sc
                if prev_c:
                    Sc = vmap(lambda a, *b: proj_c(a, torch.stack(b, 0), w_c))(
                        Sc, *tuple(prev_c))
                chk.append(vmap(lambda a: _rq_from_state(a, w_c, nu))(Sc)
                           .clamp_min(0).cpu().numpy())
                prev_c.append((Sc_raw if storage == "raw" else Sc).detach())
        lam_check = np.array(chk)
        base = np.array(lam)
        with np.errstate(divide="ignore", invalid="ignore"):
            quad_drift = np.abs(lam_check - base) / np.where(base > 0, base, np.nan)

    return {"arm": ARM_NAME[projection], "projection": projection,
            "check_n_per_block": check_n_per_block,
            **({"lam_check": lam_check, "quadrature_drift": quad_drift}
               if check_n_per_block else {}),
            "lam": np.array(lam), "shapes": np.array(shapes),
            "raw_shapes": np.array(raw_shapes), "storage": storage,
            "params": ([{k_: v.detach().cpu().numpy() for k_, v in e.params.items()}
                        for e in ensembles] if return_params else None),
            **({"span_per_seed": span_rows} if per_seed_span else {}),
            "history": hists, "stage_seconds": stage_secs,
            "seconds": float(sum(stage_secs)),
            "pts": pts.cpu().numpy(), "w": w.cpu().numpy(),
            "n_q": int(pts.shape[0]), "n_seeds": n_seeds, "iters": iters,
            "span_diagnostics": span_diag,
            "n_per_block": n_per_block,
            "disclosure": EnsembleVecMLP(1, width, depth,
                                         device="cpu").disclosure()}
