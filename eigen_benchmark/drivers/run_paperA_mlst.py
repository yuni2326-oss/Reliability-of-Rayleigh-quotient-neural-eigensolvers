"""논문 A MLST 재투고 계산 — 사전규정 `docs/paper2-cs/2026-10-14-mlst-prespecification.md`.

검토(`docs/paper2-cs/2026-10-14-mlst-plan-review.md`)가 정한 다섯 실행과 그 판정이다.

1. **P1-RF**(Rowan-style storage-orthogonalization control). 시드마다 모드 1–2를 커밋된 (a′) 경로로 한 번
   풀어(고정 접두부) 그 원시 쌍을 세 갈래가 함께 쓴다 — (a′) 원시 쌍 + 대각, RF 같은 쌍을 저장 시 직교화 +
   대각, (b) 원시 쌍 + 정확 사영. 모드 3이 주 판정이고 모드 4–10은 각 갈래가 제 경로로 이어 간다.
2. **fp32 RF**. 같은 RF 갈래를 fp32로(접두부도 fp32) — fp32 모드 6의 (b) 0.16·(b′) 0.88과 대비한다.
3. **기저 불변 시험**. 같은 부분공간의 기저만 v₂ = (q₂ + αq₁)/√(1+α²)로 바꿔 대각 규칙으로 모드 3을 푼다
   (결과 수준), 그리고 학습 없이 연산자 거리와 누설을 잰다(연산자 수준).
4. **P4-RAW**. 커밋된 P4는 사영된 상태를 저장해 세 규칙이 같은 사영이었다. P1처럼 원시 상태를 저장해 세
   규칙 × 50시드 × 모드 1–3을 돌고, 모드 3 직전 원시 쌍의 시드별 진단(P4-SPAN)과 망을 남긴다.
5. **P4-ADF**. 재진입 코너의 ADF 단면, 그리고 코너 쪽 등급 구적 3단계(0·4·8겹)에서 참조 FEM 모드와 정답
   신경 셀의 Λ 변화.

P1·P4의 커밋된 경로는 바꾸지 않는다 — 새 경로는 솔버의 저장 규약 인자(`storage`, `stored_arrays`)다.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .. import metrics as mt
from . import manifest

N_SEEDS, ITERS, N_MODES_P1 = 50, 4000, 10
# 결과 수준 — r = 0.010, 0.050, 0.243, 0.707(로그 간격). 양 끝 r = 0(RF)과 r ≈ 0.991((a′))은 고정 접두부의
# 갈래가 덮는다. Adam의 걸음은 기울기 크기에 거의 무관하므로 끌림이 작은 r에서 시작할 수 있어 전이 구간을 본다.
ALPHAS = (0.01, 0.05, 0.25, 1.0)
ALPHAS_OPERATOR = (0.0, 0.01, 0.05, 0.25, 1.0, 4.0, 7.4)
ADF_LEVELS = (0, 4, 8)                        # 코너 쪽 기하 등급의 겹 수(0 = 훈련 구적)
ADF_N_PER_CELL = 20
P4_ARMS = ("diagonal", "exact", "qr")
B_BOOT, BOOT_SEED = 20_000, 12345
A_PRIME, B_EXACT = "a_prime_projection_diagonal", "b_projection_exact"
FROZEN = {"a_prime": "a_prime_frozen", "rf": "rf_frozen", "b": "b_frozen"}


# ── 통계 — 시드 군집 쌍 부트스트랩(같은 시드가 군집, B = 20,000, 난수 12345) ─────────────────

def paired_bootstrap(a, b, B: int = B_BOOT, seed: int = BOOT_SEED):
    """짝지은 차이 a − b의 (점추정, 2.5 %, 97.5 %). 시드가 군집이다(`p5_stats.paired_bootstrap`과 같은 산술)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("짝지은 비교는 같은 시드 집합이어야 한다")
    idx = np.random.default_rng(seed).integers(0, len(a), size=(B, len(a)))
    lo, hi = np.percentile(a[idx].mean(1) - b[idx].mean(1), [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def rule(ci) -> str:
    """규칙 1: 구간이 0을 제외하고 양수, 규칙 3: 0을 제외하고 음수, 규칙 2: 0을 포함."""
    return "rule1" if ci[1] > 0 else "rule3" if ci[2] < 0 else "rule2"


def _indicator(recs, mode: int, outcome: str = "correct", n_seeds: int | None = None) -> np.ndarray:
    by = {int(r["seed"]): float(r["outcome"] == outcome) for r in recs if int(r["mode"]) == int(mode)}
    n = len(by) if n_seeds is None else int(n_seeds)
    if sorted(by) != list(range(n)):
        raise ValueError(f"모드 {mode}의 시드가 0..{n - 1}이 아니다: {sorted(by)[:5]}…")
    return np.array([by[s] for s in range(n)])


# ── P1 — 고정 접두부, RF, fp32, 기저 불변 ─────────────────────────────────────────────

def p1_prefix(n_seeds: int = N_SEEDS, iters: int = ITERS, dtype=None, device=None) -> dict:
    """모드 1–2를 커밋된 (a′) 경로(망 저장, 대각 규칙)로 푼다. 모드 1은 사영이 없고 모드 2는 저장 모드 하나라
    세 규칙이 같은 알고리즘이므로, 이 쌍을 세 갈래가 함께 쓰는 것이 정당하다(계획 v2 §3.1)."""
    from ..neural import deflation as df
    from ..neural import sequential as sq
    return sq.solve_sequential(2, df.ProjectionDiagonal(), n_seeds=n_seeds, iters=iters,
                               dtype=dtype, device=device)


def _pair(out, dtype=None):
    import torch
    dt = torch.float64 if dtype is None else dtype
    return [(torch.as_tensor(out["raw_shapes"][k]).to(dt), torch.as_tensor(out["raw_d2"][k]).to(dt))
            for k in range(2)]


def _pair_rows(arm: str, pair, wq, cand, rule_name: str) -> list:
    """시드마다 저장 쌍의 진단과, 그 갈래 규칙으로 사영한 모드 3 원시 후보의 누설(P1 `mode3_leak_rel`과 같은 정의)."""
    import torch
    from ..neural import deflation as df
    from ..neural import diagnostics as dgn
    w = torch.as_tensor(wq, dtype=pair[0][0].dtype)
    rows = []
    for s_ in range(pair[0][0].shape[0]):
        P = torch.stack([pair[0][0][s_], pair[1][0][s_]])
        P2 = torch.stack([pair[0][1][s_], pair[1][1][s_]])
        phi = torch.as_tensor(cand[s_], dtype=P.dtype)
        d = dgn.span_diagnostics(P, P2, w, phi=phi, d2=phi)
        c = (df.project_diagonal if rule_name == "diagonal" else df.project_exact)(phi, P, w)
        tphi = phi - c @ P
        G = torch.einsum("ki,i,li->kl", P, w, P)
        leak = torch.einsum("ki,i,i->k", P, w, tphi) / torch.sqrt(torch.diagonal(G))
        nrm = torch.sqrt((w * tphi * tphi).sum())
        rows.append({"arm": arm, "seed": s_, "k": d["k"], "dev_identity": d["dev_identity"],
                     "max_offdiag": d["max_offdiag"], "kappa_gram": d["kappa_gram"],
                     "kappa_gram_times_eps": d["kappa_gram_times_eps"], "eps_machine": d["eps_machine"],
                     "rank_1e12": d["rank_1e12"], "qr_orth_defect": d["qr_orth_defect"],
                     "mode3_leak_rel": float(torch.linalg.vector_norm(leak) / torch.clamp(nrm, min=1e-300))})
    return rows


def _gate_p1(prefix, data_dir: str, n_seeds: int) -> dict:
    """재현 게이트 — 접두부의 모드 1–2 기록이 커밋된 (a′) 기록(I0, gauss)과 같은가(e_λ·MAC·분류의 비트 일치)."""
    from .run_p1_neural import _cell
    p = os.path.join(data_dir, "p1_neural_records.jsonl")
    if not os.path.exists(p):
        return {"checked": False, "why": "p1_neural_records.jsonl 없음"}
    old = [json.loads(x) for x in open(p, encoding="utf-8")]
    old = {(int(r["mode"]), int(r["seed"])): r for r in old
           if r["arm"] == A_PRIME and r.get("ladder") == "I0" and r.get("nodes") == "gauss"}
    same, n = 0, 0
    for m in (1, 2):
        recs, _ = _cell(prefix, m, A_PRIME, "I0", "gauss")
        for r in recs:
            o = old.get((m, int(r["seed"])))
            n += 1
            same += int(o is not None and o["outcome"] == r["outcome"] and o["e_lam"] == r["e_lam"]
                        and o["mac"] == r["mac"])
    return {"checked": True, "n": n, "bit_identical": same, "pass": same == n == 2 * n_seeds}


def p1_rf(outdir=None, n_modes: int = N_MODES_P1, n_seeds: int = N_SEEDS, iters: int = ITERS,
          device=None, quick: bool = False) -> dict:
    """P1-RF fp64 — 고정 접두부 세 갈래. `p1_rf_*`와 접두부 쌍(`p1_mlst_prefix.npz`)을 쓴다."""
    import torch
    from ..neural import deflation as df
    from ..neural import sequential as sq
    from .run_p1_neural import _cell
    d = manifest.ensure_outdir(outdir)
    launch = manifest.launch_provenance()
    if quick:
        n_modes, n_seeds, iters = 4, 2, 12
    pre = p1_prefix(n_seeds, iters, device=device)
    gate = _gate_p1(pre, d, n_seeds) if not quick else {"checked": False, "why": "quick"}
    raw = _pair(pre)
    wq = torch.as_tensor(pre["wq"])
    rf = sq.orthogonalize_pair(raw, wq)
    np.savez_compressed(os.path.join(d, "p1_mlst_prefix.npz"),
                        **{f"raw_phi{k + 1}": raw[k][0].numpy() for k in range(2)},
                        **{f"raw_d2_{k + 1}": raw[k][1].numpy() for k in range(2)},
                        xs=pre["xs"], wq=pre["wq"])
    arms = ((FROZEN["a_prime"], df.ProjectionDiagonal(), raw, "raw", "diagonal"),
            (FROZEN["rf"], df.ProjectionDiagonal(), rf, "orthogonalized", "diagonal"),
            (FROZEN["b"], df.ProjectionExact(), raw, "raw", "exact"))
    rows, recs, span = [], [], []
    for name, defl, pair, storage, rname in arms:
        out = sq.solve_sequential(n_modes, defl, n_seeds=n_seeds, iters=iters, device=device,
                                  stored_arrays=pair, storage=storage)
        for m in range(3, n_modes + 1):
            r, agg = _cell(out, m, name, "I0", "gauss", n_ref=n_modes + 4)
            agg.update({"storage": storage, "frozen_prefix": True})
            recs += r
            rows.append(agg)
        span += _pair_rows(name, pair, pre["wq"], out["raw_shapes"][2], rname)
        print(f"  [P1-RF {name}] 모드3 correct {rows[-(n_modes - 2)]['correct']}/{n_seeds}", flush=True)
    manifest.write_csv(os.path.join(d, "p1_rf.csv"), rows)
    manifest.write_jsonl(os.path.join(d, "p1_rf_records.jsonl"), recs)
    manifest.write_csv(os.path.join(d, "p1_rf_span.csv"), span)
    manifest.write_json(os.path.join(d, "manifest_p1_rf.json"), manifest.build({
        "driver": "run_paperA_mlst.p1_rf", "launch": launch, "quick": bool(quick),
        "n_modes": n_modes, "n_seeds": n_seeds, "iters": iters, "dtype": "float64",
        "prefix": "modes 1-2 solved once per seed on the committed (a') path (network storage, "
                  "diagonal rule); the raw stored pair is shared by the three arms",
        "arms": {FROZEN["a_prime"]: "raw pair + diagonal rule",
                 FROZEN["rf"]: "the same pair orthogonalized at storage time (classical Gram-Schmidt, "
                               "Rowan-style) + diagonal rule; later stages store their deflated state",
                 FROZEN["b"]: "raw pair + exact stored-span projection"},
        "storage": "modes from 3 on are kept as arrays at the fixed Gauss nodes (the same deflation "
                   "as re-evaluating the network there; tested)",
        "reproduction_gate": gate, "prespecification": "2026-10-14-mlst-prespecification.md"}))
    print(f"[p1_rf] 게이트 {gate}", flush=True)
    return {"rows": rows, "gate": gate, "outdir": d}


def p1_rf_fp32(outdir=None, n_modes: int = N_MODES_P1, n_seeds: int = N_SEEDS, iters: int = ITERS,
               device=None, quick: bool = False) -> dict:
    """fp32 RF — 접두부도 fp32(초기화가 fp64와 다르다). `p1_rf_fp32_*`를 쓴다."""
    import torch
    from ..neural import deflation as df
    from ..neural import sequential as sq
    from .run_p1_neural import _cell
    d = manifest.ensure_outdir(outdir)
    launch = manifest.launch_provenance()
    if quick:
        n_modes, n_seeds, iters = 4, 2, 12
    pre = p1_prefix(n_seeds, iters, dtype=torch.float32, device=device)
    raw = _pair(pre, torch.float32)
    rf = sq.orthogonalize_pair(raw, torch.as_tensor(pre["wq"]).to(torch.float32))
    out = sq.solve_sequential(n_modes, df.ProjectionDiagonal(), n_seeds=n_seeds, iters=iters,
                              dtype=torch.float32, device=device, stored_arrays=rf,
                              storage="orthogonalized")
    rows, recs = [], []
    for m in range(1, n_modes + 1):
        src = pre if m <= 2 else out
        r, agg = _cell(src, m, "rf_fp32", "I0", "gauss", n_ref=n_modes + 4)
        agg.update({"dtype": "float32", "storage": "orthogonalized" if m > 2 else "network",
                    "frozen_prefix": m <= 2})
        recs += r
        rows.append(agg)
    span = [{"stage_mode": m, **sd} for m, sd in enumerate(out["span_diagnostics"], start=1) if sd]
    manifest.write_csv(os.path.join(d, "p1_rf_fp32.csv"), rows)
    manifest.write_jsonl(os.path.join(d, "p1_rf_fp32_records.jsonl"), recs)
    manifest.write_csv(os.path.join(d, "p1_rf_fp32_span.csv"), span)
    manifest.write_json(os.path.join(d, "manifest_p1_rf_fp32.json"), manifest.build({
        "driver": "run_paperA_mlst.p1_rf_fp32", "launch": launch, "quick": bool(quick),
        "n_modes": n_modes, "n_seeds": n_seeds, "iters": iters, "dtype": "float32",
        "prefix": "modes 1-2 in float32 on the network-storage path with the diagonal rule",
        "arm": "storage-time orthogonalization (classical Gram-Schmidt of the deflated state, "
               "float32) + diagonal rule from mode 3 on",
        "compare": "committed float32 cells p1_fp32_stress.csv (b) and (b') at the same seeds; the "
                   "float32 initial weights are paired across arms, the prefixes are not bitwise equal",
        "prespecification": "2026-10-14-mlst-prespecification.md"}))
    print(f"[p1_rf_fp32] 모드6 correct {rows[5]['correct'] if len(rows) > 5 else '-'}", flush=True)
    return {"rows": rows, "outdir": d}


def _orthonormal_pair(pair, wq):
    """저장 쌍을 질량 정규직교 기저 (q₁, q₂)로 — 고전 GS 뒤 정규화. φ와 φ″에 같은 계수."""
    import torch
    from ..neural import sequential as sq
    rf = sq.orthogonalize_pair(pair, wq)
    out = []
    for ph, d2 in rf:
        n = torch.sqrt((wq * ph * ph).sum(-1))[..., None]
        out.append((ph / n, d2 / n))
    return out


def p1_basis(outdir=None, alphas=ALPHAS, alphas_operator=ALPHAS_OPERATOR, n_seeds: int = N_SEEDS,
             iters: int = ITERS, device=None, quick: bool = False) -> dict:
    """기저 불변 시험 — `p1_mlst_prefix.npz`(P1-RF가 쓴 접두부)에서 같은 부분공간의 기저만 바꾼다."""
    import torch
    from ..neural import core
    from ..neural import deflation as df
    from ..neural import sequential as sq
    from ..neural.net import EnsembleMLP
    from .run_p1_neural import _cell
    d = manifest.ensure_outdir(outdir)
    launch = manifest.launch_provenance()
    if quick:
        n_seeds, iters, alphas = 2, 12, alphas[:2]
    z = np.load(os.path.join(d, "p1_mlst_prefix.npz"))
    pair = [(torch.as_tensor(z[f"raw_phi{k}"]), torch.as_tensor(z[f"raw_d2_{k}"])) for k in (1, 2)]
    if int(pair[0][0].shape[0]) != n_seeds:
        raise ValueError(f"접두부의 시드 수 {pair[0][0].shape[0]} ≠ {n_seeds}")
    wq = torch.as_tensor(z["wq"])
    Q = _orthonormal_pair(pair, wq)
    Qp, Qd = torch.stack([Q[0][0], Q[1][0]]), torch.stack([Q[0][1], Q[1][1]])
    # 연산자 수준 — 학습 없음. 후보는 모드 3 단계의 초기 망(seed·1000 + 2)이다.
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ens = EnsembleMLP(n_seeds, seed=2, device=dev)
    with torch.no_grad():
        cand = core.eval_phi(ens, torch.as_tensor(z["xs"], device=dev))[0].cpu()
    op_rows = []
    for a in alphas_operator:
        Vp = sq.mix_basis(Qp, a)
        for s_ in range(n_seeds):
            V = Vp[:, s_]
            phi = cand[s_]
            c = df.project_diagonal(phi, V, wq)
            tphi = phi - c @ V
            pe = phi - df.project_exact(phi, V, wq) @ V
            nrm = lambda v: float(torch.sqrt((wq * v * v).sum()))
            op_rows.append({"alpha": a, "seed": s_, "r": float((wq * V[0] * V[1]).sum()),
                            "op_gap": sq.diagonal_operator_gap(V, wq),
                            "leak_init_rel": float(torch.linalg.vector_norm(
                                torch.einsum("ki,i,i->k", V, wq, tphi)) / max(nrm(tphi), 1e-300)),
                            "diag_vs_exact_rel": nrm(tphi - pe) / max(nrm(pe), 1e-300)})
    # 결과 수준 — 대각 규칙으로 모드 3만 푼다(초기화는 순차 풀이와 같다).
    rows, recs = [], []
    for a in alphas:
        Vp, Vd = sq.mix_basis(Qp, a), sq.mix_basis(Qd, a)
        out = sq.solve_sequential(3, df.ProjectionDiagonal(), n_seeds=n_seeds, iters=iters,
                                  device=device, stored_arrays=[(Vp[0], Vd[0]), (Vp[1], Vd[1])])
        r, agg = _cell(out, 3, f"basis_alpha={a:g}", "I0", "gauss")
        agg.update({"alpha": a, "r": a / math.sqrt(1 + a * a)})
        for x in r:
            x["alpha"] = a
        recs += r
        rows.append(agg)
        print(f"  [basis α={a:g}] 모드3 correct {agg['correct']}/{n_seeds} basin "
              f"{agg['lower_mode_basin']}", flush=True)
    manifest.write_csv(os.path.join(d, "p1_basis_invariance.csv"), rows)
    manifest.write_jsonl(os.path.join(d, "p1_basis_invariance_records.jsonl"), recs)
    manifest.write_csv(os.path.join(d, "p1_basis_operator.csv"), op_rows)
    manifest.write_json(os.path.join(d, "manifest_p1_basis_invariance.json"), manifest.build({
        "driver": "run_paperA_mlst.p1_basis", "launch": launch, "quick": bool(quick),
        "n_seeds": n_seeds, "iters": iters, "alphas": list(alphas),
        "alphas_operator": list(alphas_operator),
        "basis": "q1, q2 = mass-orthonormal basis of the frozen prefix pair (classical Gram-Schmidt); "
                 "v1 = q1, v2 = (q2 + alpha q1)/sqrt(1 + alpha^2), the same linear combination on phi''",
        "subspace": "unchanged for every alpha, so the exact projection is basis-invariant; alpha = 0 is "
                    "the rf_frozen arm and alpha about 7.4 (r = 0.991) the a_prime_frozen arm",
        "candidate": "operator level: the untrained mode-3 networks (seed 1000*0 + 2)",
        "prespecification": "2026-10-14-mlst-prespecification.md"}))
    return {"rows": rows, "op_rows": op_rows, "outdir": d}


# ── 재검토 후속(2026-10-15-mlst-followup-prespecification.md) ───────────────────────────────────
# (1) fp32 고정 스팬 직접 시험 — 같은 저장 집합·같은 후보에 같은 사영자의 세 실현을 fp32로 적용한다. 세 학습
#     이력이 달라서 생기는 차이와 산술만의 차이를 가른다. (2) 정확 사영의 기저 불변을 결과 수준에서 — 섞은 기저로
#     모드 3을 정확 사영으로 학습한다.

ROUTES = ("gram", "cgs", "mgs")
ALPHAS_EXACT = (0.01, 0.25)
FIXED_SPAN_STAGE = 6          # 판정 단계 — 모드 6에 넘기는 저장 집합(k = 5)


def fixed_span_routes(Phi, Phi2, phi, phi2, wq) -> dict:
    """같은 원시 저장 집합 Φ(k×n)와 같은 후보 φ에 세 실현을 **입력 정밀도로** 적용하고 fp64 기준과 비교한다.

    gram = 정규방정식 c = G⁻¹ΦᵀWφ(arm (b)), cgs = 저장 시 고전 Gram–Schmidt(질량 내적, 정규화 없음) 뒤 대각 규칙(RF),
    mgs = 가중 modified Gram–Schmidt(√ε 폐기, arm (b′)). 기준은 fp64 SVD 최소제곱(√W Φᵀ c ≈ √W φ) — 조건수가 커지면
    fp64 정규방정식도 기준이 못 된다. 곡률에는 같은 계수를 쓴다. 측정은 모두 fp64에서 한다:
    rel_err = ‖φ̃ − φ̃_ref‖_W/‖φ̃_ref‖_W, leak = ‖P φ̃‖_W/‖φ̃‖_W(사영 노름 — 기저에 불변, ≤ 1),
    rq_rel_err = |R(φ̃) − R(φ̃_ref)|/R(φ̃_ref)."""
    import torch
    from ..neural import deflation as df
    from ..neural import sequential as sq
    with torch.no_grad():
        try:
            c = df.project_exact(phi, Phi, wq)
            g, g2 = phi - c @ Phi, phi2 - c @ Phi2
        except RuntimeError:                       # 수치적으로 특이한 그람
            g = g2 = torch.full_like(phi, float("nan"))
        U = sq.orthogonalize_pair([(Phi[j], Phi2[j]) for j in range(Phi.shape[0])], wq)
        Uh, Uh2 = torch.stack([u for u, _ in U]), torch.stack([u2 for _, u2 in U])
        cd = df.project_diagonal(phi, Uh, wq)
        h, h2 = phi - cd @ Uh, phi2 - cd @ Uh2
        Q, Q2, kept = df.w_orthonormalize(Phi, Phi2, wq)
        a = torch.einsum("ki,i,i->k", Q, wq, phi)
        m, m2 = phi - a @ Q, phi2 - a @ Q2
    P, P2, f, f2, w = (t.detach().cpu().double() for t in (Phi, Phi2, phi, phi2, wq))
    sw = torch.sqrt(w)
    A = sw[:, None] * P.T

    def coef(v):
        return torch.linalg.lstsq(A, sw * v, driver="gelsd").solution

    def nrm(v):
        return float(torch.sqrt(torch.clamp((w * v * v).sum(), min=0.0)))

    def rq(v, v2):
        return float((w * v2 * v2).sum() / (w * v * v).sum())

    cr = coef(f)
    ref, ref2 = f - cr @ P, f2 - cr @ P2
    r_ref = rq(ref, ref2)
    out = {}
    for name, (v, v2) in (("gram", (g, g2)), ("cgs", (h, h2)), ("mgs", (m, m2))):
        v, v2 = v.detach().cpu().double(), v2.detach().cpu().double()
        if not bool(torch.isfinite(v).all()):
            out[name] = {"rel_err": float("inf"), "leak": float("nan"), "rq_rel_err": float("inf")}
            continue
        out[name] = {"rel_err": nrm(v - ref) / max(nrm(ref), 1e-300),
                     "leak": nrm(coef(v) @ P) / max(nrm(v), 1e-300),
                     "rq_rel_err": abs(rq(v, v2) - r_ref) / r_ref}
    U64 = Uh.detach().cpu().double()
    Gu = (U64 * w) @ U64.T
    du = torch.sqrt(torch.diagonal(Gu))
    C = Gu / (du[:, None] * du[None, :])
    out["cgs"]["dev_identity"] = float(torch.linalg.matrix_norm(torch.eye(C.shape[0], dtype=C.dtype) - C, ord=2))
    Qk = Q.detach().cpu().double()[kept.detach().cpu().bool()]
    Gq = (Qk * w) @ Qk.T
    out["mgs"]["orth_defect"] = (float(torch.linalg.matrix_norm(torch.eye(Gq.shape[0], dtype=Gq.dtype) - Gq, ord=2))
                                 if Gq.numel() else float("nan"))
    out["mgs"]["dropped"] = int((kept.detach().cpu() == 0).sum())
    ev = torch.linalg.eigvalsh((P * w) @ P.T)
    out["kappa_gram"] = float(ev[-1] / ev[0]) if float(ev[0]) > 0 else float("inf")
    out["eps_machine"] = float(torch.finfo(Phi.dtype).eps)
    return out


def p1_fp32_fixed_span(outdir=None, n_modes: int = N_MODES_P1, n_seeds: int = N_SEEDS, iters: int = ITERS,
                       device=None, quick: bool = False) -> dict:
    """arm (b) fp32를 커밋된 구성(망 저장, 50시드, 4000회, 가우스 256) 그대로 다시 돌려 단계마다 원시 출력 배열을
    남기고, 각 단계의 (저장 집합, 후보)에 `fixed_span_routes`를 적용한다. 다시 돌린 실행의 판정은 기록만 하고(GPU
    비결정성 — 커밋된 fp32 셀과 비트 일치를 요구하지 않는다) 신뢰도 표에는 쓰지 않는다."""
    import torch
    from ..neural import deflation as df
    from ..neural import sequential as sq
    from .run_p1_neural import _cell
    d = manifest.ensure_outdir(outdir)
    launch = manifest.launch_provenance()
    if quick:
        n_modes, n_seeds, iters = 4, 2, 12
    out = sq.solve_sequential(n_modes, df.ProjectionExact(), n_seeds=n_seeds, iters=iters,
                              dtype=torch.float32, device=device)
    np.savez_compressed(os.path.join(d, "p1_fp32_raw_sets.npz"),
                        raw_shapes=np.asarray(out["raw_shapes"]), raw_d2=np.asarray(out["raw_d2"]),
                        shapes=np.asarray(out["shapes"]), lam=np.asarray(out["lam"]),
                        xs=np.asarray(out["xs"]), wq=np.asarray(out["wq"]),
                        converged=np.asarray(out["converged"]))
    recs, cells = [], []
    for m in range(1, n_modes + 1):
        r, agg = _cell(out, m, "b_projection_exact_fp32_rerun", "I0", "gauss", n_ref=n_modes + 4)
        agg["dtype"] = "float32"
        recs += r
        cells.append(agg)
    wq = torch.as_tensor(np.asarray(out["wq"]), dtype=torch.float32)
    RS = torch.as_tensor(np.asarray(out["raw_shapes"]), dtype=torch.float32)       # (모드, 시드, 절점)
    RD = torch.as_tensor(np.asarray(out["raw_d2"]), dtype=torch.float32)
    rows = []
    for m in range(2, n_modes + 1):
        for s_ in range(n_seeds):
            res = fixed_span_routes(RS[:m - 1, s_], RD[:m - 1, s_], RS[m - 1, s_], RD[m - 1, s_], wq)
            rows.append({"stage_mode": m, "seed": s_, "k": m - 1, "dtype": "float32",
                         "kappa_gram": res["kappa_gram"],
                         "kappa_gram_times_eps": res["kappa_gram"] * res["eps_machine"],
                         **{f"err_{k}": res[k]["rel_err"] for k in ROUTES},
                         **{f"leak_{k}": res[k]["leak"] for k in ROUTES},
                         **{f"rq_err_{k}": res[k]["rq_rel_err"] for k in ROUTES},
                         "cgs_dev_identity": res["cgs"]["dev_identity"],
                         "mgs_orth_defect": res["mgs"]["orth_defect"], "mgs_dropped": res["mgs"]["dropped"]})
    manifest.write_csv(os.path.join(d, "p1_fp32_fixed_span.csv"), rows)
    manifest.write_csv(os.path.join(d, "p1_fp32_rerun.csv"), cells)
    manifest.write_jsonl(os.path.join(d, "p1_fp32_rerun_records.jsonl"), recs)
    manifest.write_json(os.path.join(d, "manifest_p1_fp32_fixed_span.json"), manifest.build({
        "driver": "run_paperA_mlst.p1_fp32_fixed_span", "launch": launch, "quick": bool(quick),
        "n_modes": n_modes, "n_seeds": n_seeds, "iters": iters, "dtype": "float32",
        "rerun": "arm (b) float32 on the committed configuration (network storage, Gauss 256); raw outputs of "
                 "every stage saved; its classification is recorded, not used in the reliability tables",
        "routes": "gram = normal equations; cgs = classical Gram-Schmidt of the raw stored set at storage time "
                  "+ diagonal rule; mgs = weighted modified Gram-Schmidt with sqrt(eps) discard",
        "reference": "float64 SVD least squares on the same float32 arrays",
        "prespecification": "2026-10-15-mlst-followup-prespecification.md"}))
    return {"rows": rows, "cells": cells, "outdir": d}


def p1_basis_exact(outdir=None, alphas=ALPHAS_EXACT, n_seeds: int = N_SEEDS, iters: int = ITERS,
                   device=None, quick: bool = False) -> dict:
    """정확 사영의 기저 불변을 결과 수준에서 — `p1_basis`와 같은 섞은 기저로 모드 3을 정확 사영(arm (b))으로 푼다.
    연산자 수준 검사: 같은 후보를 섞은 기저와 원시 쌍에 정확 사영한 장의 상대 차이."""
    import torch
    from ..neural import core
    from ..neural import deflation as df
    from ..neural import sequential as sq
    from ..neural.net import EnsembleMLP
    from .run_p1_neural import _cell
    d = manifest.ensure_outdir(outdir)
    launch = manifest.launch_provenance()
    if quick:
        n_seeds, iters, alphas = 2, 12, alphas[:1]
    z = np.load(os.path.join(d, "p1_mlst_prefix.npz"))
    pair = [(torch.as_tensor(z[f"raw_phi{k}"]), torch.as_tensor(z[f"raw_d2_{k}"])) for k in (1, 2)]
    if int(pair[0][0].shape[0]) != n_seeds:
        raise ValueError(f"접두부의 시드 수 {pair[0][0].shape[0]} ≠ {n_seeds}")
    wq = torch.as_tensor(z["wq"])
    Q = _orthonormal_pair(pair, wq)
    Qp, Qd = torch.stack([Q[0][0], Q[1][0]]), torch.stack([Q[0][1], Q[1][1]])
    Rp = torch.stack([pair[0][0], pair[1][0]])
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ens = EnsembleMLP(n_seeds, seed=2, device=dev)
    with torch.no_grad():
        cand = core.eval_phi(ens, torch.as_tensor(z["xs"], device=dev))[0].cpu()
    rows, recs = [], []
    for a in alphas:
        Vp, Vd = sq.mix_basis(Qp, a), sq.mix_basis(Qd, a)
        nrm = lambda v: float(torch.sqrt((wq * v * v).sum()))      # noqa: E731
        op = []
        for s_ in range(n_seeds):
            phi = cand[s_]
            pm = phi - df.project_exact(phi, Vp[:, s_], wq) @ Vp[:, s_]
            pr = phi - df.project_exact(phi, Rp[:, s_], wq) @ Rp[:, s_]
            op.append(nrm(pm - pr) / max(nrm(pr), 1e-300))
        out = sq.solve_sequential(3, df.ProjectionExact(), n_seeds=n_seeds, iters=iters,
                                  device=device, stored_arrays=[(Vp[0], Vd[0]), (Vp[1], Vd[1])])
        r, agg = _cell(out, 3, f"basis_exact_alpha={a:g}", "I0", "gauss")
        agg.update({"alpha": a, "r": a / math.sqrt(1 + a * a), "op_exact_vs_raw_rel": max(op)})
        for x in r:
            x["alpha"] = a
        recs += r
        rows.append(agg)
        print(f"  [basis exact α={a:g}] 모드3 correct {agg['correct']}/{n_seeds}", flush=True)
    manifest.write_csv(os.path.join(d, "p1_basis_exact.csv"), rows)
    manifest.write_jsonl(os.path.join(d, "p1_basis_exact_records.jsonl"), recs)
    manifest.write_json(os.path.join(d, "manifest_p1_basis_exact.json"), manifest.build({
        "driver": "run_paperA_mlst.p1_basis_exact", "launch": launch, "quick": bool(quick),
        "n_seeds": n_seeds, "iters": iters, "alphas": list(alphas),
        "basis": "as p1_basis: v1 = q1, v2 = (q2 + alpha q1)/sqrt(1 + alpha^2) from the frozen prefix pair",
        "rule": "exact M-orthogonal projection (arm (b)); compare with b_frozen (raw pair, exact) at mode 3",
        "prespecification": "2026-10-15-mlst-followup-prespecification.md"}))
    return {"rows": rows, "outdir": d}


def _boot_median(x, B: int = B_BOOT, seed: int = BOOT_SEED):
    """시드를 다시 뽑는 부트스트랩의 중앙값 95 % 백분위 구간."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, x.size, size=(B, x.size))], axis=1)
    return float(np.median(x)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def followup_stats_from(d: str, B: int = B_BOOT, seed: int = BOOT_SEED) -> dict:
    """후속 사전규정의 판정량 — 있는 것만 계산한다."""
    res = {"fixed_span": None, "basis_exact": None}
    fs = os.path.join(d, "p1_fp32_fixed_span.csv")
    if os.path.exists(fs):
        rows = [r for r in manifest.read_csv(fs)]
        st = {}
        for m in sorted({int(r["stage_mode"]) for r in rows}):
            rr = sorted((r for r in rows if int(r["stage_mode"]) == m), key=lambda r: int(r["seed"]))
            lg = {k: np.log10(np.clip([float(r[f"err_{k}"]) for r in rr], 1e-300, np.inf)) for k in ROUTES}
            gc, cm = _boot_median(lg["gram"] - lg["cgs"], B, seed), _boot_median(lg["cgs"] - lg["mgs"], B, seed)
            st[m] = {"n_seeds": len(rr),
                     "median_err": {k: float(np.median([float(r[f"err_{k}"]) for r in rr])) for k in ROUTES},
                     "median_leak": {k: float(np.nanmedian([float(r[f"leak_{k}"]) for r in rr])) for k in ROUTES},
                     "median_cgs_dev_identity": float(np.median([float(r["cgs_dev_identity"]) for r in rr])),
                     "median_mgs_orth_defect": float(np.nanmedian([float(r["mgs_orth_defect"]) for r in rr])),
                     "median_kappa_gram_times_eps": float(np.median([float(r["kappa_gram_times_eps"]) for r in rr])),
                     "log10_gram_minus_cgs": {"median_ci": gc, "verdict": rule(gc)},
                     "log10_cgs_minus_mgs": {"median_ci": cm, "verdict": rule(cm)}}
        res["fixed_span"] = {"by_stage": st, "decision_stage": FIXED_SPAN_STAGE}
    be = os.path.join(d, "p1_basis_exact_records.jsonl")
    rf = os.path.join(d, "p1_rf_records.jsonl")
    if os.path.exists(be) and os.path.exists(rf):
        recs, base = _load(d, "p1_basis_exact_records.jsonl"), _load(d, "p1_rf_records.jsonl")
        n = len({int(r["seed"]) for r in base if r["arm"] == FROZEN["b"] and int(r["mode"]) == 3})
        b0 = _indicator([r for r in base if r["arm"] == FROZEN["b"]], 3, n_seeds=n)
        lam0 = {int(r["seed"]): float(r["e_lam"]) for r in base if r["arm"] == FROZEN["b"] and int(r["mode"]) == 3}
        out = {}
        for a in sorted({float(r["alpha"]) for r in recs}):
            ra = [r for r in recs if float(r["alpha"]) == a]
            ia = _indicator(ra, 3, n_seeds=n)
            ci = paired_bootstrap(ia, b0, B=B, seed=seed)
            de = [abs(float(r["e_lam"]) - lam0[int(r["seed"])]) for r in ra]
            out[f"{a:g}"] = {"correct": int(ia.sum()), "b_frozen_correct": int(b0.sum()),
                             "contrast_vs_b_frozen": {"ci": ci, "verdict": rule(ci)},
                             "max_abs_e_lam_difference": float(max(de))}
        res["basis_exact"] = out
    return res


def followup_stats(outdir=None, B: int = B_BOOT, seed: int = BOOT_SEED) -> dict:
    """후속 판정 — `paperA_mlst_followup_stats.json`을 쓴다(원래 판정 파일은 건드리지 않는다)."""
    d = manifest.ensure_outdir(outdir)
    res = followup_stats_from(d, B=B, seed=seed)
    res["prespecification"] = "2026-10-15-mlst-followup-prespecification.md"
    res["bootstrap"] = {"B": B, "seed": seed}
    manifest.write_json(os.path.join(d, "paperA_mlst_followup_stats.json"), res)
    return res


# ── P4 — 원시 저장 격자(P4-RAW, P4-SPAN)와 코너 계기(P4-ADF) ────────────────────────────────

def _gate_p4(arm: str, out, data_dir: str, modes=(1, 2)) -> dict:
    """재현 게이트 — `modes`의 Λ가 커밋된 같은 규칙의 기록과 비트 단위로 같은가. 원시 저장은 모드 1–2만(모드 3부터
    저장 규약이 갈린다), 사영 저장 재실행은 커밋된 경로 그대로라 모드 1–3 전부를 본다."""
    f = "p4_neural_records.jsonl" if arm == "exact" else "p4_neural_arms_records.jsonl"
    p = os.path.join(data_dir, f)
    if not os.path.exists(p):
        return {"checked": False, "why": f"{f} 없음"}
    from ..neural.p4_neural import ARM_NAME
    old = {(int(r["mode"]), int(r["seed"])): float(r["Lam"]) for r in map(json.loads, open(p, encoding="utf-8"))
           if r["arm"] == ARM_NAME[arm]}
    by_mode = {}
    for m in modes:
        by_mode[m] = sum(int(old.get((m, s_)) == float(out["lam"][m - 1][s_])) for s_ in range(int(out["n_seeds"])))
    n, same = len(modes) * int(out["n_seeds"]), sum(by_mode.values())
    return {"checked": True, "n": n, "bit_identical": same, "pass": same == n, "by_mode": by_mode}


def _p4_storage_run(storage: str, stem: str, meta: dict, gate_modes, outdir, arms, n_seeds: int, iters: int,
                    device, quick: bool) -> dict:
    """P4 세 규칙 × 모드 1–3을 한 저장 규약으로 돌려 기록·시드별 스팬 진단·망(`p4_params_<storage>.npz`, 저장소
    밖)을 남긴다 — `p4_raw`와 `p4_projected_rerun`의 공용 본체."""
    from ..neural.p4_neural import solve_p4_neural
    from .run_p4_neural import _reference, _score
    d = manifest.ensure_outdir(outdir)
    launch = manifest.launch_provenance()
    kw = dict(n_per_block=20, width=64, depth=4)
    n_ref, beta = 8, 3.0
    if quick:
        n_seeds, iters, kw, n_ref = 2, 6, dict(n_per_block=6, width=16, depth=2), 4
    rows, recs, span, gates, params = [], [], [], {}, {}
    for arm in arms:
        out = solve_p4_neural(n_modes=3, n_seeds=n_seeds, iters=iters, projection=arm, device=device,
                              storage=storage, return_params=True, per_seed_span=True, **kw)
        ext, lam_ref, shapes = _reference(d, n_ref, beta, 3, out["pts"])
        r0 = len(rows)
        _score(out, rows, recs, 3, n_seeds, iters, ext, lam_ref, shapes, out["w"])
        for r in rows[r0:]:
            r["storage"] = storage
        for r in recs:
            r.setdefault("storage", storage)
        span += [{"arm": arm, **x} for x in out["span_per_seed"]]
        gates[arm] = (_gate_p4(arm, out, d, modes=gate_modes) if not quick
                      else {"checked": False, "why": "quick"})
        for m, pm in enumerate(out["params"], start=1):
            for k_, v in pm.items():
                params[f"{arm}_m{m}_{k_}"] = v
        print(f"  [P4 {storage} {arm}] 게이트 {gates[arm]}", flush=True)
    np.savez_compressed(os.path.join(d, f"p4_params_{storage}.npz"), **params)
    manifest.write_csv(os.path.join(d, f"{stem}.csv"), rows)
    manifest.write_jsonl(os.path.join(d, f"{stem}_records.jsonl"), recs)
    manifest.write_csv(os.path.join(d, f"{stem}_span_per_seed.csv"), span)
    manifest.write_json(os.path.join(d, f"manifest_{stem}.json"), manifest.build({
        "driver": f"run_paperA_mlst.{stem}", "launch": launch, "quick": bool(quick),
        "arms": list(arms), "n_modes": 3, "n_seeds": n_seeds, "iters": iters, **kw,
        "n_ref": n_ref, "beta": beta, "storage": meta["storage"],
        "jitter": "the exact path adds 1e-14 times the identity to the Gram (absolute), as committed",
        "reproduction_gate": gates, "params": f"p4_params_{storage}.npz (not in the repository)",
        **{k: v for k, v in meta.items() if k != "storage"}}))
    return {"rows": rows, "gates": gates, "outdir": d}


def p4_raw(outdir=None, arms=P4_ARMS, n_seeds: int = N_SEEDS, iters: int = ITERS, device=None,
           quick: bool = False) -> dict:
    """P4-RAW — 원시 저장의 세 규칙 × 모드 1–3. 망은 `p4_params_raw.npz`(저장소 밖)에 남긴다."""
    return _p4_storage_run("raw", "p4_raw", {
        "storage": "raw: the pre-projection state of each stage is stored, as on P1; the committed P4 runs "
                   "stored the projected state",
        "prespecification": "2026-10-14-mlst-prespecification.md"}, (1, 2), outdir, arms, n_seeds, iters,
        device, quick)


PRESPEC_P4_PROJECTED = "2026-10-16-mlst-p4-projected-prespecification.md"


def p4_projected_rerun(outdir=None, arms=P4_ARMS, n_seeds: int = N_SEEDS, iters: int = ITERS, device=None,
                       quick: bool = False) -> dict:
    """사영 저장 재실행 — 커밋된 P4 경로(사영된 상태를 저장)를 그대로 다시 돌려 망을 `p4_params_projected.npz`에
    남긴다(사전규정 보완 2). 게이트는 모드 1–3 전부의 비트 일치 — 통과하면 남긴 망이 커밋된 셀의 망이다."""
    return _p4_storage_run("projected", "p4_projected_rerun", {
        "storage": "projected: the projected state of each stage is stored (the committed P4 path), rerun to "
                   "keep the networks",
        "reproduction_gate_modes": [1, 2, 3],
        "prespecification": PRESPEC_P4_PROJECTED}, (1, 2, 3), outdir, arms, n_seeds, iters, device, quick)


def _p4_fem_modes(n_ref: int = 8, beta: float = 3.0, n_modes: int = 3):
    from ..problems import p4_lshape as p4
    res = p4.solve(n_ref, beta=beta, n_modes=max(n_modes + 3, 6))
    mesh, free = res["mesh"], res["free_dofs"]
    order = np.argsort(res["Lam"])
    fulls = []
    for j in order[:n_modes]:
        f = np.zeros(2 * len(mesh["nodes"]))
        f[free] = res["vec"][:, j]
        fulls.append(f)
    return mesh, fulls, np.sort(res["Lam"])[:n_modes]


def _rq_plane_stress(U, Gr, w, nu: float = 0.29) -> float:
    """U (n, 2), Gr (n, 2, 2) = [성분, d/dx·d/dy] → Ω²(`p4_neural._rq_from_state`와 같은 평면응력 식)."""
    ux, uy, vx, vy = Gr[:, 0, 0], Gr[:, 0, 1], Gr[:, 1, 0], Gr[:, 1, 1]
    c = 1.0 / (1.0 - nu ** 2)
    num = (w * c * (ux ** 2 + vy ** 2 + 2.0 * nu * ux * vy + 0.5 * (1.0 - nu) * (uy + vx) ** 2)).sum()
    return float(num / (w * (U[:, 0] ** 2 + U[:, 1] ** 2)).sum())


def _p4_ensemble(z, arm: str, m: int, width: int, depth: int, dev):
    """`p4_params_*.npz`에 남긴 (규칙, 모드)의 망을 앙상블로 되살린다."""
    import torch
    from ..neural import p4_neural as pn
    e = pn.EnsembleVecMLP(int(z[f"{arm}_m{m}_net.0.weight"].shape[0]), width, depth, seed=0, device=dev)
    for k_ in list(e.params):
        e.params[k_] = torch.as_tensor(z[f"{arm}_m{m}_{k_}"], device=dev)
    return e


def _p4_states(e, pts, chunk: int):
    """앙상블의 상태 (시드, 6, n_q) = 변위 두 성분과 네 도함수."""
    import torch
    from torch.func import vmap
    from ..neural import p4_neural as pn
    return torch.cat([vmap(lambda p, c=c: pn._state_one(p, e.base, c))(e.params)
                      for c in torch.split(pts, chunk)], dim=-1)


def _p4_mass(A, w):
    """(…, 6, n) → (…, 2n): √w를 곱한 변위 두 성분 — 유클리드 내적이 질량 내적이 된다."""
    import torch
    sw = torch.sqrt(w)
    return torch.cat([A[..., 0, :] * sw, A[..., 1, :] * sw], dim=-1)


def _relevel_neural(d: str, storage: str, stem: str, quads: dict, dev, width: int, depth: int,
                    chunk: int) -> list:
    """`p4_params_<storage>.npz`의 망을 각 구적에서 다시 재고 그 실행의 저장 규약으로 다시 사영한다 — 솔버와 같게,
    원시 규약은 앞 모드들의 원시 상태에, 사영 규약은 그 구적에서 사영된 앞 모드들의 상태에 사영한다."""
    import torch
    from torch.func import vmap
    from ..neural import p4_neural as pn
    z = np.load(os.path.join(d, f"p4_params_{storage}.npz"))
    oc = {(r["arm"], int(r["mode"]), int(r["seed"])): r["outcome"] for r in _load(d, f"{stem}_records.jsonl")}
    nu = 0.29
    nn_rows = []
    for arm in P4_ARMS:
        keys = [k for k in z.files if k.startswith(f"{arm}_m")]
        if not keys:
            continue
        ens = [_p4_ensemble(z, arm, m, width, depth, dev) for m in (1, 2, 3)]
        proj = pn.PROJECTIONS[arm]
        for L, (pts, w) in quads.items():
            pts_d, w_d = pts.to(dev), w.to(dev)
            with torch.no_grad():
                raw = [_p4_states(e, pts_d, chunk) for e in ens]
                stored = []
                for m in (1, 2, 3):
                    S = raw[m - 1]
                    if m > 1:
                        S = vmap(lambda a, *b: proj(a, torch.stack(b, 0), w_d))(S, *stored)
                    stored.append(raw[m - 1] if storage == "raw" else S)
                    lam = vmap(lambda a: pn._rq_from_state(a, w_d, nu))(S).cpu().numpy()
                    for s_ in range(len(lam)):
                        nn_rows.append({"arm": arm, "mode": m, "seed": s_, "level": L,
                                        "n_pts": int(len(w)), "Lam": float(lam[s_]),
                                        "outcome": oc.get((pn.ARM_NAME[arm], m, s_), ""), "storage": storage})
    return nn_rows


def p4_adf(outdir=None, levels=ADF_LEVELS, n_per_cell: int = ADF_N_PER_CELL, device=None,
           quick: bool = False, chunk: int = 20_000) -> dict:
    """P4-ADF — ADF 코너 단면, 등급 구적 단계별 참조 FEM 모드와 신경 셀(P4-RAW의 망)의 Λ."""
    import torch
    from ..neural import p4_neural as pn
    from ..problems import p4_lshape as p4
    d = manifest.ensure_outdir(outdir)
    launch = manifest.launch_provenance()
    n_ref, width, depth = 8, 64, 4
    if quick:
        levels, n_per_cell, n_ref, width, depth = (0, 2), 6, 4, 16, 2     # p4_raw(quick)와 같은 격자·망
    corner = pn.adf_corner_profile()
    mesh, fulls, lam_h = _p4_fem_modes(n_ref)
    ref_rows = []
    quads = {L: pn.quadrature_graded(n_per_cell, L, device="cpu") for L in levels}
    for m, f in enumerate(fulls, start=1):
        for L, (pts, w) in quads.items():
            P, W = pts.numpy(), w.numpy()
            U, Gr = p4.eval_mode_at(mesh, f, P), p4.eval_mode_grad_at(mesh, f, P)
            ref_rows.append({"mode": m, "level": L, "n_pts": len(W), "Lam": _rq_plane_stress(U, Gr, W),
                             "Lam_fem": float(lam_h[m - 1])})
    # 신경 셀 — 원시 저장 규약 그대로 각 구적에서 다시 사영한다.
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nn_rows = _relevel_neural(d, "raw", "p4_raw", quads, dev, width, depth, chunk)
    manifest.write_csv(os.path.join(d, "p4_adf_corner.csv"), corner)
    manifest.write_csv(os.path.join(d, "p4_adf_reference.csv"), ref_rows)
    manifest.write_csv(os.path.join(d, "p4_adf_neural.csv"), nn_rows)
    manifest.write_json(os.path.join(d, "manifest_p4_adf.json"), manifest.build({
        "driver": "run_paperA_mlst.p4_adf", "launch": launch, "quick": bool(quick),
        "levels": list(levels), "n_per_cell": n_per_cell, "ratio": 0.5,
        "quadrature": "three unit blocks, each axis cut toward the re-entrant corner (1, 1) at "
                      "L*0.5^j (j = 1..level), n_per_cell^2 Gauss points per cell; level 0 is the "
                      "training quadrature",
        "reference": f"graded Q2 FEM (n = {n_ref}, beta = 3) modes 1-3, displacement and gradient "
                     "evaluated at the quadrature points; Lam_fem is the FEM eigenvalue",
        "neural": "p4_params_raw.npz networks, re-projected on each quadrature with the raw-storage "
                  "convention of their own arm",
        "prespecification": "2026-10-14-mlst-prespecification.md"}))
    return {"corner": corner, "ref_rows": ref_rows, "nn_rows": nn_rows, "outdir": d}


def p4_adf_projected(outdir=None, levels=ADF_LEVELS, n_per_cell: int = ADF_N_PER_CELL, device=None,
                     quick: bool = False, chunk: int = 20_000) -> dict:
    """사영 저장 재실행의 망을 P4-ADF와 같은 등급 구적에서 사영 저장 규약으로 다시 잰다 — 두 저장 정책을 같은
    구적에서 재채점하기 위한 입력(사전규정 보완 2). 참조 FEM과 코너 단면은 구적이 같으므로 P4-ADF의 것을 쓴다."""
    import torch
    from ..neural import p4_neural as pn
    d = manifest.ensure_outdir(outdir)
    launch = manifest.launch_provenance()
    width, depth = 64, 4
    if quick:
        levels, n_per_cell, width, depth = (0, 2), 6, 16, 2     # p4_projected_rerun(quick)와 같은 격자·망
    quads = {L: pn.quadrature_graded(n_per_cell, L, device="cpu") for L in levels}
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nn_rows = _relevel_neural(d, "projected", "p4_projected_rerun", quads, dev, width, depth, chunk)
    manifest.write_csv(os.path.join(d, "p4_adf_neural_projected.csv"), nn_rows)
    manifest.write_json(os.path.join(d, "manifest_p4_adf_projected.json"), manifest.build({
        "driver": "run_paperA_mlst.p4_adf_projected", "launch": launch, "quick": bool(quick),
        "levels": list(levels), "n_per_cell": n_per_cell, "ratio": 0.5,
        "quadrature": "the P4-ADF graded quadrature (manifest_p4_adf.json); level 0 is the training quadrature",
        "neural": "p4_params_projected.npz networks, re-projected on each quadrature with the projected-storage "
                  "convention: each mode is projected against the projected lower modes on that quadrature",
        "prespecification": PRESPEC_P4_PROJECTED}))
    return {"nn_rows": nn_rows, "outdir": d}


def _span_sine(A, B) -> float:
    """두 행 공간 사이 최대 주각의 사인 ‖(I − Q_A Q_Aᵀ) Q_B‖₂ — 행은 √w를 곱한 질량 내적 좌표의 벡터."""
    qa, _ = np.linalg.qr(np.asarray(A, float).T)
    qb, _ = np.linalg.qr(np.asarray(B, float).T)
    return float(np.linalg.norm(qb - qa @ (qa.T @ qb), 2))


def p4_policy_fields(outdir=None, device=None, quick: bool = False, chunk: int = 20_000) -> dict:
    """두 저장 정책이 모드 3에 넘기는 입력의 장 수준 대조(메모 [11]) — `p4_params_raw.npz`(P4-RAW)와
    `p4_params_projected.npz`(사영 저장 재실행)의 망을 훈련 구적에서 시드마다 잰다.

    - 모드 1·2 망 파라미터의 비트 동일성(SHA-256)과 두 실행의 상태 차이
    - 두 정책의 저장 집합 [φ₁, φ₂](원시)·[φ₁, φ̃₂](사영, φ̃₂ = 그 규칙으로 φ₁에 사영한 φ₂)의 스팬 거리(최대 주각의 사인)
    - 같은 입력 — 모드 3의 초기 망(`solve_p4_neural`의 seed·1000 + k, seed = 0, k = 2) — 에 두 저장 집합으로 그
      규칙을 적용한 결과의 상대 차이 `op_rel_diff`(대각 규칙만 저장 기저에 의존한다)
    - 학습을 마친 모드 3 상태(각 정책의 제 사영 뒤)의 두 실행 사이 상대 차이(서술)
    """
    import hashlib
    import torch
    from torch.func import vmap
    from ..neural import p4_neural as pn
    d = manifest.ensure_outdir(outdir)
    launch = manifest.launch_provenance()
    n_per_block, width, depth = 20, 64, 4
    if quick:
        n_per_block, width, depth = 6, 16, 2
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pts, w = pn.quadrature(n_per_block, dev)
    zr = np.load(os.path.join(d, "p4_params_raw.npz"))
    zp = np.load(os.path.join(d, "p4_params_projected.npz"))

    def ens_from(z, arm, m):
        return _p4_ensemble(z, arm, m, width, depth, dev)

    def states(e):
        return _p4_states(e, pts, chunk)

    def mass(A):
        return _p4_mass(A, w)

    def rel(a, b):
        return (torch.linalg.vector_norm(mass(a - b), dim=-1)
                / torch.linalg.vector_norm(mass(b), dim=-1)).cpu().numpy()

    def net(z, arm, m, s_):
        ks = sorted(k for k in z.files if k.startswith(f"{arm}_m{m}_"))
        return [np.ascontiguousarray(z[k][s_]) for k in ks]

    def sha(arrs):
        h = hashlib.sha256()
        for a in arrs:
            h.update(a.tobytes())
        return h.hexdigest()

    rows = []
    for arm in P4_ARMS:
        if not any(k.startswith(f"{arm}_m") for k in zp.files):
            continue
        proj = pn.PROJECTIONS[arm]
        n_seeds = int(zp[f"{arm}_m1_net.0.weight"].shape[0])
        with torch.no_grad():
            S1r, S2r, S3r = (states(ens_from(zr, arm, m)) for m in (1, 2, 3))
            S1p, S2p_raw, S3p = (states(ens_from(zp, arm, m)) for m in (1, 2, 3))
            S2p = vmap(lambda a, b: proj(a, b[None], w))(S2p_raw, S1p)            # 사영 저장 실행이 저장한 φ̃₂
            Praw, Pproj = torch.stack([S1r, S2r], 1), torch.stack([S1p, S2p], 1)
            phi = states(pn.EnsembleVecMLP(n_seeds, width, depth, seed=2, device=dev))
            out_r = vmap(lambda a, P: proj(a, P, w))(phi, Praw)
            out_p = vmap(lambda a, P: proj(a, P, w))(phi, Pproj)
            op = (torch.linalg.vector_norm(mass(out_r - out_p), dim=-1)
                  / torch.linalg.vector_norm(mass(phi), dim=-1)).cpu().numpy()
            tr = rel(vmap(lambda a, P: proj(a, P, w))(S3r, Praw), vmap(lambda a, P: proj(a, P, w))(S3p, Pproj))
            d1, d2 = rel(S1r, S1p), rel(S2r, S2p_raw)
            Mr, Mp = mass(Praw).cpu().numpy(), mass(Pproj).cpu().numpy()
        for s_ in range(n_seeds):
            n1r, n1p, n2r, n2p = net(zr, arm, 1, s_), net(zp, arm, 1, s_), net(zr, arm, 2, s_), net(zp, arm, 2, s_)
            a, b = Mr[s_]
            rows.append({"arm": arm, "seed": s_,
                         "mode1_params_identical": all(np.array_equal(x, y) for x, y in zip(n1r, n1p)),
                         "mode2_params_identical": all(np.array_equal(x, y) for x, y in zip(n2r, n2p)),
                         "mode1_sha256": sha(n1p), "mode2_sha256": sha(n2p),
                         "stored_mode1_rel_diff": float(d1[s_]), "stored_mode2_raw_rel_diff": float(d2[s_]),
                         "raw_pair_r": float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))),
                         "span_sin_max": _span_sine(Mr[s_], Mp[s_]), "op_rel_diff": float(op[s_]),
                         "mode3_trained_rel_diff": float(tr[s_])})
    manifest.write_csv(os.path.join(d, "p4_policy_fields.csv"), rows)
    manifest.write_json(os.path.join(d, "manifest_p4_policy_fields.json"), manifest.build({
        "driver": "run_paperA_mlst.p4_policy_fields", "launch": launch, "quick": bool(quick),
        "n_per_block": n_per_block, "width": width, "depth": depth,
        "networks": "p4_params_raw.npz (P4-RAW) and p4_params_projected.npz (projected-storage rerun)",
        "norm": "mass norm of the two displacement components on the training quadrature",
        "probe": "the mode-3 initial network of solve_p4_neural (EnsembleVecMLP seed 2), identical in both runs",
        "prespecification": PRESPEC_P4_PROJECTED}))
    return {"rows": rows, "outdir": d}


POSTHOC = "post hoc: added after the prespecified outputs were seen; not part of the prespecified judgments"


def p4_policy_posthoc(outdir=None, device=None, quick: bool = False, chunk: int = 20_000) -> dict:
    """사후 검사(사전규정 밖, 결과를 본 뒤) — `p4_policy_fields`를 보완한다.

    - 모드 1·2 상태의 여섯 성분(변위와 네 도함수) 전부에서 두 실행의 최대 절대 차이(메모 [11]의 도함수 배열 대조)
    - 학습을 마친 모드 3 상태(각 정책의 제 사영 뒤)의 두 실행 간 질량 MAC과 진폭 비. `mode3_trained_rel_diff`는 진폭에
      좌우된다 — Rayleigh 몫이 척도 불변이라 망 출력의 크기가 정해지지 않는다.
    - 사영 저장 집합 [φ₁, φ̃₂]의 정규화 비직교 |⟨φ₁, φ̃₂⟩|/(‖φ₁‖‖φ̃₂‖)와, 정확 규칙의 그람 jitter(1e-14, 절대)가 남기는
      몫 |⟨φ₁, φ₂⟩|·1e-14/(⟨φ₁, φ₁⟩ + 1e-14)/(‖φ₁‖‖φ̃₂‖). 대각·QR 경로에는 jitter가 없다(0).
    """
    import torch
    from torch.func import vmap
    from ..neural import p4_neural as pn
    d = manifest.ensure_outdir(outdir)
    launch = manifest.launch_provenance()
    n_per_block, width, depth = 20, 64, 4
    if quick:
        n_per_block, width, depth = 6, 16, 2
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pts, w = pn.quadrature(n_per_block, dev)
    zr = np.load(os.path.join(d, "p4_params_raw.npz"))
    zp = np.load(os.path.join(d, "p4_params_projected.npz"))
    rows = []
    for arm in P4_ARMS:
        if not any(k.startswith(f"{arm}_m") for k in zp.files):
            continue
        proj = pn.PROJECTIONS[arm]
        with torch.no_grad():
            R = [_p4_states(_p4_ensemble(zr, arm, m, width, depth, dev), pts, chunk) for m in (1, 2, 3)]
            P = [_p4_states(_p4_ensemble(zp, arm, m, width, depth, dev), pts, chunk) for m in (1, 2, 3)]
            S2p = vmap(lambda a, b: proj(a, b[None], w))(P[1], P[0])
            T_r = vmap(lambda a, Q: proj(a, Q, w))(R[2], torch.stack([R[0], R[1]], 1))
            T_p = vmap(lambda a, Q: proj(a, Q, w))(P[2], torch.stack([P[0], S2p], 1))
            dmax = [(R[i] - P[i]).abs().amax(dim=(1, 2)).cpu().numpy() for i in (0, 1)]
            a, b = _p4_mass(T_r, w), _p4_mass(T_p, w)
            ab, aa, bb = (a * b).sum(-1), (a * a).sum(-1), (b * b).sum(-1)
            mac, amp = (ab ** 2 / (aa * bb)).cpu().numpy(), torch.sqrt(aa / bb).cpu().numpy()
            f1, f2, f2p = _p4_mass(P[0], w), _p4_mass(P[1], w), _p4_mass(S2p, w)
            g11 = (f1 * f1).sum(-1)
            den = torch.sqrt(g11) * torch.linalg.vector_norm(f2p, dim=-1)
            defect = ((f1 * f2p).sum(-1).abs() / den).cpu().numpy()
            jit = ((f1 * f2).sum(-1).abs() * 1e-14 / (g11 + 1e-14) / den).cpu().numpy()
        if arm != "exact":
            jit = np.zeros_like(jit)
        for s_ in range(len(mac)):
            rows.append({"arm": arm, "seed": s_, "mode1_state_max_abs_diff": float(dmax[0][s_]),
                         "mode2_state_max_abs_diff": float(dmax[1][s_]), "mode3_mac_between_runs": float(mac[s_]),
                         "mode3_amp_ratio": float(amp[s_]), "stage3_defect": float(defect[s_]),
                         "stage3_defect_jitter": float(jit[s_])})
    manifest.write_csv(os.path.join(d, "p4_policy_posthoc.csv"), rows)
    manifest.write_json(os.path.join(d, "manifest_p4_policy_posthoc.json"), manifest.build({
        "driver": "run_paperA_mlst.p4_policy_posthoc", "launch": launch, "quick": bool(quick),
        "n_per_block": n_per_block, "width": width, "depth": depth, "status": POSTHOC,
        "state_diff": "max absolute difference over all six state components (u, v and four derivatives) "
                      "and all training quadrature points, raw-storage run minus projected-storage rerun",
        "mode3": "mass MAC and amplitude ratio between the trained mode-3 states, each after its own "
                 "policy's projection",
        "defect": "normalized non-orthogonality of the projected stored set [phi1, P(phi2)] before mode 3; "
                  "the jitter column is the share left by the absolute 1e-14 Gram jitter of the exact path",
        "prespecification": PRESPEC_P4_PROJECTED}))
    return {"rows": rows, "outdir": d}


# ── 판정 ────────────────────────────────────────────────────────────────────────────────

P4_UNCERTAINTY = {1: 9.849675673425165e-05, 2: 1.542817999384032e-04, 3: 3.334312108996915e-04}


def _load(d, name):
    return [json.loads(x) for x in open(os.path.join(d, name), encoding="utf-8")]


def stats_from(d: str, B: int = B_BOOT, seed: int = BOOT_SEED) -> dict:
    """사전규정 §3–§7의 판정. 자료가 없는 실험은 None이다."""
    out = {}
    # P1-RF — 주 판정: 모드 3 정답, RF − (a′); 보조: RF − (b)(동등성 문구 금지)
    try:
        rr = _load(d, "p1_rf_records.jsonl")
        by = {a: [r for r in rr if r["arm"] == a] for a in FROZEN.values()}
        n = len({r["seed"] for r in by[FROZEN["rf"]]})
        rf3, a3, b3 = (_indicator(by[FROZEN[k]], 3, n_seeds=n) for k in ("rf", "a_prime", "b"))
        ci_a = paired_bootstrap(rf3, a3, B, seed)
        ci_b = paired_bootstrap(rf3, b3, B, seed)
        per_mode = {a: {m: int(sum(1 for r in v if int(r["mode"]) == m and r["outcome"] == "correct"))
                        for m in sorted({int(r["mode"]) for r in v})} for a, v in by.items()}
        out["p1_rf"] = {"rf_minus_a_prime": {"contrast": ci_a, "verdict": rule(ci_a)},
                        "rf_minus_b": {"contrast": ci_b, "verdict": rule(ci_b),
                                       "wording": "no detectable difference within the tested sample"
                                       if rule(ci_b) == "rule2" else "difference detected"},
                        "correct_by_mode": per_mode, "n_seeds": n}
    except FileNotFoundError:
        out["p1_rf"] = None
    # fp32 RF — 모드 6, RF − (b), RF − (b′)(커밋된 fp32 기록, 같은 시드)
    try:
        rf = _load(d, "p1_rf_fp32_records.jsonl")
        old = _load(d, "p1_fp32_records.jsonl")
        n = len({r["seed"] for r in rf})
        x = _indicator(rf, 6, n_seeds=n)
        res = {"correct_6": {"rf_fp32": int(x.sum())}}
        for arm in sorted({r["arm"] for r in old}):
            y = _indicator([r for r in old if r["arm"] == arm and r.get("ladder", "I0") == "I0"], 6, n_seeds=n)
            ci = paired_bootstrap(x, y, B, seed)
            res[f"rf_minus_{arm}"] = {"contrast": ci, "verdict": rule(ci)}
            res["correct_6"][arm] = int(y.sum())
        out["p1_rf_fp32"] = res
    except FileNotFoundError:
        out["p1_rf_fp32"] = None
    # 기저 불변 — 주 판정: 모드 3 정답, α = 0.5 − α = 4
    try:
        br = _load(d, "p1_basis_invariance_records.jsonl")
        alphas = sorted({float(r["alpha"]) for r in br})
        n = len({r["seed"] for r in br})
        ind = {a: _indicator([r for r in br if float(r["alpha"]) == a], 3, n_seeds=n) for a in alphas}
        res = {"correct_by_alpha": {f"{a:g}": int(v.sum()) for a, v in ind.items()},
               "basin_by_alpha": {f"{a:g}": int(_indicator([r for r in br if float(r["alpha"]) == a], 3,
                                                           "lower_mode_basin", n).sum()) for a in alphas}}
        lo_a, hi_a = min(ALPHAS), max(ALPHAS)
        if lo_a in ind and hi_a in ind:
            ci = paired_bootstrap(ind[lo_a], ind[hi_a], B, seed)
            res["smallest_minus_largest_alpha"] = {"alphas": [lo_a, hi_a], "contrast": ci, "verdict": rule(ci)}
        vals = [int(ind[a].sum()) for a in alphas]
        res["non_increasing_in_alpha"] = all(x >= y for x, y in zip(vals, vals[1:]))
        out["p1_basis"] = res
    except FileNotFoundError:
        out["p1_basis"] = None
    # P4-RAW — 주 판정: 대각 규칙의 모드 3 정답, 원시 저장 − 사영 저장(커밋된 대각 arm, 같은 시드)
    try:
        raw = _load(d, "p4_raw_records.jsonl")
        from ..neural.p4_neural import ARM_NAME
        olds = {"exact": _load(d, "p4_neural_records.jsonl"), "diagonal": _load(d, "p4_neural_arms_records.jsonl"),
                "qr": _load(d, "p4_neural_arms_records.jsonl")}
        res = {}
        for arm in P4_ARMS:
            new = [r for r in raw if r["arm"] == ARM_NAME[arm]]
            if not new:
                continue
            old = [r for r in olds[arm] if r["arm"] == ARM_NAME[arm]]
            n = len({r["seed"] for r in new})
            c_new, c_old = _indicator(new, 3, n_seeds=n), _indicator(old, 3, n_seeds=n)
            ci = paired_bootstrap(c_new, c_old, B, seed)
            comp = {o: int(sum(1 for r in new if int(r["mode"]) == 3 and r["outcome"] == o)) for o in mt.OUTCOMES}
            res[arm] = {"correct3_raw_minus_projected": {"contrast": ci, "verdict": rule(ci)},
                        "mode3_raw": comp,
                        "mode3_projected": {o: int(sum(1 for r in old if int(r["mode"]) == 3 and r["outcome"] == o))
                                            for o in mt.OUTCOMES}}
        sp = manifest.read_csv(os.path.join(d, "p4_raw_span_per_seed.csv"))
        for arm in res:
            r3 = [r for r in sp if r["arm"] == arm and int(r["stage_mode"]) == 3]
            dv = np.array([float(r["dev_identity"]) for r in r3])
            lk = np.array([float(r["leak_rel"]) for r in r3])
            res[arm]["raw_pair"] = {"dev_identity_median": float(np.median(dv)), "dev_identity_max": float(dv.max()),
                                    "dev_identity_min": float(dv.min()), "leak_rel_median": float(np.median(lk))}
        out["p4_raw"] = res
    except FileNotFoundError:
        out["p4_raw"] = None
    # P4-ADF — 등급 4겹 → 8겹의 Λ 변화 ≤ 참조 불확실도(모드별)
    try:
        ref = manifest.read_csv(os.path.join(d, "p4_adf_reference.csv"))
        nn = manifest.read_csv(os.path.join(d, "p4_adf_neural.csv"))
        lv = sorted({int(r["level"]) for r in ref})
        hi, mid, lo = lv[-1], lv[-2], lv[0]
        res = {"levels": lv, "reference": {}, "neural": {}}
        for m in (1, 2, 3):
            L = {int(r["level"]): float(r["Lam"]) for r in ref if int(r["mode"]) == m}
            ch = abs(L[hi] - L[mid]) / abs(L[hi])
            res["reference"][m] = {"rel_change_hi_mid": ch, "rel_change_hi_train": abs(L[hi] - L[lo]) / abs(L[hi]),
                                   "uncertainty": P4_UNCERTAINTY[m], "pass": bool(ch <= P4_UNCERTAINTY[m])}
            for arm in P4_ARMS:
                cells = {}
                for r in nn:
                    if r["arm"] == arm and int(r["mode"]) == m and r["outcome"] == "correct":
                        cells.setdefault(int(r["seed"]), {})[int(r["level"])] = float(r["Lam"])
                if not cells:
                    continue
                chs = np.array([abs(c[hi] - c[mid]) / abs(c[hi]) for c in cells.values()])
                trs = np.array([abs(c[hi] - c[lo]) / abs(c[hi]) for c in cells.values()])
                res["neural"].setdefault(arm, {})[m] = {
                    "n_correct": int(len(chs)), "rel_change_hi_mid_median": float(np.median(chs)),
                    "rel_change_hi_mid_max": float(chs.max()), "rel_change_hi_train_median": float(np.median(trs)),
                    "n_pass": int((chs <= P4_UNCERTAINTY[m]).sum())}
        out["p4_adf"] = res
    except FileNotFoundError:
        out["p4_adf"] = None
    return out


def stats(outdir=None) -> dict:
    d = manifest.ensure_outdir(outdir)
    res = stats_from(d)
    manifest.write_json(os.path.join(d, "paperA_mlst_stats.json"), res)
    manifest.write_json(os.path.join(d, "manifest_paperA_mlst_stats.json"), manifest.build({
        "driver": "run_paperA_mlst.stats", "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED,
                                                         "cluster": "seed (paired)"},
        "prespecification": "2026-10-14-mlst-prespecification.md"}))
    return res


def projected_stats_from(d: str, B: int = B_BOOT, seed: int = BOOT_SEED) -> dict:
    """사전규정 보완 2의 판정 — 두 저장 정책의 셀을 **같은 최고 등급 구적**에서 재채점하고(e_λ 문턱만, 수렴·모드
    식별은 훈련 구적의 것 — 감사의 `rescore`), 규칙마다 모드 3 정답을 원시 − 사영의 시드 쌍으로 대비한다."""
    from ..neural.p4_neural import ARM_NAME
    from .paperA_mlst_audit import discordance, rescore
    ref = {int(r["mode"]): float(r["lambda_extrapolated"])
           for r in manifest.read_csv(os.path.join(d, "p4_reference.csv"))}
    nn = {"raw": manifest.read_csv(os.path.join(d, "p4_adf_neural.csv")),
          "projected": manifest.read_csv(os.path.join(d, "p4_adf_neural_projected.csv"))}
    level = max(int(r["level"]) for r in nn["projected"])
    recs = {"raw": _load(d, "p4_raw_records.jsonl"), "projected": _load(d, "p4_projected_rerun_records.jsonl")}
    res = {"level": level, "rescore": {}}
    for arm in P4_ARMS:
        per = {}
        for pol in ("raw", "projected"):
            lam = {}
            for r in nn[pol]:
                if r["arm"] == arm and int(r["level"]) == level:
                    lam.setdefault(int(r["mode"]), {})[int(r["seed"])] = float(r["Lam"])
            per[pol] = {}
            for m in (1, 2, 3):
                rs = [r for r in recs[pol] if r["arm"] == ARM_NAME[arm] and int(r["mode"]) == m]
                o = rescore(rs, lam[m], ref[m])
                old = {int(r["seed"]): r["outcome"] for r in rs}
                per[pol][m] = {"n_correct_training": o["n_correct"][0], "n_correct_level": o["n_correct"][1],
                               "flips": {f"{a} -> {b}": n for (a, b), n in o["flips"].items()},
                               "flipped_seeds": sorted(s for s, v in o["outcomes"].items() if v != old[s]),
                               "outcomes": {s: o["outcomes"][s] for s in sorted(o["outcomes"])}}
        seeds = sorted(per["raw"][3]["outcomes"])
        x = np.array([per["raw"][3]["outcomes"][s] == "correct" for s in seeds], float)
        y = np.array([per["projected"][3]["outcomes"][s] == "correct" for s in seeds], float)
        ci = paired_bootstrap(x, y, B, seed)
        per["raw_minus_projected_mode3"] = {"contrast": ci, "verdict": rule(ci),
                                            "discordance": discordance(list(x > 0), list(y > 0))}
        res["rescore"][arm] = per
    man = os.path.join(d, "manifest_p4_projected_rerun.json")
    res["gate"] = json.load(open(man, encoding="utf-8"))["reproduction_gate"] if os.path.exists(man) else None
    committed = [f for f in ("p4_neural_records.jsonl", "p4_neural_arms_records.jsonl")
                 if os.path.exists(os.path.join(d, f))]
    if committed:
        old = {(r["arm"], int(r["mode"]), int(r["seed"])): r["outcome"] for f in committed for r in _load(d, f)}
        keys = [(r["arm"], int(r["mode"]), int(r["seed"])) for r in recs["projected"]]
        agree = sum(old.get(k) == r["outcome"] for k, r in zip(keys, recs["projected"]))
        res["outcome_agreement_with_committed"] = {"n": len(keys), "agree": int(agree)}
    else:
        res["outcome_agreement_with_committed"] = None
    pf = manifest.read_csv(os.path.join(d, "p4_policy_fields.csv"))
    yes = lambda v: str(v) == "True"                              # noqa: E731
    by_arm = {}
    for arm in P4_ARMS:
        rr = [r for r in pf if r["arm"] == arm]
        if not rr:
            continue
        op, tr = (np.array([float(r[k]) for r in rr]) for k in ("op_rel_diff", "mode3_trained_rel_diff"))
        by_arm[arm] = {"op_rel_diff_median": float(np.median(op)), "op_rel_diff_min": float(op.min()),
                       "op_rel_diff_max": float(op.max()), "mode3_trained_rel_diff_median": float(np.median(tr)),
                       "mode3_trained_rel_diff_max": float(tr.max())}
    res["fields"] = {"n_networks": 2 * len(pf),
                     "n_networks_identical": int(sum(yes(r["mode1_params_identical"]) + yes(r["mode2_params_identical"])
                                                     for r in pf)),
                     "stored_rel_diff_max": max(max(float(r["stored_mode1_rel_diff"]),
                                                    float(r["stored_mode2_raw_rel_diff"])) for r in pf),
                     "span_sin_max": max(float(r["span_sin_max"]) for r in pf), "by_arm": by_arm}
    sp = manifest.read_csv(os.path.join(d, "p4_projected_rerun_span_per_seed.csv"))
    res["projected_span"] = {}
    for arm in P4_ARMS:
        r3 = [r for r in sp if r["arm"] == arm and int(r["stage_mode"]) == 3]
        if r3:
            dv, lk = (np.array([float(r[k]) for r in r3]) for k in ("dev_identity", "leak_rel"))
            res["projected_span"][arm] = {"n": len(r3), "dev_identity_median": float(np.median(dv)),
                                          "dev_identity_max": float(dv.max()),
                                          "leak_rel_median": float(np.median(lk)), "leak_rel_max": float(lk.max())}
    return res


def projected_stats(outdir=None, B: int = B_BOOT, seed: int = BOOT_SEED) -> dict:
    """사전규정 보완 2의 판정 파일 `paperA_mlst_projected_stats.json`(원래·후속 판정 파일은 건드리지 않는다)."""
    d = manifest.ensure_outdir(outdir)
    res = projected_stats_from(d, B=B, seed=seed)
    res["prespecification"] = PRESPEC_P4_PROJECTED
    res["bootstrap"] = {"B": B, "seed": seed, "cluster": "seed (paired)"}
    manifest.write_json(os.path.join(d, "paperA_mlst_projected_stats.json"), res)
    return res


def projected_posthoc_from(d: str) -> dict:
    """사후 요약(사전규정 밖) — 사영 저장 셀의 보충 표 S23 대칭 열, `p4_policy_posthoc.csv`의 요약, 두 정책의 모드 3
    e_λ 차이(두 정책 모두 정답인 시드)."""
    from ..neural.p4_neural import ARM_NAME
    from .paperA_mlst_audit import adf_all_cells
    ref = {int(r["mode"]): float(r["lambda_extrapolated"])
           for r in manifest.read_csv(os.path.join(d, "p4_reference.csv"))}
    rows = manifest.read_csv(os.path.join(d, "p4_adf_neural_projected.csv"))
    lv = sorted({int(r["level"]) for r in rows})
    cells = adf_all_cells(rows, ref, mid=lv[-2], hi=lv[-1])
    lam = {}
    for r in rows:
        lam.setdefault((r["arm"], int(r["mode"]), int(r["seed"])), {})[int(r["level"])] = (float(r["Lam"]), r["outcome"])
    hi = lv[-1]
    kept = [abs(v[0][0] - v[hi][0]) / abs(v[hi][0]) for (a, m, s), v in lam.items()
            if v[0][1] == "correct" and abs(v[hi][0] - ref[m]) / ref[m] <= mt.ELAM_MAX]
    ph = manifest.read_csv(os.path.join(d, "p4_policy_posthoc.csv"))
    recs = {"raw": _load(d, "p4_raw_records.jsonl"), "projected": _load(d, "p4_projected_rerun_records.jsonl")}
    sp = {(r["arm"], int(r["seed"])): float(r["dev_identity"])
          for r in manifest.read_csv(os.path.join(d, "p4_projected_rerun_span_per_seed.csv")) if int(r["stage_mode"]) == 3}
    between = {}
    for arm in P4_ARMS:
        rr = {int(r["seed"]): r for r in ph if r["arm"] == arm}
        if not rr:
            continue
        oc = {pol: {int(r["seed"]): r for r in recs[pol] if r["arm"] == ARM_NAME[arm] and int(r["mode"]) == 3}
              for pol in recs}
        both = [s for s in rr if oc["raw"][s]["outcome"] == oc["projected"][s]["outcome"] == "correct"]
        rest = [s for s in rr if s not in both]
        mac = {s: float(rr[s]["mode3_mac_between_runs"]) for s in rr}
        amp = [float(rr[s]["mode3_amp_ratio"]) for s in rr]
        de = [abs(float(oc["raw"][s]["e_lam"]) - float(oc["projected"][s]["e_lam"])) for s in both]
        dfx = np.array([float(rr[s]["stage3_defect"]) for s in rr])
        jit = np.array([float(rr[s]["stage3_defect_jitter"]) for s in rr])
        solver = np.array([sp[(arm, s)] for s in rr])
        pairs = {}
        for s in rest:
            k_ = f"{oc['raw'][s]['outcome']} | {oc['projected'][s]['outcome']}"
            pairs[k_] = pairs.get(k_, 0) + 1
        between[arm] = {
            "n_both_correct": len(both),
            "mac_both_correct_min": min((mac[s] for s in both), default=None),
            "max_abs_e_lam_difference_both_correct": max(de, default=None),
            "n_other": len(rest), "mac_other_median": float(np.median([mac[s] for s in rest])) if rest else None,
            "mac_other_min": min((mac[s] for s in rest), default=None),
            "outcome_pairs_other": pairs,
            "amp_ratio_min": min(amp), "amp_ratio_max": max(amp),
            "stage3_defect_max": float(dfx.max()), "stage3_defect_jitter_max": float(jit.max()),
            "stage3_defect_minus_jitter_max": float(np.abs(dfx - jit).max()),
            "stage3_defect_vs_solver_max_abs": float(np.max(np.abs(dfx - solver)))}     # 반올림 수준이라 절대 차이
    return {"status": POSTHOC, "level": hi,
            "adf_all_cells_projected": {f"{a}_mode{m}": v for (a, m), v in cells.items()},
            "correct_kept_max_rel_0_level": max(kept) if kept else None,
            "state_max_abs_diff": max(max(float(r["mode1_state_max_abs_diff"]), float(r["mode2_state_max_abs_diff"]))
                                      for r in ph),
            "mode3_between_runs": between}


def projected_posthoc(outdir=None) -> dict:
    """사후 요약 파일 `paperA_mlst_projected_posthoc.json` — 사전규정 판정 파일과 분리한다."""
    d = manifest.ensure_outdir(outdir)
    res = projected_posthoc_from(d)
    manifest.write_json(os.path.join(d, "paperA_mlst_projected_posthoc.json"), res)
    return res
