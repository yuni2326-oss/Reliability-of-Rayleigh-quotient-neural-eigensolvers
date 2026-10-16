"""P4(L형) 2차원 신경 arm — 논문이 "mesh-free 주장이 진짜 시험되는 무대"라 부른 자리.

P1의 arm (b)(정확 질량-직교 사영)와 같은 deflation을 2차원 재진입 코너 정의역에 올려,
같은 사전등록 판정규칙(MAC ≥ 0.9, e_λ ≤ 0.05, 4분류, 50시드, Wilson)으로 채점한다.
참조는 Richardson 외삽 Q2 등급메시이고, MAC은 FEM 모드를 신경 arm과 **같은 구적점**에서
평가해 잰다(`p4_lshape.mode_at`).

비교 기준을 P1과 맞추기 위해 고전 쪽도 같은 축에 올린다 — 등급 Q2 FEM의 (n_dof, 초, e_λ).
"""
from __future__ import annotations

import math
import os

import numpy as np

from .. import metrics as mt
from ..cost import expected_time_to_success
from ..problems import p4_lshape as p4
from . import manifest


def _mass_mac(A, B, w) -> float:
    """질량 가중 MAC — 벡터장이므로 두 성분을 함께 내적한다."""
    ab = float((w * (A[0] * B[0] + A[1] * B[1])).sum())
    aa = float((w * (A[0] ** 2 + A[1] ** 2)).sum())
    bb = float((w * (B[0] ** 2 + B[1] ** 2)).sum())
    if aa <= 0 or bb <= 0:
        return float("nan")
    return ab ** 2 / (aa * bb)


def main(outdir=None, n_modes: int = 3, n_seeds: int = 50, iters: int = 4000,
         n_per_block: int = 20, n_ref: int = 8, beta: float = 3.0,
         quick: bool = False, device=None) -> dict:
    from ..neural.p4_neural import solve_p4_neural
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_modes, n_seeds, iters, n_per_block, n_ref = 1, 3, 20, 8, 4

    out = solve_p4_neural(n_modes=n_modes, n_seeds=n_seeds, iters=iters,
                          n_per_block=n_per_block, device=device)
    pts, w = out["pts"], out["w"]

    # 참조: 외삽값이 있으면 그것을, 없으면 세밀 격자 직접해
    ref = p4.solve(n_ref, beta=beta, n_modes=max(n_modes + 3, 6))
    lam_ref = np.sort(ref["Lam"])
    try:
        ext = {int(r["mode"]): float(r["lambda_extrapolated"])
               for r in manifest.read_csv(os.path.join(d, "p4_reference.csv"))}
    except Exception:
        ext = {}
    ref_shapes = [np.asarray(p4.mode_at(ref, m, pts)).T for m in
                  range(1, len(lam_ref) + 1)]        # 각 (2, n_q)

    rows, recs = [], []
    _score(out, rows, recs, n_modes, n_seeds, iters, ext, lam_ref, ref_shapes, w)
    return _finish(d, rows, recs, out, n_modes, n_seeds, iters, n_per_block,
                   n_ref, beta)


def _score(out, rows, recs, n_modes, n_seeds, iters, ext, lam_ref, ref_shapes, w):
    """사전등록 4분기로 한 arm을 채점한다 — main과 run_arms가 **같은 게이트**를 쓴다."""
    drift = out.get("quadrature_drift")
    for m in range(1, n_modes + 1):
        target = ext.get(m, float(lam_ref[m - 1]))
        dvec = (np.asarray(drift)[m - 1] if drift is not None
                and len(np.asarray(drift)) >= m else None)
        cnt = {k: 0 for k in mt.OUTCOMES}
        el, macs, matched = [], [], []
        for s in range(n_seeds):
            lam = float(out["lam"][m - 1][s])
            phi = out["shapes"][m - 1][s]                  # (2, n_q)
            mac_all = [_mass_mac(phi, r, w) for r in ref_shapes]
            j = int(np.nanargmax(mac_all)) + 1
            mac = float(mac_all[j - 1])
            e = abs(lam - target) / target if math.isfinite(lam) else float("nan")
            conv = _converged(out["history"][m - 1], s)
            oc = ("non_converged" if not conv or not math.isfinite(e)
                  else "correct" if (j == m and mac >= mt.MAC_MIN
                                     and e <= mt.ELAM_MAX)
                  else "lower_mode_basin" if (j < m and mac >= mt.MAC_MIN)
                  else "spurious")
            cnt[oc] += 1
            el.append(e); macs.append(mac); matched.append(j)
            recs.append({"arm": out["arm"], "problem": "P4", "mode": m, "seed": s,
                         "ladder": "I0", "nodes": "gauss2d", "outcome": oc,
                         "matched_mode": j, "mac": mac, "e_lam": e,
                         "Lam": lam, "Lam_ref": target,
                         "seconds": out["stage_seconds"][m - 1],
                         "iters": iters, "n_q": out["n_q"],
                         **({"quad_drift": float(dvec[s]),
                             "Lam_check": float(out["lam_check"][m - 1][s])}
                            if dvec is not None else {})})
        lo, hi = mt.wilson(cnt["correct"], n_seeds)
        acc = sum(1 for j, mm, e in zip(matched, macs, el)
                  if j == m and mm >= mt.MAC_MIN and math.isfinite(e)
                  and e <= mt.ELAM_MAX)
        secs = float(out["stage_seconds"][m - 1])
        agg_drift = ({} if dvec is None else {
            "quad_drift_median": float(np.median(dvec[np.isfinite(dvec)]))
            if np.isfinite(dvec).any() else float("nan"),
            "quad_drift_max": float(np.max(dvec[np.isfinite(dvec)]))
            if np.isfinite(dvec).any() else float("nan"),
            "check_n_per_block": int(out.get("check_n_per_block") or 0)})
        rows.append({**agg_drift,
                     "problem": "P4", "arm": out["arm"], "mode": m,
                     "ladder": "I0", "nodes": "gauss2d", "n": n_seeds,
                     "p_correct": cnt["correct"] / n_seeds,
                     "wilson_lo": lo, "wilson_hi": hi, **cnt,
                     "p_accurate": acc / n_seeds,
                     "e_lam_median": float(np.nanmedian(el)),
                     "mac_median": float(np.nanmedian(macs)),
                     "Lam_median": float(np.median(out["lam"][m - 1])),
                     "Lam_ref": target, "iters": iters,
                     "seconds": secs, "n_q": out["n_q"],
                     "E_T_success": expected_time_to_success(
                         secs, cnt["correct"] / n_seeds)})
        print(f"  [P4 neural m{m}] p_correct={rows[-1]['p_correct']:.2f} "
              f"e_lam={rows[-1]['e_lam_median']:.3e} "
              f"MAC={rows[-1]['mac_median']:.4f} {secs:.0f}s", flush=True)

    return rows


def _finish(d, rows, recs, out, n_modes, n_seeds, iters, n_per_block, n_ref, beta):
    manifest.write_csv(os.path.join(d, "p4_neural.csv"), rows)
    manifest.write_jsonl(os.path.join(d, "p4_neural_records.jsonl"), recs)
    manifest.write_json(os.path.join(d, "manifest_p4_neural.json"),
                        manifest.build({
                            "driver": "run_p4_neural.main",
                            "n_modes": n_modes, "n_seeds": n_seeds, "iters": iters,
                            "n_per_block": n_per_block, "n_q": out["n_q"],
                            "n_ref": n_ref, "beta": beta,
                            "disclosure": out["disclosure"],
                            "reference": "Richardson-extrapolated graded Q2 where "
                                         "available, else direct fine-grid solve",
                            "mac": "mass-weighted, FEM reference evaluated at the "
                                   "neural arm's own quadrature points"}))
    print(f"[run_p4_neural] {d} — {len(rows)}행")
    return {"p4_neural": rows, "outdir": d}


def _reference(d, n_ref: int, beta: float, n_modes: int, pts):
    """참조 고유값과 모드형 — main과 run_arms가 **같은 참조**를 써야 비교가 성립한다."""
    ref = p4.solve(n_ref, beta=beta, n_modes=max(n_modes + 3, 6))
    lam_ref = np.sort(ref["Lam"])
    try:
        ext = {int(r["mode"]): float(r["lambda_extrapolated"])
               for r in manifest.read_csv(os.path.join(d, "p4_reference.csv"))}
    except Exception:
        ext = {}
    shapes = [np.asarray(p4.mode_at(ref, m, pts)).T
              for m in range(1, len(lam_ref) + 1)]
    return ext, lam_ref, shapes


def run_arms(outdir=None, arms=("diagonal", "qr"), n_modes: int = 3,
             n_seeds: int = 50, iters: int = 4000, n_per_block: int = 20,
             n_ref: int = 8, beta: float = 3.0, quick: bool = False,
             device=None) -> dict:
    """P4에 사영 규칙을 바꿔 단 arm들 — 커밋된 `p4_neural.csv`는 건드리지 않는다.

    묻는 것은 두 가지다. (i) **lower-mode basin이 2차원에서도 재현되는가** — P1에서
    (a′)의 벽을 만든 기전이 차원에 무관한지. (ii) (b′)이 2차원에서도 (b)와 같은 신뢰도를
    내는가 — 같은 사영이므로 같아야 하고, 아니면 구현이 틀린 것이다.

    **공정성 단서 하나.** (b) 경로의 그람 풀이에는 1e-14·I jitter가 있고(원래 구현),
    (a′)·(b′)에는 없다 — 전자는 역행렬을 풀지 않고 후자는 그람을 만들지 않기 때문이다.
    이 비대칭은 manifest와 표 각주에 남긴다.

    모드 4–6 확장은 하지 않는다. **참조가 없어서가 아니다** — 그 참조는 이미 있고 목표
    불확실도를 만족한다. 이유는 근접축퇴다: 외삽 고유값이 12.105·12.613·12.724라 인접
    상대간격이 4.2 %와 0.9 %이고 둘 다 사전등록 허용오차 5 % 안이므로, 고유값만으로는
    세 모드를 가를 수 없다. 확장하려면 P2의 고유공간 채점 분기를 세 모드 군집에
    적용해야 하고 그것은 사전 선언이 필요한 별도 결정이다.
    """
    from ..neural.p4_neural import solve_p4_neural
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_modes, n_seeds, iters, n_per_block, n_ref = 1, 3, 20, 8, 4

    rows, recs, disc = [], [], {}
    for kind in arms:
        out = solve_p4_neural(n_modes=n_modes, n_seeds=n_seeds, iters=iters,
                              n_per_block=n_per_block, projection=kind,
                              device=device)
        ext, lam_ref, shapes = _reference(d, n_ref, beta, n_modes, out["pts"])
        print(f"  [P4 arm {kind}]", flush=True)
        _score(out, rows, recs, n_modes, n_seeds, iters, ext, lam_ref, shapes,
               out["w"])
        disc = out["disclosure"]

    manifest.write_csv(os.path.join(d, "p4_neural_arms.csv"), rows)
    manifest.write_jsonl(os.path.join(d, "p4_neural_arms_records.jsonl"), recs)
    manifest.write_json(os.path.join(d, "manifest_p4_neural_arms.json"),
                        manifest.build({
                            "driver": "run_p4_neural.run_arms",
                            "arms": list(arms), "n_modes": n_modes,
                            "n_seeds": n_seeds, "iters": iters,
                            "n_per_block": n_per_block, "n_ref": n_ref,
                            "beta": beta, "disclosure": disc,
                            "gate": "the pre-specified rules, unchanged: MAC >= 0.9, "
                                    "|dlambda|/lambda <= 0.05, four outcomes, Wilson 95 %",
                            "jitter": "the exact-projection path adds 1e-14 times the "
                                      "identity to the Gram before solving, as in the "
                                      "committed arm (b) run; the diagonal and QR paths "
                                      "add nothing, because one does not invert a matrix "
                                      "and the other never forms the Gram",
                            "not_run": "modes 4-6. Their Richardson references exist and "
                                       "meet the uncertainty target; the obstacle is that "
                                       "they form a near-degenerate cluster (12.105, "
                                       "12.613, 12.724; consecutive relative gaps 4.2 % "
                                       "and 0.9 %, both inside the 5 % eigenvalue "
                                       "tolerance), so scoring them needs the P2 "
                                       "eigenspace branch declared in advance",
                            "added": "2026-08-26, after the pre-specified grid; the "
                                     "judgment rules were not changed"}))
    print(f"[run_p4_neural.run_arms] {d} — {len(rows)}행")
    return {"p4_arms": rows, "outdir": d}


def run_span_diagnostics(outdir=None, projection: str = "diagonal", n_seeds: int = 8,
                         iters: int = 4000, n_per_block: int = 20,
                         n_modes: int = 3, quick: bool = False, device=None) -> dict:
    """P4에서 각 스테이지가 넘겨받는 저장 스팬의 직교성·조건수.

    **왜 별도 실행인가.** P1에서 (a′)가 모드 3에서 무너진 이유는 저장 스팬이 강하게
    비직교하기 때문이었다(최대 비대각 0.99). 2차원에서 같은 규칙이 살아남는다면 그 스팬이
    다르다는 뜻이고, 그것은 채점이 아니라 측정의 문제다. 시드 수가 채점 실행과 다르므로
    산출물을 섞지 않고 자기 파일에 쓴다.
    """
    from ..neural.p4_neural import solve_p4_neural
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_modes, n_seeds, iters, n_per_block = 2, 2, 2, 4

    out = solve_p4_neural(n_modes=n_modes, n_seeds=n_seeds, iters=iters,
                          n_per_block=n_per_block, projection=projection,
                          device=device)
    keys = ("k", "kappa_gram", "kappa_gram_times_eps", "dev_identity",
            "lam_min_normalized", "qr_orth_defect", "n_dropped", "rank_1e10",
            "rank_1e12", "rank_1e16", "proj_agreement_rel",
            "gram_solve_finite_frac", "eps_machine", "dtype", "n_seeds_summarized")
    rows = []
    for m, sd in enumerate(out["span_diagnostics"], start=1):
        if sd is None:
            continue
        row = {"problem": "P4", "arm": out["arm"], "projection": projection,
               "stage_mode": m, "n_seeds": n_seeds, "iters": iters}
        row.update({k: sd.get(k) for k in keys})
        rows.append(row)
        print(f"  [P4 span m{m} k={row['k']}] ||C-I||2 {row['dev_identity']:.3e}  "
              f"kappa(G) {row['kappa_gram']:.3e}  ||I-QWQ|| {row['qr_orth_defect']:.2e}  "
              f"agree {row['proj_agreement_rel']:.2e}", flush=True)

    manifest.write_csv(os.path.join(d, "p4_span_diagnostics.csv"), rows)
    manifest.write_json(os.path.join(d, "manifest_p4_span_diagnostics.json"),
                        manifest.build({
                            "driver": "run_p4_neural.run_span_diagnostics",
                            "projection": projection, "n_modes": n_modes,
                            "n_seeds": n_seeds, "iters": iters,
                            "n_per_block": n_per_block,
                            "metric": "the same span diagnostics as P1, taken in the "
                                      "two-component mass inner product",
                            "not_scored": "this run measures the stored span only; the "
                                          "scored 50-seed cells are in p4_neural.csv and "
                                          "p4_neural_arms.csv",
                            "added": "2026-08-26"}))
    print(f"[run_span_diagnostics] {len(rows)}행")
    return {"p4_span": rows, "outdir": d}


def _converged(hist, s: int, tol: float = 1e-3, tail: float = 0.1) -> bool:
    """사전등록 수렴규칙을 시드 s에 적용 — P1과 같은 꼬리 10 % 상대변화 기준."""
    if len(hist) < 3:
        return False
    k = max(int(len(hist) * tail), 2)
    a, b = float(hist[-k][2][s]), float(hist[-1][2][s])
    return math.isfinite(b) and abs(b - a) / max(abs(b), 1e-300) <= tol


if __name__ == "__main__":
    main()


def single_seed_cost(outdir=None, n_modes: int = 3, iters: int = 4000,
                     n_per_block: int = 20, reps: int = 3, quick: bool = False,
                     projections=("exact",), device=None) -> dict:
    """P4의 1회 시도 비용을 직접 잰다 — P1과 같은 회계로 맞춘다.

    §5.4의 P4 비율은 50시드 앙상블 벽시계를 1회 비용으로 쓰고 있어 격차를 **과대**계상했고,
    그래서 상한으로만 인용할 수 있었다. P1에서 이미 확인했듯 앙상블→1회 환산계수는 이 기계
    (GPU를 상주 추론 서비스와 공유)에서 2.53~7.82로 흔들려 쓸 수 없다. 그러니 P1과 마찬가지로
    **필요한 값을 직접 잰다**: n_seeds = 1, reps회, median과 min/max.

    결과는 `p1_single_seed_cost.csv`에 덧붙인다 — `run_p1_compare._single_seed_cost`가
    (arm, mode) 하나의 사전에서 찾도록 출처를 한 곳으로 유지한다.
    """
    from ..neural.p4_neural import solve_p4_neural
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_modes, iters, n_per_block, reps = 1, 20, 8, 1
    # **arm마다 잰다.** 3-arm 비교의 비용 칸을 앙상블 벽시계로 채우면, 이 연구가
    # 직접측정으로 교체한 상한 회계가 되살아난다.
    from ..neural.p4_neural import ARM_NAME
    path = os.path.join(d, "p1_single_seed_cost.csv")
    names = {ARM_NAME[k] for k in projections}
    keep = [r for r in manifest.read_csv(path)
            if str(r["arm"]) not in names] if os.path.exists(path) else []
    rows = []
    for kind in projections:
        per = {m: [] for m in range(1, n_modes + 1)}
        for _ in range(reps):
            out = solve_p4_neural(n_modes=n_modes, n_seeds=1, iters=iters,
                                  n_per_block=n_per_block, projection=kind,
                                  device=device)
            for m in range(1, n_modes + 1):
                per[m].append(float(out["stage_seconds"][m - 1]))
        for m in range(1, n_modes + 1):
            ts = sorted(per[m])
            med = ts[len(ts) // 2]
            rows.append({"arm": ARM_NAME[kind], "mode": m, "iters": iters,
                         "reps": len(ts), "seconds_per_attempt": med,
                         "seconds_min": ts[0], "seconds_max": ts[-1],
                         "spread_rel": (ts[-1] - ts[0]) / max(med, 1e-300),
                         "measured": "directly at n_seeds = 1 (no batching conversion); "
                                     "P4, two dimensions"})
            print(f"  [1seed P4 {kind} m{m}] median {med:7.1f}s "
                  f"[{ts[0]:.1f}, {ts[-1]:.1f}]", flush=True)
    manifest.write_csv(path, keep + rows)
    manifest.write_json(os.path.join(d, "manifest_p4_single_seed_cost.json"),
                        manifest.build({
                            "driver": "run_p4_neural.single_seed_cost",
                            "projections": list(projections),
                            "n_modes": n_modes, "iters": iters, "reps": reps,
                            "n_per_block": n_per_block,
                            "why": "the P4 cost rows used the 50-seed ensemble wall-clock "
                                   "as if it were a per-attempt cost, so the gap was an "
                                   "upper bound; measuring at one seed puts P4 on the "
                                   "same accounting as P1"}))
    print(f"[p4 single_seed_cost] {len(rows)}행 (p1_single_seed_cost.csv에 병합)")
    return {"single_seed": rows, "outdir": d}


# ── P4 구적분리 검사 (2026-09-07, 사후추가) ─────────────────────────────────
# 위협 13은 P4의 69셀을 **하한 스캔**으로 잡았다. Λ가 Λ₁ 위에 있으면서 과소적분된
# 셀은 그 스캔에 걸리지 않으므로, P1에 붙인 것과 같은 두 차수 계기를 P4에도 붙인다.
# 커밋된 `p4_neural_arms.csv`는 덮지 않고 별도 파일에 쓴다.
QCHECK_ARMS = ("exact", "diagonal", "qr")
QCHECK_N_PER_BLOCK = 40          # 학습 20의 2배 — Gauss 점이 하나도 겹치지 않는다


def quadrature_check(outdir=None, arms=QCHECK_ARMS, n_modes: int = 3,
                     n_seeds: int = 50, iters: int = 4000,
                     n_per_block: int = 20,
                     check_n_per_block: int = QCHECK_N_PER_BLOCK,
                     n_ref: int = 8, beta: float = 3.0, quick: bool = False,
                     device=None) -> dict:
    """세 규칙을 커밋된 설정 그대로 다시 풀되, 셀마다 두 번째 구적차수에서 재측정한다."""
    from ..neural.p4_neural import solve_p4_neural

    d = manifest.ensure_outdir(outdir)
    if quick:
        n_modes, n_seeds, iters, n_per_block, n_ref = 1, 3, 20, 8, 4
        check_n_per_block = 16

    rows, recs, disc = [], [], {}
    for kind in arms:
        out = solve_p4_neural(n_modes=n_modes, n_seeds=n_seeds, iters=iters,
                              n_per_block=n_per_block, projection=kind,
                              check_n_per_block=check_n_per_block, device=device)
        ext, lam_ref, shapes = _reference(d, n_ref, beta, n_modes, out["pts"])
        print(f"  [P4 구적검사 {kind}]", flush=True)
        _score(out, rows, recs, n_modes, n_seeds, iters, ext, lam_ref, shapes,
               out["w"])
        disc = out["disclosure"]
        # **arm이 끝날 때마다 저장한다.**
        manifest.write_csv(os.path.join(d, "p4_quadrature_check.csv"), rows)
        manifest.write_jsonl(
            os.path.join(d, "p4_quadrature_check_records.jsonl"), recs)

    manifest.write_csv(os.path.join(d, "p4_quadrature_check.csv"), rows)
    manifest.write_jsonl(
        os.path.join(d, "p4_quadrature_check_records.jsonl"), recs)
    manifest.write_json(
        os.path.join(d, "manifest_p4_quadrature_check.json"), manifest.build({
            "driver": "run_p4_neural.quadrature_check",
            "arms": list(arms), "n_modes": n_modes, "n_seeds": n_seeds,
            "iters": iters, "n_per_block": n_per_block,
            "check_n_per_block": check_n_per_block,
            "n_ref": n_ref, "beta": beta, "disclosure": disc, "quick": bool(quick),
            "purpose": "the sub-lambda1 scan catches only gross under-integration; a "
                       "cell can be under-integrated while its Rayleigh quotient stays "
                       "above lambda1, and only a second quadrature order sees that",
            "scoring": "unchanged from the committed arm run; the instrument adds one "
                       "forward pass after training and does not enter the objective",
            "added": "2026-09-07, post-hoc, after the P1 two-order confirmation"}))
    print(f"[P4 구적검사] {len(rows)}행", flush=True)
    return {"rows": rows, "records": recs, "outdir": d}
