"""P1 신경망 실행격자 — §5.1의 empirical content.

**v3 §5.1 격자에서 재배분했다.** docx는 "모드3 고정 × arm × 사다리"였으나 스파이크에서
정확 사영은 모드3·I0에서 이미 50/50, 순수 페널티는 0/50이라 두 축이 포화된다.
정보가 있는 축은 (i) 모드수 — 사영이 어디서 무너지는가, (ii) 페널티 가중 임계,
(iii) 구적, (iv) 사영이 무너지는 구간에서의 사다리다. 재배분 사실은 논문에 명시한다.

  docker run --rm --gpus all --ipc=host -v /home/super/project:/work -w /work \
    -e PYTHONPATH=/work gb10-pinn:26.04 python3 -u -m eigen_benchmark.drivers.run_p1_neural
"""
from __future__ import annotations

import math
import os

import numpy as np

from .. import metrics as mt
from ..cost import expected_time_to_success
from ..neural import deflation as df
from ..neural import ladder as ld
from ..neural import sequential as sq
from ..neural.curriculum import solve_curriculum
from ..neural.pielm import solve_pielm
from ..neural.subspace import solve_neural_basis_galerkin, solve_simultaneous
from ..problems import p1_beam as p1
from . import manifest

MODE_MAX = 10
PENALTY_WEIGHTS = (1e-1, 1e0, 1e1, 1e2, 1e3)
PIELM_FEATURES = (20, 40, 80, 160)


def _cell(out, target, arm, ladder_name, nodes, n_ref: int = 14):
    """한 스테이지의 시드별 레코드 + 셀 집계.

    `seconds`는 **그 스테이지**의 시간이다(전체 합계가 아니라) — 합계를 쓰면 모드 1-10을
    돈 arm과 1-3만 돈 arm의 비용이 같은 열에서 비교 불가능해진다.
    진단 열 `p_accurate`는 **수렴 인증과 무관하게** e_λ·MAC가 기준을 만족한 비율이다.
    확률적 구적이나 성격이 다른 목적함수에서는 "해는 맞았으나 인증 못 함"이 지배적이라
    이 구분이 없으면 표가 거짓 인상을 준다."""
    # 모드 15 이상을 채점하려면 기준 모드형이 그만큼 있어야 한다(기본 14개로는
    # 인덱스가 넘친다). 목표보다 넉넉히 잡아 오식별 후보를 놓치지 않는다.
    recs = sq.classify_stage(out["shapes"][target - 1], out["lam"][target - 1],
                             target=target, xs=out["xs"], wq=out["wq"],
                             n_ref=max(int(n_ref), int(target) + 4),
                             converged=out["converged"][target - 1])
    st = out.get("stage_seconds")
    secs = float(st[target - 1]) if st and len(st) >= target else float(out["seconds"])
    # 사전적합(사다리 I1·I3·I5)은 공짜가 아니다 — 청구하지 않으면 "이득 없고 비용도 없다"로
    # 읽혀 실제 결론("이득이 없는데 비용은 든다 = 열등")을 놓친다.
    isec = out.get("init_seconds") or []
    init_s = float(sum(isec[:target])) if len(isec) >= target else 0.0
    it = out["iters"]
    it = int(it[target - 1]) if isinstance(it, (list, tuple)) else int(it)
    # 두 번째 구적차수로 잰 드리프트는 **셀마다** 실린다. 학습되는 기저는 자신을 재는
    # 격자에 맞춰 변형하므로, 이 열이 없으면 "격자를 속인 셀"과 "진짜로 낮은 Λ를 찾은 셀"을
    # 표에서 구별할 수 없다(P5에서 실측). 계기를 붙이지 않은 실행에는 열 자체가 없다.
    drift = out.get("quad_drift")
    dvec = (np.asarray(drift)[target - 1] if drift is not None
            and len(np.asarray(drift)) >= target else None)
    for i, r in enumerate(recs):
        r.update({"arm": arm, "mode": target, "seed": i, "nodes": nodes,
                  "ladder": ladder_name, "seconds": secs,
                  "init_seconds": init_s, "seconds_with_init": secs + init_s,
                  "iters": it, "n_q": out["n_q"]})
        if dvec is not None:
            r["quad_drift"] = float(dvec[i])
    cnt = mt.confusion_counts(recs)
    lo, hi = mt.wilson(cnt["correct"], cnt["n"])
    acc = [r for r in recs
           if r["matched_mode"] == target and np.isfinite(r["e_lam"])
           and r["e_lam"] <= mt.ELAM_MAX and r["mac"] >= mt.MAC_MIN]
    agg = {"arm": arm, "mode": target, "ladder": ladder_name, "nodes": nodes,
           "n": cnt["n"], "p_correct": cnt["correct"] / cnt["n"],
           "wilson_lo": lo, "wilson_hi": hi,
           **{k: cnt[k] for k in mt.OUTCOMES},
           "p_accurate": len(acc) / cnt["n"],
           "n_certified": sum(1 for r in recs if r.get("certified")),
           "iters": it, "seconds": secs,
           "init_seconds": init_s, "seconds_with_init": secs + init_s}
    if dvec is not None:
        finite = dvec[np.isfinite(dvec)]
        agg["quad_drift_median"] = float(np.median(finite)) if finite.size else float("nan")
        agg["quad_drift_max"] = float(finite.max()) if finite.size else float("nan")
        agg["check_n_q"] = int(out.get("check_n_q") or 0)
    if "rank_median" in out:
        agg["rank_median"] = out["rank_median"]
        agg["rank_min"] = out["rank_min"]
    return recs, agg


def main(outdir=None, quick: bool = False, device=None) -> dict:
    d = manifest.ensure_outdir(outdir)
    n_seeds = 6 if quick else 50
    iters = 400 if quick else 4000
    n_modes = 3 if quick else MODE_MAX
    weights = (1e0, 1e2) if quick else PENALTY_WEIGHTS
    feats = (20, 40) if quick else PIELM_FEATURES

    records, mode_rows, pen_rows, quad_rows, ladder_rows, cost_rows = \
        [], [], [], [], [], []

    def checkpoint(tag: str):
        """**구간마다 증분 기록.** 마지막에 일괄 기록하면 후반 크래시로 앞선
        수 시간치 계산이 통째로 날아간다(실제로 (d) arm의 Cholesky 실패로 1h37m 소실)."""
        for name, rows in (("mode_sweep", mode_rows), ("penalty_sweep", pen_rows),
                           ("quadrature", quad_rows), ("ladder", ladder_rows)):
            if rows:
                manifest.write_csv(os.path.join(d, f"p1_neural_{name}.csv"), rows)
        if records:
            manifest.write_jsonl(os.path.join(d, "p1_neural_records.jsonl"), records)
        print(f"  [checkpoint:{tag}] 레코드 {len(records)}, 모드스윕 {len(mode_rows)}행",
              flush=True)

    # (1) 주축 — 모드수 × {대각사영(파일럿), 정확사영}
    for arm, defl in (("a_prime_projection_diagonal", df.ProjectionDiagonal()),
                      ("b_projection_exact", df.ProjectionExact())):
        out = sq.solve_sequential(n_modes, defl, n_seeds=n_seeds, iters=iters,
                                  device=device)
        for m in range(1, n_modes + 1):
            r, agg = _cell(out, m, arm, "I0", "gauss")
            records += r
            mode_rows.append(agg)
    checkpoint("1_mode_sweep")

    # (2) 페널티 가중 스윕
    n_pen = min(3, n_modes)
    for w in weights:
        out = sq.solve_sequential(n_pen, df.Penalty(w), n_seeds=n_seeds,
                                  iters=iters, device=device)
        for m in range(1, n_pen + 1):
            r, agg = _cell(out, m, f"a_penalty(w={w:g})", "I0", "gauss")
            agg["penalty_weight"] = w
            records += r
            pen_rows.append(agg)
    checkpoint("2_penalty")

    # (3) 구적 대조
    for nodes in ("gauss", "mc"):
        out = sq.solve_sequential(n_pen, df.ProjectionExact(), n_seeds=n_seeds,
                                  iters=iters, nodes=nodes, device=device)
        for m in range(1, n_pen + 1):
            r, agg = _cell(out, m, "b_projection_exact", "I0", nodes)
            records += r
            quad_rows.append(agg)
    checkpoint("3_quadrature")

    # (4) 사다리 — 파일럿 arm이 처음 실패하는 모드에서만
    fail = [row["mode"] for row in mode_rows
            if row["arm"] == "a_prime_projection_diagonal" and row["p_correct"] < 0.9]
    m_star = min(fail) if fail else n_pen
    for lvl in ("I0", "I1", "I3", "I5"):
        init = ld.make_ladder(lvl, mode=m_star, iters=max(iters // 2, 200))
        out = sq.solve_sequential(m_star, df.ProjectionDiagonal(), n_seeds=n_seeds,
                                  iters=iters, init=init, device=device)
        r, agg = _cell(out, m_star, "a_prime_projection_diagonal", lvl, "gauss")
        records += r
        ladder_rows.append(agg)
    checkpoint("4_ladder")

    # (5) 커리큘럼 (c) — I4
    out = solve_curriculum(n_pen, df.ProjectionExact(), n_seeds=n_seeds,
                           iters=iters, device=device)
    for m in range(1, n_pen + 1):
        r, agg = _cell(out, m, out["arm"], "I4", "gauss")
        records += r
        mode_rows.append(agg)
    checkpoint("5_curriculum")

    # (6) 부분공간 (d)(e) — 순차 오차누적이 없는 두 접근
    n_sub = min(6, n_modes)
    for out in (solve_neural_basis_galerkin(n_sub, n_seeds=n_seeds, iters=iters,
                                            device=device),
                solve_simultaneous(n_sub, n_seeds=n_seeds, iters=iters,
                                   device=device)):
        for m in range(1, n_sub + 1):
            r, agg = _cell(out, m, out["arm"], "I0", "gauss")
            records += r
            mode_rows.append(agg)
    checkpoint("6_subspace")

    # (7) PIELM (CPU, 학습 없음) — **다른 arm과 같은 시드 프로토콜**
    #
    # 처음 구현은 무작위특징 1회 추출(n=1, Wilson 구간 없음)로 p_correct를 냈다.
    # 그건 다른 여섯 arm에 50시드를 요구하면서 가장 싼 arm에만 면제를 준 것이다 —
    # 1회가 15~18 ms이므로 50회에 1초가 안 걸린다. 특히 질량행렬 조건수가
    # κ ≈ 1e300 규모라 단일 추출이 대표적이라는 보장이 전혀 없다.
    ref = p1.beta_roots(n_pen) ** 4
    for nf in feats:
        cnt = {m: {"correct": 0, "spurious": 0, "non_converged": 0} for m in
               range(1, n_pen + 1)}
        secs, ranks, kappas, spectra, elams = [], [], [], [], {m: [] for m in
                                                               range(1, n_pen + 1)}
        for sd in range(n_seeds):
            o = solve_pielm(nf, n_modes=n_pen, seed=sd)
            secs.append(o["seconds"])
            ranks.append(o["rank_used"])
            kappas.append(o["kappa_M"])
            spectra.append(o["rank_spectrum"])
            for m in range(1, n_pen + 1):
                e = (abs(o["Lam"][m - 1] - ref[m - 1]) / ref[m - 1]
                     if o["cholesky_ok"] and np.isfinite(o["Lam"][m - 1])
                     else float("nan"))
                elams[m].append(e)
                if not o["cholesky_ok"] or not np.isfinite(e):
                    cnt[m]["non_converged"] += 1
                elif e <= mt.ELAM_MAX:
                    cnt[m]["correct"] += 1
                else:
                    cnt[m]["spurious"] += 1
                records.append({"arm": f"f_eig_pielm(nf={nf})", "mode": m,
                                "seed": sd, "ladder": "n/a", "nodes": "gauss",
                                "outcome": ("correct" if e <= mt.ELAM_MAX
                                            else "spurious") if np.isfinite(e)
                                           else "non_converged",
                                "matched_mode": m, "mac": float("nan"),
                                "e_lam": e, "seconds": o["seconds"],
                                "rank_used": o["rank_used"],
                                "kappa_M": o["kappa_M"]})
        for m in range(1, n_pen + 1):
            c = cnt[m]["correct"]
            lo, hi = mt.wilson(c, n_seeds)
            row = {"arm": f"f_eig_pielm(nf={nf})", "mode": m, "ladder": "n/a",
                   "nodes": "gauss", "n": n_seeds, "p_correct": c / n_seeds,
                   "wilson_lo": lo, "wilson_hi": hi, "correct": c,
                   "lower_mode_basin": 0, "spurious": cnt[m]["spurious"],
                   "non_converged": cnt[m]["non_converged"],
                   "p_accurate": c / n_seeds, "n_certified": c,
                   "seconds": float(np.median(secs)),
                   "e_lam": float(np.nanmedian(elams[m])),
                   "rank_used": float(np.median(ranks)),
                   "kappa_M": float(np.median(kappas))}
            # 유효 rank는 절단 임계에 의존한다 — 단일 정수로 보고하면 임계가 숨는다
            for t in spectra[0]:
                row[f"rank_tol_{t}"] = float(np.median([sp[t] for sp in spectra]))
            mode_rows.append(row)

    # (8) 비용 — 실패를 청구한 기대비용
    for row in mode_rows + pen_rows + quad_rows + ladder_rows:
        cost_rows.append({"arm": row["arm"], "mode": row["mode"],
                          "ladder": row.get("ladder", "I0"),
                          "nodes": row.get("nodes", "gauss"),
                          "mean_seconds": row["seconds"],
                          "p_correct": row["p_correct"],
                          "E_T_success": expected_time_to_success(
                              row["seconds"], row["p_correct"])})

    manifest.write_csv(os.path.join(d, "p1_neural_mode_sweep.csv"), mode_rows)
    manifest.write_csv(os.path.join(d, "p1_neural_penalty_sweep.csv"), pen_rows)
    manifest.write_csv(os.path.join(d, "p1_neural_quadrature.csv"), quad_rows)
    manifest.write_csv(os.path.join(d, "p1_neural_ladder.csv"), ladder_rows)
    manifest.write_csv(os.path.join(d, "p1_neural_cost.csv"), cost_rows)
    manifest.write_jsonl(os.path.join(d, "p1_neural_records.jsonl"), records)

    import torch
    manifest.write_json(os.path.join(d, "manifest_p1_neural.json"),
                        manifest.build({
                            "driver": "run_p1_neural", "quick": quick,
                            "n_seeds": n_seeds, "iters": iters,
                            "mode_max": n_modes, "penalty_weights": list(weights),
                            "torch": torch.__version__,
                            "gpu": (torch.cuda.get_device_name(0)
                                    if torch.cuda.is_available() else "cpu"),
                            "mac_min": mt.MAC_MIN, "elam_max": mt.ELAM_MAX,
                            "m_star": m_star,
                            "grid_note": ("regrid of the v3 §5.1 cell layout: exact projection at mode 3 / I0 "
                                          "saturates at 50/50 while pure penalty is 0/50, "
                                          "so the primary axis was changed to mode number "
                                          "x arm. Rationale in the module docstring.")}))
    return {"records": records, "mode_sweep": mode_rows, "penalty": pen_rows,
            "quadrature": quad_rows, "ladder": ladder_rows, "cost": cost_rows,
            "outdir": d, "m_star": m_star}


if __name__ == "__main__":
    out = main()
    print(f"[run_p1_neural] {out['outdir']} — 레코드 {len(out['records'])}, "
          f"모드스윕 {len(out['mode_sweep'])}행, m*={out['m_star']}")


LONG_ITERS = 20000


def run_long(outdir=None, base_iters: int = 4000, long_iters: int = LONG_ITERS,
             n_seeds: int = 50, quick: bool = False, device=None,
             arms: tuple = ("sequential", "subspace"), append: bool = False) -> dict:
    """§7 장기변주 — **"예산 부족"과 "basin 포획"을 가른다.**

    기본 격자(4000 반복)에서 성공률이 100 %가 아니었던 셀만 골라 5배 예산으로 다시 돈다.
    설계서는 "헤드라인 3셀"을 말했지만, 실제로 실패가 관측된 지점을 정조준하는 편이
    예산 대비 정보가 크다. 셀 선택은 기본 격자 CSV에서 **데이터로 결정**하며 임의가 아니다.

    이미 100 %인 저차모드는 기본 예산 그대로 두고 실패 모드에만 큰 예산을 준다
    (순차 deflation이라 하위모드를 먼저 잡아야 하지만, 그것들은 이미 수렴한다).

      docker run ... gb10-pinn:26.04 python3 -u -c \
        "from eigen_benchmark.drivers.run_p1_neural import run_long; run_long()"
    """
    import csv

    from ..neural.subspace import (solve_neural_basis_galerkin,
                                   solve_simultaneous)

    d = manifest.ensure_outdir(outdir)
    if quick:
        base_iters, long_iters, n_seeds = 200, 600, 4

    base_path = os.path.join(d, "p1_neural_mode_sweep.csv")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"기본 격자 결과가 없습니다: {base_path} "
                                "— run_p1_neural.main()을 먼저 실행하세요")
    base = list(csv.DictReader(open(base_path, encoding="utf-8")))

    def failing(arm_prefix, thresh: float = 0.9, n_worst: int = 2):
        """장기변주 대상 모드 선택규칙(사전선언).

        1) 성공률 < thresh 인 모드만 후보로 삼는다(1.00 미만 전부가 아니다 —
           0.98은 예산 문제가 아니다).
        2) 후보가 **모두 붕괴**(≤ 0.05)면 **가장 낮은 모드 하나만** 쓴다. 순차 deflation은
           체인이라 첫 실패 이후는 그 하류 결과이고 5배 예산을 거기 쓰는 것은 낭비다.
        3) 아니면 가장 나쁜 n_worst개를 쓴다(완만한 저하의 실제 한계를 보는 것이 목적).

        이 규칙 없이 "1.00 미만 전부"를 쓰면 실측 스테이지 비용으로 약 10시간,
        이 규칙으로는 약 4시간이다."""
        cand = [(int(r["mode"]), float(r["p_correct"])) for r in base
                if r["arm"].startswith(arm_prefix)
                and float(r["p_correct"]) < thresh and int(r["n"]) > 1]
        if not cand:
            return []
        if all(p <= 0.05 for _, p in cand):
            return [min(m for m, _ in cand)]
        return sorted(m for m, _ in sorted(cand, key=lambda t: t[1])[:n_worst])

    def supersede(prefix: str):
        """재계산하는 arm의 이전 행·레코드를 **대체**한다(중복 방치 금지).

        그냥 이어붙이면 설정이 바뀐 재실행 결과가 옛 결과 옆에 남아 서로 모순되는 두 행이
        생긴다. 실제로 (e)를 K=3 교란 상태로 돌렸다가 K=6으로 재계산했을 때 모드1이
        0.92와 0.02로 동시에 존재했다 — 표를 읽는 사람이 어느 쪽을 믿을 수 없다."""
        rows[:] = [r for r in rows if not str(r.get("arm", "")).startswith(prefix)]
        records[:] = [r for r in records
                      if not str(r.get("arm", "")).startswith(prefix)]

    def flush(tag: str):
        """**셀마다 증분 기록.** run_long에 체크포인트가 없어 마지막 기록 단계의
        TypeError로 2h11m치를 잃었다(main()에는 있었는데 여기엔 없었다)."""
        if rows:
            manifest.write_csv(os.path.join(d, "p1_neural_longrun.csv"), rows)
        if records:
            manifest.write_jsonl(
                os.path.join(d, "p1_neural_longrun_records.jsonl"), records)
        print(f"  [longrun:{tag}] 셀 {len(rows)}, 레코드 {len(records)}", flush=True)

    records, rows = [], []
    if append:                       # 이미 유효한 셀을 재계산하지 않고 이어붙인다
        import json
        prev = os.path.join(d, "p1_neural_longrun.csv")
        if os.path.exists(prev):
            rows += manifest.read_csv(prev)
        prev_j = os.path.join(d, "p1_neural_longrun_records.jsonl")
        if os.path.exists(prev_j):
            records += [json.loads(l) for l in open(prev_j, encoding="utf-8")]
    plan = []
    seq_arms = ((("a_prime_projection_diagonal", df.ProjectionDiagonal()),
                 ("b_projection_exact", df.ProjectionExact()))
                if "sequential" in arms else ())
    for prefix, defl in seq_arms:
        bad = failing(prefix)
        if bad:
            top = max(bad)
            budget = [long_iters if (m + 1) in bad else base_iters
                      for m in range(top)]
            plan.append((prefix, defl, top, budget, bad))

    for prefix, defl, top, budget, bad in plan:
        supersede(prefix)
        out = sq.solve_sequential(top, defl, n_seeds=n_seeds, iters=budget,
                                  device=device)
        for m in bad:
            r, agg = _cell(out, m, prefix, "I0", "gauss")
            agg["variant"] = "long_run"
            agg["base_iters"] = base_iters
            for rec in r:
                rec["variant"] = "long_run"
            records += r
            rows.append(agg)
        flush(f"seq:{prefix[:12]}")

    # 부분공간 arm은 **기본 격자와 같은 K로** 돌려야 한다.
    # max(실패모드)를 K로 쓰면 trace가 합산하는 고유값 범위가 달라져(λ₁/λ₆≈1.4e-4 →
    # λ₁/λ₃≈3.2e-3) 저모드 경시가 46배 완화된다 → 반복 5배 효과와 K 축소 효과가 섞인다.
    # 기본 격자의 K는 그 arm이 보고한 최대 모드에서 읽는다.
    def base_n_modes(arm_prefix):
        ms = [int(r["mode"]) for r in base if r["arm"].startswith(arm_prefix)]
        return max(ms) if ms else 0

    sub_arms = ((("e_simultaneous_subspace", solve_simultaneous),
                 ("d_neural_basis_galerkin", solve_neural_basis_galerkin))
                if "subspace" in arms else ())
    for prefix, solver in sub_arms:
        sub_bad = failing(prefix)
        if not sub_bad:
            continue
        supersede(prefix)
        K = base_n_modes(prefix)
        out = solver(K, n_seeds=n_seeds, iters=long_iters, device=device)
        for m in sub_bad:
            r, agg = _cell(out, m, out["arm"], "I0", "gauss")
            agg["variant"] = "long_run"
            agg["base_iters"] = base_iters
            agg["n_modes_solved"] = K
            for rec in r:
                rec["variant"] = "long_run"
            records += r
            rows.append(agg)
        flush(f"sub:{prefix[:12]}")

    if not rows:
        print("[run_long] 기본 격자에 실패 셀이 없어 장기변주가 불필요합니다")
        return {"longrun": [], "records": [], "outdir": d}

    # 기본 격자와 나란히 놓아 "예산으로 해결되는가"를 바로 읽게 한다
    for row in rows:
        m = [r for r in base
             if r["arm"] == row["arm"] and int(r["mode"]) == row["mode"]]
        row["p_correct_base"] = float(m[0]["p_correct"]) if m else float("nan")
        row["budget_resolved"] = bool(row["p_correct"] > row["p_correct_base"])

    manifest.write_csv(os.path.join(d, "p1_neural_longrun.csv"), rows)
    manifest.write_jsonl(os.path.join(d, "p1_neural_longrun_records.jsonl"), records)
    manifest.write_json(os.path.join(d, "manifest_p1_neural_longrun.json"),
                        manifest.build({"driver": "run_p1_neural.run_long",
                                        "base_iters": base_iters,
                                        "long_iters": long_iters,
                                        "n_seeds": n_seeds, "quick": quick,
                                        "cells": [f"{r['arm']}@m{r['mode']}"
                                                  for r in rows]}))
    print(f"[run_long] {d} — 장기변주 {len(rows)}셀, 레코드 {len(records)}")
    return {"longrun": rows, "records": records, "outdir": d}


ORTH_WEIGHTS = (1e3, 1e4, 1e5, 1e6)


def run_orth_ablation(outdir=None, w_orths=ORTH_WEIGHTS, n_seeds: int = 50,
                      iters: int = 4000, quick: bool = False, device=None) -> dict:
    """(d) rank 붕괴의 귀속 확인 ablation — 직교성 압력 부재가 원인인가.

    기본 격자에서 (d)의 유효 rank가 4로 붕괴했고 예산 5배로도 5까지만 올랐는데,
    **rank가 충족된 모드조차 정확도가 0**이었다(20000반복 rank 5, 모드5 정확도 0.00).
    즉 rank 개수가 병목이 아니라 주변 방향의 근사능력이 없다.

    가설: trace 목적함수가 span만 보므로 기저함수 분리에 기울기 압력이 없고, 거의 겹친
    두 함수의 차는 질량·강성이 동시에 작아 Ritz 기여가 유한하게 남아 벌점도 없다.
    그람 비대각 벌점으로 그 압력을 공급하면 회복되는가.

    가중치는 (a) 페널티와 같이 스윕한다 — trace가 ~1e5 규모라 유효 범위를 미리 알 수 없다.
    회복되면 귀속 확정, 안 되면 원인은 표현력 자체다. **새 방법 제안이 아니라 진단이다.**
    """
    from ..neural.subspace import solve_neural_basis_galerkin

    d = manifest.ensure_outdir(outdir)
    if quick:
        n_seeds, iters, w_orths = 4, 150, (1e4,)
    base = manifest.read_csv(os.path.join(d, "p1_neural_mode_sweep.csv"))
    n_modes = max([int(r["mode"]) for r in base
                   if str(r["arm"]).startswith("d_neural_basis_galerkin")] or [6])

    records, rows = [], []
    for w in w_orths:
        out = solve_neural_basis_galerkin(n_modes, n_seeds=n_seeds, iters=iters,
                                          w_orth=w, device=device)
        for m in range(1, n_modes + 1):
            r, agg = _cell(out, m, out["arm"], "I0", "gauss")
            agg.update({"variant": "orth_ablation", "w_orth": w,
                        "n_basis": out["n_basis"]})
            for rec in r:
                rec["variant"] = "orth_ablation"
                rec["w_orth"] = w
            records += r
            rows.append(agg)
        manifest.write_csv(os.path.join(d, "p1_neural_orth_ablation.csv"), rows)
        manifest.write_jsonl(
            os.path.join(d, "p1_neural_orth_ablation_records.jsonl"), records)
        print(f"  [orth:{w:g}] rank_median={out['rank_median']:.1f} "
              f"rank_min={out['rank_min']} 셀 {len(rows)}", flush=True)

    # 기본 격자(w_orth=0)와 나란히
    zero = {int(r["mode"]): r for r in base
            if str(r["arm"]).startswith("d_neural_basis_galerkin")}
    for row in rows:
        z = zero.get(row["mode"])
        row["p_accurate_w0"] = float(z["p_accurate"]) if z else float("nan")
        row["rank_median_w0"] = float(z.get("rank_median", "nan")) if z else float("nan")

    manifest.write_csv(os.path.join(d, "p1_neural_orth_ablation.csv"), rows)
    manifest.write_json(os.path.join(d, "manifest_p1_neural_orth.json"),
                        manifest.build({"driver": "run_p1_neural.run_orth_ablation",
                                        "w_orths": list(w_orths), "n_seeds": n_seeds,
                                        "iters": iters, "n_modes": n_modes,
                                        "purpose": "(d) rank 붕괴 귀속 확인 ablation"}))
    print(f"[run_orth_ablation] {d} — {len(rows)}셀, 레코드 {len(records)}")
    return {"orth": rows, "records": records, "outdir": d}


def run_collinearity_diagnostic(outdir=None, n_seeds: int = 16, iters: int = 4000,
                                w_orths=(0.0, 1e6), quick: bool = False,
                                device=None) -> dict:
    """(d) 실패 기전 진단 — 기저함수가 **공선으로 붕괴**하는가.

    가설 둘이 반증된 뒤 남은 설명이다.
      · "더 훈련하면 rank가 더 붕괴" → 4→5로 소폭 개선(반증)
      · "직교성 압력 부재가 원인" → 벌점을 trace의 70배로 넣어도 무효(반증)

    측정으로 확정한 것: M=9 기저함수가 **거의 하나의 함수로 겹친다**.
    근거 (1) 상관 비대각 벌점 71.94 ≈ 이론최대 M²−M = 72,
         (2) λ_max(B) = 34 ≈ Σ‖φᵢ‖² = 34.1 (완전 공선일 때만 성립), 2번째 고유값은 6자리 아래,
         (3) 노름은 O(0.1–3)으로 고르므로 "0으로 수축"이 아니다.

    기전: 각 망이 받는 trace 기울기는 본질적으로 "자기 Rayleigh 몫을 낮춰라"이고 그
    최소점은 아홉 모두에게 모드1이다. 서로 다른 모드로 갈라놓아야 할 B⁻¹ 결합은 이미
    공선인 상태에서 질량·강성이 동시에 작아 0/0으로 복원력을 잃는다 → **퇴화 끌개**.

    (e) 공유 trunk도 같은 지표로 재서 **통제된 대조**를 만든다. K출력은 같은 특징의 서로
    다른 선형 읽기이므로 공선이 되려면 읽기 행렬이 rank-1이어야 하고, 그건 trace가 실제로
    벌점을 주는 구성이다 → 끌개를 구조적으로 회피한다.

    공선성 지표 = λ_max(B)/tr(B). 1이면 완전 공선, 1/M이면 고르게 퍼짐.
    """
    import torch
    from torch.func import vmap

    from ..neural import core
    from ..neural import subspace as sub

    d = manifest.ensure_outdir(outdir)
    if quick:
        n_seeds, iters, w_orths = 4, 200, (0.0,)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    xs, wq = core.gauss_nodes(256, dev)
    K, M = 6, 9
    rows, shapes = [], {}

    def measure(Phi, label, w_orth, secs):
        """Phi: (n_seeds, n_basis, n_q) — 시드별 공선성 지표.

        **척도무관 지표를 정본으로 쓴다.** 원시 그람행렬의 λ_max/tr은 노름
        불균형에 오염된다 — 함수 하나가 trace를 독차지하면 공선성이 전혀 없어도
        1에 가까워진다(실측 norm_max/min 중앙 35, 최대 643). 그래서 상관행렬
        C = D^{-1/2} B D^{-1/2}(대각이 1)에서 같은 양을 다시 재고, 그 값
        `corr_index = λ_max(C)/n_basis`를 결론의 근거로 삼는다. 1이면 모든
        쌍의 |ρ|=1(방향이 하나), 1/n_basis면 서로 직교다.
        """
        with torch.no_grad():
            nrm = torch.sqrt((wq * Phi ** 2).sum(-1))
            B = torch.einsum("smi,i,sni->smn", Phi, wq, Phi)
            ev = torch.linalg.eigvalsh(B)
            dg = torch.sqrt(torch.clamp(torch.diagonal(B, dim1=-2, dim2=-1),
                                        min=1e-300))
            C = B / (dg.unsqueeze(-1) * dg.unsqueeze(-2))
            evc = torch.linalg.eigvalsh(C)
            pen = vmap(lambda a: sub.gram_offdiagonal_penalty(a, wq))(Phi)
        n_basis = Phi.shape[1]
        for s in range(Phi.shape[0]):
            e = ev[s].cpu().numpy()
            ec = evc[s].cpu().numpy()
            tr = float(e.sum())
            rows.append({
                "arm": label, "w_orth": float(w_orth), "seed": s,
                "n_basis": n_basis,
                # --- 척도무관(정본) ---
                "corr_index": float(ec.max() / n_basis),
                "corr_eig_max": float(ec.max()),
                "corr_eig_2nd": float(sorted(ec)[-2]),
                "corr_ratio_2nd_to_max": float(sorted(ec)[-2] / ec.max()),
                "corr_rank_1e12": int((ec > ec.max() * 1e-12).sum()),
                "offdiag_penalty": float(pen[s]),
                "offdiag_penalty_max": float(n_basis ** 2 - n_basis),
                "offdiag_ratio": float(pen[s]) / float(n_basis ** 2 - n_basis),
                # --- 원시 그람(노름 불균형에 오염됨 — 참고용) ---
                "raw_index_confounded": float(e.max() / tr) if tr > 0 else float("nan"),
                "eig_max": float(e.max()),
                "eig_2nd": float(sorted(e)[-2]),
                "eig_ratio_2nd_to_max": float(sorted(e)[-2] / e.max()),
                "gram_trace": tr,
                "sum_sq_norms": float((nrm[s] ** 2).sum()),
                "norm_max_over_min": float(nrm[s].max() / nrm[s].min()),
                "rank_1e12": int((e > e.max() * 1e-12).sum()),
                "seconds": secs})
        shapes[f"{label}|w_orth={w_orth:g}"] = Phi.cpu().numpy()

    # (d) 분리된 M개 망 — w_orth 유무로
    for w in w_orths:
        out = sub.solve_neural_basis_galerkin(K, n_basis=M, n_seeds=n_seeds,
                                              iters=iters, w_orth=w, device=dev)
        # **원시 기저**로 측정한다 — 추출된 모드는 구성상 B-직교라 공선성을 잴 수 없다.
        Phi = torch.tensor(out["basis_shapes"], device=dev)      # (seed, M, nq)
        measure(Phi, f"d_neural_basis_galerkin(M={M})", w, out["seconds"])
        print(f"  [collin d w={w:g}] rank_median={out['rank_median']:.1f}", flush=True)

    # (e) 공유 trunk — 대조군
    oute = sub.solve_simultaneous(K, n_seeds=n_seeds, iters=iters, device=dev)
    Phie = torch.tensor(oute["basis_shapes"], device=dev)       # (seed, K, nq)
    measure(Phie, "e_simultaneous_subspace", 0.0, oute["seconds"])
    print(f"  [collin e] rank_median={oute['rank_median']:.1f}", flush=True)

    # 원시 기저를 남긴다 — 지표를 고칠 때마다 20분씩 재학습하지 않도록.
    np.savez_compressed(os.path.join(d, "p1_neural_collinearity_basis.npz"),
                        xs=xs.cpu().numpy(), wq=wq.cpu().numpy(), **shapes)
    manifest.write_csv(os.path.join(d, "p1_neural_collinearity.csv"), rows)
    manifest.write_json(os.path.join(d, "manifest_p1_neural_collinearity.json"),
                        manifest.build({
                            "driver": "run_p1_neural.run_collinearity_diagnostic",
                            "n_seeds": n_seeds, "iters": iters,
                            "w_orths": list(w_orths), "K": K, "M": M,
                            "metric": "corr_index = eig_max(C)/n_basis (C = 상관행렬, "
                                      "척도무관·정본); 1 = 모든 쌍 |rho|=1, "
                                      "1/n_basis = 직교. raw_index_confounded = "
                                      "eig_max(B)/trace(B) 는 노름 불균형에 오염되므로 "
                                      "참고용",
                            "basis_npz": "p1_neural_collinearity_basis.npz"}))
    print(f"[run_collinearity_diagnostic] {d} — {len(rows)}행")
    return {"collinearity": rows, "outdir": d}


# (f) PIELM은 `run_p1_pielm.py`로 분리했다 — torch를 import하지 않는 numpy 전용 경로라야
# 고전 기저와 같은 환경(호스트 numpy/CPU)에서 측정된다. 비용 회계의 요구사항이다.


def run_audit_followups(outdir=None, n_seeds: int = 50, iters: int = 4000,
                        quick: bool = False, device=None) -> dict:
    """감사가 지적한 세 공백을 실측으로 메운다.

    (1) **vmap 배칭 계수** — 비용표의 `seconds`는 시드당이 아니라 n_seeds 앙상블
        전체의 벽시계다. PIELM 단일해와 비교하는 것이 정당한지는 "배칭이 거의
        공짜인가"에 달려 있는데, 그 근거가 커밋된 벤치마크로 없었다. n_seeds를
        바꿔가며 재서 배칭 계수를 데이터로 남긴다.
    (2) **MC 스냅샷 잡음 하한** — "≈7 %, 기준의 70배"는 어디서도 계산되지 않은
        주장이었다. 확률적 목적함수의 스냅샷 간 상대변화 분포를 직접 재서
        사전등록 임계 1e-3과 나란히 놓는다.
    (3) **a′ × MC 대조군** — "구적이 원인이 아니다"가 파일럿(MC)과 a′(Gauss)의
        교차구현 비교에 의존했다. a′를 MC로 직접 돌려 자체 대조군을 만든다.
    """
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_seeds, iters = 4, 200
    rows = []

    # (1) 배칭 계수 — 같은 일을 시드 수만 바꿔 **반복** 측정한다.
    #
    # 한 번만 재면 값이 흔들린다: n=1이 서로 다른 실행에서 39.5 s와 26.5 s로 나와 계수가
    # 2.53과 3.80이 됐다(GPU 워밍업·다른 작업과의 경합). 비용 회계 전체가 이 계수에
    # 걸려 있으므로 반복해 중앙값과 산포를 함께 보고하고, 논문의 비율은 그 불확실도를
    # 명시한 채 인용한다.
    seed_counts = (1, 2, 8, n_seeds) if not quick else (1, 2)
    reps = 1 if quick else 3
    med = {}
    for ns in seed_counts:
        ts = []
        for _ in range(reps):
            out = sq.solve_sequential(1, df.ProjectionExact(), n_seeds=ns,
                                      iters=iters, device=device)
            ts.append(float(out["stage_seconds"][0]))
        ts.sort()
        med[ns] = ts[len(ts) // 2]
        rows.append({"study": "vmap_batching", "n_seeds": ns, "iters": iters,
                     "reps": reps, "seconds_median": med[ns],
                     "seconds_min": ts[0], "seconds_max": ts[-1],
                     "spread_rel": (ts[-1] - ts[0]) / max(med[ns], 1e-300),
                     "seconds_per_seed": med[ns] / ns,
                     "note": "same work (mode 1, exact projection) with only the seed "
                             "count changed; median of repeated measurements"})
        print(f"  [batch n={ns}] median {med[ns]:.1f}s "
              f"[{ts[0]:.1f}, {ts[-1]:.1f}]  per seed {med[ns] / ns:.2f}s", flush=True)
    if 1 in med and n_seeds in med:
        bf = med[n_seeds] / med[1]
        rows.append({"study": "batch_factor", "n_seeds": n_seeds, "iters": iters,
                     "reps": reps, "batch_factor": bf,
                     "speedup_vs_serial": med[1] * n_seeds / med[n_seeds],
                     "note": "ensemble wall-clock divided by a single-seed run; this is "
                             "the factor used to convert ensemble time to per-attempt "
                             "cost in the cost tables"})
        print(f"  [batch_factor] {bf:.2f}  (직렬 대비 "
              f"{med[1] * n_seeds / med[n_seeds]:.1f}배)", flush=True)

    # (2) MC 잡음 하한 — 사전등록 규칙과 같은 꼬리 구간에서
    for nodes in ("gauss", "mc"):
        out = sq.solve_sequential(1, df.ProjectionExact(), n_seeds=n_seeds,
                                  iters=iters, nodes=nodes, device=device)
        h = out["history"][0] if isinstance(out["history"][0], list) else out["history"]
        vals = np.array([np.asarray(v) for _, _, v in h])       # (n_snap, n_seeds)
        k = max(int(len(h) * 0.1), 2)                            # 규칙과 같은 꼬리
        tail = vals[-k:]
        step = np.abs(np.diff(tail, axis=0)) / np.maximum(np.abs(tail[1:]), 1e-300)
        rule = np.abs(vals[-1] - vals[-k]) / np.maximum(np.abs(vals[-1]), 1e-300)
        rows.append({"study": "snapshot_noise_floor", "nodes": nodes,
                     "n_seeds": n_seeds, "iters": iters,
                     "n_snapshots_in_tail": int(k),
                     "step_rel_change_median": float(np.median(step)),
                     "step_rel_change_p95": float(np.percentile(step, 95)),
                     "rule_rel_change_median": float(np.median(rule)),
                     "rule_rel_change_p95": float(np.percentile(rule, 95)),
                     "prereg_tol": 1e-3,
                     "ratio_median_to_tol": float(np.median(rule)) / 1e-3,
                     "note": "same trailing 10 % window the pre-registered rule inspects"})
        print(f"  [noise {nodes}] 규칙값 중앙 {np.median(rule):.3e} "
              f"(임계 1e-3의 {np.median(rule) / 1e-3:.1f}배)", flush=True)

    # (3) a′ × MC 자체 대조군
    n_m = min(3, 3)
    for nodes in ("mc",):
        out = sq.solve_sequential(n_m, df.ProjectionDiagonal(), n_seeds=n_seeds,
                                  iters=iters, nodes=nodes, device=device)
        for m in range(1, n_m + 1):
            _, agg = _cell(out, m, "a_prime_projection_diagonal", "I0", nodes)
            agg["study"] = "a_prime_mc_control"
            rows.append(agg)
            print(f"  [a' mc mode{m}] p_correct={agg['p_correct']:.2f} "
                  f"p_accurate={agg['p_accurate']:.2f}", flush=True)

    manifest.write_csv(os.path.join(d, "p1_neural_audit.csv"), rows)
    manifest.write_json(os.path.join(d, "manifest_p1_neural_audit.json"),
                        manifest.build({
                            "driver": "run_p1_neural.run_audit_followups",
                            "n_seeds": n_seeds, "iters": iters,
                            "studies": ["vmap_batching", "snapshot_noise_floor",
                                        "a_prime_mc_control"]}))
    print(f"[run_audit_followups] {d} — {len(rows)}행")
    return {"audit": rows, "outdir": d}


# 사다리 준위를 warm/cold 축으로 읽는 방법. 범위문서 §7이 논문 2로 지정한
# "PINN warm-start/cold-start 비교"는 이 축이다.
# **논문에 인쇄되는 문자열은 영문이다** — 이 딕트의 두 번째 항목이 Table 5의
# prior_information 열로 그대로 나간다. 한국어를 두면 영어 원고에 박힌다.
WARM_COLD = {
    "I0": ("cold", "random network initialization; no prior information"),
    "I1": ("cold", "BC-satisfying random polynomial; structural prior only, "
                   "no information about the target mode"),
    "I2": ("warm-lambda", "eigenvalue window only, as a barrier on the objective; "
                          "no shape information"),
    "I3": ("warm-shape", "coarse-Ritz solution pre-fitted in value and curvature"),
    "I4": ("warm-staged", "curriculum staging, consumed internally by arm (c)"),
    "I5": ("warm-oracle", "analytic mode shape and curvature pre-fitted; upper bound"),
}


def run_warm_cold(outdir=None, n_seeds: int = 50, iters: int = 4000,
                  modes=(1, 3), levels=("I0", "I1", "I2", "I3", "I5"),
                  out_name: str = "p1_neural_warm_cold",
                  quick: bool = False, device=None) -> dict:
    """warm-start 이득이 **arm에 의존하는가** — (a′)와 (b) 둘 다 사다리를 돌린다.

    기존 격자는 (a′) 한 arm의 사다리만 돌렸다(`p1_neural_ladder.csv` 4행). 그것만으로는
    "사전정보가 도움이 되는가"에 답할 수 없다 — (a′)는 모드 3에서 어떤 준위로도 0.00이라
    사다리가 **전부 무효**로 보이지만, 그것은 사전정보의 성질이 아니라 deflation 기전의
    성질이다. 정확사영 (b)에서도 같은지 봐야 비로소 축이 분리된다.

    준위를 warm/cold로 읽는 규칙은 `WARM_COLD`에 있고, 사전정보의 **종류**까지 구분한다 —
    고유값 구간만 아는 것(I2)과 형상을 아는 것(I3·I5)은 다른 자원이다.
    """
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_seeds, iters, modes, levels = 4, 200, (1,), ("I0", "I5")
    rows, records = [], []
    arms = (("a_prime_projection_diagonal", df.ProjectionDiagonal()),
            ("b_projection_exact", df.ProjectionExact()))
    for arm_name, defl in arms:
        for m in modes:
            for lvl in levels:
                init = ld.make_ladder(lvl, mode=m, iters=max(iters // 2, 200))
                kw = {}
                if lvl == "I2":
                    # I2는 초기화가 아니라 목적함수에 붙는 창 배리어다
                    kw["barrier"] = init
                    init = None
                out = sq.solve_sequential(m, defl, n_seeds=n_seeds, iters=iters,
                                          init=init, device=device, **kw)
                r, agg = _cell(out, m, arm_name, lvl, "gauss")
                kind, why = WARM_COLD[lvl]
                agg.update({"start": kind, "prior_information": why,
                            "study": "warm_cold"})
                records += r
                rows.append(agg)
                print(f"  [{arm_name.split('_')[0]} m{m} {lvl}/{kind}] "
                      f"p_correct={agg['p_correct']:.2f} "
                      f"p_accurate={agg['p_accurate']:.2f}", flush=True)

    manifest.write_csv(os.path.join(d, f"{out_name}.csv"), rows)
    manifest.write_jsonl(os.path.join(d, f"{out_name}_records.jsonl"), records)
    manifest.write_json(os.path.join(d, f"manifest_{out_name}.json"),
                        manifest.build({
                            "driver": "run_p1_neural.run_warm_cold",
                            "n_seeds": n_seeds, "iters": iters,
                            "modes": list(modes), "levels": list(levels),
                            "arms": [a for a, _ in arms],
                            "warm_cold_map": {k: v[0] for k, v in WARM_COLD.items()}}))
    print(f"[run_warm_cold] {d} — {len(rows)}행")
    return {"warm_cold": rows, "outdir": d}


def summarize_warm_cold(outdir=None, budgets=(200, 400, 4000)) -> dict:
    """예산별 warm/cold 표를 하나로 합쳐 **손익분기**를 보인다.

    사다리를 4000반복에서만 돌리면 모든 준위가 1.00으로 천장에 붙어 사전정보의 효과를
    검출할 여유가 없다. 예산을 좁히면 cold가 실패하기 시작하고 거기서 비로소 축이 분리된다.

    그리고 **사전적합 비용을 청구해야** 결론이 뒤집히지 않는다 — 본학습 단계만 세면
    oracle(I5)이 최고로 보이지만, 사전적합을 넣으면 cold보다 비싸다.
    """
    from .run_p1_compare import _base, _single_seed_cost
    d = manifest.ensure_outdir(outdir)
    direct = _single_seed_cost(d)
    src = {200: "p1_neural_warm_cold_budget200",
           400: "p1_neural_warm_cold_budget400",
           4000: "p1_neural_warm_cold"}
    # **1회/앙상블 비를 예산 4000에서 arm마다 한 번 구해 다른 예산에도 같은 비율로 쓴다.**
    #
    # 첫 판은 `scale = single_seed / ensemble_b`를 예산마다 계산하고 `tr = ensemble_b *
    # scale`을 했는데, 그러면 tr이 항상 single_seed와 같아져 **예산과 무관한 상수**가 되고
    # (200/400/4000이 모두 127 s), 사전적합은 반대로 예산이 커질수록 나눠져 줄어들었다.
    # 물리적으로 불가능한 표가 나왔고 E[T] 최소가 4000으로 옮겨가 이 절의 실험 논리와도
    # 충돌했다. 비는 **한 예산에서만** 측정된 양이므로 비율로 옮겨야 한다.
    ratio = {}
    ref_path = os.path.join(d, f"{src[4000]}.csv")
    if os.path.exists(ref_path):
        for r in manifest.read_csv(ref_path):
            if int(r["mode"]) != 3:
                continue
            hit = direct.get((_base(r["arm"]), 3))
            if hit and float(r["seconds"]) > 0:
                ratio[_base(r["arm"])] = hit[0] / float(r["seconds"])

    rows = []
    for b in budgets:
        path = os.path.join(d, f"{src[b]}.csv")
        if not os.path.exists(path):
            print(f"  [건너뜀] {src[b]}.csv 없음 — 해당 예산 미실행", flush=True)
            continue
        for r in manifest.read_csv(path):
            if int(r["mode"]) != 3:
                continue
            p = float(r["p_correct"])
            # **비용표와 같은 회계**: 앙상블 초를 환산하지 않고 n_seeds=1 직접 측정을
            # 쓴다. 환산계수는 같은 기계에서 2.53과 7.82로 나와 쓸 수 없다.
            # 예산이 다른 셀은 반복수에 비례해 맞춘다(같은 arm·같은 모드).
            sc = ratio.get(_base(r["arm"]), float("nan"))
            tr = float(r["seconds"]) * sc
            ini = float(r.get("init_seconds", 0.0)) * sc
            rows.append({
                "budget_iters": b, "arm": r["arm"], "mode": 3,
                "ladder": r["ladder"], "start": r.get("start", ""),
                "prior_information": r.get("prior_information", ""),
                "p_correct": p, "p_accurate": float(r["p_accurate"]),
                "train_seconds_per_attempt": tr,
                "init_seconds_per_attempt": ini,
                "total_seconds_per_attempt": tr + ini,
                "ensemble_seconds_reported": float(r["seconds"]),
                "per_attempt_scale": sc,
                "per_attempt_source": "p1_single_seed_cost.csv (n_seeds = 1)",
                "E_T_success": expected_time_to_success(tr + ini, p),
                # 사전적합을 청구하지 않았을 때의 값 — 회계가 결론을 뒤집는다는 증거
                "E_T_success_if_init_free": expected_time_to_success(tr, p)})
    if not rows:
        raise FileNotFoundError("합칠 warm/cold 표가 없습니다")
    # 예산별로 cold(I0) 대비 이득을 붙인다
    base = {(r["budget_iters"], r["arm"]): r["E_T_success"]
            for r in rows if r["ladder"] == "I0"}
    for r in rows:
        b0 = base.get((r["budget_iters"], r["arm"]))
        r["speedup_vs_cold"] = (b0 / r["E_T_success"]
                                if b0 and math.isfinite(r["E_T_success"])
                                and r["E_T_success"] > 0 else float("nan"))
        r["beats_cold"] = bool(r["speedup_vs_cold"] > 1.0)
    manifest.write_csv(os.path.join(d, "p1_warm_cold_summary.csv"), rows)
    manifest.write_json(os.path.join(d, "manifest_p1_warm_cold_summary.json"),
                        manifest.build({
                            "driver": "run_p1_neural.summarize_warm_cold",
                            "budgets": list(budgets),
                            "per_attempt_ratio": "single-seed / ensemble measured at "
                                                 "4000 iters per arm, applied as a ratio "
                                                 "to the other budgets",
                            "prefit_budget_note": "the pre-fit uses max(iters//2, 200) "
                                                  "iterations, so it is tied to the main "
                                                  "budget by design and is not constant "
                                                  "across budgets",
                            "note": "E[T] charges the pre-fit. "
                                    "E_T_success_if_init_free is the value when it is not "
                                    "charged, reported alongside as evidence that the "
                                    "accounting choice reverses the ranking."}))
    print(f"[summarize_warm_cold] {d} — {len(rows)}행")
    return {"summary": rows, "outdir": d}


def run_e_objective_ablation(outdir=None, n_seeds: int = 50, iters: int = 4000,
                             n_modes: int = 6, objectives=("trace", "logtrace"),
                             quick: bool = False, device=None) -> dict:
    """(e)의 모드1 악화 기전 검정 — **목적함수의 가중이 원인인가**.

    관측: (e)는 모드 순서와 정확도가 뒤집혀 있다(모드1 e_λ 4.9e-2 > 모드4 2.2e-3)이고,
    예산을 5배로 늘리면 모드1만 더 나빠진다. MAC은 전 모드에서 1에 붙어 있다.

    가설 A(목적함수 가중): 절대 trace Σλ_k는 ∂/∂λ_k = 1이라 모드1을 7 % 희생해 얻는
        0.87을 모드6의 0.1 % 개선(89)이 100배로 갚는다 — 최저 모드를 버리는 것이 이득이다.
        Σ log λ_k로 바꾸면 ∂/∂λ_k = 1/λ_k가 되어 상대 변화가 등가가 된다. 최소점은 같다.
    가설 B(공유 표현): 같은 목적함수를 쓰는 (d)(분리된 9망)는 모드1이 e_λ 8.8e-5로
        **가장 정확**하다. 즉 목적함수만으로는 역전이 강제되지 않고 표현 공유가 필요하다.

    이 드라이버는 A를 직접 검정한다 — 구조는 그대로 두고 가중만 바꾼다. logtrace에서
    프로파일이 평평해지면 A가 지배적이고, 그대로면 B가 지배적이다.

    부수로 **H²(곡률) 오차와 L²(질량) 오차를 분리**해 낸다. "곡률로 재고 질량으로
    채점한다"는 설명의 직접 검정이다 — MAC이 1인데 e_λ가 큰 이유가 그것이라면,
    모드1의 H² 오차만 크고 L² 오차는 작아야 한다.
    """
    from ..neural import subspace as sub
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_seeds, iters, n_modes, objectives = 3, 30, 3, ("trace",)
    rows, recs = [], []
    for obj in objectives:
        out = sub.solve_simultaneous(n_modes, n_seeds=n_seeds, iters=iters,
                                     objective=obj, device=device)
        xs, wq = out["xs"], out["wq"]
        for m in range(1, n_modes + 1):
            r, agg = _cell(out, m, out["arm"], "I0", "gauss")
            # H²·L² 분리 — 부호·스케일을 참조모드에 맞춰 정규화한 뒤 잰다
            ref = p1.analytic_mode(xs, m)
            refd2 = p1.analytic_mode_d2(xs, m)
            nref = float(np.sqrt((wq * ref ** 2).sum()))
            nc = float(np.sqrt((wq * refd2 ** 2).sum()))
            l2, h2 = [], []
            for s in range(n_seeds):
                ph = out["shapes"][m - 1][s]
                cu = out["curvatures"][m - 1][s]
                if not np.all(np.isfinite(ph)):
                    continue
                a = float((wq * ph * ref).sum() / max((wq * ph ** 2).sum(), 1e-300))
                l2.append(float(np.sqrt((wq * (a * ph - ref) ** 2).sum())) / nref)
                h2.append(float(np.sqrt((wq * (a * cu - refd2) ** 2).sum())) / nc)
            agg.update({"objective": obj, "study": "e_objective_ablation",
                        "l2_error_median": float(np.median(l2)) if l2 else float("nan"),
                        "h2_error_median": float(np.median(h2)) if h2 else float("nan"),
                        "h2_over_l2": (float(np.median(h2)) / float(np.median(l2))
                                       if l2 and np.median(l2) > 0 else float("nan")),
                        "e_lam_median": float(np.nanmedian(
                            [x["e_lam"] for x in r if x["e_lam"] is not None])),
                        "mac_median": float(np.nanmedian(
                            [x["mac"] for x in r if x["mac"] is not None]))})
            for x in r:
                x["objective"] = obj
            recs += r
            rows.append(agg)
            print(f"  [{obj:9s} m{m}] p={agg['p_correct']:.2f} "
                  f"e_lam={agg['e_lam_median']:.3e} MAC={agg['mac_median']:.6f} "
                  f"L2={agg['l2_error_median']:.3e} H2={agg['h2_error_median']:.3e} "
                  f"H2/L2={agg['h2_over_l2']:.1f}", flush=True)

    manifest.write_csv(os.path.join(d, "p1_e_objective_ablation.csv"), rows)
    manifest.write_jsonl(os.path.join(d, "p1_e_objective_records.jsonl"), recs)
    manifest.write_json(os.path.join(d, "manifest_p1_e_objective.json"),
                        manifest.build({
                            "driver": "run_p1_neural.run_e_objective_ablation",
                            "n_seeds": n_seeds, "iters": iters, "n_modes": n_modes,
                            "objectives": list(objectives),
                            "hypothesis": "절대 trace의 가중 비대칭이 모드1 희생을 "
                                          "보상하는가 — logtrace로 상대 가중을 주어 검정"}))
    print(f"[run_e_objective_ablation] {d} — {len(rows)}행")
    return {"ablation": rows, "outdir": d}


def run_single_seed_cost(outdir=None, iters: int = 4000, reps: int = 3,
                         modes=(1, 2, 3), quick: bool = False, device=None) -> dict:
    """1회 시도 비용을 **직접** 잰다 — 앙상블÷배칭계수 환산을 없앤다.

    환산은 계수가 안정적일 때만 성립하는데, 이 기계에서는 그렇지 않다. 같은 측정을 세 번
    해서 n=1이 39.5 s와 26.5 s, n=50이 100 s와 222 s로 나왔고 계수가 2.53에서 7.82까지
    움직였다. 원인은 GPU를 상주 LLM 스택과 공유하기 때문이다(메모리 56 GB 점유, 부하 변동).

    그래서 비용표가 필요로 하는 값을 그대로 잰다: **n_seeds = 1로 돌린 벽시계**. 계수가
    사라지므로 그 불안정이 결론에 실리지 않는다. 남는 것은 공유 GPU 자체의 산포이고,
    그것은 반복 측정의 min/max로 드러낸다(§6.2 위협).
    """
    from ..neural.curriculum import solve_curriculum
    from ..neural.subspace import solve_neural_basis_galerkin, solve_simultaneous
    d = manifest.ensure_outdir(outdir)
    if quick:
        iters, reps, modes = 30, 1, (1,)
    n_seq = max(modes)
    rows = []

    def record(arm, mode, ts):
        ts = sorted(ts)
        m = ts[len(ts) // 2]
        rows.append({"arm": arm, "mode": mode, "iters": iters, "reps": len(ts),
                     "seconds_per_attempt": m, "seconds_min": ts[0],
                     "seconds_max": ts[-1],
                     "spread_rel": (ts[-1] - ts[0]) / max(m, 1e-300),
                     "measured": "directly at n_seeds = 1 (no batching conversion)"})
        print(f"  [1seed {arm[:28]:28s} m{mode}] median {m:7.1f}s "
              f"[{ts[0]:.1f}, {ts[-1]:.1f}]", flush=True)

    seq = (("a_prime_projection_diagonal", df.ProjectionDiagonal()),
           ("b_projection_exact", df.ProjectionExact()))
    for name, defl in seq:
        per = {m: [] for m in modes}
        for _ in range(reps):
            out = sq.solve_sequential(n_seq, defl, n_seeds=1, iters=iters,
                                      device=device)
            for m in modes:
                per[m].append(float(out["stage_seconds"][m - 1]))
        for m in modes:
            record(name, m, per[m])

    per = {m: [] for m in modes}
    for _ in range(reps):
        out = solve_curriculum(n_seq, df.ProjectionExact(), n_seeds=1, iters=iters,
                               device=device)
        for m in modes:
            per[m].append(float(out["stage_seconds"][m - 1]))
    for m in modes:
        record("c_curriculum", m, per[m])

    for label, fn in (("d_neural_basis_galerkin",
                       lambda: solve_neural_basis_galerkin(
                           max(modes), n_seeds=1, iters=iters, device=device)),
                      ("e_simultaneous_subspace",
                       lambda: solve_simultaneous(
                           max(modes), n_seeds=1, iters=iters, device=device))):
        ts = [float(fn()["seconds"]) for _ in range(reps)]
        for m in modes:          # 부분공간 arm은 한 번에 전 모드를 낸다 — 같은 비용
            record(label, m, ts)

    manifest.write_csv(os.path.join(d, "p1_single_seed_cost.csv"), rows)
    manifest.write_json(os.path.join(d, "manifest_p1_single_seed_cost.json"),
                        manifest.build({
                            "driver": "run_p1_neural.run_single_seed_cost",
                            "iters": iters, "reps": reps, "modes": list(modes),
                            "why": "the ensemble-to-per-attempt conversion factor was "
                                   "measured at 2.53 and at 7.82 on the same machine, "
                                   "so it is not usable; this measures the quantity the "
                                   "cost tables need directly",
                            "machine_note": "the GPU is shared with a resident LLM "
                                            "serving stack; wall-clock spread is "
                                            "reported per row"}))
    print(f"[run_single_seed_cost] {d} — {len(rows)}행")
    return {"single_seed": rows, "outdir": d}


def run_penalty_extended(outdir=None, n_seeds: int = 50, iters: int = 4000,
                         weights=(1e4, 1e5), n_modes: int = 3,
                         quick: bool = False, device=None) -> dict:
    """페널티 가중을 두 자리 더 밀어본다 — gap-scaling 가설의 직접 검정.

    penalty의 목적함수는 정규화된 겹침 제곱이다:

        pen(φ) = w · Σ_k ⟨Φ_k, φ⟩² / (‖φ‖² ‖Φ_k‖²)

    각 항이 cos²이라 **모드당 최대 1로 유계**다. 반면 Rayleigh 몫은 Λ_m 규모로 자란다
    (Λ₁ = 12.4, Λ₂ = 485, Λ₃ = 3807). 그래서 "겹침을 벌하는 힘"과 "낮은 모드로 내려가려는
    힘"의 비가 w/Λ_m으로 스케일하고, 같은 w가 낮은 모드에서는 충분하고 높은 모드에서는
    부족할 수 있다 — 실제로 기존 스윕에서 모드 2는 w = 1000에서 50/50이 됐고 모드 3은
    0/50이었다.

    그렇다면 모드 3도 w를 Λ₃/Λ₂ ≈ 7.8배 이상 올리면 넘어갈 것이라는 예측이 나온다. 넘어가면
    "penalty는 원리적으로 실패한다"가 아니라 "penalty는 모드마다 다른 가중을 요구한다"가
    되어 서사가 더 정확해지고, 넘어가지 않으면 그 자체가 결과다. 어느 쪽이든 값이 있다.
    """
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_seeds, iters, weights, n_modes = 4, 100, (1e4,), 2
    path = os.path.join(d, "p1_neural_penalty_sweep.csv")
    keep = [r for r in manifest.read_csv(path) if os.path.exists(path)] \
        if os.path.exists(path) else []
    keep = [r for r in keep if float(r.get("penalty_weight", -1)) not in
            {float(w) for w in weights}]
    rows, recs = [], []
    for w in weights:
        out = sq.solve_sequential(n_modes, df.Penalty(w), n_seeds=n_seeds,
                                  iters=iters, device=device)
        for m in range(1, n_modes + 1):
            r, agg = _cell(out, m, f"a_penalty(w={w:g})", "I0", "gauss")
            agg["penalty_weight"] = float(w)
            recs += r
            rows.append(agg)
            print(f"  [penalty w={w:g} m{m}] p_correct={agg['p_correct']:.2f} "
                  f"p_accurate={agg['p_accurate']:.2f}", flush=True)
    manifest.write_csv(path, keep + rows)
    manifest.write_jsonl(os.path.join(d, "p1_neural_penalty_ext_records.jsonl"), recs)
    manifest.write_json(os.path.join(d, "manifest_p1_penalty_extended.json"),
                        manifest.build({
                            "driver": "run_p1_neural.run_penalty_extended",
                            "weights": [float(w) for w in weights],
                            "n_seeds": n_seeds, "iters": iters, "n_modes": n_modes,
                            "objective": "w * sum_k <Phi_k,phi>^2 / (|phi|^2 |Phi_k|^2) "
                                         "— normalized, so bounded by w per mode",
                            "hypothesis": "the penalty-to-Rayleigh ratio scales as "
                                          "w / Lambda_m, so mode 3 may need roughly "
                                          "Lambda_3/Lambda_2 ~ 7.8x the weight that "
                                          "sufficed for mode 2"}))
    print(f"[run_penalty_extended] {len(rows)}행")
    return {"penalty": rows, "outdir": d}


def run_orthogonality_diagnostic(outdir=None, n_seeds: int = 50, iters: int = 4000,
                                 n_modes: int = 10, quick: bool = False,
                                 device=None) -> dict:
    """모드별 ‖ΦᵀWΦ − I‖ — (a′)의 가정이 얼마나 깨지는지 정량화한다.

    **핵심은 무엇의 그람인가다.** deflation이 넘겨받는 Φ는 앞 단계의 **원시 망 출력**이고,
    보고되는 모드형(사영 후)이 아니다. 원시 φ⁽²⁾는 모드 1 성분을 그대로 갖고 있으므로
    (사영은 보고 시점에만 적용된다) 그 그람은 강하게 비대각이다.

    이것이 두 arm을 가르는 지점이다. 정확 사영은 `c = G⁻¹Φᵀwφ`이라 Φ의 **span에만** 의존해
    개별 벡터를 어떻게 잡든 같은 결과를 낸다. 대각근사는 개별 방향에 의존하므로 span이
    같아도 답이 달라진다.

    대각근사 사영은 c_k = ⟨φ,φ_k⟩/⟨φ_k,φ_k⟩로 계수를 잡는데, 이는 앞 모드들이 **서로
    질량-직교일 때만** 정확 사영과 일치한다. 신경 근사 모드들은 직교하지 않으므로 잔차가
    남고, §5.1은 그 잔차가 모드 1로의 하강 방향과 겹친다고 설명한다.

    그 가정이 실제로 얼마나 깨지는지는 지금까지 **측정되지 않았다.** 각 단계 k에서 앞
    모드들을 질량 정규화한 그람행렬 G의 ‖G − I‖_F를 재면, (a′)와 (b)에서 그 값이 어떻게
    갈라지는지가 기전 주장의 직접 증거가 된다.

    **모드 10까지 잰다.** headline이 "정확 사영은 모드 10까지 벽이 없다"이므로 정작
    중요한 그람은 모드 10을 풀 때 쓰이는 k = 9다. k = 6에서 이미 κ₂(G) = 2e8이므로
    "그럼 k = 7, 8, 9는?"이 곧바로 나오고, 그 답이 초록의 "남은 한계는 최적화 예산"
    주장과 대수적 조건을 분리할 수 있는지를 결정한다.

    **두 사영은 그람의 비대각을 통해서만 다르다.** 스테이지 2에서는 Φ가 1열이라 G가 1×1이고
    두 arm이 비트 단위로 동일하다 — 그래서 모드 1·2가 양쪽에서 같은 결과인 것이 당연하다.
    차이는 스테이지 3에서 2×2 비대각이 처음 생길 때 나타난다. 따라서 재야 할 것은

      · 넘겨받은 모드들의 정규화 그람이 I에서 얼마나 벗어나는가 (`gram_dev_fro`), 그리고
      · 그 벗어남이 **사영 연산자**를 얼마나 바꾸는가
        `op_rel_diff = ‖G⁻¹ − diag(G)⁻¹‖_F / ‖diag(G)⁻¹‖_F`

    후자가 φ에 무관하게 두 arm이 다를 수 있는 최대 폭이다. 이 값이 0이면 두 arm은 그
    단계에서 같은 계산을 하므로 §5.1의 설명이 틀린 것이고, 유한하면 그 크기가 기전의
    정량이 된다. 어느 쪽이든 지금까지 측정되지 않았다.
    """
    import torch
    from torch.func import vmap

    from ..neural import core
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_seeds, iters, n_modes = 4, 100, 3
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name, defl in (("a_prime_projection_diagonal", df.ProjectionDiagonal()),
                       ("b_projection_exact", df.ProjectionExact())):
        out = sq.solve_sequential(n_modes, defl, n_seeds=n_seeds, iters=iters,
                                 device=device)
        n_here = n_modes
        xs = torch.tensor(out["xs"], device=dev)
        wq = torch.tensor(out["wq"], device=dev)
        # **원시 망 출력**의 그람을 잰다 — deflation이 실제로 넘겨받는 Φ가 그것이다.
        # 사영된 `shapes`를 재면 구성상 대각이 되어 두 arm이 같아 보인다(실제로 그렇게
        # 잘못 잰 적이 있다).
        sh = torch.tensor(np.asarray(out["raw_shapes"]), device=dev)  # (mode, seed, n_q)
        for k in range(2, n_here + 1):           # k개 모드가 모였을 때의 직교성
            P = sh[:k]                                             # (k, seed, n_q)
            G = torch.einsum("asi,i,bsi->sab", P, wq, P)
            dg = torch.sqrt(torch.clamp(torch.diagonal(G, dim1=-2, dim2=-1),
                                        min=1e-300))
            C = G / (dg.unsqueeze(-1) * dg.unsqueeze(-2))           # 질량 정규화
            eye = torch.eye(k, dtype=C.dtype, device=C.device)
            dev_f = torch.linalg.matrix_norm(C - eye, ord="fro").cpu().numpy()
            offmax = torch.triu(C - eye, 1).abs().amax(dim=(-2, -1)).cpu().numpy()
            # **역그람 불일치**(옛 이름 op_rel_diff). 이것은 계수사상 G⁻¹ ↔ D⁻¹의
            # 상대거리이고 사영 연산자 자체의 거리가 아니다 — 검토자가 지적한 대로
            # 이름과 해석을 분리해서 싣는다.
            Gi = torch.linalg.inv(G)
            Di = torch.diag_embed(1.0 / torch.diagonal(G, dim1=-2, dim2=-1))
            rel = (torch.linalg.matrix_norm(Gi - Di, ord="fro")
                   / torch.linalg.matrix_norm(Di, ord="fro")).cpu().numpy()
            # **사영 연산자의 실제 차이.** 논문의 규약은 Φ의 **열**이 기저함수이고
            # G = ΦᵀWΦ이므로 사영은 Π_G = ΦG⁻¹ΦᵀW, Π_D = ΦD⁻¹ΦᵀW다. 따라서
            #   Π_G − Π_D = Φ(G⁻¹ − D⁻¹)ΦᵀW
            # 이고, 질량내적이 유도하는 작용소 노름에서 B = W^{1/2}Φ, BᵀB = G를 쓰면
            #   ‖Π_G − Π_D‖_W = ‖G^{1/2}(G⁻¹ − D⁻¹)G^{1/2}‖₂ = ‖I − G^{1/2}D⁻¹G^{1/2}‖₂
            # 인데, G^{1/2}D⁻¹G^{1/2}는 D⁻¹G와 닮았고 그 고유값은 C = D^{-1/2}GD^{-1/2}의
            # 고유값과 같다. 대칭행렬이므로 결국
            #   ‖Π_G − Π_D‖_W = ‖I − C‖₂ = max_i |1 − λ_i(C)|
            # 로 **역행렬도 제곱근도 없이** 정확히 계산된다. 상한이 아니라 실측값이다.
            evc = torch.linalg.eigvalsh(C)
            pdiff = (evc - 1.0).abs().amax(dim=-1).cpu().numpy()
            # 전체 그람 역행렬을 핵심 방법으로 쓰므로 그 조건도 함께 보고한다.
            ev = torch.linalg.eigvalsh(G)
            kap = (ev.amax(dim=-1) / torch.clamp(ev.amin(dim=-1),
                                                 min=1e-300)).cpu().numpy()
            n_sing = int((ev.amin(dim=-1) <= 0).sum())
            lminc = evc.amin(dim=-1).cpu().numpy()
            erank = (evc > 1e-12 * evc.amax(dim=-1, keepdim=True)
                     ).sum(dim=-1).double().cpu().numpy()
            rows.append({
                "arm": name, "n_prev_modes": k, "n_seeds": n_seeds, "iters": iters,
                "gram_dev_fro_median": float(np.median(dev_f)),
                "gram_dev_fro_p95": float(np.percentile(dev_f, 95)),
                "max_offdiag_median": float(np.median(offmax)),
                "max_offdiag_p95": float(np.percentile(offmax, 95)),
                "inv_gram_rel_diff_median": float(np.median(rel)),
                "inv_gram_rel_diff_p95": float(np.percentile(rel, 95)),
                "proj_op_diff_W_median": float(np.median(pdiff)),
                "proj_op_diff_W_p95": float(np.percentile(pdiff, 95)),
                "kappa_gram_median": float(np.median(kap)),
                "kappa_gram_p95": float(np.percentile(kap, 95)),
                "lam_min_normalized_median": float(np.median(lminc)),
                "eff_rank_1e12_median": float(np.median(erank)),
                "n_seeds_gram_singular": n_sing,
                "solve_method": "torch.linalg.solve on the full Gram (LU, no "
                                "regularization, no pinv) as in neural/deflation.py",
                "note": "mass-normalized Gram C of the k raw modes handed to the next "
                        "stage; G is the unnormalized Gram. inv_gram_rel_diff = "
                        "||G^-1 - diag(G)^-1||_F / ||diag(G)^-1||_F compares the two "
                        "coefficient maps. proj_op_diff_W = ||I - G^(1/2) diag(G)^-1 "
                        "proj_op_diff_W = ||I - C||_2 = max_i |1 - lambda_i(C)| is the "
                        "exact operator norm of Pi_G - Pi_D under the mass inner product, "
                        "with Pi_G = Phi G^-1 Phi^T W and Pi_D = Phi diag(G)^-1 Phi^T W"})
            print(f"  [orth {name.split('_')[0]} k={k}] "
                  f"||C-I||_F {rows[-1]['gram_dev_fro_median']:.3e}  "
                  f"max|off| {rows[-1]['max_offdiag_median']:.3e}  "
                  f"invG {rows[-1]['inv_gram_rel_diff_median']:.3e}  "
                  f"||P_G-P_D||_W {rows[-1]['proj_op_diff_W_median']:.3e}  "
                  f"kappa(G) {rows[-1]['kappa_gram_median']:.3e}", flush=True)

        # **모드 3에서의 잔여 누출** — 인과의 마지막 고리. 각 arm이 자기 규칙으로
        # 사영한 모드-3 함수가 저장 모드들과 얼마나 겹쳐 남는지. (b)는 구성상 0이고,
        # (a′)에 남는 양이 모드 1로 되돌아가는 하강 방향과 겹치는 성분이다.
        if n_here >= 3:
            P = sh[:2]                                    # (2, seed, n_q)
            phi = sh[2]                                   # (seed, n_q)
            G2 = torch.einsum("asi,i,bsi->sab", P, wq, P)
            r2 = torch.einsum("asi,i,si->sa", P, wq, phi)
            if "diagonal" in name:
                c = r2 / torch.diagonal(G2, dim1=-2, dim2=-1)
            else:
                c = torch.linalg.solve(G2, r2.unsqueeze(-1)).squeeze(-1)
            tphi = phi - torch.einsum("sa,asi->si", c, P)
            leak = torch.einsum("asi,i,si->sa", P, wq, tphi)
            nrm = torch.sqrt(torch.einsum("si,i,si->s", tphi, wq, tphi))
            dg2 = torch.sqrt(torch.diagonal(G2, dim1=-2, dim2=-1))
            rl = (torch.linalg.vector_norm(leak / dg2, dim=-1)
                  / torch.clamp(nrm, min=1e-300)).cpu().numpy()
            rows.append({
                "arm": name, "n_prev_modes": 2, "n_seeds": n_seeds, "iters": iters,
                "mode3_leak_rel_median": float(np.median(rl)),
                "mode3_leak_rel_p95": float(np.percentile(rl, 95)),
                "solve_method": "torch.linalg.solve on the full Gram (LU, no "
                                "regularization, no pinv) as in neural/deflation.py",
                "note": "residual overlap of the deflated mode-3 function with the two "
                        "stored modes, ||Phi W phi_tilde|| / ||phi_tilde||_W with the "
                        "stored modes mass-normalized; zero iff deflation removed the "
                        "stored span exactly"})
            print(f"  [orth {name.split('_')[0]} mode3 leak] "
                  f"{rows[-1]['mode3_leak_rel_median']:.3e}", flush=True)

    manifest.write_csv(os.path.join(d, "p1_orthogonality.csv"), rows)
    manifest.write_json(os.path.join(d, "manifest_p1_orthogonality.json"),
                        manifest.build({
                            "driver": "run_p1_neural.run_orthogonality_diagnostic",
                            "n_seeds": n_seeds, "iters": iters, "n_modes": n_modes,
                            "metric": "mass-normalized Gram deviation, exact operator "
                                      "norm of P_G - P_D under the mass inner product, "
                                      "Gram conditioning, and mode-3 residual leakage",
                            "why": "§5.1 attributes the (a') failure to residual "
                                   "non-orthogonality of the previously computed modes, "
                                   "which had not been measured"}))
    print(f"[run_orthogonality_diagnostic] {len(rows)}행")
    return {"orthogonality": rows, "outdir": d}
