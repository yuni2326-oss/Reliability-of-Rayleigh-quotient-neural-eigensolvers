"""arm (b′) — QR deflation의 P1 본실험과 float32 스트레스.

**(b)와 (b′)은 같은 사영이고 다른 수치경로다.** 그래서 두 실험이 서로 다른 것을 잰다.

  · 본실험(fp64)  — 신뢰도는 (b)와 **같아야 한다**. 같지 않으면 둘 중 하나가 사영이
    아니거나 구현이 틀린 것이다. 여기서 새로 얻는 것은 신뢰도가 아니라 **진단의 대비**다:
    (b′)은 ‖I − QᵀWQ‖ ≈ ε를 유지하는데 (b)가 실제로 푸는 그람은 κ₂(G) = 1.6e11까지 자란다.
  · fp32 스트레스 — κ₂(G)·ε_fp32 ≈ 1.6e11 × 1.2e-7 ≈ 1.9e4 ≫ 1이므로 그람 경로에는
    유효자리가 남지 않는다는 예측을 **직접 시험한다.** 예측이 맞으면 (b)는 무너지고
    (b′)은 산다. 틀리면 그 사실이 결과다.

fp64 기준선은 다시 돌리지 않는다 — 커밋된 `p1_neural_mode_sweep.csv`(arm (b))와 이 모듈이
쓰는 `p1_neural_qr.csv`(arm (b′))가 같은 설정(50시드·4000반복·I0·고정 Gauss·n_q = 256)이다.

**사후 추가다.** (b′)과 fp32 축은 사전등록 격자 이후인 2026-08-26에 추가됐고, 판정규칙
(MAC ≥ 0.9, |Δλ|/λ ≤ 0.05, 4분류, 50시드, Wilson 95 %, 예산 4000)은 바꾸지 않았다.

  docker run --rm --gpus all --ipc=host -v /home/super/project:/work -w /work \
    -e PYTHONPATH=/work gb10-pinn:26.04 python3 -u -m eigen_benchmark.drivers.run_p1_qr
"""
from __future__ import annotations

import os

import numpy as np

from ..neural import deflation as df
from ..neural import sequential as sq
from . import manifest
from .run_p1_neural import _cell

MODE_MAX = 10
ARM_QR = "b_prime_projection_qr"
ARM_EXACT = "b_projection_exact"
ADDED = ("2026-08-26, after the pre-specified grid; the judgment rules "
         "(MAC >= 0.9, |dlambda|/lambda <= 0.05, four outcomes, 50 seeds, "
         "Wilson 95 %, 4000 iterations) were not changed")

# 진단 CSV의 열 — 표가 이 이름을 그대로 읽는다
DIAG_KEYS = ("k", "kappa_gram", "kappa_gram_times_eps", "dev_identity",
             "lam_min_normalized", "qr_orth_defect", "n_dropped",
             "rank_1e10", "rank_1e12", "rank_1e16", "proj_agreement_rel",
             "gram_solve_finite_frac", "span_finite_frac", "eps_machine", "dtype",
             "measure_dtype",
             "n_seeds_summarized")


def _diag_rows(out, arm: str, n_seeds: int, iters: int) -> list:
    """스테이지별 스팬 진단 → CSV 행. 스테이지 m은 m−1개 저장 모드를 넘겨받았다."""
    rows = []
    for m, sd in enumerate(out["span_diagnostics"], start=1):
        if sd is None:
            continue
        row = {"arm": arm, "dtype": out["dtype"], "stage_mode": m,
               "n_seeds": n_seeds, "iters": iters}
        row.update({k: sd.get(k) for k in DIAG_KEYS})
        rows.append(row)
    return rows


def _print_diag(tag: str, rows: list) -> None:
    for r in rows:
        print(f"  [{tag} m{r['stage_mode']} k={r['k']}] "
              f"kappa(G) {r['kappa_gram']:.3e}  x eps {r['kappa_gram_times_eps']:.3e}  "
              f"||I-QWQ|| {r['qr_orth_defect']:.3e}  dropped {r['n_dropped']}  "
              f"agree {r['proj_agreement_rel']:.3e}", flush=True)


def main(outdir=None, quick: bool = False, n_modes: int = MODE_MAX,
         n_seeds: int = 50, iters: int = 4000, device=None) -> dict:
    """Stage 2 — (b′) × P1 모드 1–10 × 50시드, 기존 게이트."""
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_modes, n_seeds, iters = 3, 4, 20

    out = sq.solve_sequential(n_modes, df.ProjectionQR(), n_seeds=n_seeds,
                              iters=iters, device=device)
    rows, records = [], []
    for m in range(1, n_modes + 1):
        r, agg = _cell(out, m, ARM_QR, "I0", "gauss")
        records += r
        rows.append(agg)
        print(f"  [qr m{m}] p_correct={agg['p_correct']:.2f} "
              f"p_accurate={agg['p_accurate']:.2f} {agg['seconds']:.0f}s", flush=True)
    diag = _diag_rows(out, ARM_QR, n_seeds, iters)
    _print_diag("qr diag", diag)

    manifest.write_csv(os.path.join(d, "p1_neural_qr.csv"), rows)
    manifest.write_jsonl(os.path.join(d, "p1_neural_qr_records.jsonl"), records)
    manifest.write_csv(os.path.join(d, "p1_qr_span_diagnostics.csv"), diag)
    manifest.write_json(os.path.join(d, "manifest_p1_qr.json"), manifest.build({
        "driver": "run_p1_qr.main", "arm": ARM_QR, "quick": quick,
        "n_modes": n_modes, "n_seeds": n_seeds, "iters": iters,
        "dtype": out["dtype"],
        "deflation": "phi_tilde = phi - Q (Q^T W phi) with Q from W-inner-product "
                     "modified Gram-Schmidt on the stored raw outputs; the same "
                     "coefficients are applied to the stored second derivatives, so "
                     "the coefficients are constants with respect to the spatial "
                     "derivative exactly as in arm (b)",
        "drop_rule": "a direction is dropped when its norm after orthogonalization "
                     "falls below sqrt(eps_machine) times its norm before it",
        "diagnostics": ("median over the first "
                        f"{diag[0]['n_seeds_summarized'] if diag else 0} seeds"),
        "added": ADDED}))
    print(f"[run_p1_qr] {d} — {len(rows)}행, 진단 {len(diag)}행")
    return {"qr_sweep": rows, "records": records, "diagnostics": diag, "outdir": d}


def single_seed_cost(outdir=None, n_modes: int = 3, iters: int = 4000, reps: int = 3,
                     quick: bool = False, device=None) -> dict:
    """(b′)의 1회 시도 비용 — 다른 arm과 같은 회계(n_seeds = 1, reps회 중앙값).

    QR은 반복마다 k×k 풀이 하나를 k(k+1)/2개의 내적으로 바꾼다. 그 값이 얼마인지는
    논증이 아니라 측정의 문제다."""
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_modes, iters, reps = 2, 10, 1
    per = {m: [] for m in range(1, n_modes + 1)}
    for _ in range(reps):
        out = sq.solve_sequential(n_modes, df.ProjectionQR(), n_seeds=1,
                                  iters=iters, device=device)
        for m in range(1, n_modes + 1):
            per[m].append(float(out["stage_seconds"][m - 1]))
    path = os.path.join(d, "p1_single_seed_cost.csv")
    keep = ([r for r in manifest.read_csv(path) if str(r["arm"]) != ARM_QR]
            if os.path.exists(path) else [])
    rows = []
    for m in range(1, n_modes + 1):
        ts = sorted(per[m])
        med = ts[len(ts) // 2]
        rows.append({"arm": ARM_QR, "mode": m, "iters": iters, "reps": len(ts),
                     "seconds_per_attempt": med, "seconds_min": ts[0],
                     "seconds_max": ts[-1],
                     "spread_rel": (ts[-1] - ts[0]) / max(med, 1e-300),
                     "measured": "directly at n_seeds = 1 (no batching conversion)"})
        print(f"  [1seed qr m{m}] median {med:7.1f}s [{ts[0]:.1f}, {ts[-1]:.1f}]",
              flush=True)
    manifest.write_csv(path, keep + rows)
    manifest.write_json(os.path.join(d, "manifest_p1_qr_cost.json"), manifest.build({
        "driver": "run_p1_qr.single_seed_cost", "arm": ARM_QR,
        "n_modes": n_modes, "iters": iters, "reps": reps, "added": ADDED,
        "why": "the QR arm replaces one k x k solve per iteration with k(k+1)/2 inner "
               "products, so its per-attempt cost has to be measured, not argued"}))
    print(f"[run_p1_qr.single_seed_cost] {len(rows)}행 (p1_single_seed_cost.csv에 병합)")
    return {"single_seed": rows, "outdir": d}


def _solve_reduced(defl, n_modes: int, n_seeds: int, iters: int, dtype, device,
                   check_n_q: int | None = None):
    """축소정밀도 실행. **예외로 죽는 것도 결과다** — 죽은 지점을 남기고 모드수를 줄여
    살아남는 만큼은 건진다. bfloat16에서는 그람 경로에 인수분해 구현 자체가 없어
    첫 시도에서 죽고, 그 사실이 이 함수의 반환값으로 기록된다."""
    err = None
    for n in (n_modes, max(n_modes // 2, 3), 2):
        try:
            out = sq.solve_sequential(n, defl, n_seeds=n_seeds, iters=iters,
                                      dtype=dtype, device=device,
                                      check_n_q=check_n_q)
            return out, n, err
        except RuntimeError as e:                      # 특이 그람·미구현 등
            err = f"{type(e).__name__}: {e}"[:300]
            print(f"  [{dtype}] n_modes={n}에서 예외 — {err}", flush=True)
        if n <= 2:
            break
    return None, 0, err


def run_precision_stress(outdir=None, dtype=None, tag: str = "fp32",
                         n_modes: int = MODE_MAX, n_seeds: int = 50,
                         iters: int = 4000, quick: bool = False,
                         exact_solver: str = "library", arms=None,
                         check_n_q: int | None = None,
                         device=None) -> dict:
    """정밀도 축 — (b)와 (b′)를 지정한 작업 정밀도로 돌린다.

    순차 deflation이라 상위 모드만 따로 돌 수 없다(모드 6은 1–5의 저장 스팬을 요구한다).
    그래서 1–10을 돌리고 전 모드를 보고한다.

    `tag`는 산출물 이름에 들어간다(`p1_{tag}_stress.csv`). 정밀도마다 파일을 갈라야
    이전 정밀도의 결과가 덮이지 않는다."""
    import torch

    dtype = torch.float32 if dtype is None else dtype
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_modes, n_seeds, iters = 3, 4, 20

    dname = str(dtype).replace("torch.", "")
    rows, diag_rows, notes, records = [], [], [], []
    pool = {ARM_EXACT: df.ProjectionExact(solver=exact_solver),
            ARM_QR: df.ProjectionQR()}
    order = list(arms) if arms else [ARM_EXACT, ARM_QR]
    for arm in order:
        defl = pool[arm]
        out, n_ok, err = _solve_reduced(defl, n_modes, n_seeds, iters, dtype, device,
                                        check_n_q=check_n_q)
        if out is None:
            notes.append({"arm": arm, "dtype": dname, "aborted": True,
                          "error": err or "", "n_modes_completed": 0})
            print(f"  [{tag} {arm}] 전부 실패 — {err}", flush=True)
            continue
        # **재채점 가능하게 남긴다.** 채점 규칙이나 참조가 바뀌면 형상만 있으면 되고,
        # 없으면 몇 시간짜리 풀이를 다시 태워야 한다(해석 참조 결함에서 실제로 그랬다).
        np.savez_compressed(
            os.path.join(d, f"p1_{tag}_{arm}_shapes.npz"),
            shapes=np.asarray(out["shapes"]), lam=np.asarray(out["lam"]),
            xs=np.asarray(out["xs"]), wq=np.asarray(out["wq"]),
            converged=np.asarray(out["converged"]),
            **({"lam_check": np.asarray(out["lam_check"]),
                "quad_drift": np.asarray(out["quad_drift"])}
               if "quad_drift" in out else {}))
        for m in range(1, n_ok + 1):
            recs, agg = _cell(out, m, arm, "I0", "gauss", n_ref=n_modes + 4)
            agg["dtype"] = dname
            agg["exact_solver"] = exact_solver if arm == ARM_EXACT else ""
            agg["n_modes_completed"] = n_ok
            rows.append(agg)
            records += recs
            print(f"  [{tag} {arm[:8]} m{m}] p_correct={agg['p_correct']:.2f} "
                  f"p_accurate={agg['p_accurate']:.2f}", flush=True)
        dg = _diag_rows(out, arm, n_seeds, iters)
        _print_diag(f"{tag} {arm[:8]}", dg)
        diag_rows += dg
        notes.append({"arm": arm, "dtype": dname, "aborted": n_ok < n_modes,
                      "error": err or "", "n_modes_completed": n_ok})

    if rows:
        manifest.write_csv(os.path.join(d, f"p1_{tag}_stress.csv"), rows)
    manifest.write_jsonl(os.path.join(d, f"p1_{tag}_records.jsonl"), records)
    if diag_rows:
        manifest.write_csv(os.path.join(d, f"p1_{tag}_span_diagnostics.csv"), diag_rows)
    manifest.write_json(os.path.join(d, f"manifest_p1_{tag}.json"), manifest.build({
        "driver": "run_p1_qr.run_precision_stress",
        "arms": order, "dtype": dname, "tag": tag,
        "exact_solver": exact_solver,
        "n_modes": n_modes, "n_seeds": n_seeds, "iters": iters, "runs": notes,
        "check_n_q": check_n_q,
        "quadrature_separation": (
            "each cell is re-evaluated at a second Gauss order that training never saw; "
            "a trained basis adapts to the grid that measures it, so certifying one grid "
            "against the reference modes does not certify the function the optimizer "
            "reaches" if check_n_q else "not measured"),
        "fp64_baseline": "not re-run; the fp64 rows come from p1_neural_mode_sweep.csv "
                         "(arm b) and p1_neural_qr.csv (arm b'), which use the same "
                         "50 seeds, 4000 iterations, I0, fixed Gauss and n_q = 256",
        "quadrature": "Gauss nodes and weights are built in fp64 and cast down, so the "
                      "quadrature rule itself is not part of the precision variable",
        "prediction": "kappa_2(G) reaches 1.56e11 at k = 9 in fp64, so kappa_2(G) times "
                      "the fp32 unit roundoff is of order 1e4 and the Gram path should "
                      "have no significant digits left, while the QR path depends on "
                      "kappa_2(Phi) = sqrt(kappa_2(G)) instead. The prediction recorded "
                      "before the reduced-precision runs is in "
                      "docs/paper2-cs/2026-08-26-precision-third-point-prediction.md",
        "rescoring": "per-seed records and the projected mode shapes are saved, so a "
                     "change of scoring rule or reference can be applied without re-running "
                     "the solves",
        "measurement": "spectral diagnostics are computed on a float64 copy of the stored "
                       "span at precisions where the library provides no symmetric "
                       "eigensolver; the orthonormalization itself runs at the working "
                       "precision and each row records the measure_dtype actually used",
        "added": ADDED}))
    print(f"[run_precision_stress:{tag}] {len(rows)}행, 진단 {len(diag_rows)}행")
    return {"fp32": rows, "rows": rows, "diagnostics": diag_rows, "notes": notes,
            "outdir": d}


def run_fp32_stress(outdir=None, **kw) -> dict:
    """이전 이름 유지 — float32 축은 이 이름으로 불려 왔다."""
    import torch

    kw.setdefault("dtype", torch.float32)
    kw.setdefault("tag", "fp32")
    return run_precision_stress(outdir=outdir, **kw)


if __name__ == "__main__":
    import sys

    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "main"):
        main()
    if what in ("all", "cost"):
        single_seed_cost()
    if what in ("all", "fp32"):
        run_fp32_stress()
