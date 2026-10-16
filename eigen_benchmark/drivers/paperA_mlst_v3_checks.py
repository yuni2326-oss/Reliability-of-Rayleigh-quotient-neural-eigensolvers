"""논문 A MLST v3 Word 메모(2026-10-15) 대응 검사. 학습 없이 커밋된 기록·배열과 고전 경로만 쓴다.

- `timing_spans` — 반복 측정의 max/min을 자릿수(log10)로. 표 16의 순위가 타이밍 산포보다 큰지 본다(본문 메모 [17]).
- `quad_remainder`·`reference_check` — fp32 고정 스팬 비교(보충 표 S21)의 기준은 fp64 SVD 최소제곱이다. 같은 배열을
  binary128(aarch64의 long double)로 두 번 직교화한 MGS 잔여와 비교해 기준 자체의 정밀도를 잰다(본문 [9], 보충 [24]).
- `cholesky_boundary` — 원시 단항 N = 12의 질량행렬에서 Cholesky 실패와 eigh(K, M) 성공이 함께 나온 기록(보충 S10.7)을
  삼각형별 LAPACK potrf, scipy 경로, eigh 드라이버별로 다시 잰다(보충 [9]). 라이브러리 판마다 따로 돌린다.
- `coarse_target_seconds` — arm (c)의 조립 Ritz 목표는 학습 단계 타이머 밖에서 계산된다. 그 비용을 잰다(본문 [3]).
"""
import csv
import json
import math
import os
import time

import numpy as np
from scipy import linalg
from scipy.linalg import lapack

from ..conditioning import generalized_backward_error
from . import manifest

DRIVERS = ("gvd", "gv", "gvx")
TABLE16_NEURAL = (("c_curriculum", 1), ("p4_neural_2d[exact projection]", 1),
                  ("p4_neural_2d[exact projection]", 2), ("p4_neural_2d[exact projection]", 3))
TABLE16_CLASSICAL = (("P1", "monomial_raw", 8), ("P4", "graded Q2 (beta=3.0)", 8))


def timing_spans(rows, key) -> dict:
    """행마다 seconds_max/seconds_min과 그 log10. 가장 넓은 행도 함께."""
    out = {}
    for r in rows:
        lo, hi = float(r["seconds_min"]), float(r["seconds_max"])
        out[key(r)] = {"ratio": hi / lo, "orders": math.log10(hi / lo)}
    k = max(out, key=lambda q: out[q]["orders"])
    return {"rows": out, "max_orders": out[k]["orders"], "max_key": k}


def table16_timing_spans(data_dir) -> dict:
    """표 16에 들어간 구성 가운데 반복 측정이 있는 것의 산포. 기울기 학습은 3회, 고전은 7회."""
    rd = lambda name: list(csv.DictReader(open(os.path.join(data_dir, name), encoding="utf-8")))  # noqa: E731
    neural = [r for r in rd("p1_single_seed_cost.csv") if (r["arm"], int(r["mode"])) in TABLE16_NEURAL]
    classical = [r for r in rd("p1p4_classical_cost_repeats.csv")
                 if (r["problem"], r["solver"], int(r["size"])) in TABLE16_CLASSICAL]
    return {"neural": timing_spans(neural, key=lambda r: f"{r['arm']}/{r['mode']}"),
            "classical": timing_spans(classical, key=lambda r: f"{r['problem']}/{r['solver']}/{r['size']}"),
            "not_repeated": ["Eig-PIELM attempt times (single deterministic solve)",
                             "graded Q2 (beta=1.0) at 322 dof (P4 mode 2 baseline)"]}


def quad_remainder(P, f, w):
    """f에서 span(P의 행)을 질량 내적 ⟨u, v⟩ = Σ w u v로 지운 잔여를 binary128로. MGS를 두 번 돌린다."""
    ld = np.longdouble
    sw = np.sqrt(np.asarray(w, ld))
    A = np.asarray(P, ld) * sw
    b = np.asarray(f, ld) * sw
    Q = []
    for j in range(A.shape[0]):
        v = A[j].copy()
        for _ in range(2):
            for q in Q:
                v -= (q @ v) * q
        nv = np.sqrt(v @ v)
        if nv > 0:
            Q.append(v / nv)
    for _ in range(2):
        for q in Q:
            b -= (q @ b) * q
    return b / sw


def _fp64_svd_reference(P, f, w, solver=None):
    """커밋된 `fixed_span_routes`와 같은 기준: √W Φᵀ c ≈ √W φ의 SVD 최소제곱(gelsd)과 잔여 φ − cΦ."""
    sw = np.sqrt(w)
    A, b = sw[:, None] * P.T, sw * f
    if solver is not None:
        c = solver(A, b)
    else:
        try:
            import torch
            c = torch.linalg.lstsq(torch.as_tensor(A), torch.as_tensor(b), driver="gelsd").solution.numpy()
        except ImportError:
            c = np.linalg.lstsq(A, b, rcond=None)[0]
    return f - c @ P


def reference_check(raw, wq, solver=None) -> dict:
    """fp32 재실행의 원시 배열 raw(모드, 시드, 절점)에서 단계 m의 (저장 집합 1..m−1, 후보 m)마다 fp64 기준과 binary128
    잔여의 상대 질량노름 차이. 커밋된 비교처럼 배열과 가중치를 fp32로 반올림한 값을 쓴다."""
    raw32 = np.asarray(raw, np.float32).astype(np.float64)
    w = np.asarray(wq, np.float32).astype(np.float64)
    wl = w.astype(np.longdouble)
    by_stage, allv = {}, []
    for m in range(2, raw32.shape[0] + 1):
        vals = []
        for s in range(raw32.shape[1]):
            P, f = raw32[:m - 1, s], raw32[m - 1, s]
            r64 = _fp64_svd_reference(P, f, w, solver).astype(np.longdouble)
            r128 = quad_remainder(P, f, w)
            d = np.sqrt(np.sum(wl * (r64 - r128) ** 2)) / np.sqrt(np.sum(wl * r128 ** 2))
            vals.append(float(d))
        by_stage[m] = {"max": max(vals), "median": float(np.median(vals))}
        allv += vals
    return {"by_stage": by_stage, "max_rel_diff": max(allv), "n_pairs": len(allv),
            "reference": "binary128 weighted MGS (two passes) vs float64 SVD least squares (gelsd)"}


def cholesky_boundary(K, M) -> dict:
    """같은 (K, M)에 대해 삼각형별 potrf, scipy.linalg.cholesky, eigh(K, M)의 드라이버별 결과."""
    K, M = np.asarray(K, float), np.asarray(M, float)
    out = {"n": int(M.shape[0]),
           "asymmetry_rel": float(np.max(np.abs(M - M.T)) / np.max(np.abs(M)))}
    for tri, lower in (("lower", 1), ("upper", 0)):
        _, info = lapack.dpotrf(M, lower=lower, clean=1)
        out[f"potrf_{tri}_info"] = int(info)
        try:
            linalg.cholesky(M, lower=bool(lower))
            out[f"scipy_cholesky_{tri}"] = {"ok": True, "message": ""}
        except linalg.LinAlgError as e:
            out[f"scipy_cholesky_{tri}"] = {"ok": False, "message": str(e)}
    ev = np.linalg.eigvalsh(0.5 * (M + M.T))
    out["eig_M_min"], out["eig_M_max"] = float(ev[0]), float(ev[-1])
    out["eigh"] = {}
    for d in DRIVERS:
        try:
            lam, V = linalg.eigh(K, M, driver=d)
            out["eigh"][d] = {"ok": True, "lambda_1": float(lam[0]),
                              "backward_error": generalized_backward_error(K, M, float(lam[0]), V[:, 0])}
        except (linalg.LinAlgError, ValueError) as e:
            out["eigh"][d] = {"ok": False, "message": str(e)}
    return out


def p1_raw_monomial_pair(n: int = 12):
    """보충 표 S7의 원시 단항 행과 같은 조립: `run_p1`처럼 n_q = max(2 n_dof, 12)."""
    from ..bases.monomial import MonomialBasis
    from ..problems import p1_beam as p1
    basis = MonomialBasis(n)
    r = p1.solve(basis, n_q=max(2 * basis.n_dof, 12), n_modes=3)
    return r["K"], r["M"]


def coarse_target_seconds(reps: int = 7, modes=(1, 2, 3)) -> dict:
    """arm (c)의 목표 `coarse_ritz_lambda`(5항 직교화 단항 Ritz, n_q = 400)를 그대로 불러 벽시계를 잰다."""
    from ..neural.curriculum import coarse_ritz_lambda
    rows = []
    for m in modes:
        ts, val = [], None
        for _ in range(reps):
            t0 = time.perf_counter()
            val = coarse_ritz_lambda(m)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        rows.append({"mode": m, "reps": reps, "lambda": float(val), "seconds_median": ts[len(ts) // 2],
                     "seconds_min": ts[0], "seconds_max": ts[-1]})
    return {"rows": rows, "timed": "coarse_ritz_lambda(mode): basis construction + assembly + eigensolve"}


def _versions() -> dict:
    import scipy
    return {"numpy": np.__version__, "scipy": scipy.__version__}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join("docs", "_generated", "data", "paper2"))
    ap.add_argument("--part", choices=("all", "cholesky"), default="all")
    args = ap.parse_args(argv)
    K, M = p1_raw_monomial_pair(12)
    chol = {**_versions(), **cholesky_boundary(K, M)}
    if args.part == "cholesky":
        path = os.path.join(args.data, f"paperA_mlst_v3_cholesky_scipy{chol['scipy']}.json")
        manifest.write_json(path, manifest.build({"driver": "paperA_mlst_v3_checks.cholesky", "result": chol}))
        print(json.dumps(chol, indent=1))
        return chol
    z = np.load(os.path.join(args.data, "p1_fp32_raw_sets.npz"))
    res = {"table16_timing_spans": table16_timing_spans(args.data),
           "fp32_fixed_span_reference": reference_check(z["raw_shapes"], z["wq"]),
           "p1_raw_monomial_n12_cholesky": chol,
           "c_curriculum_coarse_target": coarse_target_seconds()}
    manifest.write_json(os.path.join(args.data, "paperA_mlst_v3_checks.json"), res)
    manifest.write_json(os.path.join(args.data, "manifest_paperA_mlst_v3_checks.json"), manifest.build({
        "driver": "paperA_mlst_v3_checks.main", "launch": manifest.launch_provenance(),
        "inputs": ["p1_single_seed_cost.csv", "p1p4_classical_cost_repeats.csv", "p1_fp32_raw_sets.npz"],
        "why": "v3 Word comments (2026-10-15): timing spread vs table 16 ranking, precision of the fp64 reference "
               "in table S21, Cholesky/eigh boundary at raw monomial N = 12, untimed coarse target of arm (c)"}))
    print(json.dumps({k: (v if k != "fp32_fixed_span_reference" else {"max_rel_diff": v["max_rel_diff"]})
                      for k, v in res.items()}, indent=1, default=str))
    return res


if __name__ == "__main__":
    main()
