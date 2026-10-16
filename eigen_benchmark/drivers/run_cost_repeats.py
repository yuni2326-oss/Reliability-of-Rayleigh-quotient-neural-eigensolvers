"""비용표에 쓰이는 셀을 **반복** 측정한다 — 고전 쪽도 산포를 갖게 한다.

신경 쪽은 `run_single_seed_cost`가 3회 반복의 median·min·max를 남기는데, 고전 쪽은 단
한 번의 측정이었다. 비율의 분모가 방어되지 않은 상태다.

그리고 **짧은 측정일수록 오염에 취약하다.** 신경은 40 s라 수백 ms의 방해가 1 %지만
고전은 0.14 ms라 같은 방해가 몇 배가 된다. 이 기계는 GPU와 호스트 CPU를 상주 추론
서비스와 공유하므로, 보호가 필요한 쪽이 오히려 무방비였다.

여기서는 비용표가 실제로 인용하는 고전 구성만 골라 여러 번 재고, 중앙값과 범위를 남긴다.
"""
from __future__ import annotations

import os
import time

import numpy as np

from ..bases.monomial import MonomialBasis, orthonormalized
from ..bases.recurrence import ChebyshevBasis, IntegratedLegendreBasis, \
    ShiftedLegendreBasis
from ..problems import p1_beam as p1
from ..problems import p4_lshape as p4
from . import manifest

BASES = {"monomial_raw": lambda n: MonomialBasis(n),
         "monomial_orthonormalized": lambda n: orthonormalized(MonomialBasis(n)),
         "shifted_legendre": lambda n: ShiftedLegendreBasis(n),
         "chebyshev": lambda n: ChebyshevBasis(n),
         "integrated_legendre": lambda n: IntegratedLegendreBasis(n)}


def main(outdir=None, reps: int = 7, dofs=(6, 8, 10, 12),
         p4_grids=((4, 3.0), (8, 3.0), (8, 1.0), (16, 3.0)),
         quick: bool = False) -> dict:
    d = manifest.ensure_outdir(outdir)
    if quick:
        reps, dofs, p4_grids = 2, (6,), ((4, 3.0),)
    rows = []

    def rec(problem, solver, size, ts):
        ts = sorted(ts)
        m = ts[len(ts) // 2]
        rows.append({"problem": problem, "solver": solver, "size": size,
                     "reps": len(ts), "seconds_median": m,
                     "seconds_min": ts[0], "seconds_max": ts[-1],
                     "spread_rel": (ts[-1] - ts[0]) / max(m, 1e-300),
                     "note": "repeated on a machine that also hosts a resident "
                             "inference service; the range is the contention exposure"})
        print(f"  [{problem} {solver[:26]:26s} {size:>5}] median {m * 1e3:9.3f} ms "
              f"[{ts[0] * 1e3:.3f}, {ts[-1] * 1e3:.3f}]", flush=True)

    # **구성도 타이머 안에서** 잰다. 이 드라이버의 첫 판은 `b = make(n)`을 타이머 밖에
    # 두어 run_p1과 똑같은 결함을 반복했다 — 고치려던 결함을 측정 코드가 재현한 셈이다.
    # 구적차수도 드라이버와 같은 규칙(max(2*dof, 12))을 쓴다.
    for name, make in BASES.items():
        for n in dofs:
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                b = make(n)
                p1.solve(b, n_q=max(2 * n, 12), n_modes=min(n, 6))
                ts.append(time.perf_counter() - t0)
            rec("P1", name, n, ts)

    for n, beta in p4_grids:
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            p4.solve(n, beta=beta, n_modes=6)
            ts.append(time.perf_counter() - t0)
        rec("P4", f"graded Q2 (beta={beta})", n, ts)

    manifest.write_csv(os.path.join(d, "p1p4_classical_cost_repeats.csv"), rows)
    manifest.write_json(os.path.join(d, "manifest_cost_repeats.json"),
                        manifest.build({
                            "driver": "run_cost_repeats.main", "reps": reps,
                            "dofs": list(dofs),
                            "timed": "basis construction + assembly + solve, "
                                     "n_q = max(2*n_dof, 12) as in run_p1",
                            "why": "the classical side of every cost ratio was a single "
                                   "unrepeated measurement while the neural side had "
                                   "three; short measurements are the more exposed to "
                                   "contention, so the denominator was the unprotected "
                                   "one"}))
    print(f"[run_cost_repeats] {d} — {len(rows)}행")
    return {"repeats": rows, "outdir": d}


if __name__ == "__main__":
    main()
