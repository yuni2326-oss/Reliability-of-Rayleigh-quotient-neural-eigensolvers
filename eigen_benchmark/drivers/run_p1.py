"""P1 드라이버 — 해석기준 모드 1–10 + 기저별 수렴·조건수 표를 재생성한다.

  PYTHONPATH=/home/super/project /home/super/equip-venv/bin/python \
      -m eigen_benchmark.drivers.run_p1
"""
from __future__ import annotations

import os

import numpy as np

from ..bases.bspline import BSplineC0Basis
from ..bases.monomial import MonomialBasis, orthonormalized
from ..bases.recurrence import (ChebyshevBasis, IntegratedLegendreBasis,
                                ShiftedLegendreBasis)
from ..conditioning import condition_report
from ..cost import timed
from ..metrics import rel_errors_padded
from ..problems import p1_beam as p1
from . import manifest

GLOBAL_DOFS = (4, 6, 8, 10, 12, 16, 20, 24)
QUICK_DOFS = (6, 12)


# **기저 구성을 비용에 포함한다.** 이전에는 `_global_bases(n)`가 루프 헤더에서 기저를
# 만들고 `timed`가 풀이만 재서, 정규직교 단항식의 Householder QR(n=6에서 70 ms)이
# 청구되지 않았다 — 풀이는 0.105 ms이므로 실제 비용의 1/400만 세고 있었다. 신경 arm은
# "처음부터 학습"으로 재면서 고전은 "기저는 이미 있다고 치고" 잰 셈이라 비용표의 분모가
# 무너진다. 구성과 풀이를 따로 재서 둘 다 남긴다 — 같은 기저로 여러 문제를 푸는 상각
# 시나리오는 독자가 두 열로 직접 계산할 수 있다.
def _global_basis_factories(n):
    """라벨 → **무인자 팩토리**. 구성 비용을 하나씩 재려면 즉시 만들면 안 된다.

    P1에 쓸 수 있는 기저 — 전부 클램프 BC를 만족하고 힌지 기구가 없다.
    (계면강화 기저는 broken form 전용이라 P1에서 제외한다: 램프의 굽힘에너지가 0)."""
    return [
        ("monomial_raw", lambda: MonomialBasis(n)),
        ("monomial_orthonormalized", lambda: orthonormalized(MonomialBasis(n))),
        ("shifted_legendre", lambda: ShiftedLegendreBasis(n)),
        ("integrated_legendre", lambda: IntegratedLegendreBasis(n)),
        ("chebyshev", lambda: ChebyshevBasis(n)),
        ("bspline_p3", lambda: BSplineC0Basis(max(n - 2, 3), degree=3)),
    ]


def _global_bases(n):
    """즉시 생성 버전 — 구성 비용을 재지 않는 곳(조건수 보고 등)에서 쓴다."""
    return [(label, make()) for label, make in _global_basis_factories(n)]


def main(outdir=None, quick: bool = False) -> dict:
    d = manifest.ensure_outdir(outdir)
    dofs = QUICK_DOFS if quick else GLOBAL_DOFS
    n_modes = 3 if quick else 10

    betas = p1.beta_roots(14)
    reference = [{"mode": i + 1, "beta_L": float(betas[i]),
                  "Lambda": float(betas[i] ** 4),
                  "reference": "analytic (Euler-Bernoulli)",
                  "uncertainty_rel": 1e-14} for i in range(10)]

    basis_rows, cond_rows = [], []
    ref_lam = betas[:n_modes] ** 4
    for n in dofs:
        for label, make in _global_basis_factories(n):
            with timed("construct") as tc:         # **그 기저 하나만** 만든다
                basis = make()
            nq = max(2 * basis.n_dof, 12)          # 구적차수 하한 = 자유도(rank 보장)
            with timed("solve") as t:
                r = p1.solve(basis, n_q=nq, n_modes=n_modes)
            row = {"basis": label, "n_dof": basis.n_dof, "n_q_per_segment": nq,
                   "cholesky_ok": r["cholesky_ok"],
                   "seconds_construct": tc["seconds"],
                   "seconds_solve": t["seconds"],
                   # 비용표가 쓰는 값 = 처음부터 답을 얻는 데 드는 시간
                   "seconds": tc["seconds"] + t["seconds"]}
            e = (rel_errors_padded(r["Lam"], ref_lam, n_modes)
                 if r["cholesky_ok"] else [float("nan")] * n_modes)
            for k in range(n_modes):
                row[f"e_lam_mode{k + 1}"] = e[k]
            basis_rows.append(row)
            rep = condition_report(r["K"], r["M"], dps=50, n_eig=3)
            cond_rows.append({"basis": label, "n_dof": basis.n_dof,
                              **{k: (v if not isinstance(v, list) else v[0])
                                 for k, v in rep.items()}})

    manifest.write_csv(os.path.join(d, "p1_reference.csv"), reference)
    manifest.write_csv(os.path.join(d, "p1_basis_study.csv"), basis_rows)
    manifest.write_csv(os.path.join(d, "p1_conditioning.csv"), cond_rows)
    manifest.write_jsonl(os.path.join(d, "p1_basis_study.jsonl"), basis_rows)
    manifest.write_json(os.path.join(d, "manifest_p1.json"),
                        manifest.build({"driver": "run_p1", "quick": quick,
                                        "dofs": list(dofs), "n_modes": n_modes}))
    return {"reference": reference, "basis_study": basis_rows,
            "conditioning": cond_rows, "outdir": d}


if __name__ == "__main__":
    out = main()
    print(f"[run_p1] {out['outdir']} — 기준 {len(out['reference'])}행, "
          f"기저 {len(out['basis_study'])}행")
