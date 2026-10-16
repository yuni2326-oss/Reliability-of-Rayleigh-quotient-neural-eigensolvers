"""P3 드라이버 — 공통약형식 기저연구(§5.3) 전체를 재생성한다.

  PYTHONPATH=/home/super/project /home/super/equip-venv/bin/python \
      -m eigen_benchmark.drivers.run_p3
"""
from __future__ import annotations

import os

import numpy as np

from ..bases.bspline import BSplineC0Basis
from ..bases.enriched import EnrichedBasis
from ..bases.monomial import MonomialBasis, orthonormalized
from ..bases.recurrence import (ChebyshevBasis, IntegratedLegendreBasis,
                                ShiftedLegendreBasis)
from ..conditioning import condition_report
from ..cost import timed
from ..metrics import rel_errors_padded
from ..problems import p1_beam as p1
from ..problems import p3_spring as p3
from ..quadrature_study import doubling_table
from ..reference.transfer_matrix import (fp64_vs_highprec_betas,
                                         kappa_from_k_hat)
from . import manifest

DOFS = (8, 12, 16, 20, 24)
QUICK_DOFS = (12,)


def _basis_factories(n, xc):
    """(라벨, **무인자 팩토리**, 점프표현가능). 구성 비용을 하나씩 재려면 지연생성해야 한다.

    점프표현 여부가 §5.3의 대조축이다."""
    return [
        ("monomial_raw", lambda: MonomialBasis(n), False),
        ("monomial_orthonormalized", lambda: orthonormalized(MonomialBasis(n)), False),
        ("shifted_legendre", lambda: ShiftedLegendreBasis(n), False),
        ("integrated_legendre", lambda: IntegratedLegendreBasis(n), False),
        ("chebyshev", lambda: ChebyshevBasis(n), False),
        ("bspline_C0",
         lambda: BSplineC0Basis(max(n - 4, 5), degree=3, xc=xc, c0=True), True),
        ("enriched_split",
         lambda: EnrichedBasis(max(n - 2, 4), xc=xc, n_enrich=2), True),
        # 같은 강화공간을 정규직교 좌표로 — 공간의 적합성(e_approx)과 좌표의 조건수
        # (e_algebraic)를 분리한다. 단항식 부모는 고차에서 Cholesky가 깨지기 때문.
        ("enriched_split_orthonormalized",
         lambda: orthonormalized(EnrichedBasis(max(n - 2, 4), xc=xc, n_enrich=2)), True),
    ]


def _bases(n, xc):
    """즉시 생성 버전 — 구성 비용을 재지 않는 곳에서 쓴다."""
    return [(label, make(), j) for label, make, j in _basis_factories(n, xc)]


def main(outdir=None, quick: bool = False) -> dict:
    d = manifest.ensure_outdir(outdir)
    xc = p3.P3_CONFIG["xc_over_L"]
    k_hats = ((p3.P3_CONFIG["k_hat_central"],) if quick
              else p3.P3_CONFIG["k_hats"])
    dofs = QUICK_DOFS if quick else DOFS
    n_modes = 3 if quick else 6
    n_elem_fem = 20 if quick else 40

    reference, prec_rows = [], []
    for kh in k_hats:
        b = p3.reference_betas(kh, n_modes=n_modes)
        kap = kappa_from_k_hat(kh)
        for i, bi in enumerate(b):
            reference.append({
                "k_hat": float(kh), "kappa": float(kap),
                "mode": i + 1, "beta_L": float(bi), "Lambda": float(bi ** 4),
                "reference": "transfer matrix (mpmath dps=50)"})
        pv = fp64_vs_highprec_betas(xc, kap, n_modes=n_modes)
        for i, rd in enumerate(pv["rel_diff"]):
            prec_rows.append({"k_hat": float(kh), "mode": i + 1,
                              "beta_fp64": pv["fp64"][i],
                              "beta_highprec": pv["highprec"][i],
                              "rel_diff": float(rd)})

    basis_rows, cond_rows = [], []
    for kh in k_hats:
        ref_lam = p3.reference_betas(kh, n_modes=n_modes) ** 4
        uni_lam = p1.beta_roots(n_modes) ** 4
        for n in dofs:
            entries = (_basis_factories(n, xc)
                       + [("hermite_fem_split_rotation", None, True)])
            for label, make, jump_ok in entries:
                nq = max(2 * n, 12)
                # 기저 구성도 비용이다 — run_p1과 같은 이유(그 주석 참조).
                # **그 기저 하나만** 만든다.
                with timed("construct") as tc:
                    basis = make() if make is not None else None
                with timed("solve") as t:
                    if basis is None:
                        r = p3.solve_fem(n_elem_fem, kh, n_modes=n_modes)
                    else:
                        r = p3.solve_basis(basis, kh, n_q=nq, n_modes=n_modes)
                row = {"k_hat": float(kh), "basis": label,
                       "n_dof": int(r["K"].shape[0]), "jump_capable": jump_ok,
                       "n_q_per_segment": nq if basis is not None else None,
                       "cholesky_ok": r["cholesky_ok"],
                       "seconds_construct": tc["seconds"],
                       "seconds_solve": t["seconds"],
                       "seconds": tc["seconds"] + t["seconds"]}
                if r["cholesky_ok"]:
                    e = rel_errors_padded(r["Lam"], ref_lam, n_modes)
                    eu = rel_errors_padded(r["Lam"], uni_lam, n_modes)
                    row["e_lam_vs_uniform_mode1"] = eu[0]
                else:
                    e = [float("nan")] * n_modes
                    row["e_lam_vs_uniform_mode1"] = float("nan")
                for k in range(n_modes):
                    row[f"e_lam_mode{k + 1}"] = e[k]
                basis_rows.append(row)
                rep = condition_report(r["K"], r["M"], dps=50, n_eig=3)
                cond_rows.append({"k_hat": float(kh), "basis": label,
                                  "n_dof": int(r["K"].shape[0]),
                                  **{k: (v if not isinstance(v, list) else v[0])
                                     for k, v in rep.items()}})

    # 구적분리는 **좌표가 건전한** 기저로만 의미가 있다 — 원시 단항식 부모의 조건수
    # 붕괴가 구적 신호를 가리기 때문(NaN 남발). 정규직교 강화기저와 C⁰ B-spline을 쓴다.
    k0 = k_hats[0] if quick else p3.P3_CONFIG["k_hat_central"]
    quad_rows = []
    def _enr():
        return orthonormalized(EnrichedBasis(12, xc=xc, n_enrich=2))

    def _bsp():
        return BSplineC0Basis(10, degree=3, xc=xc, c0=True)

    for label, factory, aligned in (
            ("enriched_orthonormalized", _enr, True),
            ("enriched_orthonormalized", _enr, False),
            ("bspline_C0", _bsp, True),
            ("bspline_C0", _bsp, False)):
        for row in doubling_table(
                factory,
                lambda b, nq, al=aligned: p3.solve_basis(
                    b, k0, n_q=nq, n_modes=1, split_at_xc=al)["Lam"][0],
                [12, 16, 24, 32, 64]):
            quad_rows.append({"basis": label, "aligned": aligned, **row})

    manifest.write_csv(os.path.join(d, "p3_reference.csv"), reference)
    manifest.write_csv(os.path.join(d, "p3_basis_study.csv"), basis_rows)
    manifest.write_csv(os.path.join(d, "p3_conditioning.csv"), cond_rows)
    manifest.write_csv(os.path.join(d, "p3_precision_fp64_vs_mp.csv"), prec_rows)
    manifest.write_csv(os.path.join(d, "p3_quadrature_separation.csv"), quad_rows)
    manifest.write_jsonl(os.path.join(d, "p3_basis_study.jsonl"), basis_rows)
    manifest.write_json(os.path.join(d, "manifest_p3.json"),
                        manifest.build({"driver": "run_p3", "quick": quick,
                                        "xc_over_L": xc, "k_hats": list(k_hats),
                                        "dofs": list(dofs), "n_modes": n_modes,
                                        "n_elem_fem": n_elem_fem}))
    return {"reference": reference, "basis_study": basis_rows,
            "conditioning": cond_rows, "precision": prec_rows,
            "quadrature": quad_rows, "outdir": d}


if __name__ == "__main__":
    out = main()
    print(f"[run_p3] {out['outdir']} — 기준 {len(out['reference'])}행, "
          f"기저 {len(out['basis_study'])}행, 구적 {len(out['quadrature'])}행")
