"""P2 드라이버 — 정확 Bessel 기준(m 0–4 × n 0–3) + 기저연구 + 조건수 + 축퇴 지표.

  cd /home/super/project && PYTHONPATH=/home/super/project \
      /home/super/equip-venv/bin/python -m eigen_benchmark.drivers.run_p2
"""
from __future__ import annotations

import os

import numpy as np

from .. import degeneracy as dg
from .. import metrics as mt
from ..bases.bspline import BSplineC0Basis
from ..bases.monomial import MonomialBasis, orthonormalized
from ..bases.radial_fem import solve_radial_fem
from ..bases.recurrence import (ChebyshevBasis, IntegratedLegendreBasis,
                                ShiftedLegendreBasis)
from ..conditioning import condition_report
from ..cost import timed
from ..metrics import rel_errors_padded
from ..problems import p2_annulus as p2
from ..reference import bessel_annulus as ba
from . import manifest

DOFS = (6, 8, 10, 12, 16)
QUICK_DOFS = (10,)


def _bases(n):
    return [
        ("monomial_raw", MonomialBasis(n)),
        ("monomial_orthonormalized", orthonormalized(MonomialBasis(n))),
        ("shifted_legendre", ShiftedLegendreBasis(n)),
        ("integrated_legendre", IntegratedLegendreBasis(n)),
        ("chebyshev", ChebyshevBasis(n)),
        ("bspline_p3", BSplineC0Basis(max(n - 2, 3), degree=3)),
    ]


def main(outdir=None, quick: bool = False) -> dict:
    d = manifest.ensure_outdir(outdir)
    ms = (0, 2) if quick else (0, 1, 2, 3, 4)
    n_rad = 2 if quick else 4
    dofs = QUICK_DOFS if quick else DOFS

    reference = []
    for m in (0, 1, 2, 3, 4):
        k = ba.annulus_k_roots(m, n_modes=4)
        for i in range(4):
            reference.append({"m": m, "radial_order": i, "kb": float(k[i]),
                              "Lambda_k4": float(k[i] ** 4),
                              "degenerate": bool(m > 0),
                              "reference": "exact Bessel (mpmath dps=50)"})

    basis_rows, cond_rows = [], []
    for m in ms:
        ref_lam = ba.annulus_k_roots(m, n_modes=n_rad) ** 4
        for n in dofs:
            entries = _bases(n) + [("radial_hermite_fem", None)]
            for label, basis in entries:
                with timed("solve") as t:
                    if basis is None:
                        r = solve_radial_fem(20 if quick else 40, m, n_modes=n_rad)
                    else:
                        r = p2.solve(basis, m, n_q=max(2 * n, 16), n_modes=n_rad)
                row = {"m": m, "basis": label, "n_dof": int(r["K"].shape[0]),
                       "cholesky_ok": r["cholesky_ok"], "seconds": t["seconds"]}
                e = (rel_errors_padded(r["Lam"], ref_lam, n_rad)
                     if r["cholesky_ok"] else [float("nan")] * n_rad)
                for k in range(n_rad):
                    row[f"e_lam_mode{k + 1}"] = e[k]
                basis_rows.append(row)
                rep = condition_report(r["K"], r["M"], dps=50, n_eig=2)
                cond_rows.append({"m": m, "basis": label,
                                  "n_dof": int(r["K"].shape[0]),
                                  **{k: (v if not isinstance(v, list) else v[0])
                                     for k, v in rep.items()}})

    grid = dg.polar_grid(30 if quick else 48, 48 if quick else 96)
    Wt = np.diag(grid[2])
    deg_rows = []
    for m in (0, 1, 2, 3, 4):
        k = ba.annulus_k_roots(m, n_modes=1)[0]
        if m == 0:
            deg_rows.append({"m": m, "individual_mac_rotated": 1.0,
                             "subspace_mac_rotated": 1.0,
                             "max_principal_angle_rad": 0.0,
                             # **논문에 인쇄되는 문자열은 영문이다.** 이 열이 Table 7의 note로 그대로 나가는데
                             # 한국어가 영어 원고에 박혀 있었다.
                             "note": "not degenerate: sin(0*theta) = 0"})
            continue
        P = dg.degenerate_pair(k, m, grid)
        Q = dg.rotated_pair(k, m, grid, alpha=np.pi / 5)
        deg_rows.append({
            "m": m,
            "individual_mac_rotated": float(mt.mac(P[:, 0], Q[:, 0], Wt)),
            "subspace_mac_rotated": float(mt.subspace_mac(P, Q, Wt)),
            "max_principal_angle_rad": float(np.max(mt.principal_angles(P, Q, Wt))),
            "note": "rotated pair spans the same eigenspace"})

    manifest.write_csv(os.path.join(d, "p2_reference.csv"), reference)
    manifest.write_csv(os.path.join(d, "p2_basis_study.csv"), basis_rows)
    manifest.write_csv(os.path.join(d, "p2_conditioning.csv"), cond_rows)
    manifest.write_csv(os.path.join(d, "p2_degeneracy.csv"), deg_rows)
    manifest.write_jsonl(os.path.join(d, "p2_basis_study.jsonl"), basis_rows)
    manifest.write_json(os.path.join(d, "manifest_p2.json"),
                        manifest.build({"driver": "run_p2", "quick": quick,
                                        "nondimensional": "Lambda = (k b)^4, b = 1; "
                                        "문제는 a/b와 nu만으로 정해진다",
                                        "geometry": p2.P2_GEOMETRY,
                                        "m_values": list(ms), "dofs": list(dofs)}))
    return {"reference": reference, "basis_study": basis_rows,
            "conditioning": cond_rows, "degeneracy": deg_rows, "outdir": d}


if __name__ == "__main__":
    out = main()
    print(f"[run_p2] {out['outdir']} — 기준 {len(out['reference'])}행, "
          f"기저 {len(out['basis_study'])}행, 축퇴 {len(out['degeneracy'])}행")
