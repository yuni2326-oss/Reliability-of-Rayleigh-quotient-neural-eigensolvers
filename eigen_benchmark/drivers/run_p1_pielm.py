"""(f) Eig-PIELM 전용 드라이버 — **torch를 import하지 않는다.**

PIELM은 역전파가 없어 numpy/CPU만 쓴다. 그런데 `run_p1_neural`이 최상단에서 torch를
import하므로 거기에 두면 GPU 컨테이너에서만 돌게 되고, 그 결과 **고전 기저(호스트
numpy/CPU)와 다른 환경에서 측정**된다. 실제로 그 상태에서 nf=20이 2.01 s, nf=160이
0.52 s로 역전하는 값이 나왔다(같은 코드를 호스트에서 직접 재면 둘 다 15~47 ms).

비용 비교의 회계를 지키려면 같은 계열은 같은 환경에서 재야 한다. 그래서 분리했다 —
이 드라이버는 `equip-venv`(호스트)에서 돌고, 계획 1의 고전 기저와 동일 환경이다.

    PYTHONPATH=/home/super/project /home/super/equip-venv/bin/python \
      -m eigen_benchmark.drivers.run_p1_pielm
"""
from __future__ import annotations

import os

import numpy as np

from .. import metrics as mt
from ..neural.pielm import solve_pielm
from ..problems import p1_beam as p1
from . import manifest


def run_pielm_seeds(outdir=None, n_seeds: int = 50, feats=(20, 40, 80, 160),
                    n_modes: int = 3) -> dict:
    """(f) PIELM만 50시드로 재실행해 기본 격자의 PIELM 행을 **교체**한다.

    감사 지적: 다른 여섯 arm에 50시드+Wilson을 요구하면서 PIELM만 단일 추출
    (n=1, 구간 없음)로 p_correct=1.00을 보고했다. 1회 15~18 ms이므로 면제할
    이유가 없다 — 특히 κ(M) ≈ 1e300 규모에서 단일 추출의 대표성은 보장되지 않는다.
    전체 격자(수 시간)를 다시 돌리지 않고 PIELM 행만 갈아끼운다."""
    d = manifest.ensure_outdir(outdir)
    path = os.path.join(d, "p1_neural_mode_sweep.csv")
    keep = [r for r in manifest.read_csv(path)
            if not str(r["arm"]).startswith("f_eig_pielm")]
    ref = p1.beta_roots(n_modes) ** 4
    rows, recs = [], []
    for nf in feats:
        cnt = {m: {"correct": 0, "spurious": 0, "non_converged": 0}
               for m in range(1, n_modes + 1)}
        secs, ranks, kappas, spectra = [], [], [], []
        elams = {m: [] for m in range(1, n_modes + 1)}
        for sd in range(n_seeds):
            o = solve_pielm(nf, n_modes=n_modes, seed=sd)
            secs.append(o["seconds"]); ranks.append(o["rank_used"])
            kappas.append(o["kappa_M"]); spectra.append(o["rank_spectrum"])
            for m in range(1, n_modes + 1):
                e = (abs(o["Lam"][m - 1] - ref[m - 1]) / ref[m - 1]
                     if o["cholesky_ok"] and np.isfinite(o["Lam"][m - 1])
                     else float("nan"))
                elams[m].append(e)
                oc = ("non_converged" if not np.isfinite(e)
                      else "correct" if e <= mt.ELAM_MAX else "spurious")
                cnt[m][oc] += 1
                recs.append({"arm": f"f_eig_pielm(nf={nf})", "mode": m, "seed": sd,
                             "ladder": "n/a", "nodes": "gauss", "outcome": oc,
                             "matched_mode": m, "mac": float("nan"), "e_lam": e,
                             "seconds": o["seconds"], "rank_used": o["rank_used"],
                             "kappa_M": o["kappa_M"]})
        for m in range(1, n_modes + 1):
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
            for t in spectra[0]:
                row[f"rank_tol_{t}"] = float(np.median([sp[t] for sp in spectra]))
            rows.append(row)
        print(f"  [pielm nf={nf}] p_correct(mode1..{n_modes}) = "
              f"{[cnt[m]['correct'] / n_seeds for m in range(1, n_modes + 1)]}",
              flush=True)
    manifest.write_csv(path, keep + rows)
    manifest.write_jsonl(os.path.join(d, "p1_neural_pielm_records.jsonl"), recs)
    return {"pielm": rows, "outdir": d}


if __name__ == "__main__":
    run_pielm_seeds()
