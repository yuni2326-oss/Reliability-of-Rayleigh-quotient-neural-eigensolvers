"""P4 드라이버 — L형 평면탄성 기준값(Richardson 외삽 + 불확실도) + 등급/균일 비교.

  cd /home/super/project && PYTHONPATH=/home/super/project \
      /home/super/equip-venv/bin/python -m eigen_benchmark.drivers.run_p4
"""
from __future__ import annotations

import os

from ..cost import timed
from ..problems import p4_lshape as p4
from . import manifest

UNCERTAINTY_TARGET = 1e-3          # §4.4 기준 불확실도 목표 0.1 %


def main(outdir=None, quick: bool = False) -> dict:
    d = manifest.ensure_outdir(outdir)
    # 세분 수준이 기준 정확도를 좌우한다. (3,6,12)에서는 모드 1–2만 0.1 %를 충족했고
    # (4,8,16)에서 6모드 전부 충족한다. **등급을 더 주는 것(β=5)은 오히려 악화** —
    # 과도 등급은 요소를 왜곡시키고 성근 영역의 해상도를 떨어뜨린다. β 세 수준을
    # 모두 산출해 그 비단조성을 데이터로 남긴다.
    ns = (2, 4, 8) if quick else (4, 8, 16)
    n_modes = 3 if quick else 6
    betas = (1.0, 3.0) if quick else (1.0, 3.0, 5.0)

    conv_rows, studies = [], {}
    for beta in betas:
        with timed(f"beta={beta}") as t:
            st = p4.convergence_study(ns, beta=beta, n_modes=n_modes)
        studies[beta] = st
        for row in st["rows"]:
            conv_rows.append({"beta": beta, "n": row["n"], "n_dof": row["n_dof"],
                              # 격자별 실측(이전에는 sweep 전체 시간을 모든 행에
                              # 복사해 n=4와 n=16이 같은 초로 기록됐다)
                              "seconds": row["seconds"],
                              "sweep_seconds": t["seconds"],
                              **{f"lam_mode{i + 1}": v
                                 for i, v in enumerate(row["Lam"])}})

    # **채택 규칙(사전선언)**: 모드별 불확실도의 최댓값이 가장 작은 β를 기준으로 쓴다.
    # "가장 강한 등급"을 쓰지 않는 이유는 위 주석대로 β가 클수록 좋지 않기 때문이다.
    best = min(betas, key=lambda b: max(e["uncertainty_rel"]
                                        for e in studies[b]["reference"]))
    reference = []
    for entry in studies[best]["reference"]:
        reference.append({
            "mode": entry["mode"], "beta": best,
            "lambda_extrapolated": entry["extrapolated"],
            "convergence_rate": entry["rate"],
            "uncertainty_rel": entry["uncertainty_rel"],
            "within_target": bool(entry["uncertainty_rel"] <= UNCERTAINTY_TARGET),
            "reference": f"Q2 graded mesh (beta={best}) + Richardson"})

    manifest.write_csv(os.path.join(d, "p4_reference.csv"), reference)
    manifest.write_csv(os.path.join(d, "p4_convergence.csv"), conv_rows)
    manifest.write_json(os.path.join(d, "manifest_p4.json"),
                        manifest.build({"driver": "run_p4", "quick": quick,
                                        "geometry": p4.P4_GEOMETRY,
                                        "levels": list(ns), "betas": list(betas),
                                        "uncertainty_target": UNCERTAINTY_TARGET}))
    return {"reference": reference, "convergence": conv_rows, "outdir": d}


if __name__ == "__main__":
    out = main()
    n_ok = sum(1 for r in out["reference"] if r["within_target"])
    print(f"[run_p4] {out['outdir']} — 기준 {len(out['reference'])}모드"
          f"(목표 불확실도 충족 {n_ok}개), 수렴 {len(out['convergence'])}행")
