"""(g) 시공간 PINN — 범위문서 §7의 "time-domain PINN negative result".

논문 1에서 관측된 것(준정적 문제에서 미수렴, 고전 다항이 압도)을 논문 2의 오차귀속 틀
안에서 재현 가능하게 만든다. 논문 1 코드는 수정하지 않고 **재구현**했다(폴더 규약).

negative result를 정직하게 쓰려면 "튜닝 부족"과 "구조적 한계"를 가려야 한다. 관측창
n_periods를 쓸어 실측한 결과, **구조적 한계 쪽이었고 그 정체는 예상과 달랐다.**

| 주기 | e_λ | 빈 간격 | ω̂ | PDE 잔차 loss |
|---|---|---|---|---|
| 4 | 3.9e−3 | 0.50 | 3.509 | 5.8e−1 |
| 8 | 2.4e−1 | 0.25 | 3.071 | 3.6e−1 |
| 16 | 3.4e−1 | 0.125 | 2.851 | 5.0e−2 |
| 32 | 4.4e−1 | 0.0625 | 2.632 | 7.1e−2 |

두 가지가 읽힌다.

1. **잔차와 고유값 오차가 역상관이다.** loss가 한 자리 내려가는 동안 e_λ가 100배 오른다.
   목적함수가 고유값이 아니면 목적함수를 잘 최소화해도 고유값이 개선되지 않는다 —
   이것이 (A) Rayleigh 몫 접근과 (B)를 가르는 구조적 차이이고 §5.4의 논지다.
2. **시간 지평이 길어질수록 나빠진다.** ω̂가 −25 % 단조 감소한다. 여러 주기에 걸쳐
   진동을 유지하지 못한다.

**빈 간격은 하한이 아니다.** 창을 정수 주기로 잡으므로 ω/Δω = n_periods로 참조 주파수가
항상 정확히 빈에 떨어진다. 그래서 창이 짧으면(격자가 거칠면) 틀린 주파수도 맞는 빈으로
반올림돼 오차가 **작아 보인다**(4주기 e_λ = 3.9e−3). 거친 격자가 모형오차를 가리고 고운
격자가 드러낸다 — 빈 간격은 "이 오차가 격자 잡음인가 실제 모형오차인가"를 가르는 데만 쓴다.

비교 기준: 같은 고유값을 Rayleigh 몫 접근은 (a)~(f) arm이, 고전 기저는 6 DOF
정규직교 단항식이 0.14 ms에 e_λ = 1.6e-10으로 낸다(`p1_cheapest_to_epsilon.csv`).
"""
from __future__ import annotations

import os

from ..neural.spacetime import bin_spacing, solve_spacetime
from . import manifest

PERIOD_SWEEP = (4.0, 8.0, 16.0, 32.0)


def main(outdir=None, n_periods=PERIOD_SWEEP, iters: int = 4000, n_seeds: int = 3,
         n_col: int = 4096, quick: bool = False, device=None) -> dict:
    d = manifest.ensure_outdir(outdir)
    if quick:
        n_periods, iters, n_seeds, n_col = (4.0,), 30, 1, 256
    rows = []
    for T in n_periods:
        for sd in range(n_seeds):
            o = solve_spacetime(n_periods=T, iters=iters, n_col=n_col,
                                seed=sd, device=device)
            binw = o["rel_lambda_bin"]
            rows.append({
                "arm": o["arm"], "n_periods": T, "seed": sd, "iters": iters,
                "n_col": n_col, "n_params": o["n_params"],
                "T_nondim": o["T_nondim"],
                "omega_hat": o["omega_hat"], "omega_ref": o["omega_ref"],
                "Lam_hat": float(o["Lam"][0]), "Lam_ref": o["Lam_ref"],
                "e_lam": o["e_lam"],
                "rel_lambda_bin": binw,
                # 오차가 빈 간격을 넘는가 = 격자 잡음이 아니라 실제 모형오차인가
                "e_lam_over_bin": (o["e_lam"] / binw if binw > 0 else float("nan")),
                "error_exceeds_one_bin": bool(o["e_lam"] > binw),
                "loss_final": o["loss_final"], "seconds": o["seconds"]})
            print(f"  [spacetime T={T:g} s{sd}] e_lam={o['e_lam']:.3e} "
                  f"빈={binw:.3e} 비={rows[-1]['e_lam_over_bin']:.1f} "
                  f"loss={o['loss_final']:.2e} {o['seconds']:.0f}s", flush=True)

    # 참고: 하한만 계산한 순수 이론표 — 학습 없이도 알 수 있는 것
    theory = [{**bin_spacing(T),
               "note": "FFT bin width. An integer-period window places the reference "
                       "frequency exactly on bin n_periods, so this is NOT an error "
                       "floor; it only separates grid noise from model error"}
              for T in (4, 8, 16, 32, 64, 128, 256)]

    manifest.write_csv(os.path.join(d, "p1_spacetime.csv"), rows)
    manifest.write_csv(os.path.join(d, "p1_spacetime_bins.csv"), theory)
    manifest.write_json(os.path.join(d, "manifest_p1_spacetime.json"),
                        manifest.build({
                            "driver": "run_p1_spacetime.main",
                            "n_periods": list(n_periods), "iters": iters,
                            "n_seeds": n_seeds, "n_col": n_col,
                            "purpose": "negative result — 잔차와 고유값 오차의 역상관, "
                                       "그리고 시간 지평에 따른 악화",
                            "reimplemented_from":
                                "impeller_pinn(논문1) 시간영역 PINN을 수정 없이 재구현"}))
    print(f"[run_p1_spacetime] {d} — {len(rows)}행 + 하한 {len(theory)}행")
    return {"spacetime": rows, "floor": theory, "outdir": d}


if __name__ == "__main__":
    main()
