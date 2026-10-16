"""P1에서 **고전 기저와 신경 arm을 같은 축에 올린다** — 범위문서 §7의 미이관 항목.

논문 2의 부제가 "classical solver 비교"인데, 계획 1(고전 기저)과 계획 3(신경 arm)이
서로 다른 격자에 있어 같은 축에서 비교된 표가 없었다. 공통축은 둘뿐이다:
**달성 정확도 e_λ**와 **벽시계**. DOF와 반복수는 서로 번역되지 않으므로 축으로 쓸 수 없다.

## 회계 기준을 명시한다 (감사 지적)

세 시간이 서로 다른 것을 센다.

| 계열 | `seconds`가 세는 것 | 1회 시도 비용 |
|---|---|---|
| 고전 (numpy/CPU) | 조립+`eigh` 1회 | 그대로 |
| 신경 (torch/GPU) | 50시드 vmap 앙상블 전체 | **n_seeds = 1 직접 측정** |
| PIELM (numpy/CPU) | 특징 뽑기+rank절단 GEP 1회 | 그대로 |

**환산계수는 쓰지 않는다.** 처음에는 앙상블 시간을 실측 배칭계수로 나눴는데, 그 계수가
같은 기계에서 2.53과 7.82로 나왔다(n=1이 39.5 s와 26.5 s, n=50이 100 s와 222 s). 원인은
GPU를 상주 LLM 스택과 공유하기 때문이고, 계수를 반복 측정해도 안정되지 않는다. 계수의
불안정이 결론에 실리는 것을 막는 유일한 방법은 **필요한 값을 직접 재는 것**이다 —
`run_single_seed_cost`가 각 arm을 n_seeds = 1로 돌려 1회 시도 벽시계를 median과
min/max로 남기고, 이 모듈은 그것을 그대로 쓴다. 앙상블 원값도 함께 싣는다.

**하드웨어가 다르다는 점은 보정하지 않는다.** 고전은 CPU, gradient arm은 GB10 GPU다.
이것은 정직하게 남기는 교란이지 없앨 수 있는 것이 아니다(§6 타당성 위협).

단 **PIELM은 같은 환경에서 재야 한다** — 역전파가 없어 GPU가 필요없으므로 고전 기저와
같은 numpy/CPU에서 돌린다. 처음에는 PIELM을 컨테이너에서, 고전을 호스트 venv에서 재서
nf=20이 2.01 s·nf=160이 0.52 s로 **역전**하는 결과가 나왔다(호스트 직접 측정은 둘 다
15–47 ms). 같은 계열 안에서도 환경이 섞이면 비교가 무너진다는 실례다.

## e_λ의 출처

기본격자 CSV는 gradient arm의 `e_λ`를 담지 않는다(`_cell`이 결과 카운트만 집계했다).
그대로 두면 그 arm들이 ε-표에서 **조용히 빠진다** — 그래서 `p1_neural_records.jsonl`의
시드별 레코드에서 `outcome == "correct"`인 것만 골라 중앙값을 낸다. 기본격자를 특정하려면
`ladder`와 `nodes`로도 걸러야 한다(같은 arm이 구적·사다리 하위연구에도 등장한다).

## 계열을 셋으로 쪼갠다

"classical vs neural"로 두 계열만 쓰면, 논문의 핵심 대조가 가려진다 — PIELM(역전파 없음)이
gradient arm보다 3자리 싸서 "neural" 슬롯을 독차지하고 gradient arm은 표에서 사라진다.
그런데 이 논문의 주제는 **gradient로 학습하는 Rayleigh 솔버**다. 그래서

| family | 무엇 | 학습 |
|---|---|---|
| `classical` | 고전 기저 + `eigh` | 없음 |
| `neural_randfeat` | Eig-PIELM (고정 랜덤특징 + rank절단 GEP) | 없음 |
| `neural_gradient` | (a)(a′)(b)(c)(d)(e) | Adam |

로 나눈다. 그러면 표가 세 질문에 각각 답한다 — 고전이 얼마나 싼가, 학습을 없애면 얼마나
싸지는가, 학습을 하면 얼마를 더 내는가.

## 성공확률

고전은 결정론적이므로 `cholesky_ok`이고 `e_λ ≤ ELAM_MAX`이면 p = 1, 아니면 0이다.
신경은 50시드의 사전등록 `p_correct`를 그대로 쓴다. E[T_success] = 1회비용 / p.
"""
from __future__ import annotations

import math
import os

from .. import metrics as mt
from ..cost import expected_time_to_success
from . import manifest

EPS_LEVELS = (1e-2, 1e-3, 1e-4, 1e-6)
FAMILIES = ("classical", "neural_randfeat", "neural_gradient")


def _classical_rows(data_dir: str, modes=(1, 2, 3)) -> list:
    """고전 기저: (기저, dof)마다 모드별 e_λ와 1회 벽시계."""
    rows = []
    for r in manifest.read_csv(os.path.join(data_dir, "p1_basis_study.csv")):
        ok = bool(r["cholesky_ok"])
        for m in modes:
            e = r.get(f"e_lam_mode{m}")
            e = float(e) if e not in (None, "") else float("nan")
            hit = bool(ok and math.isfinite(e) and e <= mt.ELAM_MAX)
            rows.append({
                "family": "classical", "solver": r["basis"], "mode": m,
                "e_lam_source": "p1_basis_study.csv",
                "size_label": "n_dof", "size": r["n_dof"],
                "iterations": None,       # 직접 풀이 — 반복이 없다
                "e_lam": e, "p_correct": float(hit),
                "seconds_reported": r["seconds"],
                "seconds_per_attempt": r["seconds"],       # 결정론 1회 = 보고값
                "accounting": "single deterministic solve (numpy/CPU)",
                "E_T_success": expected_time_to_success(r["seconds"], float(hit)),
                "factorization_ok": ok})
    return rows


def _base(arm) -> str:
    return str(arm).split("(")[0]


def _single_seed_cost(data_dir: str) -> dict:
    """(arm기본이름, mode) → 1회 시도 벽시계(median, min, max). 없으면 빈 dict."""
    try:
        rows = manifest.read_csv(os.path.join(data_dir, "p1_single_seed_cost.csv"))
    except FileNotFoundError:
        return {}
    return {(_base(r["arm"]), int(r["mode"])):
            (float(r["seconds_per_attempt"]), float(r["seconds_min"]),
             float(r["seconds_max"])) for r in rows}


# **사후 추가된 arm은 자기 파일에 쓴다.** 기본격자 CSV를 덮어쓰면 재실행 순서에 따라
# 행이 사라지고, 커밋된 데이터가 조용히 바뀐다. 그래서 소스를 목록으로 두고 병합한다.
NEURAL_SWEEPS = ("p1_neural_mode_sweep.csv", "p1_neural_qr.csv")
NEURAL_RECORDS = ("p1_neural_records.jsonl", "p1_neural_qr_records.jsonl")
P4_NEURAL_SWEEPS = ("p4_neural.csv", "p4_neural_arms.csv")


def _read_many(data_dir: str, names) -> list:
    """있는 파일만 읽어 합친다 — 사후 추가 파일이 없어도 기존 실행이 돌아야 한다."""
    rows = []
    for n in names:
        path = os.path.join(data_dir, n)
        if os.path.exists(path):
            rows += manifest.read_csv(path)
    return rows


def _elam_from_records(data_dir: str) -> dict:
    """(arm기본이름, mode) → correct 시드들의 e_λ 중앙값.

    기본격자만 보도록 `nodes == "gauss"`이고 `ladder`가 그 arm의 기본준위인 것만 쓴다.
    이 집계가 없으면 gradient arm이 ε-표에서 조용히 빠진다."""
    import json
    import statistics as st
    lines = []
    for n in NEURAL_RECORDS:
        path = os.path.join(data_dir, n)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                lines += f.readlines()
    bucket = {}
    for line in lines:
        r = json.loads(line)
        if r.get("nodes") != "gauss" or r.get("outcome") != "correct":
            continue
        base = str(r["arm"]).split("(")[0]
        lad = r.get("ladder")
        if lad not in ("I0", "I4"):        # I0=기본, I4=커리큘럼의 기본
            continue
        e = r.get("e_lam")
        if e is None or not math.isfinite(float(e)):
            continue
        bucket.setdefault((base, int(r["mode"])), []).append(float(e))
    return {k: st.median(v) for k, v in bucket.items()}


CORE_WIDTH, CORE_DEPTH = 64, 4      # neural/core.py의 기본 MLP — 부록 B와 같은 값


def _neural_rows(data_dir: str, modes=(1, 2, 3)) -> list:
    """신경 arm: 기본격자의 arm×모드. 앙상블 시간을 배칭계수로 1회 비용으로 환산."""
    from_recs = _elam_from_records(data_dir)
    direct = _single_seed_cost(data_dir)
    rows = []
    for r in _read_many(data_dir, NEURAL_SWEEPS):
        m = int(r["mode"])
        if m not in modes:
            continue
        arm = str(r["arm"])
        pielm = arm.startswith("f_eig_pielm")
        secs = r["seconds"]
        # PIELM은 시드 루프이므로 보고값이 이미 1회(중앙값)다. gradient arm은
        # n_seeds=1 직접 측정을 쓴다 — 환산계수가 불안정해 결론에 실을 수 없다.
        hit = direct.get((_base(arm), m))
        per = float(secs) if pielm else (hit[0] if hit else float("nan"))
        lo_hi = ("", "") if pielm or not hit else (hit[1], hit[2])
        e = r.get("e_lam")
        e = float(e) if e not in (None, "") else float("nan")
        if not math.isfinite(e):                      # gradient arm은 CSV에 없다
            e = from_recs.get((_base(arm), m), float("nan"))
        rows.append({
            "family": "neural_randfeat" if pielm else "neural_gradient",
            "solver": arm, "mode": m,
            "e_lam_source": ("mode_sweep.csv" if r.get("e_lam") not in (None, "")
                             else "records.jsonl (median over correct seeds)"),
            # **`size`가 두 가지를 뜻하고 있었다.** PIELM 행에도 gradient 행에도
            # 4000이 찍혔는데 그건 반복예산이지 시행공간의 크기가 아니다. PIELM은
            # backprop이 없어 반복이 아예 없고, 크기는 무작위 특징 수다.
            "size_label": "random features" if pielm else "MLP width x depth",
            "size": (int(arm.split("nf=")[-1].rstrip(")")) if pielm
                     else f"{CORE_WIDTH}x{CORE_DEPTH}"),
            "iterations": None if pielm else int(r.get("iters", 4000)),
            "e_lam": e, "p_correct": float(r["p_correct"]),
            "ensemble_seconds": float(secs),
            "seconds_per_attempt": per,
            "seconds_per_attempt_min": lo_hi[0],
            "seconds_per_attempt_max": lo_hi[1],
            "accounting": ("single deterministic solve (numpy/CPU, no backprop)"
                           if pielm else
                           "measured directly at n_seeds = 1 (GB10 GPU, shared)"),
            "E_T_success": expected_time_to_success(per, float(r["p_correct"])),
            "factorization_ok": True})
    return rows


def _pareto(rows, mode: int) -> list:
    """(정확도, 1회비용) 평면의 Pareto 전선 — 더 정확하고 더 싼 것이 없는 점들."""
    pts = [r for r in rows if r["mode"] == mode
           and math.isfinite(r["e_lam"]) and r["p_correct"] > 0
           and math.isfinite(r["seconds_per_attempt"])]
    out = []
    for r in pts:
        dominated = any(
            (s["e_lam"] <= r["e_lam"] and s["seconds_per_attempt"] <= r["seconds_per_attempt"]
             and (s["e_lam"] < r["e_lam"]
                  or s["seconds_per_attempt"] < r["seconds_per_attempt"]))
            for s in pts)
        if not dominated:
            out.append({**r, "pareto_mode": mode})
    return sorted(out, key=lambda r: r["seconds_per_attempt"])


def _cheapest_to_reach(rows, mode: int) -> list:
    """정확도 수준 ε마다 **가장 싼 성공 경로**를 계열별로 낸다 — t(ε)의 통합판.

    논문 §5.4가 요구하는 "언제 신경 쪽이 값어치를 하는가"에 직접 답하는 표다."""
    out = []
    for eps in EPS_LEVELS:
        for fam in FAMILIES:
            cand = [r for r in rows
                    if r["mode"] == mode and r["family"] == fam
                    and math.isfinite(r["e_lam"]) and r["e_lam"] <= eps
                    and r["p_correct"] > 0 and math.isfinite(r["E_T_success"])]
            best = min(cand, key=lambda r: r["E_T_success"]) if cand else None
            out.append({
                "mode": mode, "eps": eps, "family": fam,
                "solver": best["solver"] if best else "",
                "size_label": best["size_label"] if best else "",
                "size": best["size"] if best else "",
                "iterations": (best.get("iterations") if best else None),
                "e_lam": best["e_lam"] if best else float("nan"),
                "p_correct": best["p_correct"] if best else float("nan"),
                "seconds_per_attempt": (best["seconds_per_attempt"] if best
                                        else float("nan")),
                "E_T_success": best["E_T_success"] if best else float("inf"),
                "reached": bool(best)})
    return out


def main(outdir=None, modes=(1, 2, 3)) -> dict:
    d = manifest.ensure_outdir(outdir)
    rows = _classical_rows(d, modes) + _neural_rows(d, modes)
    pareto = [p for m in modes for p in _pareto(rows, m)]
    cheapest = [c for m in modes for c in _cheapest_to_reach(rows, m)]

    manifest.write_csv(os.path.join(d, "p1_classical_vs_neural.csv"), rows)
    manifest.write_csv(os.path.join(d, "p1_pareto.csv"), pareto)
    manifest.write_csv(os.path.join(d, "p1_cheapest_to_epsilon.csv"), cheapest)
    manifest.write_json(
        os.path.join(d, "manifest_p1_compare.json"),
        manifest.build({
            "driver": "run_p1_compare.main",
            "per_attempt_source": "p1_single_seed_cost.csv (measured at n_seeds = 1); "
                                  "no batching conversion is applied",
            "eps_levels": list(EPS_LEVELS), "modes": list(modes),
            "families": list(FAMILIES),
            "accounting_note":
                "classical = one numpy/CPU solve; gradient arms = measured at "
                "n_seeds = 1 on a shared GB10 GPU; Eig-PIELM = one numpy/CPU solve. "
                "The CPU/GPU difference is not corrected for and the shared-GPU "
                "wall-clock spread is reported per row.",
            "inputs": ["p1_basis_study.csv", "p1_neural_mode_sweep.csv"]}))
    print(f"[run_p1_compare] {d} — 대조 {len(rows)}행, Pareto {len(pareto)}행, "
          f"t(eps) {len(cheapest)}행")
    return {"compare": rows, "pareto": pareto, "cheapest": cheapest, "outdir": d}


def p4_compare(outdir=None, modes=(1, 2, 3)) -> dict:
    """P4에서 고전 등급 Q2와 2차원 신경 arm을 같은 축에 올린다.

    P1에서 한 것과 같은 회계다 — 고전은 numpy/CPU 1회 조립+희소 해석, 신경은 50시드 GPU
    앙상블을 실측 배칭계수로 나눈 1회 시도 비용. 하드웨어 차이는 보정하지 않는다.

    P4는 참조해가 Richardson 외삽이므로 **오차의 하한이 참조 불확실도**다. 그보다 작은
    e_λ는 솔버의 정확도가 아니라 참조의 잡음이므로 그 사실을 열로 남긴다.

    신경 쪽 1회 비용은 `p1_single_seed_cost.csv`의 P4 행(n_seeds=1 직접 측정)을 쓴다. 그
    측정이 없으면 앙상블 벽시계로 대체하되 `per_attempt_is_upper_bound`를 참으로 표시한다 —
    그 경우 격차가 과대계상되므로 자릿수는 상한이다.
    """
    d = manifest.ensure_outdir(outdir)
    unc = {int(r["mode"]): float(r["uncertainty_rel"])
           for r in manifest.read_csv(os.path.join(d, "p4_reference.csv"))}
    ext = {int(r["mode"]): float(r["lambda_extrapolated"])
           for r in manifest.read_csv(os.path.join(d, "p4_reference.csv"))}
    rows = []
    for r in manifest.read_csv(os.path.join(d, "p4_convergence.csv")):
        for m in modes:
            lam = r.get(f"lam_mode{m}")
            if lam in (None, ""):
                continue
            e = abs(float(lam) - ext[m]) / ext[m]
            hit = e <= mt.ELAM_MAX
            rows.append({
                "problem": "P4", "family": "classical", "mode": m,
                "solver": f"graded Q2 (beta={r['beta']})",
                "size_label": "n_dof", "size": r["n_dof"],
                "iterations": None,       # 직접 풀이 — 반복이 없다
                "e_lam": e, "p_correct": float(hit),
                "seconds_reported": float(r["seconds"]),
                "seconds_per_attempt": float(r["seconds"]),
                "accounting": "single deterministic solve (numpy/CPU)",
                "E_T_success": expected_time_to_success(float(r["seconds"]),
                                                        float(hit)),
                "at_reference_floor": bool(e <= unc.get(m, 0.0))})
    direct = _single_seed_cost(d)
    skipped = set()
    for r in _read_many(d, P4_NEURAL_SWEEPS):
        m = int(r["mode"])
        if m not in modes:
            continue
        hit = direct.get((_base(r["arm"]), m))
        # **1회 비용 실측이 없는 arm은 비용표에 넣지 않는다.** 앙상블 벽시계를 1회 비용으로
        # 쓰면 격차가 과대계상되고, 그 상한은 이 논문이 직접측정으로 교체한 바로 그 값이다.
        # 신뢰도 비교(표 16)에는 그대로 남으므로 결과가 사라지지는 않는다.
        if hit is None:
            skipped.add(str(r["arm"]))
            continue
        per = hit[0]
        upper = False
        rows.append({
            "problem": "P4", "family": "neural_gradient", "mode": m,
            "solver": str(r["arm"]), "size_label": "iters", "size": r["iters"],
            "e_lam": float(r["e_lam_median"]), "p_correct": float(r["p_correct"]),
            "seconds_reported": float(r["seconds"]),
            "seconds_per_attempt": per,
            "accounting": ("measured directly at n_seeds = 1 (GB10 GPU, shared)"
                           if not upper else
                           "50-seed vmap ensemble wall-clock used as a per-attempt cost "
                           "(GB10 GPU, shared) — an upper bound"),
            "per_attempt_is_upper_bound": upper,
            "seconds_per_attempt_min": (hit[1] if hit else ""),
            "seconds_per_attempt_max": (hit[2] if hit else ""),
            "ensemble_seconds": float(r["seconds"]),
            "E_T_success": expected_time_to_success(per, float(r["p_correct"])),
            "at_reference_floor": bool(float(r["e_lam_median"]) <= unc.get(m, 0.0))})
    manifest.write_csv(os.path.join(d, "p4_classical_vs_neural.csv"), rows)
    if skipped:
        print(f"[p4_compare] 1회 비용 미측정으로 비용표에서 제외: {sorted(skipped)}")
    print(f"[p4_compare] {len(rows)}행")
    return {"p4_compare": rows}


def headline(outdir=None, eps: float = 1e-4, mode: int = 1) -> dict:
    """논문이 인용하는 **정본 수치**를 하나의 표로 낸다 — 손으로 옮기지 않는다.

    같은 비율이 초록·§5.4·Figure 캡션·결론에 손으로 적혀 있다가 서로 어긋났다
    (5.34 / 5.4 / "5.40–5.50" / 5.42가 한 문서에 공존). 파생값도 함께 어긋났다
    (narrowing 1.6 대 1.7, 정확도 격차 two/three orders). 원인은 하나다 — 같은 숫자를
    다섯 곳에 손으로 썼다. 그러니 여기서 계산해 렌더링하고, 본문은 그 표를 가리킨다.
    """
    d = manifest.ensure_outdir(outdir)
    ch = manifest.read_csv(os.path.join(d, "p1_cheapest_to_epsilon.csv"))
    best = {}
    for r in ch:
        if int(r["mode"]) != mode or abs(float(r["eps"]) - eps) > 1e-12:
            continue
        if not bool(r["reached"]):
            continue
        best[str(r["family"])] = r
    rows = []

    def add(a, b, label):
        if a not in best or b not in best:
            return
        ta, tb = float(best[a]["seconds_per_attempt"]), float(best[b]["seconds_per_attempt"])
        ea, eb = float(best[a]["e_lam"]), float(best[b]["e_lam"])
        # **기대성공 기준을 함께 낸다.** 논문은 E[T_success]를 주 비용 지표로 선언했으므로
        # 정본표가 1회 기준만 싣고 기대성공을 캡션 산문에 숨기면 앞뒤가 맞지 않는다.
        sa, sb = float(best[a]["E_T_success"]), float(best[b]["E_T_success"])
        rows.append({"bound": "measured", "quantity": label,
                     "numerator": b, "denominator": a,
                     "cost_ratio": tb / ta,
                     "cost_orders": math.log10(tb / ta),
                     "expected_success_ratio": sb / sa,
                     "expected_success_orders": math.log10(sb / sa),
                     "accuracy_ratio": eb / ea,
                     "accuracy_orders": math.log10(eb / ea),
                     "numerator_solver": best[b]["solver"],
                     "denominator_solver": best[a]["solver"],
                     "numerator_seconds": tb, "denominator_seconds": ta,
                     "numerator_e_lam": eb, "denominator_e_lam": ea})

    add("classical", "neural_gradient", "gradient-trained vs classical")
    add("classical", "neural_randfeat", "random-feature vs classical")
    add("neural_randfeat", "neural_gradient", "gradient-trained vs random-feature")

    # P4의 같은 비교를 **모드마다** 낸다. 손으로 쓴 별도 표가 §5.4에 있었는데 그 표는
    # 폐기한 배칭계수(ensemble÷2.53)로 계산돼 같은 비교를 3.7과 4.13 두 값으로 만들었고,
    # 번호·캡션·출처도 없었다. 여기서 계산해 흡수하면 그 종류의 분기가 불가능해진다.
    p4 = manifest.read_csv(os.path.join(d, "p4_classical_vs_neural.csv"))
    o4_by_mode, o4s_by_mode = {}, {}
    for m4 in sorted({int(r["mode"]) for r in p4}):
        p4m = [r for r in p4 if int(r["mode"]) == m4]
        ne = [r for r in p4m if r["family"] == "neural_gradient"]
        if not ne:
            continue
        en, tn = float(ne[0]["e_lam"]), float(ne[0]["seconds_per_attempt"])
        cand = [r for r in p4m if r["family"] == "classical"
                and float(r["e_lam"]) <= en]
        if not cand:
            continue
        bq = min(cand, key=lambda r: float(r["seconds_per_attempt"]))
        o4 = math.log10(tn / float(bq["seconds_per_attempt"]))
        o4_by_mode[m4] = o4
        ub = bool(ne[0].get("per_attempt_is_upper_bound"))
        pc = float(ne[0].get("p_correct") or 0.0)
        pcq = float(bq.get("p_correct") or 1.0)
        es4 = ((tn / pc) / (float(bq["seconds_per_attempt"]) / pcq)
               if pc > 0 and pcq > 0 else float("nan"))
        o4s_by_mode[m4] = math.log10(es4) if es4 == es4 else float("nan")
        rows.append({"bound": "upper" if ub else "measured",
                     "quantity": f"gradient-trained vs classical (P4, mode {m4})",
                     "numerator": "neural_gradient", "denominator": "classical",
                     "cost_ratio": tn / float(bq["seconds_per_attempt"]),
                     "cost_orders": o4,
                     "expected_success_ratio": es4,
                     "expected_success_orders": (math.log10(es4) if es4 == es4
                                                 else float("nan")),
                     "accuracy_ratio": en / float(bq["e_lam"]),
                     "accuracy_orders": math.log10(en / float(bq["e_lam"])),
                     "numerator_solver": ne[0]["solver"],
                     "denominator_solver": f"{bq['solver']}, {bq['size']} dof",
                     "numerator_seconds": tn,
                     "denominator_seconds": float(bq["seconds_per_attempt"]),
                     "numerator_e_lam": en, "denominator_e_lam": float(bq["e_lam"])})
    p1o = next((r["cost_orders"] for r in rows
                if r["quantity"] == "gradient-trained vs classical"), None)
    p1s = next((r["expected_success_orders"] for r in rows
                if r["quantity"] == "gradient-trained vs classical"), None)
    if p1o is not None and 1 in o4_by_mode:
        nb = next((r["bound"] for r in rows
                   if r["quantity"] == "gradient-trained vs classical (P4, mode 1)"),
                  "measured")
        rows.append({"bound": "lower" if nb == "upper" else "measured",
                     # **행 이름도 관찰값이다.** "narrowing"은 무언가가 줄였다고 읽히고,
                     # 이 비교는 corner를 분리하지 못한다(§5.4).
                     "quantity": ("observed P1-to-P4 gap difference (mode 1"
                                  + (", lower bound)" if nb == "upper" else ")")),
                     "numerator": "", "denominator": "",
                     "cost_ratio": float("nan"),
                     "cost_orders": p1o - o4_by_mode[1],
                     "expected_success_ratio": float("nan"),
                     "expected_success_orders": ((p1s - o4s_by_mode[1])
                                                 if p1s is not None
                                                 and 1 in o4s_by_mode
                                                 else float("nan")),
                     "accuracy_ratio": float("nan"),
                     "accuracy_orders": float("nan"),
                     "numerator_solver": "", "denominator_solver": "",
                     "numerator_seconds": float("nan"),
                     "denominator_seconds": float("nan"),
                     "numerator_e_lam": float("nan"),
                     "denominator_e_lam": float("nan")})
    manifest.write_csv(os.path.join(d, "p1_cost_headline.csv"), rows)
    manifest.write_json(os.path.join(d, "manifest_cost_headline.json"),
                        manifest.build({
                            "driver": "run_p1_compare.headline",
                            "eps": eps, "mode": mode,
                            "why": "the same ratios were hand-copied into the abstract, "
                                   "§5.4, a figure caption and the conclusion and drifted "
                                   "apart; they are computed here and rendered"}))
    print(f"[headline] {len(rows)}행")
    return {"headline": rows}

if __name__ == "__main__":
    main()
    p4_compare()
    headline()
