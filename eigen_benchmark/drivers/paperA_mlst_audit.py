"""논문 A MLST 결과 재검토(2026-10-15, 사용자 문서 `paperA_MLST_results_critical_review_2026-10-15.md`) 대응 감사.

학습 없이 커밋된 기록만 읽는다. 판정 규칙을 바꾸지 않고, 이미 판정된 결과를 다른 각도에서 센다(사후 진단).

- `discordance` — 같은 시드의 이진 결과 쌍 2×2표, 정확 McNemar(불일치 쌍의 이항), 불일치 비율의 Clopper–Pearson 95 %
  구간. 모든 쌍이 같으면 부트스트랩 구간이 [1, 1]·[0, 0]으로 퇴화한다 — 불일치가 0이어도 동등성이 아니라 불일치
  비율의 상한만 준다.
- `transitions` — P4 사영 저장 → 원시 저장의 시드별 전이와 p_accurate, 잃은 시드의 모드 식별(수렴 문턱과 분리).
- `span_algebra` — 정규화한 저장 두 방향 v₁·v₂(상관 r)의 v± = (v₁ ± v₂)/√(2(1 ± r)): A_D v± = (1 ± r)v±,
  (I − A_D)v± = ∓r v±, (I − P_G)v± = 0, R[(I − A_D)v±] = R(v±). r ≠ 0이면 대각 규칙은 저장 스팬을 지우지 않고 제자리로
  보낸다 — 스케일 불변 Rayleigh 몫은 그 스팬의 최소 Ritz 값에 그대로 닿는다.
- `adf_all_cells` — 코너 등급 구적의 Λ 변화를 정답 셀만이 아니라 채점된 모든 셀에서, 그리고 e_λ 문턱을 넘나드는 셀 수.
- `rescore` — 8겹 구적의 Λ로 e_λ 문턱만 다시 적용한다(수렴·모드 식별은 그대로). 훈련 구적을 착취해 정답이 된 셀을 센다.
"""
import collections
import csv
import json
import math
import os

import numpy as np
from scipy import linalg, stats

from .. import metrics as mt

ALPHAS = (0.0, 0.01, 0.05, 0.25, 1.0)
P4_RULES = {"diagonal": "diagonal projection", "exact": "exact projection", "qr": "qr projection"}


def discordance(a, b) -> dict:
    """같은 시드의 이진 결과 a·b(참 = 정답)의 2×2표와 정확 McNemar p(양측), 불일치 비율의 Clopper–Pearson 95 % 구간."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    if a.shape != b.shape:
        raise ValueError("paired outcomes must cover the same seeds")
    n11, n10 = int(np.sum(a & b)), int(np.sum(a & ~b))
    n01, n00 = int(np.sum(~a & b)), int(np.sum(~a & ~b))
    m, n = n10 + n01, int(a.size)
    p = 1.0 if m == 0 else min(1.0, 2.0 * float(stats.binom.cdf(min(n10, n01), m, 0.5)))
    lo = 0.0 if m == 0 else float(stats.beta.ppf(0.025, m, n - m + 1))
    hi = 1.0 if m == n else float(stats.beta.ppf(0.975, m + 1, n - m))
    return {"n": n, "n11": n11, "n10": n10, "n01": n01, "n00": n00, "p_mcnemar": p,
            "discordant_ci": (lo, hi)}


def _accurate(r) -> bool:
    e = float(r["e_lam"])
    return (int(r["matched_mode"]) == int(r["mode"]) and float(r["mac"]) >= mt.MAC_MIN
            and math.isfinite(e) and e <= mt.ELAM_MAX)


def transitions(proj_recs, raw_recs) -> dict:
    """한 규칙의 사영 저장(proj)·원시 저장(raw) 기록(같은 모드, 같은 시드)의 시드별 전이."""
    P = {int(r["seed"]): r for r in proj_recs}
    R = {int(r["seed"]): r for r in raw_recs}
    if set(P) != set(R):
        raise ValueError("projected and raw records must cover the same seeds")
    seeds = sorted(P)
    table = collections.Counter((P[s]["outcome"], R[s]["outcome"]) for s in seeds)
    lost = [s for s in seeds if P[s]["outcome"] == "correct" and R[s]["outcome"] != "correct"]
    rescued = sum(1 for s in seeds if P[s]["outcome"] != "correct" and R[s]["outcome"] == "correct")
    lower_id = [s for s in lost if int(R[s]["matched_mode"]) < int(R[s]["mode"])
                and float(R[s]["mac"]) >= mt.MAC_MIN]
    return {"table": dict(table),
            "lost": {"n": len(lost), "seeds": lost,
                     "by_raw_outcome": dict(collections.Counter(R[s]["outcome"] for s in lost)),
                     "mode1_identity": sum(1 for s in lower_id if int(R[s]["matched_mode"]) == 1),
                     "lower_identity": len(lower_id),
                     "matched_lower": sum(1 for s in lost if int(R[s]["matched_mode"]) < int(R[s]["mode"]))},
            "rescued": rescued,
            "p_accurate": {"projected": sum(_accurate(P[s]) for s in seeds),
                           "raw": sum(_accurate(R[s]) for s in seeds)}}


def _ip(w, a, b) -> float:
    return float(np.sum(w * a * b))


def _rq(w, f, fd) -> float:
    return _ip(w, fd, fd) / _ip(w, f, f)


def _min_ritz(w, B, Bd) -> float:
    K = (w * Bd) @ Bd.T
    M = (w * B) @ B.T
    return float(linalg.eigh(K, M, eigvals_only=True)[0])


def span_algebra(a, b, w, alpha: float) -> dict:
    """원시 저장 쌍 a = (φ₁, φ₁″), b = (φ₂, φ₂″)를 질량 내적으로 직교·정규화(RF 쌍 q₁·q₂)한 뒤
    v₁ = q₁, v₂ = (q₂ + αq₁)/√(1 + α²)로 섞어 대각 규칙의 스팬 작용을 잰다. 곡률에도 같은 결합을 쓴다."""
    (p1, d1), (p2, d2) = a, b
    n1 = math.sqrt(_ip(w, p1, p1))
    q1, q1d = p1 / n1, d1 / n1
    c = _ip(w, p2, q1)
    u, ud = p2 - c * q1, d2 - c * q1d
    n2 = math.sqrt(_ip(w, u, u))
    q2, q2d = u / n2, ud / n2
    s = math.sqrt(1.0 + alpha * alpha)
    V = np.stack([q1, (q2 + alpha * q1) / s])
    Vd = np.stack([q1d, (q2d + alpha * q1d) / s])
    r = _ip(w, V[0], V[1])

    def coef_diag(f):
        return np.array([_ip(w, V[j], f) / _ip(w, V[j], V[j]) for j in range(2)])

    def coef_exact(f):
        return np.linalg.solve((w * V) @ V.T, (w * V) @ f)

    def nrm(f):
        return math.sqrt(max(_ip(w, f, f), 0.0))

    vp = (V[0] + V[1]) / math.sqrt(2 * (1 + r))
    vm = (V[0] - V[1]) / math.sqrt(2 * (1 - r))
    vpd = (Vd[0] + Vd[1]) / math.sqrt(2 * (1 + r))
    vmd = (Vd[0] - Vd[1]) / math.sqrt(2 * (1 - r))
    ad_err, resid, gain, rq_diff, defl, defl_d = 0.0, 0.0, 0.0, 0.0, [], []
    for f, fd, sign in ((vp, vpd, 1.0), (vm, vmd, -1.0)):
        cd = coef_diag(f)
        g, gd = f - cd @ V, fd - cd @ Vd                    # (I − A_D)f, 같은 계수의 곡률
        ad_err = max(ad_err, nrm(cd @ V - (1 + sign * r) * f))
        resid = max(resid, nrm(f - coef_exact(f) @ V))
        gain = max(gain, nrm(g))
        defl.append(g)
        defl_d.append(gd)
        if abs(r) > 0:
            rq_diff = max(rq_diff, abs(_rq(w, g, gd) - _rq(w, f, fd)) / _rq(w, f, fd))
    span_min = _min_ritz(w, V, Vd)
    return {"alpha": float(alpha), "r": r, "ad_eig_err": ad_err, "proj_resid": resid,
            "deflated_gain": gain, "rq_rel_diff": rq_diff if abs(r) > 0 else float("nan"),
            "span_rq_min": span_min,
            "deflated_rq_min": (_min_ritz(w, np.stack(defl), np.stack(defl_d)) if abs(r) > 0
                                else float("nan")),
            "rq_vp": _rq(w, vp, vpd), "rq_vm": _rq(w, vm, vmd)}


def adf_all_cells(rows, ref_lam, mid: int = 4, hi: int = 8) -> dict:
    """`p4_adf_neural.csv`의 행(규칙·모드·시드·겹 수·Λ·판정)에서 (규칙, 모드)마다 모든 셀의 Λ 변화와 e_λ 문턱 횡단 수.
    열 이름은 기본 등급(0·4·8겹)을 따른다 — `mid`·`hi`는 다른 등급 집합(소형 시험)에서 같은 양을 잴 때만 바꾼다."""
    lam = collections.defaultdict(dict)
    outcome = {}
    for r in rows:
        key = (r["arm"], int(r["mode"]), int(r["seed"]))
        lam[key][int(r["level"])] = float(r["Lam"])
        outcome[key] = r["outcome"]
    out = {}
    for arm, mode in sorted({(k[0], k[1]) for k in lam}):
        ks = [k for k in lam if k[0] == arm and k[1] == mode]
        rel48 = [abs(lam[k][mid] - lam[k][hi]) / abs(lam[k][hi]) for k in ks]
        rel08 = [abs(lam[k][0] - lam[k][hi]) / abs(lam[k][hi]) for k in ks]
        ref = float(ref_lam[mode])
        cross = sum(1 for k in ks if (abs(lam[k][0] - ref) / ref <= mt.ELAM_MAX)
                    != (abs(lam[k][hi] - ref) / ref <= mt.ELAM_MAX))
        out[(arm, mode)] = {"n": len(ks), "n_correct": sum(outcome[k] == "correct" for k in ks),
                            "max_rel_4_8": max(rel48), "median_rel_0_8": float(np.median(rel08)),
                            "max_rel_0_8": max(rel08), "gate_crossings": cross}
    return out


def rescore(recs, lam_hi, ref_lam: float) -> dict:
    """한 (규칙, 모드)의 기록을 다른 구적에서 잰 Λ(`lam_hi`, 시드 → Λ)로 다시 판정한다. 바뀌는 것은 e_λ 문턱뿐이다 —
    수렴 판정과 모드 식별(MAC·대응 모드)은 훈련 구적의 것을 그대로 쓴다."""
    outcomes, flips, n_old, n_new = {}, collections.Counter(), 0, 0
    for r in recs:
        s, m, old = int(r["seed"]), int(r["mode"]), r["outcome"]
        if old == "non_converged":
            new = old
        else:
            e = abs(float(lam_hi[s]) - ref_lam) / ref_lam
            j, mac = int(r["matched_mode"]), float(r["mac"])
            new = ("correct" if (j == m and mac >= mt.MAC_MIN and e <= mt.ELAM_MAX)
                   else "lower_mode_basin" if (j < m and mac >= mt.MAC_MIN) else "spurious")
        outcomes[s] = new
        n_old += old == "correct"
        n_new += new == "correct"
        if new != old:
            flips[(old, new)] += 1
    return {"outcomes": outcomes, "flips": dict(flips), "n_correct": (n_old, n_new)}


EPS_LEVELS = (1e-2, 1e-3, 1e-4, 1e-6)
TARGET_FAMILIES = ("classical", "neural_randfeat", "neural_gradient")


def target_cost_grid(rows, recs, eps_levels=EPS_LEVELS, modes=(1, 2, 3)) -> list:
    """목표 ε마다 계열별 최소 기대 시간 경로 — 성공 확률을 **목표 자체**로 정한다(5 % 벤치마크 문턱이 아니라).

    rows: `p1_classical_vs_neural.csv`의 행(계열·풀이·모드·e_λ·1회 시도 비용). recs: {(풀이, 모드): 시드별 기록}.
    고전 풀이는 결정론적이라 e_λ ≤ ε이면 p = 1, 아니면 0이다. 신경 풀이는 p = (정답이면서 e_λ ≤ ε인 시드)/(시드 수),
    기대 시간 = 1회 비용 / p. 계열마다 기대 시간이 가장 작은 경로를 고른다."""
    out = []
    for m in modes:
        for eps in eps_levels:
            for fam in TARGET_FAMILIES:
                cands = []
                for r in rows:
                    if r["family"] != fam or int(r["mode"]) != m:
                        continue
                    per = float(r["seconds_per_attempt"])
                    if not math.isfinite(per):
                        continue
                    if fam == "classical":
                        e = float(r["e_lam"])
                        n_ok, n = (1, 1) if (math.isfinite(e) and e <= eps) else (0, 1)
                        e_ok = [e] if n_ok else []
                    else:
                        rs = recs.get((r["solver"], m), [])
                        n = len(rs)
                        e_ok = [float(x["e_lam"]) for x in rs if x["outcome"] == "correct" and float(x["e_lam"]) <= eps]
                        n_ok = len(e_ok)
                    if n and n_ok:
                        cands.append((per * n / n_ok, r["solver"], per, n_ok, n, float(np.median(e_ok))))
                if cands:
                    et, solver, per, n_ok, n, e_med = min(cands)
                    out.append({"mode": m, "eps": eps, "family": fam, "solver": solver, "seconds_per_attempt": per,
                                "n_target": n_ok, "n_seeds": n, "p_target": n_ok / n, "E_T_target": et,
                                "e_lam_median_target": e_med, "reached": True})
                else:
                    out.append({"mode": m, "eps": eps, "family": fam, "solver": "", "seconds_per_attempt": float("nan"),
                                "n_target": 0, "n_seeds": 0, "p_target": 0.0, "E_T_target": float("inf"),
                                "e_lam_median_target": float("nan"), "reached": False})
    return out


def _target_records(data_dir):
    """비용표 후보의 시드별 기록 — 기본 격자(가우스, 4000회, I0·커리큘럼 I4·무작위 특징)만."""
    recs = collections.defaultdict(list)
    for name in ("p1_neural_records.jsonl", "p1_neural_qr_records.jsonl", "p1_neural_pielm_records.jsonl"):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        for r in _jsonl(path):
            if r.get("nodes") != "gauss" or r.get("ladder") not in ("I0", "I4", "n/a"):
                continue
            if "iters" in r and r["iters"] not in (None, "") and int(r["iters"]) != 4000 and "pielm" not in r["arm"]:
                continue
            key = (r["arm"], int(r["mode"]))
            if all(int(x["seed"]) != int(r["seed"]) for x in recs[key]):   # 같은 셀이 두 표에 실린 중복은 한 번만
                recs[key].append(r)
    return recs


def _jsonl(path):
    return [json.loads(x) for x in open(path, encoding="utf-8")]


def _correct(recs, arm, mode):
    rows = sorted((r for r in recs if r["arm"] == arm and int(r["mode"]) == mode), key=lambda r: int(r["seed"]))
    return [r["outcome"] == "correct" for r in rows]


def _p4(recs, rule, mode=3):
    return [r for r in recs if P4_RULES[rule] in r["arm"] and int(r["mode"]) == mode]


def audit(data_dir: str, outdir: str | None = None) -> dict:
    """커밋된 MLST 기록에서 재검토 대응 수치를 만든다 — `paperA_mlst_audit.json`과 세 CSV."""
    outdir = outdir or data_dir
    j = lambda name: _jsonl(os.path.join(data_dir, name))     # noqa: E731
    rf, basis = j("p1_rf_records.jsonl"), j("p1_basis_invariance_records.jsonl")
    rf32, fp32 = j("p1_rf_fp32_records.jsonl"), j("p1_fp32_records.jsonl")
    raw4 = j("p4_raw_records.jsonl")
    proj4 = j("p4_neural_records.jsonl") + j("p4_neural_arms_records.jsonl")

    disc = {
        "p1_rf_mode3_rf_vs_a_prime": discordance(_correct(rf, "rf_frozen", 3), _correct(rf, "a_prime_frozen", 3)),
        "p1_rf_mode3_rf_vs_b": discordance(_correct(rf, "rf_frozen", 3), _correct(rf, "b_frozen", 3)),
        "p1_basis_mode3_alpha0.01_vs_alpha1": discordance(_correct(basis, "basis_alpha=0.01", 3),
                                                          _correct(basis, "basis_alpha=1", 3)),
        "p1_basis_mode3_rf_vs_alpha0.01": discordance(_correct(rf, "rf_frozen", 3),
                                                      _correct(basis, "basis_alpha=0.01", 3)),
        "p1_fp32_mode6_rf_vs_b": discordance(_correct(rf32, "rf_fp32", 6), _correct(fp32, "b_projection_exact", 6)),
        "p1_fp32_mode6_rf_vs_b_prime": discordance(_correct(rf32, "rf_fp32", 6),
                                                   _correct(fp32, "b_prime_projection_qr", 6)),
    }
    trans = {}
    for rule in P4_RULES:
        pr, rw = _p4(proj4, rule), _p4(raw4, rule)
        P = sorted(pr, key=lambda r: int(r["seed"]))
        R = sorted(rw, key=lambda r: int(r["seed"]))
        disc[f"p4_mode3_{rule}_projected_vs_raw"] = discordance([r["outcome"] == "correct" for r in P],
                                                                [r["outcome"] == "correct" for r in R])
        trans[rule] = transitions(pr, rw)

    z = np.load(os.path.join(data_dir, "p1_mlst_prefix.npz"))
    alg_rows = []
    for s in range(z["raw_phi1"].shape[0]):
        for a in ALPHAS:
            row = span_algebra((z["raw_phi1"][s], z["raw_d2_1"][s]), (z["raw_phi2"][s], z["raw_d2_2"][s]),
                               z["wq"], a)
            alg_rows.append({"seed": s, **row})
    pos = [r for r in alg_rows if r["alpha"] > 0]
    alg = {"n_seeds": int(z["raw_phi1"].shape[0]), "alphas": list(ALPHAS),
           "max_ad_eig_err": max(r["ad_eig_err"] for r in alg_rows),
           "max_proj_resid": max(r["proj_resid"] for r in alg_rows),
           "max_gain_minus_r": max(abs(r["deflated_gain"] - abs(r["r"])) for r in alg_rows),
           "max_rq_rel_diff": max(r["rq_rel_diff"] for r in pos),
           "max_deflated_min_rel_diff": max(abs(r["deflated_rq_min"] - r["span_rq_min"]) / r["span_rq_min"]
                                            for r in pos),
           "span_rq_min_median": float(np.median([r["span_rq_min"] for r in alg_rows if r["alpha"] == 0]))}

    ref = {int(r["mode"]): float(r["lambda_extrapolated"])
           for r in csv.DictReader(open(os.path.join(data_dir, "p4_reference.csv"), encoding="utf-8"))}
    adf_rows = list(csv.DictReader(open(os.path.join(data_dir, "p4_adf_neural.csv"), encoding="utf-8")))
    adf = adf_all_cells(adf_rows, ref)
    adf_ref = list(csv.DictReader(open(os.path.join(data_dir, "p4_adf_reference.csv"), encoding="utf-8")))
    lam8 = collections.defaultdict(dict)
    rel08 = {}
    for r in adf_rows:
        k = (r["arm"], int(r["mode"]), int(r["seed"]))
        if int(r["level"]) == 8:
            lam8[(k[0], k[1])][k[2]] = float(r["Lam"])
        if int(r["level"]) in (0, 8):
            rel08.setdefault(k, {})[int(r["level"])] = float(r["Lam"])
    rescored, kept = {}, []
    for rule in P4_RULES:
        for mode in (1, 2, 3):
            rs = _p4(raw4, rule, mode)
            out = rescore(rs, lam8[(rule, mode)], ref[mode])
            flipped = sorted(s for s, o in out["outcomes"].items()
                             if o != next(r["outcome"] for r in rs if int(r["seed"]) == s))
            rescored[f"{rule}_mode{mode}"] = {"n_correct_training_rule": out["n_correct"][0],
                                             "n_correct_eight_layers": out["n_correct"][1],
                                             "flips": out["flips"], "flipped_seeds": flipped}
            kept += [abs(rel08[(rule, mode, int(r["seed"]))][0] - rel08[(rule, mode, int(r["seed"]))][8])
                     / abs(rel08[(rule, mode, int(r["seed"]))][8])
                     for r in rs if r["outcome"] == "correct" and int(r["seed"]) not in flipped]

    cost_rows = list(csv.DictReader(open(os.path.join(data_dir, "p1_classical_vs_neural.csv"), encoding="utf-8")))
    target_grid = target_cost_grid(cost_rows, _target_records(data_dir))
    res = {"discordance": disc, "p4_transitions": trans, "p1_span_algebra": alg,
           "p1_cheapest_to_target": target_grid,
           "p4_adf_all_cells": {f"{a}_mode{m}": v for (a, m), v in adf.items()},
           "p4_adf_all_cells_overall": {
               "n": sum(v["n"] for v in adf.values()),
               "max_rel_4_8": max(v["max_rel_4_8"] for v in adf.values()),
               "max_rel_0_8": max(v["max_rel_0_8"] for v in adf.values()),
               "gate_crossings": sum(v["gate_crossings"] for v in adf.values())},
           "p4_adf_rescore_eight_layers": rescored,
           "p4_adf_correct_kept_max_rel_0_8": max(kept),
           "p4_adf_reference_levels": [{k: (float(v) if k != "mode" and k != "level" and k != "n_pts" else int(v))
                                        for k, v in r.items()} for r in adf_ref]}
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "paperA_mlst_audit.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(res), f, indent=1, ensure_ascii=False)
    with open(os.path.join(outdir, "p1_cheapest_to_epsilon_target.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(target_grid[0]))
        wr.writeheader()
        wr.writerows(target_grid)
    with open(os.path.join(outdir, "p4_raw_transitions.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["rule", "projected_outcome", "raw_outcome", "n_seeds"])
        for rule, t in trans.items():
            for (po, ro), n in sorted(t["table"].items()):
                wr.writerow([rule, po, ro, n])
    with open(os.path.join(outdir, "p1_span_algebra.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(alg_rows[0]))
        wr.writeheader()
        wr.writerows(alg_rows)
    with open(os.path.join(outdir, "p4_adf_all_cells.csv"), "w", newline="", encoding="utf-8") as f:
        keys = ["n", "n_correct", "max_rel_4_8", "median_rel_0_8", "max_rel_0_8", "gate_crossings"]
        wr = csv.writer(f)
        wr.writerow(["rule", "mode"] + keys)
        for (a, m), v in adf.items():
            wr.writerow([a, m] + [v[k] for k in keys])
    return res


def _jsonable(x):
    if isinstance(x, dict):
        return {(" -> ".join(k) if isinstance(k, tuple) else str(k)): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join("docs", "_generated", "data", "paper2"))
    args = ap.parse_args(argv)
    res = audit(args.data)
    print(json.dumps(_jsonable({k: res[k] for k in ("p1_span_algebra", "p4_adf_all_cells_overall")}), indent=1))


if __name__ == "__main__":
    main()
