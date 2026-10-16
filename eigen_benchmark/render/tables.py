"""논문 §5의 표를 CSV에서 조립한다. **여기서 새로 계산하는 숫자는 없다.**

각 함수는 마크다운 표 문자열 하나를 낸다. 출처 CSV와 git sha를 캡션에 남긴다 — 표만 보고도
어느 실행에서 나왔는지 추적할 수 있어야 한다(§3.7).

집계가 필요하면 **드라이버에서** 한다. 렌더러가 집계하면 논문의 숫자가 어느 코드에서
나왔는지 추적선이 끊긴다.
"""
from __future__ import annotations

import json
import math
import os

from ..drivers import manifest
from . import md

DATA_DIR = os.path.join("docs", "_generated", "data", "paper2")

ARM_LABEL = {
    "a_penalty": "(a) penalty deflation",
    "a_prime_projection_diagonal": "(a′) diagonal-approx. projection (pilot)",
    "b_projection_exact": "(b) exact M-orthogonal projection",
    "b_prime_projection_qr": "(b′) QR-orthonormalized projection",
    "c_curriculum": "(c) coarse-Ritz curriculum",
    "d_neural_basis_galerkin": "(d) separate-net neural basis + Galerkin",
    "e_simultaneous_subspace": "(e) shared-trunk subspace trace",
    "f_eig_pielm": "(f) Eig-PIELM (random features, no backprop)",
    "g_spacetime_pinn": "(g) space–time PINN (time domain)",
}
ARM_ORDER = list(ARM_LABEL)

# P1·P2·P3의 참조해는 초월방정식의 근을 mpmath dps=50으로 찾은 것이라 불확도가 근찾기
# 허용오차이고, fp64로 보고하는 순간 **표현 자체가 하한**이 된다. 공란으로 두면 "재지
# 않았다"로 읽히므로 그 값을 명시한다.
HIGHPREC_UNC = "< 1e-15 (fp64 representation)"

FAMILY_LABEL = {"classical": "classical basis + `eigh`",
                "neural_randfeat": "neural, random features (no training)",
                "neural_gradient": "neural, gradient-trained"}


def sha(data_dir: str = DATA_DIR, name: str = "manifest_p1_neural.json") -> str:
    with open(os.path.join(data_dir, name), encoding="utf-8") as f:
        return json.load(f).get("git_sha", "unknown")


def _base(arm) -> str:
    return str(arm).split("(")[0]


def _rows(data_dir: str, name: str) -> list:
    return manifest.read_csv(os.path.join(data_dir, f"{name}.csv"))


def _rows_many(data_dir: str, *names) -> list:
    """있는 파일만 읽어 합친다 — 사후 추가된 arm은 자기 파일에 있고, 그 파일이 없어도
    기존 표가 렌더링돼야 한다."""
    out = []
    for n in names:
        path = os.path.join(data_dir, f"{n}.csv")
        if os.path.exists(path):
            out += manifest.read_csv(path)
    return out


def _frac(r, key: str) -> str:
    return f"{r[key]}/{r['n']}"


def _ci(r) -> str:
    return f"[{md.fmt(r['wilson_lo'])}, {md.fmt(r['wilson_hi'])}]"


# ---------------------------------------------------------------- 표 1


def table1_attribution(data_dir: str = DATA_DIR, n_table: int = 1) -> str:
    """arm × 목표모드의 4분류 카운트 — 논문의 귀속 표.

    v3 초안의 껍데기는 `arm × 초기화 × mode 3`이었으나, 초기화 사다리는 (a′)에서 I5
    (해석해 oracle)까지 전부 0.00이라 변별력이 없었다. 변별이 실제로 일어나는 축은
    **모드 번호**이므로 격자를 arm × mode로 바꿨다. 초기화는 표 4에서 예산 축과 함께 다룬다.
    """
    rows = _rows_many(data_dir, "p1_neural_mode_sweep", "p1_neural_qr")
    by = {}
    for r in rows:
        by.setdefault((_base(r["arm"]), int(r["mode"])), r)
    out = []
    for a in ARM_ORDER:
        for m in sorted({int(r["mode"]) for r in rows}):
            r = by.get((a, m))
            if r is None:
                continue
            out.append([ARM_LABEL[a], m, _frac(r, "correct"),
                        _frac(r, "lower_mode_basin"), _frac(r, "spurious"),
                        _frac(r, "non_converged"), r["p_correct"], _ci(r),
                        r.get("p_accurate")])
    body = md.table(["arm", "mode", "correct", "lower-mode basin", "spurious",
                     "non-converged", "p_correct", "95 % Wilson", "p_accurate"],
                    out, align="llrrrrrcr")
    cap = md.caption(
        n_table,
        "Outcome attribution on P1, 50 seeds per cell, I0 initialization, fixed "
        "Gauss–Legendre quadrature, 4000 iterations. Categories and thresholds "
        "(MAC ≥ 0.9, |Δλ|/λ ≤ 0.05) were pre-specified before any run and were not "
        "adjusted after seeing data. `p_accurate` — a post-hoc diagnostic, not part of "
        "the pre-specification — counts seeds whose recovered pair meets the same "
        "accuracy thresholds while skipping the convergence gate.",
        "p1_neural_mode_sweep.csv", sha(data_dir))
    return body + "\n" + cap + "\n"


# ---------------------------------------------------------------- 표 2–5


def table2_penalty(data_dir: str = DATA_DIR, n_table: int = 2) -> str:
    rows = _rows(data_dir, "p1_neural_penalty_sweep")
    # **non-converged 열을 반드시 싣는다.** w = 1e5·모드 2는 correct 7, lower 0,
    # spurious 0이므로 나머지 43/50이 어디로 갔는지 표에서 사라진다 — 그 43이
    # 이 결과의 핵심(인증 실패)이므로 숨은 잔여가 있어서는 안 된다.
    out = [[r["penalty_weight"], r["mode"], _frac(r, "correct"),
            _frac(r, "lower_mode_basin"), _frac(r, "spurious"),
            _frac(r, "non_converged"),
            r["p_correct"], _ci(r), r.get("p_accurate")] for r in rows]
    body = md.table(["penalty weight w", "mode", "correct", "lower-mode basin",
                     "spurious", "non-converged", "p_correct", "95 % Wilson",
                     "p_accurate"],
                    out, align="rrrrrrrcr")
    cap = md.caption(
        n_table,
        "Penalty deflation on P1, 50 seeds per cell, six decades of weight. A one-line "
        "argument predicts the thresholds. If the stored modes were exact and "
        "mass-normalized, a trial function sitting on stored mode j would see an objective of "
        "λ_j + w, against λ_m for the target, so excluding mode 1 requires w > λ_m − λ₁. "
        "That bound is 473 for mode 2 and 3794 for mode 3, and the sweep turns those modes on "
        "at w = 1e3 and w = 1e4 — in each case the first decade above the bound. "
        "**Penalty deflation therefore does recover mode 3, given a large enough weight** — "
        "an earlier draft of this benchmark stopped at w = 1e3 and concluded it never did. "
        "What the mechanism costs is not capability but the need to know that weight: it is "
        "mode-dependent, it is set by a gap that is not available before the modes are "
        "computed, and the window closes from above — at w = 1e5 mode 2 falls to 0.14, with "
        "43 of 50 seeds failing the pre-specified convergence criterion while `p_accurate` "
        "stays at 1.00, so the recovered mode is still right and only the convergence gate "
        "fails. The contrast is arm (a′), which holds mode 2 "
        "at 1.00 with no weight to choose at all.",
        "p1_neural_penalty_sweep.csv", sha(data_dir))
    return body + "\n" + cap + "\n"


def table3_quadrature(data_dir: str = DATA_DIR, n_table: int = 3) -> str:
    rows = _rows(data_dir, "p1_neural_quadrature")
    audit = _rows(data_dir, "p1_neural_audit")
    noise = [r for r in audit if r.get("study") == "snapshot_noise_floor"]
    # (a′)의 MC 자체 대조군은 감사 후속에서 나왔다 — 같은 표에 실어야 증거가 보인다.
    rows = rows + [r for r in audit if r.get("study") == "a_prime_mc_control"]
    out = [[ARM_LABEL.get(_base(r["arm"]), r["arm"]), r["nodes"], r["mode"],
            r["p_correct"], r.get("p_accurate"), _frac(r, "non_converged"), _ci(r)]
           for r in rows]
    body = md.table(["arm", "quadrature", "mode", "p_correct", "p_accurate",
                     "non-converged", "95 % Wilson"], out, align="llrrrrc")
    extra = ""
    if noise:
        nrows = [[r["nodes"], r["rule_rel_change_median"], r["rule_rel_change_p95"],
                  r["prereg_tol"], r["ratio_median_to_tol"]] for r in noise]
        extra = ("\n" + md.table(
            ["quadrature", "tail relative change (median)", "(p95)",
             "pre-specified tolerance", "ratio"], nrows, align="lrrrr"))
    cap = md.caption(
        n_table,
        "Monte-Carlo collocation yields p_accurate = 1.00 with p_correct = 0.00: the "
        "recovered pairs meet the accuracy thresholds but the pre-specified "
        "convergence rule cannot certify them. The lower panel measures why — over the "
        "same tail window the rule uses, the stochastic objective's relative change is "
        "8.6e-2, about 86× the 1e-3 tolerance, while deterministic Gauss sits at 2.3e-6 "
        "(0.002×). The rule presumes a deterministic objective; under it the MC arm is "
        "inconclusive rather than wrong (§6). The (a′) rows are the same-arm control for "
        "the quadrature question: only the quadrature changed, and `p_accurate` separates "
        "the two effects cleanly — modes 1 and 2 stay accurate and are merely "
        "uncertifiable, while mode 3 is genuinely wrong. The mode-3 failure therefore "
        "survives a change of quadrature and cannot be attributed to it.",
        ["p1_neural_quadrature.csv", "p1_neural_audit.csv"], sha(data_dir))
    return body + extra + "\n" + cap + "\n"


def table4_warm_cold(data_dir: str = DATA_DIR, n_table: int = 4) -> str:
    """초기화 축 — 예산과 함께 보아야 의미가 있고, 사전적합 비용을 청구해야 순위가 맞다."""
    rows = [r for r in _rows(data_dir, "p1_warm_cold_summary")
            if _base(r["arm"]) == "b_projection_exact"]
    out = [[r["budget_iters"], r["ladder"], r["start"], r["p_correct"],
            r["train_seconds_per_attempt"], r["init_seconds_per_attempt"],
            r["E_T_success"], r["speedup_vs_cold"],
            r["E_T_success_if_init_free"]] for r in rows]
    body = md.table(["budget (iters)", "level", "start", "p_correct",
                     "train [s/attempt]", "pre-fit [s/attempt]",
                     "E[T_success] [s]", "vs cold",
                     "E[T] if pre-fit were free [s]"],
                    out, align="rllrrrrrr")
    cap = md.caption(
        n_table,
        "Initialization on arm (b), mode 3, 50 seeds. At the 4000-iteration budget every "
        "level reaches 1.00, so that grid has no headroom to detect an initialization "
        "effect; the axis only separates once the budget is tight enough for cold starts "
        "to fail. Two readings follow. First, what helps is not knowledge of the answer: "
        "a BC-satisfying random polynomial (I1) lifts 0.58 → 1.00 while knowing the "
        "eigenvalue window (I2) changes nothing. Second, charging the pre-fit reverses "
        "the ranking — the analytic oracle (I5) reaches 1.00 but costs about a third more "
        "per success than a cold start; the last column shows what the table would say if "
        "the pre-fit were not charged. All times are per attempt, on the same basis as the cost "
        "table: the "
        "single-seed cost measured directly at the 4000-iteration budget and carried to "
        "the other budgets as a ratio, not converted by a batching factor. The pre-fit "
        "uses max(iters//2, 200) iterations, so it is tied to the main budget by design "
        "and is not a fixed one-off cost.",
        "p1_warm_cold_summary.csv", sha(data_dir, "manifest_p1_warm_cold_summary.json"))
    return body + "\n" + cap + "\n"


def _budget14_control(data_dir: str):
    """모드 14의 5배 예산 대조를 예산 표의 한 행으로. **다른 실행이지만 같은 실험이다** —
    모드 9·10 옆에 놓여야 깊이에 따라 예산이 무엇을 사는지가 한 축에서 읽힌다.

    기준선은 확장 실행의 같은 셀이고, 그 두 실행은 모드 1–13에서 판정까지 일치하므로
    모드 14의 유일한 변수는 예산이다(테스트가 그 일치를 검사한다)."""
    try:
        ctl = {int(float(r["mode"])): r
               for r in _rows(data_dir, "p1_fp64_budget14_stress")}
        ext = {int(float(r["mode"])): r
               for r in _rows(data_dir, "p1_fp64_deep_b_stress")}
    except FileNotFoundError:
        return None
    top = max(ctl)
    c, e = ctl.get(top), ext.get(top)
    if not c or not e or int(float(c["iters"])) == int(float(e["iters"])):
        return None
    # **정수로 넘긴다.** md.fmt는 문자열 "14"를 float으로 파싱해 "14.0"으로 찍는다.
    return {"arm": "b_projection_exact", "mode": int(top),
            "p_correct_base": e["p_correct"], "p_correct": c["p_correct"],
            "wilson_lo": c["wilson_lo"], "wilson_hi": c["wilson_hi"],
            "iters": c["iters"], "n": c["n"],
            "budget_resolved": float(c["p_correct"]) > float(e["p_correct"]),
            "non_converged": c["non_converged"], "spurious": c["spurious"],
            "nc_base": e["non_converged"], "spur_base": e["spurious"]}


def _wilson_disjoint(k1, k2, n, z=1.96):
    def iv(k):
        p, den = k / n, 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    (a, b), (c, d) = iv(k1), iv(k2)
    return b < c or d < a


def table5_longrun(data_dir: str = DATA_DIR, n_table: int = 5) -> str:
    rows = list(_rows(data_dir, "p1_neural_longrun"))
    ctl = _budget14_control(data_dir)
    if ctl:
        rows = rows + [ctl]
    out = [[ARM_LABEL.get(_base(r["arm"]), r["arm"]), r["mode"],
            r.get("p_correct_base"), r["p_correct"], _ci(r), r["iters"],
            r.get("budget_resolved")] for r in rows]
    body = md.table(["arm", "mode", "p_correct @ 4000", "p_correct @ 20000",
                     "95 % Wilson", "iters", "budget-resolved"],
                    out, align="lrrrcrl")
    # 이 열이 실제로 무엇인지 — 점추정 비교다. 어느 행이 사전등록 구간 규칙으로도
    # 갈리는지 세어 그대로 적는다. 손으로 적으면 행이 늘 때마다 어긋난다.
    sep, worse = [], []
    for r in rows:
        try:
            n_ = int(float(r["n"]))
            k1 = round(float(r["p_correct_base"]) * n_)
            k2 = round(float(r["p_correct"]) * n_)
        except (KeyError, TypeError, ValueError):
            continue
        if _wilson_disjoint(k1, k2, n_):
            lab = (ARM_LABEL.get(_base(r["arm"]), r["arm"])[:4].strip()
                   + f" mode {r['mode']}")
            (worse if k2 < k1 else sep).append(lab)
    extra = ""
    if ctl:
        extra = (
            f" The last row is the five-fold control at mode {ctl['mode']}, added in "
            "revision to ask whether the decline with depth is bought back by budget. "
            f"Its base cell comes from the mode-16 extension run, whose modes 1-13 "
            "reproduce this control's outcomes exactly, so the budget is the only "
            f"variable there. What moves is the convergence residue, from "
            f"{ctl['nc_base']} of {ctl['n']} non-converged to {ctl['non_converged']}; "
            f"the misidentification residue does not ({ctl['spur_base']} against "
            f"{ctl['spurious']}).")
    cap = md.caption(
        n_table,
        "Cells below p_correct = 1 in the base grid, re-run at five times the iteration "
        "budget. The `budget-resolved` flag is the bare comparison of the two point "
        "estimates and is **not a significance test**: under the pre-specified Wilson "
        "rule only " + (", ".join(sep) if sep else "no cell")
        + (" separate" if len(sep) != 1 else " separates")
        + ", and " + (", ".join(worse) if worse else "no cell")
        + (" separate" if len(worse) != 1 else " separates")
        + " in the other direction — the extra budget makes it **worse**, "
        "which the flag records as unresolved rather than as harm. A cell whose point "
        "estimate does not move at all is limited by something the budget cannot buy."
        + extra,
        (["p1_neural_longrun.csv", "p1_fp64_budget14_stress.csv",
          "p1_fp64_deep_b_stress.csv"] if ctl else "p1_neural_longrun.csv"),
        sha(data_dir, "manifest_p1_neural_longrun.json"))
    return body + "\n" + cap + "\n"


# ---------------------------------------------------------------- 표 6–7


def _construct_solve(data_dir: str) -> dict:
    """(기저, dof) → (구성초, 풀이초). 고전 계열만 있다."""
    try:
        rows = _rows(data_dir, "p1_basis_study")
    except FileNotFoundError:
        return {}
    return {(str(r["basis"]), str(r["n_dof"])):
            (float(r["seconds_construct"]), float(r["seconds_solve"])) for r in rows}


def table6_cost(data_dir: str = DATA_DIR, n_table: int = 6) -> str:
    """정확도 ε를 가장 싸게 달성하는 경로 — 세 계열을 같은 축에서."""
    rows = _rows(data_dir, "p1_cheapest_to_epsilon")
    out = []
    for r in rows:
        reached = bool(r["reached"])
        cs = _construct_solve(data_dir).get(
            (str(r["solver"]), str(r["size"]))) if reached else None
        # size는 계열마다 뜻이 다르므로 단위를 칸에 붙여 쓴다. 반복예산은 별개 열이다
        # — 이전에는 두 값이 한 열에 섞여 PIELM 행에 4000(=반복예산)이 찍혔다.
        size = (f"{md.fmt(r['size'])} {r.get('size_label', '')}".strip()
                if reached else None)
        it = r.get("iterations")
        out.append([r["mode"], r["eps"], FAMILY_LABEL[r["family"]],
                    r["solver"] if reached else "not reached", size,
                    (int(float(it)) if reached and it not in (None, "") else None),
                    r["e_lam"] if reached else None,
                    r["p_correct"] if reached else None,
                    (cs[0] * 1e3) if cs else None,
                    (cs[1] * 1e3) if cs else None,
                    (r["seconds_per_attempt"] * 1e3) if reached else None,
                    r["E_T_success"] if reached else None])
    body = md.table(["mode", "target ε", "family", "cheapest solver", "trial size",
                     "iterations", "e_λ", "p_correct", "construct [ms]", "solve [ms]",
                     "per attempt [ms]", "E[T_success] [s]"],
                    out, align="rrllrrrrrrrr")
    cap = md.caption(
        n_table,
        "Cheapest route to a target accuracy on P1. Families are split three ways "
        "because lumping the random-feature arm with the gradient-trained ones hides "
        "the latter entirely — Eig-PIELM is three orders cheaper and would take the "
        "single 'neural' slot. Trial size and iteration budget are separate columns "
        "because they are separate things: a classical size is a number of degrees of "
        "freedom, Eig-PIELM's is a number of random features, and neither iterates at all, "
        "while the gradient arms have a fixed architecture and a 4000-iteration budget. "
        "Two accounting statements, which apply to different quantities: **accuracy "
        "statistics use the 50-seed ensembles**, and **timing uses a single seed, measured "
        "directly as the median of three runs.** The two are separated because the "
        "ensemble-to-per-attempt conversion factor measured 2.53, 3.80 and 7.82 in three "
        "sessions on this shared GPU and is therefore not usable in a result. Classical "
        "solves and Eig-PIELM are single deterministic numpy/CPU solves. The "
        "CPU/GPU hardware difference is not corrected for and is reported as a threat "
        "in §6. Rows marked 'not reached' are kept rather than dropped.",
        ["p1_cheapest_to_epsilon.csv", "p1_classical_vs_neural.csv"],
        sha(data_dir, "manifest_p1_compare.json"))
    return body + "\n" + cap + "\n"


def _shape_certification(data_dir: str) -> dict:
    """(problem, label) → 모드형 인증 문자열. 인증 실행이 없으면 빈 사전."""
    rows = _rows_many(data_dir, "reference_certification")
    out = {}
    for r in rows:
        if r.get("quantity_certified") != "mode shape":
            continue
        key = (r["problem"], str(r["label"]).strip())
        d = float(r["deficit"])
        against = str(r.get("certified_against", ""))
        if r["problem"] == "P3":
            out[key] = "not used in scoring"
        elif "mesh" in against or "n = " in against:
            out[key] = f"1 − MAC = {md.fmt(d, 2)} vs refined mesh"
        else:
            out[key] = f"1 − MAC ≤ {md.fmt(max(d, 1e-16), 2)} vs high precision"
    return out


def table7_references(data_dir: str = DATA_DIR, n_table: int = 7) -> str:
    cert = _shape_certification(data_dir)

    def c(problem, label):
        return cert.get((problem, label), md.MISSING)

    out = []
    for r in _rows(data_dir, "p1_reference"):
        u = r.get("uncertainty_rel")
        out.append(["P1", f"mode {r['mode']}", "(βL)⁴", r["Lambda"], "",
                    r["reference"], u if u not in (None, "") else HIGHPREC_UNC,
                    c("P1", f"mode {r['mode']}")])
    for r in _rows(data_dir, "p2_reference"):
        out.append(["P2", f"m={r['m']}, n={r['radial_order']}", "(kb)⁴",
                    r["Lambda_k4"], "degenerate" if r["degenerate"] else "",
                    r["reference"], HIGHPREC_UNC,
                    c("P2", f"m={r['m']}, n={r['radial_order']}")])
    for r in _rows(data_dir, "p3_reference"):
        out.append(["P3", f"k̂={md.fmt(r['k_hat'])}, mode {r['mode']}", "(βL)⁴",
                    r["Lambda"], "", r["reference"], HIGHPREC_UNC, c("P3", "all")])
    for r in _rows(data_dir, "p4_reference"):
        out.append(["P4", f"mode {r['mode']}", "Λ_el=ω²ρL²/E",
                    r["lambda_extrapolated"],
                    f"rate {md.fmt(r['convergence_rate'])}",
                    r["reference"], r["uncertainty_rel"],
                    c("P4", f"mode {r['mode']}")])
    body = md.table(["problem", "label", "quantity", "value", "note",
                     "reference method", "relative uncertainty",
                     "mode-shape certification"],
                    out, align="llllllrr")
    cap = md.caption(
        n_table,
        "Reference eigenvalues, all dimensionless. References are analytic (P1 "
        "Euler–Bernoulli, P2 exact Bessel for the annular Kirchhoff plate, P3 "
        "transfer-matrix determinant at mpmath dps = 50) or Richardson-extrapolated Q2 "
        "FEM (P4). The P4 uncertainty is the extrapolation residual and sets the floor "
        "below which an error cannot be attributed to the solver. Degenerate P2 pairs "
        "are flagged: they must be scored by subspace MAC, not by individual MAC (§5.2). "
        "Displayed values are rounded; the stored references carry full precision, and the "
        "stated uncertainties refer to the stored values, not the rounded display. "
        "**The last two columns certify different things, and both are needed.** The "
        "pre-specified rules read two quantities from the reference: the eigenvalue, "
        "through |Δλ|/λ ≤ 0.05, and the mode shape, through MAC ≥ 0.9. Those reach the "
        "scorer by different numerical paths, so certifying one says nothing about the "
        "other. On P1 the eigenvalue is a root of cos β cosh β + 1, which is well "
        "conditioned and accurate to 1.7e-16, while the textbook form of the mode shape "
        "subtracts two terms of size e^β/2 to leave a quantity of order one. That "
        "cancellation cost every significant digit beyond mode 13: the reference for mode "
        "14 scored MAC 0.887 against a high-precision evaluation of the same formula, "
        "below the 0.9 gate, so a solver recovering the exact mode would have been "
        "classified spurious. The shape is now evaluated in an exponential form with no "
        "cancellation and certified against higher precision at every mode the scorer can "
        "use; the certification itself is run at a precision chosen per mode so that it is "
        "converged in its own arithmetic. P4's reference is a discretization rather than a "
        "closed form, so it is certified against a mesh refinement and its tolerance is "
        "set at the discretization level, not the round-off level.",
        ["p1_reference.csv", "p2_reference.csv", "p3_reference.csv",
         "p4_reference.csv", "reference_certification.csv"],
        sha(data_dir, "manifest_p1.json"))
    return body + "\n" + cap + "\n"


# ---------------------------------------------------------------- 표 8–10


def table8_conditioning(data_dir: str = DATA_DIR, n_table: int = 8) -> str:
    rows = _rows(data_dir, "p1_conditioning")
    out = [[r["basis"], r["n_dof"], r["kappa_M_raw"], r["kappa_K_raw"],
            r["kappa_M_massnorm"], r["kappa_K_equilibrated"],
            r.get("kappa_A_transformed"),
            "ok" if r["cholesky_ok"] else "**fail**",
            r.get("backward_error")] for r in rows]
    body = md.table(["basis", "n_dof", "κ(M) raw", "κ(K) raw", "κ(M) mass-norm",
                     "κ(K) equilibrated", "κ transformed operator",
                     "Cholesky", "backward error"], out, align="lrrrrrrlr")
    cap = md.caption(
        n_table,
        "Conditioning of the discrete pencil on P1 under four normalizations. The raw "
        "condition numbers are a property of the coordinates, not of the trial space: "
        "three coordinate systems spanning the same space agree only after the "
        "transformed-operator normalization. Factorization failures are shown, not "
        "hidden — raw monomials lose positive-definiteness from N = 12, the smallest size "
        "tested, and high precision does not recover it because the information is already "
        "lost at assembly.",
        "p1_conditioning.csv", sha(data_dir, "manifest_p1.json"))
    return body + "\n" + cap + "\n"


def table9_effective_rank(data_dir: str = DATA_DIR, n_table: int = 9) -> str:
    """세 계열의 명목 / 수치 rank / 사용 가능 차원 — §5.3의 통합 논지."""
    out = []
    cond = _rows(data_dir, "p1_conditioning")
    for r in cond:
        if r["basis"] == "monomial_raw" and int(r["n_dof"]) in (12, 16, 20, 24):
            out.append(["classical monomial (raw)", r["n_dof"],
                        "n/a", "n/a",
                        "ok" if r["cholesky_ok"] else "**fail**",
                        "deterministic coordinate choice; orthonormalizing the same "
                        "span restores factorization"])
    pie = [r for r in _rows(data_dir, "p1_neural_mode_sweep")
           if _base(r["arm"]) == "f_eig_pielm" and int(r["mode"]) == 1]
    for r in pie:
        nf = str(r["arm"]).split("nf=")[-1].rstrip(")")
        out.append(["random features (Eig-PIELM)", nf,
                    r.get("rank_tol_1e-16"), r.get("rank_tol_1e-10"), "ok",
                    "redundancy from the random draw; independent of training"])
    col = _rows(data_dir, "p1_neural_collinearity")
    for arm in sorted({str(r["arm"]) for r in col}):
        g = [r for r in col if str(r["arm"]) == arm and float(r["w_orth"]) == 0.0]
        if not g:
            continue
        nb = g[0]["n_basis"]
        med = lambda k: sorted(float(r[k]) for r in g)[len(g) // 2]      # noqa: E731
        # (d)와 (e)를 한 문구로 묶으면 표가 본문과 어긋난다. (e)는 수치 rank가
        # 6/6으로 온전히 유지되므로 "붕괴"가 아니다 — 붕괴한 것은 (d)뿐이고
        # (e)의 문제는 rank를 잃지 않은 채로 쌍별 상관이 1에 가깝다는 것이다.
        r12 = med("rank_1e12")
        mech = (f"high pairwise collinearity, full numerical rank "
                f"(Σρ²/max = {md.fmt(med('offdiag_ratio'))})"
                if r12 >= float(nb) else
                f"optimization-induced collapse to rank {md.fmt(r12)}; pairwise "
                f"|ρ| ≈ 1 (Σρ²/max = {md.fmt(med('offdiag_ratio'))})")
        out.append([f"trained neural basis — {_base(arm)}", nb,
                    md.fmt(med("corr_rank_1e12")), md.fmt(r12), "ok", mech])
    body = md.table(["family", "nominal dimension", "numerical rank @1e-16",
                     "numerical rank @1e-10/1e-12", "factorization", "mechanism"],
                    out, align="lrrrll")
    cap = md.caption(
        n_table,
        "Three notions of dimension, and three distinct mechanisms by which a nominal "
        "dimension can overstate the usable one: raw monomials by a deterministic choice of "
        "coordinates, random features by the draw, and trained neural bases by the "
        "optimization itself. The mechanism is present in all three representation families "
        "but does not cost every arm its rank. The two trained arms in particular are not "
        "the same failure: the separate-net basis (d) "
        "loses numerical rank outright, whereas the shared-trunk subspace (e) keeps its full "
        "rank of 6 and is redundant only in the weaker sense of high pairwise correlation, "
        "which is what its span-invariant objective permits (§5.3.2). Numerical rank is "
        "threshold-dependent, so it is reported at more than "
        "one tolerance — a single integer would hide the choice. This is a shared "
        "diagnostic requirement, not one mechanism under three names (§5.3).",
        ["p1_conditioning.csv", "p1_neural_mode_sweep.csv",
         "p1_neural_collinearity.csv"],
        sha(data_dir, "manifest_p1_neural_collinearity.json"))
    return body + "\n" + cap + "\n"


def table10_spacetime(data_dir: str = DATA_DIR, n_table: int = 10) -> str:
    """(g) 시공간 PINN — 잔차와 고유값 오차의 역상관."""
    rows = _rows(data_dir, "p1_spacetime")
    g = {}
    for r in rows:
        g.setdefault(float(r["n_periods"]), []).append(r)
    out = []
    for T in sorted(g):
        rs = g[T]
        med = lambda k: sorted(float(r[k]) for r in rs)[len(rs) // 2]    # noqa: E731
        out.append([T, rs[0]["T_nondim"], med("omega_hat"), rs[0]["omega_ref"],
                    med("e_lam"), rs[0]["rel_lambda_bin"], med("e_lam_over_bin"),
                    med("loss_final"), med("seconds")])
    body = md.table(["periods in window", "T (nondim.)", "ω̂ (median)", "ω reference",
                     "e_λ (median)", "first-order relative Λ bin, 2/periods",
                     "e_λ / bin",
                     "final PDE residual", "seconds"], out, align="rrrrrrrrr")
    cap = md.caption(
        n_table,
        "Space–time PINN on P1, 4000 iterations, 3 seeds per window. From 4 to 32 periods the "
        "PDE residual falls by 8.2× between the endpoints while the eigenvalue error rises "
        "monotonically from below the resolution limit to 7 bins. The residual is not "
        "monotone — it rises again in the last interval — so its figure is an endpoint ratio "
        "and not a trend at every step. Minimizing the residual does not minimize the "
        "quantity of interest, because the eigenvalue is read off afterwards rather than "
        "optimized. The recovered frequency drifts monotonically downward by 25 % over the "
        "same sweep. **The bin column sets what this estimator can resolve, and the "
        "4-period row must be read with it.** An FFT over a window of T = periods·2π/ω has "
        "Δω/ω = 1/periods; since Λ = ω², a one-bin change in ω gives an exact relative Λ "
        "change of (1 + 1/periods)² − 1 = 2/periods + 1/periods², of which the column keeps "
        "the leading term (0.500 at 4 periods, against 0.5625 exact; the angular spacing "
        "Δω = 2π/T is 0.879 there). The windows are integer numbers of reference periods and "
        "the record length is exactly T, so the reference frequency falls on bin `periods` "
        "exactly — verified by passing the analytic mode through the same estimator, which "
        "then returns zero error. A peak in that bin therefore means the error is **below "
        "half a bin**, not that it is zero: the 4-period entry of order 1e-16 is an upper "
        "bound of about 0.25, not a measurement of exactness. From 16 periods e_λ/bin exceeds "
        "one, so those errors are resolved and the growth is real. That the windows were "
        "built from integer periods of the reference frequency is prior information, and it "
        "favours this negative control rather than penalizing it.",
        "p1_spacetime.csv", sha(data_dir, "manifest_p1_spacetime.json"))
    return body + "\n" + cap + "\n"


def table_degeneracy(data_dir: str = DATA_DIR, n_table: int = 8) -> str:
    """축퇴쌍을 개별 MAC으로 채점하면 안 된다는 근거 — 본문 산문이 아니라 표로 낸다."""
    rows = _rows(data_dir, "p2_degeneracy")
    out = [[r["m"], r["individual_mac_rotated"], r["subspace_mac_rotated"],
            r["max_principal_angle_rad"], r["note"]] for r in rows]
    body = md.table(["nodal diameters m", "individual MAC (rotated pair)",
                     "subspace MAC", "max principal angle [rad]", "note"],
                    out, align="rrrrl")
    cap = md.caption(
        n_table,
        "Degenerate nodal-diameter pairs on P2. A rotation within the eigenspace leaves "
        "the space unchanged but destroys individual MAC, which falls as low as 0.095, "
        "while the subspace MAC stays at 1.0000 and the largest principal angle is 3e-8. "
        "Scoring degenerate modes one at a time therefore reports failures that are not "
        "failures. This is a scoring result about an intact structure; it is unrelated "
        "to the splitting of a degenerate pair by a defect.",
        "p2_degeneracy.csv", sha(data_dir, "manifest_p2.json"))
    return body + "\n" + cap + "\n"


def table11_objective_weighting(data_dir: str = DATA_DIR, n_table: int = 11) -> str:
    """(e)의 모드1 악화 귀속 — 목적함수의 가중만 바꾼 대조."""
    rows = _rows(data_dir, "p1_e_objective_ablation")
    tr = {int(r["mode"]): r for r in rows if r["objective"] == "trace"}
    lg = {int(r["mode"]): r for r in rows if r["objective"] == "logtrace"}
    out = []
    for m in sorted(tr):
        a, b = tr[m], lg.get(m)
        if b is None:
            continue
        out.append([m, a["p_correct"], a["e_lam_median"], a["h2_over_l2"],
                    b["p_correct"], b["e_lam_median"], b["h2_over_l2"],
                    float(a["e_lam_median"]) / float(b["e_lam_median"])])
    body = md.table(["mode", "p_correct (Σλ)", "e_λ (Σλ)", "H²/L² (Σλ)",
                     "p_correct (Σlog λ)", "e_λ (Σlog λ)", "H²/L² (Σlog λ)",
                     "e_λ improvement"], out, align="rrrrrrrr")
    cap = md.caption(
        n_table,
        "Arm (e) with the objective weighting changed and nothing else — same "
        "architecture, budget, quadrature and seeds; 651 s versus 653 s, since the "
        "log-determinant form needs no eigendecomposition. The absolute trace Σλ_k "
        "weights every mode equally in absolute terms, so on this problem sacrificing "
        "7 % of λ₁ (0.87 in the objective) buys a 0.1 % gain on λ₆ (89) at 100:1; "
        "Σ log λ_k makes relative changes equivalent instead. Three predictions of that "
        "account are met: the inverted accuracy profile disappears, the improvement "
        "decreases monotonically with mode order — 2122×, 516×, 52.6×, 33.3×, 15.7× and "
        "11.3× for modes 1 to 6, i.e. largest exactly where the under-weighting was "
        "worst — and the cost is unchanged. The H²/L² ratio, however, still falls monotonically with "
        "mode number under both objectives — the low modes' error stays concentrated in "
        "curvature, which is a property of the shared representation and not of the "
        "weighting. This is an attribution experiment; the log-determinant objective is "
        "standard in subspace methods and is not proposed here as a contribution.",
        "p1_e_objective_ablation.csv",
        sha(data_dir, "manifest_p1_e_objective.json"))
    return body + "\n" + cap + "\n"


def _seed_note(*rowsets) -> str:
    """진단 열이 요약한 시드 수를 문장으로. **셀의 시드 수와 다를 수 있다.**

    실제로 달랐다 — 한때 셀은 50시드인데 진단은 앞 8시드만 요약했고, 그 결과 같은 κ₂(G)가
    표마다 최대 3배 어긋났다. 캡션이 셀의 시드 수만 적으면 독자는 진단도 그 수라고 읽는다."""
    seeds = sorted({int(float(v)) for rows in rowsets for r in (rows or ())
                    for v in (r.get("n_seeds_summarized") or r.get("n_seeds"),)
                    if v not in (None, "")})
    if not seeds:
        return ""
    n = str(seeds[0]) if len(seeds) == 1 else f"{seeds[0]}\u2013{seeds[-1]}"
    return f" The per-stage diagnostics summarize {n} seeds."


def table_orthogonality(data_dir: str = DATA_DIR, n_table: int = 5) -> str:
    """저장된 원시 출력의 직교성 — (a′)와 (b)가 갈라지는 지점의 정량.

    **두 지표를 구분해 싣는다.** ‖G⁻¹ − D⁻¹‖/‖D⁻¹‖는 계수사상의 거리이지 연산자의
    거리가 아니다. 논문의 규약은 Φ의 열이 기저함수이고 G = ΦᵀWΦ이므로 정확 사영은
    Π_G = ΦG⁻¹ΦᵀW이고 대각 규칙은 A_D = ΦD⁻¹ΦᵀW다. 실제 차이는 Π_G − A_D =
    Φ(G⁻¹ − D⁻¹)ΦᵀW이고, C = D^{-1/2}GD^{-1/2}로 두면 ‖Π_G − A_D‖_W = ‖I − C‖₂로
    정확히 계산된다 — 상한이 아니라 실측값이다.

    **A_D는 사영이 아니다.** A_D² = A_D는 D⁻¹G가 멱등일 때, 즉 C = I일 때만 성립한다.
    저장 모드가 서로 직교하지 않으면 대각 공식은 부정확한 사영이 아니라 **사영이 아닌
    연산자**다 — 이 논문의 핵심 메시지를 더 정확히 만드는 사실이므로 표기를 갈라 쓴다.
    arm 이름 "diagonal-approximation projection"은 실험명으로 유지한다.
    """
    rows = _rows(data_dir, "p1_orthogonality")
    main = [r for r in rows if r.get("gram_dev_fro_median") not in (None, "")]
    leak = {(_base(r["arm"])): r for r in rows
            if r.get("mode3_leak_rel_median") not in (None, "")}
    out = []
    for r in main:
        b = _base(r["arm"])
        # 모드-3 누출은 그 풀이가 쓰는 기저(k = 2)의 행에만 놓는다
        lk = (leak.get(b, {}).get("mode3_leak_rel_median")
              if int(float(r["n_prev_modes"])) == 2 else None)
        out.append([ARM_LABEL.get(b, r["arm"]), r["n_prev_modes"],
                    r["gram_dev_fro_median"], r["max_offdiag_median"],
                    r.get("lam_min_normalized_median"), r.get("kappa_gram_median"),
                    r.get("inv_gram_rel_diff_median"),
                    r.get("proj_op_diff_W_median"), lk, None, None])
    # **(b′)의 진단은 그 실행 자신이 남긴다.** 같은 스팬 성질을 재지만 출처가 다르므로
    # 행을 뒤에 붙이고, QR만 가진 두 열(직교성 결손·두 사영의 일치도)을 함께 싣는다.
    qr = [r for r in _rows_many(data_dir, "p1_qr_span_diagnostics")
          if str(r.get("dtype", "float64")) == "float64"]
    for r in sorted(qr, key=lambda r: int(float(r["k"]))):
        out.append([ARM_LABEL["b_prime_projection_qr"], r["k"], None, None,
                    r.get("lam_min_normalized"), r.get("kappa_gram"), None,
                    r.get("dev_identity"), None, r.get("qr_orth_defect"),
                    r.get("proj_agreement_rel")])
    body = md.table(["arm", "stored modes k", "‖C − I‖_F", "max |off-diagonal|",
                     "λ_min(C)", "κ₂(G)", "inverse-Gram discrepancy",
                     "‖Π_G − A_D‖_W", "mode-3 residual leakage",
                     "‖I − QᵀWQ‖₂", "(b) vs (b′) agreement"],
                    out, align="lrrrrrrrrrr")
    meth = next((r.get("solve_method") for r in rows if r.get("solve_method")), "")
    deep_qr = max(qr, key=lambda r: int(float(r["k"]))) if qr else {}
    qr_defect = float(deep_qr.get("qr_orth_defect", float("nan")) or "nan")
    qr_kappa = float(deep_qr.get("kappa_gram", float("nan")) or "nan")
    qr_agree = float(deep_qr.get("proj_agreement_rel", float("nan")) or "nan")
    # 정확 사영 arm에서 **실제로 풀이에 넘겨지는** 가장 깊은 그람 — 손으로 적으면
    # 재실행마다 어긋난다. k개가 저장된 그람은 모드 k+1을 풀 때 쓰이므로, 모드를 n개까지
    # 계산했으면 쓰이는 최대는 k = n−1이다(k = n 행은 마지막 저장 집합의 성질일 뿐이다).
    ex = [r for r in main if _base(r["arm"]) == "b_projection_exact"]
    kmax = max((int(float(r["n_prev_modes"])) for r in ex), default=0)
    deep = next((r for r in ex
                 if int(float(r["n_prev_modes"])) == max(kmax - 1, 2)), {}) if ex else {}
    kb = int(float(deep.get("n_prev_modes", 0)))
    kap_b = float(deep.get("kappa_gram_median", float("nan")))
    lmin_b = float(deep.get("lam_min_normalized_median", float("nan")))
    # fp64 유효자리 ≈ 16 − log10(κ₂)
    # 자릿수는 κ₂·u 기반 **최악 추정**이다 — 실제 손실은 보통 이보다 적다
    digits_b = (max(int(round(-math.log10(kap_b * 2.22e-16))), 0)
                if kap_b == kap_b and kap_b > 0 else 0)
    # 효과 rank가 모든 행에서 k와 같은지 확인해 그 사실만 서술한다 — 한 행의 값을
    # 대표로 쓰면 "stays full (2)"처럼 읽힌다.
    inv_max = max((float(r["inv_gram_rel_diff_median"]) for r in main
                   if r.get("inv_gram_rel_diff_median") not in (None, "")),
                  default=float("nan"))
    # 유효 rank는 arm마다 다르게 끝난다 — (a′)는 마지막 단계에서 처음 결손된다.
    # "모든 행에서 k와 같다"로 뭉치면 그 사실이 사라지고, 조건 대신 특이성으로 읽힌다.
    er = [r for r in main if r.get("eff_rank_1e12_median") not in (None, "")]
    ex_full = all(float(r["eff_rank_1e12_median"]) == float(r["n_prev_modes"])
                  for r in er if _base(r["arm"]) == "b_projection_exact")
    short = [(int(float(r["n_prev_modes"])), int(float(r["eff_rank_1e12_median"])))
             for r in er if _base(r["arm"]) != "b_projection_exact"
             and float(r["eff_rank_1e12_median"]) < float(r["n_prev_modes"])]
    cap = md.caption(
        n_table,
        "Orthogonality of the stored raw network outputs that each deflation stage is "
        "handed, on P1 with 50 seeds; all entries are medians over seeds. C is their "
        "mass-normalized Gram, `C = D^{-1/2}GD^{-1/2}` with `G = ΦᵀWΦ` the unnormalized "
        "Gram of the stored set and `D = diag(G)`. Two different quantities are "
        "reported and should not be conflated. The **inverse-Gram discrepancy** "
        "‖G⁻¹ − D⁻¹‖_F / ‖D⁻¹‖_F compares the two coefficient maps; it is not "
        "a bound on the error of the map. The **operator difference** is the exact "
        "norm of the two maps themselves. With Φ holding the stored modes "
        "as columns, the exact rule is the W-orthogonal projector `Π_G = ΦG⁻¹ΦᵀW` and the "
        "diagonal rule is the subtraction map `A_D = ΦD⁻¹ΦᵀW`, so "
        "`Π_G − A_D = Φ(G⁻¹ − D⁻¹)ΦᵀW`, and under the mass inner "
        "product that operator has norm ‖Π_G − A_D‖_W = ‖I − C‖₂ = max_i |1 − λ_i(C)| — a "
        "measured value, not a "
        "bound. **A_D is not a projection**: `A_D² = A_D` requires `D⁻¹G` to be idempotent, "
        "i.e. `C = I`. Once the stored set is not mutually orthogonal the diagonal formula "
        "is not an inexact projector but an operator that does not project at all, which is "
        "the sharper form of this paper\u2019s claim. The operator difference is O(1) while "
        f"the inverse-Gram discrepancy reaches {md.fmt(inv_max)}, which is why the "
        "two are kept apart. The last column is defined as "
        "‖D^{-1/2}ΦᵀWφ̃‖₂ / ‖φ̃‖_W with φ̃ the deflated mode-3 function, i.e. the "
        "root-sum-square over stored modes of cos∠(φ_k, φ̃) in the mass inner product; it "
        "is zero if and only if φ̃ is mass-orthogonal to every stored mode, and for a "
        "near-parallel stored pair its ceiling is √2 ≈ 1.414. Four things are visible. The stored outputs are strongly "
        "non-orthogonal in **both** arms — the largest off-diagonal is about 0.99 "
        "throughout — because deflation is applied when a mode is used, not to what is "
        "stored. The k = 2 rows agree to every digit across the two arms, and necessarily "
        "so: that basis was produced by the mode-1 solve, which deflates against nothing, "
        "and the mode-2 solve, whose Gram is 1 × 1, where diag(G) = G and the diagonal "
        "approximation is not an approximation; mode 3 is the first solve handed a Gram "
        "with an off-diagonal. Third, the projections do differ materially there — "
        "‖Π_G − A_D‖_W is comparable to ‖Π_G‖ = 1 — and the consequence is measured "
        "directly by the last column, the residual overlap of the deflated mode-3 function "
        "with the stored set: machine zero for the exact rule against order unity for the "
        "diagonal one, meaning the diagonally \"deflated\" function still lies largely in "
        "the span it was supposed to leave. Fourth, since the exact arm inverts this Gram, "
        "its conditioning is itself a result and is reported rather than assumed away. "
        f"κ₂(G) grows with k and λ_min(C) approaches zero: at k = {kb} — the stored set the "
        f"mode-{kb + 1} solve is handed, i.e. the deepest one behind the no-wall-through-mode-10 "
        f"claim — the exact arm reaches κ₂(G) = {md.fmt(kap_b)} and "
        f"λ_min(C) = {md.fmt(lmin_b)}. With u the fp64 unit roundoff, κ₂(G)·u is then of "
        f"order {md.fmt(kap_b * 2.22e-16, 1)}, i.e. roughly four to five decimal digits in "
        f"a worst-case conditioning estimate. That is thin but not "
        "exhausted, and it is a limit of the exact rule itself rather than an argument for "
        "the diagonal one, whose Gram at the same depth is worse by two further orders. "
        "Arm (a′) reaches an "
        "off-diagonal of exactly 1.0000, i.e. its stored set holds "
        "a duplicate — the collapse of mode 3 onto mode 1 feeding back into the next stage. "
        + (f"At a 1e-12 relative cutoff the exact arm\u2019s median effective rank equals k "
           f"in every row, so its Gram is ill-conditioned rather than numerically singular "
           f"there. " if ex_full else "")
        + (f"Arm (a\u2032) instead loses rank outright at k = {short[0][0]}, where the "
           f"median effective rank is {short[0][1]}. " if short else "")
        + (f"The k = {kmax} row is the set left after all {kmax} modes have been stored; in "
           f"a {kmax}-mode run it is handed to no solve and is reported only to show that "
           f"the trend continues. " if kmax else "")
        + f"The exact arm solves with {meth}. "
        + (("The last two columns belong to arm (b′), which was added on 2026-08-26 and "
            "reaches the same projection by a different numerical route: it never forms "
            "G, orthonormalizing the stored span in the mass inner product instead, so "
            "the quantity its accuracy depends on is not κ₂(G) but the orthonormality "
            f"defect ‖I − QᵀWQ‖₂, which stays at {md.fmt(qr_defect)} while κ₂(G) reaches "
            f"{md.fmt(qr_kappa)} at the same depth. Its own diagnostics come from its own "
            "run, so the two blocks are separate measurements of the same span property, "
            "not the same numbers rearranged. The agreement column is the relative "
            "difference between the two deflated functions in the mass norm on the run's "
            "own data: the two rules are the same projection, and the measured agreement "
            f"of {md.fmt(qr_agree)} is the numerical check of that statement, not an "
            "assumption. Non-orthogonality of the stored set is therefore not an artifact "
            "of a badly chosen arm — it arises in the exact arm too, which is why its "
            "own κ₂(G) column is a result rather than a caveat.") if qr else "")
        + _seed_note(main, qr),
        ["p1_orthogonality.csv", "p1_qr_span_diagnostics.csv"],
        sha(data_dir, "manifest_p1_orthogonality.json"))
    return body + "\n" + cap + "\n"


def table_headline(data_dir: str = DATA_DIR, n_table: int = 11) -> str:
    """정본 비율 — 초록·본문·캡션·결론이 **모두 이 표를 가리킨다**.

    같은 비율을 다섯 곳에 손으로 적었다가 서로 어긋났다(5.34 / 5.4 / "5.40–5.50" / 5.42가
    한 문서에 공존, narrowing 1.6 대 1.7, 정확도 격차 two 대 three orders). 하나의 계산에서
    렌더링하면 그 종류의 불일치가 구조적으로 불가능해진다."""
    rows = _rows(data_dir, "p1_cost_headline")
    out = []
    for r in rows:
        out.append([r["quantity"], r.get("denominator_solver"),
                    r.get("numerator_solver"),
                    r.get("cost_ratio"), r["cost_orders"],
                    r.get("expected_success_orders"),
                    r.get("accuracy_orders"), r.get("bound")])
    body = md.table(["comparison", "cheaper side", "costlier side", "cost ratio",
                     "per attempt, orders", "expected success, orders",
                     "accuracy advantage of the cheaper side, orders", "bound"],
                    out, align="lllrrrrl")
    cap = md.caption(
        n_table,
        "Canonical cost and accuracy ratios. **The two problems are not on a common "
        "accuracy target.** The P1 rows are taken at the 1e-4 target on mode 1. The P4 "
        "rows cannot be: the neural arm there plateaus at e_λ ≈ 3e-3, so its comparison is "
        "against the cheapest classical mesh that matches or beats that accuracy, which is "
        "a weaker target. The P4 gap and the difference derived from it are therefore "
        "read at each problem\u2019s own achievable accuracy, and the two are compared as "
        "such rather than as a single controlled cross-problem contrast. Every ratio quoted "
        "anywhere in this paper is read from this "
        "table rather than restated, because an earlier draft carried the same number in "
        "five places and they drifted apart. Costs are per attempt: one numpy/CPU run "
        "counted from basis construction onward for the classical and random-feature "
        "families, and a directly measured single-seed run — median of three — for the "
        "gradient-trained arms on both P1 and P4, so the two problems are on the same "
        "footing. The last column states whether a row is a direct measurement or a "
        "bound: a row would be an upper bound if only an ensemble wall-clock were "
        "available for the neural side, since that overstates its per-attempt cost, and "
        "the difference row would then be a lower bound. The last row is the difference "
        "between the two problems — the observed P1-to-P4 gap difference, named as an "
        "observation and not as an effect of the corner, for the reasons given in §5.4. Both "
        "accounting bases are given as columns rather than one being described in prose, "
        "since expected time-to-success is the paper\u2019s declared cost metric: the "
        "per-attempt column divides wall-clock by attempt, the expected-success column "
        "divides it further by p_correct on each side. On P1 mode 1 the two coincide because "
        "p_correct = 1.00 for both sides; on P4 they separate because the neural arm\u2019s "
        "p_correct is 0.76, 0.68 and 0.66. The verbal claim that the observed gap is about two "
        "orders smaller on P4 "
        "holds on either basis. Rounded "
        "verbal characterizations such as \"about two orders\" appear in the text; exact "
        "values appear only here.",
        ["p1_cost_headline.csv", "p1_cheapest_to_epsilon.csv",
         "p4_classical_vs_neural.csv"],
        sha(data_dir, "manifest_cost_headline.json"))
    return body + "\n" + cap + "\n"


def P4_KIND(arm) -> str:
    """`p4_neural_2d[qr projection]` → `qr`."""
    a = str(arm)
    return a.split("[")[-1].rstrip("]").split()[0] if "[" in a else a


P4_LABEL = {"exact": "(b) exact", "diagonal": "(a′) diagonal", "qr": "(b′) QR"}


def _p4_sub_lambda1(data_dir: str) -> dict:
    """Λ < Λ₁ 인 셀을 기록에서 센다. 정확 적분이면 어떤 허용함수에서도 R(u) ≥ λ₁ 이므로,
    이 셀들은 구적이 최적화가 도달한 함수를 분해하지 못했다는 뜻이다 — 사영 규칙과 무관하다
    (모드 1에는 deflation이 없다). 없는 파일은 건너뛴다: 사후추가 arm은 자기 파일에 있다."""
    recs = []
    for n in ("p4_neural_records", "p4_neural_arms_records"):
        path = os.path.join(data_dir, f"{n}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            recs += [json.loads(l) for l in f if l.strip()]
    if not recs:
        return {}
    lam1 = next((r["Lam_ref"] for r in recs if int(r["mode"]) == 1), None)
    if lam1 is None:
        return {}
    viol = [r for r in recs if r["Lam"] < lam1]
    if not viol:
        return {}
    depth = sorted((lam1 - r["Lam"]) / lam1 for r in viol)
    per = {}
    for r in viol:
        per[P4_KIND(r["arm"])] = per.get(P4_KIND(r["arm"]), 0) + 1
    return {"n": len(viol), "total": len(recs),
            "median_pct": 100.0 * depth[len(depth) // 2],
            "max_pct": 100.0 * depth[-1],
            "per_arm": tuple(per[k] for k in sorted(per)),
            "at_mode1": sum(1 for r in viol if int(r["mode"]) == 1),
            "all_nonconv": all(r["outcome"] == "non_converged" for r in viol)}


def table_p4(data_dir: str = DATA_DIR, n_table: int = 13) -> str:
    """P4 2차원 신경 arm의 사전등록 4분류 — P1과 같은 잣대, 세 사영 규칙."""
    rows = _rows_many(data_dir, "p4_neural", "p4_neural_arms")
    order = {"exact": 0, "diagonal": 1, "qr": 2}
    rows = sorted(rows, key=lambda r: (order.get(P4_KIND(r["arm"]), 9),
                                       int(float(r["mode"]))))
    out = [[P4_LABEL.get(P4_KIND(r["arm"]), r["arm"]), r["mode"],
            _frac(r, "correct"), _frac(r, "lower_mode_basin"),
            _frac(r, "spurious"), _frac(r, "non_converged"), r["p_correct"],
            _ci(r), r["p_accurate"], r["e_lam_median"], r["mac_median"],
            r["Lam_median"], r["Lam_ref"]] for r in rows]
    body = md.table(["arm", "mode", "correct", "lower-mode basin", "spurious",
                     "non-converged", "p_correct", "95 % Wilson", "p_accurate",
                     "e_λ (median)", "MAC (median)", "Λ_el (median)",
                     "Λ_el reference"],
                    out, align="lrrrrrrcrrrrr")
    n_runs = sum(int(float(r["n"])) for r in rows)
    # 전제(저장 스팬의 직교성)는 별도 진단 실행에서 온다 — 시드 수가 다르므로 파일도 다르다.
    # **캡션이 말하는 행을 읽는다.** 문장은 "the stored set handed to the mode-3 solve"이고
    # 같은 문장에서 P1의 0.991·286에 견주는데, 그 두 값은 P1의 k=2다. 최심 행(k=5)을
    # 읽으면 문장은 모드 3을 말하면서 숫자는 다른 깊이를 가리켜 대응이 깨진다.
    span = _rows_many(data_dir, "p4_span_diagnostics")
    at_k2 = next((r for r in span if int(float(r["k"])) == 2), None)
    ref = at_k2 or (max(span, key=lambda r: int(float(r["k"]))) if span else {})
    p4_dev = float(ref.get("dev_identity", float("nan")) or "nan")
    p4_kap = float(ref.get("kappa_gram", float("nan")) or "nan")
    n_basin = sum(int(float(r["lower_mode_basin"])) for r in rows)
    kinds = sorted({P4_KIND(r["arm"]) for r in rows})
    ex3 = next((r for r in rows if P4_KIND(r["arm"]) == "exact"
                and int(float(r["mode"])) == 3), {})
    sub = _p4_sub_lambda1(data_dir)
    cap = md.caption(
        n_table,
        f"Three deflation rules on P4, in two dimensions, 50 seeds per cell, 4000 "
        "iterations, clamped condition imposed exactly by the distance-function "
        "construction of §3.3. Scored by the same pre-specified rules as P1, with MAC "
        "computed against the FEM reference evaluated at the neural arm's own quadrature "
        "points. The diagonal and QR rules were added on 2026-08-26 and the scoring was "
        "not changed for them. **The rules are not distinguishable here.** "
        + (f"Across {n_runs} runs there is not one lower-mode-basin outcome under any of "
           f"the {len(kinds)} rules — including the diagonal rule, which on P1 collapses "
           "completely from mode 3 on. That is a bound on the P1 mechanism rather than a "
           "confirmation of it: the wall there follows from a strongly non-orthogonal "
           "stored set (Table 2), and whether that precondition arises is a property of "
           "the problem, not of the rule. On P4 it is measured and it does not arise: the "
           f"stored set handed to the mode-3 solve has ‖I − C‖₂ = {md.fmt(p4_dev, 2)} and "
           f"κ₂(G) = {md.fmt(p4_kap)}, i.e. mass-orthogonal to machine precision, against "
           "0.991 and 286 for the same quantities on the beam, so by the identity of "
           "§5.1 the diagonal rule **is** the exact projection here. This table therefore "
           "cannot be read as evidence that the exact projection removes lower-mode "
           "attraction in two dimensions; nothing here needed removing, and that is what "
           "the mechanism predicts once its precondition is measured rather than assumed. "
           if n_basin == 0 else
           f"Across {n_runs} runs there are {n_basin} lower-mode-basin outcomes. ")
        + "What does carry over is the failure that is left. e_λ settles near 3e-3 in every "
        "arm, one to two orders above the reference uncertainty, and the residue is a mix "
        "of non-convergence and spurious solutions rather than convergence failures alone — "
        + (f"the exact rule at mode 3 has {ex3.get('spurious', '?')} spurious against "
           f"{ex3.get('non_converged', '?')} non-converged" if ex3 else "")
        + ". "
        + ((f"Part of that non-convergence is an instrument failure rather than an "
            f"optimizer failure: {sub['n']} of the {sub['total']} runs report Λ below the "
            f"mode-1 reference, by a median of {sub['median_pct']:.0f} % and up to "
            f"{sub['max_pct']:.0f} %, which exact integration forbids for any admissible "
            f"trial function. They split "
            + "/".join(str(x) for x in sub["per_arm"]) +
            f" over the three rules and {sub['at_mode1']} of them occur at mode 1, where no "
            f"deflation is applied, so the quadrature and not the rule is the cause. "
            f"{'All' if sub['all_nonconv'] else 'Not all'} of them are classified "
            f"non-converged, so none of them enters a scored claim; they are attributed to "
            f"e_measurement rather than e_optimization (§6.2, item 13). ")
           if sub.get("all_nonconv") else "")
        + "A fair-comparison note: the exact path adds 1e-14 times the identity to the "
        "Gram before solving, as in the committed arm-(b) run, while the diagonal and QR "
        "paths add nothing because one inverts nothing and the other never forms the Gram. "
        "Modes 4-6 are not scored, and not for want of a reference: the Richardson "
        "references for those modes exist and meet the uncertainty target. They form a "
        "near-degenerate cluster, with extrapolated eigenvalues 12.105, 12.613 and "
        "12.724 and consecutive relative gaps of 4.2 % and 0.9 %, both inside the "
        "pre-specified 5 % eigenvalue tolerance, so the simple-mode rule cannot "
        "separate them and scoring them needs the eigenspace branch used on P2. All three "
        "arms are costed in §5.4 on the same basis, one seed and the median of three "
        "repetitions, measured in a single session; the mode-1 cell, where the three "
        "arms perform identical computations because no deflation is applied, differs "
        "by 9 %, which calibrates the timing noise and is why no cost difference "
        "between the rules is claimed here." + _seed_note(span),
        ["p4_neural.csv", "p4_neural_arms.csv", "p4_span_diagnostics.csv"],
        sha(data_dir, "manifest_p4_neural.json"))
    return body + "\n" + cap + "\n"


def table_fp32(data_dir: str = DATA_DIR, n_table: int = 16) -> str:
    """float32 스트레스 — 같은 사영, 다른 수치경로가 정밀도에서 갈리는 지점.

    **이 표가 새 논거의 전부다.** (b)와 (b′)은 fp64에서 같은 답을 내므로 둘 중 하나를
    고를 이유가 없어 보인다. 그러나 (b)의 정확도는 κ₂(G)에, (b′)의 정확도는 κ₂(Φ) =
    √κ₂(G)에 걸린다. 유효자리를 다 쓰는 정밀도로 내리면 그 차이가 결과로 나타난다.
    """
    fp32 = _rows_many(data_dir, "p1_fp32_stress")
    if not fp32:
        return ""
    diag = {(str(r["arm"]), int(float(r["stage_mode"]))): r
            for r in _rows_many(data_dir, "p1_fp32_span_diagnostics")}
    base = {(_base(r["arm"]), int(float(r["mode"]))): r
            for r in _rows_many(data_dir, "p1_neural_mode_sweep", "p1_neural_qr")}
    out = []
    for a in ARM_ORDER:
        for r in sorted((x for x in fp32 if _base(x["arm"]) == a),
                        key=lambda x: int(float(x["mode"]))):
            m = int(float(r["mode"]))
            b64 = base.get((a, m), {})
            dg = diag.get((str(r["arm"]), m), {})
            out.append([ARM_LABEL.get(a, r["arm"]), m,
                        b64.get("p_correct"), r["p_correct"], _ci(r),
                        _frac(r, "lower_mode_basin"), _frac(r, "spurious"),
                        _frac(r, "non_converged"),
                        dg.get("kappa_gram_times_eps"), dg.get("qr_orth_defect"),
                        dg.get("n_dropped")])
    body = md.table(["arm", "mode", "p_correct (fp64)", "p_correct (fp32)",
                     "95 % Wilson (fp32)", "lower-mode basin", "spurious",
                     "non-converged", "κ₂(G)·u (fp32)", "‖I − QᵀWQ‖₂ (fp32)",
                     "dropped directions"],
                    out, align="lrrrcrrrrrr")

    # 캡션의 수치는 데이터에서 읽는다 — 손으로 적으면 재실행에서 어긋난다
    prod = [(str(r["arm"]), int(float(r["stage_mode"])), float(r["kappa_gram_times_eps"]))
            for r in diag.values() if r.get("kappa_gram_times_eps") not in (None, "")]
    # 기준의 주어는 **그람 경로**다 — 최대값도 그 arm의 스팬에서 읽는다
    finite = [v for a, _, v in prod
              if a == "b_projection_exact" and math.isfinite(v)]
    worst = max(finite, default=float("nan"))
    over = sorted(m for a, m, v in prod
                  if a == "b_projection_exact" and (v > 1.0 or not math.isfinite(v)))
    first_over = over[0] if over else None
    sing = sorted(m for a, m, v in prod
                  if a == "b_projection_exact" and not math.isfinite(v))
    first_sing = sing[0] if sing else None
    qdef = max((float(r["qr_orth_defect"]) for r in diag.values()
                if str(r.get("arm")) == "b_prime_projection_qr"
                and r.get("qr_orth_defect") not in (None, "")), default=float("nan"))
    drop = max((int(float(r["n_dropped"])) for r in diag.values()
                if str(r.get("arm")) == "b_prime_projection_qr"
                and r.get("n_dropped") not in (None, "")), default=0)
    ex = [r for r in fp32 if _base(r["arm"]) == "b_projection_exact"]
    qr = [r for r in fp32 if _base(r["arm"]) == "b_prime_projection_qr"]
    hi = [int(float(r["mode"])) for r in ex if int(float(r["mode"])) >= 6]
    mean_ex = (sum(float(r["p_correct"]) for r in ex if int(float(r["mode"])) >= 6)
               / len(hi)) if hi else float("nan")
    mean_qr = (sum(float(r["p_correct"]) for r in qr if int(float(r["mode"])) >= 6)
               / max(len([1 for r in qr if int(float(r["mode"])) >= 6]), 1))
    cap = md.caption(
        n_table,
        "The same deflation, two numerical routes, at float32. Both arms compute the "
        "W-orthogonal projection onto the stored span and agree in float64 to the accuracy "
        "the Gram route itself can hold (Table 2, last column), so this table varies one "
        "thing: the working "
        "precision. Arm (b) forms the Gram G = ΦᵀWΦ and solves with it, which costs "
        "κ₂(G); arm (b′) orthonormalizes the stored span instead and costs κ₂(Φ) = "
        "√κ₂(G). The criterion is read from the product of the conditioning with the "
        "unit roundoff u of the working precision: where κ₂(G)·u approaches or exceeds "
        "one, the Gram route has no significant digits left. At float32 that product first "
        f"exceeds one at the stored set handed to the mode-{first_over} solve"
        + (f", is {md.fmt(worst)} at its largest finite value, and is not finite from "
           f"mode {first_sing} on, where the Gram is numerically singular at this "
           "precision" if first_sing else f" and reaches {md.fmt(worst)}")
        + ". The orthonormality defect of the QR basis over the same range stays below "
        f"{md.fmt(qdef, 2)}. That column is measured the same way for both arms because it "
        "is a property of the stored span, but only (b′) computes with it; for (b) it is "
        "reported to show that the span itself is still usable where the Gram is not. "
        "Averaged over modes 6–10 — where the fp64 conditioning "
        f"is worst — p_correct is {md.fmt(mean_ex)} for (b) against {md.fmt(mean_qr)} for "
        "(b′). The last column is what makes the QR route rank-aware rather than merely "
        "better conditioned: a stored direction whose norm after orthogonalization falls "
        "below √u times its norm before it is dropped instead of inverted, and the count "
        f"of dropped directions ({drop} at the deepest stage here) is the arm's own report "
        "that the stored set has become numerically rank-deficient at this precision. A "
        "dropped direction is not deflated against, so the arm trades a guaranteed loss of "
        "orthogonality for a declared loss of coverage; both are visible in the table. "
        "The fp64 column is the same configuration at double precision and is not re-run "
        "here — it is the corresponding row of Table 3. Precision is the only variable: "
        "seeds, budget, initialization, quadrature rule and network are identical, and the "
        "Gauss nodes and weights are built in float64 and cast down so the quadrature "
        "itself does not move." + _seed_note(diag.values()),
        ["p1_fp32_stress.csv", "p1_fp32_span_diagnostics.csv"],
        sha(data_dir, "manifest_p1_fp32.json"))
    return body + "\n" + cap + "\n"


def table_deep_modes(data_dir: str = DATA_DIR, n_table: int = 5) -> str:
    """P1 모드 11–16으로 확장한 두 사영 경로 — 벽이 아니라 점진적 열화다.

    이 표가 답하는 질문은 "fp64에서 그람 경로가 무너지는 깊이에 닿는가"였다. 닿지 않는다:
    가장 깊은 저장 집합에서도 κ₂(G)·u가 0.082에 그친다. 그래서 표의 내용은 붕괴가 아니라
    **붕괴하지 않는다는 확인**과, 깊이에 따라 신뢰도가 완만히 떨어진다는 관측이다.

    **이 셀들은 정정된 참조로 채점됐다.** 처음 채점에 쓴 모드형 참조는 상쇄소거로 모드
    13부터 열화해 모드 14의 천장을 MAC 0.887로 만들었고, 그 아래 셀들이 전부 spurious로
    분류됐다(§3.2). 풀이는 그대로이고 채점만 다시 했다 — 진단은 두 실행에서 자릿수까지
    같다."""
    rows = _rows_many(data_dir, "p1_fp64_deep_b_stress", "p1_fp64_deep_qr_stress")
    if not rows:
        return ""
    diag = {}
    for r in _rows_many(data_dir, "p1_fp64_deep_b_span_diagnostics",
                        "p1_fp64_deep_qr_span_diagnostics"):
        diag[(_base(r["arm"]), int(float(r["stage_mode"])))] = r
    by = {(_base(r["arm"]), int(float(r["mode"]))): r for r in rows}
    modes = sorted({int(float(r["mode"])) for r in rows if int(float(r["mode"])) >= 8})
    out = []
    for a in ("b_projection_exact", "b_prime_projection_qr"):
        for m in modes:
            r = by.get((a, m))
            if r is None:
                continue
            g = diag.get((a, m), {})
            out.append([ARM_LABEL[a], m, _frac(r, "correct"),
                        _frac(r, "lower_mode_basin"), _frac(r, "spurious"),
                        _frac(r, "non_converged"), r["p_correct"], _ci(r),
                        g.get("kappa_gram"), g.get("kappa_gram_times_eps"),
                        g.get("qr_orth_defect")])
    body = md.table(["arm", "mode", "correct", "lower-mode basin", "spurious",
                     "non-converged", "p_correct", "95 % Wilson", "κ₂(G)",
                     "κ₂(G)·u", "‖I − QᵀWQ‖₂"], out, align="lrrrrrrcrrr")

    n_runs = sum(int(float(r["n"])) for r in rows)
    n_basin = sum(int(float(r["lower_mode_basin"])) for r in rows)
    # 기준의 주어는 **그람 경로**다 — 그 arm의 스팬에서 읽고, 다른 arm의 값도 함께 밝힌다.
    def _ku(arm):
        v = [float(g["kappa_gram_times_eps"]) for (a, _), g in diag.items()
             if a == arm and g.get("kappa_gram_times_eps") not in (None, "")]
        return max(v) if v else float("nan")

    deep = _ku("b_projection_exact")
    deep_qr = _ku("b_prime_projection_qr")
    qmax = max((float(g["qr_orth_defect"]) for g in diag.values()
                if g.get("qr_orth_defect") not in (None, "")), default=float("nan"))
    pooled = {}
    for a in ("b_projection_exact", "b_prime_projection_qr"):
        cs = [int(float(by[(a, m)]["correct"])) for m in modes
              if (a, m) in by and m >= 13]
        ns = [int(float(by[(a, m)]["n"])) for m in modes if (a, m) in by and m >= 13]
        pooled[a] = (sum(cs), sum(ns))
    cap = md.caption(
        n_table,
        "Extension of the two projection routes to mode 16 on P1, 50 seeds per cell, 4000 "
        "iterations, I0, fixed Gauss; modes 8-10 are repeated from Table 3 so the trend is "
        "read on one axis. The question this answers is whether double precision reaches "
        "the depth at which the normal-equation route loses its conditioning. It does not: "
        f"κ₂(G)·u reaches only {md.fmt(deep, 2)} for the Gram route at the deepest stored "
        f"set ({md.fmt(deep_qr, 2)} on the other arm's own span), an order of magnitude "
        "below one, and the criterion accordingly predicts no algebraic breakdown here. The prediction holds. There is "
        f"**no lower-mode-basin outcome in any of the {n_runs} runs**, no stored span "
        f"becomes non-finite, and the orthonormality defect stays below {md.fmt(qmax, 2)}. "
        "What the extension shows is not a wall but a gradual decline with depth, from 1.00 "
        "through mode 7 to roughly one half by mode 16, with the residue dominated by the "
        "convergence gate rather than by misidentification. The two routes remain "
        f"indistinguishable there: pooled over modes 13-16 they score "
        f"{pooled['b_projection_exact'][0]}/{pooled['b_projection_exact'][1]} and "
        f"{pooled['b_prime_projection_qr'][0]}/{pooled['b_prime_projection_qr'][1]}. "
        "These cells are scored against the corrected mode-shape reference of §3.2. The "
        "first scoring used the textbook form, whose cancellation capped MAC at 0.887 from "
        "mode 14 upward and classified correct solutions as spurious; the solves are "
        "unchanged and were re-scored, and the per-stage diagnostics reproduce to every "
        "digit across the two scorings, which is what identifies the difference as scoring "
        "rather than solving." + _seed_note(diag.values()),
        ["p1_fp64_deep_b_stress.csv", "p1_fp64_deep_qr_stress.csv",
         "p1_fp64_deep_b_span_diagnostics.csv",
         "p1_fp64_deep_qr_span_diagnostics.csv"],
        sha(data_dir, "manifest_p1_fp64_deep_b.json"))
    return body + "\n" + cap + "\n"


# ---------------------------------------------------------------- 부록


def appendix_a_nondimensional(n_table: int | None = None) -> str:
    """부록 A — 무차원 문제 정의. **절대 치수·물성·Hz는 넣지 않는다**(경계 정본 §1)."""
    from ..problems.p2_annulus import P2_GEOMETRY as G2
    from ..problems.p3_spring import P3_CONFIG as G3
    from ..problems.p4_lshape import P4_GEOMETRY as G4
    rows = [
        ["P1", "clamped–free uniform Euler–Bernoulli beam", "—",
         "Λ = (βL)⁴; independent of material and dimensions"],
        ["P2", "annular Kirchhoff plate, clamped inner / free outer",
         f"a/b = {md.fmt(G2['a'] / G2['b'])}, ν = {md.fmt(G2['nu'])}",
         "Λ = (kb)⁴ with b = 1"],
        ["P3", "beam with an internal rotational spring",
         f"x_c/L = {md.fmt(G3['xc_over_L'])}, "
         f"k̂ ∈ {{{', '.join(md.fmt(k) for k in G3['k_hats'])}}}",
         "Λ = (βL)⁴; k̂ = k_θL/(EI)"],
        ["P4", "L-shaped plane-stress domain",
         f"unit-length arms, ν = {md.fmt(G4['nu'])}",
         "Λ_el = ω²ρL²/E with L = E = ρ = 1 (Ω denotes the domain)"]]
    body = md.table(["problem", "domain", "dimensionless parameters", "eigenvalue"],
                    rows, align="llll")
    note = ("\nEvery test problem is stated dimensionlessly. Absolute dimensions, "
            "material properties and frequencies are deliberately absent: none of the "
            "benchmark's conclusions depend on them, and quoting them would tie the "
            "problems to one particular structure.\n")
    return body + note


# 공시값의 정본은 코드다. 그런데 렌더링은 torch가 없는 호스트에서 돌아야 하므로 여기에
# 사본을 두고, torch가 있는 환경의 테스트
# (`test_render_tables_gpu.py::test_disclosure_defaults_match_the_code`)가 사본과 코드의
# 일치를 강제한다. 사본이 낡으면 그 테스트가 실패한다.
def _spacetime_defaults() -> dict:
    """`solve_spacetime`의 기본 인자를 그대로 읽는다 — 부록이 코드와 어긋나지 않게."""
    import inspect
    try:
        from ..neural import spacetime
    except Exception:                      # torch 없는 환경(문서 렌더)에서는 사본을 쓴다
        return {"n_col": 4096, "n_ic": 256, "n_bc": 256, "n_probe": 512,
                "w_ic": 100.0, "w_bc": 10.0, "iters": 4000, "lr": 2e-3}
    sig = inspect.signature(spacetime.solve_spacetime)
    return {k: sig.parameters[k].default
            for k in ("n_col", "n_ic", "n_bc", "n_probe", "w_ic", "w_bc",
                      "iters", "lr")}


ST = _spacetime_defaults()

DISCLOSED = {"width": 64, "depth": 4, "activation": "tanh",
             "spacetime_params": 12737, "spacetime_width": 64, "spacetime_depth": 4,
             "p4_width": 64, "p4_depth": 4, "p4_params": 12802,
             "p4_n_q": 1200, "p4_n_per_block": 20}


def _working_precisions(data_dir: str = DATA_DIR) -> str:
    """커밋된 결과가 실제로 어떤 작업 정밀도에서 나왔는지. **환경 기록은 이걸 모른다.**

    환경 기록은 프로세스의 기본 dtype을 적을 뿐이고, 축소정밀도 축은 그 기본을 일부러
    바꿔 돌린 실행이다. "float64 throughout"만 남으면 §5.1과 부록 D의 존재 자체와
    어긋나므로, 결과 CSV의 dtype 열을 세어 실제 목록을 덧붙인다."""
    import csv
    import glob

    order = ["float64", "float32", "bfloat16", "float16"]
    seen = set()
    for p_ in glob.glob(os.path.join(data_dir, "*.csv")):
        try:
            with open(p_, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    # 드라이버마다 "float64"로도 "torch.float64"로도 적었다.
                    # 정규화하지 않으면 같은 정밀도가 두 번 세어진다.
                    if r.get("dtype"):
                        seen.add(str(r["dtype"]).replace("torch.", "").strip())
        except Exception:
            continue
    if not seen:
        return ""
    named = [d for d in order if d in seen] + sorted(seen - set(order))
    if named == ["float64"]:
        return ". Every committed solve runs at float64"
    rest = ", ".join(named[1:])
    return (f". The committed solves run at {named[0]} except the declared "
            f"reduced-precision axis of §5.1 and Appendix D, which runs at {rest}; "
            "each row and each manifest records the working precision it used")


def availability_record(data_dir: str = DATA_DIR) -> str:
    """Data and code availability 절 — **환경 기록에서 렌더링한다.**

    손으로 쓰면 어긋난다. 실제로 이 저장소의 호스트 venv는 작업 중에 numpy를 올렸고,
    그래서 산출물 절반은 numpy 2.1.0으로, 절반은 2.5.0으로 만들어졌다. 그 사실은 각
    manifest에 남아 있으므로 여기서 세어 그대로 적는다 — "라이브러리 버전 하나"로 뭉개면
    거짓이 된다.

    DOI와 최종 SHA는 제출 시점에 정해지므로 자리만 만들고 값은 비운다.
    """
    import glob
    import json

    def env(tag):
        p_ = os.path.join(data_dir, f"environment_{tag}.json")
        if not os.path.exists(p_):
            return {}
        with open(p_, encoding="utf-8") as f:
            return json.load(f)

    h, g = env("host"), env("gpu")

    # 산출물을 실제로 만든 버전 조합을 manifest에서 센다
    combos = {}
    for f_ in sorted(glob.glob(os.path.join(data_dir, "manifest*.json"))):
        with open(f_, encoding="utf-8") as fh:
            d = json.load(fh)
        key = (d.get("python"), d.get("numpy"), d.get("scipy"), d.get("mpmath"))
        combos[key] = combos.get(key, 0) + 1
    combo_txt = "; ".join(
        f"Python {k[0]}, NumPy {k[1]}, SciPy {k[2]}, mpmath {k[3]} ({n} artifacts)"
        for k, n in sorted(combos.items(), key=lambda kv: -kv[1]) if k[0])

    rows = [
        ["Repository DOI", "*minted at submission*"],
        ["Final git SHA", "*recorded at submission; every artifact also carries the "
                          "SHA of the commit that produced it, in its own manifest*"],
        ["Hardware", f"{h.get('cpu') or g.get('cpu', '—')}; "
                     f"{h.get('gpu') or g.get('gpu', '—')}"
                     + (f" (driver {h.get('nvidia_driver') or g.get('nvidia_driver')})"
                        if (h.get("nvidia_driver") or g.get("nvidia_driver")) else "")],
        ["Operating system", f"{h.get('os', '—')}, kernel {h.get('kernel', '—')}, "
                             f"{h.get('platform', '—').split()[-1]}"],
        ["Classical solves, references, rendering",
         f"CPU only. Python {h.get('python', '—')}, "
         f"NumPy {h.get('numpy', '—')}, SciPy {h.get('scipy', '—')}, "
         f"mpmath {h.get('mpmath', '—')}, Matplotlib {h.get('matplotlib', '—')}"],
        ["Neural arms",
         f"GPU. PyTorch {g.get('torch', '—')} on CUDA {g.get('torch_cuda', '—')}"
         + (f", cuDNN {g.get('cudnn')}" if g.get("cudnn") else "")
         + f"; Python {g.get('python', '—')}, NumPy {g.get('numpy', '—')}, "
           f"SciPy {g.get('scipy', '—')}"],
        ["Versions that produced the committed artifacts",
         combo_txt or "—"],
        ["Precision",
         f"{h.get('fp64', '—')}. On the GPU side, {g.get('fp64', '—')}"
         + _working_precisions(data_dir)],
    ]
    body = md.table(["field", "value"], rows, align="ll")
    note = ("\nEvery CSV in the repository is accompanied by a JSON manifest carrying the "
            "git SHA, timestamp, library versions, platform and precision of the run that "
            "wrote it, together with the driver name and its arguments. The two library "
            "combinations above are a fact of the record rather than an inconsistency: the "
            "host environment was upgraded during the work, and each artifact states which "
            "combination produced it. The eigenvalue conclusions do not depend on that "
            "difference — the regression suite runs against the committed data under the "
            "current environment.\n")
    return body + note


def appendix_d_bfloat16(data_dir: str = DATA_DIR) -> str:
    """부록 D — bfloat16을 세 번째 정밀도 점으로 쓰려던 시도와, 쓰지 못한 이유.

    **왜 싣는가.** 리뷰어의 첫 질문은 "왜 fp32에서 멈췄는가"다. 답이 "라이브러리가
    막아서"라면 방법의 한계가 아니라 구현의 한계이므로 고쳐야 하고, 실제로 고쳤다.
    고치고 나서 드러난 진짜 이유는 채점 쪽이었다 — 수렴 게이트가 산술보다 곱다.
    그것은 이 논문이 e_measurement라 부르는 바로 그 실패이므로, 자기 분류 체계를
    자기 실험에 적용한 사례로 기록한다."""
    ch = _rows(data_dir, "p1_bf16_chol_stress")
    qr = [r for r in _rows(data_dir, "p1_bf16_stress")
          if _base(r["arm"]) == "b_prime_projection_qr"]
    dch = {int(float(r["stage_mode"])): r
           for r in _rows(data_dir, "p1_bf16_chol_span_diagnostics")}
    dqr = {int(float(r["stage_mode"])): r
           for r in _rows(data_dir, "p1_bf16_span_diagnostics")}
    if not ch or not qr:
        return ""
    byq = {int(r["mode"]): r for r in qr}
    out = []
    for r in sorted(ch, key=lambda r: int(r["mode"])):
        m = int(r["mode"])
        q, gc, gq = byq.get(m, {}), dch.get(m, {}), dqr.get(m, {})
        out.append([m, r["p_correct"], r["p_accurate"], f"{r['non_converged']}/{r['n']}",
                    q.get("p_correct"), q.get("p_accurate"),
                    (f"{q['non_converged']}/{q['n']}" if q else None),
                    gc.get("span_finite_frac"), gq.get("qr_orth_defect"),
                    gq.get("n_dropped")])
    body = md.table(["mode", "(b) p_correct", "(b) p_accurate", "(b) non-conv.",
                     "(b′) p_correct", "(b′) p_accurate", "(b′) non-conv.",
                     "(b) span finite", "(b′) ‖I − QᵀWQ‖₂", "(b′) dropped"],
                    out, align="rrrrrrrrrr")
    first = next((r for r in ch if int(r["mode"]) == 1), {})
    nc1, n1 = int(float(first.get("non_converged", 0))), int(float(first.get("n", 0)))
    eps = float(next(iter(dqr.values()), {}).get("eps_machine", float("nan")))
    dead = sorted(m for m, g in dch.items()
                  if g.get("span_finite_frac") not in (None, "")
                  and float(g["span_finite_frac"]) == 0.0)
    drops = max((int(float(g["n_dropped"])) for g in dqr.values()
                 if g.get("n_dropped") not in (None, "")), default=0)
    coll = sorted(int(r["mode"]) for r in ch
                  if float(r["p_correct"]) <= 0.05 and int(r["mode"]) > 1)
    cap = md.caption(
        0,
        "Arm (b) and arm (b′) at bfloat16 on P1, 50 seeds per cell, 4000 iterations, "
        "I0, fixed Gauss — the same configuration as Table 4 with only the working "
        "precision changed. PyTorch provides no `lu`, `cholesky` or `eigh` kernel at "
        "bfloat16 on either device, so arm (b) as defined cannot be executed; the rows "
        "here use an unblocked Cholesky written in elementary tensor operations, which "
        "for k ≤ 20 is the algorithm the library would use anyway and which reproduces "
        "`torch.linalg.solve` to rtol 1e-10 in float64. That removes the library "
        "artifact: the arm runs. It does not make bfloat16 a third point on the "
        "precision axis, and the reason is in the scoring rather than in either arm. "
        f"At mode 1 **no deflation is applied at all**, yet {nc1} of {n1} seeds fail "
        "the pre-specified convergence gate while p_accurate = 1.00 — every seed found "
        "the right mode. The gate asks for a relative change below 1e-3 over the final "
        f"tenth of the iterations and the unit roundoff here is {md.fmt(eps, 2)}, so "
        "the criterion is finer than the arithmetic it judges: this is an "
        "`e_measurement` failure in the sense of §3.0, and no attribution among the "
        "other four components is available at this precision. Both arms then collapse "
        f"together from mode {coll[0] if coll else 3}, so the cells that remain cannot "
        "discriminate between the two rules either. What the diagnostics do separate is "
        "the *manner* of failure, and it is the same distinction the criterion predicts "
        "at float32: the Gram route's stored span becomes non-finite in every seed from "
        f"mode {dead[0] if dead else 7} onward, while the QR route's span stays finite "
        f"in every seed and instead declares the loss, dropping up to {drops} stored "
        "directions as numerically dependent. A rank-aware route degrades by reporting "
        "what it discarded; a normal-equation route degrades by producing nothing. "
        "Because the gate cannot certify any of it, none of these cells is used in a "
        "reported reliability claim.",
        ["p1_bf16_chol_stress.csv", "p1_bf16_stress.csv",
         "p1_bf16_chol_span_diagnostics.csv", "p1_bf16_span_diagnostics.csv"],
        sha(data_dir, "manifest_p1_bf16.json"))
    cap = cap.replace("**Table 0.** ", "") + _seed_note(dch.values(), dqr.values())
    return body + "\n" + cap + "\n"


def appendix_c_numerics(data_dir: str = DATA_DIR) -> str:
    """부록 C — 재현에 필요한 수치 세부. **값은 코드·데이터에서 읽는다.**

    검토 체크리스트가 요구한 항목들이다. 산문으로 흩어 두면 어긋나므로 한 곳에 모으고,
    격자·자유도·회전각 같은 값은 커밋된 CSV와 드라이버에서 가져온다.
    """
    import inspect

    conv = _rows(data_dir, "p4_convergence")
    best = "3.0"
    lv = sorted({(int(float(r["n"])), int(float(r["n_dof"])))
                 for r in conv if str(float(r["beta"])) == best})
    seq = ", ".join(f"n = {n} ({d} dof)" for n, d in lv)

    alpha = "π/5"
    try:                                     # 회전각의 정본은 드라이버다
        from ..drivers import run_p2
        src = inspect.getsource(run_p2.main)
        if "alpha=np.pi / 5" in src:
            alpha = "π/5 (36°)"
    except Exception:
        pass

    bad = [r for r in _rows(data_dir, "p1_conditioning")
           if str(r["basis"]) == "monomial_raw" and not bool(r["cholesky_ok"])]
    bad_n = min((int(r["n_dof"]) for r in bad), default=None)
    bad_row = next((r for r in bad if int(r["n_dof"]) == bad_n), {})

    items = [
        ("P4 reference by Richardson extrapolation",
         f"Graded Q2 meshes with grading exponent β = {best} on the sequence {seq}, each "
         f"level halving the element size, so the refinement ratio is 2. `h` is the "
         f"parameter of the graded map, i.e. the uniform reference spacing 1/n before "
         f"grading, not a local element diameter — grading redistributes elements but does "
         f"not change the ratio between levels. With the three coarsest-to-finest values "
         f"λ₁, λ₂, λ₃ the model is λ_h = λ_* + C·h^p; the observed order is "
         f"p = log₂((λ₁ − λ₂)/(λ₂ − λ₃)) and the reference is "
         f"λ_* = λ₃ − (λ₂ − λ₃)/(2^p − 1). The quoted uncertainty is the relative "
         f"extrapolation step |λ₃ − λ_*|/|λ_*|, i.e. how far the finest computed level "
         f"still sits from the extrapolant; it is not a statistical interval. A level "
         f"sequence that is not monotone in the required sense (d₁/d₂ ≤ 1) returns "
         f"rate = NaN and uncertainty = ∞ rather than a number. Per-level eigenvalues for "
         f"all three gradings are in `p4_convergence.csv`."),
        ("P2 degenerate-pair rotation",
         f"The rotated pair of Table 10 is generated by rotating the plate through "
         f"α = {alpha} about its axis. For a nodal-diameter number m that rotation acts on "
         f"the (cos mθ, sin mθ) pair as a plane rotation by mα, which is how it is applied "
         f"in the script; the angle is a declared input and is not inferred from the "
         f"resulting MAC values."),
        ("Subspace-trace and log-determinant objectives",
         "Both are evaluated from the reduced pencil (K_Φ, M_Φ) with "
         "M_Φ = ΦᵀWΦ and K_Φ = (Φ″)ᵀWΦ″ formed at the quadrature nodes and symmetrized as "
         "½(X + Xᵀ) before use. The absolute trace is "
         "`trace(torch.linalg.solve(M_Φ, K_Φ))`; the log-determinant form is "
         "`slogdet(K_Φ) − slogdet(M_Φ)`, which is why it needs no eigendecomposition and "
         "costs the same. **No jitter, no regularization and no pseudoinverse are used, and "
         "no rank truncation happens inside either objective** — truncation occurs only at "
         "extraction for arm (d). The one guard is on the log form: if either `slogdet` "
         "returns a non-positive sign, meaning the reduced matrix is numerically indefinite, "
         "the objective returns a fixed penalty of 1e6 for that step instead of a "
         "non-finite value, so the optimizer is pushed away from that region rather than "
         "receiving a NaN. The absolute-trace form has no such guard: a singular M_Φ would "
         "propagate whatever `solve` returns."),
        ("Backward error, including rows where the Cholesky column is `no`",
         f"η = ‖Kx − λMx‖₂ / ((‖K‖₂ + |λ|‖M‖₂)‖x‖₂) for the reported eigenpair. The "
         f"`factorization` column reports an **explicit** Cholesky of M, which is used only "
         f"to build the transformed operator M^(−1/2)KM^(−1/2); the eigenpairs come from a "
         f"separate `scipy.linalg.eigh(K, M)` call. The two can disagree, and at raw "
         f"monomials with N = {bad_n} they do: the explicit factorization fails while `eigh` "
         f"still returns, so η is defined there "
         f"({md.fmt(bad_row.get('backward_error'))}) even though the column reads `fail` and "
         f"κ₂ of the transformed operator is left blank. That disagreement is itself the "
         f"measurement — the raw mass matrix is at the edge of fp64 positive definiteness "
         f"(κ₂(M) = {md.fmt(bad_row.get('kappa_M_raw'))}), and the mpmath cross-check at "
         f"dps = 50 differs from fp64 by "
         f"{md.fmt(bad_row.get('Lam_absdiff_rel'))} relative on the lowest eigenvalue."),
        ("Numerical-rank cutoffs",
         "Ranks are reported at relative cutoffs 1e-10 and 1e-12, which are the two "
         "primary conventions used in the text. The 1e-16 column is retained only as a "
         "sensitivity diagnostic at the edge of fp64 resolution and no claim rests on it; a "
         "single integer would hide the choice, which is why more than one appears."),
        ("Repository metadata",
         "The data and code availability record carries the repository DOI, the git SHA of "
         "the commit that produced each artifact (in every manifest), the CPU and GPU "
         "models, library versions, the operating system, and confirmation that every "
         "reported solve runs in fp64. The DOI is minted at submission; the remaining "
         "fields are written by the manifest builder rather than typed."),
    ]
    body = md.table(["item", "specification"],
                    [[k, v] for k, v in items], align="ll")
    return body + "\n"


def appendix_b_disclosure(data_dir: str = DATA_DIR) -> str:
    """부록 B — arm별 구현 공시. 값의 정본은 코드이고 `DISCLOSED`가 그 사본이다."""
    w, dep = DISCLOSED["width"], DISCLOSED["depth"]
    npar_st = DISCLOSED["spacetime_params"]
    rows = [
        ["(a) penalty deflation", f"MLP {w}×{dep}, tanh", "Adam 2e-3, StepLR",
         "4000", "50", "Gauss 256",
         "φ = x²·N(x); objective R(φ) + w·Σ_k ⟨φ,φ_k⟩²/(‖φ‖²‖φ_k‖²), i.e. a sum of "
         "squared cosines — each term is bounded by 1, so the whole penalty is bounded "
         "by w·k while R grows like Λ_m; the penalty-to-objective ratio therefore "
         "scales as w/Λ_m"],
        ["(a′) diagonal-approx. projection", f"MLP {w}×{dep}, tanh", "Adam 2e-3, StepLR",
         "4000", "50", "Gauss 256", "cₖ = ⟨φ,φₖ⟩/⟨φₖ,φₖ⟩"],
        ["(b) exact M-orthogonal projection", f"MLP {w}×{dep}, tanh",
         "Adam 2e-3, StepLR", "4000", "50", "Gauss 256",
         "φ̃ = φ − Φ(ΦᵀWΦ)⁻¹ΦᵀWφ"],
        ["(b′) QR-orthonormalized projection", f"MLP {w}×{dep}, tanh",
         "Adam 2e-3, StepLR", "4000", "50", "Gauss 256",
         "φ̃ = φ − Q(QᵀWφ) with Q from modified Gram–Schmidt on the stored raw "
         "outputs in the mass inner product; the same coefficients are applied to the "
         "stored second derivatives, so they are constants with respect to the spatial "
         "derivative as in (b). The Gram is never formed. A stored direction whose norm "
         "after orthogonalization falls below √u times its norm before it is dropped "
         "rather than inverted, u the unit roundoff of the working precision; the number "
         "of dropped directions is logged per stage. Added 2026-08-26, scored by the "
         "unchanged pre-specified rules"],
        ["(b) and (b′) at reduced precision", f"MLP {w}×{dep}, tanh",
         "Adam 2e-3, StepLR", "4000", "50", "Gauss 256",
         "the two arms above, re-run with the working precision set to **float32** and "
         "nothing else changed; the Gauss nodes and weights are built in float64 and "
         "cast down so the quadrature rule is not part of the precision variable. "
         "Added 2026-08-26"],
        ["(c) coarse-Ritz curriculum", f"MLP {w}×{dep}, tanh", "Adam 2e-3, StepLR",
         "4000", "50", "Gauss 256", "stage-1 fraction 0.5, then (b)"],
        ["(d) separate-net neural basis", f"9 × MLP {w}×{dep}, tanh",
         "Adam 2e-3, StepLR", "4000", "50", "Gauss 256",
         "subspace trace; reduced GEP with rank truncation 1e-12"],
        ["(e) shared-trunk subspace", f"MLP {w}×{dep} with 6 outputs, tanh",
         "Adam 2e-3, StepLR", "4000", "50", "Gauss 256", "subspace trace"],
        ["(f) Eig-PIELM", "random tanh features, fixed", "none (no backprop)",
         "—", "50", "Gauss 512",
         "φⱼ = x²·tanh(aⱼx+bⱼ); rank-truncated GEP"],
        ["(g) space–time PINN",
         f"MLP {DISCLOSED['spacetime_width']}×{DISCLOSED['spacetime_depth']} "
         f"on (x,t), tanh", f"Adam {ST['lr']:g}, StepLR",
         f"{int(ST['iters'])}", "3",
         f"{int(ST['n_col'])} random collocation points per iteration, resampled "
         f"every iteration",
         # **자유단 조건이 손실에 실제로 들어간다.** 예전 공시는 `w = x²·N`만 적어
         # "자연경계조건을 강형식에서 부과하지 않았다"로 읽힐 수 있었다. 코드는
         # `spacetime.py`에서 x = 1의 w_xx·w_xxx 잔차를 벌점으로 넣는다.
         f"strong form `w_tt + w_xxxx = 0` on (0,1) × (0,T); trial field "
         f"w = x²·N(x,t), which imposes the clamped end `w(0,t) = w_x(0,t) = 0` "
         f"exactly; the free end `w_xx(1,t) = w_xxx(1,t) = 0` is imposed by penalty "
         f"on {int(ST['n_bc'])} times drawn once from U(0,T) and held fixed; initial "
         f"state is the analytic clamped–free mode 1 at rest, `w(x,0) = φ₁(x)` and "
         f"`w_t(x,0) = 0`, on {int(ST['n_ic'])} uniform x; loss = PDE residual "
         f"(weight 1) + {ST['w_ic']:g} · initial-state residual + {ST['w_bc']:g} · "
         f"free-end residual, weights chosen so the terms are of comparable size and "
         f"not tuned; the eigenvalue is read afterwards from the tip response at "
         f"x = 1 sampled at {int(ST['n_probe'])} uniform times over [0,T] "
         f"(Δt = T/{int(ST['n_probe'])}; the window endpoint is excluded so the record "
         f"length is exactly T and the reference frequency falls on a bin), mean removed, "
         f"**no taper or window**, "
         f"peak taken as the largest |rFFT| bin excluding DC with no interpolation; "
         f"{npar_st} parameters"],
        ["(a′) on P4, two dimensions",
         f"MLP {DISCLOSED['p4_width']}×{DISCLOSED['p4_depth']} on (x,y) with two "
         f"outputs, tanh", "Adam 2e-3, StepLR", "4000", "50",
         f"tensor Gauss, {DISCLOSED['p4_n_per_block']}² per unit block "
         f"({DISCLOSED['p4_n_q']} points)",
         "the diagonal rule cₖ = ⟨u,uₖ⟩/⟨uₖ,uₖ⟩ in the two-component mass inner "
         "product, coefficients applied to the field and its gradients; **no jitter** — "
         "nothing is inverted. Modes 1–3 only. Added 2026-08-26"],
        ["(b′) on P4, two dimensions",
         f"MLP {DISCLOSED['p4_width']}×{DISCLOSED['p4_depth']} on (x,y) with two "
         f"outputs, tanh", "Adam 2e-3, StepLR", "4000", "50",
         f"tensor Gauss, {DISCLOSED['p4_n_per_block']}² per unit block "
         f"({DISCLOSED['p4_n_q']} points)",
         "mass-inner-product Gram–Schmidt on the stored states, the same coefficients "
         "applied to the field and its gradients; **no jitter** — the Gram is never "
         "formed. Modes 1–3 only. Added 2026-08-26"],
        ["(b) on P4, two dimensions",
         f"MLP {DISCLOSED['p4_width']}×{DISCLOSED['p4_depth']} on (x,y) with two "
         f"outputs, tanh", "Adam 2e-3, StepLR", "4000", "50",
         f"tensor Gauss, {DISCLOSED['p4_n_per_block']}² per unit block "
         f"({DISCLOSED['p4_n_q']} points)",
         f"u = φ_Ω(x,y)·N(x,y) with φ_Ω the R-equivalence approximate distance "
         f"function of Sukumar and Srivastava; {DISCLOSED['p4_params']} parameters; exact "
         f"mass-orthogonal projection applied to the field and its gradients, with "
         f"**1e-14 times the identity added to the Gram before solving** (this "
         f"stabilization is present in the committed arm-(b) run and is absent from the "
         f"(a′) and (b′) rows above, which invert nothing); "
         f"per-mode wall-clock measured directly at a single seed as the median of "
         f"three runs, the same basis as the canonical ratio table"]]
    body = md.table(["arm", "architecture", "optimizer", "iterations", "seeds",
                     "quadrature", "notes"], rows, align="lllrrll")
    note = ("\nArms (a), (a\u2032), (b) and (c) share architecture, optimizer, schedule, "
            "quadrature and seed derivation, so any difference among those four is "
            "attributable to the one component named in the notes column — they are the "
            "paired ablations of \u00a73.3. Arms (d) to (g) deliberately do not: (d) trains "
            "nine separate networks, (e) a single trunk with six outputs, (f) draws random "
            "features and does not backpropagate at all, and (g) trains on a space\u2013time "
            "domain. Those four are read against (b) as separate designs, not as "
            "one-component ablations. Clamped "
            "boundary conditions are satisfied exactly by construction, not by penalty: "
            "an x^2 factor on the one-dimensional problems, and the R-equivalence "
            "distance function of Sukumar and Srivastava (\u00a73.2) on the L-shaped "
            "domain, where the domain is "
            "non-convex and a product of edge functions would vanish inside it.\n")
    return body + note


