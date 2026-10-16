"""표 렌더러 — 값이 CSV와 일치하는지, 그리고 숨김이 없는지."""
import math

import pytest

from eigen_benchmark.drivers import manifest
from eigen_benchmark.render import tables
from ._paper import needs_paper

DATA = "docs/_generated/data/paper2"


def csv(name):
    return manifest.read_csv(f"{DATA}/{name}.csv")


# ------------------------------------------------------------------ 표 1
def test_table1_counts_match_the_csv_exactly():
    """렌더러는 계산하지 않는다 — 표의 카운트가 CSV에 그대로 있어야 한다."""
    s = tables.table1_attribution(DATA)
    for r in csv("p1_neural_mode_sweep"):
        if int(r["mode"]) != 3:
            continue
        assert f"{r['correct']}/{r['n']}" in s, (r["arm"], r["correct"])


def test_table1_denominator_is_fifty_not_thirty():
    s = tables.table1_attribution(DATA)
    assert "/30" not in s and "/50" in s


def test_table1_has_no_unfilled_placeholder():
    s = tables.table1_attribution(DATA)
    assert "[ ]" not in s and "TO RUN" not in s


def test_table1_reports_the_four_preregistered_categories():
    s = tables.table1_attribution(DATA).lower()
    for o in ("correct", "lower-mode basin", "spurious", "non-converged"):
        assert o in s


def test_table1_marks_p_accurate_as_post_hoc():
    """사전등록되지 않은 지표를 그렇게 표시하지 않으면 사전등록의 의미가 없다."""
    assert "post-hoc" in tables.table1_attribution(DATA).lower()


# ------------------------------------------------------------------ 표 2–5
def test_table2_penalty_spans_four_decades():
    s = tables.table2_penalty(DATA)
    ws = sorted({float(r["penalty_weight"]) for r in csv("p1_neural_penalty_sweep")})
    assert len(ws) >= 4
    for w in ws:
        assert str(int(math.log10(w))) in s or f"{w:.2e}"[:4] in s


def test_table3_contrasts_correct_against_accurate_and_quantifies_the_noise():
    """MC의 논점은 p_correct=0인데 p_accurate=1이라는 불일치이고,
    그 이유(잡음이 임계의 86배)가 실측으로 붙어야 한다."""
    s = tables.table3_quadrature(DATA)
    assert "p_correct" in s and "p_accurate" in s
    assert any(r["nodes"] == "mc" for r in csv("p1_neural_quadrature"))
    assert "tolerance" in s.lower() and "86" in s


def test_table4_shows_both_the_prefit_cost_and_the_counterfactual():
    """사전적합을 청구하지 않으면 oracle이 최고로 보인다 — 두 값을 다 보여야 한다."""
    s = tables.table4_warm_cold(DATA)
    assert "pre-fit" in s and "if pre-fit were free" in s
    assert "200" in s and "4000" in s


def test_table5_longrun_states_both_budgets():
    s = tables.table5_longrun(DATA)
    assert "4000" in s and "20000" in s


# ------------------------------------------------------------------ 표 6–7
def test_table6_splits_three_families_so_gradient_arms_stay_visible():
    s = tables.table6_cost(DATA)
    for f in tables.FAMILY_LABEL.values():
        assert f in s


def test_table6_keeps_unreached_rows_instead_of_dropping_them():
    s = tables.table6_cost(DATA)
    if any(not bool(r["reached"]) for r in csv("p1_cheapest_to_epsilon")):
        assert "not reached" in s


def test_table6_declares_the_timing_accounting():
    """앙상블 시간과 1회 시간을 섞으면 비교가 무너진다 — 회계를 캡션에 밝힌다."""
    s = tables.table6_cost(DATA).lower()
    assert "2.53" in s and "ensemble" in s and "cpu" in s


def test_table7_covers_all_four_problems_with_uncertainty():
    s = tables.table7_references(DATA)
    for p in ("P1", "P2", "P3", "P4"):
        assert p in s
    assert "uncertainty" in s.lower()
    assert "degenerate" in s.lower()


def test_table7_carries_no_absolute_units():
    """논문 경계: 참조표에 Hz·mm·GPa가 재유입되면 안 된다."""
    s = tables.table7_references(DATA).lower()
    for u in (" hz", " mm", " gpa", "kg/m"):
        assert u not in s


# ------------------------------------------------------------------ 표 8–10
def test_table8_reports_all_four_normalizations_and_flags_failures():
    s = tables.table8_conditioning(DATA)
    for c in ("raw", "mass-norm", "equilibrated", "transformed"):
        assert c in s.lower()
    if any(not bool(r["cholesky_ok"]) for r in csv("p1_conditioning")):
        assert "fail" in s.lower()


def test_table9_shows_three_families_and_tolerance_dependence():
    s = tables.table9_effective_rank(DATA)
    for fam in ("monomial", "random features", "neural basis"):
        assert fam.lower() in s.lower()
    assert "1e-16" in s and "tolerance" in s.lower()


def test_table9_names_a_different_mechanism_for_each_family():
    """증상이 같다고 기전이 같은 것은 아니다 — 표가 그것을 구분해야 한다."""
    s = tables.table9_effective_rank(DATA).lower()
    for m in ("coordinate", "draw", "optimization"):
        assert m in s


def test_table10_shows_the_residual_error_anticorrelation():
    """잔차는 내려가고 목표 오차는 올라간다 — 그리고 4주기 항은 해상도로 읽어야 한다.

    옛 버전은 "113×"를 실었는데 그 분모(4주기 e_λ = 3.9e-3)가 추정기 편향이었다. 표본추출을
    고친 뒤 4주기는 기계영이고, 그 값은 정확함이 아니라 **반 빈 이하**라는 상한이다."""
    s = tables.table10_spacetime(DATA)
    assert "residual" in s.lower()
    assert "8.2×" in s
    assert "113" not in s, "폐기한 비율이 남았다"
    assert "below the resolution limit to 7 bins" in s
    assert "below half a bin" in s
    assert "not an error floor" not in s.lower(), \
        "이제 빈은 검출 해상도로 설명한다 — 오차 하한이 아니라는 부정형만으로는 부족하다"
    rows = csv("p1_spacetime")
    import statistics as _st
    med = lambda T, k: _st.median(  # noqa: E731
        [float(r[k]) for r in rows if float(r["n_periods"]) == T])
    assert med(4.0, "e_lam") < 1e-12, "4주기는 참조 빈에 정확히 떨어져야 한다"
    e = [med(T, "e_lam") for T in (4.0, 8.0, 16.0, 32.0)]
    assert e == sorted(e), e
    assert med(32.0, "e_lam_over_bin") > 1.0, "긴 창의 오차는 해상도를 넘어야 한다"


# ------------------------------------------------------------------ 부록
def test_appendix_a_is_read_from_code_not_typed():
    from eigen_benchmark.problems.p2_annulus import P2_GEOMETRY
    from eigen_benchmark.render import md
    a = tables.appendix_a_nondimensional()
    assert md.fmt(P2_GEOMETRY["a"] / P2_GEOMETRY["b"]) in a
    assert md.fmt(P2_GEOMETRY["nu"]) in a


def test_appendix_a_carries_no_absolute_units():
    a = tables.appendix_a_nondimensional().lower()
    for u in (" mm", " gpa", " kg/m", " hz", "193", "7930", "36.56", "15.4"):
        assert u not in a, u


def test_appendix_b_discloses_every_arm_and_the_parameter_count():
    b = tables.appendix_b_disclosure(DATA)
    for tag in ("(a)", "(a′)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"):
        assert tag in b
    assert "parameters" in b.lower()
    assert "seeds" in b.lower()


def test_table11_reports_the_controlled_objective_swap():
    """가중만 바꾼 대조 — 개선폭이 모드 순서에 단조여야 가중 가설이 지지된다."""
    s_ = tables.table11_objective_weighting(DATA)
    assert "Σlog λ" in s_ and "Σλ" in s_
    assert "attribution experiment" in s_
    assert "not proposed here as a contribution" in s_
    rows = csv("p1_e_objective_ablation")
    tr = {int(r["mode"]): float(r["e_lam_median"]) for r in rows if r["objective"] == "trace"}
    lg = {int(r["mode"]): float(r["e_lam_median"]) for r in rows if r["objective"] == "logtrace"}
    gain = [tr[m] / lg[m] for m in sorted(tr)]
    assert gain == sorted(gain, reverse=True), gain      # 모드1에서 최대, 단조 감소


def test_table11_shows_the_inversion_is_removed():
    """절대 trace에서는 e_λ가 모드와 함께 감소(역전), 로그에서는 증가(정상)."""
    rows = csv("p1_e_objective_ablation")
    tr = [float(r["e_lam_median"]) for r in sorted(
        (r for r in rows if r["objective"] == "trace"), key=lambda r: int(r["mode"]))]
    lg = [float(r["e_lam_median"]) for r in sorted(
        (r for r in rows if r["objective"] == "logtrace"), key=lambda r: int(r["mode"]))]
    assert tr[0] > tr[-1], "절대 trace의 역전이 데이터에 없다"
    assert lg[0] < lg[-1], "로그 목적함수에서 프로파일이 정상화되지 않았다"


def test_table11_curvature_concentration_survives_the_fix():
    """H2/L2가 두 목적함수 모두에서 모드 순서에 단조 감소 — 공유표현의 성질."""
    rows = csv("p1_e_objective_ablation")
    for obj in ("trace", "logtrace"):
        v = [float(r["h2_over_l2"]) for r in sorted(
            (r for r in rows if r["objective"] == obj), key=lambda r: int(r["mode"]))]
        assert v[0] > v[-1] * 5, (obj, v)


def test_table_p4_reports_the_three_rules_with_the_same_scoring():
    """P4는 P1과 같은 사전등록 4분류로 채점되고, **세 규칙이 구분되지 않는다는 사실**과
    그 함의를 캡션이 말해야 한다. 없앨 것이 없던 자리에서 "정확 사영이 없앴다"고 읽히면
    P1의 기전 주장이 근거 없이 2차원으로 확장된다."""
    s_ = tables.table_p4(DATA)
    for c in ("correct", "lower-mode basin", "spurious", "non-converged",
              "95 % Wilson", "p_accurate"):
        assert c in s_, c
    assert "not distinguishable" in s_
    assert "lower-mode-basin outcome under any" in s_
    assert "cannot be read as evidence" in s_
    assert "1e-14" in s_          # jitter 비대칭 공시


def test_table_p4_counts_match_the_csv():
    s_ = tables.table_p4(DATA)
    for r in csv("p4_neural"):
        assert f"{r['correct']}/{r['n']}" in s_, (r["mode"], r["correct"])


def test_table_p4_caption_reports_the_sub_lambda1_cells_from_the_records():
    """Λ < Λ₁은 정확 적분에서 불가능하다 — 그 셀 수를 렌더러가 **기록에서 세어** 캡션에
    실어야 한다. 숫자를 캡션에 손으로 박으면 데이터가 바뀔 때 조용히 틀어진다."""
    import json
    recs = []
    for f in ("p4_neural_records", "p4_neural_arms_records"):
        with open(f"{DATA}/{f}.jsonl", encoding="utf-8") as fh:
            recs += [json.loads(l) for l in fh]
    lam1 = next(r["Lam_ref"] for r in recs if r["mode"] == 1)
    viol = [r for r in recs if r["Lam"] < lam1]
    assert viol, "이 테스트는 위반 셀이 있는 데이터를 전제한다"
    s_ = tables.table_p4(DATA)
    assert f"{len(viol)} of the {len(recs)}" in s_, s_[-1200:]
    # 주장에 들어간 셀이 없다는 사실이 캡션의 요점이다
    assert all(r["outcome"] == "non_converged" for r in viol)
    assert "none of them enters a scored claim" in s_


def test_p4_deflation_result_transfers_to_two_dimensions():
    """논문이 §5.4에서 '하위모드 basin이 하나도 없다'고 쓰는 근거 — 데이터로 확인."""
    rows = csv("p4_neural")
    assert rows
    assert all(int(r["lower_mode_basin"]) == 0 for r in rows), \
        [(r["mode"], r["lower_mode_basin"]) for r in rows]


def test_p4_accuracy_plateau_is_above_the_reference_uncertainty():
    """e_λ가 참조 불확실도보다 커야 '솔버 오차'라 말할 수 있다."""
    unc = {int(r["mode"]): float(r["uncertainty_rel"]) for r in csv("p4_reference")}
    for r in csv("p4_neural"):
        m = int(r["mode"])
        assert float(r["e_lam_median"]) > unc[m], (m, r["e_lam_median"], unc[m])


def test_orthogonality_table_shows_where_the_two_arms_diverge():
    """(a′)와 (b)는 저장된 모드 그람의 비대각을 통해서만 다르다 — k=2에서 두 arm이
    동일하고 그 지점의 사영 연산자 차이가 기전의 정량이다."""
    s_ = tables.table_orthogonality(DATA)
    # 계수사상의 거리와 사영 연산자의 거리를 **따로** 실어야 한다 — 전자를
    # "사영 차이의 상한"이라 쓰면 1e8이 O(1)인 양을 과장하게 된다.
    assert "inverse-Gram discrepancy" in s_
    # 사영은 Π_G = ΦG⁻¹ΦᵀW다 — Φ와 Φᵀ가 뒤바뀐 식이 한때 실려 차원이 맞지 않았다
    assert "‖Π_G − A_D‖_W" in s_
    assert "`Π_G − A_D = Φ(G⁻¹ − D⁻¹)ΦᵀW`" in s_
    assert "A_D is not a projection" in s_
    assert "Φᵀ(G⁻¹ − D⁻¹)ΦW" not in s_
    assert "‖I − C‖₂" in s_ and "C = D^{-1/2}GD^{-1/2}" in s_
    assert "operator discrepancy" not in s_
    assert "both" in s_ and "0.99" in s_
    # 전체 그람을 역행렬로 쓰는 방법이므로 그 조건도 함께 보고한다
    assert "κ₂(G)" in s_ and "λ_min(C)" in s_ and "torch.linalg.solve" in s_
    rows = csv("p1_orthogonality")
    k2 = {r["arm"]: r for r in rows if int(r["n_prev_modes"]) == 2
          and r.get("gram_dev_fro_median") not in (None, "")}
    assert len(k2) == 2
    a, b = (k2[k] for k in sorted(k2))
    # 모드 2까지 두 arm은 비트 단위 동일하므로 k=2 행이 같아야 한다
    for key in ("inv_gram_rel_diff_median", "proj_op_diff_W_median",
                "kappa_gram_median"):
        assert float(a[key]) == pytest.approx(float(b[key]), rel=1e-9), key


def test_mode3_leakage_closes_the_causal_chain():
    """사영 후에도 남는 겹침 — (b)는 기계영, (a′)는 O(1)이어야 기전 주장이 닫힌다."""
    rows = [r for r in csv("p1_orthogonality")
            if r.get("mode3_leak_rel_median") not in (None, "")]
    assert len(rows) == 2, "두 arm 각각 한 행이어야 한다"
    g = {r["arm"]: float(r["mode3_leak_rel_median"]) for r in rows}
    ap = next(v for a, v in g.items() if a.startswith("a_prime"))
    bx = next(v for a, v in g.items() if a.startswith("b_"))
    assert bx < 1e-10, f"정확 사영은 저장 span을 기계정밀도로 제거해야 한다: {bx}"
    assert ap > 0.1, f"대각근사에는 O(1) 잔여가 남아야 한다: {ap}"
    s_ = tables.table_orthogonality(DATA)
    assert "mode-3 residual leakage" in s_
    # 중심 증거이므로 정의식이 캡션에 있어야 재현 가능하다
    assert "‖D^{-1/2}ΦᵀWφ̃‖₂ / ‖φ̃‖_W" in s_ and "√2" in s_


def test_projection_operator_difference_is_not_the_inverse_gram_number():
    """실측 ‖P_G − P_D‖_W는 O(1)이고 역그람 불일치는 1e8까지 간다 — 서로 다른 양이다."""
    rows = [r for r in csv("p1_orthogonality")
            if r.get("proj_op_diff_W_median") not in (None, "")]
    for r in rows:
        pd = float(r["proj_op_diff_W_median"])
        k = int(r["n_prev_modes"])
        # ‖I − G^{1/2}D⁻¹G^{1/2}‖₂ = max|1 − λ(C)| ≤ k − 1 (C는 단위대각 PSD)
        assert 0 < pd <= k - 1 + 1e-9, (k, pd)
        if float(r["inv_gram_rel_diff_median"]) > 1e6:
            assert pd < 10, "역그람 수치가 커도 사영 차이는 O(1)이다"


def test_orthogonality_diverges_from_k_equals_three():
    rows = csv("p1_orthogonality")
    for k in (3, 4, 5, 6):
        g = {r["arm"]: float(r["inv_gram_rel_diff_median"])
             for r in rows if int(r["n_prev_modes"]) == k
             and r.get("inv_gram_rel_diff_median") not in (None, "")}
        ap = next(v for a, v in g.items() if a.startswith("a_prime"))
        bx = next(v for a, v in g.items() if a.startswith("b_"))
        assert ap > 100 * bx, (k, ap, bx)


def test_penalty_caption_reports_that_a_large_enough_weight_works():
    """이전 초안은 w=1e3에서 멈추고 '결코 회복하지 못한다'고 결론했다 — 틀렸다."""
    s_ = tables.table2_penalty(DATA)
    assert "does recover mode 3" in s_
    rows = {(float(r["penalty_weight"]), int(r["mode"])): float(r["p_correct"])
            for r in csv("p1_neural_penalty_sweep")}
    assert rows[(1e4, 3)] > 0.9
    assert rows[(1e3, 3)] == 0.0
    assert rows[(1e5, 2)] < 0.5          # 위쪽에서 창이 닫힌다


def test_cost_table_does_not_call_an_iteration_budget_a_trial_size():
    """`size` 한 열에 n_dof와 반복예산 4000이 섞여 있었다 — PIELM은 반복이 아예 없다."""
    rows = csv("p1_cheapest_to_epsilon")
    for r in rows:
        if not str(r["reached"]).lower().startswith("t"):
            continue
        lab, sz, it = r["size_label"], str(r["size"]), str(r.get("iterations", ""))
        if r["family"] == "classical":
            assert lab == "n_dof" and it == "", (lab, it)
        elif r["family"] == "neural_randfeat":
            assert lab == "random features", lab
            assert it == "", "Eig-PIELM은 backprop이 없으므로 반복예산이 없다"
            # 무작위 특징 수는 arm 라벨의 nf와 일치해야 한다(격자에 20·40이 있다)
            assert sz == r["solver"].split("nf=")[-1].rstrip(")"), (sz, r["solver"])
        else:
            assert lab == "MLP width x depth" and sz == "64x4", (lab, sz)
            assert int(float(it)) == 4000
    s_ = tables.table6_cost(DATA)
    assert "trial size" in s_ and "| iterations" in s_
    assert "accuracy statistics use the 50-seed ensembles" in s_
    assert "timing uses a single seed" in s_


def test_headline_table_carries_both_accounting_bases_as_columns():
    """E[T_success]가 주 비용지표라고 선언했으므로 캡션 산문에 숨겨선 안 된다."""
    s_ = tables.table_headline(DATA)
    assert "per attempt, orders" in s_ and "expected success, orders" in s_
    rows = {r["quantity"]: r for r in csv("p1_cost_headline")}
    p1 = rows["gradient-trained vs classical"]
    # P1 모드1은 양쪽 p_correct가 1.00이므로 두 기준이 일치해야 한다
    assert abs(float(p1["cost_orders"])
               - float(p1["expected_success_orders"])) < 1e-6
    p4 = rows["gradient-trained vs classical (P4, mode 1)"]
    # P4는 신경 쪽 p_correct < 1이므로 기대성공 격차가 더 커야 한다
    assert float(p4["expected_success_orders"]) > float(p4["cost_orders"])


def test_rank_table_does_not_call_the_shared_trunk_a_collapse():
    """(e)는 수치 rank 6/6을 유지한다 — 붕괴는 (d)뿐이다."""
    s_ = tables.table9_effective_rank(DATA)
    assert "high pairwise collinearity, full numerical rank" in s_
    assert "optimization-induced collapse to rank" in s_
    rows = [r for r in csv("p1_neural_collinearity") if float(r["w_orth"]) == 0.0]
    e = [r for r in rows if "simultaneous" in r["arm"]]
    assert e and float(e[0]["rank_1e12"]) == float(e[0]["n_basis"])


def test_appendix_b_scopes_the_shared_architecture_claim():
    """(d)-(g)는 구조가 서로 다르므로 '모든 gradient arm이 동일'은 거짓이다."""
    s_ = tables.appendix_b_disclosure(DATA)
    assert "All gradient-trained arms share architecture" not in s_
    assert "nine separate networks" in s_ and "six outputs" in s_
    # P4 행의 시간 회계가 정본표와 같아야 한다
    assert "50-seed ensemble" not in s_
    assert "median of three runs" in s_


def test_gram_conditioning_is_reported_up_to_the_depth_the_claim_needs():
    """"모드 10까지 벽이 없다"면 정작 중요한 그람은 k = 9다.

    k = 6까지만 보고하면 검토자가 곧바로 "k = 7, 8, 9는?"을 묻는다. 그 답이 초록의
    "남은 한계는 최적화 예산" 주장과 대수적 조건을 분리할 수 있는지를 결정한다."""
    rows = [r for r in csv("p1_orthogonality")
            if r.get("kappa_gram_median") not in (None, "")]
    ks = {(r["arm"], int(r["n_prev_modes"])) for r in rows}
    for k in range(2, 11):
        assert ("b_projection_exact", k) in ks, f"정확 사영 arm의 k={k}가 없다"
    ex = {int(r["n_prev_modes"]): r for r in rows
          if r["arm"] == "b_projection_exact"}
    # κ₂는 k에 대해 단조증가해야 한다(모드를 더 저장하면 그람이 더 나빠진다)
    kap = [float(ex[k]["kappa_gram_median"]) for k in sorted(ex)]
    assert kap == sorted(kap), kap
    # k=9는 실제로 모드 10 풀이에 넘겨지는 그람 — 특이하지 않아야 결과가 유효하다
    r9 = ex[9]
    assert int(float(r9["eff_rank_1e12_median"])) == 9
    assert int(float(r9["n_seeds_gram_singular"])) == 0
    assert float(r9["lam_min_normalized_median"]) > 0
    s_ = tables.table_orthogonality(DATA)
    assert "at k = 9" in s_, "캡션이 실제로 쓰이는 최대 깊이를 지목해야 한다"
    assert "mode-11 solve" not in s_, "k=10 그람은 어느 풀이에도 넘겨지지 않는다"
    assert "decimal digits in a worst-case conditioning estimate" in s_


@needs_paper
def test_paper_does_not_claim_comfortable_conditioning_at_mode_ten():
    """κ₂ = 1.6e11은 fp64에서 약 5자리다 — '여유가 충분하다'고 쓸 수 없다."""
    import pathlib
    t = (pathlib.Path(__file__).resolve().parents[2]
         / "docs/paper2-cs/paperA_CS_benchmark.md").read_text(encoding="utf-8")
    assert "1.6e11" in t, "k=9 조건수를 본문이 밝혀야 한다"
    # 최악 추정임을 밝혀야 한다 — κu ≈ 3e-5는 4~5자리다
    flat = " ".join(t.split())
    assert "four to five decimal digits in a worst-case" in flat
    assert "about five significant digits" not in flat and "about five fp64" not in flat
    assert "makes no\nclaim past it" in t or "makes no claim past it" in t


def test_rank_table_caption_does_not_contradict_its_own_rows():
    """캡션이 "모든 계열이 usable dimension을 과대표시한다"고 하면서 바로 아래 행에서
    (e)가 full numerical rank라고 적으면 자기모순이다."""
    s_ = tables.table9_effective_rank(DATA)
    assert "Every family overstates its usable dimension" not in s_
    assert "three distinct mechanisms by which a nominal dimension can overstate" in s_
    assert "does not cost every arm its rank" in s_
    # 실제로 rank를 유지하는 arm이 존재해야 이 문장이 필요하다
    rows = [r for r in csv("p1_neural_collinearity") if float(r["w_orth"]) == 0.0]
    assert any(float(r["rank_1e12"]) == float(r["n_basis"]) for r in rows)
    assert any(float(r["rank_1e12"]) < float(r["n_basis"]) for r in rows)


@needs_paper
def test_p4_domain_and_eigenvalue_do_not_share_a_symbol():
    """§3.1이 영역을 Ω로 정의하므로 고유값에 Ω²를 쓰면 같은 기호가 두 뜻을 갖는다."""
    import pathlib
    t = (pathlib.Path(__file__).resolve().parents[2]
         / "docs/paper2-cs/paperA_CS_benchmark.md").read_text(encoding="utf-8")
    assert "Ω²" not in t, "고유값 기호가 영역 기호와 겹친다"
    assert "Λ_el = ω²ρL²/E" in t
    assert "Ω denotes the domain" in t
    for s_ in (tables.table7_references(DATA), tables.table_p4(DATA),
               tables.appendix_a_nondimensional()):
        assert "Ω²" not in s_


def test_appendix_c_matches_the_implementations_it_describes():
    """부록 C는 재현용 공시다 — 코드와 어긋나면 그 자체가 결함이다."""
    import inspect

    # 표 칸의 파이프는 이스케이프되므로 벗겨 비교한다
    s_ = tables.appendix_c_numerics(DATA).replace("\\|", "|")
    # 목적함수 구현과의 대조는 torch가 필요하므로 GPU 스위트에 있다
    # (`test_neural_subspace.py::test_appendix_c_objective_claims_match_the_code`).
    assert "fixed penalty of 1e6" in s_
    assert "torch.linalg.solve(M_Φ, K_Φ)" in s_
    assert "No jitter, no regularization and no pseudoinverse are used" in s_

    # 후향오차 공식은 conditioning의 독스트링과 같은 식이어야 한다
    from eigen_benchmark import conditioning
    assert "‖Kx − λMx‖" in inspect.getdoc(conditioning.generalized_backward_error)
    assert "η = ‖Kx − λMx‖₂ / ((‖K‖₂ + |λ|‖M‖₂)‖x‖₂)" in s_

    # Richardson: 격자열과 자유도를 데이터에서 읽는가
    conv = csv("p4_convergence")
    for n, dof in {(int(float(r["n"])), int(float(r["n_dof"]))) for r in conv
                   if float(r["beta"]) == 3.0}:
        assert f"n = {n} ({dof} dof)" in s_, (n, dof)
    assert "log₂((λ₁ − λ₂)/(λ₂ − λ₃))" in s_
    assert "rate = NaN and uncertainty = ∞" in s_

    # P2 회전각은 드라이버에서 읽는가
    from eigen_benchmark.drivers import run_p2
    assert "alpha=np.pi / 5" in inspect.getsource(run_p2.main)
    assert "α = π/5 (36°)" in s_


def test_availability_record_is_rendered_not_typed():
    """가용성 절의 환경 값은 `environment_*.json`에서 와야 한다.

    손으로 쓰면 어긋난다 — 실제로 호스트 venv가 작업 중에 numpy를 올려서 산출물이 두
    버전 조합으로 나뉘었다. 그 사실을 뭉개지 않고 세어 적는지까지 확인한다."""
    import glob
    import json
    import os

    s_ = tables.availability_record(DATA).replace("\\|", "|")
    for tag in ("host", "gpu"):
        p_ = os.path.join(DATA, f"environment_{tag}.json")
        assert os.path.exists(p_), f"{p_}가 없다 — run_environment를 돌려라"
        with open(p_, encoding="utf-8") as f:
            d = json.load(f)
        for k in ("cpu", "os", "kernel", "python", "numpy", "scipy", "fp64"):
            assert k in d, (tag, k)
    with open(os.path.join(DATA, "environment_gpu.json"), encoding="utf-8") as f:
        g = json.load(f)
    assert g["torch"] in s_ and g["torch_cuda"] in s_
    with open(os.path.join(DATA, "environment_host.json"), encoding="utf-8") as f:
        h = json.load(f)
    assert h["cpu"] in s_ and h["os"] in s_ and h["kernel"] in s_
    # DOI와 최종 SHA는 제출 시점 값이므로 비워 둔다
    assert "minted at submission" in s_
    assert "recorded at submission" in s_
    # 산출물을 만든 버전 조합을 manifest에서 세어 적는가
    combos = set()
    for f_ in glob.glob(os.path.join(DATA, "manifest*.json")):
        with open(f_, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("numpy"):
            combos.add((d["numpy"], d["scipy"]))
    assert len(combos) >= 1
    for npv, spv in combos:
        assert f"NumPy {npv}, SciPy {spv}" in s_, (npv, spv)
    if len(combos) > 1:
        assert "the host environment was upgraded during the work" in s_
    # fp64는 환경 기본값이 아니라 솔버가 설정한다는 사실을 흐리지 않는다
    assert "process default is float32" in s_


# ------------------------------------------------- (b′) QR arm · fp32 (2026-08-26)
def _sweep_row(arm, mode, p=1.0, secs=300.0, correct=50):
    return {"arm": arm, "mode": mode, "ladder": "I0", "nodes": "gauss", "n": 50,
            "p_correct": p, "wilson_lo": max(p - 0.1, 0.0),
            "wilson_hi": min(p + 0.05, 1.0), "correct": correct,
            "lower_mode_basin": 50 - correct, "spurious": 0, "non_converged": 0,
            "p_accurate": p, "n_certified": 50, "iters": 4000, "seconds": secs}


def _diag_row(arm, stage, k, kappa, defect, dtype="float64", agree=1e-14,
              dropped=0, ranks=(1, 1, 1)):
    return {"arm": arm, "dtype": dtype, "stage_mode": stage, "n_seeds": 50,
            "iters": 4000, "k": k, "kappa_gram": kappa,
            "kappa_gram_times_eps": kappa * 2.22e-16, "dev_identity": 3.0,
            "lam_min_normalized": 1e-3, "qr_orth_defect": defect,
            "n_dropped": dropped, "rank_1e10": ranks[0], "rank_1e12": ranks[1],
            "rank_1e16": ranks[2], "proj_agreement_rel": agree,
            "gram_solve_finite_frac": 1.0, "eps_machine": 2.22e-16,
            "n_seeds_summarized": 8}


@pytest.fixture
def qr_data(tmp_path):
    """합성 데이터 — 렌더러가 새 CSV를 실제로 읽는지만 본다."""
    import json
    import os
    d = str(tmp_path)
    manifest.write_csv(os.path.join(d, "p1_neural_mode_sweep.csv"),
                       [_sweep_row("b_projection_exact", 9, 0.78, 606.0, 39)])
    manifest.write_csv(os.path.join(d, "p1_neural_qr.csv"),
                       [_sweep_row("b_prime_projection_qr", 9, 0.80, 611.0, 40)])
    manifest.write_csv(os.path.join(d, "p1_qr_span_diagnostics.csv"),
                       [_diag_row("b_prime_projection_qr", 10, 9, 1.56e11, 4.4e-15,
                                  ranks=(9, 9, 9))])
    manifest.write_csv(os.path.join(d, "p1_orthogonality.csv"),
                       [{"arm": "b_projection_exact", "n_prev_modes": 9,
                         "n_seeds": 50, "iters": 4000,
                         "gram_dev_fro_median": 6.26, "gram_dev_fro_p95": 7.0,
                         "max_offdiag_median": 0.99, "max_offdiag_p95": 1.0,
                         "inv_gram_rel_diff_median": 1e3,
                         "inv_gram_rel_diff_p95": 1e4,
                         "proj_op_diff_W_median": 5.59, "proj_op_diff_W_p95": 6.0,
                         "kappa_gram_median": 1.562e11, "kappa_gram_p95": 1e12,
                         "lam_min_normalized_median": 1e-4,
                         "eff_rank_1e12_median": 9, "n_seeds_gram_singular": 0,
                         "solve_method": "torch.linalg.solve", "note": "x"}])
    manifest.write_csv(os.path.join(d, "p1_fp32_stress.csv"),
                       [{**_sweep_row("b_projection_exact", 8, 0.0, 300.0, 0),
                         "dtype": "float32", "n_modes_completed": 10},
                        {**_sweep_row("b_prime_projection_qr", 8, 0.9, 300.0, 45),
                         "dtype": "float32", "n_modes_completed": 10}])
    manifest.write_csv(os.path.join(d, "p1_fp32_span_diagnostics.csv"),
                       [_diag_row("b_projection_exact", 9, 8, 4.0e10, 1e-3,
                                  dtype="float32", ranks=(5, 6, 8)),
                        _diag_row("b_prime_projection_qr", 9, 8, 4.0e10, 2e-6,
                                  dtype="float32", dropped=1, ranks=(5, 6, 8))])
    for n, extra in (("manifest_p1_qr", {}), ("manifest_p1_fp32", {}),
                     ("manifest_p1_orthogonality", {}), ("manifest_p1_neural", {})):
        with open(os.path.join(d, n + ".json"), "w") as f:
            json.dump({"git_sha": "abc12345", **extra}, f)
    return d


def test_attribution_table_carries_the_qr_arm(qr_data):
    """(b′)이 arm 비교표에 들어와야 한다 — 별도 표로 떼면 (b)와 비교가 안 된다."""
    s = tables.table1_attribution(qr_data)
    assert "b_prime_projection_qr" in tables.ARM_LABEL
    assert tables.ARM_LABEL["b_prime_projection_qr"] in s
    assert "40/50" in s          # QR 행의 correct 카운트
    assert "39/50" in s          # (b) 행이 사라지지 않았다


def test_orthogonality_table_contrasts_gram_conditioning_with_qr_defect(qr_data):
    """표의 논점 — κ₂(G)는 1.6e11까지 자라는데 ‖I − QᵀWQ‖는 ε 수준에 머문다."""
    s = tables.table_orthogonality(qr_data)
    assert "QᵀWQ" in s or "Q^T W Q" in s
    assert "1.56e+11" in s or "1.562e+11" in s
    assert "4.40e-15" in s or "4.4e-15" in s


def test_fp32_table_puts_the_two_arms_side_by_side_with_the_criterion(qr_data):
    s = tables.table_fp32(qr_data)
    assert tables.ARM_LABEL["b_projection_exact"] in s
    assert tables.ARM_LABEL["b_prime_projection_qr"] in s
    assert "float32" in s
    assert "unit roundoff" in s          # 안정화 기준을 캡션이 말해야 한다
    assert "source:" in s and "git " in s


def test_fp32_table_reports_dropped_directions_so_rank_awareness_is_visible(qr_data):
    s = tables.table_fp32(qr_data)
    assert "dropped" in s.lower()


def test_p4_table_merges_the_added_arms_when_present(tmp_path):
    import json
    import os
    d = str(tmp_path)
    base = {"problem": "P4", "mode": 1, "ladder": "I0", "nodes": "gauss2d", "n": 50,
            "p_correct": 0.76, "wilson_lo": 0.6, "wilson_hi": 0.87, "correct": 38,
            "lower_mode_basin": 0, "spurious": 6, "non_converged": 6,
            "p_accurate": 0.76, "e_lam_median": 3e-3, "mac_median": 0.99,
            "Lam_median": 100.1, "Lam_ref": 100.0, "iters": 4000,
            "seconds": 590.0, "n_q": 1200, "E_T_success": 776.0}
    manifest.write_csv(os.path.join(d, "p4_neural.csv"),
                       [{**base, "arm": "p4_neural_2d[exact projection]"}])
    manifest.write_csv(os.path.join(d, "p4_neural_arms.csv"),
                       [{**base, "arm": "p4_neural_2d[diagonal projection]",
                         "p_correct": 0.0, "correct": 0, "lower_mode_basin": 44},
                        {**base, "arm": "p4_neural_2d[qr projection]"}])
    for n in ("manifest_p4_neural", "manifest_p4_neural_arms"):
        with open(os.path.join(d, n + ".json"), "w") as f:
            json.dump({"git_sha": "abc12345"}, f)
    s = tables.table_p4(d)
    assert "exact" in s and "diagonal" in s and "qr" in s.lower()
    assert "44/50" in s          # 2차원에서도 lower-mode basin이 재현되면 그 숫자


def test_appendix_b_discloses_the_qr_arm_the_precision_axis_and_the_p4_arms():
    """새 arm과 새 축이 공시에 없으면 재현 지시가 불완전하다."""
    b = tables.appendix_b_disclosure(DATA)
    assert "(b′)" in b
    assert "Gram–Schmidt" in b or "Gram-Schmidt" in b
    assert "float32" in b
    assert "(a′) on P4" in b and "(b′) on P4" in b
    # (b)에만 있는 jitter를 공시해야 3-arm 비교가 공정하다
    assert "1e-14" in b


def test_the_reference_table_certifies_the_shapes_not_only_the_eigenvalues():
    """MAC 게이트는 모드형과 비교한다 — 고유값만 인증하면 채점의 절반이 인증되지 않는다."""
    s = tables.table7_references(DATA)
    assert "mode-shape certification" in s
    # 각 문제가 자기 인증값을 갖는다
    for problem in ("P1", "P2", "P4"):
        rows = [ln for ln in s.splitlines() if ln.startswith(f"| {problem} ")]
        assert rows, problem
        assert all(ln.rstrip().rstrip("|").split("|")[-1].strip() not in ("", "—")
                   for ln in rows), problem


def test_the_reference_caption_states_why_shape_certification_exists():
    s = tables.table7_references(DATA)
    assert "cancellation" in s
    assert "0.887" in s          # 실제로 물었던 값
    assert "certified" in s.lower()


def test_the_deep_mode_table_reports_both_routes_and_the_basin_count():
    """⑦의 내용은 '벽'이 아니라 '점진적 열화 + basin 0'이다 — 표가 그걸 보여야 한다."""
    s = tables.table_deep_modes(DATA)
    assert tables.ARM_LABEL["b_projection_exact"] in s
    assert tables.ARM_LABEL["b_prime_projection_qr"] in s
    for m in ("11", "13", "16"):
        assert f"| {m} " in s or f" {m} " in s
    assert "lower-mode basin" in s
    assert "κ₂(G)·u" in s
    assert "source:" in s


def test_the_deep_mode_caption_states_what_the_criterion_predicted():
    """끝점 수치는 데이터에서 읽는다 — 여기에 상수를 박으면 테스트 자신이
    재실행 때마다 어긋나는 두 번째 손글씨가 된다(실제로 그랬다)."""
    import csv
    import os
    s = tables.table_deep_modes(DATA)
    rows = sorted(csv.DictReader(
        open(os.path.join(DATA, "p1_fp64_deep_b_span_diagnostics.csv"))),
        key=lambda r: int(float(r["k"])))
    ku = float(rows[-1]["kappa_gram_times_eps"])
    assert f"{ku:.3f}" in s, f"그람 경로의 가장 깊은 κ₂·u({ku:.3f})가 캡션에 없다"
    assert "no lower-mode-basin outcome" in s
    assert "reference" in s.lower()             # 참조 정정 사실을 밝힌다


# ─────────────────────────────────────────────────────────────────────────────
# 진단 열의 요약 시드 수는 셀의 시드 수와 다를 수 있다.
# 실제로 8시드 요약과 50시드 요약이 한 논문 안에 섞여 κ₂(G)가 최대 3배 어긋났고,
# 두 표를 대조한 독자가 그 차이를 먼저 발견했다. 캡션이 말하지 않으면 재발한다.
# ─────────────────────────────────────────────────────────────────────────────

DIAG_TABLES = (
    ("table_orthogonality", ("p1_orthogonality", "p1_qr_span_diagnostics")),
    ("table_p4", ("p4_span_diagnostics",)),
    ("table_fp32", ("p1_fp32_span_diagnostics",)),
    ("table_deep_modes", ("p1_fp64_deep_b_span_diagnostics",
                          "p1_fp64_deep_qr_span_diagnostics")),
)


def _summary_seeds(stem):
    """그 진단 파일이 실제로 몇 시드를 요약했는가."""
    import csv
    import os
    p = os.path.join(tables.DATA_DIR, stem + ".csv")
    if not os.path.exists(p):
        return set()
    rows = list(csv.DictReader(open(p)))
    key = ("n_seeds_summarized" if rows and "n_seeds_summarized" in rows[0]
           else "n_seeds")
    return {int(float(r[key])) for r in rows if r.get(key) not in (None, "")}


@pytest.mark.parametrize("name,stems", DIAG_TABLES)
def test_diagnostic_captions_state_how_many_seeds_the_diagnostics_summarize(name, stems):
    cap = [l for l in getattr(tables, name)().splitlines()
           if l.startswith("**Table")][0]
    seeds = set().union(*(_summary_seeds(s) for s in stems))
    assert len(seeds) == 1, f"{name}: 요약 시드 수가 파일마다 다르다 — {sorted(seeds)}"
    n = seeds.pop()
    assert f"diagnostics summarize {n} seeds" in cap, (
        f"{name}: 캡션이 진단 요약 시드 수({n})를 밝히지 않는다")


# ─────────────────────────────────────────────────────────────────────────────
# 부록 D — bfloat16 시도. 데이터는 있는데 원고에 없던 결과다.
# 세 번째 정밀도 점이 되지 못한 이유가 하필 이 논문의 e_measurement와 같은 구조다:
# 채점 기준(상대변화 1e-3)이 산술의 단위 반올림(7.8e-3)보다 곱다.
# ─────────────────────────────────────────────────────────────────────────────

def test_appendix_d_reports_the_bfloat16_attempt_and_why_it_is_not_a_third_point():
    import csv
    import os
    s = tables.appendix_d_bfloat16(DATA)
    ch = {int(r["mode"]): r for r in
          csv.DictReader(open(os.path.join(DATA, "p1_bf16_chol_stress.csv")))}
    # 모드 1은 deflation을 전혀 쓰지 않는데도 게이트를 통과하지 못하는 시드가 있고,
    # 그 시드들조차 옳은 모드를 찾았다 — 채점이 산술보다 곱다는 직접 증거다.
    assert float(ch[1]["p_accurate"]) == 1.0
    assert f"{int(ch[1]['non_converged'])} of {int(ch[1]['n'])} seeds" in s
    assert "no deflation" in s
    # 단위 반올림은 데이터에서 읽는다
    from eigen_benchmark.render import md as _md
    dg = list(csv.DictReader(open(os.path.join(DATA, "p1_bf16_span_diagnostics.csv"))))
    eps = float(dg[0]["eps_machine"])
    assert _md.fmt(eps, 2) in s, f"bfloat16 단위 반올림 {eps}이 부록에 없다"
    # 두 경로의 서로 다른 실패 방식
    assert "non-finite" in s
    assert "dropped" in s
    assert "source:" in s


def test_appendix_d_is_registered_for_injection():
    from eigen_benchmark.render import inject
    assert "appendix_d" in inject.render_all()


def test_the_availability_record_names_every_working_precision_in_the_data():
    """"float64 throughout"는 축소정밀도 축이 생긴 순간 거짓이 됐다. 가용성 기록은
    리뷰어가 데이터 번들과 대조하는 첫 표이므로, 실제로 존재하는 dtype을 읽어야 한다."""
    import csv
    import glob
    import os
    s = tables.availability_record(DATA)
    seen = set()
    for p in glob.glob(os.path.join(DATA, "*.csv")):
        try:
            rows = list(csv.DictReader(open(p)))
        except Exception:
            continue
        for r in rows:
            if r.get("dtype"):
                seen.add(r["dtype"])
    assert seen, "데이터에 dtype 열이 하나도 없다 — 테스트가 무의미하다"
    for d in sorted(seen):
        assert d in s, f"가용성 기록이 {d} 실행을 언급하지 않는다 (데이터에는 있다)"


def test_the_availability_record_normalizes_dtype_spellings():
    """드라이버마다 dtype을 'float64'로도 'torch.float64'로도 적었다. 정규화하지 않으면
    가용성 문장이 같은 정밀도를 두 번 세어 'float32, bfloat16, torch.float64'처럼 읽힌다."""
    import re
    s = tables.availability_record(DATA)
    # torch.set_default_dtype(torch.float64)은 함수 호출 표기이므로 정당하다.
    # 검사 대상은 **정밀도 목록**이다.
    m = re.search(r"which runs at ([^;]+);", s)
    assert m, "정밀도 목록 문장을 찾지 못했다"
    named = [x.strip() for x in m.group(1).split(",")]
    assert all(not n.startswith("torch.") for n in named), f"정규화 안 됨: {named}"
    assert len(named) == len(set(named)), f"중복 나열: {named}"
    assert "float64" not in named, f"기준 정밀도가 예외 목록에 들어갔다: {named}"
    assert "bfloat16" in named


def test_the_budget_table_carries_the_mode_14_control():
    """5배 예산 대조는 다른 예산 실험과 같은 표에 있어야 비교가 된다 —
    모드 9·10 옆에 놓여야 깊이에 따라 예산이 무엇을 사는지가 한 축에서 읽힌다."""
    s = tables.table5_longrun(DATA)
    assert "14" in s
    import csv
    import os
    ctl = {int(r["mode"]): r for r in csv.DictReader(
        open(os.path.join(DATA, "p1_fp64_budget14_stress.csv")))}
    assert f"{float(ctl[14]['p_correct']):.3g}"[:4] in s.replace("0.820", "0.82")


def test_the_budget_flag_is_defined_as_the_bare_comparison_it_is():
    """`budget-resolved`는 점추정 비교일 뿐 유의성 검정이 아니다. 실제로 모드 9는
    'yes'인데 Wilson 구간이 겹치고, (e) 모드 1은 'no'인데 구간이 분리된다(악화).
    캡션이 그 사실을 말하지 않으면 표만 읽는 독자가 두 경우를 모두 오독한다."""
    s = tables.table5_longrun(DATA)
    assert "not a significance test" in s
    # "움직이지 않으면 예산으로 못 산다"는 서술은 악화 사례를 덮지 못한다
    assert "does not move is limited by something the budget cannot buy" not in s
    assert "worse" in s.lower()


def test_table_p4_quotes_the_span_row_its_own_sentence_names():
    """캡션은 "the stored set handed to the mode-3 solve"라고 쓰고, 같은 문장에서 P1의
    0.991과 286에 견준다. 그 두 값은 P1의 **k=2**이므로 P4도 k=2를 읽어야 대응이 성립한다.
    최심 행(k=5)을 읽으면 문장은 모드 3을 말하면서 숫자는 다른 깊이를 가리킨다."""
    span = csv("p4_span_diagnostics")
    k2 = next(r for r in span if int(float(r["k"])) == 2)
    s_ = tables.table_p4(DATA)
    assert "stored set handed to the mode-3 solve" in s_
    for key in ("dev_identity", "kappa_gram"):
        v = float(k2[key])
        assert any(f"{v:.{d}g}" in s_ or f"{v:.{d}f}" in s_ for d in (2, 3)), (key, v)
    # 최심 행의 값이 그 문장에 들어가 있으면 안 된다
    deep = max(span, key=lambda r: int(float(r["k"])))
    assert f"{float(deep['kappa_gram']):.3g}" not in s_, "k=5의 kappa가 실려 있다"
