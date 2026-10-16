"""논문 A MLST 결과 재검토(2026-10-15) 대응 감사 — 판정 불일치표, P4 저장 전이, 저장 스팬 대수, 코너 구적 전 셀.

학습 없이 커밋된 기록만 읽는다. 결과가 아니라 산술과 배선을 본다(마지막 시험만 커밋된 자료를 읽는다).
"""
import json
import math
import os

import numpy as np
import pytest

from eigen_benchmark.drivers import paperA_mlst_audit as au

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "_generated", "data", "paper2")


def test_discordance_counts_and_exact_mcnemar():
    a = [1] * 12 + [0] * 8
    b = [0] * 19 + [1]
    d = au.discordance(a, b)
    assert (d["n11"], d["n10"], d["n01"], d["n00"]) == (0, 12, 1, 7)
    assert d["p_mcnemar"] == pytest.approx(2 * (1 + 13) / 2 ** 13)
    lo, hi = d["discordant_ci"]
    assert lo < 13 / 20 < hi
    with pytest.raises(ValueError):
        au.discordance(a, b[:5])


def test_discordance_without_discordant_pairs_bounds_the_rate_and_claims_no_equivalence():
    d = au.discordance([1] * 50, [1] * 50)
    assert d["n10"] == d["n01"] == 0 and d["p_mcnemar"] == 1.0
    assert d["discordant_ci"][0] == 0.0
    assert d["discordant_ci"][1] == pytest.approx(1 - 0.025 ** (1 / 50))


def _rec(seed, outcome, matched=3, mac=0.99, e_lam=0.01):
    return {"seed": seed, "mode": 3, "outcome": outcome, "matched_mode": matched, "mac": mac, "e_lam": e_lam}


def test_transitions_separate_mode_identity_from_the_convergence_gate():
    proj = [_rec(0, "correct"), _rec(1, "correct"), _rec(2, "correct"),
            _rec(3, "spurious", matched=4, mac=0.5)]
    raw = [_rec(0, "correct"), _rec(1, "lower_mode_basin", matched=1, mac=0.999, e_lam=0.5),
           _rec(2, "non_converged", matched=1, mac=0.95, e_lam=0.4), _rec(3, "correct")]
    t = au.transitions(proj, raw)
    assert t["table"][("correct", "lower_mode_basin")] == 1
    assert t["table"][("spurious", "correct")] == 1
    assert t["lost"]["n"] == 2 and t["rescued"] == 1
    assert t["lost"]["mode1_identity"] == 2          # 분지 1 + 비수렴이지만 모드 1과 MAC ≥ 0.9
    assert t["lost"]["by_raw_outcome"] == {"lower_mode_basin": 1, "non_converged": 1}
    assert t["p_accurate"] == {"projected": 3, "raw": 2}
    with pytest.raises(ValueError):
        au.transitions(proj, raw[:3])


def _pair():
    x = np.linspace(0.0, 1.0, 401)
    w = np.full_like(x, 1.0 / 400)
    w[[0, -1]] /= 2
    p1, d1 = np.sin(np.pi * x), -np.pi ** 2 * np.sin(np.pi * x)
    p2 = np.sin(2 * np.pi * x) + 0.3 * p1                 # 비직교 원시 쌍 — 곡률도 같은 결합
    d2 = -(2 * np.pi) ** 2 * np.sin(2 * np.pi * x) + 0.3 * d1
    return (p1, d1), (p2, d2), w


@pytest.mark.parametrize("alpha", [0.0, 0.01, 0.25, 1.0])
def test_diagonal_rule_keeps_the_stored_span_for_any_nonzero_correlation(alpha):
    a, b, w = _pair()
    out = au.span_algebra(a, b, w, alpha)
    r = alpha / math.sqrt(1 + alpha ** 2)
    assert out["r"] == pytest.approx(r, abs=1e-12)
    assert out["ad_eig_err"] < 1e-12                   # A_D v± = (1 ± r) v±
    assert out["proj_resid"] < 1e-12                   # (I − P_G) v± = 0
    assert out["deflated_gain"] == pytest.approx(r, abs=1e-12)   # ‖(I − A_D) v±‖_W = |r|
    if alpha > 0:
        assert out["rq_rel_diff"] < 1e-12              # R[(I − A_D) v±] = R(v±)
        assert out["deflated_rq_min"] == pytest.approx(out["span_rq_min"], rel=1e-10)
    else:
        assert math.isnan(out["rq_rel_diff"]) and math.isnan(out["deflated_rq_min"])


def test_adf_all_cells_counts_gate_crossings_over_every_scored_cell():
    rows = []
    cells = [(5.20, 5.19, "correct"), (5.60, 5.50, "spurious"), (5.42, 5.44, "non_converged")]
    for seed, (l0, l8, oc) in enumerate(cells):
        for lv, lam in ((0, l0), (4, l8 + 1e-7), (8, l8)):
            rows.append({"arm": "diagonal", "mode": "1", "seed": str(seed), "level": str(lv),
                         "Lam": repr(lam), "outcome": oc})
    out = au.adf_all_cells(rows, {1: 5.17})
    c = out[("diagonal", 1)]
    assert c["n"] == 3 and c["max_rel_4_8"] < 1e-6
    assert c["gate_crossings"] == 1                    # 5.42(e 0.048) → 5.44(e 0.052)
    assert c["max_rel_0_8"] == pytest.approx(0.10 / 5.50)


def test_audit_reproduces_the_reported_counts_from_committed_records(tmp_path):
    if not os.path.exists(os.path.join(DATA, "p4_raw_records.jsonl")):
        pytest.skip("committed MLST records absent")
    res = au.audit(DATA, outdir=str(tmp_path))
    d = res["p4_transitions"]["diagonal"]
    assert d["lost"]["n"] == 14 and d["rescued"] == 0 and d["p_accurate"] == {"projected": 33, "raw": 19}
    assert res["p4_transitions"]["exact"]["lost"]["n"] == 0
    assert res["discordance"]["p1_rf_mode3_rf_vs_a_prime"]["n10"] == 50
    assert res["discordance"]["p1_rf_mode3_rf_vs_b"]["n10"] + res["discordance"]["p1_rf_mode3_rf_vs_b"]["n01"] == 0
    assert res["discordance"]["p4_mode3_diagonal_projected_vs_raw"]["n10"] == 14
    alg = res["p1_span_algebra"]
    assert alg["max_ad_eig_err"] < 1e-10 and alg["max_proj_resid"] < 1e-10
    g = {(r["mode"], r["eps"], r["family"]): r for r in res["p1_cheapest_to_target"]}
    assert g[(2, 1e-6, "neural_randfeat")]["solver"] == "f_eig_pielm(nf=80)"
    assert g[(2, 1e-6, "neural_randfeat")]["n_target"] == 45 and g[(2, 1e-6, "neural_randfeat")]["n_seeds"] == 50
    assert g[(1, 1e-4, "neural_gradient")]["p_target"] == 1.0
    for name in ("paperA_mlst_audit.json", "p4_raw_transitions.csv", "p1_span_algebra.csv",
                 "p4_adf_all_cells.csv", "p1_cheapest_to_epsilon_target.csv"):
        assert os.path.exists(os.path.join(str(tmp_path), name))
    json.load(open(os.path.join(str(tmp_path), "paperA_mlst_audit.json"), encoding="utf-8"))


def test_rescore_changes_only_the_eigenvalue_gate_and_keeps_identity_and_convergence():
    recs = [{"seed": 0, "mode": 2, "outcome": "correct", "matched_mode": 2, "mac": 0.99, "e_lam": 0.02},
            {"seed": 1, "mode": 2, "outcome": "non_converged", "matched_mode": 2, "mac": 0.99, "e_lam": 0.02},
            {"seed": 2, "mode": 2, "outcome": "lower_mode_basin", "matched_mode": 1, "mac": 0.99, "e_lam": 0.2},
            {"seed": 3, "mode": 2, "outcome": "spurious", "matched_mode": 2, "mac": 0.99, "e_lam": 0.08}]
    lam_hi = {0: 7.0, 1: 7.0, 2: 6.0, 3: 6.5}           # 기준 Λ = 6.48 — 0: 문턱 밖으로, 3: 안으로
    out = au.rescore(recs, lam_hi, ref_lam=6.48)
    assert out["outcomes"] == {0: "spurious", 1: "non_converged", 2: "lower_mode_basin", 3: "correct"}
    assert out["flips"] == {("correct", "spurious"): 1, ("spurious", "correct"): 1}
    assert out["n_correct"] == (1, 1)


def test_audit_rescoring_finds_the_quadrature_artifacts_in_two_seeds(tmp_path):
    if not os.path.exists(os.path.join(DATA, "p4_adf_neural.csv")):
        pytest.skip("committed MLST records absent")
    res = au.audit(DATA, outdir=str(tmp_path))
    rs = res["p4_adf_rescore_eight_layers"]
    assert sum(v["n_correct_training_rule"] for v in rs.values()) == 300
    assert {s for v in rs.values() for s in v["flipped_seeds"]} == {18, 30}
    assert rs["diagonal_mode3"]["n_correct_eight_layers"] == 19
    assert res["p4_adf_correct_kept_max_rel_0_8"] < 0.01


def test_target_cost_grid_uses_target_specific_success_and_picks_the_cheapest_expected_time():
    rows = [{"family": "classical", "solver": "monomial_raw", "mode": "2", "e_lam": "1e-9", "seconds_per_attempt": "0.0002"},
            {"family": "neural_randfeat", "solver": "A", "mode": "2", "e_lam": "2e-7", "seconds_per_attempt": "0.013"},
            {"family": "neural_randfeat", "solver": "B", "mode": "2", "e_lam": "3e-7", "seconds_per_attempt": "0.014"},
            {"family": "neural_gradient", "solver": "G", "mode": "2", "e_lam": "1e-5", "seconds_per_attempt": "75.0"}]
    ok = lambda e: {"outcome": "correct", "e_lam": e}                    # noqa: E731
    recs = {("A", 2): [ok(1e-7)] * 37 + [ok(1e-5)] * 13,
            ("B", 2): [ok(1e-7)] * 50,
            ("G", 2): [ok(1e-5)] * 49 + [{"outcome": "non_converged", "e_lam": 1e-7}]}
    g = {(r["eps"], r["family"]): r for r in au.target_cost_grid(rows, recs, eps_levels=(1e-4, 1e-6), modes=(2,))}
    rf = g[(1e-6, "neural_randfeat")]
    assert rf["solver"] == "B" and rf["n_target"] == 50 and rf["E_T_target"] == pytest.approx(0.014)
    assert g[(1e-4, "neural_randfeat")]["solver"] == "A"                 # 1e-4에서는 둘 다 50/50 — A가 싸다
    assert g[(1e-4, "neural_gradient")]["n_target"] == 49                # 비수렴 시드는 정확해도 세지 않는다
    assert g[(1e-6, "neural_gradient")]["reached"] is False
    assert g[(1e-6, "classical")]["p_target"] == 1.0
