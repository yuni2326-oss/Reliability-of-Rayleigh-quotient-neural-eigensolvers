"""논문 A MLST 재투고 계산의 드라이버 — 배선과 판정 산술(사전규정 2026-10-14-mlst-prespecification.md).

소형(quick) 설정으로 CPU에서 끝까지 돈다. 결과가 아니라 배선·파일·불변식을 본다.
"""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eigen_benchmark.drivers import run_paperA_mlst as ml     # noqa: E402


def test_paired_bootstrap_is_seed_clustered_and_rules_read_the_interval():
    a = np.array([1.0] * 12 + [0.0] * 8)
    b = np.array([0.0] * 19 + [1.0])
    pt, lo, hi = ml.paired_bootstrap(a, b, B=4000)
    assert pt == pytest.approx(0.55) and lo > 0 and ml.rule((pt, lo, hi)) == "rule1"
    assert ml.rule((0.0, -0.1, 0.1)) == "rule2" and ml.rule((-0.5, -0.7, -0.3)) == "rule3"
    with pytest.raises(ValueError):
        ml.paired_bootstrap(a, b[:5])


@pytest.fixture(scope="module")
def p1_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("p1"))
    ml.p1_rf(outdir=d, quick=True, device="cpu")
    return d


def test_p1_rf_shares_one_prefix_across_the_three_frozen_arms(p1_dir):
    recs = [json.loads(x) for x in open(os.path.join(p1_dir, "p1_rf_records.jsonl"), encoding="utf-8")]
    assert {r["arm"] for r in recs} == set(ml.FROZEN.values())
    assert {int(r["mode"]) for r in recs} == {3, 4}
    span = list(open(os.path.join(p1_dir, "p1_rf_span.csv"), encoding="utf-8"))
    assert len(span) == 1 + 3 * 2
    from eigen_benchmark.drivers import manifest
    rows = {(r["arm"], int(r["seed"])): r for r in manifest.read_csv(os.path.join(p1_dir, "p1_rf_span.csv"))}
    for s in (0, 1):
        raw = float(rows[(ml.FROZEN["a_prime"], s)]["dev_identity"])
        assert float(rows[(ml.FROZEN["b"], s)]["dev_identity"]) == pytest.approx(raw, rel=1e-12)
        assert float(rows[(ml.FROZEN["rf"], s)]["dev_identity"]) < 1e-12          # 저장 시 직교화
        assert float(rows[(ml.FROZEN["rf"], s)]["mode3_leak_rel"]) < 1e-10
        assert float(rows[(ml.FROZEN["b"], s)]["mode3_leak_rel"]) < 1e-10
    z = np.load(os.path.join(p1_dir, "p1_mlst_prefix.npz"))
    assert z["raw_phi1"].shape == (2, 256) and z["raw_d2_2"].shape == (2, 256)
    man = json.load(open(os.path.join(p1_dir, "manifest_p1_rf.json"), encoding="utf-8"))
    assert man["reproduction_gate"]["checked"] is False and "launch" in man


def test_p1_basis_keeps_the_subspace_and_moves_only_the_diagonal_rule(p1_dir):
    ml.p1_basis(outdir=p1_dir, quick=True, device="cpu")
    from eigen_benchmark.drivers import manifest
    op = manifest.read_csv(os.path.join(p1_dir, "p1_basis_operator.csv"))
    for r in op:
        a = float(r["alpha"])
        rr = a / np.sqrt(1 + a * a)
        assert float(r["r"]) == pytest.approx(rr, abs=1e-10)
        assert float(r["op_gap"]) == pytest.approx(rr, abs=1e-10)
        if a == 0.0:
            assert float(r["diag_vs_exact_rel"]) < 1e-10 and float(r["leak_init_rel"]) < 1e-10
    recs = [json.loads(x) for x in open(os.path.join(p1_dir, "p1_basis_invariance_records.jsonl"),
                                        encoding="utf-8")]
    assert {float(r["alpha"]) for r in recs} == set(ml.ALPHAS[:2]) and {int(r["mode"]) for r in recs} == {3}


def test_p1_rf_fp32_scores_every_mode_from_the_float32_prefix(tmp_path):
    ml.p1_rf_fp32(outdir=str(tmp_path), quick=True, device="cpu")
    from eigen_benchmark.drivers import manifest
    rows = manifest.read_csv(os.path.join(str(tmp_path), "p1_rf_fp32.csv"))
    assert [int(r["mode"]) for r in rows] == [1, 2, 3, 4]
    assert {r["dtype"] for r in rows} == {"float32"}


@pytest.fixture(scope="module")
def p4_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("p4"))
    ml.p4_raw(outdir=d, quick=True, device="cpu")
    return d


def test_p4_raw_writes_records_per_seed_span_and_networks(p4_dir):
    recs = [json.loads(x) for x in open(os.path.join(p4_dir, "p4_raw_records.jsonl"), encoding="utf-8")]
    assert len(recs) == 3 * 3 * 2 and {r["storage"] for r in recs} == {"raw"}
    from eigen_benchmark.drivers import manifest
    sp = manifest.read_csv(os.path.join(p4_dir, "p4_raw_span_per_seed.csv"))
    assert {(r["arm"], int(r["stage_mode"])) for r in sp} == {(a, m) for a in ml.P4_ARMS for m in (2, 3)}
    for r in sp:
        if r["arm"] in ("exact", "qr") and int(r["stage_mode"]) == 3:
            assert float(r["leak_rel"]) < 1e-8
    z = np.load(os.path.join(p4_dir, "p4_params_raw.npz"))
    assert {k.split("_m")[0] for k in z.files} == set(ml.P4_ARMS)


def test_p4_adf_measures_reference_and_neural_cells_on_every_level(p4_dir):
    out = ml.p4_adf(outdir=p4_dir, quick=True, device="cpu")
    levels = {int(r["level"]) for r in out["ref_rows"]}
    assert levels == {0, 2} and len(out["ref_rows"]) == 3 * 2
    assert len(out["nn_rows"]) == 3 * 3 * 2 * 2
    lv = {(r["arm"], r["mode"], r["seed"], r["level"]): r["Lam"] for r in out["nn_rows"]}
    assert all(np.isfinite(v) and v > 0 for v in lv.values())
    assert out["corner"][0]["grad_max"] < 10
    # 등급 0겹은 훈련 구적과 같은 점이다 — P4-RAW가 기록한 Λ를 그대로 재현한다.
    recs = [json.loads(x) for x in open(os.path.join(p4_dir, "p4_raw_records.jsonl"), encoding="utf-8")]
    from eigen_benchmark.neural.p4_neural import ARM_NAME
    for r in recs:
        arm = [a for a, n in ARM_NAME.items() if n == r["arm"]][0]
        assert lv[(arm, int(r["mode"]), int(r["seed"]), 0)] == pytest.approx(float(r["Lam"]), rel=1e-10)


def test_stats_read_whatever_has_been_run(p1_dir):
    res = ml.stats_from(p1_dir, B=500)
    assert res["p1_rf"]["n_seeds"] == 2
    assert res["p1_rf"]["rf_minus_a_prime"]["verdict"] in ("rule1", "rule2", "rule3")
    assert res["p4_raw"] is None and res["p4_adf"] is None


# ── 재검토 후속(2026-10-15 사전규정 보완): fp32 고정 스팬 직접 시험, 정확 사영의 결과 수준 기저 불변 ──

def _sines(k, n=257):
    x = torch.linspace(0, 1, n, dtype=torch.float64)
    w = torch.full_like(x, 1.0 / (n - 1))
    w[[0, -1]] /= 2
    B = torch.stack([torch.sin((j + 1) * np.pi * x) for j in range(k + 1)])
    B2 = torch.stack([-((j + 1) * np.pi) ** 2 * B[j] for j in range(k + 1)])
    return x, w, B, B2


def test_fixed_span_routes_agree_on_an_orthogonal_set_and_separate_on_an_ill_conditioned_one():
    _, w, B, B2 = _sines(5)
    phi, phi2 = B[5] + 0.5 * B[0], B2[5] + 0.5 * B2[0]
    f = lambda t: t.to(torch.float32)                                   # noqa: E731
    r = ml.fixed_span_routes(f(B[:5]), f(B2[:5]), f(phi), f(phi2), f(w))
    assert max(r[k]["rel_err"] for k in ("gram", "cgs", "mgs")) < 1e-5
    Phi, Phi2 = [B[0]], [B2[0]]                                         # 열마다 앞 열 + 작은 새 성분
    for j in range(1, 5):
        Phi.append(Phi[-1] + 1e-3 * B[j])
        Phi2.append(Phi2[-1] + 1e-3 * B2[j])
    r = ml.fixed_span_routes(f(torch.stack(Phi)), f(torch.stack(Phi2)), f(phi), f(phi2), f(w))
    assert r["kappa_gram"] > 1e5
    assert r["mgs"]["rel_err"] < r["gram"]["rel_err"]
    assert np.isfinite(r["cgs"]["rel_err"]) and r["cgs"]["dev_identity"] > r["mgs"]["orth_defect"]
    for k in ("gram", "cgs", "mgs"):
        assert 0.0 <= r[k]["leak"] <= 1.0 + 1e-12                       # 사영 노름 기준 누설은 1 이하


def test_p1_fp32_fixed_span_saves_raw_sets_and_scores_three_routes_per_stage(tmp_path):
    out = ml.p1_fp32_fixed_span(outdir=str(tmp_path), quick=True, device="cpu")
    z = np.load(os.path.join(str(tmp_path), "p1_fp32_raw_sets.npz"))
    assert z["raw_shapes"].shape[:2] == (4, 2) and z["raw_d2"].shape == z["raw_shapes"].shape
    from eigen_benchmark.drivers import manifest
    rows = manifest.read_csv(os.path.join(str(tmp_path), "p1_fp32_fixed_span.csv"))
    assert {(int(r["stage_mode"]), int(r["seed"])) for r in rows} == {(m, s) for m in (2, 3, 4) for s in (0, 1)}
    assert all(r["dtype"] == "float32" for r in rows) and "err_cgs" in rows[0] and "err_mgs" in rows[0]
    assert out["outdir"] == str(tmp_path)


def test_p1_basis_exact_trains_mode3_with_the_exact_rule_on_mixed_bases(p1_dir):
    ml.p1_basis_exact(outdir=p1_dir, quick=True, device="cpu")
    recs = [json.loads(x) for x in open(os.path.join(p1_dir, "p1_basis_exact_records.jsonl"), encoding="utf-8")]
    assert {r["arm"] for r in recs} == {"basis_exact_alpha=0.01"} and {int(r["mode"]) for r in recs} == {3}
    from eigen_benchmark.drivers import manifest
    op = manifest.read_csv(os.path.join(p1_dir, "p1_basis_exact.csv"))
    assert all(float(r["op_exact_vs_raw_rel"]) < 1e-10 for r in op)  # 같은 스팬 → 같은 사영(연산자 수준)


def test_followup_stats_read_both_followups(p1_dir):
    ml.p1_fp32_fixed_span(outdir=p1_dir, quick=True, device="cpu")
    ml.p1_basis_exact(outdir=p1_dir, quick=True, device="cpu")
    res = ml.followup_stats_from(p1_dir, B=500)
    assert set(res["fixed_span"]["by_stage"]) == {2, 3, 4}
    assert res["fixed_span"]["by_stage"][3]["log10_gram_minus_cgs"]["verdict"] in ("rule1", "rule2", "rule3")
    be = res["basis_exact"]["0.01"]
    assert be["contrast_vs_b_frozen"]["verdict"] in ("rule1", "rule2", "rule3") and be["max_abs_e_lam_difference"] >= 0


def test_followup_stats_writes_the_judgment_file(p1_dir):
    if not os.path.exists(os.path.join(p1_dir, "p1_fp32_fixed_span.csv")):
        ml.p1_fp32_fixed_span(outdir=p1_dir, quick=True, device="cpu")
    if not os.path.exists(os.path.join(p1_dir, "p1_basis_exact_records.jsonl")):
        ml.p1_basis_exact(outdir=p1_dir, quick=True, device="cpu")
    ml.followup_stats(outdir=p1_dir, B=300)
    res = json.load(open(os.path.join(p1_dir, "paperA_mlst_followup_stats.json"), encoding="utf-8"))
    assert res["fixed_span"]["decision_stage"] == 6 and "basis_exact" in res


# ── P4 사영 저장 재실행(2026-10-16 사전규정 보완 2): 망 보존, 같은 8겹 구적의 대칭 재채점, 두 저장 정책의 장 수준 대조 ──

def _jsonl(d, name):
    return [json.loads(x) for x in open(os.path.join(d, name), encoding="utf-8")]


def test_gate_p4_counts_every_requested_mode(tmp_path):
    from eigen_benchmark.neural.p4_neural import ARM_NAME
    lam = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])            # (모드, 시드)
    recs = [{"arm": ARM_NAME["diagonal"], "mode": m + 1, "seed": s, "Lam": float(lam[m, s])}
            for m in range(3) for s in range(2)]
    recs[-1]["Lam"] = 6.0 + 1e-15                                    # 모드 3 시드 1만 한 ulp 다르다
    with open(tmp_path / "p4_neural_arms_records.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    out = {"n_seeds": 2, "lam": lam}
    g2 = ml._gate_p4("diagonal", out, str(tmp_path))
    assert (g2["n"], g2["bit_identical"], g2["pass"]) == (4, 4, True)
    g3 = ml._gate_p4("diagonal", out, str(tmp_path), modes=(1, 2, 3))
    assert (g3["n"], g3["bit_identical"], g3["pass"]) == (6, 5, False)
    assert g3["by_mode"] == {1: 2, 2: 2, 3: 1}


@pytest.fixture(scope="module")
def p4_both(p4_dir):
    ml.p4_projected_rerun(outdir=p4_dir, quick=True, device="cpu")
    return p4_dir


def test_p4_projected_rerun_stores_projected_states_and_keeps_the_networks(p4_both):
    from eigen_benchmark.drivers import manifest
    recs = _jsonl(p4_both, "p4_projected_rerun_records.jsonl")
    assert len(recs) == 3 * 3 * 2 and {r["storage"] for r in recs} == {"projected"}
    sp = manifest.read_csv(os.path.join(p4_both, "p4_projected_rerun_span_per_seed.csv"))
    assert {(r["arm"], int(r["stage_mode"])) for r in sp} == {(a, m) for a in ml.P4_ARMS for m in (2, 3)}
    for r in sp:
        if int(r["stage_mode"]) == 3:
            assert float(r["dev_identity"]) < 1e-10       # 사영된 상태를 저장하면 저장 집합은 구성상 직교다
            assert float(r["leak_rel"]) < 1e-8            # 그래서 대각 규칙도 저장 스팬을 지운다
    z = np.load(os.path.join(p4_both, "p4_params_projected.npz"))
    assert {k.split("_m")[0] for k in z.files} == set(ml.P4_ARMS)
    man = json.load(open(os.path.join(p4_both, "manifest_p4_projected_rerun.json"), encoding="utf-8"))
    assert man["storage"].startswith("projected") and set(man["reproduction_gate"]) == set(ml.P4_ARMS)
    # 원시 저장 실행의 기록은 그대로다
    assert {r["storage"] for r in _jsonl(p4_both, "p4_raw_records.jsonl")} == {"raw"}


def test_p4_adf_projected_relevels_with_the_projected_convention(p4_both):
    before = open(os.path.join(p4_both, "p4_raw_records.jsonl"), encoding="utf-8").read()
    out = ml.p4_adf_projected(outdir=p4_both, quick=True, device="cpu")
    assert len(out["nn_rows"]) == 3 * 3 * 2 * 2 and {r["storage"] for r in out["nn_rows"]} == {"projected"}
    lv = {(r["arm"], r["mode"], r["seed"], r["level"]): r["Lam"] for r in out["nn_rows"]}
    from eigen_benchmark.neural.p4_neural import ARM_NAME
    for r in _jsonl(p4_both, "p4_projected_rerun_records.jsonl"):
        arm = [a for a, n in ARM_NAME.items() if n == r["arm"]][0]
        # 등급 0겹은 훈련 구적이다 — 사영 저장 규약으로 다시 사영하면 재실행이 기록한 Λ가 나온다
        assert lv[(arm, int(r["mode"]), int(r["seed"]), 0)] == pytest.approx(float(r["Lam"]), rel=1e-10)
    assert os.path.exists(os.path.join(p4_both, "p4_adf_neural_projected.csv"))
    assert open(os.path.join(p4_both, "p4_raw_records.jsonl"), encoding="utf-8").read() == before


def test_p4_adf_keeps_the_raw_convention_after_the_refactor(p4_both):
    out = ml.p4_adf(outdir=p4_both, quick=True, device="cpu")
    lv = {(r["arm"], r["mode"], r["seed"], r["level"]): r["Lam"] for r in out["nn_rows"]}
    from eigen_benchmark.neural.p4_neural import ARM_NAME
    for r in _jsonl(p4_both, "p4_raw_records.jsonl"):
        arm = [a for a, n in ARM_NAME.items() if n == r["arm"]][0]
        assert lv[(arm, int(r["mode"]), int(r["seed"]), 0)] == pytest.approx(float(r["Lam"]), rel=1e-10)


def test_p4_policy_fields_hash_the_shared_prefix_and_compare_the_mode3_input(p4_both):
    out = ml.p4_policy_fields(outdir=p4_both, quick=True, device="cpu")
    rows = out["rows"]
    assert len(rows) == 3 * 2
    for r in rows:
        assert r["mode1_params_identical"] and r["mode2_params_identical"]
        assert r["stored_mode1_rel_diff"] == 0.0 and r["stored_mode2_raw_rel_diff"] == 0.0
        assert r["span_sin_max"] < 1e-10                   # 두 정책이 모드 3에 넘기는 스팬은 같다
        if r["arm"] in ("exact", "qr"):
            assert r["op_rel_diff"] < 1e-9                 # 기저에 무관한 사영은 같은 입력에 같은 결과
    assert min(r["op_rel_diff"] for r in rows if r["arm"] == "diagonal") > 1e-6   # 대각 규칙만 저장 기저를 탄다
    from eigen_benchmark.drivers import manifest
    assert len(manifest.read_csv(os.path.join(p4_both, "p4_policy_fields.csv"))) == 3 * 2


def test_projected_stats_rescore_both_policies_on_the_same_quadrature(p4_both):
    from eigen_benchmark.drivers import manifest
    ref = [{"mode": m, "lambda_extrapolated": v} for m, v in ((1, 1.0), (2, 2.0), (3, 3.0))]
    manifest.write_csv(os.path.join(p4_both, "p4_reference.csv"), ref)
    if not os.path.exists(os.path.join(p4_both, "p4_adf_neural_projected.csv")):
        ml.p4_adf_projected(outdir=p4_both, quick=True, device="cpu")
    if not os.path.exists(os.path.join(p4_both, "p4_adf_neural.csv")):
        ml.p4_adf(outdir=p4_both, quick=True, device="cpu")
    if not os.path.exists(os.path.join(p4_both, "p4_policy_fields.csv")):
        ml.p4_policy_fields(outdir=p4_both, quick=True, device="cpu")
    res = ml.projected_stats_from(p4_both, B=300)
    q = res["level"]
    assert q == 2                                          # quick의 최고 등급
    for arm in ml.P4_ARMS:
        r = res["rescore"][arm]
        assert set(r) == {"raw", "projected", "raw_minus_projected_mode3"}
        assert set(r["raw"]) == set(r["projected"]) == {1, 2, 3}
        assert r["raw_minus_projected_mode3"]["verdict"] in ("rule1", "rule2", "rule3")
        # 모드 1·2는 두 정책이 같은 망이다 — 재채점도 같아야 한다
        for m in (1, 2):
            assert r["raw"][m]["outcomes"] == r["projected"][m]["outcomes"]
    assert res["fields"]["n_networks_identical"] == res["fields"]["n_networks"] == 3 * 2 * 2
    assert res["projected_span"]["diagonal"]["dev_identity_max"] < 1e-10
    ml.projected_stats(outdir=p4_both, B=300)
    js = json.load(open(os.path.join(p4_both, "paperA_mlst_projected_stats.json"), encoding="utf-8"))
    assert js["prespecification"] == "2026-10-16-mlst-p4-projected-prespecification.md"


# ── 사후 검사(사전규정 밖, 결과를 본 뒤): 여섯 성분 배열 대조, 진폭에 무관한 모드 3 비교, 정확 규칙의 jitter 결손 ──

def test_p4_policy_posthoc_checks_all_six_components_and_scale_free_mode3(p4_both):
    out = ml.p4_policy_posthoc(outdir=p4_both, quick=True, device="cpu")
    rows = out["rows"]
    assert len(rows) == 3 * 2
    for r in rows:
        assert r["mode1_state_max_abs_diff"] == 0.0 and r["mode2_state_max_abs_diff"] == 0.0   # 값과 도함수 전부
        assert 0.0 <= r["mode3_mac_between_runs"] <= 1.0 + 1e-12
        assert r["mode3_amp_ratio"] > 0
        if r["arm"] == "exact":
            assert r["stage3_defect_jitter"] > 0
            assert abs(r["stage3_defect"] - r["stage3_defect_jitter"]) < 1e-12
        else:
            assert r["stage3_defect"] < 1e-12 and r["stage3_defect_jitter"] == 0.0
    from eigen_benchmark.drivers import manifest
    assert len(manifest.read_csv(os.path.join(p4_both, "p4_policy_posthoc.csv"))) == 3 * 2


def test_projected_posthoc_reads_the_symmetric_cells_and_the_posthoc_rows(p4_both):
    from eigen_benchmark.drivers import manifest
    if not os.path.exists(os.path.join(p4_both, "p4_reference.csv")):
        manifest.write_csv(os.path.join(p4_both, "p4_reference.csv"),
                           [{"mode": m, "lambda_extrapolated": v} for m, v in ((1, 1.0), (2, 2.0), (3, 3.0))])
    for f, fn in (("p4_adf_neural_projected.csv", ml.p4_adf_projected), ("p4_adf_neural.csv", ml.p4_adf),
                  ("p4_policy_posthoc.csv", ml.p4_policy_posthoc)):
        if not os.path.exists(os.path.join(p4_both, f)):
            fn(outdir=p4_both, quick=True, device="cpu")
    res = ml.projected_posthoc_from(p4_both)
    cells = res["adf_all_cells_projected"]
    assert set(cells) == {f"{a}_mode{m}" for a in ml.P4_ARMS for m in (1, 2, 3)}
    for v in cells.values():
        assert v["n"] == 2 and v["gate_crossings"] >= 0
    assert res["state_max_abs_diff"] == 0.0
    assert set(res["mode3_between_runs"]) == set(ml.P4_ARMS)
    for v in res["mode3_between_runs"].values():
        assert v["stage3_defect_vs_solver_max_abs"] < 1e-12      # 다시 잰 비직교가 솔버의 진단과 같다
    ml.projected_posthoc(outdir=p4_both)
    js = json.load(open(os.path.join(p4_both, "paperA_mlst_projected_posthoc.json"), encoding="utf-8"))
    assert js["status"].startswith("post hoc")
