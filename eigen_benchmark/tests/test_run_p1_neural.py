import json

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU 필요")

from eigen_benchmark.drivers import run_p1_neural as rn      # noqa: E402


@pytest.fixture(scope="module")
def quick(tmp_path_factory):
    d = tmp_path_factory.mktemp("neural")
    return rn.main(outdir=str(d), quick=True), d


def test_quick_run_writes_expected_artifacts(quick):
    res, d = quick
    for name in ("p1_neural_mode_sweep.csv", "p1_neural_penalty_sweep.csv",
                 "p1_neural_quadrature.csv", "p1_neural_ladder.csv",
                 "p1_neural_cost.csv", "p1_neural_records.jsonl",
                 "manifest_p1_neural.json"):
        assert (d / name).exists()
    with open(d / "manifest_p1_neural.json") as f:
        m = json.load(f)
    assert "git_sha" in m and "gpu" in m and "grid_note" in m


def test_every_record_carries_the_preregistered_fields(quick):
    res, _ = quick
    r = res["records"][0]
    for key in ("arm", "mode", "seed", "outcome", "matched_mode", "mac",
                "e_lam", "seconds", "nodes", "ladder"):
        assert key in r


def test_outcomes_use_only_the_preregistered_categories(quick):
    from eigen_benchmark.metrics import OUTCOMES
    res, _ = quick
    assert {r["outcome"] for r in res["records"]} <= set(OUTCOMES)


def test_wilson_intervals_are_reported_for_every_cell(quick):
    res, _ = quick
    for row in res["mode_sweep"]:
        if row["n"] < 2:            # PIELM은 단일 결정론 실행이라 구간이 없다
            continue
        assert 0.0 <= row["p_correct"] <= 1.0
        assert row["wilson_lo"] <= row["p_correct"] <= row["wilson_hi"]


def test_expected_cost_charges_failures(quick):
    res, _ = quick
    for row in res["cost"]:
        if row["p_correct"] == 0.0:
            assert row["E_T_success"] == float("inf")
        else:
            assert row["E_T_success"] >= row["mean_seconds"] * (1 - 1e-12)


def test_all_seven_arms_are_represented(quick):
    res, _ = quick
    arms = {r["arm"].split("(")[0] for r in res["mode_sweep"]}
    for expected in ("a_prime_projection_diagonal", "b_projection_exact",
                     "c_curriculum", "d_neural_basis_galerkin",
                     "e_simultaneous_subspace", "f_eig_pielm"):
        assert expected in arms, f"{expected} 누락"
    pen_arms = {r["arm"].split("(")[0] for r in res["penalty"]}
    assert "a_penalty" in pen_arms


def test_long_run_requires_the_base_grid(tmp_path):
    import pytest as _pt
    with _pt.raises(FileNotFoundError):
        rn.run_long(outdir=str(tmp_path), quick=True)


def test_long_run_targets_only_failing_cells(quick):
    """장기변주 셀 선택은 임의가 아니라 기본 격자에서 **데이터로** 결정된다 —
    성공률 100 % 미만인 셀만 고른다."""
    res, d = quick
    out = rn.run_long(outdir=str(d), quick=True)
    import csv
    base = list(csv.DictReader(open(d / "p1_neural_mode_sweep.csv", encoding="utf-8")))
    bad = {(r["arm"], int(r["mode"])) for r in base
           if float(r["p_correct"]) < 0.9 and int(r["n"]) > 1}
    for row in out["longrun"]:
        assert (row["arm"], row["mode"]) in bad
        assert row["variant"] == "long_run"
        assert "p_correct_base" in row and "budget_resolved" in row


def test_per_mode_iteration_budget_is_honoured():
    """이미 100 %인 저차모드까지 5배로 돌리는 것은 낭비다 — 모드별 예산이 실제로 적용되는지."""
    from eigen_benchmark.neural import deflation as df
    from eigen_benchmark.neural import sequential as sq
    out = sq.solve_sequential(2, df.ProjectionExact(), n_seeds=3,
                              iters=[60, 200], snapshot_every=20)
    assert out["iters"] == [60, 200]
    assert out["stage_seconds"][1] > out["stage_seconds"][0]


def test_mismatched_budget_length_raises():
    from eigen_benchmark.neural import deflation as df
    from eigen_benchmark.neural import sequential as sq
    with pytest.raises(ValueError):
        sq.solve_sequential(3, df.NoDeflation(), n_seeds=2, iters=[10, 20])
