"""(b′) 드라이버 스모크 — 출력은 tmp_path로만 쓴다(커밋 데이터 보호)."""
import numpy as np
import pytest

pytest.importorskip("torch")
from eigen_benchmark.drivers import run_p1_qr    # noqa: E402


def test_quick_run_writes_the_qr_sweep_with_the_standard_columns(tmp_path):
    """표 병합이 되려면 열 구조가 mode_sweep과 같아야 한다."""
    out = run_p1_qr.main(outdir=str(tmp_path), quick=True, device="cpu")
    rows = out["qr_sweep"]
    assert rows
    assert {"arm", "mode", "n", "p_correct", "wilson_lo", "wilson_hi", "correct",
            "lower_mode_basin", "spurious", "non_converged", "p_accurate",
            "seconds", "iters"} <= set(rows[0])
    assert all(r["arm"] == "b_prime_projection_qr" for r in rows)
    assert (tmp_path / "p1_neural_qr.csv").exists()
    assert (tmp_path / "manifest_p1_qr.json").exists()


def test_quick_run_records_span_diagnostics_for_every_stage_that_had_a_span(tmp_path):
    out = run_p1_qr.main(outdir=str(tmp_path), quick=True, device="cpu")
    diag = out["diagnostics"]
    assert [r["stage_mode"] for r in diag] == [2, 3]      # 모드 1은 스팬이 없다
    assert {"kappa_gram", "kappa_gram_times_eps", "qr_orth_defect", "n_dropped",
            "rank_1e10", "rank_1e12", "rank_1e16",
            "proj_agreement_rel"} <= set(diag[0])
    assert (tmp_path / "p1_qr_span_diagnostics.csv").exists()


def test_the_manifest_declares_the_arm_as_added_after_the_pre_specification(tmp_path):
    """사후 추가는 날짜와 함께 선언한다 — 설계기록 §7.1 규약."""
    import json
    run_p1_qr.main(outdir=str(tmp_path), quick=True, device="cpu")
    m = json.loads((tmp_path / "manifest_p1_qr.json").read_text())
    assert "2026-08-26" in m["added"]
    assert "not changed" in m["added"]


def test_fp32_stress_labels_the_dtype_and_keeps_both_arms(tmp_path):
    out = run_p1_qr.run_fp32_stress(outdir=str(tmp_path), quick=True, device="cpu")
    rows = out["fp32"]
    assert rows, "fp32 실행이 한 행도 남기지 못했다"
    assert {r["dtype"] for r in rows} == {"float32"}
    assert {r["arm"] for r in rows} == {"b_projection_exact", "b_prime_projection_qr"}
    assert (tmp_path / "p1_fp32_stress.csv").exists()
    assert (tmp_path / "p1_fp32_span_diagnostics.csv").exists()
    # 진단이 fp32 정밀도를 실제로 보고하는지
    assert all(float(r["eps_machine"]) > 1e-8 for r in out["diagnostics"])


def test_single_seed_cost_merges_without_dropping_the_other_arms(tmp_path):
    from eigen_benchmark.drivers import manifest
    path = tmp_path / "p1_single_seed_cost.csv"
    manifest.write_csv(str(path), [{"arm": "b_projection_exact", "mode": 1,
                                    "iters": 4000, "reps": 3,
                                    "seconds_per_attempt": 40.0,
                                    "seconds_min": 39.0, "seconds_max": 44.0,
                                    "spread_rel": 0.1, "measured": "x"}])
    run_p1_qr.single_seed_cost(outdir=str(tmp_path), quick=True, device="cpu")
    arms = {r["arm"] for r in manifest.read_csv(str(path))}
    assert arms == {"b_projection_exact", "b_prime_projection_qr"}


def test_the_stress_driver_takes_the_precision_as_a_parameter(tmp_path):
    """정밀도 축이 두 점에서 멈추지 않도록 dtype을 인자로 받는다."""
    import torch
    out = run_p1_qr.run_precision_stress(outdir=str(tmp_path), dtype=torch.bfloat16,
                                         tag="bf16", quick=True, device="cpu")
    assert (tmp_path / "p1_bf16_stress.csv").exists()
    assert (tmp_path / "p1_bf16_span_diagnostics.csv").exists()
    # 그람 경로는 bf16에 구현이 없다 — 그 사실이 데이터로 남아야 한다
    notes = {n["arm"]: n for n in out["notes"]}
    assert notes["b_projection_exact"]["aborted"] is True
    assert "not implemented" in notes["b_projection_exact"]["error"].lower()
    assert notes["b_prime_projection_qr"]["n_modes_completed"] > 0


def test_run_fp32_stress_still_works_as_before(tmp_path):
    out = run_p1_qr.run_fp32_stress(outdir=str(tmp_path), quick=True, device="cpu")
    assert (tmp_path / "p1_fp32_stress.csv").exists()
    assert {r["dtype"] for r in out["fp32"]} == {"float32"}


def test_the_stress_driver_saves_enough_to_rescore_without_rerunning(tmp_path):
    """채점 규칙이나 참조가 바뀌면 재채점이 필요하다. 형상을 버리면 그때마다 몇 시간을
    다시 태워야 한다 — 실제로 해석 참조 결함 때문에 그 일이 있었다."""
    import numpy as np
    out = run_p1_qr.run_precision_stress(outdir=str(tmp_path), tag="t", quick=True,
                                         arms=["b_prime_projection_qr"], device="cpu")
    npz = tmp_path / "p1_t_b_prime_projection_qr_shapes.npz"
    assert npz.exists()
    z = np.load(npz)
    assert {"shapes", "lam", "xs", "wq", "converged"} <= set(z.files)
    assert z["shapes"].ndim == 3                      # (mode, seed, n_q)
    assert z["shapes"].shape[0] == z["lam"].shape[0]
    assert (tmp_path / "p1_t_records.jsonl").exists()
    # 저장분만으로 다시 채점할 수 있는가
    from eigen_benchmark.neural import sequential as sq
    recs = sq.classify_stage(z["shapes"][0], z["lam"][0], target=1,
                             xs=z["xs"], wq=z["wq"], converged=z["converged"][0])
    assert len(recs) == z["shapes"].shape[1]


def test_the_stress_driver_can_measure_at_a_second_quadrature_order(tmp_path):
    """P5에서 학습된 기저가 고정 인증격자를 이겼다. P1에 그 병리가 **없다**고 말하려면
    같은 계기로 재야 한다 — 없음의 주장은 계기 없이 성립하지 않는다. 셀마다 드리프트가
    기록·표·재채점본 셋 다에 실려야 나중에 표에서 구별할 수 있다."""
    import numpy as np
    out = run_p1_qr.run_precision_stress(outdir=str(tmp_path), tag="t2", quick=True,
                                         arms=["b_prime_projection_qr"],
                                         check_n_q=512, device="cpu")
    z = np.load(tmp_path / "p1_t2_b_prime_projection_qr_shapes.npz")
    assert {"lam_check", "quad_drift"} <= set(z.files), z.files
    assert z["quad_drift"].shape == z["lam"].shape

    import json
    with open(tmp_path / "p1_t2_records.jsonl", encoding="utf-8") as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    assert recs and all("quad_drift" in r for r in recs)

    for r in out["rows"]:
        assert "quad_drift_median" in r and "quad_drift_max" in r, r.keys()
        assert float(r["quad_drift_median"]) >= 0.0


def test_the_second_quadrature_order_is_off_by_default(tmp_path):
    """커밋된 실행의 결과를 바꾸지 않는다 — 계기를 붙인 것과 붙이지 않은 것이
    같은 열 이름을 쓰면 어느 쪽인지 알 수 없게 된다."""
    out = run_p1_qr.run_precision_stress(outdir=str(tmp_path), tag="t3", quick=True,
                                         arms=["b_prime_projection_qr"], device="cpu")
    assert all("quad_drift_median" not in r for r in out["rows"])
