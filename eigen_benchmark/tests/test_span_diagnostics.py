"""저장 스팬 진단 — 이후 단계의 stabilization criterion이 여기서 나온다."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from eigen_benchmark.neural import diagnostics as dg    # noqa: E402


def _w(n=400):
    x, w = np.polynomial.legendre.leggauss(n)
    return (torch.tensor(0.5 * x + 0.5), torch.tensor(0.5 * w))


def test_orthonormal_span_reports_unit_conditioning_and_full_rank():
    xs, wq = _w()
    Phi = torch.stack([np.sqrt(2) * torch.sin(np.pi * xs),
                       np.sqrt(2) * torch.sin(2 * np.pi * xs)])
    d = dg.span_diagnostics(Phi, Phi, wq)
    assert d["k"] == 2
    assert abs(d["kappa_gram"] - 1.0) < 1e-8
    assert d["dev_identity"] < 1e-10
    assert d["qr_orth_defect"] < 1e-12
    assert d["rank_1e10"] == 2 and d["rank_1e12"] == 2 and d["rank_1e16"] == 2
    assert d["n_dropped"] == 0
    assert d["gram_solve_finite"] is True


def test_rank_is_reported_at_each_threshold_separately():
    """단일 정수 rank는 임계를 숨긴다 — 세 임계에서 답이 갈리는 스팬으로 고정한다."""
    xs, wq = _w()
    v1, v2 = torch.sin(np.pi * xs), torch.sin(2 * np.pi * xs)
    Phi = torch.stack([v1, v1 + 1e-7 * v2, v1 + 1e-13 * v2])
    d = dg.span_diagnostics(Phi, Phi, wq)
    assert d["rank_1e10"] < d["rank_1e16"], d


def test_projection_agreement_is_machine_zero_in_fp64():
    """(b)와 (b′)이 같은 사영이라는 주장을 실행마다 재는 열."""
    xs, wq = _w()
    Phi = torch.stack([torch.sin(np.pi * xs), torch.sin(2 * np.pi * xs)])
    Phi2 = torch.stack([-(np.pi ** 2) * torch.sin(np.pi * xs),
                        -4 * (np.pi ** 2) * torch.sin(2 * np.pi * xs)])
    phi = torch.sin(5 * np.pi * xs) + Phi[0]
    d2 = -((5 * np.pi) ** 2) * torch.sin(5 * np.pi * xs) + Phi2[0]
    d = dg.span_diagnostics(Phi, Phi2, wq, phi=phi, d2=d2)
    assert d["proj_agreement_rel"] < 1e-12


def test_a_duplicate_direction_is_counted_as_dropped():
    xs, wq = _w(200)
    v = torch.sin(np.pi * xs)
    d = dg.span_diagnostics(torch.stack([v, v]), torch.stack([v, v]), wq)
    assert d["n_dropped"] == 1
    assert d["rank_1e12"] == 1


def test_the_stabilization_product_is_reported_so_the_criterion_can_be_read_off():
    """κ₂(G)·u는 그람 경로가 언제 무너지는지의 예측량이다 — 값으로 남긴다."""
    xs, wq = _w()
    Phi = torch.stack([torch.sin(np.pi * xs), torch.sin(np.pi * xs) + 1e-6 *
                       torch.sin(2 * np.pi * xs)])
    d = dg.span_diagnostics(Phi, Phi, wq)
    assert d["kappa_gram_times_eps"] == pytest.approx(
        d["kappa_gram"] * float(np.finfo(np.float64).eps), rel=1e-9)


def test_sequential_logs_span_diagnostics_at_every_store():
    """모든 순차 실행이 자기 진단을 남긴다 — 별도 진단 실행이 필요 없어야 한다."""
    from eigen_benchmark.neural import deflation as df
    from eigen_benchmark.neural import sequential as sq

    out = sq.solve_sequential(3, df.ProjectionExact(), n_seeds=2, iters=5,
                              n_q=64, device="cpu")
    sd = out["span_diagnostics"]
    assert len(sd) == 3
    assert sd[0] is None                      # 모드 1은 넘겨받은 스팬이 없다
    assert sd[2]["k"] == 2 and sd[2]["kappa_gram"] >= 1.0
    assert {"qr_orth_defect", "rank_1e10", "rank_1e12", "rank_1e16",
            "proj_agreement_rel", "kappa_gram_times_eps",
            "n_seeds_summarized"} <= set(sd[2])
    assert sd[2]["proj_agreement_rel"] < 1e-10


def test_sequential_runs_in_float32_when_asked():
    from eigen_benchmark.neural import deflation as df
    from eigen_benchmark.neural import sequential as sq

    out = sq.solve_sequential(2, df.ProjectionQR(), n_seeds=2, iters=5, n_q=64,
                              dtype=torch.float32, device="cpu")
    assert out["dtype"] == "float32"
    assert np.isfinite(out["lam"]).all()
    assert out["span_diagnostics"][1]["eps_machine"] > 1e-8


def test_float64_remains_the_default_so_existing_runs_are_unchanged():
    from eigen_benchmark.neural import deflation as df
    from eigen_benchmark.neural import sequential as sq

    out = sq.solve_sequential(1, df.NoDeflation(), n_seeds=2, iters=3, n_q=32,
                              device="cpu")
    assert out["dtype"] == "float64"


def test_reduced_precision_runs_and_reports_that_the_gram_solve_is_unavailable():
    """bf16에는 인수분해 루틴이 없다 — QR 경로는 돌고 그람 경로는 못 돈다는 사실이
    데이터로 남아야 한다. 진단의 스펙트럼 계산은 측정이므로 fp64로 올린다."""
    from eigen_benchmark.neural import deflation as df
    from eigen_benchmark.neural import sequential as sq

    out = sq.solve_sequential(3, df.ProjectionQR(), n_seeds=2, iters=5, n_q=64,
                              dtype=torch.bfloat16, device="cpu")
    assert out["dtype"] == "bfloat16"
    assert np.isfinite(out["lam"]).all()
    sd = out["span_diagnostics"][2]
    assert sd["kappa_gram"] >= 1.0                      # fp64로 측정됐다
    assert sd["qr_orth_defect"] >= 0.0
    assert sd["gram_solve_finite_frac"] == 0.0          # 그람 경로는 사용 불가
    assert sd["measure_dtype"] == "float64"


def test_the_measurement_precision_is_recorded_so_the_convention_is_visible():
    xs, wq = _w(200)
    Phi = torch.stack([torch.sin(np.pi * xs), torch.sin(2 * np.pi * xs)])
    d64 = dg.span_diagnostics(Phi, Phi, wq)
    assert d64["measure_dtype"] == "float64"
    d32 = dg.span_diagnostics(Phi.float(), Phi.float(), wq.float())
    assert d32["measure_dtype"] == "float32", "fp32 진단은 기존 관례를 유지해야 한다"


def test_a_non_finite_stored_span_is_reported_not_raised():
    """저장 스팬이 발산하면 그것이 결과다 — 진단이 예외로 실행을 죽이면 그 사실이
    데이터에서 사라지고, 몇 모드까지 갔는지도 알 수 없게 된다."""
    xs, wq = _w(200)
    good = torch.sin(np.pi * xs)
    bad = torch.full_like(good, float("nan"))
    d = dg.span_diagnostics(torch.stack([good, bad]), torch.stack([good, bad]), wq)
    assert d["span_finite"] is False
    assert d["kappa_gram"] != d["kappa_gram"]          # NaN
    assert d["k"] == 2


def test_a_finite_span_is_flagged_finite():
    xs, wq = _w(200)
    Phi = torch.stack([torch.sin(np.pi * xs), torch.sin(2 * np.pi * xs)])
    assert dg.span_diagnostics(Phi, Phi, wq)["span_finite"] is True


def test_solving_against_a_fixed_span_deflates_exactly_against_that_span():
    """저장 집합을 해석해로 주입할 수 있어야 '누적 오차'와 '그 단계의 풀이'를 가른다."""
    from eigen_benchmark.neural import deflation as df
    from eigen_benchmark.neural import sequential as sq
    from eigen_benchmark.problems import p1_beam as p1

    out = sq.solve_fixed_span(3, df.ProjectionExact(), n_ref_modes=2, n_seeds=2,
                              iters=5, n_q=128, device="cpu")
    assert {"lam", "shapes", "converged", "xs", "wq", "target"} <= set(out)
    assert out["target"] == 3
    assert out["shapes"].shape[0] == 2
    xs, wq = out["xs"], out["wq"]
    PHI = np.stack([p1.analytic_mode(xs, n) for n in (1, 2)])
    leak = np.abs((PHI * wq) @ out["shapes"].T).max()
    assert leak < 1e-10, leak          # 주입한 스팬과 질량직교여야 한다


def test_the_fixed_span_solver_records_that_the_span_is_analytic():
    from eigen_benchmark.neural import deflation as df
    from eigen_benchmark.neural import sequential as sq

    out = sq.solve_fixed_span(3, df.ProjectionExact(), n_ref_modes=2, n_seeds=2,
                              iters=3, n_q=128, device="cpu")
    assert "analytic" in out["span_source"]


def test_diagnostics_summarize_over_every_seed_in_the_cell_by_default():
    """요약 시드 수가 표마다 다르면 같은 양이 두 값을 갖는다 — 실제로 arm (b)의
    kappa(G)가 표에 따라 2~3배 달랐다. 기본을 '전 시드'로 고정한다."""
    from eigen_benchmark.neural import deflation as df
    from eigen_benchmark.neural import sequential as sq

    out = sq.solve_sequential(3, df.ProjectionExact(), n_seeds=12, iters=5, n_q=64,
                              device="cpu")
    assert out["span_diagnostics"][2]["n_seeds_summarized"] == 12


def test_the_summary_seed_count_can_still_be_capped_explicitly():
    from eigen_benchmark.neural import deflation as df
    from eigen_benchmark.neural import sequential as sq

    out = sq.solve_sequential(3, df.ProjectionExact(), n_seeds=12, iters=5, n_q=64,
                              n_diag_seeds=2, device="cpu")
    assert out["span_diagnostics"][2]["n_seeds_summarized"] == 2


def test_diagnostics_report_the_max_off_diagonal_of_the_normalized_gram():
    """P1의 `p1_orthogonality.csv`가 max off-diag를 싣고, P5의 갈래 판정 문턱도
    그 양을 지목한다(0.10 / 0.01). 스펙트럼 지표 `dev_identity`로 대체하면 사전 고정된
    문턱이 지목한 양과 재는 양이 달라진다. k = 2에서는 둘이 일치하고 그보다 깊으면 갈린다."""
    import numpy as np
    import torch

    from eigen_benchmark.neural import diagnostics as dgn
    w = torch.ones(64, dtype=torch.float64) / 64
    # 서로 강하게 겹치는 두 방향 — max off-diag가 커야 한다
    t = torch.linspace(0, 1, 64, dtype=torch.float64)
    Phi = torch.stack([t, t + 0.05 * torch.sin(6 * t)], 0)
    d = dgn.span_diagnostics(Phi, Phi, w)
    assert "max_offdiag" in d
    assert 0.0 <= d["max_offdiag"] <= 1.0
    # k = 2이므로 단위대각 C의 ‖I − C‖₂와 일치한다
    assert abs(d["max_offdiag"] - d["dev_identity"]) < 1e-10, (
        d["max_offdiag"], d["dev_identity"])
    # 직교한 두 방향이면 0에 가깝다
    Q = torch.stack([torch.ones(64, dtype=torch.float64),
                     torch.cos(torch.pi * t) * 0 + torch.sign(t - 0.5)], 0)
    assert dgn.span_diagnostics(Q, Q, w)["max_offdiag"] < 1e-12
