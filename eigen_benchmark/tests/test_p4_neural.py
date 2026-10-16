"""P4 2차원 신경 arm — 구현이 옳은지부터 고정한다.

이 arm은 "mesh-free 주장이 시험되는 무대"의 결과를 내므로, 하드 BC나 Rayleigh 몫이 틀리면
논문의 결론이 무의미해진다. 그래서 성능이 아니라 **정확성**을 회귀로 잡는다.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from eigen_benchmark.neural import p4_neural as pn        # noqa: E402
from eigen_benchmark.problems import p4_lshape as p4      # noqa: E402


def _boundary_points(n: int = 7):
    pts = []
    for a, b in pn.EDGES:
        for s in np.linspace(0.03, 0.97, n):
            pts.append([a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])])
    return torch.tensor(np.array(pts))


def test_adf_vanishes_exactly_on_the_boundary():
    """하드 BC의 근거 — 비볼록 정의역에서 곱셈 다항식으로는 얻을 수 없는 성질."""
    assert float(pn.boundary_adf(_boundary_points()).abs().max()) == 0.0


def test_adf_is_strictly_positive_inside_including_near_the_reentrant_corner():
    inside = torch.tensor([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5],
                           [0.98, 0.98], [1.02, 0.5], [0.5, 1.02]])
    assert float(pn.boundary_adf(inside).min()) > 0.0


def test_trial_field_satisfies_the_clamped_condition_bitwise():
    ens = pn.EnsembleVecMLP(1, device="cpu")
    p0 = {k: v[0] for k, v in ens.params.items()}
    b = _boundary_points()
    u = torch.stack([torch.stack([pn._u_comp(p0, ens.base, t, j) for j in (0, 1)])
                     for t in b])
    assert float(u.abs().max()) == 0.0


def test_quadrature_integrates_the_L_shaped_area_exactly():
    """세 단위블록 텐서 Gauss — 면적 3을 정확히 내야 구적이 맞다."""
    _, w = pn.quadrature(12)
    assert float(w.sum()) == pytest.approx(3.0, abs=1e-12)


def test_quadrature_points_are_all_inside_the_domain():
    pts, _ = pn.quadrature(10)
    x, y = pts[:, 0].numpy(), pts[:, 1].numpy()
    assert np.all((x > 0) & (x < 2) & (y > 0) & (y < 2))
    assert not np.any((x > 1) & (y > 1)), "제외 사분면에 구적점이 있다"


def test_rayleigh_quotient_respects_the_upper_bound_property():
    """임의 시행함수의 Ω²는 최저 고유값 이상이어야 한다 — 약형식이 맞다는 필요조건."""
    pts, w = pn.quadrature(16)
    nu = p4.P4_GEOMETRY["nu"]
    lam1 = float(np.sort(p4.solve(5, beta=3.0, n_modes=3)["Lam"])[0])
    for sd in range(4):
        ens = pn.EnsembleVecMLP(1, seed=sd, device="cpu")
        p0 = {k: v[0] for k, v in ens.params.items()}
        with torch.no_grad():
            R = float(pn._rq_from_state(pn._state_one(p0, ens.base, pts), w, nu))
        assert R >= lam1, (sd, R, lam1)


def test_exact_projection_removes_the_previous_mode_in_the_mass_inner_product():
    """정확 질량-직교 사영 — 사영 후 앞 모드와의 질량 내적이 0이어야 한다."""
    pts, w = pn.quadrature(10)
    g = torch.Generator().manual_seed(0)
    P = torch.randn(2, 6, pts.shape[0], generator=g, dtype=torch.float64)
    S = torch.randn(6, pts.shape[0], generator=g, dtype=torch.float64)
    Sp = pn._project_exact(S, P, w)
    for a in range(P.shape[0]):
        ip = float((w * (P[a, 0] * Sp[0] + P[a, 1] * Sp[1])).sum())
        assert abs(ip) < 1e-10, (a, ip)


def test_projection_applies_the_same_coefficients_to_the_derivatives():
    """장만 빼고 도함수를 안 빼면 사영된 함수의 Rayleigh 몫이 그 함수의 것이 아니다."""
    pts, w = pn.quadrature(8)
    g = torch.Generator().manual_seed(1)
    P = torch.randn(1, 6, pts.shape[0], generator=g, dtype=torch.float64)
    S = torch.randn(6, pts.shape[0], generator=g, dtype=torch.float64)
    Sp = pn._project_exact(S, P, w)
    c = (S - Sp)[0] / P[0, 0]                     # 성분 0에서 역산한 계수
    for n in range(1, 6):                          # 나머지 다섯 성분도 같은 계수여야
        assert torch.allclose((S - Sp)[n], c * P[0, n], atol=1e-10), n


def test_fem_reference_can_be_evaluated_at_the_neural_quadrature_points():
    """MAC을 재려면 두 해를 같은 격자에서 봐야 한다 — 평가기가 빈틈없이 덮는지."""
    pts, w = pn.quadrature(12)
    r = p4.solve(5, beta=3.0, n_modes=2)
    U = p4.mode_at(r, 1, pts.numpy())
    assert not np.isnan(U).any(), int(np.isnan(U[:, 0]).sum())
    m = float((w.numpy() * (U[:, 0] ** 2 + U[:, 1] ** 2)).sum())
    assert m > 0 and math.isfinite(m)


def test_fem_reference_vanishes_on_the_clamped_boundary():
    r = p4.solve(5, beta=3.0, n_modes=1)
    b = _boundary_points(5).numpy()
    assert float(np.nanmax(np.abs(p4.mode_at(r, 1, b)))) == 0.0


def test_q2_assembly_still_works_after_adding_the_evaluator():
    """평가기를 붙이며 `_q2_shape`를 덮어써 조립이 조용히 깨진 적이 있다 — 회귀로 막는다."""
    mesh = p4.build_mesh(3, beta=2.0)
    K, M = p4.assemble(mesh)
    assert K.shape == M.shape and K.shape[0] == 2 * len(mesh["nodes"])
    assert float(abs(M).sum()) > 0


def test_p4_neural_and_fem_use_the_same_plane_stress_form():
    """FEM과 신경 arm이 같은 구성행렬·전단규약·질량밀도·단위두께를 쓰는가.

    FEM은 εᵀDε(공학전단 γ = u_y + v_x, D = E/(1−ν²)[[1,ν,0],[ν,1,0],[0,0,(1−ν)/2]])를,
    신경 arm은 전개식을 쓴다. 둘이 다르면 같은 물리문제를 푸는 것이 아니므로 P4 비교가
    무의미해진다. 두 형식의 적분항을 무작위 변형률에서 직접 대조한다."""
    import numpy as np

    from eigen_benchmark.neural.p4_neural import _rq_from_state
    from eigen_benchmark.problems.p4_lshape import P4_GEOMETRY, _plane_stress_D

    nu, E, rho = P4_GEOMETRY["nu"], P4_GEOMETRY["E"], P4_GEOMETRY["rho"]
    assert (E, rho) == (1.0, 1.0), "무차원 규약이 E = ρ = 1이다"
    D = _plane_stress_D(E, nu)
    rng = np.random.default_rng(0)
    for _ in range(500):
        ux, uy, vx, vy, u, v = rng.normal(size=6)
        eps = np.array([ux, vy, uy + vx])
        r_fem = float(eps @ D @ eps) / float(rho * (u ** 2 + v ** 2))
        S = torch.tensor([[u], [v], [ux], [uy], [vx], [vy]], dtype=torch.float64)
        r_nn = float(_rq_from_state(S, torch.ones(1, dtype=torch.float64), nu))
        assert abs(r_fem - r_nn) <= 1e-12 * abs(r_fem), (r_fem, r_nn)


def test_exact_projection_coefficients_are_constant_in_space():
    """사영 계수 c는 **구적 적분**이므로 공간미분에 대해 상수여야 한다.

    c가 실수로 공간에 의존하면 (b)의 φ″ 처리가 틀리고 중앙 결과가 무너진다. 확인 세 가지:
    c에 구적축이 없다, 같은 c가 장과 도함수에 함께 적용된다, 그리고 파라미터에 대해서는
    여전히 미분 가능하다."""
    from eigen_benchmark.neural import deflation as df
    from eigen_benchmark.neural.net import EnsembleMLP
    from eigen_benchmark.neural import core

    torch.set_default_dtype(torch.float64)
    n_q = 200
    xs = torch.linspace(1e-3, 1.0, n_q)
    wq = torch.full((n_q,), 1.0 / n_q)
    ens = EnsembleMLP(n_seeds=3, width=8, depth=2, device="cpu")
    phi_all, d2_all = core.eval_phi(ens, xs)
    Phi, Phi2 = phi_all[:2], d2_all[:2]
    phi = phi_all[2].clone().requires_grad_(True)
    d2 = d2_all[2]

    c = df.project_exact(phi, Phi, wq)
    assert c.shape == (2,), f"c에 구적축이 있다: {tuple(c.shape)}"
    ph_t, d2_t, _ = df.ProjectionExact().apply(phi, d2, Phi, Phi2, wq)
    # φ̃″ = φ″ − cᵀΦ″ 가 정확히 성립해야 한다(c가 x에 무관하므로)
    assert torch.equal(d2_t, d2 - c @ Phi2)
    # 저장 span과 질량직교
    leak = torch.einsum("ki,i,i->k", Phi, wq, ph_t).abs().max()
    assert float(leak) < 1e-14, float(leak)
    # 파라미터(장)에 대해서는 미분 가능해야 한다 — detach가 없어야 한다
    g = torch.autograd.grad(c.sum(), phi, allow_unused=True)[0]
    assert g is not None and float(g.abs().sum()) > 0


def _fake_state(n_prev: int = 2, seed: int = 3):
    pts, w = pn.quadrature(8)
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(n_prev, 6, pts.shape[0], generator=g, dtype=torch.float64)
    S = torch.randn(6, pts.shape[0], generator=g, dtype=torch.float64)
    return S, P, w


def test_p4_exact_path_is_unchanged_by_the_strategy_refactor():
    """(b)의 커밋된 결과가 흔들리면 안 된다 — 비트 단위로 같은 경로여야 한다."""
    S, P, w = _fake_state()
    assert torch.equal(pn.project(S, P, w, "exact"), pn._project_exact(S, P, w))


def test_every_p4_strategy_applies_one_coefficient_set_to_all_six_components():
    """장만 빼고 변형률을 안 빼면 사영된 시행함수의 Rayleigh 몫이 그 함수의 것이 아니다."""
    S, P, w = _fake_state(n_prev=1)
    for kind in ("exact", "diagonal", "qr"):
        d = S - pn.project(S, P, w, kind)
        c = d[0] / P[0, 0]
        for n in range(1, 6):
            assert torch.allclose(d[n], c * P[0, n], atol=1e-10), (kind, n)


def test_p4_qr_and_exact_agree_in_fp64_because_they_are_the_same_projection():
    S, P, w = _fake_state(n_prev=3)
    a = pn.project(S, P, w, "exact")
    b = pn.project(S, P, w, "qr")
    assert float((a - b).abs().max()) < 1e-9 * float(S.abs().max())


def test_p4_diagonal_leaves_a_residual_when_the_stored_modes_are_not_orthogonal():
    """2차원에서도 대각근사는 사영이 아니다 — (a′)를 P4에 올리는 근거."""
    S, P, w = _fake_state(n_prev=1)
    P2 = torch.stack([P[0], 0.7 * P[0] + 0.3 * torch.roll(P[0], 5, dims=-1)])
    leak_d = max(abs(float((w * (P2[a, 0] * pn.project(S, P2, w, "diagonal")[0]
                                 + P2[a, 1] * pn.project(S, P2, w, "diagonal")[1])).sum()))
                 for a in range(2))
    leak_e = max(abs(float((w * (P2[a, 0] * pn.project(S, P2, w, "exact")[0]
                                 + P2[a, 1] * pn.project(S, P2, w, "exact")[1])).sum()))
                 for a in range(2))
    assert leak_e < 1e-9
    assert leak_d > 1e-3


def test_solve_p4_neural_accepts_a_projection_choice_and_names_it():
    out = pn.solve_p4_neural(n_modes=2, n_seeds=2, iters=2, n_per_block=4,
                             projection="qr", device="cpu")
    assert "qr" in out["arm"]
    assert np.isfinite(out["lam"]).all()


def test_p4_driver_writes_the_added_arms_to_their_own_file(tmp_path):
    """커밋된 `p4_neural.csv`(arm (b))는 건드리지 않고 새 arm만 따로 쓴다."""
    from eigen_benchmark.drivers import run_p4_neural as rp4

    out = rp4.run_arms(outdir=str(tmp_path), arms=("diagonal", "qr"), quick=True,
                       device="cpu")
    rows = out["p4_arms"]
    assert {r["arm"] for r in rows} == {"p4_neural_2d[diagonal projection]",
                                        "p4_neural_2d[qr projection]"}
    assert (tmp_path / "p4_neural_arms.csv").exists()
    assert not (tmp_path / "p4_neural.csv").exists()
    assert {"p_correct", "wilson_lo", "wilson_hi", "e_lam_median", "mac_median",
            "lower_mode_basin", "E_T_success"} <= set(rows[0])


def test_p4_added_arms_manifest_declares_the_jitter_asymmetry(tmp_path):
    """(b)에만 있는 1e-14 jitter를 공시하지 않으면 3-arm 비교가 공정하지 않다."""
    import json
    from eigen_benchmark.drivers import run_p4_neural as rp4

    rp4.run_arms(outdir=str(tmp_path), arms=("qr",), quick=True, device="cpu")
    m = json.loads((tmp_path / "manifest_p4_neural_arms.json").read_text())
    assert "jitter" in json.dumps(m)
    assert "2026-08-26" in m["added"]


def test_p4_reports_the_stored_span_diagnostics_it_was_handed():
    """2차원에서 대각규칙이 무너지지 않는다면 그 이유는 저장 스팬의 직교성이다 —
    설명할 수 있으려면 그 값을 재야 한다."""
    out = pn.solve_p4_neural(n_modes=3, n_seeds=2, iters=2, n_per_block=4,
                             device="cpu")
    sd = out["span_diagnostics"]
    assert len(sd) == 3 and sd[0] is None
    assert sd[2]["k"] == 2
    assert {"kappa_gram", "dev_identity", "qr_orth_defect", "rank_1e12",
            "n_seeds_summarized"} <= set(sd[2])


def test_p4_span_diagnostics_use_the_two_component_mass_inner_product():
    """질량 내적은 u·v 두 성분을 함께 쓴다 — 한 성분만 쓰면 값이 틀린다."""
    import numpy as np
    _, P, w = _fake_state(n_prev=2)
    d = pn.span_diagnostics_2d(P, w)
    G = torch.einsum("aij,j,bij->ab", P[:, :2], w, P[:, :2]).double()
    ev = torch.linalg.eigvalsh(0.5 * (G + G.T))
    assert d["kappa_gram"] == pytest.approx(
        float(ev.amax() / ev.amin()), rel=1e-6)


def test_p4_span_diagnostic_driver_writes_its_own_file_only(tmp_path):
    """진단 실행은 채점 데이터와 섞이면 안 된다 — 시드 수가 다르기 때문이다."""
    from eigen_benchmark.drivers import run_p4_neural as rp4

    out = rp4.run_span_diagnostics(outdir=str(tmp_path), projection="diagonal",
                                   n_seeds=2, iters=2, n_per_block=4, device="cpu")
    rows = out["p4_span"]
    assert [r["stage_mode"] for r in rows] == [2, 3]
    assert {"kappa_gram", "dev_identity", "qr_orth_defect", "rank_1e10",
            "rank_1e12", "rank_1e16", "n_seeds_summarized"} <= set(rows[0])
    assert (tmp_path / "p4_span_diagnostics.csv").exists()
    assert not (tmp_path / "p4_neural_arms.csv").exists()
    assert not (tmp_path / "p4_neural.csv").exists()


def test_p4_single_seed_cost_covers_each_projection_rule(tmp_path):
    """3-arm 비교의 비용 칸을 비워 두지 않으려면 arm마다 실측이 있어야 한다."""
    from eigen_benchmark.drivers import manifest
    from eigen_benchmark.drivers import run_p4_neural as rp4

    out = rp4.single_seed_cost(outdir=str(tmp_path), projections=("diagonal", "qr"),
                               quick=True, device="cpu")
    arms = {r["arm"] for r in out["single_seed"]}
    assert arms == {"p4_neural_2d[diagonal projection]", "p4_neural_2d[qr projection]"}
    rows = manifest.read_csv(str(tmp_path / "p1_single_seed_cost.csv"))
    assert {r["arm"] for r in rows} == arms
    assert all(float(r["seconds_per_attempt"]) > 0 for r in rows)


def test_p4_single_seed_cost_still_defaults_to_the_exact_arm(tmp_path):
    """기존 호출부(인자 없음)가 그대로 동작해야 한다."""
    from eigen_benchmark.drivers import run_p4_neural as rp4

    out = rp4.single_seed_cost(outdir=str(tmp_path), quick=True, device="cpu")
    assert {r["arm"] for r in out["single_seed"]} == {"p4_neural_2d[exact projection]"}


def test_p4_span_diagnostics_summarize_every_seed_by_default():
    """P1 경로와 같은 규칙이어야 한다 — 한쪽만 고치면 두 문제의 진단이 다른 관례로 남는다.
    실제로 그렇게 남았고, P4만 앞 8시드를 요약하고 있었다."""
    import torch

    from eigen_benchmark.neural.p4_neural import span_diagnostics_2d
    torch.manual_seed(0)
    n_seeds, k, n_q = 12, 3, 16
    P = torch.randn(n_seeds, k, 6, n_q, dtype=torch.float64)
    w = torch.rand(n_q, dtype=torch.float64)
    out = span_diagnostics_2d(P, w)
    assert out["n_seeds_summarized"] == n_seeds


def test_p4_span_diagnostics_can_still_be_capped_explicitly():
    import torch

    from eigen_benchmark.neural.p4_neural import span_diagnostics_2d
    torch.manual_seed(0)
    P = torch.randn(12, 3, 6, 16, dtype=torch.float64)
    w = torch.rand(16, dtype=torch.float64)
    assert span_diagnostics_2d(P, w, n_diag_seeds=5)["n_seeds_summarized"] == 5


def test_check_block_reports_the_same_eigenvalue_at_a_second_quadrature_order():
    """P5에서 학습된 기저가 고정 인증격자를 이겼고, P1도 같은 계기로 재서 고차모드
    드리프트를 찾았다. P4는 하한 스캔에서 69셀을 잡았으나 두 구적차수 계기가 없었다 —
    Λ가 Λ₁ 위에 있으면서 과소적분된 셀은 하한 스캔에 걸리지 않는다."""
    out = pn.solve_p4_neural(n_modes=2, n_seeds=4, iters=200, width=16, depth=2,
                              check_n_per_block=40)
    assert out["check_n_per_block"] == 40
    for k in ("lam_check", "quadrature_drift"):
        assert k in out, k
        assert np.asarray(out[k]).shape == (2, 4), (k, np.asarray(out[k]).shape)
    assert np.isfinite(out["quadrature_drift"]).all()


def test_check_block_is_off_by_default_so_committed_p4_runs_are_unchanged():
    out = pn.solve_p4_neural(n_modes=1, n_seeds=2, iters=50, width=16, depth=2)
    assert out.get("check_n_per_block") is None
    assert "quadrature_drift" not in out
