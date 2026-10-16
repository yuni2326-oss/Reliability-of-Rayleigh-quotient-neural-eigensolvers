"""P4 저장 규약·코너 계기(MLST 재투고 계산, 사전규정 2026-10-14-mlst-prespecification.md).

커밋된 P4는 학습한 단계의 **사영된** 상태를 저장한다(그래서 세 규칙이 같은 사영이다). P1의 기전을 2차원에서
시험하려면 P1처럼 **원시** 출력을 저장해야 한다. 코너 계기는 재진입 코너 쪽으로 등급을 준 구적, FEM 참조 모드의
기울기, ADF의 코너 기울기다. CPU 소형 설정으로 돈다.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from eigen_benchmark.neural import p4_neural as pn        # noqa: E402
from eigen_benchmark.problems import p4_lshape as p4      # noqa: E402

KW = dict(n_seeds=2, iters=6, n_per_block=6, width=16, depth=2, device="cpu", snapshot_every=2)


def test_raw_storage_keeps_modes_one_and_two_and_changes_what_mode_three_is_deflated_against():
    """모드 1은 사영이 없고 모드 2는 저장 모드 하나(St₁ = u₁)에 사영하므로 두 규약에서 같다. 모드 3부터 원시 쌍과
    사영된 쌍이 갈린다 — 대각 규칙은 달라지고, 정확 사영은 같은 스팬이라 그대로다."""
    for proj in ("diagonal", "exact"):
        a = pn.solve_p4_neural(n_modes=3, projection=proj, **KW)
        b = pn.solve_p4_neural(n_modes=3, projection=proj, storage="raw", **KW)
        np.testing.assert_array_equal(a["lam"][:2], b["lam"][:2])
        np.testing.assert_array_equal(a["shapes"][:2], b["shapes"][:2])
        if proj == "diagonal":
            assert not np.allclose(a["lam"][2], b["lam"][2], rtol=1e-6)
        else:
            np.testing.assert_allclose(a["lam"][2], b["lam"][2], rtol=1e-8)
    assert a["storage"] == "projected" and b["storage"] == "raw"


def test_span_diagnostics_measure_what_was_stored():
    """사영 저장이면 모드 3에 넘겨지는 집합은 구성상 직교이고, 원시 저장이면 원시 쌍의 비직교가 보인다."""
    a = pn.solve_p4_neural(n_modes=3, projection="diagonal", **KW)
    b = pn.solve_p4_neural(n_modes=3, projection="diagonal", storage="raw", **KW)
    assert a["span_diagnostics"][2]["dev_identity"] < 1e-12
    assert b["span_diagnostics"][2]["dev_identity"] > 1e-6


def test_per_seed_span_rows_carry_the_raw_pair_and_the_applied_leakage():
    out = pn.solve_p4_neural(n_modes=3, projection="diagonal", storage="raw", per_seed_span=True, **KW)
    rows = out["span_per_seed"]
    assert {r["stage_mode"] for r in rows} == {2, 3} and {r["seed"] for r in rows} == {0, 1}
    r3 = [r for r in rows if r["stage_mode"] == 3]
    for r in r3:
        assert r["k"] == 2 and r["max_offdiag"] == pytest.approx(r["dev_identity"], rel=1e-9)
        assert r["leak_rel"] > 0 and np.isfinite(r["kappa_gram"])
    ex = pn.solve_p4_neural(n_modes=3, projection="exact", storage="raw", per_seed_span=True, **KW)
    assert max(r["leak_rel"] for r in ex["span_per_seed"] if r["stage_mode"] == 3) < 1e-10


def test_returned_parameters_rebuild_each_stage_network():
    out = pn.solve_p4_neural(n_modes=2, projection="exact", return_params=True, **KW)
    assert len(out["params"]) == 2
    ens = pn.EnsembleVecMLP(2, 16, 2, seed=0, device="cpu")
    for k_, v in out["params"][1].items():
        ens.params[k_] = torch.as_tensor(v)
    pts = torch.as_tensor(out["pts"])
    S = torch.func.vmap(lambda p: pn._state_one(p, ens.base, pts))(ens.params)
    np.testing.assert_allclose(S[:, :2].numpy(), out["raw_shapes"][1], rtol=1e-12, atol=1e-14)


def test_default_storage_is_the_committed_projected_path():
    a = pn.solve_p4_neural(n_modes=2, projection="diagonal", **KW)
    b = pn.solve_p4_neural(n_modes=2, projection="diagonal", storage="projected", **KW)
    np.testing.assert_array_equal(a["lam"], b["lam"])
    with pytest.raises(ValueError):
        pn.solve_p4_neural(n_modes=2, projection="diagonal", storage="orthogonal", **KW)


def test_corner_graded_quadrature_integrates_polynomials_and_refines_toward_the_corner():
    for layers in (0, 2, 4):
        pts, w = pn.quadrature_graded(6, layers, device="cpu")
        P, W = pts.numpy(), w.numpy()
        assert W.sum() == pytest.approx(3.0, rel=1e-13)
        # ∫ x² y dΩ = A 1/6 + B 7/6 + C 1/2 = 11/6 — 셀마다 6점 가우스면 정확하다.
        assert float((W * P[:, 0] ** 2 * P[:, 1]).sum()) == pytest.approx(11.0 / 6.0, rel=1e-12)
        assert (P[:, 0] > 0).all() and (P[:, 1] > 0).all()
        assert not ((P[:, 0] > 1) & (P[:, 1] > 1)).any()        # 정의역 밖(첫 사분면 쪽)은 없다
    d = [np.hypot(*(pn.quadrature_graded(6, L, device="cpu")[0].numpy() - 1.0).T).min() for L in (0, 2, 4)]
    assert d[0] > d[1] > d[2]


def test_fem_mode_gradient_matches_finite_differences():
    res = p4.solve(4, beta=3.0, n_modes=2)
    mesh, free = res["mesh"], res["free_dofs"]
    full = np.zeros(2 * len(mesh["nodes"]))
    full[free] = res["vec"][:, 0]
    pts = np.array([[0.3137, 0.4211], [1.6123, 0.2719], [0.5173, 1.7123]])
    g = p4.eval_mode_grad_at(mesh, full, pts)                   # (n, 2, 2): [성분, d/dx·d/dy]
    h = 1e-6
    for j, (dx, dy) in enumerate(((h, 0.0), (0.0, h))):
        fd = (p4.eval_mode_at(mesh, full, pts + [dx, dy]) - p4.eval_mode_at(mesh, full, pts - [dx, dy])) / (2 * h)
        np.testing.assert_allclose(g[:, :, j], fd, rtol=1e-5, atol=1e-7)


def test_adf_gradient_stays_bounded_toward_the_reentrant_corner():
    """ADF는 꼭짓점에서 매끄럽지 않지만 1차 미분은 유계다 — 평면응력 에너지는 1차 미분만 쓴다."""
    rows = pn.adf_corner_profile(radii=(1e-1, 1e-3, 1e-5), n_angles=7)
    g = np.array([r["grad_max"] for r in rows])
    assert np.isfinite(g).all() and g.max() < 10.0 and g.min() > 1e-3
    phi = np.array([r["phi_max"] for r in rows])
    assert phi[0] > phi[1] > phi[2]                             # 코너에서 0으로 간다
