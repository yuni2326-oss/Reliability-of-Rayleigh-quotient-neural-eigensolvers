import numpy as np
import pytest

torch = pytest.importorskip("torch")
from eigen_benchmark.neural import core                 # noqa: E402
from eigen_benchmark.neural import subspace as sub      # noqa: E402
from eigen_benchmark.problems import p1_beam as p1      # noqa: E402


def _analytic_basis(k, n_q=256, device="cpu"):
    xs, wq = core.gauss_nodes(n_q, device)
    x = xs.cpu().numpy()
    Phi = torch.tensor(np.stack([p1.analytic_mode(x, n) for n in range(1, k + 1)]))
    Phi2 = torch.tensor(np.stack([p1.analytic_mode_d2(x, n) for n in range(1, k + 1)]))
    return Phi.to(device), Phi2.to(device), xs, wq


def test_trace_of_the_exact_eigenbasis_equals_the_sum_of_eigenvalues():
    Phi, Phi2, xs, wq = _analytic_basis(3)
    got = float(sub.subspace_trace(Phi, Phi2, wq))
    want = float((p1.beta_roots(3) ** 4).sum())
    assert got == pytest.approx(want, rel=1e-8)


def test_galerkin_recovers_individual_eigenvalues_from_a_mixed_basis():
    """기저를 섞어도 Galerkin은 개별 고유값을 복원해야 한다(span만 맞으면 됨)."""
    Phi, Phi2, xs, wq = _analytic_basis(3)
    Mx = torch.tensor([[1.0, 0.4, 0.1], [0.0, 1.0, 0.3], [0.2, 0.0, 1.0]],
                      dtype=torch.float64)
    lam4, _, rank = sub.galerkin_solve(Mx @ Phi, Mx @ Phi2, wq)
    assert rank == 3
    assert np.allclose(lam4.cpu().numpy(), p1.beta_roots(3) ** 4, rtol=1e-8)


def test_trace_is_minimized_by_the_lowest_eigenspace():
    """최저 3개 고유공간의 trace가, 하나를 4번째로 바꾼 공간보다 작아야 한다."""
    Phi, Phi2, xs, wq = _analytic_basis(4)
    lo = float(sub.subspace_trace(Phi[:3], Phi2[:3], wq))
    sw = torch.stack([Phi[0], Phi[1], Phi[3]])
    sw2 = torch.stack([Phi2[0], Phi2[1], Phi2[3]])
    assert lo < float(sub.subspace_trace(sw, sw2, wq))


def test_multi_output_net_satisfies_hard_bc_on_every_output():
    net = sub.MultiOutputMLP(2, n_out=4, seed=0, device="cpu")
    z = torch.zeros(1, dtype=torch.float64)
    with torch.no_grad():
        phi, _ = sub.eval_multi(net, z)
    assert phi.shape == (2, 4, 1)
    assert torch.allclose(phi, torch.zeros_like(phi), atol=1e-14)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU 필요")
def test_simultaneous_arm_recovers_the_three_lowest_modes():
    out = sub.solve_simultaneous(3, n_seeds=6, iters=4000)
    ref = p1.beta_roots(3)
    for m in range(3):
        assert np.median(np.abs(out["lam"][m] - ref[m]) / ref[m]) < 5e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU 필요")
def test_neural_basis_galerkin_recovers_the_three_lowest_modes():
    out = sub.solve_neural_basis_galerkin(3, n_basis=6, n_seeds=4, iters=3000)
    ref = p1.beta_roots(3)
    for m in range(3):
        assert np.median(np.abs(out["lam"][m] - ref[m]) / ref[m]) < 1e-1


def test_both_arms_return_sequential_compatible_keys():
    out = sub.solve_simultaneous(2, n_seeds=3, iters=60, device="cpu")
    for k in ("lam", "shapes", "history", "converged", "xs", "wq", "seconds",
              "arm"):
        assert k in out
    assert out["lam"].shape[0] == 2
    assert out["shapes"].shape[0] == 2


def test_galerkin_survives_a_rank_deficient_basis():
    """**본 실행에서 실제로 터진 구성**: 학습된 신경기저가 수치적으로 rank 결손이 되어
    그람행렬이 양정부호를 잃는다(M=9). trace 목적함수는 span만 보므로 개별 기저함수가
    서로 겹쳐도 벌점이 없다 — Cholesky로 풀면 크래시하고, rank 절단으로 풀어야 한다."""
    Phi, Phi2, xs, wq = _analytic_basis(3)
    dup = torch.cat([Phi, Phi[:1] * (1 + 1e-15), Phi[1:2]])      # 의도적 중복
    dup2 = torch.cat([Phi2, Phi2[:1] * (1 + 1e-15), Phi2[1:2]])
    w, V, rank = sub.galerkin_solve(dup, dup2, wq)
    assert rank == 3                                   # 중복분이 제거됨
    assert np.allclose(w.cpu().numpy(), p1.beta_roots(3) ** 4, rtol=1e-6)


def test_subspace_arms_report_effective_rank():
    out = sub.solve_simultaneous(2, n_seeds=3, iters=60, device="cpu")
    assert "rank_median" in out and "rank_min" in out
    assert out["rank_min"] >= 0


def test_missing_modes_are_nan_not_filled():
    """rank가 부족하면 그 모드는 NaN이어야 한다 — 0이나 마지막 값으로 채우면
    표가 거짓말을 하고, NaN은 사전등록 규칙에서 non_converged로 잡힌다."""
    Phi, Phi2, xs, wq = _analytic_basis(2)
    dup = torch.cat([Phi[:1], Phi[:1]])
    dup2 = torch.cat([Phi2[:1], Phi2[:1]])
    lam4, shapes, ranks, curv = sub._extract(dup.unsqueeze(0), dup2.unsqueeze(0), wq,
                                             n_seeds=1, n_modes=3)
    assert ranks[0] == 1
    assert np.isfinite(lam4[0, 0]) and np.all(np.isnan(lam4[1:, 0]))
    # 곡률도 같은 규칙을 따라야 한다 — 도달 못한 모드는 채우지 않는다
    assert np.all(np.isfinite(curv[0, 0])) and np.all(np.isnan(curv[1:, 0]))


def test_gram_offdiagonal_penalty_is_zero_for_orthogonal_and_positive_for_duplicates():
    """trace는 span만 보므로 중복에 벌점이 없다 — 이 항이 없던 압력을 공급한다."""
    Phi, Phi2, xs, wq = _analytic_basis(3)
    assert float(sub.gram_offdiagonal_penalty(Phi, wq)) < 1e-12
    dup = torch.stack([Phi[0], Phi[0] * 1.7, Phi[1]])
    assert float(sub.gram_offdiagonal_penalty(dup, wq)) > 1.0


def test_penalty_is_scale_invariant():
    """상관 정규화했으므로 기저함수 크기를 바꿔도 벌점은 같아야 한다."""
    Phi, Phi2, xs, wq = _analytic_basis(3)
    a = float(sub.gram_offdiagonal_penalty(Phi, wq))
    scaled = torch.stack([Phi[0] * 1e3, Phi[1] * 1e-3, Phi[2]])
    b = float(sub.gram_offdiagonal_penalty(scaled, wq))
    assert abs(a - b) < 1e-10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU 필요")
def test_orth_penalty_changes_the_arm_label_and_is_recorded():
    out = sub.solve_neural_basis_galerkin(2, n_basis=4, n_seeds=3, iters=100,
                                          w_orth=1e4)
    assert "w_orth" in out["arm"] and out["w_orth"] == 1e4
    assert "rank_median" in out


def test_orthogonality_penalty_actually_reaches_the_gradients():
    """`w_orth`가 옵티마이저에 **연결되어 있는지** — 조용한 단절을 잡는 회귀.

    감사에서 나온 문제다. ablation은 벌점을 trace의 70배까지 올려도 rank가 4.0으로
    불변이었는데, 그 결과는 **배선이 끊겨 있어도 똑같이** 나온다. 코드를 읽어
    연결을 확인했지만(detach 없이 backward), 그것을 지키는 테스트가 없었다.

    같은 시드로 w_orth만 바꿔 몇 스텝 돌리면 결과가 달라야 한다 — 비트 단위로
    같으면 벌점이 손실에 도달하지 않은 것이다."""
    kw = dict(n_basis=3, n_seeds=2, iters=5, snapshot_every=5, device="cpu")
    a = sub.solve_neural_basis_galerkin(2, w_orth=0.0, **kw)["basis_shapes"]
    b = sub.solve_neural_basis_galerkin(2, w_orth=1e4, **kw)["basis_shapes"]
    assert a.shape == b.shape
    assert not np.allclose(a, b, rtol=0, atol=0), \
        "w_orth를 바꿨는데 기저가 비트 단위로 같다 — 벌점이 손실에 도달하지 않는다"
    # 그리고 실제로 비대각을 **줄이는** 방향이어야 한다(부호 확인)
    _, wq = core.gauss_nodes(a.shape[-1], "cpu")
    pen = [float(np.median([sub.gram_offdiagonal_penalty(
        torch.tensor(P[s]), wq) for s in range(P.shape[0])])) for P in (a, b)]
    assert pen[1] <= pen[0] * 1.001, f"벌점이 비대각을 늘렸다: {pen}"


def test_appendix_c_objective_claims_match_the_code():
    """부록 C가 두 목적함수 구현을 정확히 적는지 — 소스와 직접 대조한다."""
    import inspect

    from eigen_benchmark.neural import subspace
    from eigen_benchmark.render import tables

    s_ = tables.appendix_c_numerics().replace("\\|", "|")
    src_lt = inspect.getsource(subspace.subspace_logtrace)
    src_tr = inspect.getsource(subspace.subspace_trace)
    # 로그 형태는 slogdet + 1e6 벌점 가드를 쓴다
    assert "slogdet" in src_lt and "1e6" in src_lt
    assert "fixed penalty of 1e6" in s_
    # 절대 trace는 solve의 대각합이고 가드가 없다
    assert "linalg.solve" in src_tr and "diagonal" in src_tr
    assert "torch.linalg.solve(M_Φ, K_Φ)" in s_
    assert "The absolute-trace form has no such guard" in s_
    # 두 함수 모두 jitter·pinv를 쓰지 않는다
    for bad in ("jitter", "pinv", "pseudo"):
        assert bad not in src_lt.lower() and bad not in src_tr.lower(), bad
    assert "No jitter, no regularization and no pseudoinverse are used" in s_
    # 대칭화가 실제로 들어 있다
    assert "0.5 * (B + B.T)" in src_tr and "0.5 * (A + A.T)" in src_lt
    assert "symmetrized as ½(X + Xᵀ)" in s_
