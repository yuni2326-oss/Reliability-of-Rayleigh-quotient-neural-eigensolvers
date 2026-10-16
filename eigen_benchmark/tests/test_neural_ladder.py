import numpy as np
import pytest

torch = pytest.importorskip("torch")
from eigen_benchmark.neural import core, ladder, net    # noqa: E402
from eigen_benchmark.problems import p1_beam as p1      # noqa: E402


def test_i0_is_no_initializer():
    assert ladder.make_ladder("I0") is None


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        ladder.make_ladder("I9")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU 없으면 느림")
def test_i5_oracle_fits_both_value_and_curvature():
    """값만 맞추면 K = ∫(φ″)²가 부정확해진다 — H² 적합이어야 한다."""
    dev = "cuda"
    ens = net.EnsembleMLP(2, seed=0, device=dev)
    xs, wq = core.gauss_nodes(128, dev)
    ladder.make_ladder("I5", mode=2, iters=2500)(ens, 1, xs, wq)
    with torch.no_grad():
        ph, d2 = core.eval_phi(ens, xs)
    x = xs.cpu().numpy(); w = wq.cpu().numpy()
    tgt = p1.analytic_mode(x, 2); tgt = tgt / np.sqrt((w * tgt ** 2).sum())
    got = ph[0].cpu().numpy()
    s = np.sign((w * got * tgt).sum())
    assert np.sqrt((w * (s * got - tgt) ** 2).sum()) < 5e-2
    ref = p1.beta_roots(2)[1] ** 4
    assert abs(float(core.rayleigh(ph, d2, wq)[0]) - ref) / ref < 0.05


def test_i3_uses_a_computable_non_oracle_warm_start():
    """I3는 4–6항 조립 Ritz — 해석해를 쓰지 않는다(oracle이 아니다)."""
    from eigen_benchmark.bases.monomial import MonomialBasis
    r = p1.solve(MonomialBasis(5), n_q=200, n_modes=1)
    assert r["cholesky_ok"]
    init = ladder.make_ladder("I3", mode=1, n_terms=5)
    assert "analytic" not in (init.__doc__ or "").lower()


def test_i2_returns_a_window_barrier_not_an_initializer():
    b = ladder.make_ladder("I2", lam_lo=100.0, lam_hi=1000.0)
    assert callable(b)
    assert float(b(torch.tensor(500.0, dtype=torch.float64))) == pytest.approx(0.0)
    assert float(b(torch.tensor(5.0, dtype=torch.float64))) > 0.0
    assert float(b(torch.tensor(5000.0, dtype=torch.float64))) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU 없으면 느림")
def test_i1_satisfies_clamped_bc_and_is_not_the_analytic_mode():
    dev = "cuda"
    ens = net.EnsembleMLP(2, seed=3, device=dev)
    xs, wq = core.gauss_nodes(128, dev)
    ladder.make_ladder("I1", mode=1, iters=800)(ens, 0, xs, wq)
    z = torch.zeros(1, dtype=torch.float64, device=dev)
    with torch.no_grad():
        ph0, _ = core.eval_phi(ens, z)
        ph, d2 = core.eval_phi(ens, xs)
    assert torch.allclose(ph0, torch.zeros_like(ph0), atol=1e-13)
    ref = p1.beta_roots(1)[0] ** 4
    assert abs(float(core.rayleigh(ph, d2, wq)[0]) - ref) / ref > 1e-3


def test_all_declared_levels_are_constructible():
    for lvl in ladder.LEVELS:
        ladder.make_ladder(lvl, mode=1, iters=10)      # 예외 없이 만들어져야 한다


def test_i2_barrier_reaches_the_objective():
    """I2는 초기화가 아니라 목적함수 항이다 — `solve_sequential(barrier=)` 배선이 없으면
    사다리에서 조용히 빠진다(실제로 `("I0","I1","I3","I5")`만 돌고 있었다)."""
    from eigen_benchmark.neural import deflation as df
    from eigen_benchmark.neural import sequential as sq
    bar = ladder.make_ladder("I2", mode=1, lam_lo=1e9, lam_hi=2e9, barrier_weight=1e-6)
    assert callable(bar)
    kw = dict(n_seeds=2, iters=8, snapshot_every=8, device="cpu")
    a = sq.solve_sequential(1, df.NoDeflation(), **kw)
    b = sq.solve_sequential(1, df.NoDeflation(), barrier=bar, **kw)
    assert not np.allclose(a["lam"], b["lam"], rtol=0, atol=0), \
        "barrier를 줬는데 결과가 비트 단위로 같다 — 목적함수에 도달하지 않는다"
