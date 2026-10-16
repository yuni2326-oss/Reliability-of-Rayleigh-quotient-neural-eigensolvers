import numpy as np
import pytest

torch = pytest.importorskip("torch")
from eigen_benchmark.neural import core, net          # noqa: E402
from eigen_benchmark.problems import p1_beam as p1    # noqa: E402


def test_gauss_nodes_integrate_polynomials_exactly():
    xs, wq = core.gauss_nodes(64, device="cpu")
    got = float((wq * xs ** 9).sum())
    assert got == pytest.approx(0.1, rel=1e-13)


def test_hard_boundary_conditions_hold_for_every_seed():
    ens = net.EnsembleMLP(4, seed=0, device="cpu")
    xs = torch.zeros(1, dtype=torch.float64)
    phi, _ = core.eval_phi(ens, xs)
    assert phi.shape == (4, 1)
    assert torch.allclose(phi, torch.zeros_like(phi), atol=1e-14)


def test_rayleigh_of_the_analytic_mode_equals_beta_fourth():
    """코어 정합성의 기준: 해석 모드형을 넣으면 RQ가 정확히 β⁴여야 한다."""
    xs, wq = core.gauss_nodes(256, device="cpu")
    x = xs.cpu().numpy()
    for n in (1, 2, 3):
        phi = torch.tensor(p1.analytic_mode(x, n)).reshape(1, -1)
        d2 = torch.tensor(p1.analytic_mode_d2(x, n)).reshape(1, -1)
        r = float(core.rayleigh(phi, d2, wq)[0])
        assert r == pytest.approx(p1.beta_roots(3)[n - 1] ** 4, rel=1e-8)


def test_train_records_snapshots_for_time_to_accuracy():
    ens = net.EnsembleMLP(3, seed=0, device="cpu")
    xs, wq = core.gauss_nodes(64, device="cpu")

    def loss(p):
        ph, d2 = core.eval_phi_one(p, ens.base, xs)
        return (wq * d2 ** 2).sum() / (wq * ph ** 2).sum()

    out = core.train(ens, loss, iters=50, snapshot_every=10)
    assert len(out["history"]) >= 5
    it, sec, r = out["history"][-1]
    assert it <= 50 and sec > 0 and r.shape == (3,)


def test_training_reduces_the_rayleigh_quotient():
    ens = net.EnsembleMLP(3, seed=1, device="cpu")
    xs, wq = core.gauss_nodes(64, device="cpu")

    def loss(p):
        ph, d2 = core.eval_phi_one(p, ens.base, xs)
        return (wq * d2 ** 2).sum() / (wq * ph ** 2).sum()

    out = core.train(ens, loss, iters=200, snapshot_every=50)
    first, last = out["history"][0][2], out["history"][-1][2]
    assert np.median(last) < np.median(first)


def test_mc_nodes_have_unit_total_weight():
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    xs, wq = core.mc_nodes(500, g, device="cpu")
    assert float(wq.sum()) == pytest.approx(1.0, rel=1e-12)
    assert xs.shape == (500,)


def test_disclosure_reports_architecture_for_appendix_b():
    ens = net.EnsembleMLP(5, width=32, depth=3, seed=2, device="cpu")
    d = ens.disclosure()
    assert d["width"] == 32 and d["depth"] == 3 and d["n_seeds"] == 5
    assert d["dtype"] == "float64" and "x^2" in d["hard_bc"]
    assert d["n_params_per_seed"] > 0
