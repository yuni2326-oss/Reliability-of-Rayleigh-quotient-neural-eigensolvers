import numpy as np
import pytest

torch = pytest.importorskip("torch")
from eigen_benchmark.neural import curriculum as cu     # noqa: E402
from eigen_benchmark.neural import deflation as df      # noqa: E402
from eigen_benchmark.problems import p1_beam as p1      # noqa: E402


def test_coarse_ritz_lambda_is_close_but_not_analytic():
    """λ*는 조립 Ritz에서 온다 — 해석해가 아니어야 한다(비-oracle)."""
    exact = p1.beta_roots(3)[2] ** 4
    lam = cu.coarse_ritz_lambda(3, n_terms=5)
    assert lam > exact                      # Ritz는 상한
    assert abs(lam - exact) / exact > 1e-8  # 해석해와 구별됨
    assert abs(lam - exact) / exact < 0.5   # 그래도 쓸 만한 창


def test_curriculum_returns_the_same_keys_as_sequential():
    out = cu.solve_curriculum(2, df.ProjectionExact(), n_seeds=3, iters=120)
    for k in ("lam", "shapes", "history", "converged", "xs", "wq", "seconds",
              "arm"):
        assert k in out
    assert out["lam"].shape[0] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU 필요")
def test_curriculum_reaches_the_targeted_mode():
    out = cu.solve_curriculum(3, df.ProjectionExact(), n_seeds=8, iters=4000)
    b3 = p1.beta_roots(3)[2]
    assert np.median(np.abs(out["lam"][2] - b3) / b3) < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU 필요")
def test_stage1_pulls_the_quotient_toward_the_target():
    """1단만 돌리면 R이 λ*로 끌려가야 한다(모드형 정보 없이 고유값만)."""
    tgt = p1.beta_roots(2)[1] ** 4
    out = cu.solve_curriculum(1, df.NoDeflation(), n_seeds=6, iters=2500,
                              stage1_frac=1.0, lam_targets=[tgt])
    r = out["lam"][0] ** 4
    assert np.median(np.abs(r - tgt) / tgt) < 0.5


def test_stage_fraction_is_recorded_for_disclosure():
    out = cu.solve_curriculum(1, df.NoDeflation(), n_seeds=3, iters=100,
                              stage1_frac=0.3)
    assert out["stage1_frac"] == pytest.approx(0.3)
    assert out["arm"].startswith("c_curriculum")


def test_history_spans_both_stages():
    out = cu.solve_curriculum(1, df.NoDeflation(), n_seeds=3, iters=100,
                              stage1_frac=0.5, snapshot_every=10)
    its = [h[0] for h in out["history"][0]]
    assert min(its) == 0 and max(its) >= 90
    assert len(its) > len(set(range(0, 51, 10)))       # 2단 스냅샷이 이어붙음
