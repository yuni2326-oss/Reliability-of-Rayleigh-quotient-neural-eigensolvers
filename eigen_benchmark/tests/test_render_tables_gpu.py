"""공시 사본이 코드와 일치하는지 — torch가 있는 환경에서만 검증한다."""
import pytest

torch = pytest.importorskip("torch")
from eigen_benchmark.neural.net import EnsembleMLP          # noqa: E402
from eigen_benchmark.neural.spacetime import SpaceTimeNet   # noqa: E402
from eigen_benchmark.render.tables import DISCLOSED         # noqa: E402


def test_disclosure_defaults_match_the_code():
    """부록 B는 torch 없는 호스트에서 렌더링되므로 값의 사본을 둔다 —
    사본이 낡으면 논문 부록이 조용히 틀린다. 여기서 강제한다."""
    d = EnsembleMLP(1, device="cpu").disclosure()
    assert d["width"] == DISCLOSED["width"]
    assert d["depth"] == DISCLOSED["depth"]
    assert str(d.get("activation", "tanh")).lower().startswith("tanh")


def test_spacetime_parameter_count_matches():
    net = SpaceTimeNet(DISCLOSED["spacetime_width"], DISCLOSED["spacetime_depth"])
    assert sum(p.numel() for p in net.parameters()) == DISCLOSED["spacetime_params"]


def test_p4_disclosure_matches_the_code():
    """부록 B의 P4 행은 torch 없는 호스트에서 렌더링되므로 사본을 둔다 —
    사본이 낡으면 논문 부록이 조용히 틀린다(실제로 12866으로 잘못 적었다)."""
    from eigen_benchmark.neural.p4_neural import EnsembleVecMLP, quadrature
    d = EnsembleVecMLP(1, DISCLOSED["p4_width"], DISCLOSED["p4_depth"],
                       device="cpu").disclosure()
    assert d["n_params_per_seed"] == DISCLOSED["p4_params"]
    pts, _ = quadrature(DISCLOSED["p4_n_per_block"])
    assert int(pts.shape[0]) == DISCLOSED["p4_n_q"]
