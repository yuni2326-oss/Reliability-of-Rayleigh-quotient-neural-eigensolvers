"""참조 인증 — 채점이 참조에서 읽는 **모든 양**이 인증돼야 한다."""
import pytest

from eigen_benchmark.drivers import manifest
from eigen_benchmark.drivers import run_reference_certification as rc


DATA = "docs/_generated/data/paper2"


@pytest.fixture(scope="module")
def cert(tmp_path_factory):
    """출력은 임시 디렉터리로 — 커밋된 데이터를 덮지 않는다. 입력은 실데이터를 복사한다."""
    import os
    import shutil
    out = str(tmp_path_factory.mktemp("cert"))
    for f in ("p2_reference.csv",):
        shutil.copy(os.path.join(DATA, f), out)
    return rc.main(outdir=out, n_modes_p1=16)


def test_every_scored_reference_quantity_is_certified(cert):
    """고유값만 인증하고 모드형을 빼면, MAC 게이트가 인증되지 않은 양에 걸린다."""
    rows = cert["rows"]
    kinds = {(r["problem"], r["quantity_certified"]) for r in rows}
    assert ("P1", "mode shape") in kinds
    assert ("P1", "mode curvature") in kinds
    assert ("P2", "mode shape") in kinds
    assert ("P4", "mode shape") in kinds


def test_the_p1_shape_reference_is_certified_over_every_mode_the_scorer_can_use(cert):
    """`classify_stage`의 기본 n_ref는 14다 — 그 위까지 덮어야 한다."""
    modes = {int(r["label"].split()[-1]) for r in cert["rows"]
             if r["problem"] == "P1" and r["quantity_certified"] == "mode shape"}
    assert modes >= set(range(1, 15))


def test_all_certifications_pass_their_threshold(cert):
    bad = [(r["problem"], r["label"], r["quantity_certified"], r["deficit"])
           for r in cert["rows"] if not r["passes"]]
    assert not bad, bad[:4]


def test_the_tolerance_matches_the_kind_of_reference(cert):
    """닫힌 형식은 반올림 수준, 이산해는 이산화 수준 — 같은 잣대를 쓰면 한쪽이 무의미해진다."""
    tol = {(r["problem"], r["quantity_certified"]): r["tolerance"] for r in cert["rows"]}
    assert tol[("P1", "mode shape")] <= 1e-9
    assert tol[("P2", "mode shape")] <= 1e-9
    assert 1e-9 < tol[("P4", "mode shape")] <= 1e-2
    # 어느 쪽이든 MAC 게이트 여유 0.1에 비해 무시할 만해야 한다
    worst = max(r["deficit"] for r in cert["rows"])
    assert worst < 0.01, worst


def test_the_certification_is_written_with_its_comparison_basis(cert):
    import os
    rows = manifest.read_csv(os.path.join(cert["outdir"], "reference_certification.csv"))
    assert rows
    for r in rows:
        assert r["certified_against"], r
        assert r["deficit"] != ""


def test_p3_is_recorded_as_out_of_scope_rather_than_uncertified(cert):
    """규칙은 "모든 참조를 인증하라"가 아니라 "채점이 읽는 모든 양을 인증하라"다.
    P3가 읽는 양은 고유값뿐이므로 인증 대상이 "모드형"이면 어색하다 — 해당 없음이 맞다."""
    p3 = [r for r in cert["rows"] if r["problem"] == "P3"]
    assert p3
    assert all("not applicable" in r["quantity_certified"] for r in p3)
    assert any("no per-seed classification" in r["note"] for r in p3)
    assert any("only the eigenvalue" in r["note"] for r in p3)


def test_the_local_p4_blocks_match_the_solver_definition():
    """드라이버는 torch 없이 돌아야 해서 블록을 지역 정의한다 — 어긋나면 인증이 무의미하다."""
    torch = pytest.importorskip("torch")
    from eigen_benchmark.neural import p4_neural as pn
    assert tuple(tuple(map(tuple, b)) for b in pn.BLOCKS) == rc.P4_BLOCKS
