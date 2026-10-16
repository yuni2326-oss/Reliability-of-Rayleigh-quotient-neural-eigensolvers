import json
import os

import pytest

from eigen_benchmark.drivers import manifest, run_p1, run_p3


def test_manifest_has_provenance_fields():
    m = manifest.build()
    for key in ("git_sha", "git_dirty", "timestamp_utc", "python", "numpy",
                "scipy", "mpmath", "platform"):
        assert key in m


def test_manifest_accepts_extra_fields():
    m = manifest.build({"gpu_note": "vLLM 3개 상주"})
    assert m["gpu_note"] == "vLLM 3개 상주"


def test_run_p1_quick_writes_expected_artifacts(tmp_path):
    res = run_p1.main(outdir=str(tmp_path), quick=True)
    assert (tmp_path / "p1_reference.csv").exists()
    assert (tmp_path / "p1_basis_study.csv").exists()
    assert (tmp_path / "manifest_p1.json").exists()
    assert len(res["reference"]) == 10
    with open(tmp_path / "manifest_p1.json") as f:
        assert "git_sha" in json.load(f)


def test_run_p1_reference_is_dimensionless_and_increasing(tmp_path):
    """논문 경계: 참조표는 무차원 Λ=(βL)⁴만 낸다 — Hz 열은 없다."""
    res = run_p1.main(outdir=str(tmp_path), quick=True)
    assert not any("hz" in k.lower() for k in res["reference"][0])
    lam = [r["Lambda"] for r in res["reference"]]
    assert lam == sorted(lam)
    assert abs(lam[0] - 1.875104068711961 ** 4) / lam[0] < 1e-12


def test_run_p3_quick_writes_expected_artifacts(tmp_path):
    res = run_p3.main(outdir=str(tmp_path), quick=True)
    for name in ("p3_reference.csv", "p3_basis_study.csv",
                 "p3_conditioning.csv", "p3_quadrature_separation.csv",
                 "manifest_p3.json"):
        assert (tmp_path / name).exists()
    assert len(res["reference"]) >= 3


def test_run_p3_records_saturation_of_c1_bases(tmp_path):
    res = run_p3.main(outdir=str(tmp_path), quick=True)
    rows = [r for r in res["basis_study"]
            if r["jump_capable"] is False and r["cholesky_ok"]]
    assert rows, "C¹ 기저 행이 있어야 한다"
    # C¹ 기저는 스프링 기준 대비 상대오차가 1% 이상 남는다(포화)
    assert min(r["e_lam_mode1"] for r in rows) > 1e-2


def test_run_p3_records_convergence_of_jump_capable_bases(tmp_path):
    res = run_p3.main(outdir=str(tmp_path), quick=True)
    rows = [r for r in res["basis_study"]
            if r["jump_capable"] is True and r["cholesky_ok"]]
    assert rows
    assert min(r["e_lam_mode1"] for r in rows) < 1e-4


def test_run_p3_conditioning_rows_carry_all_four_normalizations(tmp_path):
    res = run_p3.main(outdir=str(tmp_path), quick=True)
    r = res["conditioning"][0]
    for key in ("kappa_M_raw", "kappa_M_massnorm", "kappa_K_equilibrated",
                "kappa_A_transformed", "cholesky_ok", "highprec_path"):
        assert key in r


def test_classical_cost_includes_basis_construction(tmp_path):
    """비용표의 분모 — 신경은 '처음부터 학습'으로 재므로 고전도 '처음부터'여야 한다.

    이전에는 기저를 timed 밖에서 만들어 정규직교화 QR(n=6에서 70 ms)이 빠지고 풀이
    0.105 ms만 셌다. 실제 비용의 1/400이라 자릿수 주장이 무너진다."""
    res = run_p1.main(outdir=str(tmp_path), quick=True)
    rows = res["basis_study"]
    assert rows
    for r in rows:
        for k in ("seconds_construct", "seconds_solve", "seconds"):
            assert k in r, k
        assert r["seconds"] == pytest.approx(r["seconds_construct"]
                                             + r["seconds_solve"], rel=1e-12)
    # 정규직교 단항식은 구성이 풀이보다 비싸다 — 그것이 이 수정의 이유다
    orth = [r for r in rows if r["basis"] == "monomial_orthonormalized"]
    assert orth and any(r["seconds_construct"] > r["seconds_solve"] for r in orth)
