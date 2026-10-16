import pytest
import json

import numpy as np

from eigen_benchmark.drivers import run_p2, run_p4


def test_run_p2_quick_writes_expected_artifacts(tmp_path):
    run_p2.main(outdir=str(tmp_path), quick=True)
    for name in ("p2_reference.csv", "p2_basis_study.csv", "p2_conditioning.csv",
                 "p2_degeneracy.csv", "manifest_p2.json"):
        assert (tmp_path / name).exists()
    with open(tmp_path / "manifest_p2.json") as f:
        assert "git_sha" in json.load(f)


def test_run_p2_reference_covers_the_declared_grid(tmp_path):
    res = run_p2.main(outdir=str(tmp_path), quick=True)
    ms = sorted({r["m"] for r in res["reference"]})
    ns = sorted({r["radial_order"] for r in res["reference"]})
    assert ms == [0, 1, 2, 3, 4]
    assert ns == [0, 1, 2, 3]
    assert len(res["reference"]) == 20


def test_run_p2_bases_match_the_exact_reference(tmp_path):
    res = run_p2.main(outdir=str(tmp_path), quick=True)
    ok = [r for r in res["basis_study"] if r["cholesky_ok"]]
    assert ok
    assert min(r["e_lam_mode1"] for r in ok) < 1e-6


def test_run_p2_degeneracy_rows_show_subspace_advantage(tmp_path):
    res = run_p2.main(outdir=str(tmp_path), quick=True)
    rows = [r for r in res["degeneracy"] if r["m"] > 0]
    assert rows
    for r in rows:
        assert r["subspace_mac_rotated"] > 0.999999
        assert r["individual_mac_rotated"] < r["subspace_mac_rotated"]


def test_run_p4_quick_writes_expected_artifacts(tmp_path):
    res = run_p4.main(outdir=str(tmp_path), quick=True)
    for name in ("p4_reference.csv", "p4_convergence.csv", "manifest_p4.json"):
        assert (tmp_path / name).exists()
    assert len(res["reference"]) >= 3


def test_run_p4_reports_uncertainty_for_every_reference_mode(tmp_path):
    res = run_p4.main(outdir=str(tmp_path), quick=True)
    for entry in res["reference"]:
        assert "uncertainty_rel" in entry
        assert "within_target" in entry


def test_run_p4_compares_uniform_and_graded(tmp_path):
    res = run_p4.main(outdir=str(tmp_path), quick=True)
    betas = sorted({row["beta"] for row in res["convergence"]})
    assert 1.0 in betas and len(betas) >= 2


def test_run_p4_picks_the_beta_with_the_smallest_uncertainty(tmp_path):
    """채택 규칙은 '가장 강한 등급'이 아니라 '불확실도 최댓값이 가장 작은 β'다.
    과도 등급(β=5)이 오히려 수렴을 악화시키므로 max(betas)를 쓰면 안 된다."""
    res = run_p4.main(outdir=str(tmp_path), quick=True)
    chosen = {r["beta"] for r in res["reference"]}
    assert len(chosen) == 1
    betas = sorted({row["beta"] for row in res["convergence"]})
    assert chosen.pop() in betas


def test_p4_convergence_times_each_grid_separately(tmp_path):
    """격자마다 따로 재야 한다 — 이전에는 sweep 전체 시간이 모든 행에 복사되어
    n=4와 n=16이 같은 초로 기록됐고, 비용 비교를 그 열에 세울 수 없었다."""
    from eigen_benchmark.drivers import run_p4
    res = run_p4.main(outdir=str(tmp_path), quick=True)
    rows = res["convergence"]
    by = {}
    for r in rows:
        by.setdefault(r["beta"], []).append((r["n"], r["seconds"]))
    for beta, v in by.items():
        v.sort()
        secs = [s for _, s in v]
        assert len(set(secs)) == len(secs), (beta, v)
        assert secs == sorted(secs), f"격자를 키웠는데 시간이 줄었다: {v}"


def test_no_csv_field_that_reaches_a_rendered_table_contains_non_ascii_prose():
    """논문은 영문이다. 표의 note·reference 같은 문자열 열은 그대로 인쇄되므로
    한국어가 섞이면 원고에 박힌다 — 실제로 Table 7에 그렇게 들어갔다."""
    import csv
    import glob
    import os
    bad = []
    for f in glob.glob("docs/_generated/data/paper2/*.csv"):
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        for r in rows[:200]:
            for k, v in r.items():
                if not isinstance(v, str) or not v:
                    continue
                # 숫자·경로가 아닌 산문 열만 본다
                if any(c > "\x7f" for c in v):
                    bad.append((os.path.basename(f), k, v[:40]))
    assert not bad, bad[:6]


def test_p4_quadrature_check_driver_carries_the_drift_per_cell(tmp_path):
    """위협 13은 P4에서 하한 스캔만으로 69셀을 잡았다고 적는다. Λ가 Λ₁ 위에 있으면서
    과소적분된 셀은 그 스캔에 안 걸리므로, P1과 같은 두 차수 계기를 P4에도 붙인다.
    커밋된 `p4_neural_arms.csv`는 건드리지 않고 별도 파일에 쓴다."""
    import json
    pytest.importorskip("torch")
    from eigen_benchmark.drivers import run_p4_neural as r4
    out = r4.quadrature_check(outdir=str(tmp_path), quick=True, device="cpu")
    assert (tmp_path / "p4_quadrature_check.csv").exists()
    assert not (tmp_path / "p4_neural_arms.csv").exists(), "커밋 산출물을 덮었다"
    for row in out["rows"]:
        assert "quad_drift_median" in row and "quad_drift_max" in row, row.keys()
    with open(tmp_path / "p4_quadrature_check_records.jsonl", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    assert recs and all("quad_drift" in r for r in recs)
    man = json.load(open(tmp_path / "manifest_p4_quadrature_check.json",
                         encoding="utf-8"))
    assert man["check_n_per_block"] > man["n_per_block"]
