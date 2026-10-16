"""장기변주 셀 선택규칙 — GPU 없이 규칙만 검증한다.

`run_p1_neural.run_long`의 `failing()`과 같은 로직을 합성 격자에 적용해,
"붕괴 arm은 첫 실패 모드만 / 완만한 arm은 가장 나쁜 n개"가 지켜지는지 본다.
규칙이 무너지면 장기변주가 정보 없는 셀에 예산을 쏟는다(실측 4시간 → 10시간).
"""
import csv
import tempfile

from eigen_benchmark.drivers import manifest


def _failing(base, pre, thresh=0.9, n_worst=2):
    cand = [(int(r["mode"]), float(r["p_correct"])) for r in base
            if r["arm"].startswith(pre)
            and float(r["p_correct"]) < thresh and int(r["n"]) > 1]
    if not cand:
        return []
    if all(p <= 0.05 for _, p in cand):
        return [min(m for m, _ in cand)]
    return sorted(m for m, _ in sorted(cand, key=lambda t: t[1])[:n_worst])


def _grid():
    d = tempfile.mkdtemp()
    rows = [{"arm": "collapse_arm", "mode": m, "n": 50,
             "p_correct": (1.0 if m < 3 else 0.0), "seconds": 1.0}
            for m in range(1, 11)]
    rows += [{"arm": "graceful_arm", "mode": m, "n": 50,
              "p_correct": (1.0 if m < 9 else (0.78 if m == 9 else 0.82)),
              "seconds": 1.0} for m in range(1, 11)]
    rows += [{"arm": "clean_arm", "mode": m, "n": 50,
              "p_correct": (1.0 if m < 5 else 0.98), "seconds": 1.0}
             for m in range(1, 11)]
    rows += [{"arm": "single_shot", "mode": 1, "n": 1,
              "p_correct": 0.0, "seconds": 1.0}]
    manifest.write_csv(f"{d}/g.csv", rows)
    return list(csv.DictReader(open(f"{d}/g.csv", encoding="utf-8")))


def test_collapsing_arm_gets_only_its_first_failing_mode():
    """순차 deflation은 체인이라 첫 실패 이후 모드는 그 하류 결과다."""
    assert _failing(_grid(), "collapse_arm") == [3]


def test_gracefully_degrading_arm_gets_its_worst_modes():
    assert _failing(_grid(), "graceful_arm") == [9, 10]


def test_nearly_perfect_cells_are_not_selected():
    """0.98은 예산 문제가 아니다 — 임계 0.9 미만만 후보."""
    assert _failing(_grid(), "clean_arm") == []


def test_single_run_cells_are_excluded():
    """PIELM처럼 시드 개념이 없는 단일 결정론 실행(n=1)은 대상이 아니다."""
    assert _failing(_grid(), "single_shot") == []


def test_worst_count_is_capped():
    base = _grid()
    assert len(_failing(base, "graceful_arm", n_worst=1)) == 1


def test_appended_csv_rows_are_type_restored():
    """CSV에서 읽은 행은 전부 문자열이다. 그대로 float와 비교하면 TypeError로 죽고,
    실제로 그 때문에 2시간 11분치 계산을 마지막 기록 단계에서 잃었다."""
    raw = {"arm": "b_projection_exact", "mode": "9", "n": "50",
           "p_correct": "0.96", "budget_resolved": "True",
           "ladder": "I0", "seconds": "3025.4"}
    got = manifest.coerce_row(raw)
    assert got["mode"] == 9 and isinstance(got["mode"], int)
    assert got["p_correct"] == 0.96 and isinstance(got["p_correct"], float)
    assert got["budget_resolved"] is True
    assert got["arm"] == "b_projection_exact"      # 이름은 문자열로 남는다
    assert got["seconds"] > 3000.0                 # float 비교가 가능해야 한다


def test_coerce_leaves_non_numeric_strings_alone():
    got = manifest.coerce_row({"arm": "c_curriculum(stage1=0.5)", "nodes": "gauss"})
    assert got["arm"] == "c_curriculum(stage1=0.5)"
    assert got["nodes"] == "gauss"
