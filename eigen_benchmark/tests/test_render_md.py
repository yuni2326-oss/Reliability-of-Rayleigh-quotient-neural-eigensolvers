import pytest

from eigen_benchmark.render import md


def test_fmt_handles_missing_and_infinite():
    assert md.fmt(float("nan")) == "—"
    assert md.fmt(None) == "—"
    assert md.fmt("") == "—"
    assert md.fmt(float("inf")) == "∞"
    assert md.fmt(1.0) == "1.00"
    assert md.fmt(0.9384, sig=3) == "0.938"
    assert md.fmt(1.2345e-5, sig=3) == "1.23e-05"


def test_fmt_passes_through_non_numeric_strings():
    assert md.fmt("I0") == "I0"
    assert md.fmt("b_projection_exact") == "b_projection_exact"


def test_fmt_renders_booleans_as_words_not_numbers():
    """CSV의 bool을 1.00/0으로 찍으면 표가 거짓말한다."""
    assert md.fmt(True) == "yes" and md.fmt(False) == "no"


def test_fmt_keeps_integers_integral():
    """모드 번호와 카운트가 1.00으로 찍히면 표가 흐려진다."""
    assert md.fmt(1) == "1" and md.fmt(20000) == "20000"
    assert md.fmt(1.0) == "1.00"          # float는 유효숫자를 유지


def test_table_emits_gfm_with_alignment_row():
    s = md.table(["a", "bb"], [[1, "x"], [2.5, "y"]], align="rl")
    lines = s.strip().splitlines()
    assert len(lines) == 4
    assert lines[1].replace(" ", "") == "|---:|:---|"
    assert "| 1 " in lines[2] and "2.50" in lines[3]


def test_table_rejects_ragged_rows():
    with pytest.raises(ValueError):
        md.table(["a", "b"], [[1]])


def test_table_rejects_mismatched_alignment():
    with pytest.raises(ValueError):
        md.table(["a", "b"], [[1, 2]], align="r")


def test_table_with_no_rows_still_has_a_header():
    s = md.table(["a", "b"], [])
    assert len(s.strip().splitlines()) == 2


def test_caption_names_the_source_and_sha():
    c = md.caption(3, "Mode-order reliability.", "p1_neural_mode_sweep.csv", "abcdef1234")
    assert "Table 3." in c and "p1_neural_mode_sweep.csv" in c and "abcdef12" in c


def test_caption_accepts_several_sources():
    c = md.caption(7, "References.", ["p1_reference.csv", "p2_reference.csv"], "0123456789")
    assert "p1_reference.csv, p2_reference.csv" in c


def test_pipes_inside_cells_are_escaped_so_columns_do_not_shift():
    """칸 안의 `|`를 그대로 두면 그 칸이 쪼개져 열이 어긋난다.

    부록 B의 (g) 행이 `|rFFT|`로 두 칸 밀려 표가 무너진 적이 있다. 열 수 검사는 파이썬
    리스트 기준이라 이 종류를 잡지 못하므로, 출력된 줄의 파이프 개수로 확인한다."""
    out = md.table(["a", "b"], [["x", "peak |rFFT| bin"], ["y", "no pipe"]])
    lines = [l for l in out.splitlines() if l.startswith("|")]
    assert len({l.count("|") - l.count("\\|") for l in lines}) == 1, \
        [l.count("|") for l in lines]
    assert "\\|rFFT\\|" in out
    # 헤더에 파이프가 있어도 같다
    out2 = md.table(["a|b", "c"], [["1", "2"]])
    lines2 = [l for l in out2.splitlines() if l.startswith("|")]
    assert len({l.count("|") - l.count("\\|") for l in lines2}) == 1
