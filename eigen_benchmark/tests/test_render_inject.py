import pytest

from eigen_benchmark.render import inject

DOC = "before\n<!-- AUTO:t1 -->\nold\n<!-- /AUTO -->\nafter\n"


def test_inject_replaces_block_content():
    out = inject.inject(DOC, {"t1": "NEW"})
    assert "NEW" in out and "old" not in out
    assert out.startswith("before") and out.rstrip().endswith("after")


def test_inject_is_idempotent():
    once = inject.inject(DOC, {"t1": "NEW"})
    assert inject.inject(once, {"t1": "NEW"}) == once


def test_inject_keeps_the_markers_so_it_can_run_again():
    out = inject.inject(DOC, {"t1": "NEW"})
    assert "<!-- AUTO:t1 -->" in out and "<!-- /AUTO -->" in out


def test_unplaced_block_is_an_error_not_a_silent_drop():
    with pytest.raises(ValueError, match="t2"):
        inject.inject(DOC, {"t1": "A", "t2": "B"})


def test_empty_marker_in_document_is_an_error():
    with pytest.raises(ValueError, match="t1"):
        inject.inject(DOC, {})


def test_render_all_produces_every_declared_table_and_figure(tmp_path):
    b = inject.render_all(fig_dir=str(tmp_path))
    for k in inject.TABLES:
        assert k in b and b[k].strip()
    for i in range(1, 7):
        assert f"figure_{i}" in b
    assert "appendix_a" in b and "appendix_b" in b


def test_figure_blocks_link_a_relative_path_that_resolves(tmp_path):
    """논문에서 본 상대경로가 실제 파일을 가리켜야 한다 — docx 변환이 여기서 깨진다."""
    import os
    import re
    fig = tmp_path / "fig"
    b = inject.render_all(fig_dir=str(fig), paper_dir=str(tmp_path))
    src = re.search(r"!\[Figure 1\]\(([^)]+)\)", b["figure_1"]).group(1)
    assert os.path.exists(os.path.join(str(tmp_path), src)), src


def test_every_figure_block_carries_a_caption(tmp_path):
    b = inject.render_all(fig_dir=str(tmp_path))
    for i in range(1, 7):
        assert f"**Figure {i}.**" in b[f"figure_{i}"]


def test_the_reliability_caption_describes_what_the_figure_now_shows():
    """그림이 모드 16까지 가고 arm (b′)를 담게 됐다. 캡션이 "through mode 10"에
    머물면 그림과 설명이 어긋나고, 확장이 사후 추가라는 사실도 사라진다."""
    cap = inject.FIGURE_CAPTION["fig2_mode_reliability"]
    assert "mode 16" in cap
    assert "(b′)" in cap
    assert "pre-specified grid" in cap and "added in revision" in cap
    assert "through mode 10." not in cap


def test_figure_file_names_carry_the_number_they_are_rendered_as():
    """파일명의 숫자와 논문 그림 번호가 어긋나면, 저장소를 훑는 사람이 엉뚱한 파일을 연다.
    실제로 그랬다 — fig3_error_vs_dof.png는 Figure 3이고, Figure 2는 fig2_mode_reliability
    였다. 순서 정본은 FIGURE_ORDER이므로 이름이 그것을 따라야 한다."""
    for i, name in enumerate(inject.FIGURE_ORDER, start=1):
        assert name.startswith(f"fig{i}_"), (
            f"Figure {i}로 렌더되는 파일 이름이 {name}이다")
