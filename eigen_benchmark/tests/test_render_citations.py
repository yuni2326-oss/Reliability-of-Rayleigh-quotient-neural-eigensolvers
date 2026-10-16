"""인용 재번호 — 손으로 유지할 수 없는 종류라 빌드 단계로 만들었다."""
import pytest

from ._paper import require_paper

require_paper()

from eigen_benchmark.render import citations

DOC = """Intro cites [3] then [1].
Later [2,3] and [5].

## References

*note line*

1. first
2. second
3. third
5. fifth
"""


def test_first_appearance_order_ignores_the_reference_list():
    assert citations.first_appearance_order(DOC) == [3, 1, 2, 5]


def test_renumber_maps_to_first_appearance():
    out = citations.renumber(DOC)
    body = out.split("## References")[0]
    assert "[1]" in body and "[2]" in body          # 3→1, 1→2
    assert citations.is_ordered(out)


def test_renumber_reorders_the_reference_list_to_match():
    out = citations.renumber(DOC)
    refs = out.split("## References")[1]
    lines = [l.strip() for l in refs.splitlines() if l.strip().startswith("[")]
    assert lines == ["[1] third", "[2] first", "[3] second", "[4] fifth"]


def test_renumber_keeps_non_entry_lines_in_place():
    assert "*note line*" in citations.renumber(DOC)


def test_grouped_citations_are_sorted_after_mapping():
    """사상은 3→1, 1→2, 2→3이므로 [2,3]은 {3,1} → 오름차순 [1,3]이 된다."""
    out = citations.renumber(DOC)
    assert "[1,3]" in out.split("## References")[0]


def test_renumber_is_idempotent():
    once = citations.renumber(DOC)
    assert citations.renumber(once) == once


def test_uncited_reference_is_an_error_not_a_silent_drop():
    doc = DOC.replace("Later [2,3] and [5].", "Later [2,3].")
    with pytest.raises(ValueError, match="5"):
        citations.renumber(doc)


def test_citation_without_a_reference_entry_is_an_error():
    doc = DOC.replace("5. fifth\n", "")
    with pytest.raises(ValueError, match="5"):
        citations.renumber(doc)


def test_committed_paper_is_in_first_appearance_order():
    """C&S는 번호식 참고문헌을 최초 등장 순서로 요구한다. 초안을 고치는 동안 22개 중
    20개가 순서를 벗어난 적이 있다 — 빌드가 지켜야 한다."""
    with open("docs/paper2-cs/paperA_CS_benchmark.md", encoding="utf-8") as f:
        t = f.read()
    assert citations.is_ordered(t), citations.first_appearance_order(t)


def test_renderers_do_not_hardcode_citation_numbers():
    """렌더러 소스에 [n]을 박으면 재번호 뒤 조용히 틀린 문헌을 가리킨다.

    부록 B가 거리함수를 [3]으로 박아 두었는데, 본문 재번호로 Sukumar가 [22]가 되면서
    [3]은 Deep Ritz를 가리키게 됐다 — 번호는 유효하므로 어떤 검사도 걸리지 않는다.
    본문 인용은 원고가 쓰고, 렌더러는 저자명으로 가리킨다."""
    import ast
    import pathlib
    import re
    src = pathlib.Path(__file__).resolve().parents[1] / "render"
    cite = re.compile(r"\[\d+\]")
    for f in sorted(src.glob("*.py")):
        if f.name == "citations.py":          # 재번호 모듈 자신은 예외
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        bad = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and cite.search(n.value)]
        assert not bad, f"{f.name}에 박힌 인용 번호: {bad}"


def test_renumber_accepts_the_korean_reference_heading():
    """P5 한글본은 목록 제목이 "## 참고문헌"이다. 제목을 영문으로만 찾으면 목록을 못 찾아
    재번호가 조용히 건너뛰어진다(v3 한글본의 [28]이 목록 끝에 남은 이유)."""
    doc = DOC.replace("## References", "## 참고문헌")
    out = citations.renumber(doc, heading="## 참고문헌")
    refs = out.split("## 참고문헌")[1]
    lines = [l.strip() for l in refs.splitlines() if l.strip().startswith("[")]
    assert lines == ["[1] third", "[2] first", "[3] second", "[4] fifth"]
    assert citations.is_ordered(out, heading="## 참고문헌")


def test_integer_intervals_are_not_citations():
    """격자 범위를 인용으로 오인하면 재번호가 그 범위를 조용히 고쳐 쓴다. P5 v4 초안에서
    "n_x ∈ [28, 32]"가 "[28,29]"로 바뀌었다(32번 문헌이 29번으로 재번호되며). 형식 규약상
    인용은 콤마 뒤 공백이 없고, 집합 기호 ∈ 뒤의 괄호는 구간이다."""
    doc = ("Grid n ∈ [28, 32] and m ∈ [16,20], also a range [8, 10].\n"
           "Cites [2] then [1] and [1,2].\n\n## References\n\n[1] a\n[2] b\n")
    out = citations.renumber(doc)
    assert "n ∈ [28, 32]" in out and "m ∈ [16,20]" in out and "range [8, 10]" in out
    assert "Cites [1] then [2] and [1,2]." in out
    assert citations.first_appearance_order(doc) == [2, 1]


def test_blank_line_separated_list_stays_separated():
    """P5 원고 목록은 항목 사이에 빈 줄을 둔다. 다시 쓸 때 빈 줄을 목록 끝으로 몰면 마크다운에서
    항목들이 한 문단으로 합쳐진다."""
    doc = "Cites [2] then [1].\n\n## References\n\n[1] a\n\n[2] b\n\n---\n\n## Appendix\n"
    out = citations.renumber(doc)
    refs = out.split("## References")[1].split("## Appendix")[0]
    assert "[1] b\n\n[2] a\n" in refs
    assert "\n\n\n" not in refs
