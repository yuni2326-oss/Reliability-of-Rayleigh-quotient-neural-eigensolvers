"""인용 번호를 **본문 최초 등장 순서**로 다시 매긴다 — 손으로 유지할 수 없는 종류의 일.

Elsevier 번호식(C&S 포함)은 참고문헌을 본문 최초 등장 순서로 요구한다. 그런데 초안을
고치는 동안 §2에 문헌이 추가되고 §3에서만 쓰이는 항목이 앞번호를 차지하면서, 22개 중
**20개**가 순서를 벗어났다. 손으로 맞추면 다음 편집에서 다시 깨진다.

그래서 빌드 단계로 만든다. `render_paper`가 AUTO 블록을 주입한 **뒤** 이 함수를 돌려
본문 인용과 참고문헌 목록을 함께 다시 매긴다. 표 캡션 안의 인용(부록 B의 [3] 등)도 주입
후에 처리되므로 같이 따라온다.

**멱등이다.** 이미 등장순이면 사상이 항등이므로 아무것도 바뀌지 않는다 — 그래서
`--check`가 안정적으로 동작한다.
"""
from __future__ import annotations

import re

# 인용은 콤마 뒤 공백이 없다(형식 규약 §6: `[23,24]`). 공백이 있는 괄호와 집합 기호 ∈ 뒤의
# 괄호는 **구간**이다 — 이것까지 인용으로 잡으면 재번호가 구간을 조용히 고쳐 쓴다. P5 v4
# 초안에서 "n_x ∈ [28, 32]"가 "[28,29]"로 바뀐 적이 있다(32번 문헌이 29번으로 재번호되며).
CITE = re.compile(r"(?<!∈)(?<!∈ )\[(\d+(?:,\d+)*)\]")
# 목록 항목은 `[1] Author…` 형태로 쓴다. `1. Author…`(마크다운 번호목록)로 두면
# docx 변환에서 Word의 `List Number` 스타일이 붙어 번호가 **자동생성**된다 — 편집자가
# 목록을 건드리면 본문의 [n]과 소리 없이 어긋난다. 대괄호 형태는 문자 그대로 실린다.
# 옛 형태도 읽어서 다시 쓸 때 대괄호로 정규화한다.
REF_LINE = re.compile(r"^\[(\d+)\]\s+(.*)$|^(\d+)\.\s+(.*)$")
REF_HEADING = "## References"


def _split(text: str, heading: str = REF_HEADING):
    """(본문, 참고문헌 목록, 꼬리) — 목록 뒤의 부록은 **꼬리로 따로 돌려준다.**

    부록이 참고문헌 뒤에 오는 편집 순서라, 목록 이후 전체를 "참고문헌 절"로 묶으면
    부록의 인용이 재번호에서 빠진다. 실제로 부록 B의 거리함수 인용이 [3]으로 남아
    본문의 [22]와 어긋났다 — 파이프라인이 일관성을 보장한다는 주장을 정면으로 깨는
    종류의 버그이므로 절을 셋으로 나눈다."""
    i = text.find(heading)
    if i < 0:
        return text, "", ""
    j = text.find("\n## ", i + len(heading))
    if j < 0:
        return text[:i], text[i:], ""
    return text[:i], text[i:j], text[j:]


def first_appearance_order(text: str, heading: str = REF_HEADING) -> list:
    """인용이 처음 나오는 순서의 번호 목록 — 본문을 먼저, 그다음 부록을 훑는다."""
    body, _, tail = _split(text, heading)
    seen, order = set(), []
    for m in CITE.finditer(body + "\n" + tail):
        for n in (int(x) for x in re.split(r",", m.group(1))):
            if n not in seen:
                seen.add(n)
                order.append(n)
    return order


def is_ordered(text: str, heading: str = REF_HEADING) -> bool:
    o = first_appearance_order(text, heading)
    return o == list(range(1, len(o) + 1))


def _parse_refs(refs: str) -> dict:
    """번호 → 항목 본문. 목록 항목이 아닌 줄(설명문)은 무시한다."""
    out = {}
    for line in refs.splitlines():
        m = REF_LINE.match(line.strip())
        if m:
            n, body = (m.group(1), m.group(2)) if m.group(1) else (m.group(3),
                                                                   m.group(4))
            out[int(n)] = body.strip()
    return out


def renumber(text: str, heading: str = REF_HEADING) -> str:
    """본문 최초 등장 순서로 인용과 목록을 다시 매긴다. 이미 맞으면 그대로 반환.

    `heading`은 목록 제목이다 — P5 한글본은 "## 참고문헌"을 쓴다."""
    body, refs, tail = _split(text, heading)
    order = first_appearance_order(text, heading)
    if not order:
        return text
    entries = _parse_refs(refs)
    if entries:
        missing = [n for n in order if n not in entries]
        if missing:
            raise ValueError(f"본문이 인용하는데 목록에 없는 번호: {missing}")
        extra = sorted(set(entries) - set(order))
        if extra:
            raise ValueError(f"목록에만 있고 인용되지 않는 번호: {extra}")
    if order == list(range(1, len(order) + 1)):
        if entries and all(not line.strip().startswith("[")
                           for line in refs.splitlines()
                           if REF_LINE.match(line.strip())):
            return body + _rewrite_refs(refs, order, entries) + tail
        return text

    old2new = {o: i + 1 for i, o in enumerate(order)}

    def sub(m):
        ns = sorted(old2new[int(x)] for x in re.split(r",", m.group(1)))
        return "[" + ",".join(str(n) for n in ns) + "]"

    new_body = CITE.sub(sub, body)
    new_tail = CITE.sub(sub, tail)      # 부록도 같은 사상을 받는다

    if not entries:
        return new_body + refs + new_tail
    return new_body + _rewrite_refs(refs, order, entries) + new_tail


def _rewrite_refs(refs: str, order: list, entries: dict) -> str:
    """목록을 새 순서로 다시 쓴다. 항목 줄이 아닌 줄은 위치를 유지한다.

    **항목 사이의 구분 방식을 보존한다.** 원래 목록이 항목마다 빈 줄로 떨어져 있으면 다시 쓴
    목록도 그렇게 쓴다. 빈 줄을 그대로 제자리에 두면 새 항목들이 한 덩어리로 붙고 빈 줄은 목록
    끝으로 밀려, 마크다운에서 항목들이 한 문단으로 합쳐진다."""
    lines = refs.splitlines()
    idx = [i for i, line in enumerate(lines) if REF_LINE.match(line.strip())]
    if not idx:
        return refs
    first, last = idx[0], idx[-1]
    spaced = any(lines[i + 1].strip() == "" for i in idx[:-1])
    inner = [line for line in lines[first:last + 1]
             if line.strip() and not REF_LINE.match(line.strip())]
    body = []
    for i, o in enumerate(order, start=1):
        body.append(f"[{i}] {entries[o]}")
        if spaced and i < len(order):
            body.append("")
    out = lines[:first] + body + inner + lines[last + 1:]
    return "\n".join(out) + ("\n" if refs.endswith("\n") else "")
