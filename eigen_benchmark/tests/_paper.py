"""원고 정본의 위치와, 원고가 없을 때의 처리.

**왜 이 모듈이 있는가.** 공개 아티팩트에는 코드와 데이터만 싣고 심사 중인 원고는 싣지
않는다. 그러면 원고를 읽는 검사들이 갈 곳을 잃는데, 그 검사들을 지우면 무엇을 검사하고
있었는지가 사라지고, 그대로 두면 내려받은 사람의 스위트가 처음부터 빨갛다. 그래서
남겨 두되 원고가 없으면 건너뛴다 — 무엇이 검사되지 않았는지가 skip 사유로 남는다.
"""
import os

import pytest

PAPER = os.path.join("docs", "paper2-cs", "paperA_CS_benchmark.md")
REASON = (f"원고 정본({PAPER})이 없다 — 코드·데이터만 담은 배포본에서는 "
          "원고를 읽는 검사를 건너뛴다")


def have_paper() -> bool:
    return os.path.exists(PAPER)


def require_paper():
    """모듈 전체가 원고에 의존할 때 import 시점에 부른다."""
    if not have_paper():
        pytest.skip(REASON, allow_module_level=True)


needs_paper = pytest.mark.skipif(not os.path.exists(PAPER), reason=REASON)
