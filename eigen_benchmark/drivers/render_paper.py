"""정본 마크다운의 AUTO 블록을 현재 데이터로 갱신한다.

`--check`는 쓰지 않고 어긋남만 보고한다 — 표가 데이터와 다른 채로 커밋되는 것을 막는 것이
§3.7("드라이버가 모든 표를 재생성한다")의 실질이다.
"""
from __future__ import annotations

import argparse
import os

from ..render import citations, figures, inject, tables

PAPER = os.path.join("docs", "paper2-cs", "paperA_CS_benchmark.md")


def main(paper: str = PAPER, data_dir: str = tables.DATA_DIR,
         fig_dir: str = figures.FIG_DIR, check: bool = False) -> dict:
    blocks = inject.render_all(data_dir, fig_dir, os.path.dirname(paper))
    # **원고가 없을 수 있다.** 코드·데이터만 담은 배포본이 그렇다. 표와 그림은 원고 없이도
    # 만들어지므로 거기까지 하고 멈추되, 무엇이 없어서 멈췄는지 말한다 — 트레이스백으로
    # 죽으면 받은 사람은 아티팩트가 깨진 줄 안다.
    if not os.path.exists(paper):
        # 이 메시지만 영어다 — 공개 아티팩트의 README가 안내하는 경로이고,
        # 원고가 없는 상황은 그 배포본에서만 생긴다. 나머지 콘솔 출력은 한국어 그대로.
        print(f"[render_paper] {paper} not found: rendered {len(blocks)} table "
              f"and figure blocks into {fig_dir}, and skipped injection and the "
              "drift check.")
        return {"paper": paper, "paper_present": False, "drift": None,
                "n_blocks": len(blocks)}
    with open(paper, encoding="utf-8") as f:
        old = f.read()
    new = inject.inject(old, blocks)
    # **AUTO 주입 뒤에** 인용을 등장순으로 다시 매긴다 — 표 캡션 안의 인용까지 함께
    # 따라오게 하려면 순서가 이래야 한다. 멱등이므로 --check가 안정적이다.
    new = citations.renumber(new)
    drift = new != old
    if not check and drift:
        with open(paper, "w", encoding="utf-8") as f:
            f.write(new)
    return {"paper": paper, "paper_present": True, "drift": drift,
            "n_blocks": len(blocks)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    r = main(check=a.check)
    print(r)
    if a.check and r["drift"]:
        raise SystemExit(1)
