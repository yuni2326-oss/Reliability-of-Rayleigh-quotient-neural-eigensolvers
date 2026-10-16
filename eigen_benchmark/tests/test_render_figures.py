import os

import pytest

pytest.importorskip("matplotlib")
from eigen_benchmark.render import figures      # noqa: E402

DATA = "docs/_generated/data/paper2"


@pytest.mark.parametrize("name", sorted(figures.ALL))
def test_every_figure_writes_a_nonempty_png(tmp_path, name):
    p = figures.ALL[name](DATA, str(tmp_path))
    assert os.path.exists(p) and os.path.getsize(p) > 5000


def test_figures_do_not_depend_on_a_display():
    """헤드리스에서 돌아야 한다 — Agg 백엔드를 import 시점에 강제하는지."""
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg"


def test_figure_labels_are_ascii_so_no_font_is_missing(tmp_path):
    """한글 라벨은 폰트 누락 시 두부글자가 된다 — 소스에 비ASCII 축라벨이 없는지."""
    import re
    src = open(figures.__file__, encoding="utf-8").read()
    for call in re.findall(r'set_(?:xlabel|ylabel|title)\(\s*(r?["\'][^"\']*["\'])',
                           src):
        assert call.isascii(), call


# ─────────────────────────────────────────────────────────────────────────────
# 그림 2는 이 논문의 deflation 논증을 나르는 그림이다. 본문이 모드 16까지 주장하고
# 개정에서 arm (b′)를 추가했는데 그림이 모드 10에서 끊기고 (b′)가 없으면,
# 그림만 보는 독자는 개정 내용을 전혀 보지 못한다.
# ─────────────────────────────────────────────────────────────────────────────

def test_the_reliability_figure_includes_the_arm_added_in_revision():
    s = figures._reliability_series(DATA)
    assert any("(b′)" in k for k in s), f"QR arm이 없다: {sorted(s)}"


def test_the_reliability_figure_reaches_the_depth_the_claim_reaches():
    import csv
    s = figures._reliability_series(DATA)
    drawn = max(max(v["mode"]) for v in s.values())
    ext = max(int(r["mode"]) for r in
              csv.DictReader(open(os.path.join(DATA, "p1_fp64_deep_b_stress.csv"))))
    assert drawn == ext, f"그림은 모드 {drawn}까지, 확장 데이터는 {ext}까지"


def test_the_reliability_figure_separates_prespecified_from_extension():
    """사전등록 격자와 사후 확장을 한 축에 그리면서 경계를 표시하지 않으면,
    독자는 16까지가 전부 사전등록된 것으로 읽는다."""
    import re
    s = figures._reliability_series(DATA)
    pre = {v["n_prespecified"] for v in s.values() if v["n_prespecified"]}
    assert pre == {10}, f"사전등록 격자 경계가 일관되지 않다: {pre}"
    src = open(figures.__file__, encoding="utf-8").read()
    body = src[src.index("def fig2_mode_reliability"):]
    body = body[:body.index("\ndef ", 1)]
    assert re.search(r"axvline|axvspan", body), "사전등록 경계 표시가 없다"


def test_the_reliability_series_matches_the_committed_outcomes():
    """그림의 점이 표와 다른 수를 그리면 안 된다 — 두 곳이 같은 CSV를 읽는지."""
    import csv
    s = figures._reliability_series(DATA)
    ext = {int(r["mode"]): r for r in
           csv.DictReader(open(os.path.join(DATA, "p1_fp64_deep_b_stress.csv")))}
    key = next(k for k in s if k.startswith("(b) "))
    got = dict(zip(s[key]["mode"], s[key]["p"]))
    for m in (10, 14, 16):
        assert abs(got[m] - float(ext[m]["p_correct"])) < 1e-12, (
            f"모드 {m}: 그림 {got[m]} vs 데이터 {ext[m]['p_correct']}")
