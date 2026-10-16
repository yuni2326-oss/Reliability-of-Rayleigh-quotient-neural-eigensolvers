"""논문 경계 규약을 코드로 강제하는 테스트.

`docs/paper2-cs/README.md`의 논문 구분 규약: 각 논문의 코드 패키지는 서로 수정하지
않고, 읽기전용 import 또는 **재구현 + 교차검증 테스트**로만 관계한다.
논문2(`eigen_benchmark`)는 2026-08-01 P3 재파라미터화 이후 논문1(`impeller_pinn`)에
대한 의존이 완전히 사라졌다 — 그 상태를 회귀로 지킨다.
"""
from __future__ import annotations

import pathlib
import re

PKG = pathlib.Path(__file__).resolve().parent.parent
_IMPORT = re.compile(r"^\s*(?:from|import)\s+impeller_pinn\b", re.MULTILINE)


def _sources():
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_module_imports_paper1_package():
    offenders = [str(p.relative_to(PKG)) for p in _sources()
                 if _IMPORT.search(p.read_text(encoding="utf-8"))]
    assert offenders == [], f"논문1 패키지를 import하는 모듈: {offenders}"


def test_package_has_sources_so_the_scan_is_meaningful():
    # 위 테스트가 빈 목록을 훑고 통과하는 위양성을 막는다.
    assert len(_sources()) > 20


def test_p3_is_parameterized_by_dimensionless_spring_stiffness_only():
    """논문 분리 규약: crack-depth fingerprint는 논문3 전유. 논문2의 P3는 회전스프링을
    generic 계면 벤치마크로만 쓰므로 손상 기전 파라미터가 노출되면 안 된다."""
    from eigen_benchmark.problems import p3_spring as p3
    assert set(p3.P3_CONFIG) == {"xc_over_L", "k_hats", "k_hat_central"}
    assert not hasattr(p3, "k_hat_from_crack")
    from eigen_benchmark.reference import transfer_matrix as tm
    assert not hasattr(tm, "kappa_from_crack")
    assert hasattr(tm, "kappa_from_k_hat")


def test_no_test_writes_into_the_committed_data_directory():
    """테스트가 `docs/_generated/data/paper2`에 쓰면 커밋된 데이터가 조용히 바뀐다.

    실제로 `test_run_p1_compare`가 드라이버를 실데이터 경로로 돌려 manifest의 git_sha를
    테스트 실행 시점의 HEAD로 갈아치웠고, 그 결과 캡션이 커밋마다 바뀌어
    `render_paper --check`가 드리프트를 보고했다. 출력은 tmp로 보낸다.

    주석·문자열이 아니라 **실제 키워드 인자**만 보도록 AST로 검사한다."""
    import ast
    import pathlib as _pl
    tests = _pl.Path(__file__).resolve().parent
    bad = []
    for f in sorted(tests.glob("test_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg not in ("outdir", "out_dir", "out"):
                    continue
                v = kw.value
                lit = (v.value if isinstance(v, ast.Constant)
                       and isinstance(v.value, str) else None)
                name = v.id if isinstance(v, ast.Name) else None
                if (lit and lit.startswith("docs/")) or name == "DATA":
                    bad.append(f"{f.name}:{node.lineno} {kw.arg}={lit or name}")
    assert not bad, "테스트가 커밋된 데이터에 쓴다: " + "; ".join(bad)


def test_render_paper_says_what_is_missing_instead_of_raising(tmp_path, capsys):
    """코드·데이터만 담은 배포본에는 원고가 없다. README가 안내하는 명령이 거기서
    트레이스백으로 죽으면, 받은 사람은 아티팩트가 깨진 줄 안다 — 무엇이 없는지 말해야 한다."""
    from eigen_benchmark.drivers import render_paper
    r = render_paper.main(paper=str(tmp_path / "absent.md"),
                          fig_dir=str(tmp_path / "fig"))
    assert r["paper_present"] is False
    assert r["drift"] is None
    assert r["n_blocks"] > 0            # 표·그림은 원고 없이도 만들어진다
    out = capsys.readouterr().out
    assert "absent.md" in out and "not found" in out


# ─────────────────────────────────────────────────────────────────────────────
# README는 조용히 낡는다. 실제로 "torch·GPU 없음"이라 적힌 채 신경망 arm이 들어왔고,
# 테스트 수는 117로 남아 있었으며, 완료된 계획 둘이 미완으로 표시돼 있었다.
# 사람이 다시 읽어야만 발견되는 종류라, 기계로 확인 가능한 것만 고정한다.
# ─────────────────────────────────────────────────────────────────────────────

README = PKG / "README.md"


def _readme():
    return README.read_text(encoding="utf-8")


def test_readme_lists_every_solver_module():
    """새 모듈이 생겼는데 README가 모르면, 다음 사람은 그 모듈이 없는 줄 안다.
    드라이버와 렌더러는 개수가 많아 디렉터리 단위로 적으므로 여기서 제외한다."""
    t = _readme()
    missing = []
    for p in sorted(PKG.rglob("*.py")):
        if "__pycache__" in p.parts or p.name == "__init__.py":
            continue
        rel = p.relative_to(PKG)
        if rel.parts[0] in ("tests", "drivers", "render"):
            continue
        if p.stem not in t:
            missing.append(str(rel))
    assert not missing, f"README가 언급하지 않는 모듈: {missing}"


def test_readme_names_every_deflation_arm():
    from eigen_benchmark.render.tables import ARM_LABEL
    t = _readme()
    missing = [v for v in ARM_LABEL.values() if v.split(")")[0] + ")" not in t]
    assert not missing, f"README에 없는 arm: {missing}"


def test_readme_quotes_the_prespecified_constants_it_claims():
    """사전등록값은 README가 인용하는 유일한 수치다. 코드와 어긋나면 둘 중 하나가 거짓이다."""
    from eigen_benchmark import metrics
    t = _readme()
    for name, val in (("MAC_MIN", metrics.MAC_MIN), ("ELAM_MAX", metrics.ELAM_MAX)):
        assert f"metrics.{name} = {val}" in t, f"{name}이 코드({val})와 다르게 적혀 있다"


def test_readme_does_not_still_claim_the_package_is_torch_free():
    """한때 사실이었고 지금은 아니다 — neural/이 생기면서 거짓이 됐다."""
    t = _readme()
    assert "torch·GPU 없음" not in t and "no torch" not in t.lower()
    assert "neural/" in t
