"""논문 정본이 데이터와 일치하고 범위를 지키는지."""
import re
import shutil

import pytest

from eigen_benchmark.drivers import render_paper

from ._paper import PAPER, require_paper  # noqa: E402

require_paper()


def read():
    with open(PAPER, encoding="utf-8") as f:
        return f.read()


def flat():
    """산문 검사용 — 줄바꿈이 문장을 쪼개므로 공백을 정규화한다."""
    return " ".join(read().split())


def test_no_unresolved_placeholders_remain():
    t = read()
    for bad in ("[TO RUN]", "[FILL", "[DECIDE]", "TBD", "[ ]/", "to be completed"):
        assert bad not in t, bad


def test_every_auto_marker_is_closed():
    t = read()
    assert t.count("<!-- AUTO:") == t.count("<!-- /AUTO -->") == 29


def test_pilot_section_states_seed_count_and_disclaims_evidence():
    m = re.search(r"##\s*4\.\s*Pilot results.*?(?=\n##\s)", read(), re.DOTALL)
    assert m, "§4가 없다"
    s = m.group(0)
    assert "4 seeds" in s
    assert "motivation" in s.lower() or "not evidence" in s.lower()
    assert "superseded" in s.lower()


def test_pilot_numbers_are_not_retrofitted():
    """파일럿을 사후에 고쳐 쓰면 '먼저 알았다'는 거짓 인상을 준다."""
    m = re.search(r"##\s*4\.\s*Pilot results.*?(?=\n##\s)", read(), re.DOTALL)
    assert "not retro-fitted" in m.group(0).lower()


def test_paper_stays_inside_its_scope():
    """논문 1·3의 소재는 여기 없어야 한다(경계 정본 §1)."""
    t = read().lower()
    for out in ("cavitation", "impeller", "crack depth", "npsh", "pump"):
        assert out not in t, out


def test_paper_quotes_no_absolute_units():
    """무차원 규칙 — **문제 정의**에 치수·물성·주파수가 재유입되면 안 된다.

    벽시계(ms, s)는 대상이 아니다: 비용은 무차원화할 수 없고, 그 값이 시험문제를 특정
    기계와 동일시하지도 않는다. 금지 대상은 길이·물성·고유주파수다."""
    t = read()
    for u in (" mm", " GPa", "kg m⁻³", "kg/m³", " Hz", "7930 kg", "193 GPa"):
        assert u not in t, u


def test_arm_a_prime_is_named_as_the_pilot_implementation():
    """파일럿은 대각근사 사영 = (a′)다. 개정 이력이 아니라 평서문으로 적혀야 한다."""
    t = flat()
    assert "this is the pilot's implementation" in t.lower()
    assert "the earlier pilot used (a′)" in t


def test_seed_count_is_fifty_everywhere_it_is_stated():
    t = read()
    assert "≥30 seeds" not in t and "≥ 30 seeds" not in t
    assert "50 seeds" in t


def test_preregistration_and_post_hoc_are_both_declared():
    t = read()
    assert "pre-specification" in t.lower() or "pre-specified" in t.lower()
    assert "post-hoc" in t.lower()
    for d in ("p_accurate", "n_certified"):
        assert d in t


def test_mode1_degradation_is_reported_as_measured_not_conjectured():
    """§6.2에 미측정 가설로 남아 있던 항목이 실측으로 대체됐는지 —
    가설로 되돌아가면 논문이 약해진다."""
    t = read()
    t = flat()
    assert "we did not measure it" not in t
    assert "capacity-competition" not in t.lower()
    assert "2122" in t and "100:1" in t
    assert "log-determinant objective is standard" in t   # 방법 제안이 아님을 명시


def test_threats_to_validity_names_every_known_limitation():
    m = re.search(r"###\s*6\.2\s*Threats to validity.*?(?=\n###\s|\n##\s)",
                  read(), re.DOTALL)
    assert m
    s = m.group(0).lower()
    for k in ("deterministic objective", "tolerance-dependent", "richardson",
              "shared", "fp64", "ky fan", "post-hoc", "spectral spread"):
        assert k in s, k


def test_conclusion_makes_no_overclaim():
    m = re.search(r"##\s*7\.\s*Conclusion.*?(?=\n##\s|\Z)", read(), re.DOTALL)
    assert m
    s = m.group(0).lower()
    for over in ("in general", "always", "any problem", "proves that", "never fail"):
        assert over not in s, over


def test_committed_paper_matches_committed_data():
    """정본이 데이터와 일치하는지 — 이 테스트가 §3.7의 실질적 보증이다."""
    r = render_paper.main(check=True)
    assert r["drift"] is False, "논문 표가 데이터와 어긋났다: render_paper를 실행하라"


def test_render_is_idempotent(tmp_path):
    p = tmp_path / "paper.md"
    shutil.copy(PAPER, p)
    render_paper.main(paper=str(p), fig_dir=str(tmp_path / "fig"))
    a = p.read_text(encoding="utf-8")
    render_paper.main(paper=str(p), fig_dir=str(tmp_path / "fig"))
    assert p.read_text(encoding="utf-8") == a


def test_check_mode_reports_drift_without_writing(tmp_path):
    p = tmp_path / "paper.md"
    shutil.copy(PAPER, p)
    render_paper.main(paper=str(p), fig_dir=str(tmp_path / "fig"))
    before = p.read_text(encoding="utf-8")
    assert "**Table 1.**" in before          # AUTO 블록 안의 실재 문자열
    p.write_text(before.replace("**Table 1.**", "**Table 1 TAMPERED.**", 1),
                 encoding="utf-8")
    r = render_paper.main(paper=str(p), fig_dir=str(tmp_path / "fig"), check=True)
    assert r["drift"] is True
    assert "TAMPERED" in p.read_text(encoding="utf-8")     # 쓰지 않았다


def test_tables_and_figures_are_numbered_in_order_of_appearance():
    """리뷰 지적: 표가 7→1-5→8,9→6→11→10, 그림이 1,2,4,5,3,6 순으로 나왔다."""
    t = read()
    tn = [int(m) for m in re.findall(r"\*\*Table (\d+)\.\*\*", t)]
    fn = [int(m) for m in re.findall(r"\*\*Figure (\d+)\.\*\*", t)]
    assert tn == sorted(tn) == list(range(1, len(tn) + 1)), tn
    assert fn == sorted(fn) == list(range(1, len(fn) + 1)), fn


def test_every_caption_carries_a_resolvable_source_and_sha():
    """'본문-데이터 불일치는 빌드 실패'라는 주장과 'git unknown'은 양립할 수 없다."""
    t = read()
    caps = re.findall(r"\*\*(?:Table|Figure) \d+\.\*\*.*", t)
    assert caps
    for c in caps:
        assert "(source: `" in c, c[:80]
        m = re.search(r"git ([0-9a-f]{8}|unknown)", c)
        assert m and m.group(1) != "unknown", c[-90:]


def test_neural_scope_is_limited_to_the_problems_actually_run():
    """P2·P4에 신경 arm을 돌리지 않았다는 사실이 초록·기여·§5·결론에 모두 있어야 한다."""
    t = flat()
    # §4에 4시드 pilot이 있으므로 "돌리지 않았다"는 내부모순이다 — 벤치마크 행렬로 한정한다
    assert "No neural arm is included in the 50-seed benchmark matrix on P2" in t
    assert "four-seed pilot of §4" in t
    assert "No neural arm was run on" not in t
    assert t.lower().count(
        "neural arm is included in the 50-seed benchmark matrix on p2") == 2, \
        "§1 범위 문장과 §5.2 두 곳에 모두 있어야 한다"
    assert "on every problem tested, classical bases" not in t
    # P4는 이제 돌렸다 — "안 돌렸다"가 남아 있으면 안 된다
    assert "no neural arm was run on P4" not in t
    # 초록을 C&S 250단어로 줄이면서 이 문구는 §1으로 옮겼다 — 사실이 남아 있는지를 본다
    flatt = " ".join(t.split())
    assert ("Three of them — the diagonal, exact and QR-orthonormalized projections — are "
            "additionally run on P4 in two dimensions" in flatt)


def test_p3_neural_limit_is_stated_as_derivation_not_measurement():
    t = flat()
    assert "no neural training run on P3 is reported, and none is needed" in t


def test_seed_count_exception_is_declared():
    t = flat()
    assert "One declared exception to the seed count" in t
    assert "3 seeds" in t


def test_repeated_run_discrepancy_is_disclosed():
    t = flat()
    assert "independent runs of an identical configuration" in t


def test_no_internal_metadata_or_revision_history_leaks():
    t = read()
    for leak in ("design_record", "audit_record", "scope_boundary",
                 "Correction to the v3 draft", "status: draft"):
        assert leak not in t, leak


def test_references_have_no_placeholders_and_every_number_is_cited():
    t = read()
    body, refs = t.split("## References", 1)
    nums = {int(n) for grp in re.findall(r"\[([0-9,\s]+)\]", body)
            for n in grp.split(",") if n.strip().isdigit()}
    # 목록은 `[n] Author…` 형태다 — Word 자동번호를 피하기 위한 정본 형식
    listed = {int(m) for m in re.findall(r"^\[(\d+)\] ", refs, re.MULTILINE)}
    assert listed == set(range(1, max(listed) + 1)), listed
    assert nums == listed, (sorted(nums ^ listed))
    assert "[verify" not in refs and "[XFEM" not in refs
    assert not re.search(r"^\[\d+\] \[", refs, re.MULTILINE), "빈 대괄호 항목이 남았다"
    assert not re.search(r"^\d+\. [A-Z]", refs, re.MULTILINE), \
        "마크다운 번호목록으로 남은 항목이 있다 — docx에서 Word 자동번호가 붙는다"


def test_inline_table_references_point_to_the_right_numbers():
    """표를 재번호했으면 본문의 상호참조도 따라와야 한다 — docx에서는 굵게가 벗겨져
    캡션과 본문 참조가 한 흐름으로 읽히므로, 순서가 어긋나면 바로 드러난다."""
    t = read()
    caps = {int(m) for m in re.findall(r"\*\*Table (\d+)\.\*\*", t)}
    body = re.sub(r"\*\*Table \d+\.\*\*[^\n]*", "", t)      # 캡션 제거 후 본문만
    refs = {int(m) for m in re.findall(r"\bTable (\d+)\b", body)}
    assert refs <= caps, f"존재하지 않는 표를 참조: {sorted(refs - caps)}"
    # 회계 통일 후 비용표는 10번이다 — 6번(장기변주)을 가리키면 안 된다
    import re as _re
    assert "same basis as the cost table" in " ".join(t.split())
    assert _re.search(r"declared per row\*\* \(Table \d+\)", t)


def test_citation_three_is_used_for_what_that_paper_actually_says():
    """[3]은 거리함수로 경계조건을 정확 부과하는 논문이다. 빈 자리표를 채우려고
    'mesh-free beam/plate variants'의 근거로 쓴 것은 인용-주장 불일치였다."""
    t = flat()
    import re as _re
    assert "mesh-free beam/plate variants" not in t
    # 번호는 빌드가 등장순으로 다시 매기므로 번호에 의존하지 않는다
    m = _re.search(r"R-equivalence of Sukumar and Srivastava \[(\d+)\]", t)
    assert m, "ADF 문장에서 [3]을 인용해야 한다"
    n = m.group(1)
    assert f"[{n}] N. Sukumar, A. Srivastava" in t, f"[{n}]이 Sukumar 항목을 가리켜야 한다"


def test_every_reference_number_is_cited_exactly_where_it_belongs():
    """한 번호가 서로 다른 두 주장의 근거로 쓰이면 하나는 참조가 실종된다."""
    t = read()
    body = t.split("## References", 1)[0]
    # Banerjee–Osborn(수치적분)은 한 주장에만 쓰인다 — 번호는 빌드가 정한다
    import re as _re
    m = _re.search(r"the effect of numerical integration on computed eigenvalues \[(\d+)\]",
                   body)
    assert m
    assert body.count(f"[{m.group(1)}]") == 1


def test_arm_and_basis_counts_agree_across_sections():
    t = flat()
    assert "the nine arms of §3.3" in t
    assert "**Neural arms.** Nine arms" in t
    assert "Eight arms" not in t
    assert "Seven arms" not in t
    assert "Nine classical basis families" not in t
    assert "The eight classical basis families of §3.3" in t


def test_order_of_magnitude_claims_name_their_comparison_pair():
    """'three to five orders'는 세 비교쌍을 뭉갠 것이었다. 이제 쌍마다 한 행을 갖는
    정본표에서 렌더링되고 본문은 그것을 가리킨다."""
    t = flat()
    assert "three to five orders" not in t
    for k in ("gradient-trained vs classical", "random-feature vs classical",
              "gradient-trained vs random-feature",
              "observed P1-to-P4 gap difference"):
        assert k in t, k


def test_section_six_one_counts_its_own_paragraphs_correctly():
    t = flat()
    assert "Two of the four results below" not in t


def test_paper_body_is_ascii_prose():
    """영문 원고다 — CSV 소스의 한국어가 표에 박힌 적이 있다(Table 7 note)."""
    t = read()
    bad = [ln for ln in t.splitlines()
           if any("가" <= c <= "힣" for c in ln)]
    assert not bad, bad[:3]


def test_cost_conversion_factor_was_replaced_by_direct_measurement():
    """환산계수는 같은 기계에서 2.53·3.80·7.82로 나왔다 — 헤드라인을 실을 수 없다.
    본문이 그 사실과 직접 측정으로 바꾼 것을 밝혀야 한다."""
    t = flat()
    assert "measured directly at one seed" in t
    assert "2.53, 3.80 and 7.82" in t
    assert "divided by a **measured** batching factor" not in t


def test_classical_cost_is_counted_from_scratch():
    """신경은 '처음부터 학습'으로, 고전은 '기저는 이미 있다고 치고' 재던 결함의 회귀.
    정규직교 단항식은 구성 71 ms 대 풀이 0.2 ms라 400배 과소계상이었다."""
    t = flat()
    assert "Everything needed to produce the answer from scratch" in t
    assert "one four-hundredth of its true cost" in t
    assert "Construction and solve are now reported in separate columns" in t


def test_cheapest_classical_selection_is_explained_not_just_asserted():
    """회계를 고치니 선택되는 기저가 바뀌었다 — §5.3의 'N>=16에서 실패'와 충돌해
    보이므로 그 긴장을 본문이 직접 풀어야 한다."""
    t = flat()
    assert "Orthonormalized monomials are no longer selected" in t
    assert "There is no contradiction with §5.3" in t


def test_no_cost_ratio_is_hand_written_in_the_prose():
    """같은 비율을 다섯 곳에 손으로 적었다가 서로 어긋났다(5.34/5.4/5.40–5.50/5.42,
    narrowing 1.6 대 1.7, two 대 three orders). 정본표만 값을 갖는다."""
    t = read()
    body = re.sub(r"\*\*Table \d+\.\*\*[^\n]*", "", t)          # 캡션 제외
    body = re.sub(r"\|[^\n]*\|", "", body)                       # 표 본문 제외
    for stale in ("10⁵·⁴", "10⁵·³", "10³·⁷", "10³·⁵", "10^5.4", "5.40–5.50",
                  "1.7 orders", "1.6 orders", "two orders more accurate",
                  "three orders more accurate"):
        assert stale not in body, stale


def test_stale_batching_factor_is_gone_from_every_caption():
    t = read()
    for stale in ("divided by the batching factor", "batching factor of 2.53",
                  "costs 34 % more"):
        assert stale not in t, stale


def test_p4_arm_is_disclosed_as_designed_after_the_p1_analysis():
    """§3.7이 '실행 전 동결'을 선언하므로, 나중에 설계된 arm은 그렇게 적어야 한다."""
    t = flat()
    assert "One arm was designed after the P1 analysis" in t
    assert "confirmatory experiment on a pre-specified scoring rule" in t


def test_appendix_b_discloses_the_two_dimensional_arm():
    t = read()
    assert "(b) on P4, two dimensions" in t
    assert "R-equivalence approximate distance" in t
    assert "12802" in t and "1200 points" in t


def test_section_six_one_states_what_p4_actually_tested():
    """P4는 기전을 **확인**한 것이 아니라 그 기전이 거기서 일어나지 않음을 보였다.
    "2차원으로 전이된다"로 쓰면 없던 것을 없앴다고 읽힌다."""
    t = flat()
    assert "Both results below are measured on P1 and P3 only" not in t
    assert "in two dimensions, on" in t
    assert "the mechanism does not arise" in t
    assert "P4 therefore does not exercise the mechanism" in t


def test_no_unnumbered_uncaptioned_ratio_table_survives():
    """§5.4에 번호·캡션·출처 없는 P4 비용표가 손으로 적혀 있었고, 폐기한 배칭계수로
    계산돼(590/2.53=233 s) 같은 비교를 3.7과 4.13 두 값으로 만들었다."""
    t = read()
    assert "| mode | neural: e_λ @ per attempt |" not in t
    for stale in ("@ 233 s", "@ 236 s", "@ 239 s"):
        assert stale not in t, stale
    # 모든 GFM 표는 캡션을 갖는다(프레임워크·사다리·잡음패널·부록은 AUTO 안에 있다)
    assert t.count("**Table ") >= 14


def test_p4_cost_is_measured_on_the_same_footing_as_the_beam():
    """P4는 앙상블 벽시계를 1회로 쓰다가(격차 과대계상) n_seeds=1 직접 측정으로 바꿨다.
    상한 유보 문장이 남아 있으면 데이터와 어긋난다."""
    t = flat()
    assert "measured at one seed, three repetitions, exactly as on the beam" in t
    for stale in ("upper bound** on the gap", "no single-seed measurement was made on P4",
                  "narrowing from P1 to P4 (mode 1, lower bound)"):
        assert stale not in t, stale
    # bound 열이 어느 행이 측정이고 어느 행이 경계인지 밝혀야 한다
    assert "`bound` column of Table" in t


def test_objective_ablation_reference_points_at_table_twelve():
    """정본표 신설로 objective ablation이 11→12로 밀렸다 — 참조가 따라와야 한다."""
    t = flat()
    import re as _re
    assert _re.search(r"absolute-trace column of Table \d+", t)
    assert _re.search(r"matched pair inside Table \d+", t)


def test_cross_references_point_at_a_table_about_the_right_thing():
    """번호가 존재하는지가 아니라 **그 번호의 표가 그 내용인지**를 본다.

    렌더러가 표 순서를 바꾸면 refs ⊆ caps는 계속 통과하면서 본문이 엉뚱한 표를
    가리킬 수 있다 — 직교성 논의가 초기화 표를 가리킨 채 통과한 적이 있다.
    문장의 단서와 캡션의 주제어를 짝지어 확인한다."""
    t = read()
    caps = {int(m.group(1)): m.group(2) for m in
            re.finditer(r"\*\*Table (\d+)\.\*\* (.*)", t)}
    # (본문 단서, 그 표 캡션에 있어야 하는 주제어)
    pairs = [("off-diagonal entry is 0.99 in both arms", "Orthogonality of the stored"),
             ("arm-(e) mode-1 row of", "Outcome attribution on P1"),
             ("with the spectral gap**", "Penalty deflation on P1"),
             ("Timing accounting is declared per row**", "Cheapest route to a target"),
             ("`bound` column of", "Canonical cost and accuracy ratios"),
             ("absolute-trace column of", "objective weighting changed")]
    for cue, topic in pairs:
        m = re.search(re.escape(cue) + r"[^.]*?Table (\d+)", t)
        assert m, f"본문 단서를 못 찾았다: {cue!r}"
        n = int(m.group(1))
        assert topic in caps.get(n, ""), \
            f"{cue!r} → Table {n}은 {caps.get(n, '없음')[:60]!r}, {topic!r}를 기대"


def test_stnet_claims_match_the_published_record():
    """[10] STNet(NeurIPS 2025)에 대해 원문에서 확인한 것만 주장한다.

    확인한 것: 연산자 deflation `L − QΣQᵀ`(식 9, 유사역행렬이 아니라 전치),
    Proposition B.2가 **orthogonal Q**와 normal 연산자·기지의 고유벡터를 가정한다는 것,
    ablation의 "converges exclusively to the first eigenfunction"(§4.5),
    2D Harmonic 표 6의 λ₂ 절대오차 3.04e-2 → 2.96e+1(모듈 제거 시), 쪽수 56537–56563.
    확인하지 못한 것(권 번호 등)은 쓰지 않는다.

    모드순 정확도 저하(1차원 λ₁ 대 λ₄)는 **일부러 쓰지 않는다** — deflation ablation과
    직접 관계가 없고, 2D·5D 결과가 같은 방식으로 단조롭지도 않아 경쟁 방법의 가장 불리한
    행을 고른 것으로 읽힐 수 있다."""
    t = read()
    body, refs = t.split("## References", 1)
    m = re.search(r"\[(\d+)\] H\. Wang, Y\. Jiang, J\. Wang, X\. Li, J\. Luo, H\. Dong, "
                  r"STNet", refs)
    assert m, "STNet 항목이 있어야 한다"
    n = m.group(1)
    entry = next(l for l in refs.splitlines() if l.startswith(f"[{n}] "))
    assert "56537–56563" in entry and "10.52202/085713-1695" in entry
    assert "Advances in Neural Information Processing Systems 38" not in entry, \
        "권 번호는 원문에 인쇄되어 있지 않다"
    assert "39th Conference on Neural Information Processing Systems" in entry
    # 본문 주장
    assert "`L − QΣQᵀ`" in body, "전치를 쓰는 연산자 deflation 형태를 밝혀야 한다"
    assert 'converges exclusively to the first eigenfunction' in body
    assert "3.04e-2 to 2.96e+1" in body
    assert "6e9" not in body, "모드순 저하 수치는 싣지 않기로 했다"
    # 정리가 가정하는 것을 정확히 적어야 한다
    assert "assumes an orthogonal `Q`" in body
    assert "states no mutual re-orthogonalization or Gram correction" in body
    # 과잉 동일시·방어적 표현 금지
    for bad in ("STNet is arm (a′)", "STNet uses the diagonal approximation",
                "same as arm (a′)", "not a straw man"):
        assert bad not in body, bad


def test_deep_ritz_is_not_cited_as_an_eigensolver():
    """[1] E & Yu의 deep Ritz는 변분 PDE 방법이고 고유값 솔버 논문이 아니다.

    "Differential-operator eigensolvers [1]"로 쓰면 라벨과 문헌이 어긋난다. [1]은 변분적
    기반으로 인용하고, 고유값 솔버 자리에는 실제 고유값 논문을 둔다."""
    t = read()
    body, refs = t.split("## References", 1)
    n = re.search(r"\[(\d+)\] W\. E, B\. Yu, The deep Ritz method", refs)
    assert n, "deep Ritz 항목이 있어야 한다"
    k = n.group(1)
    assert f"Differential-operator eigensolvers [{k}]" not in body
    assert f"deep Ritz method [{k}]" in body
    # 실제 고유값 솔버 문헌이 목록에 있고 본문에서 그 역할로 인용되는지
    han = re.search(r"\[(\d+)\] J\. Han, J\. Lu, M\. Zhou, Solving high-dimensional "
                    r"eigenvalue problems", refs)
    bs = re.search(r"\[(\d+)\] I\. Ben-Shaul, L\. Bar, D\. Fishelov, N\. Sochen, Deep "
                   r"learning solution of the eigenvalue problem", refs)
    assert han and bs, "고유값 솔버 문헌 두 건이 있어야 한다"
    for m in (han, bs):
        assert f"[{m.group(1)}]" in body, f"[{m.group(1)}]이 본문에서 인용되어야 한다"
    assert "eigensolvers proper include" in body


def test_projection_operators_use_the_papers_own_matrix_convention():
    """§3.3이 `Φ(ΦᵀWΦ)⁻¹ΦᵀWφ`로 쓰므로 Φ의 **열**이 기저함수다.

    그 규약에서 사영은 Π = ΦG⁻¹ΦᵀW이고 차이는 Φ(G⁻¹ − D⁻¹)ΦᵀW다. 한때 Φ와 Φᵀ가
    뒤바뀐 `Φᵀ(G⁻¹ − D⁻¹)ΦW`로 적혀 차원이 맞지 않았다 — 선형대수를 따라가면 바로 보인다."""
    t = read()
    assert "Φ(ΦᵀWΦ)⁻¹ΦᵀWφ" in t, "§3.3의 규약이 바뀌었다면 이 검사를 다시 써라"
    assert "Φᵀ(G⁻¹ − D⁻¹)ΦW" not in t, "Φ와 Φᵀ가 뒤바뀐 식이 남아 있다"
    # A_D는 사영이 아니다 — C = I가 아니면 멱등이 아니므로 Π로 쓰지 않는다
    assert "`Π_G = ΦG⁻¹ΦᵀW`" in t and "`A_D = ΦD⁻¹ΦᵀW`" in t
    assert "`Π_G − A_D = Φ(G⁻¹ − D⁻¹)ΦᵀW`" in t
    assert "Π_D" not in t, "대각 규칙을 사영 기호로 쓰면 안 된다"
    # 멱등성은 **조건부**로만 쓴다 — 무조건 사영으로 읽히면 안 된다
    assert "`A_D² = A_D`" in t
    i = t.index("`A_D² = A_D`")
    assert "C = I" in t[i:i + 400] or "if and only if" in t[i:i + 200]
    # 읽기 쉬운 형태로 정리했는지 + C의 정의가 명시됐는지
    assert "‖I − C‖₂" in t
    assert "C = D^{-1/2}GD^{-1/2}" in t
    # 정리 전 중간형태는 **증명 줄 안에서만** 허용한다. 주장을 그 형태로 쓰면 읽는 사람이
    # 상한과 실측값을 구분하지 못한다(원래 지적). 증명은 그 중간형태를 거쳐야 하므로
    # 인용 블록의 증명 줄은 예외다.
    outside = [ln for ln in t.splitlines() if "G^{1/2}D⁻¹G^{1/2}" in ln
               and not ln.lstrip().startswith(">")]
    assert not outside, outside


def test_mode3_leakage_has_an_explicit_definition():
    """중심 인과 증거이므로 캡션만으로 재현 가능해야 한다."""
    t = read()
    assert "‖D^{-1/2}ΦᵀWφ̃‖₂ / ‖φ̃‖_W" in t
    assert "cos∠(φ_k, φ̃)" in t
    assert "√2" in t, "근평행 저장쌍에서의 상한을 밝히면 1.41의 의미가 읽힌다"


def test_invariant_price_sentence_reads_the_right_way_round():
    """(a′)가 대가를 치르는 것은 불변량을 **잃은** 것이지 불변량 자체가 아니다."""
    t = read()
    flat = " ".join(t.split())
    assert "measured price of **losing** that invariant" in flat
    assert "measured price of the invariant" not in flat


def test_p4_problem_statement_is_closed_in_the_text():
    """벤치마크 논문이면 형상과 경계조건을 코드에 가지 않고 본문만으로 알 수 있어야 한다.

    좌표는 `neural/p4_neural.py`의 BLOCKS(세 단위블록)와 `problems/p4_lshape.py`의
    boundary 마스크(∂Ω 전체)에서 확인했다."""
    t = read()
    flat = " ".join(t.split())
    assert "`Ω = (0,2)² \\ ([1,2] × [1,2])`" in flat
    assert "`u = 0` on all of `∂Ω`" in flat
    assert "re-entrant corner at `(1,1)`" in flat
    assert "ν = 0.29" in flat and "L = E = ρ = 1" in flat


def test_abstract_has_no_dash_semicolon_artifact_and_no_overstated_cost():
    """— ; 는 편집 잔재다. 그리고 random-feature 격차는 1.88 orders이므로
    "several orders … than either neural family"는 한쪽에 대해 과하다."""
    t = read()
    # 줄바꿈이 문구 중간에 걸리므로 공백을 정규화해서 본다
    abstract = " ".join(
        t.split("## Abstract", 1)[1].split("## 1.", 1)[0].split())
    assert "— ;" not in abstract and "—;" not in abstract
    assert "several orders of magnitude more cheaply than either" not in abstract
    assert "cheaper and more accurate than both neural families" in abstract
    assert "is paid to sacrifice" not in " ".join(t.split()), "수동태 표현이 남아 있다"
    assert "creates an incentive to sacrifice" in abstract


def test_p1_and_p3_redundancy_claims_are_scoped_separately():
    """신경/무작위특징 중복성은 P1에서만 측정했다 — P3에서는 신경 arm을 학습하지 않았다."""
    t = read()
    flat = " ".join(t.split())
    assert "dominant algebraic failure mode across all three solver families" not in flat
    assert "On P1, redundancy appears as an algebraic hazard" in flat
    assert "no neural arm was trained on it" in flat


def test_figure_one_is_the_problem_schematic_and_the_rest_shift_down():
    """개요도를 그림 1로 넣으면 나머지 그림 번호가 하나씩 밀려야 한다."""
    t = read()
    caps = [int(m) for m in re.findall(r"\*\*Figure (\d+)\.\*\*", t)]
    assert caps == list(range(1, len(caps) + 1)), caps
    assert len(caps) == 7, "개요도 + 결과 그림 6개"
    first = re.search(r"\*\*Figure 1\.\*\* ([^\n]+)", t).group(1)
    assert "four dimensionless benchmark problems (schematic)" in first
    assert "no quantitative claim\nis read from it" in t or \
        "no quantitative claim is read from it" in " ".join(t.split())
    # 그림 1은 §3.1 문제정의 뒤, §3.2 앞에 온다
    i1 = t.index("![Figure 1](")
    assert t.index("### 3.1 Problems") < i1 < t.index("### 3.2 References")


def test_motivation_connects_the_problems_to_a_rotor_without_naming_the_machine():
    """네 문제의 선정 근거가 본문에 있어야 하고, 그 근거가 경계를 깨서는 안 된다.

    경계 정본 §1은 논문 2가 특정 기계와 동일시되는 것을 금지한다(`test_paper_stays_
    inside_its_scope`가 낱말로 막는다). 그래서 동기는 **회전 블레이드 디스크** 수준으로
    쓰고, 네 문제가 각각 무엇을 떼어내는지를 밝힌다."""
    flat_ = " ".join(read().split())
    assert "Why these four problems" in flat_
    assert "rotating\nbladed disc" in read() or "rotating bladed disc" in flat_
    for feature in ("slender cantilevers", "rotationally symmetric",
                    "localized\ncompliance", "re-entrant fillet"):
        assert feature.replace("\n", " ") in flat_, feature
    for iso in ("P1 isolates the cantilever", "P2 isolates the symmetry",
                "P3 isolates the localized compliance", "P4 isolates the corner"):
        assert iso in flat_, iso
    assert "nothing here depends on which rotor motivated the problems" in flat_


def test_p4_residual_failures_are_not_called_convergence_only():
    """Table 15는 모드 3에서 spurious 9 > non-converged 8이다 — "수렴 문제일 뿐"이 아니다.

    exact projection이 P4에서 없앤 것은 **하위모드 유인**(150회 전부 0)이고, 남은 실패는
    비수렴과 spurious가 섞여 있다. "not mode identification"은 표와 어긋난다."""
    flat_ = " ".join(read().split())
    assert "not mode identification" not in flat_
    assert "9 spurious against 8 non-converged" in flat_
    # 세 규칙 전부 basin 0이므로 "정확 사영이 없앴다"는 서술은 데이터와 어긋난다
    assert "removes lower-mode attraction there completely" not in flat_
    assert "nothing there for a projection to remove" in flat_


def test_abstract_scope_matches_the_actual_design():
    """아홉 arm은 P1에만 적용됐고 P4는 세 사영 규칙뿐이며 space-time은 3시드다."""
    ab = " ".join(read().split("## Abstract", 1)[1].split("**Keywords", 1)[0].split())
    import re as _re
    words = len(_re.sub(r"\*\*|\*", "", ab).split())
    assert words <= 250, f"C&S 초록 상한 250단어 — 현재 {words}"
    assert "All nine arms run on P1" in ab
    assert "three deflation arms run" in ab
    assert "control uses three" in ab
    assert "Every cell uses 50 seeds" not in ab


def test_lower_mode_basin_scope_matches_mode_six():
    """모드 6은 49 basin + 1 non-converged이므로 "100 % of failures"가 아니다."""
    flat_ = " ".join(read().split())
    assert "100 % of failures are" not in flat_
    assert "every *converged* incorrect" in flat_ or \
        "every converged incorrect" in flat_
    assert "mode 6 is 49 basin and 1 non-converged" in flat_


def test_objective_gain_direction_and_mixed_quantities():
    """개선비는 모드순으로 감소한다. 앙상블 확률과 단일 타이밍을 한 문장에 섞지 않는다."""
    flat_ = " ".join(read().split())
    assert "improvement decreases monotonically with mode order" in flat_
    assert "across 50 seeds p_correct ≥ 0.98 for every mode" in flat_
    assert "the measured per-attempt cost is 651 s" in flat_
    assert "delicate cancellation among high-frequency features" not in flat_
    assert "consistent with a shared-representation limitation" in flat_


def test_relative_lambda_bin_is_labelled_first_order():
    """Λ = ω²이므로 한 빈 변화는 2/p + 1/p²다 — 2/p는 1차 근사다(p=4에서 0.5625)."""
    flat_ = " ".join(read().split())
    assert "first-order relative Λ bin" in flat_
    assert "2/periods + 1/periods²" in flat_
    assert "0.5625 exact" in flat_
    assert "is prior information" in flat_
    # 4주기 항은 상한이다 — 추정기가 반 빈 이하를 구분하지 못한다
    assert "below half a bin" in flat_
    assert "an upper bound of about 0.25" in flat_
    # 폐기한 편향값과 비율이 서술에만 남고 결과로는 쓰이지 않아야 한다
    assert "113×" not in flat_


def test_p4_section_heading_and_body_are_observational():
    """§5.4 제목과 본문이 표와 같은 이야기를 해야 한다.

    제목이 "the re-entrant corner narrows the gap"이면 corner에 인과를 귀속하는데, 본문은
    스스로 corner를 분리할 수 없다고 인정한다. 본문의 "What limits the arm instead is
    convergence"도 모드 3의 spurious 9 > non-converged 8과 어긋난다."""
    flat_ = " ".join(read().split())
    assert "P4 — the observed gap is two orders smaller, and it does not reverse." in flat_
    assert "re-entrant corner narrows the gap" not in flat_
    assert "What limits the arm instead is convergence" not in flat_
    assert "Lower-mode attraction is removed; the remaining failures are a mixture of " \
           "non-convergence and spurious solutions" in flat_
    assert "0, 2 and 9 converge to a mode outside the target" in flat_


def test_headline_row_is_named_as_an_observation():
    """정본표의 마지막 행 이름도 관찰값이어야 한다 — "narrowing"은 원인을 함축한다."""
    t = read()
    assert "observed P1-to-P4 gap difference" in t
    assert "narrowing" not in t, "원고에 인과를 함축하는 행 이름이 남았다"
    assert "how much the re-entrant corner narrows" not in t


def test_protocol_is_called_pre_specified_not_pre_registered():
    """제3자 타임스탬프 등록이 없으므로 "pre-registered"는 과한 주장이다.

    실질은 그대로다 — 임계값을 고정한 커밋이 첫 벤치마크 데이터 커밋보다 앞선다. 그 근거를
    적고 용어만 정확히 한다."""
    flat_ = " ".join(read().split())
    # 용어를 쓰지 않는다고 **밝히는** 한 문장만 예외다
    disclaimer = 'the term used here is deliberately "pre-specified" and not "pre-registered"'
    assert disclaimer in flat_
    assert "pre-regist" not in flat_.replace(disclaimer, "").lower(), \
        "외부 등록을 함축하는 용어가 남았다"
    assert "Pre-specified analysis protocol" in flat_
    assert "no third-party timestamped registration" in flat_
    assert "the commit fixing the thresholds precedes the first committed benchmark data" \
        in flat_
    # 실질(무엇을 고정했는지)은 삭제하지 않는다
    for frozen in ("MAC ≥ 0.9", "|Δλ|/λ ≤ 0.05", "50 seeds per cell",
                   "Wilson 95 % intervals"):
        assert frozen in flat_, frozen


def test_e_model_row_declares_the_space_time_exception():
    """arm (g)는 같은 약형식의 다른 이산화가 아니라 다른 formulation이다."""
    flat_ = " ".join(read().split())
    assert "eliminated by design at the level of the **physical eigenproblem**" in flat_
    assert "Arm (g) is the declared exception" in flat_
    assert "one weak form per problem, discretized identically by every solver" not in flat_


def test_cost_claim_carries_its_hardware_qualifier():
    """고전/랜덤특징은 CPU, gradient arm은 공유 GPU다 — 알고리즘 복잡도 주장으로 읽히면 안 된다."""
    ab = " ".join(read().split("## Abstract", 1)[1].split("**Keywords", 1)[0].split())
    assert "In measured wall-clock cost on the reported hardware" in ab
    assert "at no measurable added cost" in ab
    assert "reweighting removes at no cost" not in ab


def test_space_time_arm_states_the_problem_it_actually_solves():
    """arm (g)가 P1과 같은 물리문제를 푼다는 것이 negative result의 전제다.

    시행함수만 적어 두면(`w = x²·N(x,t)`) 자유단 조건을 부과했는지 알 수 없고, 부과하지
    않았다면 다른 경계조건 문제를 푸는 것이므로 해석이 무너진다. 코드는 부과하며
    (`spacetime.py`의 w_bc 항, `test_spacetime_loss_actually_contains_the_free_end_residual`
    이 동작으로 확인), 본문·부록이 그것을 적어야 한다."""
    flat_ = " ".join(read().split())
    assert "`w_tt + w_xxxx = 0` on (0,1) × (0,T)" in flat_
    assert "`w_xx(1,t) = w_xxx(1,t) = 0`" in flat_
    assert "`w(x,0) = φ₁(x)`" in flat_ and "`w_t(x,0) = 0`" in flat_
    assert "the free end does not follow from the strong form and is imposed by" in flat_
    # 부록에 이산화·추정기 세부가 있어야 재현 가능하다
    for item in ("random collocation points per iteration", "uniform times over [0,T]",
                 "no taper or window", "excluding DC with no interpolation",
                 "initial-state residual", "free-end residual"):
        assert item in flat_, item


def test_boundary_enforcement_claim_excludes_the_space_time_arm():
    """"경계 강제로는 어떤 실패도 귀속되지 않는다"는 (g)의 자유단에는 성립하지 않는다."""
    flat_ = " ".join(read().split())
    assert ("none of the reported failures can be attributed to boundary enforcement"
            not in flat_)
    assert ("none of the reported failures of the variational arms can be attributed to "
            "boundary enforcement" in flat_)
    assert "Arm (g) is the exception and is scoped as such" in flat_
    assert "boundary enforcement is one of the things that arm is testing" in flat_


def test_generative_ai_declaration_is_present_and_correctly_placed():
    """Elsevier는 이 절을 **References 바로 앞**에 요구하고, AI를 저자로 올리는 것을 금지한다.

    이 원고는 산문·코드·검토 전반에 AI 도구를 썼으므로 선언 대상이다(문법 검사 같은 기초
    도구 예외에 해당하지 않는다)."""
    t = read()
    head = "## Declaration of generative AI and AI-assisted technologies"
    assert head in t
    assert t.index(head) < t.index("## References"), "References 앞에 와야 한다"
    flat_ = " ".join(t.split())
    # Elsevier 예시 문형을 유지한다
    assert "During the preparation of this work the author(s) used" in flat_
    assert ("the author(s) reviewed and edited the content as needed and take(s) full "
            "responsibility for the content of the published article") in flat_
    # 실제로 무엇에 썼는지 밝힌다 — 산문·코드·검토 세 용도
    for use in ("draft and revise manuscript text", "regression tests",
                "critical readings of the authors' own draft"):
        assert use in flat_, use
    # 그림이 생성형 이미지가 아니라는 점을 밝힌다
    assert "The figures are not AI-generated imagery" in flat_
    # AI를 저자·공동저자로 올리지 않는다
    author_block = t.split("## Abstract", 1)[0]
    for banned in ("Claude", "GPT", "Anthropic", "OpenAI"):
        assert banned not in author_block, f"저자 항목에 {banned}가 있다"


# ------------------------------------------- (b′) QR arm · fp32 · P4 3-arm (2026-08-26)
def test_the_qr_arm_and_the_precision_axis_are_declared_as_added_after_the_fact():
    """사후 추가를 날짜와 함께 선언하지 않으면 사전등록 절이 무의미해진다."""
    t = flat()
    assert "2026-08-26" in t
    i = t.index("2026-08-26")
    window = t[max(i - 700, 0):i + 700]
    assert "(b′)" in window or "QR" in window
    assert "were not changed" in window or "unchanged" in window


def test_the_two_projection_rules_are_stated_as_propositions_with_a_numerical_check():
    """§5.1의 대수는 이미 있었다 — 형식 서술로 올리고 수치 검증을 가리켜야 한다."""
    t = flat()
    assert "Proposition 1" in t and "Proposition 2" in t and "Corollary" in t
    assert "depends only on the span" in t or "only on the span" in t
    assert "machine precision" in t


def test_the_qr_arm_does_not_claim_a_reliability_gain_at_double_precision():
    """(b)와 (b′)은 fp64에서 같은 사영이다 — 신뢰도 향상을 주장하면 거짓이 된다."""
    t = flat()
    # 두 arm이 fp64에서 통계적으로 구분되지 않는다는 것을 **본문이 말해야** 한다
    assert "statistically indistinguishable" in t or "indistinguishable" in t
    assert "same projection" in t
    for bad in ("QR improves reliability", "QR is more reliable in float64",
                "the QR arm is more accurate in double precision"):
        assert bad not in t


def test_the_float32_claim_names_its_criterion():
    t = flat()
    assert "unit roundoff" in t
    # 정밀도 축이 **결과로** 서술돼야 한다 — 부록의 환경 서술만으로는 부족하다
    assert "at float32" in t
    assert "κ₂(G)·u" in t or "kappa_2(G) u" in t


def test_the_p4_three_arm_comparison_states_what_was_not_run():
    """모드 4–6은 참조 고유값이 선행 의존성이라 돌지 않았다 — 그 사실을 적어야 한다."""
    t = flat()
    assert "modes 4-6" in t or "modes 4–6" in t
    # 사유가 "참조 부재"가 아니라 **근접축퇴**여야 한다 — 참조는 실제로 존재한다
    assert "near-degenerate cluster" in t
    assert "12.105, 12.613 and 12.724" in t
    assert "which do not exist yet" not in t


def test_the_jitter_asymmetry_between_the_p4_arms_is_disclosed():
    """(b)에만 있는 1e-14 jitter를 숨기면 3-arm 비교가 공정하지 않다."""
    t = flat()
    assert "jitter" in t.lower() or "1e-14" in t


def test_the_qr_pooled_counts_in_the_prose_match_the_committed_data():
    """본문이 손으로 적은 476/500·474/500이 CSV와 같아야 한다."""
    from eigen_benchmark.drivers import manifest
    d = "docs/_generated/data/paper2"
    qr = sum(int(r["correct"]) for r in manifest.read_csv(f"{d}/p1_neural_qr.csv"))
    ex = sum(int(r["correct"]) for r in manifest.read_csv(f"{d}/p1_neural_mode_sweep.csv")
             if r["arm"] == "b_projection_exact")
    t = flat()
    assert f"recovers {qr} of 500 correct against {ex} of 500" in t, (qr, ex)


def test_every_qr_mode_has_overlapping_wilson_intervals_with_the_exact_arm():
    """"통계적으로 구분되지 않는다"는 주장을 데이터로 확인한다 — 산문이 앞서면 안 된다."""
    from eigen_benchmark.drivers import manifest
    d = "docs/_generated/data/paper2"
    qr = {int(r["mode"]): r for r in manifest.read_csv(f"{d}/p1_neural_qr.csv")}
    ex = {int(r["mode"]): r for r in manifest.read_csv(f"{d}/p1_neural_mode_sweep.csv")
          if r["arm"] == "b_projection_exact"}
    for m in sorted(qr):
        a, b = ex[m], qr[m]
        assert not (float(a["wilson_hi"]) < float(b["wilson_lo"])
                    or float(b["wilson_hi"]) < float(a["wilson_lo"])), m


# `test_the_diagnostic_numbers_quoted_for_the_qr_arm_match_its_own_run`은 삭제했다.
# 같은 것을 검사하되 기대값을 전부 상수로 박아 두어, 진단을 재실행하면 산문과 함께
# 이 테스트도 고쳐야 하는 두 번째 손글씨였다(실제로 7.4e10 → 9.6e10에서 그랬다).
# 대체: `test_the_qr_exposure_prose_quotes_its_own_runs_diagnostics`가 산문에서 수치를
# 정규식으로 뽑아 CSV와 대조하고, `test_the_attribution_of_the_residual_rests_on_an_
# ordering_the_data_supports`가 그 논증이 기대는 순서 관계를 검사한다.


def test_the_qr_overhead_quoted_in_the_prose_matches_the_measurement():
    """§5.4가 손으로 적은 1회 비용이 `p1_single_seed_cost.csv`와 같아야 한다."""
    from eigen_benchmark.drivers import manifest
    rows = {(r["arm"], int(r["mode"])): r for r in manifest.read_csv(
        "docs/_generated/data/paper2/p1_single_seed_cost.csv")}
    t_ = flat()
    for arm, mode in (("b_prime_projection_qr", 1), ("b_projection_exact", 1),
                      ("b_prime_projection_qr", 3), ("b_projection_exact", 3)):
        v = float(rows[(arm, mode)]["seconds_per_attempt"])
        assert f"{v:.1f} s" in t_, (arm, mode, v)
    # 범위가 실제로 겹치는지 — "세 번으로는 구분되지 않는다"의 근거
    for mode in (1, 2, 3):
        a = rows[("b_projection_exact", mode)]
        b = rows[("b_prime_projection_qr", mode)]
        assert not (float(a["seconds_max"]) < float(b["seconds_min"])
                    or float(b["seconds_max"]) < float(a["seconds_min"])), mode


def test_the_p4_three_arm_counts_in_the_prose_match_the_committed_data():
    """450회·basin 0이라는 서술이 실제 CSV와 같아야 한다."""
    from eigen_benchmark.drivers import manifest
    d = "docs/_generated/data/paper2"
    rows = (manifest.read_csv(f"{d}/p4_neural.csv")
            + manifest.read_csv(f"{d}/p4_neural_arms.csv"))
    n = sum(int(r["n"]) for r in rows)
    basin = sum(int(r["lower_mode_basin"]) for r in rows)
    assert basin == 0, basin
    t_ = flat()
    assert f"Not one of the {n} runs lands in the lower-mode basin" in t_
    assert f"in {n} runs" in t_
    assert {str(r["arm"]) for r in rows} == {
        "p4_neural_2d[exact projection]", "p4_neural_2d[diagonal projection]",
        "p4_neural_2d[qr projection]"}


def test_the_p4_precondition_is_measured_not_asserted():
    """"2차원에서는 전제가 성립하지 않는다"는 서술이 진단 CSV와 같아야 한다."""
    from eigen_benchmark.drivers import manifest
    rows = manifest.read_csv("docs/_generated/data/paper2/p4_span_diagnostics.csv")
    deep = max(rows, key=lambda r: int(float(r["k"])))
    assert float(deep["dev_identity"]) < 1e-14, deep["dev_identity"]
    t_ = flat()
    assert "2.2e-16" in t_ and "1.92" in t_
    assert "mass-orthogonal to machine precision" in t_
    # P1 대조값을 함께 적어야 비교가 성립한다
    assert "0.991 and 286" in t_


def test_the_paper_states_that_references_are_certified_for_the_shapes_too():
    """MAC 게이트가 인증되지 않은 양에 걸려 있었다 — 그 사실과 규칙이 본문에 있어야 한다."""
    t_ = flat()
    assert "certified for every quantity the scoring reads" in t_
    assert "0.887" in t_
    assert "converged in its own arithmetic" in t_


def test_the_framework_states_what_its_elimination_licenses():
    """분류가 풀이 과정만 덮는다는 사실과, 공유 목록에 계기가 들어간다는 사실을
    §3.0이 직접 말해야 한다 — 그러지 않으면 다음 사람이 같은 추론을 반복한다."""
    t_ = flat()
    assert "the inference the framework supports is an elimination" in t_
    assert "the scoring rules and the references they read" in t_
    assert "what a perfect solver would score" in t_


def test_the_threat_list_records_the_instance_and_points_at_the_logic():
    t_ = flat()
    assert "Arm-independence is an elimination, not an attribution" in t_
    assert "0.887" in t_
    # 이 논문의 결과가 그 오독에 기대지 않는다는 사실을 밝혀야 한다
    assert "No result reported in this paper rests on the earlier reading" in t_


def test_the_shape_certification_in_the_paper_matches_the_certification_run():
    """표의 인증값이 인증 실행과 같아야 한다 — 손으로 적으면 어긋난다."""
    from eigen_benchmark.drivers import manifest
    rows = manifest.read_csv("docs/_generated/data/paper2/reference_certification.csv")
    assert rows, "인증 실행 산출물이 없다"
    assert all(str(r["passes"]).lower() in ("true", "1") for r in rows)
    worst = max(float(r["deficit"]) for r in rows)
    assert worst < 1e-3, worst
    # 채점이 쓰는 모드 범위를 덮는가
    p1 = {int(r["label"].split()[-1]) for r in rows
          if r["problem"] == "P1" and r["quantity_certified"] == "mode shape"}
    assert p1 >= set(range(1, 15))


def test_the_attribution_scheme_names_the_instrument_as_its_own_component():
    """계측 오차에 칸이 없으면 그 원인이 남은 칸 중 하나로 잘못 배정된다 —
    실제로 그렇게 배정한 적이 있다(§6.2 항목 11)."""
    t_ = flat()
    assert "e_measurement" in t_
    assert "five sources" in t_
    assert "not a peer of the other four but a **precondition**" in t_
    assert "what score a perfect solution would receive there" in t_


def test_the_instrument_component_is_excluded_before_the_others():
    """순서가 주장의 일부다 — 계기를 나중에 보면 이미 잘못 귀속한 뒤다."""
    t_ = flat()
    assert "it is excluded first and the other four are considered only afterwards" in t_
    assert "That is why it is excluded first rather than considered alongside the others" in t_


# ─────────────────────────────────────────────────────────────────────────────
# §5.1의 깊은-모드 문단은 손으로 적혀 있고 진단 수치를 인용한다. 진단을 재실행하면
# 표는 렌더러가 갱신하지만 이 문단은 갱신되지 않는다 — 실제로 그렇게 어긋났고,
# 리뷰어가 표의 앵커에서 증분을 더해 끝점을 맞춰보다 발견할 수 있는 자리였다.
# ─────────────────────────────────────────────────────────────────────────────

def _deep_diag():
    import csv
    import os
    p = os.path.join("docs", "_generated", "data", "paper2",
                     "p1_fp64_deep_b_span_diagnostics.csv")
    rows = sorted(csv.DictReader(open(p)), key=lambda r: int(float(r["k"])))
    return rows


def test_the_deep_mode_prose_quotes_the_endpoint_the_data_reports():
    import re
    rows = _deep_diag()
    ku = float(rows[-1]["kappa_gram_times_eps"])
    m = re.search(r"κ₂\(G\)·u reaches only ([0-9.]+) at the deepest stored set", flat())
    assert m, "깊은-모드 문단의 끝점 문장을 찾지 못했다"
    assert abs(float(m.group(1)) - ku) < 5e-4, (
        f"산문 {m.group(1)} vs 데이터 {ku:.4g} — 재실행 뒤 문단이 갱신되지 않았다")


def test_the_quoted_increments_reproduce_the_endpoint_from_the_table_anchor():
    """리뷰어가 실제로 할 계산이다: 표의 앵커 κ₂에 산문의 증분을 더하면 끝점이 나와야 한다."""
    import math
    import re
    rows = _deep_diag()
    m = re.search(r"increments of κ₂\(G\) per stage decay across the extension — "
                  r"([0-9.,\s]+?) from k = (\d+) to k = (\d+)", flat())
    assert m, "증분 문장을 찾지 못했다"
    inc = [float(x) for x in m.group(1).replace("\n", " ").split(",")]
    k0, k1 = int(m.group(2)), int(m.group(3))
    assert len(inc) == k1 - k0, f"증분 {len(inc)}개 vs 구간 {k1 - k0}단"
    by_k = {int(float(r["k"])): r for r in rows}
    anchor = math.log10(float(by_k[k0]["kappa_gram"]))
    got = 10 ** (anchor + sum(inc)) * float(rows[0]["eps_machine"])
    want = float(by_k[k1]["kappa_gram_times_eps"])
    assert abs(got - want) / want < 0.05, (
        f"앵커+증분 = {got:.4g} vs 표의 끝점 {want:.4g} — 두 수가 같은 실행에서 나오지 않았다")


def _qr_diag():
    import csv
    import os
    p = os.path.join("docs", "_generated", "data", "paper2",
                     "p1_qr_span_diagnostics.csv")
    return sorted(csv.DictReader(open(p)), key=lambda r: int(float(r["k"])))


def test_the_qr_exposure_prose_quotes_its_own_runs_diagnostics():
    """§5.1의 노출 문단도 손으로 적혀 있다. 진단 요약 관례가 바뀌면 이 값들이 먼저 어긋난다."""
    import re
    rows = _qr_diag()
    deep = rows[-1]
    t = flat()

    def near(quoted, actual, tol=0.06):
        return abs(float(quoted) - actual) <= tol * abs(actual)

    m = re.search(r"climbs by eleven orders to ([0-9.e+]+) at the set handed to the "
                  r"mode-\d+ solve, so κ₂\(G\)·u reaches ([0-9.e-]+)", t)
    assert m, "노출 문단의 κ₂ 문장을 찾지 못했다"
    assert near(m.group(1), float(deep["kappa_gram"])), (
        f"산문 κ₂ {m.group(1)} vs 데이터 {float(deep['kappa_gram']):.3g}")
    assert near(m.group(2), float(deep["kappa_gram_times_eps"])), (
        f"산문 κ₂·u {m.group(2)} vs 데이터 {float(deep['kappa_gram_times_eps']):.3g}")

    m = re.search(r"agree to ([0-9.e-]+) at the first stage and to ([0-9.e-]+) at the deepest", t)
    assert m, "일치도 문장을 찾지 못했다"
    assert near(m.group(1), float(rows[0]["proj_agreement_rel"])), (
        f"산문 최초 일치도 {m.group(1)} vs {float(rows[0]['proj_agreement_rel']):.3g}")
    assert near(m.group(2), float(deep["proj_agreement_rel"])), (
        f"산문 최심 일치도 {m.group(2)} vs {float(deep['proj_agreement_rel']):.3g}")

    m = re.search(r"stays below ([0-9.e-]+) over the same range", t)
    assert m, "직교성 결함 상한 문장을 찾지 못했다"
    worst = max(float(r["qr_orth_defect"]) for r in rows)
    assert worst <= float(m.group(1)), (
        f"산문 상한 {m.group(1)} < 실측 최대 {worst:.3g} — 상한이 거짓이다")


def test_the_paper_states_which_seeds_the_diagnostics_summarize():
    """시드 수가 두 가지 뜻으로 쓰인다 — 셀의 시드와 진단이 요약한 시드. 둘이 달랐던 적이
    있고, 그때 같은 κ₂(G)가 표마다 3배 어긋났다. 규칙을 본문이 말해야 캡션이 검증 가능해진다."""
    t = flat()
    assert "Two senses of the seed count" in t
    assert ("diagnostics logged inside a scored run summarize every seed of that cell"
            in t)
    assert "dedicated diagnostic runs state their own seed count in the caption" in t


def test_the_paper_says_why_the_precision_axis_stops_at_float32():
    """리뷰어의 첫 질문이다. "라이브러리가 막아서"는 방법의 한계가 아니므로 답이 될 수 없고,
    실제 이유(게이트가 산술보다 곱다)는 이 논문 자신의 e_measurement 사례다."""
    t = flat()
    assert "two usable precisions" in t
    assert "Appendix D" in t
    # 구현 한계가 아니라 채점 한계였다는 사실을 한계 목록이 직접 말해야 한다
    assert "not because either arm cannot be implemented there" in t
    assert "the convergence gate is finer than the arithmetic" in t


def test_section_five_reports_that_the_library_obstacle_was_removed_not_worked_around():
    """"라이브러리에 커널이 없다"는 방법에 관한 진술이 아니다. 그 구분을 본문이 해야
    독자가 arm (b)의 한계와 PyTorch의 한계를 섞지 않는다."""
    t = flat()
    assert "no `lu`, `cholesky` or `eigh` kernel" in t
    assert "a statement about a library and not about the method" in t


def test_the_attribution_of_the_residual_rests_on_an_ordering_the_data_supports():
    """§5.1은 잔차를 그람 계수 탓으로 돌린다. 그 논증은 세 값의 **순서**에 걸려 있다:
    직교성 결함 ≪ 두 사영의 불일치 ≪ κ₂(G)·u. 숫자만 맞추는 테스트는 순서가 뒤집혀도
    통과하므로, 순서 자체를 검사한다."""
    rows = _qr_diag()
    deep = rows[-1]
    od = float(deep["qr_orth_defect"])
    dis = float(deep["proj_agreement_rel"])
    ku = float(deep["kappa_gram_times_eps"])
    assert od < dis < ku, f"순서가 무너졌다: 결함 {od:.3g}, 불일치 {dis:.3g}, 한계 {ku:.3g}"
    # 직교성 쪽으로는 여유가 커야 "설명할 수 없다"가 성립한다
    assert dis / od > 1e4, f"불일치가 직교성 결함의 {dis / od:.3g}배뿐이다"
    # 그람 한계는 넘지 않아야 한다 — 넘으면 한계가 한계가 아니다
    assert dis < ku


def test_the_seed_convention_paragraph_says_where_its_own_number_can_be_checked():
    """"k = 10에서 최대 3배"는 구 8시드 요약과의 비교인데, 진단은 시드별 값이 아니라
    중앙값만 저장하므로 공개 CSV로는 재현되지 않는다. 재현 불가한 수치는 출처를 밝혀야
    한다 — 밝히지 않으면 리뷰어가 확인하려다 못 찾는다."""
    t = flat()
    assert "the earlier numbers cannot be recomputed from the released CSVs" in t
    assert "2026-08-28-seed-convention-unification.md" in t


def test_the_reliability_bands_quoted_for_the_extension_are_true_of_the_data():
    """§5.1은 깊이에 따른 쇠퇴를 구간별 수치로 요약한다. 그 구간들이 실제 셀을 덮는지
    검사한다 — "1.00 through mode 7"이라 적혀 있었지만 모드 5는 0.98, 모드 7은 0.96이었다."""
    import csv
    import os
    import re
    d = os.path.join("docs", "_generated", "data", "paper2")
    cells = []
    for f in ("p1_fp64_deep_b_stress.csv", "p1_fp64_deep_qr_stress.csv"):
        cells += list(csv.DictReader(open(os.path.join(d, f))))

    def band(lo, hi):
        v = [float(r["p_correct"]) for r in cells if lo <= int(r["mode"]) <= hi]
        return min(v), max(v)

    t = flat()
    m = re.search(r"Reliability stays at or above ([0-9.]+) through mode (\d+)", t)
    assert m, "얕은 구간 서술을 찾지 못했다"
    floor_, upto = float(m.group(1)), int(m.group(2))
    assert band(1, upto)[0] >= floor_, (
        f"모드 ≤{upto}의 최소 {band(1, upto)[0]:.2f} < 산문의 하한 {floor_}")

    m = re.search(r"sits at ([0-9.]+)[–-]([0-9.]+) at mode (\d+)", t)
    assert m, "모드 8 구간 서술을 찾지 못했다"
    lo_, hi_, md_ = float(m.group(1)), float(m.group(2)), int(m.group(3))
    assert (lo_, hi_) == band(md_, md_), f"모드 {md_}: 산문 {lo_}-{hi_} vs 데이터 {band(md_, md_)}"

    m = re.search(r"runs between ([0-9.]+) and ([0-9.]+) across modes (\d+)-(\d+)", t)
    assert m, "중간 구간 서술을 찾지 못했다"
    lo_, hi_ = float(m.group(1)), float(m.group(2))
    a, b_ = int(m.group(3)), int(m.group(4))
    assert (lo_, hi_) == band(a, b_), f"모드 {a}-{b_}: 산문 {lo_}-{hi_} vs 데이터 {band(a, b_)}"


def test_the_budget_control_claim_matches_the_control_run():
    """§5.1은 깊이에 따른 잔차를 예산 탓으로 돌린다. 그 주장은 모드 14의 5배 예산 대조에
    걸려 있고, 대조가 성립하려면 모드 1-13이 확장 실행과 판정까지 같아야 한다."""
    import csv
    import os
    import re
    d = os.path.join("docs", "_generated", "data", "paper2")
    ctl = {int(r["mode"]): r for r in
           csv.DictReader(open(os.path.join(d, "p1_fp64_budget14_stress.csv")))}
    ext = {int(r["mode"]): r for r in
           csv.DictReader(open(os.path.join(d, "p1_fp64_deep_b_stress.csv")))}
    keys = ("correct", "lower_mode_basin", "spurious", "non_converged")
    for m in range(1, 14):
        assert all(ctl[m][k] == ext[m][k] for k in keys), (
            f"모드 {m}이 두 실행에서 다르다 — 예산만 다른 대조가 아니다")
    assert int(float(ctl[14]["iters"])) == 5 * int(float(ext[14]["iters"]))

    t = flat()
    m = re.search(r"non-convergence residue falls from (\d+) of (\d+) to (\d+) of \d+", t)
    assert m, "미수렴 잔차 문장을 찾지 못했다"
    assert (int(m.group(1)), int(m.group(3))) == (int(ext[14]["non_converged"]),
                                                  int(ctl[14]["non_converged"]))
    m = re.search(r"spurious count does not move \((\d+) of \d+ against (\d+) of \d+\)", t)
    assert m, "가짜 판정 문장을 찾지 못했다"
    assert (int(m.group(1)), int(m.group(2))) == (int(ext[14]["spurious"]),
                                                  int(ctl[14]["spurious"]))
    # 사전등록 규칙(Wilson 겹침)으로는 p_correct가 분리되지 않는다 — 그 사실을 밝혀야 한다
    lo_a, hi_a = float(ext[14]["wilson_lo"]), float(ext[14]["wilson_hi"])
    lo_b, hi_b = float(ctl[14]["wilson_lo"]), float(ctl[14]["wilson_hi"])
    assert not (hi_a < lo_b or hi_b < lo_a), "구간이 분리됐다면 서술을 강화해야 한다"
    assert "not separated on that outcome" in t
