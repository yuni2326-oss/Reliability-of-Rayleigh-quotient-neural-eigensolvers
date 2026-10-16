"""논문 마크다운의 `<!-- AUTO:name -->` 블록에 렌더 결과를 주입한다.

표시를 남겨두므로 **멱등**이다 — 데이터를 다시 돌린 뒤 같은 명령으로 갱신한다.
표시와 블록의 짝이 안 맞으면 조용히 넘기지 않고 실패한다: 표 하나가 조용히 빠진 논문은
표가 틀린 논문보다 찾기 어렵다.
"""
from __future__ import annotations

import os
import re

from . import figures, md, tables

BLOCK_RE = re.compile(r"<!-- AUTO:(\w+) -->.*?<!-- /AUTO -->", re.DOTALL)

# **표 번호는 본문 등장 순서다.** 리뷰 지적: 이전 판은 7 → 1–5 → 8,9 → 6 → 11 → 10 순으로
# 나와 읽는 순서와 번호가 어긋났다. 키는 의미를 유지하고 번호만 순서에 맞춘다.
TABLES = {
    "table_references": (tables.table7_references, 1),        # §3.2
    "table_attribution": (tables.table1_attribution, 3),      # §5.1
    "table_penalty": (tables.table2_penalty, 6),
    "table_quadrature": (tables.table3_quadrature, 7),
    "table_orthogonality": (tables.table_orthogonality, 2),
    "table_fp32": (tables.table_fp32, 4),                     # §5.1 정밀도 축
    "table_deep_modes": (tables.table_deep_modes, 5),         # §5.1 깊은 모드
    "table_warm_cold": (tables.table4_warm_cold, 8),
    "table_longrun": (tables.table5_longrun, 9),
    "table_degeneracy": (tables.table_degeneracy, 10),        # §5.2
    "table_conditioning": (tables.table8_conditioning, 11),   # §5.3
    "table_rank": (tables.table9_effective_rank, 12),
    "table_cost": (tables.table6_cost, 13),                   # §5.4
    "table_headline": (tables.table_headline, 14),            # 정본 비율
    "table_objective": (tables.table11_objective_weighting, 15),
    "table_spacetime": (tables.table10_spacetime, 16),
    "table_p4": (tables.table_p4, 17),
}

# **그림 번호도 본문 등장 순서다**(이전 판은 1,2,4,5,3,6으로 나왔다).
FIGURE_ORDER = ("fig1_benchmark_problems",
                "fig2_mode_reliability", "fig3_error_vs_dof", "fig4_conditioning",
                "fig5_rank_collapse", "fig6_accuracy_cost_pareto",
                "fig7_spacetime_negative")

FIGURE_CAPTION = {
    "fig1_benchmark_problems":
        "The four dimensionless benchmark problems (schematic). Color shows normalized "
        "modal displacement, blending into plain geometry; amplitudes are exaggerated for "
        "visibility. (a) P1, clamped–free Euler–Bernoulli beam (mode 3 shown). (b) P2, "
        "annular Kirchhoff plate, clamped inner / free outer, a/b = 0.42, ν = 0.29; "
        "degenerate cos mθ / sin mθ pairs are scored as eigenspaces (m = 2, n = 0 shown); "
        "reference only. (c) P3, P1's beam with a zero-width rotational spring at "
        "x_c/L = 0.2; ⟦u⟧ = 0, the rotation jump ⟦u′⟧ carries the spring energy "
        "(k̂ = 1, mode 2 shown). (d) P4, L-shaped plane-stress eigenproblem, clamped "
        "boundary, re-entrant corner (mode 1 shown). All displayed fields are computed "
        "from the stated equations; the figure is illustrative and no quantitative claim "
        "is read from it.",
    "fig2_mode_reliability":
        "Mode-order reliability on P1. Diagonal-approximation projection (a′) holds "
        "1.00 through mode 2 and drops to 0.00 from mode 3 on; the two exact-projection "
        "routes (b) and (b′) show no such wall anywhere, and the extension carries them "
        "to mode 16 without a single lower-mode-basin outcome. What replaces the wall is "
        "a gradual decline to about one half, dominated by the convergence gate rather "
        "than by misidentification. Bars are 95 % Wilson intervals over 50 seeds. The "
        "shaded region beyond the dashed line is past the pre-specified grid: arm (b′) "
        "and the modes beyond 10 were added in revision, and the two projection arms "
        "reproduce the pre-specified cells exactly where the two overlap, which is why "
        "they are drawn as one curve each.",
    "fig3_error_vs_dof":
        "Convergence on the common P3 weak form. The C¹ global bases saturate at a "
        "fixed error because the jump term vanishes identically for them, so refining "
        "the space cannot help — an e_approx floor, not an algebraic one.",
    "fig6_accuracy_cost_pareto":
        "Accuracy against cost on P1 mode 1. The three families occupy separate regions: "
        "the classical points sit both to the left of and below every neural point, i.e. "
        "cheaper and more accurate at once. The ratios are not restated here; they are in "
        "the canonical ratio table, and the timing accounting is in the cost table.",
    "fig4_conditioning":
        "Conditioning on P1. Only the transformed-operator normalization is a property "
        "of the trial space; the raw condition numbers reflect the coordinates. Crosses "
        "mark Cholesky failures, plotted at their raw κ(M).",
    "fig5_rank_collapse":
        "Effective rank and pairwise collinearity of the trained neural bases. The "
        "separate-net basis (d) never attains its nominal dimension at any tolerance, "
        "while the shared-trunk subspace (e) retains full rank; both are heavily "
        "correlated, so collinearity alone does not separate them.",
    "fig7_spacetime_negative":
        "Space–time PINN on P1. Lengthening the time window lowers the PDE residual while "
        "the eigenvalue error rises: the objective is not the quantity of interest. The bin "
        "curve is the estimator's resolution, not an error floor — the record length is "
        "exactly an integer number of reference periods, so the reference frequency falls on "
        "a bin exactly and a peak there only bounds the error by half a bin. The 4-period "
        "point sits at machine zero for that reason and is an upper bound; from 16 periods "
        "the error exceeds one bin and is resolved.",
}


def inject(text: str, blocks: dict) -> str:
    """`<!-- AUTO:k -->…<!-- /AUTO -->` 사이를 blocks[k]로 치환. 짝이 안 맞으면 실패."""
    found = {m.group(1) for m in BLOCK_RE.finditer(text)}
    missing_marker = sorted(set(blocks) - found)
    if missing_marker:
        raise ValueError(f"논문에 자리(<!-- AUTO:… -->)가 없는 블록: {missing_marker}")
    missing_block = sorted(found - set(blocks))
    if missing_block:
        raise ValueError(f"렌더 결과가 없는 표시: {missing_block}")

    def sub(m):
        k = m.group(1)
        return f"<!-- AUTO:{k} -->\n{blocks[k].strip()}\n<!-- /AUTO -->"

    return BLOCK_RE.sub(sub, text)


def render_all(data_dir: str = tables.DATA_DIR, fig_dir: str = figures.FIG_DIR,
               paper_dir: str = os.path.join("docs", "paper2-cs")) -> dict:
    """표 10개 + 그림 6개의 마크다운 조각. 그림 경로는 논문 파일 기준 상대경로다."""
    out = {k: fn(data_dir, n) for k, (fn, n) in TABLES.items()}
    assert set(FIGURE_ORDER) == set(figures.ALL), "그림 순서 목록과 렌더러 목록 불일치"
    out["appendix_a"] = tables.appendix_a_nondimensional()
    out["appendix_b"] = tables.appendix_b_disclosure(data_dir)
    out["appendix_c"] = tables.appendix_c_numerics(data_dir)
    out["appendix_d"] = tables.appendix_d_bfloat16(data_dir)
    out["availability"] = tables.availability_record(data_dir)
    rel = os.path.relpath(fig_dir, paper_dir)
    for i, name in enumerate(FIGURE_ORDER, start=1):
        path = figures.ALL[name](data_dir, fig_dir)
        src = os.path.join(rel, os.path.basename(path)).replace(os.sep, "/")
        cap = md.caption(i, FIGURE_CAPTION[name], os.path.basename(path),
                         tables.sha(data_dir)).replace("**Table", "**Figure")
        out[f"figure_{i}"] = f"![Figure {i}]({src})\n\n{cap}"
    return out
