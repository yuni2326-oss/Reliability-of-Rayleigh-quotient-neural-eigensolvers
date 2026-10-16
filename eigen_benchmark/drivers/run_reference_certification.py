"""참조 인증 — 채점이 참조에서 읽는 **모든 양**을 더 높은 정밀도와 대조한다.

§3.2의 불확실도 열은 **고유값**을 인증한다(`quantity` 열이 (βL)⁴다). 그런데 사전등록
판정은 두 가지를 쓴다: e_λ는 고유값과, MAC은 **모드형**과 비교한다. 두 양은 수치적으로
전혀 다른 경로로 나온다 — P1에서 β는 cos·cosh + 1 = 0의 근찾기로 얻어 조건이 매우 좋지만
(상대오차 1.7e-16), 모드형을 교과서 형태로 평가하면 e^β/2 크기의 두 항이 O(1)로 상쇄돼
β ≳ 40에서 유효자리가 사라진다. 하나를 인증해도 다른 하나에 대해 아무 말도 하지 않는다.

실제로 그 틈으로 결함이 빠졌다: 모드 14의 참조 모드형이 자기 자신(고정밀 평가)과 MAC
0.887이어서, **정확한 해를 찾은 솔버가 게이트 0.9에서 spurious로 분류**됐다.

이 드라이버는 각 문제의 모드형 참조를 그 문제에 맞는 더 높은 정밀도와 대조한다.

  · P1  해석 모드형 vs mpmath 고정밀 (같은 식; 기본 30자리 — 목표 1e-9의 스무 자릿수 위)
  · P2  Bessel 조합 vs mpmath 재조립 (같은 계수)
  · P4  참조 격자(n = 8)의 FEM 모드 vs 세밀 격자(n = 16), 채점이 실제로 쓰는 구적점에서
  · P3  모드형이 채점에 쓰이지 않는다는 사실 자체를 기록

**torch를 쓰지 않는다.** 렌더·분석 경로(호스트 venv)에서 돌아야 하므로 P4 구적점을 numpy로
직접 만든다. 그 정의가 솔버와 어긋나면 인증이 무의미하므로 컨테이너 테스트가 강제한다.
"""
from __future__ import annotations

import os

import numpy as np

from .. import metrics as mt
from ..problems import p1_beam as p1
from ..problems import p4_lshape as p4
from ..quadrature import gauss_legendre
from ..reference import bessel_annulus as ba
from . import manifest

# `neural/p4_neural.py`의 BLOCKS와 같아야 한다(그 모듈은 torch를 끌어온다).
P4_BLOCKS = (((0.0, 1.0), (0.0, 1.0)),
             ((1.0, 2.0), (0.0, 1.0)),
             ((0.0, 1.0), (1.0, 2.0)))
# 허용치는 **참조의 종류**에 따라 다르다. 닫힌 형식은 반올림 수준까지 요구할 수 있지만,
# 이산해는 이산화 오차가 하한이므로 같은 잣대를 들이대면 의미가 없다. 어느 쪽이든 기준은
# 하나다 — MAC 게이트 0.9에서 남는 여유 0.1에 비해 무시할 만한가.
MAC_TOL = 1e-9          # 닫힌 형식(P1·P2): 반올림 수준
CURV_TOL = 1e-9
DISCRETE_TOL = 1e-3     # 이산 참조(P4): 게이트 여유 0.1의 백분의 일


def required_dps(scale: float, digits: int = 25) -> int:
    """지수적으로 커지는 항이 상쇄될 때 필요한 고정밀 자릿수.

    **계기도 인증해야 한다.** 처음 이 드라이버를 고정 dps = 30으로 돌렸을 때 모드 20의
    곡률이 1.8e-5 어긋나는 것으로 나왔는데, 원인은 fp64 식이 아니라 **대조 쪽**이었다
    — e^β ≈ 4e26이라 30자리로는 상쇄 뒤에 세 자리밖에 남지 않는다. dps를 45로 올리자
    2.5e-15로 떨어지고 더 올려도 변하지 않았다.

    필요한 자릿수는 상쇄되는 항의 크기 log10(e^scale) = scale/ln(10)에 원하는 유효자릿수를
    더한 것이다."""
    return int(scale / 2.302585092994046) + int(digits)


def _nodes(n_q: int = 256):
    x, w = np.polynomial.legendre.leggauss(n_q)
    return 0.5 * x + 0.5, 0.5 * w


def _p4_points(n_per_block: int = 20):
    """`p4_neural.quadrature`와 같은 점·가중 — MAC이 실제로 평가되는 자리."""
    P, W = [], []
    for (x0, x1), (y0, y1) in P4_BLOCKS:
        xs, wx = gauss_legendre(n_per_block, x0, x1)
        ys, wy = gauss_legendre(n_per_block, y0, y1)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        P.append(np.stack([X.ravel(), Y.ravel()], 1))
        W.append(np.outer(wx, wy).ravel())
    return np.concatenate(P), np.concatenate(W)


def _mass_mac_2d(A, B, w) -> float:
    ab = float((w * (A[:, 0] * B[:, 0] + A[:, 1] * B[:, 1])).sum())
    aa = float((w * (A[:, 0] ** 2 + A[:, 1] ** 2)).sum())
    bb = float((w * (B[:, 0] ** 2 + B[:, 1] ** 2)).sum())
    if aa <= 0 or bb <= 0:
        return float("nan")
    return ab ** 2 / (aa * bb)


def _row(problem, label, quantity, method, against, agreement, tol, note=""):
    deficit = 1.0 - float(agreement)
    return {"problem": problem, "label": label, "quantity_certified": quantity,
            "method": method, "certified_against": against,
            "agreement": float(agreement), "deficit": deficit,
            "tolerance": tol, "passes": bool(deficit <= tol), "note": note}


def _certify_p1(rows, n_modes: int, dps: int, n_q: int) -> None:
    from mpmath import cos, cosh, mp, mpf, sin, sinh

    x, w = _nodes(n_q)
    W = np.diag(w)
    old = mp.dps
    try:
        for n in range(1, n_modes + 1):
            bf = float(p1.beta_roots(n)[n - 1])
            use = max(int(dps), required_dps(bf))
            mp.dps = use
            b = mpf(bf)
            s = (cosh(b) + cos(b)) / (sinh(b) + sin(b))
            ex = np.array([float((cosh(b * xx) - cos(b * xx))
                                 - s * (sinh(b * xx) - sin(b * xx))) for xx in x])
            ex2 = np.array([float(b ** 2 * ((cosh(b * xx) + cos(b * xx))
                                            - s * (sinh(b * xx) + sin(b * xx))))
                            for xx in x])
            rows.append(_row("P1", f"mode {n}", "mode shape",
                             "analytic Euler-Bernoulli, exponential form",
                             "same formula at higher precision",
                             mt.mac(p1.analytic_mode(x, n), ex, W), MAC_TOL,
                             f"comparison at dps = {use}"))
            got2 = p1.analytic_mode_d2(x, n)
            rel = float(np.abs(got2 - ex2).max() / max(np.abs(ex2).max(), 1e-300))
            rows.append(_row("P1", f"mode {n}", "mode curvature",
                             "analytic second derivative, exponential form",
                             "same formula at higher precision",
                             1.0 - rel, CURV_TOL,
                             "agreement reported as 1 - max relative error; "
                             f"comparison at dps = {use}"))
    finally:
        mp.dps = old


def _certify_p2(rows, data_dir: str, dps: int, n_r: int = 96) -> None:
    from mpmath import mp, mpf

    geo = getattr(ba, "P2_GEOMETRY", None)
    a = float(geo["a"]) if geo and "a" in geo else 0.42
    r = np.linspace(a, 1.0, n_r)
    w = np.full_like(r, (1.0 - a) / n_r) * r          # 축대칭 반경 가중
    W = np.diag(w)
    old = mp.dps
    try:
        for row in manifest.read_csv(os.path.join(data_dir, "p2_reference.csv")):
            m, k = int(row["m"]), float(row["kb"])
            use = max(int(dps), required_dps(k))     # I_m(kb) ~ e^{kb}
            mp.dps = use
            c = ba._null_coefficients(k, m)
            ex = np.array([float(sum(mpf(float(c[j])) * ba._derivs_mp(mpf(k), mpf(float(rr)), m)[j][0]
                                     for j in range(4))) for rr in r])
            rows.append(_row("P2", f"m={m}, n={row['radial_order']}", "mode shape",
                             "exact Bessel combination (fp64)",
                             "same coefficients at higher precision",
                             mt.mac(ba.mode_shape(r, k, m), ex, W), MAC_TOL,
                             ("degenerate pair; " if str(row["degenerate"]).lower()
                              in ("true", "1") else "") + f"comparison at dps = {use}"))
    finally:
        mp.dps = old


def _certify_p4(rows, n_ref: int, n_fine: int, beta: float, n_modes: int) -> None:
    pts, w = _p4_points()
    coarse = p4.solve(n_ref, beta=beta, n_modes=max(n_modes + 3, 6))
    fine = p4.solve(n_fine, beta=beta, n_modes=max(n_modes + 3, 6))
    lam_c, lam_f = np.sort(coarse["Lam"]), np.sort(fine["Lam"])
    for m in range(1, n_modes + 1):
        A = np.asarray(p4.mode_at(coarse, m, pts))
        B = np.asarray(p4.mode_at(fine, m, pts))
        gap = min((abs(lam_c[m - 1] - lam_c[j]) / lam_c[m - 1]
                   for j in range(len(lam_c)) if j != m - 1), default=float("inf"))
        # 근접축퇴는 **고유값 기준**이 세 모드를 못 가른다는 뜻이지, 모드형이 분해되지
        # 않는다는 뜻이 아니다 — 격자를 두 배로 세밀하게 해도 개별 모드형이 유지되는지가
        # 여기서 측정된다. 그 구분을 note에 명시한다.
        note = (f"adjacent eigenvalue gap {gap:.3f} is inside the pre-specified "
                "tolerance 0.05, so the simple-mode rule cannot separate this mode by "
                "eigenvalue; the shape itself is individually resolved, as this row shows"
                if gap < mt.ELAM_MAX else "")
        rows.append(_row("P4", f"mode {m}", "mode shape",
                         f"graded Q2 FEM, n = {n_ref} (the mesh the scorer uses)",
                         f"same construction at n = {n_fine}",
                         _mass_mac_2d(A, B, w), DISCRETE_TOL, note))


def main(outdir=None, n_modes_p1: int = 20, dps: int = 30, n_q: int = 192,
         p4_n_ref: int = 8, p4_n_fine: int = 16, p4_beta: float = 3.0,
         p4_modes: int = 6) -> dict:
    d = manifest.ensure_outdir(outdir)
    rows: list = []
    _certify_p1(rows, n_modes_p1, dps, n_q)
    _certify_p2(rows, d, dps)
    _certify_p4(rows, p4_n_ref, p4_n_fine, p4_beta, p4_modes)
    # **P3는 인증 틀의 경계 사례다.** 규칙은 "모든 참조를 인증하라"가 아니라 "채점이
    # 참조에서 읽는 모든 양을 인증하라"이고, P3가 읽는 양은 고유값 하나뿐이다. 그래서
    # 인증 대상을 "mode shape"으로 적으면 어색하다 — 해당 없음이 정확하다.
    rows.append(_row("P3", "all", "not applicable (no mode shape is read)",
                     "transfer-matrix determinant, root-found at mpmath dps = 50",
                     "not applicable", 1.0, 0.0,
                     "P3 carries no neural training run and no per-seed classification: "
                     "its outputs are eigenvalue errors and a structural jump-capability "
                     "flag, with no MAC gate, no seed ensemble and no Wilson interval. The "
                     "scoring therefore reads only the eigenvalue, which is certified by "
                     "the high-precision root-finding of the reference itself. The largest "
                     "beta reported on P3 is 17.3, so the cancellation that damaged the P1 "
                     "shape at beta = 42.4 does not arise here"))

    manifest.write_csv(os.path.join(d, "reference_certification.csv"), rows)
    worst = max((r["deficit"] for r in rows
                 if r["quantity_certified"] != "mode shape"
                 or r["tolerance"] < 1.0), default=0.0)
    manifest.write_json(os.path.join(d, "manifest_reference_certification.json"),
                        manifest.build({
                            "driver": "run_reference_certification.main",
                            "n_modes_p1": n_modes_p1, "dps": dps, "n_q": n_q,
                            "p4_meshes": [p4_n_ref, p4_n_fine], "p4_beta": p4_beta,
                            "worst_deficit": worst,
                            "why": "the reference-uncertainty column certifies the "
                                   "eigenvalues; the MAC gate compares mode shapes, "
                                   "which reach the scorer by a different numerical "
                                   "path. A defect in the P1 shape reference at high "
                                   "mode order was found this way: catastrophic "
                                   "cancellation in the textbook cosh/sinh form put the "
                                   "mode-14 reference at MAC 0.887 against its own "
                                   "high-precision evaluation, below the 0.9 gate",
                            "added": "2026-08-27"}))
    print(f"[reference_certification] {len(rows)}행, 최악 결손 {worst:.2e}")
    return {"rows": rows, "outdir": d, "worst_deficit": worst}


if __name__ == "__main__":
    main()
