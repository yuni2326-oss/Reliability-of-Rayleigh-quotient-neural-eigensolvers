"""해석 모드형의 수치 안정성 — 채점 참조이므로 여기가 틀리면 전부가 틀린다."""
import numpy as np
import pytest

from eigen_benchmark import metrics as mt
from eigen_benchmark.problems import p1_beam as p1

mp = pytest.importorskip("mpmath")


def _nodes(n_q=256):
    x, w = np.polynomial.legendre.leggauss(n_q)
    return 0.5 * x + 0.5, 0.5 * w


def _exact(x, n, dps=60):
    """고정밀 모드형 — 같은 식을 dps자리로 평가한 기준."""
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        b = mp.mpf(float(p1.beta_roots(n)[n - 1]))
        s = (mp.cosh(b) + mp.cos(b)) / (mp.sinh(b) + mp.sin(b))
        return np.array([float((mp.cosh(b * xx) - mp.cos(b * xx))
                               - s * (mp.sinh(b * xx) - mp.sin(b * xx))) for xx in x])
    finally:
        mp.mp.dps = old


def _exact_d2(x, n, dps=60):
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        b = mp.mpf(float(p1.beta_roots(n)[n - 1]))
        s = (mp.cosh(b) + mp.cos(b)) / (mp.sinh(b) + mp.sin(b))
        return np.array([float(b ** 2 * ((mp.cosh(b * xx) + mp.cos(b * xx))
                                         - s * (mp.sinh(b * xx) + mp.sin(b * xx))))
                         for xx in x])
    finally:
        mp.mp.dps = old


@pytest.mark.parametrize("n", [1, 5, 10, 12, 14, 16, 18, 20])
def test_the_reference_mode_shape_survives_its_own_scoring_gate(n):
    """참조가 자기 자신과 MAC 1을 못 이루면, 정확한 해도 spurious로 채점된다.

    소박한 cosh/sinh 조합은 β ≳ 40에서 상쇄소거로 무너진다 — n = 14에서 MAC이
    0.887까지 떨어져 사전등록 게이트 0.9 아래로 내려간 적이 있다."""
    x, w = _nodes()
    got = p1.analytic_mode(x, n)
    ref = _exact(x, n)
    m = mt.mac(got, ref, np.diag(w))
    assert m > 1 - 1e-9, f"mode {n}: MAC vs high precision = {m}"


@pytest.mark.parametrize("n", [12, 14, 16, 20])
def test_the_reference_curvature_is_stable_too(n):
    """φ″는 K = ∫(φ″)²를 만들므로 값보다 더 민감하다."""
    x, w = _nodes()
    got = p1.analytic_mode_d2(x, n)
    ref = _exact_d2(x, n)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 1e-9, f"mode {n}: max rel err {rel:.2e}"


@pytest.mark.parametrize("n", [10, 14, 18, 20])
def test_the_reference_reproduces_its_own_eigenvalue(n):
    x, w = _nodes()
    ph, d2 = p1.analytic_mode(x, n), p1.analytic_mode_d2(x, n)
    R = (w * d2 ** 2).sum() / (w * ph ** 2).sum()
    lam = p1.beta_roots(n)[n - 1] ** 4
    assert abs(R - lam) / lam < 1e-9


def test_the_clamped_end_conditions_hold_at_high_mode():
    for n in (14, 20):
        x = np.array([0.0])
        assert abs(float(p1.analytic_mode(x, n)[0])) < 1e-10
        h = 1e-6
        d = (p1.analytic_mode(np.array([h]), n)[0]
             - p1.analytic_mode(np.array([-h]), n)[0]) / (2 * h)
        assert abs(d) < 1e-4, (n, d)


def test_committed_classifications_are_robust_to_reference_level_perturbations():
    """참조를 안정형으로 고쳤을 때 커밋된 판정이 뒤집히지 않는지 — 여유로 보인다.

    수정이 모드 10 이하에서 MAC을 바꾸는 크기는 1e-9 수준이다(수정 전 참조도 그
    범위에서는 고정밀과 MAC 1.000000이었다). 커밋된 기록 중 게이트 0.9에 가장 가까운
    것의 여유가 그보다 훨씬 크면, 판정은 참조 수정에 영향받지 않는다."""
    import json
    import math

    path = "docs/_generated/data/paper2/p1_neural_records.jsonl"
    margins = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line).get("mac")
            if m is not None and math.isfinite(float(m)):
                margins.append(abs(float(m) - 0.9))
    assert margins, "커밋된 기록이 없다"
    assert min(margins) > 1e-5, (
        f"게이트 여유가 {min(margins):.2e}로 참조 수준 섭동과 같은 자릿수다 — "
        "참조를 바꾸면 판정이 뒤집힐 수 있으므로 재채점이 필요하다")


def test_the_scoring_reference_is_certified_over_the_modes_actually_scored():
    """채점에 쓰이는 모든 참조가 자기 정밀도 검사를 통과해야 한다.

    `classify_stage`의 기본 n_ref는 14이고 깊은 목표에서는 target+4까지 쓴다."""
    x, w = _nodes()
    W = np.diag(w)
    worst = min(mt.mac(p1.analytic_mode(x, n), _exact(x, n), W) for n in range(1, 21))
    assert worst > 1 - 1e-9, f"가장 나쁜 참조의 MAC = {worst}"
