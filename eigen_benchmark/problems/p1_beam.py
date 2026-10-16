"""P1 — 각기둥 보(Euler–Bernoulli 캔틸레버), 모드 1–10. 가장 깨끗한 e_optimization 분리.

정규화 x̃ = x/L, 약형식
    a(u,v) = ∫₀¹ u″v″ dx̃,   b(u,v) = ∫₀¹ uv dx̃,   clamped(0)–free(1)
고유값 Λ = (βL)⁴, 주파수 f = √Λ·ω_b/(2π), ω_b = (h/√12)√(E/ρ)/L².

**근찾기 주의**: 특성방정식 1 + cos β cosh β = 0을 그대로 쓰면 β₁₀ ≈ 29.8에서
cos β ≈ −sech β ≈ −2.5e−13이라 파멸적 상쇄가 난다. cosh로 나눈 등가형
    cos β + sech β = 0
을 쓰면 두 항의 크기가 O(1)·O(1e−13)으로 분리되어 fp64에서도 상대 1e−14을 얻는다.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

# **무차원 문제다.** Λ = (βL)⁴는 재료·치수에 무관하므로 기하 딕트가 없다.
# 이전에는 논문 1 §4.2의 캔틸레버 vane 기하(L=30mm, h=1.2mm, E=193GPa, ρ=8000)를
# 담아 Hz를 인쇄했는데, 그 수치는 이 논문의 어떤 결론도 지탱하지 않으면서 시험문제를
# 특정 기계와 동일시하게 만들었다(논문 경계 정리, 2026-08-05).


def _char(b: float) -> float:
    """클램프-자유 특성함수(수치안정형) cos β + sech β."""
    return math.cos(b) + 1.0 / math.cosh(b)


def beta_roots(n_modes: int = 14) -> np.ndarray:
    """캔틸레버 βL 근 n_modes개(오름차순). 부호변화 스캔 + brentq."""
    if n_modes < 1:
        raise ValueError("n_modes는 1 이상이어야 합니다")
    b_max = (2 * n_modes + 1) * math.pi / 2 + 1.0
    grid = np.linspace(0.3, b_max, 200 * n_modes + 2000)
    vals = np.array([_char(b) for b in grid])
    roots = []
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            roots.append(float(grid[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            roots.append(float(brentq(_char, grid[i], grid[i + 1],
                                      xtol=1e-15, rtol=8.9e-16, maxiter=200)))
        if len(roots) >= n_modes:
            break
    if len(roots) < n_modes:
        raise RuntimeError(f"근 {n_modes}개를 못 찾음(찾은 수 {len(roots)})")
    return np.array(roots[:n_modes])





def _sigma(b: float) -> float:
    return (math.cosh(b) + math.cos(b)) / (math.sinh(b) + math.sin(b))


def _delta(b: float) -> float:
    """1 − σ를 **상쇄 없이** 계산한다.

    σ = (cosh b + cos b)/(sinh b + sin b)이므로
      1 − σ = (sinh b − cosh b + sin b − cos b)/(sinh b + sin b)
            = (−e^{−b} + sin b − cos b)/(sinh b + sin b)
    이고, 분자·분모 모두 상쇄가 없다. δ = O(e^{−b})다."""
    return (-math.exp(-b) + math.sin(b) - math.cos(b)) / (math.sinh(b) + math.sin(b))


def analytic_mode(x, n: int) -> np.ndarray:
    """n차(1-based) 해석 모드형 Φ_n(x̃) — 비정규화.

    **소박한 cosh/sinh 조합을 쓰면 안 된다.** cosh(βx) − σ·sinh(βx)는 β ≳ 40에서
    ~e^{β}/2 크기의 두 항이 O(1)로 상쇄돼 fp64 유효자리가 전부 사라진다(β = 42.4,
    즉 모드 14에서 x = 1의 값이 −2 대신 0으로 나왔다). 채점이 이 모드형으로 MAC을
    재므로, 정확한 해를 찾은 솔버가 spurious로 분류됐다.

    지수형으로 다시 쓰면 상쇄가 없다. δ = 1 − σ에 대해
      cosh(βx) − σ sinh(βx) = ½[δ·e^{βx} + (2 − δ)·e^{−βx}]
    이고 δ = O(e^{−β})이므로 δ·e^{βx}는 x = 1에서도 O(1)이다."""
    b = float(beta_roots(max(n, 1))[n - 1])
    s = _sigma(b)
    d = _delta(b)
    x = np.asarray(x, dtype=float)
    z = b * x
    hyp = 0.5 * (d * np.exp(z) + (2.0 - d) * np.exp(-z))
    return hyp - np.cos(z) + s * np.sin(z)


def analytic_mode_d2(x, n: int) -> np.ndarray:
    """Φ_n''(x̃) — H² 사전학습·Rayleigh 몫 검증용. 값과 같은 이유로 지수형을 쓴다."""
    b = float(beta_roots(max(n, 1))[n - 1])
    s = _sigma(b)
    d = _delta(b)
    x = np.asarray(x, dtype=float)
    z = b * x
    hyp = 0.5 * (d * np.exp(z) + (2.0 - d) * np.exp(-z))
    return b ** 2 * (hyp + np.cos(z) - s * np.sin(z))


def assemble(basis, x, w):
    """P1 약형식의 이산 (K, M). basis는 eigen_benchmark.bases.base.Basis.

    K_ij = Σ w ψ_i″ψ_j″,  M_ij = Σ w ψ_iψ_j — 대칭화까지 수행(부동소수 비대칭 제거)."""
    P = basis.eval(x)
    D2 = basis.d2(x)
    K = (D2 * w) @ D2.T
    M = (P * w) @ P.T
    return 0.5 * (K + K.T), 0.5 * (M + M.T)


def solve(basis, n_q: int = 200, n_modes=None, breaks=None):
    """일반화 대칭 고유문제 Kc = ΛMc를 풀어 Λ·고유벡터·Hz를 반환.

    구적은 **기저의 조각경계(basis.breaks)에 정렬**된다. 정렬하지 않으면 조각다항
    기저(B-spline·강화·FEM)를 매끄러운 것으로 오적분해 강성을 과소평가하고 Ritz 값이
    정확해보다 낮아진다(비적합). n_q는 **구간당** 점수다. breaks를 직접 주면
    기저의 경계를 무시하는데, 이는 구적분리 실험에서 오정렬 효과를 재는 데 쓴다.

    **fp64 Cholesky 실패는 예외로 터뜨리지 않고 기록한다** — 원시 단항식이 차수 ≈10을
    넘으면 질량행렬이 수치적으로 양정부호를 잃는다(파일럿 소견). 그 실패 자체가 §3.6의
    보고항목이므로 Lam=NaN + cholesky_ok=False로 남기고, 참값은 고정밀 경로
    (conditioning.highprec_eigenvalues)에서 얻는다.

    반환 dict: Lam(오름차순 = (βL)⁴, 무차원), vec, K, M, cholesky_ok."""
    from scipy.linalg import LinAlgError, eigh

    from ..quadrature import piecewise_gauss

    br = list(getattr(basis, "breaks", (0.0, 1.0))) if breaks is None else list(breaks)
    x, w = piecewise_gauss(br, n_q)
    K, M = assemble(basis, x, w)
    n = K.shape[0]
    try:
        Lam, V = eigh(K, M)
        ok = True
    except (LinAlgError, np.linalg.LinAlgError):
        Lam = np.full(n, np.nan)
        V = np.full((n, n), np.nan)
        ok = False
    if n_modes is not None:
        Lam, V = Lam[:n_modes], V[:, :n_modes]
    return {"Lam": Lam, "vec": V,
            "K": K, "M": M, "cholesky_ok": ok}
