"""§3.6 — 선언된 정규화 아래의 조건수 보고, 일반화 backward error, 고정밀 대조.

논문의 e_algebraic 성분을 측정하는 유일한 창구다. 원시 κ만 보고하면 "기저가 나쁘다"와
"좌표가 나쁘다"를 구분할 수 없으므로, 같은 (K,M)에 대해 네 가지 정규화를 모두 보고한다.
1) 원시  2) 질량노름 정규화  3) 열평형(column equilibration)  4) 변환연산자 M^(−1/2)KM^(−1/2)
그리고 fp64 고유값과 mpmath(dps=50) 고유값의 차를 함께 남긴다.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError, cholesky, eigh, solve_triangular


def _cond(A) -> float:
    try:
        return float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        return float("inf")


def _sym_scale(A, d):
    return A * np.outer(d, d)


def generalized_backward_error(K, M, lam: float, vec) -> float:
    """η = ‖Kx − λMx‖ / ((‖K‖ + |λ|‖M‖)‖x‖) — 일반화 고유쌍의 후방오차."""
    K = np.asarray(K, float); M = np.asarray(M, float)
    x = np.asarray(vec, float).ravel()
    nx = np.linalg.norm(x)
    if nx == 0.0:
        return float("inf")
    num = np.linalg.norm(K @ x - lam * (M @ x))
    den = (np.linalg.norm(K, 2) + abs(lam) * np.linalg.norm(M, 2)) * nx
    return float(num / den) if den > 0 else float("inf")


def highprec_eigenvalues(K, M, dps: int = 50, n_eig: int = 3, return_info: bool = False):
    """mpmath dps자리 정밀도로 Kc = ΛMc의 최저 n_eig개 Λ.

    두 경로가 있다.
    - `cholesky`: M = LLᵀ가 성립하면 eigsy(L⁻¹KL⁻ᵀ) — 대칭 정부호 정공법.
    - `general_pencil`: **fp64 조립이 이미 양정부호를 파괴한 경우**의 대체경로.
      질량행렬이 fp64로 반올림되며 λ_min보다 큰 오차를 얻으면(예: Hilbert형 기저 고차)
      그 행렬은 정확산술에서 양정부호가 아니므로 Cholesky가 옳게 거부한다. 이때는
      M⁻¹K의 일반 고유값을 고정밀로 구해 **fp64가 건네준 그 행렬쌍의 참 고유값**을 보고한다.
      → 고정밀 풀이는 조립에서 잃은 정보를 복원하지 못한다는 사실 자체가 §3.6의 소견이다.

    return_info=True면 (값 목록, {"path": ...}) 튜플을 반환."""
    from mpmath import cholesky as mp_cholesky
    from mpmath import eig, eigsy, inverse, lu_solve, matrix, mp

    old = mp.dps
    mp.dps = dps
    try:
        n = len(K)
        Km = matrix([[mp.mpf(float(K[i][j])) for j in range(n)] for i in range(n)])
        Mm = matrix([[mp.mpf(float(M[i][j])) for j in range(n)] for i in range(n)])
        try:
            L = mp_cholesky(Mm)
        except (ValueError, ZeroDivisionError):
            E, _ = eig(inverse(Mm) * Km)
            vals = sorted(float(mp.re(e)) for e in E)[:n_eig]
            return (vals, {"path": "general_pencil"}) if return_info else vals
        # A = L⁻¹ K L⁻ᵀ  (삼각계 두 번)
        Y = matrix(n, n)
        for j in range(n):
            col = lu_solve(L, matrix([Km[i, j] for i in range(n)]))
            for i in range(n):
                Y[i, j] = col[i]
        A = matrix(n, n)
        for j in range(n):
            row = lu_solve(L, matrix([Y[j, i] for i in range(n)]))
            for i in range(n):
                A[j, i] = row[i]
        for i in range(n):                      # 대칭화
            for j in range(i + 1, n):
                s = (A[i, j] + A[j, i]) / 2
                A[i, j] = A[j, i] = s
        ev, _ = eigsy(A)
        vals = sorted(float(ev[i]) for i in range(n))[:n_eig]
        return (vals, {"path": "cholesky"}) if return_info else vals
    finally:
        mp.dps = old


def condition_report(K, M, dps: int = 50, n_eig: int = 3) -> dict:
    """(K,M) 한 쌍의 조건수·후방오차·고정밀 대조를 한 딕트로."""
    K = np.asarray(K, float); M = np.asarray(M, float)
    out = {
        "n_dof": int(len(K)),
        "kappa_M_raw": _cond(M),
        "kappa_K_raw": _cond(K),
    }
    dm = np.sqrt(np.abs(np.diag(M)))
    dm[dm == 0.0] = 1.0
    out["kappa_M_massnorm"] = _cond(_sym_scale(M, 1.0 / dm))
    out["kappa_K_massnorm"] = _cond(_sym_scale(K, 1.0 / dm))
    dk = np.sqrt(np.abs(np.diag(K)))
    dk[dk == 0.0] = 1.0
    out["kappa_K_equilibrated"] = _cond(_sym_scale(K, 1.0 / dk))

    try:
        L = cholesky(M, lower=True)
        out["cholesky_ok"] = True
        Y = solve_triangular(L, K, lower=True)
        A = solve_triangular(L, Y.T, lower=True).T
        A = 0.5 * (A + A.T)
        out["kappa_A_transformed"] = _cond(A)
    except (LinAlgError, np.linalg.LinAlgError):
        out["cholesky_ok"] = False
        out["kappa_A_transformed"] = float("nan")

    try:
        Lam, V = eigh(K, M)
        out["Lam_fp64"] = [float(v) for v in Lam[:n_eig]]
        out["backward_error"] = [generalized_backward_error(K, M, Lam[i], V[:, i])
                                 for i in range(min(n_eig, len(Lam)))]
    except (LinAlgError, np.linalg.LinAlgError):
        out["Lam_fp64"] = [float("nan")] * n_eig
        out["backward_error"] = [float("nan")] * n_eig

    try:
        hp, info = highprec_eigenvalues(K, M, dps=dps, n_eig=n_eig, return_info=True)
        out["Lam_highprec"] = hp
        out["highprec_path"] = info["path"]
        out["Lam_absdiff_rel"] = [
            (abs(a - b) / abs(b) if np.isfinite(a) and b != 0 else float("nan"))
            for a, b in zip(out["Lam_fp64"], hp)]
    except Exception:                           # 고정밀 실패도 데이터로 남긴다
        out["Lam_highprec"] = [float("nan")] * n_eig
        out["highprec_path"] = "failed"
        out["Lam_absdiff_rel"] = [float("nan")] * n_eig
    return out
