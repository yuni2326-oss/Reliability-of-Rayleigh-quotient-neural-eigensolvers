"""P4 — L형 평면탄성 고유치문제. 재진입 코너에서 mesh-free 주장이 실제로 시험되는 곳.

정의역(단위셀 배수, L_unit = 30 mm): [0,2]×[0,1] ∪ [0,1]×[1,2], 재진입 코너 (1,1).
∂Ω 전체 클램프. SUS304 평면응력.

**등급메시가 핵심이다.** 재진입 코너의 응력특이성(u ~ r^λ, λ<1) 탓에 균일메시에서는
고유값이 O(h^{2λ})로 느리게 수렴해 0.1 % 기준 불확실도를 얻기 어렵다. 코너 쪽으로
기하등급을 주면 최적 차수를 회복한다. β=1이면 균일메시가 되므로 두 경우를 같은
코드로 비교할 수 있다(그 비교 자체가 §5.4의 데이터).

블록 규약: A=[0,1]², B=[1,2]×[0,1], C=[0,1]×[1,2].
  g(t) = 1 − (1−t)^β (t=1 쪽 집중), h(t) = t^β (t=0 쪽 집중)
  A: (g(s), g(t))   B: (1+h(s), g(t))   C: (g(s), 1+h(t))
이 규칙이면 A|B 공유변의 y분포와 A|C 공유변의 x분포가 자동 일치해 절점이 맞물린다.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..quadrature import gauss_legendre

# **무차원 문제다.** Ω² ≡ ω²ρL²/E는 형상(L형 종횡비)과 ν에만 의존하므로 L=E=ρ=1로 잡고
# solve가 그 Ω²를 그대로 낸다. 이전에는 L_unit=30mm·SUS304 물성으로 Hz를 인쇄했는데,
# 그 수치는 결론을 지탱하지 않으면서 시험문제를 논문 1의 기계와 같은 치수로 만들었다.
# (논문 경계 정리, 2026-08-05)
P4_GEOMETRY = {"L_unit": 1.0, "E": 1.0, "nu": 0.29, "rho": 1.0}
_TOL = 1e-9


def grade(t, beta: float = 3.0):
    """g(t) = 1 − (1−t)^β — t=1 쪽으로 집중. β=1이면 항등(균일)."""
    t = np.asarray(t, float)
    return 1.0 - (1.0 - t) ** beta


def _with_midpoints(v):
    """꼭짓점 좌표 (n+1,) → Q2 절점 좌표 (2n+1,). 중간절점은 산술 중점."""
    v = np.asarray(v, float)
    out = np.empty(2 * len(v) - 1)
    out[0::2] = v
    out[1::2] = 0.5 * (v[:-1] + v[1:])
    return out


def _shape_1d(s):
    """1D 2차 Lagrange(절점 −1,0,1)의 값과 도함수. 반환 (3,), (3,)."""
    return (np.array([s * (s - 1) / 2, 1 - s ** 2, s * (s + 1) / 2]),
            np.array([s - 0.5, -2 * s, s + 0.5]))


def _q2_shape(xi, eta):
    """Q2 형상함수 N(9,)와 국소도함수 dN(9,2). 절점 순서 3*j+i."""
    Lx, dLx = _shape_1d(xi)
    Ly, dLy = _shape_1d(eta)
    N = np.empty(9)
    dN = np.empty((9, 2))
    for j in range(3):
        for i in range(3):
            a = 3 * j + i
            N[a] = Lx[i] * Ly[j]
            dN[a, 0] = dLx[i] * Ly[j]
            dN[a, 1] = Lx[i] * dLy[j]
    return N, dN


def build_mesh(n: int, beta: float = 3.0, geometry=None) -> dict:
    """n×n Q2 요소를 블록마다. 반환 {nodes(n_node,2), elems(n_elem,9), boundary(bool)}."""
    if n < 1:
        raise ValueError("n은 1 이상이어야 합니다")
    g = P4_GEOMETRY if geometry is None else geometry
    L = g["L_unit"]
    # **꼭짓점만 등급으로 배치하고 중간절점은 기하학적 중점에 둔다.**
    # 등급 파라미터의 중점에 두면 강한 등급(β≥2)에서 중간절점이 변의 중앙 1/2 구간을
    # 벗어나 등매개변수 사상이 퇴화하고 야코비안이 비양수가 된다.
    tv = np.linspace(0.0, 1.0, n + 1)              # 요소 꼭짓점 파라미터
    gg = _with_midpoints(grade(tv, beta))
    hh = _with_midpoints(tv ** beta)

    blocks = [(gg, gg), (1.0 + hh, gg), (gg, 1.0 + hh)]      # A, B, C
    index, nodes, elems = {}, [], []

    def nid(x, y):
        key = (round(float(x), 10), round(float(y), 10))
        if key not in index:
            index[key] = len(nodes)
            nodes.append(key)
        return index[key]

    for xs, ys in blocks:
        grid = [[nid(xs[i], ys[j]) for i in range(2 * n + 1)]
                for j in range(2 * n + 1)]
        for je in range(n):
            for ie in range(n):
                elems.append([grid[2 * je + jj][2 * ie + ii]
                              for jj in range(3) for ii in range(3)])

    xy = np.array(nodes, float)
    x, y = xy[:, 0], xy[:, 1]
    boundary = ((np.abs(x) < _TOL) | (np.abs(y) < _TOL)
                | (np.abs(x - 2) < _TOL) | (np.abs(y - 2) < _TOL)
                | ((np.abs(y - 1) < _TOL) & (x >= 1 - _TOL))
                | ((np.abs(x - 1) < _TOL) & (y >= 1 - _TOL)))
    return {"nodes": xy * L, "elems": np.array(elems, int),
            "boundary": boundary, "n": int(n), "beta": float(beta)}


def _plane_stress_D(E, nu):
    return E / (1 - nu ** 2) * np.array([[1.0, nu, 0.0],
                                         [nu, 1.0, 0.0],
                                         [0.0, 0.0, (1 - nu) / 2]])


def assemble(mesh, geometry=None, n_q: int = 3):
    """BC 미부과 전체 (K, M) — CSR 희소행렬. 두께 1(고유값에서 상쇄)."""
    g = P4_GEOMETRY if geometry is None else geometry
    D = _plane_stress_D(g["E"], g["nu"])
    rho = g["rho"]
    xy, elems = mesh["nodes"], mesh["elems"]
    n_dof = 2 * xy.shape[0]
    gp, gw = gauss_legendre(n_q, -1.0, 1.0)

    rows, cols, kv, mv = [], [], [], []
    for el in elems:
        ce = xy[el]                                   # (9,2)
        Ke = np.zeros((18, 18))
        Me = np.zeros((18, 18))
        for a, xi in enumerate(gp):
            for b, eta in enumerate(gp):
                N, dN = _q2_shape(xi, eta)
                J = dN.T @ ce                          # (2,2)
                detJ = np.linalg.det(J)
                if detJ <= 0:
                    raise RuntimeError("야코비안이 비양수 — 메시가 뒤집혔다")
                dNx = np.linalg.solve(J, dN.T).T       # (9,2) 물리 도함수
                B = np.zeros((3, 18))
                B[0, 0::2] = dNx[:, 0]
                B[1, 1::2] = dNx[:, 1]
                B[2, 0::2] = dNx[:, 1]
                B[2, 1::2] = dNx[:, 0]
                Nm = np.zeros((2, 18))
                Nm[0, 0::2] = N
                Nm[1, 1::2] = N
                wq = gw[a] * gw[b] * detJ
                Ke += wq * (B.T @ D @ B)
                Me += wq * rho * (Nm.T @ Nm)
        dofs = np.empty(18, int)
        dofs[0::2] = 2 * el
        dofs[1::2] = 2 * el + 1
        R, C = np.meshgrid(dofs, dofs, indexing="ij")
        rows.append(R.ravel()); cols.append(C.ravel())
        kv.append(Ke.ravel()); mv.append(Me.ravel())

    rows = np.concatenate(rows); cols = np.concatenate(cols)
    K = sp.coo_matrix((np.concatenate(kv), (rows, cols)),
                      shape=(n_dof, n_dof)).tocsr()
    M = sp.coo_matrix((np.concatenate(mv), (rows, cols)),
                      shape=(n_dof, n_dof)).tocsr()
    return K, M


def apply_clamped(K, M, mesh):
    """∂Ω 전체 클램프 → 자유 DOF만 남긴 (K, M, free_dofs)."""
    b = mesh["boundary"]
    free = np.where(np.repeat(~b, 2))[0]
    return K[free][:, free], M[free][:, free], free


def reflection_dof_permutation(mesh, free=None):
    """y=x 반사 (x,y)→(y,x)에 대응하는 DOF 치환.

    벡터장은 u′(Rx) = R u(x)로 변환하므로 성분 u↔v가 뒤바뀐다.
    free를 주면 축소된(클램프 제거) 벡터 인덱스로 돌려준다."""
    xy = mesh["nodes"]
    index = {(round(x, 9), round(y, 9)): i for i, (x, y) in enumerate(xy)}
    perm = np.empty(2 * len(xy), int)
    for i, (x, y) in enumerate(xy):
        j = index[(round(y, 9), round(x, 9))]
        perm[2 * i] = 2 * j + 1                 # u_i ← v_{R(i)}
        perm[2 * i + 1] = 2 * j                 # v_i ← u_{R(i)}
    if free is None:
        return perm
    pos = -np.ones(2 * len(xy), int)
    pos[free] = np.arange(len(free))
    red = pos[perm[free]]
    if np.any(red < 0):
        raise RuntimeError("반사가 자유 DOF 집합을 보존하지 않는다")
    return red


def solve(n: int, beta: float = 3.0, n_modes: int = 6, geometry=None) -> dict:
    """클램프 L형의 최저 n_modes개 ω²와 주파수. 희소 shift-invert."""
    import math

    from scipy.sparse.linalg import eigsh

    mesh = build_mesh(n, beta, geometry)
    K, M = assemble(mesh, geometry)
    Kb, Mb, free = apply_clamped(K, M, mesh)
    k = min(n_modes, Kb.shape[0] - 2)
    vals, vecs = eigsh(Kb.tocsc(), k=k, M=Mb.tocsc(), sigma=0.0, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    return {"Lam": vals, "vec": vecs,
            "n_dof": int(Kb.shape[0]), "n": int(n), "beta": float(beta),
            "free_dofs": free, "mesh": mesh}


def richardson(levels, values) -> dict:
    """세 수준(비 2배)의 Richardson 외삽. 수렴하지 않으면 rate=nan·불확실도=inf."""
    import math

    v = [float(x) for x in values]
    if len(v) < 3:
        raise ValueError("세 수준이 필요합니다")
    d1, d2 = v[-3] - v[-2], v[-2] - v[-1]
    if d2 == 0.0 or d1 / d2 <= 1.0:
        return {"rate": float("nan"), "extrapolated": v[-1],
                "uncertainty_rel": float("inf")}
    p = math.log(d1 / d2) / math.log(2.0)
    # λ ≈ λ₃ + (λ₃−λ₂)/(2^p−1) 이고 d2 = λ₂−λ₃ 이므로 **빼야** 한다.
    ext = v[-1] - d2 / (2.0 ** p - 1.0)
    unc = abs(v[-1] - ext) / abs(ext) if ext != 0 else float("inf")
    return {"rate": p, "extrapolated": ext, "uncertainty_rel": unc}


def convergence_study(ns, beta: float = 3.0, n_modes: int = 6,
                      geometry=None) -> dict:
    """격자 수준별 고유값 + 모드별 Richardson 기준값·불확실도."""
    import time
    rows = []
    for n in ns:
        # **격자마다 따로 잰다.** 이전에는 드라이버가 sweep 전체 시간을 모든 행에 복사해
        # n=4와 n=16이 같은 초로 기록됐다 — 비용 비교를 그 열에 세우면 무너진다.
        t0 = time.perf_counter()
        r = solve(n, beta, n_modes, geometry)
        rows.append({"n": int(n), "beta": float(beta), "n_dof": r["n_dof"],
                     "seconds": time.perf_counter() - t0,
                     "Lam": [float(v) for v in r["Lam"]]})
    reference = []
    for k in range(n_modes):
        vals = [row["Lam"][k] for row in rows]
        out = richardson([row["n"] for row in rows], vals)
        out["mode"] = k + 1
        reference.append(out)
    return {"rows": rows, "reference": reference}


def _q2_tensor_1d(s):
    """1차원 3절점 Lagrange 값 (좌, 중, 우). `eval_mode_at`의 텐서곱 평가에 쓴다.

    이름을 `_q2_shape`로 두면 `assemble`이 쓰는 (N, dN) 반환 함수를 덮어써
    조립이 조용히 깨진다 — 실제로 한 번 그렇게 만들었다."""
    return np.stack([0.5 * s * (s - 1), 1 - s ** 2, 0.5 * s * (s + 1)])


def eval_mode_at(mesh, vec_full, pts) -> np.ndarray:
    """FEM 모드를 임의 점에서 평가 — 신경 arm과 MAC을 재려면 같은 격자에서 봐야 한다.

    등급이 1차원 좌표열의 **텐서곱**이라 모든 요소가 축정렬 사각형이고, 따라서 등매개변수
    사상의 역변환이 축별 선형식으로 정확히 닫힌다. 요소 밖 점은 NaN을 낸다(채우지 않는다).

    `vec_full`은 전체 DOF 벡터(클램프 포함, 2·n_node)이고 반환은 (n_pts, 2)다.
    """
    xy, elems = mesh["nodes"], mesh["elems"]
    pts = np.asarray(pts, float)
    out = np.full((len(pts), 2), np.nan)
    # 요소별 경계상자 — 축정렬이므로 이것으로 정확히 판정된다
    ex = xy[elems][:, :, 0]
    ey = xy[elems][:, :, 1]
    x0, x1 = ex.min(1), ex.max(1)
    y0, y1 = ey.min(1), ey.max(1)
    for i, (px, py) in enumerate(pts):
        hit = np.where((px >= x0 - _TOL) & (px <= x1 + _TOL)
                       & (py >= y0 - _TOL) & (py <= y1 + _TOL))[0]
        if len(hit) == 0:
            continue
        e = int(hit[0])
        xi = 2.0 * (px - x0[e]) / max(x1[e] - x0[e], _TOL) - 1.0
        eta = 2.0 * (py - y0[e]) / max(y1[e] - y0[e], _TOL) - 1.0
        lx = _q2_tensor_1d(np.clip(xi, -1, 1))
        ly = _q2_tensor_1d(np.clip(eta, -1, 1))
        nd = elems[e]
        # elems의 9절점을 (3x3) 격자로 되돌린다: x 오름차순, y 오름차순
        order = np.lexsort((xy[nd][:, 0], xy[nd][:, 1])).reshape(3, 3)
        acc = np.zeros(2)
        for a in range(3):            # y 방향
            for b in range(3):        # x 방향
                n = nd[order[a, b]]
                acc += ly[a] * lx[b] * vec_full[2 * n:2 * n + 2]
        out[i] = acc
    return out


def eval_mode_grad_at(mesh, vec_full, pts) -> np.ndarray:
    """FEM 모드의 기울기를 임의 점에서 평가 — (n_pts, 2, 2), [성분 u·v, ∂/∂x·∂/∂y] (P4-ADF).

    `eval_mode_at`과 같은 축정렬 Q2 텐서곱의 도함수다. 요소 경계에서는 그 점을 담은 첫 요소의 값을 쓴다
    (C⁰ 장이라 경계에서 기울기가 불연속일 수 있다). 요소 밖 점은 NaN이다."""
    xy, elems = mesh["nodes"], mesh["elems"]
    pts = np.asarray(pts, float)
    out = np.full((len(pts), 2, 2), np.nan)
    ex = xy[elems][:, :, 0]
    ey = xy[elems][:, :, 1]
    x0, x1 = ex.min(1), ex.max(1)
    y0, y1 = ey.min(1), ey.max(1)
    for i, (px, py) in enumerate(pts):
        hit = np.where((px >= x0 - _TOL) & (px <= x1 + _TOL)
                       & (py >= y0 - _TOL) & (py <= y1 + _TOL))[0]
        if len(hit) == 0:
            continue
        e = int(hit[0])
        hx, hy = max(x1[e] - x0[e], _TOL), max(y1[e] - y0[e], _TOL)
        xi = np.clip(2.0 * (px - x0[e]) / hx - 1.0, -1, 1)
        eta = np.clip(2.0 * (py - y0[e]) / hy - 1.0, -1, 1)
        lx, ly = _q2_tensor_1d(xi), _q2_tensor_1d(eta)
        dlx = np.array([xi - 0.5, -2.0 * xi, xi + 0.5]) * (2.0 / hx)
        dly = np.array([eta - 0.5, -2.0 * eta, eta + 0.5]) * (2.0 / hy)
        nd = elems[e]
        order = np.lexsort((xy[nd][:, 0], xy[nd][:, 1])).reshape(3, 3)
        acc = np.zeros((2, 2))
        for a in range(3):            # y 방향
            for b in range(3):        # x 방향
                n = nd[order[a, b]]
                v = vec_full[2 * n:2 * n + 2]
                acc[:, 0] += ly[a] * dlx[b] * v
                acc[:, 1] += dly[a] * lx[b] * v
        out[i] = acc
    return out


def mode_at(res: dict, mode: int, pts) -> np.ndarray:
    """`solve`의 결과에서 mode번째(1-기반) 모드를 pts에서 평가. (n_pts, 2)."""
    mesh, free = res["mesh"], res["free_dofs"]
    full = np.zeros(2 * len(mesh["nodes"]))
    full[free] = res["vec"][:, mode - 1]
    return eval_mode_at(mesh, full, pts)
