import numpy as np
import pytest

from eigen_benchmark.problems import p4_lshape as p4


def test_grade_is_monotone_and_clusters_toward_one():
    t = np.linspace(0, 1, 11)
    g = p4.grade(t, beta=3.0)
    assert g[0] == pytest.approx(0.0)
    assert g[-1] == pytest.approx(1.0)
    assert np.all(np.diff(g) > 0)
    assert np.diff(g)[-1] < np.diff(g)[0]        # 끝쪽 간격이 더 촘촘
    assert np.allclose(p4.grade(t, beta=1.0), t)  # β=1이면 균일


def test_mesh_node_count_and_element_shape():
    mesh = p4.build_mesh(2, beta=1.0)
    assert mesh["elems"].shape[1] == 9
    assert mesh["elems"].shape[0] == 3 * 2 * 2
    assert mesh["nodes"].shape[1] == 2
    assert mesh["boundary"].shape[0] == mesh["nodes"].shape[0]


def test_blocks_share_nodes_on_internal_edges():
    # A|B는 x=1(y≤1)에서, A|C는 y=1(x≤1)에서 절점을 공유해야 한다(중복 없음).
    mesh = p4.build_mesh(3, beta=2.5)
    L = p4.P4_GEOMETRY["L_unit"]
    xy = mesh["nodes"] / L
    uniq = {(round(x, 10), round(y, 10)) for x, y in xy}
    assert len(uniq) == len(xy)                   # 중복 절점 없음
    shared = [(x, y) for x, y in xy if abs(x - 1) < 1e-12 and y < 1 - 1e-12]
    assert len(shared) > 0


def test_boundary_flags_include_reentrant_edges_and_exclude_internal():
    mesh = p4.build_mesh(3, beta=1.0)
    L = p4.P4_GEOMETRY["L_unit"]
    xy = mesh["nodes"] / L

    def flag(px, py):
        i = int(np.argmin((xy[:, 0] - px) ** 2 + (xy[:, 1] - py) ** 2))
        assert abs(xy[i, 0] - px) < 1e-9 and abs(xy[i, 1] - py) < 1e-9
        return bool(mesh["boundary"][i])

    assert flag(0.0, 0.5)          # 외곽
    assert flag(2.0, 0.5)          # 외곽
    assert flag(1.5, 1.0)          # 재진입 수평변
    assert flag(1.0, 1.5)          # 재진입 수직변
    assert not flag(1.0, 0.5)      # A|B 내부 공유변
    assert not flag(0.5, 1.0)      # A|C 내부 공유변


def test_free_stiffness_has_exactly_three_rigid_body_modes():
    """BC 없는 평면탄성 K는 강체운동 3개(병진2+회전1)에 대해 정확히 0이어야 한다.
    조립·B행렬·야코비안이 맞는지 보는 표준 패치시험."""
    mesh = p4.build_mesh(2, beta=1.0)
    K, M = p4.assemble(mesh)
    xy = mesh["nodes"]
    n = xy.shape[0]
    tx = np.zeros(2 * n); tx[0::2] = 1.0
    ty = np.zeros(2 * n); ty[1::2] = 1.0
    rot = np.zeros(2 * n); rot[0::2] = -xy[:, 1]; rot[1::2] = xy[:, 0]
    scale = abs(K).max()
    for v in (tx, ty, rot):
        assert np.linalg.norm(K @ v) < 1e-8 * scale * np.linalg.norm(v)


def test_mass_matrix_total_equals_domain_mass():
    mesh = p4.build_mesh(3, beta=2.0)
    K, M = p4.assemble(mesh)
    g = p4.P4_GEOMETRY
    area = 3.0 * g["L_unit"] ** 2
    n = mesh["nodes"].shape[0]
    tx = np.zeros(2 * n); tx[0::2] = 1.0
    assert float(tx @ (M @ tx)) == pytest.approx(g["rho"] * area, rel=1e-10)


def test_mesh_is_symmetric_about_the_diagonal():
    """정의역은 y=x 반사에 대해 대칭이고(A→A, B↔C) 등급규칙도 대칭이므로
    절점 집합이 반사에 닫혀 있어야 한다. 메시가 대칭성을 깨면 스펙트럼이 오염된다."""
    mesh = p4.build_mesh(3, beta=2.5)
    xy = mesh["nodes"]
    keys = {(round(x, 9), round(y, 9)) for x, y in xy}
    for x, y in xy:
        assert (round(y, 9), round(x, 9)) in keys


def test_clamped_removes_all_boundary_dofs():
    mesh = p4.build_mesh(3, beta=1.0)
    K, M = p4.assemble(mesh)
    Kb, Mb, free = p4.apply_clamped(K, M, mesh)
    n_b = int(mesh["boundary"].sum())
    assert Kb.shape[0] == 2 * (mesh["nodes"].shape[0] - n_b)
    assert len(free) == Kb.shape[0]
