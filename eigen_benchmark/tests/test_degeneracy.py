import numpy as np
import pytest

from eigen_benchmark import degeneracy as dg
from eigen_benchmark import metrics as mt
from eigen_benchmark.reference import bessel_annulus as ba


def test_polar_grid_weights_sum_to_annulus_area():
    from eigen_benchmark.problems.p2_annulus import P2_GEOMETRY as G
    r, th, w = dg.polar_grid(60, 96)
    exact = np.pi * (G["b"] ** 2 - G["a"] ** 2)
    assert np.sum(w) == pytest.approx(exact, rel=1e-3)


def test_individual_mac_fails_for_rotated_degenerate_pair():
    """회전된 쌍은 물리적으로 같은 고유공간인데도 **개별 MAC은 1이 아니다** —
    모드별 비교로 축퇴를 판정하면 안 되는 이유."""
    grid = dg.polar_grid(40, 64)
    k = ba.annulus_k_roots(2, n_modes=1)[0]
    P = dg.degenerate_pair(k, 2, grid)
    Q = dg.rotated_pair(k, 2, grid, alpha=np.pi / 5)
    W = np.diag(grid[2])
    assert mt.mac(P[:, 0], Q[:, 0], W) < 0.75


def test_subspace_metrics_recognize_the_rotated_pair_as_identical():
    grid = dg.polar_grid(40, 64)
    k = ba.annulus_k_roots(2, n_modes=1)[0]
    P = dg.degenerate_pair(k, 2, grid)
    Q = dg.rotated_pair(k, 2, grid, alpha=np.pi / 5)
    W = np.diag(grid[2])
    assert mt.subspace_mac(P, Q, W) == pytest.approx(1.0, abs=1e-8)
    assert np.max(mt.principal_angles(P, Q, W)) < 1e-6


def test_different_nodal_diameter_families_are_orthogonal_subspaces():
    grid = dg.polar_grid(40, 64)
    W = np.diag(grid[2])
    P = dg.degenerate_pair(ba.annulus_k_roots(2, n_modes=1)[0], 2, grid)
    R = dg.degenerate_pair(ba.annulus_k_roots(3, n_modes=1)[0], 3, grid)
    assert mt.subspace_mac(P, R, W) < 1e-6
    assert np.min(mt.principal_angles(P, R, W)) > np.pi / 2 - 1e-3


def test_axisymmetric_family_has_no_degenerate_partner():
    # m=0은 축퇴가 없다(sin(0·θ)≡0) → 쌍을 만들려 하면 2열이 나오되 한 열이 0.
    grid = dg.polar_grid(30, 48)
    P = dg.degenerate_pair(ba.annulus_k_roots(0, n_modes=1)[0], 0, grid)
    assert np.allclose(P[:, 1], 0.0)
