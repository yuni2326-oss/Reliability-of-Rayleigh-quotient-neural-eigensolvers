import numpy as np
import pytest

from eigen_benchmark.problems import p4_lshape as p4


def test_all_eigenvalues_are_positive_for_clamped_domain():
    r = p4.solve(3, beta=1.0, n_modes=6)
    assert np.all(r["Lam"] > 0)
    assert np.all(np.diff(r["Lam"]) >= -1e-9 * r["Lam"][-1])


def test_refinement_lowers_eigenvalues_monotonically():
    # 적합 Galerkin → 위에서 수렴
    a = p4.solve(2, beta=1.0, n_modes=3)["Lam"]
    b = p4.solve(4, beta=1.0, n_modes=3)["Lam"]
    c = p4.solve(6, beta=1.0, n_modes=3)["Lam"]
    assert np.all(b <= a * (1 + 1e-12))
    assert np.all(c <= b * (1 + 1e-12))


def test_graded_mesh_beats_uniform_at_equal_dof():
    """재진입 코너 특이성 탓에 균일메시는 수렴이 느리다. 같은 자유도에서
    등급메시가 더 낮은(=더 정확한) 고유값을 줘야 한다."""
    uni = p4.solve(6, beta=1.0, n_modes=1)
    gra = p4.solve(6, beta=3.0, n_modes=1)
    assert gra["n_dof"] == uni["n_dof"]
    assert gra["Lam"][0] < uni["Lam"][0]


def test_richardson_recovers_a_known_rate():
    # λ_h = 10 + 3·h^2, h = 1, 1/2, 1/4 → rate 2, 외삽 10
    vals = [10 + 3 * h ** 2 for h in (1.0, 0.5, 0.25)]
    out = p4.richardson([1, 2, 4], vals)
    assert out["rate"] == pytest.approx(2.0, rel=1e-9)
    assert out["extrapolated"] == pytest.approx(10.0, rel=1e-9)


def test_richardson_flags_nonconvergent_sequences():
    out = p4.richardson([1, 2, 4], [10.0, 11.0, 9.0])
    assert not np.isfinite(out["rate"]) or out["uncertainty_rel"] == float("inf")


def test_convergence_study_produces_reference_with_uncertainty():
    st = p4.convergence_study([2, 4, 8], beta=3.0, n_modes=3)
    assert len(st["rows"]) == 3
    ref = st["reference"]
    assert len(ref) == 3
    for entry in ref:
        assert entry["extrapolated"] > 0
        assert entry["uncertainty_rel"] >= 0


def test_every_mode_is_symmetric_or_antisymmetric_under_the_reflection():
    """정의역이 y=x 반사에 대칭이므로 각 고유모드는 반사 연산자의 고유벡터,
    즉 **대칭(+1) 또는 반대칭(−1)** 이어야 한다. 조립·경계처리·등급메시 중
    어느 하나라도 대칭을 깨면 여기서 잡힌다.

    거울대칭(Z₂)은 1차원 기약표현만 가지므로 **축퇴를 강제하지 않는다** —
    이중근을 기대하면 안 된다(그건 C₄ᵥ 같은 비가환군에서 나온다)."""
    r = p4.solve(4, beta=2.5, n_modes=6)
    red = p4.reflection_dof_permutation(r["mesh"], r["free_dofs"])
    for i in range(r["vec"].shape[1]):
        phi = r["vec"][:, i]
        refl = phi[red]
        cos = abs(float(refl @ phi)) / (np.linalg.norm(refl) * np.linalg.norm(phi))
        assert cos == pytest.approx(1.0, abs=1e-6), f"모드 {i+1}이 반사 고유벡터가 아님"


def test_reflection_permutation_is_an_involution():
    mesh = p4.build_mesh(3, beta=2.0)
    perm = p4.reflection_dof_permutation(mesh)
    assert np.array_equal(perm[perm], np.arange(len(perm)))
