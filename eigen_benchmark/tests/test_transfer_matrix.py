import numpy as np
import pytest

from eigen_benchmark.problems import p1_beam as p1
from eigen_benchmark.reference import transfer_matrix as tm


def test_zero_spring_compliance_recovers_uniform_cantilever():
    # κ→0 (강성 무한) 극한에서 균일보 βL을 회복해야 한다 = 자기검증.
    b = tm.spring_beam_betas(xc=0.2, kappa=0.0, n_modes=3, dps=30)
    assert np.allclose(b, p1.beta_roots(3), rtol=1e-9)


def test_spring_lowers_frequencies_monotonically():
    ref = p1.beta_roots(3)
    b_small = tm.spring_beam_betas(0.2, tm.kappa_from_k_hat(1000.0), 3, dps=30)
    b_large = tm.spring_beam_betas(0.2, tm.kappa_from_k_hat(1.0), 3, dps=30)
    assert np.all(b_small < ref)
    assert np.all(b_large < b_small)


def test_kappa_is_the_reciprocal_of_k_hat():
    assert tm.kappa_from_k_hat(10.0) == pytest.approx(0.1, rel=1e-15)
    assert tm.kappa_from_k_hat(1000.0) < tm.kappa_from_k_hat(1.0)
    with pytest.raises(ValueError):
        tm.kappa_from_k_hat(-1.0)


def test_module_does_not_depend_on_paper1_code():
    """논문2 코드는 논문1 패키지에 의존하지 않는다(폴더 규약). k̂를 직접 선언하므로
    파괴역학 유연도 함수를 빌려올 이유가 없다."""
    import inspect
    src = inspect.getsource(tm)
    assert "impeller_pinn" not in src


def test_fp64_and_highprec_agree_for_low_modes():
    d = tm.fp64_vs_highprec_betas(0.2, tm.kappa_from_k_hat(10.0), n_modes=4)
    assert len(d["fp64"]) == 4
    assert max(d["rel_diff"]) < 1e-9


def test_char_det_changes_sign_around_a_root():
    kap = tm.kappa_from_k_hat(10.0)
    b = tm.spring_beam_betas(0.2, kap, 1, dps=30)[0]
    assert tm.char_det(b - 0.05, 0.2, kap) * tm.char_det(b + 0.05, 0.2, kap) < 0.0


def test_six_modes_are_found_and_increasing():
    kap = tm.kappa_from_k_hat(10.0)
    b = tm.spring_beam_betas(0.2, kap, 6, dps=50)
    assert len(b) == 6
    assert np.all(np.diff(b) > 0)
    assert np.all(b < p1.beta_roots(6))
