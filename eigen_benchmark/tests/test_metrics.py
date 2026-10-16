import numpy as np
import pytest

from eigen_benchmark import metrics as mt


def test_mac_is_one_for_identical_and_zero_for_orthogonal():
    M = np.eye(3)
    a = np.array([1.0, 2.0, 3.0])
    assert mt.mac(a, 2.5 * a, M) == pytest.approx(1.0)          # 스케일 불변
    assert mt.mac(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), M) == pytest.approx(0.0)


def test_mac_uses_mass_weighting():
    M = np.diag([1.0, 100.0])
    a = np.array([1.0, 0.0]); b = np.array([0.0, 1.0])
    assert mt.mac(a, b, M) == pytest.approx(0.0)
    assert mt.mac(a, a + 1e-9 * b, M) == pytest.approx(1.0, abs=1e-6)


def test_normalized_residual_zero_for_exact_pair():
    K = np.diag([1.0, 4.0]); M = np.eye(2)
    assert mt.normalized_residual(K, M, 1.0, np.array([1.0, 0.0])) < 1e-15


def test_orthogonality_matrix_is_identity_for_m_orthonormal_set():
    M = np.eye(3)
    Phi = np.eye(3)
    assert np.allclose(mt.orthogonality_matrix(Phi, M), np.eye(3))


def test_projection_coeffs_recover_known_components():
    M = np.eye(3)
    Phi = np.eye(3)[:, :2]
    c = mt.projection_coeffs(np.array([3.0, -1.0, 7.0]), Phi, M)
    assert np.allclose(c, [3.0, -1.0])


def test_principal_angles_of_known_subspaces():
    M = np.eye(3)
    P1 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    P2 = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    ang = mt.principal_angles(P1, P2, M)
    assert np.allclose(np.sort(ang), [0.0, np.pi / 2], atol=1e-10)
    assert mt.subspace_mac(P1, P1, M) == pytest.approx(1.0)


def test_classify_correct():
    ref = np.eye(4)
    r = mt.classify(ref[:, 2], 100.0, ref, np.array([1.0, 10.0, 100.0, 1000.0]),
                    target=3, converged=True)
    assert r["outcome"] == "correct"
    assert r["matched_mode"] == 3


def test_classify_lower_mode_basin():
    ref = np.eye(4)
    r = mt.classify(ref[:, 0], 1.0, ref, np.array([1.0, 10.0, 100.0, 1000.0]),
                    target=3, converged=True)
    assert r["outcome"] == "lower_mode_basin"
    assert r["matched_mode"] == 1


def test_classify_spurious_when_no_reference_mode_matches():
    ref = np.eye(4)
    mixed = (ref[:, 0] + ref[:, 1] + ref[:, 2]) / np.sqrt(3)
    r = mt.classify(mixed, 55.0, ref, np.array([1.0, 10.0, 100.0, 1000.0]),
                    target=3, converged=True)
    assert r["outcome"] == "spurious"


def test_classify_spurious_when_shape_matches_but_eigenvalue_is_off():
    ref = np.eye(4)
    r = mt.classify(ref[:, 2], 140.0, ref, np.array([1.0, 10.0, 100.0, 1000.0]),
                    target=3, converged=True)
    assert r["outcome"] == "spurious"


def test_classify_higher_mode_escape_is_spurious():
    ref = np.eye(4)
    r = mt.classify(ref[:, 3], 1000.0, ref, np.array([1.0, 10.0, 100.0, 1000.0]),
                    target=3, converged=True)
    assert r["outcome"] == "spurious"
    assert r["matched_mode"] == 4


def test_classify_non_converged_takes_priority():
    ref = np.eye(4)
    r = mt.classify(ref[:, 2], 100.0, ref, np.array([1.0, 10.0, 100.0, 1000.0]),
                    target=3, converged=False)
    assert r["outcome"] == "non_converged"


def test_classify_handles_nan_as_non_converged():
    ref = np.eye(4)
    r = mt.classify(np.array([np.nan, 0.0, 0.0, 0.0]), float("nan"), ref,
                    np.array([1.0, 10.0, 100.0, 1000.0]), target=3, converged=True)
    assert r["outcome"] == "non_converged"


def test_wilson_interval_matches_textbook_value():
    lo, hi = mt.wilson(10, 20)
    assert lo == pytest.approx(0.2993, abs=5e-4)
    assert hi == pytest.approx(0.7007, abs=5e-4)


def test_wilson_at_zero_successes_is_bounded():
    lo, hi = mt.wilson(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < hi < 0.1


def test_confusion_counts_aggregates_outcomes():
    recs = [{"outcome": "correct"}, {"outcome": "correct"},
            {"outcome": "spurious"}, {"outcome": "non_converged"}]
    c = mt.confusion_counts(recs)
    assert c["correct"] == 2 and c["spurious"] == 1
    assert c["lower_mode_basin"] == 0 and c["non_converged"] == 1
    assert c["n"] == 4


def test_rel_errors_padded_pads_missing_modes_with_nan():
    # 4항 기저로 6모드를 물으면 뒤 두 모드는 표현 불가 → NaN(0이나 반복값 금지)
    got = mt.rel_errors_padded([1.0, 2.2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 6)
    assert got[0] == pytest.approx(0.0)
    assert got[1] == pytest.approx(0.1)
    assert all(np.isnan(v) for v in got[2:])


def test_rel_errors_padded_propagates_nan_input():
    got = mt.rel_errors_padded([np.nan, 2.0], [1.0, 2.0], 2)
    assert np.isnan(got[0])
    assert got[1] == pytest.approx(0.0)


def test_preregistered_thresholds_are_frozen_constants():
    # 사전등록값이므로 코드에서 바뀌면 테스트가 잡아야 한다.
    assert mt.MAC_MIN == 0.9
    assert mt.ELAM_MAX == 0.05


def test_non_converged_still_reports_the_measurement():
    """수렴 인증 실패와 오답은 다르다. outcome은 사전등록 규칙을 따르되 mac·e_lam은
    항상 측정해야 "해는 맞았으나 인증 못 함"(확률적 구적·부분공간 trace)이 데이터에 남는다."""
    ref = np.eye(4)
    r = mt.classify(ref[:, 2], 100.0, ref, np.array([1.0, 10.0, 100.0, 1000.0]),
                    target=3, converged=False)
    assert r["outcome"] == "non_converged"
    assert r["certified"] is False
    assert r["matched_mode"] == 3
    assert r["mac"] == pytest.approx(1.0)
    assert r["e_lam"] == pytest.approx(0.0)


def test_certified_flag_marks_rule_applicable_records():
    ref = np.eye(4)
    r = mt.classify(ref[:, 2], 100.0, ref, np.array([1.0, 10.0, 100.0, 1000.0]),
                    target=3, converged=True)
    assert r["certified"] is True and r["outcome"] == "correct"
