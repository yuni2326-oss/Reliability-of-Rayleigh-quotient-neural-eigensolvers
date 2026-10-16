"""P1 저장 규약과 고정 접두부(MLST 재투고 계산, 사전규정 2026-10-14-mlst-prespecification.md).

P1의 가우스 절점은 고정이므로 저장 모드를 망 대신 절점 위 배열로 들고 있어도 같은 deflation이다. 그 위에서
(i) 같은 원시 쌍에서 모드 3만 푸는 고정 접두부, (ii) 저장 시 직교화(Rowan식, RF), (iii) 같은 부분공간의 기저만
바꾸는 기저 불변 시험이 선다. CPU 소형 설정으로 돈다(기존 P1 솔버 시험은 GPU 전용).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eigen_benchmark.neural import deflation as df      # noqa: E402
from eigen_benchmark.neural import sequential as sq     # noqa: E402

KW = dict(n_seeds=2, iters=12, device="cpu", snapshot_every=4)


def _pair(out, n=2):
    """순차 풀이의 원시 저장 쌍을 (φ, φ″) 배열 목록으로."""
    return [(torch.as_tensor(out["raw_shapes"][k]), torch.as_tensor(out["raw_d2"][k]))
            for k in range(n)]


def test_raw_curvature_is_returned_with_the_raw_shapes():
    out = sq.solve_sequential(2, df.ProjectionDiagonal(), **KW)
    assert np.asarray(out["raw_d2"]).shape == (2, 2, 256)
    np.testing.assert_array_equal(out["raw_shapes"][0], out["shapes"][0])   # 모드 1은 사영이 없다
    assert np.isfinite(out["raw_d2"]).all()


@pytest.mark.parametrize("defl", [df.ProjectionDiagonal(), df.ProjectionExact()])
def test_a_stage_started_from_the_stored_raw_pair_is_the_sequential_stage(defl):
    """고정 접두부 — 순차 풀이의 원시 쌍을 배열로 주고 모드 3만 풀면 순차 풀이의 모드 3과 같다
    (같은 초기화 시드 seed·1000 + 2, 같은 절점, 같은 deflation)."""
    seq = sq.solve_sequential(3, defl, **KW)
    st = sq.solve_sequential(3, defl, stored_arrays=_pair(seq), **KW)
    np.testing.assert_allclose(st["lam"][2], seq["lam"][2], rtol=1e-12, atol=0)
    np.testing.assert_allclose(st["shapes"][2], seq["shapes"][2], rtol=1e-10, atol=1e-13)
    assert st["n_stored"] == 2 and seq["n_stored"] == 0
    assert len(st["history"][0]) == 0 and len(st["history"][2]) > 0


def test_orthogonalized_storage_turns_the_diagonal_rule_into_the_exact_projection():
    """RF — 같은 원시 쌍을 저장 시 고전 GS로 직교화하면 대각 규칙이 정확 사영과 같은 연산자다. 모드 3의 결과가
    원시 쌍 + 정확 사영과 같아야 하고, 원시 쌍 + 대각 규칙과는 달라야 한다."""
    seq = sq.solve_sequential(2, df.ProjectionDiagonal(), **KW)
    raw = _pair(seq)
    rf = sq.orthogonalize_pair(raw, torch.as_tensor(seq["wq"]))
    a = sq.solve_sequential(3, df.ProjectionDiagonal(), stored_arrays=rf,
                            storage="orthogonalized", **KW)
    b = sq.solve_sequential(3, df.ProjectionExact(), stored_arrays=raw, **KW)
    d = sq.solve_sequential(3, df.ProjectionDiagonal(), stored_arrays=raw, **KW)
    np.testing.assert_allclose(a["lam"][2], b["lam"][2], rtol=1e-8)
    assert not np.allclose(d["lam"][2], b["lam"][2], rtol=1e-6)
    w = torch.as_tensor(seq["wq"])
    G = torch.einsum("ki,i,li->kl", torch.stack([p for p, _ in rf])[:, 0], w,
                     torch.stack([p for p, _ in rf])[:, 0])
    assert abs(float(G[0, 1])) < 1e-12 * float(torch.sqrt(G[0, 0] * G[1, 1]))


def test_orthogonalized_storage_keeps_storing_orthogonal_states_in_later_stages():
    seq = sq.solve_sequential(2, df.ProjectionDiagonal(), **KW)
    rf = sq.orthogonalize_pair(_pair(seq), torch.as_tensor(seq["wq"]))
    out = sq.solve_sequential(4, df.ProjectionDiagonal(), stored_arrays=rf,
                              storage="orthogonalized", **KW)
    w = out["wq"]
    S = np.stack([out["stored"][k][0] for k in range(4)])          # (k, n_seeds, n_q)
    G = np.einsum("asi,i,bsi->sab", S, w, S)
    C = G / np.sqrt(np.einsum("saa,sbb->sab", G, G))
    off = np.abs(C - np.eye(4)[None]).max()
    assert off < 1e-10


def test_stored_span_projector_identity_and_basis_dependence_of_the_diagonal_rule():
    """사전규정 단위 시험 — 고정 원시 Φ에서 Φ(ΦᵀWΦ)⁻¹ΦᵀW = QQᵀW이고, 같은 부분공간의 기저를
    v₂ = (q₂ + αq₁)/√(1+α²)로 바꾸면 정확 사영은 그대로, 대각 규칙은 α에 따라 바뀌며
    ‖P_G − A_D‖_W = |r|, r = α/√(1+α²)다."""
    torch.manual_seed(0)
    n = 256
    w = torch.rand(n, dtype=torch.float64) + 0.5
    Phi = torch.randn(2, n, dtype=torch.float64)
    Phi[1] = Phi[1] + 30.0 * Phi[0]                                # 강한 비직교
    phi = torch.randn(n, dtype=torch.float64)
    pe = phi - df.project_exact(phi, Phi, w) @ Phi
    Q, _, _ = df.w_orthonormalize(Phi, Phi, w)
    pq = phi - torch.einsum("ki,i,i->k", Q, w, phi) @ Q
    nrm = lambda v: float(torch.sqrt((w * v * v).sum()))
    assert nrm(pe - pq) <= 1e-8 * nrm(pe)
    for alpha in (0.0, 0.5, 2.0, 7.4):
        V = sq.mix_basis(Q, alpha)
        r = alpha / np.sqrt(1 + alpha ** 2)
        G = torch.einsum("ki,i,li->kl", V, w, V)
        assert abs(float(G[0, 1]) - r) < 1e-12
        assert sq.diagonal_operator_gap(V, w) == pytest.approx(abs(r), abs=1e-12)
        pv = phi - df.project_exact(phi, V, w) @ V
        assert nrm(pv - pe) <= 1e-10 * nrm(pe)                     # 정확 사영은 기저에 불변
        pd = phi - df.project_diagonal(phi, V, w) @ V
        if alpha == 0.0:
            assert nrm(pd - pe) <= 1e-10 * nrm(pe)
        else:
            assert nrm(pd - pe) > 1e-3 * nrm(pe)                   # 대각 규칙은 기저에 의존


def test_array_storage_refuses_what_it_cannot_represent():
    seq = sq.solve_sequential(2, df.ProjectionDiagonal(), **KW)
    raw = _pair(seq)
    with pytest.raises(ValueError):                 # MC 절점은 매 반복 바뀌어 배열이 그 점의 값이 아니다
        sq.solve_sequential(3, df.ProjectionDiagonal(), stored_arrays=raw, nodes="mc", **KW)
    with pytest.raises(ValueError):                 # 검사 격자는 망을 다시 재야 한다
        sq.solve_sequential(3, df.ProjectionDiagonal(), stored_arrays=raw, check_n_q=512, **KW)
    with pytest.raises(ValueError):                 # 풀 단계가 남지 않는다
        sq.solve_sequential(2, df.ProjectionDiagonal(), stored_arrays=raw, **KW)
    with pytest.raises(ValueError):
        sq.solve_sequential(2, df.ProjectionDiagonal(), storage="projected", **KW)
    with pytest.raises(ValueError):
        sq.solve_sequential(2, df.ProjectionDiagonal(), storage="orthogonalized", nodes="mc", **KW)


def test_default_storage_is_the_committed_raw_network_path():
    a = sq.solve_sequential(2, df.ProjectionDiagonal(), **KW)
    b = sq.solve_sequential(2, df.ProjectionDiagonal(), storage="raw", **KW)
    np.testing.assert_array_equal(a["lam"], b["lam"])
    np.testing.assert_array_equal(a["shapes"], b["shapes"])
    assert a["storage"] == "raw" and a["n_stored"] == 0
