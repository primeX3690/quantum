"""Grover, Deutsch-Jozsa, QFT, VQE, Shor."""
import numpy as np
from algorithms.grover_search import run_grover, optimal_iterations
from algorithms.shor import (
    find_factors_via_shor, shor_period_finding, modexp_unitary, qft_matrix,
    check_memory,
)


def test_grover_finds_target():
    for n, target in [(2, 3), (3, 5), (4, 9)]:
        res = run_grover(n, target, optimal_iterations(n) or 1)
        best = max(res, key=res.get)
        assert int(best, 2) == target
        assert res[best] > 0.9


def test_deutsch_jozsa():
    from algorithms.deutsch_jozsa import demo_both_cases
    out = demo_both_cases(n_qubits=4)
    verdicts = {v["verdict"].lower() for v in out.values()}
    assert len(verdicts) == 2          # one constant, one balanced


def test_qft_matches_definition():
    from algorithms.qft import verify_qft_against_definition
    for n in (2, 3, 4):
        assert verify_qft_against_definition(n)["passed"]


def test_vqe_reaches_exact_ground_state():
    from algorithms.vqe_toy import run_vqe
    assert run_vqe()["absolute_error"] < 1e-3


def test_shor_fast_path_matches_dense_matrix():
    a, N, nc = 7, 15, 4
    nw = N.bit_length()
    W, Q = 2 ** nw, 2 ** nc
    st = np.zeros(2 ** (nc + nw), dtype=complex)
    for x in range(Q):
        st[x * W + 1] = 1 / np.sqrt(Q)
    st = modexp_unitary(a, N, nc, nw) @ st
    st = np.kron(qft_matrix(nc), np.eye(W)) @ st
    p = np.abs(st) ** 2
    dense = np.array([p[x * W:(x + 1) * W].sum() for x in range(Q)])
    assert np.allclose(dense, shor_period_finding(a, N, nc), atol=1e-12)


def test_shor_factors_several_numbers():
    for N, expect in [(15, (3, 5)), (21, (3, 7)), (35, (5, 7)),
                      (55, (5, 11)), (77, (7, 11)), (91, (7, 13))]:
        assert find_factors_via_shor(N, a=2, verbose=False) == expect


def test_shor_memory_guard():
    try:
        check_memory(2 ** 20 - 1, 40)
    except MemoryError:
        return
    raise AssertionError("expected MemoryError")


def test_grover_fast_matches_old_dense_operators():
    from algorithms.grover_search import (
        grover_statevector, diffusion_operator, optimal_iterations)
    for n, t in [(2, 3), (3, 5), (4, 9), (5, 17)]:
        N = 2 ** n
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        Hn = H
        for _ in range(n - 1):
            Hn = np.kron(Hn, H)
        psi = Hn @ np.eye(N)[0]
        O = np.eye(N)
        O[t, t] = -1
        D = diffusion_operator(n)
        it = optimal_iterations(n)
        for _ in range(it):
            psi = D @ (O @ psi)
        assert np.allclose(psi, grover_statevector(n, t, it), atol=1e-12)


def test_grover_scales_and_supports_multiple_targets():
    res = run_grover(18, 200000, optimal_iterations(18))
    assert int(next(iter(res)), 2) == 200000 and next(iter(res.values())) > 0.99
    res2 = run_grover(6, [3, 40], optimal_iterations(6, 2))
    assert abs(res2["000011"] - 0.5) < 0.02 and abs(res2["101000"] - 0.5) < 0.02
