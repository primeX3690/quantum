"""MPS engine: exactness vs dense, scaling, sampling, truncation."""
import random
import numpy as np
from core_engine.mps_engine import MPSState
from core_engine.statevector import QuantumState
from core_engine.tensor_network import statevector_to_mps, mps_to_statevector


def _random_pair(n, depth, seed):
    random.seed(seed)
    m, q = MPSState(n), QuantumState(n)
    for _ in range(depth):
        k = random.choice(["h", "t", "rx", "ry", "cnot", "cz", "cphase", "swap"])
        a, b = random.sample(range(n), 2)
        th = random.uniform(0, 6.28)
        if k == "h": m.h(a); q.apply_h(a)
        elif k == "t": m.t(a); q.apply_t(a)
        elif k == "rx": m.rx(a, th); q.apply_rx(a, th)
        elif k == "ry": m.ry(a, th); q.apply_ry(a, th)
        elif k == "cnot": m.cnot(a, b); q.apply_cnot_general(a, b)
        elif k == "cz": m.cz(a, b); q.apply_cz_general(a, b)
        elif k == "cphase": m.cphase(a, b, th); q.apply_cphase_general(a, b, th)
        elif k == "swap": m.swap(a, b); q.apply_swap_general(a, b)
    return m, q


def test_matches_dense_on_random_circuits():
    for seed in range(6):
        m, q = _random_pair(6, 50, seed)
        assert abs(abs(np.vdot(q.state, m.to_statevector())) - 1) < 1e-9
        assert abs(m.norm() - 1) < 1e-9


def test_expectation_and_amplitude():
    m, q = _random_pair(5, 40, 11)
    Z = np.diag([1, -1]).astype(complex)
    probs = np.abs(q.state.reshape([2] * 5)) ** 2
    pz = probs.sum(axis=(0, 1, 3, 4))
    assert abs(m.expectation_1q(Z, 2).real - (pz[0] - pz[1])) < 1e-9
    assert abs(m.amplitude("01011") - q.state[int("01011", 2)]) < 1e-9


def test_ghz_100_qubits_tiny_memory():
    m = MPSState(100)
    m.h(0)
    for i in range(99):
        m.cnot(i, i + 1)
    assert m.max_bond() == 2
    assert m.memory_bytes() < 100_000            # dense would be ~2e22 GB
    counts = m.sample(100, seed=0)
    assert set(counts) <= {"0" * 100, "1" * 100}
    assert abs(m.entanglement_entropy(50) - 1.0) < 1e-9


def test_dense_refused_when_too_large():
    try:
        MPSState(40).to_statevector()
    except MemoryError:
        return
    raise AssertionError("expected MemoryError")


def test_sampling_distribution():
    m = MPSState(3)
    m.h(0); m.cnot(0, 1); m.ry(2, 1.0)
    q = QuantumState(3)
    q.apply_h(0); q.apply_cnot(0, 1); q.apply_ry(2, 1.0)
    shots = 6000
    counts = m.sample(shots, seed=5)
    for k, p in q.measure_probabilities_dict().items():
        assert abs(counts.get(k, 0) / shots - p) < 0.03


def test_bond_truncation_reports_error():
    m_full, _ = _random_pair(8, 120, 2)
    random.seed(2)
    m_cut = MPSState(8, max_bond_dim=2)
    for _ in range(120):
        a, b = random.sample(range(8), 2)
        m_cut.h(a); m_cut.cnot(a, b); m_cut.ry(b, 0.7)
    assert m_cut.max_bond() <= 2
    assert m_cut.truncation_error > 0
    assert abs(m_cut.norm() - 1) < 1e-9


def test_legacy_statevector_to_mps_roundtrip():
    q = QuantumState(4)
    q.apply_h(0); q.apply_cnot(0, 1); q.apply_cnot(1, 2)
    back = mps_to_statevector(statevector_to_mps(q.state, 4), 4)
    assert np.allclose(back, q.state)


def test_estimated_fidelity_tracks_true_fidelity():
    """With canonical gauge, the product-of-kept-weights estimate must agree
    with the real overlap against the untruncated state."""
    rs = np.random.default_rng(7)
    exact, cut = MPSState(10), MPSState(10, max_bond_dim=4)
    for layer in range(6):
        for i in range(layer % 2, 9, 2):
            th = rs.uniform(0, 3)
            for m in (exact, cut):
                m.ry(i, th); m.ry(i + 1, th / 2); m.cnot(i, i + 1); m.rz(i + 1, th)
    true_f = abs(np.vdot(exact.to_statevector(), cut.to_statevector())) ** 2
    assert cut.max_bond() <= 4
    assert abs(cut.estimated_fidelity() - true_f) < 0.03
    assert abs(cut.norm() - 1) < 1e-9
