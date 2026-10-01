"""Core statevector engine, gates, measurement."""
import numpy as np
from core_engine.statevector import QuantumState
from core_engine.custom_gates import (
    gate_H, gate_X, gate_Y, gate_Z, gate_S, gate_T, gate_RX, gate_RY, gate_RZ,
    gate_CNOT, gate_CZ, gate_SWAP, gate_CPHASE,
)


def _is_unitary(U):
    return np.allclose(U @ U.conj().T, np.eye(U.shape[0]))


def test_all_gates_unitary():
    for U in [gate_H(), gate_X(), gate_Y(), gate_Z(), gate_S(), gate_T(),
              gate_RX(0.7), gate_RY(1.1), gate_RZ(2.3), gate_CNOT(), gate_CZ(),
              gate_SWAP(), gate_CPHASE(0.9)]:
        assert _is_unitary(U)


def test_bell_state():
    qs = QuantumState(2)
    qs.apply_h(0)
    qs.apply_cnot(0, 1)
    p = qs.measure_probabilities_dict()
    assert set(p) == {"00", "11"}
    assert abs(p["00"] - 0.5) < 1e-9


def test_ghz_state():
    qs = QuantumState(4)
    qs.apply_h(0)
    for i in range(3):
        qs.apply_cnot(i, i + 1)
    p = qs.measure_probabilities_dict()
    assert set(p) == {"0000", "1111"}


def test_norm_preserved_after_random_gates():
    rng = np.random.default_rng(0)
    qs = QuantumState(5)
    for _ in range(50):
        a, b = rng.choice(5, 2, replace=False)
        qs.apply_h(int(a))
        qs.apply_cnot_general(int(a), int(b))
        qs.apply_rz(int(b), float(rng.uniform(0, 6)))
    assert abs(np.linalg.norm(qs.state) - 1) < 1e-10


def test_non_adjacent_cnot_matches_swap_trick():
    a = QuantumState(3)
    a.apply_h(0)
    a.apply_cnot_general(0, 2)
    b = QuantumState(3)
    b.apply_h(0)
    b.apply_swap_general(1, 2)
    b.apply_cnot(0, 1)
    b.apply_swap_general(1, 2)
    assert np.allclose(a.state, b.state)


def test_partial_measurement_collapses_and_renormalises():
    rng = np.random.default_rng(3)
    qs = QuantumState(2)
    qs.apply_h(0)
    qs.apply_cnot(0, 1)
    out = qs.measure_qubit(0, rng)
    assert abs(np.linalg.norm(qs.state) - 1) < 1e-12
    # entangled partner must now agree
    assert qs.measure_qubit(1, rng) == out


def test_partial_measurement_statistics():
    rng = np.random.default_rng(4)
    ones = 0
    trials = 2000
    for _ in range(trials):
        qs = QuantumState(1)
        qs.apply_ry(0, 2 * np.arcsin(np.sqrt(0.3)))   # P(1) = 0.3
        ones += qs.measure_qubit(0, rng)
    assert abs(ones / trials - 0.3) < 0.04


def test_new_gates_are_unitary_and_correct():
    from core_engine.custom_gates import (
        gate_SX, gate_T_dag, gate_P, gate_U3, gate_ISWAP, gate_CCX, gate_T, gate_RY)
    for U in [gate_SX(), gate_T_dag(), gate_P(0.4), gate_U3(0.3, 0.5, 0.7),
              gate_ISWAP(), gate_CCX()]:
        assert _is_unitary(U)
    assert np.allclose(gate_SX() @ gate_SX(), gate_X())
    assert np.allclose(gate_T() @ gate_T_dag(), np.eye(2))
    assert np.allclose(gate_U3(0.7, 0, 0), gate_RY(0.7))


def test_measure_shots_seed_is_local_and_marginals_work():
    from core_engine.measurement_channel import measure_shots
    np.random.seed(5)
    expected = np.random.rand()
    np.random.seed(5)
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    a = measure_shots(bell, 2, 200, seed=99)
    assert np.random.rand() == expected               # global RNG untouched
    assert a == measure_shots(bell, 2, 200, seed=99)  # reproducible
    st = np.zeros(8, dtype=complex)
    st[0b101] = 1
    assert measure_shots(st, 3, 50, seed=1, qubits=[2, 0]) == {"11": 50}
