"""
custom_gates.py
----------------
Quantum gate matrices, implemented from first principles using only
NumPy. No external quantum SDK (Qiskit/Cirq) used or referenced.
"""

import numpy as np


def gate_I():
    return np.array([[1, 0], [0, 1]], dtype=complex)


def gate_X():
    return np.array([[0, 1], [1, 0]], dtype=complex)


def gate_H():
    factor = 1 / np.sqrt(2)
    return factor * np.array([[1, 1], [1, -1]], dtype=complex)


def gate_Z():
    return np.array([[1, 0], [0, -1]], dtype=complex)


def gate_CNOT():
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)


def oracle_mark_state(target_index, n_qubits):
    """Generic phase-flip oracle used by Grover's algorithm."""
    size = 2 ** n_qubits
    oracle = np.eye(size, dtype=complex)
    oracle[target_index, target_index] = -1
    return oracle


# ---------------------------------------------------------------------------
# Extra single/two-qubit gates -- added to support Deutsch-Jozsa, QFT and the
# toy VQE ansatz. Still first-principles NumPy matrices, no SDK involved.
# ---------------------------------------------------------------------------

def gate_Y():
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def gate_S():
    return np.array([[1, 0], [0, 1j]], dtype=complex)


def gate_S_dag():
    return np.array([[1, 0], [0, -1j]], dtype=complex)


def gate_T():
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def gate_RX(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def gate_RY(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def gate_RZ(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                      [0, np.exp(1j * theta / 2)]], dtype=complex)


def gate_CZ():
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ], dtype=complex)


def gate_SWAP():
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=complex)


def gate_CPHASE(theta):
    """Controlled phase rotation -- the workhorse gate of the QFT."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * theta)],
    ], dtype=complex)


def multi_controlled_z_oracle(marked_indices, n_qubits):
    """Phase-flip oracle that marks an arbitrary set of computational-basis
    indices with a -1 phase. Used for the Deutsch-Jozsa balanced oracle."""
    size = 2 ** n_qubits
    oracle = np.eye(size, dtype=complex)
    for idx in marked_indices:
        oracle[idx, idx] = -1
    return oracle