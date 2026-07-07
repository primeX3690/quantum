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