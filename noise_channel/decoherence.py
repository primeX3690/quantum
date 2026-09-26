"""
decoherence.py
----------------
Noise/error channel modeling using density matrix formalism, built
from scratch. Pure statevectors cannot represent probabilistic noise
mixtures, so we use rho = |psi><psi| and Kraus operators instead.
"""

import numpy as np


def statevector_to_density_matrix(state):
    return np.outer(state, state.conj())


def _expand_single_qubit_op(op, qubit, n_qubits):
    I = np.eye(2, dtype=complex)
    ops = [op if i == qubit else I for i in range(n_qubits)]
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


def bit_flip_channel(rho, p, qubit, n_qubits):
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    K0 = np.sqrt(1 - p) * _expand_single_qubit_op(I, qubit, n_qubits)
    K1 = np.sqrt(p) * _expand_single_qubit_op(X, qubit, n_qubits)
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def amplitude_damping_channel(rho, gamma, qubit, n_qubits):
    K0_single = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1_single = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    K0 = _expand_single_qubit_op(K0_single, qubit, n_qubits)
    K1 = _expand_single_qubit_op(K1_single, qubit, n_qubits)
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def phase_damping_channel(rho, lam, qubit, n_qubits):
    """
    Pure dephasing channel (T2-driven), Kraus operators
    K0 = diag(1, sqrt(1-lambda)), K1 = diag(0, sqrt(lambda)).
    Added alongside the existing bit-flip / amplitude-damping channels
    so IBM's published T2 numbers can be modeled as what they actually
    are physically (dephasing), not folded into the T1 (amplitude
    damping) channel as an approximation.
    """
    K0_single = np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex)
    K1_single = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)
    K0 = _expand_single_qubit_op(K0_single, qubit, n_qubits)
    K1 = _expand_single_qubit_op(K1_single, qubit, n_qubits)
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def get_probabilities_from_density_matrix(rho):
    n_qubits = int(np.log2(rho.shape[0]))
    diag = np.real(np.diag(rho))
    result = {}
    for i, p in enumerate(diag):
        label = format(i, f'0{n_qubits}b')
        if p > 1e-6:
            result[label] = round(float(p), 6)
    return result