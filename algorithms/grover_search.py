"""
grover_search.py
-----------------
Grover's Search Algorithm. The underlying algorithm is public research
(Lov Grover, 1996) — implementing it does not make it "proprietary."
What is original here is the ground-up implementation: no quantum SDK
is used anywhere in this code.
"""

import numpy as np
from core_engine.custom_gates import gate_H, oracle_mark_state


def diffusion_operator(n_qubits):
    size = 2 ** n_qubits
    H = gate_H()
    H_n = H
    for _ in range(n_qubits - 1):
        H_n = np.kron(H_n, H)
    zero_state = np.zeros(size, dtype=complex)
    zero_state[0] = 1.0
    s = H_n @ zero_state
    projector = 2 * np.outer(s, s.conj())
    identity = np.eye(size, dtype=complex)
    return projector - identity


def run_grover(n_qubits, target_index, iterations):
    size = 2 ** n_qubits
    H = gate_H()
    H_n = H
    for _ in range(n_qubits - 1):
        H_n = np.kron(H_n, H)
    state = np.zeros(size, dtype=complex)
    state[0] = 1.0
    state = H_n @ state

    oracle = oracle_mark_state(target_index, n_qubits)
    diffusion = diffusion_operator(n_qubits)

    for _ in range(iterations):
        state = oracle @ state
        state = diffusion @ state

    probs = np.abs(state) ** 2
    result = {}
    for i, p in enumerate(probs):
        label = format(i, f'0{n_qubits}b')
        result[label] = round(float(p), 6)
    return result


def optimal_iterations(n_qubits):
    N = 2 ** n_qubits
    return int(np.floor((np.pi / 4) * np.sqrt(N)))