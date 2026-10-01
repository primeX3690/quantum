"""
grover_search.py
-----------------
Grover's Search Algorithm (Lov Grover, 1996). The algorithm is public
research; what is original here is the ground-up implementation, no quantum
SDK anywhere.

Upgrade: the old version built dense 2^n x 2^n oracle and diffusion matrices
(O(4^n) memory, stuck around 12 qubits). Now both operators act directly on
the statevector:
  oracle     : flip the sign of the marked amplitudes          O(#targets)
  diffusion  : 2*mean(psi) - psi  (inversion about the mean)   O(2^n)
so 20+ qubits become possible on a laptop (cost per iteration is O(2^n) and
the number of iterations grows like sqrt(2^n), so total time is ~2^(1.5 n):
roughly a few seconds at 18-20 qubits, tens of seconds at 22). Multiple
marked items are supported. The old dense builders are kept (diffusion_operator) for
verification only.
"""

import numpy as np
from core_engine.custom_gates import gate_H

MAX_QUBITS = 24          # 2^24 * 16 B = 256 MiB per state vector


def diffusion_operator(n_qubits):
    """Dense diffusion matrix 2|s><s| - I. Verification only (small n)."""
    size = 2 ** n_qubits
    H_n = gate_H()
    for _ in range(n_qubits - 1):
        H_n = np.kron(H_n, gate_H())
    zero_state = np.zeros(size, dtype=complex)
    zero_state[0] = 1.0
    s = H_n @ zero_state
    return 2 * np.outer(s, s.conj()) - np.eye(size, dtype=complex)


def _targets(target_index, n_qubits):
    idx = [target_index] if np.isscalar(target_index) else list(target_index)
    idx = sorted({int(i) for i in idx})
    if not idx:
        raise ValueError("need at least one target")
    if idx[0] < 0 or idx[-1] >= 2 ** n_qubits:
        raise ValueError(f"target index out of range for {n_qubits} qubits")
    if len(idx) >= 2 ** n_qubits:
        raise ValueError("every state marked: nothing to search")
    return idx


def grover_statevector(n_qubits, target_index, iterations):
    """Final amplitudes after `iterations` Grover steps (O(2^n) each)."""
    if not 1 <= n_qubits <= MAX_QUBITS:
        raise ValueError(f"n_qubits must be in [1, {MAX_QUBITS}]")
    targets = _targets(target_index, n_qubits)
    N = 2 ** n_qubits
    # Grover amplitudes stay real (H, oracle, diffusion are real operators),
    # so a float64 vector is exact and ~2x faster / half the memory.
    psi = np.full(N, 1 / np.sqrt(N))                     # H^n |0...0>
    for _ in range(int(iterations)):
        psi[targets] *= -1                               # oracle
        np.subtract(2 * psi.mean(), psi, out=psi)        # diffusion, in place
    return psi.astype(complex)


def run_grover(n_qubits, target_index, iterations, top_k=None):
    """Returns {bitstring: probability}. For n <= 12 the full distribution
    (as before); for larger n only the top_k (default 16) outcomes."""
    psi = grover_statevector(n_qubits, target_index, iterations)
    probs = np.abs(psi) ** 2
    if top_k is None and n_qubits <= 12:
        return {format(i, f"0{n_qubits}b"): round(float(p), 6) for i, p in enumerate(probs)}
    k = min(top_k or 16, len(probs))
    top = np.argpartition(probs, -k)[-k:]
    top = top[np.argsort(probs[top])[::-1]]
    return {format(int(i), f"0{n_qubits}b"): round(float(probs[i]), 6) for i in top}


def optimal_iterations(n_qubits, n_targets=1):
    N = 2 ** n_qubits
    return int(np.floor((np.pi / 4) * np.sqrt(N / n_targets)))
