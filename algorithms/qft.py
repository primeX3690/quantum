"""
qft.py
--------
Quantum Fourier Transform (Coppersmith, 1994). Public, published
algorithm -- what's original here is the from-scratch circuit
implementation using the engine's own gate set (H + controlled-phase
+ SWAP), with zero quantum SDK involved.

This is also what forced the engine-level upgrade in
kronecker_solver.apply_two_qubit_gate_general(): QFT needs
controlled-phase gates between qubits that are NOT adjacent
(qubit 0 talks to qubit n-1), which the original adjacent-only
two-qubit gate application could not do.
"""

import numpy as np
from core_engine.statevector import QuantumState


def apply_qft(qs: QuantumState, qubits=None):
    """
    Applies the QFT in place to the given QuantumState, on the given
    qubit indices (defaults to all qubits), using the standard
    textbook circuit:
        for each qubit i (from most-significant to least):
            H(i)
            for each qubit j > i:
                controlled-phase(2*pi / 2^(j-i+1)) between j (control) and i (target)
        then reverse qubit order with SWAPs (QFT naturally outputs
        the result in bit-reversed order).
    """
    if qubits is None:
        qubits = list(range(qs.n_qubits))
    n = len(qubits)

    for i in range(n):
        qs.apply_h(qubits[i])
        for j in range(i + 1, n):
            theta = 2 * np.pi / (2 ** (j - i + 1))
            qs.apply_cphase_general(qubits[j], qubits[i], theta)

    for i in range(n // 2):
        qs.apply_swap_general(qubits[i], qubits[n - 1 - i])

    return qs


def qft_matrix(n_qubits):
    """The textbook N x N unitary DFT matrix (Fourier basis), built
    directly from its closed-form definition -- used only to
    mathematically verify the circuit above reproduces the correct
    unitary. No SDK, just the definition of QFT itself."""
    N = 2 ** n_qubits
    omega = np.exp(2j * np.pi / N)
    j, k = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return (omega ** (j * k)) / np.sqrt(N)


def verify_qft_against_definition(n_qubits, trials=5, seed=0):
    """
    Applies the circuit to `trials` random input states and checks the
    output matches state = QFT_matrix @ input to within numerical
    tolerance. This is an internal mathematical correctness check
    (definition vs. circuit), independent of any external library.
    """
    rng = np.random.default_rng(seed)
    F = qft_matrix(n_qubits)
    max_err = 0.0

    for _ in range(trials):
        vec = rng.normal(size=2 ** n_qubits) + 1j * rng.normal(size=2 ** n_qubits)
        vec = vec / np.linalg.norm(vec)

        qs = QuantumState(n_qubits)
        qs.state = vec.copy()
        apply_qft(qs)

        expected = F @ vec
        err = np.max(np.abs(qs.state - expected))
        max_err = max(max_err, err)

    return {"n_qubits": n_qubits, "trials": trials, "max_abs_error": float(max_err),
            "passed": bool(max_err < 1e-9)}


if __name__ == "__main__":
    for n in [2, 3, 4]:
        print(verify_qft_against_definition(n))
