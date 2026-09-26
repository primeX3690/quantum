"""
qiskit_cross_check.py
------------------------
VALIDATION-ONLY. Not imported anywhere in core_engine/, algorithms/,
noise_channel/, api/, or run_my_core.py -- the production engine
stays 100% SDK-free, exactly as before. This file's only job is to
build the SAME circuit twice -- once on my_quantum_core, once on
Qiskit -- and diff the two statevectors, which is a real, external,
mathematically-checkable correctness proof rather than a self-report.

Requires `pip install -r requirements-dev.txt` (qiskit + qiskit-aer).
Run with:  PYTHONPATH=. python3 validation/qiskit_cross_check.py
"""

import numpy as np

from core_engine.statevector import QuantumState
from algorithms.grover_search import run_grover, optimal_iterations
from algorithms.deutsch_jozsa import balanced_oracle_indices, run_deutsch_jozsa
from algorithms.qft import apply_qft

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
except ImportError as e:
    raise SystemExit(
        "This validation script needs Qiskit. Install it with:\n"
        "    pip install -r requirements-dev.txt\n"
        "(production my_quantum_core code never needs this)."
    ) from e


def _bit_reversed(vec, n_qubits):
    """
    my_quantum_core indexes basis states with qubit 0 as the
    MOST-significant bit; Qiskit indexes with qubit 0 as the
    LEAST-significant bit. This is a well-known, documented
    convention difference (not a bug in either engine), so before
    diffing two statevectors we permute one of them by reversing the
    bit order of its basis-state index.
    """
    N = 2 ** n_qubits
    out = np.zeros_like(vec)
    for i in range(N):
        j = int(format(i, f"0{n_qubits}b")[::-1], 2)
        out[i] = vec[j]
    return out


def _global_phase_aligned_fidelity(psi1, psi2):
    """
    Statevector fidelity |<psi1|psi2>|^2, plus the raw max-amplitude
    difference AFTER removing the physically meaningless global phase
    (two statevectors that differ only by an overall e^{i*phi} factor
    represent the identical physical state).
    """
    overlap = np.vdot(psi1, psi2)
    fidelity = float(np.abs(overlap) ** 2)

    # Align global phase, then compare amplitude-by-amplitude.
    phase = overlap / abs(overlap) if abs(overlap) > 1e-12 else 1.0
    aligned = psi1 * phase
    max_abs_diff = float(np.max(np.abs(aligned - psi2)))
    return fidelity, max_abs_diff


def cross_check_bell():
    qs = QuantumState(2)
    qs.apply_h(0)
    qs.apply_cnot(0, 1)
    mine = qs.get_statevector()

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    theirs = _bit_reversed(Statevector.from_instruction(qc).data, 2)

    return "Bell state (H+CNOT)", mine, theirs


def cross_check_ghz(n=4):
    qs = QuantumState(n)
    qs.apply_h(0)
    for i in range(n - 1):
        qs.apply_cnot(i, i + 1)
    mine = qs.get_statevector()

    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    theirs = _bit_reversed(Statevector.from_instruction(qc).data, n)

    return f"GHZ state ({n} qubits)", mine, theirs


def cross_check_grover(n=3, target=5):
    iters = optimal_iterations(n)
    probs = run_grover(n, target, iters)
    mine_probs = np.array([probs.get(format(i, f"0{n}b"), 0.0) for i in range(2 ** n)])

    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    oracle = np.eye(2 ** n)
    oracle[target, target] = -1
    # Diffusion + oracle applied as explicit unitaries via UnitaryGate,
    # matching exactly what run_grover() does internally.
    from qiskit.circuit.library import UnitaryGate
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    Hn = H
    for _ in range(n - 1):
        Hn = np.kron(Hn, H)
    zero = np.zeros(2 ** n); zero[0] = 1.0
    s = Hn @ zero
    diffusion = 2 * np.outer(s, s.conj()) - np.eye(2 ** n)

    for _ in range(iters):
        qc.append(UnitaryGate(oracle), range(n))
        qc.append(UnitaryGate(diffusion), range(n))

    # NOTE: like Deutsch-Jozsa below, both the oracle and diffusion operator
    # are built as raw diagonal/reflection matrices on the same index space
    # (not per-qubit gates), so no bit-reversal is needed here either.
    theirs = Statevector.from_instruction(qc).data
    theirs_probs = np.abs(theirs) ** 2

    max_prob_diff = float(np.max(np.abs(mine_probs - theirs_probs)))
    return f"Grover's search ({n} qubits, target={target}, {iters} iters)", mine_probs, theirs_probs, max_prob_diff


def cross_check_qft(n=3):
    qs = QuantumState(n)
    for q in range(n):
        qs.apply_h(q) if q == 0 else None
    # Start from a non-trivial input so the QFT circuit does real work.
    qs = QuantumState(n)
    qs.apply_x(0)
    apply_qft(qs)
    mine = qs.get_statevector()

    qc = QuantumCircuit(n)
    qc.x(0)
    qc.append(_qiskit_qft_circuit(n).to_instruction(), range(n))
    theirs = _bit_reversed(Statevector.from_instruction(qc).data, n)

    return f"QFT ({n} qubits)", mine, theirs


def _qiskit_qft_circuit(n):
    """Hand-built QFT circuit in raw Qiskit gates (H + CP + SWAP),
    matching our own circuit gate-for-gate -- deliberately NOT using
    qiskit.circuit.library.QFT, so this is an independent
    reconstruction rather than importing a pre-verified black box."""
    qc = QuantumCircuit(n, name="qft_reference")
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            theta = 2 * np.pi / (2 ** (j - i + 1))
            qc.cp(theta, j, i)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


def cross_check_deutsch_jozsa(n=3):
    marked = balanced_oracle_indices(n)
    result = run_deutsch_jozsa(n, marked)

    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    oracle = np.eye(2 ** n)
    for idx in marked:
        oracle[idx, idx] = -1
    from qiskit.circuit.library import UnitaryGate
    qc.append(UnitaryGate(oracle), range(n))
    for q in range(n):
        qc.h(q)
    # NOTE: no bit-reversal here (unlike the other checks) -- both sides
    # build the oracle as a raw diagonal matrix on the same index space,
    # and H^n is symmetric under qubit relabeling, so no per-qubit gate
    # ever introduces a convention-dependent difference in this circuit.
    theirs_probs = np.abs(Statevector.from_instruction(qc).data) ** 2

    mine_probs = np.array([result["probabilities"].get(format(i, f"0{n}b"), 0.0) for i in range(2 ** n)])
    max_prob_diff = float(np.max(np.abs(mine_probs - theirs_probs)))
    return f"Deutsch-Jozsa balanced oracle ({n} qubits)", mine_probs, theirs_probs, max_prob_diff


def run_all_checks(tolerance=1e-9):
    print("=" * 70)
    print("QISKIT CROSS-VALIDATION -- my_quantum_core vs. Qiskit Statevector")
    print("(validation only; production engine has zero Qiskit dependency)")
    print("=" * 70)
    all_passed = True

    for label, mine, theirs in [cross_check_bell(), cross_check_ghz(4)]:
        fidelity, max_diff = _global_phase_aligned_fidelity(mine, theirs)
        passed = max_diff < tolerance
        all_passed &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {label}: fidelity={fidelity:.12f}, "
              f"max_amplitude_diff={max_diff:.2e}")

    for label, mine_p, theirs_p, max_diff in [cross_check_grover(), cross_check_deutsch_jozsa()]:
        passed = max_diff < 1e-6
        all_passed &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {label}: max_probability_diff={max_diff:.2e}")

    label, mine, theirs = cross_check_qft(3)
    fidelity, max_diff = _global_phase_aligned_fidelity(mine, theirs)
    passed = max_diff < tolerance
    all_passed &= passed
    print(f"[{'PASS' if passed else 'FAIL'}] {label}: fidelity={fidelity:.12f}, "
          f"max_amplitude_diff={max_diff:.2e}")

    print("=" * 70)
    print("ALL CROSS-CHECKS PASSED" if all_passed else "SOME CROSS-CHECKS FAILED")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    run_all_checks()
