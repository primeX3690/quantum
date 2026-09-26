"""
deutsch_jozsa.py
------------------
Deutsch-Jozsa Algorithm (Deutsch & Jozsa, 1992). Public, published
algorithm -- what's original here is the from-scratch implementation:
no quantum SDK is used anywhere in this file.

Problem: given a black-box function f: {0,1}^n -> {0,1} that is
promised to be either CONSTANT (same output for every input) or
BALANCED (0 for exactly half the inputs, 1 for the other half),
decide which -- using exactly ONE query to f, something no classical
algorithm can guarantee.

We build the oracle as a diagonal phase-flip unitary acting on the
n input qubits directly (the standard "phase kickback" formulation),
which keeps the whole thing inside the existing statevector engine
without needing a separate ancilla register.
"""

import numpy as np
from core_engine.statevector import QuantumState


def constant_oracle_indices(n_qubits, output_bit=0):
    """A constant function marks either ALL basis states (f=1) or NONE (f=0)."""
    size = 2 ** n_qubits
    return list(range(size)) if output_bit == 1 else []


def balanced_oracle_indices(n_qubits):
    """
    A simple, canonical balanced function: f(x) = x_0 (the value of the
    first input bit). Exactly half of all basis states have x_0 = 1,
    so this marks exactly half the indices -- genuinely balanced.
    """
    size = 2 ** n_qubits
    marked = [i for i in range(size) if (i >> (n_qubits - 1)) & 1 == 1]
    assert len(marked) == size // 2, "Oracle is not balanced"
    return marked


def build_oracle(n_qubits, marked_indices):
    from core_engine.custom_gates import multi_controlled_z_oracle
    return multi_controlled_z_oracle(marked_indices, n_qubits)


def run_deutsch_jozsa(n_qubits, marked_indices):
    """
    Runs the algorithm: H^n -> oracle (phase-flip on marked indices)
    -> H^n -> measure. If the all-zeros state has probability ~1, f is
    CONSTANT. Otherwise f is BALANCED.
    """
    qs = QuantumState(n_qubits)
    for q in range(n_qubits):
        qs.apply_h(q)

    oracle = build_oracle(n_qubits, marked_indices)
    qs.state = oracle @ qs.state

    for q in range(n_qubits):
        qs.apply_h(q)

    probs = qs.measure_probabilities_dict()
    p_all_zero = probs.get("0" * n_qubits, 0.0)
    verdict = "CONSTANT" if p_all_zero > 0.999 else "BALANCED"
    return {
        "probabilities": probs,
        "p_all_zero": p_all_zero,
        "verdict": verdict,
    }


def demo_both_cases(n_qubits=3):
    results = {}
    results["constant_f=0"] = run_deutsch_jozsa(n_qubits, constant_oracle_indices(n_qubits, 0))
    results["constant_f=1"] = run_deutsch_jozsa(n_qubits, constant_oracle_indices(n_qubits, 1))
    results["balanced"] = run_deutsch_jozsa(n_qubits, balanced_oracle_indices(n_qubits))
    return results


if __name__ == "__main__":
    for name, res in demo_both_cases(4).items():
        print(f"{name}: verdict={res['verdict']}  p(all-zero)={res['p_all_zero']:.4f}")
