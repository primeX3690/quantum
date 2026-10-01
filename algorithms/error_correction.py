"""
error_correction.py
----------------------
3-qubit bit-flip error correction code. Encodes 1 logical qubit into
3 physical qubits so a single bit-flip error can be detected and
corrected via majority vote. (Foundational QEC concept — not a
research-level code like Shor's 9-qubit or surface codes.)
"""

import numpy as np
from core_engine.statevector import QuantumState


def encode_bit_flip(logical_state_is_one=False):
    """Encodes |0> -> |000> or |1> -> |111> using 2 CNOTs."""
    qs = QuantumState(n_qubits=3)
    if logical_state_is_one:
        qs.apply_x(0)
    qs.apply_cnot(0, 1)
    qs.apply_cnot(1, 2)
    return qs


def introduce_bit_flip_error(qs, qubit_index):
    """Simulates a physical error flipping one specific qubit."""
    qs.apply_x(qubit_index)
    return qs


def detect_and_correct(qs):
    """Majority-vote recovery of the original logical bit."""
    probs = qs.measure_probabilities_dict()
    most_likely = max(probs, key=probs.get)
    bits = [int(b) for b in most_likely]
    majority = 1 if sum(bits) >= 2 else 0
    return majority, most_likely

# ===========================================================================
# REAL syndrome-based error correction (ancilla qubits, indirect measurement)
# ---------------------------------------------------------------------------
# The functions above measure the data qubits directly and take a majority
# vote. That destroys a logical superposition and is not how real QEC works.
# Below: 3 data qubits (0,1,2) + 2 ancillas (3,4). Ancillas record the parities
# q0^q1 and q1^q2 via CNOTs; ONLY the ancillas are measured. The syndrome
# says WHICH qubit flipped without revealing the logical state, so
# alpha|0>+beta|1> survives. Correction is classical feed-forward.
# ===========================================================================

_SYNDROME_TABLE = {(0, 0): None, (1, 0): 0, (1, 1): 1, (0, 1): 2}


def encode_logical(alpha_beta_theta=0.0, code="bit"):
    """Prepare cos(t/2)|0>+sin(t/2)|1> and encode into 3 data qubits of a
    5-qubit register. code='bit' guards X errors, 'phase' guards Z errors."""
    qs = QuantumState(n_qubits=5)
    qs.apply_ry(0, alpha_beta_theta)
    qs.apply_cnot_general(0, 1)
    qs.apply_cnot_general(0, 2)
    if code == "phase":
        for q in range(3):
            qs.apply_h(q)
    elif code != "bit":
        raise ValueError("code must be 'bit' or 'phase'")
    return qs


def inject_error(qs, qubit, code="bit"):
    if code == "bit":
        qs.apply_x(qubit)
    else:
        qs.apply_z(qubit)
    return qs


def extract_syndrome(qs, code="bit", rng=None):
    """Ancilla-based parity checks. Only ancillas 3,4 are measured."""
    if code == "phase":                 # rotate so phase flips look like bit flips
        for q in range(3):
            qs.apply_h(q)
    qs.reset_qubit(3, rng)
    qs.reset_qubit(4, rng)
    qs.apply_cnot_general(0, 3)
    qs.apply_cnot_general(1, 3)
    qs.apply_cnot_general(1, 4)
    qs.apply_cnot_general(2, 4)
    s1 = qs.measure_qubit(3, rng)
    s2 = qs.measure_qubit(4, rng)
    return (s1, s2)


def correct_from_syndrome(qs, syndrome, code="bit"):
    """Classical feed-forward: apply X (bit code) on the flagged qubit."""
    bad = _SYNDROME_TABLE[syndrome]
    if bad is not None:
        qs.apply_x(bad)
    if code == "phase":                 # rotate back to the phase-code basis
        for q in range(3):
            qs.apply_h(q)
    return bad


def logical_fidelity(qs, theta, code="bit", syndrome=(0, 0)):
    """|<expected|state>|^2 where expected = encoded state (x) |syndrome>."""
    ref = encode_logical(theta, code)
    ref.apply_x(3) if syndrome[0] else None
    ref.apply_x(4) if syndrome[1] else None
    return float(abs(np.vdot(ref.state, qs.state)) ** 2)


def run_qec_cycle(theta=1.0, error_qubit=None, code="bit", rng=None):
    """One full encode -> error -> syndrome -> correct cycle.
    Returns dict with syndrome, corrected qubit, and fidelity vs the
    original encoded logical state (1.0 = perfectly preserved)."""
    qs = encode_logical(theta, code)
    if error_qubit is not None:
        inject_error(qs, error_qubit, code)
    syn = extract_syndrome(qs, code, rng)
    fixed = correct_from_syndrome(qs, syn, code)
    fid = logical_fidelity(qs, theta, code, syn)
    return {"syndrome": syn, "corrected_qubit": fixed, "fidelity": fid}


def monte_carlo_logical_error(p, trials=500, code="bit", theta=1.0, seed=0):
    """Independent physical error prob p per data qubit. Returns measured
    logical failure rate vs. theory 3p^2-2p^3 (uncoded would be p)."""
    rng = np.random.default_rng(seed)
    fails = 0
    for _ in range(trials):
        qs = encode_logical(theta, code)
        for q in range(3):
            if rng.random() < p:
                inject_error(qs, q, code)
        syn = extract_syndrome(qs, code, rng)
        correct_from_syndrome(qs, syn, code)
        if logical_fidelity(qs, theta, code, syn) < 0.99:
            fails += 1
    return {"p_physical": p, "logical_error_rate": fails / trials,
            "theory": 3 * p ** 2 - 2 * p ** 3}
