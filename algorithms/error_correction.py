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