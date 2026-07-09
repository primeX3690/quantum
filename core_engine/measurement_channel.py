"""
measurement_channel.py
------------------------
Converts ideal statevector probabilities into real-hardware-style
"shot counts" — how actual quantum computers report results
(e.g. {'00': 512, '11': 512} instead of exact probabilities).
"""

import numpy as np


def measure_shots(state, n_qubits, shots=1024, seed=None):
    if seed is not None:
        np.random.seed(seed)
    probs = np.abs(state) ** 2
    probs = probs / np.sum(probs)
    size = 2 ** n_qubits
    outcomes = np.random.choice(size, size=shots, p=probs)
    counts = {}
    for outcome in outcomes:
        label = format(outcome, f'0{n_qubits}b')
        counts[label] = counts.get(label, 0) + 1
    return counts