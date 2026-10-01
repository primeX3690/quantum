"""
measurement_channel.py
------------------------
Converts ideal statevector probabilities into real-hardware-style
"shot counts" (e.g. {'00': 512, '11': 512} instead of exact probabilities).

Upgrade: uses a LOCAL random generator, so a `seed` gives reproducible
counts without silently re-seeding NumPy's global RNG (the old version's
np.random.seed(seed) changed random behaviour everywhere else in the
program). Counting is vectorised, and you can measure a subset of qubits.
"""

import numpy as np


def measure_shots(state, n_qubits, shots=1024, seed=None, qubits=None, rng=None):
    """Sample `shots` measurement outcomes.

    qubits : optional list of qubit indices; the returned bitstrings then
             contain only those qubits (in the order given) = marginal counts.
    rng    : optional numpy Generator; otherwise one is built from `seed`.
    """
    if shots < 1:
        raise ValueError("shots must be >= 1")
    rng = rng if rng is not None else np.random.default_rng(seed)
    probs = np.abs(np.asarray(state)) ** 2
    probs = probs / probs.sum()
    outcomes = rng.choice(len(probs), size=shots, p=probs)
    if qubits is None:
        values, freq = np.unique(outcomes, return_counts=True)
        return {format(int(v), f"0{n_qubits}b"): int(c) for v, c in zip(values, freq)}
    for q in qubits:
        if not 0 <= q < n_qubits:
            raise ValueError(f"qubit {q} out of range")
    sub = np.zeros(len(outcomes), dtype=np.int64)
    for q in qubits:
        sub = (sub << 1) | ((outcomes >> (n_qubits - 1 - q)) & 1)
    values, freq = np.unique(sub, return_counts=True)
    return {format(int(v), f"0{len(qubits)}b"): int(c) for v, c in zip(values, freq)}
