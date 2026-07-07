"""
statevector.py
----------------
Core quantum simulation engine. Represents an n-qubit system as a
complex vector of size 2^n and applies gates via matrix multiplication.
"""

import numpy as np
from core_engine.custom_gates import gate_H, gate_X, gate_Z, gate_CNOT
from core_engine.kronecker_solver import expand_gate_to_n_qubits


class QuantumState:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.size = 2 ** n_qubits
        self.state = np.zeros(self.size, dtype=complex)
        self.state[0] = 1.0

    def apply_gate(self, gate, target_qubits):
        full_operator = expand_gate_to_n_qubits(gate, target_qubits, self.n_qubits)
        self.state = full_operator @ self.state

    def apply_h(self, qubit):
        self.apply_gate(gate_H(), [qubit])

    def apply_x(self, qubit):
        self.apply_gate(gate_X(), [qubit])

    def apply_z(self, qubit):
        self.apply_gate(gate_Z(), [qubit])

    def apply_cnot(self, control, target):
        if target != control + 1:
            raise NotImplementedError(
                "This version supports CNOT only on adjacent qubits (control, control+1)."
            )
        self.apply_gate(gate_CNOT(), [control, target])

    def probabilities(self):
        return np.abs(self.state) ** 2

    def measure_probabilities_dict(self):
        probs = self.probabilities()
        result = {}
        for i, p in enumerate(probs):
            label = format(i, f'0{self.n_qubits}b')
            if p > 1e-10:
                result[label] = round(float(p), 6)
        return result

    def sample(self, shots=1000):
        probs = self.probabilities()
        outcomes = np.random.choice(self.size, size=shots, p=probs)
        counts = {}
        for outcome in outcomes:
            label = format(outcome, f'0{self.n_qubits}b')
            counts[label] = counts.get(label, 0) + 1
        return counts

    def __repr__(self):
        return f"QuantumState({self.n_qubits} qubits, state={np.round(self.state, 3)})"