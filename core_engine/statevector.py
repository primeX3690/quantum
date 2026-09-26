"""
statevector.py
----------------
Core quantum simulation engine. Represents an n-qubit system as a
complex vector of size 2^n and applies gates via matrix multiplication.
"""

import numpy as np
from core_engine.custom_gates import (
    gate_H, gate_X, gate_Z, gate_CNOT,
    gate_Y, gate_S, gate_S_dag, gate_T,
    gate_RX, gate_RY, gate_RZ, gate_CZ, gate_SWAP, gate_CPHASE,
)
from core_engine.kronecker_solver import (
    expand_gate_to_n_qubits,
    apply_two_qubit_gate_general,
    apply_single_qubit_gate_efficient,
)


class QuantumState:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.size = 2 ** n_qubits
        self.state = np.zeros(self.size, dtype=complex)
        self.state[0] = 1.0

    def apply_gate(self, gate, target_qubits):
        """
        Dispatches to the efficient O(2^n) tensor-reshape path for
        single-qubit gates and adjacent two-qubit gates (used by every
        apply_* convenience method below), scaling to 20+ qubits.
        Falls back to the older O(4^n) full-matrix path
        (expand_gate_to_n_qubits) only for anything that path doesn't
        cover, kept for backward compatibility.
        """
        if len(target_qubits) == 1:
            self.state = apply_single_qubit_gate_efficient(self.state, gate, target_qubits[0], self.n_qubits)
            return
        if len(target_qubits) == 2 and gate.shape == (4, 4):
            q0, q1 = target_qubits
            self.state = apply_two_qubit_gate_general(self.state, gate, q0, q1, self.n_qubits)
            return
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

    # -- Added for Deutsch-Jozsa / QFT / VQE -------------------------------
    def apply_y(self, qubit):
        self.apply_gate(gate_Y(), [qubit])

    def apply_s(self, qubit):
        self.apply_gate(gate_S(), [qubit])

    def apply_s_dag(self, qubit):
        self.apply_gate(gate_S_dag(), [qubit])

    def apply_t(self, qubit):
        self.apply_gate(gate_T(), [qubit])

    def apply_rx(self, qubit, theta):
        self.apply_gate(gate_RX(theta), [qubit])

    def apply_ry(self, qubit, theta):
        self.apply_gate(gate_RY(theta), [qubit])

    def apply_rz(self, qubit, theta):
        self.apply_gate(gate_RZ(theta), [qubit])

    def apply_cz_general(self, q0, q1):
        """CZ between ANY two qubits (adjacent or not)."""
        self.state = apply_two_qubit_gate_general(self.state, gate_CZ(), q0, q1, self.n_qubits)

    def apply_swap_general(self, q0, q1):
        """SWAP between ANY two qubits (adjacent or not)."""
        self.state = apply_two_qubit_gate_general(self.state, gate_SWAP(), q0, q1, self.n_qubits)

    def apply_cphase_general(self, q0, q1, theta):
        """Controlled-phase between ANY two qubits -- the gate the QFT needs."""
        self.state = apply_two_qubit_gate_general(self.state, gate_CPHASE(theta), q0, q1, self.n_qubits)

    def apply_cnot_general(self, control, target):
        """CNOT between ANY two qubits (adjacent or not), unlike apply_cnot()."""
        self.state = apply_two_qubit_gate_general(self.state, gate_CNOT(), control, target, self.n_qubits)

    def get_statevector(self):
        return self.state.copy()

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