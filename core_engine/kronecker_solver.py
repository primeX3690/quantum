"""
kronecker_solver.py
--------------------
Tensor product (Kronecker product) logic for combining multi-qubit
systems into a single state space.
"""

import numpy as np


def tensor_product(*matrices):
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def expand_gate_to_n_qubits(gate, target_qubits, n_qubits):
    from core_engine.custom_gates import gate_I

    if len(target_qubits) == 1:
        q = target_qubits[0]
        ops = [gate if i == q else gate_I() for i in range(n_qubits)]
        return tensor_product(*ops)

    elif len(target_qubits) == 2 and gate.shape == (4, 4):
        q0, q1 = target_qubits
        if q1 != q0 + 1:
            raise NotImplementedError(
                "Only adjacent qubits are supported for 2-qubit gates in this version."
            )
        ops = []
        i = 0
        while i < n_qubits:
            if i == q0:
                ops.append(gate)
                i += 2
            else:
                ops.append(gate_I())
                i += 1
        return tensor_product(*ops)

    else:
        raise ValueError("Unsupported gate size or target_qubits combination.")