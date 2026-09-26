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


def apply_single_qubit_gate_efficient(state, gate, qubit, n_qubits):
    """
    Applies a 2x2 gate to one qubit directly on the statevector via
    tensor reshaping, WITHOUT building the full 2^n x 2^n operator
    matrix that expand_gate_to_n_qubits() does. The old kron-based
    path allocates O(4^n) memory and is only practical up to ~14
    qubits on an 8GB laptop; this path is O(2^n) and scales far
    higher (20+ qubits), which matters because it's the same
    statevector that later gets fed into the MPS compressor -- there
    is no point compressing a state you could not afford to build in
    the first place.
    """
    tensor = state.reshape([2] * n_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    front_shape = tensor.shape
    flat = tensor.reshape(2, -1)
    flat = gate @ flat
    tensor = flat.reshape(front_shape)
    tensor = np.moveaxis(tensor, 0, qubit)
    return tensor.reshape(-1)


def apply_two_qubit_gate_general(state, gate, q0, q1, n_qubits):
    """
    Applies an arbitrary 4x4 two-qubit gate to ANY pair of qubits
    (adjacent or not) directly on the statevector, using tensor
    reshaping instead of building a full 2^n x 2^n matrix.

    This is what unlocks QFT's controlled-phase rotations between
    distant qubits and general SWAP, which the older
    expand_gate_to_n_qubits() (adjacent-only) could not do.

    Implementation: reshape the flat state into an n-qubit tensor of
    shape (2,2,...,2), move the two target axes to the front, apply
    the gate as a 4x4 matrix on the flattened (2x2) leading block,
    then move the axes back.
    """
    tensor = state.reshape([2] * n_qubits)
    tensor = np.moveaxis(tensor, [q0, q1], [0, 1])
    front_shape = tensor.shape
    flat = tensor.reshape(4, -1)
    flat = gate @ flat
    tensor = flat.reshape(front_shape)
    tensor = np.moveaxis(tensor, [0, 1], [q0, q1])
    return tensor.reshape(-1)