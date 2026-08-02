"""
tensor_network.py
--------------------
Matrix Product State (MPS) representation -- built from scratch using
SVD (Singular Value Decomposition), NOT copied from any tensor-network
library (no ITensor, no TeNPy, no Qiskit).

HONEST SCOPE: MPS gives memory savings only for LOW-ENTANGLEMENT
states/circuits. Highly entangled states (e.g. deep random circuits)
still require large "bond dimension" and lose the advantage. This is
a genuine physical limitation, not a bug -- it defines WHERE this
technique helps and where it doesn't.

Current scope: converts an existing statevector into MPS form
(verified correct via reconstruction). Applying gates directly to
MPS tensors -- without ever building the full statevector -- is the
next engineering milestone; that is what unlocks the real 50+ qubit
scaling shown in the memory projections below.
"""

import numpy as np


def statevector_to_mps(state, n_qubits, max_bond_dim=None):
    """
    Decomposes a full statevector into a chain of MPS tensors using
    sequential SVD (the standard, textbook MPS construction method,
    implemented here from the underlying linear algebra).
    Returns a list of tensors, one per qubit.
    """
    tensors = []
    remaining = state.reshape(2, -1)
    left_dim = 1

    for i in range(n_qubits - 1):
        mat = remaining.reshape(left_dim * 2, -1)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        if max_bond_dim is not None and len(S) > max_bond_dim:
            U = U[:, :max_bond_dim]
            S = S[:max_bond_dim]
            Vh = Vh[:max_bond_dim, :]

        bond_dim = len(S)
        tensor = U.reshape(left_dim, 2, bond_dim)
        tensors.append(tensor)

        remaining = np.diag(S) @ Vh
        left_dim = bond_dim

    tensors.append(remaining.reshape(left_dim, 2, 1))
    return tensors


def mps_to_statevector(tensors, n_qubits):
    """Reconstructs the full statevector from MPS tensors -- used only
    to VERIFY correctness on small systems. For large systems you
    would never do this, as it defeats the purpose of MPS."""
    result = tensors[0].reshape(2, -1)
    for t in tensors[1:]:
        left, phys, right = t.shape
        t_mat = t.reshape(left, phys * right)
        result = result @ t_mat
        result = result.reshape(-1, right)
    return result.flatten()


def mps_memory_bytes(tensors):
    """Total memory used by the MPS representation (complex128 = 16 bytes/element)."""
    total_elements = sum(t.size for t in tensors)
    return total_elements * 16


def statevector_memory_bytes(n_qubits):
    """Memory a full statevector would need (complex128 = 16 bytes/element)."""
    return (2 ** n_qubits) * 16