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

BUG FIXED (see benchmarks/external_reference_comparison.py for how
this was caught): the original version only dropped singular values
when an explicit max_bond_dim was passed. With max_bond_dim=None (the
default used everywhere, including the earlier "50 qubits -> 6.1KB"
projection), np.linalg.svd's full_matrices=False mode still returns
EVERY singular value down to float noise, so the bond dimension
always grew to the generic worst-case profile 2^min(i, n-i) --
even for a fully separable, zero-entanglement product state. That
made the MPS representation LARGER than the plain statevector, not
smaller, for every circuit actually run through it. The fix below
truncates singular values that are numerically zero (below
`svd_cutoff`, relative to the largest singular value at that cut)
by default, which is the standard, lossless-to-numerical-precision
practice every real MPS implementation uses. Genuine physically
motivated truncation (accepting some error to bound bond dimension
on a highly entangled state) is a separate, explicit choice via
max_bond_dim -- the two knobs are independent.

Scope of THIS module: converts an existing statevector into MPS form
(verified by reconstruction) and reports memory. Gates applied directly
on MPS tensors -- never building the full statevector -- now live in
core_engine/mps_engine.py (MPSState), which is what unlocks 50-100+
qubit scaling for low-entanglement circuits.
"""

import numpy as np

DEFAULT_SVD_CUTOFF = 1e-10


def statevector_to_mps(state, n_qubits, max_bond_dim=None, svd_cutoff=DEFAULT_SVD_CUTOFF):
    """
    Decomposes a full statevector into a chain of MPS tensors using
    sequential SVD (the standard, textbook MPS construction method,
    implemented here from the underlying linear algebra).

    svd_cutoff: singular values smaller than `svd_cutoff * max(S)` at
    a given cut are dropped -- this is what actually captures "how
    entangled is this state" as a small bond dimension for
    low-entanglement circuits. Set svd_cutoff=0 to restore the old
    (lossless but non-memory-saving) exact-SVD behavior.

    Returns a list of tensors, one per qubit.
    """
    tensors = []
    remaining = state.reshape(2, -1)
    left_dim = 1

    for i in range(n_qubits - 1):
        mat = remaining.reshape(left_dim * 2, -1)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        if svd_cutoff is not None and svd_cutoff > 0 and len(S) > 0:
            threshold = svd_cutoff * S[0]
            keep = int(np.sum(S > threshold))
            keep = max(keep, 1)
            U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]

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