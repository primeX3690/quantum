"""
mps_engine.py
----------------
Matrix Product State (MPS) simulator that applies gates DIRECTLY on the
MPS tensors. The full 2^n statevector is never built, so memory grows as
O(n * chi^2) (chi = bond dimension) instead of O(2^n).

This closes the gap noted in tensor_network.py: that module only
compresses an already-built statevector. Here the state is *born* in MPS
form, so a 50-100 qubit low-entanglement circuit (GHZ, cluster/graph
states, shallow nearest-neighbour circuits) runs comfortably on an
8 GB laptop.

Tensor layout: tensors[i] has shape (chi_left, 2, chi_right); qubit 0 is
site 0 and is the most significant bit, exactly matching QuantumState
(statevector.py), so results are directly comparable.

Built with NumPy only. No quantum SDK.

Honest scope: cost is governed by entanglement. A random deep circuit
drives chi toward 2^(n/2) and MPS loses its advantage (use max_bond_dim
to trade accuracy for memory; `truncation_error` reports what was lost).
"""

import numpy as np
from core_engine.custom_gates import (
    gate_H, gate_X, gate_Y, gate_Z, gate_S, gate_S_dag, gate_T,
    gate_RX, gate_RY, gate_RZ, gate_CNOT, gate_CZ, gate_SWAP, gate_CPHASE,
    gate_T_dag, gate_SX, gate_P, gate_U3, gate_ISWAP,
)

DEFAULT_CUTOFF = 1e-12


class MPSState:
    def __init__(self, n_qubits, max_bond_dim=None, svd_cutoff=DEFAULT_CUTOFF):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = n_qubits
        self.max_bond_dim = max_bond_dim
        self.svd_cutoff = svd_cutoff
        self.truncation_error = 0.0   # accumulated discarded weight (sum of s^2)
        self._fidelity_estimate = 1.0  # product of per-gate kept weights
        # Orthogonality centre: sites < center are left-orthonormal, sites >
        # center are right-orthonormal. Keeping this gauge makes every local
        # SVD a true Schmidt decomposition, so truncation is optimal and
        # `truncation_error` / the norm are exact, not approximate.
        self.center = 0
        self.tensors = []
        for _ in range(n_qubits):
            t = np.zeros((1, 2, 1), dtype=complex)
            t[0, 0, 0] = 1.0
            self.tensors.append(t)

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------
    def apply_1q(self, gate, q):
        self._check(q)
        self.tensors[q] = np.einsum('ps,lsr->lpr', gate, self.tensors[q])

    def _move_center(self, target):
        """Shift the orthogonality centre with QR sweeps (exact, O(chi^3))."""
        while self.center < target:
            c = self.center
            l, _, r = self.tensors[c].shape
            Q, R = np.linalg.qr(self.tensors[c].reshape(l * 2, r))
            self.tensors[c] = Q.reshape(l, 2, Q.shape[1])
            self.tensors[c + 1] = np.tensordot(R, self.tensors[c + 1], axes=(1, 0))
            self.center = c + 1
        while self.center > target:
            c = self.center
            l, _, r = self.tensors[c].shape
            Q, R = np.linalg.qr(self.tensors[c].reshape(l, 2 * r).T)
            self.tensors[c] = Q.T.reshape(Q.shape[1], 2, r)
            self.tensors[c - 1] = np.tensordot(self.tensors[c - 1], R.T, axes=(2, 0))
            self.center = c - 1

    def _apply_adjacent(self, gate4, q):
        """gate4: 4x4 acting on (q, q+1) with q as the more significant bit."""
        self._move_center(q)
        A, B = self.tensors[q], self.tensors[q + 1]
        l, r = A.shape[0], B.shape[2]
        theta = np.tensordot(A, B, axes=(2, 0))               # (l,s,t,r)
        G = gate4.reshape(2, 2, 2, 2)                         # (s',t',s,t)
        theta = np.einsum('abst,lstr->labr', G, theta)
        U, S, Vh = np.linalg.svd(theta.reshape(l * 2, 2 * r), full_matrices=False)

        keep = int(np.sum(S > self.svd_cutoff * max(S[0], 1e-300)))
        keep = max(keep, 1)
        if self.max_bond_dim is not None:
            keep = min(keep, self.max_bond_dim)
        total = float(np.sum(S ** 2))
        kept = float(np.sum(S[:keep] ** 2))
        if total > 0:
            self.truncation_error += (total - kept) / total
            self._fidelity_estimate *= kept / total
        S = S[:keep]
        if kept > 0:
            S = S * np.sqrt(total / kept)   # renormalise so the state stays unit-norm
        self.tensors[q] = U[:, :keep].reshape(l, 2, keep)
        self.tensors[q + 1] = (np.diag(S) @ Vh[:keep, :]).reshape(keep, 2, r)
        self.center = q + 1

    def _swap_adjacent(self, q):
        self._apply_adjacent(gate_SWAP(), q)

    def apply_2q(self, gate4, q0, q1):
        """Any 4x4 gate on ANY pair. Non-adjacent pairs are routed with a
        SWAP network (exact), applied, and routed back."""
        self._check(q0)
        self._check(q1)
        if q0 == q1:
            raise ValueError("two-qubit gate needs two distinct qubits")
        swap = gate_SWAP()
        if q0 > q1:                                   # re-orient gate so lower index is first
            gate4 = swap @ gate4 @ swap
            q0, q1 = q1, q0
        # move q1 down next to q0
        for k in range(q1 - 1, q0, -1):
            self._swap_adjacent(k)
        self._apply_adjacent(gate4, q0)
        for k in range(q0 + 1, q1):
            self._swap_adjacent(k)

    # Convenience API (mirrors QuantumState)
    def h(self, q): self.apply_1q(gate_H(), q)
    def x(self, q): self.apply_1q(gate_X(), q)
    def y(self, q): self.apply_1q(gate_Y(), q)
    def z(self, q): self.apply_1q(gate_Z(), q)
    def s(self, q): self.apply_1q(gate_S(), q)
    def sdg(self, q): self.apply_1q(gate_S_dag(), q)
    def t(self, q): self.apply_1q(gate_T(), q)
    def tdg(self, q): self.apply_1q(gate_T_dag(), q)
    def sx(self, q): self.apply_1q(gate_SX(), q)
    def p(self, q, th): self.apply_1q(gate_P(th), q)
    def u3(self, q, th, ph, lam): self.apply_1q(gate_U3(th, ph, lam), q)
    def iswap(self, a, b): self.apply_2q(gate_ISWAP(), a, b)
    def rx(self, q, th): self.apply_1q(gate_RX(th), q)
    def ry(self, q, th): self.apply_1q(gate_RY(th), q)
    def rz(self, q, th): self.apply_1q(gate_RZ(th), q)
    def cnot(self, c, t): self.apply_2q(gate_CNOT(), c, t)
    def cz(self, a, b): self.apply_2q(gate_CZ(), a, b)
    def swap(self, a, b): self.apply_2q(gate_SWAP(), a, b)
    def cphase(self, a, b, th): self.apply_2q(gate_CPHASE(th), a, b)

    # ------------------------------------------------------------------
    # Queries (none of these build the 2^n vector, except to_statevector)
    # ------------------------------------------------------------------
    def bond_dims(self):
        return [t.shape[2] for t in self.tensors[:-1]]

    def max_bond(self):
        return max(self.bond_dims()) if self.n_qubits > 1 else 1

    def estimated_fidelity(self):
        """Standard MPS accuracy estimate: product over all truncating gates
        of (kept weight / total weight). ~1.0 means (almost) exact; values
        near 0 mean the bond cap is destroying the state. (Also see
        `truncation_error`, the raw summed discarded weight.)"""
        return self._fidelity_estimate

    def memory_bytes(self):
        return sum(t.nbytes for t in self.tensors)

    def norm(self):
        env = np.ones((1, 1), dtype=complex)
        for A in self.tensors:
            env = np.einsum('ab,asc,bsd->cd', env, A, A.conj())
        return float(np.sqrt(abs(env[0, 0])))

    def amplitude(self, bitstring):
        """<bitstring|psi> without building the full vector."""
        if len(bitstring) != self.n_qubits:
            raise ValueError("bitstring length must equal n_qubits")
        v = np.ones((1,), dtype=complex)
        for A, b in zip(self.tensors, bitstring):
            v = v @ A[:, int(b), :]
        return complex(v[0])

    def expectation_1q(self, op, q):
        """<psi| op_q |psi> for a 2x2 operator on qubit q. O(n * chi^3)."""
        self._check(q)
        env = np.ones((1, 1), dtype=complex)
        for i, A in enumerate(self.tensors):
            B = np.einsum('ps,lsr->lpr', op, A) if i == q else A
            env = np.einsum('ab,asc,bsd->cd', env, B, A.conj())
        return complex(env[0, 0])

    def _canonical_copy(self):
        """Right-canonical copy: sites 1..n-1 have orthonormal rows, so
        sequential sampling probabilities are exact."""
        T = [t.copy() for t in self.tensors]
        for i in range(self.n_qubits - 1, 0, -1):
            l, _, r = T[i].shape
            M = T[i].reshape(l, 2 * r)
            Q, R = np.linalg.qr(M.T)                # M.T = Q R  ->  M = R.T Q.T
            k = Q.shape[1]
            T[i] = Q.T.reshape(k, 2, r)
            T[i - 1] = np.tensordot(T[i - 1], R.T, axes=(2, 0))
        return T

    def sample(self, shots=1024, seed=None):
        """Draw measurement bitstrings directly from the MPS
        (O(shots * n * chi^2)); never materialises 2^n amplitudes."""
        rng = np.random.default_rng(seed)
        T = self._canonical_copy()
        counts = {}
        for _ in range(shots):
            v = np.ones((1,), dtype=complex)
            bits = []
            for A in T:
                w0, w1 = v @ A[:, 0, :], v @ A[:, 1, :]
                p0, p1 = float(np.vdot(w0, w0).real), float(np.vdot(w1, w1).real)
                tot = p0 + p1
                b = 0 if rng.random() < p0 / tot else 1
                v = (w0 if b == 0 else w1)
                v = v / np.linalg.norm(v)
                bits.append(str(b))
            key = ''.join(bits)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def entanglement_entropy(self, bond):
        """Von Neumann entropy (bits) across the cut between qubit `bond`
        and `bond+1`."""
        if not 0 <= bond < self.n_qubits - 1:
            raise ValueError("bond must be in [0, n_qubits-2]")
        T = [t.copy() for t in self.tensors]
        for i in range(bond + 1):                    # left-canonicalise 0..bond
            l, _, r = T[i].shape
            Q, R = np.linalg.qr(T[i].reshape(l * 2, r))
            T[i] = Q.reshape(l, 2, Q.shape[1])
            T[i + 1] = np.tensordot(R, T[i + 1], axes=(1, 0))
        for i in range(self.n_qubits - 1, bond + 1, -1):   # right-canonicalise
            l, _, r = T[i].shape
            Q, R = np.linalg.qr(T[i].reshape(l, 2 * r).T)
            T[i] = Q.T.reshape(Q.shape[1], 2, r)
            T[i - 1] = np.tensordot(T[i - 1], R.T, axes=(2, 0))
        C = T[bond + 1]
        s = np.linalg.svd(C.reshape(C.shape[0], -1), compute_uv=False)
        p = s ** 2
        p = p / p.sum()
        p = p[p > 1e-15]
        return float(-np.sum(p * np.log2(p)))

    def to_statevector(self, max_qubits=24):
        """Contract to a dense vector. Only for verification on small n."""
        if self.n_qubits > max_qubits:
            raise MemoryError(
                f"Refusing to build a 2^{self.n_qubits} vector; "
                f"use sample()/amplitude()/expectation_1q() instead.")
        psi = self.tensors[0]
        for A in self.tensors[1:]:
            psi = np.tensordot(psi, A, axes=(psi.ndim - 1, 0))
        return psi.reshape(-1)

    def _check(self, q):
        if not 0 <= q < self.n_qubits:
            raise IndexError(f"qubit {q} out of range for {self.n_qubits} qubits")

    def __repr__(self):
        return (f"MPSState({self.n_qubits} qubits, max_bond={self.max_bond()}, "
                f"memory={self.memory_bytes()/1024:.1f} KiB, "
                f"est_fidelity={self._fidelity_estimate:.4f})")


if __name__ == "__main__":
    import time
    for n in [20, 50, 100]:
        t0 = time.time()
        m = MPSState(n)
        m.h(0)
        for i in range(n - 1):
            m.cnot(i, i + 1)
        dt = time.time() - t0
        counts = m.sample(shots=200, seed=1)
        dense_gb = (2 ** n) * 16 / 1e9
        print(f"GHZ n={n:3d}: {dt:.2f}s  MPS mem={m.memory_bytes()/1024:.1f} KiB "
              f"(dense would need {dense_gb:.3g} GB)  samples={dict(list(counts.items())[:2])}")
