"""
test_mps.py
-------------
Verifies MPS correctness and demonstrates memory scaling advantage
for structured, low-entanglement states like GHZ.
"""
import numpy as np
from core_engine.statevector import QuantumState
from core_engine.tensor_network import (
    statevector_to_mps, mps_to_statevector,
    mps_memory_bytes, statevector_memory_bytes
)


def test_correctness():
    print("=" * 55)
    print("TEST 1: MPS CORRECTNESS (3-qubit GHZ state)")
    print("=" * 55)
    qs = QuantumState(3)
    qs.apply_h(0)
    qs.apply_cnot(0, 1)
    qs.apply_cnot(1, 2)
    original = qs.state

    tensors = statevector_to_mps(original, 3)
    reconstructed = mps_to_statevector(tensors, 3)

    print("Original:     ", np.round(original, 4))
    print("Reconstructed:", np.round(reconstructed, 4))
    print("Exact match:", np.allclose(original, reconstructed))
    print("Bond dimensions:", [t.shape for t in tensors])
    print()


def build_ghz_mps_directly(n):
    """
    Constructs GHZ-state MPS tensors analytically -- this represents
    what a full gate-on-MPS engine would produce without ever building
    the exponential-size statevector.
    """
    tensors = []
    t0 = np.zeros((1, 2, 2), dtype=complex)
    t0[0, 0, 0] = 1
    t0[0, 1, 1] = 1
    tensors.append(t0)
    for i in range(1, n - 1):
        t = np.zeros((2, 2, 2), dtype=complex)
        t[0, 0, 0] = 1
        t[1, 1, 1] = 1
        tensors.append(t)
    tlast = np.zeros((2, 2, 1), dtype=complex)
    tlast[0, 0, 0] = 1
    tlast[1, 1, 0] = 1
    tensors.append(tlast)
    return tensors


def test_memory_scaling():
    print("=" * 55)
    print("TEST 2: MEMORY SCALING (GHZ-like state, direct MPS)")
    print("=" * 55)
    print(f"{'Qubits':<8}{'Statevector':<20}{'MPS':<15}{'Savings':<10}")
    for n in [10, 20, 30, 50]:
        tensors = build_ghz_mps_directly(n)
        sv_mem = statevector_memory_bytes(n)
        mps_mem = mps_memory_bytes(tensors)
        if sv_mem < 1e9:
            sv_str = f"{sv_mem/1e6:,.1f} MB"
        elif sv_mem < 1e15:
            sv_str = f"{sv_mem/1e9:,.1f} GB"
        else:
            sv_str = f"{sv_mem/1e15:,.0f} PB (impossible)"
        print(f"{n:<8}{sv_str:<20}{mps_mem:<15,}{sv_mem/mps_mem:,.0f}x")
    print()
    print("NOTE: This demonstrates the MPS *representation's* efficiency")
    print("for structured/low-entanglement states. A full gate-application")
    print("engine (applying gates directly to MPS tensors, never building")
    print("the full statevector) is the next milestone to make this general.")


if __name__ == "__main__":
    test_correctness()
    test_memory_scaling()