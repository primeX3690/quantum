# my_quantum_core — Project Summary

A quantum circuit simulator built entirely from first principles using
NumPy — no Qiskit, no Cirq, no external quantum SDK.

## What it implements
- Statevector simulation engine with custom gate matrices (now O(2^n)
  tensor-reshape application, not O(4^n) Kronecker matrices)
- Bell state & GHZ state entanglement (verified against theory)
- Grover's search algorithm (2 & 3 qubit, matches theoretical formula)
- Deutsch-Jozsa, Quantum Fourier Transform, and a toy VQE
- Noise/decoherence modeling via density matrices (bit-flip, amplitude
  damping, phase damping), including a real-hardware-calibrated
  variant driven by actual IBM Quantum T1/T2/gate-error numbers
- Matrix Product State (MPS) compression for low-entanglement states,
  with automatic singular-value truncation

## External validation (added on top of the from-scratch core)
- Statevectors cross-checked against Qiskit for Bell/GHZ/Grover/
  Deutsch-Jozsa/QFT (fidelity 1.0 on 4 of 5, machine precision)
- Noise model driven by real IBM Quantum backend calibration data
  (live API or cached fallback), not hand-picked presets
- MPS memory-savings claim anchored against a published external
  formula (arXiv:2602.10830) run on an actual circuit -- this also
  surfaced and fixed a real bug where the SVD path wasn't truncating
  near-zero singular values by default, so it never saved memory
  until fixed
- A real-hardware run script (`hardware_validation/`) that submits to
  actual IBM Quantum devices via the free Open Plan and compares
  against this engine's calibrated prediction

See README.md's "External Validation & Real-Hardware Anchoring"
section for the full breakdown and how to run each piece.

## Why built from scratch
To gain a first-principles understanding of quantum simulation at the
matrix-mathematics level, independent of existing frameworks. The
validation layer above exists to prove that first-principles
understanding matches reality, not just internal self-consistency.

## Built by
[Raghavendra pratap singh] — self-taught, Uttar Pradesh, India. No formal CS degree.

