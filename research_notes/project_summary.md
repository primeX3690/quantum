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

## v2.0 additions (all six roadmap gaps closed)
1. **MPS engine that applies gates on tensors** (`core_engine/mps_engine.py`):
   no dense statevector ever; canonical-gauge truncation, sampling,
   expectation values, entanglement entropy. 100-qubit GHZ in ~12 KiB.
2. **Real syndrome error correction** (`algorithms/error_correction.py`):
   ancilla parity checks, only ancillas measured (new partial projective
   measurement in `QuantumState`), bit- and phase-flip codes, Monte Carlo
   logical error rate matches 3p^2 - 2p^3.
3. **Scalable Shor** (`algorithms/shor.py`): permutation modexp + axis QFT,
   factors up to N=255 on a laptop; integrated into `run_my_core.py`.
4. **General circuit builder / CLI / API** (`core_engine/circuit.py`,
   `cli.py`, `api/main.py`): JSON + text formats, statevector / MPS / noisy
   backends, validated + size-capped HTTP endpoints.
5. **Tests + packaging** (`tests/`, `run_tests.py`, `pyproject.toml`):
   43 tests; dependency-free runner; pip-installable.
6. **Pulse-level physics -> noise model** (`core_engine/pulse_control.py`,
   `noise_channel/pulse_noise.py`): drive error, detuning, leaked-drive
   crosstalk, ZZ coupling, T1/T2, readout error, on a density matrix.

Bugs found and fixed while building these (kept for the record):
- MPS truncation drifted the norm because local SVDs were not taken in
  canonical gauge; fixed by tracking the orthogonality centre.
- QEC Monte Carlo initially reported 0 failures: the failure threshold
  (fidelity < 0.5) was wrong for a non-equal superposition; now < 0.99.
- `QuantumCircuit.from_text` raised TypeError instead of ValueError on
  malformed input.

Known limits: MPS cost grows with entanglement; QEC handles one error
type at a time with noiseless ancillas; pulse parameters are illustrative,
not calibrated to a specific chip; noisy backend is density-matrix (<=9
qubits); Shor is a classical simulation (exponential in bit-length).

## Round-2 additions
- Grover rewritten to act on the statevector (no dense matrices), multi-target, up to 22-24 qubits; API cap raised to 20.
- `measure_shots` no longer re-seeds NumPy's global RNG; supports marginal counts.
- Extra gates: tdg, sx, p, u3, iswap, Toffoli (ccx, decomposed); wired through statevector, MPS and pulse-noise backends.

