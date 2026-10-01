# 🌌 Quantum Computing Simulation Engine from Scratch

A quantum computing simulator built in Python **from scratch: no Qiskit, no Cirq, no quantum SDK, no cloud API** in the core engine. Only NumPy for linear algebra. It runs on a modest laptop (developed on a Ryzen 3 / 8 GB machine).

The core engine has:
- a dense **statevector** simulator (O(2^n) tensor-reshape gates, ~22 qubits),
- a **Matrix Product State (MPS)** simulator that applies gates *directly on tensors* (50-100+ qubits for low-entanglement circuits),
- **density-matrix noise** driven by **pulse-level physics** (drive error, detuning, crosstalk, T1/T2),
- **real ancilla-based syndrome error correction**,
- a **general circuit builder** with JSON/text formats, a CLI and an HTTP API,
- Grover, Deutsch-Jozsa, QFT, toy VQE and a scalable **Shor**.

---

## 🚀 Quick start

```bash
pip install numpy psutil scipy          # scipy is used only by the VQE optimiser
python run_my_core.py                   # all 17 demos
python run_tests.py                     # 49 tests, no pytest needed (pytest -q also works)
python cli.py run examples/bell.txt --draw
python cli.py shor 91
python cli.py qec --p 0.05
python cli.py mps-demo --qubits 100
python cli.py noisy examples/ghz3.txt --profile noisy_nisq
uvicorn api.main:app --reload           # needs: pip install fastapi uvicorn ; docs at /docs
```

Optional install as a package: `pip install -e .` (see `pyproject.toml`).

---

## 🧩 What each part does (and its honest limits)

### 1. MPS engine: `core_engine/mps_engine.py`
Gates (any 1-qubit gate, any 2-qubit gate on **any** pair via an exact SWAP network) are applied straight onto the MPS tensors. The 2^n vector is never built. The state is kept in canonical gauge (tracked orthogonality centre), so truncation with `max_bond_dim` is optimal and `estimated_fidelity()` tracks the true fidelity closely (verified against the dense state in `tests/test_mps.py`). Also provides `sample()` (measurement without a dense vector), `amplitude()`, `expectation_1q()` and `entanglement_entropy()`.
*Limit:* cost is set by entanglement. GHZ / cluster / shallow nearest-neighbour circuits are cheap; deep random circuits blow up the bond dimension and MPS loses its advantage (that is physics, not a bug).

### 2. Syndrome-based error correction: `algorithms/error_correction.py`
3 data qubits + 2 ancillas. Ancillas record the parities q0^q1 and q1^q2 via CNOTs and **only the ancillas are measured** (new `QuantumState.measure_qubit()` does true partial projective measurement). The syndrome names the flipped qubit, a classical feed-forward X fixes it, and a superposition `cos(t/2)|0> + sin(t/2)|1>` survives with fidelity 1. Both **bit-flip** and **phase-flip** codes. A Monte Carlo of the logical error rate matches the theory `3p^2 - 2p^3`. (The old majority-vote functions are kept for backward compatibility.)
*Limit:* it corrects one kind of error at a time (X *or* Z), not both at once (that needs Shor's 9-qubit or a surface code), and the ancilla operations themselves are noiseless here.

### 3. Shor's algorithm: `algorithms/shor.py`
The dense modular-exponentiation matrix is gone. The register is a (count x work) array, modexp is an exact index permutation, the QFT runs along one axis. It factors **N = 15, 21, 35, 77, 91, 143, 255** on a laptop (peak RAM ~0.55 GB at the top end) and tries other bases automatically when a base gives an odd/trivial period. Verified equal to the old dense-matrix result (diff ~1e-30).
*Limit:* still a classical simulation, so cost grows exponentially with the bit-length of N. A memory guard refuses sizes that will not fit.

### 4. Circuit builder, CLI and API: `core_engine/circuit.py`, `cli.py`, `api/main.py`
```python
qc = QuantumCircuit(3).h(0).cnot(0, 1).cnot(1, 2)
qc.run(shots=1000)                              # auto: statevector or MPS
qc.run(backend="mps", max_bond_dim=32)
qc.run(noise=PulseNoiseModel.from_profile("superconducting"))
QuantumCircuit.from_text("qubits 3\nh 0\ncx 0 2")   # or .from_json(...)
print(qc.draw())
```
The API exposes `/run/circuit`, `/run/shor`, `/run/qec`, `/run/bell`, `/run/grover`, `/gates`, with input validation and size caps.
*Note:* the API handler logic was exercised with stubs while building; run `uvicorn api.main:app` once on your machine with fastapi installed to check the live server.

### 5. Grover, measurement and gate set (round-2 fixes)
- **Grover** (`algorithms/grover_search.py`) no longer builds dense `2^n x 2^n` matrices: the oracle flips marked amplitudes and diffusion is `2*mean - psi`, directly on the vector (O(2^n) memory). Equal to the old dense version to ~1e-15, supports several marked items, and runs 18 qubits in ~0.1 s, 20 in ~0.8 s, 22 in ~7 s (sandbox timings; time grows ~2^(1.5 n) because the iteration count grows like sqrt(2^n)). The API cap went from 12 to 20 qubits.
- **`measure_shots`** now uses a local random generator (the old `np.random.seed()` silently re-seeded NumPy globally), is vectorised, and can return marginal counts for a subset of qubits.
- **More gates:** `tdg`, `sx` (sqrt-X), `p` (phase), `u3` (any 1-qubit unitary), `iswap`, and **Toffoli** `ccx` (expanded into 15 one/two-qubit gates so it works on the statevector, MPS and noisy backends alike; checked against the 8x8 matrix on all 8 inputs).

### 6. Tests and packaging
`tests/` (49 tests) covers the engine, every algorithm, MPS vs dense on random circuits, QEC statistics, circuit validation, pulse physics and noise. `run_tests.py` runs them with no dependencies; `pytest -q` works too. `pyproject.toml` makes it pip-installable and exposes a `quantum-core` command.

### 7. Pulse-level physics wired into the noise model: `core_engine/pulse_control.py`, `noise_channel/pulse_noise.py`
A Gaussian pi-pulse is integrated (RK4) with amplitude error and detuning; a 2-qubit simulation adds leaked-drive crosstalk and always-on ZZ coupling. The resulting **error unitaries** are turned into gate noise (scaled to each gate's rotation angle), together with T1/T2 damping over gate time and readout error, and applied to a density matrix. Profiles: `ideal`, `superconducting`, `noisy_nisq`. Verified: the ideal profile reproduces the ideal simulation, T1/T2 matches the Kraus reference channels, more noise gives lower fidelity, and crosstalk is measurable on spectator qubits.
*Limit:* illustrative physically-motivated parameters, **not** a calibrated model of a specific chip; linear nearest-neighbour coupling; density matrices cap this backend at 9 qubits.

---

## 📂 Project architecture

```text
├── core_engine/
│   ├── custom_gates.py       # gate matrices
│   ├── kronecker_solver.py   # O(2^n) tensor-reshape gate application
│   ├── statevector.py        # dense simulator + partial measurement
│   ├── mps_engine.py         # MPS simulator (gates on tensors)      [new]
│   ├── tensor_network.py     # statevector <-> MPS compression
│   ├── circuit.py            # circuit builder, JSON/text, backends  [new]
│   ├── pulse_control.py      # pulse physics, crosstalk, fidelity    [extended]
│   └── measurement_channel.py
├── algorithms/               # grover, deutsch_jozsa, qft, vqe_toy, shor [scalable], error_correction [syndrome QEC]
├── noise_channel/            # decoherence, hardware_profiles, ibm_calibration, pulse_noise [new]
├── api/main.py               # FastAPI service                        [extended]
├── cli.py                    # command line                           [new]
├── tests/  run_tests.py      # 49 tests                               [new]
├── examples/                 # sample circuit files                   [new]
├── validation/  benchmarks/  hardware_validation/   # optional dev-only Qiskit / IBM anchoring
├── pyproject.toml  requirements*.txt  run_my_core.py
```

---

## 📊 Measured results (from the build sandbox, not from a Ryzen 3: run `python run_my_core.py` for your own numbers)

| What | Result |
|---|---|
| GHZ on 100 qubits via MPS | ~0.01 s, 12.4 KiB (dense vector would need ~2e22 GB) |
| MPS vs dense on random 6-qubit circuits | overlap 1.0 to 1e-9, incl. non-adjacent gates |
| Truncated MPS (bond cap 4, 10 qubits) | estimated fidelity 0.834 vs true 0.834 |
| Syndrome QEC, single errors | logical fidelity 1.0 for all positions, both codes |
| Logical error rate at p=0.05 / 0.10 / 0.20 | 0.0063 / 0.031 / 0.100 (theory 0.0073 / 0.028 / 0.104) |
| Shor N=143 | (11, 13) in ~3.6 s, ~545 MB peak |
| Noisy 3-qubit GHZ fidelity | ideal 1.000, superconducting 0.989, noisy_nisq 0.926 |
| Noisy density backend, 9 qubits | ~2.3 s, ~60 MB |

---

## 🧭 Roadmap (not done yet)
- Simultaneous X+Z correction (Shor 9-qubit / surface code) and noisy ancillas
- Noise inside the MPS backend (quantum trajectories) to combine noise with 50+ qubits
- Replace the scipy optimiser in `vqe_toy.py` with an in-house one (parameter-shift + gradient descent) to make the core NumPy-only
- Calibrating the pulse model to a specific device's published data
- Circuit optimisation / gate decomposition (transpiler)

---

## 🔬 External Validation & Real-Hardware Anchoring

Everything above is self-contained and SDK-free. The additions below exist
for one purpose: **anchor every headline claim to something outside this
repo** -- an external library, a published formula, or an actual quantum
computer -- while keeping the production engine untouched (still zero
Qiskit/Cirq dependency; these live in separate, clearly-marked folders and
`requirements-dev.txt`).

| # | Claim being anchored | How | Where |
|---|---|---|---|
| 1 | "The statevector math is correct" | Build the identical circuit in Qiskit and diff statevectors directly (fidelity + max amplitude error) | `validation/qiskit_cross_check.py` |
| 2 | "Algorithm coverage goes beyond Grover" | Deutsch-Jozsa (1992), QFT (1994, checked against the closed-form DFT matrix to 1e-15), toy VQE (2014, checked against exact diagonalization to 1e-16) | `algorithms/deutsch_jozsa.py`, `algorithms/qft.py`, `algorithms/vqe_toy.py` |
| 3 | "The noise model resembles real hardware" | Converts real IBM Quantum T1/T2/gate-error calibration numbers (live API pull or cached fallback) into this engine's own Kraus-operator noise channels via standard open-quantum-systems formulas | `noise_channel/ibm_calibration.py` |
| 4 | "MPS gives real memory savings" | Anchored against a published external formula (Chatelain et al., arXiv:2602.10830, `M = L·d·χ²`) run on an actual circuit through the general SVD pipeline -- **this also caught and fixed a real bug**: `statevector_to_mps()` wasn't truncating near-zero singular values by default, so it never actually saved memory until now | `benchmarks/external_reference_comparison.py`, fix in `core_engine/tensor_network.py` |
| 5 | "Validated on a real quantum computer" | Submits a circuit to actual IBM Quantum hardware (free Open Plan, just needs an API token) and compares the measured distribution against this engine's noise-calibrated prediction | `hardware_validation/run_on_real_ibm_device.py` |

Run them:
```bash
pip install -r requirements-dev.txt
PYTHONPATH=. python3 validation/qiskit_cross_check.py
PYTHONPATH=. python3 benchmarks/external_reference_comparison.py
PYTHONPATH=. python3 noise_channel/ibm_calibration.py
export IBM_QUANTUM_API_TOKEN="..."   # from https://quantum.cloud.ibm.com
PYTHONPATH=. python3 hardware_validation/run_on_real_ibm_device.py
```

**Honest note on the engine itself, found while building the above:** the
original gate-application path built full `2^n x 2^n` matrices via Kronecker
products, which ran out of memory above ~14 qubits on 8GB RAM. Single- and
two-qubit gates now apply directly to the statevector via tensor reshaping
(`O(2^n)` instead of `O(4^n)`), which is what makes the qubit counts in the
MPS benchmark above actually reachable rather than theoretical.

---

---

## 🛠️ Tech Stack & Concepts Covered
- **Language:** Python 3.9+, NumPy (core), psutil (benchmarks), scipy (VQE optimiser only), FastAPI (optional API)
- **Mathematics:** linear algebra, tensor networks (MPS/SVD/QR canonical forms), Kraus channels, Runge-Kutta integration of the Schrödinger equation
- **Quantum:** superposition, entanglement, projective measurement, syndrome error correction, decoherence (T1/T2), pulse control, period finding

---
*Developed with 💻 as a deep-tech engineering sandbox.*
