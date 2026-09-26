# 🌌 Quantum Computing Simulation Engine from Scratch

A lightweight, high-performance Quantum Computing Simulator built entirely in Python **from scratch, with zero external quantum SDKs** (No Qiskit, No Cirq). This project implements and optimizes the foundational mathematics and physics of quantum mechanics, statevector evolution, noisy channels, and error-correcting codes.

---

## 🚀 Core Features

- **Built From Scratch:** Pure linear algebra implementation of quantum states and operators without high-level wrappers.
- **Statevector Simulator:** Manages complex probability amplitudes and tracks exact quantum state evolution.
- **Quantum Noise & Decoherence:** Advanced physical error injection simulating phase-flip, bit-flip, and environmental noise channels.
- **Quantum Error Correction (QEC):** Active implementation of stabilizer/syndrome measurement logic to automatically detect and correct simulated hardware faults.
- **ASCII Circuit Visualization:** Built-in terminal renderer to visualize quantum circuits (Hadamard, CNOT, etc.) directly in the console.

---

## 📂 Project Architecture

```text
QUAN.../
├── algorithms/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── error_correction.py   # 3-Qubit Bit-Flip Error Correction logic
│   └── grover_search.py      # Grover's Search Algorithm implementation
├── config/
│   └── virtual_hardware.json # Dynamic virtual hardware & qubit constraints
├── core_engine/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── custom_gates.py       # Matrix definitions for standard quantum gates
│   ├── kronecker_solver.py   # High-efficiency multi-qubit tensor products
│   └── statevector.py        # Core state manipulation engine
├── noise_channel/
│   ├── __pycache__/
│   └── __init__.py           # Environmental noise modeling tools
├── research_notes/
│   └── project_summary.md    # Theoretical math formulas & notes
├── utils/
├── .gitignore
├── LICENSE
├── requirements.txt
└── run_my_core.py            # Main execution script running benchmarks
```

---

## 📊 Performance Benchmarks & Demo Outputs

### 1. Quantum Entanglement (Bell State)
Simulating a standard Bell State (\(\vert{}\Phi^+\rangle\)) over **1024 shots** demonstrates a near-perfect theoretical probabilistic distribution:
```text
1024 shots on Bell state: {'00': 524, '11': 500}
```

### 2. Qubit Scaling Benchmark
Testing the computational efficiency of the Kronecker Product solver by scaling statevectors from 4 to 10 qubits under strict RAM controls:
- **4 Qubits:** 0.0012s | RAM Delta: 0.00MB
- **6 Qubits:** 0.0060s | RAM Delta: -0.30MB
- **8 Qubits:** 0.0420s | RAM Delta: 0.20MB
- **10 Qubits:** 0.5343s | Total RAM: 34.44MB

### 3. 3-Qubit Bit-Flip Error Correction
Injecting a hardware fault on Qubit 1 proves the active syndrome measurement can safely isolate and auto-correct errors:
```text
State after error injected on qubit 1: {'101': 1.0}
Recovered logical bit: 1 (raw measurement: 101) -> ERROR CORRECTED
```

### 4. Terminal ASCII Circuit Rendering
```text
q0: -[H]-------
q1: --------O--
```

---

## ⚙️ Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com
   cd quantum-simulation-engine
   ```

2. Run the full simulation and benchmark suite:
   ```bash
   python run_my_core.py
   ```

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

## 🛠️ Tech Stack & Concepts Covered
- **Language:** Python 3
- **Mathematics:** Advanced Linear Algebra, Complex Numbers, Kronecker/Tensor Products, Probability Matrices.
- **Quantum Mechanics:** Quantum Superposition, Entanglement, Statevectors, Mixed States/Decoherence.

---
*Developed with 💻 as a deep-tech engineering sandbox.*

