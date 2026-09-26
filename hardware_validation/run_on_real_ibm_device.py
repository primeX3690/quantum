"""
run_on_real_ibm_device.py
----------------------------
Runs a small circuit on ACTUAL IBM Quantum hardware (free Open Plan,
just needs an API token from https://quantum.cloud.ibm.com -- no
hardware purchase, runs from this laptop) and compares the real
device's measured distribution against:
  (a) my_quantum_core's IDEAL (noiseless) simulator prediction
  (b) my_quantum_core's NOISY prediction, using the real T1/T2/gate
      error numbers pulled by noise_channel/ibm_calibration.py for
      that same backend

This is the "validated on a real quantum computer" claim: not a
simulation of noise, an actual comparison against actual hardware.

SETUP (one-time):
    pip install -r requirements-dev.txt
    export IBM_QUANTUM_API_TOKEN="your token from quantum.cloud.ibm.com"

RUN:
    PYTHONPATH=. python3 hardware_validation/run_on_real_ibm_device.py

If no token is set, this script explains what it WOULD do and exits
cleanly rather than silently faking a result -- there is no offline
substitute for "ran on a real device", unlike the noise model (which
has a legitimate offline cached-calibration fallback).
"""

import os
import sys
import numpy as np

from core_engine.statevector import QuantumState
from noise_channel.decoherence import (
    statevector_to_density_matrix,
    get_probabilities_from_density_matrix,
)
from noise_channel.ibm_calibration import apply_ibm_profile_to_density_matrix, get_calibration

TOKEN_ENV = "IBM_QUANTUM_API_TOKEN"
N_QUBITS = 2
SHOTS = 1024


def build_bell_circuit_qiskit():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def our_ideal_prediction():
    qs = QuantumState(N_QUBITS)
    qs.apply_h(0)
    qs.apply_cnot(0, 1)
    return qs.measure_probabilities_dict()


def our_noisy_prediction(backend_name):
    qs = QuantumState(N_QUBITS)
    qs.apply_h(0)
    qs.apply_cnot(0, 1)
    rho = statevector_to_density_matrix(qs.state)
    noisy_rho, calibration = apply_ibm_profile_to_density_matrix(rho, N_QUBITS, backend_name=backend_name)
    return get_probabilities_from_density_matrix(noisy_rho), calibration


def total_variation_distance(p, q, n_qubits):
    labels = [format(i, f"0{n_qubits}b") for i in range(2 ** n_qubits)]
    return 0.5 * sum(abs(p.get(l, 0.0) - q.get(l, 0.0)) for l in labels)


def counts_to_probs(counts, shots, n_qubits):
    return {k.replace(" ", ""): v / shots for k, v in counts.items()}


def run_on_hardware():
    token = os.environ.get(TOKEN_ENV)
    if not token:
        print("=" * 70)
        print("No IBM_QUANTUM_API_TOKEN set -- explaining what this script does")
        print("instead of running it (no offline substitute for real hardware).")
        print("=" * 70)
        print(f"""
1. Connects to IBM Quantum Platform with QiskitRuntimeService(token=...)
2. Picks the least-busy REAL (non-simulator) backend you have access to
   under the free Open Plan
3. Submits a 2-qubit Bell-state circuit ({SHOTS} shots) via the
   Qiskit Runtime SamplerV2 primitive
4. Pulls that backend's OWN calibration data via
   noise_channel/ibm_calibration.py and computes:
     - my_quantum_core's ideal (noiseless) prediction
     - my_quantum_core's noisy prediction using that backend's real
       T1/T2/gate-error numbers
5. Reports total variation distance between:
     real hardware counts  vs. ideal prediction   (expected: largest)
     real hardware counts  vs. noisy prediction    (expected: smallest --
       this is the actual claim: our noise-calibrated simulator predicts
       real hardware better than assuming an ideal, noiseless device)

To actually run it:
    pip install -r requirements-dev.txt
    export {TOKEN_ENV}="<your token from https://quantum.cloud.ibm.com>"
    PYTHONPATH=. python3 hardware_validation/run_on_real_ibm_device.py
""")
        return None

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    except ImportError:
        print("qiskit-ibm-runtime not installed. Run: pip install -r requirements-dev.txt")
        sys.exit(1)

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.least_busy(operational=True, simulator=False)
    print(f"Selected real backend: {backend.name}")

    from qiskit import transpile
    qc = build_bell_circuit_qiskit()
    transpiled = transpile(qc, backend=backend)

    sampler = Sampler(mode=backend)
    job = sampler.run([transpiled], shots=SHOTS)
    print(f"Job submitted: {job.job_id()}  (waiting for result...)")
    result = job.result()
    counts = result[0].data.c.get_counts()
    hw_probs = counts_to_probs(counts, SHOTS, N_QUBITS)

    ideal_probs = our_ideal_prediction()
    noisy_probs, calibration = our_noisy_prediction(backend.name)

    tvd_ideal = total_variation_distance(hw_probs, ideal_probs, N_QUBITS)
    tvd_noisy = total_variation_distance(hw_probs, noisy_probs, N_QUBITS)

    print("\n" + "=" * 70)
    print(f"REAL HARDWARE RESULT ({backend.name}, {SHOTS} shots)")
    print("=" * 70)
    print(f"Hardware counts (probabilities): {hw_probs}")
    print(f"my_quantum_core IDEAL prediction: {ideal_probs}")
    print(f"my_quantum_core NOISY prediction (this backend's real calibration): {noisy_probs}")
    print(f"\nTotal variation distance, hardware vs IDEAL prediction:  {tvd_ideal:.4f}")
    print(f"Total variation distance, hardware vs NOISY prediction:  {tvd_noisy:.4f}")
    print(f"\n{'NOISY model is closer to real hardware, as expected.' if tvd_noisy < tvd_ideal else 'NOISY model was NOT closer this run -- report both numbers honestly; shot noise at 1024 shots can flip this on a very quiet device.'}")
    print("=" * 70)

    return {
        "backend": backend.name,
        "hw_probs": hw_probs,
        "ideal_probs": ideal_probs,
        "noisy_probs": noisy_probs,
        "tvd_ideal": tvd_ideal,
        "tvd_noisy": tvd_noisy,
    }


if __name__ == "__main__":
    run_on_hardware()
