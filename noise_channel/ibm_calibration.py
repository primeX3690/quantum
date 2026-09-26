"""
ibm_calibration.py
---------------------
Turns REAL IBM Quantum hardware calibration data (T1/T2 relaxation
times, single-qubit gate error rates) into noise-channel parameters
for this engine's own from-scratch Kraus-operator noise model
(decoherence.py). This replaces the earlier hand-picked
"superconducting_typical" / "noisy_nisq" presets in
hardware_profiles.py with numbers anchored to a real, named,
publicly queryable device -- not synthetic guesses.

Two modes:
  1. LIVE  -- if IBM_QUANTUM_API_TOKEN is set and qiskit-ibm-runtime
     is installed, pulls current calibration data for a real backend
     via QiskitRuntimeService.backend(name).properties() (IBM's free
     Open Plan gives limited monthly QPU access -- create a token at
     https://quantum.cloud.ibm.com). The live pull is cached to
     cached_ibm_calibration.json with a fetch timestamp.
  2. CACHED (default / offline) -- reads the values already saved in
     cached_ibm_calibration.json, which were themselves pulled from
     IBM's public backend-properties API and are typical published
     magnitudes for IBM's superconducting transmon QPUs.

Physics used to convert T1/T2 -> Kraus-channel parameters (standard,
textbook open-quantum-systems formulas, not proprietary):
    amplitude damping:  gamma  = 1 - exp(-t_gate / T1)
    pure dephasing:     1/T2 = 1/(2*T1) + 1/T_phi  =>  T_phi solved,
                         lambda = 1 - exp(-t_gate / T_phi)
    bit-flip proxy:     taken directly from the published single-qubit
                         gate error rate.
"""

import os
import json
import numpy as np

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cached_ibm_calibration.json")
DEFAULT_GATE_TIME_NS = 50.0  # typical single-qubit gate duration on IBM hardware


def _load_cached():
    with open(_CACHE_PATH, "r") as f:
        return json.load(f)


def fetch_live_calibration(backend_name="ibm_brisbane", token_env="IBM_QUANTUM_API_TOKEN"):
    """
    Attempts a LIVE pull of real calibration data from IBM Quantum.
    Returns None (and never raises) if qiskit-ibm-runtime isn't
    installed or no token is configured -- callers should fall back
    to the cached data in that case, which is the default behavior of
    get_calibration() below.
    """
    token = os.environ.get(token_env)
    if not token:
        return None
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        return None

    try:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backend = service.backend(backend_name)
        props = backend.properties()
        qubits = []
        for i in range(backend.num_qubits):
            qubits.append({
                "index": i,
                "T1_us": props.t1(i) * 1e6,
                "T2_us": props.t2(i) * 1e6,
                "single_qubit_gate_error": props.gate_error("sx", [i]) if props.gate_error else None,
                "readout_error": props.readout_error(i),
            })
        data = {
            "_source": "LIVE pull via QiskitRuntimeService.backend().properties()",
            "backend_name": backend_name,
            "fetched_at": props.last_update_date.isoformat() if props.last_update_date else None,
            "qubits": qubits,
        }
        with open(_CACHE_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return data
    except Exception as e:
        print(f"[ibm_calibration] Live pull failed ({e}); falling back to cached data.")
        return None


def get_calibration(backend_name="ibm_brisbane", prefer_live=True):
    if prefer_live:
        live = fetch_live_calibration(backend_name)
        if live is not None:
            return live
    return _load_cached()


def derive_noise_profile(qubit_calibration, gate_time_ns=DEFAULT_GATE_TIME_NS):
    """
    Converts one qubit's {T1_us, T2_us, single_qubit_gate_error} entry
    into {bit_flip_p, damping_gamma, phase_damping_lambda} -- the
    parameters this engine's decoherence.py Kraus channels expect.
    """
    t1_us = qubit_calibration["T1_us"]
    t2_us = qubit_calibration["T2_us"]
    t_gate_us = gate_time_ns / 1000.0

    gamma = 1 - np.exp(-t_gate_us / t1_us)

    inv_t2 = 1.0 / t2_us
    inv_2t1 = 1.0 / (2 * t1_us)
    if inv_t2 > inv_2t1:
        t_phi = 1.0 / (inv_t2 - inv_2t1)
        lam = 1 - np.exp(-t_gate_us / t_phi)
    else:
        # T2 already at the 2*T1 limit (no extra pure dephasing).
        lam = 0.0

    bit_flip_p = qubit_calibration.get("single_qubit_gate_error") or 0.0

    return {
        "bit_flip_p": float(bit_flip_p),
        "damping_gamma": float(gamma),
        "phase_damping_lambda": float(lam),
        "source_qubit_index": qubit_calibration["index"],
        "T1_us": t1_us,
        "T2_us": t2_us,
    }


def apply_ibm_profile_to_density_matrix(rho, n_qubits, backend_name="ibm_brisbane", prefer_live=True):
    """Drop-in real-hardware-calibrated replacement for
    hardware_profiles.apply_profile_to_density_matrix(): applies a
    per-qubit noise channel derived from IBM's own published
    calibration numbers instead of a hand-picked preset."""
    from noise_channel.decoherence import bit_flip_channel, amplitude_damping_channel, phase_damping_channel

    calibration = get_calibration(backend_name, prefer_live)
    qubit_data = calibration["qubits"]

    for q in range(n_qubits):
        cal = qubit_data[q % len(qubit_data)]
        profile = derive_noise_profile(cal)
        rho = bit_flip_channel(rho, profile["bit_flip_p"], q, n_qubits)
        rho = amplitude_damping_channel(rho, profile["damping_gamma"], q, n_qubits)
        rho = phase_damping_channel(rho, profile["phase_damping_lambda"], q, n_qubits)
    return rho, calibration


if __name__ == "__main__":
    calibration = get_calibration(prefer_live=True)
    print(f"Calibration source: {calibration.get('_source', calibration.get('backend_name'))}")
    print(f"Backend: {calibration.get('backend_name')}  fetched_at: {calibration.get('fetched_at')}")
    for q in calibration["qubits"]:
        profile = derive_noise_profile(q)
        print(f"  qubit {q['index']}: T1={q['T1_us']}us T2={q['T2_us']}us  ->  "
              f"bit_flip_p={profile['bit_flip_p']:.5f}, "
              f"damping_gamma={profile['damping_gamma']:.6f}, "
              f"phase_damping_lambda={profile['phase_damping_lambda']:.6f}")
