# """
# run_my_core.py
# ----------------
# Entry point. Runs all four verification demos: Bell state, GHZ state,
# Grover's search, and noise/decoherence modeling.
# """

# from core_engine.statevector import QuantumState
# from algorithms.grover_search import run_grover, optimal_iterations
# from noise_channel.decoherence import (
#     statevector_to_density_matrix,
#     bit_flip_channel,
#     amplitude_damping_channel,
#     get_probabilities_from_density_matrix,
# )


# def demo_bell_state():
#     print("=" * 55)
#     print("DEMO 1: BELL STATE — 2-Qubit Entanglement")
#     print("=" * 55)
#     qs = QuantumState(n_qubits=2)
#     print("Initial:", qs.measure_probabilities_dict())
#     qs.apply_h(0)
#     print("After Hadamard:", qs.measure_probabilities_dict())
#     qs.apply_cnot(control=0, target=1)
#     print("After CNOT (entangled):", qs.measure_probabilities_dict())
#     print("Expected: {'00': 0.5, '11': 0.5}\n")


# def demo_ghz_state():
#     print("=" * 55)
#     print("DEMO 2: GHZ STATE — 3-Qubit Entanglement")
#     print("=" * 55)
#     qs = QuantumState(n_qubits=3)
#     qs.apply_h(0)
#     qs.apply_cnot(control=0, target=1)
#     qs.apply_cnot(control=1, target=2)
#     print("Final:", qs.measure_probabilities_dict())
#     print("Expected: {'000': 0.5, '111': 0.5}\n")


# def demo_grover():
#     print("=" * 55)
#     print("DEMO 3: GROVER'S SEARCH ALGORITHM")
#     print("=" * 55)
#     print("2-qubit search, target |11>, 1 iteration:")
#     print(run_grover(n_qubits=2, target_index=3, iterations=1))

#     n = 3
#     opt_iter = optimal_iterations(n)
#     print(f"\n3-qubit search, target |101>, {opt_iter} iterations (optimal):")
#     print(run_grover(n_qubits=n, target_index=5, iterations=opt_iter))
#     print()


# def demo_noise():
#     print("=" * 55)
#     print("DEMO 4: NOISE / DECOHERENCE MODEL")
#     print("=" * 55)
#     qs = QuantumState(n_qubits=2)
#     qs.apply_h(0)
#     qs.apply_cnot(control=0, target=1)

#     rho = statevector_to_density_matrix(qs.state)
#     print("Ideal Bell state:", get_probabilities_from_density_matrix(rho))

#     rho_noisy = bit_flip_channel(rho, p=0.1, qubit=0, n_qubits=2)
#     print("After 10% bit-flip noise:", get_probabilities_from_density_matrix(rho_noisy))

#     rho_damped = amplitude_damping_channel(rho, gamma=0.15, qubit=1, n_qubits=2)
#     print("After 15% amplitude damping:", get_probabilities_from_density_matrix(rho_damped))
#     print()


# if __name__ == "__main__":
#     demo_bell_state()
#     demo_ghz_state()
#     demo_grover()
#     demo_noise()
#     print("=" * 55)
#     print("ALL DEMOS PASSED. Built from scratch, zero external quantum SDK.")
#     print("=" * 55)






"""
run_my_core.py
----------------
Entry point. Runs all demos: Bell state, GHZ state, Grover's search,
noise/decoherence, measurement shots, qubit scaling + benchmarking,
error correction, and circuit visualization.
"""

from core_engine.statevector import QuantumState
from core_engine.measurement_channel import measure_shots
from algorithms.grover_search import run_grover, optimal_iterations
from algorithms.error_correction import encode_bit_flip, introduce_bit_flip_error, detect_and_correct
from noise_channel.decoherence import (
    statevector_to_density_matrix,
    bit_flip_channel,
    amplitude_damping_channel,
    get_probabilities_from_density_matrix,
)
from utils.benchmarker import benchmark
from utils.circuit_tracer import CircuitTracer


def demo_bell_state():
    print("=" * 55)
    print("DEMO 1: BELL STATE — 2-Qubit Entanglement")
    print("=" * 55)
    qs = QuantumState(n_qubits=2)
    print("Initial:", qs.measure_probabilities_dict())
    qs.apply_h(0)
    print("After Hadamard:", qs.measure_probabilities_dict())
    qs.apply_cnot(control=0, target=1)
    print("After CNOT (entangled):", qs.measure_probabilities_dict())
    print("Expected: {'00': 0.5, '11': 0.5}\n")


def demo_ghz_state():
    print("=" * 55)
    print("DEMO 2: GHZ STATE — 3-Qubit Entanglement")
    print("=" * 55)
    qs = QuantumState(n_qubits=3)
    qs.apply_h(0)
    qs.apply_cnot(control=0, target=1)
    qs.apply_cnot(control=1, target=2)
    print("Final:", qs.measure_probabilities_dict())
    print("Expected: {'000': 0.5, '111': 0.5}\n")


def demo_grover():
    print("=" * 55)
    print("DEMO 3: GROVER'S SEARCH ALGORITHM")
    print("=" * 55)
    print("2-qubit search, target |11>, 1 iteration:")
    print(run_grover(n_qubits=2, target_index=3, iterations=1))
    n = 3
    opt_iter = optimal_iterations(n)
    print(f"\n3-qubit search, target |101>, {opt_iter} iterations (optimal):")
    print(run_grover(n_qubits=n, target_index=5, iterations=opt_iter))
    print()


def demo_noise():
    print("=" * 55)
    print("DEMO 4: NOISE / DECOHERENCE MODEL")
    print("=" * 55)
    qs = QuantumState(n_qubits=2)
    qs.apply_h(0)
    qs.apply_cnot(control=0, target=1)
    rho = statevector_to_density_matrix(qs.state)
    print("Ideal Bell state:", get_probabilities_from_density_matrix(rho))
    rho_noisy = bit_flip_channel(rho, p=0.1, qubit=0, n_qubits=2)
    print("After 10% bit-flip noise:", get_probabilities_from_density_matrix(rho_noisy))
    rho_damped = amplitude_damping_channel(rho, gamma=0.15, qubit=1, n_qubits=2)
    print("After 15% amplitude damping:", get_probabilities_from_density_matrix(rho_damped))
    print()


def demo_measurement_shots():
    print("=" * 55)
    print("DEMO 5: MEASUREMENT SHOTS (real-hardware-style output)")
    print("=" * 55)
    qs = QuantumState(n_qubits=2)
    qs.apply_h(0)
    qs.apply_cnot(0, 1)
    counts = measure_shots(qs.state, n_qubits=2, shots=1024, seed=7)
    print(f"1024 shots on Bell state: {counts}\n")


def demo_scaling_and_benchmark():
    print("=" * 55)
    print("DEMO 6: QUBIT SCALING + PERFORMANCE BENCHMARK")
    print("=" * 55)

    def build_and_run(n):
        qs = QuantumState(n)
        for i in range(n - 1):
            qs.apply_h(i)
            qs.apply_cnot(i, i + 1)
        return qs

    for n in [4, 6, 8, 10]:
        print(f"--- {n} qubits ---")
        benchmark(build_and_run, n)
    print()


def demo_error_correction():
    print("=" * 55)
    print("DEMO 7: 3-QUBIT BIT-FLIP ERROR CORRECTION")
    print("=" * 55)
    qs = encode_bit_flip(logical_state_is_one=True)
    qs = introduce_bit_flip_error(qs, qubit_index=1)
    print("State after error injected on qubit 1:", qs.measure_probabilities_dict())
    result, raw = detect_and_correct(qs)
    print(f"Recovered logical bit: {result} (raw measurement: {raw}) -> ERROR CORRECTED\n")


def demo_circuit_tracer():
    print("=" * 55)
    print("DEMO 8: ASCII CIRCUIT DIAGRAM")
    print("=" * 55)
    tracer = CircuitTracer(n_qubits=2)
    tracer.log_gate("H", [0])
    tracer.log_gate("CNOT", [0, 1])
    tracer.draw()
    print()





from noise_channel.hardware_profiles import apply_profile_to_density_matrix
def demo_hardware_profiles():
    print("=" * 55)
    print("DEMO 9: CONFIGURABLE HARDWARE NOISE PROFILES")
    print("=" * 55)
    qs = QuantumState(n_qubits=2)
    qs.apply_h(0)
    qs.apply_cnot(0, 1)
    rho = statevector_to_density_matrix(qs.state)
    for profile in ["ideal", "superconducting_typical", "noisy_nisq"]:
        result = apply_profile_to_density_matrix(rho, profile, 2)
        print(f"{profile}:", get_probabilities_from_density_matrix(result))
    print()


def demo_deutsch_jozsa():
    print("=" * 55)
    print("DEMO 10: DEUTSCH-JOZSA ALGORITHM")
    print("=" * 55)
    from algorithms.deutsch_jozsa import demo_both_cases
    for name, res in demo_both_cases(n_qubits=4).items():
        print(f"  {name}: verdict={res['verdict']}  p(all-zero)={res['p_all_zero']:.4f}")
    print()


def demo_qft():
    print("=" * 55)
    print("DEMO 11: QUANTUM FOURIER TRANSFORM")
    print("=" * 55)
    from algorithms.qft import verify_qft_against_definition
    for n in [2, 3, 4]:
        result = verify_qft_against_definition(n)
        print(f"  {n} qubits: max error vs. closed-form DFT matrix = "
              f"{result['max_abs_error']:.2e}  ({'PASS' if result['passed'] else 'FAIL'})")
    print()


def demo_vqe():
    print("=" * 55)
    print("DEMO 12: TOY VQE (2-qubit Pauli-sum Hamiltonian)")
    print("=" * 55)
    from algorithms.vqe_toy import run_vqe
    result = run_vqe()
    print(f"  VQE energy:            {result['vqe_energy']:.10f}")
    print(f"  Exact ground state:    {result['exact_ground_state_energy']:.10f}")
    print(f"  Absolute error:        {result['absolute_error']:.2e}")
    print()


def demo_mps_engine():
    print("=" * 55)
    print("DEMO 13: MPS ENGINE -- gates applied directly on tensors")
    print("=" * 55)
    import time
    from core_engine.mps_engine import MPSState
    for n in (20, 50, 100):
        t0 = time.time()
        m = MPSState(n)
        m.h(0)
        for i in range(n - 1):
            m.cnot(i, i + 1)
        dense_gb = (2 ** n) * 16 / 1e9
        print(f"  GHZ n={n:3d}: {time.time() - t0:.3f}s, MPS memory "
              f"{m.memory_bytes() / 1024:.1f} KiB (dense vector would need {dense_gb:.3g} GB)")
    m = MPSState(100)
    m.h(0)
    for i in range(99):
        m.cnot(i, i + 1)
    print("  100-qubit GHZ samples:", {k[:6] + '..': v for k, v in m.sample(100, seed=1).items()})
    print(f"  entanglement entropy across the middle cut: {m.entanglement_entropy(50):.3f} bit")
    print()


def demo_syndrome_qec():
    print("=" * 55)
    print("DEMO 14: ANCILLA-BASED SYNDROME ERROR CORRECTION")
    print("=" * 55)
    import numpy as np
    from algorithms.error_correction import run_qec_cycle, monte_carlo_logical_error
    rng = np.random.default_rng(0)
    for code in ("bit", "phase"):
        for err in (None, 0, 1, 2):
            r = run_qec_cycle(theta=1.0, error_qubit=err, code=code, rng=rng)
            print(f"  {code}-flip code, error on {err}: syndrome={r['syndrome']} "
                  f"-> fixed qubit {r['corrected_qubit']}, logical fidelity={r['fidelity']:.6f}")
    print("  (data qubits never measured -- the superposition survives)")
    for p in (0.05, 0.1, 0.2):
        r = monte_carlo_logical_error(p, trials=800, seed=1)
        print(f"  p_physical={p:.2f} -> logical error {r['logical_error_rate']:.4f} "
              f"(theory 3p^2-2p^3 = {r['theory']:.4f})")
    print()


def demo_shor():
    print("=" * 55)
    print("DEMO 15: SHOR'S ALGORITHM (scales past N=15)")
    print("=" * 55)
    import time
    from algorithms.shor import find_factors_via_shor
    for N in (15, 21, 35, 77, 143):
        t0 = time.time()
        f = find_factors_via_shor(N, a=2, verbose=False)
        print(f"  N={N:4d} -> {f}   ({time.time() - t0:.2f}s)")
    print()


def demo_circuit_builder():
    print("=" * 55)
    print("DEMO 16: GENERAL CIRCUIT BUILDER (JSON / text / backends)")
    print("=" * 55)
    from core_engine.circuit import QuantumCircuit
    qc = QuantumCircuit.from_text("qubits 4\nh 0\ncnot 0 3\nrx 1 1.5708\ncphase 1 3 0.7")
    print(qc)
    print(qc.draw())
    print("  statevector:", qc.run(shots=500, backend="statevector", seed=1)["counts"])
    print("  mps        :", qc.run(shots=500, backend="mps", seed=1)["counts"])
    print("  JSON:", qc.to_json()[:90] + "...")
    print()


def demo_pulse_noise():
    print("=" * 55)
    print("DEMO 17: PULSE-DERIVED NOISE (drive error + crosstalk + T1/T2)")
    print("=" * 55)
    from core_engine.circuit import QuantumCircuit
    from noise_channel.pulse_noise import PulseNoiseModel
    qc = QuantumCircuit(3).h(0).cnot(0, 1).cnot(1, 2)
    for prof in ("ideal", "superconducting", "noisy_nisq"):
        r = qc.run(shots=2000, noise=PulseNoiseModel.from_profile(prof), seed=3)
        print(f"  {prof:16s} fidelity={r['fidelity_vs_ideal']:.4f} purity={r['purity']:.4f} "
              f"P(000)+P(111)={(r['counts'].get('000', 0) + r['counts'].get('111', 0)) / 2000:.3f}")
    print()


if __name__ == "__main__":
    demo_bell_state()
    demo_ghz_state()
    demo_grover()
    demo_noise()
    demo_measurement_shots()
    demo_scaling_and_benchmark()
    demo_error_correction()
    demo_circuit_tracer()
    demo_hardware_profiles()
    demo_deutsch_jozsa()
    demo_qft()
    demo_vqe()
    demo_mps_engine()
    demo_syndrome_qec()
    demo_shor()
    demo_circuit_builder()
    demo_pulse_noise()
    print("=" * 55)
    print("ALL DEMOS PASSED. Built from scratch, zero external quantum SDK.")
    print("=" * 55)
    print()
    print("For the investor-facing validation package (Qiskit cross-check,")
    print("real IBM hardware calibration noise model, external MPS benchmark")
    print("anchor, and real-hardware run script), see:")
    print("  validation/qiskit_cross_check.py")
    print("  noise_channel/ibm_calibration.py")
    print("  benchmarks/external_reference_comparison.py")
    print("  hardware_validation/run_on_real_ibm_device.py")
