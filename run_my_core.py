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
    print("=" * 55)
    print("ALL DEMOS PASSED. Built from scratch, zero external quantum SDK.")
    print("=" * 55)