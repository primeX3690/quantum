"""Density-matrix noise, pulse physics, and their integration."""
import numpy as np
from core_engine import pulse_control as pc
from core_engine.circuit import QuantumCircuit
from noise_channel.pulse_noise import PulseNoiseModel
from noise_channel.decoherence import (
    statevector_to_density_matrix, amplitude_damping_channel, phase_damping_channel,
)

X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_calibrated_pi_pulse_is_x_gate():
    T, sig = 40.0, 10.0
    amp = pc.calibrated_amplitude(np.pi, T, sig)
    U = pc.simulate_pulse_unitary(T, sig, amp)
    assert pc.average_gate_fidelity(U, -1j * X) > 0.99999


def test_amplitude_error_lowers_fidelity_monotonically():
    T, sig = 40.0, 10.0
    amp = pc.calibrated_amplitude(np.pi, T, sig)
    f = [pc.average_gate_fidelity(pc.simulate_pulse_unitary(T, sig, amp * (1 + e)), -1j * X)
         for e in (0.0, 0.01, 0.05)]
    assert f[0] > f[1] > f[2]


def test_unitary_power_is_consistent():
    T, sig = 40.0, 10.0
    amp = pc.calibrated_amplitude(np.pi, T, sig)
    E = pc.simulate_two_qubit_drive(T, sig, amp, xtalk_drive=0.05, zz=0.001) @ \
        pc.simulate_two_qubit_drive(T, sig, amp).conj().T
    h = pc.unitary_power(E, 0.5)
    assert np.allclose(h @ h, E, atol=1e-10)


def test_ideal_profile_equals_ideal_simulation():
    qc = QuantumCircuit(3).h(0).cnot(0, 1).cnot(1, 2)
    r = qc.run(shots=100, noise=PulseNoiseModel.from_profile("ideal"), seed=0)
    assert r["fidelity_vs_ideal"] > 1 - 1e-9 and r["purity"] > 1 - 1e-9


def test_noisier_profile_gives_lower_fidelity():
    qc = QuantumCircuit(3).h(0).cnot(0, 1).cnot(1, 2)
    f = [qc.run(shots=10, noise=PulseNoiseModel.from_profile(p), seed=0)["fidelity_vs_ideal"]
         for p in ("ideal", "superconducting", "noisy_nisq")]
    assert f[0] > f[1] > f[2]


def test_t1_t2_matches_reference_kraus_channels():
    m = PulseNoiseModel(t1_ns=1000, t2_ns=800, t_1q_ns=100, t_2q_ns=300, amp_error=0,
                        detuning_mhz=0, xtalk_drive=0, zz_khz=0, readout_error=0)
    got = QuantumCircuit(1).x(0).run(shots=5, noise=m, seed=0)["fidelity_vs_ideal"]
    g, l = m.damping_params(100)
    rho = statevector_to_density_matrix(np.array([0, 1], dtype=complex))
    rho = phase_damping_channel(amplitude_damping_channel(rho, g, 0, 1), l, 0, 1)
    assert abs(got - np.real(rho[1, 1])) < 1e-9


def test_crosstalk_is_visible_on_spectators():
    base = dict(t1_ns=float("inf"), t2_ns=float("inf"), t_1q_ns=40, t_2q_ns=300,
                amp_error=0, detuning_mhz=0, readout_error=0)
    qc = QuantumCircuit(3).h(1).x(0).x(2).x(0).x(2)
    clean = qc.run(shots=5, noise=PulseNoiseModel(**base, xtalk_drive=0, zz_khz=0), seed=0)
    xt = qc.run(shots=5, noise=PulseNoiseModel(**base, xtalk_drive=0.05, zz_khz=300), seed=0)
    assert clean["fidelity_vs_ideal"] > 1 - 1e-9
    assert xt["fidelity_vs_ideal"] < clean["fidelity_vs_ideal"] - 1e-3


def test_physical_constraint_t2_le_2t1():
    try:
        PulseNoiseModel(t1_ns=100, t2_ns=500, t_1q_ns=40, t_2q_ns=300, amp_error=0,
                        detuning_mhz=0, xtalk_drive=0, zz_khz=0, readout_error=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_density_backend_guard():
    try:
        QuantumCircuit(12).h(0).run(noise=PulseNoiseModel.from_profile("ideal"))
    except MemoryError:
        return
    raise AssertionError("expected MemoryError")
