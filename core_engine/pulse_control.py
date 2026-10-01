"""
pulse_control.py
-------------------
Simplified pulse-level control simulation. Real quantum hardware sends
electromagnetic/laser pulses that drive qubit rotations via the
time-dependent Schrodinger equation, not abstract gates directly.
This module numerically integrates that equation for a single qubit
driven by a Gaussian pulse envelope.

Scope: pulse-level physics for 1 and 2 qubits, now including drive
detuning, amplitude (calibration) error, leaked-drive crosstalk to a
neighbouring qubit, and always-on ZZ coupling. The resulting error
unitaries feed noise_channel/pulse_noise.py, so pulse imperfections
turn into gate noise in circuit simulations. Units: time in ns,
frequencies/amplitudes in rad/ns. Illustrative parameters, not a
hardware-ready pulse compiler or calibrated to a specific device.
"""
import numpy as np


def gaussian_pulse(t, t0, sigma, amplitude):
    return amplitude * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))


def simulate_pulse_evolution(total_time, dt, t0, sigma, amplitude):
    """4th-order Runge-Kutta integration of the Schrodinger equation,
    implemented from scratch."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    psi = np.array([1, 0], dtype=complex)
    n_steps = int(total_time / dt)

    def derivative(psi, t):
        omega_t = gaussian_pulse(t, t0, sigma, amplitude)
        H = (omega_t / 2) * sigma_x
        return -1j * H @ psi

    t = 0.0
    for _ in range(n_steps):
        k1 = derivative(psi, t)
        k2 = derivative(psi + dt/2 * k1, t + dt/2)
        k3 = derivative(psi + dt/2 * k2, t + dt/2)
        k4 = derivative(psi + dt * k3, t + dt)
        psi = psi + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt

    psi = psi / np.linalg.norm(psi)
    return psi


def pulse_area(t0, sigma, amplitude, total_time, dt):
    ts = np.arange(0, total_time, dt)
    return np.sum(gaussian_pulse(ts, t0, sigma, amplitude)) * dt

# ===========================================================================
# Unitary-level pulse simulation, crosstalk, and error extraction
# ===========================================================================
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def _reunitarize(U):
    """Project a numerically-drifted matrix back onto the unitary group."""
    W, _, Vh = np.linalg.svd(U)
    return W @ Vh


def _propagate(hamiltonian, dim, total_time, dt):
    """RK4 integration of dU/dt = -i H(t) U starting from identity."""
    U = np.eye(dim, dtype=complex)
    n_steps = int(round(total_time / dt))

    def f(U, t):
        return -1j * hamiltonian(t) @ U

    t = 0.0
    for _ in range(n_steps):
        k1 = f(U, t)
        k2 = f(U + dt / 2 * k1, t + dt / 2)
        k3 = f(U + dt / 2 * k2, t + dt / 2)
        k4 = f(U + dt * k3, t + dt)
        U = U + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return _reunitarize(U)


def calibrated_amplitude(target_angle, total_time, sigma, dt=0.05):
    """Gaussian amplitude whose *numerical* (truncated) area equals
    target_angle -- what a calibration routine would find."""
    t0 = total_time / 2
    unit_area = pulse_area(t0, sigma, 1.0, total_time, dt)
    return target_angle / unit_area


def simulate_pulse_unitary(total_time, sigma, amplitude, detuning=0.0, dt=0.05):
    """2x2 unitary of an X-drive Gaussian pulse with detuning (rad/ns).
    H(t) = (Omega(t)/2) X + (detuning/2) Z."""
    t0 = total_time / 2

    def H(t):
        return 0.5 * gaussian_pulse(t, t0, sigma, amplitude) * _X + 0.5 * detuning * _Z

    return _propagate(H, 2, total_time, dt)


def simulate_two_qubit_drive(total_time, sigma, amplitude, detuning=0.0,
                             xtalk_drive=0.0, zz=0.0, dt=0.05):
    """Drive qubit 0 with a Gaussian X pulse while qubit 1 is a spectator.
    xtalk_drive : fraction of the drive that leaks onto qubit 1
    zz          : always-on ZZ coupling, H_zz = (zz/2) Z(x)Z  (rad/ns)
    Returns the 4x4 unitary on (q0, q1), q0 most significant."""
    t0 = total_time / 2
    XI, IX = np.kron(_X, _I2), np.kron(_I2, _X)
    ZI, ZZop = np.kron(_Z, _I2), np.kron(_Z, _Z)

    def H(t):
        om = gaussian_pulse(t, t0, sigma, amplitude)
        return (0.5 * om * XI + 0.5 * xtalk_drive * om * IX
                + 0.5 * detuning * ZI + 0.5 * zz * ZZop)

    return _propagate(H, 4, total_time, dt)


def average_gate_fidelity(U, U_target):
    """F_avg = (d*F_pro + 1)/(d+1), F_pro = |Tr(U_target^dag U)|^2 / d^2."""
    d = U.shape[0]
    f_pro = abs(np.trace(U_target.conj().T @ U)) ** 2 / d ** 2
    return float((d * f_pro + 1) / (d + 1))


def unitary_power(U, s):
    """Fractional power U**s of a unitary via a shared eigenbasis
    (NumPy only). Used to scale a pi-pulse error to other rotation angles."""
    A = (U + U.conj().T) / 2
    B = (U - U.conj().T) / 2j
    rng = np.random.default_rng(12345)
    c1, c2 = rng.uniform(0.5, 1.5, 2)
    _, V = np.linalg.eigh(c1 * A + c2 * B)
    D = V.conj().T @ U @ V
    phases = np.angle(np.diag(D))
    return V @ np.diag(np.exp(1j * s * phases)) @ V.conj().T


if __name__ == "__main__":
    total_time, dt = 20.0, 0.01
    t0, sigma = total_time / 2, 2.0
    target_area = np.pi
    amplitude = target_area / (sigma * np.sqrt(2 * np.pi))

    print(f"Target pulse area: {target_area:.4f} (should give X gate / pi-rotation)")
    print(f"Achieved pulse area: {pulse_area(t0, sigma, amplitude, total_time, dt):.4f}")

    final_state = simulate_pulse_evolution(total_time, dt, t0, sigma, amplitude)
    print(f"\nFinal state after pulse: {np.round(final_state, 4)}")
    prob_1 = np.abs(final_state[1]) ** 2
    print(f"Probability of |1>: {prob_1:.4f} (ideal X gate = 1.0000)")