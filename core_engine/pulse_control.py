"""
pulse_control.py
-------------------
Simplified pulse-level control simulation. Real quantum hardware sends
electromagnetic/laser pulses that drive qubit rotations via the
time-dependent Schrodinger equation, not abstract gates directly.
This module numerically integrates that equation for a single qubit
driven by a Gaussian pulse envelope.

Honest scope: simplified single-qubit model (no crosstalk, no real
device calibration data) -- demonstrates the physics/math of pulse
control, not a hardware-ready pulse compiler.
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