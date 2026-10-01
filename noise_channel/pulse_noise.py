"""
pulse_noise.py
----------------
Connects the pulse-level physics (core_engine/pulse_control.py) to the
noise/decoherence model, so gate errors in a circuit come from physical
pulse imperfections instead of hand-picked probabilities.

Per physical gate the model applies, on a density matrix:
  1. coherent drive error   -- extracted by simulating a Gaussian pi-pulse
                               with amplitude (calibration) error + detuning,
                               scaled to the gate's rotation angle
  2. crosstalk to neighbours -- leaked drive + ZZ coupling on qubits q+-1,
                               extracted from a 2-qubit pulse simulation
  3. T1 / T2 decoherence     -- amplitude damping + pure dephasing over the
                               gate duration (all qubits; gates run serially,
                               a conservative simplification)
  4. readout error           -- symmetric bit flip when sampling

Z-type gates (rz, s, sdg, t, z) are "virtual" (frame changes): zero time,
zero error, as on real superconducting hardware.

Honest scope: illustrative, physically-motivated parameters -- NOT a
calibrated digital twin of any specific chip. Linear nearest-neighbour
coupling. Density matrices limit this to ~9 qubits.

NumPy only, no quantum SDK.
"""
import numpy as np

from core_engine import pulse_control as pc
from core_engine.custom_gates import (
    gate_H, gate_X, gate_Y, gate_Z, gate_S, gate_S_dag, gate_T,
    gate_RX, gate_RY, gate_RZ, gate_CNOT, gate_CZ, gate_SWAP, gate_CPHASE,
    gate_T_dag, gate_SX, gate_P, gate_U3, gate_ISWAP,
)
from core_engine.kronecker_solver import (
    apply_single_qubit_gate_efficient, apply_two_qubit_gate_general,
)

MAX_DENSITY_QUBITS = 9

# name -> (ideal-gate matrix factory, physical?, pulse ratio vs a pi pulse)
_VIRTUAL = {"z", "s", "sdg", "t", "tdg", "rz", "p"}

PROFILES = {
    # ns / rad-per-ns; values in the ballpark of published superconducting
    # devices, chosen to be illustrative.
    "ideal": dict(t1_ns=float("inf"), t2_ns=float("inf"), t_1q_ns=40.0,
                  t_2q_ns=300.0, amp_error=0.0, detuning_mhz=0.0,
                  xtalk_drive=0.0, zz_khz=0.0, readout_error=0.0),
    "superconducting": dict(t1_ns=100_000.0, t2_ns=80_000.0, t_1q_ns=40.0,
                            t_2q_ns=300.0, amp_error=0.005, detuning_mhz=0.05,
                            xtalk_drive=0.01, zz_khz=30.0, readout_error=0.01),
    "noisy_nisq": dict(t1_ns=20_000.0, t2_ns=15_000.0, t_1q_ns=50.0,
                       t_2q_ns=400.0, amp_error=0.02, detuning_mhz=0.3,
                       xtalk_drive=0.03, zz_khz=150.0, readout_error=0.04),
}
_ALIASES = {"superconducting_typical": "superconducting"}


class PulseNoiseModel:
    def __init__(self, t1_ns, t2_ns, t_1q_ns, t_2q_ns, amp_error, detuning_mhz,
                 xtalk_drive, zz_khz, readout_error):
        if t2_ns > 2 * t1_ns:
            raise ValueError("physical constraint violated: T2 <= 2*T1")
        self.t1_ns, self.t2_ns = t1_ns, t2_ns
        self.t_1q_ns, self.t_2q_ns = t_1q_ns, t_2q_ns
        self.amp_error = amp_error
        self.detuning = 2 * np.pi * detuning_mhz * 1e-3      # rad/ns
        self.xtalk_drive = xtalk_drive
        self.zz = 2 * np.pi * zz_khz * 1e-6                  # rad/ns
        self.readout_error = readout_error
        self._derive_from_pulses()

    @classmethod
    def from_profile(cls, name):
        name = _ALIASES.get(name, name)
        if name not in PROFILES:
            raise ValueError(f"Unknown profile '{name}'. Options: {sorted(PROFILES)}")
        return cls(**PROFILES[name])

    # ------------------------------------------------------------------
    def _derive_from_pulses(self):
        T, sigma = self.t_1q_ns, self.t_1q_ns / 4
        amp = pc.calibrated_amplitude(np.pi, T, sigma)
        ideal_pi = pc.simulate_pulse_unitary(T, sigma, amp)
        real_pi = pc.simulate_pulse_unitary(T, sigma, amp * (1 + self.amp_error),
                                            detuning=self.detuning)
        # relative coherent error of a pi-pulse (identity if perfect)
        self.E_drive_pi = real_pi @ ideal_pi.conj().T
        self.fidelity_1q_coherent = pc.average_gate_fidelity(real_pi, ideal_pi)

        # crosstalk-only error on (driven, spectator) pair
        U_ref = pc.simulate_two_qubit_drive(T, sigma, amp)
        U_xt = pc.simulate_two_qubit_drive(T, sigma, amp,
                                           xtalk_drive=self.xtalk_drive, zz=self.zz)
        self.E_xtalk_pi = U_xt @ U_ref.conj().T
        self.fidelity_xtalk = pc.average_gate_fidelity(U_xt, U_ref)

        # decoherence per unit time
        self.gamma_rate = 0.0 if np.isinf(self.t1_ns) else 1.0 / self.t1_ns
        if np.isinf(self.t2_ns):
            self.phi_rate = 0.0
        else:
            self.phi_rate = max(1.0 / self.t2_ns - 0.5 * self.gamma_rate, 0.0)

    def damping_params(self, duration_ns):
        gamma = 1 - np.exp(-duration_ns * self.gamma_rate)
        lam = 1 - np.exp(-duration_ns * self.phi_rate)
        return float(gamma), float(lam)

    def summary(self):
        g1, l1 = self.damping_params(self.t_1q_ns)
        return {
            "fidelity_1q": self.fidelity_1q_coherent
            * (1 - (g1 / 2 + l1 / 2) * 2 / 3),
            "fidelity_1q_coherent_only": self.fidelity_1q_coherent,
            "crosstalk_fidelity": self.fidelity_xtalk,
            "gamma_1q": g1, "lambda_1q": l1,
            "t1_ns": self.t1_ns, "t2_ns": self.t2_ns,
            "readout_error": self.readout_error,
        }


# ---------------------------------------------------------------------------
# Density-matrix helpers (tensor reshape, O(4^n) memory but no kron blow-up)
# ---------------------------------------------------------------------------
def _apply_op(rho, K, qubits, n):
    """rho -> K rho K^dag for a 1- or 2-qubit operator K on `qubits`."""
    flat = rho.reshape(-1)
    if len(qubits) == 1:
        q = qubits[0]
        flat = apply_single_qubit_gate_efficient(flat, K, q, 2 * n)
        flat = apply_single_qubit_gate_efficient(flat, K.conj(), n + q, 2 * n)
    else:
        a, b = qubits
        flat = apply_two_qubit_gate_general(flat, K, a, b, 2 * n)
        flat = apply_two_qubit_gate_general(flat, K.conj(), n + a, n + b, 2 * n)
    return flat.reshape(rho.shape)


def _apply_kraus(rho, kraus, q, n):
    out = np.zeros_like(rho)
    for K in kraus:
        out += _apply_op(rho, K, [q], n)
    return out


def _decohere_all(rho, model, duration_ns, n):
    if duration_ns <= 0:
        return rho
    gamma, lam = model.damping_params(duration_ns)
    if gamma == 0 and lam == 0:
        return rho
    ad = [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
          np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)]
    pd = [np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex),
          np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)]
    for q in range(n):
        if gamma > 0:
            rho = _apply_kraus(rho, ad, q, n)
        if lam > 0:
            rho = _apply_kraus(rho, pd, q, n)
    return rho


_GATE_1Q = {
    "h": lambda p: gate_H(), "x": lambda p: gate_X(), "y": lambda p: gate_Y(),
    "z": lambda p: gate_Z(), "s": lambda p: gate_S(), "sdg": lambda p: gate_S_dag(),
    "t": lambda p: gate_T(), "rx": lambda p: gate_RX(p[0]),
    "ry": lambda p: gate_RY(p[0]), "rz": lambda p: gate_RZ(p[0]),
    "tdg": lambda p: gate_T_dag(), "sx": lambda p: gate_SX(),
    "p": lambda p: gate_P(p[0]), "u3": lambda p: gate_U3(*p),
}
_GATE_2Q = {
    "cnot": lambda p: gate_CNOT(), "cz": lambda p: gate_CZ(),
    "swap": lambda p: gate_SWAP(), "cphase": lambda p: gate_CPHASE(p[0]),
    "iswap": lambda p: gate_ISWAP(),
}


def _pulse_ratio(name, params):
    if name in ("rx", "ry"):
        return min(abs(params[0]) / np.pi, 2.0)
    if name == "sx":
        return 0.5
    return 1.0            # h, x, y are pi-scale rotations


def run_noisy_circuit(circuit, model, shots=1024, seed=None):
    """Execute a QuantumCircuit on a density matrix with pulse-derived
    noise. Returns counts + fidelity/purity diagnostics."""
    n = circuit.n_qubits
    if n > MAX_DENSITY_QUBITS:
        raise MemoryError(
            f"density-matrix noise backend limited to {MAX_DENSITY_QUBITS} qubits "
            f"(needs 4^n memory); got {n}. Use backend='statevector'/'mps' for ideal runs.")
    rng = np.random.default_rng(seed)

    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rho[0, 0] = 1.0
    ideal = None
    from core_engine.statevector import QuantumState
    ideal_qs = circuit._run_statevector()
    ideal = ideal_qs.get_statevector()

    for name, q, p in circuit.ops:
        if name in _GATE_1Q:
            U = _GATE_1Q[name](p)
            rho = _apply_op(rho, U, [q[0]], n)
            if name in _VIRTUAL:
                continue                              # virtual Z: no time, no error
            r = _pulse_ratio(name, p)
            rho = _apply_op(rho, pc.unitary_power(model.E_drive_pi, r), [q[0]], n)
            E_x = pc.unitary_power(model.E_xtalk_pi, r)
            for nb in (q[0] - 1, q[0] + 1):           # linear-chain neighbours
                if 0 <= nb < n:
                    rho = _apply_op(rho, E_x, [q[0], nb], n)
            rho = _decohere_all(rho, model, model.t_1q_ns, n)
        else:
            U = _GATE_2Q[name](p)
            rho = _apply_op(rho, U, list(q), n)
            reps = 3 if name == "swap" else 1
            dur = model.t_2q_ns * reps
            E_half = pc.unitary_power(model.E_drive_pi, 0.5)
            for qq in q:
                rho = _apply_op(rho, E_half, [qq], n)
            ang = model.zz * dur
            ZZ = np.diag(np.exp(-1j * ang / 2 * np.array([1, -1, -1, 1]))).astype(complex)
            rho = _apply_op(rho, ZZ, list(q), n)
            rho = _decohere_all(rho, model, dur, n)

    probs = np.real(np.diag(rho)).clip(min=0)
    probs = probs / probs.sum()
    fidelity = float(np.real(np.vdot(ideal, rho @ ideal)))
    purity = float(np.real(np.trace(rho @ rho)))

    draws = rng.choice(len(probs), size=shots, p=probs)
    counts = {}
    for d in draws:
        bits = [(int(d) >> (n - 1 - i)) & 1 for i in range(n)]
        if model.readout_error > 0:
            bits = [b ^ int(rng.random() < model.readout_error) for b in bits]
        key = ''.join(map(str, bits))
        counts[key] = counts.get(key, 0) + 1
    return {
        "backend": "density", "shots": shots, "counts": counts,
        "fidelity_vs_ideal": fidelity, "purity": purity,
        "noise_summary": model.summary(),
    }
