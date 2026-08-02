"""
hardware_profiles.py
-----------------------
Configurable noise presets based on PUBLICLY REPORTED hardware specs
(approximate, from published papers/docs). NOT a prediction of future
hardware -- a calibration tool so simulation results resemble known
real device characteristics.
"""

HARDWARE_PROFILES = {
    "ideal": {"bit_flip_p": 0.0, "damping_gamma": 0.0},
    "superconducting_typical": {
        "bit_flip_p": 0.01,
        "damping_gamma": 0.03,
    },
    "noisy_nisq": {
        "bit_flip_p": 0.05,
        "damping_gamma": 0.08,
    },
}


def get_profile(name):
    if name not in HARDWARE_PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Options: {list(HARDWARE_PROFILES.keys())}")
    return HARDWARE_PROFILES[name]


def apply_profile_to_density_matrix(rho, profile_name, n_qubits):
    from noise_channel.decoherence import bit_flip_channel, amplitude_damping_channel
    profile = get_profile(profile_name)
    for q in range(n_qubits):
        rho = bit_flip_channel(rho, profile["bit_flip_p"], q, n_qubits)
        rho = amplitude_damping_channel(rho, profile["damping_gamma"], q, n_qubits)
    return rho