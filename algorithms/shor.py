"""
Simplified Shor's Algorithm demo -- factors N=15 using quantum
period-finding, built from scratch (QFT matrix + modular exponentiation
permutation matrix), then classical post-processing.

Honest scope: N=15 is the standard textbook-scale demonstration.
Larger N needs more qubits than a laptop can statevector-simulate.
"""
import numpy as np
from math import gcd
from fractions import Fraction


def qft_matrix(n_qubits):
    """QFT unitary built directly from its definition:
    F_jk = (1/sqrt(N)) * exp(2*pi*i*j*k/N)"""
    N = 2 ** n_qubits
    F = np.zeros((N, N), dtype=complex)
    for j in range(N):
        for k in range(N):
            F[j, k] = np.exp(2j * np.pi * j * k / N) / np.sqrt(N)
    return F


def modexp_unitary(a, N_mod, n_count, n_work):
    """Permutation matrix for |x>|y> -> |x>|y * a^x mod N_mod>"""
    total_qubits = n_count + n_work
    size = 2 ** total_qubits
    work_size = 2 ** n_work
    U = np.zeros((size, size), dtype=complex)
    for x in range(2 ** n_count):
        a_x_mod_N = pow(a, x, N_mod)
        for y in range(work_size):
            in_index = x * work_size + y
            new_y = (y * a_x_mod_N) % N_mod if y < N_mod else y
            out_index = x * work_size + new_y
            U[out_index, in_index] = 1.0
    return U


def shor_period_finding(a, N_mod, n_count=4):
    n_work = N_mod.bit_length()
    total = n_count + n_work
    size = 2 ** total
    work_size = 2 ** n_work

    state = np.zeros(size, dtype=complex)
    count_amp = 1 / np.sqrt(2 ** n_count)
    for x in range(2 ** n_count):
        idx = x * work_size + 1
        state[idx] = count_amp

    U = modexp_unitary(a, N_mod, n_count, n_work)
    state = U @ state

    F = qft_matrix(n_count)
    full_qft = np.kron(F, np.eye(work_size))
    state = full_qft @ state

    probs = np.abs(state) ** 2
    count_probs = np.zeros(2 ** n_count)
    for x in range(2 ** n_count):
        count_probs[x] = sum(probs[x*work_size:(x+1)*work_size])
    return count_probs


def find_factors_via_shor(N_mod=15, a=7, n_count=4):
    print(f"Running Shor's period-finding for N={N_mod}, a={a}")
    count_probs = shor_period_finding(a, N_mod, n_count)

    top_indices = np.argsort(count_probs)[::-1][:4]
    print("Top measured counting-register values (with probability):")
    for idx in top_indices:
        print(f"  x={idx}, P={count_probs[idx]:.4f}")

    Q = 2 ** n_count
    best_r = None
    for idx in top_indices:
        if idx == 0:
            continue
        frac = Fraction(idx, Q).limit_denominator(N_mod)
        r = frac.denominator
        if r > 1 and pow(a, r, N_mod) == 1:
            best_r = r
            break

    if best_r is None:
        print("Period not found from top measurements (try different 'a').")
        return None

    print(f"Found period r={best_r}")
    if best_r % 2 != 0:
        print("Odd period, cannot proceed with this a.")
        return None

    guess1 = gcd(pow(a, best_r // 2) - 1, N_mod)
    guess2 = gcd(pow(a, best_r // 2) + 1, N_mod)
    print(f"Candidate factors: {guess1}, {guess2}")
    return guess1, guess2


if __name__ == "__main__":
    find_factors_via_shor(N_mod=15, a=7, n_count=4)