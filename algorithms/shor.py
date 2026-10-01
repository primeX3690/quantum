"""
shor.py
---------
Shor's algorithm (quantum period finding + classical post-processing),
built from scratch with NumPy only.

Upgrade over the original toy version: the old code built a dense
(2^(n_count+n_work))^2 modular-exponentiation matrix, which capped it at
N=15. Here the register is stored as a (count x work) array, modular
exponentiation is applied as an index permutation (exact, O(2^n)), and the
QFT over the counting register is applied along one axis. That lets a laptop
factor N up to ~100+ (see `max_supported_N`). Cost is still exponential in
bit-length(N) -- this is a simulation, not a speed-up -- and that honest
limit is enforced by `check_memory`.

Convention: qubit ordering / QFT sign match core_engine + algorithms.qft
(F_jk = exp(2*pi*i*j*k/N)/sqrt(N)).
"""
import numpy as np
from math import gcd
from fractions import Fraction

MAX_STATE_BYTES = 1.5e9   # keep well inside an 8 GB laptop


def qft_matrix(n_qubits):
    """QFT unitary from its definition F_jk = exp(2*pi*i*j*k/N)/sqrt(N).
    Kept for verification; the fast path uses an FFT along one axis."""
    N = 2 ** n_qubits
    j = np.arange(N).reshape(-1, 1)
    k = np.arange(N).reshape(1, -1)
    return np.exp(2j * np.pi * j * k / N) / np.sqrt(N)


def modexp_unitary(a, N_mod, n_count, n_work):
    """Dense permutation matrix |x>|y> -> |x>|y*a^x mod N>. Small cases /
    verification only; the fast path never builds this."""
    size = 2 ** (n_count + n_work)
    work_size = 2 ** n_work
    U = np.zeros((size, size), dtype=complex)
    for x in range(2 ** n_count):
        ax = pow(a, x, N_mod)
        for y in range(work_size):
            new_y = (y * ax) % N_mod if y < N_mod else y
            U[x * work_size + new_y, x * work_size + y] = 1.0
    return U


def default_n_count(N_mod):
    """2*bit_length(N) counting qubits gives a clean period peak."""
    return 2 * N_mod.bit_length()


def check_memory(N_mod, n_count):
    n_work = N_mod.bit_length()
    need = (2 ** (n_count + n_work)) * 16
    if need > MAX_STATE_BYTES:
        raise MemoryError(
            f"N={N_mod}, n_count={n_count} needs ~{need/1e9:.2f} GB of state; "
            f"exceeds the {MAX_STATE_BYTES/1e9:.1f} GB safety cap.")
    return need


def max_supported_N(n_count_rule=default_n_count):
    """Largest bit-length of N that fits under the memory cap."""
    b = 2
    while (2 ** (n_count_rule(2 ** b) + (2 ** b).bit_length())) * 16 <= MAX_STATE_BYTES:
        b += 1
    return 2 ** b - 1


def shor_period_finding(a, N_mod, n_count=None):
    """Returns the probability distribution over the counting register
    after modexp + inverse-free QFT (P[x], x in [0, 2^n_count))."""
    if n_count is None:
        n_count = default_n_count(N_mod)
    check_memory(N_mod, n_count)
    n_work = N_mod.bit_length()
    Q, W = 2 ** n_count, 2 ** n_work

    # H^n on counting register, work register = |1>, then controlled modexp:
    # amplitude sits at (x, a^x mod N) -- exact permutation, no matrix.
    vals = np.empty(Q, dtype=np.int64)
    cur = 1
    for x in range(Q):
        vals[x] = cur
        cur = (cur * a) % N_mod
    state = np.zeros((Q, W), dtype=complex)
    state[np.arange(Q), vals] = 1 / np.sqrt(Q)

    # QFT over the counting axis (sign/normalisation identical to qft_matrix)
    state = np.fft.ifft(state, axis=0) * np.sqrt(Q)
    return (np.abs(state) ** 2).sum(axis=1)


def _period_from_peak(x, Q, a, N_mod):
    frac = Fraction(x, Q).limit_denominator(N_mod)
    r0 = frac.denominator
    for m in range(1, 6):            # peak may give a divisor of the period
        r = r0 * m
        if r > 1 and pow(a, r, N_mod) == 1:
            return r
    return None


def _factors_from_period(a, r, N_mod):
    if r % 2:
        return None
    half = pow(a, r // 2, N_mod)
    if half == N_mod - 1:
        return None
    for g in (gcd(half - 1, N_mod), gcd(half + 1, N_mod)):
        if 1 < g < N_mod:
            return tuple(sorted((g, N_mod // g)))
    return None


def find_factors_via_shor(N_mod=15, a=7, n_count=None, verbose=True, max_attempts=6):
    """Factor N. Tries `a` first, then other coprime bases if that one
    yields an odd/trivial period (as real Shor does). Returns (p, q) or None."""
    if N_mod % 2 == 0:
        return (2, N_mod // 2)
    if n_count is None:
        n_count = default_n_count(N_mod)
    Q = 2 ** n_count
    say = print if verbose else (lambda *a, **k: None)

    bases = [a] + [b for b in range(2, N_mod) if b != a]
    tried = 0
    for base in bases:
        if tried >= max_attempts:
            break
        g = gcd(base, N_mod)
        if g != 1:
            say(f"a={base} shares factor {g} with N (lucky classical hit)")
            return tuple(sorted((g, N_mod // g)))
        tried += 1
        say(f"Running Shor's period-finding for N={N_mod}, a={base} "
            f"({n_count} counting + {N_mod.bit_length()} work qubits)")
        probs = shor_period_finding(base, N_mod, n_count)
        peaks = np.argsort(probs)[::-1][:8]
        say("  top peaks: " + ", ".join(f"x={int(p)} (P={probs[p]:.3f})" for p in peaks[:4]))
        for x in peaks:
            if x == 0:
                continue
            r = _period_from_peak(int(x), Q, base, N_mod)
            if r is None:
                continue
            f = _factors_from_period(base, r, N_mod)
            if f:
                say(f"  period r={r}  ->  {N_mod} = {f[0]} x {f[1]}")
                return f
        say("  no usable period for this base, trying another")
    say("Failed to factor within attempt budget.")
    return None


if __name__ == "__main__":
    for n in (15, 21, 35, 55, 77, 91):
        find_factors_via_shor(n, a=2)
