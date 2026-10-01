"""Ancilla-based syndrome error correction."""
import numpy as np
from algorithms.error_correction import (
    run_qec_cycle, monte_carlo_logical_error, encode_bit_flip,
    introduce_bit_flip_error, detect_and_correct,
)


def test_single_errors_corrected_and_superposition_preserved():
    rng = np.random.default_rng(0)
    for code in ("bit", "phase"):
        for err in (None, 0, 1, 2):
            r = run_qec_cycle(theta=1.0, error_qubit=err, code=code, rng=rng)
            assert r["fidelity"] > 1 - 1e-9, (code, err, r)
            assert r["corrected_qubit"] == err


def test_syndrome_identifies_error_location():
    rng = np.random.default_rng(1)
    expected = {0: (1, 0), 1: (1, 1), 2: (0, 1)}
    for q, syn in expected.items():
        assert run_qec_cycle(1.0, q, "bit", rng)["syndrome"] == syn
    assert run_qec_cycle(1.0, None, "bit", rng)["syndrome"] == (0, 0)


def test_logical_error_rate_matches_theory():
    for p in (0.05, 0.2):
        r = monte_carlo_logical_error(p, trials=1200, seed=7)
        assert abs(r["logical_error_rate"] - r["theory"]) < 0.025


def test_coding_helps_below_threshold():
    r = monte_carlo_logical_error(0.05, trials=1500, seed=3)
    assert r["logical_error_rate"] < 0.05           # better than an uncoded qubit


def test_legacy_majority_vote_still_works():
    qs = encode_bit_flip(True)
    introduce_bit_flip_error(qs, 1)
    assert detect_and_correct(qs)[0] == 1
