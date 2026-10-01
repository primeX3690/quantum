"""
cli.py -- command line front-end for the from-scratch quantum engine.

  python cli.py run circuit.txt --shots 2000            # text or .json circuit
  python cli.py run circuit.json --backend mps --draw
  python cli.py shor 21
  python cli.py qec --p 0.05 --trials 500
  python cli.py mps-demo --qubits 100
  python cli.py noisy circuit.txt --profile superconducting
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core_engine.circuit import QuantumCircuit


def _load_circuit(path):
    text = open(path, encoding="utf-8").read()
    if path.endswith(".json"):
        return QuantumCircuit.from_json(text)
    return QuantumCircuit.from_text(text)


def _print_counts(counts, limit=16):
    total = sum(counts.values())
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:limit]:
        print(f"  {k}  {v:6d}  {v / total:6.1%}")
    if len(counts) > limit:
        print(f"  ... {len(counts) - limit} more outcomes")


def cmd_run(a):
    qc = _load_circuit(a.file)
    print(qc)
    if a.draw:
        print(qc.draw())
    res = qc.run(shots=a.shots, backend=a.backend, seed=a.seed)
    print(f"backend: {res['backend']}")
    _print_counts(res["counts"])
    if res["backend"] == "mps":
        print(f"max bond dim: {res['max_bond_dim_used']}, "
              f"memory: {res['mps_memory_bytes'] / 1024:.1f} KiB")


def cmd_noisy(a):
    from noise_channel.pulse_noise import PulseNoiseModel
    qc = _load_circuit(a.file)
    model = PulseNoiseModel.from_profile(a.profile)
    res = qc.run(shots=a.shots, seed=a.seed, noise=model)
    print(qc)
    print(f"noise model: {a.profile}  (avg 1q gate fidelity "
          f"{res['noise_summary']['fidelity_1q']:.5f})")
    print(f"fidelity vs ideal state: {res['fidelity_vs_ideal']:.4f}")
    _print_counts(res["counts"])


def cmd_shor(a):
    from algorithms.shor import find_factors_via_shor
    r = find_factors_via_shor(a.N, a=a.a)
    print("Result:", r)


def cmd_qec(a):
    from algorithms.error_correction import monte_carlo_logical_error
    for code in ("bit", "phase"):
        r = monte_carlo_logical_error(a.p, trials=a.trials, code=code, seed=a.seed)
        print(f"{code}-flip code: physical p={a.p}  logical error="
              f"{r['logical_error_rate']:.4f}  (theory {r['theory']:.4f})")


def cmd_mps(a):
    import time
    from core_engine.mps_engine import MPSState
    t0 = time.time()
    m = MPSState(a.qubits)
    m.h(0)
    for i in range(a.qubits - 1):
        m.cnot(i, i + 1)
    print(f"GHZ on {a.qubits} qubits in {time.time() - t0:.2f}s, "
          f"{m.memory_bytes() / 1024:.1f} KiB "
          f"(a dense vector would need {(2 ** a.qubits) * 16 / 1e9:.3g} GB)")
    _print_counts(m.sample(200, seed=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a circuit file")
    r.add_argument("file")
    r.add_argument("--shots", type=int, default=1024)
    r.add_argument("--backend", default="auto", choices=["auto", "statevector", "mps"])
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--draw", action="store_true")
    r.set_defaults(fn=cmd_run)

    n = sub.add_parser("noisy", help="run with pulse-derived hardware noise")
    n.add_argument("file")
    n.add_argument("--profile", default="superconducting",
                   choices=["ideal", "superconducting", "noisy_nisq"])
    n.add_argument("--shots", type=int, default=1024)
    n.add_argument("--seed", type=int, default=None)
    n.set_defaults(fn=cmd_noisy)

    s = sub.add_parser("shor", help="factor N with simulated Shor")
    s.add_argument("N", type=int)
    s.add_argument("--a", type=int, default=2)
    s.set_defaults(fn=cmd_shor)

    q = sub.add_parser("qec", help="syndrome-based error-correction Monte Carlo")
    q.add_argument("--p", type=float, default=0.05)
    q.add_argument("--trials", type=int, default=500)
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(fn=cmd_qec)

    m = sub.add_parser("mps-demo", help="GHZ on many qubits via MPS")
    m.add_argument("--qubits", type=int, default=100)
    m.set_defaults(fn=cmd_mps)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
