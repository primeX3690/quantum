"""
circuit.py
-----------
General-purpose circuit builder for the from-scratch engine.

    qc = QuantumCircuit(3)
    qc.h(0).cnot(0, 1).cnot(1, 2)
    qc.run(shots=1000)                  # -> {'counts': {...}, ...}

* Fluent API, JSON (de)serialisation, and a tiny text format
  ("h 0", "cnot 0 1", "rx 2 1.5708").
* Three backends, same circuit:
    - "statevector"  dense 2^n vector (fast, exact, n <= ~22 on 8 GB)
    - "mps"          matrix-product state (50-100+ qubits if entanglement
                     stays low; see core_engine/mps_engine.py)
    - "density"      density matrix with a pulse-derived noise model
                     (n <= ~9; see noise_channel/pulse_noise.py)
  backend="auto" picks statevector for small n, MPS for large n.
* Input is validated (gate names, qubit ranges, parameter counts) so it is
  safe to expose through the HTTP API.

NumPy only, no quantum SDK.
"""
import json
import numpy as np

# name -> (number of qubit args, number of angle params)
GATE_SPEC = {
    "h": (1, 0), "x": (1, 0), "y": (1, 0), "z": (1, 0),
    "s": (1, 0), "sdg": (1, 0), "t": (1, 0),
    "rx": (1, 1), "ry": (1, 1), "rz": (1, 1),
    "cnot": (2, 0), "cz": (2, 0), "swap": (2, 0), "cphase": (2, 1),
    "tdg": (1, 0), "sx": (1, 0), "p": (1, 1), "u3": (1, 3),
    "iswap": (2, 0),
    "ccx": (3, 0),      # Toffoli -- expanded into 1q/2q gates on add()
}
ALIASES = {"cx": "cnot", "sdag": "sdg", "cp": "cphase", "tdag": "tdg",
           "toffoli": "ccx", "ccnot": "ccx", "phase": "p"}

STATEVECTOR_LIMIT = 22      # 2^22 * 16 B = 64 MiB per copy; fine on 8 GB
MPS_LIMIT = 500


class QuantumCircuit:
    def __init__(self, n_qubits):
        if not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError("n_qubits must be a positive integer")
        self.n_qubits = n_qubits
        self.ops = []          # list of (name, qubits tuple, params tuple)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def add(self, name, qubits, params=()):
        name = ALIASES.get(name.lower(), name.lower())
        if name not in GATE_SPEC:
            raise ValueError(f"unknown gate '{name}'. Available: {sorted(GATE_SPEC)}")
        nq, npar = GATE_SPEC[name]
        qubits = tuple(int(q) for q in (qubits if hasattr(qubits, '__iter__') else [qubits]))
        params = tuple(float(p) for p in params)
        if len(qubits) != nq:
            raise ValueError(f"gate '{name}' needs {nq} qubit(s), got {len(qubits)}")
        if len(params) != npar:
            raise ValueError(f"gate '{name}' needs {npar} parameter(s), got {len(params)}")
        for q in qubits:
            if not 0 <= q < self.n_qubits:
                raise ValueError(f"qubit {q} out of range [0, {self.n_qubits - 1}]")
        if len(set(qubits)) != nq:
            raise ValueError(f"gate '{name}' needs {nq} distinct qubits")
        if name == "ccx":                      # standard 6-CNOT Toffoli decomposition
            a, b, c = qubits
            for g, qs in [("h", (c,)), ("cnot", (b, c)), ("tdg", (c,)), ("cnot", (a, c)),
                          ("t", (c,)), ("cnot", (b, c)), ("tdg", (c,)), ("cnot", (a, c)),
                          ("t", (b,)), ("t", (c,)), ("h", (c,)), ("cnot", (a, b)),
                          ("t", (a,)), ("tdg", (b,)), ("cnot", (a, b))]:
                self.ops.append((g, qs, ()))
            return self
        self.ops.append((name, qubits, params))
        return self

    def h(self, q): return self.add("h", [q])
    def x(self, q): return self.add("x", [q])
    def y(self, q): return self.add("y", [q])
    def z(self, q): return self.add("z", [q])
    def s(self, q): return self.add("s", [q])
    def sdg(self, q): return self.add("sdg", [q])
    def t(self, q): return self.add("t", [q])
    def tdg(self, q): return self.add("tdg", [q])
    def sx(self, q): return self.add("sx", [q])
    def p(self, q, th): return self.add("p", [q], [th])
    def u3(self, q, th, ph, lam): return self.add("u3", [q], [th, ph, lam])
    def iswap(self, a, b): return self.add("iswap", [a, b])
    def ccx(self, a, b, c): return self.add("ccx", [a, b, c])
    def toffoli(self, a, b, c): return self.add("ccx", [a, b, c])
    def rx(self, q, th): return self.add("rx", [q], [th])
    def ry(self, q, th): return self.add("ry", [q], [th])
    def rz(self, q, th): return self.add("rz", [q], [th])
    def cnot(self, c, t): return self.add("cnot", [c, t])
    def cx(self, c, t): return self.add("cnot", [c, t])
    def cz(self, a, b): return self.add("cz", [a, b])
    def swap(self, a, b): return self.add("swap", [a, b])
    def cphase(self, a, b, th): return self.add("cphase", [a, b], [th])

    def depth(self):
        level = [0] * self.n_qubits
        for _, qs, _ in self.ops:
            d = max(level[q] for q in qs) + 1
            for q in qs:
                level[q] = d
        return max(level) if self.ops else 0

    def __len__(self):
        return len(self.ops)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self):
        return {"n_qubits": self.n_qubits,
                "gates": [{"gate": n, "qubits": list(q), "params": list(p)}
                          for n, q, p in self.ops]}

    @classmethod
    def from_dict(cls, d):
        qc = cls(int(d["n_qubits"]))
        for g in d.get("gates", []):
            qc.add(g["gate"], g["qubits"], g.get("params", []))
        return qc

    def to_json(self, **kw):
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_text(cls, text, n_qubits=None):
        """Lines: 'qubits N' (optional), then '<gate> <q...> <angle...>'.
        '#' starts a comment. Example:  cphase 0 2 1.5708"""
        rows = []
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tok = line.split()
            if tok[0].lower() == "qubits":
                n_qubits = int(tok[1])
                continue
            rows.append(tok)
        if n_qubits is None:
            hi = 0
            for tok in rows:
                name = ALIASES.get(tok[0].lower(), tok[0].lower())
                if name not in GATE_SPEC:
                    raise ValueError(f"unknown gate '{tok[0]}'")
                nq, npar = GATE_SPEC[name]
                if len(tok) != 1 + nq + npar:
                    raise ValueError(f"bad arguments for '{tok[0]}': {' '.join(tok)}")
                try:
                    hi = max([hi] + [int(t) for t in tok[1:1 + nq]])
                except ValueError:
                    raise ValueError(f"qubit indices must be integers: {' '.join(tok)}")
            n_qubits = hi + 1
        qc = cls(n_qubits)
        for tok in rows:
            name = ALIASES.get(tok[0].lower(), tok[0].lower())
            if name not in GATE_SPEC:
                raise ValueError(f"unknown gate '{tok[0]}'")
            nq, npar = GATE_SPEC[name]
            if len(tok) != 1 + nq + npar:
                raise ValueError(f"bad arguments for '{tok[0]}': {' '.join(tok)}")
            qc.add(name, [int(t) for t in tok[1:1 + nq]], [float(t) for t in tok[1 + nq:]])
        return qc

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(self):
        rows = [f"q{i}: " for i in range(self.n_qubits)]
        pad = max(len(r) for r in rows)
        rows = [r.ljust(pad) for r in rows]
        for name, qs, ps in self.ops:
            label = name.upper()
            if ps:
                label += "(" + ",".join(f"{p:.2f}" for p in ps) + ")"
            w = max(len(label) + 2, 5)
            cells = ["─" * w] * self.n_qubits
            if len(qs) == 1:
                cells[qs[0]] = f"[{label}]".center(w, "─")
            else:
                a, b = qs
                lo, hi = min(a, b), max(a, b)
                for k in range(lo + 1, hi):
                    cells[k] = "┼".center(w, "─")
                if name == "cnot":
                    cells[a], cells[b] = "●".center(w, "─"), "⊕".center(w, "─")
                elif name == "cz":
                    cells[a] = cells[b] = "●".center(w, "─")
                elif name in ("swap", "iswap"):
                    cells[a] = cells[b] = ("x" if name == "swap" else "iSW").center(w, "─")
                else:  # cphase
                    cells[a] = "●".center(w, "─")
                    cells[b] = f"[{label}]".center(w, "─")
            rows = [r + c for r, c in zip(rows, cells)]
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _pick_backend(self, backend):
        if backend == "auto":
            return "statevector" if self.n_qubits <= 16 else "mps"
        if backend not in ("statevector", "mps", "density"):
            raise ValueError("backend must be 'auto', 'statevector', 'mps' or 'density'")
        return backend

    def _run_statevector(self):
        from core_engine.statevector import QuantumState
        if self.n_qubits > STATEVECTOR_LIMIT:
            raise MemoryError(
                f"{self.n_qubits} qubits is too many for the dense statevector "
                f"backend (limit {STATEVECTOR_LIMIT}). Use backend='mps'.")
        qs = QuantumState(self.n_qubits)
        for name, q, p in self.ops:
            if name == "h": qs.apply_h(q[0])
            elif name == "x": qs.apply_x(q[0])
            elif name == "y": qs.apply_y(q[0])
            elif name == "z": qs.apply_z(q[0])
            elif name == "s": qs.apply_s(q[0])
            elif name == "sdg": qs.apply_s_dag(q[0])
            elif name == "t": qs.apply_t(q[0])
            elif name == "tdg": qs.apply_tdg(q[0])
            elif name == "sx": qs.apply_sx(q[0])
            elif name == "p": qs.apply_p(q[0], p[0])
            elif name == "u3": qs.apply_u3(q[0], *p)
            elif name == "iswap": qs.apply_iswap_general(q[0], q[1])
            elif name == "rx": qs.apply_rx(q[0], p[0])
            elif name == "ry": qs.apply_ry(q[0], p[0])
            elif name == "rz": qs.apply_rz(q[0], p[0])
            elif name == "cnot": qs.apply_cnot_general(q[0], q[1])
            elif name == "cz": qs.apply_cz_general(q[0], q[1])
            elif name == "swap": qs.apply_swap_general(q[0], q[1])
            elif name == "cphase": qs.apply_cphase_general(q[0], q[1], p[0])
        return qs

    def _run_mps(self, max_bond_dim=None):
        from core_engine.mps_engine import MPSState
        if self.n_qubits > MPS_LIMIT:
            raise ValueError(f"MPS backend limited to {MPS_LIMIT} qubits")
        m = MPSState(self.n_qubits, max_bond_dim=max_bond_dim)
        for name, q, p in self.ops:
            getattr(m, {"sdg": "sdg", "cnot": "cnot"}.get(name, name))(*q, *p)
        return m

    def statevector(self):
        """Dense final state (small n only)."""
        return self._run_statevector().get_statevector()

    def run(self, shots=1024, backend="auto", seed=None, max_bond_dim=None,
            noise=None):
        """Execute and return {'counts', 'backend', 'shots', ...extras}."""
        if shots < 1:
            raise ValueError("shots must be >= 1")
        backend = "density" if noise is not None else self._pick_backend(backend)
        rng = np.random.default_rng(seed)
        if backend == "statevector":
            qs = self._run_statevector()
            probs = qs.probabilities()
            probs = probs / probs.sum()
            draws = rng.choice(len(probs), size=shots, p=probs)
            counts = {}
            for d in draws:
                key = format(int(d), f"0{self.n_qubits}b")
                counts[key] = counts.get(key, 0) + 1
            return {"backend": "statevector", "shots": shots, "counts": counts}
        if backend == "mps":
            m = self._run_mps(max_bond_dim)
            return {"backend": "mps", "shots": shots,
                    "counts": m.sample(shots, seed=seed),
                    "max_bond_dim_used": m.max_bond(),
                    "truncation_error": m.truncation_error,
                    "mps_memory_bytes": m.memory_bytes()}
        # density-matrix backend with pulse-derived noise
        from noise_channel.pulse_noise import run_noisy_circuit
        return run_noisy_circuit(self, noise, shots=shots, seed=seed)

    def __repr__(self):
        return f"QuantumCircuit({self.n_qubits} qubits, {len(self.ops)} gates, depth {self.depth()})"
