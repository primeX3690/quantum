"""
circuit_tracer.py
--------------------
Simple ASCII circuit diagram printer for the terminal — renders a
text-based visualization of gate sequences, built independently.
"""

class CircuitTracer:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gate_log = []

    def log_gate(self, gate_name, qubits):
        self.gate_log.append((gate_name, qubits))

    def draw(self):
        lines = [f"q{i}: " for i in range(self.n_qubits)]
        for gate_name, qubits in self.gate_log:
            col_width = max(len(gate_name) + 2, 5)
            for q in range(self.n_qubits):
                if q in qubits:
                    if gate_name == "CNOT":
                        symbol = "●" if q == qubits[0] else "⊕"
                    else:
                        symbol = f"[{gate_name}]"
                    lines[q] += symbol.center(col_width, "─")
                else:
                    lines[q] += "─" * col_width
        print("\n".join(lines))