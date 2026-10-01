"""Circuit builder, formats, backends."""
import numpy as np
from core_engine.circuit import QuantumCircuit


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


def test_fluent_build_and_run():
    qc = QuantumCircuit(3).h(0).cnot(0, 1).cnot(1, 2)
    assert len(qc) == 3 and qc.depth() == 3
    counts = qc.run(shots=500, seed=1)["counts"]
    assert set(counts) == {"000", "111"}


def test_json_and_text_roundtrip():
    qc = QuantumCircuit(3).h(0).rx(1, 0.5).cphase(0, 2, 1.2).swap(1, 2)
    assert QuantumCircuit.from_json(qc.to_json()).ops == qc.ops
    txt = "qubits 3\nh 0\nrx 1 0.5  # comment\ncphase 0 2 1.2\nswap 1 2\n"
    assert QuantumCircuit.from_text(txt).ops == qc.ops


def test_validation_errors():
    assert _raises(lambda: QuantumCircuit(2).add("nope", [0]), ValueError)
    assert _raises(lambda: QuantumCircuit(2).h(5), ValueError)
    assert _raises(lambda: QuantumCircuit(2).cnot(1, 1), ValueError)
    assert _raises(lambda: QuantumCircuit(2).add("rx", [0]), ValueError)
    assert _raises(lambda: QuantumCircuit.from_text("h"), ValueError)


def test_statevector_and_mps_agree():
    import random
    random.seed(9)
    qc = QuantumCircuit(6)
    for _ in range(50):
        a, b = random.sample(range(6), 2)
        qc.h(a).cnot(a, b).ry(b, random.uniform(0, 3))
    sv = qc.statevector()
    mp = qc._run_mps().to_statevector()
    assert abs(abs(np.vdot(sv, mp)) - 1) < 1e-9


def test_auto_backend_switches_to_mps_and_scales():
    qc = QuantumCircuit(80).h(0)
    for i in range(79):
        qc.cnot(i, i + 1)
    r = qc.run(shots=20, seed=1)
    assert r["backend"] == "mps"
    assert set(r["counts"]) <= {"0" * 80, "1" * 80}


def test_statevector_backend_guard():
    assert _raises(lambda: QuantumCircuit(30).h(0).run(backend="statevector"), MemoryError)


def test_draw_contains_gates():
    art = QuantumCircuit(2).h(0).cnot(0, 1).draw()
    assert "[H]" in art and "●" in art and "⊕" in art


def test_toffoli_truth_table_all_backends():
    from core_engine.custom_gates import gate_CCX
    for bits in range(8):
        qc = QuantumCircuit(3)
        for i in range(3):
            if (bits >> (2 - i)) & 1:
                qc.x(i)
        qc.ccx(0, 1, 2)
        expected = gate_CCX() @ np.eye(8)[bits]
        assert abs(abs(np.vdot(expected, qc.statevector())) - 1) < 1e-9
        assert abs(abs(np.vdot(expected, qc._run_mps().to_statevector())) - 1) < 1e-9
    assert _raises(lambda: QuantumCircuit(3).ccx(0, 1, 1), ValueError)


def test_new_gates_statevector_mps_and_noise_agree():
    from noise_channel.pulse_noise import PulseNoiseModel
    qc = QuantumCircuit(3).h(0).h(1).ccx(0, 2, 1).sx(2).p(0, 0.3).u3(1, 0.4, 0.5, 0.6) \
        .iswap(0, 2).tdg(1)
    assert abs(abs(np.vdot(qc.statevector(), qc._run_mps().to_statevector())) - 1) < 1e-9
    r = qc.run(shots=10, noise=PulseNoiseModel.from_profile("ideal"), seed=0)
    assert r["fidelity_vs_ideal"] > 1 - 1e-9
