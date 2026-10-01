"""
main.py -- FastAPI wrapper for my_quantum_core.
Run with:  uvicorn api.main:app --reload
Docs at http://localhost:8000/docs

Endpoints
  POST /run/bell      quick Bell-state demo
  POST /run/grover    Grover search
  POST /run/circuit   ANY circuit (JSON gate list), statevector / mps / noisy
  POST /run/shor      factor N with simulated Shor
  POST /run/qec       ancilla-syndrome error-correction cycle / Monte Carlo
  GET  /gates         list supported gates
All inputs are validated and size-capped so the server cannot be asked to
allocate unbounded memory.
"""
from typing import List, Optional
import sys
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_engine.statevector import QuantumState
from core_engine.circuit import QuantumCircuit, GATE_SPEC
from algorithms.grover_search import run_grover, optimal_iterations

app = FastAPI(title="my_quantum_core API", version="2.0")

MAX_SHOTS = 100_000
MAX_GATES = 20_000


class GroverRequest(BaseModel):
    n_qubits: int = Field(ge=1, le=20)
    target_index: int = Field(ge=0)


class GateModel(BaseModel):
    gate: str
    qubits: List[int]
    params: List[float] = []


class CircuitRequest(BaseModel):
    n_qubits: int = Field(ge=1, le=500)
    gates: List[GateModel]
    shots: int = Field(default=1024, ge=1, le=MAX_SHOTS)
    backend: str = "auto"
    seed: Optional[int] = None
    max_bond_dim: Optional[int] = Field(default=None, ge=1, le=1024)
    noise_profile: Optional[str] = None   # ideal | superconducting | noisy_nisq


class ShorRequest(BaseModel):
    N: int = Field(ge=4, le=255)
    a: int = Field(default=2, ge=2)


class QECRequest(BaseModel):
    code: str = "bit"
    error_qubit: Optional[int] = Field(default=None, ge=0, le=2)
    theta: float = 1.0
    p_physical: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    trials: int = Field(default=300, ge=1, le=5000)


@app.get("/")
def root():
    return {"status": "my_quantum_core API is running", "version": "2.0",
            "endpoints": ["/run/bell", "/run/grover", "/run/circuit",
                          "/run/shor", "/run/qec", "/gates"]}


@app.get("/gates")
def gates():
    return {g: {"qubits": nq, "params": npar} for g, (nq, npar) in GATE_SPEC.items()}


@app.post("/run/bell")
def run_bell():
    qs = QuantumState(n_qubits=2)
    qs.apply_h(0)
    qs.apply_cnot(0, 1)
    return {"result": qs.measure_probabilities_dict()}


@app.post("/run/grover")
def run_grover_endpoint(req: GroverRequest):
    if req.target_index >= 2 ** req.n_qubits:
        raise HTTPException(400, "target_index out of range for n_qubits")
    iters = optimal_iterations(req.n_qubits)
    result = run_grover(n_qubits=req.n_qubits, target_index=req.target_index, iterations=iters)
    return {"target_index": req.target_index, "iterations_used": iters, "result": result}


@app.post("/run/circuit")
def run_circuit(req: CircuitRequest):
    if len(req.gates) > MAX_GATES:
        raise HTTPException(400, f"too many gates (max {MAX_GATES})")
    try:
        qc = QuantumCircuit.from_dict(req.model_dump() if hasattr(req, "model_dump") else req.dict())
        noise = None
        if req.noise_profile:
            from noise_channel.pulse_noise import PulseNoiseModel
            noise = PulseNoiseModel.from_profile(req.noise_profile)
        res = qc.run(shots=req.shots, backend=req.backend, seed=req.seed,
                     max_bond_dim=req.max_bond_dim, noise=noise)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    except MemoryError as e:
        raise HTTPException(413, str(e))
    res["depth"] = qc.depth()
    res["gate_count"] = len(qc)
    return res


@app.post("/run/shor")
def run_shor(req: ShorRequest):
    from algorithms.shor import find_factors_via_shor
    try:
        f = find_factors_via_shor(req.N, a=req.a, verbose=False)
    except MemoryError as e:
        raise HTTPException(413, str(e))
    return {"N": req.N, "factors": list(f) if f else None}


@app.post("/run/qec")
def run_qec(req: QECRequest):
    from algorithms.error_correction import run_qec_cycle, monte_carlo_logical_error
    if req.code not in ("bit", "phase"):
        raise HTTPException(400, "code must be 'bit' or 'phase'")
    if req.p_physical is not None:
        return monte_carlo_logical_error(req.p_physical, trials=req.trials,
                                         code=req.code, theta=req.theta)
    return run_qec_cycle(req.theta, req.error_qubit, req.code)
