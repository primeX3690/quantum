"""
main.py — FastAPI wrapper for my_quantum_core.
Run with: uvicorn api.main:app --reload
Then visit http://localhost:8000/docs for interactive API testing.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_engine.statevector import QuantumState
from algorithms.grover_search import run_grover, optimal_iterations

app = FastAPI(title="my_quantum_core API", version="1.0")


class GroverRequest(BaseModel):
    n_qubits: int
    target_index: int


@app.get("/")
def root():
    return {"status": "my_quantum_core API is running", "endpoints": ["/run/bell", "/run/grover"]}


@app.post("/run/bell")
def run_bell():
    qs = QuantumState(n_qubits=2)
    qs.apply_h(0)
    qs.apply_cnot(0, 1)
    return {"result": qs.measure_probabilities_dict()}


@app.post("/run/grover")
def run_grover_endpoint(req: GroverRequest):
    iters = optimal_iterations(req.n_qubits)
    result = run_grover(n_qubits=req.n_qubits, target_index=req.target_index, iterations=iters)
    return {"target_index": req.target_index, "iterations_used": iters, "result": result}