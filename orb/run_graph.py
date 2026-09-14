"""
orb/run_graph.py
----------------
Full pipeline: graph JSON path → compile → L1 → decode.

Usage:
    python -m orb.run_graph path/to/graph.json

The L1 LP here supports hard balance constraints (A_eq x = b_eq), which
the existing estimator.py does not.  This module never calls estimator.py.

LP formulation
--------------
Variables: x = state vector (n_vars),  t = residual slacks (m_claims)

  minimize    w_sink @ x[:n]  +  w @ t
  subject to:
     H x - t  ≤  y          (residual upper bound)
    -H x - t  ≤ -y          (residual lower bound)
    A_eq x     = b_eq        (hard balance)
    t          ≥ 0
    sink cols of x ≥ 0       (leaks are non-negative)

Solved with scipy.optimize.linprog(method="highs").
"""

from __future__ import annotations
import json
import sys

import numpy as np
from scipy.optimize import linprog

from .compile import compile as compile_graph
from .decode  import decode


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_graph(graph_path: str) -> dict:
    """
    Parameters
    ----------
    graph_path  Path to a graph JSON file (see orb/compile.py for schema).

    Returns
    -------
    {
        "x":         list[float]   state vector
        "residuals": list[float]   y - H @ x_hat
        "decoded":   dict          human-readable nodes / edges / sinks / flagged
        "report":    dict          identifiability summary from compile()
    }
    """
    with open(graph_path) as fh:
        graph = json.load(fh)

    compiled  = compile_graph(graph)
    x_hat, residuals = _l1_solve(compiled)
    decoded   = decode(x_hat, residuals, compiled, graph)

    return {
        "x":         x_hat.tolist(),
        "residuals": residuals.tolist(),
        "decoded":   decoded,
        "report":    compiled["report"],
    }


# ---------------------------------------------------------------------------
# Internal LP solver
# ---------------------------------------------------------------------------

def _l1_solve(compiled: dict) -> tuple[np.ndarray, np.ndarray]:
    """Solve the L1 LP with hard balance constraints. Returns (x_hat, residuals)."""
    H      = compiled["H"]
    y      = compiled["y"]
    w      = compiled["w"]
    w_sink = compiled["w_sink"]
    A_eq   = compiled["A_eq"]
    b_eq   = compiled["b_eq"]
    idx    = compiled["index"]

    m, n = H.shape
    p    = A_eq.shape[0]

    # Objective: sink penalties on state vars + slack costs on residuals
    c_obj = np.concatenate([w_sink, w])

    # Inequality: H x - t ≤ y  and  -H x - t ≤ -y
    A_ub = np.block([
        [ H, -np.eye(m)],
        [-H, -np.eye(m)],
    ])
    b_ub = np.concatenate([y, -y])

    # Equality: A_eq x = b_eq  (pad t block with zeros)
    A_eq_full = np.hstack([A_eq, np.zeros((p, m))])

    # Bounds: sink cols ≥ 0, others unrestricted; t ≥ 0
    sink_col_set = {col for name, col in idx.items() if name.startswith("sink_")}
    x_bounds = [
        (0.0, None) if c in sink_col_set else (None, None)
        for c in range(n)
    ]
    bounds = x_bounds + [(0.0, None)] * m

    result = linprog(
        c_obj,
        A_ub=A_ub,   b_ub=b_ub,
        A_eq=A_eq_full, b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if result.status != 0:
        raise RuntimeError(
            f"LP solver failed (status {result.status}): {result.message}"
        )

    x_hat    = result.x[:n]
    residuals = y - H @ x_hat
    return x_hat, residuals


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m orb.run_graph <graph.json>", file=sys.stderr)
        sys.exit(1)

    result = run_graph(sys.argv[1])
    report  = result["report"]
    decoded = result["decoded"]

    print(f"\n{'='*56}")
    print(f"  Identifiable: {report['identifiable']}   "
          f"rank {report['rank']}/{report['n_vars']}   "
          f"correctable k={report['correctable_k']}")
    print(f"{'='*56}")

    print("\nNodes:")
    for nd in decoded["nodes"]:
        print(f"  {nd['id']:<12}  qty = {nd['qty']:.2f}")

    print("\nEdges:")
    for ed in decoded["edges"]:
        print(f"  {ed['from']:<10} → {ed['to']:<10}  flow = {ed['flow']:.2f}")

    if decoded["sinks"]:
        print("\nSinks (unknown losses):")
        for sk in decoded["sinks"]:
            print(f"  {sk['id']:<12}  sink = {sk['sink']:.4f}")

    if decoded["flagged"]:
        print("\nFlagged claims (|residual| > 0.5):")
        for f in decoded["flagged"]:
            print(f"  [{f['claim_id']}] {f['source']}  r={f['residual']:.2f}")
    else:
        print("\nNo claims flagged.")
