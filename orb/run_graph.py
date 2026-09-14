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
Variables: x (n),  t+ (m),  t- (m)

  minimize    w_sink @ x  +  w @ (t+ + t-)
  subject to:
     H x + t+ - t-  =  y     (residual r = t+ - t- = y - Hx)
     A_eq x         =  b_eq  (hard conservation)
     t+, t-         >= 0
     sink cols of x >= 0
     all x bounded  in [-M, M] so rank-deficient graphs stay bounded

Solved with scipy.optimize.linprog(method="highs").
If the hard-balance LP is infeasible or unbounded, balance is relaxed
with a large slack penalty and (last resort) a least-squares fallback.
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
    H      = np.asarray(compiled["H"], dtype=float)
    y      = np.asarray(compiled["y"], dtype=float)
    w      = np.asarray(compiled["w"], dtype=float)
    w_sink = np.asarray(compiled["w_sink"], dtype=float)
    A_eq   = np.asarray(compiled["A_eq"], dtype=float)
    b_eq   = np.asarray(compiled["b_eq"], dtype=float)
    idx    = compiled["index"]

    if H.ndim != 2:
        H = H.reshape(len(y), -1)
    m, n = H.shape
    if A_eq.size == 0:
        A_eq = np.zeros((0, n))
        b_eq = np.zeros(0)

    if n == 0:
        return np.zeros(0), np.zeros(m)

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    b_eq = np.nan_to_num(b_eq, nan=0.0, posinf=0.0, neginf=0.0)
    w_sink = np.nan_to_num(w_sink, nan=0.0)

    sink_col_set = {col for name, col in idx.items() if name.startswith("sink_")}

    x_hat = _solve_l1_lp(H, y, w, A_eq, b_eq, w_sink, sink_col_set, soft_balance=False)
    if x_hat is None:
        x_hat = _solve_l1_lp(H, y, w, A_eq, b_eq, w_sink, sink_col_set, soft_balance=True)
    if x_hat is None:
        x_hat = _lstsq_fallback(H, y, A_eq, b_eq, sink_col_set)

    residuals = y - H @ x_hat if m else np.zeros(0)
    return x_hat, residuals


def _bound_M(y: np.ndarray, b_eq: np.ndarray) -> float:
    mag = [abs(float(v)) for v in np.concatenate([y.ravel(), b_eq.ravel()]) if np.isfinite(v)]
    peak = max(mag) if mag else 1.0
    return max(1e6, 1e3 * peak)


def _solve_l1_lp(
    H, y, w, A_eq, b_eq, w_sink, sink_col_set, *, soft_balance: bool,
) -> np.ndarray | None:
    m, n = H.shape
    p = A_eq.shape[0]
    M = _bound_M(y, b_eq)

    # x, t+, t- [, s+, s-]
    n_slack_bal = p if soft_balance else 0
    n_tot = n + 2 * m + 2 * n_slack_bal

    c_obj = np.zeros(n_tot)
    c_obj[:n] = w_sink
    if m:
        c_obj[n:n + m] = w
        c_obj[n + m:n + 2 * m] = w
    if n_slack_bal:
        penalty = 1e6 * max(float(np.max(w)) if m else 1.0, 1.0)
        c_obj[n + 2 * m:] = penalty

    eq_rows = []
    eq_rhs = []
    if m:
        # H x + t+ - t- = y
        row = np.zeros((m, n_tot))
        row[:, :n] = H
        row[:, n:n + m] = np.eye(m)
        row[:, n + m:n + 2 * m] = -np.eye(m)
        eq_rows.append(row)
        eq_rhs.append(y)
    if p:
        # A_eq x [+ s+ - s-] = b_eq
        row = np.zeros((p, n_tot))
        row[:, :n] = A_eq
        if n_slack_bal:
            row[:, n + 2 * m:n + 2 * m + p] = np.eye(p)
            row[:, n + 2 * m + p:] = -np.eye(p)
        eq_rows.append(row)
        eq_rhs.append(b_eq)

    if eq_rows:
        A_full = np.vstack(eq_rows)
        b_full = np.concatenate(eq_rhs)
    else:
        A_full = None
        b_full = None

    bounds = []
    for i in range(n):
        if i in sink_col_set:
            bounds.append((0.0, M))
        else:
            bounds.append((-M, M))
    bounds.extend([(0.0, None)] * (2 * m + 2 * n_slack_bal))

    result = linprog(
        c_obj,
        A_eq=A_full,
        b_eq=b_full,
        bounds=bounds,
        method="highs",
        options={"presolve": True, "time_limit": 60},
    )
    if result.status != 0 or result.x is None:
        return None
    return np.asarray(result.x[:n], dtype=float)


def _lstsq_fallback(H, y, A_eq, b_eq, sink_col_set) -> np.ndarray:
    n = H.shape[1]
    blocks, rhs = [], []
    if H.shape[0]:
        blocks.append(H)
        rhs.append(y)
    if A_eq.shape[0]:
        blocks.append(1e3 * A_eq)
        rhs.append(1e3 * b_eq)
    if not blocks:
        x = np.zeros(n)
    else:
        x, *_ = np.linalg.lstsq(np.vstack(blocks), np.concatenate(rhs), rcond=None)
    for i in sink_col_set:
        if 0 <= i < n:
            x[i] = max(0.0, x[i])
    return x


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
