"""
orb/decode.py
-------------
Convert L1 solver output (x_hat, residuals) into human-readable dicts.

decode(x_hat, residuals, compiled, graph, threshold=5.0) -> dict
  nodes    — [{id, qty}]
  edges    — [{id, from, to, flow}]
  sinks    — [{id, sink}]  (only nodes whose sink variable is in state vector)
  flagged  — [{claim_id, residual, source, type}]  claims with |residual| > threshold
"""

from __future__ import annotations
import numpy as np


def decode(
    x_hat: np.ndarray,
    residuals: np.ndarray,
    compiled: dict,
    graph: dict,
    threshold: float = 5.0,
) -> dict:
    """
    Parameters
    ----------
    x_hat       State vector returned by the solver.
    residuals   y - H @ x_hat  (one entry per claim).
    compiled    Output of orb.compile.compile().
    graph       Original graph dict (for claim metadata).
    threshold   |residual| above this → flagged.

    Returns
    -------
    dict with keys: nodes, edges, sinks, flagged.
    """
    idx       = compiled["index"]
    claim_ids = compiled["claim_ids"]

    # Build a fast claim lookup
    claims_by_id = {c["id"]: c for c in graph.get("claims", [])}

    # ── Node quantities ──────────────────────────────────────────────────────
    nodes_out = []
    for n in graph["nodes"]:
        col = idx[f"qty_{n['id']}"]
        nodes_out.append({"id": n["id"], "qty": _r(x_hat[col])})

    # ── Edge flows ───────────────────────────────────────────────────────────
    edges_out = []
    for e in graph["edges"]:
        col = idx[f"flow_{e['id']}"]
        edges_out.append({
            "id":   e["id"],
            "from": e["from"],
            "to":   e["to"],
            "flow": _r(x_hat[col]),
        })

    # ── Sink amounts (unknown leaks / losses) ────────────────────────────────
    sinks_out = []
    for n in graph["nodes"]:
        key = f"sink_{n['id']}"
        if key in idx:
            col = idx[key]
            sinks_out.append({"id": n["id"], "sink": _r(x_hat[col])})

    # ── Flagged claims ───────────────────────────────────────────────────────
    flagged = []
    for cid, r in zip(claim_ids, residuals):
        if abs(r) > threshold:
            claim = claims_by_id.get(cid, {})
            flagged.append({
                "claim_id": cid,
                "residual": _r(r),
                "source":   claim.get("source", "?"),
                "type":     claim.get("type", "?"),
            })

    # Sort flagged by descending |residual|
    flagged.sort(key=lambda f: abs(f["residual"]), reverse=True)

    return {
        "nodes":   nodes_out,
        "edges":   edges_out,
        "sinks":   sinks_out,
        "flagged": flagged,
    }


def _r(v) -> float:
    """Round to 4 decimal places."""
    return round(float(v), 4)
