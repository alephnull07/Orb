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

    # Prefer compile-time (possibly coalesced) claims; fall back to the graph.
    claims_by_id: dict[str, dict] = {}
    for c in graph.get("claims", []) or []:
        cid = c.get("id")
        if cid is not None:
            claims_by_id[cid] = c
    for c in compiled.get("claims") or []:
        cid = c.get("id")
        if cid is not None:
            claims_by_id[cid] = c

    n_x = len(x_hat)

    # ── Node quantities ──────────────────────────────────────────────────────
    nodes_out = []
    for n in graph.get("nodes") or []:
        col = idx.get(f"qty_{n['id']}")
        if col is None or col >= n_x:
            continue
        nodes_out.append({"id": n["id"], "qty": _r(x_hat[col])})

    # ── Edge flows ───────────────────────────────────────────────────────────
    edges_out = []
    for e in graph.get("edges") or []:
        col = idx.get(f"flow_{e['id']}")
        if col is None or col >= n_x:
            continue
        edges_out.append({
            "id":   e["id"],
            "from": e["from"],
            "to":   e["to"],
            "flow": _r(x_hat[col]),
        })

    # ── Sink amounts (unknown leaks / losses) ────────────────────────────────
    sinks_out = []
    for n in graph.get("nodes") or []:
        key = f"sink_{n['id']}"
        col = idx.get(key)
        if col is None or col >= n_x:
            continue
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
