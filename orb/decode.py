"""
orb/decode.py
-------------
Convert L1 solver output (x_hat, residuals) into human-readable dicts
and classify leftover residual as noise, loss, or corruption.

decode(x_hat, residuals, compiled, graph) -> dict
  nodes       — [{id, qty}]
  edges       — [{id, from, to, flow}]
  sinks       — [{id, sink, status}]  status: loss | ambiguous | estimate
  loss        — sinks we are willing to name as physical loss
  ambiguous   — unmetered sinks when correctable_k == 0
                ({loss, downward corruption} cannot be told apart)
  flagged     — corruption: |residual| > max(3σ, deadband)
  noise       — claims whose residual is within the noise band
  sigma       — robust residual scale (MAD)
  threshold   — flag cutoff used
  undetectable— hops with no third channel (coordinated-lie blind)
"""

from __future__ import annotations

import numpy as np


def robust_sigma(residuals) -> float:
    """Scale of inlier residuals. 0 when everything is exact.

    L1 inliers sit at 0. Scale from the shortest half of |r| so a 50/50
    split like [0, 0, 320, 320] does not pull the median off zero and
    swallow the sparse residuals as 'noise'.
    """
    r = np.asarray(residuals, dtype=float).ravel()
    if r.size == 0:
        return 0.0
    finite = r[np.isfinite(r)]
    if finite.size == 0:
        return 0.0
    abs_r = np.abs(finite)
    half = max(1, (finite.size + 1) // 2)
    smallest = np.partition(abs_r, half - 1)[:half]
    sig = 1.4826 * float(np.median(smallest))
    if sig < 1e-12:
        return 0.0
    inliers = finite[abs_r <= max(3.0 * sig, 0.5)]
    if inliers.size >= 2:
        mad2 = float(np.median(np.abs(inliers)))
        sig = 1.4826 * mad2
    return float(sig)


def flag_threshold(residuals, override: float | None = None) -> tuple[float, float]:
    """Return (sigma, threshold). override forces an absolute cutoff."""
    sigma = robust_sigma(residuals)
    if override is not None:
        return sigma, float(override)
    # Exact conservation → any discrepancy above numerical deadband is sparse a.
    # Otherwise 3σ. Floor 0.5 so 1e-10 solver noise is not flagged.
    if sigma < 1e-12:
        return 0.0, 0.5
    return sigma, max(3.0 * sigma, 0.5)


def decode(
    x_hat: np.ndarray,
    residuals: np.ndarray,
    compiled: dict,
    graph: dict,
    threshold: float | None = None,
) -> dict:
    """
    Parameters
    ----------
    x_hat       State vector returned by the solver.
    residuals   y - H @ x_hat  (one entry per compiled claim).
    compiled    Output of orb.compile.compile().
    graph       Original graph dict (for claim metadata).
    threshold   If set, |residual| above this → flagged.  If None, use 3σ.
    """
    idx       = compiled["index"]
    claim_ids = compiled["claim_ids"]
    report    = compiled.get("report") or {}
    residuals = np.asarray(residuals, dtype=float).ravel()

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
    sigma, thresh = flag_threshold(residuals, override=threshold)
    T = int(report.get("T") or 1)
    k = int(report.get("correctable_k") or 0)
    metered = set(report.get("metered_sink_ids") or [])
    qty_obs = report.get("qty_obs_counts") or {}

    nodes_out = []
    for n in graph.get("nodes") or []:
        col = idx.get(f"qty_{n['id']}")
        if col is None or col >= n_x:
            continue
        nodes_out.append({"id": n["id"], "qty": _r(x_hat[col])})

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

    sinks_out = []
    loss_out = []
    ambiguous_out = []
    for n in graph.get("nodes") or []:
        key = f"sink_{n['id']}"
        col = idx.get(key)
        if col is None or col >= n_x:
            continue
        mag = _r(x_hat[col])
        nid = n["id"]
        two_type = nid in metered and int(qty_obs.get(nid, 0) or 0) >= 1
        can_name = two_type or (T >= 3 and k >= 1)
        if abs(mag) <= thresh:
            status = "estimate"
        elif can_name:
            status = "loss"
        else:
            status = "ambiguous"
        row = {"id": nid, "sink": mag, "status": status}
        sinks_out.append(row)
        if status == "loss":
            loss_out.append(row)
        elif status == "ambiguous":
            ambiguous_out.append({
                **row,
                "hypotheses": ["loss", "downward_corruption"],
                "reason": (
                    "unmetered unknown sink: a leak and a low qty/EOD lie "
                    "are the same residual signature"
                ),
            })

    flagged = []
    noise = []
    for cid, r in zip(claim_ids, residuals):
        claim = claims_by_id.get(cid, {})
        entry = {
            "claim_id": cid,
            "residual": _r(r),
            "source":   claim.get("source", "?"),
            "type":     claim.get("type", "?"),
        }
        if abs(r) > thresh:
            entry["kind"] = "corruption"
            flagged.append(entry)
        else:
            entry["kind"] = "noise"
            noise.append(entry)
    flagged.sort(key=lambda f: abs(f["residual"]), reverse=True)

    return {
        "nodes":         nodes_out,
        "edges":         edges_out,
        "sinks":         sinks_out,
        "loss":          loss_out,
        "ambiguous":     ambiguous_out,
        "flagged":       flagged,
        "noise":         noise,
        "sigma":         _r(sigma),
        "threshold":     _r(thresh),
        "undetectable":  list(report.get("blind_edges") or []),
        "correctable_k": k,
        "T":             T,
    }


def _r(v) -> float:
    """Round to 4 decimal places."""
    return round(float(v), 4)
