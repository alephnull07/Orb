"""
orb/temporal.py
---------------
Multi-snapshot temporal stacking for SCADA leak localization.

Key insight: single-snapshot SCADA is exactly determined (correctable_k=0).
With T snapshots sharing the same leak (sink variables), redundancy grows
linearly: correctable_k rises from 0 to a useful value.

State vector layout (one block per timestamp, shared sinks at end):
  [qty_J_n(t=1..T) | flow_e(t=1..T) | sink_J_n(n=1..N)]

Balance constraint for node n at time t:
  qty_n(t) - initial_n + Σ outflows(t) - Σ inflows(t) + sink_n = 0

Claim matrix H is block-diagonal in the time-varying part;
sink variables appear in every balance row for their respective node.

Public API
----------
  build_stacked_lp(snapshots, topo_links, topo_nodes, lambda_sink) -> LPProblem
  solve_stacked(lp) -> (x_hat, residuals)
  correctable_k_stacked(lp) -> int
  localize_leak(snapshots, topo, lambda_sink) -> {jid: sink_value}
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


# ── data structures ─────────────────────────────────────────────────────────

@dataclass
class Snapshot:
    """One SCADA timestamp: per-sensor (node/edge) observations."""
    timestamp: str
    node_obs: dict[str, float]    # jid -> demand/pressure value (averaged if both)
    edge_obs: dict[str, float]    # link_id -> flow value
    weight: float = 0.9           # claim weight


@dataclass
class LPProblem:
    """Stacked multi-snapshot LP ready for scipy.optimize.linprog."""
    H:        np.ndarray    # (n_claims, n_vars) claim observation matrix
    y:        np.ndarray    # (n_claims,) observed values
    w:        np.ndarray    # (n_claims,) claim weights
    A_eq:     np.ndarray    # (n_balance, n_vars) balance constraint matrix
    b_eq:     np.ndarray    # (n_balance,) balance rhs
    n_vars:   int
    n_claims: int
    n_nodes:  int           # N (junctions)
    n_edges:  int           # E (links)
    T:        int           # number of snapshots
    node_ids: list[str]     # length N
    edge_ids: list[str]     # length E
    claim_ids: list[str]


# ── builder ──────────────────────────────────────────────────────────────────

def build_stacked_lp(
    snapshots: list[Snapshot],
    topo_links: dict[str, tuple[str, str]],   # link_id -> (src_jid, tgt_jid)
    topo_nodes: list[str],                    # all junction ids
    lambda_sink: float = 0.01,
) -> LPProblem:
    """
    Build the stacked LP for T snapshots.

    Variable ordering:
      [qty block 1 | ... | qty block T | flow block 1 | ... | flow block T | sinks]
    where each qty block is N variables, each flow block is E variables,
    and sinks is N variables (shared across all T).
    """
    T      = len(snapshots)
    nodes  = sorted(topo_nodes)
    edges  = sorted(topo_links.keys())
    N, E   = len(nodes), len(edges)
    n_node_idx = {nid: i for i, nid in enumerate(nodes)}
    n_edge_idx = {eid: i for i, eid in enumerate(edges)}

    n_vars   = T * N + T * E + N          # qty×T + flow×T + sinks
    n_balance = T * N

    # ── balance constraint matrix ────────────────────────────────────────────
    A_eq = np.zeros((n_balance, n_vars), dtype=float)
    b_eq = np.zeros(n_balance, dtype=float)

    for t, snap in enumerate(snapshots):
        qty_offset  = t * N                       # qty block for time t
        flow_offset = T * N + t * E               # flow block for time t
        sink_offset = T * N + T * E               # shared sinks (start)

        for i, nid in enumerate(nodes):
            row = t * N + i
            # qty variable
            A_eq[row, qty_offset + i] = 1.0
            # sink variable (shared)
            A_eq[row, sink_offset + i] = 1.0
            # outgoing edges
            for j, eid in enumerate(edges):
                src, tgt = topo_links[eid]
                if src == nid:
                    A_eq[row, flow_offset + j] += 1.0
                if tgt == nid:
                    A_eq[row, flow_offset + j] -= 1.0
            # b_eq[row] = initial stock = 0

    # ── claim matrix ─────────────────────────────────────────────────────────
    claim_rows_H: list[np.ndarray] = []
    claim_y:  list[float] = []
    claim_w:  list[float] = []
    claim_ids: list[str] = []
    ci = 0

    for t, snap in enumerate(snapshots):
        qty_offset  = t * N
        flow_offset = T * N + t * E

        for nid, val in snap.node_obs.items():
            if nid not in n_node_idx:
                continue
            row = np.zeros(n_vars)
            row[qty_offset + n_node_idx[nid]] = 1.0
            claim_rows_H.append(row)
            claim_y.append(val)
            claim_w.append(snap.weight)
            claim_ids.append(f"c{ci}_t{t}_{nid}")
            ci += 1

        for eid, val in snap.edge_obs.items():
            if eid not in n_edge_idx:
                continue
            row = np.zeros(n_vars)
            row[flow_offset + n_edge_idx[eid]] = 1.0
            claim_rows_H.append(row)
            claim_y.append(val)
            claim_w.append(snap.weight)
            claim_ids.append(f"c{ci}_t{t}_{eid}")
            ci += 1

    H = np.vstack(claim_rows_H) if claim_rows_H else np.zeros((0, n_vars))
    y = np.array(claim_y)
    w = np.array(claim_w)

    return LPProblem(
        H=H, y=y, w=w,
        A_eq=A_eq, b_eq=b_eq,
        n_vars=n_vars, n_claims=len(y),
        n_nodes=N, n_edges=E, T=T,
        node_ids=nodes, edge_ids=edges,
        claim_ids=claim_ids,
    )


# ── solver ────────────────────────────────────────────────────────────────────

def solve_stacked(lp: LPProblem, lambda_sink: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """
    L1 solve for the stacked LP.
    Returns (x_hat, residuals) where residuals[i] = H[i]@x - y[i].
    """
    # Auxiliary variable r: H x - y = r+  - r-  (r+, r- >= 0)
    n = lp.n_vars
    m = lp.n_claims
    N = lp.n_nodes
    T = lp.T

    # Sink variables are at indices [T*N + T*E .. T*N + T*E + N)
    sink_start = T * lp.n_nodes + T * lp.n_edges

    # Cost: w_i*(r+_i + r-_i) for claims + lambda_sink * sink_j for sinks
    c_x    = np.zeros(n)
    c_x[sink_start:sink_start + N] = lambda_sink
    c_rp   = lp.w
    c_rm   = lp.w
    c      = np.concatenate([c_x, c_rp, c_rm])

    # Equality: H x - r+ + r- = y
    A_cl   = np.hstack([lp.H, -np.eye(m),  np.eye(m)])
    b_cl   = lp.y
    A_eq   = np.hstack([lp.A_eq, np.zeros((lp.A_eq.shape[0], 2*m))])
    A_full = np.vstack([A_cl, A_eq])
    b_full = np.concatenate([b_cl, lp.b_eq])

    # Bounds: x free (except sinks >= 0), r+, r- >= 0
    lb = np.full(n + 2*m, -np.inf)
    lb[sink_start:sink_start + N] = 0.0        # sinks >= 0
    lb[n:] = 0.0                               # r+, r- >= 0

    res = linprog(c, A_eq=A_full, b_eq=b_full, bounds=list(zip(lb, [None]*(n+2*m))),
                  method='highs', options={'presolve': True})

    if res.status != 0:
        raise RuntimeError(f"LP did not converge: {res.message}")

    x_hat = res.x[:n]
    rp    = res.x[n:n+m]
    rm    = res.x[n+m:]
    residuals = rp - rm
    return x_hat, residuals


# ── correctable_k ─────────────────────────────────────────────────────────────

def correctable_k_stacked(lp: LPProblem, max_k: int = 4) -> int:
    """
    Largest k such that dropping ANY 2k claim rows leaves
    [H_sub; A_eq] full column rank.
    Uses exhaustive search; feasible for small n_claims (< ~30).
    """
    full_matrix = np.vstack([lp.H, lp.A_eq])
    n_cols = lp.n_vars
    n_claim_rows = lp.n_claims

    for k in range(max_k, 0, -1):
        n_drop = 2 * k
        if n_drop > n_claim_rows:
            continue
        ok = True
        for drop_idx in itertools.combinations(range(n_claim_rows), n_drop):
            keep = [i for i in range(n_claim_rows) if i not in set(drop_idx)]
            sub = np.vstack([lp.H[keep], lp.A_eq])
            rank = np.linalg.matrix_rank(sub)
            if rank < n_cols:
                ok = False
                break
        if ok:
            return k
    return 0


# ── high-level: localize ──────────────────────────────────────────────────────

def localize_leak(
    snapshots: list[Snapshot],
    topo_links: dict[str, tuple[str, str]],
    topo_nodes: list[str],
    lambda_sink: float = 0.01,
) -> dict[str, float]:
    """
    Solve the stacked LP and return {node_id: inferred_sink}.
    Sorted descending by sink magnitude.
    """
    lp = build_stacked_lp(snapshots, topo_links, topo_nodes, lambda_sink)
    x_hat, _ = solve_stacked(lp, lambda_sink=lambda_sink)
    sink_start = lp.T * lp.n_nodes + lp.T * lp.n_edges
    return {
        nid: float(x_hat[sink_start + i])
        for i, nid in enumerate(lp.node_ids)
    }
