"""
orb/compile.py
--------------
Compile a domain-free graph JSON into the linear system for L1 estimation.

The model:  y ≈ H @ x + a + ε,  A_eq @ x = b_eq  (hard balance)

State vector layout (columns of H, rows of A_eq):
  [qty_<node_id>, ...]          one per node  (final quantity)
  [flow_<edge_id>, ...]         one per edge  (flow amount)
  [sink_<node_id>, ...]         only for nodes where sinks == "unknown"

Graph JSON schema:
  {
    "nodes":  [{"id", "initial", "sinks": "none"|"unknown"}, ...],
    "edges":  [{"id", "from", "to"}, ...],
    "claims": [{"id", "type", "ref"|"refs", "value", "source", "weight"}, ...],
    "lambda_sink": float   (optional, default 0.01 — L1 penalty per unit of sink)
  }

Claim types:
  "node"      — measures qty[ref]            H row: +1 at qty col
  "edge"      — measures flow[ref]           H row: +1 at flow col
  "aggregate" — measures sum of qty[refs]    H row: +1 at each qty col

Balance constraint for node i  (one row of A_eq):
  qty[i]  -  Σ flow_in[j]  +  Σ flow_out[j]  +  sink[i]  =  initial[i] - known_sinks[i]

Returns dict:
  H, y, w       — observation matrix, values, weights   (shapes m×n, m, m)
  w_sink        — objective penalty per state column     (shape n; nonzero only at sink cols)
  A_eq, b_eq    — hard balance constraints               (shapes p×n, p)
  index         — {var_name: col_idx}  bidirectional map
  claim_ids     — [claim["id"], ...]  same order as H rows
  report        — identifiability summary dict
"""

import itertools
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile(graph: dict) -> dict:  # noqa: A001
    """Compile graph dict → linear system."""
    nodes       = graph["nodes"]
    edges       = graph["edges"]
    claims      = graph["claims"]
    lambda_sink = float(graph.get("lambda_sink", 0.01))

    edge_map = {e["id"]: e for e in edges}

    # ── 1. State vector index ────────────────────────────────────────────────
    idx: dict[str, int] = {}
    col = 0

    qty_cols:  dict[str, int] = {}
    for n in nodes:
        key = f"qty_{n['id']}"
        idx[key] = col
        qty_cols[n["id"]] = col
        col += 1

    flow_cols: dict[str, int] = {}
    for e in edges:
        key = f"flow_{e['id']}"
        idx[key] = col
        flow_cols[e["id"]] = col
        col += 1

    sink_cols: dict[str, int] = {}
    for n in nodes:
        if n.get("sinks") == "unknown":
            key = f"sink_{n['id']}"
            idx[key] = col
            sink_cols[n["id"]] = col
            col += 1

    n_vars = col

    # ── 2. Observation matrix H, y, w ────────────────────────────────────────
    m = len(claims)
    H = np.zeros((m, n_vars))
    y = np.zeros(m)
    w = np.zeros(m)
    claim_ids: list[str] = []

    for i, c in enumerate(claims):
        claim_ids.append(c["id"])
        y[i] = float(c["value"])
        w[i] = float(c.get("weight", 1.0))

        ctype = c["type"]
        if ctype == "node":
            H[i, qty_cols[c["ref"]]] = 1.0
        elif ctype == "edge":
            H[i, flow_cols[c["ref"]]] = 1.0
        elif ctype == "sink":
            # Direct measurement of an unknown sink variable (e.g. a drain meter).
            # Only valid if the referenced node has sinks == "unknown".
            nid = c["ref"]
            if nid not in sink_cols:
                raise ValueError(
                    f"Claim {c['id']!r} has type 'sink' but node {nid!r} "
                    f"has sinks != 'unknown'"
                )
            H[i, sink_cols[nid]] = 1.0
        elif ctype == "aggregate":
            for ref in c["refs"]:
                H[i, qty_cols[ref]] = 1.0
        else:
            raise ValueError(f"Unknown claim type: {ctype!r}")

    # ── 3. Balance constraints A_eq, b_eq ────────────────────────────────────
    # qty[i] − Σflow_in + Σflow_out + sink[i] = initial[i]
    in_edges:  dict[str, list[str]] = {n["id"]: [] for n in nodes}
    out_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        out_edges[e["from"]].append(e["id"])
        in_edges[e["to"]].append(e["id"])

    p = len(nodes)
    A_eq = np.zeros((p, n_vars))
    b_eq = np.zeros(p)

    for i, n in enumerate(nodes):
        nid = n["id"]
        A_eq[i, qty_cols[nid]] = 1.0
        for eid in in_edges[nid]:
            A_eq[i, flow_cols[eid]] = -1.0
        for eid in out_edges[nid]:
            A_eq[i, flow_cols[eid]] = 1.0
        if nid in sink_cols:
            A_eq[i, sink_cols[nid]] = 1.0
        b_eq[i] = float(n.get("initial", 0.0)) - float(n.get("known_sinks", 0.0))

    # ── 4. Sink penalty in objective ─────────────────────────────────────────
    w_sink = np.zeros(n_vars)
    for c in sink_cols.values():
        w_sink[c] = lambda_sink

    # ── 5. Identifiability ───────────────────────────────────────────────────
    HA        = np.vstack([H, A_eq])
    rank_HA   = int(np.linalg.matrix_rank(HA))
    identifiable   = rank_HA >= n_vars
    correctable_k, k_exact = correctable_k_search(H, A_eq, n_vars)

    report = {
        "n_vars":         n_vars,
        "n_claims":       m,
        "rank":           rank_HA,
        "identifiable":   identifiable,
        "correctable_k":  correctable_k,
        "correctable_k_exact": k_exact,
        "sink_node_ids":  list(sink_cols.keys()),
    }

    return {
        "H":          H,
        "y":          y,
        "w":          w,
        "w_sink":     w_sink,
        "A_eq":       A_eq,
        "b_eq":       b_eq,
        "index":      idx,
        "claim_ids":  claim_ids,
        "report":     report,
    }


# ---------------------------------------------------------------------------
# Identifiability helpers
# ---------------------------------------------------------------------------

_K_CAP         = 5          # never report more than this
_SEARCH_BUDGET = 300_000    # rank evaluations before giving up on exactness


def _correctable_k(H: np.ndarray, A_eq: np.ndarray, n_vars: int) -> int:
    """Backward-compatible wrapper: just the k."""
    return correctable_k_search(H, A_eq, n_vars)[0]


def correctable_k_search(H: np.ndarray, A_eq: np.ndarray, n_vars: int) -> tuple[int, bool]:
    """
    Largest k such that dropping ANY 2k rows of H still leaves
    rank([H_remaining; A_eq]) == n_vars, capped at min(m // 2, 5).

    Equivalent statement: let d be the smallest number of H rows whose
    removal makes the system rank-deficient; then k = floor((d - 1) / 2).

    When every H row measures a single variable (node / edge / sink claims)
    d is found by searching VARIABLE subsets instead of row subsets:
    removing rows breaks rank iff the variables they free (every claim on
    the variable removed), together with the never-claimed variables, admit
    a nonzero vector in the nullspace of the balance equations.  Freeing
    variable j costs m_j rows (its claim count), so d = min cost(F) over
    deficient F.  This is exact, and its size depends on the number of
    variables rather than on how many reports were kept — which matters now
    that the merge keeps every sender / receiver / per-file row as evidence.

    Aggregate claims (rows with several nonzeros) fall back to the
    row-subset search.  Both searches are budgeted; if the budget runs out
    the returned k is the largest one fully verified and exact is False.

    Returns (k, exact).
    """
    m = H.shape[0]
    max_k = min(m // 2, _K_CAP)
    if max_k == 0:
        return 0, True
    if _unit_rows(H):
        return _k_by_variables(H, A_eq, n_vars, max_k)
    return _k_by_rows(H, A_eq, n_vars, max_k)


def _unit_rows(H: np.ndarray) -> bool:
    return bool(H.shape[0] > 0 and np.all(np.count_nonzero(H, axis=1) == 1))


def _k_by_variables(H: np.ndarray, A_eq: np.ndarray, n_vars: int, max_k: int) -> tuple[int, bool]:
    counts = np.count_nonzero(H, axis=0)                 # claims per variable
    unclaimed = [j for j in range(n_vars) if counts[j] == 0]
    claimed   = sorted((j for j in range(n_vars) if counts[j] > 0), key=lambda j: counts[j])
    cost      = {j: int(counts[j]) for j in claimed}

    def deficient(cols: list[int]) -> bool:
        if not cols:
            return False
        return np.linalg.matrix_rank(A_eq[:, cols]) < len(cols)

    if deficient(unclaimed):          # unidentifiable before dropping anything
        return 0, True

    limit = 2 * max_k                 # only costs <= limit can change the answer
    best: int | None = None           # cheapest deficient F found
    evals = 0
    sorted_costs = [cost[j] for j in claimed]

    for size in range(1, len(claimed) + 1):
        min_cost = sum(sorted_costs[:size])
        if min_cost > limit or (best is not None and min_cost >= best):
            break
        for F in itertools.combinations(claimed, size):
            c = sum(cost[j] for j in F)
            if c > limit or (best is not None and c >= best):
                continue
            evals += 1
            if evals > _SEARCH_BUDGET:
                # Every F of size < `size` was examined, and cost >= size,
                # so no deficient F costs less than `size`.
                if best is not None and best <= size:
                    d = best
                    exact = True
                else:
                    d = size
                    exact = False
                return min(max_k, (d - 1) // 2), exact
            if deficient(unclaimed + list(F)):
                best = c

    d = best if best is not None else limit + 1
    return min(max_k, (d - 1) // 2), True


def _k_by_rows(H: np.ndarray, A_eq: np.ndarray, n_vars: int, max_k: int) -> tuple[int, bool]:
    """Original exhaustive row-subset search, budgeted."""
    m = H.shape[0]
    evals = 0
    for k in range(1, max_k + 1):
        n_drop = 2 * k
        if n_drop > m:
            return k - 1, True
        for dropped in itertools.combinations(range(m), n_drop):
            evals += 1
            if evals > _SEARCH_BUDGET:
                return k - 1, False
            kept = [i for i in range(m) if i not in dropped]
            HA_sub = np.vstack([H[np.array(kept)], A_eq]) if kept else A_eq
            if np.linalg.matrix_rank(HA_sub) < n_vars:
                return k - 1, True
    return max_k, True
