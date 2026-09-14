"""
orb/compile.py
--------------
Compile a domain-free graph JSON into the linear system for L1 estimation.

The model:  y ≈ H @ x + a + ε,  A_eq @ x = b_eq  (hard balance)

State vector layout (columns of H, rows of A_eq):
  [qty_<node_id>, ...]          one per node  (final quantity)
  [flow_<edge_id>, ...]         one per edge  (flow amount)
  [sink_<node_id>, ...]         nodes with sinks == "unknown", or a known
                                withdrawal claim (consumption) on a stocked node

Graph JSON schema:
  {
    "nodes":  [{"id", "initial", "sinks": "none"|"unknown"|"known", "sink"?}, ...],
    "edges":  [{"id", "from", "to"}, ...],
    "claims": [{"id", "type", "ref"|"refs", "value", "source", "weight"}, ...],
    "lambda_sink": float   (optional, default 0.01 — L1 penalty per unit of sink)
  }

Claim types:
  "node"      — measures qty[ref]            H row: +1 at qty col
  "edge"      — measures flow[ref]           H row: +1 at flow col
  "sink"      — measures sink[ref]           H row: +1 at sink col
  "aggregate" — measures sum of qty[refs]    H row: +1 at each qty col

Additive edge claims that share (ref, source) — two shipment_sent rows
on the same hop — are summed before H is built. Replicate sensors
(flow, demand) stay as separate rows.

Balance constraint for node i  (one row of A_eq):
  qty[i]  -  Σ flow_in[j]  +  Σ flow_out[j]  +  sink[i]  =  initial[i] - known_sink[i]

Known sinks: consumption/used/issued claims are subtracted from b_eq
(qty = initial + in − out − consumption). They are not qty sensors and not
unknown-leak variables. Water `demand` stays a qty observation unless the
same node also has an EOD/stock claim.

Returns dict:
  H, y, w       — observation matrix, values, weights   (shapes m×n, m, m)
  w_sink        — objective penalty per state column     (shape n; nonzero only at sink cols)
  A_eq, b_eq    — hard balance constraints               (shapes p×n, p)
  index         — {var_name: col_idx}  bidirectional map
  claim_ids     — [claim["id"], ...]  same order as H rows
  report        — identifiability summary dict
"""

import itertools
import math
import re

import numpy as np

# Edge channels whose rows add up over a snapshot (two trucks on one hop),
# as opposed to replicate sensors of the same flow (water `flow` / `demand`).
_ADDITIVE_TOKENS = {
    "ship", "shipment", "shipped", "sent", "send", "sending",
    "received", "receive", "got", "transfer", "dispatch", "load",
}

# Withdrawals that are known sinks (not final stock, not unknown leaks).
# `demand` is only treated as a known sink when the same node also has EOD/stock.
_KNOWN_SINK_TOKENS = {
    "consumption", "consumed", "consume", "used", "usage", "issued", "issue",
    "burned", "burn", "withdraw", "withdrawal", "drawn",
}
_DEMAND_TOKENS = {"demand"}
_LEAK_METER_TOKENS = {"meter", "drain"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _source_tokens(source) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(source or "").lower()) if t}


def _is_additive_edge_claim(claim: dict) -> bool:
    if claim.get("type") != "edge":
        return False
    return bool(_source_tokens(claim.get("source")) & _ADDITIVE_TOKENS)


def _is_leak_meter_claim(claim: dict) -> bool:
    src = str(claim.get("source") or "").lower().replace("-", "_")
    toks = _source_tokens(src)
    if "sink_meter" in src or toks & _LEAK_METER_TOKENS:
        return True
    # Explicit sink type that is not a named consumption channel → leak meter
    if claim.get("type") == "sink" and not (toks & _KNOWN_SINK_TOKENS):
        return True
    return False


def _is_known_offset_claim(claim: dict, stock_nodes: set[str]) -> bool:
    """Consumption/use is a known withdrawal folded into balance, not a state var."""
    if not claim.get("ref") or claim.get("type") not in {"node", "sink"}:
        return False
    if _is_leak_meter_claim(claim):
        return False
    toks = _source_tokens(claim.get("source"))
    if toks & _KNOWN_SINK_TOKENS:
        return True
    if toks & _DEMAND_TOKENS:
        return claim["ref"] in stock_nodes
    return False


def _is_stock_claim(claim: dict) -> bool:
    """Node claim that reports on-hand / EOD, not a withdrawal."""
    if claim.get("type") != "node" or not claim.get("ref"):
        return False
    toks = _source_tokens(claim.get("source"))
    if toks & _KNOWN_SINK_TOKENS or toks & _DEMAND_TOKENS:
        return False
    if _is_leak_meter_claim(claim):
        return False
    return True


def _node_known_sink(n: dict) -> float:
    """Hard-wired known withdrawal on the node (not a flaggable claim)."""
    mode = str(n.get("sinks") or "").lower()
    if mode not in {"known", "fixed"}:
        return 0.0
    for key in ("sink", "known_sink", "consumption"):
        val = n.get(key)
        if val is not None and _finite(val):
            return float(val)
    return 0.0


def _is_additive_claim(claim: dict) -> bool:
    if _is_additive_edge_claim(claim):
        return True
    if claim.get("type") in {"node", "sink"} and _source_tokens(claim.get("source")) & _KNOWN_SINK_TOKENS:
        return True
    return False


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _coalesce_claims(claims: list[dict]) -> list[dict]:
    """Sum additive edge rows that share (ref, source). Keep replicates intact."""
    buckets: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for c in claims:
        if not _is_additive_claim(c):
            order.append(("raw", id(c), c))
            continue
        key = (c.get("ref"), str(c.get("source") or "").lower())
        if key not in buckets:
            buckets[key] = []
            order.append(("bucket", key, None))
        buckets[key].append(c)

    out: list[dict] = []
    for kind, item, raw in order:
        if kind == "raw":
            out.append(raw)
            continue
        group = buckets[item]
        if len(group) == 1:
            out.append(group[0])
            continue
        summed = dict(group[0])
        summed["value"] = float(sum(float(c.get("value") or 0) for c in group))
        summed["weight"] = max(float(c.get("weight", 1.0) or 1.0) for c in group)
        summed["members"] = [c.get("id") for c in group]
        out.append(summed)
    return out


def compile(graph: dict) -> dict:  # noqa: A001
    """Compile graph dict → linear system."""
    nodes       = [dict(n) for n in (graph.get("nodes") or [])]
    edges       = [dict(e) for e in (graph.get("edges") or [])]
    raw_claims  = [dict(c) for c in (graph.get("claims") or [])]
    lambda_sink = float(graph.get("lambda_sink", 0.01) or 0.01)

    stock_nodes = {c["ref"] for c in raw_claims if _is_stock_claim(c)}
    claims = _coalesce_claims(raw_claims)

    known_offsets: dict[str, float] = {}
    kept: list[dict] = []
    for c in claims:
        if _is_known_offset_claim(c, stock_nodes) and _finite(c.get("value")):
            nid = str(c["ref"])
            known_offsets[nid] = known_offsets.get(nid, 0.0) + float(c["value"])
            continue
        kept.append(c)
    claims = kept

    node_ids = {str(n["id"]) for n in nodes}
    for e in edges:
        for nid in (e.get("from"), e.get("to")):
            if nid and str(nid) not in node_ids:
                nodes.append({"id": str(nid), "initial": 0.0, "sinks": "none"})
                node_ids.add(str(nid))

    for c in claims:
        if c.get("type") in {"node", "sink"}:
            nid = c.get("ref")
            if nid and str(nid) not in node_ids:
                nodes.append({"id": str(nid), "initial": 0.0, "sinks": "none"})
                node_ids.add(str(nid))
        if c.get("type") == "sink" and c.get("ref") and not _is_known_offset_claim(c, stock_nodes):
            for n in nodes:
                if n["id"] == c["ref"]:
                    n["sinks"] = "unknown"
                    break

    for nid in known_offsets:
        if nid not in node_ids:
            nodes.append({"id": nid, "initial": 0.0, "sinks": "none"})
            node_ids.add(nid)

    # ── 1. State vector index ────────────────────────────────────────────────
    idx: dict[str, int] = {}
    col = 0

    qty_cols:  dict[str, int] = {}
    seen_nodes: set[str] = set()
    unique_nodes: list[dict] = []
    for n in nodes:
        nid = str(n.get("id", ""))
        if not nid or nid in seen_nodes:
            continue
        seen_nodes.add(nid)
        n["id"] = nid
        unique_nodes.append(n)
        key = f"qty_{nid}"
        idx[key] = col
        qty_cols[nid] = col
        col += 1
    nodes = unique_nodes

    flow_cols: dict[str, int] = {}
    seen_edges: set[str] = set()
    unique_edges: list[dict] = []
    for e in edges:
        eid = str(e.get("id") or f"e_{e.get('from')}_{e.get('to')}")
        if eid in seen_edges:
            continue
        seen_edges.add(eid)
        e["id"] = eid
        unique_edges.append(e)
        key = f"flow_{eid}"
        idx[key] = col
        flow_cols[eid] = col
        col += 1
    edges = unique_edges

    sink_cols: dict[str, int] = {}
    for n in nodes:
        if n.get("sinks") == "unknown":
            key = f"sink_{n['id']}"
            idx[key] = col
            sink_cols[n["id"]] = col
            col += 1

    n_vars = col

    # ── 2. Observation matrix H, y, w ────────────────────────────────────────
    usable: list[dict] = []
    for c in claims:
        if not _finite(c.get("value")):
            continue
        ctype = c.get("type")
        if ctype == "node" and c.get("ref") in qty_cols:
            usable.append(c)
        elif ctype == "edge" and c.get("ref") in flow_cols:
            usable.append(c)
        elif ctype == "sink" and c.get("ref") in sink_cols:
            usable.append(c)
        elif ctype == "aggregate" and any(r in qty_cols for r in (c.get("refs") or [])):
            usable.append(c)
    claims = usable

    m = len(claims)
    H = np.zeros((m, n_vars)) if n_vars else np.zeros((m, 0))
    y = np.zeros(m)
    w = np.zeros(m)
    claim_ids: list[str] = []

    for i, c in enumerate(claims):
        claim_ids.append(c["id"])
        y[i] = float(c["value"])
        w[i] = max(float(c.get("weight", 1.0) or 0.0), 1e-12)

        ctype = c["type"]
        if ctype == "node":
            H[i, qty_cols[c["ref"]]] = 1.0
        elif ctype == "edge":
            H[i, flow_cols[c["ref"]]] = 1.0
        elif ctype == "sink":
            H[i, sink_cols[c["ref"]]] = 1.0
        elif ctype == "aggregate":
            for ref in c["refs"]:
                if ref in qty_cols:
                    H[i, qty_cols[ref]] = 1.0

    # ── 3. Balance constraints A_eq, b_eq ────────────────────────────────────
    # qty[i] − Σflow_in + Σflow_out + sink[i] = initial[i]
    in_edges:  dict[str, list[str]] = {n["id"]: [] for n in nodes}
    out_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        frm, to = e.get("from"), e.get("to")
        if frm in out_edges:
            out_edges[frm].append(e["id"])
        if to in in_edges:
            in_edges[to].append(e["id"])

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
        b_eq[i] = float(n.get("initial", 0.0) or 0.0)
        b_eq[i] -= known_offsets.get(nid, 0.0)
        if nid not in sink_cols:
            b_eq[i] -= _node_known_sink(n)

    # ── 4. Sink penalty in objective ─────────────────────────────────────────
    # Observed (known) sinks are pinned by claims — do not shrink them with λ.
    # Unobserved unknown sinks (leaks) keep the L1 leak penalty.
    w_sink = np.zeros(n_vars)
    observed_sinks = {c["ref"] for c in claims if c.get("type") == "sink"}
    for nid, col in sink_cols.items():
        w_sink[col] = 0.0 if nid in observed_sinks else lambda_sink

    # ── 5. Identifiability ───────────────────────────────────────────────────
    HA        = np.vstack([H, A_eq]) if n_vars else np.zeros((m + p, 0))
    rank_HA   = int(np.linalg.matrix_rank(HA)) if HA.size else 0
    identifiable   = rank_HA >= n_vars
    correctable_k  = _correctable_k(H, A_eq, n_vars)

    report = {
        "n_vars":         n_vars,
        "n_claims":       m,
        "rank":           rank_HA,
        "identifiable":   identifiable,
        "correctable_k":  correctable_k,
        "sink_node_ids":  list(sink_cols.keys()),
        "known_sinks":    known_offsets,
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
        "claims":     claims,
        "known_sinks": known_offsets,
        "report":     report,
    }


# ---------------------------------------------------------------------------
# Identifiability helpers
# ---------------------------------------------------------------------------

def _correctable_k(H: np.ndarray, A_eq: np.ndarray, n_vars: int) -> int:
    """
    Largest k such that dropping ANY 2k rows of H still leaves
    rank([H_remaining; A_eq]) == n_vars.

    Exhaustive search capped at min(m // 2, 5) for tractability, and
    stopped before C(m, 2k) blows up (LeakDB-scale claim sets).
    Returns 0 if even one corruption cannot be corrected.
    """
    if n_vars <= 0:
        return 0
    m = H.shape[0]
    max_k = min(m // 2, 5)
    max_combos = 20_000

    for k in range(1, max_k + 1):
        n_drop = 2 * k
        if n_drop > m:
            return k - 1
        n_combos = math.comb(m, n_drop) if hasattr(math, "comb") else None
        if n_combos is None:
            n_combos = 1
            for i in range(n_drop):
                n_combos = n_combos * (m - i) // (i + 1)
        if n_combos > max_combos:
            return k - 1
        for dropped in itertools.combinations(range(m), n_drop):
            kept = [i for i in range(m) if i not in dropped]
            if kept:
                H_sub = H[np.array(kept)]
                HA_sub = np.vstack([H_sub, A_eq])
            else:
                HA_sub = A_eq
            if np.linalg.matrix_rank(HA_sub) < n_vars:
                return k - 1

    return max_k
