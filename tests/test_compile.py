"""
tests/test_compile.py
---------------------
Unit tests for the compile / run_graph layer, including two documented
limitation tests (lambda_sink sensitivity and sink/corruption confusability).

Run with:   pytest tests/test_compile.py -v

Known limitations documented here:
  test_lambda_sink_sensitivity — valid range for lambda_sink
  test_sink_corruption_confusability — when corruption is absorbed as fake leak
"""

import copy
import json
import os
import sys

import numpy as np
import pytest

# Make the Orb root importable without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orb.compile   import compile as compile_graph
from orb.run_graph import run_graph, _l1_solve
from orb.decode    import decode

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name: str) -> dict:
    with open(os.path.join(FIXTURES, name)) as fh:
        return json.load(fh)


def _run(name: str) -> dict:
    return run_graph(os.path.join(FIXTURES, name))


# ---------------------------------------------------------------------------
# Test 1 — index round-trip
# ---------------------------------------------------------------------------

def test_index_round_trip():
    """
    The index must contain exactly one column for every node qty and edge flow,
    column indices must be unique, and no sink cols must appear in toy_supply.json
    (which has sinks='none').
    """
    graph = _load("toy_supply.json")
    c = compile_graph(graph)
    idx = c["index"]

    # Every node has a qty col
    for n in graph["nodes"]:
        assert f"qty_{n['id']}" in idx, f"Missing qty col for {n['id']}"

    # Every edge has a flow col
    for e in graph["edges"]:
        assert f"flow_{e['id']}" in idx, f"Missing flow col for {e['id']}"

    # No sinks (sinks='none' for all nodes)
    for key in idx:
        assert not key.startswith("sink_"), f"Unexpected sink col: {key}"

    # Columns are unique and contiguous 0..n-1
    vals = sorted(idx.values())
    assert vals == list(range(len(vals))), "Column indices not contiguous"


# ---------------------------------------------------------------------------
# Test 2 — toy_supply: L1 flags c2 as corrupted
# ---------------------------------------------------------------------------

def test_toy_supply_flags_c2():
    """
    toy_supply.json plants a corruption on c2 (edge e2 claims 99 instead of 400).
    The L1 estimator must flag c2 with the largest absolute residual (≥ 100).
    """
    result  = _run("toy_supply.json")
    flagged = result["decoded"]["flagged"]

    assert len(flagged) >= 1, "Expected at least one flagged claim"

    # Flagged list is sorted by descending |residual| — c2 must be first
    top = flagged[0]
    assert top["claim_id"] == "c2", (
        f"Expected c2 to have the largest residual, got {top['claim_id']} "
        f"(r={top['residual']:.2f})"
    )
    assert abs(top["residual"]) > 100, (
        f"c2 residual too small: {top['residual']:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — toy_water_sinks: L1 recovers the 20-unit leak at BASE_B
# ---------------------------------------------------------------------------

def test_toy_water_recovers_leak():
    """
    toy_water_sinks.json has sinks='unknown' at BASE_B with a true leak of 20.
    All claims are truthful; the solver must recover sink_BASE_B ≈ 20.
    """
    result = _run("toy_water_sinks.json")
    sinks  = result["decoded"]["sinks"]

    base_b = next((s for s in sinks if s["id"] == "BASE_B"), None)
    assert base_b is not None, "BASE_B must have a sink entry (sinks='unknown')"

    assert abs(base_b["sink"] - 20.0) < 1.0, (
        f"Expected BASE_B sink ≈ 20, got {base_b['sink']:.4f}"
    )

    # No claims should be flagged (all claims are truthful)
    flagged = result["decoded"]["flagged"]
    assert len(flagged) == 0, (
        f"No claims should be flagged in the water fixture, got: {flagged}"
    )


# ---------------------------------------------------------------------------
# Test 4 — identifiability: both fixtures are full-rank
# ---------------------------------------------------------------------------

def test_identifiability():
    """
    [H; A_eq] must span the full state space for both fixtures.
    report['identifiable'] must be True and rank == n_vars.
    """
    for fname in ["toy_supply.json", "toy_water_sinks.json"]:
        graph = _load(fname)
        c     = compile_graph(graph)
        rep   = c["report"]

        assert rep["identifiable"], (
            f"{fname}: not identifiable  "
            f"(rank={rep['rank']}, n_vars={rep['n_vars']})"
        )
        assert rep["rank"] == rep["n_vars"], (
            f"{fname}: rank {rep['rank']} < n_vars {rep['n_vars']}"
        )


# ---------------------------------------------------------------------------
# Test 5 — claim_ids are preserved in declaration order
# ---------------------------------------------------------------------------

def test_claim_ids_preserved():
    """
    compile() must return claim_ids in the same order as graph['claims'].
    This ensures residuals[i] matches claims[i] for downstream flagging.
    """
    graph     = _load("toy_supply.json")
    c         = compile_graph(graph)
    expected  = [claim["id"] for claim in graph["claims"]]

    assert c["claim_ids"] == expected, (
        f"claim_ids order mismatch:\n  got:      {c['claim_ids']}\n  expected: {expected}"
    )


# ---------------------------------------------------------------------------
# Test 6 — lambda_sink valid range (robustness check)
# ---------------------------------------------------------------------------

def test_lambda_sink_sensitivity():
    """
    lambda_sink must be strictly less than the minimum claim weight for the
    estimator to prefer absorbing a true leak into a sink variable rather than
    flagging the truthful node claim.

    Mechanically: the LP compares
        lambda_sink * sink_amount  vs  claim_weight * residual_if_no_sink
    and picks whichever is cheaper.  With sink_amount == residual_if_no_sink == 20
    and claim_weight == 0.8, the break-even is lambda_sink == 0.8.

    This test verifies:
      - lambda in [0.0001, 0.5]  → sink_BASE_B ≈ 20, no claims flagged  (correct)
      - lambda in [1.0, 10.0]    → estimator flags instead; sink = 0    (wrong regime)
    """
    base_graph = _load("toy_water_sinks.json")

    # Valid range: lambda < claim_weight (0.8)
    for lam in [0.0001, 0.001, 0.01, 0.1, 0.5]:
        g = copy.deepcopy(base_graph)
        g["lambda_sink"] = lam
        result = run_graph(os.path.join(FIXTURES, "toy_water_sinks.json"))
        # Cheat: recompile with lam override to avoid re-reading file
        compiled = compile_graph(g)
        x_hat, residuals = _l1_solve(compiled)
        decoded = decode(x_hat, residuals, compiled, g)
        sink_bb = next((s["sink"] for s in decoded["sinks"] if s["id"] == "BASE_B"), None)
        assert abs(sink_bb - 20.0) < 1.0, (
            f"lambda={lam}: expected sink_BASE_B≈20, got {sink_bb:.4f}"
        )
        assert not decoded["flagged"], (
            f"lambda={lam}: spurious flagging with truthful claims"
        )

    # Invalid range: lambda >= claim_weight → sink suppressed, claim flagged
    for lam in [1.0, 5.0, 10.0]:
        g = copy.deepcopy(base_graph)
        g["lambda_sink"] = lam
        compiled = compile_graph(g)
        x_hat, residuals = _l1_solve(compiled)
        decoded = decode(x_hat, residuals, compiled, g)
        sink_bb = next((s["sink"] for s in decoded["sinks"] if s["id"] == "BASE_B"), None)
        assert sink_bb < 1.0, (
            f"lambda={lam}: expected sink suppressed, got {sink_bb:.4f}"
        )
        assert decoded["flagged"], (
            f"lambda={lam}: expected claim flagged when sink suppressed"
        )


# ---------------------------------------------------------------------------
# Test 7 — sink / corruption confusability (documented limitation)
# ---------------------------------------------------------------------------

def test_sink_corruption_confusability():
    """
    DOCUMENTED LIMITATION: when a node claim for a sink-enabled node is
    corrupted *downward*, the LP silently absorbs the discrepancy into an
    enlarged sink rather than flagging the corruption.

    This is the correct LP optimum — the fake sink is cheaper than flagging —
    but it means the estimator produces false certainty: it returns a state
    with zero flagged claims even though one claim is wrong.

    The confusability arises because 'claim reports less than received' and
    'a real leak occurred' produce identical residual signatures on the
    claim vector.  The only resolution is an independent corroborating claim
    on sink_BASE_B itself (e.g. a physical meter).

    correctable_k == 0 on toy_water_sinks.json is the formal indicator:
    the system cannot tolerate even one corruption when leaks are unknown.
    """
    base_graph = _load("toy_water_sinks.json")

    # Find c6: node claim for BASE_B (true value 380)
    c6_idx = next(i for i, c in enumerate(base_graph["claims"]) if c["id"] == "c6")

    # Corrupt c6 downward by 50 (claims 330 instead of 380)
    g = copy.deepcopy(base_graph)
    g["claims"][c6_idx]["value"] = 330

    compiled = compile_graph(g)
    x_hat, residuals = _l1_solve(compiled)
    decoded = decode(x_hat, residuals, compiled, g)

    sink_bb = next((s["sink"] for s in decoded["sinks"] if s["id"] == "BASE_B"), 0.0)
    flagged_ids = [f["claim_id"] for f in decoded["flagged"]]

    # The estimator silently inflates the sink rather than flagging c6
    assert not flagged_ids, (
        "If this assertion fails, confusability is resolved — "
        f"update this test.  Flagged: {flagged_ids}"
    )
    assert sink_bb > 25, (
        f"Expected inflated sink (>25), got {sink_bb:.2f}.  "
        "Confusability may no longer apply."
    )
    # The inflated sink is the false certainty: the estimator is confident
    # but wrong about what is happening at BASE_B.
    assert any(a["id"] == "BASE_B" for a in decoded["ambiguous"]), (
        "Unmetered downward lie must be reported as {loss, downward_corruption}, "
        f"got {decoded.get('ambiguous')}"
    )
    assert not any(s["id"] == "BASE_B" for s in decoded.get("loss") or [])


# ---------------------------------------------------------------------------
# Test 8 — metered fixture: correctable_k rises to 1
# ---------------------------------------------------------------------------

def test_metered_correctable_k():
    """
    toy_water_metered.json adds two independent sensors at BASE_B:
      m1  (type='sink', drain meter)  — directly observes sink_BASE_B
      m2  (type='node', level sensor) — independently observes qty_BASE_B

    WHY two sensors are needed:
      With only m1, the pair {c6, m1} can be dropped simultaneously, leaving
      BASE_B balance as the only constraint on {qty_BASE_B, sink_BASE_B} — one
      equation in two unknowns, rank-deficient.  m2 breaks that by providing
      an independent path to qty_BASE_B, so NO pair of dropped H rows can
      leave any variable underdetermined.

    Result: correctable_k rises from 0 → 1.
    """
    graph = _load("toy_water_metered.json")
    c     = compile_graph(graph)
    rep   = c["report"]

    assert rep["correctable_k"] >= 1, (
        f"Expected correctable_k >= 1 with two meters, got {rep['correctable_k']}"
    )
    assert rep["identifiable"], "Metered fixture must be identifiable"


# ---------------------------------------------------------------------------
# Test 9 — metered fixture: downward corruption on c6 is now flagged
# ---------------------------------------------------------------------------

def test_metered_detects_downward_corruption():
    """
    In toy_water_sinks.json, corrupting c6 downward by 50 was silently
    absorbed into an enlarged sink (test_sink_corruption_confusability).

    With two corroborating sensors (m1 + m2), the same corruption is detected:
      - m2 (level sensor) still reports 380, contradicting c6's false 330
      - m1 (drain meter) still reports 20, contradicting any fake-sink explanation
      - The LP flags c6 as the cheapest inconsistency to discard

    The resolving claim is m2 (or m1 via the balance constraint): the independent
    second reading makes the coordinated-silence attack visible.
    """
    base_graph = _load("toy_water_metered.json")

    c6_idx = next(i for i, c in enumerate(base_graph["claims"]) if c["id"] == "c6")
    g = copy.deepcopy(base_graph)
    g["claims"][c6_idx]["value"] = 330  # corrupt c6 downward by 50

    compiled = compile_graph(g)
    x_hat, residuals = _l1_solve(compiled)
    decoded = decode(x_hat, residuals, compiled, g)

    flagged_ids = [f["claim_id"] for f in decoded["flagged"]]
    sink_bb = next((s["sink"] for s in decoded["sinks"] if s["id"] == "BASE_B"), 0.0)

    # c6 must be flagged; the fake sink must NOT be inflated
    assert "c6" in flagged_ids, (
        f"Expected c6 flagged, got: {flagged_ids}  (sink_BASE_B={sink_bb:.2f})"
    )
    assert sink_bb < 25, (
        f"Sink should remain ~20, not inflated to {sink_bb:.2f} (corruption absorbed)"
    )


# ---------------------------------------------------------------------------
# Test 10 — direction asymmetry on c6 (the only sink-enabled node claim)
# ---------------------------------------------------------------------------

def test_direction_asymmetry_c6():
    """
    In the unmetered water fixture, corruption on c6 (node BASE_B, the only
    node with sinks='unknown') is direction-asymmetric:

      Downward corruption (qty claimed < true):
        Absorbed silently into sink_BASE_B.  Zero flagged claims.
        The estimator is falsely certain: it returns a plausible state with a
        large leak and no inconsistency signal.

      Upward corruption (qty claimed > true):
        Cannot be absorbed: sink >= 0 would need to go negative.
        The LP flags c6 correctly.

    This gives a clean characterisation:
      "With unknown sinks, downward lies on sink-node claims are invisible;
       upward lies are not.  This is the mirror image of one-directional
       corruption tolerance baked into threshold-based approaches."

    Also verify edge claims (c3: flow_e2) are detectable in BOTH directions,
    since edge flows do not interact with the sink variable in the balance.
    """
    base_graph = _load("toy_water_sinks.json")
    c6_idx = next(i for i, c in enumerate(base_graph["claims"]) if c["id"] == "c6")
    c3_idx = next(i for i, c in enumerate(base_graph["claims"]) if c["id"] == "c3")

    # c6 downward → absorbed (sink inflates, nothing flagged)
    g_down = copy.deepcopy(base_graph)
    g_down["claims"][c6_idx]["value"] = 230  # 380 - 150
    compiled = compile_graph(g_down)
    x_hat, residuals = _l1_solve(compiled)
    decoded_down = decode(x_hat, residuals, compiled, g_down)

    sink_down = next(s["sink"] for s in decoded_down["sinks"] if s["id"] == "BASE_B")
    assert not decoded_down["flagged"], "Downward c6 corruption must be absorbed (no flags)"
    assert sink_down > 25, f"Sink should inflate above 25, got {sink_down:.2f}"

    # c6 upward → flagged (sink can't go negative)
    g_up = copy.deepcopy(base_graph)
    g_up["claims"][c6_idx]["value"] = 530  # 380 + 150
    compiled = compile_graph(g_up)
    x_hat, residuals = _l1_solve(compiled)
    decoded_up = decode(x_hat, residuals, compiled, g_up)

    flagged_up = [f["claim_id"] for f in decoded_up["flagged"]]
    assert "c6" in flagged_up, f"Upward c6 corruption must be flagged, got {flagged_up}"

    # Edge claim c3 (flow_e2): both directions flagged because edge flows
    # can't be absorbed — sink only enters the balance at BASE_B's qty side
    for label, delta in [("up", +150), ("down", -150)]:
        g = copy.deepcopy(base_graph)
        g["claims"][c3_idx]["value"] += delta
        compiled = compile_graph(g)
        x_hat, residuals = _l1_solve(compiled)
        decoded = decode(x_hat, residuals, compiled, g)
        flagged_ids = [f["claim_id"] for f in decoded["flagged"]]
        assert flagged_ids, (
            f"Edge claim c3 corruption ({label}) must be flagged, got none"
        )


# ---------------------------------------------------------------------------
# Helpers for ingested logistics fixtures
# ---------------------------------------------------------------------------

def _ingest(name: str) -> dict:
    from orb.ingest import ingest_paths
    graph, report = ingest_paths([os.path.join(FIXTURES, name)])
    assert graph and graph.get("claims"), f"{name}: no graph ({report})"
    return graph


def _qty(decoded: dict, nid: str) -> float:
    node = next((n for n in decoded["nodes"] if n["id"] == nid), None)
    assert node is not None, f"missing node {nid} in { [n['id'] for n in decoded['nodes']] }"
    return node["qty"]


def _flow(decoded: dict, frm: str, to: str) -> float:
    edge = next((e for e in decoded["edges"] if e["from"] == frm and e["to"] == to), None)
    assert edge is not None, f"missing edge {frm}->{to}"
    return edge["flow"]


# ---------------------------------------------------------------------------
# Test 11 — two shipments on one hop are net flow, not a fight
# ---------------------------------------------------------------------------

def test_supply_logistics_nets_two_shipments():
    """
    supply_logistics.csv has two Depot→Alpha trucks (800 then 300) plus a
    planted +250 lie on OP_DELTA's EOD.

    Before coalescing, L1 treated 800 and 300 as conflicting sensors of one
    flow, picked ~800, and then falsely flagged Depot/Alpha EOD. Net flow
    must be 1100, books must match, and only Delta's EOD is the lie.
    """
    graph = _ingest("supply_logistics.csv")
    compiled = compile_graph(graph)
    x_hat, residuals = _l1_solve(compiled)
    decoded = decode(x_hat, residuals, compiled, graph)

    assert abs(_flow(decoded, "MAIN_DEPOT", "FOB_ALPHA") - 1100) < 1.0
    assert abs(_qty(decoded, "MAIN_DEPOT") - 2300) < 1.0
    assert abs(_qty(decoded, "FOB_ALPHA") - 1300) < 1.0
    assert abs(_qty(decoded, "FOB_BRAVO") - 900) < 1.0
    assert abs(_qty(decoded, "OP_DELTA") - 270) < 1.0  # 90 + 180, not the 520 lie

    flagged_ids = {f["claim_id"] for f in decoded["flagged"]}
    by_id = {c["id"]: c for c in compiled["claims"]}
    flagged_node_refs = {
        by_id[cid]["ref"] for cid in flagged_ids
        if by_id.get(cid, {}).get("type") == "node"
    }
    assert "OP_DELTA" in flagged_node_refs, (
        f"Delta EOD must be flagged, got {decoded['flagged']}"
    )
    assert "MAIN_DEPOT" not in flagged_node_refs
    assert "FOB_ALPHA" not in flagged_node_refs


def test_fuel_tanks_flags_terminal_eod():
    """Terminal EOD 3600 vs conserved 3300. L1 recovers 3300 and flags the +300."""
    graph = _ingest("fuel_tanks.csv")
    compiled = compile_graph(graph)
    x_hat, residuals = _l1_solve(compiled)
    decoded = decode(x_hat, residuals, compiled, graph)

    assert abs(_qty(decoded, "TERMINAL") - 3300) < 1.0
    by_id = {c["id"]: c for c in compiled["claims"]}
    flagged_refs = {
        by_id[f["claim_id"]]["ref"]
        for f in decoded["flagged"]
        if f["claim_id"] in by_id
    }
    assert "TERMINAL" in flagged_refs, decoded["flagged"]
    assert abs(_qty(decoded, "TANK_FARM") - 4500) < 1.0


def test_kilo_lima_flags_kilo_eod():
    """Kilo EOD 900 vs books 500. L1 recovers 500 and flags the EOD lie."""
    graph = _ingest("kilo_lima_logistics.csv")
    compiled = compile_graph(graph)
    x_hat, residuals = _l1_solve(compiled)
    decoded = decode(x_hat, residuals, compiled, graph)

    assert abs(_qty(decoded, "FOB_KILO") - 500) < 1.0
    by_id = {c["id"]: c for c in compiled["claims"]}
    flagged_refs = {
        by_id[f["claim_id"]]["ref"]
        for f in decoded["flagged"]
        if f["claim_id"] in by_id
    }
    assert "FOB_KILO" in flagged_refs, decoded["flagged"]
    assert "RAILHEAD" not in flagged_refs
    assert abs(_qty(decoded, "RAILHEAD") - 1200) < 1.0


def test_fuel_b_two_corruptions():
    """
    Ironside→Talon sent 1300 / received 900. Ironside EOD matches 900.
    Talon EOD 1450 is a second lie (would be 1900 if flow=900).
    L1 must pick flow=900, keep Ironside, flag the sent 1300 and Talon EOD.
    """
    graph = _ingest("fuel_B_two_corruptions.csv")
    compiled = compile_graph(graph)
    x_hat, residuals = _l1_solve(compiled)
    decoded = decode(x_hat, residuals, compiled, graph)

    assert abs(_flow(decoded, "FOB_IRONSIDE", "OP_TALON") - 900) < 1.0
    assert abs(_qty(decoded, "FOB_IRONSIDE") - 7100) < 1.0
    assert abs(_qty(decoded, "OP_TALON") - 1900) < 1.0

    by_id = {c["id"]: c for c in compiled["claims"]}
    flagged = [by_id[f["claim_id"]] for f in decoded["flagged"] if f["claim_id"] in by_id]
    flagged_edge = [
        c for c in flagged
        if c.get("type") == "edge" and c.get("ref") == "e_FOB_IRONSIDE_OP_TALON"
    ]
    flagged_nodes = {c["ref"] for c in flagged if c.get("type") == "node"}
    assert any(abs(float(c["value"]) - 1300) < 1 or "sent" in str(c.get("source", "")).lower()
               for c in flagged_edge), flagged
    assert "OP_TALON" in flagged_nodes, decoded["flagged"]
    assert "FOB_IRONSIDE" not in flagged_nodes


def test_coalesce_sums_shipments_not_flow_replicates():
    """shipment_sent rows add; water-style `flow` replicates stay separate."""
    graph = {
        "nodes": [
            {"id": "A", "initial": 2000, "sinks": "none"},
            {"id": "B", "initial": 0, "sinks": "none"},
        ],
        "edges": [{"id": "e_A_B", "from": "A", "to": "B"}],
        "claims": [
            {"id": "s1", "type": "edge", "ref": "e_A_B", "value": 800, "source": "shipment_sent", "weight": 1},
            {"id": "s2", "type": "edge", "ref": "e_A_B", "value": 300, "source": "shipment_sent", "weight": 1},
            {"id": "r1", "type": "edge", "ref": "e_A_B", "value": 800, "source": "shipment_received", "weight": 1},
            {"id": "r2", "type": "edge", "ref": "e_A_B", "value": 300, "source": "shipment_received", "weight": 1},
            {"id": "eod_a", "type": "node", "ref": "A", "value": 900, "source": "eod_on_hand", "weight": 1},
            {"id": "eod_b", "type": "node", "ref": "B", "value": 1100, "source": "eod_on_hand", "weight": 1},
        ],
    }
    compiled = compile_graph(graph)
    # 2 coalesced edge rows (sent sum, received sum) + 2 node rows
    assert compiled["report"]["n_claims"] == 4
    assert 1100.0 in set(compiled["y"])

    x_hat, residuals = _l1_solve(compiled)
    decoded = decode(x_hat, residuals, compiled, graph)
    assert abs(_flow(decoded, "A", "B") - 1100) < 1e-3
    assert not decoded["flagged"]

    flow_graph = {
        "nodes": [
            {"id": "J1", "initial": 0, "sinks": "none"},
            {"id": "J2", "initial": 0, "sinks": "none"},
        ],
        "edges": [{"id": "e_J1_J2", "from": "J1", "to": "J2"}],
        "claims": [
            {"id": "f1", "type": "edge", "ref": "e_J1_J2", "value": 10.0, "source": "flow", "weight": 1},
            {"id": "f2", "type": "edge", "ref": "e_J1_J2", "value": 10.2, "source": "flow", "weight": 1},
        ],
    }
    flow_c = compile_graph(flow_graph)
    assert flow_c["report"]["n_claims"] == 2, "flow replicates must not be summed"


def test_l1_skips_missing_refs_and_empty_claims():
    """Missing claim refs must not crash; empty claim sets must still solve."""
    graph = {
        "nodes": [{"id": "A", "initial": 5, "sinks": "none"}],
        "edges": [],
        "claims": [
            {"id": "bad", "type": "edge", "ref": "nope", "value": 1, "source": "x", "weight": 1},
            {"id": "ok", "type": "node", "ref": "A", "value": 5, "source": "eod", "weight": 1},
        ],
    }
    compiled = compile_graph(graph)
    assert compiled["claim_ids"] == ["ok"]
    x_hat, residuals = _l1_solve(compiled)
    assert abs(x_hat[compiled["index"]["qty_A"]] - 5) < 1e-6

    empty = compile_graph({
        "nodes": [{"id": "A", "initial": 3, "sinks": "none"}],
        "edges": [],
        "claims": [],
    })
    x_hat, residuals = _l1_solve(empty)
    assert abs(x_hat[empty["index"]["qty_A"]] - 3) < 1e-6
    assert list(residuals) == []


def test_known_consumption_enters_balance():
    """
    Consumption is a known withdrawal, not a second reading of final stock.

    FOB: opening 600 + 1000 received - 200 used = 1400 EOD.
    If consumption were compiled as qty, L1 would fight 200 vs 1400.
    """
    graph = {
        "nodes": [
            {"id": "DEPOT", "initial": 4000, "sinks": "none"},
            {"id": "FOB", "initial": 600, "sinks": "none"},
        ],
        "edges": [{"id": "e_DEPOT_FOB", "from": "DEPOT", "to": "FOB"}],
        "claims": [
            {"id": "s", "type": "edge", "ref": "e_DEPOT_FOB", "value": 1000, "source": "shipment_sent", "weight": 1},
            {"id": "r", "type": "edge", "ref": "e_DEPOT_FOB", "value": 1000, "source": "shipment_received", "weight": 1},
            {"id": "use", "type": "node", "ref": "FOB", "value": 200, "source": "consumption", "weight": 1},
            {"id": "eod_d", "type": "node", "ref": "DEPOT", "value": 3000, "source": "eod_on_hand", "weight": 1},
            {"id": "eod_f", "type": "node", "ref": "FOB", "value": 1400, "source": "eod_on_hand", "weight": 1},
        ],
    }
    compiled = compile_graph(graph)
    assert "sink_FOB" not in compiled["index"]
    assert compiled["known_sinks"].get("FOB") == 200
    assert all(c.get("type") != "sink" for c in compiled["claims"])
    assert compiled["report"]["n_claims"] == 4  # 2 edge + 2 eod; consumption is a balance offset

    x_hat, residuals = _l1_solve(compiled)
    decoded = decode(x_hat, residuals, compiled, graph)
    assert abs(_qty(decoded, "FOB") - 1400) < 1.0
    assert abs(_qty(decoded, "DEPOT") - 3000) < 1.0
    assert not decoded["sinks"], decoded["sinks"]
    assert not decoded["flagged"], decoded["flagged"]


def test_node_known_sink_constant():
    """sinks='known' with a numeric sink subtracts from the balance (no extra var)."""
    graph = {
        "nodes": [
            {"id": "A", "initial": 1000, "sinks": "known", "sink": 200},
            {"id": "B", "initial": 0, "sinks": "none"},
        ],
        "edges": [{"id": "e", "from": "A", "to": "B"}],
        "claims": [
            {"id": "f", "type": "edge", "ref": "e", "value": 300, "source": "flow", "weight": 1},
            {"id": "ea", "type": "node", "ref": "A", "value": 500, "source": "eod_on_hand", "weight": 1},
            {"id": "eb", "type": "node", "ref": "B", "value": 300, "source": "eod_on_hand", "weight": 1},
        ],
    }
    compiled = compile_graph(graph)
    assert "sink_A" not in compiled["index"]
    x_hat, residuals = _l1_solve(compiled)
    decoded = decode(x_hat, residuals, compiled, graph)
    assert abs(_qty(decoded, "A") - 500) < 1e-3
    assert abs(_qty(decoded, "B") - 300) < 1e-3
    assert not decoded["flagged"]


def test_water_demand_stays_qty():
    """Water demand claims must remain qty observations, not known-sink withdrawals."""
    graph = _load("toy_water_sinks.json")
    compiled = compile_graph(graph)
    assert all(c.get("type") != "sink" for c in compiled["claims"])
    assert "sink_BASE_B" in compiled["index"]
    assert compiled["report"]["n_claims"] == len(graph["claims"])


def test_fuel_e_consumption_is_clean():
    """fuel_E_consumption.csv is an uncorrupted daily snapshot with known use."""
    graph = _ingest("fuel_E_consumption.csv")
    compiled = compile_graph(graph)
    x_hat, residuals = _l1_solve(compiled)
    decoded = decode(x_hat, residuals, compiled, graph)

    assert compiled["known_sinks"].get("FOB_IRONSIDE") == 800
    assert compiled["known_sinks"].get("OP_VIPER") == 120
    assert "sink_FOB_IRONSIDE" not in compiled["index"]
    assert not decoded["sinks"], decoded["sinks"]
    assert not decoded["flagged"], decoded["flagged"]
    assert abs(_qty(decoded, "FOB_IRONSIDE") - 6300) < 1.0
    assert abs(_qty(decoded, "FOB_KESTREL") - 4600) < 1.0
    assert abs(_qty(decoded, "OP_TALON") - 1750) < 1.0
    assert abs(_qty(decoded, "OP_VIPER") - 930) < 1.0
    assert abs(_qty(decoded, "PORT_HAVEN") - 11000) < 1.0


def test_unmetered_leak_is_ambiguous():
    """True leak at BASE_B cannot be named as loss without a meter (k=0)."""
    result = _run("toy_water_sinks.json")
    decoded = result["decoded"]
    assert any(a["id"] == "BASE_B" for a in decoded["ambiguous"])
    assert not decoded["loss"]
    assert not decoded["flagged"]


def test_metered_leak_is_named_loss():
    """Drain meter + level sensor: BASE_B sink is identifiable loss."""
    graph = _load("toy_water_metered.json")
    compiled = compile_graph(graph)
    x_hat, residuals = _l1_solve(compiled)
    decoded = decode(x_hat, residuals, compiled, graph)
    assert "BASE_B" in compiled["report"]["metered_sink_ids"]
    assert any(s["id"] == "BASE_B" for s in decoded["loss"])
    assert not decoded["ambiguous"]


def test_mad_threshold_separates_noise_from_sparse_a():
    from orb.decode import flag_threshold
    r = np.array([0.1, -0.2, 0.15, -0.05, 0.0, 0.08, 80.0])
    sigma, thresh = flag_threshold(r)
    assert 80.0 > thresh
    assert abs(0.2) <= thresh


def test_half_and_half_residuals_still_flag():
    """[0, 0, 320, 320] must not inflate σ until 320 looks like noise."""
    from orb.decode import flag_threshold
    sigma, thresh = flag_threshold(np.array([320.0, 320.0, 0.0, 0.0]))
    assert 320.0 > thresh
    assert thresh <= 0.5 or sigma < 1.0


def test_blind_edge_without_third_channel():
    graph = {
        "nodes": [
            {"id": "A", "initial": 0, "sinks": "none"},
            {"id": "B", "initial": 0, "sinks": "none"},
        ],
        "edges": [{"id": "e_A_B", "from": "A", "to": "B"}],
        "claims": [
            {"id": "s", "type": "edge", "ref": "e_A_B", "value": 10, "source": "shipment_sent", "weight": 1},
            {"id": "r", "type": "edge", "ref": "e_A_B", "value": 10, "source": "shipment_received", "weight": 1},
        ],
    }
    compiled = compile_graph(graph)
    assert compiled["report"]["blind_edges"]
    decoded = decode(*_l1_solve(compiled), compiled, graph)
    assert decoded["undetectable"]
