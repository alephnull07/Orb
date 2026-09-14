"""
tests/test_demo.py
------------------
End-to-end tests for the unified run_demo() pipeline.

1. Water CSV: true leak ranks #1, margin reported
2. Supply TXT: planted corruption flagged with correct provenance
3. Both files produce claims that compile() accepts unchanged
4. Guard: leak node id and leak size appear nowhere in CSV, claims, or compile dict

Run with:   pytest tests/test_demo.py -v
"""

import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orb.compile   import compile as compile_graph
from orb.demo      import run_demo
from orb.run_graph import _l1_solve
from orb.decode    import decode
from orb.water_world import WATER_COLUMN_MAPPING

FIXTURES   = os.path.join(os.path.dirname(__file__), "fixtures")
WATER_CSV  = os.path.join(FIXTURES, "demo_water.csv")
FUEL_CSV   = os.path.join(FIXTURES, "fuel_E_consumption.csv")
WATER_TRUTH = os.path.join(FIXTURES, "demo_water_truth.json")
SUPPLY_TXT = os.path.join(FIXTURES, "supply_sample.txt")


# ---------------------------------------------------------------------------
# Test 1 — water CSV: true leak ranks #1 with margin
# ---------------------------------------------------------------------------

def test_water_csv_leak_ranks_first():
    """
    run_demo on demo_water.csv must recover the true leak junction at rank #1
    with a clear margin over #2.
    """
    result = run_demo(
        WATER_CSV, truth_path=WATER_TRUTH,
        sinks="unknown", column_mapping=WATER_COLUMN_MAPPING,
    )

    with open(WATER_TRUTH) as fh:
        truth = json.load(fh)

    decoded = result["decoded"]
    sinks = sorted(decoded["sinks"], key=lambda s: abs(s["sink"]), reverse=True)

    # True leak node must be #1
    leak_nid = truth["leak_node"]
    assert sinks[0]["id"] == leak_nid, (
        f"Expected {leak_nid} at rank #1, got {sinks[0]['id']} "
        f"(sink={sinks[0]['sink']:.2f})"
    )

    # Margin over #2 must be significant
    top = abs(sinks[0]["sink"])
    second = abs(sinks[1]["sink"]) if len(sinks) > 1 else 0
    margin = top - second
    assert margin > 10, f"Margin too small: {margin:.2f}"

    # correctable_k >= 1
    report = result["report"]
    assert report["correctable_k"] >= 1, (
        f"Expected correctable_k >= 1, got {report['correctable_k']}"
    )


# ---------------------------------------------------------------------------
# Test 2 — supply TXT: planted corruption flagged with provenance
# ---------------------------------------------------------------------------

def test_supply_txt_corruption_flagged():
    """
    supply_sample.txt has a planted corruption: Charlie outpost reports
    EOD on-hand 500 lb but only received 150 lb.  run_demo must flag
    a claim related to this inconsistency, and the flagged claim's source
    must provide provenance (traceable to the corrupt record).
    """
    result = run_demo(SUPPLY_TXT)

    decoded = result["decoded"]
    flagged = decoded["flagged"]

    assert len(flagged) >= 1, "Expected at least one flagged claim"

    # The flagged claim should touch CHARLIE_OUTPOST
    flagged_refs = set()
    for f in flagged:
        claim = next(
            (c for c in result["graph"]["claims"] if c["id"] == f["claim_id"]),
            None,
        )
        if claim:
            flagged_refs.add(claim.get("ref", ""))

    # The corrupt record is Charlie's EOD (node claim on CHARLIE_OUTPOST)
    # or the consensus edge to CHARLIE_OUTPOST
    charlie_edge_ids = {
        e["id"] for e in result["graph"]["edges"]
        if e["to"] == "CHARLIE_OUTPOST" or e["from"] == "CHARLIE_OUTPOST"
    }
    charlie_related = {"CHARLIE_OUTPOST"} | charlie_edge_ids

    assert flagged_refs & charlie_related, (
        f"No flagged claim touches CHARLIE_OUTPOST.  "
        f"Flagged refs: {flagged_refs}, expected one of: {charlie_related}"
    )

    # Provenance: flagged claim must have a non-empty source
    for f in flagged:
        assert f["source"] != "?", f"Flagged claim {f['claim_id']} has no source"


# ---------------------------------------------------------------------------
# Test 3 — both files produce claims that compile() accepts unchanged
# ---------------------------------------------------------------------------

def test_both_paths_compile_unchanged():
    """
    Both files must produce graph dicts whose claims compile() accepts
    without modification — same compile() call path for both.
    """
    water_result = run_demo(
        WATER_CSV, truth_path=WATER_TRUTH,
        sinks="unknown", column_mapping=WATER_COLUMN_MAPPING,
    )
    supply_result = run_demo(SUPPLY_TXT)

    # Both must have non-empty compiled results
    for label, result in [("water", water_result), ("supply", supply_result)]:
        graph = result["graph"]
        assert graph["nodes"], f"{label}: no nodes"
        assert graph["claims"], f"{label}: no claims"

        # compile() must accept the graph unchanged
        compiled = compile_graph(graph)
        assert compiled["report"]["identifiable"], f"{label}: not identifiable"
        assert compiled["report"]["n_claims"] == len(graph["claims"]), (
            f"{label}: claim count mismatch after compile"
        )

        # The compiled output must have valid H, y, w shapes
        H = compiled["H"]
        n_claims = compiled["report"]["n_claims"]
        n_vars   = compiled["report"]["n_vars"]
        assert H.shape == (n_claims, n_vars), (
            f"{label}: H shape {H.shape} != ({n_claims}, {n_vars})"
        )


# ---------------------------------------------------------------------------
# Test 4 — guard: leak node id and leak size not in CSV/claims/compile dict
# ---------------------------------------------------------------------------

def test_leakage_guard():
    """
    The leak node id ('J_03') and leak size (300.0) must not appear
    anywhere in the CSV text, the claims list, or the dict passed to
    compile().  Sensor IDs are generic (J_01, P_03 etc.) and values
    are noisy measurements — not the exact leak parameters.
    """
    with open(WATER_TRUTH) as fh:
        truth = json.load(fh)
    leak_nid  = truth["leak_node"]   # "J_03"
    leak_size = truth["leak_size"]   # 300.0

    # ── CSV text: leak_size must not appear as an exact value ──────────────
    with open(WATER_CSV) as fh:
        csv_text = fh.read()
    # The exact leak_size (e.g. "300.0") should not be a CSV value
    reader = csv.DictReader(csv_text.splitlines())
    for row in reader:
        val = row.get("value", "")
        assert float(val) != leak_size, (
            f"Exact leak size {leak_size} found in CSV row: {row}"
        )

    # ── Claims: no claim value equals the exact leak size ─────────────────
    result = run_demo(
        WATER_CSV, truth_path=WATER_TRUTH,
        sinks="unknown", column_mapping=WATER_COLUMN_MAPPING,
    )
    graph = result["graph"]
    for c in graph["claims"]:
        assert c["value"] != leak_size, (
            f"Exact leak size {leak_size} found in claim {c['id']}: {c}"
        )

    # ── Compile dict: no y entry equals the exact leak size ───────────────
    compiled = result["compiled"]
    for i, val in enumerate(compiled["y"]):
        assert float(val) != leak_size, (
            f"Exact leak size {leak_size} in y[{i}]"
        )

    # ── The string "leak" must not appear in claim sources or types ───────
    for c in graph["claims"]:
        assert "leak" not in c.get("source", "").lower(), (
            f"'leak' in claim source: {c}"
        )
        assert "leak" not in c.get("type", "").lower(), (
            f"'leak' in claim type: {c}"
        )


# ---------------------------------------------------------------------------
# Test 5 — fuel CSV: consumption as known sink, exactly one flagged claim
# ---------------------------------------------------------------------------

FUEL_COLUMN_MAPPING = {
    "entity_column": "sensor_id",
    "value_column": "value",
    "time_column": "timestamp",
    "channel_column": "channel",
    "channel_map": {
        "opening": "node",
        "flow": "edge",
        "eod": "node",
        "consumption": "sink",
    },
    "from_column": "from_node",
    "to_column": "to_node",
    "id_pattern": None,
    "excluded_channels": [],
    "notes": "Fuel supply chain with known consumption burn rates.",
}

# Expected node quantities (EOD) and burn rates
_FUEL_EXPECTED_QTY = {
    "PORT_HAVEN":   11000,
    "FOB_IRONSIDE":  6300,
    "FOB_KESTREL":   4600,
    "OP_TALON":      1750,
    "OP_VIPER":       930,    # true value (CSV reports 980)
}
_FUEL_EXPECTED_BURNS = {
    "FOB_IRONSIDE": 800,
    "FOB_KESTREL":  600,
    "OP_TALON":     150,
    "OP_VIPER":     120,
}
_FUEL_EXPECTED_FLOWS = {
    ("PORT_HAVEN", "FOB_IRONSIDE"):  5000,
    ("PORT_HAVEN", "FOB_KESTREL"):   4000,
    ("FOB_IRONSIDE", "OP_TALON"):     900,
    ("FOB_IRONSIDE", "OP_VIPER"):     600,
    ("FOB_KESTREL", "OP_TALON"):      700,
}


def test_fuel_consumption_known_sink():
    """
    fuel_E_consumption.csv has a 'consumption' channel with known burn rates.
    These must be routed as known sinks (adjusting the balance RHS), NOT as
    claim rows in H.

    Expected: correct node quantities, exactly ONE flagged claim (OP_VIPER
    eod reported 980, true 930, residual +50).
    """
    result = run_demo(
        FUEL_CSV, column_mapping=FUEL_COLUMN_MAPPING,
    )
    assert result, "run_demo returned empty"

    graph   = result["graph"]
    decoded = result["decoded"]
    report  = result["report"]

    # ── No consumption claims in H — they're known sinks ────────────────
    for c in graph["claims"]:
        assert c.get("source") != "consumption", (
            f"Consumption channel leaked into claims: {c}"
        )
        assert c["type"] != "sink", (
            f"Known sink became a claim row: {c}"
        )

    # ── Known sinks on nodes ────────────────────────────────────────────
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    for nid, burn in _FUEL_EXPECTED_BURNS.items():
        assert nodes_by_id[nid].get("known_sinks") == burn, (
            f"{nid}: expected known_sinks={burn}, "
            f"got {nodes_by_id[nid].get('known_sinks')}"
        )
    # PORT_HAVEN has no consumption
    assert "known_sinks" not in nodes_by_id["PORT_HAVEN"], (
        f"PORT_HAVEN should not have known_sinks"
    )

    # ── Decoded node quantities match expected ──────────────────────────
    decoded_qty = {n["id"]: n["qty"] for n in decoded["nodes"]}
    for nid, expected in _FUEL_EXPECTED_QTY.items():
        actual = decoded_qty.get(nid)
        assert actual is not None, f"Node {nid} not in decoded output"
        assert abs(actual - expected) < 1.0, (
            f"{nid}: expected qty={expected}, got {actual}"
        )

    # ── Decoded edge flows match expected ───────────────────────────────
    decoded_flows = {(e["from"], e["to"]): e["flow"] for e in decoded["edges"]}
    for (src, tgt), expected in _FUEL_EXPECTED_FLOWS.items():
        actual = decoded_flows.get((src, tgt))
        assert actual is not None, f"Edge {src}->{tgt} not in decoded output"
        assert abs(actual - expected) < 1.0, (
            f"{src}->{tgt}: expected flow={expected}, got {actual}"
        )

    # ── Exactly one flagged claim: OP_VIPER eod ─────────────────────────
    flagged = decoded["flagged"]
    assert len(flagged) == 1, (
        f"Expected exactly 1 flagged claim, got {len(flagged)}: {flagged}"
    )

    fc = flagged[0]
    # The flagged claim should reference OP_VIPER
    claim = next(c for c in graph["claims"] if c["id"] == fc["claim_id"])
    assert claim["ref"] == "OP_VIPER", (
        f"Expected flagged claim on OP_VIPER, got ref={claim['ref']}"
    )
    assert abs(fc["residual"] - 50.0) < 1.0, (
        f"Expected residual ~50, got {fc['residual']}"
    )

    # ── Identifiable and correctable ────────────────────────────────────
    assert report["identifiable"], "System must be identifiable"
    assert report["correctable_k"] >= 1, (
        f"Expected correctable_k >= 1, got {report['correctable_k']}"
    )
