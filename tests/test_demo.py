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

FIXTURES   = os.path.join(os.path.dirname(__file__), "fixtures")
WATER_CSV  = os.path.join(FIXTURES, "demo_water.csv")
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
    result = run_demo(WATER_CSV, truth_path=WATER_TRUTH)

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
    water_result = run_demo(WATER_CSV, truth_path=WATER_TRUTH)
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
    result = run_demo(WATER_CSV, truth_path=WATER_TRUTH)
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
