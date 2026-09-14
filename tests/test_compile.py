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
