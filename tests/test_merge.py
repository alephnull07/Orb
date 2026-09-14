"""
tests/test_merge.py
-------------------
Pure-Python tests for orb/_merge.py — no LLM, no files.

The invariant under test: every independent observation survives the merge
as its own claim row.  Only separate events on one edge within one observer
stream (same file, same channel) are summed.

Run with:   pytest tests/test_merge.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orb._merge import merge_graphs
from orb.compile import compile as compile_graph


def _graph(nodes, edges, claims, ts="all"):
    return {
        "nodes":  [{"id": n, "initial": 0.0, "sinks": "none"} for n in nodes],
        "edges":  [{"id": f"e_{a}_{b}", "from": a, "to": b} for a, b in edges],
        "claims": [dict(c, id=f"c{i}") for i, c in enumerate(claims)],
        "_timestamp": ts,
    }


def _edge(ref, value, source, ts=None, weight=1.0):
    c = {"type": "edge", "ref": ref, "value": value, "source": source, "weight": weight}
    if ts is not None:
        c["timestamp"] = ts
    return c


def _node(ref, value, source, ts=None, weight=1.0):
    c = {"type": "node", "ref": ref, "value": value, "source": source, "weight": weight}
    if ts is not None:
        c["timestamp"] = ts
    return c


def _by_ref(graph):
    out = {}
    for c in graph["claims"]:
        out.setdefault(c.get("ref"), []).append(c)
    return out


# ---------------------------------------------------------------------------
# 1. Sender and receiver rows at the same timestamp stay separate
# ---------------------------------------------------------------------------

def test_sender_receiver_same_timestamp_stay_separate():
    g = _graph(["A", "B"], [("A", "B")], [
        _edge("e_A_B", 1300, "shipment_sent",     "2026-01-01T10:00:00Z"),
        _edge("e_A_B",  900, "shipment_received", "2026-01-01T10:00:00Z"),
    ])
    graph, report = merge_graphs([([g], {"source_file": "log.csv"})])

    edge_claims = [c for c in graph["claims"] if c["type"] == "edge"]
    assert sorted(c["value"] for c in edge_claims) == [900, 1300], edge_claims
    assert {c["channel"] for c in edge_claims} == {"shipment_sent", "shipment_received"}
    # The averaged value must not appear anywhere
    assert all(c["value"] != 1100 for c in graph["claims"])
    assert report["n_edge_streams"] == 2


# ---------------------------------------------------------------------------
# 2. Separate events on one edge are summed within a stream, per stream
# ---------------------------------------------------------------------------

def test_events_sum_within_stream_not_across():
    g = _graph(["A", "B"], [("A", "B")], [
        _edge("e_A_B", 800, "shipment_sent",     "2026-01-01T06:40:00Z"),
        _edge("e_A_B", 800, "shipment_received", "2026-01-01T06:40:00Z"),
        _edge("e_A_B", 300, "shipment_sent",     "2026-01-01T13:20:00Z"),
        _edge("e_A_B", 300, "shipment_received", "2026-01-01T13:20:00Z"),
    ])
    graph, _ = merge_graphs([([g], {"source_file": "log.csv"})])

    edge_claims = [c for c in graph["claims"] if c["type"] == "edge"]
    assert len(edge_claims) == 2
    assert all(c["value"] == 1100 for c in edge_claims)
    assert all(c["n_events"] == 2 for c in edge_claims)
    assert {c["channel"] for c in edge_claims} == {"shipment_sent", "shipment_received"}


# ---------------------------------------------------------------------------
# 3. Conflicting values across files are both kept — no median
# ---------------------------------------------------------------------------

def test_cross_file_conflict_both_rows_survive():
    sender = _graph(["FWD_HOSP_NORTH", "AID_STN_3"], [("FWD_HOSP_NORTH", "AID_STN_3")], [
        _edge("e_FWD_HOSP_NORTH_AID_STN_3", 140, "sender", "2026-09-15T14:10:00Z"),
    ])
    receiver = _graph(["FWD_HOSP_NORTH", "AID_STN_3"], [("FWD_HOSP_NORTH", "AID_STN_3")], [
        _edge("e_FWD_HOSP_NORTH_AID_STN_3", 340, "receiver", "2026-09-15T14:40:00Z"),
    ])
    graph, report = merge_graphs([
        ([sender],   {"source_file": "convoy.jsonl"}),
        ([receiver], {"source_file": "radio.txt"}),
    ])

    edge_claims = [c for c in graph["claims"] if c["type"] == "edge"]
    assert sorted(c["value"] for c in edge_claims) == [140, 340]
    assert {c["source"] for c in edge_claims} == {"convoy.jsonl", "radio.txt"}
    assert all(c["value"] != 240 for c in graph["claims"])       # the old median
    assert report["multi_source_claims"] == 1


# ---------------------------------------------------------------------------
# 4. Identical node readings from two files are two rows, not one
# ---------------------------------------------------------------------------

def test_identical_node_claims_from_two_files_not_deduped():
    a = _graph(["X"], [], [_node("X", 500, "eod_on_hand", "2026-01-01T17:30:00Z")])
    b = _graph(["X"], [], [_node("X", 500, "closing",     "2026-01-01T17:35:00Z")])
    graph, _ = merge_graphs([([a], {"source_file": "a.csv"}), ([b], {"source_file": "b.txt"})])

    node_claims = [c for c in graph["claims"] if c["type"] == "node"]
    assert len(node_claims) == 2
    assert {c["source"] for c in node_claims} == {"a.csv", "b.txt"}
    assert all("merged" not in c["source"] for c in node_claims)


# ---------------------------------------------------------------------------
# 5. Same-file node readings at one timestamp are separate sensors
# ---------------------------------------------------------------------------

def test_same_file_same_timestamp_node_rows_kept():
    g = _graph(["J_01"], [], [
        _node("J_01", 112.09, "demand", "2026-01-01T00:00:00Z"),
        _node("J_01", 111.58, "demand", "2026-01-01T00:00:00Z"),
    ])
    graph, _ = merge_graphs([([g], {"source_file": "water.csv"})])
    vals = sorted(c["value"] for c in graph["claims"] if c["type"] == "node")
    assert vals == [111.58, 112.09]


# ---------------------------------------------------------------------------
# 6. Openings become initials; conflicts are reported, never silent
# ---------------------------------------------------------------------------

def test_openings_to_initial_and_conflicts_reported(capsys):
    a = _graph(["D"], [], [_node("D", 90, "opening_on_hand", "2026-01-01T06:00:00Z")])
    b = _graph(["D"], [], [_node("D", 90, "opening",         "2026-01-01T06:05:00Z")])
    graph, report = merge_graphs([([a], {"source_file": "a.csv"}), ([b], {"source_file": "b.txt"})])
    assert graph["nodes"][0]["initial"] == 90
    assert report["opening_conflicts"] == {}
    assert not [c for c in graph["claims"] if c["type"] == "node"]   # openings are not claims

    c = _graph(["D"], [], [_node("D", 120, "opening", "2026-01-01T06:05:00Z")])
    graph, report = merge_graphs([([a], {"source_file": "a.csv"}), ([c], {"source_file": "c.txt"})])
    assert "D" in report["opening_conflicts"]
    assert {r["value"] for r in report["opening_conflicts"]["D"]} == {90, 120}
    assert "Conflicting opening" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 7. Names are canonicalized across files; the merged graph compiles
# ---------------------------------------------------------------------------

def test_canonical_names_and_compile():
    csv = _graph(["MAIN_DEPOT", "FOB_ALPHA"], [("MAIN_DEPOT", "FOB_ALPHA")], [
        _node("MAIN_DEPOT", 4000, "opening_on_hand", "2026-01-01T06:00:00Z"),
        _node("FOB_ALPHA",   600, "opening_on_hand", "2026-01-01T06:00:00Z"),
        _edge("e_MAIN_DEPOT_FOB_ALPHA", 800, "shipment_sent",     "2026-01-01T06:40:00Z"),
        _edge("e_MAIN_DEPOT_FOB_ALPHA", 800, "shipment_received", "2026-01-01T06:40:00Z"),
        _node("MAIN_DEPOT", 3200, "eod_on_hand", "2026-01-01T17:30:00Z"),
        _node("FOB_ALPHA",  1400, "eod_on_hand", "2026-01-01T17:30:00Z"),
    ])
    txt = _graph(["MAIN DEPOT", "FOB ALPHA"], [("MAIN DEPOT", "FOB ALPHA")], [
        _edge("e_MAIN DEPOT_FOB ALPHA", 800, "sender",   "2026-01-01T06:40:00Z"),
        _edge("e_MAIN DEPOT_FOB ALPHA", 800, "receiver", "2026-01-01T08:05:00Z"),
        _node("FOB ALPHA", 1400, "closing", "2026-01-01T17:35:00Z"),
    ])
    graph, report = merge_graphs([([csv], {"source_file": "s.csv"}), ([txt], {"source_file": "s.txt"})])

    assert [n["id"] for n in graph["nodes"]] == ["FOB_ALPHA", "MAIN_DEPOT"]
    assert len(graph["edges"]) == 1
    assert report["canon_map"] == {"MAIN DEPOT": "MAIN_DEPOT", "FOB ALPHA": "FOB_ALPHA"}

    by_ref = _by_ref(graph)
    assert len(by_ref["e0"]) == 4                    # 2 streams x 2 files
    assert len(by_ref["FOB_ALPHA"]) == 2             # csv eod + txt closing
    assert len(by_ref["MAIN_DEPOT"]) == 1

    compiled = compile_graph(graph)
    assert compiled["report"]["identifiable"]
    assert compiled["report"]["n_claims"] == len(graph["claims"])
