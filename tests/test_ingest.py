"""
tests/test_ingest.py
--------------------
5 tests for the ingest layer (orb/ingest.py + helpers).

Run with:   pytest tests/test_ingest.py -v

Tests
-----
1. supply_sample.txt → RECORD mode; every record has {record_id, file, line}
2. LeakDB scada_snapshot.csv → TABULAR; mapping produced; LLM calls == 1 (not N_rows)
3. leak_demand channel excluded; exclusion appears in report
4. Claim with type='leak' raises ValueError from validate_claims
5. Second call on same file → zero LLM calls (pure cache hit)
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orb.compile       import compile as compile_graph
from orb.ingest        import build_graph, leakage_guard, validate_claims
from orb._ingest_record import split_records

FIXTURES   = os.path.join(os.path.dirname(__file__), "fixtures")
SUPPLY_TXT = os.path.join(FIXTURES, "supply_sample.txt")
LEAKDB_CSV = os.path.join(
    os.path.dirname(__file__), "..",
    "data", "supply_drops", "user_run", "corrupted", "scada_snapshot.csv",
)


# ---------------------------------------------------------------------------
# Test 1 — provenance fields on every record
# ---------------------------------------------------------------------------

def test_record_provenance():
    """
    split_records on supply_sample.txt must return records, each with
    {record_id, file, line} fields populated and non-empty.
    """
    text = open(SUPPLY_TXT, encoding="utf-8").read()
    records = split_records(text, SUPPLY_TXT)

    assert len(records) >= 4, f"Expected >= 4 records, got {len(records)}"

    for r in records:
        assert "record_id" in r and r["record_id"], f"Missing record_id: {r}"
        assert "file"      in r and r["file"],      f"Missing file: {r}"
        assert "line"      in r and r["line"] >= 1,  f"Missing/bad line: {r}"


# ---------------------------------------------------------------------------
# Test 2 — TABULAR mode: ONE LLM call for LeakDB CSV (not N_rows)
# ---------------------------------------------------------------------------

def test_tabular_one_llm_call():
    """
    build_graph on scada_snapshot.csv must:
      - route to TABULAR mode (report['mode'] == 'TABULAR')
      - return a column mapping dict in report['mapping']
      - make exactly 1 LLM call regardless of row count
    Requires ANTHROPIC_API_KEY; skips if absent.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    with tempfile.TemporaryDirectory() as tmp:
        graphs, report = build_graph(LEAKDB_CSV, api_key=api_key, cache_dir=tmp)

    assert report["mode"] == "TABULAR", f"Expected TABULAR mode, got {report['mode']}"
    assert "mapping" in report,         "report must contain 'mapping'"

    mapping = report["mapping"]
    assert mapping.get("entity_column"), "mapping must identify entity_column"
    assert mapping.get("value_column"),  "mapping must identify value_column"

    # LLM calls: 1 on cache miss, 0 on cache hit (both are valid here)
    assert report["llm_calls"] <= 1, (
        f"Expected <= 1 LLM call, got {report['llm_calls']}. "
        "TABULAR mode must map columns with ONE call, not one per row."
    )


# ---------------------------------------------------------------------------
# Test 3 — leak_demand channel excluded from claims
# ---------------------------------------------------------------------------

def test_leak_demand_excluded():
    """
    scada_snapshot.csv contains a 'leak_demand' channel.
    After ingest, the report['exclusions'] must list 'leak_demand' and
    no claim in any graph must have source == 'leak_demand'.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    with tempfile.TemporaryDirectory() as tmp:
        graphs, report = build_graph(LEAKDB_CSV, api_key=api_key, cache_dir=tmp)

    exclusion_names = {e["name"] for e in report.get("exclusions", [])}
    assert "leak_demand" in exclusion_names, (
        f"'leak_demand' must appear in report exclusions; got: {exclusion_names}"
    )

    for g in graphs:
        for claim in g.get("claims", []):
            assert claim.get("source") != "leak_demand", (
                f"Claim {claim['id']} has source='leak_demand'; should be excluded"
            )


# ---------------------------------------------------------------------------
# Test 4 — validate_claims rejects invalid primitive type
# ---------------------------------------------------------------------------

def test_validate_claims_rejects_bad_type():
    """
    validate_claims must raise ValueError when a claim has type='leak'
    (or any type not in {node, edge, aggregate}).
    """
    bad_claims = [
        {"id": "c0", "type": "node",  "ref": "A", "value": 10.0, "source": "s", "weight": 1.0},
        {"id": "c1", "type": "leak",  "ref": "B", "value":  5.0, "source": "s", "weight": 1.0},
    ]
    with pytest.raises(ValueError, match="leak"):
        validate_claims(bad_claims)

    # Valid claims must not raise
    good_claims = [
        {"id": "c0", "type": "node",      "ref": "A", "value": 10.0, "source": "s", "weight": 1.0},
        {"id": "c1", "type": "edge",      "ref": "e0", "value": 5.0, "source": "s", "weight": 1.0},
        {"id": "c2", "type": "aggregate", "refs": ["A", "B"], "value": 15.0, "source": "s", "weight": 1.0},
    ]
    validate_claims(good_claims)  # must not raise


# ---------------------------------------------------------------------------
# Test 5 — cache hit → zero LLM calls on second run
# ---------------------------------------------------------------------------

def test_cache_hit_zero_llm_calls():
    """
    Running build_graph twice on the same file (same cache_dir) must
    yield zero LLM calls on the second run.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    with tempfile.TemporaryDirectory() as tmp:
        # First run: may call LLM (1 call) or hit disk cache (0)
        _, report1 = build_graph(LEAKDB_CSV, api_key=api_key, cache_dir=tmp)

        # Second run: same cache_dir → must be pure cache hit
        _, report2 = build_graph(LEAKDB_CSV, api_key=api_key, cache_dir=tmp)

    assert report2["llm_calls"] == 0, (
        f"Second run must make 0 LLM calls (cache hit), got {report2['llm_calls']}"
    )


# ---------------------------------------------------------------------------
# LeakDB mixed upload: skip labels, bind topology, don't LLM the sidecars
# ---------------------------------------------------------------------------

LEAKDB_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "data", "leakdb", "runs", "run_07", "corrupted",
)


def test_leakdb_bundle_skips_labels_and_joins_topology():
    """
    The UI upload of a LeakDB corrupted/ folder must:
      - skip Labels.csv / scenario_info (ground truth + metadata)
      - treat topology.json as a link schema, not a free-text LLM record
      - resolve Link_* flow rows onto real endpoints
      - produce a compile-able graph without an API key
    """
    from orb.ingest import ingest_paths

    bundle = [
        os.path.join(LEAKDB_DIR, name)
        for name in (
            "Labels.csv",
            "messages.jsonl",
            "scada_snapshot.csv",
            "scenario_info.csv",
            "topology.json",
        )
    ]
    for p in bundle:
        assert os.path.exists(p), p

    graph, report = ingest_paths(bundle)
    skipped_names = {s["file"] for s in report.get("skipped") or []}
    assert "Labels.csv" in skipped_names
    assert "scenario_info.csv" in skipped_names
    assert report.get("schema_links", 0) > 0

    assert graph is not None, f"expected a graph; report={report}"
    assert graph["edges"], "topology/SCADA must produce edges (Link_* → endpoints)"
    assert graph["claims"], "SCADA must produce claims"
    assert any(c["type"] == "edge" for c in graph["claims"]), (
        "flow rows must become edge claims once topology is applied"
    )
    assert not any("leak" in str(c.get("source", "")).lower() for c in graph["claims"])

    compiled = compile_graph(graph)
    assert compiled["report"]["n_claims"] == len(graph["claims"])


def test_observation_csv_named_corruptions_is_not_skipped():
    """Scenario names like fuel_B_two_corruptions.csv are data, not GT sidecars."""
    from orb._ingest_schema import skip_reason
    from orb.ingest import build_graph

    path = os.path.join(FIXTURES, "fuel_B_two_corruptions.csv")
    assert os.path.exists(path)
    assert skip_reason(path) is None
    graphs, report = build_graph(path)
    assert report["mode"] != "SKIP", report
    assert graphs, report
    g = graphs[0]
    assert len(g["nodes"]) == 5, [n["id"] for n in g["nodes"]]
    assert len(g["edges"]) == 5, g["edges"]


def test_site_core_does_not_smush_prefixed_sites():
    from src.etl.textutil import site_core

    assert site_core("FOB ALPHA") == "alpha"
    assert site_core("FOB BRAVO") == "bravo"
    assert site_core("OP CRESCENT") == "crescent"
    assert site_core("OP DELTA") == "delta"
    assert site_core("OP ECHO") == "echo"
    assert site_core("MAIN") == site_core("MAIN DEPOT") == "depot"
    assert site_core("FOB ALPHA") != site_core("FOB BRAVO")
    assert site_core("Alpha-1") == site_core("FOB ALPHA") == site_core("Alpha depot")
    assert site_core("MAIN DEPOT") == site_core("Depot-Main") == site_core("the depot") == site_core("MAIN")
    assert site_core("Junction 10") != site_core("Junction 11")
    assert site_core("Junction 10") == "junction 10"


def test_collapse_keeps_fob_and_op_sites_distinct():
    from orb._ingest_record import _collapse_node_aliases
    from src.etl.textutil import canon, slug

    def node(display):
        return {
            "id": slug(display),
            "key": canon(display),
            "display": display,
            "aliases": [display],
        }

    graphs = {
        "scout": {
            "nodes": [
                node("FOB ALPHA"),
                node("FOB BRAVO"),
                node("OP CRESCENT"),
                node("OP DELTA"),
                node("OP ECHO"),
                node("MAIN DEPOT"),
                node("Alpha-1"),
            ],
            "edges": [],
            "trusted_constraint_rows": [],
        }
    }
    _collapse_node_aliases(graphs)
    keys = {n["key"] for n in graphs["scout"]["nodes"]}
    ids = {n["id"] for n in graphs["scout"]["nodes"]}
    assert len(keys) == 6, keys
    assert any("BRAVO" in i for i in ids)
    assert any("DELTA" in i for i in ids)
    assert any("ECHO" in i for i in ids)
    assert canon("Alpha-1") not in keys


def test_field_reports_keeps_six_sites():
    """supply_field_reports.txt is six sites, five hops — not 3 nodes / 2 edges."""
    from orb.ingest import build_graph

    path = os.path.join(FIXTURES, "supply_field_reports.txt")
    if not os.path.exists(path):
        return
    graphs, _report = build_graph(path)
    assert graphs, "expected a graph from field reports"
    g = graphs[0]
    ids = {n["id"] for n in g["nodes"]}
    joined = " ".join(ids).upper()
    assert len(g["nodes"]) >= 6, f"smushed sites: {ids}"
    for token in ("ALPHA", "BRAVO", "CRESCENT", "DELTA", "ECHO", "DEPOT"):
        assert token in joined, f"missing {token} in {ids}"
    assert len(g["edges"]) >= 5, f"too few hops: {g['edges']}"
