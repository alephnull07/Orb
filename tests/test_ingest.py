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
