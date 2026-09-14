"""
tests/test_presets.py
---------------------
Every demo preset runs the FULL pipeline every time.

For each preset: run it twice, assert the graph / decoded / report output is
identical, and assert the L1 solver was actually invoked on both runs rather
than short-circuited by any cache.  Extraction caching by prompt hash is
expected; a cached *result* would fail the invocation check.

Run with:   pytest tests/test_presets.py -v -s      (-s prints the table)
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import orb.demo as demo_mod                                  # noqa: E402
from orb.demo_json import _clean                            # noqa: E402
from orb.presets import PRESETS, format_table, run_preset, summary_row  # noqa: E402

_ROWS: list[dict] = []


def _snapshot(result: dict) -> str:
    keep = {k: result[k] for k in ("graph", "decoded", "report")}
    return json.dumps(_clean(keep), sort_keys=True)


@pytest.mark.parametrize("preset", PRESETS, ids=[p["id"] for p in PRESETS])
def test_preset_runs_full_pipeline_twice(preset, monkeypatch):
    calls = {"n": 0}
    real_solve = demo_mod._l1_solve

    def counting_solve(*a, **kw):
        calls["n"] += 1
        return real_solve(*a, **kw)

    monkeypatch.setattr(demo_mod, "_l1_solve", counting_solve)

    with contextlib.redirect_stdout(io.StringIO()):
        first = run_preset(preset["id"])
    assert calls["n"] == 1, "estimator not invoked on first run"

    with contextlib.redirect_stdout(io.StringIO()):
        second = run_preset(preset["id"])
    assert calls["n"] == 2, "estimator not invoked on second run (short-circuited?)"

    assert _snapshot(first) == _snapshot(second), "second run differs from first"

    # The preset's own mode ran, and every claim source is a fixture path (no temp copies)
    assert first["ingest_report"]["sinks_mode"] == preset["sinks"]
    assert first["preset"]["sinks"] == preset["sinks"]
    for c in first["graph"]["claims"]:
        src = str(c.get("source", ""))
        assert "tests/fixtures" in src.replace("\\", "/"), src
        assert "/var/folders" not in src and "/tmp/" not in src

    _ROWS.append(summary_row(first))


def test_zz_print_summary_table():
    """Runs last: prints the table collected by the parametrized test."""
    assert len(_ROWS) == len(PRESETS), "run the whole module to get the table"
    print("\n\n" + format_table(_ROWS) + "\n")
