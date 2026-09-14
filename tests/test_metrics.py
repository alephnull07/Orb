"""
tests/test_metrics.py
---------------------
Tests for the metrics harness.

1. Corruption mask never reaches the estimator (compile dict has no mask info).
2. Deterministic: same seed → byte-identical results.
3. Smoke test: a small sweep runs without error.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orb.compile import compile as compile_graph
from orb.supply_world import generate_supply_graph
from orb.water_world import generate_water_graph


# ---------------------------------------------------------------------------
# Test 1 — corruption mask never reaches compile dict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("generator", ["supply", "water"])
@pytest.mark.parametrize("k", [0, 2, 4])
@pytest.mark.parametrize("corruption_type", ["random", "correlated", "directional"])
def test_corruption_mask_isolated(generator, k, corruption_type):
    """
    The graph dict passed to compile() must contain NO information about
    which claims are corrupted.  The corruption mask, indices, true state,
    and the word 'corrupt' must not appear in the compile input or output.
    """
    seed = 42
    if generator == "supply":
        graph, truth = generate_supply_graph(
            seed=seed, k=k, corruption_type=corruption_type,
        )
    else:
        graph, truth = generate_water_graph(
            seed=seed, k=k, corruption_type=corruption_type,
        )

    # Serialize graph to JSON — no mask info should be present
    graph_json = json.dumps(graph, default=str)
    assert "corrupt" not in graph_json.lower(), (
        f"'corrupt' found in graph dict: {graph_json[:200]}"
    )
    assert "mask" not in graph_json.lower(), (
        f"'mask' found in graph dict: {graph_json[:200]}"
    )
    assert "true_state" not in graph_json.lower(), (
        f"'true_state' found in graph dict"
    )

    # Compile the graph
    compiled = compile_graph(graph)

    # Compiled output must also be clean
    # Check all string-typed values in report
    report_json = json.dumps(compiled["report"], default=str)
    assert "corrupt" not in report_json.lower()
    assert "mask" not in report_json.lower()
    assert "true_state" not in report_json.lower()

    # claim_ids must not encode corruption info
    for cid in compiled["claim_ids"]:
        assert "corrupt" not in cid.lower()


# ---------------------------------------------------------------------------
# Test 2 — deterministic: same seed → identical results
# ---------------------------------------------------------------------------

def test_deterministic():
    """Same seed must produce byte-identical results."""
    from orb.metrics import run_single

    r1 = run_single("supply", k=2, estimator="l1", corruption_type="random", seed=7)
    r2 = run_single("supply", k=2, estimator="l1", corruption_type="random", seed=7)

    # Remove compute_ms (timing varies)
    for r in [r1, r2]:
        r.pop("compute_ms", None)

    assert r1 == r2, f"Non-deterministic results:\n{r1}\nvs\n{r2}"

    # Water world too
    r3 = run_single("water", k=3, estimator="least_squares", corruption_type="directional", seed=13)
    r4 = run_single("water", k=3, estimator="least_squares", corruption_type="directional", seed=13)
    for r in [r3, r4]:
        r.pop("compute_ms", None)
    assert r3 == r4


# ---------------------------------------------------------------------------
# Test 3 — smoke: small sweep runs without error
# ---------------------------------------------------------------------------

def test_smoke_sweep():
    """A small sweep (2 seeds, k=0..2) must complete without error."""
    from orb.metrics import run_sweep

    sweep = run_sweep(
        generators=["supply", "water"],
        k_range=[0, 1, 2],
        estimators=["l1", "least_squares"],
        corruption_types=["random"],
        seeds=[0, 1],
        output_path="/tmp/orb_test_sweep.json",
    )

    assert len(sweep["rows"]) > 0
    assert "meta" in sweep

    # Each row must have required keys
    required = {
        "generator", "k", "estimator", "corruption_type", "seed",
        "false_certainty", "exact_recovery", "state_error_max",
        "state_error_mean", "detection_precision", "detection_recall",
        "compute_ms", "correctable_k",
    }
    for row in sweep["rows"]:
        missing = required - set(row.keys())
        assert not missing, f"Missing keys: {missing}"
