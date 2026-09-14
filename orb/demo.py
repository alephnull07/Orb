"""
orb/demo.py
-----------
run_demo(path) — unified demo entry point for any file.

    any file  →  ingest (routes by extension)  →  claims  →  compile  →  L1  →  decode

CSV and text both reach compile() through the same claim contract.
No per-path special casing downstream of ingest.
"""

from __future__ import annotations

import json
from pathlib import Path

from .compile   import compile as compile_graph
from .decode    import decode
from .ingest    import build_graph
from .run_graph import _l1_solve
from .water_world import WATER_COLUMN_MAPPING


def run_demo(path: str | Path, truth_path: str | Path | None = None) -> dict:
    """
    Run the full ORB pipeline on *path* and print a formatted report.

    Parameters
    ----------
    path        Input file (CSV or TXT).
    truth_path  Optional ground-truth JSON for scoring (water CSV only).

    Returns
    -------
    dict with keys: graphs, compiled, decoded, report, ingest_report
    """
    path = Path(path)
    ext  = path.suffix.lower()

    # ── ingest (routes by extension) ──────────────────────────────────────────
    if ext in {".csv", ".tsv", ".xlsx"}:
        graphs, ingest_report = build_graph(
            path, sinks="unknown", column_mapping=WATER_COLUMN_MAPPING,
        )
    else:
        # lambda_w=0.1: unanimous consensus edges get weight 10× vs single-source
        graphs, ingest_report = build_graph(path, lambda_w=0.1)

    if not graphs:
        print(f"[demo] No graphs produced from {path}")
        return {}

    graph = graphs[0]

    # ── compile → L1 → decode (SAME path for both) ───────────────────────────
    compiled  = compile_graph(graph)
    x_hat, residuals = _l1_solve(compiled)
    decoded   = decode(x_hat, residuals, compiled, graph)
    report    = compiled["report"]

    # ── print report ──────────────────────────────────────────────────────────
    mode = ingest_report.get("mode", "?")
    print(f"\n{'='*60}")
    print(f"  run_demo({path.name})   mode={mode}")
    print(f"{'='*60}")

    if mode == "TABULAR":
        _print_tabular_report(ingest_report, report, decoded, truth_path)
    elif mode == "RECORD":
        _print_record_report(ingest_report, report, decoded, graph)

    return {
        "graph":         graph,
        "compiled":      compiled,
        "decoded":        decoded,
        "report":        report,
        "ingest_report": ingest_report,
    }


# ---------------------------------------------------------------------------
# Formatted output printers
# ---------------------------------------------------------------------------

def _print_tabular_report(ingest_report, report, decoded, truth_path):
    mapping = ingest_report.get("mapping", {})
    print(f"\n  Column mapping:")
    for k, v in mapping.items():
        print(f"    {k}: {v}")

    n_claims = report["n_claims"]
    print(f"\n  Claim count: {n_claims}")

    print(f"\n  Identifiability:")
    print(f"    rank:           {report['rank']} / {report['n_vars']}")
    print(f"    identifiable:   {report['identifiable']}")
    print(f"    correctable_k:  {report['correctable_k']}")

    sinks = sorted(decoded["sinks"], key=lambda s: abs(s["sink"]), reverse=True)
    print(f"\n  Top 5 sinks by magnitude:")
    for i, s in enumerate(sinks[:5]):
        print(f"    #{i+1}  {s['id']:<8}  sink = {s['sink']:.2f}")

    # Scoring against truth
    if truth_path and Path(truth_path).exists():
        with open(truth_path) as fh:
            truth = json.load(fh)
        leak_nid = truth["leak_node"]
        leak_size = truth["leak_size"]

        leak_rank = None
        for i, s in enumerate(sinks):
            if s["id"] == leak_nid:
                leak_rank = i + 1
                break

        top_mag = abs(sinks[0]["sink"]) if sinks else 0
        second_mag = abs(sinks[1]["sink"]) if len(sinks) > 1 else 0
        margin = top_mag - second_mag

        print(f"\n  Scoring:")
        print(f"    true leak node: {leak_nid}")
        print(f"    true leak size: {leak_size}")
        print(f"    recovered rank: #{leak_rank}")
        print(f"    margin over #2: {margin:.2f}")
    print()


def _print_record_report(ingest_report, report, decoded, graph):
    record_count = ingest_report.get("record_count", 0)
    n_claims = report["n_claims"]
    print(f"\n  Record count: {record_count}")
    print(f"  Claim count:  {n_claims}")

    # Weights and reader agreement
    claims = graph.get("claims", [])
    weights = [c.get("weight", 1.0) for c in claims]
    if weights:
        print(f"  Claim weights: min={min(weights):.4f}  max={max(weights):.4f}  "
              f"mean={sum(weights)/len(weights):.4f}")

    # Reader agreement from consensus edges
    edges = graph.get("edges", [])
    print(f"  Reader agreement: {len(edges)} edges extracted")

    print(f"\n  Identifiability:")
    print(f"    rank:           {report['rank']} / {report['n_vars']}")
    print(f"    identifiable:   {report['identifiable']}")
    print(f"    correctable_k:  {report['correctable_k']}")

    # Decoded node values
    print(f"\n  Decoded nodes:")
    for nd in decoded["nodes"]:
        print(f"    {nd['id']:<16}  qty = {nd['qty']:.2f}")

    # Flagged claims with provenance
    if decoded["flagged"]:
        print(f"\n  Flagged claims ({len(decoded['flagged'])}):")
        for f in decoded["flagged"]:
            # Find the source record for provenance
            claim = next((c for c in claims if c["id"] == f["claim_id"]), {})
            source = claim.get("source", "?")
            print(f"    [{f['claim_id']}]  |r|={abs(f['residual']):.2f}  "
                  f"source={source}  type={f['type']}")
    else:
        print("\n  No claims flagged.")
    print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m orb.demo <file> [truth.json]", file=sys.stderr)
        sys.exit(1)
    truth = sys.argv[2] if len(sys.argv) > 2 else None
    run_demo(sys.argv[1], truth_path=truth)
