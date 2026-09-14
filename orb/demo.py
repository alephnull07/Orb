"""
orb/demo.py
-----------
run_demo(path) — unified demo entry point for any file.
run_demo_multi(paths) — multi-file merge entry point.

    any file  →  ingest (routes by extension)  →  claims  →  compile  →  L1  →  decode

CSV and text both reach compile() through the same claim contract.
No per-path special casing downstream of ingest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .compile   import compile as compile_graph
from .decode    import decode
from .ingest    import build_graph, build_graph_multi
from .run_graph import _l1_solve


def _get_api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or None


def run_demo_multi(paths: list[str | Path], sinks: str = "none") -> dict:
    """
    Run the full ORB pipeline on multiple files simultaneously.
    Each file is ingested via its native mode (CSV→TABULAR, TXT→RECORD, etc.),
    then all graphs are merged into one via build_graph_multi.
    """
    paths = [Path(p) for p in paths]

    graph, ingest_report = build_graph_multi(
        paths,
        api_key=_get_api_key(),
        sinks=sinks,
    )

    if not graph or not graph.get("claims"):
        print(f"[demo] No graphs produced from {[p.name for p in paths]}")
        return {}

    compiled  = compile_graph(graph)
    x_hat, residuals = _l1_solve(compiled)
    decoded   = decode(x_hat, residuals, compiled, graph)
    report    = compiled["report"]

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    stem = "_".join(p.stem for p in paths)[:80]
    graph_out = out_dir / f"graph_{stem}.json"
    with open(graph_out, "w") as fh:
        json.dump(graph, fh, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  run_demo_multi({[p.name for p in paths]})   mode=MERGED")
    print(f"{'='*60}")
    _print_merge_header(ingest_report, graph)
    _print_common(report, decoded, graph)
    print(f"\n  graph.json → {graph_out}")
    print()

    return {
        "graph":         graph,
        "compiled":      compiled,
        "decoded":       decoded,
        "report":        report,
        "ingest_report": ingest_report,
    }


def run_demo(
    path: str | Path,
    truth_path: str | Path | None = None,
    sinks: str = "none",
    column_mapping: dict | None = None,
) -> dict:
    """
    Run the full ORB pipeline on *path* and print a formatted report.

    Parameters
    ----------
    path            Input file (CSV, TXT, JSONL, etc.).
    truth_path      Optional ground-truth JSON for scoring (water CSV only).
    sinks           "none" | "known" | "unknown" — a caller decision, never
                    detected.  none: strict conservation (declared consumption
                    channels are fixed draws).  known: consumption channels are
                    metered sink variables with claim rows.  unknown: every
                    claimed node gets a free sink variable (leak search).
    column_mapping  Pre-supplied mapping for TABULAR mode — skips LLM call.

    Returns
    -------
    dict with keys: graph, compiled, decoded, report, ingest_report
    """
    path = Path(path)

    # Single file: use build_graph_multi for consistent timestamp collapse
    graph, ingest_report = build_graph_multi(
        [path],
        api_key=_get_api_key(),
        sinks=sinks,
        column_mapping=column_mapping,
    )

    if not graph or not graph.get("claims"):
        print(f"[demo] No graphs produced from {path}")
        return {}

    # ── compile → L1 → decode (SAME path for both) ───────────────────────────
    compiled  = compile_graph(graph)
    x_hat, residuals = _l1_solve(compiled)
    decoded   = decode(x_hat, residuals, compiled, graph)
    report    = compiled["report"]

    # ── write graph.json ────────────────────────────────────────────────────
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    graph_out = out_dir / f"graph_{path.stem}.json"
    with open(graph_out, "w") as fh:
        json.dump(graph, fh, indent=2, default=str)

    # ── print report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  run_demo({path.name})   mode=MERGED")
    print(f"{'='*60}")
    _print_merge_header(ingest_report, graph)

    # Shared sections: claims, identifiability, decoded result, flagged
    _print_common(report, decoded, graph)

    # Truth scoring (water CSV only)
    if truth_path and Path(truth_path).exists():
        _print_truth_scoring(decoded, truth_path)

    print(f"\n  graph.json → {graph_out}")
    print()

    return {
        "graph":         graph,
        "compiled":      compiled,
        "decoded":       decoded,
        "report":        report,
        "ingest_report": ingest_report,
    }


# ---------------------------------------------------------------------------
# Formatted output printers
# ---------------------------------------------------------------------------

def _print_merge_header(ingest_report, graph):
    n_files = ingest_report.get("n_files", 1)
    total = ingest_report.get("total_claims", len(graph.get("claims", [])))
    multi = ingest_report.get("multi_source_claims", 0)
    span = ingest_report.get("timestamp_span_hours", 0)
    window = ingest_report.get("window_hours", 24)
    canon_map = ingest_report.get("canon_map", {})
    initials = ingest_report.get("opening_initials", {})

    print(f"\n  Sinks mode: {ingest_report.get('sinks_mode', 'none')}"
          + (f"  (sink variables: {', '.join(ingest_report['sink_variable_nodes'])})"
             if ingest_report.get('sink_variable_nodes') else ""))
    print(f"  Files merged: {n_files}")
    print(f"  Total claims: {total}  (multi-source: {multi})")
    print(f"  Timestamp span: {span:.1f}h  (window: {window}h)")
    if canon_map:
        print(f"  Canonicalized: {len(canon_map)} aliases")
        for raw, canon in sorted(canon_map.items()):
            print(f"    {raw} -> {canon}")
    if initials:
        print(f"  Opening initials:")
        for nid, val in sorted(initials.items()):
            print(f"    {nid}: {val}")


def _print_common(report, decoded, graph):
    """Sections printed for every file type."""
    claims = graph.get("claims", [])

    # Claims
    print(f"\n  Claims ({report['n_claims']}):")
    for c in claims:
        ref = c.get("ref") or c.get("refs", "?")
        print(f"    {c['id']:<10} {c['type']:<6}  ref={ref!s:<20}  "
              f"value={c['value']:<10}  w={c.get('weight',1.0):.2f}")

    # Identifiability
    print(f"\n  Identifiability:")
    print(f"    rank:           {report['rank']} / {report['n_vars']}")
    print(f"    identifiable:   {report['identifiable']}")
    print(f"    correctable_k:  {report['correctable_k']}")

    # Decoded nodes
    print(f"\n  Decoded nodes:")
    for nd in decoded["nodes"]:
        print(f"    {nd['id']:<20}  qty = {nd['qty']:.2f}")

    # Decoded edges
    if decoded["edges"]:
        print(f"\n  Decoded edges:")
        for e in decoded["edges"]:
            print(f"    {e['id']:<10}  {e['from']:<12} → {e['to']:<12}  "
                  f"flow = {e['flow']:.2f}")

    # Decoded sinks
    if decoded["sinks"]:
        sinks = sorted(decoded["sinks"], key=lambda s: abs(s["sink"]), reverse=True)
        print(f"\n  Sinks (ranked by magnitude):")
        for i, s in enumerate(sinks):
            marker = " ◄" if i == 0 and abs(s["sink"]) > 1.0 else ""
            print(f"    #{i+1}  {s['id']:<16}  sink = {s['sink']:.2f}{marker}")

    # Flagged claims
    if decoded["flagged"]:
        print(f"\n  Flagged claims ({len(decoded['flagged'])}):")
        for f in decoded["flagged"]:
            claim = next((c for c in claims if c["id"] == f["claim_id"]), {})
            source = claim.get("source", "?")
            print(f"    [{f['claim_id']}]  |r|={abs(f['residual']):.2f}  "
                  f"source={source}  type={f['type']}")
    else:
        print("\n  No claims flagged.")


def _print_truth_scoring(decoded, truth_path):
    with open(truth_path) as fh:
        truth = json.load(fh)
    leak_nid = truth["leak_node"]
    leak_size = truth["leak_size"]

    sinks = sorted(decoded["sinks"], key=lambda s: abs(s["sink"]), reverse=True)
    leak_rank = None
    for i, s in enumerate(sinks):
        if s["id"] == leak_nid:
            leak_rank = i + 1
            break

    top_mag = abs(sinks[0]["sink"]) if sinks else 0
    second_mag = abs(sinks[1]["sink"]) if len(sinks) > 1 else 0
    margin = top_mag - second_mag

    print(f"\n  Truth scoring:")
    print(f"    true leak node: {leak_nid}")
    print(f"    true leak size: {leak_size}")
    print(f"    recovered rank: #{leak_rank}")
    print(f"    margin over #2: {margin:.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m orb.demo <file> [truth.json]", file=sys.stderr)
        sys.exit(1)
    truth = sys.argv[2] if len(sys.argv) > 2 else None
    run_demo(sys.argv[1], truth_path=truth)
