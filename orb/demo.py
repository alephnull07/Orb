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

from .advise    import generate_advice
from .compile   import compile as compile_graph
from .decode    import decode
from .ingest    import ingest_paths
from .run_graph import _l1_solve
from .temporal  import try_stack_sinks


def run_demo_multi(paths: list[str | Path]) -> dict:
    """Run the pipeline on any mix of files (tables, logs, jsonl)."""
    return run_demo(paths)


def run_demo(path: str | Path | list[str | Path], truth_path: str | Path | None = None) -> dict:
    """
    Run the full ORB pipeline on one file or an assortment of files.

    LLM is the primary reader (column mapping + record extraction).
    Regex only fills gaps when the model is missing or returns nothing.
    """
    paths = [Path(p) for p in (path if isinstance(path, list) else [path])]

    graph, ingest_report = ingest_paths(paths, lambda_w=0.1)

    if not graph or not graph.get("claims"):
        names = [p.name for p in paths]
        print(f"[demo] No graphs produced from {names}")
        skipped = ingest_report.get("skipped") or []
        if skipped:
            print("  skipped:")
            for s in skipped:
                print(f"    {s.get('file')}: {s.get('reason')}")
        return {
            "graph": None,
            "ingest_report": ingest_report,
        }

    # ── compile → L1 → decode (SAME path for both) ───────────────────────────
    compiled  = compile_graph(graph)
    x_hat, residuals = _l1_solve(compiled)
    x_hat = try_stack_sinks(graph, compiled, x_hat, lambda_sink=float(graph.get("lambda_sink", 0.01) or 0.01))
    decoded   = decode(x_hat, residuals, compiled, graph)
    report    = compiled["report"]
    advice    = generate_advice(graph, compiled, decoded)

    # ── write graph.json ────────────────────────────────────────────────────
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    stem = paths[0].stem if len(paths) == 1 else "_".join(p.stem for p in paths)[:80]
    graph_out = out_dir / f"graph_{stem}.json"
    with open(graph_out, "w") as fh:
        json.dump(graph, fh, indent=2, default=str)

    # ── print report ──────────────────────────────────────────────────────────
    mode = ingest_report.get("mode", "?")
    label = paths[0].name if len(paths) == 1 else [p.name for p in paths]
    print(f"\n{'='*60}")
    print(f"  run_demo({label})   mode={mode}")
    print(f"{'='*60}")

    # Mode-specific header
    if mode == "TABULAR":
        _print_tabular_header(ingest_report)
    elif mode == "RECORD":
        _print_record_header(ingest_report, graph)

    # Shared sections: claims, identifiability, decoded result, flagged
    _print_common(report, decoded, graph, advice)

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
        "advice":        advice,
        "ingest_report": ingest_report,
    }


# ---------------------------------------------------------------------------
# Formatted output printers
# ---------------------------------------------------------------------------

def _print_tabular_header(ingest_report):
    mapping = ingest_report.get("mapping") or {}
    if not mapping:
        return
    print(f"\n  Column mapping:")
    for k, v in mapping.items():
        print(f"    {k}: {v}")


def _print_record_header(ingest_report, graph):
    record_count = ingest_report.get("record_count", 0)
    claims = graph.get("claims", [])
    weights = [c.get("weight", 1.0) for c in claims]
    edges = graph.get("edges", [])
    print(f"\n  Record count: {record_count}")
    if weights:
        print(f"  Claim weights: min={min(weights):.4f}  max={max(weights):.4f}  "
              f"mean={sum(weights)/len(weights):.4f}")
    print(f"  Reader agreement: {len(edges)} edges extracted")


def _print_common(report, decoded, graph, advice=None):
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
    if decoded.get("sigma") is not None:
        print(f"    sigma (MAD):    {decoded['sigma']:.4f}   flag |r| > {decoded.get('threshold', 0):.2f}")
    if report.get("T"):
        print(f"    snapshots T:    {report['T']}")

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
            status = s.get("status") or ""
            print(f"    #{i+1}  {s['id']:<16}  sink = {s['sink']:.2f}  {status}{marker}")

    if decoded.get("loss"):
        print(f"\n  Named loss ({len(decoded['loss'])}):")
        for s in decoded["loss"]:
            print(f"    {s['id']:<16}  sink = {s['sink']:.2f}")

    if decoded.get("ambiguous"):
        print(f"\n  Ambiguous (cannot tell loss from downward corruption):")
        for a in decoded["ambiguous"]:
            print(f"    {a['id']:<16}  sink = {a['sink']:.2f}  {a.get('reason','')}")

    if decoded.get("undetectable"):
        print(f"\n  Coordinated-lie blind hops (no third channel):")
        for b in decoded["undetectable"]:
            print(f"    {b.get('from')} → {b.get('to')}  {b.get('reason')}")

    # Flagged claims
    if decoded["flagged"]:
        print(f"\n  Flagged claims / corruption ({len(decoded['flagged'])}):")
        for f in decoded["flagged"]:
            claim = next((c for c in claims if c["id"] == f["claim_id"]), {})
            source = claim.get("source", "?")
            print(f"    [{f['claim_id']}]  |r|={abs(f['residual']):.2f}  "
                  f"source={source}  type={f['type']}")
    else:
        print("\n  No claims flagged.")

    if advice:
        print(f"\n  Ops ({advice.get('severity')}): {advice.get('headline')}")
        for i, a in enumerate(advice.get("actions") or [], 1):
            print(f"    {i}. [{a.get('kind')}] {a.get('do')}")
        sensors = advice.get("sensors") or []
        if sensors:
            print("\n  Sensor placement:")
            for s in sensors:
                hops = s.get("covers_hops") or []
                hop_txt = ", ".join(f"{h.get('from')}→{h.get('to')}" for h in hops) or "—"
                print(f"    {s.get('at')}: {s.get('kind')}  covers {hop_txt}")
                print(f"      {s.get('effect')}")


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
        print("Usage: python -m orb.demo <file> [file ...] [truth.json]", file=sys.stderr)
        sys.exit(1)
    args = [Path(a) for a in sys.argv[1:]]
    truth = None
    if args[-1].suffix.lower() == ".json":
        try:
            obj = json.loads(args[-1].read_text(encoding="utf-8"))
            if isinstance(obj, dict) and "leak_node" in obj:
                truth = args.pop()
        except Exception:
            pass
    run_demo(args if len(args) > 1 else args[0], truth_path=truth)
