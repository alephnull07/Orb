"""
orb/summarize.py
----------------
Print a markdown summary table from sweep results, averaged over seeds.

Usage:
    python -m orb.summarize [results/sweep.json]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def print_summary(sweep: dict) -> str:
    """Print and return a markdown table averaged over seeds."""
    rows = sweep["rows"]
    meta = sweep["meta"]

    # Group by (generator, k, estimator, corruption_type)
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r["generator"], r["k"], r["estimator"], r["corruption_type"])
        groups[key].append(r)

    # Metric columns to average
    metric_keys = [
        "false_certainty", "exact_recovery",
        "state_error_max", "state_error_mean",
        "detection_precision", "detection_recall",
        "compute_ms", "correctable_k",
    ]

    # Build summary rows
    summary_rows = []
    for key, group in sorted(groups.items()):
        gen, k, est, ct = key
        n = len(group)
        avgs = {}
        for mk in metric_keys:
            vals = [r[mk] for r in group]
            avgs[mk] = sum(vals) / len(vals)
        summary_rows.append({
            "generator": gen,
            "k": k,
            "estimator": est,
            "corruption_type": ct,
            "n_seeds": n,
            **{mk: round(avgs[mk], 4) for mk in metric_keys},
        })

    # Format markdown table
    lines = []
    lines.append(f"## ORB Sweep Results")
    lines.append(f"")
    lines.append(f"- **Git hash**: `{meta.get('git_hash', '?')}`")
    lines.append(f"- **Timestamp**: {meta.get('timestamp', '?')}")
    lines.append(f"- **Seeds**: {len(meta.get('seeds', []))}")
    lines.append(f"- **Total runs**: {len(rows)}")
    lines.append(f"")

    # Table header
    header = "| gen | k | estimator | corrupt | n | false_cert | exact_rec | err_max | err_mean | prec | recall | ms | corr_k |"
    sep    = "|-----|---|-----------|---------|---|------------|-----------|---------|----------|------|--------|-----|--------|"
    lines.append(header)
    lines.append(sep)

    for sr in summary_rows:
        line = (
            f"| {sr['generator']:<5} "
            f"| {sr['k']} "
            f"| {sr['estimator']:<13} "
            f"| {sr['corruption_type']:<9} "
            f"| {sr['n_seeds']:>1} "
            f"| {sr['false_certainty']:.2f}       "
            f"| {sr['exact_recovery']:.2f}      "
            f"| {sr['state_error_max']:>7.1f} "
            f"| {sr['state_error_mean']:>8.2f} "
            f"| {sr['detection_precision']:.2f} "
            f"| {sr['detection_recall']:.2f}   "
            f"| {sr['compute_ms']:>5.0f} "
            f"| {sr['correctable_k']:.0f}      |"
        )
        lines.append(line)

    md = "\n".join(lines)
    print(md)
    return md


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/sweep.json"
    with open(path) as fh:
        sweep = json.load(fh)
    print_summary(sweep)
