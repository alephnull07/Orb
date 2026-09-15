"""
orb/presets.py
--------------
Demo scenarios: fixture files + a losses mode + a caption.  Single source of
truth for the UI ("Try a demo"), the JSON entry point (--preset) and the tests.

Every run goes through the FULL pipeline — ingest, merge, compile, solve,
decode.  The only caching anywhere is the disk cache of LLM extraction
keyed by prompt hash (column mapping, per-report text extraction); a
finished run is never snapshotted.

    python3 -m orb.presets --json     # list presets as JSON
    python3 -m orb.presets --table    # run all presets, print a summary table
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

# kind → what the scenario demonstrates (drives the pill colour in the UI):
#   clean       nothing wrong; the control
#   corruption  independent false reports, caught and corrected
#   correlated  several false reports from one source
#   overload    more corruption than the guarantee covers; the system says so
#   leak        physical loss found via sink variables
#   merge       evidence that only exists across files / formats
#   mode        same data, different losses mode, different verdict
PRESET_KINDS = ("clean", "corruption", "correlated", "overload", "leak", "merge", "mode")

# Order matters for the demo panel: the most unstructured inputs (free-text
# radio and field reports, JSON lines) come first.
PRESETS: list[dict] = [
    {
        "id": "supply-three-formats",
        "tag": "multi-format merge",
        "kind": "merge",
        "name": "Supply drop — three formats",
        "files": ["supply/supply_field_reports.txt", "supply/supply_logistics.csv", "supply/supply_reports.jsonl"],
        "sinks": "none",
        "caption": "Three file formats describing one network. One false end-of-day count.",
    },
    {
        "id": "supply-five-channels",
        "tag": "five channels, one day",
        "kind": "merge",
        "name": "Supply drop — five channels",
        "files": ["supply_large/radio_net.txt", "supply_large/email_thread.txt", "supply_large/sms_export.txt",
                  "supply_large/whatsapp_export.txt", "supply_large/field_logs.txt"],
        "sinks": "none",
        "caption": "Radio, email, SMS, WhatsApp and photographed notebooks from one day: 88 messages, 17 sites, six altered figures and four fabricated messages planted across the channels.",
        "warning": "warning — long run, 1-2 min",
    },
    {
        "id": "medical-four-sources",
        "tag": "cross-source detection",
        "kind": "merge",
        "name": "Medical supply — four sources",
        "files": ["medical/med_radio_traffic.txt", "medical/med_convoy_log.jsonl",
                  "medical/med_opening_manifest.csv", "medical/med_eod_counts.csv"],
        "sinks": "none",
        "caption": "Neither corruption is detectable from the file it lives in.",
    },
    {
        "id": "fuel-two-false-reports",
        "tag": "two independent corruptions",
        "kind": "corruption",
        "name": "Field logistics — two false reports",
        "files": ["fuel/fuel_B_two_corruptions.csv"],
        "sinks": "none",
        "caption": "A driver overstating a delivery and a site overstating its stock, caught separately.",
    },
    {
        "id": "fuel-one-bad-reporter",
        "tag": "correlated corruption",
        "kind": "correlated",
        "name": "One bad reporter",
        "files": ["fuel/fuel_D_correlated.csv"],
        "sinks": "none",
        "caption": "Three flags, all tracing to the same source. A broken process, not three mistakes.",
    },
    {
        "id": "fuel-consumption",
        "tag": "known consumption",
        "kind": "mode",
        "name": "Consumption tracking",
        "files": ["fuel/fuel_E_consumption.csv"],
        "sinks": "none",
        "caption": "Sites burning fuel daily. A stock report is off by 50 against burn rates of 600-800.",
    },
    {
        "id": "fuel-audit-meters",
        "tag": "metered sinks",
        "kind": "mode",
        "name": "Audit the meters",
        "files": ["fuel/fuel_E_consumption.csv"],
        "sinks": "known",
        "caption": "Same file as above. Burn meters become claims, so the meter gets flagged instead of the stock count.",
    },
    {
        "id": "fuel-past-guarantee",
        "tag": "too much corruption — system says so",
        "kind": "overload",
        "name": "Past the guarantee",
        "files": ["fuel/fuel_F_overcorrupted.csv"],
        "sinks": "none",
        "caption": "Four bad claims on a network that can provably survive one. It fails, and we predicted how.",
    },
    {
        "id": "water-find-the-loss",
        "tag": "leak search, no false reports",
        "kind": "leak",
        "name": "Water network — find the loss",
        "files": ["water/water_A_leak_only.csv"],
        "sinks": "unknown",
        "caption": "No false reports at all, but 480 m3 is missing. Where?",
    },
    {
        "id": "water-lie-or-leak",
        "tag": "lie vs leak",
        "kind": "leak",
        "name": "Lie or leak?",
        "files": ["water/water_B_leak_and_lie.csv"],
        "sinks": "unknown",
        "caption": "One real loss and one false report, separated in a single run.",
    },
    {
        "id": "large-network",
        "tag": "no corruption — clean baseline",
        "kind": "clean",
        "name": "Large network",
        "files": ["large/fifteen_node.csv"],
        "sinks": "none",
        "caption": "15 sites, 14 routes, clean data.",
    },
]

_BY_ID = {p["id"]: p for p in PRESETS}


def get_preset(preset_id: str) -> dict:
    if preset_id not in _BY_ID:
        raise KeyError(f"Unknown preset {preset_id!r}. Known: {sorted(_BY_ID)}")
    return _BY_ID[preset_id]


def preset_paths(preset: dict) -> list[str]:
    """
    Fixture paths for a preset, relative to the current directory when it is
    inside the repo (so claim sources read 'tests/fixtures/...'), absolute
    otherwise.  Never a temp copy.
    """
    out = []
    for f in preset["files"]:
        p = FIXTURES / f
        if not p.exists():
            raise FileNotFoundError(p)
        try:
            rel = os.path.relpath(p, os.getcwd())
            out.append(rel if not rel.startswith("..") else str(p))
        except ValueError:          # different drive on Windows
            out.append(str(p))
    return out


def run_preset(preset_id: str) -> dict:
    """
    Run the full pipeline for a preset.  The preset's own losses mode is
    used — nothing the caller has selected elsewhere can override it.
    Returns the run_demo result dict with a "preset" entry attached.
    """
    from .demo import run_demo, run_demo_multi

    preset = get_preset(preset_id)
    paths = preset_paths(preset)
    if len(paths) == 1:
        result = run_demo(paths[0], sinks=preset["sinks"])
    else:
        result = run_demo_multi(paths, sinks=preset["sinks"])
    if not result:
        raise RuntimeError(f"Preset {preset_id!r} produced no graph")
    result["preset"] = {
        "id": preset["id"],
        "name": preset["name"],
        "tag": preset["tag"],
        "kind": preset["kind"],
        "caption": preset["caption"],
        "warning": preset.get("warning"),
        "sinks": preset["sinks"],
        "files": [Path(f).name for f in preset["files"]],
    }
    return result


def summary_row(result: dict) -> dict:
    g, d, r = result["graph"], result["decoded"], result["report"]
    return {
        "preset": result["preset"]["name"],
        "sinks": result["preset"]["sinks"],
        "nodes": len(g["nodes"]),
        "edges": len(g["edges"]),
        "claims": len(g["claims"]),
        "correctable_k": r["correctable_k"],
        "flagged": len(d["flagged"]),
    }


def format_table(rows: list[dict]) -> str:
    cols = ["preset", "sinks", "nodes", "edges", "claims", "correctable_k", "flagged"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) if c == "preset" else c.rjust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = [
        "  ".join(str(r[c]).ljust(widths[c]) if c == "preset" else str(r[c]).rjust(widths[c]) for c in cols)
        for r in rows
    ]
    return "\n".join([line, sep, *body])


if __name__ == "__main__":
    import argparse
    import contextlib
    import io

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print the preset list as JSON")
    ap.add_argument("--table", action="store_true", help="run every preset and print a summary table")
    args = ap.parse_args()

    if args.json:
        json.dump([{k: p[k] for k in ("id", "name", "sinks", "caption", "tag", "kind")}
                   | {"warning": p.get("warning"), "files": [Path(f).name for f in p["files"]], "paths": list(p["files"])}
                   for p in PRESETS], sys.stdout, ensure_ascii=False)
        print()
    elif args.table:
        rows = []
        for p in PRESETS:
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_preset(p["id"])
            rows.append(summary_row(res))
        print(format_table(rows))
    else:
        ap.print_help()
