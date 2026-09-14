#!/usr/bin/env python3
"""
scripts/verify_fixtures.py
--------------------------
Run every fixture set through the full pipeline and print, per set:
  - the merged claim list (ref, value, channel, timestamp, source file)
  - flagged claims with residuals
  - recovered node quantities against the documented truth

Usage:
    python3 scripts/verify_fixtures.py            # all sets
    python3 scripts/verify_fixtures.py medical    # one set (supply|fuel|water|medical)

Requires ANTHROPIC_API_KEY (read from .env) for column mapping and text
extraction; both are disk-cached after the first run.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orb.demo import run_demo, run_demo_multi          # noqa: E402
from orb.water_world import WATER_COLUMN_MAPPING       # noqa: E402

FIX = ROOT / "tests" / "fixtures"

FUEL_E_MAPPING = {
    "entity_column": "sensor_id", "value_column": "value", "time_column": "timestamp",
    "channel_column": "channel",
    "channel_map": {"opening": "node", "flow": "edge", "eod": "node", "consumption": "sink"},
    "from_column": "from_node", "to_column": "to_node", "id_pattern": None,
    "excluded_channels": [], "notes": "fuel with known burn",
}

# Documented truths (README_FUEL.txt, supply README_FIXTURES.txt, medical derived
# from openings + convoy manifests).
TRUTH = {
    "fuel_C_clean":            {"PORT_HAVEN": 11000, "FOB_IRONSIDE": 7100, "FOB_KESTREL": 5200, "OP_TALON": 1900, "OP_VIPER": 1050},
    "fuel_B_two_corruptions":  {"PORT_HAVEN": 11000, "FOB_IRONSIDE": 7100, "FOB_KESTREL": 5200, "OP_TALON": 1900, "OP_VIPER": 1050},
    "fuel_D_correlated":       {"PORT_HAVEN": 11000, "FOB_IRONSIDE": 7100, "FOB_KESTREL": 5200, "OP_TALON": 1900, "OP_VIPER": 1050},
    "fuel_E_consumption":      {"PORT_HAVEN": 11000, "FOB_IRONSIDE": 6300, "FOB_KESTREL": 4600, "OP_TALON": 1750, "OP_VIPER": 930},
    # README_FUEL: four bad claims on a k=1 graph. The documented outcome IS the
    # failure: KESTREL -350, VIPER +350, four flags incl. one honest receiver row.
    "fuel_F_overcorrupted":    {"PORT_HAVEN": 11000, "FOB_IRONSIDE": 7100, "FOB_KESTREL": 4850, "OP_TALON": 1900, "OP_VIPER": 1400},
    "supply":                  {"MAIN_DEPOT": 2300, "FOB_ALPHA": 1300, "FOB_BRAVO": 900, "OP_CRESCENT": 340, "OP_DELTA": 270, "OP_ECHO": 225},
    "medical":                 {"REGIONAL_HUB": 3400, "FWD_HOSP_NORTH": 1810, "FWD_HOSP_SOUTH": 1150,
                                "AID_STN_1": 440, "AID_STN_2": 370, "AID_STN_3": 415, "AID_STN_4": 330},
}

EXPECT_FLAGS = {
    "fuel_C_clean": 0, "fuel_B_two_corruptions": 2, "fuel_D_correlated": 3,
    "fuel_E_consumption": 1, "fuel_F_overcorrupted": 4,
    "supply": 1, "medical": 2,
}


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **kw)


def _short(p) -> str:
    return Path(str(p)).name if p else "?"


def _ts(c) -> str:
    t = c.get("timestamp", "")
    if isinstance(t, list):
        return f"{t[0][11:16]}..{t[-1][11:16]} ({len(t)})" if all(len(x) >= 16 for x in t) else f"[{len(t)} ts]"
    return t[11:16] if isinstance(t, str) and len(t) >= 16 else (t or "-")


def print_result(title: str, result: dict, truth: dict | None, expect_flags) -> bool:
    graph, decoded, report = result["graph"], result["decoded"], result["report"]
    print(f"\n{'=' * 96}\n  {title}\n{'=' * 96}")
    print(f"  identifiability: rank {report['rank']}/{report['n_vars']}  "
          f"correctable_k={report['correctable_k']}  claims={report['n_claims']}")
    conflicts = result["ingest_report"].get("opening_conflicts") or {}
    if conflicts:
        print(f"  OPENING CONFLICTS: {conflicts}")

    edge_name = {e["id"]: f"{e['from']}->{e['to']}" for e in graph["edges"]}
    flagged = {f["claim_id"]: f["residual"] for f in decoded["flagged"]}

    print(f"\n  {'id':<5} {'type':<5} {'ref':<32} {'value':>9}  {'channel':<18} {'time':<16} {'file':<28} {'w':>5}  resid")
    for c in graph["claims"]:
        ref = edge_name.get(c.get("ref"), c.get("ref") or ",".join(c.get("refs", [])))
        flag = f"  <-- FLAGGED {flagged[c['id']]:+.1f}" if c["id"] in flagged else ""
        print(f"  {c['id']:<5} {c['type']:<5} {ref:<32} {c['value']:>9.1f}  {c.get('channel', ''):<18} "
              f"{_ts(c):<16} {_short(c.get('source')):<28} {c.get('weight', 1):>5.2f}{flag}")

    ok = True
    if truth:
        print(f"\n  {'node':<18} {'estimated':>10} {'truth':>8} {'err':>8}")
        est = {n["id"]: n["qty"] for n in decoded["nodes"]}
        for nid, tv in truth.items():
            ev = est.get(nid)
            err = None if ev is None else ev - tv
            mark = "" if err is not None and abs(err) < 1.0 else "   <-- MISMATCH"
            if mark:
                ok = False
            print(f"  {nid:<18} {ev if ev is not None else float('nan'):>10.1f} {tv:>8} "
                  f"{(err if err is not None else float('nan')):>8.1f}{mark}")
    if isinstance(expect_flags, int):
        n = len(decoded["flagged"])
        print(f"\n  flagged: {n}  (expected {expect_flags}){'' if n == expect_flags else '   <-- MISMATCH'}")
        ok = ok and n == expect_flags
    else:
        print(f"\n  flagged: {len(decoded['flagged'])}  (expected: {expect_flags})")
    return ok


def run_supply() -> dict[str, bool]:
    d = FIX / "supply"
    out = {}
    for f in ["supply_logistics.csv", "supply_reports.jsonl", "supply_field_reports.txt"]:
        r = _quiet(run_demo, d / f)
        out[f] = print_result(f"SUPPLY — {f}", r, TRUTH["supply"], EXPECT_FLAGS["supply"])
    r = _quiet(run_demo_multi, [d / "supply_logistics.csv", d / "supply_reports.jsonl", d / "supply_field_reports.txt"])
    out["supply (3 files merged)"] = print_result("SUPPLY — all three files merged", r, TRUTH["supply"], EXPECT_FLAGS["supply"] * 3)
    return out


def run_fuel() -> dict[str, bool]:
    d = FIX / "fuel"
    out = {}
    for stem in ["fuel_C_clean", "fuel_B_two_corruptions", "fuel_D_correlated", "fuel_E_consumption", "fuel_F_overcorrupted"]:
        kw = {"column_mapping": FUEL_E_MAPPING} if stem == "fuel_E_consumption" else {}
        r = _quiet(run_demo, d / f"{stem}.csv", **kw)
        label = f"FUEL — {stem}" + ("  (documented failure past k=1: expect KESTREL -350, VIPER +350)" if stem == "fuel_F_overcorrupted" else "")
        out[stem] = print_result(label, r, TRUTH[stem], EXPECT_FLAGS[stem])
    return out


def run_water() -> dict[str, bool]:
    r = _quiet(run_demo, FIX / "demo_water.csv", truth_path=FIX / "demo_water_truth.json",
               sinks="unknown", column_mapping=WATER_COLUMN_MAPPING)
    ok = print_result("WATER — demo_water.csv (leak search)", r, None, "leak at J_03 ranks #1")
    sinks = sorted(r["decoded"]["sinks"], key=lambda s: abs(s["sink"]), reverse=True)
    print("  sinks ranked:", ", ".join(f"{s['id']}={s['sink']:.1f}" for s in sinks[:4]))
    ok = sinks[0]["id"] == "J_03"
    print(f"  leak node J_03 at rank #1: {ok}")
    return {"demo_water": ok}


WATER_A_TRUE = {"RESERVOIR": 29500, "PUMP_A": 7100, "PUMP_B": 2400, "TOWER_N": 1100,
                "ZONE_1": 900, "ZONE_2": 1280, "ZONE_3": 480, "ZONE_4": 1010, "ZONE_5": 1090}


def run_water_a() -> dict[str, bool]:
    f = FIX / "water" / "water_A_leak_only.csv"
    out = {}
    r = _quiet(run_demo, f, sinks="none")
    truth_none = dict(WATER_A_TRUE, ZONE_3=WATER_A_TRUE["ZONE_3"] + 480)   # strict conservation puts the leak on the node
    out["water_A sinks=none"] = print_result("WATER_A — sinks=none (strict; expect ZONE_3 flagged -480)", r, truth_none, 1)
    r = _quiet(run_demo, f, sinks="unknown")
    ok = print_result("WATER_A — sinks=unknown (leak search; expect exact, sink ZONE_3=480, no flags)", r, WATER_A_TRUE, 0)
    sinks = {s["id"]: s["sink"] for s in r["decoded"]["sinks"]}
    print("  sinks:", ", ".join(f"{k}={v:.1f}" for k, v in sorted(sinks.items(), key=lambda kv: -abs(kv[1]))))
    ok = ok and abs(sinks.get("ZONE_3", 0) - 480) < 1.0
    out["water_A sinks=unknown"] = ok
    r = _quiet(run_demo, f, sinks="known")
    out["water_A sinks=known"] = print_result("WATER_A — sinks=known (metered draws become sink claims)", r, None, "informational")
    return out


def run_medical() -> dict[str, bool]:
    d = FIX / "medical"
    files = [d / "med_opening_manifest.csv", d / "med_convoy_log.jsonl", d / "med_radio_traffic.txt", d / "med_eod_counts.csv"]
    r = _quiet(run_demo_multi, files)
    ok = print_result("MEDICAL — four files merged", r, TRUTH["medical"], EXPECT_FLAGS["medical"])
    return {"medical": ok}


if __name__ == "__main__":
    which = sys.argv[1:] or ["supply", "fuel", "water", "water_a", "medical"]
    results: dict[str, bool] = {}
    for w in which:
        results.update({"supply": run_supply, "fuel": run_fuel, "water": run_water,
                        "water_a": run_water_a, "medical": run_medical}[w]())
    print(f"\n{'=' * 96}\n  SUMMARY\n{'=' * 96}")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL':<5} {k}")
    sys.exit(0 if all(results.values()) else 1)
