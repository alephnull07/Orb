"""
orb/_merge.py
-------------
Post-extraction merge: collapse timestamp buckets, canonicalize node names
across files, deduplicate claims, sum edge events, and emit one graph.

merge_graphs(per_file_results, window_hours, sinks) -> (graph_dict, report)
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def canon_id(name: str) -> str:
    """Lowercase, strip all non-alphanumeric. 'MAIN_DEPOT' -> 'maindepot'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def pick_display(raw_ids: list[str]) -> str:
    """Pick the most readable form: prefer underscores/separators, longest."""
    if not raw_ids:
        return "UNKNOWN"
    # Score: length + bonus for separators (underscores, spaces)
    def score(s: str) -> tuple[int, int]:
        sep_count = s.count("_") + s.count(" ") + s.count("-")
        return (sep_count, len(s))
    return max(raw_ids, key=score)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

_TS_FMTS = [
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%MZ",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]


def _parse_ts(s: str) -> datetime | None:
    if not s or s == "all":
        return None
    for fmt in _TS_FMTS:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------

def merge_graphs(
    per_file_results: list[tuple[list[dict], dict]],
    window_hours: float = 24.0,
    sinks: str = "none",
) -> tuple[dict, dict]:
    """
    Merge per-file (graphs, report) into one unified graph.

    Returns (graph_dict, merge_report).
    """

    # ------------------------------------------------------------------
    # Phase 1: collect raw data from all files/graphs
    # ------------------------------------------------------------------
    raw_node_ids: dict[str, set[str]] = defaultdict(set)  # canon -> {raw, ...}
    opening_values: dict[str, list[float]] = defaultdict(list)  # canon -> [values]
    edge_events: dict[tuple[str, str], list[dict]] = defaultdict(list)
    eod_claims: list[dict] = []
    passthrough_claims: list[dict] = []  # sink, aggregate — kept as-is
    all_timestamps: list[str] = []
    known_sinks_merged: dict[str, float] = defaultdict(float)  # canon -> total

    for file_idx, (graphs, report) in enumerate(per_file_results):
        file_name = report.get("source_file", report.get("source_files", [f"file_{file_idx}"]))
        if isinstance(file_name, list):
            file_name = file_name[0] if file_name else f"file_{file_idx}"

        # Collect known_sinks per file (not per graph — every timestamp-
        # bucket graph repeats the same node dicts, so only count once).
        file_known_sinks: dict[str, float] = {}

        for g in graphs:
            ts = g.get("_timestamp", "all")
            if ts and ts != "all":
                all_timestamps.append(ts)

            # Build edge_id -> (canon_from, canon_to) map for this graph
            edge_endpoint_map: dict[str, tuple[str, str]] = {}
            for e in g.get("edges", []):
                cf = canon_id(e["from"])
                ct = canon_id(e["to"])
                edge_endpoint_map[e["id"]] = (cf, ct)
                raw_node_ids[cf].add(e["from"])
                raw_node_ids[ct].add(e["to"])

            # Collect raw node IDs and known sinks
            for n in g.get("nodes", []):
                cn = canon_id(n["id"])
                raw_node_ids[cn].add(n["id"])
                if "known_sinks" in n and cn not in file_known_sinks:
                    file_known_sinks[cn] = n["known_sinks"]

            # Classify claims
            for c in g.get("claims", []):
                ctype = c.get("type", "")
                source = c.get("source", "")

                if ctype == "node":
                    cn = canon_id(c["ref"])
                    is_opening = "opening" in source.lower()
                    if is_opening:
                        opening_values[cn].append(c["value"])
                    else:
                        eod_claims.append({
                            "canon_ref": cn,
                            "value": c["value"],
                            "source": source,
                            "file": file_name,
                            "weight": c.get("weight", 1.0),
                        })
                elif ctype == "edge":
                    endpoints = edge_endpoint_map.get(c["ref"])
                    if endpoints:
                        edge_events[endpoints].append({
                            "value": c["value"],
                            "ts": ts,
                            "file": file_name,
                            "source": source,
                            "weight": c.get("weight", 1.0),
                        })
                elif ctype in ("sink", "aggregate"):
                    # Pass through with canonicalized ref
                    cn = canon_id(c["ref"]) if ctype == "sink" else None
                    passthrough_claims.append({
                        "type": ctype,
                        "canon_ref": cn,
                        "ref": c.get("ref"),
                        "refs": c.get("refs"),
                        "value": c["value"],
                        "source": source,
                        "file": file_name,
                        "weight": c.get("weight", 1.0),
                    })

        # Merge per-file known sinks into the global total
        for cn, val in file_known_sinks.items():
            known_sinks_merged[cn] += val

    # ------------------------------------------------------------------
    # Phase 2: timestamp span check
    # ------------------------------------------------------------------
    parsed_ts = [_parse_ts(t) for t in all_timestamps]
    parsed_ts = [t for t in parsed_ts if t is not None]
    if len(parsed_ts) >= 2:
        span = (max(parsed_ts) - min(parsed_ts)).total_seconds() / 3600.0
    else:
        span = 0.0

    if span > window_hours:
        print(
            f"  [WARNING] Timestamp span {span:.1f}h exceeds window {window_hours}h. "
            f"Merging anyway; splitting not yet implemented.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Phase 3: canonical node set
    # ------------------------------------------------------------------
    canon_display: dict[str, str] = {}  # canon -> display form
    for cn, raws in raw_node_ids.items():
        canon_display[cn] = pick_display(list(raws))

    # ------------------------------------------------------------------
    # Phase 4: opening claims -> initial
    # ------------------------------------------------------------------
    initials: dict[str, float] = {}
    for cn, vals in opening_values.items():
        initials[cn] = median(vals)

    # ------------------------------------------------------------------
    # Phase 5: edge claims -> one claim per unique edge
    # ------------------------------------------------------------------
    merged_edges: list[dict] = []
    for (cf, ct), events in sorted(edge_events.items()):
        if not events:
            continue

        # Group by (file, timestamp) -> one physical event
        per_file_ts: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for ev in events:
            per_file_ts[ev["file"]][ev["ts"]].append(ev["value"])

        # Per file: mean per event (dedup sent/received), sum across events
        file_totals: dict[str, float] = {}
        for f, ts_groups in per_file_ts.items():
            event_sum = 0.0
            for ts_key, vals in ts_groups.items():
                event_sum += sum(vals) / len(vals)  # mean of sent/received
            file_totals[f] = event_sum

        total = median(list(file_totals.values()))
        merged_edges.append({
            "from": cf,
            "to": ct,
            "value": total,
            "files": list(file_totals.keys()),
            "per_file": dict(file_totals),
        })

    # ------------------------------------------------------------------
    # Phase 6: EOD node claims -> dedup ACROSS files only
    # ------------------------------------------------------------------
    # Group by (canon_ref, rounded value). Within a group, claims from
    # the SAME file are distinct observations (different sensors) and
    # must be kept. Claims from DIFFERENT files with the same value
    # are the same event and merge into one claim.
    eod_groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for ec in eod_claims:
        key = (ec["canon_ref"], round(ec["value"]))
        eod_groups[key].append(ec)

    merged_eod: list[dict] = []
    for (cn, _rval), group in sorted(eod_groups.items()):
        # Keep one claim per unique (value, file) pair — different
        # readings from the same file are separate observations.
        seen: set[tuple[float, str]] = set()
        for g in group:
            pair = (g["value"], g["file"])
            if pair in seen:
                continue
            seen.add(pair)
            # Collect all files that reported this exact value
            same_val_files = list({
                g2["file"] for g2 in group if g2["value"] == g["value"]
            })
            merged_eod.append({
                "canon_ref": cn,
                "value": g["value"],
                "files": same_val_files,
                "weight": max(
                    g2["weight"] for g2 in group if g2["value"] == g["value"]
                ),
            })

    # ------------------------------------------------------------------
    # Phase 7: assemble unified graph
    # ------------------------------------------------------------------
    # Ensure all edge endpoints are in node set
    all_canons = set(canon_display.keys())
    for me in merged_edges:
        all_canons.add(me["from"])
        all_canons.add(me["to"])
        # Ensure display names exist
        if me["from"] not in canon_display:
            canon_display[me["from"]] = me["from"].upper()
        if me["to"] not in canon_display:
            canon_display[me["to"]] = me["to"].upper()

    # Only set sinks="unknown" on nodes that are direct claim targets,
    # not on edge-endpoint-only nodes (e.g. reservoir).
    claimed_canons = {ne["canon_ref"] for ne in merged_eod}

    graph_nodes = []
    for cn in sorted(all_canons):
        disp = canon_display.get(cn, cn.upper())
        node_sinks = sinks if (sinks != "unknown" or cn in claimed_canons) else "none"
        node: dict = {
            "id": disp,
            "initial": initials.get(cn, 0.0),
            "sinks": node_sinks,
        }
        if cn in known_sinks_merged:
            node["known_sinks"] = known_sinks_merged[cn]
        graph_nodes.append(node)

    graph_edges = []
    edge_id_map: dict[tuple[str, str], str] = {}
    for ei, me in enumerate(merged_edges):
        eid = f"e{ei}"
        graph_edges.append({
            "id": eid,
            "from": canon_display[me["from"]],
            "to": canon_display[me["to"]],
        })
        edge_id_map[(me["from"], me["to"])] = eid

    claims = []
    ci = 0
    multi_source = 0

    for me in merged_edges:
        eid = edge_id_map[(me["from"], me["to"])]
        n_files = len(me["files"])
        claims.append({
            "id": f"c{ci}",
            "type": "edge",
            "ref": eid,
            "value": me["value"],
            "source": f"merged({n_files})" if n_files > 1 else me["files"][0],
            "weight": 1.0,
        })
        if n_files > 1:
            multi_source += 1
        ci += 1

    for ne in merged_eod:
        disp = canon_display.get(ne["canon_ref"], ne["canon_ref"].upper())
        n_files = len(ne["files"])
        claims.append({
            "id": f"c{ci}",
            "type": "node",
            "ref": disp,
            "value": ne["value"],
            "source": f"merged({n_files})" if n_files > 1 else ne["files"][0],
            "weight": ne["weight"],
        })
        if n_files > 1:
            multi_source += 1
        ci += 1

    # Sink and aggregate claims: pass through with canonicalized refs
    for pc in passthrough_claims:
        claim: dict = {
            "id": f"c{ci}",
            "type": pc["type"],
            "value": pc["value"],
            "source": pc["source"],
            "weight": pc["weight"],
        }
        if pc["type"] == "sink" and pc["canon_ref"]:
            claim["ref"] = canon_display.get(pc["canon_ref"], pc["ref"])
        elif pc["type"] == "aggregate" and pc.get("refs"):
            claim["refs"] = pc["refs"]
        else:
            claim["ref"] = pc.get("ref", "")
        claims.append(claim)
        ci += 1

    graph = {
        "nodes": graph_nodes,
        "edges": graph_edges,
        "claims": claims,
    }

    # ------------------------------------------------------------------
    # Phase 8: merge report
    # ------------------------------------------------------------------
    canon_map = {}
    for cn, raws in raw_node_ids.items():
        disp = canon_display[cn]
        for r in raws:
            if r != disp:
                canon_map[r] = disp

    merge_report = {
        "n_files": len(per_file_results),
        "total_claims": len(claims),
        "multi_source_claims": multi_source,
        "timestamp_span_hours": round(span, 2),
        "window_hours": window_hours,
        "canon_map": canon_map,
        "opening_initials": {canon_display.get(cn, cn): v for cn, v in initials.items()},
        "per_file": [r for _, r in per_file_results],
    }

    return graph, merge_report
