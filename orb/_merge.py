"""
orb/_merge.py
-------------
Post-extraction merge: canonicalize node names across files, turn opening
counts into initials, sum separate events on an edge within one observer
stream, and emit one graph in which EVERY independent observation is its
own claim row.

merge_graphs(per_file_results, window_hours, sinks) -> (graph_dict, report)

The rule that matters
---------------------
An observer stream is (file, channel) on one edge — e.g. the sender-side
rows of a CSV, or the receiver-side reports in a radio log.  Within a
stream, rows at DIFFERENT timestamps are separate physical events on the
same edge and are SUMMED, because the edge variable is the total flow over
the window.  Rows from different streams — sender vs receiver, file A vs
file B — are independent observations of that total and stay as separate
claims.  Nothing in this module averages, medians, or deduplicates
observations by value.  Conflicts are left for the L1 estimator, which is
the only component that can tell an honest row from a corrupted one.

The one exception is the opening count: it is a hard constant in the
balance equation, not a claim row, so conflicting openings are resolved by
median and reported loudly in merge_report["opening_conflicts"].
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
    def score(s: str) -> tuple[int, int, str]:
        sep_count = s.count("_") + s.count(" ") + s.count("-")
        return (sep_count, len(s), s)
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
    "%Y-%m-%d %H:%M",
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


def _ts_sort_key(s: str) -> tuple[int, float, str]:
    dt = _parse_ts(s)
    return (0, dt.timestamp(), s) if dt else (1, 0.0, s)


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
    # Phase 1: collect every observation row from every file
    # ------------------------------------------------------------------
    raw_node_ids: dict[str, set[str]] = defaultdict(set)     # canon -> {raw}
    opening_rows: dict[str, list[dict]] = defaultdict(list)   # canon -> rows
    edge_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    node_rows: list[dict] = []
    passthrough_rows: list[dict] = []                         # sink, aggregate
    all_timestamps: list[str] = []
    known_sinks_merged: dict[str, float] = defaultdict(float)
    file_order: dict[str, int] = {}

    def _file_idx(name: str) -> int:
        if name not in file_order:
            file_order[name] = len(file_order)
        return file_order[name]

    for file_idx, (graphs, report) in enumerate(per_file_results):
        file_name = report.get("source_file", report.get("source_files", [f"file_{file_idx}"]))
        if isinstance(file_name, list):
            file_name = file_name[0] if file_name else f"file_{file_idx}"
        _file_idx(file_name)

        # known_sinks once per file (every timestamp bucket repeats the node dicts)
        file_known_sinks: dict[str, float] = {}

        for g in graphs:
            bucket_ts = g.get("_timestamp", "all") or "all"
            if bucket_ts != "all":
                all_timestamps.append(bucket_ts)

            edge_endpoint_map: dict[str, tuple[str, str]] = {}
            for e in g.get("edges", []):
                cf, ct = canon_id(e["from"]), canon_id(e["to"])
                edge_endpoint_map[e["id"]] = (cf, ct)
                raw_node_ids[cf].add(e["from"])
                raw_node_ids[ct].add(e["to"])

            for n in g.get("nodes", []):
                cn = canon_id(n["id"])
                raw_node_ids[cn].add(n["id"])
                if "known_sinks" in n and cn not in file_known_sinks:
                    file_known_sinks[cn] = n["known_sinks"]

            for order, c in enumerate(g.get("claims", [])):
                ctype   = c.get("type", "")
                channel = str(c.get("source", "") or "")
                ts      = str(c.get("timestamp") or bucket_ts or "all") or "all"
                if ts != "all" and ts != bucket_ts:
                    all_timestamps.append(ts)
                row = {
                    "value":     float(c["value"]),
                    "ts":        ts,
                    "file":      c.get("file") or file_name,
                    "channel":   channel,
                    "weight":    float(c.get("weight", 1.0)),
                    "record_id": c.get("record_id"),
                    "order":     order,
                }
                _file_idx(row["file"])

                if ctype == "node":
                    cn = canon_id(c["ref"])
                    row["canon_ref"] = cn
                    if "opening" in channel.lower():
                        opening_rows[cn].append(row)
                    else:
                        node_rows.append(row)
                elif ctype == "edge":
                    endpoints = edge_endpoint_map.get(c["ref"])
                    if endpoints:
                        edge_rows[endpoints].append(row)
                elif ctype in ("sink", "aggregate"):
                    row["type"] = ctype
                    row["canon_ref"] = canon_id(c["ref"]) if ctype == "sink" else None
                    row["ref"] = c.get("ref")
                    row["refs"] = c.get("refs")
                    passthrough_rows.append(row)

        for cn, val in file_known_sinks.items():
            known_sinks_merged[cn] += val

    # ------------------------------------------------------------------
    # Phase 2: timestamp span check
    # ------------------------------------------------------------------
    parsed_ts = [t for t in (_parse_ts(s) for s in all_timestamps) if t is not None]
    span = (max(parsed_ts) - min(parsed_ts)).total_seconds() / 3600.0 if len(parsed_ts) >= 2 else 0.0
    if span > window_hours:
        print(
            f"  [WARNING] Timestamp span {span:.1f}h exceeds window {window_hours}h. "
            f"Merging anyway; splitting not yet implemented.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Phase 3: canonical node set
    # ------------------------------------------------------------------
    canon_display: dict[str, str] = {cn: pick_display(sorted(raws)) for cn, raws in raw_node_ids.items()}

    # ------------------------------------------------------------------
    # Phase 4: opening counts -> initial (hard constant; conflicts reported)
    # ------------------------------------------------------------------
    initials: dict[str, float] = {}
    opening_conflicts: dict[str, list[dict]] = {}
    for cn, rows in opening_rows.items():
        vals = [r["value"] for r in rows]
        initials[cn] = float(median(vals))
        if len(set(vals)) > 1:
            disp = canon_display.get(cn, cn.upper())
            opening_conflicts[disp] = [
                {"value": r["value"], "file": r["file"], "timestamp": r["ts"]} for r in rows
            ]
            print(
                f"  [WARNING] Conflicting opening counts for {disp}: "
                + ", ".join(f"{r['value']:g} ({r['file']})" for r in rows)
                + f". Using median {initials[cn]:g}; the balance equation cannot flag this.",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Phase 5: edge rows -> one claim per observer stream (and lane)
    # ------------------------------------------------------------------
    # stream = (file, channel).  Different timestamps within a stream are
    # separate events on the edge and are summed.  Several rows at the SAME
    # timestamp in one stream are separate observations ("lanes"): lane i
    # at each timestamp sums with lane i at the others.  Streams never mix.
    merged_edges: list[dict] = []          # one entry per claim
    edge_keys = sorted(edge_rows.keys())
    for (cf, ct) in edge_keys:
        streams: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in edge_rows[(cf, ct)]:
            streams[(r["file"], r["channel"])].append(r)
        for (fname, channel) in sorted(streams, key=lambda k: (file_order[k[0]], k[1])):
            rows = streams[(fname, channel)]
            by_ts: dict[str, list[dict]] = defaultdict(list)
            for r in sorted(rows, key=lambda r: r["order"]):
                by_ts[r["ts"]].append(r)
            n_lanes = max(len(v) for v in by_ts.values())
            for lane in range(n_lanes):
                lane_rows = [
                    by_ts[ts][lane]
                    for ts in sorted(by_ts, key=_ts_sort_key)
                    if lane < len(by_ts[ts])
                ]
                merged_edges.append({
                    "from": cf, "to": ct,
                    "value": sum(r["value"] for r in lane_rows),
                    "file": fname,
                    "channel": channel,
                    "lane": lane,
                    "timestamps": [r["ts"] for r in lane_rows],
                    "record_ids": [r["record_id"] for r in lane_rows if r["record_id"]],
                    "weight": min(r["weight"] for r in lane_rows),
                    "n_events": len(lane_rows),
                })

    # ------------------------------------------------------------------
    # Phase 6: node rows -> one claim per row (levels are never summed)
    # ------------------------------------------------------------------
    node_rows.sort(key=lambda r: (r["canon_ref"], file_order[r["file"]],
                                  _ts_sort_key(r["ts"]), r["channel"], r["order"]))

    # ------------------------------------------------------------------
    # Phase 7: assemble unified graph
    # ------------------------------------------------------------------
    all_canons = set(canon_display.keys())
    for me in merged_edges:
        for cn in (me["from"], me["to"]):
            all_canons.add(cn)
            canon_display.setdefault(cn, cn.upper())
    for r in node_rows:
        all_canons.add(r["canon_ref"])
        canon_display.setdefault(r["canon_ref"], r["canon_ref"].upper())

    claimed_canons = {r["canon_ref"] for r in node_rows}

    graph_nodes = []
    for cn in sorted(all_canons):
        disp = canon_display[cn]
        node_sinks = sinks if (sinks != "unknown" or cn in claimed_canons) else "none"
        node: dict = {"id": disp, "initial": initials.get(cn, 0.0), "sinks": node_sinks}
        if cn in known_sinks_merged:
            node["known_sinks"] = known_sinks_merged[cn]
        graph_nodes.append(node)

    graph_edges = []
    edge_id_map: dict[tuple[str, str], str] = {}
    for (cf, ct) in edge_keys:
        eid = f"e{len(graph_edges)}"
        graph_edges.append({"id": eid, "from": canon_display[cf], "to": canon_display[ct]})
        edge_id_map[(cf, ct)] = eid

    claims: list[dict] = []
    files_per_ref: dict[str, set[str]] = defaultdict(set)

    def _ts_field(ts_list: list[str]):
        uniq = list(dict.fromkeys(ts_list))
        return uniq[0] if len(uniq) == 1 else uniq

    for me in merged_edges:
        eid = edge_id_map[(me["from"], me["to"])]
        claims.append({
            "id":        f"c{len(claims)}",
            "type":      "edge",
            "ref":       eid,
            "value":     me["value"],
            "source":    me["file"],
            "channel":   me["channel"],
            "timestamp": _ts_field(me["timestamps"]),
            "n_events":  me["n_events"],
            "record_id": (me["record_ids"][0] if len(me["record_ids"]) == 1
                          else (me["record_ids"] or None)),
            "weight":    me["weight"],
        })
        files_per_ref[eid].add(me["file"])

    for r in node_rows:
        disp = canon_display[r["canon_ref"]]
        claims.append({
            "id":        f"c{len(claims)}",
            "type":      "node",
            "ref":       disp,
            "value":     r["value"],
            "source":    r["file"],
            "channel":   r["channel"],
            "timestamp": r["ts"],
            "n_events":  1,
            "record_id": r["record_id"],
            "weight":    r["weight"],
        })
        files_per_ref[disp].add(r["file"])

    for r in passthrough_rows:
        claim: dict = {
            "id":        f"c{len(claims)}",
            "type":      r["type"],
            "value":     r["value"],
            "source":    r["file"],
            "channel":   r["channel"],
            "timestamp": r["ts"],
            "n_events":  1,
            "record_id": r["record_id"],
            "weight":    r["weight"],
        }
        if r["type"] == "sink" and r["canon_ref"]:
            claim["ref"] = canon_display.get(r["canon_ref"], r["ref"])
        elif r["type"] == "aggregate" and r.get("refs"):
            claim["refs"] = r["refs"]
        else:
            claim["ref"] = r.get("ref", "")
        claims.append(claim)

    # Drop None record_id keys to keep JSON tidy
    for c in claims:
        if c.get("record_id") is None:
            c.pop("record_id", None)

    graph = {"nodes": graph_nodes, "edges": graph_edges, "claims": claims}

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
        "multi_source_claims": sum(1 for fs in files_per_ref.values() if len(fs) > 1),
        "n_edge_streams": len(merged_edges),
        "timestamp_span_hours": round(span, 2),
        "window_hours": window_hours,
        "canon_map": canon_map,
        "opening_initials": {canon_display.get(cn, cn): v for cn, v in initials.items()},
        "opening_conflicts": opening_conflicts,
        "per_file": [r for _, r in per_file_results],
    }

    return graph, merge_report
