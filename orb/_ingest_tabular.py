"""
orb/_ingest_tabular.py
----------------------
TABULAR mode:  CSV / TSV / XLSX → claims via ONE LLM column-mapping call.

Steps
-----
1. Sample header + 20 rows spread across the file.
2. ONE agent call → column mapping dict (cached to disk).
3. Apply the mapping in pure Python → claims.  Zero LLM calls per row.
4. Print the mapping for human inspection.
5. If value_column contains prose, fall back to RECORD mode (note in report).

Column mapping schema (returned by LLM):
  {
    "entity_column":      "sensor",
    "value_column":       "value",
    "time_column":        "timestamp",        // or null
    "channel_column":     "channel",          // or null
    "channel_map":        {"demand": "node", "flow": "edge"},
    "from_column":        null,               // for edge rows with explicit src
    "to_column":          null,               // for edge rows with explicit tgt
    "id_pattern":         null,               // regex to clean entity_column values
    "excluded_channels":  ["leak_demand"],    // leakage + ground-truth channels
    "notes":              "..."
  }
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

from ._ingest_utils import leakage_guard, validate_claims

_MAPPING_PROMPT = """\
You are analyzing a tabular dataset to map its columns to a conserved-flow network \
(supply chain, water distribution, power grid, etc.).

Sample of the file (header + rows):
<sample>
{sample}
</sample>

Return ONLY valid JSON matching this exact schema — no markdown fences, no commentary:
{{
  "entity_column":     "<column whose values identify nodes or links>",
  "value_column":      "<column containing the numeric quantity>",
  "time_column":       "<column containing timestamps, or null>",
  "channel_column":    "<column that distinguishes claim types, or null>",
  "channel_map":       {{
    "<channel_value>": "node" | "edge"
  }},
  "from_column":       "<explicit source-node column for edge rows, or null>",
  "to_column":         "<explicit target-node column for edge rows, or null>",
  "id_pattern":        "<regex to extract clean ID from entity_column, or null>",
  "excluded_channels": ["<channels encoding labels, anomalies, ground truth — not raw observations>"],
  "notes":             "<one sentence explaining the mapping>"
}}

Rules:
- Include ONLY channels that represent raw physical observations \
(flow, demand, pressure, stock, temperature, current…).
- EXCLUDE any channel whose name or values encode labels, anomalies, \
faults, corruptions, ground truth, or the quantity being estimated \
(e.g. leak_demand, label, is_anomaly, ground_truth, fault_flag).
- For "node" channels the entity_column value identifies a node.
- For "edge" channels the entity_column value identifies a link; \
  from_column / to_column supply explicit endpoints if present, \
  otherwise the link ID is used as the edge identifier.
"""


def _load_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Return (headers, rows) for a CSV/TSV file."""
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh, delimiter=sep)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    return headers, rows


def _sample_rows(headers: list[str], rows: list[dict], n: int = 20) -> str:
    """Return a TSV-formatted sample: header + n rows spread across the file."""
    step = max(1, len(rows) // n)
    sampled = [rows[i] for i in range(0, len(rows), step)][:n]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=headers, delimiter="\t")
    w.writeheader()
    w.writerows(sampled)
    return buf.getvalue()


def _get_mapping(sample: str, api_key: str | None, cache) -> dict:
    """
    Call LLM once (cached) to get the column mapping.
    Returns the mapping dict; raises if API key missing and no cache hit.
    """
    prompt = _MAPPING_PROMPT.format(sample=sample)
    blob = cache.call(prompt, api_key=api_key)

    match = re.search(r"\{.*\}", blob, re.DOTALL)
    if not match:
        raise ValueError(f"LLM returned no JSON for column mapping:\n{blob[:500]}")
    return json.loads(match.group(0))


_ENTITY_COLS = (
    "sensor_id", "entity_id", "sensor", "entity", "node_id", "site",
    "junction", "name", "id",
)
_VALUE_COLS = ("value", "qty", "quantity", "amount", "lb", "flow", "demand")
_TIME_COLS = ("timestamp", "time", "ts")
_CHANNEL_COLS = ("channel", "metric", "kind")
_FROM_COLS = ("from_node", "from", "source_name", "source", "src")
_TO_COLS = ("to_node", "to", "target_name", "target", "tgt")
_INITIAL_CHANNELS = {"opening_on_hand", "opening", "initial", "start_on_hand"}
_NON_CONSERVED = {
    "pressure", "head", "temperature", "temp", "voltage", "status", "quality",
}
_ARROW_RE = re.compile(r"\s*(?:->|→|=>|>)\s*")


def _header_lookup(headers: list[str]) -> dict[str, str]:
    return {h.lower(): h for h in headers}


def _pick_col(lookup: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def _numeric_fraction(rows: list[dict], col: str, n: int = 40) -> float:
    if not rows or not col:
        return 0.0
    sample = rows[:n]
    ok = 0
    for row in sample:
        try:
            float(row.get(col, ""))
            ok += 1
        except (TypeError, ValueError):
            pass
    return ok / max(len(sample), 1)


def _infer_channel_map(
    rows: list[dict],
    channel_col: str | None,
    from_col: str | None,
    to_col: str | None,
) -> dict[str, str]:
    if not channel_col or not rows:
        return {}
    from collections import defaultdict

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        ch = str(row.get(channel_col, "") or "").strip()
        if ch:
            buckets[ch].append(row)

    cmap: dict[str, str] = {}
    for ch, rs in buckets.items():
        cl = ch.lower()
        _, excl = leakage_guard([ch])
        if excl:
            continue
        if cl in _NON_CONSERVED:
            continue
        n_ep = 0
        if from_col and to_col:
            n_ep = sum(
                1 for r in rs
                if str(r.get(from_col, "") or "").strip()
                and str(r.get(to_col, "") or "").strip()
            )
        if from_col and to_col and n_ep >= max(1, 0.5 * len(rs)):
            cmap[ch] = "edge"
        elif "sink" in cl or "drain" in cl or any(
            k in cl for k in ("consum", "issued", "burned", "usage")
        ):
            cmap[ch] = "sink"
        elif any(k in cl for k in ("flow", "ship", "transfer", "pipe", "link")):
            cmap[ch] = "edge"
        else:
            cmap[ch] = "node"
    return cmap


def infer_column_mapping(
    headers: list[str],
    rows: list[dict] | None = None,
) -> dict | None:
    """Build a column mapping from headers and (optionally) row values."""
    rows = rows or []
    lookup = _header_lookup(headers)
    skip = {*(h.lower() for h in headers if h.lower() in {
        "timestamp", "time", "ts", "channel", "metric", "kind",
        "from_node", "to_node", "from", "to",
    })}

    entity = _pick_col(lookup, _ENTITY_COLS)
    if entity and entity.lower() in skip:
        entity = None
    value = _pick_col(lookup, _VALUE_COLS)
    if value and value.lower() in {"id", "timestamp", "time"}:
        value = None

    if not value and rows:
        best, best_f = None, 0.4
        for h in headers:
            if h.lower() in skip or h.lower() in {"id", "channel"}:
                continue
            frac = _numeric_fraction(rows, h)
            if frac > best_f:
                best, best_f = h, frac
        value = best

    if not entity:
        for h in headers:
            hl = h.lower()
            if hl in skip or h == value:
                continue
            if _numeric_fraction(rows, h) < 0.5:
                entity = h
                break

    if not entity or not value:
        return None

    from_col = _pick_col(lookup, _FROM_COLS)
    to_col = _pick_col(lookup, _TO_COLS)
    channel_col = _pick_col(lookup, _CHANNEL_COLS)
    channel_map = _infer_channel_map(rows, channel_col, from_col, to_col)

    return {
        "entity_column": entity,
        "value_column": value,
        "time_column": _pick_col(lookup, _TIME_COLS),
        "channel_column": channel_col,
        "channel_map": channel_map,
        "from_column": from_col,
        "to_column": to_col,
        "id_pattern": None,
        "excluded_channels": [],
        "notes": "Inferred from column headers and values",
    }


def _split_edge_name(entity: str) -> tuple[str, str] | None:
    parts = [p.strip() for p in _ARROW_RE.split(entity.strip()) if p.strip()]
    if len(parts) == 2:
        return parts[0], parts[1]
    return None


def _apply_mapping(
    headers: list[str],
    rows: list[dict],
    mapping: dict,
    exclusions: list[dict],
    schema: dict[str, tuple[str, str]] | None = None,
    aliases: dict[str, str] | None = None,
) -> tuple[dict[str, list[dict]], set[str], set[tuple], dict[str, float]]:
    """
    Apply the column mapping to every row in pure Python.
    Returns (claims_by_ts, node_ids_seen, edge_tuples_seen, initials).
    Zero LLM calls.
    """
    entity_col   = mapping.get("entity_column", "")
    value_col    = mapping.get("value_column", "")
    time_col     = mapping.get("time_column")
    channel_col  = mapping.get("channel_column")
    channel_map  = mapping.get("channel_map", {})
    from_col     = mapping.get("from_column")
    to_col       = mapping.get("to_column")
    id_pattern   = mapping.get("id_pattern")
    excluded     = set(mapping.get("excluded_channels", []))
    schema = schema or {}
    aliases = aliases or {}
    excluded_col_names = {e["name"] for e in exclusions}

    def _nid(raw: str) -> str:
        raw = (raw or "").strip()
        return aliases.get(raw, raw) if raw else raw

    id_re = re.compile(id_pattern) if id_pattern else None

    claims_by_ts: dict[str, list[dict]] = {}
    nodes_seen: set[str] = set()
    edges_seen: set[tuple] = set()
    initials: dict[str, float] = {}
    claim_idx = 0

    for row in rows:
        channel = str(row.get(channel_col, "") or "").strip() if channel_col else ""

        # Skip excluded / leakage channels (labels, ground truth, leak_demand, …)
        if channel in excluded:
            continue
        if channel_col and channel_col in excluded_col_names:
            continue
        if channel:
            _, ch_excl = leakage_guard([channel])
            if ch_excl:
                continue
            if channel.lower() in _NON_CONSERVED:
                continue

        entity = (row.get(entity_col, "") or "").strip()
        if not entity:
            continue

        # Clean entity ID with optional regex
        if id_re:
            m = id_re.search(entity)
            entity = m.group(0) if m else entity
        entity = _nid(entity)

        # Parse value
        raw_val = row.get(value_col, "")
        try:
            value = float(raw_val)
        except (ValueError, TypeError):
            # Prose in value column → caller should fall back to RECORD mode
            continue

        src = _nid((row.get(from_col, "") or "").strip() if from_col else "")
        tgt = _nid((row.get(to_col, "") or "").strip() if to_col else "")
        if not src or not tgt:
            parsed = _split_edge_name(entity)
            if parsed:
                src, tgt = _nid(parsed[0]), _nid(parsed[1])
        if (not src or not tgt) and entity in schema:
            src, tgt = schema[entity]
        if (not src or not tgt) and schema:
            for key in (entity, f"Link_{entity}", f"P_{entity}"):
                if key in schema:
                    src, tgt = schema[key]
                    break
        src, tgt = _nid(src), _nid(tgt)

        # Opening stock is the node's initial, not a measurement of final qty.
        if str(channel).strip().lower() in _INITIAL_CHANNELS:
            nid = src or entity
            if nid:
                initials[nid] = value
                nodes_seen.add(nid)
            continue

        ts = row.get(time_col, "all") if time_col else "all"
        primitive = channel_map.get(channel) or channel_map.get(str(channel).lower())
        if primitive is None:
            primitive = "edge" if (src and tgt) else "node"

        if primitive == "edge":
            if not src or not tgt:
                # No endpoints and no schema — cannot place a conservation edge.
                continue
            edge_key = (src, tgt)
            edges_seen.add(edge_key)
            ref = f"e_{src}_{tgt}"
            claim = {
                "id": f"c{claim_idx}",
                "type": "edge",
                "ref": ref,
                "value": value,
                "source": channel or entity_col,
                "weight": 1.0,
                "timestamp": None if ts == "all" else ts,
            }
        elif primitive == "sink":
            nodes_seen.add(entity)
            claim = {
                "id": f"c{claim_idx}",
                "type": "sink",
                "ref": entity,
                "value": value,
                "source": channel or entity_col,
                "weight": 1.0,
                "timestamp": None if ts == "all" else ts,
            }
        else:
            nodes_seen.add(entity)
            claim = {
                "id": f"c{claim_idx}",
                "type": "node",
                "ref": entity,
                "value": value,
                "source": channel or entity_col,
                "weight": 1.0,
                "timestamp": None if ts == "all" else ts,
            }

        claims_by_ts.setdefault(ts, []).append(claim)
        claim_idx += 1

    return claims_by_ts, nodes_seen, edges_seen, initials


def _build_graphs(
    claims_by_ts: dict[str, list[dict]],
    nodes_seen: set[str],
    edges_seen: set[tuple],
    sinks: str | None = None,
    initials: dict[str, float] | None = None,
) -> list[dict]:
    """Build one compile.py-format graph per timestamp bucket."""
    # Include all edge endpoints as nodes so compile.py balance rows are valid
    endpoint_nodes = {n for pair in edges_seen for n in pair}
    all_nodes = nodes_seen | endpoint_nodes
    initials = initials or {}

    # When sinks="unknown", mark nodes that were direct claim targets
    # (nodes_seen) as unknown; endpoint-only nodes stay "none".
    def _sink_mode(nid: str) -> str:
        if sinks == "unknown" and nid in nodes_seen:
            return "unknown"
        return "none"

    graph_nodes = [
        {"id": nid, "initial": float(initials.get(nid, 0.0)), "sinks": _sink_mode(nid)}
        for nid in sorted(all_nodes)
    ]
    graph_edges = [
        {"id": f"e_{src}_{tgt}", "from": src, "to": tgt}
        for (src, tgt) in sorted(edges_seen)
    ]
    graphs = []
    for ts, claims in sorted(claims_by_ts.items()):
        graphs.append({
            "nodes":  graph_nodes,
            "edges":  graph_edges,
            "claims": claims,
            "_timestamp": ts,
        })
    return graphs


def run_tabular_rows(
    headers: list[str],
    rows: list[dict],
    *,
    path: Path | None = None,
    api_key: str | None = None,
    cache=None,
    sinks: str | None = None,
    column_mapping: dict | None = None,
    schema: dict[str, tuple[str, str]] | None = None,
    aliases: dict[str, str] | None = None,
) -> tuple[list[dict], dict]:
    """Turn in-memory tabular rows into compile-format graphs."""
    _, col_exclusions = leakage_guard(headers)

    if column_mapping is not None:
        mapping = column_mapping
    else:
        mapping = None
        if cache is not None:
            try:
                from src.etl.llm import get_api_key
                if get_api_key(api_key):
                    mapping = _get_mapping(_sample_rows(headers, rows), api_key, cache)
            except Exception as e:
                print(f"[ingest] LLM column mapping failed ({e}); using header inference.", file=sys.stderr)
                mapping = None
        if mapping is None:
            mapping = infer_column_mapping(headers, rows)
        else:
            inferred = infer_column_mapping(headers, rows)
            if inferred:
                cm = dict(mapping.get("channel_map") or {})
                for k, v in (inferred.get("channel_map") or {}).items():
                    cm.setdefault(k, v)
                mapping["channel_map"] = cm
                for k in (
                    "from_column", "to_column", "entity_column",
                    "value_column", "channel_column", "time_column",
                ):
                    if not mapping.get(k) and inferred.get(k):
                        mapping[k] = inferred[k]
        if mapping is None:
            return [], {
                "mode": "TABULAR",
                "source_file": str(path) if path else "",
                "mapping": None,
                "exclusions": col_exclusions,
                "timestamp_buckets": [],
                "llm_calls": cache.llm_calls if cache else 0,
                "empty_reason": "could not infer a column mapping",
            }

    print(f"\n[ingest] Column mapping:\n{json.dumps(mapping, indent=2)}\n")

    llm_excluded = set(mapping.get("excluded_channels", []))
    all_exclusions = [*col_exclusions, *[
        {"name": ch, "reason": "excluded by column-mapping LLM"}
        for ch in llm_excluded
        if ch not in {e["name"] for e in col_exclusions}
    ]]

    claims_by_ts, nodes_seen, edges_seen, initials = _apply_mapping(
        headers, rows, mapping, all_exclusions, schema=schema, aliases=aliases,
    )
    if schema:
        for src, tgt in schema.values():
            if src and tgt:
                edges_seen.add((src, tgt))
                nodes_seen.add(src)
                nodes_seen.add(tgt)
    all_claims = [c for cs in claims_by_ts.values() for c in cs]
    validate_claims(all_claims)

    if sinks is None:
        channels = {str(k).lower() for k in (mapping.get("channel_map") or {})}
        leak_search = "flow" in channels and "demand" in channels
        sinks = "unknown" if leak_search else "none"

    graphs = _build_graphs(
        claims_by_ts, nodes_seen, edges_seen, sinks=sinks, initials=initials,
    )

    report = {
        "mode": "TABULAR",
        "source_file": str(path) if path else "",
        "mapping": mapping,
        "exclusions": all_exclusions,
        "timestamp_buckets": [
            {"timestamp": ts, "n_claims": len(cs)}
            for ts, cs in sorted(claims_by_ts.items())
        ],
        "llm_calls": cache.llm_calls if cache else 0,
    }
    return graphs, report


def run_tabular_mode(
    path: Path,
    api_key: str | None = None,
    lambda_w: float = 1.0,
    cache=None,
    sinks: str | None = None,
    column_mapping: dict | None = None,
    schema: dict[str, tuple[str, str]] | None = None,
    aliases: dict[str, str] | None = None,
) -> tuple[list[dict], dict]:
    """
    TABULAR mode for a single CSV/TSV/XLSX file.
    Returns (graphs, report).
    """
    if path.suffix.lower() == ".xlsx":
        try:
            import openpyxl
            wb = openpyxl.load_workbook(path, data_only=True)
            ws = wb.active
            all_rows_raw = list(ws.values)
            headers = [str(h) for h in all_rows_raw[0]]
            rows = [
                {headers[i]: str(v) if v is not None else "" for i, v in enumerate(row)}
                for row in all_rows_raw[1:]
            ]
        except ImportError:
            raise RuntimeError("Install openpyxl to read .xlsx:  pip install openpyxl")
    else:
        headers, rows = _load_csv(path)

    return run_tabular_rows(
        headers, rows,
        path=path, api_key=api_key, cache=cache,
        sinks=sinks, column_mapping=column_mapping, schema=schema, aliases=aliases,
    )
