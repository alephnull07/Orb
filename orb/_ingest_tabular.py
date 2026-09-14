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


def _apply_mapping(
    headers: list[str],
    rows: list[dict],
    mapping: dict,
    exclusions: list[dict],
) -> tuple[dict[str, list[dict]], set[str], set[tuple]]:
    """
    Apply the column mapping to every row in pure Python.
    Returns (claims_by_ts, node_ids_seen, edge_tuples_seen).
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
    excluded_col_names = {e["name"] for e in exclusions}

    id_re = re.compile(id_pattern) if id_pattern else None

    claims_by_ts: dict[str, list[dict]] = {}
    nodes_seen: set[str] = set()
    edges_seen: set[tuple] = set()
    claim_idx = 0

    for row in rows:
        channel = row.get(channel_col, "") if channel_col else ""

        # Skip excluded channels (leakage guard)
        if channel in excluded:
            continue
        if channel_col and channel_col in excluded_col_names:
            continue

        entity = row.get(entity_col, "").strip()
        if not entity:
            continue

        # Clean entity ID with optional regex
        if id_re:
            m = id_re.search(entity)
            entity = m.group(0) if m else entity

        # Parse value
        raw_val = row.get(value_col, "")
        try:
            value = float(raw_val)
        except (ValueError, TypeError):
            # Prose in value column → caller should fall back to RECORD mode
            continue

        ts = row.get(time_col, "all") if time_col else "all"
        primitive = channel_map.get(channel, "node")

        if primitive == "edge":
            src = row.get(from_col, entity).strip() if from_col else entity
            tgt = row.get(to_col, "NETWORK").strip() if to_col else "NETWORK"
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
            }

        claims_by_ts.setdefault(ts, []).append(claim)
        claim_idx += 1

    return claims_by_ts, nodes_seen, edges_seen


def _build_graphs(
    claims_by_ts: dict[str, list[dict]],
    nodes_seen: set[str],
    edges_seen: set[tuple],
) -> list[dict]:
    """Build one compile.py-format graph per timestamp bucket."""
    # Include all edge endpoints as nodes so compile.py balance rows are valid
    endpoint_nodes = {n for pair in edges_seen for n in pair}
    all_nodes = nodes_seen | endpoint_nodes
    graph_nodes = [{"id": nid, "initial": 0.0, "sinks": "none"} for nid in sorted(all_nodes)]
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


def run_tabular_mode(
    path: Path,
    api_key: str | None = None,
    lambda_w: float = 1.0,
    cache=None,
) -> tuple[list[dict], dict]:
    """
    TABULAR mode for a single CSV/TSV/XLSX file.
    Returns (graphs, report).  One LLM call total (cached).
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

    # ── leakage guard on column names ─────────────────────────────────────────
    _, col_exclusions = leakage_guard(headers)

    # Build sample
    sample = _sample_rows(headers, rows)

    # ONE LLM call (or cache hit)
    mapping = _get_mapping(sample, api_key, cache)
    print(f"\n[ingest] Column mapping:\n{json.dumps(mapping, indent=2)}\n")

    # Merge LLM-requested exclusions with leakage-guard exclusions
    llm_excluded = set(mapping.get("excluded_channels", []))
    all_exclusions = [*col_exclusions, *[
        {"name": ch, "reason": "excluded by column-mapping LLM"}
        for ch in llm_excluded
        if ch not in {e["name"] for e in col_exclusions}
    ]]

    # Apply mapping (pure Python, zero LLM)
    claims_by_ts, nodes_seen, edges_seen = _apply_mapping(
        headers, rows, mapping, all_exclusions
    )
    validate_claims([c for cs in claims_by_ts.values() for c in cs])

    graphs = _build_graphs(claims_by_ts, nodes_seen, edges_seen)

    report = {
        "mode": "TABULAR",
        "source_file": str(path),
        "mapping": mapping,
        "exclusions": all_exclusions,
        "timestamp_buckets": [
            {"timestamp": ts, "n_claims": len(cs)}
            for ts, cs in sorted(claims_by_ts.items())
        ],
        "llm_calls": cache.llm_calls if cache else 1,
    }
    return graphs, report
