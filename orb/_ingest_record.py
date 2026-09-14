"""
orb/_ingest_record.py
---------------------
RECORD mode: split a free-text / JSONL file into records, run the 3-reader
extraction (LLM primary, regex fallback), consensus, compile.py claims.

Record contract:  {record_id, text, file, line}
Claim contract:   {id, type, value, ref|refs, source, weight}
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import defaultdict

from src.etl import agent_scout, agent_receiver, agent_auditor
from src.etl.consensus import merge
from src.etl.llm import extract_llm, get_api_key
from src.etl.textutil import canon, slug, site_core

# ── record splitting ─────────────────────────────────────────────────────────

_BRACKET_TS_RE  = re.compile(r"^\[(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}[^\]]*)\]", re.M)
_SPEAKER_PRE_RE = re.compile(r"^[\w\s]+:[ \t]", re.M)
_ISO_TS_RE      = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?")


def split_records(text: str, file_path: str) -> list[dict]:
    """
    Split *text* into records on message boundaries.
    Boundaries: bracketed timestamps  [2026-09-12T06:15:47Z],
                blank lines between non-empty paragraphs,
                or JSONL (one JSON object per line).
    Each record: {record_id, text, file, line}.
    """
    # Try JSONL first
    lines = text.splitlines()
    json_objects: list[tuple[int, dict]] = []
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                json_objects.append((lineno, obj))
        except json.JSONDecodeError:
            break
    if len(json_objects) == len([l for l in lines if l.strip()]):
        # Uniform JSON lines → each line is a record (raw_text preferred)
        records = []
        for lineno, obj in json_objects:
            raw = obj.get("raw_text") or json.dumps(obj)
            ts = obj.get("timestamp", "")
            records.append({
                "record_id": f"{Path(file_path).stem}-L{lineno:04d}",
                "text": raw,
                "file": file_path,
                "line": lineno,
                "timestamp": ts,
            })
        return records

    # Bracket-timestamp splitting (most structured free-text logs)
    bracket_spans = [m.start() for m in _BRACKET_TS_RE.finditer(text)]
    if bracket_spans:
        records = []
        for i, start in enumerate(bracket_spans):
            end = bracket_spans[i + 1] if i + 1 < len(bracket_spans) else len(text)
            chunk = text[start:end].strip()
            if not chunk:
                continue
            lineno = text[:start].count("\n") + 1
            ts_match = _BRACKET_TS_RE.match(chunk)
            ts = ts_match.group(1) if ts_match else ""
            records.append({
                "record_id": f"{Path(file_path).stem}-L{lineno:04d}",
                "text": chunk,
                "file": file_path,
                "line": lineno,
                "timestamp": ts,
            })
        return records

    # Fallback: split on blank lines
    records = []
    para_start = 0
    for m in re.finditer(r"\n\s*\n", text):
        chunk = text[para_start:m.start()].strip()
        if chunk:
            lineno = text[:para_start].count("\n") + 1
            ts_m = _ISO_TS_RE.search(chunk)
            ts = ts_m.group(0) if ts_m else ""
            records.append({
                "record_id": f"{Path(file_path).stem}-L{lineno:04d}",
                "text": chunk,
                "file": file_path,
                "line": lineno,
                "timestamp": ts,
            })
        para_start = m.end()
    tail = text[para_start:].strip()
    if tail:
        lineno = text[:para_start].count("\n") + 1
        ts_m = _ISO_TS_RE.search(tail)
        ts = ts_m.group(0) if ts_m else ""
        records.append({
            "record_id": f"{Path(file_path).stem}-L{lineno:04d}",
            "text": tail,
            "file": file_path,
            "line": lineno,
            "timestamp": ts,
        })
    return records


# ── message formatting ────────────────────────────────────────────────────────

def records_to_messages(records: list[dict]) -> list[dict]:
    """
    Format records as the message dicts the existing readers expect.
    {message_id, timestamp, channel, from_name, callsign, to, raw_text}
    """
    msgs = []
    for r in records:
        # Try to parse bracket-style header from the record text
        text = r["text"]
        channel, from_name, callsign, to = "text_record", "", "", ""
        m = re.match(
            r"\[[^\]]+\]\s*\((\w+)\)\s*([^/]+?)\s*/\s*([^\s]+)\s*->\s*(\S+)",
            text,
        )
        if m:
            channel   = m.group(1)
            from_name = m.group(2).strip()
            callsign  = m.group(3).strip()
            to        = m.group(4).strip()
        else:
            # Field-log header: [ts] SITE / CALLSIGN   or   [ts] SITE
            m = re.match(
                r"\[[^\]]+\]\s+([^/\n]+?)(?:\s*/\s*(\S+))?\s*$",
                text.splitlines()[0] if text else "",
            )
            if m:
                from_name = m.group(1).strip()
                callsign  = (m.group(2) or "").strip()
                channel   = "field_log"

        msgs.append({
            "message_id": r["record_id"],
            "timestamp":  r.get("timestamp", ""),
            "channel":    channel,
            "from_name":  from_name,
            "callsign":   callsign,
            "to":         to,
            "raw_text":   text,
        })
    return msgs


# ── extraction + consensus ────────────────────────────────────────────────────

_REGEX_AGENTS = {
    "scout":    agent_scout.extract,
    "receiver": agent_receiver.extract,
    "auditor":  agent_auditor.extract,
}


def _union_agent(primary: dict, secondary: dict, allow_new_edges: bool = True) -> dict:
    """Merge two agent graphs (primary wins on conflicts). Mirrors src/etl/run.py."""
    nodes = {n["key"]: dict(n) for n in primary.get("nodes") or []}
    for n in secondary.get("nodes") or []:
        if n["key"] not in nodes:
            nodes[n["key"]] = dict(n)
        else:
            existing = set(nodes[n["key"]].get("aliases", []))
            existing.update(n.get("aliases", []))
            nodes[n["key"]]["aliases"] = sorted(existing)

    edges = {(e["source_key"], e["target_key"]): dict(e) for e in primary.get("edges") or []}
    for e in secondary.get("edges") or []:
        k = (e["source_key"], e["target_key"])
        if k not in edges:
            if allow_new_edges:
                edges[k] = dict(e)
        else:
            ev = list(edges[k].get("evidence") or [])
            for mid in e.get("evidence") or []:
                if mid not in ev:
                    ev.append(mid)
            edges[k]["evidence"] = ev

    constraints = list(primary.get("trusted_constraint_rows") or [])
    seen_nodes = {c["node"] for c in constraints}
    for c in secondary.get("trusted_constraint_rows") or []:
        if c["node"] not in seen_nodes:
            constraints.append(c)
            seen_nodes.add(c["node"])

    out = dict(primary)
    out["nodes"] = list(nodes.values())
    out["edges"] = list(edges.values())
    out["trusted_constraint_rows"] = constraints
    return out


def _run_agent(agent_name: str, messages: list[dict],
               api_key: str | None = None, cache=None,
               topology: dict | None = None) -> dict:
    """LLM extracts the graph; regex only fills gaps if the LLM is missing or thin."""
    regex_graph = _REGEX_AGENTS[agent_name](messages, topology=topology)

    key = get_api_key(api_key)
    if not key:
        return regex_graph

    try:
        llm_graph = extract_llm(agent_name, messages, api_key=key, topology=topology)
        if cache:
            cache.llm_calls += 1
    except Exception as e:
        print(f"  [warn] LLM call failed for {agent_name}: {e}", file=sys.stderr)
        llm_graph = None

    if not llm_graph:
        return regex_graph

    llm_n = len(llm_graph.get("nodes") or []) + len(llm_graph.get("edges") or [])
    if llm_n == 0:
        return regex_graph

    # LLM is primary for every persona, including Scout — regex may add
    # extra edges/nodes the model missed, but cannot veto LLM topology.
    return _union_agent(llm_graph, regex_graph, allow_new_edges=True)


def _collapse_node_aliases(graphs: dict[str, dict]) -> None:
    """
    Collapse true aliases, not sites that merely share a facility prefix.

    ``Alpha depot`` / ``Alpha-1`` / ``FOB Alpha`` → one node (core ``alpha``).
    ``FOB Alpha`` / ``FOB Bravo`` stay two nodes (cores ``alpha`` vs ``bravo``).
    ``OP Crescent`` / ``OP Delta`` / ``OP Echo`` stay three nodes.
    """
    all_nodes: list[dict] = []
    for g in graphs.values():
        all_nodes.extend(g.get("nodes") or [])

    groups: dict[str, list[dict]] = defaultdict(list)
    for n in all_nodes:
        groups[site_core(n.get("key") or n.get("display") or "")].append(n)

    # Build rewrite maps: old_key -> canonical_key, old_id -> canonical_id
    key_map: dict[str, str] = {}   # old canon key → canonical canon key
    id_map: dict[str, str] = {}    # old slug id → canonical slug id
    id_display: dict[str, str] = {}  # old slug id → canonical display

    for core, members in groups.items():
        if not core:
            continue
        unique_keys = {n["key"] for n in members}
        if len(unique_keys) <= 1:
            continue  # no aliasing needed

        # Pick canonical = longest display name
        canonical_node = max(members, key=lambda n: len(n.get("display", "")))
        c_key = canonical_node["key"]
        c_id = canonical_node["id"]
        c_display = canonical_node["display"]

        # Collect all aliases from the group
        all_aliases: set[str] = set()
        for n in members:
            all_aliases.update(n.get("aliases", []))
            all_aliases.add(n.get("display", ""))

        for n in members:
            if n["key"] != c_key:
                key_map[n["key"]] = c_key
                id_map[n["id"]] = c_id
                id_display[n["id"]] = c_display

        # Update canonical node's aliases to include all
        canonical_node["aliases"] = sorted(all_aliases - {""})

    if not key_map:
        return

    # Rewrite every graph in-place
    for g in graphs.values():
        # Rewrite nodes: merge aliased nodes into canonical
        seen_keys: set[str] = set()
        new_nodes = []
        for n in (g.get("nodes") or []):
            old_key = n["key"]
            new_key = key_map.get(old_key, old_key)
            if new_key in seen_keys:
                # Already have canonical node; just merge aliases
                existing = next(nn for nn in new_nodes if nn["key"] == new_key)
                for a in n.get("aliases", []):
                    if a not in existing["aliases"]:
                        existing["aliases"].append(a)
                existing["aliases"] = sorted(existing["aliases"])
                continue
            if old_key in key_map:
                old_id = n["id"]
                n["key"] = new_key
                n["id"] = id_map.get(old_id, old_id)
                n["display"] = id_display.get(old_id, n["display"])
            seen_keys.add(new_key)
            new_nodes.append(n)
        g["nodes"] = new_nodes

        # Rewrite edges
        for e in (g.get("edges") or []):
            e["source_key"] = key_map.get(e["source_key"], e["source_key"])
            e["target_key"] = key_map.get(e["target_key"], e["target_key"])
            e["source"] = id_map.get(e["source"], e["source"])
            e["target"] = id_map.get(e["target"], e["target"])

        # Rewrite trusted_constraint_rows
        for c in (g.get("trusted_constraint_rows") or []):
            old_id = c.get("node", "")
            if old_id in id_map:
                c["node"] = id_map[old_id]
                c["display"] = id_display.get(old_id, c.get("display", ""))


def extract_records(
    records: list[dict],
    api_key: str | None = None,
    cache=None,
    max_workers: int = 10,
    topology: dict | None = None,
) -> tuple[list[dict], dict]:
    """
    Run 3-reader extraction on *records*.
    Returns (consensus, meta).
    Concurrency: 3 agent calls run concurrently inside this function;
    callers may call extract_records() concurrently for multiple batches
    (max_workers is forwarded from build_graph).
    """
    if not records:
        return [], {}

    messages = records_to_messages(records)

    # Run 3 agents concurrently (regex + LLM when api_key present)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            pool.submit(
                _run_agent, name, messages,
                api_key=api_key, cache=cache, topology=topology,
            ): name
            for name in ["scout", "receiver", "auditor"]
        }
        graphs = {}
        for fut in as_completed(futs):
            name = futs[fut]
            graphs[name] = fut.result()

    # Collapse true aliases (FOB Alpha / Alpha-1) before merge; do not
    # merge sites that only share a facility prefix (FOB Alpha / FOB Bravo).
    _collapse_node_aliases(graphs)

    consensus = merge([graphs["scout"], graphs["receiver"], graphs["auditor"]])
    return consensus, {}


# ── consensus → compile.py format ─────────────────────────────────────────────

def consensus_to_graph(consensus: dict, lambda_w: float = 1.0) -> dict:
    """
    Convert a consensus graph dict to the format compile.py expects:
      {nodes, edges, claims}

    Nodes: initial=0 (unknown starting stock); override via EOD if available.
    Edges: each consensus edge becomes a graph edge + an edge claim.
    EOD rows: become node claims.
    w = 1 / (spread + lambda_w)  where spread = max(reader values) - min(reader values).
    """
    c_nodes: list[dict] = []
    c_edges: list[dict] = []
    c_claims: list[dict] = []

    # Nodes — initial=0 for all (starting stock unknown);
    # EOD inventory is used ONLY as a node claim, not as initial,
    # to avoid double-counting in the balance constraint.
    node_ids = {n["id"] for n in consensus.get("nodes", [])}
    for n in consensus.get("nodes", []):
        c_nodes.append({"id": n["id"], "initial": 0.0, "sinks": "none"})

    # Edges → edge claims
    claim_idx = 0
    for ei, e in enumerate(consensus.get("edges", [])):
        src, tgt = e["source"], e["target"]
        eid = f"e{ei}"
        c_edges.append({"id": eid, "from": src, "to": tgt})

        # Spread across agent values for this edge
        agent_vals = [v for v in e.get("agent_values", {}).values()]
        if not agent_vals:
            agent_vals = [e["value_lb"]]
        spread = max(agent_vals) - min(agent_vals)
        w = round(1.0 / (spread + lambda_w), 6)
        median_val = sorted(agent_vals)[len(agent_vals) // 2]

        c_claims.append({
            "id": f"c{claim_idx}",
            "type": "edge",
            "ref": eid,
            "value": float(median_val),
            "source": "consensus",
            "weight": w,
        })
        claim_idx += 1

    # EOD inventory → node claims  (only if non-zero)
    for row in consensus.get("trusted_constraint_rows", []):
        nid = row.get("node")
        inv = row.get("inventory_eod_lb")
        if nid and inv and float(inv) != 0.0 and nid in node_ids:
            c_claims.append({
                "id": f"c{claim_idx}",
                "type": "node",
                "ref": nid,
                "value": float(inv),
                "source": row.get("source", "auditor_eod"),
                "weight": 1.0,
            })
            claim_idx += 1

    return {"nodes": c_nodes, "edges": c_edges, "claims": c_claims}


# ── mode entry point ──────────────────────────────────────────────────────────

def run_record_mode(
    path: Path,
    api_key: str | None = None,
    lambda_w: float = 1.0,
    cache=None,
    max_workers: int = 10,
    topology: dict | None = None,
) -> tuple[list[dict], dict]:
    """
    RECORD mode for a single file.
    Returns (graphs, report).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    records = split_records(text, str(path))

    consensus, _ = extract_records(
        records, api_key=api_key, cache=cache, max_workers=max_workers,
        topology=topology,
    )
    graph = consensus_to_graph(consensus, lambda_w=lambda_w)

    # Group by timestamp bucket ("all" window for text files)
    ts_bucket = "all"
    graphs = [graph] if (graph["nodes"] or graph["claims"]) else []

    report_records = [
        {"record_id": r["record_id"], "file": r["file"], "line": r["line"]}
        for r in records
    ]

    report = {
        "mode": "RECORD",
        "source_file": str(path),
        "record_count": len(records),
        "records": report_records,
        "timestamp_buckets": [
            {"timestamp": ts_bucket, "n_claims": len(graph.get("claims", []))}
        ],
        "llm_calls": cache.llm_calls if cache else 0,
    }
    return graphs, report


def run_record_mode_multi(
    paths: list[Path],
    api_key: str | None = None,
    lambda_w: float = 1.0,
    cache=None,
    max_workers: int = 10,
    topology: dict | None = None,
) -> tuple[list[dict], dict]:
    """
    RECORD mode for multiple files — reads each, concatenates records,
    then runs a single extraction + consensus pipeline.
    """
    all_records: list[dict] = []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        all_records.extend(split_records(text, str(p)))

    consensus, _ = extract_records(
        all_records, api_key=api_key, cache=cache, max_workers=max_workers,
        topology=topology,
    )
    graph = consensus_to_graph(consensus, lambda_w=lambda_w)

    graphs = [graph] if (graph["nodes"] or graph["claims"]) else []

    report_records = [
        {"record_id": r["record_id"], "file": r["file"], "line": r["line"]}
        for r in all_records
    ]

    report = {
        "mode": "RECORD",
        "source_files": [str(p) for p in paths],
        "record_count": len(all_records),
        "records": report_records,
        "timestamp_buckets": [
            {"timestamp": "all", "n_claims": len(graph.get("claims", []))}
        ],
        "llm_calls": cache.llm_calls if cache else 0,
    }
    return graphs, report
