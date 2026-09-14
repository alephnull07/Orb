"""
orb/_ingest_record.py
---------------------
RECORD mode: split a free-text / JSONL file into records, run the existing
3-reader regex extraction, run consensus, and convert to compile.py claims.

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

from src.etl import agent_scout, agent_receiver, agent_auditor
from src.etl.consensus import merge

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

def _run_agent(agent_name: str, messages: list[dict]) -> dict:
    agents = {
        "scout":    agent_scout.extract,
        "receiver": agent_receiver.extract,
        "auditor":  agent_auditor.extract,
    }
    return agents[agent_name](messages, topology=None)


def extract_records(
    records: list[dict],
    api_key: str | None = None,
    cache=None,
    max_workers: int = 10,
) -> tuple[list[dict], dict]:
    """
    Run 3-reader extraction on *records*.
    Returns (claims, meta) where meta = {agent_values_per_edge}.
    Concurrency: 3 agent calls run concurrently inside this function;
    callers may call extract_records() concurrently for multiple batches
    (max_workers is forwarded from build_graph).
    """
    if not records:
        return [], {}

    messages = records_to_messages(records)

    # Run 3 agents concurrently (regex; LLM optional via cache)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            pool.submit(_run_agent, name, messages): name
            for name in ["scout", "receiver", "auditor"]
        }
        graphs = {}
        for fut in as_completed(futs):
            name = futs[fut]
            graphs[name] = fut.result()

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

    # Build initial values from trusted_constraint_rows (EOD inventory)
    eod_by_id: dict[str, float] = {}
    for row in consensus.get("trusted_constraint_rows", []):
        nid = row.get("node")
        inv = row.get("inventory_eod_lb")
        if nid and inv is not None:
            eod_by_id[nid] = float(inv)

    # Nodes
    node_ids = {n["id"] for n in consensus.get("nodes", [])}
    for n in consensus.get("nodes", []):
        initial = eod_by_id.get(n["id"], 0.0)
        c_nodes.append({"id": n["id"], "initial": initial, "sinks": "none"})

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
) -> tuple[list[dict], dict]:
    """
    RECORD mode for a single file.
    Returns (graphs, report).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    records = split_records(text, str(path))

    consensus, _ = extract_records(records, api_key=api_key, cache=cache, max_workers=max_workers)
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
