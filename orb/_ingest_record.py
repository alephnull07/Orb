"""
orb/_ingest_record.py
---------------------
RECORD mode: split a free-text / JSONL file into records, extract ONE
observation per report, run reader consensus per observation, and emit
compile.py claims that keep every report as its own row.

Why per-report:  a sender's "800 cases departed" and the receiver's "800
cases arrived" are two independent observations of one transfer.  Keeping
both is what makes a false report catchable.  Nothing in this module sums,
averages, or majority-votes across *different* reports.  The only voting is
across the three READERS of the *same* report, which is extraction noise,
not evidence.

Record contract:  {record_id, text, file, line, timestamp}
Claim contract:   {id, type, value, ref|refs, source, weight,
                   timestamp, record_id, file, perspective|kind}
  source  = channel semantics: "sender" | "receiver" | "third_party" for
            transfers, "opening" | "closing" | "stock" for stock levels.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.etl.extract import GraphBuilder
from src.etl.llm import claude_generate, get_api_key
from src.etl.textutil import canon, corpus_text, is_site_name


def _key(name: str) -> str:
    """Grouping key: lowercase alphanumerics only. 'FOB_ALPHA' == 'FOB Alpha'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _node_id(display: str) -> str:
    """Stable id that keeps the source's separators: 'FOB Alpha' -> 'FOB_ALPHA'."""
    nid = re.sub(r"[^A-Za-z0-9]+", "_", display.strip()).strip("_").upper()
    return nid[:80] or "UNKNOWN"

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
    Each record: {record_id, text, file, line, timestamp}.
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
            ts = obj.get("timestamp") or obj.get("ts") or ""
            if not ts:
                m = _ISO_TS_RE.search(raw)
                ts = m.group(0) if m else ""
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
    Format records as the message dicts the readers expect.
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


# ── per-report extraction ─────────────────────────────────────────────────────
#
# An "item" is one observation from one report:
#   transfer: {kind:"transfer", src, dst, value, perspective, evidence, approx}
#   stock:    {kind:"stock",    site, value, stock_kind, evidence, approx}
# src/dst/site are display names; keys are canon(display).

PERSPECTIVES = ("sender", "receiver", "third_party")
STOCK_KINDS  = ("opening", "closing", "other")

_PERSONAS = {
    "scout": (
        "You are Scout, a literal field-log reader. You record exactly what each "
        "message states, one item per message, with no interpretation."
    ),
    "receiver": (
        "You are Receiver, an arrivals-focused reader. You are careful about which "
        "side of a transfer a message speaks from and what was actually counted."
    ),
    "auditor": (
        "You are Auditor, a skeptical conservation checker. You record each message's "
        "figure separately even when it contradicts another message — contradictions "
        "are evidence, not errors to reconcile."
    ),
}

_ITEM_PROMPT = """{persona}

Below are messages from a conserved-flow network (supplies, fuel, water, or any
commodity moving between physical sites). Each message starts with a header line
"[<message_id>] ...". Extract every quantitative observation as a SEPARATE item.

Output ONLY valid JSON matching this schema — no markdown fences, no commentary:
{{
  "transfers": [
    {{"from": "<site>", "to": "<site>", "quantity": 800,
      "perspective": "sender" | "receiver" | "third_party",
      "evidence": "<message_id>", "approx": false}}
  ],
  "stocks": [
    {{"site": "<site>", "quantity": 600,
      "kind": "opening" | "closing" | "other",
      "evidence": "<message_id>", "approx": false}}
  ]
}}

Rules:
1. ONE item per message per stated figure. Never add, combine, or reconcile
   numbers from different messages. If two messages describe the same transfer,
   output two items — one per message.
2. "evidence" is the single message_id the item came from.
3. Sites are physical places or assets only: never people, callsigns, convoys,
   drivers, TOC, HQ, or "ALL". Use the fullest site name as written in the
   corpus (e.g. "FOB ALPHA", "MAIN DEPOT") and spell it the same way everywhere.
   A convoy or driver reporting from a site speaks FOR that site.
4. "perspective": "sender" when the message comes from the origin side
   (dispatched, departed, sent, pushed, loaded); "receiver" when it comes from
   the destination side (received, arrived, got, took, counted off the truck);
   "third_party" otherwise.
5. "kind": "opening" for start-of-day / opening counts; "closing" for
   end-of-day / EOD / closing counts; "other" for any other on-hand level.
6. If a figure is vague ("about 300", "roughly"), include it with "approx": true.
   If a message states no number, do not invent one — skip it.
7. Do not output totals, balances, or figures the corpus does not state.

Messages:
{corpus}
"""


def _parse_items(blob: str, agent: str) -> list[dict]:
    """Parse the LLM JSON blob into a flat list of items."""
    match = re.search(r"\{.*\}", blob, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    items: list[dict] = []
    for t in data.get("transfers") or []:
        src = str(t.get("from") or "").strip()
        dst = str(t.get("to") or "").strip()
        val = _num(t.get("quantity"))
        ev  = str(t.get("evidence") or "").strip()
        if val is None or not src or not dst or not ev:
            continue
        if not is_site_name(src) or not is_site_name(dst) or _key(src) == _key(dst):
            continue
        persp = str(t.get("perspective") or "third_party").strip().lower()
        if persp not in PERSPECTIVES:
            persp = "third_party"
        items.append({
            "kind": "transfer", "agent": agent,
            "src": src, "dst": dst, "value": val,
            "perspective": persp, "evidence": ev,
            "approx": bool(t.get("approx")),
        })
    for s in data.get("stocks") or []:
        site = str(s.get("site") or "").strip()
        val  = _num(s.get("quantity"))
        ev   = str(s.get("evidence") or "").strip()
        if val is None or not site or not ev or not is_site_name(site):
            continue
        sk = str(s.get("kind") or "other").strip().lower()
        if sk not in STOCK_KINDS:
            sk = "other"
        items.append({
            "kind": "stock", "agent": agent,
            "site": site, "value": val,
            "stock_kind": sk, "evidence": ev,
            "approx": bool(s.get("approx")),
        })
    return items


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(str(v).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _extract_llm_items(agent: str, messages: list[dict], api_key: str) -> list[dict]:
    prompt = _ITEM_PROMPT.format(persona=_PERSONAS[agent], corpus=corpus_text(messages))
    blob = claude_generate(prompt, api_key=api_key)
    return _parse_items(blob, agent)


_REGEX_BIAS_PERSPECTIVE = {
    "send": "sender", "transfer": "sender", "flow": "third_party",
    "also_sent": "sender", "got": "receiver",
}


def _extract_regex_items(agent: str, messages: list[dict]) -> list[dict]:
    """Regex fallback (no API key): per-mention items from GraphBuilder."""
    g = GraphBuilder()
    if agent == "scout":
        g.ingest_text(messages, outbound=True, also_sent=True)
    elif agent == "receiver":
        g.ingest_text(messages, inbound=True)
    else:
        g.ingest_text(messages, outbound=True, inbound=True, also_sent=True,
                      eod=True, skip_suspicious=True)

    items: list[dict] = []
    for (sk, dk), rec in g.edges.items():
        src = g.nodes[sk]["display"]
        dst = g.nodes[dk]["display"]
        persp = _REGEX_BIAS_PERSPECTIVE.get(rec.get("bias", ""), "third_party")
        for m in rec["mentions"]:
            items.append({
                "kind": "transfer", "agent": agent,
                "src": src, "dst": dst, "value": float(m["value_lb"]),
                "perspective": persp, "evidence": m["mid"], "approx": False,
            })
    for key, row in g.eod.items():
        inv = row.get("inventory_eod_lb")
        if inv is None:
            continue
        items.append({
            "kind": "stock", "agent": agent,
            "site": g.nodes[key]["display"], "value": float(inv),
            "stock_kind": "closing", "evidence": row["evidence"], "approx": False,
        })
    return items


def _run_agent(agent_name: str, messages: list[dict],
               api_key: str | None = None, cache=None) -> list[dict]:
    """Return per-report items for one reader persona."""
    key = get_api_key(api_key)
    if not key:
        return _extract_regex_items(agent_name, messages)
    try:
        items = _extract_llm_items(agent_name, messages, key)
        if cache:
            cache.llm_calls += 1
        return items
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] LLM call failed for {agent_name}: {e}", file=sys.stderr)
        return _extract_regex_items(agent_name, messages)


# ── alias collapse (regex mode only) ─────────────────────────────────────────

def _collapse_aliases(items: list[dict]) -> None:
    """
    Regex extractors surface 'Alpha depot', 'Alpha-1', 'ALPHA' as different
    sites. Group by leading token and rewrite to the longest display, in place.
    Not used with LLM readers (they are told to spell sites consistently, and
    leading-token grouping would merge 'FOB ALPHA' with 'FOB BRAVO').
    """
    names: set[str] = set()
    for it in items:
        names.update([it.get("src", ""), it.get("dst", ""), it.get("site", "")])
    names.discard("")
    groups: dict[str, list[str]] = defaultdict(list)
    for n in names:
        k = canon(n)
        lead = k.split()[0] if k.split() else k
        groups[lead].append(n)
    rewrite: dict[str, str] = {}
    for members in groups.values():
        if len({canon(m) for m in members}) <= 1:
            continue
        best = max(members, key=len)
        for m in members:
            rewrite[m] = best
    if not rewrite:
        return
    for it in items:
        for f in ("src", "dst", "site"):
            if it.get(f) in rewrite:
                it[f] = rewrite[it[f]]


# ── consensus per report ──────────────────────────────────────────────────────

def _vote_value(vals: list[float], lambda_w: float) -> tuple[float, float]:
    """
    Resolve the READERS' values for one report.  Majority value; on a
    three-way split the middle value.  Weight shrinks with reader spread.
    This is extraction noise on a single report — not a merge of reports.
    """
    counts = Counter(vals)
    top = max(counts.values())
    winners = sorted(v for v, c in counts.items() if c == top)
    value = winners[len(winners) // 2]
    spread = max(vals) - min(vals)
    weight = round(1.0 / (spread + lambda_w), 6)
    return value, weight


def _vote_label(labels: list[str], default: str) -> str:
    counts = Counter(labels)
    top = max(counts.values())
    winners = [l for l, c in counts.items() if c == top]
    return winners[0] if len(winners) == 1 else (default if default in winners else winners[0])


def consensus_items(agent_items: dict[str, list[dict]], lambda_w: float = 1.0) -> dict:
    """
    Cross-reader consensus, per report.

    Returns {nodes: [{id, display, key, votes}], transfers: [...], stocks: [...]}
    where every transfer / stock is ONE report that >= majority of readers saw.
    """
    n_agents = len(agent_items)
    majority = max(1, min(2, (n_agents + 1) // 2))

    # Node votes: a site counts once per agent that mentioned it anywhere.
    site_votes: dict[str, set[str]] = defaultdict(set)
    site_display: dict[str, Counter] = defaultdict(Counter)
    for agent, items in agent_items.items():
        for it in items:
            for name in (it.get("src"), it.get("dst"), it.get("site")):
                if name:
                    k = _key(name)
                    site_votes[k].add(agent)
                    site_display[k][name] += 1

    kept: dict[str, dict] = {}
    for k, agents in site_votes.items():
        display = max(site_display[k].items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        if len(agents) < majority or not is_site_name(display):
            continue
        kept[k] = {"id": _node_id(display), "display": display, "key": k,
                   "votes": sorted(agents)}

    # Transfer votes keyed by (src, dst, evidence) — one report.
    t_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    s_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for items in agent_items.values():
        for it in items:
            if it["kind"] == "transfer":
                t_groups[(_key(it["src"]), _key(it["dst"]), it["evidence"])].append(it)
            else:
                s_groups[(_key(it["site"]), it["evidence"])].append(it)

    transfers = []
    for (sk, dk, ev), votes in sorted(t_groups.items()):
        agents = {v["agent"] for v in votes}
        if len(agents) < majority or sk not in kept or dk not in kept:
            continue
        value, weight = _vote_value([v["value"] for v in votes], lambda_w)
        if any(v["approx"] for v in votes):
            weight = round(weight * 0.5, 6)
        transfers.append({
            "src": kept[sk]["id"], "dst": kept[dk]["id"],
            "value": value, "weight": weight,
            "perspective": _vote_label([v["perspective"] for v in votes], "third_party"),
            "evidence": ev, "votes": sorted(agents),
            "reader_values": {v["agent"]: v["value"] for v in votes},
        })

    stocks = []
    for (k, ev), votes in sorted(s_groups.items()):
        agents = {v["agent"] for v in votes}
        if len(agents) < majority or k not in kept:
            continue
        value, weight = _vote_value([v["value"] for v in votes], lambda_w)
        if any(v["approx"] for v in votes):
            weight = round(weight * 0.5, 6)
        stocks.append({
            "site": kept[k]["id"], "value": value, "weight": weight,
            "stock_kind": _vote_label([v["stock_kind"] for v in votes], "other"),
            "evidence": ev, "votes": sorted(agents),
            "reader_values": {v["agent"]: v["value"] for v in votes},
        })

    return {"nodes": list(kept.values()), "transfers": transfers, "stocks": stocks}


def extract_records(
    records: list[dict],
    api_key: str | None = None,
    cache=None,
    max_workers: int = 10,
) -> tuple[dict, dict]:
    """
    Run 3-reader extraction on *records*.
    Returns (consensus, meta).  consensus has per-report transfers and stocks.
    """
    if not records:
        return {"nodes": [], "transfers": [], "stocks": []}, {}

    messages = records_to_messages(records)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            pool.submit(_run_agent, name, messages, api_key=api_key, cache=cache): name
            for name in ["scout", "receiver", "auditor"]
        }
        agent_items: dict[str, list[dict]] = {}
        for fut in as_completed(futs):
            agent_items[futs[fut]] = fut.result()

    if not get_api_key(api_key):
        for items in agent_items.values():
            _collapse_aliases(items)

    consensus = consensus_items(agent_items)
    meta = {
        "n_readers": len(agent_items),
        "items_per_reader": {a: len(v) for a, v in agent_items.items()},
    }
    return consensus, meta


# ── consensus → compile.py format ─────────────────────────────────────────────

_STOCK_SOURCE = {"opening": "opening", "closing": "closing", "other": "stock"}


def consensus_to_graph(consensus: dict, lambda_w: float = 1.0,
                       records: list[dict] | None = None) -> dict:
    """
    Convert per-report consensus into the format compile.py expects:
      {nodes, edges, claims}

    One claim per report.  Edge claims carry perspective as `source`
    ("sender" / "receiver" / "third_party"); stock claims carry
    "opening" / "closing" / "stock".  The merge step turns "opening" into
    the node's initial and sums same-perspective transfers across events.
    Nothing here derives a figure the corpus did not state.
    """
    ts_by_record = {r["record_id"]: r.get("timestamp", "") for r in (records or [])}
    file_by_record = {r["record_id"]: r.get("file", "") for r in (records or [])}

    c_nodes = [{"id": n["id"], "initial": 0.0, "sinks": "none"}
               for n in sorted(consensus.get("nodes", []), key=lambda n: n["id"])]
    node_ids = {n["id"] for n in c_nodes}

    edge_ids: dict[tuple[str, str], str] = {}
    c_edges: list[dict] = []
    for t in consensus.get("transfers", []):
        key = (t["src"], t["dst"])
        if key not in edge_ids and t["src"] in node_ids and t["dst"] in node_ids:
            eid = f"e{len(c_edges)}"
            edge_ids[key] = eid
            c_edges.append({"id": eid, "from": t["src"], "to": t["dst"]})

    c_claims: list[dict] = []
    for t in consensus.get("transfers", []):
        eid = edge_ids.get((t["src"], t["dst"]))
        if not eid:
            continue
        c_claims.append({
            "id": f"c{len(c_claims)}",
            "type": "edge",
            "ref": eid,
            "value": float(t["value"]),
            "source": t["perspective"],
            "weight": t["weight"],
            "timestamp": ts_by_record.get(t["evidence"], ""),
            "record_id": t["evidence"],
            "file": file_by_record.get(t["evidence"], ""),
            "readers": t["votes"],
        })
    for s in consensus.get("stocks", []):
        if s["site"] not in node_ids:
            continue
        c_claims.append({
            "id": f"c{len(c_claims)}",
            "type": "node",
            "ref": s["site"],
            "value": float(s["value"]),
            "source": _STOCK_SOURCE.get(s["stock_kind"], "stock"),
            "weight": s["weight"],
            "timestamp": ts_by_record.get(s["evidence"], ""),
            "record_id": s["evidence"],
            "file": file_by_record.get(s["evidence"], ""),
            "readers": s["votes"],
        })

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

    key = get_api_key(api_key)
    using_llm = bool(key)
    extraction_mode = "LLM" if using_llm else "REGEX_ONLY"

    if not using_llm:
        print(
            f"\n  *** LLM UNAVAILABLE — using regex extractors only ***\n"
            f"  Regex patterns are narrow and may miss most records.\n"
            f"  Set ANTHROPIC_API_KEY for LLM-based extraction.\n",
            file=sys.stderr,
        )

    consensus, meta = extract_records(records, api_key=api_key, cache=cache, max_workers=max_workers)
    graph = consensus_to_graph(consensus, lambda_w=lambda_w, records=records)
    graph["_timestamp"] = "all"

    graphs = [graph] if (graph["nodes"] or graph["claims"]) else []

    if not graphs and not using_llm:
        print(
            f"  *** NO GRAPH PRODUCED — regex extractors found nothing. ***\n"
            f"  This file requires LLM readers. Set ANTHROPIC_API_KEY.\n",
            file=sys.stderr,
        )

    report_records = [
        {"record_id": r["record_id"], "file": r["file"], "line": r["line"]}
        for r in records
    ]

    report = {
        "mode": "RECORD",
        "extraction_mode": extraction_mode,
        "source_file": str(path),
        "record_count": len(records),
        "records": report_records,
        "readers": meta,
        "timestamp_buckets": [
            {"timestamp": "all", "n_claims": len(graph.get("claims", []))}
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
) -> tuple[list[dict], dict]:
    """
    RECORD mode for multiple files — reads each, concatenates records,
    then runs a single extraction + consensus pipeline.
    """
    all_records: list[dict] = []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        all_records.extend(split_records(text, str(p)))

    consensus, meta = extract_records(all_records, api_key=api_key, cache=cache, max_workers=max_workers)
    graph = consensus_to_graph(consensus, lambda_w=lambda_w, records=all_records)
    graph["_timestamp"] = "all"

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
        "readers": meta,
        "timestamp_buckets": [
            {"timestamp": "all", "n_claims": len(graph.get("claims", []))}
        ],
        "llm_calls": cache.llm_calls if cache else 0,
    }
    return graphs, report
