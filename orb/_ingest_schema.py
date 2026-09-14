"""
Classify sidecar JSON and build a link-id → endpoint schema.

Used so mixed uploads (SCADA CSV + topology.json + Labels.csv) don't send
ground-truth / network-schema files through the LLM readers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_SKIP_STEMS = {
    "labels",
    "scenario_info",
    "corruption_labels",
    "true_graph",
    "corrupted_graph",
}

_INFO_STEM_RE = re.compile(r"^leak_.*_info$", re.I)

_SCADA_KEYS = {"timestamp", "channel", "sensor", "value"}


def peek_json(path: Path) -> Any | None:
    if path.suffix.lower() not in {".json", ".jsonl"}:
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def skip_reason(path: Path) -> str | None:
    """Return a reason to skip this file as labels / GT / metadata, else None.

    Only known sidecars are skipped. Observation files may have 'corruptions'
    in the name (e.g. fuel_B_two_corruptions.csv) without being ground truth.
    Column/channel leakage_guard still runs inside tabular ingest.
    """
    stem = Path(path).stem.lower()
    if stem in _SKIP_STEMS or _INFO_STEM_RE.match(stem):
        return f"{path.name} is a labels/metadata sidecar, not an observation file"
    return None


def classify_json(obj: Any) -> str | None:
    """GRAPH | SCHEMA | TRUTH | None."""
    if not isinstance(obj, dict):
        return None
    if "leak_node" in obj or "leak_size" in obj:
        return "TRUTH"
    if isinstance(obj.get("nodes"), list) and isinstance(obj.get("claims"), list):
        return "GRAPH"
    links = obj.get("links")
    if isinstance(links, dict) and links:
        rec = next(iter(links.values()), None)
        if isinstance(rec, dict) and ("source_name" in rec or "from" in rec):
            return "SCHEMA"
    if isinstance(obj.get("nodes"), list) and isinstance(obj.get("pipes"), list):
        return "SCHEMA"
    return None


def is_structured_observation(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    keys = {k.lower() for k in obj}
    return "value" in keys and (
        "sensor" in keys or "entity" in keys or "entity_id" in keys
    ) and ("channel" in keys or "metric" in keys)


def load_jsonl_rows(path: Path) -> tuple[list[str], list[dict]] | None:
    rows: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return None
            if not isinstance(obj, dict):
                return None
            rows.append({str(k): v for k, v in obj.items()})
    if not rows or not is_structured_observation(rows[0]):
        return None
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                headers.append(k)
    return headers, [{h: row.get(h, "") for h in headers} for row in rows]


def _canon_node(n: dict) -> str:
    return str(n.get("id") or n.get("display") or n.get("nato") or "").strip()


def alias_map(topology: dict) -> dict[str, str]:
    """Any observed name → canonical node id."""
    out: dict[str, str] = {}

    def add(alias: str, nid: str) -> None:
        a = (alias or "").strip()
        if a and nid:
            out[a] = nid

    nodes = topology.get("nodes") or []
    junctions = [n for n in nodes if str(n.get("type") or "").lower() not in {"source", "reservoir"}]
    sources = [n for n in nodes if str(n.get("type") or "").lower() in {"source", "reservoir"}
               or "reservoir" in str(n.get("kind") or "").lower()]

    for group in (junctions, sources):
        for n in group:
            nid = _canon_node(n)
            if not nid:
                continue
            add(nid, nid)
            add(n.get("display") or "", nid)
            nato = str(n.get("nato") or "").strip()
            kind = str(n.get("kind") or n.get("type") or "").lower()
            for a in n.get("aliases") or []:
                add(str(a), nid)
            if nato:
                add(f"J_{nato}", nid)
                add(f"J-{nato}", nid)
                add(f"J{nato}", nid)
                add(f"Junction {nato}", nid)
                add(f"Node_{nato}", nid)
                add(f"N_{nato}", nid)
                if "reservoir" in kind or n.get("type") == "source":
                    add(f"Reservoir {nato}", nid)
                    add(f"R_{nato}", nid)
                    add(f"R-{nato}", nid)
                    add(f"RES_{nato}", nid)
    return out


def schema_from_topology(topology: dict) -> dict[str, tuple[str, str]]:
    """Link / pipe id → (source_id, target_id)."""
    aliases = alias_map(topology)
    schema: dict[str, tuple[str, str]] = {}

    def resolve(name: str) -> str:
        name = (name or "").strip()
        return aliases.get(name, name)

    links = topology.get("links") or {}
    if isinstance(links, dict):
        for key, rec in links.items():
            if not isinstance(rec, dict):
                continue
            src = resolve(rec.get("source_name") or rec.get("from") or rec.get("source") or "")
            tgt = resolve(rec.get("target_name") or rec.get("to") or rec.get("target") or "")
            if not src or not tgt:
                continue
            pair = (src, tgt)
            pipe = str(rec.get("pipe_id") or key)
            for k in {str(key), pipe, f"Link_{pipe}", f"Link_{key}", f"P_{pipe}"}:
                schema[k] = pair

    for p in topology.get("pipes") or []:
        if not isinstance(p, dict):
            continue
        src = resolve(str(p.get("n1") or p.get("from") or ""))
        tgt = resolve(str(p.get("n2") or p.get("to") or ""))
        pid = str(p.get("pipe_id") or p.get("id") or "")
        if src and tgt and pid:
            schema[pid] = (src, tgt)
            schema[f"Link_{pid}"] = (src, tgt)

    return schema


def load_topology(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return obj if isinstance(obj, dict) else {}


def looks_like_scada_table(headers: list[str]) -> bool:
    return {h.lower() for h in headers} >= _SCADA_KEYS
