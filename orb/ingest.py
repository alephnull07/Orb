"""
orb/ingest.py
-------------
Public entry point: build_graph(path) → (list[graph_dict], report_dict)

Routing (by extension + a cheap JSON peek, no LLM):
  .csv / .tsv / .xlsx  → TABULAR
  .txt / .log / .md    → RECORD
  .jsonl               → structured SCADA rows → TABULAR, else RECORD
  .json                → compile GRAPH / network SCHEMA / TRUTH (skipped)

Cross-cutting:
  leakage_guard()   — regex-flag columns/channels before they enter extraction
  validate_claims() — reject any claim whose type is not in VALID_TYPES
  Cache             — shared disk cache passed to both mode modules
"""

from __future__ import annotations

from pathlib import Path

from ._cache import Cache
from ._ingest_record  import run_record_mode
from ._ingest_schema  import (
    alias_map,
    classify_json,
    load_jsonl_rows,
    load_topology,
    looks_like_scada_table,
    peek_json,
    schema_from_topology,
    skip_reason,
)
from ._ingest_tabular import run_tabular_mode, run_tabular_rows
from ._ingest_utils   import leakage_guard, validate_claims, VALID_TYPES  # re-export


# ── routing ────────────────────────────────────────────────────────────────────

_TABULAR_EXTS = {".csv", ".tsv", ".xlsx"}
_RECORD_EXTS  = {".txt", ".log", ".md"}


def _route(path: Path) -> str:
    """Return TABULAR, RECORD, JSON, or SKIP."""
    reason = skip_reason(path)
    if reason:
        return "SKIP"
    ext = path.suffix.lower()
    if ext in _TABULAR_EXTS:
        return "TABULAR"
    if ext in _RECORD_EXTS:
        return "RECORD"
    if ext == ".json":
        return "JSON"
    if ext == ".jsonl":
        return "JSONL"
    return "RECORD"


def _skipped(path: Path, reason: str) -> tuple[list[dict], dict]:
    return [], {
        "mode": "SKIP",
        "source_file": str(path),
        "skipped": True,
        "empty_reason": reason,
        "llm_calls": 0,
        "record_count": 0,
        "exclusions": [],
    }


def _topology_graph(topology: dict) -> dict:
    schema = schema_from_topology(topology)
    nodes: dict[str, dict] = {}
    for n in topology.get("nodes") or []:
        nid = str(n.get("id") or n.get("display") or "").strip()
        if nid:
            nodes[nid] = {"id": nid, "initial": 0.0, "sinks": "none"}
    edges = []
    seen: set[tuple[str, str]] = set()
    for src, tgt in schema.values():
        key = (src, tgt)
        if key in seen or not src or not tgt:
            continue
        seen.add(key)
        edges.append({"id": f"e_{src}_{tgt}", "from": src, "to": tgt})
        for nid in (src, tgt):
            nodes.setdefault(nid, {"id": nid, "initial": 0.0, "sinks": "none"})
    return {"nodes": list(nodes.values()), "edges": edges, "claims": []}


def _maybe_record_fallback(path: Path, graphs: list[dict], report: dict, **kwargs) -> tuple[list[dict], dict]:
    """Prose tables with no numeric mapping fall through to the LLM readers.

    Do not fall back when a mapping existed (label files, SCADA without schema)
    — that path is what used to dump Labels.csv into three Claude calls.
    """
    if any(g.get("claims") or g.get("nodes") for g in graphs):
        return graphs, report
    if report.get("mapping"):
        return graphs, report
    rec_graphs, rec_report = run_record_mode(path, **kwargs)
    if rec_graphs:
        rec_report["fell_back_from"] = "TABULAR"
        return rec_graphs, rec_report
    return graphs, report


# ── main entry point ───────────────────────────────────────────────────────────

def build_graph(
    path: str | Path,
    api_key: str | None = None,
    lambda_w: float = 1.0,
    cache_dir: str | Path | None = None,
    sinks: str | None = None,
    column_mapping: dict | None = None,
    schema: dict[str, tuple[str, str]] | None = None,
    aliases: dict[str, str] | None = None,
    topology: dict | None = None,
) -> tuple[list[dict], dict]:
    """
    Convert *path* (any supported file) into one or more compile.py-format graphs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    cache = Cache(cache_dir)
    mode = _route(path)
    rec_kw = dict(api_key=api_key, lambda_w=lambda_w, cache=cache, topology=topology)

    if mode == "SKIP":
        return _skipped(path, skip_reason(path) or "skipped")

    if mode == "JSON":
        obj = peek_json(path) or {}
        kind = classify_json(obj)
        if kind == "TRUTH":
            return _skipped(path, f"{path.name} is a truth/eval sidecar (leak_node)")
        if kind == "GRAPH":
            g = {
                "nodes": obj.get("nodes") or [],
                "edges": obj.get("edges") or [],
                "claims": obj.get("claims") or [],
            }
            if obj.get("lambda_sink") is not None:
                g["lambda_sink"] = obj["lambda_sink"]
            return [g], {
                "mode": "GRAPH",
                "source_file": str(path),
                "llm_calls": 0,
                "exclusions": [],
            }
        if kind == "SCHEMA":
            g = _topology_graph(obj)
            return ([g] if g["nodes"] or g["edges"] else []), {
                "mode": "SCHEMA",
                "source_file": str(path),
                "llm_calls": 0,
                "n_links": len(schema_from_topology(obj)),
                "exclusions": [],
            }
        return run_record_mode(path, **rec_kw)

    if mode == "JSONL":
        loaded = load_jsonl_rows(path)
        if loaded:
            headers, rows = loaded
            graphs, report = run_tabular_rows(
                headers, rows,
                path=path, api_key=api_key, cache=cache,
                sinks=sinks, column_mapping=column_mapping,
                schema=schema, aliases=aliases,
            )
            return _maybe_record_fallback(path, graphs, report, **rec_kw)
        return run_record_mode(path, **rec_kw)

    if mode == "TABULAR":
        graphs, report = run_tabular_mode(
            path, api_key=api_key, lambda_w=lambda_w, cache=cache,
            sinks=sinks, column_mapping=column_mapping,
            schema=schema, aliases=aliases,
        )
        return _maybe_record_fallback(path, graphs, report, **rec_kw)

    graphs, report = run_record_mode(path, **rec_kw)
    return graphs, report


def merge_graphs(graphs: list[dict]) -> dict:
    """Union nodes/edges/claims from any number of compile-format graphs."""
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], str] = {}
    claims: list[dict] = []
    cid = 0
    lambda_sink = None

    for g in graphs:
        if g.get("lambda_sink") is not None:
            lambda_sink = g["lambda_sink"]
        id_map: dict[str, str] = {}
        for n in g.get("nodes") or []:
            nid = str(n["id"])
            prev = nodes.get(nid)
            if prev is None:
                nodes[nid] = {
                    "id": nid,
                    "initial": float(n.get("initial", 0) or 0),
                    "sinks": n.get("sinks") or "none",
                }
            else:
                if float(n.get("initial", 0) or 0) and not prev["initial"]:
                    prev["initial"] = float(n["initial"])
                if n.get("sinks") == "unknown":
                    prev["sinks"] = "unknown"

        for e in g.get("edges") or []:
            frm, to = e["from"], e["to"]
            key = (frm, to)
            if key not in edges:
                edges[key] = e.get("id") or f"e_{frm}_{to}"
            id_map[e.get("id", "")] = edges[key]
            for nid in (frm, to):
                if nid not in nodes:
                    nodes[nid] = {"id": nid, "initial": 0.0, "sinks": "none"}

        for c in g.get("claims") or []:
            nc = dict(c)
            nc["id"] = f"c{cid}"
            if nc.get("type") == "edge" and nc.get("ref") in id_map:
                nc["ref"] = id_map[nc["ref"]]
            if nc.get("type") == "sink" and nc.get("ref") in nodes:
                nodes[nc["ref"]]["sinks"] = "unknown"
            if not nc.get("timestamp") and g.get("_timestamp"):
                nc["timestamp"] = g["_timestamp"]
            claims.append(nc)
            cid += 1

    out = {
        "nodes": list(nodes.values()),
        "edges": [{"id": eid, "from": a, "to": b} for (a, b), eid in edges.items()],
        "claims": claims,
    }
    if lambda_sink is not None:
        out["lambda_sink"] = lambda_sink
    return out


def ingest_paths(
    paths: list[str | Path],
    api_key: str | None = None,
    lambda_w: float = 1.0,
    cache_dir: str | Path | None = None,
    sinks: str | None = None,
    column_mapping: dict | None = None,
) -> tuple[dict | None, dict]:
    """Ingest one or many mixed files into a single compile graph.

    Sidecars are classified first so topology.json becomes a schema, Labels.csv
    is skipped, and SCADA rows bind Link_* sensors to real endpoints.
    """
    cache = Cache(cache_dir)
    paths = [Path(p) for p in paths]

    skipped: list[dict] = []
    schema_files: list[Path] = []
    graph_files: list[Path] = []
    jsonl_files: list[Path] = []
    tabular_files: list[Path] = []
    record_files: list[Path] = []

    for path in paths:
        reason = skip_reason(path)
        if reason:
            skipped.append({"file": path.name, "reason": reason})
            continue
        ext = path.suffix.lower()
        if ext == ".json":
            kind = classify_json(peek_json(path) or {})
            if kind == "TRUTH":
                skipped.append({"file": path.name, "reason": "truth/eval sidecar"})
            elif kind == "SCHEMA":
                schema_files.append(path)
            elif kind == "GRAPH":
                graph_files.append(path)
            else:
                record_files.append(path)
        elif ext == ".jsonl":
            jsonl_files.append(path)
        elif ext in _TABULAR_EXTS:
            tabular_files.append(path)
        else:
            record_files.append(path)

    topology: dict = {"links": {}, "nodes": []}
    for sp in schema_files:
        topo = load_topology(sp)
        topology.setdefault("links", {}).update(topo.get("links") or {})
        topology.setdefault("nodes", []).extend(topo.get("nodes") or [])
        if topo.get("pipes"):
            topology.setdefault("pipes", []).extend(topo["pipes"])

    schema = schema_from_topology(topology) if (topology.get("links") or topology.get("nodes")) else {}
    aliases = alias_map(topology) if topology.get("nodes") else {}

    has_scada_csv = False
    for tp in tabular_files:
        try:
            from ._ingest_tabular import _load_csv
            headers, _ = _load_csv(tp)
            if looks_like_scada_table(headers):
                has_scada_csv = True
                break
        except Exception:
            pass

    parts: list[dict] = []
    reports: list[dict] = []

    shared = dict(
        api_key=api_key, lambda_w=lambda_w, cache_dir=cache.dir,
        sinks=sinks, column_mapping=column_mapping,
        schema=schema or None, aliases=aliases or None, topology=topology or None,
    )

    for path in schema_files:
        gs, report = build_graph(path, **shared)
        # Schema contributes topology; observations come from SCADA / logs.
        # Keep the edges so flow claims have somewhere to sit even if the CSV
        # only names Link_* ids.
        parts.extend(gs)
        reports.append(report)

    for path in graph_files:
        gs, report = build_graph(path, **shared)
        parts.extend(gs)
        reports.append(report)

    for path in tabular_files:
        gs, report = build_graph(path, **shared)
        parts.extend(gs)
        reports.append(report)

    for path in jsonl_files:
        loaded = load_jsonl_rows(path)
        if loaded and has_scada_csv:
            skipped.append({
                "file": path.name,
                "reason": "duplicate of SCADA CSV (structured jsonl skipped)",
            })
            reports.append(_skipped(path, skipped[-1]["reason"])[1])
            continue
        gs, report = build_graph(path, **shared)
        parts.extend(gs)
        reports.append(report)

    for path in record_files:
        gs, report = build_graph(path, **shared)
        parts.extend(gs)
        reports.append(report)

    graph = merge_graphs(parts) if parts else None
    if graph and not (graph["nodes"] or graph["claims"] or graph["edges"]):
        graph = None

    modes = {r.get("mode") for r in reports if r.get("mode") != "SKIP"}
    combined = {
        "mode": "MIXED" if len(modes) > 1 else (next(iter(modes), "SKIP")),
        "source_files": [str(p) for p in paths],
        "files": reports,
        "skipped": skipped,
        "schema_links": len(schema),
        "llm_calls": sum(r.get("llm_calls", 0) for r in reports),
        "record_count": sum(r.get("record_count", 0) for r in reports),
        "mapping": reports[0].get("mapping") if len(reports) == 1 else None,
        "exclusions": [e for r in reports for e in (r.get("exclusions") or [])],
    }
    return graph, combined
