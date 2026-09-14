"""
orb/ingest.py
-------------
Public entry points:
  build_graph(path)            → (list[graph_dict], report_dict)  per-file
  build_graph_multi(paths)     → (graph_dict, merge_report)       merged

Routing (by extension, no LLM):
  .csv / .tsv / .xlsx  → TABULAR mode  (_ingest_tabular.run_tabular_mode)
  .txt / .log / .md    → RECORD  mode  (_ingest_record.run_record_mode)
  .jsonl               → peek first line; object → RECORD, plain text → RECORD

Cross-cutting concerns handled here:
  leakage_guard()   — regex-flag columns/channels before they enter extraction
  validate_claims() — reject any claim whose type is not in VALID_TYPES
  Cache             — shared disk cache passed to both mode modules
"""

from __future__ import annotations

from pathlib import Path

from ._cache import Cache
from ._ingest_record  import run_record_mode
from ._ingest_tabular import run_tabular_mode
from ._ingest_utils   import leakage_guard, validate_claims, VALID_TYPES  # re-export
from ._merge import merge_graphs


# ── routing ────────────────────────────────────────────────────────────────────

_TABULAR_EXTS = {".csv", ".tsv", ".xlsx"}
_RECORD_EXTS  = {".txt", ".log", ".md"}


def _route(path: Path) -> str:
    """Return 'TABULAR' or 'RECORD' for the given file."""
    ext = path.suffix.lower()
    if ext in _TABULAR_EXTS:
        return "TABULAR"
    if ext in _RECORD_EXTS:
        return "RECORD"
    if ext == ".jsonl":
        # Peek: if first non-empty line is a JSON object → RECORD
        return "RECORD"
    # Default to RECORD for unknown types
    return "RECORD"


# ── main entry point ───────────────────────────────────────────────────────────

def build_graph(
    path: str | Path,
    api_key: str | None = None,
    lambda_w: float = 1.0,
    cache_dir: str | Path | None = None,
    sinks: str | None = None,
    column_mapping: dict | None = None,
) -> tuple[list[dict], dict]:
    """
    Convert *path* (any supported file) into one or more compile.py-format graphs.

    Parameters
    ----------
    sinks : "unknown" | "none" | None
        Override the default sinks setting on nodes.  When "unknown", nodes
        that appear as claim targets get sinks="unknown" (leak search mode).
    column_mapping : dict | None
        Pre-supplied column mapping for TABULAR mode — skips the LLM call.

    Returns
    -------
    graphs : list[dict]
        One dict per timestamp bucket.  Each dict has {nodes, edges, claims}.
    report : dict
        Ingest report: mode, source_file, exclusions, timestamp_buckets, llm_calls, …
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    cache = Cache(cache_dir)
    mode  = _route(path)

    if mode == "TABULAR":
        graphs, report = run_tabular_mode(
            path, api_key=api_key, lambda_w=lambda_w, cache=cache,
            sinks=sinks, column_mapping=column_mapping,
        )
    else:
        graphs, report = run_record_mode(path, api_key=api_key, lambda_w=lambda_w, cache=cache)

    return graphs, report


def build_graph_multi(
    paths: list[str | Path],
    api_key: str | None = None,
    lambda_w: float = 1.0,
    cache_dir: str | Path | None = None,
    sinks: str | None = None,
    window_hours: float = 24.0,
    column_mapping: dict | None = None,
) -> tuple[dict, dict]:
    """
    Ingest multiple files, merge into ONE graph.

    Each file is ingested via build_graph (per-file routing unchanged).
    Then merge_graphs canonicalizes node names, sums edge events,
    deduplicates claims, and collapses timestamp buckets within *window_hours*.

    Returns (graph_dict, merge_report).
    """
    per_file: list[tuple[list[dict], dict]] = []
    for p in paths:
        graphs, report = build_graph(
            Path(p), api_key=api_key, lambda_w=lambda_w,
            cache_dir=cache_dir, sinks=sinks,
            column_mapping=column_mapping,
        )
        per_file.append((graphs, report))

    return merge_graphs(
        per_file,
        window_hours=window_hours,
        sinks=sinks or "none",
    )
