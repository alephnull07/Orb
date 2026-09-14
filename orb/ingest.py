"""
orb/ingest.py
-------------
Public entry point: build_graph(path) → (list[graph_dict], report_dict)

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
) -> tuple[list[dict], dict]:
    """
    Convert *path* (any supported file) into one or more compile.py-format graphs.

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
        graphs, report = run_tabular_mode(path, api_key=api_key, lambda_w=lambda_w, cache=cache)
    else:
        graphs, report = run_record_mode(path, api_key=api_key, lambda_w=lambda_w, cache=cache)

    return graphs, report
