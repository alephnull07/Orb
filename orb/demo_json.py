"""JSON output for the frontend — calls run_demo with stdout suppressed."""
from __future__ import annotations

import contextlib
import io
import json
import sys


def _clean(obj):
    """Convert numpy types to native Python for JSON serialization."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def demo_json(path: str, truth_path: str | None = None) -> dict:
    from .demo import run_demo

    # Suppress print output from run_demo
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_demo(path, truth_path=truth_path)

    return _clean(_result_or_error(result))


def demo_json_multi(paths: list[str]) -> dict:
    from .demo import run_demo_multi

    with contextlib.redirect_stdout(io.StringIO()):
        result = run_demo_multi(paths)

    return _clean(_result_or_error(result))


def _result_or_error(result: dict | None) -> dict:
    result = result or {}
    if not result.get("graph"):
        skipped = (result.get("ingest_report") or {}).get("skipped") or []
        extra = ""
        if skipped:
            extra = " Skipped: " + "; ".join(
                f"{s.get('file')} ({s.get('reason')})" for s in skipped
            )
        return {
            "error": "No graphs produced." + extra,
            "ingest_report": result.get("ingest_report"),
        }
    return _result_to_dict(result)


def _result_to_dict(result: dict) -> dict:
    return {
        "ingest_report": result["ingest_report"],
        "report": result["report"],
        "decoded": result["decoded"],
        "advice": result.get("advice") or {},
        "graph": {
            "nodes": result["graph"]["nodes"],
            "edges": result["graph"]["edges"],
            "claims": result["graph"]["claims"],
            "lambda_sink": result["graph"].get("lambda_sink"),
        },
    }


if __name__ == "__main__":
    import traceback

    paths = sys.argv[1:]
    if not paths:
        print('{"error": "No file paths provided"}')
        sys.exit(1)

    try:
        if len(paths) == 1:
            out = demo_json(paths[0])
        else:
            out = demo_json_multi(paths)
        json.dump(out, sys.stdout, default=str)
        if out.get("error"):
            print(out["error"], file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(msg, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        json.dump({"error": msg}, sys.stdout)
        sys.exit(1)
