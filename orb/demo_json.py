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


def demo_json(path: str, truth_path: str | None = None, sinks: str = "none") -> dict:
    from .demo import run_demo

    # Suppress print output from run_demo
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_demo(path, truth_path=truth_path, sinks=sinks)

    if not result:
        return {"error": "No graphs produced"}

    return _clean(_result_to_dict(result))


def demo_json_multi(paths: list[str], sinks: str = "none") -> dict:
    from .demo import run_demo_multi

    with contextlib.redirect_stdout(io.StringIO()):
        result = run_demo_multi(paths, sinks=sinks)

    if not result:
        return {"error": "No graphs produced"}

    return _clean(_result_to_dict(result))


def _result_to_dict(result: dict) -> dict:
    return {
        "ingest_report": result["ingest_report"],
        "report": result["report"],
        "decoded": result["decoded"],
        "graph": {
            "nodes": result["graph"]["nodes"],
            "edges": result["graph"]["edges"],
            "claims": result["graph"]["claims"],
            "lambda_sink": result["graph"].get("lambda_sink"),
        },
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ORB demo → JSON")
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--sinks", default="none", choices=["none", "known", "unknown"],
                    help="sink mode (caller decision, never detected)")
    ap.add_argument("--preset", default=None,
                    help="run a demo preset from orb/presets.py (its own sinks mode applies)")
    args = ap.parse_args()
    if args.preset:
        from .presets import run_preset
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_preset(args.preset)
            out = _clean(_result_to_dict(result))
            out["preset"] = result["preset"]
        except Exception as e:  # noqa: BLE001
            out = {"error": f"{type(e).__name__}: {e}"}
        json.dump(out, sys.stdout, default=str)
        sys.exit(0)
    if not args.paths:
        print('{"error": "No file paths provided"}')
        sys.exit(1)
    try:
        if len(args.paths) == 1:
            out = demo_json(args.paths[0], sinks=args.sinks)
        else:
            out = demo_json_multi(args.paths, sinks=args.sinks)
    except Exception as e:  # noqa: BLE001
        out = {"error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout, default=str)
