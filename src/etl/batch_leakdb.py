#!/usr/bin/env python3
"""Import official LeakDB Hanoi_CMH scenarios and/or run the ETL agents on them."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from .run import ROOT, run_sources

GEN_PATH = ROOT / "data" / "leakdb" / "import_scenarios.py"
RUNS_DIR = ROOT / "data" / "leakdb" / "runs"


def _load():
    spec = importlib.util.spec_from_file_location("leakdb_import_scenarios", GEN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GEN_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compact(score: dict | None) -> dict | None:
    if not score:
        return None
    return {
        "matched_hops": score.get("matched_hops"),
        "target_edges": score.get("target_edges"),
        "consensus_edges": score.get("consensus_edges"),
        "exact_weight_matches": score.get("exact_weight_matches"),
        "mae_lb": score.get("mean_abs_weight_error_on_matches"),
        "n_differences": len(score.get("differences") or []),
        "n_extra_hops": len(score.get("extra_hops") or []),
    }


def score_runs(n: int = 10, use_llm: bool = False) -> list[dict]:
    existing = []
    manifest_path = RUNS_DIR / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {row.get("run_id"): row for row in existing}
    need_topo = any(
        not (RUNS_DIR / f"run_{i:02d}" / "true" / "topology.json").exists()
        for i in range(1, n + 1)
        if (RUNS_DIR / f"run_{i:02d}" / "true" / "messages.jsonl").exists()
    )
    if need_topo:
        _load().attach_topologies(n=n)
    summary = []
    for i in range(1, n + 1):
        run_id = f"run_{i:02d}"
        out_root = RUNS_DIR / run_id
        if not (out_root / "true" / "messages.jsonl").exists():
            print(f"skip {run_id}: no imported corpus")
            continue
        print(f"\n======== LeakDB {run_id} ========")
        scores = run_sources(data_root=out_root, source="both", use_llm=use_llm)
        row = dict(by_id.get(run_id) or {"run_id": run_id, "out_root": str(out_root)})
        row["true_vs_truth"] = _compact(scores.get("true_vs_truth"))
        row["corrupted_vs_corrupted"] = _compact(scores.get("corrupted_vs_corrupted"))
        row["corrupted_vs_truth"] = _compact(scores.get("corrupted_vs_truth"))
        summary.append(row)
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote scores for {len(summary)} runs + {manifest_path}")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Import official LeakDB scenarios and/or run ETL agents")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--import-only", action="store_true")
    p.add_argument("--score-only", action="store_true", help="Run agents on already-imported runs")
    p.add_argument("--llm", action="store_true")
    args = p.parse_args()
    if not args.score_only:
        _load().import_all(n=args.n)
    if not args.import_only:
        score_runs(n=args.n, use_llm=args.llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
