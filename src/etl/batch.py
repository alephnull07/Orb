#!/usr/bin/env python3
"""Generate extra supply-drop corpora and run the three ETL agents on each.

Leaves the canonical data/supply_drops/{true,corrupted,eval} demo untouched.
New runs land in data/supply_drops/runs/run_01 ... run_10.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from .run import DATA, ROOT, run_sources

GEN_PATH = ROOT / "data" / "supply_drops" / "generate_corpus.py"
RUNS_DIR = DATA / "runs"
SEEDS = [8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108, 8109, 8110]


def _load_generator():
    spec = importlib.util.spec_from_file_location("supply_drops_generate_corpus", GEN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generator at {GEN_PATH}")
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
    summary: list[dict] = []
    for i in range(1, n + 1):
        run_id = f"run_{i:02d}"
        out_root = RUNS_DIR / run_id
        if not (out_root / "true" / "messages.jsonl").exists():
            print(f"skip {run_id}: no generated corpus")
            continue
        print(f"\n======== {run_id} ========")
        scores = run_sources(data_root=out_root, source="both", use_llm=use_llm)
        row = dict(by_id.get(run_id) or {"run_id": run_id, "out_root": str(out_root)})
        row["true_vs_truth"] = _compact(scores.get("true_vs_truth"))
        row["corrupted_vs_corrupted"] = _compact(scores.get("corrupted_vs_corrupted"))
        row["corrupted_vs_truth"] = _compact(scores.get("corrupted_vs_truth"))
        summary.append(row)
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote scores for {len(summary)} runs + {manifest_path}")
    return summary


def generate_and_run(
    n: int = 10,
    seeds: list[int] | None = None,
    fallback: bool = True,
    use_llm: bool = False,
) -> list[dict]:
    gen = _load_generator()
    chosen = (seeds or SEEDS)[:n]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    for i, seed in enumerate(chosen, start=1):
        run_id = f"run_{i:02d}"
        out_root = RUNS_DIR / run_id
        print(f"\n======== {run_id} seed={seed} ========")
        stats = gen.generate(out_root=out_root, seed=seed, fallback=fallback)
        print(
            f"  generated nodes={stats['n_nodes']} edges={stats['n_edges']} "
            f"msgs={stats['true_messages']}/{stats['corrupted_messages']}"
        )
        scores = run_sources(data_root=out_root, source="both", use_llm=use_llm)
        row = {
            "run_id": run_id,
            "seed": seed,
            "out_root": str(out_root),
            **stats,
            "true_vs_truth": _compact(scores.get("true_vs_truth")),
            "corrupted_vs_corrupted": _compact(scores.get("corrupted_vs_corrupted")),
            "corrupted_vs_truth": _compact(scores.get("corrupted_vs_truth")),
        }
        summary.append(row)
    manifest = RUNS_DIR / "manifest.json"
    manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(summary)} runs + {manifest}")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Generate extra corpora and run ETL agents")
    p.add_argument("--n", type=int, default=10, help="How many extra datasets to generate")
    p.add_argument("--llm", action="store_true", help="Use Claude if an API key is present")
    p.add_argument("--score-only", action="store_true", help="Run agents on already-generated runs")
    args = p.parse_args()
    if args.score_only:
        score_runs(n=args.n, use_llm=args.llm)
    else:
        generate_and_run(n=args.n, fallback=True, use_llm=args.llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
