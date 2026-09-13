#!/usr/bin/env python3
"""Run the three ETL agents (Scout, Receiver, Auditor) in parallel, then write a consensus graph.

Uses Anthropic Claude agents by default when an API key is present.

Example:
  python -m src.etl.run --source true
  python -m src.etl.run --source corrupted
  python -m src.etl.run --source both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import agent_auditor, agent_receiver, agent_scout
from .consensus import merge
from .llm import extract_llm, get_api_key
from .textutil import load_messages, load_topology

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "supply_drops"
OUT = DATA / "graphs"

AGENTS = {
    "scout": agent_scout.extract,
    "receiver": agent_receiver.extract,
    "auditor": agent_auditor.extract,
}


def run_dataset(
    name: str,
    use_llm: bool = True,
    model: str | None = None,
    api_key: str | None = None,
    data_root: Path | None = None,
) -> dict:
    data_root = Path(data_root) if data_root else DATA
    dataset_dir = data_root / name
    out_dir = data_root / "graphs"
    messages = load_messages(dataset_dir)
    topology = load_topology(dataset_dir)
    graphs = {}

    def job(agent_name: str):
        # Base regex extraction as baseline / fallback
        regex_graph = AGENTS[agent_name](messages, topology=topology)
        if use_llm:
            llm_graph = extract_llm(
                agent_name,
                messages,
                model=model,
                api_key=api_key,
                topology=topology,
            )
            if llm_graph:
                # Scout regex is the topology gate so Claude cannot add Watchtower / person hops.
                if agent_name == "scout":
                    return agent_name, _union_agent(regex_graph, llm_graph, allow_new_edges=False)
                return agent_name, _union_agent(llm_graph, regex_graph)
        return agent_name, regex_graph

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(job, agent) for agent in AGENTS]
        for fut in as_completed(futs):
            agent_name, graph = fut.result()
            graphs[agent_name] = graph
            print(f"  {name}/{agent_name}: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

    ordered = [graphs["scout"], graphs["receiver"], graphs["auditor"]]
    consensus = merge(ordered)
    print(
        f"  {name}/consensus: {consensus['stats']['nodes']} nodes, "
        f"{consensus['stats']['edges']} edges, "
        f"{consensus['stats']['discrepancies']} discrepancies logged"
    )

    dest = out_dir / name
    dest.mkdir(parents=True, exist_ok=True)
    for g in ordered:
        (dest / f"agent_{g['agent']}.json").write_text(
            json.dumps(g, indent=2) + "\n", encoding="utf-8"
        )
    (dest / "consensus.json").write_text(json.dumps(consensus, indent=2) + "\n", encoding="utf-8")
    (dest / "graph.json").write_text(json.dumps(consensus, indent=2) + "\n", encoding="utf-8")
    return consensus


def _union_agent(primary: dict, secondary: dict, allow_new_edges: bool = True) -> dict:
    nodes = {n["key"]: dict(n) for n in primary.get("nodes") or []}
    for n in secondary.get("nodes") or []:
        if n["key"] not in nodes:
            nodes[n["key"]] = dict(n)
        else:
            # Merge aliases
            existing_aliases = set(nodes[n["key"]].get("aliases", []))
            existing_aliases.update(n.get("aliases", []))
            nodes[n["key"]]["aliases"] = sorted(existing_aliases)

    edges = {(e["source_key"], e["target_key"]): dict(e) for e in primary.get("edges") or []}
    for e in secondary.get("edges") or []:
        k = (e["source_key"], e["target_key"])
        if k not in edges:
            if allow_new_edges:
                edges[k] = dict(e)
        else:
            # Keep primary weight; merge evidence only.
            ev = list(edges[k].get("evidence") or [])
            for mid in e.get("evidence") or []:
                if mid not in ev:
                    ev.append(mid)
            edges[k]["evidence"] = ev

    constraints = list(primary.get("trusted_constraint_rows") or [])
    seen_nodes = {c["node"] for c in constraints}
    for c in secondary.get("trusted_constraint_rows") or []:
        if c["node"] not in seen_nodes:
            constraints.append(c)
            seen_nodes.add(c["node"])

    out = dict(primary)
    out["nodes"] = list(nodes.values())
    out["edges"] = list(edges.values())
    out["trusted_constraint_rows"] = constraints
    return out


def score_against_eval(
    consensus: dict,
    target_filename: str,
    data_root: Path | None = None,
) -> dict | None:
    data_root = Path(data_root) if data_root else DATA
    eval_path = data_root / "eval" / target_filename
    if not eval_path.exists():
        return None
    target = json.loads(eval_path.read_text(encoding="utf-8"))
    t_edges = {}
    for e in target["edges"]:
        t_edges[(e["source_name"].lower().strip(), e["target_name"].lower().strip())] = e["value_lb"]

    c_edges = {
        (e["source_display"].lower().strip(), e["target_display"].lower().strip()): e["value_lb"]
        for e in consensus["edges"]
    }

    matched = 0
    weight_err = []
    diffs = []
    for k, v in t_edges.items():
        if k in c_edges:
            matched += 1
            err = abs(c_edges[k] - v)
            weight_err.append(err)
            if err > 0:
                diffs.append({"hop": list(k), "target_lb": v, "consensus_lb": c_edges[k], "delta": c_edges[k] - v})
        else:
            diffs.append({"hop": list(k), "missing": True, "target_lb": v})

    extra_hops = [list(k) for k in c_edges if k not in t_edges]
    return {
        "target_file": target_filename,
        "target_edges": len(t_edges),
        "consensus_edges": len(c_edges),
        "matched_hops": matched,
        "exact_weight_matches": sum(1 for err in weight_err if err == 0),
        "mean_abs_weight_error_on_matches": (
            round(sum(weight_err) / len(weight_err), 2) if weight_err else 0.0
        ),
        "differences": diffs,
        "extra_hops": extra_hops,
    }


def run_sources(
    data_root: Path,
    source: str = "both",
    use_llm: bool = False,
    model: str | None = None,
    api_key: str | None = None,
) -> dict:
    data_root = Path(data_root)
    names = ["true", "corrupted"] if source == "both" else [source]
    scores: dict = {}
    for name in names:
        print(f"\nProcessing dataset: {name}")
        consensus = run_dataset(
            name,
            use_llm=use_llm,
            model=model,
            api_key=api_key,
            data_root=data_root,
        )
        dest = data_root / "graphs" / name
        if name == "true":
            score_truth = score_against_eval(consensus, "true_graph.json", data_root=data_root)
            if score_truth:
                print(
                    f"  vs eval/true_graph.json: matched {score_truth['matched_hops']}/"
                    f"{score_truth['target_edges']} hops, MAE = "
                    f"{score_truth['mean_abs_weight_error_on_matches']} lb"
                )
                (dest / "score_vs_truth.json").write_text(
                    json.dumps(score_truth, indent=2) + "\n", encoding="utf-8"
                )
            scores["true_vs_truth"] = score_truth
        elif name == "corrupted":
            score_corrupt = score_against_eval(consensus, "corrupted_graph.json", data_root=data_root)
            if score_corrupt:
                print(
                    f"  vs eval/corrupted_graph.json: matched {score_corrupt['matched_hops']}/"
                    f"{score_corrupt['target_edges']} hops, MAE = "
                    f"{score_corrupt['mean_abs_weight_error_on_matches']} lb"
                )
                (dest / "score_vs_corrupted.json").write_text(
                    json.dumps(score_corrupt, indent=2) + "\n", encoding="utf-8"
                )
            score_truth = score_against_eval(consensus, "true_graph.json", data_root=data_root)
            if score_truth:
                print(
                    f"  vs eval/true_graph.json (hidden truth): matched {score_truth['matched_hops']}/"
                    f"{score_truth['target_edges']} hops, MAE = "
                    f"{score_truth['mean_abs_weight_error_on_matches']} lb"
                )
                (dest / "score_vs_truth.json").write_text(
                    json.dumps(score_truth, indent=2) + "\n", encoding="utf-8"
                )
            scores["corrupted_vs_corrupted"] = score_corrupt
            scores["corrupted_vs_truth"] = score_truth
    print(f"\nAll graphs successfully written to {data_root / 'graphs'}")
    return scores


def main() -> int:
    p = argparse.ArgumentParser(description="3-agent Claude ETL → consensus supply graph")
    p.add_argument("--source", choices=["true", "corrupted", "both"], default="both")
    p.add_argument("--no-llm", action="store_true", help="Disable LLM; use only regex personas")
    p.add_argument("--model", default=None, help="Claude model name (default: claude-haiku-4-5-20251001)")
    p.add_argument("--api-key", default=None, help="Anthropic API key")
    p.add_argument(
        "--data-root",
        type=Path,
        default=DATA,
        help="Corpus root containing true/, corrupted/, eval/ (default: data/supply_drops).",
    )
    args = p.parse_args()

    use_llm = not args.no_llm
    if use_llm:
        key = get_api_key(args.api_key)
        if not key:
            print("[warn] No Anthropic API key found. Falling back to regex-only extraction.", file=sys.stderr)
            use_llm = False
        else:
            print(f"[info] Using Claude model: {args.model or 'claude-haiku-4-5-20251001'}")

    run_sources(
        data_root=args.data_root,
        source=args.source,
        use_llm=use_llm,
        model=args.model,
        api_key=args.api_key,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
