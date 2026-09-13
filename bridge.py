"""
bridge.py
---------
Connects the reader pipeline (friend's ETL agents) to the L1 estimator.

The reader outputs graph.json (consensus of 3 agents).
This script converts that into a flow estimation problem and runs L1/LS on it.

UNKNOWNS: flow on each directed edge (lb)
  - consensus graph has 35 edges (including 1 ghost edge)
  - true graph has 34 edges

CLAIMS (rows of H):
  1. Edge flow claim — consensus reports V lb on edge (F, T)
     H row: +1 for edge column. y = V. Confidence from voting method.
  2. Node conservation claim — auditor EOD report says: inflow - outflow = inventory
     H row: +1 for entering edges, -1 for leaving edges. y = inventory - external_injection.
     These are independent observations, not derived from the flow claims.

The ghost edge and corrupted flows violate conservation.
L1 finds the flow assignment that satisfies the most claims and flags the rest.
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add parent so we can import estimator.py
sys.path.insert(0, str(Path(__file__).parent))
from estimator import least_squares, l1_estimate, flag_claims


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_flow_problem(consensus_graph, use_true_constraints=None):
    """
    Build H matrix and y vector from consensus graph.

    Parameters
    ----------
    consensus_graph : dict
        Output of consensus.merge() — nodes, edges, trusted_constraint_rows.
    use_true_constraints : dict or None
        If provided, use true_graph's inventory_eod_lb as conservation targets.
        (For ablation: shows what perfect conservation constraints would give.)
        If None, use the auditor's EOD reports from the consensus graph.

    Returns
    -------
    H, y, claims, edges
    """
    edges = consensus_graph["edges"]
    nodes = consensus_graph["nodes"]
    n_edges = len(edges)

    # Edge index: (source_id, target_id) -> column index
    edge_idx = {(e["source"], e["target"]): i for i, e in enumerate(edges)}

    # Node -> entering/leaving edge indices
    node_entering = {n["id"]: [] for n in nodes}
    node_leaving   = {n["id"]: [] for n in nodes}
    for i, e in enumerate(edges):
        node_leaving[e["source"]].append(i)
        if e["target"] in node_entering:
            node_entering[e["target"]].append(i)

    H_rows = []
    y_vals = []
    claims = []

    # ------------------------------------------------------------------ #
    # CLAIM TYPE 1: edge flow claims from consensus                        #
    # ------------------------------------------------------------------ #
    method_conf = {
        "unanimous":        1.0,
        "majority":         0.7,
        "corrupted_report": 0.5,
        "single":           0.4,
    }
    for i, e in enumerate(edges):
        conf = method_conf.get(e.get("method", "single"), 0.7)
        row = np.zeros(n_edges)
        row[i] = 1.0
        H_rows.append(row)
        y_vals.append(float(e["value_lb"]))
        claims.append({
            "type":       "flow",
            "source":     e["source"],
            "target":     e["target"],
            "amount":     e["value_lb"],
            "confidence": conf,
            "method":     e.get("method", "?"),
            "source_str": f"flow_{e['source']}_to_{e['target']}",
        })

    # ------------------------------------------------------------------ #
    # CLAIM TYPE 2: node inflow + outflow claims (from auditor EOD)       #
    # ------------------------------------------------------------------ #
    # The auditor independently counts total arrivals (in_lb) and total
    # departures (out_lb) at each node. These are SEPARATE observations
    # from the individual transfer claims — a different person counted.
    #
    # Adding them as two separate claims gives L1 a direct contradiction
    # when a corrupted flow claim disagrees with what the auditor counted.
    #
    # Example: driver says 225 lb left LIMA for NOVEMBER.
    #          Auditor at NOVEMBER says 200 lb arrived total.
    #          L1 sees two claims constraining f[LIMA→NOVEMBER]:
    #            flow claim: f[LIMA→NOV] = 225 (conf 1.0, unanimous)
    #            auditor inflow claim: f[LIMA→NOV] = 200 (conf 1.2, independent)
    #          L1 favors the auditor (higher weight) → recovers 200.
    #
    # Source weights:
    #   "auditor_eod"    → conf 1.2 (independent physical count, higher than
    #                       any single agent's transfer claim at 1.0)
    #   "consensus_flow" → conf 1.0 for sources (source manifest, trusted)

    node_type_map = {n["id"]: n.get("type", "unknown") for n in nodes}

    if use_true_constraints:
        constraint_map = {row["node"]: {**row, "source": "true_ground_truth"}
                          for row in use_true_constraints}
    else:
        constraint_map = {row["node"]: row
                          for row in consensus_graph.get("trusted_constraint_rows", [])}

    for node_id in [n["id"] for n in nodes]:
        if node_id not in constraint_map:
            continue
        tc = constraint_map[node_id]
        src = tc.get("source", "consensus_flow")
        is_source = node_type_map.get(node_id) == "source"

        if is_source:
            # Source nodes: their "outflow" is computed by consensus_flow (just
            # sums the corrupted transfer claims). This is NOT independent —
            # adding it as a constraint would just reinforce the corruption.
            # Skip it. The downstream auditor_eod inflow claims provide the
            # independent check instead.
            pass
        else:
            # Non-source nodes: auditor counted total arrivals and departures.
            # These are independent of the individual transfer claims.
            conf = 1.2 if src == "auditor_eod" else 0.7

            # Inflow claim: Σ(entering edges) = in_lb
            if node_entering[node_id]:
                in_row = np.zeros(n_edges)
                for ei in node_entering[node_id]:
                    in_row[ei] = 1.0
                H_rows.append(in_row)
                y_vals.append(float(tc["in_lb"]))
                claims.append({
                    "type": "inflow", "node": node_id,
                    "amount": tc["in_lb"], "confidence": conf,
                    "source_str": f"auditor_inflow_{node_id}", "cons_source": src,
                })

            # Outflow claim: Σ(leaving edges) = out_lb
            if node_leaving[node_id]:
                out_row = np.zeros(n_edges)
                for ei in node_leaving[node_id]:
                    out_row[ei] = 1.0
                H_rows.append(out_row)
                y_vals.append(float(tc["out_lb"]))
                claims.append({
                    "type": "outflow", "node": node_id,
                    "amount": tc["out_lb"], "confidence": conf,
                    "source_str": f"auditor_outflow_{node_id}", "cons_source": src,
                })

    H = np.array(H_rows)
    y = np.array(y_vals)
    return H, y, claims, edges


def evaluate(recovered_flows, edges, true_graph):
    """Compare recovered flows to true flows. Returns per-edge errors."""
    # Build true flow map by display name pair (since IDs differ between graphs)
    true_by_display = {}
    for e in true_graph["edges"]:
        key = (e["source_name"].lower().strip(), e["target_name"].lower().strip())
        true_by_display[key] = e["value_lb"]

    results = []
    for i, e in enumerate(edges):
        key = (e["source_display"].lower().strip(), e["target_display"].lower().strip())
        true_val = true_by_display.get(key)
        est_val  = recovered_flows[i]
        results.append({
            "source":    e["source"],
            "target":    e["target"],
            "src_disp":  e["source_display"],
            "tgt_disp":  e["target_display"],
            "estimated": round(est_val, 1),
            "true":      true_val,
            "error":     round(abs(est_val - true_val), 1) if true_val is not None else None,
            "ghost":     true_val is None,
        })
    return results


def print_results(label, flow_results, flagged_claims, claims, graph_edges, corruption_labels=None):
    # Build corrupted edge set matched by display name
    # (corruption_labels uses true_graph IDs like HUB_NOVEMBER;
    #  consensus graph uses its own IDs like NOVEMBER_RIDGE_HUB)
    corrupt_display_pairs = set()
    if corruption_labels:
        for atk in corruption_labels.get("attacks", []):
            sn = atk.get("source_name", "").lower().strip()
            tn = atk.get("target_name", "").lower().strip()
            if sn and tn:
                corrupt_display_pairs.add((sn, tn))

    corrupt_edges = set()
    for e in graph_edges:
        key = (e["source_display"].lower().strip(), e["target_display"].lower().strip())
        if key in corrupt_display_pairs:
            corrupt_edges.add((e["source"], e["target"]))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    errors = [r["error"] for r in flow_results if r["error"] is not None]
    ghost_flows = [r for r in flow_results if r["ghost"]]

    print(f"\n  Flow recovery (true vs estimated):")
    print(f"  {'Edge':<45} {'True':>6} {'Est':>6} {'Err':>6} {'Flag':>5}")
    flagged_edge_pairs = set()
    for fc in flagged_claims:
        c = claims[fc["index"]]
        if c["type"] == "flow":
            flagged_edge_pairs.add((c["source"], c["target"]))

    for r in flow_results:
        true_str = f"{r['true']}" if r["true"] is not None else "GHOST"
        err_str  = f"{r['error']}" if r["error"] is not None else "?"
        flagged  = "FLAG" if (r["source"], r["target"]) in flagged_edge_pairs else ""
        corrupt  = " <LIE>" if (r["source"], r["target"]) in corrupt_edges else ""
        print(f"  {r['src_disp'][:20]:<20} -> {r['tgt_disp'][:20]:<20} "
              f"{true_str:>6} {r['estimated']:>6} {err_str:>6} {flagged:>5}{corrupt}")

    print(f"\n  Summary:")
    if errors:
        print(f"    Max flow error:  {max(errors):.1f} lb")
        print(f"    Mean flow error: {sum(errors)/len(errors):.1f} lb")
        print(f"    Exact matches:   {sum(1 for e in errors if e < 2)} / {len(errors)}")
    if ghost_flows:
        print(f"    Ghost edges recovered: {len(ghost_flows)}")
        for r in ghost_flows:
            flagged = "FLAGGED" if (r["source"], r["target"]) in flagged_edge_pairs else "NOT FLAGGED"
            print(f"      {r['src_disp']} -> {r['tgt_disp']}: est={r['estimated']} ({flagged})")

    # Corruption detection
    if corrupt_edges:
        true_pos = sum(1 for s, t in corrupt_edges if (s, t) in flagged_edge_pairs)
        false_pos = len(flagged_edge_pairs - corrupt_edges - {(r["source"], r["target"]) for r in ghost_flows})
        print(f"\n  Corruption detection:")
        print(f"    Corrupted edges:     {len(corrupt_edges)}")
        print(f"    Flagged correctly:   {true_pos} / {len(corrupt_edges)}")
        print(f"    False flags:         {false_pos}")


def save_l1_graph(recovered_flows, edges, nodes, out_path):
    """
    Write L1-recovered flows back to a graph JSON that the viz can load.
    Same schema as consensus.json: nodes[] + edges[] with value_lb.
    """
    out_edges = []
    for i, e in enumerate(edges):
        val = round(float(recovered_flows[i]), 1)
        if abs(val) < 1.0:   # ghost edges suppressed to ~0 — drop them
            continue
        out_edges.append({
            "source":         e["source"],
            "target":         e["target"],
            "source_display": e.get("source_display", e["source"]),
            "target_display": e.get("target_display", e["target"]),
            "value_lb":       val,
            "method":         "l1_recovered",
        })
    out = {
        "scenario": "l1_recovered",
        "nodes":    nodes,
        "edges":    out_edges,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  L1 graph saved → {out_path}")


def main(run_dir=None):
    if run_dir:
        base = Path(run_dir)
    else:
        base = Path("data/supply_drops")

    corrupted_graph  = load_json(base / "graphs/corrupted/graph.json")
    true_graph       = load_json(base / "eval/true_graph.json")
    corruption_labels = load_json(base / "eval/corruption_labels.json")

    print("\n" + "="*60)
    print("  ORB — READER PIPELINE + L1 INTEGRATION")
    print(f"  {len(corrupted_graph['edges'])} consensus edges, "
          f"{len(true_graph['edges'])} true edges")
    print("="*60)

    # Baseline: what does consensus give us (no L1)?
    consensus_score = load_json(base / "graphs/corrupted/score_vs_truth.json")
    print(f"\n  CONSENSUS BASELINE (voting only, no conservation):")
    print(f"    Edges matched:     {consensus_score['matched_hops']} / {consensus_score['target_edges']}")
    print(f"    Exact matches:     {consensus_score['exact_weight_matches']}")
    print(f"    Mean weight error: {consensus_score['mean_abs_weight_error_on_matches']:.2f} lb")
    print(f"    Ghost edges:       {len(consensus_score['extra_hops'])}")

    # Build the L1 problem
    weights = None  # placeholder for confidence weighting (set below)

    H, y, claims, edges = build_flow_problem(corrupted_graph)
    weights = np.array([c["confidence"] for c in claims])

    print(f"\n  H shape: {H.shape} ({H.shape[0]} claims, {H.shape[1]} edges)")
    print(f"  Rank of H: {np.linalg.matrix_rank(H)}")
    print(f"  Flow claims: {sum(1 for c in claims if c['type'] == 'flow')}")
    print(f"  Conservation claims: {sum(1 for c in claims if c['type'] == 'conservation')}")

    # --- Least squares ---
    x_ls, res_ls = least_squares(H, y)
    flagged_ls = flag_claims(claims, res_ls, threshold=8.0)
    flow_results_ls = evaluate(x_ls, edges, true_graph)
    print_results("LEAST SQUARES", flow_results_ls, flagged_ls, claims, edges, corruption_labels)

    # --- L1 with confidence weights ---
    x_l1, res_l1 = l1_estimate(H, y, weights=weights)
    if x_l1 is not None:
        flagged_l1 = flag_claims(claims, res_l1, threshold=8.0)
        flow_results_l1 = evaluate(x_l1, edges, true_graph)
        print_results("L1 (confidence-weighted)", flow_results_l1, flagged_l1, claims, edges, corruption_labels)
        save_l1_graph(x_l1, edges, corrupted_graph["nodes"],
                      base / "graphs/corrupted/l1_recovered.json")
    else:
        print("\nL1: solver failed")

    # --- Summary comparison ---
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    methods = [
        ("Consensus (voting)", consensus_score["mean_abs_weight_error_on_matches"],
         len(consensus_score["extra_hops"])),
    ]
    for label, flow_results in [("Least Squares", flow_results_ls),
                                  ("L1 estimator",  flow_results_l1 if x_l1 is not None else [])]:
        errors = [r["error"] for r in flow_results if r["error"] is not None]
        ghosts_flagged = sum(
            1 for r in flow_results
            if r["ghost"] and r["estimated"] < 5.0  # ghost edge pushed near 0
        )
        mean_err = sum(errors)/len(errors) if errors else float('inf')
        methods.append((label, mean_err, ghosts_flagged))

    print(f"\n  {'Method':<28} {'MeanErr':>9} {'GhostsSuppressed':>18}")
    print(f"  {'-'*55}")
    for label, mean_err, ghosts in methods:
        print(f"  {label:<28} {mean_err:>9.2f} {ghosts:>18}")


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else None
    main(run_dir)
