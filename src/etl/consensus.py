"""Smush three agent graphs into one majority/median view."""

from __future__ import annotations

from collections import defaultdict

from .textutil import is_site_name


def merge(agent_graphs: list[dict]) -> dict:
    node_votes: dict[str, list[dict]] = defaultdict(list)
    edge_votes: dict[tuple[str, str], list[dict]] = defaultdict(list)
    constraints: list[dict] = []

    for g in agent_graphs:
        agent = g["agent"]
        for n in g.get("nodes") or []:
            node_votes[n["key"]].append({"agent": agent, **n})
        for e in g.get("edges") or []:
            edge_votes[(e["source_key"], e["target_key"])].append({"agent": agent, **e})
        constraints.extend(g.get("trusted_constraint_rows") or [])

    n_agents = len(agent_graphs)
    majority = max(2, (n_agents + 1) // 2)  # 2 of 3

    nodes = []
    node_meta = {}
    for key, votes in node_votes.items():
        display_guess = max((v["display"] for v in votes), key=len)
        if not is_site_name(display_guess) or not is_site_name(key):
            continue
        if len({v["agent"] for v in votes}) < majority:
            continue
        display = max((v["display"] for v in votes), key=len)
        aliases = sorted({a for v in votes for a in v.get("aliases", [])})
        node = {
            "id": votes[0]["id"],
            "key": key,
            "display": display,
            "aliases": aliases,
            "votes": sorted({v["agent"] for v in votes}),
        }
        nodes.append(node)
        node_meta[key] = node

    kept_keys = set(node_meta)
    edges = []
    disagreements = []
    for (sk, dk), votes in edge_votes.items():
        agents = {v["agent"] for v in votes}

        if sk not in kept_keys or dk not in kept_keys:
            continue

        # Scout-only hops (ghost runner, TOC as a warehouse) must not ride in on evidence ids.
        if len(agents) < majority:
            disagreements.append(
                {
                    "reason": "edge_outvoted",
                    "source": sk,
                    "target": dk,
                    "agents": sorted(agents),
                    "values": [v["value_lb"] for v in votes],
                }
            )
            continue

        values = [v["value_lb"] for v in votes]
        smushed = _smush_values(values, votes)
        agent_vals = {v["agent"]: v["value_lb"] for v in votes}

        edges.append(
            {
                "source": node_meta[sk]["id"],
                "target": node_meta[dk]["id"],
                "source_display": node_meta[sk]["display"],
                "target_display": node_meta[dk]["display"],
                "value_lb": smushed["value"],
                "agent_values": agent_vals,
                "votes": sorted(agents),
                "method": smushed["method"],
                "evidence": sorted({mid for v in votes for mid in (v.get("evidence") or []) if mid}),
            }
        )

        if len(set(values)) > 1:
            disagreements.append(
                {
                    "reason": "sender_receiver_disagreement",
                    "source": sk,
                    "target": dk,
                    "agent_values": agent_vals,
                    "kept": smushed["value"],
                    "delta": max(values) - min(values),
                }
            )

    type_map = _infer_types(nodes, edges)
    for n in nodes:
        n["type"] = type_map.get(n["id"], "unknown")

    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    for e in edges:
        incoming[e["target"]] += e["value_lb"]
        outgoing[e["source"]] += e["value_lb"]

    eod_by_display = {}
    eod_by_node = {}
    for row in constraints:
        if row.get("display"):
            eod_by_display[row["display"].strip().lower()] = row
        if row.get("node"):
            eod_by_node[row["node"]] = row

    trusted = []
    for n in nodes:
        eod = eod_by_node.get(n["id"]) or eod_by_display.get(n["display"].strip().lower())
        if not eod:
            # Do not invent closeout = in − out. That fabricates a third
            # channel and makes a coordinated send/receive lie look like EOD.
            continue
        inv = eod.get("inventory_eod_lb")
        if inv is None:
            continue
        trusted.append(
            {
                "node": n["id"],
                "display": n["display"],
                "law": "mass_balance",
                "in_lb": eod.get("in_lb"),
                "out_lb": eod.get("out_lb"),
                "inventory_eod_lb": inv,
                "source": "auditor_eod",
            }
        )

    opening_by_id: dict[str, float] = {}
    for g in agent_graphs:
        for o in g.get("openings") or []:
            nid = o.get("node")
            try:
                val = float(o.get("value"))
            except (TypeError, ValueError):
                continue
            if nid:
                opening_by_id[nid] = val
    for n in nodes:
        if n["id"] in opening_by_id:
            n["initial"] = opening_by_id[n["id"]]

    return {
        "scenario": "supply_drops_people_as_sensors",
        "consensus_rule": "keep node/edge if >=2 of 3 agents agree; drop TOC/person sites; if a hop has a corrupted field report, keep that reported weight",
        "agents": [g["agent"] for g in agent_graphs],
        "nodes": nodes,
        "edges": edges,
        "trusted_constraint_rows": trusted,
        "disagreements": disagreements,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "outvoted_edges": sum(1 for d in disagreements if d["reason"] == "edge_outvoted"),
            "discrepancies": sum(1 for d in disagreements if d["reason"] == "sender_receiver_disagreement"),
        },
        "by_id": {n["id"]: n["display"] for n in nodes},
    }


def _smush_values(values: list[int], votes: list[dict] | None = None) -> dict:
    if len(values) == 1:
        return {"value": values[0], "method": "single", "kept": values, "dropped": None}

    flagged = [v["value_lb"] for v in (votes or []) if v.get("from_corrupted")]
    if flagged:
        # The rewritten field report is the observation, up or down.
        kept = flagged[-1]
        return {
            "value": kept,
            "method": "corrupted_report",
            "kept": [kept],
            "dropped": [x for x in values if x != kept],
        }

    if len(set(values)) == 1:
        return {"value": values[0], "method": "unanimous", "kept": values, "dropped": None}

    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    majority_val = max(counts, key=lambda x: (counts[x], -x))
    return {
        "value": majority_val,
        "method": "majority",
        "kept": [majority_val],
        "dropped": [x for x in values if x != majority_val],
    }


def _infer_types(nodes: list[dict], edges: list[dict]) -> dict[str, str]:
    inn = defaultdict(int)
    out = defaultdict(int)
    for e in edges:
        inn[e["target"]] += 1
        out[e["source"]] += 1
    types = {}
    for n in nodes:
        nid = n["id"]
        display = n["display"].lower()
        if any(w in display for w in ("airdrop", "convoy origin", "fob", "railhead", "port dump")):
            types[nid] = "source"
        elif out[nid] and not inn[nid]:
            types[nid] = "source"
        elif inn[nid] and not out[nid]:
            types[nid] = "sink"
        elif "hub" in display or "junction" in display or "cache" in display or "yard" in display:
            types[nid] = "hub"
        elif inn[nid] and out[nid]:
            types[nid] = "junction"
        else:
            types[nid] = "unknown"
    return types
