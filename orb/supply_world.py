"""
orb/supply_world.py
-------------------
Generate a synthetic supply-chain network with planted corruption,
producing compile()-ready graph dicts directly (no LLM, no file I/O).

Public API
----------
generate_supply_graph(seed, n_sites, n_corruptions, corruption_type, ...)
    -> (graph_dict, truth_dict)
"""

from __future__ import annotations
import numpy as np


def generate_supply_graph(
    seed: int,
    n_sites: int = 6,
    k: int = 0,
    corruption_type: str = "random",
    base_stock: float = 100.0,
    noise_sigma: float = 1.0,
) -> tuple[dict, dict]:
    """
    Build a supply-chain graph with planted corruption.

    Parameters
    ----------
    seed             RNG seed for reproducibility.
    n_sites          Number of sites (first is depot, rest are bases).
    k                Number of claims to corrupt.
    corruption_type  "random", "correlated", or "directional".
    base_stock       Nominal stock per site (jittered).
    noise_sigma      Small honest noise on claim values.

    Returns
    -------
    (graph, truth) where:
      graph  = compile()-ready dict {nodes, edges, claims}
      truth  = {true_state, corrupt_indices, corrupt_mask}
    """
    rng = np.random.default_rng(seed)

    # ── topology: star from depot + one chain for redundancy ────────────────
    depot = "DEPOT"
    bases = [f"BASE_{chr(65 + i)}" for i in range(n_sites - 1)]
    all_sites = [depot] + bases

    # Star edges: depot -> each base
    edges = []
    for i, b in enumerate(bases):
        edges.append({"id": f"e{i}", "from": depot, "to": b})

    # Chain edges among bases for extra redundancy: B0->B1, B1->B2, ...
    chain_start = len(edges)
    for i in range(len(bases) - 1):
        edges.append({
            "id": f"e{chain_start + i}",
            "from": bases[i],
            "to": bases[i + 1],
        })

    # ── true state: depot starts with total, bases get transfers ────────────
    true_stock = {}
    transfers = {}

    # Each base gets a random stock
    for b in bases:
        true_stock[b] = round(float(base_stock + rng.uniform(-20, 20)), 2)

    # Star transfers: depot sends stock to each base
    for e in edges[:len(bases)]:
        transfers[e["id"]] = true_stock[e["to"]]

    # Chain transfers: small inter-base flows
    for e in edges[len(bases):]:
        flow = round(float(rng.uniform(5, 30)), 2)
        transfers[e["id"]] = flow
        # Adjust stocks for conservation: from loses, to gains
        true_stock[e["from"]] = round(true_stock.get(e["from"], 0) - flow, 2)
        true_stock[e["to"]] = round(true_stock[e["to"]] + flow, 2)

    # Depot stock = total supply - sum of star transfers
    total_supply = sum(transfers[f"e{i}"] for i in range(len(bases)))
    # Add chain outflows from depot (none in this topology)
    true_stock[depot] = round(float(1000.0 - total_supply), 2)

    # ── nodes ───────────────────────────────────────────────────────────────
    nodes = []
    for s in all_sites:
        # Depot starts with 1000, others start with 0
        initial = 1000.0 if s == depot else 0.0
        nodes.append({"id": s, "initial": initial, "sinks": "none"})

    # ── claims: node stock + edge transfers ─────────────────────────────────
    claims = []
    cid = 0

    # Node claims (stock at each site)
    for s in all_sites:
        noisy = true_stock[s] + rng.normal(0, noise_sigma)
        claims.append({
            "id": f"c{cid}",
            "type": "node",
            "ref": s,
            "value": round(float(noisy), 2),
            "source": f"agent_{s.lower()}",
            "weight": 1.0,
        })
        cid += 1

    # Edge claims (transfers)
    for e in edges:
        noisy = transfers[e["id"]] + rng.normal(0, noise_sigma)
        claims.append({
            "id": f"c{cid}",
            "type": "edge",
            "ref": e["id"],
            "value": round(float(noisy), 2),
            "source": "consensus",
            "weight": 1.0,
        })
        cid += 1

    # ── extra redundant claims for higher correctable_k ─────────────────────
    # Duplicate node claims from a second source
    for s in all_sites:
        noisy = true_stock[s] + rng.normal(0, noise_sigma)
        claims.append({
            "id": f"c{cid}",
            "type": "node",
            "ref": s,
            "value": round(float(noisy), 2),
            "source": f"agent2_{s.lower()}",
            "weight": 0.9,
        })
        cid += 1

    # ── corruption injection ────────────────────────────────────────────────
    n_claims = len(claims)
    corrupt_mask = [False] * n_claims
    corrupt_indices = []

    if k > 0 and k <= n_claims:
        if corruption_type == "random":
            # Random k claims corrupted with random magnitude
            indices = rng.choice(n_claims, size=min(k, n_claims), replace=False)
            for idx in indices:
                delta = float(rng.uniform(50, 200)) * rng.choice([-1, 1])
                claims[idx]["value"] = round(claims[idx]["value"] + delta, 2)
                corrupt_mask[idx] = True
                corrupt_indices.append(int(idx))

        elif corruption_type == "correlated":
            # Correlated: corrupt a sender-receiver pair on the same edge
            # (hardest for L1 — looks like legitimate state change)
            edge_claims = [i for i, c in enumerate(claims) if c["type"] == "edge"]
            node_claims_map = {}
            for i, c in enumerate(claims):
                if c["type"] == "node":
                    node_claims_map.setdefault(c["ref"], []).append(i)

            corrupted = 0
            for ei in rng.permutation(len(edge_claims)):
                if corrupted >= k:
                    break
                ec_idx = edge_claims[ei]
                ec = claims[ec_idx]
                edge = next(e for e in edges if e["id"] == ec["ref"])
                delta = float(rng.uniform(50, 200))

                # Corrupt the edge claim
                claims[ec_idx]["value"] = round(claims[ec_idx]["value"] + delta, 2)
                corrupt_mask[ec_idx] = True
                corrupt_indices.append(ec_idx)
                corrupted += 1

                if corrupted >= k:
                    break

                # Corrupt a matching node claim on the target
                target_claims = node_claims_map.get(edge["to"], [])
                if target_claims:
                    nc_idx = target_claims[0]
                    claims[nc_idx]["value"] = round(claims[nc_idx]["value"] + delta, 2)
                    corrupt_mask[nc_idx] = True
                    corrupt_indices.append(nc_idx)
                    corrupted += 1

        elif corruption_type == "directional":
            # Directional: all corruptions inflate values (one-sided bias)
            indices = rng.choice(n_claims, size=min(k, n_claims), replace=False)
            for idx in indices:
                delta = float(rng.uniform(50, 200))  # always positive
                claims[idx]["value"] = round(claims[idx]["value"] + delta, 2)
                corrupt_mask[idx] = True
                corrupt_indices.append(int(idx))

    graph = {"nodes": nodes, "edges": edges, "claims": claims}

    true_state = np.array([true_stock[s] for s in all_sites])
    truth = {
        "true_state": true_state,
        "true_stock": true_stock,
        "true_transfers": transfers,
        "corrupt_indices": sorted(corrupt_indices),
        "corrupt_mask": corrupt_mask,
        "all_sites": all_sites,
    }

    return graph, truth
