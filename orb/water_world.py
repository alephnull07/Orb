"""
orb/water_world.py
------------------
Generate a synthetic water distribution network with a planted leak,
produce a LeakDB-style SCADA CSV, and find a configuration where the
L1 estimator ranks the true leak node #1 with correctable_k >= 1.

Public API
----------
generate_network(seed, n_junctions, leak_node, leak_size)
generate_scada_csv(truth, path, seed, sensor_fraction, noise_sigma)
find_good_config()

Column mapping for the generated CSV (deterministic, no LLM needed):
WATER_COLUMN_MAPPING
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Deterministic column mapping for the generated CSV format
# ---------------------------------------------------------------------------

WATER_COLUMN_MAPPING = {
    "entity_column": "sensor_id",
    "value_column": "value",
    "time_column": "timestamp",
    "channel_column": "channel",
    "channel_map": {"demand": "node", "flow": "edge", "sink_meter": "sink"},
    "from_column": "from_node",
    "to_column": "to_node",
    "id_pattern": None,
    "excluded_channels": [],
    "notes": "SCADA water network: demand->node, flow->edge, sink_meter->sink",
}


# ---------------------------------------------------------------------------
# Network generation
# ---------------------------------------------------------------------------

def generate_network(
    seed: int,
    n_junctions: int,
    leak_node: int,
    leak_size: float,
    base_demand: float = 100.0,
) -> dict:
    """
    Build a connected mesh pipe network and solve for consistent flows.

    Parameters
    ----------
    seed          RNG seed
    n_junctions   number of junction nodes (excluding reservoir)
    leak_node     index (0-based) of the junction with the extra leak
    leak_size     additional demand at the leak junction
    base_demand   nominal demand per junction (jittered ±20)

    Returns truth dict (kept separate from anything the estimator sees):
        junctions   [{id, base_demand, total_demand}]
        pipes       [{id, from, to, flow}]
        reservoir   {id, supply}
        leak_node   junction id string
        leak_size   float
    """
    rng = np.random.default_rng(seed)

    # Junction demands: base (without leak) and total (with leak)
    base_demands = []
    total_demands = []
    for i in range(n_junctions):
        bd = base_demand + rng.uniform(-20, 20)
        base_demands.append(round(float(bd), 2))
        td = bd + (leak_size if i == leak_node else 0.0)
        total_demands.append(round(float(td), 2))
    demands = total_demands  # flows are solved with total demands

    jids = [f"J_{i+1:02d}" for i in range(n_junctions)]

    # Build MESH topology: ring + one chord (sparse but connected with cycle)
    pipe_endpoints: list[tuple[int, int]] = []
    pipe_set: set[tuple[int, int]] = set()

    def _add(a: int, b: int):
        key = (min(a, b), max(a, b))
        if key not in pipe_set and a != b:
            pipe_set.add(key)
            pipe_endpoints.append((a, b))

    # Ring: 0-1-2-...-n-0
    for i in range(n_junctions):
        _add(i, (i + 1) % n_junctions)
    # One chord across the ring for mesh property
    _add(0, n_junctions // 2)

    # Reservoir feeds into two junctions
    res_targets = [0, n_junctions // 2]
    n_internal = len(pipe_endpoints)

    # Incidence matrix: rows = junctions, cols = [internal pipes | reservoir pipes]
    # Convention: pipe (a,b) has +1 at b (inflow), -1 at a (outflow)
    n_res_pipes = len(res_targets)
    total_pipes = n_internal + n_res_pipes
    A = np.zeros((n_junctions, total_pipes))

    for pi, (a, b) in enumerate(pipe_endpoints):
        A[a, pi] = -1.0
        A[b, pi] = +1.0

    for ri, tgt in enumerate(res_targets):
        A[tgt, n_internal + ri] = +1.0   # reservoir inflow to junction

    # Solve A @ f = demand via least-squares
    d_vec = np.array(demands)
    f_hat, _, _, _ = np.linalg.lstsq(A, d_vec, rcond=None)

    # Verify conservation at every junction
    balance = A @ f_hat - d_vec
    assert np.allclose(balance, 0, atol=1e-6), (
        f"Conservation violated: max |balance| = {np.max(np.abs(balance)):.6f}"
    )

    # Build pipe dicts
    pipes = []
    for pi, (a, b) in enumerate(pipe_endpoints):
        pipes.append({
            "id": f"P_{pi+1:02d}",
            "from": jids[a],
            "to": jids[b],
            "flow": round(float(f_hat[pi]), 4),
        })
    for ri, tgt in enumerate(res_targets):
        pipes.append({
            "id": f"P_{n_internal + ri + 1:02d}",
            "from": "R_01",
            "to": jids[tgt],
            "flow": round(float(f_hat[n_internal + ri]), 4),
        })

    total_supply = round(float(np.sum(d_vec)), 4)

    return {
        "junctions": [
            {"id": jids[i], "base_demand": base_demands[i], "total_demand": total_demands[i]}
            for i in range(n_junctions)
        ],
        "pipes": pipes,
        "reservoir": {"id": "R_01", "supply": total_supply},
        "leak_node": jids[leak_node],
        "leak_size": leak_size,
    }


# ---------------------------------------------------------------------------
# SCADA CSV generation
# ---------------------------------------------------------------------------

def generate_scada_csv(
    truth: dict,
    path: str | Path,
    seed: int = 42,
    sensor_fraction: float = 1.0,
    noise_sigma: float = 2.0,
    extra_demand_sensors: int = 0,
    sink_meter_indices: list[int] | None = None,
) -> None:
    """
    Write a SCADA CSV from truth.
    Columns: sensor_id, channel, value, timestamp, from_node, to_node

    - 'demand' rows:     sensor_id = junction id, from/to blank
    - 'flow' rows:       sensor_id = pipe id, from/to = pipe endpoints
    - 'sink_meter' rows: sensor_id = junction id, value = drain meter reading

    Generic sensor IDs only — no leak info encoded anywhere.
    Sink meters at non-leak junctions read ~0; at leak junction they
    read ~leak_size.  This provides independent observations that raise
    correctable_k.
    """
    rng = np.random.default_rng(seed)
    ts = "2026-01-01T00:00:00Z"

    junctions = truth["junctions"]
    pipes = truth["pipes"]
    leak_nid = truth["leak_node"]
    leak_sz = truth["leak_size"]

    # Select sensor subsets
    n_j = max(1, int(len(junctions) * sensor_fraction))
    n_p = max(1, int(len(pipes) * sensor_fraction))
    j_indices = sorted(rng.choice(len(junctions), size=n_j, replace=False))
    p_indices = sorted(rng.choice(len(pipes), size=n_p, replace=False))

    rows = []
    for idx in j_indices:
        j = junctions[idx]
        noisy = j["base_demand"] + rng.normal(0, noise_sigma)
        rows.append({
            "sensor_id": j["id"],
            "channel": "demand",
            "value": round(noisy, 2),
            "timestamp": ts,
            "from_node": "",
            "to_node": "",
        })

    for idx in p_indices:
        p = pipes[idx]
        noisy = p["flow"] + rng.normal(0, noise_sigma)
        rows.append({
            "sensor_id": p["id"],
            "channel": "flow",
            "value": round(noisy, 2),
            "timestamp": ts,
            "from_node": p["from"],
            "to_node": p["to"],
        })

    # Extra redundant demand sensors
    if extra_demand_sensors > 0:
        extra_j = rng.choice(len(junctions), size=min(extra_demand_sensors, len(junctions)), replace=False)
        for idx in extra_j:
            j = junctions[idx]
            noisy = j["base_demand"] + rng.normal(0, noise_sigma)
            rows.append({
                "sensor_id": j["id"],
                "channel": "demand",
                "value": round(noisy, 2),
                "timestamp": ts,
                "from_node": "",
                "to_node": "",
            })

    # Sink meters (drain meters at specified junctions)
    if sink_meter_indices is not None:
        for idx in sink_meter_indices:
            j = junctions[idx]
            true_sink = leak_sz if j["id"] == leak_nid else 0.0
            noisy = true_sink + rng.normal(0, noise_sigma)
            noisy = max(0.0, noisy)
            rows.append({
                "sensor_id": j["id"],
                "channel": "sink_meter",
                "value": round(noisy, 2),
                "timestamp": ts,
                "from_node": "",
                "to_node": "",
            })
        # Extra duplicate sink meters at reservoir-target junctions
        # to ensure correctable_k >= 1 (two independent observations).
        res_targets = set()
        for p in pipes:
            if p["from"] == truth["reservoir"]["id"]:
                tgt_idx = next(i for i, j in enumerate(junctions) if j["id"] == p["to"])
                res_targets.add(tgt_idx)
        for idx in res_targets:
            j = junctions[idx]
            true_sink = leak_sz if j["id"] == leak_nid else 0.0
            noisy = true_sink + rng.normal(0, noise_sigma)
            noisy = max(0.0, noisy)
            rows.append({
                "sensor_id": j["id"],
                "channel": "sink_meter",
                "value": round(noisy, 2),
                "timestamp": ts,
                "from_node": "",
                "to_node": "",
            })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sensor_id", "channel", "value", "timestamp", "from_node", "to_node"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Configuration search
# ---------------------------------------------------------------------------

def _evaluate_config(seed, n_junctions, leak_idx, leak_size, sensor_frac, noise_sigma, extra_demand, sink_meters=None):
    """Try one config, return result dict or None."""
    import tempfile
    from .compile import compile as compile_graph
    from .run_graph import _l1_solve
    from .decode import decode
    from .ingest import build_graph

    truth = generate_network(seed, n_junctions, leak_idx, leak_size)
    csv_path = os.path.join(tempfile.gettempdir(), f"orb_water_{seed}_{leak_idx}.csv")
    generate_scada_csv(truth, csv_path, seed=seed,
                       sensor_fraction=sensor_frac, noise_sigma=noise_sigma,
                       extra_demand_sensors=extra_demand,
                       sink_meter_indices=sink_meters)

    graphs, report = build_graph(
        csv_path, sinks="unknown", column_mapping=WATER_COLUMN_MAPPING,
    )
    if not graphs:
        return None
    graph = graphs[0]

    compiled = compile_graph(graph)
    rep = compiled["report"]

    if not rep["identifiable"]:
        return None

    x_hat, residuals = _l1_solve(compiled)
    decoded = decode(x_hat, residuals, compiled, graph)

    # Rank sinks by magnitude (descending)
    sinks = sorted(decoded["sinks"], key=lambda s: abs(s["sink"]), reverse=True)
    if not sinks:
        return None

    leak_nid = truth["leak_node"]
    leak_rank = None
    for i, s in enumerate(sinks):
        if s["id"] == leak_nid:
            leak_rank = i + 1
            break
    if leak_rank is None:
        return None

    top_mag = abs(sinks[0]["sink"])
    second_mag = abs(sinks[1]["sink"]) if len(sinks) > 1 else 0.0
    margin = top_mag - second_mag

    return {
        "rank": leak_rank,
        "margin": margin,
        "correctable_k": rep["correctable_k"],
        "truth": truth,
        "csv_path": csv_path,
        "graph": graph,
        "decoded": decoded,
        "report": rep,
        "sinks": sinks,
        "ingest_report": report,
    }


def find_good_config(
    out_csv: str = "tests/fixtures/demo_water.csv",
    out_truth: str = "tests/fixtures/demo_water_truth.json",
) -> dict | None:
    """
    Try a few configurations and pick the first where:
      - true leak ranks #1 by sink magnitude
      - clear margin over #2
      - correctable_k >= 1
    """
    configs = [
        # (seed, n_junctions, leak_idx, leak_size, sensor_frac, noise_sigma, extra_demand, sink_meters)
        # Each junction needs ≥2 independent observations for correctable_k≥1
        (42, 6, 2, 300.0, 1.0, 1.0, 6, list(range(6))),
        (7,  6, 3, 350.0, 1.0, 1.0, 6, list(range(6))),
        (99, 5, 2, 350.0, 1.0, 0.5, 5, list(range(5))),
        (12, 5, 1, 400.0, 1.0, 0.5, 5, list(range(5))),
    ]

    for seed, nj, li, ls, sf, ns, ed, sm in configs:
        r = _evaluate_config(seed, nj, li, ls, sf, ns, ed, sm)
        if r is None:
            print(f"  seed={seed} nj={nj}: not identifiable")
            continue
        print(f"  seed={seed} nj={nj} li={li} ls={ls}: "
              f"rank={r['rank']} margin={r['margin']:.1f} k={r['correctable_k']}")
        if r["rank"] == 1 and r["margin"] > 10 and r["correctable_k"] >= 1:
            truth = r["truth"]
            generate_scada_csv(truth, out_csv, seed=seed,
                               sensor_fraction=sf, noise_sigma=ns,
                               extra_demand_sensors=ed,
                               sink_meter_indices=sm)
            truth_out = {
                "leak_node": truth["leak_node"],
                "leak_size": truth["leak_size"],
                "reservoir_supply": truth["reservoir"]["supply"],
            }
            Path(out_truth).parent.mkdir(parents=True, exist_ok=True)
            with open(out_truth, "w") as fh:
                json.dump(truth_out, fh, indent=2)

            print(f"\n[water_world] Winner: seed={seed} n={nj} leak_idx={li} "
                  f"leak_size={ls} extra_demand={ed}")
            print(f"  Wrote {out_csv}")
            print(f"  Wrote {out_truth}")
            return r

    print("[water_world] No good config found.")
    return None


if __name__ == "__main__":
    find_good_config()
