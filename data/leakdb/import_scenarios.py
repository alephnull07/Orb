#!/usr/bin/env python3
"""Import official LeakDB Hanoi_CMH scenarios. No synthetic chatter.

Each Scenario-N becomes run_NN with:
  true/      official sensors ~1 hour before the leak (last healthy Label=0
             before Label=1), or the first healthy snapshot if there is no leak
  corrupted/ official sensors at the first leak-active timestamp (Labels==1),
             or the same healthy snapshot if that scenario has no leak
  eval/      graphs built only from the official INP + official Flows/Leaks CSVs
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "raw" / "Hanoi_CMH.zip"
RUNS_DIR = ROOT / "runs"


def _open_zip() -> zipfile.ZipFile:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ZIP_PATH}. Download "
            "https://github.com/KIOS-Research/LeakDB/raw/master/CCWI-WDSA2018/Benchmarks/Hanoi_CMH.zip"
        )
    return zipfile.ZipFile(ZIP_PATH)


def _read(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8", errors="replace")


def _parse_info(text: str) -> dict:
    out = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2 or row[0].strip().lower() == "description":
            continue
        out[row[0].strip()] = row[1].strip()
    return out


def _parse_kv_csv(text: str) -> dict:
    return _parse_info(text)


def _timeseries(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0].strip().lower() == "timestamp":
            continue
        try:
            out[row[0].strip()] = float(row[1])
        except (IndexError, ValueError):
            continue
    return out


def _first_label(text: str, want: float) -> str | None:
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0].strip().lower() == "timestamp":
            continue
        try:
            if float(row[1]) == want:
                return row[0].strip()
        except (IndexError, ValueError):
            continue
    return None


def _parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")


def _pre_leak_timestamp(labels_text: str, leak_ts: str, hours: float = 1.0) -> str:
    """Healthy snapshot ~1 hour before the first Label=1 row."""
    rows: list[tuple[str, float]] = []
    for row in csv.reader(io.StringIO(labels_text)):
        if not row or row[0].strip().lower() == "timestamp":
            continue
        try:
            rows.append((row[0].strip(), float(row[1])))
        except (IndexError, ValueError):
            continue
    want = (_parse_ts(leak_ts) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    for ts, lab in rows:
        if ts == want and lab == 0.0:
            return ts
    leak_dt = _parse_ts(leak_ts)
    last = None
    for ts, lab in rows:
        if lab == 0.0 and _parse_ts(ts) < leak_dt:
            last = ts
    if last:
        return last
    raise RuntimeError(f"No Label=0 row before leak at {leak_ts}")


def parse_inp(text: str) -> dict:
    section = None
    nodes: dict[str, dict] = {}
    pipes: list[dict] = []
    for raw in text.splitlines():
        line = raw.split(";")[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.upper()
            continue
        parts = line.split()
        if section == "[JUNCTIONS]" and len(parts) >= 2:
            jid = parts[0]
            demand = float(parts[2]) if len(parts) >= 3 else 0.0
            nodes[jid] = {
                "id": f"J_{jid}",
                "nato": jid,
                "type": "junction",
                "kind": "demand junction",
                "display": f"Junction {jid}",
                "demand_cmh": demand,
                "aliases": [f"Junction {jid}", f"J{jid}", f"J-{jid}", jid],
            }
        elif section == "[RESERVOIRS]" and len(parts) >= 2:
            rid = parts[0]
            nodes[rid] = {
                "id": f"RES_{rid}",
                "nato": rid,
                "type": "source",
                "kind": "reservoir",
                "display": f"Reservoir {rid}",
                "demand_cmh": 0.0,
                "head_m": float(parts[1]),
                "aliases": [f"Reservoir {rid}", f"R{rid}", rid],
            }
        elif section == "[PIPES]" and len(parts) >= 4:
            pipes.append(
                {
                    "pipe_id": parts[0],
                    "n1": parts[1],
                    "n2": parts[2],
                    "length_m": float(parts[3]),
                    "diameter": float(parts[4]) if len(parts) > 4 else None,
                    "roughness": float(parts[5]) if len(parts) > 5 else None,
                }
            )
    return {"nodes": nodes, "pipes": pipes}


def topology_from_net(net: dict) -> dict:
    links = {}
    for p in net["pipes"]:
        src = net["nodes"][p["n1"]]["display"]
        tgt = net["nodes"][p["n2"]]["display"]
        rec = {"source_name": src, "target_name": tgt, "pipe_id": p["pipe_id"]}
        links[str(p["pipe_id"])] = rec
        links[f"Link_{p['pipe_id']}"] = rec
    return {"links": links, "nodes": list(net["nodes"].values())}


def write_topology(out_dir: Path, topology: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "topology.json").write_text(json.dumps(topology, indent=2) + "\n", encoding="utf-8")


def attach_topologies(n: int = 10) -> None:
    """Write official INP topology into already-imported runs (no re-download)."""
    with _open_zip() as zf:
        for i in range(1, n + 1):
            prefix = f"Scenario-{i}/"
            inp_name = next(x for x in zf.namelist() if x.startswith(prefix) and x.endswith(".inp"))
            topo = topology_from_net(parse_inp(_read(zf, inp_name)))
            for side in ("true", "corrupted"):
                dest = RUNS_DIR / f"run_{i:02d}" / side
                if dest.exists():
                    write_topology(dest, topo)


def _node(nodes: dict, raw: str) -> dict:
    if raw in nodes:
        return nodes[raw]
    raise KeyError(raw)


def graph_at(
    net: dict,
    flows: dict[str, float],
    leaks: list[dict],
    timestamp: str,
    scenario: int,
    info: dict,
    kind: str,
) -> dict:
    nodes = []
    seen_leak = False
    for n in net["nodes"].values():
        nodes.append(
            {
                "id": n["id"],
                "nato": n["nato"],
                "type": n["type"],
                "kind": n["kind"],
                "display": n["display"],
                "aliases": n["aliases"],
                "demand_cmh": n.get("demand_cmh", 0),
            }
        )
    edges = []
    for p in net["pipes"]:
        src, tgt = _node(net["nodes"], p["n1"]), _node(net["nodes"], p["n2"])
        flow = flows.get(p["pipe_id"], 0.0)
        src_name, tgt_name = src["display"], tgt["display"]
        src_id, tgt_id = src["id"], tgt["id"]
        if flow < 0:
            flow = -flow
            src_name, tgt_name = tgt_name, src_name
            src_id, tgt_id = tgt_id, src_id
        edges.append(
            {
                "source": src_id,
                "target": tgt_id,
                "value_lb": round(flow, 4),
                "source_name": src_name,
                "target_name": tgt_name,
                "pipe_id": p["pipe_id"],
                "length_m": p["length_m"],
                "kind": "pipe",
            }
        )
    leak_meta = []
    for leak in leaks:
        q = leak.get("demand_at_ts") or 0.0
        if q <= 0:
            continue
        seen_leak = True
        j = _node(net["nodes"], leak["node"])
        if not any(n["id"] == "SINK_LEAK" for n in nodes):
            nodes.append(
                {
                    "id": "SINK_LEAK",
                    "nato": "LEAK",
                    "type": "sink",
                    "kind": "leak discharge",
                    "display": "Leak discharge",
                    "aliases": ["Leak discharge"],
                    "demand_cmh": 0,
                }
            )
        edges.append(
            {
                "source": j["id"],
                "target": "SINK_LEAK",
                "value_lb": round(q, 4),
                "source_name": j["display"],
                "target_name": "Leak discharge",
                "pipe_id": f"LEAK_{leak['node']}",
                "kind": "leak",
            }
        )
        leak_meta.append(
            {
                "node": leak["node"],
                "node_name": j["display"],
                "leak_cmh": round(q, 4),
                "leak_type": leak.get("type"),
                "start": leak.get("start"),
                "end": leak.get("end"),
                "diameter": leak.get("diameter"),
            }
        )
    if seen_leak:
        total = sum(e["value_lb"] for e in edges if e.get("kind") == "leak")
        for n in nodes:
            if n["id"] == "SINK_LEAK":
                n["demand_cmh"] = round(total, 4)
    return {
        "scenario": f"leakdb_hanoi_cmh_scenario_{scenario}_{kind}",
        "network": "Hanoi_CMH",
        "source": "KIOS LeakDB Hanoi_CMH.zip",
        "scenario_id": scenario,
        "timestamp": timestamp,
        "window": {
            "commodity": "potable_water",
            "unit": "m3/h",
            "time_step": info.get("Time_Step"),
        },
        "model_uncertainty": {
            "topology": info.get("Uncertainty_Topology_(%)"),
            "length_pct": info.get("Uncertainty_Length_(%)"),
            "diameter_pct": info.get("Uncertainty_Diameter_(%)"),
            "roughness_pct": info.get("Uncertainty_Roughness_(%)"),
        },
        "notes": (
            "Graph assembled only from the official EPANET .inp and official "
            f"Flows/Leaks CSVs at {timestamp}. No synthetic reports."
        ),
        "leaks": leak_meta,
        "nodes": nodes,
        "edges": edges,
    }


def _sensor_records(kind: str, sensor: str, ts: str, value: float, line: str) -> dict:
    iso = ts.replace(" ", "T") + "Z"
    return {
        "message_id": f"{kind}-{sensor}-{ts.replace(' ', 'T')}",
        "timestamp": iso,
        "channel": kind,
        "sensor": sensor,
        "value": value,
        "raw_text": line.strip(),
    }


def snapshot_records(
    zf: zipfile.ZipFile,
    scenario: int,
    ts: str,
    leaks: list[dict],
) -> list[dict]:
    prefix = f"Scenario-{scenario}/"
    rows = []
    names = zf.namelist()
    for folder, kind in (("Flows", "flow"), ("Pressures", "pressure"), ("Demands", "demand")):
        for name in names:
            if not name.startswith(prefix + folder + "/") or not name.endswith(".csv"):
                continue
            sensor = Path(name).stem
            series = _timeseries(_read(zf, name))
            if ts not in series:
                continue
            val = series[ts]
            rows.append(_sensor_records(kind, sensor, ts, val, f"{ts},{val}"))
    for leak in leaks:
        q = leak.get("demand_at_ts")
        if q is None:
            continue
        rows.append(
            _sensor_records(
                "leak_demand",
                f"Leak_{leak['node']}",
                ts,
                q,
                f"{ts},{q}",
            )
        )
    rows.sort(key=lambda r: (r["channel"], r["sensor"]))
    return rows


def write_side(out_dir: Path, records: list[dict], extras: dict[str, str]) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    with (out_dir / "messages.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Official CSV snapshot (not invented chat).
    lines = ["timestamp,channel,sensor,value"]
    for r in records:
        lines.append(f"{r['timestamp']},{r['channel']},{r['sensor']},{r['value']}")
    (out_dir / "scada_snapshot.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, text in extras.items():
        (out_dir / name).write_text(text, encoding="utf-8")


def collect_leaks(zf: zipfile.ZipFile, scenario: int, ts: str) -> list[dict]:
    prefix = f"Scenario-{scenario}/Leaks/"
    infos = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith("_info.csv")]
    leaks = []
    for info_name in infos:
        info = _parse_kv_csv(_read(zf, info_name))
        node = (info.get("Leak Node") or "").strip()
        demand_name = prefix + f"Leak_{node}_demand.csv"
        demand_at = None
        if demand_name in zf.namelist():
            series = _timeseries(_read(zf, demand_name))
            demand_at = series.get(ts)
        leaks.append(
            {
                "node": node,
                "area": info.get("Leak Area"),
                "diameter": info.get("Leak Diameter"),
                "type": info.get("Leak Type"),
                "start": info.get("Leak Start"),
                "end": info.get("Leak End"),
                "peak": info.get("Peak Time"),
                "demand_at_ts": demand_at,
                "info_csv": _read(zf, info_name),
            }
        )
    return leaks


def import_scenario(zf: zipfile.ZipFile, scenario: int) -> dict:
    prefix = f"Scenario-{scenario}/"
    inp_name = next(n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".inp"))
    labels_text = _read(zf, prefix + "Labels.csv")
    info = _parse_info(_read(zf, prefix + f"Scenario-{scenario}_info.csv"))
    first_healthy = _first_label(labels_text, 0.0)
    leak_ts = _first_label(labels_text, 1.0)
    if not first_healthy:
        raise RuntimeError(f"Scenario-{scenario} has no Label=0 row")
    # true = ~1 hour before the leak. corrupted keeps those same pipe meters
    # and adds the official leak as an extra sink (supply-drop ghost analog).
    # Official leak-day Flows are a new EPANET solve and are not used as pipes.
    true_ts = _pre_leak_timestamp(labels_text, leak_ts) if leak_ts else first_healthy
    leak_on_ts = leak_ts or true_ts

    net = parse_inp(_read(zf, inp_name))
    flow_files = {
        re.sub(r"^Link_", "", Path(n).stem): n
        for n in zf.namelist()
        if n.startswith(prefix + "Flows/") and n.endswith(".csv")
    }

    def flows_at(ts: str) -> dict[str, float]:
        out = {}
        for pid, name in flow_files.items():
            series = _timeseries(_read(zf, name))
            if ts in series:
                out[pid] = series[ts]
        return out

    pipe_flows = flows_at(true_ts)
    leak_at_onset = collect_leaks(zf, scenario, leak_on_ts)
    true_graph = graph_at(net, pipe_flows, [], true_ts, scenario, info, "true")
    corrupt_graph = graph_at(
        net, pipe_flows, leak_at_onset, true_ts, scenario, info, "corrupted"
    )

    run_id = f"run_{scenario:02d}"
    out_root = RUNS_DIR / run_id
    if out_root.exists():
        shutil.rmtree(out_root)
    eval_dir = out_root / "eval"
    eval_dir.mkdir(parents=True)

    extras_true = {
        "scenario_info.csv": _read(zf, prefix + f"Scenario-{scenario}_info.csv"),
        "Labels.csv": labels_text,
    }
    extras_cor = dict(extras_true)
    for leak in leak_at_onset:
        extras_cor[f"Leak_{leak['node']}_info.csv"] = leak["info_csv"]

    write_side(out_root / "true", snapshot_records(zf, scenario, true_ts, []), extras_true)
    write_side(
        out_root / "corrupted",
        snapshot_records(zf, scenario, true_ts, leak_at_onset),
        extras_cor,
    )
    topo = topology_from_net(net)
    write_topology(out_root / "true", topo)
    write_topology(out_root / "corrupted", topo)
    (eval_dir / "true_graph.json").write_text(json.dumps(true_graph, indent=2) + "\n", encoding="utf-8")
    (eval_dir / "corrupted_graph.json").write_text(
        json.dumps(corrupt_graph, indent=2) + "\n", encoding="utf-8"
    )
    labels = {
        "source": "KIOS LeakDB Hanoi_CMH",
        "scenario": scenario,
        "true_timestamp": true_ts,
        "corrupted_timestamp": leak_on_ts,
        "has_leak": bool(leak_ts),
        "leaks": [
            {k: v for k, v in leak.items() if k != "info_csv"} for leak in leak_at_onset
        ],
        "attacks": [
            {
                "id": "leak_discharge",
                "source_name": f"Junction {leak['node']}",
                "target_name": "Leak discharge",
                "reported_lb": leak.get("demand_at_ts"),
                "true_lb": 0,
            }
            for leak in leak_at_onset
            if (leak.get("demand_at_ts") or 0) > 0
        ],
        "model_uncertainty": true_graph["model_uncertainty"],
    }
    (eval_dir / "corruption_labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    return {
        "run_id": run_id,
        "scenario": scenario,
        "true_timestamp": true_ts,
        "corrupted_timestamp": leak_on_ts,
        "has_leak": bool(leak_ts),
        "n_leaks": len([x for x in leak_at_onset if (x.get("demand_at_ts") or 0) > 0]),
        "true_nodes": len(true_graph["nodes"]),
        "true_edges": len(true_graph["edges"]),
        "corrupted_nodes": len(corrupt_graph["nodes"]),
        "corrupted_edges": len(corrupt_graph["edges"]),
        "true_messages": len(list((out_root / "true" / "messages.jsonl").open(encoding="utf-8"))),
        "corrupted_messages": len(list((out_root / "corrupted" / "messages.jsonl").open(encoding="utf-8"))),
        "out_root": str(out_root),
    }


def import_all(n: int = 10) -> list[dict]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    with _open_zip() as zf:
        for i in range(1, n + 1):
            print(f"Importing official Scenario-{i} ...")
            summary.append(import_scenario(zf, i))
    manifest = RUNS_DIR / "manifest.json"
    manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary)} scenarios + {manifest}")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Import official LeakDB Hanoi_CMH scenarios")
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    import_all(n=args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
