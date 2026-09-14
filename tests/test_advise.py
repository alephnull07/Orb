"""
tests/test_advise.py
--------------------
Ops playbook: actionable suggestions + sensor placement for coordinated
lies and unidentifiable (k=0) snapshots.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orb.advise    import generate_advice
from orb.compile   import compile as compile_graph
from orb.decode    import decode
from orb.run_graph import _l1_solve

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _run(graph):
    compiled = compile_graph(graph)
    x_hat, residuals = _l1_solve(compiled)
    decoded = decode(x_hat, residuals, compiled, graph)
    advice = generate_advice(graph, compiled, decoded)
    return compiled, decoded, advice


def _blind_hop_graph():
    return {
        "nodes": [
            {"id": "WEST_YARD", "initial": 1000, "sinks": "none"},
            {"id": "OP_GHOST", "initial": 40, "sinks": "none"},
        ],
        "edges": [{"id": "e_WEST_YARD_OP_GHOST", "from": "WEST_YARD", "to": "OP_GHOST"}],
        "claims": [
            {"id": "s", "type": "edge", "ref": "e_WEST_YARD_OP_GHOST", "value": 400,
             "source": "shipment_sent", "weight": 1},
            {"id": "r", "type": "edge", "ref": "e_WEST_YARD_OP_GHOST", "value": 400,
             "source": "shipment_received", "weight": 1},
        ],
    }


def test_blind_hop_recommends_closeout_sensor():
    compiled, decoded, advice = _run(_blind_hop_graph())
    assert decoded["undetectable"]
    assert not decoded["flagged"]
    sensors = advice["sensors"]
    assert sensors, advice
    assert any(s["kind"] == "independent_count" for s in sensors)
    covered = {(h.get("from"), h.get("to")) for s in sensors for h in s.get("covers_hops") or []}
    assert ("WEST_YARD", "OP_GHOST") in covered
    kinds = {a["kind"] for a in advice["actions"]}
    assert "verify_hop" in kinds
    assert "place_sensor" in kinds
    assert advice["severity"] in {"watch", "critical"}
    assert advice.get("unable_to_detect")
    assert "unable to detect" in (advice.get("headline") or "").lower()
    for row in advice["on_hand"]:
        assert "do not plan" in row["action"].lower(), row


def test_spoke_blind_does_not_resupply_negative_qty():
    """80 opening − 350 unverified out → L1 −270 is not a real shortage."""
    graph = {
        "nodes": [
            {"id": "MAIN_DEPOT", "initial": 4000, "sinks": "none"},
            {"id": "FOB_ALPHA", "initial": 600, "sinks": "none"},
            {"id": "SHADOW_YARD", "initial": 80, "sinks": "none"},
            {"id": "OP_GHOST", "initial": 10, "sinks": "none"},
        ],
        "edges": [
            {"id": "e_depot_alpha", "from": "MAIN_DEPOT", "to": "FOB_ALPHA"},
            {"id": "e_shadow_ghost", "from": "SHADOW_YARD", "to": "OP_GHOST"},
        ],
        "claims": [
            {"id": "s1", "type": "edge", "ref": "e_depot_alpha", "value": 800,
             "source": "shipment_sent", "weight": 1},
            {"id": "r1", "type": "edge", "ref": "e_depot_alpha", "value": 800,
             "source": "shipment_received", "weight": 1},
            {"id": "d", "type": "node", "ref": "MAIN_DEPOT", "value": 3200,
             "source": "eod_on_hand", "weight": 1},
            {"id": "a", "type": "node", "ref": "FOB_ALPHA", "value": 1400,
             "source": "eod_on_hand", "weight": 1},
            {"id": "s2", "type": "edge", "ref": "e_shadow_ghost", "value": 350,
             "source": "shipment_sent", "weight": 1},
            {"id": "r2", "type": "edge", "ref": "e_shadow_ghost", "value": 350,
             "source": "shipment_received", "weight": 1},
        ],
    }
    compiled, decoded, advice = _run(graph)
    assert decoded["undetectable"]
    assert not any(a["kind"] == "resupply" and "SHADOW_YARD" in (a.get("where") or [])
                   for a in advice["actions"]), advice["actions"]
    shadow = next(r for r in advice["on_hand"] if r["id"] == "SHADOW_YARD")
    assert shadow["l1"] is not None and shadow["l1"] < 0
    assert "do not plan" in shadow["action"].lower()


def test_cooked_books_recommends_independent_sensor():
    """Sent=received=400 and EOD rewritten to match. L1 is silent; still place a sensor."""
    graph = {
        "nodes": [
            {"id": "WEST_YARD", "initial": 1000, "sinks": "none"},
            {"id": "OP_GHOST", "initial": 40, "sinks": "none"},
        ],
        "edges": [{"id": "e_WEST_YARD_OP_GHOST", "from": "WEST_YARD", "to": "OP_GHOST"}],
        "claims": [
            {"id": "s", "type": "edge", "ref": "e_WEST_YARD_OP_GHOST", "value": 400,
             "source": "shipment_sent", "weight": 1},
            {"id": "r", "type": "edge", "ref": "e_WEST_YARD_OP_GHOST", "value": 400,
             "source": "shipment_received", "weight": 1},
            {"id": "yd", "type": "node", "ref": "WEST_YARD", "value": 600,
             "source": "eod_on_hand", "weight": 1},
            {"id": "gd", "type": "node", "ref": "OP_GHOST", "value": 440,
             "source": "eod_on_hand", "weight": 1},
        ],
    }
    compiled, decoded, advice = _run(graph)
    assert not decoded["flagged"], decoded["flagged"]
    assert not decoded["undetectable"]
    assert advice["sensors"]
    assert any(s["kind"] == "independent_count" for s in advice["sensors"])
    assert any(s.get("required") is False for s in advice["sensors"])


def test_honest_closeout_flags_coordinated_lie():
    graph = {
        "nodes": [
            {"id": "WEST_YARD", "initial": 1000, "sinks": "none"},
            {"id": "OP_GHOST", "initial": 40, "sinks": "none"},
        ],
        "edges": [{"id": "e_WEST_YARD_OP_GHOST", "from": "WEST_YARD", "to": "OP_GHOST"}],
        "claims": [
            {"id": "s", "type": "edge", "ref": "e_WEST_YARD_OP_GHOST", "value": 400,
             "source": "shipment_sent", "weight": 1},
            {"id": "r", "type": "edge", "ref": "e_WEST_YARD_OP_GHOST", "value": 400,
             "source": "shipment_received", "weight": 1},
            {"id": "yd", "type": "node", "ref": "WEST_YARD", "value": 920,
             "source": "eod_on_hand", "weight": 1},
            {"id": "gd", "type": "node", "ref": "OP_GHOST", "value": 120,
             "source": "eod_on_hand", "weight": 1},
        ],
    }
    compiled, decoded, advice = _run(graph)
    assert abs(next(e["flow"] for e in decoded["edges"]) - 80) < 1.0
    assert decoded["flagged"], "the 400 send/receive claims must be flagged"
    kinds = {a["kind"] for a in advice["actions"]}
    assert "verify_hop" in kinds or "recount" in kinds
    assert advice["severity"] == "critical"
    assert any("80" in a["do"] for a in advice["actions"])


def test_two_blind_hops_sensor_covers_both():
    graph = {
        "nodes": [
            {"id": "WEST_YARD", "initial": 2000, "sinks": "none"},
            {"id": "MID_HUB", "initial": 100, "sinks": "none"},
            {"id": "OP_RIDGE", "initial": 20, "sinks": "none"},
        ],
        "edges": [
            {"id": "e1", "from": "WEST_YARD", "to": "MID_HUB"},
            {"id": "e2", "from": "MID_HUB", "to": "OP_RIDGE"},
        ],
        "claims": [
            {"id": "s1", "type": "edge", "ref": "e1", "value": 600, "source": "shipment_sent", "weight": 1},
            {"id": "r1", "type": "edge", "ref": "e1", "value": 600, "source": "shipment_received", "weight": 1},
            {"id": "s2", "type": "edge", "ref": "e2", "value": 500, "source": "shipment_sent", "weight": 1},
            {"id": "r2", "type": "edge", "ref": "e2", "value": 500, "source": "shipment_received", "weight": 1},
        ],
    }
    compiled, decoded, advice = _run(graph)
    assert len(decoded["undetectable"]) == 2
    hub_sensor = next(
        (s for s in advice["sensors"] if s["at"] == "MID_HUB" and s["kind"] == "independent_count"),
        None,
    )
    assert hub_sensor is not None, advice["sensors"]
    assert len(hub_sensor["covers_hops"]) == 2
    verify = [a for a in advice["actions"] if a["kind"] == "verify_hop"]
    assert len(verify) == 2
    assert all("MID_HUB" in a["do"] for a in verify), [a["do"] for a in verify]
    assert not any(a["kind"] == "cannot_certify" for a in advice["actions"])


def test_ambiguous_sink_recommends_drain_meter():
    import json
    with open(os.path.join(FIXTURES, "toy_water_sinks.json")) as fh:
        graph = json.load(fh)
    compiled, decoded, advice = _run(graph)
    assert decoded["ambiguous"]
    assert any(
        s["kind"] == "drain_meter"
        for s in advice["sensors"]
    )
    assert any(a["kind"] == "do_not_treat_as_leak" for a in advice["actions"])


def test_named_loss_is_resupply_not_paperwork():
    import json
    with open(os.path.join(FIXTURES, "toy_water_metered.json")) as fh:
        graph = json.load(fh)
    compiled, decoded, advice = _run(graph)
    assert decoded["loss"]
    assert any(a["kind"] == "resupply" for a in advice["actions"])
    assert advice["severity"] == "critical"


def test_coordinated_csv_fixtures_ingest():
    from orb.ingest import ingest_paths

    path = os.path.join(FIXTURES, "coordinated", "hop_no_closeout.csv")
    graph, report = ingest_paths([path])
    assert graph and graph.get("claims"), report
    compiled, decoded, advice = _run(graph)
    assert decoded["undetectable"]
    assert advice["sensors"]


def test_demo_json_includes_advice():
    from orb.demo_json import demo_json
    path = os.path.join(FIXTURES, "coordinated", "hop_no_closeout.csv")
    out = demo_json(path)
    assert "advice" in out
    assert out["advice"].get("actions")
    assert out["advice"].get("sensors")


def test_clean_fifteen_is_certified():
    """Honest 15-node net with EOD + tank_level: no flags, not unable_to_detect."""
    from orb.ingest import ingest_paths

    path = os.path.join(FIXTURES, "clean", "fifteen_node.csv")
    graph, report = ingest_paths([path])
    assert graph and graph.get("claims"), report
    assert len(graph["nodes"]) == 15
    compiled, decoded, advice = _run(graph)
    assert compiled["report"]["correctable_k"] >= 1
    assert not decoded["flagged"], decoded["flagged"]
    assert not decoded["loss"], decoded["loss"]
    assert not decoded["ambiguous"], decoded["ambiguous"]
    assert not decoded["undetectable"], decoded["undetectable"]
    assert not advice.get("unable_to_detect")
    assert advice["severity"] == "clear"
    assert "unable to detect" not in (advice.get("headline") or "").lower()
    assert not any(
        s["kind"] == "independent_count" for s in advice["sensors"]
    )
