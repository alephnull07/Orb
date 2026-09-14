"""False-certainty fixtures: data is insufficient; ORB must not name a lie or leak."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orb.advise    import generate_advice
from orb.compile   import compile as compile_graph
from orb.decode    import decode
from orb.ingest    import ingest_paths
from orb.run_graph import _l1_solve

DIR = os.path.join(os.path.dirname(__file__), "fixtures", "false_certainty")


def _run_csv(name: str):
    graph, report = ingest_paths([os.path.join(DIR, name)])
    assert graph and graph.get("claims"), report
    compiled = compile_graph(graph)
    decoded = decode(*_l1_solve(compiled), compiled, graph)
    advice = generate_advice(graph, compiled, decoded)
    return compiled, decoded, advice


def test_blind_hop_is_undetectable_not_named_corruption():
    compiled, decoded, advice = _run_csv("blind_hop.csv")
    assert decoded["undetectable"]
    assert not decoded["flagged"]
    assert not decoded["loss"]
    assert compiled["report"]["correctable_k"] == 0
    assert any(s.get("required") for s in advice["sensors"])
    assert advice["severity"] in {"watch", "critical"}
    assert advice.get("unable_to_detect")
    assert "unable to detect" in (advice.get("headline") or "").lower()


def test_blind_chain_one_sensor_covers_both_hops():
    compiled, decoded, advice = _run_csv("blind_chain.csv")
    assert len(decoded["undetectable"]) == 2
    assert not decoded["flagged"]
    hub = next(
        (s for s in advice["sensors"] if s["at"] == "RELAY_NINE" and s.get("required")),
        None,
    )
    assert hub is not None, advice["sensors"]
    assert len(hub["covers_hops"]) == 2


def test_unmetered_leak_is_ambiguous_not_named_loss():
    compiled, decoded, advice = _run_csv("unmetered_leak.csv")
    assert compiled["report"]["correctable_k"] == 0
    assert decoded["ambiguous"]
    assert not decoded["loss"]
    assert any(a["kind"] == "do_not_treat_as_leak" for a in advice["actions"])
    assert any(s["kind"] == "drain_meter" for s in advice["sensors"])


def test_one_qty_no_drain_cannot_name_loss():
    compiled, decoded, advice = _run_csv("one_qty_no_drain.csv")
    assert not decoded["loss"]
    assert decoded["ambiguous"] or compiled["report"]["correctable_k"] == 0
    assert any(s["kind"] in {"drain_meter", "second_qty"} for s in advice["sensors"])


def test_sender_only_hop_is_unverified():
    compiled, decoded, advice = _run_csv("sender_only.csv")
    assert decoded["undetectable"]
    assert not decoded["loss"]
    assert any(a["kind"] == "verify_hop" for a in advice["actions"])
