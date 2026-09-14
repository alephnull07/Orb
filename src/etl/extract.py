"""Shared graph assembly for the three reader personas.

Personas differ in which free-text claims they trust. Structured sensor
rows (channel/sensor/value) and an optional official topology sidecar are
domain-agnostic and available to every persona so consensus can form on
any conserved-flow corpus.
"""

from __future__ import annotations

from .textutil import (
    ALSO_SENT_RE,
    DEPARTED_FOR_RE,
    EOD_ONHAND_RE,
    EOD_RE,
    FLOW_ARROW_RE,
    GOT_RE,
    RECEIVED_FROM_RE,
    ROLLING_RE,
    SEND_RE,
    SENT_TO_RE,
    SUSPICIOUS,
    TRANSFER_RE,
    as_number,
    canon,
    is_site_name,
    pick_reported_weight,
    resolve_sensor_edge,
    slug,
    speaker_stations,
)


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: dict[tuple[str, str], dict] = {}
        self.eod: dict[str, dict] = {}

    def touch(self, raw: str) -> str:
        key = canon(raw)
        label = raw.strip()
        if key not in self.nodes:
            self.nodes[key] = {
                "id": slug(raw),
                "display": label,
                "aliases": [label],
                "key": key,
            }
        elif label and label not in self.nodes[key]["aliases"]:
            self.nodes[key]["aliases"].append(label)
            if len(label) > len(self.nodes[key]["display"]):
                self.nodes[key]["display"] = label
        return key

    def add_edge(self, src: str, dst: str, value, mid, bias: str, corrupted: bool = False) -> None:
        if not is_site_name(src) or not is_site_name(dst):
            return
        qty = as_number(value)
        if qty is None:
            return
        sk, dk = self.touch(src), self.touch(dst)
        key = (sk, dk)
        mention = {"value_lb": qty, "corrupted": corrupted, "mid": mid}
        if key not in self.edges:
            self.edges[key] = {"mentions": [mention], "evidence": [mid], "bias": bias}
        else:
            self.edges[key]["mentions"].append(mention)
            self.edges[key]["evidence"].append(mid)
            self.edges[key]["bias"] = bias

    def ingest_text(
        self,
        messages: list[dict],
        *,
        outbound: bool = False,
        inbound: bool = False,
        also_sent: bool = False,
        eod: bool = False,
        skip_suspicious: bool = False,
    ) -> None:
        stations = speaker_stations(messages) if also_sent else {}
        for m in messages:
            text = m.get("raw_text") or ""
            mid = m.get("message_id")
            flagged = bool(m.get("corrupted"))
            if skip_suspicious and SUSPICIOUS.search(text):
                continue
            if eod:
                for match in EOD_RE.finditer(text):
                    place = match.group(1).strip()
                    key = self.touch(place)
                    self.eod[key] = {
                        "in_lb": as_number(match.group(2)),
                        "out_lb": as_number(match.group(3)),
                        "inventory_eod_lb": as_number(match.group(4)),
                        "evidence": mid,
                    }
                for match in EOD_ONHAND_RE.finditer(text):
                    place = (match.group(1) or match.group(3) or "").strip()
                    qty = as_number(match.group(2) or match.group(4))
                    if not place or qty is None:
                        speaker = (m.get("from_name") or "").strip()
                        if speaker and is_site_name(speaker):
                            place = speaker
                    if not place or qty is None or not is_site_name(place):
                        continue
                    key = self.touch(place)
                    self.eod[key] = {
                        "in_lb": None,
                        "out_lb": None,
                        "inventory_eod_lb": qty,
                        "evidence": mid,
                    }
            speaker = (m.get("from_name") or "").strip()
            if outbound:
                for match in SEND_RE.finditer(text):
                    self.add_edge(match.group(2), match.group(3), match.group(1), mid, "send", flagged)
                for match in TRANSFER_RE.finditer(text):
                    self.add_edge(match.group(2), match.group(3), match.group(1), mid, "transfer", flagged)
                for match in FLOW_ARROW_RE.finditer(text):
                    self.add_edge(match.group(2), match.group(3), match.group(1), mid, "flow", flagged)
                for match in DEPARTED_FOR_RE.finditer(text):
                    self.add_edge(match.group(1), match.group(2), match.group(3), mid, "departed", flagged)
                for match in ROLLING_RE.finditer(text):
                    self.add_edge(match.group(1), match.group(2), match.group(3), mid, "rolling", flagged)
                if speaker and is_site_name(speaker):
                    for match in SENT_TO_RE.finditer(text):
                        self.add_edge(speaker, match.group(2), match.group(1), mid, "sent_to", flagged)
            if inbound:
                for match in GOT_RE.finditer(text):
                    self.add_edge(match.group(2), match.group(3), match.group(1), mid, "got", flagged)
                if speaker and is_site_name(speaker):
                    for match in RECEIVED_FROM_RE.finditer(text):
                        self.add_edge(match.group(2), speaker, match.group(1), mid, "received_from", flagged)
            if also_sent:
                for match in ALSO_SENT_RE.finditer(text):
                    dst = match.group(2).strip()
                    if not is_site_name(dst):
                        continue
                    who = (m.get("from_name") or "").strip()
                    call = (m.get("callsign") or "").strip()
                    src_key = stations.get(who) or stations.get(who.lower()) or stations.get(call)
                    if not src_key or src_key == canon(dst):
                        continue
                    src_display = next(
                        (n["display"] for n in self.nodes.values() if n["key"] == src_key), src_key
                    )
                    self.add_edge(src_display, dst, match.group(1), mid, "also_sent", flagged)

    def ingest_structured(self, messages: list[dict], topology: dict | None) -> None:
        for m in messages:
            channel = (m.get("channel") or "").lower()
            if channel not in {"flow", "leak_demand", "transfer"}:
                continue
            qty = as_number(m.get("value"))
            if qty is None:
                continue
            if channel == "leak_demand" and qty <= 0:
                continue
            pair = resolve_sensor_edge(m.get("sensor") or "", topology or {}, channel)
            if not pair:
                continue
            src, dst = pair
            if qty < 0:
                src, dst = dst, src
                qty = -qty
            self.add_edge(src, dst, qty, m.get("message_id"), channel, bool(m.get("corrupted")))

    def finish(self, agent: str, role: str) -> dict:
        edges = []
        for (sk, dk), rec in self.edges.items():
            value, flagged = pick_reported_weight(rec["mentions"])
            edges.append(
                {
                    "source_key": sk,
                    "target_key": dk,
                    "source": self.nodes[sk]["id"],
                    "target": self.nodes[dk]["id"],
                    "source_display": self.nodes[sk]["display"],
                    "target_display": self.nodes[dk]["display"],
                    "value_lb": value,
                    "evidence": rec["evidence"],
                    "channel_bias": rec.get("bias"),
                    "from_corrupted": flagged,
                }
            )
        constraints = []
        for key, row in self.eod.items():
            constraints.append(
                {
                    "node": self.nodes[key]["id"],
                    "display": self.nodes[key]["display"],
                    "law": "mass_balance",
                    "in_lb": row["in_lb"],
                    "out_lb": row["out_lb"],
                    "inventory_eod_lb": row["inventory_eod_lb"],
                    "evidence": row["evidence"],
                }
            )
        out = {
            "agent": agent,
            "role": role,
            "nodes": list(self.nodes.values()),
            "edges": edges,
        }
        if constraints:
            out["trusted_constraint_rows"] = constraints
        return out
