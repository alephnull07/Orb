"""Agent Auditor — EOD books plus conservation-shaped transfers.

Ignores suspicious chatter (yesterday's OCR, kg mixups, 'second drop').
Parses receipts/sends only when the message does not look injected, and
always records EOD mass-balance rows as trusted constraints.
"""

from __future__ import annotations

from .textutil import (
    ALSO_SENT_RE,
    EOD_RE,
    GOT_RE,
    SEND_RE,
    SUSPICIOUS,
    canon,
    is_site_name,
    pick_reported_weight,
    slug,
    speaker_stations,
)


NAME = "auditor"
ROLE = "schema validator / conservation-law tagger"


def extract(messages: list[dict]) -> dict:
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = {}
    eod: dict[str, dict] = {}

    def touch(raw: str) -> str:
        key = canon(raw)
        if key not in nodes:
            nodes[key] = {
                "id": slug(raw),
                "display": raw.strip(),
                "aliases": [raw.strip()],
                "key": key,
            }
        elif raw.strip() not in nodes[key]["aliases"]:
            nodes[key]["aliases"].append(raw.strip())
            if len(raw.strip()) > len(nodes[key]["display"]):
                nodes[key]["display"] = raw.strip()
        return key

    stations = speaker_stations(messages)
    for m in messages:
        text = m.get("raw_text") or ""
        mid = m.get("message_id")
        if SUSPICIOUS.search(text):
            continue
        for match in EOD_RE.finditer(text):
            place = match.group(1).strip()
            key = touch(place)
            eod[key] = {
                "in_lb": int(match.group(2)),
                "out_lb": int(match.group(3)),
                "inventory_eod_lb": int(match.group(4)),
                "evidence": mid,
            }
        for match in SEND_RE.finditer(text):
            lb, src, dst = int(match.group(1)), match.group(2).strip(), match.group(3).strip()
            if not is_site_name(src) or not is_site_name(dst):
                continue
            sk, dk = touch(src), touch(dst)
            _acc(edges, sk, dk, lb, mid, "send", bool(m.get("corrupted")))
        for match in GOT_RE.finditer(text):
            lb, src, dst = int(match.group(1)), match.group(2).strip(), match.group(3).strip()
            if not is_site_name(src) or not is_site_name(dst):
                continue
            sk, dk = touch(src), touch(dst)
            _acc(edges, sk, dk, lb, mid, "got", bool(m.get("corrupted")))
        for match in ALSO_SENT_RE.finditer(text):
            dst = match.group(2).strip()
            if not is_site_name(dst):
                continue
            who = (m.get("from_name") or "").strip()
            call = (m.get("callsign") or "").strip()
            src_key = stations.get(who) or stations.get(who.lower()) or stations.get(call)
            if not src_key or src_key == canon(dst):
                continue
            src_display = next((n["display"] for n in nodes.values() if n["key"] == src_key), src_key)
            sk, dk = touch(src_display), touch(dst)
            _acc(edges, sk, dk, int(match.group(1)), mid, "send", bool(m.get("corrupted")))

    out_edges = []
    for (sk, dk), rec in edges.items():
        value, flagged = pick_reported_weight(rec["mentions"])
        out_edges.append(
            {
                "source_key": sk,
                "target_key": dk,
                "source": nodes[sk]["id"],
                "target": nodes[dk]["id"],
                "source_display": nodes[sk]["display"],
                "target_display": nodes[dk]["display"],
                "value_lb": value,
                "evidence": rec["evidence"],
                "channel_bias": "audited",
                "from_corrupted": flagged,
            }
        )

    constraints = []
    for key, row in eod.items():
        constraints.append(
            {
                "node": nodes[key]["id"],
                "display": nodes[key]["display"],
                "law": "mass_balance",
                "in_lb": row["in_lb"],
                "out_lb": row["out_lb"],
                "inventory_eod_lb": row["inventory_eod_lb"],
                "evidence": row["evidence"],
            }
        )

    return {
        "agent": NAME,
        "role": ROLE,
        "nodes": list(nodes.values()),
        "edges": out_edges,
        "trusted_constraint_rows": constraints,
    }


def _acc(edges, sk, dk, lb, mid, kind, corrupted: bool = False):
    key = (sk, dk)
    mention = {"value_lb": lb, "corrupted": corrupted, "mid": mid, "kind": kind}
    if key not in edges:
        edges[key] = {"mentions": [mention], "evidence": [mid], "by_kind": {kind: [lb]}}
        return
    edges[key]["mentions"].append(mention)
    edges[key]["evidence"].append(mid)
    edges[key]["by_kind"].setdefault(kind, []).append(lb)
