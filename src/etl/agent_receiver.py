"""Agent Receiver — inbound receipts and topology.

Only believes 'got X pounds off A here at B'. If Scout hallucinates a
ghost shipment with no receipt, Receiver never writes that edge.
"""

from __future__ import annotations

from .textutil import GOT_RE, canon, pick_reported_weight, slug


NAME = "receiver"
ROLE = "topology / inbound receipt mapping"


def extract(messages: list[dict]) -> dict:
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = {}

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

    for m in messages:
        text = m.get("raw_text") or ""
        mid = m.get("message_id")
        for match in GOT_RE.finditer(text):
            lb, src, dst = int(match.group(1)), match.group(2).strip(), match.group(3).strip()
            sk, dk = touch(src), touch(dst)
            key = (sk, dk)
            mention = {"value_lb": lb, "corrupted": bool(m.get("corrupted")), "mid": mid}
            if key not in edges:
                edges[key] = {"mentions": [mention], "evidence": [mid]}
            else:
                edges[key]["mentions"].append(mention)
                edges[key]["evidence"].append(mid)

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
                "channel_bias": "receipt",
                "from_corrupted": flagged,
            }
        )

    return {
        "agent": NAME,
        "role": ROLE,
        "nodes": list(nodes.values()),
        "edges": out_edges,
    }
