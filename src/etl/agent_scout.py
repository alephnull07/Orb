"""Agent Scout — outbound claims and named places.

Trusts 'sending X lb from A to B' (and similar). Gullible about extra
outbound boasts, which is why the other two agents exist.
"""

from __future__ import annotations

from .textutil import (
    ALSO_SENT_RE,
    SEND_RE,
    SUSPICIOUS,
    canon,
    is_site_name,
    pick_reported_weight,
    slug,
    speaker_stations,
)


NAME = "scout"
ROLE = "entity extraction / outbound transfer claims"


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

    stations = speaker_stations(messages)
    for m in messages:
        text = m.get("raw_text") or ""
        mid = m.get("message_id")
        for match in SEND_RE.finditer(text):
            lb, src, dst = int(match.group(1)), match.group(2).strip(), match.group(3).strip()
            if not is_site_name(src) or not is_site_name(dst):
                continue
            sk, dk = touch(src), touch(dst)
            _put_edge(edges, sk, dk, lb, mid, "send", bool(m.get("corrupted")))
        if SUSPICIOUS.search(text):
            continue
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
            _put_edge(edges, sk, dk, int(match.group(1)), mid, "also_sent", bool(m.get("corrupted")))

    return {
        "agent": NAME,
        "role": ROLE,
        "nodes": list(nodes.values()),
        "edges": [
            {
                "source_key": sk,
                "target_key": dk,
                "source": nodes[sk]["id"],
                "target": nodes[dk]["id"],
                "source_display": nodes[sk]["display"],
                "target_display": nodes[dk]["display"],
                "value_lb": rec["value_lb"],
                "evidence": rec["evidence"],
                "channel_bias": rec["bias"],
                "from_corrupted": rec["from_corrupted"],
            }
            for (sk, dk), rec in edges.items()
        ],
    }


def _put_edge(edges, sk, dk, lb, mid, bias, corrupted: bool = False):
    key = (sk, dk)
    mention = {"value_lb": lb, "corrupted": corrupted, "mid": mid}
    if key not in edges:
        edges[key] = {"mentions": [mention], "evidence": [mid], "bias": bias}
    else:
        edges[key]["mentions"].append(mention)
        edges[key]["evidence"].append(mid)
        edges[key]["bias"] = bias
    value, flagged = pick_reported_weight(edges[key]["mentions"])
    edges[key]["value_lb"] = value
    edges[key]["from_corrupted"] = flagged
