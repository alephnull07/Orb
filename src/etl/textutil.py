from __future__ import annotations

import json
import re
from pathlib import Path

SEND_RE = re.compile(
    r"(?:sending|sent)\s+(\d+)\s*(?:lb|lbs|pounds)\s+from\s+(.+?)\s+to\s+(.+?)(?:\.|,|count|;|$)",
    re.IGNORECASE,
)
GOT_RE = re.compile(
    r"got\s+(\d+)\s*(?:pounds|lbs|lb)\s+off\s+(.+?)\s+here at\s+(.+?)(?:\.|,|logged|;|$)",
    re.IGNORECASE,
)
EOD_RE = re.compile(
    r"EOD\s+(.+?):\s*in\s+(\d+)\s*lb,\s*out\s+(\d+)\s*lb,\s*on-hand\s+(\d+)\s*lb",
    re.IGNORECASE,
)
ALSO_SENT_RE = re.compile(
    r"also sent\s+(\d+)\s*(?:lbs|lb|pounds|kg)\s+over to\s+(.+?)(?:,|\.|$)",
    re.IGNORECASE,
)
# Domain-agnostic transfer language (any unit, any conserved commodity).
TRANSFER_RE = re.compile(
    r"(?:sending|sent|shipping|pumping|moving|transfer(?:red|ring)?|flowing)\s+"
    r"(\d+(?:\.\d+)?)\s*(?:lb|lbs|pounds|kg|cases?|m3/?h|cmh|gpm|lps)?\s+"
    r"from\s+(.+?)\s+to\s+(.+?)(?:\.|,|count|;|$)",
    re.IGNORECASE,
)
FLOW_ARROW_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:m3/?h|cmh|lb|lbs|pounds|kg|cases?)?\s+"
    r"(.+?)\s*(?:-->|->|→)\s+(.+?)(?:\.|,|;|$)",
    re.IGNORECASE,
)
# "Pushed 220 cases to Crescent" / "Sent 150 over to Echo" (source = speaker site).
SENT_TO_RE = re.compile(
    r"(?:sent|pushed)\s+(\d+(?:\.\d+)?)\s*(?:lb|lbs|pounds|kg|cases?)?\s+"
    r"(?:forward\s+|over\s+)?to\s+(?:the\s+)?(.+?)(?:\s+on\b|\s+site\b|[.,;]|$)",
    re.IGNORECASE,
)
# "Received 220 from Alpha" / "took 600 from Depot-Main" / "confirms 150 in from Bravo".
RECEIVED_FROM_RE = re.compile(
    r"(?:received|got|took|confirms)\s+(\d+(?:\.\d+)?)\s*(?:lb|lbs|pounds|kg|cases?)?\s+"
    r"(?:in\s+)?from\s+(?:the\s+)?(.+?)(?:[.,;]|$)",
    re.IGNORECASE,
)
# "Departed the depot for Alpha with 800 cases"
DEPARTED_FOR_RE = re.compile(
    r"departed\s+(?:the\s+)?(.+?)\s+for\s+(.+?)\s+with\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# "Rolling out of MAIN toward Bravo, 600 cases"
ROLLING_RE = re.compile(
    r"rolling out of\s+(\S+)\s+toward\s+([^,]+),\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# "Alpha EOD on hand 1300" / "EOD depot on hand 2300" / "Crescent EOD 340 on hand"
EOD_ONHAND_RE = re.compile(
    r"(?:^|\n)(?:EOD\s+(.+?)\s+on hand\s+(\d+)|(.+?)\s+EOD(?:\s+on hand)?\s+(\d+)(?:\s+on hand)?)",
    re.IGNORECASE,
)
SUSPICIOUS = re.compile(
    r"(yesterday|smudged|second drop|double it|\bkg\b|do not know if this is today)",
    re.IGNORECASE,
)
# Radio addressees, vehicles, and invented person-sites are not supply nodes.
NON_SITE = re.compile(
    r"^(watchtower|toc|hq|all|unknown|site of\b|convoy\b|driver\b)",
    re.IGNORECASE,
)

# Facility-type words. Shared prefixes like "FOB" / "OP" are not identity.
_FACILITY_WORDS = {
    "fob", "op", "ops", "camp", "base", "site", "depot", "outpost",
    "junction", "node", "hub", "cache", "yard", "station", "post",
    "checkpoint", "warehouse", "facility", "terminal", "reservoir",
    "plant", "pump", "origin", "point", "anchor",
}
_MODIFIERS = {
    "main", "the", "central", "old", "new", "fwd", "forward", "primary",
}


def is_site_name(name: str) -> bool:
    return bool(name and name.strip()) and not NON_SITE.search(canon(name))


def site_core(name: str) -> str:
    """Distinctive identity of a site after stripping facility-type affixes.

    ``FOB Alpha`` and ``Alpha-1`` share core ``alpha`` (true aliases).
    ``FOB Alpha`` and ``FOB Bravo`` do not (``alpha`` vs ``bravo``).
    ``Junction 10`` keeps its number so it does not merge with ``Junction 11``.
    """
    key = canon(name)
    toks = [t for t in key.split() if t and t not in {"the", "a", "an"}]
    if not toks:
        return key

    def _proper(ts: list[str]) -> bool:
        return any(
            t not in _FACILITY_WORDS and t not in _MODIFIERS and not t.isdigit()
            for t in ts
        )

    # Strip callsign numbers ("alpha 1") but keep "junction 10".
    if len(toks) >= 2 and toks[-1].isdigit() and _proper(toks[:-1]):
        toks = toks[:-1]

    while len(toks) > 1 and toks[-1] in _FACILITY_WORDS | _MODIFIERS and _proper(toks[:-1]):
        toks = toks[:-1]
    while len(toks) > 1 and toks[0] in _FACILITY_WORDS | _MODIFIERS and _proper(toks[1:]):
        toks = toks[1:]

    if not _proper(toks):
        if toks == ["main"] or (toks and toks[0] == "main" and all(
            t in _FACILITY_WORDS | _MODIFIERS for t in toks
        )):
            return "depot"
        for preferred in ("depot", "warehouse", "fob", "base", "camp", "hub", "cache"):
            if preferred in toks:
                return preferred
        return " ".join(toks)
    return " ".join(toks)


def speaker_stations(messages: list[dict]) -> dict[str, str]:
    """Map speaker name/callsign -> most common 'here at' site (physical only)."""
    from collections import Counter, defaultdict

    counts: dict[str, Counter] = defaultdict(Counter)
    for m in messages:
        text = m.get("raw_text") or ""
        keys = [m.get("from_name") or "", m.get("callsign") or ""]
        for match in GOT_RE.finditer(text):
            dst = match.group(3).strip()
            if not is_site_name(dst):
                continue
            ck = canon(dst)
            for k in keys:
                if k.strip():
                    counts[k.strip()][ck] += 2
                    counts[k.strip().lower()][ck] += 2
        for match in SEND_RE.finditer(text):
            src = match.group(2).strip()
            if not is_site_name(src):
                continue
            ck = canon(src)
            for k in keys:
                if k.strip():
                    counts[k.strip()][ck] += 1
                    counts[k.strip().lower()][ck] += 1
    out = {}
    for who, ctr in counts.items():
        if ctr:
            out[who] = ctr.most_common(1)[0][0]
    return out


def canon(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for junk in (" count it on arrival", " over", " logged"):
        if s.endswith(junk.strip()):
            s = s[: -len(junk.strip())].strip()
    return s


def slug(name: str) -> str:
    c = canon(name).replace(" ", "_")
    return c.upper()[:80] or "UNKNOWN"


def load_messages(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "messages.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_topology(dataset_dir: Path) -> dict:
    path = dataset_dir / "topology.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"links": {}, "nodes": []}


def as_number(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    try:
        n = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(n) if n.is_integer() else n


def resolve_sensor_edge(sensor: str, topology: dict, channel: str) -> tuple[str, str] | None:
    """Map a dataset sensor id onto a directed hop using official topology."""
    raw = (sensor or "").strip()
    if not raw:
        return None
    ch = (channel or "").lower()
    nodes = {str(n.get("nato")): n for n in (topology.get("nodes") or [])}
    if ch == "leak_demand":
        nid = raw.split("_")[-1]
        node = nodes.get(nid)
        display = node["display"] if node else f"Junction {nid}"
        return display, "Leak discharge"
    links = topology.get("links") or {}
    for key in (raw, raw.replace("Link_", "").replace("link_", ""), raw.replace("Pipe_", "").replace("P-", "")):
        if key in links:
            rec = links[key]
            return rec["source_name"], rec["target_name"]
    return None


def pick_reported_weight(mentions: list[dict]) -> tuple[int | float, bool]:
    """Choose a hop weight from extracted mentions.

    A mutated field report is marked ``corrupted`` on the message. That
    number is what the observation graph recorded, whether the lie is an
    inflation or a deflation. Fall back to the unanimous / median claim
    when nothing was rewritten.
    """
    if not mentions:
        return 0, False
    flagged = [m for m in mentions if m.get("corrupted")]
    if flagged:
        return as_number(flagged[-1]["value_lb"]) or 0, True
    vals = [as_number(m["value_lb"]) for m in mentions]
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0, False
    if len(set(vals)) == 1:
        return vals[0], False
    return sorted(vals)[len(vals) // 2], False


def corpus_text(messages: list[dict], topology: dict | None = None) -> str:
    parts = []
    for m in messages:
        extra = []
        if m.get("sensor") is not None:
            extra.append(f"sensor={m.get('sensor')}")
        if m.get("value") is not None:
            extra.append(f"value={m.get('value')}")
        head = (
            f"[{m.get('message_id')}] ({m.get('channel')}) "
            f"{m.get('from_name') or ''} / {m.get('callsign') or ''} -> {m.get('to') or ''}"
        )
        if extra:
            head += " " + " ".join(extra)
        parts.append(head + "\n" + (m.get("raw_text") or "") + "\n")
    if topology and topology.get("links"):
        parts.append("Official network schema (links):")
        seen = set()
        for key, rec in topology["links"].items():
            pair = (rec["source_name"], rec["target_name"], rec.get("pipe_id"))
            if pair in seen:
                continue
            seen.add(pair)
            parts.append(f"  {rec.get('pipe_id') or key}: {rec['source_name']} -> {rec['target_name']}")
        parts.append("")
    return "\n".join(parts)
