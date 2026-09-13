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
SUSPICIOUS = re.compile(
    r"(yesterday|smudged|second drop|double it|\bkg\b|do not know if this is today)",
    re.IGNORECASE,
)
# Radio addressees and invented person-sites are not supply nodes.
NON_SITE = re.compile(
    r"^(watchtower|toc|hq|all|unknown|site of\b)",
    re.IGNORECASE,
)


def is_site_name(name: str) -> bool:
    return bool(name and name.strip()) and not NON_SITE.search(canon(name))


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
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    for junk in (" count it on arrival", " over", " logged"):
        if s.endswith(junk.strip()):
            s = s[: -len(junk.strip())].strip()
    return s.strip()


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


def pick_reported_weight(mentions: list[dict]) -> tuple[int, bool]:
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
        return int(flagged[-1]["value_lb"]), True
    vals = [int(m["value_lb"]) for m in mentions]
    if len(set(vals)) == 1:
        return vals[0], False
    return sorted(vals)[len(vals) // 2], False


def corpus_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        parts.append(
            f"[{m.get('message_id')}] ({m.get('channel')}) "
            f"{m.get('from_name')} / {m.get('callsign')} -> {m.get('to')}\n"
            f"{m.get('raw_text')}\n"
        )
    return "\n".join(parts)
