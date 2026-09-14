"""
orb/_ingest_utils.py
--------------------
Shared utilities used by both ingest mode modules to avoid circular imports.
"""

from __future__ import annotations

import re

# ── leakage guard ─────────────────────────────────────────────────────────────

_LEAKAGE_RE = re.compile(
    r"leak|fault|anomal|corrupt|error|label|ground.truth|\bis_",
    re.I,
)

VALID_TYPES = {"node", "edge", "aggregate"}


def leakage_guard(names: list[str]) -> tuple[list[str], list[dict]]:
    """
    Partition *names* (column/channel names) into (clean, exclusions).
    exclusions: list of {name, reason} dicts for the ingest report.
    """
    clean: list[str] = []
    exclusions: list[dict] = []
    for n in names:
        if _LEAKAGE_RE.search(n):
            exclusions.append({
                "name":   n,
                "reason": f"matches leakage pattern: {_LEAKAGE_RE.pattern}",
            })
        else:
            clean.append(n)
    return clean, exclusions


def validate_claims(claims: list[dict]) -> None:
    """Raise ValueError if any claim type is not in VALID_TYPES."""
    for c in claims:
        t = c.get("type", "")
        if t not in VALID_TYPES:
            raise ValueError(
                f"Invalid claim type {t!r} in claim {c.get('id')!r}. "
                f"Valid types: {sorted(VALID_TYPES)}"
            )
