"""Agent Auditor — conservation-shaped transfers and books."""

from __future__ import annotations

from .extract import GraphBuilder

NAME = "auditor"
ROLE = "schema validator / conservation-law tagger"


def extract(messages: list[dict], topology: dict | None = None) -> dict:
    g = GraphBuilder()
    g.ingest_text(
        messages,
        outbound=True,
        inbound=True,
        also_sent=True,
        eod=True,
        skip_suspicious=True,
    )
    g.ingest_structured(messages, topology)
    return g.finish(NAME, ROLE)
