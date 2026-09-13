"""Agent Scout — outbound / link-flow claims and named places."""

from __future__ import annotations

from .extract import GraphBuilder

NAME = "scout"
ROLE = "entity extraction / outbound and link-flow claims"


def extract(messages: list[dict], topology: dict | None = None) -> dict:
    g = GraphBuilder()
    g.ingest_text(messages, outbound=True, also_sent=True)
    g.ingest_structured(messages, topology)
    return g.finish(NAME, ROLE)
