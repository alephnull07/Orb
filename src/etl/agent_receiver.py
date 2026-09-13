"""Agent Receiver — inbound receipts and measured arrivals."""

from __future__ import annotations

from .extract import GraphBuilder

NAME = "receiver"
ROLE = "topology / inbound receipt mapping"


def extract(messages: list[dict], topology: dict | None = None) -> dict:
    g = GraphBuilder()
    g.ingest_text(messages, inbound=True)
    g.ingest_structured(messages, topology)
    return g.finish(NAME, ROLE)
