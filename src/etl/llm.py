"""Anthropic Claude (and optional Ollama) LLM extraction for the 3 ETL personas.

Used when --llm is set. Runs Scout, Receiver, and Auditor with tailored
prompts to extract structured nodes, edges, weights, and evidence.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .textutil import canon, corpus_text, is_site_name, slug

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def get_api_key(explicit_key: str | None = None) -> str:
    """Find Anthropic API key from explicit arg, environment, or .env file."""
    if explicit_key:
        return explicit_key.strip()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"].strip()

    # Search upwards for .env
    cur = Path(__file__).resolve().parent
    for _ in range(5):
        env_file = cur / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    k = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if k:
                        return k
        if cur.parent == cur:
            break
        cur = cur.parent
    return ""


PERSONAS = {
    "scout": (
        "You are Scout, an entity-extraction and outbound transfer analyst.\n"
        "Your mission:\n"
        "1. Extract physical supply sites only (pads, hubs, clinics, LZs). "
        "Never create a node for a person, callsign, Watchtower, TOC, or HQ.\n"
        "2. Extract OUTBOUND claims of the form 'sending/sent X lb from A to B' where A and B are both physical sites.\n"
        "   - Ignore 'also sent X over to Y' unless both A and Y already appear as named places in a from/to sending line.\n"
        "   - Ignore second drops, yesterday/OCR, kilograms, and 'double it on the board'.\n"
        "   - Do NOT treat the radio addressee (to: Watchtower) as a destination site.\n"
        "   - Do NOT record inbound receipts ('got X lb off A here at B') — that is Receiver's job.\n"
    ),
    "receiver": (
        "You are Receiver, an inbound logistics receipt specialist.\n"
        "Your mission:\n"
        "1. Extract every unique site (node) mentioned.\n"
        "2. Extract every INBOUND arrival receipt ('got X pounds/lb off A here at B', 'received X lb off A at B').\n"
        "   - Edge source is A, edge target is B, value_lb is X.\n"
        "   - Do NOT record outbound dispatch boasts ('sending X lb from A to B') — that is Scout's job.\n"
    ),
    "auditor": (
        "You are Auditor, an operational supply auditor.\n"
        "Your mission:\n"
        "1. Extract every unique site (node) mentioned.\n"
        "2. Extract all mass-balance EOD reports ('EOD Place: in X lb, out Y lb, on-hand Z lb').\n"
        "3. Cross-reference all field transfer claims. If a transfer has conflicting numbers between sender and receiver or runner dispatches, record the edge and capture the active reported/disputed transfer observation.\n"
    ),
}


def claude_generate(prompt: str, model: str = DEFAULT_CLAUDE_MODEL, api_key: str | None = None) -> str:
    key = get_api_key(api_key)
    if not key:
        raise ValueError("No Anthropic API key found. Set ANTHROPIC_API_KEY or provide in .env")

    body = {
        "model": model,
        "max_tokens": 4000,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }

    req = urllib.request.Request(
        DEFAULT_ANTHROPIC_URL,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        data=json.dumps(body).encode("utf-8"),
        method="POST",
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["content"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Claude API HTTP {e.code}: {err_body}") from e
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            raise e
    return ""


def extract_llm(
    agent: str,
    messages: list[dict],
    host: str = "",
    model: str | None = None,
    api_key: str | None = None,
) -> dict | None:
    text = corpus_text(messages)
    chosen_model = model or DEFAULT_CLAUDE_MODEL

    prompt = f"""{PERSONAS[agent]}

Read these field communications and output ONLY valid JSON adhering to this schema:
{{
  "nodes": [{{"display": "Full Site Name", "type": "source|junction|hub|sink|unknown"}}],
  "edges": [{{"source": "Full Source Name", "target": "Full Target Name", "value_lb": 123, "evidence": ["MSG-001"]}}],
  "eod": [{{"display": "Full Site Name", "in_lb": 100, "out_lb": 50, "inventory_eod_lb": 50, "evidence": "MSG-010"}}]
}}

Field Communications:
{text}
"""

    try:
        blob = claude_generate(prompt, model=chosen_model, api_key=api_key)
    except Exception as e:
        print(f"  [warn] Claude call failed for {agent}: {e}")
        return None

    match = re.search(r"\{.*\}", blob, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    nodes = {}
    for n in data.get("nodes") or []:
        display = (n.get("display") or n.get("id") or "").strip()
        if not display or not is_site_name(display):
            continue
        key = canon(display)
        nodes[key] = {
            "id": slug(display),
            "display": display,
            "aliases": [display],
            "key": key,
            "type": n.get("type", "unknown"),
        }

    edges = []
    for e in data.get("edges") or []:
        src = (e.get("source") or e.get("source_display") or "").strip()
        dst = (e.get("target") or e.get("target_display") or "").strip()
        try:
            lb = int(e.get("value_lb"))
        except (TypeError, ValueError):
            continue
        if not src or not dst or not is_site_name(src) or not is_site_name(dst):
            continue
        sk, dk = canon(src), canon(dst)
        if sk not in nodes:
            nodes[sk] = {"id": slug(src), "display": src, "aliases": [src], "key": sk}
        if dk not in nodes:
            nodes[dk] = {"id": slug(dst), "display": dst, "aliases": [dst], "key": dk}

        ev = e.get("evidence") or []
        if isinstance(ev, str):
            ev = [ev]

        edges.append(
            {
                "source_key": sk,
                "target_key": dk,
                "source": nodes[sk]["id"],
                "target": nodes[dk]["id"],
                "source_display": nodes[sk]["display"],
                "target_display": nodes[dk]["display"],
                "value_lb": lb,
                "evidence": ev,
                "channel_bias": f"claude:{agent}",
            }
        )

    constraints = []
    for row in data.get("eod") or []:
        place = (row.get("display") or row.get("node") or "").strip()
        if not place:
            continue
        pk = canon(place)
        if pk not in nodes:
            nodes[pk] = {"id": slug(place), "display": place, "aliases": [place], "key": pk}
        try:
            constraints.append(
                {
                    "node": nodes[pk]["id"],
                    "display": nodes[pk]["display"],
                    "law": "mass_balance",
                    "in_lb": int(row.get("in_lb", 0)),
                    "out_lb": int(row.get("out_lb", 0)),
                    "inventory_eod_lb": int(row.get("inventory_eod_lb", row.get("on_hand_lb", 0))),
                    "evidence": row.get("evidence", ""),
                }
            )
        except (ValueError, TypeError):
            continue

    return {
        "agent": agent,
        "role": f"claude:{agent}",
        "nodes": list(nodes.values()),
        "edges": edges,
        "trusted_constraint_rows": constraints,
    }
