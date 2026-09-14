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

from .textutil import as_number, canon, corpus_text, is_site_name, slug

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


_SITE_IDENTITY = (
    "SITE IDENTITY (critical):\n"
    "- Each physically distinct place is its own node. Facility words (FOB, OP, camp, "
    "depot, junction, node, base, site) are types, not names.\n"
    "- FOB Alpha and FOB Bravo are TWO nodes. OP Crescent, OP Delta, and OP Echo are THREE.\n"
    "- Junction 10 and Junction 11 are TWO nodes. Do not merge sites that only share a prefix.\n"
    "- Aliases of the SAME place may merge: 'FOB Alpha', 'Alpha-1', 'Alpha depot' → one node.\n"
    "- People, callsigns, convoys, and vehicles are not nodes. Use the places they travel between.\n"
    "- Prefer the full site name from the log header (the token after the timestamp).\n"
    "- Opening counts are STARTING stock, not closeout. They are not EOD.\n"
    "- EOD / closeout / end-of-day on-hand is a separate measurement. Do not "
    "compute implied on-hand as opening minus shipments. If the log has no "
    "EOD or closeout, do not emit eod rows.\n"
)

PERSONAS = {
    "scout": (
        "You are Scout, a domain-agnostic outbound / link-flow extractor.\n"
        "Recover a conserved-flow network from whatever observations you are given "
        "(field chatter, SCADA rows, work orders). Commodity and unit are unknown.\n"
        + _SITE_IDENTITY +
        "1. Nodes are physical places or assets only — never people, callsigns, Watchtower, TOC, HQ, or 'ALL'.\n"
        "2. Record OUTBOUND or along-link quantity claims: A to B with a numeric value. "
        "Accept any phrasing and any unit (lb, cases, m3/h, raw sensor value).\n"
        "3. If a structured row has channel=flow and sensor=Link_N (or similar), and a network schema "
        "is provided, attach that value to the schema's endpoints for that link.\n"
        "4. Ignore stale/yesterday, unit-swap, and 'double it' injections.\n"
        "5. Do not invent hops that are not evidenced. Inbound-only receipts are Receiver's job.\n"
        "6. Separate shipments on the same hop stay as separate evidence; you may output one edge "
        "with the summed quantity if they are the same directed hop.\n"
    ),
    "receiver": (
        "You are Receiver, a domain-agnostic inbound / arrival extractor.\n"
        "Recover destinations and incoming quantities from any conserved-flow domain.\n"
        + _SITE_IDENTITY +
        "1. Nodes are physical places or assets only.\n"
        "2. Record INBOUND receipts: quantity arriving at B from A, any unit (including cases).\n"
        "3. Structured flow/leak_demand rows with a network schema are measured arrivals — record them.\n"
        "4. Do not record outbound-only boasts that have no arrival evidence, except structured link sensors.\n"
    ),
    "auditor": (
        "You are Auditor, a domain-agnostic conservation checker.\n"
        + _SITE_IDENTITY +
        "1. Extract every physical node. Do not drop a site just because it shares a prefix with another.\n"
        "2. Extract mass-balance / closeout rows (in, out, on-hand) in any unit. Each site's EOD stays on that site.\n"
        "3. Cross-reference every transfer or link-flow claim. If numbers disagree, keep the "
        "active reported observation (including a marked corrupted/rewritten report).\n"
        "4. Structured leak_demand > 0 is a real extra outflow to a leak sink.\n"
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
    topology: dict | None = None,
) -> dict | None:
    text = corpus_text(messages, topology=topology)
    chosen_model = model or DEFAULT_CLAUDE_MODEL

    prompt = f"""{PERSONAS[agent]}

Read these observations and output ONLY valid JSON adhering to this schema:
{{
  "nodes": [{{"display": "Full Site Name", "type": "source|junction|hub|sink|unknown"}}],
  "edges": [{{"source": "Full Source Name", "target": "Full Target Name", "value_lb": 123, "evidence": ["MSG-001"]}}],
  "eod": [{{"display": "Full Site Name", "in_lb": 100, "out_lb": 50, "inventory_eod_lb": 50, "evidence": "MSG-010"}}],
  "opening": [{{"display": "Full Site Name", "value_lb": 1000}}]
}}
value_lb is the numeric quantity in whatever unit the observations use (pounds, m3/h, etc.).
If a network schema is listed, bind link/flow sensors to those endpoints. Do not invent extra hops.
eod is ONLY an explicit closeout / EOD / end-of-day on-hand. Never invent it from opening minus flow.
opening is start-of-day / opening count. Omit eod entirely when the log has no closeout.

Observations:
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
        lb = as_number(e.get("value_lb"))
        if lb is None:
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
    body = "\n".join(
        ln for ln in text.splitlines() if not ln.strip().startswith("#")
    )
    has_closeout = bool(re.search(
        r"\beod\b|close[- ]?out|end of (the )?day|eod_on_hand",
        body,
        re.I,
    ))
    if has_closeout:
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

    openings = []
    for row in data.get("opening") or []:
        place = (row.get("display") or row.get("node") or "").strip()
        if not place:
            continue
        lb = as_number(row.get("value_lb", row.get("value")))
        if lb is None:
            continue
        pk = canon(place)
        if pk not in nodes:
            nodes[pk] = {"id": slug(place), "display": place, "aliases": [place], "key": pk}
        openings.append({"node": nodes[pk]["id"], "value": float(lb)})

    out = {
        "agent": agent,
        "role": f"claude:{agent}",
        "nodes": list(nodes.values()),
        "edges": edges,
        "trusted_constraint_rows": constraints,
    }
    if openings:
        out["openings"] = openings
    return out
