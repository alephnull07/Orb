#!/usr/bin/env python3
"""Generate messy people-as-sensor supply-drop corpora.

Each run samples a random 15–30 node deployment graph whose flows conserve
mass. Ollama (or a template fallback) writes radio/SMS/WhatsApp chatter so
later ETL agents have to recover nodes and edges from noise. A second corpus
copies the true dump and injects corrupt reports.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import shutil
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRUE_DIR = ROOT / "true"
CORRUPT_DIR = ROOT / "corrupted"
EVAL_DIR = ROOT / "eval"

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODELS = [
    os.environ.get("OLLAMA_MODEL", "").strip(),
    "llama3.2",
    "llama3.1",
    "llama3.2:1b",
    "qwen2.5:7b",
    "mistral",
    "phi3",
    "gemma2:2b",
]

NATO = [
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel",
    "India", "Juliet", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa",
    "Quebec", "Romeo", "Sierra", "Tango", "Uniform", "Victor", "Whiskey",
    "Xray", "Yankee", "Zulu",
    "Maple", "Cedar", "Quarry", "Harbor", "Overlook", "Sandpit", "Iron",
    "Copper", "Granite", "Willow",
]
PLACE_KINDS = [
    "pad", "well", "tent city", "clinic", "schoolhouse", "checkpoint",
    "market", "warehouse", "bridge", "farm", "quarry", "mill", "orchard",
    "crossroads", "mosque courtyard", "aid cage", "water point", "helipad",
]
SOURCE_KINDS = ["airdrop LZ", "FOB depot", "convoy origin", "port dump", "railhead"]
HUB_KINDS = ["ridge hub", "junction", "transfer yard", "forward cache"]
RANKS = ["PVT", "PFC", "SPC", "CPL", "SGT", "SSG", "SFC", "LT", "CPT"]
FIRST = [
    "Maya", "Diego", "Aisha", "Tom", "Priya", "James", "Leo", "Nina", "Elena",
    "Sam", "Chris", "Farah", "Omar", "Riley", "Keiko", "Noah", "Sofia", "Malik",
    "June", "Ibrahim", "Hana", "Victor", "Amelia", "Yusuf", "Greta",
]
LAST = [
    "Chen", "Ruiz", "Rahman", "Hale", "Nair", "Okonkwo", "Park", "Volkov",
    "Vasquez", "Ortiz", "Bennett", "Hassan", "Kowalski", "Abebe", "Nguyen",
    "Singh", "Diaz", "Okafor", "Berg", "Khan", "MacLeod", "Petrov", "Alvarez",
]
CHANNELS = ["radio", "sms", "whatsapp", "email", "field_log", "paper_ocr"]

MSG_RE = re.compile(
    r"<<<MSG\s*"
    r"channel:\s*(?P<channel>[^\n]+)\s*"
    r"from:\s*(?P<from_name>[^\n]+)\s*"
    r"callsign:\s*(?P<callsign>[^\n]+)\s*"
    r"to:\s*(?P<to>[^\n]+)\s*"
    r"text:\s*(?P<text>.*?)\s*"
    r">>>",
    re.IGNORECASE | re.DOTALL,
)


def http_json(url: str, payload: dict | None = None, timeout: int = 180) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_local_models(host: str) -> list[str]:
    try:
        tags = http_json(f"{host.rstrip('/')}/api/tags", timeout=5)
    except Exception:
        return []
    names = []
    for m in tags.get("models", []):
        name = m.get("name") or m.get("model")
        if name:
            names.append(name)
    return names


def pick_model(host: str, requested: str | None) -> str | None:
    available = list_local_models(host)
    if requested:
        return requested
    for cand in DEFAULT_MODELS:
        if not cand:
            continue
        for n in available:
            if n == cand or n.startswith(cand + ":") or cand in n:
                return n
    return available[0] if available else None


def ollama_generate(host: str, model: str, prompt: str, temperature: float) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 900},
    }
    out = http_json(f"{host.rstrip('/')}/api/generate", body, timeout=300)
    return (out.get("response") or "").strip()


def split_amount(total: int, k: int, rng: random.Random, step: int = 5) -> list[int]:
    """k positive parts summing to total, preferring multiples of step."""
    if k <= 0:
        return []
    if k == 1:
        return [total]
    units = total // step
    if units < k:
        step = 1
        units = total
    if units < k:
        parts = [1] * (k - 1)
        parts.append(total - (k - 1))
        return [max(p, 0) for p in parts]
    cuts = sorted(rng.sample(range(1, units), k - 1))
    parts = []
    prev = 0
    for c in cuts + [units]:
        parts.append((c - prev) * step)
        prev = c
    drift = total - sum(parts)
    parts[-1] += drift
    return parts


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Person:
    def __init__(self, rank: str, first: str, last: str, callsign: str, site_id: str | None, role: str):
        self.rank = rank
        self.first = first
        self.last = last
        self.callsign = callsign
        self.site_id = site_id
        self.role = role

    @property
    def name(self) -> str:
        if self.rank == "CIV":
            return f"{self.first} {self.last}"
        return f"{self.rank} {self.first} {self.last}"

    def blurb(self) -> str:
        where = self.site_id or "TOC"
        return f"{self.name} ({self.callsign}) — {self.role} at {where}"


def build_world(rng: random.Random, n_nodes: int) -> dict:
    n_nodes = max(15, min(30, n_nodes))
    n_sources = rng.randint(2, 3)
    n_hubs = rng.randint(3, 6)
    n_field = n_nodes - n_sources - n_hubs
    if n_field < 8:
        n_hubs = max(2, n_nodes - n_sources - 8)
        n_field = n_nodes - n_sources - n_hubs

    labels = rng.sample(NATO, n_nodes)
    nodes = []
    idx = 0

    def add_node(kind_pool: list[str], ntype: str, layer: int) -> dict:
        nonlocal idx
        letter = labels[idx]
        idx += 1
        kind = rng.choice(kind_pool)
        nick = f"{letter} {kind}"
        node = {
            "id": f"SITE_{letter.upper()}",
            "nato": letter,
            "type": ntype,
            "layer": layer,
            "kind": kind,
            "display": nick,
            "aliases": [f"Point {letter[0]}", nick, f"Anchor-{letter[0]}", letter],
        }
        if ntype == "source":
            node["id"] = f"SRC_{letter.upper()}"
            node["aliases"] = [nick, letter, f"{kind} {letter}"]
        elif ntype == "hub":
            node["id"] = f"HUB_{letter.upper()}"
        nodes.append(node)
        return node

    sources = [add_node(SOURCE_KINDS, "source", 0) for _ in range(n_sources)]
    hubs = [add_node(HUB_KINDS, "hub", 1) for _ in range(n_hubs)]
    fields = [add_node(PLACE_KINDS, "sink", 2) for _ in range(n_field)]
    by_id = {n["id"]: n for n in nodes}

    edges: list[dict] = []
    clock = datetime(2026, 9, 12, 6, 10, tzinfo=timezone.utc)

    def add_edge(src: dict, dst: dict, value: int, minutes: int) -> dict:
        nonlocal clock
        clock += timedelta(minutes=minutes)
        e = {
            "source": src["id"],
            "target": dst["id"],
            "value_lb": int(value),
            "time": iso(clock),
            "source_name": src["display"],
            "target_name": dst["display"],
        }
        edges.append(e)
        return e

    # Sources inject supply, then ship all of it into hubs (and maybe a field).
    injections = {}
    for src in sources:
        injections[src["id"]] = rng.randrange(80, 401, 10)

    for src in sources:
        amt = injections[src["id"]]
        k = rng.randint(1, min(3, len(hubs)))
        dests = rng.sample(hubs, k)
        parts = split_amount(amt, k, rng)
        for dst, part in zip(dests, parts):
            if part > 0:
                add_edge(src, dst, part, rng.randint(8, 18))

    incoming: dict[str, int] = defaultdict(int)
    outgoing: dict[str, int] = defaultdict(int)
    for e in edges:
        incoming[e["target"]] += e["value_lb"]
        outgoing[e["source"]] += e["value_lb"]

    # Hubs keep little/none and fan out to fields (and sometimes another hub).
    hub_order = list(hubs)
    rng.shuffle(hub_order)
    for i, hub in enumerate(hub_order):
        have = incoming[hub["id"]]
        if have <= 0:
            # isolated hub: siphon a bit from a source via extra edge
            src = rng.choice(sources)
            extra = rng.randrange(20, 81, 5)
            injections[src["id"]] += extra
            add_edge(src, hub, extra, rng.randint(5, 12))
            incoming[hub["id"]] += extra
            outgoing[src["id"]] += extra
            have = incoming[hub["id"]]
        keep = 0 if rng.random() < 0.7 else rng.randrange(0, min(20, have) + 1, 5)
        remaining = have - keep
        dest_pool = list(fields)
        if i < len(hub_order) - 1 and rng.random() < 0.35:
            dest_pool = dest_pool + [hub_order[i + 1]]
        k = rng.randint(2, min(5, len(dest_pool)))
        dests = rng.sample(dest_pool, k)
        parts = split_amount(remaining, k, rng) if remaining else []
        for dst, part in zip(dests, parts):
            if part > 0:
                add_edge(hub, dst, part, rng.randint(6, 16))
                incoming[dst["id"]] += part
                outgoing[hub["id"]] += part

    for site in [f for f in fields if incoming[f["id"]] == 0]:
        gift = rng.randrange(15, 46, 5)
        hub = rng.choice(hubs)
        leftover = incoming[hub["id"]] - outgoing[hub["id"]]
        if leftover >= gift:
            add_edge(hub, site, gift, rng.randint(6, 16))
            incoming[site["id"]] += gift
            outgoing[hub["id"]] += gift
        else:
            src = rng.choice(sources)
            injections[src["id"]] += gift
            add_edge(src, site, gift, rng.randint(6, 16))
            incoming[site["id"]] += gift
            outgoing[src["id"]] += gift

    # Field-to-field: some sites act as sensors AND forwarders (the A/B/C pattern, scaled).
    donors = [f for f in fields if incoming[f["id"]] >= 25]
    rng.shuffle(donors)
    n_fwd = 0
    if donors:
        hi = min(len(donors), max(1, n_field // 2 + 3))
        lo = min(hi, max(1, n_field // 4))
        n_fwd = rng.randint(lo, hi)
    for donor in donors[:n_fwd]:
        have = incoming[donor["id"]] - outgoing[donor["id"]]
        if have < 20:
            continue
        send = rng.randrange(5, min(have - 5, 60) + 1, 5)
        if send <= 0 or send >= have:
            continue
        receivers = [f for f in fields if f["id"] != donor["id"]]
        k = 1 if rng.random() < 0.55 else 2
        recs = rng.sample(receivers, min(k, len(receivers)))
        parts = split_amount(send, len(recs), rng)
        for rec, part in zip(recs, parts):
            if part > 0:
                add_edge(donor, rec, part, rng.randint(10, 22))
                incoming[rec["id"]] += part
                outgoing[donor["id"]] += part
        donor["type"] = "junction"

    inventory = {}
    for n in nodes:
        nid = n["id"]
        if n["type"] == "source":
            inventory[nid] = injections[nid] - outgoing[nid]
        else:
            inventory[nid] = incoming[nid] - outgoing[nid]
        if inventory[nid] < 0:
            raise RuntimeError(f"negative inventory at {nid}")

    # People: one primary + sometimes a civilian at sinks; TOC watchstander extra.
    used_names: set[tuple[str, str]] = set()
    people: list[Person] = []
    people_at: dict[str, list[Person]] = defaultdict(list)

    def mint_person(site: dict | None, role: str, civilian: bool = False) -> Person:
        while True:
            first, last = rng.choice(FIRST), rng.choice(LAST)
            if (first, last) not in used_names:
                used_names.add((first, last))
                break
        if civilian:
            call = f"CIV-{first[:3].upper()}"
            p = Person("CIV", first, last, call, site["id"] if site else None, role)
        else:
            letter = (site["nato"][0] if site else "W")
            call = f"Anchor-{letter}{rng.randint(1, 9)}" if site else "Watchtower"
            if site and site["type"] == "source":
                call = f"Wagon-{rng.randint(1, 6)}" if "convoy" in site["kind"] or "FOB" in site["kind"] else f"Dustoff-{letter}"
            if site and site["type"] == "hub":
                call = f"{site['nato']}-Actual"
            rank = "CIV" if civilian else rng.choice(RANKS)
            p = Person(rank, first, last, call, site["id"] if site else None, role)
        people.append(p)
        if site:
            people_at[site["id"]].append(p)
        return p

    toc = mint_person(None, "TOC collector")
    toc.callsign = "Watchtower"
    toc.rank = "MAJ"
    spare = mint_person(None, "spare radio hand")
    spare.callsign = "Spare-2"
    for n in nodes:
        mint_person(n, f"site reporter at {n['display']}")
        if n["type"] in {"sink", "junction"} and rng.random() < 0.25:
            mint_person(n, "civilian aid worker", civilian=True)

    trusted = []
    for n in nodes:
        nid = n["id"]
        row = {
            "node": nid,
            "display": n["display"],
            "law": "mass_balance",
            "in_lb": injections[nid] if n["type"] == "source" else incoming[nid],
            "out_lb": outgoing[nid],
            "inventory_eod_lb": inventory[nid],
        }
        trusted.append(row)

    graph = {
        "scenario": "supply_drops_people_as_sensors",
        "seed_nodes": n_nodes,
        "window": {
            "date": "2026-09-12",
            "start": "2026-09-12T06:00:00Z",
            "end": "2026-09-12T18:00:00Z",
            "commodity": "relief_supplies",
            "unit": "lb",
        },
        "notes": (
            "Mass is conserved at every node. People reports are sensors; "
            "conservation equalities are trusted parity rows. Node count is "
            f"sampled uniformly from 15–30 (this run: {len(nodes)})."
        ),
        "nodes": nodes,
        "edges": edges,
        "injections_lb": injections,
        "inventory_eod_lb": inventory,
        "trusted_constraint_rows": trusted,
        "people": [
            {
                "name": p.name,
                "callsign": p.callsign,
                "site_id": p.site_id,
                "role": p.role,
            }
            for p in people
        ],
    }
    return {
        "graph": graph,
        "nodes": nodes,
        "by_id": by_id,
        "edges": edges,
        "people": people,
        "people_at": people_at,
        "toc": toc,
        "spare": spare,
        "incoming": dict(incoming),
        "outgoing": dict(outgoing),
        "inventory": inventory,
        "injections": injections,
    }


def chunked(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def edge_fact(world: dict, e: dict) -> str:
    src = world["by_id"][e["source"]]["display"]
    dst = world["by_id"][e["target"]]["display"]
    return f"{src} gave {e['value_lb']} lb of supplies to {dst}."


def make_scenes(world: dict) -> list[dict]:
    edges = world["edges"]
    scenes = []
    day = datetime(2026, 9, 12, 6, 0, tzinfo=timezone.utc)

    grouped = defaultdict(list)
    for e in edges:
        src_t = world["by_id"][e["source"]]["type"]
        if src_t == "source":
            grouped["source_push"].append(e)
        elif src_t == "hub":
            grouped["hub_fanout"].append(e)
        else:
            grouped["field_fwd"].append(e)

    flavor = {
        "source_push": "engines, pallet stencils, people arguing lbs vs kg",
        "hub_fanout": "forklifts, dusty yard, overlapping radio",
        "field_fwd": "pickups, runners, schoolhouses and wells, short texts",
    }
    clocks = {
        "source_push": day + timedelta(minutes=15),
        "hub_fanout": day + timedelta(hours=3),
        "field_fwd": day + timedelta(hours=7),
    }
    for kind, batch in grouped.items():
        for i, group in enumerate(chunked(batch, 4)):
            must = [edge_fact(world, e) for e in group]
            must.append("Do not invent any other pound totals or extra routes.")
            scenes.append(
                {
                    "id": f"{kind}_{i+1}",
                    "clock": iso(clocks[kind] + timedelta(minutes=20 * i)),
                    "flavor": flavor[kind],
                    "must": must,
                    "edges": group,
                    "kind": kind,
                }
            )

    # Closeouts in batches so the model is not asked to recite 30 lines at once.
    non_src = [n for n in world["nodes"] if n["type"] != "source"]
    for i, group in enumerate(chunked(non_src, 6)):
        must = []
        for n in group:
            inv = world["inventory"][n["id"]]
            inn = world["injections"][n["id"]] if n["type"] == "source" else world["incoming"].get(n["id"], 0)
            out = world["outgoing"].get(n["id"], 0)
            must.append(
                f"{n['display']} EOD: received {inn} lb, sent {out} lb, on-hand {inv} lb."
            )
        must.append("No extra deliveries. Numbers must match those lines exactly.")
        scenes.append(
            {
                "id": f"eod_{i+1}",
                "clock": iso(day + timedelta(hours=10, minutes=15 * i)),
                "flavor": "tired closeouts, one garbled handwritten log",
                "must": must,
                "edges": [],
                "kind": "eod",
                "closeout_nodes": group,
            }
        )

    legal_nums = sorted({e["value_lb"] for e in edges} | set(world["inventory"].values()))
    scenes.append(
        {
            "id": "chatter",
            "clock": iso(day + timedelta(hours=6, minutes=40)),
            "flavor": "dust jokes, stepped-on net, leaking roofs, no new math",
            "must": [
                "Side chatter only. If a quantity appears it MUST be one of: "
                + ", ".join(str(x) for x in legal_nums[:40])
                + ".",
                "Do not invent a new transfer.",
            ],
            "edges": [],
            "kind": "chatter",
        }
    )
    return scenes


def people_for_scene(world: dict, scene: dict) -> list[Person]:
    wanted: list[Person] = [world["toc"], world["spare"]]
    seen = {id(world["toc"]), id(world["spare"])}
    ids = set()
    for e in scene.get("edges") or []:
        ids.add(e["source"])
        ids.add(e["target"])
    for n in scene.get("closeout_nodes") or []:
        ids.add(n["id"])
    if not ids:
        ids = {n["id"] for n in world["nodes"][:8]}
    for sid in ids:
        for p in world["people_at"].get(sid, []):
            if id(p) not in seen:
                wanted.append(p)
                seen.add(id(p))
    return wanted


def scene_prompt(world: dict, scene: dict, extra: str = "") -> str:
    must = "\n".join(f"- {x}" for x in scene["must"])
    people = "\n".join(f"- {p.blurb()}" for p in people_for_scene(world, scene))
    return f"""You write messy field comms for a 12 Sep 2026 humanitarian/military supply day.
This is NOT a clean log. People talk like people: typos, lowercase texts, radio brevity, overlapping chatter, nicknames.

People in this burst:
{people}

Scene time (approx): {scene['clock']}
Mood: {scene['flavor']}

HARD FACTS — if a number of pounds is mentioned it MUST match these. Never invent another transfer or another total.
{must}

Write 5 to 9 messages. Mix channels. Some are replies. One can be garbled.
Use EXACTLY this wrapper for every message, no other commentary:

<<<MSG
channel: radio|sms|whatsapp|email|field_log|paper_ocr
from: Full Name
callsign: CALLSIGN or -
to: name or ALL
text: the message body, 1-6 sentences, can include line breaks
>>>

{extra}
"""


def parse_messages(blob: str) -> list[dict]:
    found = []
    for m in MSG_RE.finditer(blob):
        text = m.group("text").strip()
        if not text:
            continue
        channel = m.group("channel").strip().lower().split()[0]
        if channel not in CHANNELS:
            channel = "sms"
        found.append(
            {
                "channel": channel,
                "from_name": m.group("from_name").strip(),
                "callsign": m.group("callsign").strip(),
                "to": m.group("to").strip(),
                "raw_text": text,
            }
        )
    return found


def fallback_thread(world: dict, scene: dict, rng: random.Random) -> list[dict]:
    msgs = []
    ch_cycle = ["radio", "sms", "whatsapp", "field_log", "email"]
    if scene["kind"] in {"source_push", "hub_fanout", "field_fwd"}:
        for i, e in enumerate(scene["edges"]):
            src_p = world["people_at"][e["source"]][0]
            dst_p = world["people_at"][e["target"]][0]
            src_n = world["by_id"][e["source"]]["display"]
            dst_n = world["by_id"][e["target"]]["display"]
            lb = e["value_lb"]
            msgs.append(
                {
                    "channel": ch_cycle[i % 3],
                    "from_name": src_p.name,
                    "callsign": src_p.callsign,
                    "to": dst_p.callsign,
                    "raw_text": (
                        f"{dst_p.callsign} this is {src_p.callsign}. sending {lb} lb "
                        f"from {src_n} to {dst_n}. count it on arrival. over"
                    ),
                }
            )
            msgs.append(
                {
                    "channel": ch_cycle[(i + 1) % 5],
                    "from_name": dst_p.name,
                    "callsign": dst_p.callsign,
                    "to": world["toc"].callsign,
                    "raw_text": (
                        f"got {lb} pounds off {src_n} here at {dst_n}. "
                        f"logged. {dst_p.last.lower()}"
                    ),
                }
            )
    elif scene["kind"] == "eod":
        for n in scene.get("closeout_nodes") or []:
            p = world["people_at"][n["id"]][0]
            inn = world["incoming"].get(n["id"], 0) or world["injections"].get(n["id"], 0)
            out = world["outgoing"].get(n["id"], 0)
            inv = world["inventory"][n["id"]]
            msgs.append(
                {
                    "channel": rng.choice(["field_log", "sms", "email"]),
                    "from_name": p.name,
                    "callsign": p.callsign,
                    "to": world["toc"].callsign,
                    "raw_text": (
                        f"EOD {n['display']}: in {inn} lb, out {out} lb, on-hand {inv} lb. "
                        f"{p.callsign} closing books."
                    ),
                }
            )
    else:
        p = rng.choice(world["people"])
        site = world["by_id"][p.site_id]["display"] if p.site_id else "TOC"
        msgs.append(
            {
                "channel": "radio",
                "from_name": world["spare"].name,
                "callsign": world["spare"].callsign,
                "to": "ALL",
                "raw_text": (
                    f"uh ignore, stepped on the net. dusty at {site}. "
                    f"no extra pallets, just looking for the toyota."
                ),
            }
        )
        msgs.append(
            {
                "channel": "whatsapp",
                "from_name": p.name,
                "callsign": p.callsign,
                "to": world["toc"].name,
                "raw_text": f"roof still leaking at {site} but the load is dry. no new numbers.",
            }
        )
    rng.shuffle(msgs)
    return msgs


def generate_scene(
    world: dict,
    scene: dict,
    host: str | None,
    model: str | None,
    fallback: bool,
    rng: random.Random,
) -> list[dict]:
    if host and model and not fallback:
        extra = "Vary punctuation. One message may be an email with a Subject line."
        for attempt in range(3):
            try:
                blob = ollama_generate(
                    host, model, scene_prompt(world, scene, extra), temperature=0.85 + 0.05 * attempt
                )
            except urllib.error.URLError as exc:
                print(f"  ollama error on {scene['id']}: {exc}", file=sys.stderr)
                break
            parsed = parse_messages(blob)
            if len(parsed) >= 3:
                return parsed
            extra = "You forgot the <<<MSG wrappers. Output ONLY wrapped messages this time."
        print(f"  parse failed for {scene['id']}; using fallback thread", file=sys.stderr)
    return fallback_thread(world, scene, rng)


def stamp_messages(scenes_out: list[tuple[dict, list[dict]]]) -> list[dict]:
    all_msgs = []
    n = 1
    for scene, msgs in scenes_out:
        base = datetime.fromisoformat(scene["clock"].replace("Z", "+00:00"))
        for i, msg in enumerate(msgs):
            ts = base.timestamp() + i * 47
            all_msgs.append(
                {
                    "message_id": f"MSG-{n:03d}",
                    "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "scene": scene["id"],
                    "channel": msg["channel"],
                    "from_name": msg["from_name"],
                    "callsign": msg["callsign"],
                    "to": msg.get("to", "ALL"),
                    "raw_text": msg["raw_text"],
                }
            )
            n += 1
    all_msgs.sort(key=lambda m: m["timestamp"])
    for i, m in enumerate(all_msgs, start=1):
        m["message_id"] = f"MSG-{i:03d}"
    return all_msgs


def write_exports(out_dir: Path, messages: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "messages.jsonl").open("w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def dump(name: str, channels: set[str], header: str) -> None:
        lines = [header, ""]
        for m in messages:
            if m["channel"] not in channels:
                continue
            who = f"{m['from_name']} / {m['callsign']}"
            lines.append(f"[{m['timestamp']}] ({m['channel']}) {who} -> {m['to']}")
            lines.append(m["raw_text"])
            lines.append("")
        (out_dir / name).write_text("\n".join(lines), encoding="utf-8")

    dump("radio_net.txt", {"radio"}, "=== ANCHOR NET TRANSCRIPT 12 SEP 26 ===")
    dump("sms_export.txt", {"sms"}, "SMS backup export — device mixed, timestamps UTC")
    dump("whatsapp_export.txt", {"whatsapp"}, "WhatsApp Chat with Logistics + CivAid")
    dump("email_thread.txt", {"email"}, "mbox-ish TOC logistics thread")
    dump("field_logs.txt", {"field_log", "paper_ocr"}, "photographed notebooks + typed logs")

    messy_csv = ["ts,who,where_maybe,note"]
    for m in messages:
        note = m["raw_text"].replace("\n", " / ").replace(",", ";")
        messy_csv.append(f"{m['timestamp']},{m['from_name']},,{note[:240]}")
    (out_dir / "phone_backup_partial.csv").write_text("\n".join(messy_csv) + "\n", encoding="utf-8")


def _messages_for_edge(msgs: list[dict], edge: dict) -> list[dict]:
    """Messages that actually name this hop and its true weight.

    Matching only the first word of the destination (e.g. 'Zulu') plus the
    pound count used to rewrite a different hop that merely mentioned that
    place. Require both endpoint display names and a send/got claim.
    """
    src = edge["source_name"]
    tgt = edge["target_name"]
    lb = str(edge["value_lb"])
    hits = []
    for m in msgs:
        if m.get("corrupted"):
            continue
        text = m.get("raw_text") or ""
        if src not in text or tgt not in text or lb not in text:
            continue
        low = text.lower()
        if "eod " in low:
            continue
        if "sending" in low or "sent" in low or "got " in low:
            hits.append(m)
    return hits


def corrupt_copy(
    world: dict,
    true_msgs: list[dict],
    host: str | None,
    model: str | None,
    fallback: bool,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    msgs = copy.deepcopy(true_msgs)
    labels: dict = {"mutated_message_ids": [], "injected_messages": [], "attacks": []}
    edges = [e for e in world["edges"] if e["value_lb"] >= 10]
    rng.shuffle(edges)

    def rewrite(msg: dict, instruction: str, fallback_text: str) -> None:
        original = msg["raw_text"]
        if host and model and not fallback:
            prompt = f"""Rewrite this field message so it includes the following LIE.
Keep the same speaker voice, channel style, typos vibe.
LIE: {instruction}
Original:
{original}

Return ONLY the rewritten message body, no wrappers."""
            try:
                msg["raw_text"] = ollama_generate(host, model, prompt, temperature=0.7) or fallback_text
            except urllib.error.URLError:
                msg["raw_text"] = fallback_text
        else:
            msg["raw_text"] = fallback_text
        msg["corrupted"] = True
        labels["mutated_message_ids"].append(msg["message_id"])

    mutated = 0
    for e in edges:
        if mutated >= 6:
            break
        lb = e["value_lb"]
        hits = _messages_for_edge(msgs, e)
        if not hits:
            continue
        msg = hits[0]
        fake = lb + rng.choice([-15, -10, 15, 25, 40])
        if fake <= 0:
            fake = lb + 20
        lie = f"Report {fake} lb instead of {lb} lb on the move {e['source_name']} -> {e['target_name']}."
        fb = mangle_number(msg["raw_text"], lb, fake)
        rewrite(msg, lie, fb)
        labels["attacks"].append(
            {
                "id": "mutate_weight",
                "message_id": msg["message_id"],
                "true_lb": lb,
                "reported_lb": fake,
                "edge": [e["source"], e["target"]],
                "source_name": e["source_name"],
                "target_name": e["target_name"],
                "lie": lie,
            }
        )
        mutated += 1

    a, b = rng.sample(world["nodes"], 2)
    ghost_lb = rng.randrange(10, 55, 5)
    p_from = world["people_at"][a["id"]][0]
    p_to = world["people_at"][b["id"]][0]
    injected = [
        {
            "message_id": "MSG-C01",
            "timestamp": "2026-09-12T14:44:00Z",
            "scene": "inject",
            "channel": "sms",
            "from_name": p_from.name,
            "callsign": p_from.callsign,
            "to": p_to.callsign,
            "raw_text": f"also sent {ghost_lb} lbs over to {b['display']}, same runner loop. log it.",
            "corrupted": True,
            "ghost": True,
        },
        {
            "message_id": "MSG-C02",
            "timestamp": "2026-09-12T07:40:00Z",
            "scene": "inject",
            "channel": "radio",
            "from_name": world["people_at"][world["nodes"][0]["id"]][0].name,
            "callsign": world["people_at"][world["nodes"][0]["id"]][0].callsign,
            "to": "Watchtower",
            "raw_text": (
                f"Watchtower, second drop, another {world['injections'][world['nodes'][0]['id']]} "
                f"pounds from {world['nodes'][0]['display']}. Double it on the board. Over."
            ),
            "corrupted": True,
            "ghost": True,
        },
        {
            "message_id": "MSG-C03",
            "timestamp": "2026-09-11T16:10:00Z",
            "scene": "inject",
            "channel": "paper_ocr",
            "from_name": "unknown",
            "callsign": "-",
            "to": "ALL",
            "raw_text": f"YESTERDAY? {b['display']} recvd 200 lb (smudged date 11/12). do not know if this is today",
            "corrupted": True,
            "ghost": True,
        },
        {
            "message_id": "MSG-C04",
            "timestamp": "2026-09-12T14:31:00Z",
            "scene": "inject",
            "channel": "sms",
            "from_name": p_from.name,
            "callsign": p_from.callsign,
            "to": p_to.name,
            "raw_text": f"correction that last runner was {ghost_lb} kg not lb wait no {ghost_lb} kg",
            "corrupted": True,
            "ghost": True,
        },
    ]
    if host and model and not fallback:
        for inj in injected:
            prompt = (
                f"Write one messy {inj['channel']} message from {inj['from_name']} ({inj['callsign']}).\n"
                f"It must convey this false claim:\n{inj['raw_text']}\nReturn ONLY the message body."
            )
            try:
                inj["raw_text"] = ollama_generate(host, model, prompt, temperature=0.8) or inj["raw_text"]
            except urllib.error.URLError:
                pass
    msgs.extend(injected)
    labels["injected_messages"] = [m["message_id"] for m in injected]
    labels["attacks"].extend(
        [
            {
                "id": "ghost_edge",
                "message_id": "MSG-C01",
                "source": a["id"],
                "target": b["id"],
                "source_name": a["display"],
                "target_name": b["display"],
                "reported_lb": ghost_lb,
                "lie": f"{a['display']} sent {ghost_lb} lb to {b['display']} (no such edge).",
            },
            {"id": "double_count_source", "message_id": "MSG-C02", "lie": "Second source drop that did not happen."},
            {"id": "stale_prior_day", "message_id": "MSG-C03", "lie": "200 lb receipt from another day."},
            {"id": "unit_swap_kg", "message_id": "MSG-C04", "lie": "kg vs lb on a runner."},
        ]
    )
    msgs.sort(key=lambda m: m["timestamp"])
    return msgs, labels


def mangle_number(text: str, old: int, new: int) -> str:
    return text.replace(str(old), str(new), 1)


def build_corrupted_graph(true_graph: dict, labels: dict) -> dict:
    g = copy.deepcopy(true_graph)
    g["scenario"] = "supply_drops_people_as_sensors_corrupted"
    g["notes"] = (
        "Corrupted graph incorporating mutated edge weights and injected "
        "ghost edges as reported by the corrupted messages."
    )
    by_pair = {(e["source"], e["target"]): e for e in g["edges"]}
    for atk in labels.get("attacks") or []:
        if atk.get("id") == "mutate_weight":
            src, tgt = atk["edge"]
            if (src, tgt) in by_pair:
                by_pair[(src, tgt)]["value_lb"] = atk["reported_lb"]
                by_pair[(src, tgt)]["corrupted"] = True
        elif atk.get("id") == "ghost_edge" and atk.get("source") and atk.get("target"):
            g["edges"].append(
                {
                    "source": atk["source"],
                    "target": atk["target"],
                    "value_lb": atk["reported_lb"],
                    "source_name": atk.get("source_name"),
                    "target_name": atk.get("target_name"),
                    "ghost": True,
                }
            )
    return g


def generate(
    out_root: Path,
    seed: int | None = None,
    nodes: int | None = None,
    min_nodes: int = 15,
    max_nodes: int = 30,
    fallback: bool = True,
    host: str | None = None,
    model: str | None = None,
) -> dict:
    seed = seed if seed is not None else random.randrange(1, 10_000_000)
    rng = random.Random(seed)
    n_nodes = max(min_nodes, min(max_nodes, nodes if nodes is not None else rng.randint(min_nodes, max_nodes)))
    host = host or DEFAULT_HOST
    use_fallback = fallback
    if not use_fallback:
        chosen = pick_model(host, model)
        if not chosen:
            use_fallback = True
            model = None
        else:
            model = chosen

    world = build_world(rng, n_nodes)
    scenes = make_scenes(world)
    scenes_out = [(scene, generate_scene(world, scene, host, model, use_fallback, rng)) for scene in scenes]
    true_msgs = stamp_messages(scenes_out)

    true_dir = out_root / "true"
    corrupt_dir = out_root / "corrupted"
    eval_dir = out_root / "eval"
    if true_dir.exists():
        shutil.rmtree(true_dir)
    write_exports(true_dir, true_msgs)

    eval_dir.mkdir(parents=True, exist_ok=True)
    graph = world["graph"]
    graph["generator_seed"] = seed
    (eval_dir / "true_graph.json").write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")

    corrupt_msgs, labels = corrupt_copy(world, true_msgs, host, model, use_fallback, rng)
    if corrupt_dir.exists():
        shutil.rmtree(corrupt_dir)
    write_exports(corrupt_dir, corrupt_msgs)
    labels["true_message_count"] = len(true_msgs)
    labels["corrupted_message_count"] = len(corrupt_msgs)
    labels["generator"] = "fallback" if use_fallback else f"ollama:{model}"
    labels["seed"] = seed
    labels["n_nodes"] = len(world["nodes"])
    (eval_dir / "corruption_labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    corrupted_graph = build_corrupted_graph(graph, labels)
    (eval_dir / "corrupted_graph.json").write_text(json.dumps(corrupted_graph, indent=2) + "\n", encoding="utf-8")
    return {
        "seed": seed,
        "n_nodes": len(world["nodes"]),
        "n_edges": len(world["edges"]),
        "true_messages": len(true_msgs),
        "corrupted_messages": len(corrupt_msgs),
        "out_root": str(out_root),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Generate true + corrupted supply-drop message corpora")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--model", default=None, help="Ollama model name")
    p.add_argument("--fallback", action="store_true", help="Skip Ollama; use canned messy threads")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--nodes", type=int, default=None, help="Exact node count (15-30). Random if omitted.")
    p.add_argument("--min-nodes", type=int, default=15)
    p.add_argument("--max-nodes", type=int, default=30)
    p.add_argument(
        "--out-root",
        type=Path,
        default=ROOT,
        help="Write true/, corrupted/, and eval/ under this directory (default: this folder).",
    )
    args = p.parse_args()
    stats = generate(
        out_root=args.out_root,
        seed=args.seed,
        nodes=args.nodes,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        fallback=args.fallback,
        host=args.host,
        model=args.model,
    )
    print(
        f"seed={stats['seed']} nodes={stats['n_nodes']} edges={stats['n_edges']} "
        f"true_msgs={stats['true_messages']} corrupted_msgs={stats['corrupted_messages']}"
    )
    print(f"wrote -> {stats['out_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
