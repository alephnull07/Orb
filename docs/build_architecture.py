#!/usr/bin/env python3
"""
docs/build_architecture.py — render docs/architecture.png: the ORB pipeline as
a monochrome, LaTeX-styled flowchart of circular stages, one sentence each.
Run:  python3 docs/build_architecture.py
"""
from __future__ import annotations

import asyncio
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
FONT = "'Latin Modern Roman', 'CMU Serif', 'Times New Roman', Times, serif"
INK, INK2, INK3 = "#000000", "#333333", "#666666"

W, H = 1780, 690
R = 42
parts: list[str] = []


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


STAGES: list[dict] = []   # filled by node(); used for the legend


def node(x, y, title, sentence, bold_ring=False):
    n = len(STAGES) + 1
    STAGES.append({"n": n, "title": title.replace("\n", " "), "sentence": sentence})
    parts.append(f'<circle cx="{x}" cy="{y}" r="{R}" fill="#ffffff" stroke="{INK}" stroke-width="{1.9 if bold_ring else 1.2}"/>')
    lines = title.split("\n")
    y0 = y + 5 - (len(lines) - 1) * 8
    for i, ln in enumerate(lines):
        parts.append(f'<text x="{x}" y="{y0 + i*17}" text-anchor="middle" font-size="14.5" font-weight="700" fill="{INK}">{esc(ln)}</text>')
    # small number badge
    parts.append(f'<circle cx="{x + R*0.72}" cy="{y - R*0.72}" r="10" fill="{INK}"/>')
    parts.append(f'<text x="{x + R*0.72}" y="{y - R*0.72 + 3.8}" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">{n}</text>')


def edge(x1, y1, x2, y2, label="", dashed=False):
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    sx, sy = x1 + ux * (R + 3), y1 + uy * (R + 3)
    ex, ey = x2 - ux * (R + 5), y2 - uy * (R + 5)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    parts.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{INK}" stroke-width="1.2"{dash} marker-end="url(#arr)"/>')
    if label:
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        off = -8 if dy < 0 else 14
        parts.append(f'<text x="{mx - 14:.1f}" y="{my + off:.1f}" text-anchor="middle" font-size="11" font-style="italic" fill="{INK2}">{esc(label)}</text>')


# ── title ────────────────────────────────────────────────────────────────────
parts.append(f'<text x="{W/2}" y="46" text-anchor="middle" font-size="24" font-weight="700" fill="{INK}">ORB: outlier-robust state estimation over conserved-flow networks</text>')
parts.append(f'<text x="{W/2}" y="72" text-anchor="middle" font-size="13.5" font-style="italic" fill="{INK2}">y = Hx + a + e, with a sparse and unknown.  Recover x, locate supp(a), and say when the data cannot decide.</text>')

Y, YT, YB = 262, 172, 352
X = [80, 226, 380, 540, 700, 860, 1020, 1180, 1340, 1520, 1700]

node(X[0], Y, "Files", "Radio logs, field notes, JSON lines, spreadsheets. No schema is assumed and sites are named however people name them.")
node(X[1], Y, "Route", "Dispatch by extension only: text and JSONL go to the readers, tables to the column mapper. No model call is made here.")
node(X[2], YT, "Readers", "Three LLM personas each emit one claim per message, (from, to, qty, perspective, evidence); a claim survives a ≥2-of-3 vote and is weighted 1/(reader spread + λ).")
node(X[2], YB, "Column\nmap", "One LLM call maps columns to node / edge / sink primitives from a stratified sample; every row then becomes a claim in pure Python.")
node(X[3], Y, "Claims", "Each report is a row yᵢ = Hᵢx + aᵢ + eᵢ over the state x = (stock, flow, sink); sender and receiver rows stay separate as two observations of one event.")
node(X[4], Y, "Squash", "Canonical ids unify names across files and events on one edge sum within an observer stream (file, channel); nothing is averaged, medianed, or deduplicated.", bold_ring=True)
node(X[5], Y, "Compile", "A hard balance per node, q − Σin + Σout + s = q₀ − draws; identifiability requires rank[H; A] = n, and the guarantee is k = ⌊(d − 1)/2⌋ with d the fewest rows whose removal blinds the balance.")
node(X[6], Y, "L1 solve", "minₓ Σ wᵢ |yᵢ − Hᵢx| + λ Σ s subject to Ax = b, solved as a linear program; sparse corruption concentrates in a few residuals instead of leaking into every honest row.")
node(X[7], Y, "Decode", "Read q̂, f̂, ŝ, rank sinks as leak candidates, and flag |rᵢ| > τ with file, channel and record; if flags exceed k the run is declared past its guarantee.")
node(X[8], Y, "Graph", "Reported versus corrected network, node health, ranked losses, and flagged rows with provenance.")
node(X[9], Y, "Analyst\nchat", "Claude receives the recovered graph, residuals, flags, health and contacts as context, writes the briefing, and answers next-step questions in a chat.")
node(X[10], Y, "Action", "Who to call, what to verify first, and how far to trust the estimate.")

edge(X[0], Y, X[1], Y)
edge(X[1], Y, X[2], YT, "text")
edge(X[1], Y, X[2], YB, "table")
edge(X[2], YT, X[3], Y)
edge(X[2], YB, X[3], Y)
for i in range(3, 10):
    edge(X[i], Y, X[i + 1], Y)

# losses mode: a caller decision entering the squash (dashed)
px, py = X[4], Y - 132
parts.append(f'<rect x="{px-112}" y="{py-17}" width="224" height="34" rx="17" fill="#ffffff" stroke="{INK}" stroke-width="1.2" stroke-dasharray="5 4"/>')
parts.append(f'<text x="{px}" y="{py+5}" text-anchor="middle" font-size="12" font-style="italic" fill="{INK}">losses mode ∈ {{none, known, unknown}}</text>')
parts.append(f'<text x="{px}" y="{py-26}" text-anchor="middle" font-size="10.8" fill="{INK2}">chosen by the operator, never inferred from data</text>')
parts.append(f'<line x1="{px}" y1="{py+17}" x2="{px}" y2="{Y-R-6}" stroke="{INK}" stroke-width="1.2" stroke-dasharray="5 4" marker-end="url(#arr)"/>')

# ── legend: numbered sentences in two columns ───────────────────────────────
import textwrap
top = 440
parts.append(f'<line x1="60" y1="{top-22}" x2="{W-60}" y2="{top-22}" stroke="{INK3}" stroke-width="0.6"/>')
col_w = (W - 120 - 40) / 2
cols = [STAGES[:6], STAGES[6:]]
for ci, items in enumerate(cols):
    x = 60 + ci * (col_w + 40)
    y = top
    for st in items:
        wrapped = textwrap.wrap(st["sentence"], width=118)
        parts.append(f'<text x="{x}" y="{y}" font-size="12" fill="{INK}"><tspan font-weight="700">{st["n"]}. {esc(st["title"])}.</tspan> {esc(wrapped[0])}</text>')
        for ln in wrapped[1:]:
            y += 15.5
            parts.append(f'<text x="{x + 18}" y="{y}" font-size="12" fill="{INK}">{esc(ln)}</text>')
        y += 24

parts.append(f'<line x1="60" y1="{H-46}" x2="{W-60}" y2="{H-46}" stroke="{INK3}" stroke-width="0.6"/>')
parts.append(f'<text x="60" y="{H-26}" font-size="11.5" fill="{INK2}">Fig. 1.  Every independent observation survives to the estimator as its own row; only the L1 objective decides which rows are wrong.  The guarantee k is computed before any answer is shown.</text>')

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
  aria-label="ORB pipeline: files are routed to text readers or a column mapper, become per-report claims, are squashed into one graph, compiled into balance equations, solved with weighted L1, decoded into state, losses and flags, shown as a graph, and explained by an analyst chat that produces actions."
  font-family="{FONT}">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="{INK}"/>
    </marker>
  </defs>
  <rect width="{W}" height="{H}" fill="#ffffff"/>
  {"".join(parts)}
</svg>'''

html = f'<!doctype html><meta charset="utf-8"><title>ORB pipeline</title><style>body{{margin:0;background:#fff}} svg{{display:block}}</style>{svg}'
(OUT / "architecture.html").write_text(html, encoding="utf-8")
(OUT / "architecture.svg").write_text(svg, encoding="utf-8")


async def render():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
        await pg.goto((OUT / "architecture.html").as_uri())
        await pg.wait_for_timeout(200)
        await pg.screenshot(path=str(OUT / "architecture.png"))
        await b.close()


if __name__ == "__main__":
    asyncio.run(render())
    print("wrote", OUT / "architecture.png")
