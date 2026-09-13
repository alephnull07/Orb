const COLORS = {
  onlyA: "#5b8def",
  onlyB: "#e05d5d",
  diff: "#e0a14a",
  match: "#8b95a5",
  source: "#6ea8ff",
  hub: "#5ec2b7",
  junction: "#5ec2b7",
  sink: "#d4a054",
  unknown: "#9aa3b5",
};

const state = {
  a: null,
  b: null,
  titleA: "Graph A",
  titleB: "Graph B",
  cmp: null,
  layout: null,
  views: [],
};

function keyName(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function shortLabel(label) {
  const parts = String(label).split(/\s+/);
  return parts.length <= 2 ? label : parts.slice(0, 2).join(" ");
}

function normalize(raw) {
  const nodes = (raw.nodes || []).map((n) => {
    const label = n.display || n.label || n.id;
    return { id: n.id, label, key: keyName(label), type: n.type || "unknown", raw: n };
  });
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const edges = (raw.edges || []).map((e, i) => {
    const fromNode = byId[e.source];
    const toNode = byId[e.target];
    const fromLabel = e.source_display || e.source_name || fromNode?.label || e.source;
    const toLabel = e.target_display || e.target_name || toNode?.label || e.target;
    return {
      id: `e${i}`,
      fromLabel,
      toLabel,
      fromKey: keyName(fromLabel),
      toKey: keyName(toLabel),
      hop: `${keyName(fromLabel)}>>${keyName(toLabel)}`,
      value: Number(e.value_lb ?? e.weight ?? e.value ?? 0),
      raw: e,
    };
  });
  return { raw, nodes, edges };
}

function compare(a, b) {
  const nodeMap = new Map();
  const addNode = (n, side) => {
    if (!nodeMap.has(n.key)) {
      nodeMap.set(n.key, { key: n.key, label: n.label, type: n.type, inA: false, inB: false });
    }
    const row = nodeMap.get(n.key);
    row[side] = true;
    if (n.type && n.type !== "unknown") row.type = n.type;
    if (n.label.length > row.label.length) row.label = n.label;
  };
  (a?.nodes || []).forEach((n) => addNode(n, "inA"));
  (b?.nodes || []).forEach((n) => addNode(n, "inB"));

  const hops = new Map();
  const addHop = (e, side) => {
    if (!hops.has(e.hop)) {
      hops.set(e.hop, {
        hop: e.hop,
        fromKey: e.fromKey,
        toKey: e.toKey,
        fromLabel: e.fromLabel,
        toLabel: e.toLabel,
        a: null,
        b: null,
      });
    }
    const row = hops.get(e.hop);
    row[side] = e.value;
    if (side === "a") {
      row.fromLabel = e.fromLabel;
      row.toLabel = e.toLabel;
    }
  };
  (a?.edges || []).forEach((e) => addHop(e, "a"));
  (b?.edges || []).forEach((e) => addHop(e, "b"));

  const hopRows = [...hops.values()].map((h) => {
    let status = "match";
    if (h.a == null) status = "onlyB";
    else if (h.b == null) status = "onlyA";
    else if (Number(h.a) !== Number(h.b)) status = "diff";
    return { ...h, status, delta: h.a != null && h.b != null ? h.b - h.a : null };
  });
  return { nodes: [...nodeMap.values()], hops: hopRows };
}

function layoutGraph(nodes, hops) {
  const preds = new Map(nodes.map((n) => [n.key, []]));
  const succs = new Map(nodes.map((n) => [n.key, []]));
  hops.forEach((h) => {
    if (!preds.has(h.fromKey) || !preds.has(h.toKey)) return;
    preds.get(h.toKey).push(h.fromKey);
    succs.get(h.fromKey).push(h.toKey);
  });

  const rank = new Map();
  const typeRank = { source: 0, hub: 1, junction: 2, sink: 3, unknown: 2 };
  nodes.forEach((n) => {
    if (!preds.get(n.key).length) rank.set(n.key, typeRank[n.type] ?? 0);
  });
  let changed = true;
  let guard = 0;
  while (changed && guard++ < nodes.length + 4) {
    changed = false;
    nodes.forEach((n) => {
      const incoming = preds.get(n.key);
      if (!incoming.length) return;
      const ready = incoming.every((p) => rank.has(p));
      if (!ready) return;
      const next = 1 + Math.max(...incoming.map((p) => rank.get(p)));
      if (!rank.has(n.key) || next > rank.get(n.key)) {
        rank.set(n.key, next);
        changed = true;
      }
    });
  }
  nodes.forEach((n) => {
    if (!rank.has(n.key)) rank.set(n.key, typeRank[n.type] ?? 1);
  });

  const cols = new Map();
  nodes.forEach((n) => {
    const r = rank.get(n.key);
    if (!cols.has(r)) cols.set(r, []);
    cols.get(r).push(n);
  });
  [...cols.values()].forEach((col) => col.sort((a, b) => a.label.localeCompare(b.label)));

  const colKeys = [...cols.keys()].sort((a, b) => a - b);
  const colGap = 210;
  const rowGap = 62;
  const padX = 70;
  const padY = 50;
  const maxRows = Math.max(1, ...[...cols.values()].map((c) => c.length));
  const positions = new Map();
  colKeys.forEach((r, i) => {
    const col = cols.get(r);
    const blockH = (col.length - 1) * rowGap;
    const startY = padY + ((maxRows - 1) * rowGap - blockH) / 2;
    col.forEach((n, j) => {
      positions.set(n.key, { x: padX + i * colGap, y: startY + j * rowGap, rank: r });
    });
  });
  return {
    positions,
    width: padX * 2 + Math.max(0, colKeys.length - 1) * colGap,
    height: padY * 2 + (maxRows - 1) * rowGap,
  };
}

function nodeFill(n) {
  return COLORS[n.type] || COLORS.unknown;
}

function edgeColor(h, side) {
  if (side === "overlay") return COLORS[h.status];
  if (h.status === "match") return COLORS.match;
  if (h.status === "diff") return COLORS.diff;
  return side === "a" ? COLORS.onlyA : COLORS.onlyB;
}

function hopsFor(side, hops) {
  if (side === "overlay") return hops;
  return hops.filter((h) => (side === "a" ? h.a != null : h.b != null));
}

function nodePresent(n, side) {
  if (side === "overlay") return n.inA || n.inB;
  return side === "a" ? n.inA : n.inB;
}

function svgEl(name, attrs, text) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
  if (text != null) el.textContent = text;
  return el;
}

function drawGraph(container, cmp, layout, side, title) {
  container.innerHTML = "";
  const pad = 36;
  const svg = svgEl("svg", {
    viewBox: `${-pad} ${-pad} ${layout.width + pad * 2} ${layout.height + pad * 2}`,
    role: "img",
    "aria-label": title,
    preserveAspectRatio: "xMidYMid meet",
  });
  const defs = svgEl("defs", {});
  const markerId = `arrow-${side}-${Math.random().toString(36).slice(2, 7)}`;
  defs.innerHTML = `
    <marker id="${markerId}" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="7" orient="auto">
      <path d="M0,0 L10,3.5 L0,7 Z" fill="#8b95a5"></path>
    </marker>`;
  svg.appendChild(defs);

  const world = svgEl("g", { class: "world" });
  svg.appendChild(world);

  hopsFor(side, cmp.hops).forEach((h) => {
    const a = layout.positions.get(h.fromKey);
    const b = layout.positions.get(h.toKey);
    if (!a || !b) return;
    const color = edgeColor(h, side);
    const value = side === "b" ? h.b : side === "a" ? h.a : h.status === "diff" ? `${h.a}/${h.b}` : h.a ?? h.b;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    const nx = dx / len;
    const ny = dy / len;
    const x1 = a.x + nx * 18;
    const y1 = a.y + ny * 18;
    const x2 = b.x - nx * 18;
    const y2 = b.y - ny * 18;
    const cx = (x1 + x2) / 2 + (dy / len) * 16;
    const cy = (y1 + y2) / 2 - (dx / len) * 16;
    const d = `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`;
    const g = svgEl("g", { class: "edge-hit", "data-hop": h.hop });
    const wide = svgEl("path", { d, fill: "none", stroke: "transparent", "stroke-width": 12 });
    const path = svgEl("path", {
      d,
      fill: "none",
      stroke: color,
      "stroke-width": h.status === "diff" ? 2.4 : 1.6,
      "stroke-dasharray": h.status === "onlyA" || h.status === "onlyB" ? "6 4" : "none",
      "marker-end": `url(#${markerId})`,
    });
    const label = svgEl("text", { x: cx, y: cy - 6, class: "edge-label", "text-anchor": "middle" }, String(value));
    g.append(wide, path, label);
    g.addEventListener("mouseenter", () => showDetail(hopText(h)));
    g.addEventListener("click", () => highlightHop(h.hop));
    world.appendChild(g);
  });

  cmp.nodes.forEach((n) => {
    const p = layout.positions.get(n.key);
    if (!p) return;
    const present = nodePresent(n, side);
    const g = svgEl("g", { class: "node-hit", "data-node": n.key, transform: `translate(${p.x},${p.y})` });
    g.appendChild(
      svgEl("circle", {
        r: n.type === "source" ? 13 : 11,
        fill: present ? nodeFill(n) : "#2a2e38",
        stroke: present ? "#e8eaef" : "#555b68",
        "stroke-width": 1.4,
        opacity: present ? 1 : 0.35,
      }),
    );
    g.appendChild(
      svgEl("text", {
        x: 0,
        y: 24,
        class: "node-label",
        "text-anchor": "middle",
        fill: present ? "#e8eaef" : "#6b7384",
      }, shortLabel(n.label)),
    );
    g.addEventListener("mouseenter", () => showDetail(nodeText(n)));
    world.appendChild(g);
  });

  const view = {
    svg,
    world,
    layout,
    vb: { x: -pad, y: -pad, w: layout.width + pad * 2, h: layout.height + pad * 2 },
  };
  bindPanZoom(view);
  container.appendChild(svg);
  applyView(view);
  return view;
}

function hopText(h) {
  return `${h.fromLabel} → ${h.toLabel} · A ${h.a ?? "—"} lb · B ${h.b ?? "—"} lb · ${h.status}`;
}

function nodeText(n) {
  const sides = [n.inA ? "A" : null, n.inB ? "B" : null].filter(Boolean).join(" + ");
  return `${n.label} · ${n.type} · in ${sides || "neither"}`;
}

function showDetail(text) {
  document.getElementById("detail").textContent = text;
}

function highlightHop(hop) {
  document.querySelectorAll(".drawer tr[data-hop]").forEach((tr) => {
    tr.classList.toggle("active", tr.dataset.hop === hop);
    if (tr.dataset.hop === hop) tr.scrollIntoView({ block: "nearest" });
  });
}

function applyView(view) {
  view.svg.setAttribute("viewBox", `${view.vb.x} ${view.vb.y} ${view.vb.w} ${view.vb.h}`);
}

function clientToVb(view, clientX, clientY) {
  const pt = view.svg.createSVGPoint();
  pt.x = clientX;
  pt.y = clientY;
  const ctm = view.svg.getScreenCTM();
  if (!ctm) return { x: 0, y: 0 };
  return pt.matrixTransform(ctm.inverse());
}

function bindPanZoom(view) {
  let dragging = false;
  let last = null;
  view.svg.addEventListener("pointerdown", (ev) => {
    if (ev.target.closest(".node-hit, .edge-hit")) return;
    dragging = true;
    last = clientToVb(view, ev.clientX, ev.clientY);
    view.svg.classList.add("dragging");
    view.svg.setPointerCapture(ev.pointerId);
  });
  view.svg.addEventListener("pointermove", (ev) => {
    if (!dragging) return;
    const now = clientToVb(view, ev.clientX, ev.clientY);
    view.vb.x -= now.x - last.x;
    view.vb.y -= now.y - last.y;
    applyView(view);
    last = clientToVb(view, ev.clientX, ev.clientY);
  });
  const end = () => {
    dragging = false;
    view.svg.classList.remove("dragging");
  };
  view.svg.addEventListener("pointerup", end);
  view.svg.addEventListener("pointercancel", end);
  view.svg.addEventListener(
    "wheel",
    (ev) => {
      ev.preventDefault();
      const factor = ev.deltaY < 0 ? 0.9 : 1.1;
      const p = clientToVb(view, ev.clientX, ev.clientY);
      const nextW = Math.min(view.layout.width * 4, Math.max(view.layout.width * 0.25, view.vb.w * factor));
      const nextH = nextW * (view.vb.h / view.vb.w);
      view.vb.x = p.x - (p.x - view.vb.x) * (nextW / view.vb.w);
      view.vb.y = p.y - (p.y - view.vb.y) * (nextH / view.vb.h);
      view.vb.w = nextW;
      view.vb.h = nextH;
      applyView(view);
    },
    { passive: false },
  );
}

function fitView(view) {
  const pad = 36;
  view.vb = { x: -pad, y: -pad, w: view.layout.width + pad * 2, h: view.layout.height + pad * 2 };
  applyView(view);
}

function render() {
  const stage = document.getElementById("stage");
  const paneB = document.getElementById("paneB");
  const view = document.getElementById("viewMode").value;
  document.getElementById("titleA").textContent = state.titleA;
  document.getElementById("titleB").textContent = state.titleB;
  document.getElementById("graphA").innerHTML = "";
  document.getElementById("graphB").innerHTML = "";
  state.views = [];

  if (!state.a && !state.b) return;
  const empty = { nodes: [], edges: [] };
  const cmp = compare(state.a || empty, state.b || empty);
  state.cmp = cmp;
  state.layout = layoutGraph(cmp.nodes, cmp.hops);
  updateStats(cmp);
  updateTable(cmp);

  if (view === "overlay" || (!state.a || !state.b)) {
    stage.className = "stage overlay";
    paneB.style.display = "none";
    document.getElementById("titleA").textContent =
      state.a && state.b ? `${state.titleA} ∩ ${state.titleB}` : state.a ? state.titleA : state.titleB;
    state.views = [drawGraph(document.getElementById("graphA"), cmp, state.layout, state.a && state.b ? "overlay" : state.a ? "a" : "b", document.getElementById("titleA").textContent)];
    return;
  }

  stage.className = "stage split";
  paneB.style.display = "";
  state.views = [
    drawGraph(document.getElementById("graphA"), cmp, state.layout, "a", state.titleA),
    drawGraph(document.getElementById("graphB"), cmp, state.layout, "b", state.titleB),
  ];
}

function updateStats(cmp) {
  const nMatch = cmp.nodes.filter((n) => n.inA && n.inB).length;
  const hopMatch = cmp.hops.filter((h) => h.status === "match").length;
  const hopDiff = cmp.hops.filter((h) => h.status === "diff").length;
  const onlyA = cmp.hops.filter((h) => h.status === "onlyA").length;
  const onlyB = cmp.hops.filter((h) => h.status === "onlyB").length;
  const shared = cmp.hops.filter((h) => h.delta != null);
  const mae = shared.length ? (shared.reduce((s, h) => s + Math.abs(h.delta), 0) / shared.length).toFixed(2) : "—";
  document.getElementById("stats").innerHTML = `
    <span>Sites <b>${nMatch}</b> shared / ${cmp.nodes.length} union</span>
    <span>Hops match <b>${hopMatch}</b></span>
    <span>Weight differs <b>${hopDiff}</b></span>
    <span>Only A <b>${onlyA}</b> · Only B <b>${onlyB}</b></span>
    <span>Mean |Δ| on shared hops <b>${mae} lb</b></span>
    <span class="legend">
      <span><i class="dot source"></i>source</span>
      <span><i class="dot hub"></i>hub</span>
      <span><i class="dot sink"></i>sink</span>
      <span><i class="dot only-a"></i>only A</span>
      <span><i class="dot only-b"></i>only B</span>
      <span><i class="dot diff"></i>weight differs</span>
      <span><i class="dot match"></i>match</span>
    </span>
  `;
}

function updateTable(cmp) {
  const order = { diff: 0, onlyA: 1, onlyB: 2, match: 3 };
  const rows = [...cmp.hops].sort((x, y) => order[x.status] - order[y.status] || x.fromLabel.localeCompare(y.fromLabel));
  const labels = { match: "match", diff: "weight", onlyA: "only A", onlyB: "only B" };
  document.getElementById("diffBody").innerHTML = rows
    .map(
      (h) => `<tr data-hop="${esc(h.hop)}">
        <td style="color:${COLORS[h.status]}">${labels[h.status]}</td>
        <td>${esc(h.fromLabel)}</td>
        <td>${esc(h.toLabel)}</td>
        <td class="tag">${h.a ?? "—"}</td>
        <td class="tag">${h.b ?? "—"}</td>
        <td class="tag">${h.delta == null ? "—" : h.delta}</td>
      </tr>`,
    )
    .join("");
  document.querySelectorAll(".drawer tr[data-hop]").forEach((tr) => {
    tr.addEventListener("click", () => {
      showDetail(hopText(cmp.hops.find((h) => h.hop === tr.dataset.hop)));
      highlightHop(tr.dataset.hop);
    });
  });
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function readFile(file) {
  return normalize(JSON.parse(await file.text()));
}

function bindFile(inputId, nameId, side) {
  document.getElementById(inputId).addEventListener("change", async (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    document.getElementById(nameId).textContent = file.name;
    state[side] = await readFile(file);
    state[side === "a" ? "titleA" : "titleB"] = file.name;
    render();
  });
}

async function loadDemo(pathA, pathB, titleA, titleB) {
  const [ra, rb] = await Promise.all([fetch(pathA), fetch(pathB)]);
  if (!ra.ok || !rb.ok) {
    alert("Could not fetch demo JSON. From the repo root run:\npython -m http.server 8765\nthen open http://localhost:8765/viz/graph-compare/");
    return;
  }
  state.a = normalize(await ra.json());
  state.b = normalize(await rb.json());
  state.titleA = titleA;
  state.titleB = titleB;
  document.getElementById("nameA").textContent = titleA;
  document.getElementById("nameB").textContent = titleB;
  render();
}

bindFile("fileA", "nameA", "a");
bindFile("fileB", "nameB", "b");
document.getElementById("viewMode").addEventListener("change", () => render());
document.getElementById("fitBtn").addEventListener("click", () => {
  state.views.forEach((v) => fitView(v));
});
document.getElementById("demoTruth").addEventListener("click", () =>
  loadDemo(
    "../../data/supply_drops/eval/true_graph.json",
    "../../data/supply_drops/graphs/true/consensus.json",
    "Hidden truth",
    "Agent consensus (true logs)",
  ),
);
document.getElementById("demoCorrupt").addEventListener("click", () =>
  loadDemo(
    "../../data/supply_drops/graphs/true/consensus.json",
    "../../data/supply_drops/graphs/corrupted/consensus.json",
    "Agents on true logs",
    "Agents on corrupted logs",
  ),
);
window.addEventListener("resize", () => {
  if (state.cmp) render();
});
