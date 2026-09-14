/**
 * Shared ORB types + pure helpers used by both the client (dashboard) and
 * the server (LLM context). No Node APIs here.
 */

// ─── Result shape returned by /api/demo ────────────────────────────────────

export interface DemoResult {
  ingest_report: {
    mode?: 'TABULAR' | 'RECORD'
    mapping?: Record<string, any>
    exclusions?: string[]
    record_count?: number
    n_files?: number
    total_claims?: number
    multi_source_claims?: number
    timestamp_span_hours?: number
    window_hours?: number
    canon_map?: Record<string, string>
    opening_initials?: Record<string, number>
    per_file?: Array<{
      mode: 'TABULAR' | 'RECORD'
      source_file: string
      mapping?: Record<string, any>
      exclusions?: string[]
      known_sinks?: Record<string, number>
      record_count?: number
    }>
  }
  report: {
    n_claims: number
    n_vars: number
    rank: number
    identifiable: boolean
    correctable_k: number
    sink_node_ids?: string[]
  }
  decoded: {
    nodes: Array<{ id: string; qty: number }>
    edges: Array<{ id: string; from: string; to: string; flow: number }>
    sinks: Array<{ id: string; sink: number }>
    flagged: Array<{ claim_id: string; residual: number; source: string; type: string }>
  }
  graph: {
    nodes: Array<{ id: string; initial?: number; sinks?: string; known_sinks?: number }>
    edges: Array<{ id: string; from: string; to: string }>
    claims: Array<{
      id: string; type: string; ref?: string; refs?: string[]
      value: any; source?: string; weight?: number
    }>
    lambda_sink?: number
  }
  error?: string
}

// ─── Mock contacts (deterministic per node id) ──────────────────────────────

export interface Contact {
  name: string
  role: string
  phone: string
  callsign: string
  email: string
}

const FIRST = ['Dana', 'Marcus', 'Priya', 'Elena', 'Tomas', 'Aisha', 'Ravi', 'Jordan',
  'Nadia', 'Felix', 'Imani', 'Lucas', 'Sofia', 'Omar', 'Hana', 'Diego']
const LAST = ['Okafor', 'Lindqvist', 'Ramirez', 'Chen', 'Haddad', 'Novak', 'Iyer',
  'Brennan', 'Mensah', 'Kowalski', 'Sato', 'Delgado', 'Fischer', 'Adeyemi', 'Moreau', 'Petrov']
const ROLES = ['Site lead', 'Ops officer', 'Logistics NCO', 'Depot manager',
  'Field supervisor', 'Shift controller', 'Station chief', 'Supply officer']
const CALLS = ['HAMMER', 'ANVIL', 'RAVEN', 'COBALT', 'SABLE', 'ORION', 'JUNO', 'VESPER',
  'MERIDIAN', 'TALLY', 'ARGUS', 'BISHOP', 'CANDOR', 'DELTA', 'EMBER', 'FALCON']

function hash(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619) >>> 0
  }
  return h
}

export function mockContact(nodeId: string): Contact {
  const h = hash(nodeId)
  const first = FIRST[h % FIRST.length]
  const last = LAST[(h >>> 4) % LAST.length]
  const role = ROLES[(h >>> 8) % ROLES.length]
  const call = CALLS[(h >>> 12) % CALLS.length]
  const area = 200 + ((h >>> 16) % 800)
  const line = 100 + ((h >>> 20) % 900)     // 555-01xx style fictional numbers
  const phone = `+1 (${area}) 555-0${String(line).slice(0, 3)}`
  const email = `${first}.${last}@ops.example`.toLowerCase()
  return { name: `${first} ${last}`, role, phone, callsign: `${call}-${1 + (h % 9)}`, email }
}

export interface ClientProfile {
  org: string
  domain: string
  deskPhone: string
  escalation: string
  escalationPhone: string
  unit: string
}

/** Infer a rough domain from node naming and produce a mock client card. */
export function mockClientProfile(result: DemoResult): ClientProfile {
  const ids = result.graph.nodes.map(n => n.id.toUpperCase()).join(' ')
  const src = (result.graph.claims[0]?.source ?? '').toLowerCase()
  let domain = 'Distribution network'
  let org = 'Northbridge Logistics Group'
  let unit = 'units'
  if (/FOB|OP_|PORT|DEPOT|BASE|POST/.test(ids) || /fuel|supply/.test(src)) {
    domain = 'Field logistics'
    org = 'Joint Sustainment Command (mock)'
    unit = /fuel/.test(src) ? 'gal' : 'lb'
  } else if (/TANK|PUMP|JUNCTION|RESERVOIR|VALVE|NODE_/.test(ids) || /water|leak/.test(src)) {
    domain = 'Water utility'
    org = 'Westfork Municipal Water (mock)'
    unit = 'm³/h'
  } else if (/WAREHOUSE|DC_|STORE|HUB/.test(ids)) {
    domain = 'Retail inventory'
    org = 'Meridian Retail Ops (mock)'
    unit = 'units'
  }
  return {
    org, domain, unit,
    deskPhone: '+1 (415) 555-0100',
    escalation: 'Duty operations director',
    escalationPhone: '+1 (415) 555-0199',
  }
}

// ─── System status derivation ───────────────────────────────────────────────

export type NodeHealth = 'ok' | 'degraded' | 'down'

export interface NodeStatus {
  id: string
  health: NodeHealth
  qty: number | null
  reported: number | null
  initial: number | null
  delta: number
  corrected: boolean
  sink: number
  flaggedClaims: string[]
  reasons: string[]
  contact: Contact
}

export interface EdgeStatus {
  id: string
  from: string
  to: string
  flow: number | null
  reported: number | null
  corrected: boolean
  flaggedClaims: string[]
}

export interface SystemStatus {
  nodes: NodeStatus[]
  edges: EdgeStatus[]
  counts: { ok: number; degraded: number; down: number }
  flagged: DemoResult['decoded']['flagged']
  client: ClientProfile
}

export const DELTA_THRESHOLD = 5.0
const HEALTH_RANK: Record<NodeHealth, number> = { down: 0, degraded: 1, ok: 2 }

/** Uploaded files land in a temp dir; only the file name is meaningful. */
export function baseName(p: string | undefined | null): string {
  if (!p) return '?'
  const parts = p.split(/[\\/]/)
  return parts[parts.length - 1] || p
}

/** Average of all claims of a given type pointing at a given ref. */
export function buildClaimedMap(
  claims: DemoResult['graph']['claims'],
  type: string,
): Map<string, number> {
  const sums = new Map<string, { total: number; count: number }>()
  for (const c of claims) {
    if (c.type !== type || c.ref == null) continue
    const v = Number(c.value)
    if (isNaN(v)) continue
    const prev = sums.get(c.ref) ?? { total: 0, count: 0 }
    sums.set(c.ref, { total: prev.total + v, count: prev.count + 1 })
  }
  const out = new Map<string, number>()
  for (const [ref, { total, count }] of sums) out.set(ref, total / count)
  return out
}

function isCorrected(est: number | null, rep: number | null): boolean {
  if (est == null || rep == null) return false
  const delta = Math.abs(est - rep)
  return delta > Math.max(DELTA_THRESHOLD, 0.05 * Math.abs(rep))
}

export function deriveStatus(result: DemoResult): SystemStatus {
  const { graph, decoded } = result
  const claimedNode = buildClaimedMap(graph.claims, 'node')
  const claimedEdge = buildClaimedMap(graph.claims, 'edge')
  const l1Qty = new Map(decoded.nodes.map(n => [n.id, n.qty]))
  const l1Flow = new Map(decoded.edges.map(e => [e.id, e.flow]))
  const sinkMap = new Map(decoded.sinks.map(s => [s.id, s.sink]))
  const flaggedIds = new Set(decoded.flagged.map(f => f.claim_id))
  const claimById = new Map(graph.claims.map(c => [c.id, c]))

  // Which claims reference which node / edge
  const nodeClaims = new Map<string, string[]>()
  const edgeClaims = new Map<string, string[]>()
  for (const c of graph.claims) {
    const refs = c.ref != null ? [c.ref] : (c.refs ?? [])
    for (const r of refs) {
      const bucket = c.type === 'edge' ? edgeClaims : nodeClaims
      bucket.set(r, [...(bucket.get(r) ?? []), c.id])
    }
  }

  const edges: EdgeStatus[] = graph.edges.map(e => {
    const flow = l1Flow.get(e.id) ?? null
    const reported = claimedEdge.get(e.id) ?? null
    const claims = edgeClaims.get(e.id) ?? []
    return {
      id: e.id, from: e.from, to: e.to, flow, reported,
      corrected: isCorrected(flow, reported),
      flaggedClaims: claims.filter(id => flaggedIds.has(id)),
    }
  })

  const nodes: NodeStatus[] = graph.nodes.map(n => {
    const qty = l1Qty.get(n.id) ?? null
    const reported = claimedNode.get(n.id) ?? null
    const sink = sinkMap.get(n.id) ?? 0
    const own = nodeClaims.get(n.id) ?? []
    const ownFlagged = own.filter(id => flaggedIds.has(id))
    const adjacent = edges.filter(e => e.from === n.id || e.to === n.id)
    const adjFlagged = adjacent.flatMap(e => e.flaggedClaims)
    const corrected = isCorrected(qty, reported)
    const delta = qty != null && reported != null ? qty - reported : 0

    const reasons: string[] = []
    let health: NodeHealth = 'ok'

    if (qty == null) {
      health = 'down'; reasons.push('No estimate could be produced for this node')
    } else if (own.length > 0 && ownFlagged.length === own.length) {
      health = 'down'; reasons.push('Every report from this node was rejected as inconsistent')
    }

    if (corrected) {
      if (health === 'ok') health = 'degraded'
      const src = ownFlagged.map(id => claimById.get(id)?.source).filter(Boolean)[0]
      reasons.push(
        `Reported ${reported!.toFixed(1)} but network conservation implies ${qty!.toFixed(1)}` +
        (src ? ` (source: ${baseName(src)})` : ''),
      )
    }
    if (ownFlagged.length > 0 && !corrected) {
      if (health === 'ok') health = 'degraded'
      reasons.push(`${ownFlagged.length} report(s) about this node flagged as inconsistent`)
    }
    if (adjFlagged.length > 0) {
      if (health === 'ok') health = 'degraded'
      reasons.push(`${adjFlagged.length} transfer report(s) on connected links flagged`)
    }
    if (Math.abs(sink) > 1) {
      if (health === 'ok') health = 'degraded'
      reasons.push(`Unaccounted loss of ${sink.toFixed(1)} detected at this node`)
    }
    if (reasons.length === 0) reasons.push('All reports consistent with the network')

    return {
      id: n.id, health, qty, reported,
      initial: n.initial ?? null,
      delta, corrected, sink,
      flaggedClaims: [...ownFlagged, ...adjFlagged],
      reasons, contact: mockContact(n.id),
    }
  })

  nodes.sort((a, b) =>
    HEALTH_RANK[a.health] - HEALTH_RANK[b.health] ||
    Math.abs(b.delta) - Math.abs(a.delta) ||
    a.id.localeCompare(b.id))

  const counts = { ok: 0, degraded: 0, down: 0 }
  for (const n of nodes) counts[n.health]++

  return { nodes, edges, counts, flagged: decoded.flagged, client: mockClientProfile(result) }
}

// ─── Text context for the LLM ───────────────────────────────────────────────

const f1 = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(1))

/** Compact, deterministic rendering of the whole run for the system prompt. */
export function buildContextText(result: DemoResult): string {
  const status = deriveStatus(result)
  const { report, ingest_report, graph } = result
  const claimById = new Map(graph.claims.map(c => [c.id, c]))
  const files = (ingest_report.per_file ?? []).map(f => `${baseName(f.source_file)} [${f.mode}]`)
  const L: string[] = []

  L.push('=== CLIENT (mock profile) ===')
  L.push(`Organisation: ${status.client.org}`)
  L.push(`Domain: ${status.client.domain}   Unit of measure: ${status.client.unit}`)
  L.push(`Operations desk: ${status.client.deskPhone}`)
  L.push(`Escalation: ${status.client.escalation} ${status.client.escalationPhone}`)
  L.push('')
  L.push('=== INPUT ===')
  L.push(`Files: ${files.length ? files.join('; ') : ingest_report.mode ?? 'unknown'}`)
  L.push(`Claims extracted: ${report.n_claims}   Unknowns: ${report.n_vars}`)
  if (ingest_report.timestamp_span_hours != null)
    L.push(`Timestamp span: ${ingest_report.timestamp_span_hours}h (window ${ingest_report.window_hours}h)`)
  L.push('')
  L.push('=== IDENTIFIABILITY ===')
  L.push(`rank ${report.rank}/${report.n_vars}  identifiable=${report.identifiable}  correctable_k=${report.correctable_k}`)
  L.push(report.correctable_k > 0
    ? `Up to ${report.correctable_k} corrupted report(s) can be provably located and corrected.`
    : 'correctable_k = 0: corruption CANNOT be guaranteed detectable; treat corrections as suggestive only.')
  L.push('')
  L.push(`=== NODES (${status.nodes.length}) — health: ok ${status.counts.ok}, degraded ${status.counts.degraded}, down ${status.counts.down} ===`)
  for (const n of status.nodes) {
    const c = n.contact
    L.push(
      `${n.id} | health=${n.health.toUpperCase()} | initial=${f1(n.initial)} reported=${f1(n.reported)} ` +
      `estimated=${f1(n.qty)} delta=${n.delta >= 0 ? '+' : ''}${n.delta.toFixed(1)} sink=${n.sink.toFixed(1)}` +
      ` | contact: ${c.name}, ${c.role}, ${c.phone}, callsign ${c.callsign}`,
    )
    for (const r of n.reasons) L.push(`    - ${r}`)
  }
  L.push('')
  L.push(`=== EDGES (${status.edges.length}) ===`)
  for (const e of status.edges) {
    L.push(
      `${e.id}: ${e.from} -> ${e.to} | reported=${f1(e.reported)} estimated=${f1(e.flow)}` +
      (e.corrected ? ' | CORRECTED' : '') +
      (e.flaggedClaims.length ? ` | flagged claims: ${e.flaggedClaims.join(', ')}` : ''),
    )
  }
  L.push('')
  L.push(`=== FLAGGED CLAIMS (${status.flagged.length}) ===`)
  if (status.flagged.length === 0) L.push('none — all reports consistent')
  for (const fl of status.flagged) {
    const c = claimById.get(fl.claim_id)
    const ref = c?.ref ?? (c?.refs ?? []).join(',')
    L.push(`${fl.claim_id} (${fl.type}) ref=${ref} claimed=${c?.value} residual=${fl.residual.toFixed(2)} source=${baseName(fl.source)}`)
  }
  L.push('')
  const MAX = 120
  L.push(`=== ALL CLAIMS (${graph.claims.length}${graph.claims.length > MAX ? `, first ${MAX} shown` : ''}) ===`)
  for (const c of graph.claims.slice(0, MAX)) {
    const ref = c.ref ?? (c.refs ?? []).join(',')
    L.push(`${c.id} ${c.type} ref=${ref} value=${c.value} w=${(c.weight ?? 1).toFixed(1)} src=${baseName(c.source)}`)
  }
  return L.join('\n')
}
