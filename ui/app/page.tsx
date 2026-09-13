'use client'

import { useMemo, useRef, useState } from 'react'
import Dagre from '@dagrejs/dagre'
import ReactFlow, {
  Background, Controls, MiniMap,
  Handle, Position,
  BaseEdge, EdgeLabelRenderer, getBezierPath,
  type EdgeProps, type NodeProps,
  type Node, type Edge,
} from 'reactflow'
import 'reactflow/dist/style.css'
import {
  Activity, AlertTriangle, ChevronDown, ChevronUp,
  Ghost, Loader2, Network, Upload, X,
} from 'lucide-react'

// ─── Types ────────────────────────────────────────────────────────────────────

type EdgeStatus = 'MATCH' | 'CORRUPTED' | 'GHOST'
type ViewMode   = 'ORIGINAL' | 'L1' | 'TRUTH'

interface RawNode { id: string; display?: string; type?: string }
interface RawEdge {
  source: string; target: string
  source_display?: string; target_display?: string
  source_name?: string;   target_name?: string
  value_lb: number; method?: string
  agent_values?: Record<string, number>
}
interface GraphData { nodes: RawNode[]; edges: RawEdge[] }

interface EnrichedEdge {
  id: string; source: string; target: string
  srcDisplay: string; tgtDisplay: string
  consensusVal: number; l1Val: number | null; truthVal: number | null
  status: EdgeStatus
  agentValues: Record<string, number>
  method: string
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const keyOf = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()

function short(s: string) {
  const p = s.split(/\s+/)
  return p.length <= 2 ? s : p.slice(0, 2).join(' ')
}

// ─── Dagre layout (LR hierarchical, auto-spaced) ─────────────────────────────

const NODE_W = 110  // bounding box width  (orb + label)
const NODE_H = 80   // bounding box height

function computeLayout(nodes: RawNode[], edges: RawEdge[]): Map<string, { x: number; y: number }> {
  const g = new Dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'LR', nodesep: 60, ranksep: 160, edgesep: 20 })

  for (const n of nodes) g.setNode(n.id, { width: NODE_W, height: NODE_H })
  // filter ghost-suppressed edges (value ~0) so they don't distort layout
  for (const e of edges) if (e.value_lb >= 1) g.setEdge(e.source, e.target)

  Dagre.layout(g)

  return new Map(
    nodes.map(n => {
      const pos = g.node(n.id)
      return [n.id, { x: pos.x - NODE_W / 2, y: pos.y - NODE_H / 2 }]
    })
  )
}

// ─── Edge enrichment ─────────────────────────────────────────────────────────

function enrichEdges(corrupted: GraphData, l1: GraphData | null, truth: GraphData | null): EnrichedEdge[] {
  // l1 and corrupted share the same node IDs — match directly
  const l1Map = new Map(l1?.edges.map(e => [`${e.source}|${e.target}`, e.value_lb]) ?? [])

  // truth uses different IDs — match by display name
  const truthMap = new Map(
    truth?.edges.map(e => {
      const src = keyOf(e.source_name || e.source_display || e.source)
      const tgt = keyOf(e.target_name || e.target_display || e.target)
      return [`${src}>>${tgt}`, e.value_lb]
    }) ?? []
  )

  return corrupted.edges.map(e => {
    const l1Val    = l1Map.get(`${e.source}|${e.target}`) ?? null
    const dispKey  = `${keyOf(e.source_display || e.source)}>>${keyOf(e.target_display || e.target)}`
    const truthVal = truthMap.get(dispKey) ?? null

    let status: EdgeStatus = 'MATCH'
    if (l1Val === null || Math.abs(l1Val) < 1)              status = 'GHOST'
    else if (Math.abs(l1Val - e.value_lb) > 3)              status = 'CORRUPTED'

    return {
      id: `${e.source}|${e.target}`,
      source: e.source, target: e.target,
      srcDisplay: e.source_display || e.source,
      tgtDisplay: e.target_display || e.target,
      consensusVal: e.value_lb, l1Val, truthVal,
      status,
      agentValues: e.agent_values ?? {},
      method: e.method ?? '',
    }
  })
}

// ─── Colors ──────────────────────────────────────────────────────────────────

const NODE_COLOR: Record<string, string> = {
  source: '#5b8def', hub: '#5ec2b7', junction: '#5ec2b7', sink: '#d4a054', unknown: '#9aa3b5',
}
const EDGE_COLOR: Record<EdgeStatus, string> = {
  MATCH: '#788395', CORRUPTED: '#e0a14a', GHOST: '#e05d5d',
}

// ─── React Flow: Node (actual orb / circle) ───────────────────────────────────

function OrbNode({ data }: NodeProps) {
  const color  = NODE_COLOR[data.type] || NODE_COLOR.unknown
  const size   = data.type === 'source' ? 48 : data.type === 'sink' ? 36 : 42
  const glow   = data.flagged
    ? `0 0 28px ${color}cc, 0 0 12px ${color}88`
    : `0 0 16px ${color}55`

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 96 }}>
      {/* Handles anchored to the orb centre, not the whole div */}
      <Handle type="target" position={Position.Left}
        style={{ top: size / 2, opacity: 0, width: 6, height: 6 }} />
      <Handle type="source" position={Position.Right}
        style={{ top: size / 2, opacity: 0, width: 6, height: 6 }} />

      {/* The orb */}
      <div style={{
        width: size, height: size, borderRadius: '50%',
        background: `radial-gradient(circle at 35% 32%, ${color}44 0%, ${color}11 70%)`,
        border: `2px solid ${color}`,
        boxShadow: glow,
        flexShrink: 0,
      }} />

      {/* Label below the orb */}
      <div style={{
        marginTop: 7, fontSize: 10, fontFamily: 'var(--font-mono, ui-monospace, monospace)',
        fontWeight: 600, color: data.flagged ? '#e0a14a' : '#c8d1dc',
        textAlign: 'center', lineHeight: 1.3, whiteSpace: 'nowrap',
        maxWidth: 96, overflow: 'hidden', textOverflow: 'ellipsis',
      }}>
        {data.shortLabel}
      </div>
      <div style={{
        fontSize: 8, color: '#3d4e60', textTransform: 'uppercase',
        letterSpacing: '0.1em', marginTop: 2,
      }}>
        {data.type}
      </div>
    </div>
  )
}

// ─── React Flow: Edge ────────────────────────────────────────────────────────

function OrbEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, selected }: EdgeProps) {
  const [path, lx, ly] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition })
  const color   = EDGE_COLOR[data.status as EdgeStatus]
  const isDashed = data.status !== 'MATCH'

  // ORIGINAL: show what was reported (wrong values highlighted)
  // L1: show corrected values, still highlight flagged edges
  // TRUTH: show truth, ghost edges are already filtered out upstream
  const displayVal: number | null =
    data.view === 'ORIGINAL' ? data.consensusVal :
    data.view === 'L1'       ? data.l1Val :
    data.truthVal

  return (
    <>
      <BaseEdge id={id} path={path} style={{
        stroke: color,
        strokeWidth: data.status === 'MATCH' ? 1.5 : 2.5,
        strokeDasharray: isDashed ? '7 5' : undefined,
        filter: selected ? `drop-shadow(0 0 6px ${color})` : undefined,
      }} className={isDashed ? 'edge-flow' : ''} />

      <EdgeLabelRenderer>
        <div className="edge-label" style={{
          position: 'absolute',
          transform: `translate(-50%,-50%) translate(${lx}px,${ly}px)`,
          borderColor: color, cursor: 'pointer',
          pointerEvents: 'all',
        }}>
          {data.status === 'GHOST' && <Ghost size={10} />}
          {/* ORIGINAL: corrupted edge shows wrong value ~~X~~ → corrected value */}
          {data.status === 'CORRUPTED' && data.view === 'ORIGINAL'
            ? <>
                <s style={{ color: '#9d6d42', marginRight: 2 }}>{data.consensusVal} lb</s>
                <span style={{ color: '#f0b967' }}>→ {data.l1Val?.toFixed(0)} lb</span>
              </>
            : displayVal != null
              ? `${Number(displayVal).toFixed(0)} lb`
              : null
          }
        </div>
      </EdgeLabelRenderer>
    </>
  )
}

const nodeTypes = { orb: OrbNode }
const edgeTypes = { orb: OrbEdge }

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function Page() {
  const [appStatus, setAppStatus]   = useState<'IDLE' | 'RUNNING' | 'COMPLETE'>('IDLE')
  const [view, setView]             = useState<ViewMode>('ORIGINAL')
  const [rawFiles, setRawFiles]     = useState<File[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [graphs, setGraphs]         = useState<{ corrupted: GraphData; l1: GraphData; truth: GraphData } | null>(null)
  const [selected, setSelected]     = useState<EnrichedEdge | null>(null)
  const [tableOpen, setTableOpen]   = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [runError, setRunError]     = useState<string | null>(null)
  const fileInputRef                = useRef<HTMLInputElement>(null)

  const acceptFiles = (list: FileList | null | undefined) => {
    if (!list) return
    const valid = Array.from(list).filter(f => /\.(txt|jsonl|csv|json)$/i.test(f.name))
    setRawFiles(prev => {
      const seen = new Set(prev.map(f => f.name))
      return [...prev, ...valid.filter(f => !seen.has(f.name))]
    })
  }

  const loadGraphs = () =>
    fetch('/api/graphs').then(r => r.json()).then(d => { setGraphs(d); setAppStatus('COMPLETE') })

  // Enriched edges: corrupted vs l1 vs truth
  const enriched = useMemo(
    () => graphs ? enrichEdges(graphs.corrupted, graphs.l1, graphs.truth) : [],
    [graphs]
  )

  const flagged = useMemo(() => enriched.filter(e => e.status !== 'MATCH'), [enriched])

  const stats = useMemo(() => {
    const real   = enriched.filter(e => e.status !== 'GHOST' && e.truthVal !== null && e.l1Val !== null)
    const errors = real.map(e => Math.abs((e.l1Val ?? 0) - (e.truthVal ?? 0)))
    return {
      meanErr:   errors.length ? errors.reduce((a, b) => a + b, 0) / errors.length : 0,
      exact:     errors.filter(e => e < 1).length,
      realEdges: enriched.filter(e => e.status !== 'GHOST').length,
      corrupted: enriched.filter(e => e.status === 'CORRUPTED').length,
      ghost:     enriched.filter(e => e.status === 'GHOST').length,
    }
  }, [enriched])

  // Build react-flow nodes (layout computed once from corrupted graph)
  const { flowNodes, baseEdges } = useMemo(() => {
    if (!graphs) return { flowNodes: [], baseEdges: [] }
    const pos         = computeLayout(graphs.corrupted.nodes, graphs.corrupted.edges)
    const flaggedIds  = new Set(flagged.flatMap(e => [e.source, e.target]))

    const flowNodes: Node[] = graphs.corrupted.nodes.map(n => ({
      id: n.id, type: 'orb',
      position: pos.get(n.id) ?? { x: 0, y: 0 },
      data: { shortLabel: short(n.display || n.id), type: n.type || 'unknown', flagged: flaggedIds.has(n.id) },
    }))

    const baseEdges: Edge[] = enriched.map(e => ({
      id: e.id, source: e.source, target: e.target, type: 'orb',
      data: { ...e, view },
    }))

    return { flowNodes, baseEdges }
  }, [graphs, enriched, flagged])

  // Only flag nodes/edges in L1 view — ORIGINAL and TRUTH look clean
  const viewNodes = useMemo(() => {
    if (view === 'L1') return flowNodes
    return flowNodes.map(n => ({ ...n, data: { ...n.data, flagged: false } }))
  }, [flowNodes, view])

  const viewEdges = useMemo(() => {
    const neutralise = view === 'ORIGINAL' || view === 'TRUTH'
    let edges = baseEdges.map(e => ({
      ...e,
      data: {
        ...e.data,
        view,
        status: neutralise ? 'MATCH' : e.data.status,
      },
    }))
    if (view === 'TRUTH') edges = edges.filter(e => e.data.truthVal !== null)
    return edges
  }, [baseEdges, view])

  const allLoaded = rawFiles.length > 0

  const handleRun = async () => {
    setAppStatus('RUNNING'); setRunError(null)
    try {
      const form = new FormData()
      for (const f of rawFiles) form.append('files', f)
      const r = await fetch('/api/run', { method: 'POST', body: form })
      const d = await r.json()
      if (!d.ok) throw new Error(d.error)
      await loadGraphs()
    } catch (e) {
      setRunError(String(e)); setAppStatus('IDLE')
    }
  }

  const VIEW_LABELS: Record<ViewMode, string> = {
    ORIGINAL: 'Original',
    L1:       'L1 minimized',
    TRUTH:    'Ground truth',
  }

  return (
    <main className="orb-shell">

      {/* ── Header ── */}
      <header className="orb-header">
        <div className="brand">
          <span className="brand-name">ORB</span>
          <div className="brand-divider" />
          <span className="brand-sub">supply chain integrity monitor</span>
        </div>

        {/* Status: no pill, just dot + text */}
        <div className="header-status">
          {appStatus === 'IDLE' && (
            <>
              <span className="status-dot idle" />
              awaiting data
            </>
          )}
          {appStatus === 'RUNNING' && (
            <>
              <span className="status-dot running" />
              computing flows
            </>
          )}
          {appStatus === 'COMPLETE' && (
            <>
              <span className="status-dot complete" />
              {stats.realEdges} / {enriched.length} flows recovered
            </>
          )}
        </div>
      </header>

      <section className="orb-workspace">

        {/* ── Left: file upload ── */}
        <aside className="side-panel left-panel">
          <div className="section-label">data feeds</div>

          {/* Multi-file drop zone for raw comms files */}
          <div
            className={`drop-zone ${isDragging ? 'dragging' : ''}`}
            onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={e => { e.preventDefault(); setIsDragging(false); acceptFiles(e.dataTransfer.files) }}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file" accept=".txt,.jsonl,.csv,.json"
              multiple style={{ display: 'none' }}
              onChange={e => acceptFiles(e.target.files)}
            />
            <Upload size={16} style={{ color: 'var(--text3)', flexShrink: 0 }} />
            <span className="drop-zone-hint">drop comms files</span>
            <span className="drop-zone-sub">.txt · .jsonl · .csv</span>
          </div>

          {rawFiles.length > 0 && (
            <div className="file-list">
              {rawFiles.map(f => {
                const ext = f.name.split('.').pop() ?? ''
                return (
                  <div key={f.name} className="file-item">
                    <span className="file-ext">{ext}</span>
                    <span className="file-name">{f.name}</span>
                    <button
                      className="file-remove"
                      onClick={() => setRawFiles(p => p.filter(x => x.name !== f.name))}
                    ><X size={10} /></button>
                  </div>
                )
              })}
            </div>
          )}

          <button className="run-button" disabled={!allLoaded || appStatus === 'RUNNING'} onClick={handleRun}>
            {appStatus === 'RUNNING'
              ? <><Loader2 className="spin" size={15} /> estimating L1...</>
              : <><Activity size={15} /> run L1 estimation</>}
          </button>

          {runError && (
            <p style={{ color: '#e84040', fontSize: 10, marginTop: 8, lineHeight: 1.5 }}>{runError}</p>
          )}

          <button className="demo-link" onClick={loadGraphs}>
            {appStatus === 'COMPLETE' ? 'reload demo data' : 'use demo data'}
          </button>
        </aside>

        {/* ── Centre: graph canvas ── */}
        <section className="graph-panel">
          <div className="graph-toolbar">
            <div className="graph-toolbar-left">
              <div className="graph-title">flow network</div>
              {graphs && (
                <div className="graph-meta">
                  {graphs.corrupted.nodes.length} nodes · {graphs.corrupted.edges.length} edges · {flagged.length} anomalies
                </div>
              )}
            </div>
            <div className="view-toggle">
              {(['ORIGINAL', 'L1', 'TRUTH'] as ViewMode[]).map(v => (
                <button key={v} className={view === v ? 'active' : ''} onClick={() => setView(v)}>
                  {VIEW_LABELS[v]}
                </button>
              ))}
            </div>
          </div>

          <div className="flow-wrap">
            {flowNodes.length > 0 ? (
              <ReactFlow
                nodes={viewNodes} edges={viewEdges}
                nodeTypes={nodeTypes} edgeTypes={edgeTypes}
                fitView minZoom={0.25}
                onEdgeClick={(_, edge) => setSelected(enriched.find(e => e.id === edge.id) ?? null)}
              >
                <Background color="#1a2230" gap={24} size={1} />
                <Controls showInteractive={false} />
                <MiniMap
                  nodeColor={n =>
                    n.data?.type === 'source' ? '#5b8def' :
                    n.data?.type === 'sink'   ? '#d4a054' : '#5ec2b7'}
                  maskColor="rgba(7,9,13,.8)"
                />
              </ReactFlow>
            ) : (
              <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                height: '100%', color: '#3d4e60', flexDirection: 'column', gap: 14,
              }}>
                <Network size={36} strokeWidth={1} />
                <span style={{ fontSize: 11, color: '#3d4e60' }}>Upload files and run, or use demo data</span>
              </div>
            )}
          </div>

          <div className="legend">
            <span><i className="dot match" /> match</span>
            <span><i className="dot corrupted" /> corrupted</span>
            <span><i className="dot ghost" /> ghost</span>
          </div>
        </section>

        {/* ── Right: results ── */}
        <aside className="side-panel right-panel">
          <div className="section-label">analysis</div>

          {/* Stats as key/value rows */}
          <div className="stats-rows">
            <div className="stat-row">
              <span className="stat-row-label">mean error</span>
              <span className={`stat-row-value ${stats.meanErr < 1 ? 'green' : stats.meanErr < 5 ? 'amber' : 'red'}`}>
                {stats.meanErr.toFixed(1)}
                <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--text3)', marginLeft: 3 }}>lb</span>
              </span>
            </div>
            <div className="stat-row">
              <span className="stat-row-label">anomalies</span>
              <span className={`stat-row-value ${(stats.corrupted + stats.ghost) > 0 ? 'amber' : 'green'}`}>
                {stats.corrupted + stats.ghost}
              </span>
            </div>
            <div className="stat-row">
              <span className="stat-row-label">exact matches</span>
              <span className={`stat-row-value ${stats.exact === stats.realEdges && stats.realEdges > 0 ? 'green' : 'amber'}`}>
                {stats.exact}
                <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--text3)', marginLeft: 3 }}>/ {stats.realEdges}</span>
              </span>
            </div>
          </div>

          <div className="anomalies-header">
            <span className="anomalies-label">anomalies</span>
            <span className="anomalies-count">{flagged.length}</span>
          </div>

          <div className="claims">
            {flagged.map(e => (
              <button
                key={e.id}
                className={`claim ${e.status === 'CORRUPTED' ? 'corrupted-border' : 'ghost-border'} ${selected?.id === e.id ? 'selected' : ''}`}
                onClick={() => setSelected(e)}
              >
                <span className="claim-content">
                  <span className="claim-edge">{short(e.srcDisplay)} → {short(e.tgtDisplay)}</span>
                  <span className="claim-residual">
                    {e.status === 'CORRUPTED'
                      ? <>residual +{Math.round(Math.abs((e.l1Val ?? 0) - e.consensusVal))} lb</>
                      : 'no matching true edge'}
                  </span>
                </span>
                <span className={`claim-tag ${e.status.toLowerCase()}`}>{e.status}</span>
              </button>
            ))}
          </div>

          <button className="table-toggle" onClick={() => setTableOpen(o => !o)}>
            {tableOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            raw edge table
          </button>
          {tableOpen && (
            <div className="raw-table">
              <div><span>edge</span><span>cons / L1 / truth</span></div>
              {enriched.map(e => (
                <p key={e.id} style={{ cursor: 'pointer' }} onClick={() => setSelected(e)}>
                  <span style={{ color: EDGE_COLOR[e.status] }}>
                    {short(e.srcDisplay)} → {short(e.tgtDisplay)}
                  </span>
                  <span>{e.consensusVal} / {e.l1Val?.toFixed(0) ?? '—'} / {e.truthVal ?? '?'}</span>
                </p>
              ))}
            </div>
          )}

          {/* Selected edge detail */}
          {selected && (
            <div className="agent-detail">
              <div className="agent-detail-subhead">
                <span className="agent-detail-label">edge detail</span>
                <span className={`agent-detail-status ${selected.status.toLowerCase()}`}>{selected.status}</span>
              </div>
              <span className="agent-detail-edge">
                {short(selected.srcDisplay)} → {short(selected.tgtDisplay)}
              </span>
              <div className="agent-values">
                {Object.entries(selected.agentValues).map(([agent, val]) => (
                  <span key={agent}>
                    {agent}
                    <strong>{val} lb</strong>
                  </span>
                ))}
                <span>
                  L1 result
                  <strong style={{ color: '#30c97e' }}>{selected.l1Val?.toFixed(0) ?? '—'} lb</strong>
                </span>
                <span>
                  truth
                  <strong style={{ color: '#7a8a9a' }}>{selected.truthVal ?? '?'} lb</strong>
                </span>
              </div>
              {selected.method && (
                <p style={{ color: '#3d4e60', marginTop: 10, fontSize: 10 }}>
                  method: <strong style={{ color: '#7a8a9a', fontFamily: 'var(--font-mono, ui-monospace, monospace)' }}>{selected.method}</strong>
                </p>
              )}
            </div>
          )}
        </aside>
      </section>

      {/* ── Bottom diff drawer ── */}
      <button className="diff-drawer" onClick={() => setDrawerOpen(o => !o)}>
        <span className="diff-drawer-left">
          <AlertTriangle size={13} />
          {enriched.length} edges · {flagged.length} anomalies
        </span>
        <span className="diff-drawer-chevron">
          {drawerOpen ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
        </span>
      </button>
      {drawerOpen && (
        <div className="diff-content">
          <div>
            <span>status</span>
            <span>from → to</span>
            <span>consensus lb</span>
            <span>L1 lb</span>
            <span>delta</span>
          </div>
          {enriched.map(e => (
            <p key={e.id} style={{ cursor: 'pointer' }} onClick={() => setSelected(e)}>
              <b className={e.status.toLowerCase()}>{e.status}</b>
              <span>{short(e.srcDisplay)} → {short(e.tgtDisplay)}</span>
              <span>{e.consensusVal}</span>
              <span>{e.l1Val?.toFixed(0) ?? '—'}</span>
              <span style={{ color: e.status === 'MATCH' ? '#3d4e60' : EDGE_COLOR[e.status] }}>
                {e.l1Val !== null ? Math.round(e.l1Val - e.consensusVal) : '—'}
              </span>
            </p>
          ))}
        </div>
      )}
    </main>
  )
}
