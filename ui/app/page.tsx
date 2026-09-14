'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
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
  Loader2, Network, ShieldAlert, ShieldCheck, Sparkles, Upload, X,
} from 'lucide-react'
import { DELTA_THRESHOLD, buildClaimedMap, type DemoResult } from '../lib/orb'
import { InsightsDashboard, type InsightsState } from './components/InsightsDashboard'
import type { ChatMessage } from './components/ChatPanel'

// ─── Types ──────────────────────────────────────────────────────────────────

type ViewMode = 'REPORTED' | 'L1'
type Screen = 'graph' | 'insights'

// ─── Helpers ────────────────────────────────────────────────────────────────

const NODE_W = 120
const NODE_H = 90

function computeLayout(
  nodes: Array<{ id: string }>,
  edges: Array<{ from: string; to: string }>,
): Map<string, { x: number; y: number }> {
  const g = new Dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'LR', nodesep: 60, ranksep: 140, edgesep: 20 })
  for (const n of nodes) g.setNode(n.id, { width: NODE_W, height: NODE_H })
  for (const e of edges) g.setEdge(e.from, e.to)
  Dagre.layout(g)
  return new Map(
    nodes.map(n => {
      const pos = g.node(n.id)
      return [n.id, { x: pos.x - NODE_W / 2, y: pos.y - NODE_H / 2 }]
    }),
  )
}

function inferNodeType(
  nodeId: string,
  edges: Array<{ from: string; to: string }>,
  node: { sinks?: string },
): string {
  const hasIncoming = edges.some(e => e.to === nodeId)
  if (!hasIncoming) return 'source'
  if (node.sinks === 'unknown') return 'sink'
  return 'junction'
}

const NODE_COLOR: Record<string, string> = {
  source: '#5b8def', junction: '#5ec2b7', sink: '#5ec2b7', unknown: '#9aa3b5',
}

// ─── ReactFlow: Node ────────────────────────────────────────────────────────

function OrbNode({ data }: NodeProps) {
  const color = NODE_COLOR[data.type] || NODE_COLOR.unknown
  const size = data.type === 'source' ? 48 : 42

  const isL1 = data.view === 'L1'
  const hasSink = isL1 && data.sinkVal > 1.0
  const hasDelta = isL1 && data.corrected
  const glow = hasSink
    ? '0 0 28px #e8952acc, 0 0 12px #e8952a88'
    : hasDelta
      ? '0 0 24px #e8952a99, 0 0 10px #e8952a66'
      : `0 0 16px ${color}55`
  const border = (hasSink || hasDelta) ? '#e8952a' : color

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 110 }}>
      <Handle type="target" position={Position.Left}
        style={{ top: size / 2, opacity: 0, width: 6, height: 6 }} />
      <Handle type="source" position={Position.Right}
        style={{ top: size / 2, opacity: 0, width: 6, height: 6 }} />
      <div style={{
        width: size, height: size, borderRadius: '50%',
        background: `radial-gradient(circle at 35% 32%, ${border}44 0%, ${border}11 70%)`,
        border: `2px solid ${border}`, boxShadow: glow, flexShrink: 0,
      }} />
      <div style={{
        marginTop: 6, fontSize: 10,
        fontFamily: 'var(--font-mono, ui-monospace, monospace)',
        fontWeight: 600,
        color: (hasSink || hasDelta) ? '#e8952a' : '#c8d1dc',
        textAlign: 'center', lineHeight: 1.3, whiteSpace: 'nowrap',
        maxWidth: 110, overflow: 'hidden', textOverflow: 'ellipsis',
      }}>
        {data.label}
      </div>
      {/* Primary value line */}
      {data.qtyLabel && (
        <div style={{
          fontSize: 9, color: hasDelta ? '#e8952a' : '#7a8a9a',
          fontFamily: 'var(--font-mono, ui-monospace, monospace)',
          marginTop: 2,
        }}>
          {data.qtyLabel}
        </div>
      )}
      {/* "was X" line in L1 view when corrected */}
      {data.wasLabel && (
        <div style={{
          fontSize: 8, color: '#9d6d42',
          fontFamily: 'var(--font-mono, ui-monospace, monospace)',
          marginTop: 1, textDecoration: 'line-through', opacity: 0.8,
        }}>
          {data.wasLabel}
        </div>
      )}
    </div>
  )
}

// ─── ReactFlow: Edge ────────────────────────────────────────────────────────

function OrbEdge({
  id, sourceX, sourceY, targetX, targetY,
  sourcePosition, targetPosition, data, selected,
}: EdgeProps) {
  const [path, lx, ly] = getBezierPath({
    sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition,
  })
  const isCorrected = data.view === 'L1' && data.corrected
  const color = isCorrected ? '#e8952a' : '#788395'

  return (
    <>
      <BaseEdge id={id} path={path} style={{
        stroke: color,
        strokeWidth: isCorrected ? 2.5 : 1.5,
        strokeDasharray: isCorrected ? '7 5' : undefined,
        filter: selected ? `drop-shadow(0 0 6px ${color})` : undefined,
      }} className={isCorrected ? 'edge-flow' : ''} />
      {data.flowLabel && (
        <EdgeLabelRenderer>
          <div className="edge-label" style={{
            position: 'absolute',
            transform: `translate(-50%,-50%) translate(${lx}px,${ly}px)`,
            borderColor: color, pointerEvents: 'all',
          }}>
            {isCorrected && data.wasFlow != null ? (
              <>
                <s style={{ color: '#9d6d42', marginRight: 3 }}>{data.wasFlow}</s>
                <span style={{ color: '#f0b967' }}>{data.flowLabel}</span>
              </>
            ) : (
              data.flowLabel
            )}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}

const nodeTypes = { orb: OrbNode }
const edgeTypes = { orb: OrbEdge }

// ─── Page ───────────────────────────────────────────────────────────────────

export default function Page() {
  const [status, setStatus]         = useState<'IDLE' | 'RUNNING' | 'DONE'>('IDLE')
  const [view, setView]             = useState<ViewMode>('L1')
  const [rawFiles, setRawFiles]     = useState<File[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [result, setResult]         = useState<DemoResult | null>(null)
  const [runError, setRunError]     = useState<string | null>(null)
  const [claimsOpen, setClaimsOpen] = useState(false)
  const [edgesOpen, setEdgesOpen]   = useState(false)
  const [nodesOpen, setNodesOpen]   = useState(false)
  const fileInputRef                = useRef<HTMLInputElement>(null)

  // ── Insights dashboard state (lifted so it survives switching screens) ──
  const [screen, setScreen]     = useState<Screen>('graph')
  const [insights, setInsights] = useState<InsightsState>({ text: '', streaming: false, error: null })
  const [chat, setChat]         = useState<ChatMessage[]>([])
  const insightsAbort           = useRef<AbortController | null>(null)

  const generateInsights = useCallback(async (r: DemoResult) => {
    insightsAbort.current?.abort()
    const ctrl = new AbortController()
    insightsAbort.current = ctrl
    setInsights({ text: '', streaming: true, error: null })
    try {
      const res = await fetch('/api/insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ result: r }),
        signal: ctrl.signal,
      })
      if (!res.ok || !res.body) {
        let msg = `HTTP ${res.status}`
        try { msg = (await res.json()).error ?? msg } catch { /* ignore */ }
        throw new Error(msg)
      }
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        const chunk = dec.decode(value, { stream: true })
        setInsights(s => ({ ...s, text: s.text + chunk }))
      }
      setInsights(s => ({ ...s, streaming: false }))
    } catch (e: any) {
      if (e?.name === 'AbortError') return
      setInsights(s => ({ ...s, streaming: false, error: e?.message ?? String(e) }))
    }
  }, [])

  // New estimation result → reset chat, stream a fresh briefing, show the dashboard.
  useEffect(() => {
    if (!result) return
    setChat([])
    setScreen('insights')
    generateInsights(result)
    return () => insightsAbort.current?.abort()
  }, [result, generateInsights])

  const acceptFiles = (list: FileList | null | undefined) => {
    if (!list) return
    const valid = Array.from(list).filter(f => /\.(txt|csv|json|jsonl|tsv|xlsx)$/i.test(f.name))
    setRawFiles(prev => {
      const names = new Set(prev.map(f => f.name))
      const fresh = valid.filter(f => !names.has(f.name))
      return [...prev, ...fresh]
    })
  }

  const handleRun = async () => {
    if (rawFiles.length === 0) return
    setStatus('RUNNING'); setRunError(null); setResult(null)
    try {
      const form = new FormData()
      for (const f of rawFiles) form.append('file', f)
      const r = await fetch('/api/demo', { method: 'POST', body: form })
      const d = await r.json()
      if (d.error) throw new Error(d.error)
      setResult(d); setStatus('DONE')
    } catch (e: any) {
      setRunError(String(e.message || e)); setStatus('IDLE')
    }
  }

  // ── Claimed (reported) value maps ──
  const { claimedNodeQty, claimedEdgeFlow } = useMemo(() => {
    if (!result) return { claimedNodeQty: new Map(), claimedEdgeFlow: new Map() }
    return {
      claimedNodeQty:  buildClaimedMap(result.graph.claims, 'node'),
      claimedEdgeFlow: buildClaimedMap(result.graph.claims, 'edge'),
    }
  }, [result])

  // ── Build ReactFlow graph — depends on result AND view ──
  const { flowNodes, flowEdges } = useMemo(() => {
    if (!result) return { flowNodes: [] as Node[], flowEdges: [] as Edge[] }
    const { graph, decoded } = result

    const l1Qty   = new Map(decoded.nodes.map(n => [n.id, n.qty]))
    const sinkMap = new Map(decoded.sinks.map(s => [s.id, s.sink]))
    const l1Flow  = new Map(decoded.edges.map(e => [e.id, e.flow]))

    const pos = computeLayout(graph.nodes, graph.edges)

    const flowNodes: Node[] = graph.nodes.map(n => {
      const reported = claimedNodeQty.get(n.id)
      const l1       = l1Qty.get(n.id)
      const sink     = sinkMap.get(n.id) ?? 0
      const delta    = (reported != null && l1 != null) ? Math.abs(l1 - reported) : 0
      const corrected = delta > DELTA_THRESHOLD

      let qtyLabel: string | undefined
      let wasLabel: string | undefined

      if (view === 'REPORTED') {
        qtyLabel = reported != null ? `qty ${reported.toFixed(1)}` : undefined
      } else {
        qtyLabel = l1 != null ? `qty ${l1.toFixed(1)}` : undefined
        if (corrected) wasLabel = `was ${reported!.toFixed(1)}`
      }

      return {
        id: n.id, type: 'orb',
        position: pos.get(n.id) ?? { x: 0, y: 0 },
        data: {
          label: n.id,
          type: inferNodeType(n.id, graph.edges, n),
          sinkVal: sink,
          view,
          corrected,
          qtyLabel,
          wasLabel,
        },
      }
    })

    const flowEdges: Edge[] = graph.edges.map(e => {
      const reported = claimedEdgeFlow.get(e.id)
      const l1       = l1Flow.get(e.id)
      const delta    = (reported != null && l1 != null) ? Math.abs(l1 - reported) : 0
      const corrected = delta > DELTA_THRESHOLD

      let flowLabel: string | undefined
      let wasFlow: string | undefined

      if (view === 'REPORTED') {
        flowLabel = reported != null ? reported.toFixed(1) : (l1 != null ? l1.toFixed(1) : undefined)
      } else {
        flowLabel = l1 != null ? l1.toFixed(1) : undefined
        if (corrected) wasFlow = reported!.toFixed(1)
      }

      return {
        id: e.id, source: e.from, target: e.to, type: 'orb',
        data: { flowLabel, wasFlow, view, corrected },
      }
    })

    return { flowNodes, flowEdges }
  }, [result, view, claimedNodeQty, claimedEdgeFlow])

  const report  = result?.report
  const decoded = result?.decoded
  const ingest  = result?.ingest_report

  const sinks = useMemo(
    () => decoded?.sinks?.slice().sort((a, b) => Math.abs(b.sink) - Math.abs(a.sink)) ?? [],
    [decoded],
  )

  const VIEW_LABELS: Record<ViewMode, string> = {
    REPORTED: 'Reported',
    L1: 'L1 corrected',
  }

  return (
    <main className="orb-shell">

      {/* ── Header ── */}
      <header className="orb-header">
        <div className="brand">
          <span className="brand-name">ORB</span>
          <div className="brand-divider" />
          <span className="brand-sub">outlier-robust estimation</span>
        </div>
        {result && (
          <div className="screen-toggle">
            <button className={screen === 'graph' ? 'active' : ''} onClick={() => setScreen('graph')}>
              <Network size={12} /> graph
            </button>
            <button className={screen === 'insights' ? 'active' : ''} onClick={() => setScreen('insights')}>
              <Sparkles size={12} /> insights
              {insights.streaming && <span className="status-dot running" style={{ marginLeft: 4 }} />}
            </button>
          </div>
        )}
        <div className="header-status">
          {status === 'IDLE' && <><span className="status-dot idle" /> awaiting data</>}
          {status === 'RUNNING' && <><span className="status-dot running" /> estimating...</>}
          {status === 'DONE' && (
            <>
              <span className="status-dot complete" />
              {ingest?.mode} · {report?.n_claims} claims · k={report?.correctable_k}
            </>
          )}
        </div>
      </header>

      {screen === 'insights' && result ? (
        <InsightsDashboard
          result={result}
          insights={insights}
          onRegenerate={() => generateInsights(result)}
          chat={chat}
          setChat={setChat}
        />
      ) : (
      <section className="orb-workspace">

        {/* ── Left: upload + mode info ── */}
        <aside className="side-panel left-panel">
          <div className="section-label">upload</div>
          <div
            className={`drop-zone ${isDragging ? 'dragging' : ''}`}
            onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={e => { e.preventDefault(); setIsDragging(false); acceptFiles(e.dataTransfer.files) }}
            onClick={() => fileInputRef.current?.click()}
          >
            <input ref={fileInputRef} type="file" multiple
              accept=".txt,.csv,.json,.jsonl,.tsv,.xlsx"
              style={{ display: 'none' }}
              onChange={e => acceptFiles(e.target.files)} />
            <Upload size={16} style={{ color: 'var(--text3)', flexShrink: 0 }} />
            <span className="drop-zone-hint">drop any data file</span>
            <span className="drop-zone-sub">.csv  .txt  .json</span>
          </div>

          {rawFiles.length > 0 && (
            <div className="file-list">
              {rawFiles.map(f => (
                <div key={f.name} className="file-item">
                  <span className="file-ext">{f.name.split('.').pop()}</span>
                  <span className="file-name">{f.name}</span>
                  <button className="file-remove"
                    onClick={() => setRawFiles(p => p.filter(x => x.name !== f.name))}
                  ><X size={10} /></button>
                </div>
              ))}
            </div>
          )}

          <button className="run-button"
            disabled={rawFiles.length === 0 || status === 'RUNNING'}
            onClick={handleRun}
          >
            {status === 'RUNNING'
              ? <><Loader2 className="spin" size={15} /> estimating...</>
              : <><Activity size={15} /> run estimation</>}
          </button>

          {runError && (
            <pre style={{
              color: '#e84040', fontSize: 9, marginTop: 8, lineHeight: 1.5,
              whiteSpace: 'pre-wrap', wordBreak: 'break-word',
              background: 'rgba(232,64,64,0.06)', borderRadius: 4, padding: '6px 8px',
              maxHeight: 120, overflow: 'auto',
              fontFamily: 'var(--font-mono), ui-monospace, monospace',
            }}>{runError}</pre>
          )}

          {/* ── Mode-specific info ── */}
          {result && ingest && (
            <>
              <div className="section-label" style={{ marginTop: 24 }}>
                {ingest.mode === 'TABULAR' ? 'column mapping' : 'record info'}
              </div>

              {ingest.mode === 'TABULAR' && ingest.mapping && (
                <div className="mode-info">
                  {Object.entries(ingest.mapping)
                    .filter(([, v]) => v != null && v !== '')
                    .map(([k, v]) => (
                      <div key={k} className="mode-info-row">
                        <span className="mode-info-key">{k}</span>
                        <span className="mode-info-val">
                          {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                        </span>
                      </div>
                    ))}
                  {ingest.exclusions && ingest.exclusions.length > 0 && (
                    <div className="mode-info-row">
                      <span className="mode-info-key">excluded</span>
                      <span className="mode-info-val">{ingest.exclusions.join(', ')}</span>
                    </div>
                  )}
                </div>
              )}

              {ingest.mode === 'RECORD' && (
                <div className="mode-info">
                  <div className="mode-info-row">
                    <span className="mode-info-key">records</span>
                    <span className="mode-info-val">{ingest.record_count ?? '?'}</span>
                  </div>
                  <div className="mode-info-row">
                    <span className="mode-info-key">edges</span>
                    <span className="mode-info-val">{result.graph.edges.length}</span>
                  </div>
                  <div className="mode-info-row">
                    <span className="mode-info-key">claims</span>
                    <span className="mode-info-val">{result.graph.claims.length}</span>
                  </div>
                  {(() => {
                    const ws = result.graph.claims.map(c => c.weight ?? 1)
                    if (ws.length === 0) return null
                    return (
                      <div className="mode-info-row">
                        <span className="mode-info-key">weight range</span>
                        <span className="mode-info-val">
                          {Math.min(...ws).toFixed(1)} – {Math.max(...ws).toFixed(1)}
                        </span>
                      </div>
                    )
                  })()}
                </div>
              )}
            </>
          )}
        </aside>

        {/* ── Centre: graph ── */}
        <section className="graph-panel">
          <div className="graph-toolbar">
            <div className="graph-toolbar-left">
              <div className="graph-title">network graph</div>
              {result && (
                <div className="graph-meta">
                  {result.graph.nodes.length} nodes · {result.graph.edges.length} edges ·{' '}
                  {decoded?.flagged.length ?? 0} flagged
                </div>
              )}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              {result && (
                <div className="view-toggle">
                  {(['REPORTED', 'L1'] as ViewMode[]).map(v => (
                    <button key={v}
                      className={view === v ? 'active' : ''}
                      onClick={() => setView(v)}
                    >
                      {VIEW_LABELS[v]}
                    </button>
                  ))}
                </div>
              )}
              {result && ingest && (
                <span className="mode-badge">{ingest.mode}</span>
              )}
            </div>
          </div>

          <div className="flow-wrap">
            {flowNodes.length > 0 ? (
              <ReactFlow
                nodes={flowNodes} edges={flowEdges}
                nodeTypes={nodeTypes} edgeTypes={edgeTypes}
                fitView minZoom={0.25}
              >
                <Background color="#1a2230" gap={24} size={1} />
                <Controls showInteractive={false} />
                <MiniMap
                  nodeColor={n =>
                    n.data?.type === 'source' ? '#5b8def'
                      : (n.data?.sinkVal ?? 0) > 1 ? '#e8952a' : '#5ec2b7'}
                  maskColor="rgba(7,9,13,.8)"
                />
              </ReactFlow>
            ) : (
              <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                height: '100%', color: '#3d4e60', flexDirection: 'column', gap: 14,
              }}>
                <Network size={36} strokeWidth={1} />
                <span style={{ fontSize: 11 }}>Upload a file and run estimation</span>
              </div>
            )}
          </div>

          {flowNodes.length > 0 && (
            <div className="legend">
              <span><i className="dot match" /> consistent</span>
              <span><i className="dot corrupted" /> corrected / leak</span>
            </div>
          )}
        </section>

        {/* ── Right: results ── */}
        <aside className="side-panel right-panel">
          {!result ? (
            <div style={{
              color: 'var(--text3)', fontSize: 11, textAlign: 'center', padding: '40px 10px',
            }}>
              Results will appear here after running estimation.
            </div>
          ) : (
            <>
              {/* ── Identifiability card ── */}
              <div className="ident-card">
                <div className="ident-header">
                  <span className="section-label" style={{ margin: 0 }}>identifiability</span>
                  {report!.correctable_k > 0
                    ? <ShieldCheck size={16} style={{ color: 'var(--green)' }} />
                    : <ShieldAlert size={16} style={{ color: 'var(--amber)' }} />}
                </div>
                <div className="ident-k">
                  <span className="ident-k-label">correctable_k</span>
                  <span className={`ident-k-value ${report!.correctable_k > 0 ? 'green' : 'red'}`}>
                    {report!.correctable_k}
                  </span>
                </div>
                {report!.correctable_k === 0 && (
                  <div className="ident-warning">
                    <AlertTriangle size={11} />
                    Cannot guarantee corruption detection
                  </div>
                )}
                <div className="ident-details">
                  <span>rank {report!.rank} / {report!.n_vars}</span>
                  <span>{report!.identifiable ? 'full rank' : 'rank deficient'}</span>
                </div>
              </div>

              {/* ── Sinks (ranked by magnitude) ── */}
              {sinks.length > 0 && (
                <>
                  <div className="section-label" style={{ marginTop: 18 }}>
                    sinks (ranked)
                    <span style={{
                      float: 'right', fontFamily: 'var(--font-mono)', letterSpacing: 0,
                    }}>{sinks.length}</span>
                  </div>
                  <div className="result-table">
                    {sinks.map((s, i) => {
                      const isTop = i === 0 && Math.abs(s.sink) > 1
                      return (
                        <div key={s.id}
                          className={`result-row ${isTop ? 'highlight' : ''}`}
                        >
                          <span className="result-rank">#{i + 1}</span>
                          <span className="result-id">{s.id}</span>
                          <span className={`result-val ${Math.abs(s.sink) > 1 ? 'amber' : ''}`}>
                            {s.sink.toFixed(2)}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </>
              )}

              {/* ── Flagged claims ── */}
              <div className="section-label" style={{ marginTop: 18 }}>
                flagged claims
                <span style={{
                  float: 'right', fontFamily: 'var(--font-mono)', letterSpacing: 0,
                }}>{decoded!.flagged.length}</span>
              </div>
              {decoded!.flagged.length > 0 ? (
                <div className="claims">
                  {decoded!.flagged.map(f => {
                    const claim = result.graph.claims.find(c => c.id === f.claim_id)
                    return (
                      <div key={f.claim_id} className="claim corrupted-border">
                        <span className="claim-content">
                          <span className="claim-edge">
                            {f.claim_id} ({f.type})
                          </span>
                          <span className="claim-residual">
                            |r| = {Math.abs(f.residual).toFixed(2)} · source: {f.source}
                          </span>
                          {claim && (
                            <span className="claim-residual">
                              ref: {claim.ref ?? (claim.refs || []).join(', ')} · claimed: {claim.value}
                            </span>
                          )}
                        </span>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div style={{
                  color: 'var(--green)', fontSize: 10, padding: '8px 0',
                  fontFamily: 'var(--font-mono)',
                }}>
                  No claims flagged — all consistent
                </div>
              )}

              {/* ── Decoded nodes (collapsible) ── */}
              <button className="table-toggle" onClick={() => setNodesOpen(o => !o)}>
                {nodesOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                decoded nodes ({decoded!.nodes.length})
              </button>
              {nodesOpen && (
                <div className="result-table">
                  {decoded!.nodes.map(n => {
                    const reported = claimedNodeQty.get(n.id)
                    const delta = reported != null ? Math.abs(n.qty - reported) : 0
                    return (
                      <div key={n.id} className={`result-row ${delta > DELTA_THRESHOLD ? 'highlight' : ''}`}>
                        <span className="result-id">{n.id}</span>
                        <span className="result-val">{n.qty.toFixed(2)}</span>
                        {delta > DELTA_THRESHOLD && (
                          <span style={{ fontSize: 9, color: '#9d6d42', textDecoration: 'line-through', flexShrink: 0 }}>
                            {reported!.toFixed(1)}
                          </span>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}

              {/* ── Decoded edges (collapsible) ── */}
              <button className="table-toggle" onClick={() => setEdgesOpen(o => !o)}>
                {edgesOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                decoded edges ({decoded!.edges.length})
              </button>
              {edgesOpen && (
                <div className="result-table">
                  {decoded!.edges.map(e => {
                    const reported = claimedEdgeFlow.get(e.id)
                    const delta = reported != null ? Math.abs(e.flow - reported) : 0
                    return (
                      <div key={e.id} className={`result-row ${delta > DELTA_THRESHOLD ? 'highlight' : ''}`}>
                        <span className="result-id" style={{ flex: 2 }}>
                          {e.from} → {e.to}
                        </span>
                        <span className="result-val">{e.flow.toFixed(2)}</span>
                        {delta > DELTA_THRESHOLD && (
                          <span style={{ fontSize: 9, color: '#9d6d42', textDecoration: 'line-through', flexShrink: 0 }}>
                            {reported!.toFixed(1)}
                          </span>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}

              {/* ── All claims (collapsible) ── */}
              <button className="table-toggle" onClick={() => setClaimsOpen(o => !o)}>
                {claimsOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                all claims ({result.graph.claims.length})
              </button>
              {claimsOpen && (
                <div className="result-table">
                  {result.graph.claims.map(c => {
                    const isFlagged = decoded!.flagged.some(f => f.claim_id === c.id)
                    return (
                      <div key={c.id}
                        className={`result-row ${isFlagged ? 'highlight' : ''}`}
                      >
                        <span className="result-rank" style={{ width: 32 }}>{c.id}</span>
                        <span className="result-id" style={{ width: 36 }}>{c.type}</span>
                        <span className="result-id" style={{ flex: 1, minWidth: 0 }}>
                          {c.ref ?? (c.refs || []).join(',')}
                        </span>
                        <span className="result-val" style={{ width: 54, textAlign: 'right' }}>
                          {typeof c.value === 'number' ? c.value.toFixed(1) : c.value}
                        </span>
                        <span className="result-val"
                          style={{ width: 36, textAlign: 'right', color: 'var(--text3)' }}
                        >
                          w{(c.weight ?? 1).toFixed(0)}
                        </span>
                      </div>
                    )
                  })}
                </div>
              )}
            </>
          )}
        </aside>

      </section>
      )}
    </main>
  )
}
