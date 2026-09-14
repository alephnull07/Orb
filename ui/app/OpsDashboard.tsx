'use client'

import { AlertTriangle, Crosshair, ListChecks, MapPin } from 'lucide-react'

export interface AdviceAction {
  priority: number
  kind: string
  title: string
  do: string
  where: string[]
  why: string
}

export interface AdviceSensor {
  at: string
  kind: string
  instrument: string
  covers_hops: Array<{ id?: string; from?: string; to?: string }>
  effect: string
  reason: string
  required?: boolean
}

export interface AdviceOnHand {
  id: string
  opening: number
  reported: number | null
  l1: number | null
  delta: number | null
  action: string
}

export interface AdviceHop {
  id?: string
  from?: string
  to?: string
  l1_flow?: number | null
  tag: string
  sent?: number
  received?: number
}

export interface Advice {
  headline: string
  severity: 'critical' | 'watch' | 'clear' | string
  correctable_k?: number
  actions: AdviceAction[]
  sensors: AdviceSensor[]
  on_hand: AdviceOnHand[]
  hops: { trust: AdviceHop[]; audit: AdviceHop[] }
}

function fmt(n: number | null | undefined, digits = 0) {
  if (n == null || Number.isNaN(n)) return '—'
  return n.toFixed(digits)
}

const KIND_LABEL: Record<string, string> = {
  resupply: 'resupply',
  recount: 'recount',
  hold: 'hold lift',
  do_not_treat_as_leak: 'count — do not assume leak',
  verify_hop: 'verify hop',
  place_sensor: 'place sensor',
  cannot_certify: 'cannot certify',
  clear: 'no action',
}

function SensorKind({ kind }: { kind: string }) {
  if (kind === 'drain_meter') return <>drain meter</>
  if (kind === 'second_qty') return <>second qty sensor</>
  return <>independent closeout</>
}

export function OpsDashboard({ advice }: { advice: Advice }) {
  const actions = advice.actions || []
  const sensors = [...(advice.sensors || [])].sort(
    (a, b) => Number(b.required !== false) - Number(a.required !== false),
  )
  const onHand = advice.on_hand || []
  const audit = advice.hops?.audit || []
  const trust = advice.hops?.trust || []
  const sev = advice.severity || 'watch'

  return (
    <div className="ops-dash">
      <div className={`ops-hero ${sev}`}>
        <div className="ops-hero-kicker">what to do</div>
        <div className="ops-hero-line">{advice.headline}</div>
      </div>

      <section className="ops-section">
        <div className="section-label">
          <ListChecks size={11} style={{ marginRight: 6, verticalAlign: -1 }} />
          do this now
          <span className="ops-count">{actions.length}</span>
        </div>
        <ol className="ops-actions">
          {actions.map((a, i) => (
            <li key={`${a.kind}-${a.title}-${i}`} className={`ops-action p${a.priority}`}>
              <div className="ops-action-meta">
                <span className="ops-pri">P{a.priority}</span>
                <span className="ops-kind">{KIND_LABEL[a.kind] || a.kind}</span>
                {a.where?.length > 0 && (
                  <span className="ops-where">{a.where.join(' · ')}</span>
                )}
              </div>
              <div className="ops-action-title">{a.title}</div>
              <div className="ops-action-do">{a.do}</div>
              {a.why && <div className="ops-action-why">{a.why}</div>}
            </li>
          ))}
        </ol>
      </section>

      <section className="ops-section">
        <div className="section-label">
          <Crosshair size={11} style={{ marginRight: 6, verticalAlign: -1 }} />
          sensor placement
          <span className="ops-count">{sensors.length}</span>
        </div>
        {sensors.length === 0 ? (
          <div className="ops-empty">
            No new sensor required for identifiability on this snapshot.
          </div>
        ) : (
          <div className="ops-sensors">
            {sensors.map((s, i) => (
              <article key={`${s.at}-${s.kind}-${i}`} className="ops-sensor">
                <div className="ops-sensor-head">
                  <MapPin size={12} />
                  <span className="ops-sensor-at">{s.at}</span>
                  <span className="ops-sensor-kind">
                    {s.required === false ? 'hardening' : 'required'} · <SensorKind kind={s.kind} />
                  </span>
                </div>
                <div className="ops-sensor-do">{s.instrument}</div>
                <div className="ops-sensor-effect">{s.effect}</div>
                {s.covers_hops?.length > 0 && (
                  <div className="ops-sensor-hops">
                    covers{' '}
                    {s.covers_hops.map(h => `${h.from} → ${h.to}`).join(' · ')}
                  </div>
                )}
                <div className="ops-action-why">{s.reason}</div>
              </article>
            ))}
          </div>
        )}
      </section>

      <div className="ops-grid">
        <section className="ops-section">
          <div className="section-label">on hand (plan on L1)</div>
          <div className="result-table">
            {onHand.map(r => (
              <div key={r.id} className={`result-row ${r.action.startsWith('recount') || r.action.startsWith('resupply') ? 'highlight' : ''}`}>
                <span className="result-id">{r.id}</span>
                <span className="result-val" title="L1">{fmt(r.l1, 0)}</span>
                {r.reported != null && r.delta != null && r.delta > 0.5 && (
                  <span style={{ fontSize: 9, color: '#9d6d42', textDecoration: 'line-through', flexShrink: 0 }}>
                    {fmt(r.reported, 0)}
                  </span>
                )}
                <span className="ops-row-note">{r.action}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="ops-section">
          <div className="section-label">
            hops to audit
            <span className="ops-count">{audit.length}</span>
          </div>
          {audit.length === 0 ? (
            <div className="ops-empty">No dual-only or flagged hops.</div>
          ) : (
            <div className="result-table">
              {audit.map(h => (
                <div key={h.id || `${h.from}-${h.to}`} className="result-row">
                  <span className="result-id" style={{ flex: 2 }}>
                    {h.from} → {h.to}
                  </span>
                  <span className="ops-row-note">{h.tag}</span>
                </div>
              ))}
            </div>
          )}
          {trust.length > 0 && (
            <div className="ops-empty" style={{ marginTop: 10 }}>
              {trust.length} hop{trust.length === 1 ? '' : 's'} consistent with conservation
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

export function NextActions({ advice, onOpen }: { advice: Advice; onOpen: () => void }) {
  const top = (advice.actions || []).slice(0, 3)
  if (!top.length) return null
  return (
    <button className="ops-next" onClick={onOpen} type="button">
      <div className="ops-next-head">
        <AlertTriangle size={12} />
        <span>what to do</span>
        <span className={`ops-pill ${advice.severity}`}>{advice.severity}</span>
      </div>
      <div className="ops-next-line">{advice.headline}</div>
      <div className="ops-next-hint">open ops · {top[0].title}</div>
    </button>
  )
}
