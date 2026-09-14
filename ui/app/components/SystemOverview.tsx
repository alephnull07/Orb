'use client'

import { useState } from 'react'
import { AlertOctagon, AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, Phone, Radio } from 'lucide-react'
import type { NodeHealth, NodeStatus, SystemStatus } from '../../lib/orb'

const HEALTH_LABEL: Record<NodeHealth, string> = { ok: 'healthy', degraded: 'degraded', down: 'down' }

function HealthIcon({ health }: { health: NodeHealth }) {
  if (health === 'down') return <AlertOctagon size={13} />
  if (health === 'degraded') return <AlertTriangle size={13} />
  return <CheckCircle2 size={13} />
}

const f1 = (v: number | null) => (v == null ? '—' : v.toFixed(1))

function NodeCard({ n, unit }: { n: NodeStatus; unit: string }) {
  const [open, setOpen] = useState(n.health !== 'ok')
  return (
    <div className={`sys-node sys-${n.health}`}>
      <button className="sys-node-head" onClick={() => setOpen(o => !o)}>
        <span className="sys-node-icon"><HealthIcon health={n.health} /></span>
        <span className="sys-node-id">{n.id}</span>
        <span className="sys-pill">{HEALTH_LABEL[n.health]}</span>
        {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
      </button>

      <div className="sys-node-nums">
        <span>est <b>{f1(n.qty)}</b> {unit}</span>
        {n.corrected && n.reported != null && (
          <span className="sys-was">reported {f1(n.reported)}</span>
        )}
        {Math.abs(n.sink) > 1 && <span className="sys-loss">loss {n.sink.toFixed(1)}</span>}
      </div>

      {open && (
        <div className="sys-node-body">
          <ul className="sys-reasons">
            {n.reasons.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
          <div className="sys-contact">
            <div className="sys-contact-name">
              {n.contact.name}
              <span className="sys-contact-role">{n.contact.role}</span>
            </div>
            <a className="sys-contact-line" href={`tel:${n.contact.phone.replace(/[^\d+]/g, '')}`}>
              <Phone size={10} /> {n.contact.phone}
            </a>
            <div className="sys-contact-line"><Radio size={10} /> {n.contact.callsign}</div>
          </div>
        </div>
      )}
    </div>
  )
}

export function SystemOverview({ status }: { status: SystemStatus }) {
  const { counts, nodes, client, flagged } = status
  const total = nodes.length || 1

  return (
    <div className="sys">
      <div className="section-label">system overview</div>

      {/* Health bar */}
      <div className="sys-bar" title={`${counts.ok} healthy · ${counts.degraded} degraded · ${counts.down} down`}>
        {counts.down > 0 && <i className="sys-bar-down" style={{ flex: counts.down / total }} />}
        {counts.degraded > 0 && <i className="sys-bar-degraded" style={{ flex: counts.degraded / total }} />}
        {counts.ok > 0 && <i className="sys-bar-ok" style={{ flex: counts.ok / total }} />}
      </div>
      <div className="sys-counts">
        <span className="sys-count-down"><b>{counts.down}</b> down</span>
        <span className="sys-count-degraded"><b>{counts.degraded}</b> degraded</span>
        <span className="sys-count-ok"><b>{counts.ok}</b> healthy</span>
        <span className="sys-count-flag"><b>{flagged.length}</b> flagged</span>
      </div>

      {/* Client card */}
      <div className="sys-client">
        <div className="sys-client-org">{client.org}</div>
        <div className="sys-client-row"><span>domain</span>{client.domain}</div>
        <div className="sys-client-row"><span>ops desk</span>{client.deskPhone}</div>
        <div className="sys-client-row"><span>escalation</span>{client.escalation}<br />{client.escalationPhone}</div>
      </div>

      <div className="section-label" style={{ marginTop: 18 }}>
        nodes
        <span style={{ float: 'right', fontFamily: 'var(--font-mono)', letterSpacing: 0 }}>{nodes.length}</span>
      </div>
      <div className="sys-nodes">
        {nodes.map(n => <NodeCard key={n.id} n={n} unit={client.unit} />)}
      </div>
    </div>
  )
}
