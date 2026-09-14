'use client'

import { useEffect, useMemo, useRef, useState } from 'react'

/**
 * Smoothly reveals `text` a few characters per frame. Network chunks arrive
 * in bursts; this paces them out so the reveal reads like steady typing, and
 * catches up quickly when far behind (e.g. text that arrived while unmounted).
 */
export function useSmoothText(text: string, streaming: boolean): string {
  const [shown, setShown] = useState(0)
  const acc = useRef(0)
  const raf = useRef<number | null>(null)
  const target = useRef(text)
  const revealedRef = useRef('')
  target.current = text

  useEffect(() => {
    // Text was replaced rather than appended (new run / new message) — restart.
    setShown(s => (s <= text.length && text.startsWith(revealedRef.current) ? s : 0))
  }, [text])

  useEffect(() => {
    const tick = () => {
      setShown(prev => {
        const total = target.current.length
        if (prev >= total) { acc.current = 0; return prev }
        const remaining = total - prev
        // ~1.6 chars/frame while live (≈95 chars/s); ramps up when far behind.
        const rate = streaming
          ? Math.min(1.6 + remaining / 60, 10)
          : Math.min(2.5 + remaining / 30, 24)
        acc.current += rate
        const step = Math.floor(acc.current)
        acc.current -= step
        return Math.min(total, prev + step)
      })
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => { if (raf.current != null) cancelAnimationFrame(raf.current) }
  }, [streaming])

  const revealed = text.slice(0, shown)
  revealedRef.current = revealed
  return revealed
}

// ─── Tiny markdown renderer with per-word fade-in ───────────────────────────

interface Inline { text: string; bold: boolean; code: boolean }

/** Split a line into bold / code / plain runs. Unclosed markers run to end. */
function parseInline(line: string): Inline[] {
  const out: Inline[] = []
  let bold = false, code = false, buf = ''
  const flush = () => { if (buf) { out.push({ text: buf, bold, code }); buf = '' } }
  for (let i = 0; i < line.length; i++) {
    if (!code && line[i] === '*' && line[i + 1] === '*') { flush(); bold = !bold; i++; continue }
    if (line[i] === '`') { flush(); code = !code; continue }
    buf += line[i]
  }
  flush()
  return out
}

/** Words (with trailing whitespace) so each can animate in independently. */
function words(run: Inline): Array<Inline & { key: string }> {
  const parts = run.text.match(/\S+\s*|\s+/g) ?? []
  return parts.map((p, i) => ({ ...run, text: p, key: String(i) }))
}

function InlineLine({ line, lineKey }: { line: string; lineKey: number }) {
  const runs = parseInline(line)
  let idx = 0
  return (
    <>
      {runs.map(run => words(run).map(w => {
        const k = `${lineKey}-${idx++}`
        const cls = `fade-w${w.bold ? ' md-b' : ''}${w.code ? ' md-code' : ''}`
        return <span key={k} className={cls}>{w.text}</span>
      }))}
    </>
  )
}

interface Block { kind: 'h' | 'li' | 'oli' | 'p' | 'blank'; text: string; n?: number; level?: number }

function toBlocks(text: string): Block[] {
  return text.split('\n').map(raw => {
    const line = raw.replace(/\s+$/, '')
    if (!line.trim()) return { kind: 'blank', text: '' }
    const h = line.match(/^(#{1,4})\s+(.*)$/)
    if (h) return { kind: 'h', level: h[1].length, text: h[2] }
    const li = line.match(/^\s*[-*•]\s+(.*)$/)
    if (li) return { kind: 'li', text: li[1] }
    const oli = line.match(/^\s*(\d+)[.)]\s+(.*)$/)
    if (oli) return { kind: 'oli', n: Number(oli[1]), text: oli[2] }
    return { kind: 'p', text: line }
  })
}

export function StreamText({
  text, streaming, className,
}: { text: string; streaming: boolean; className?: string }) {
  const shown = useSmoothText(text, streaming)
  const blocks = useMemo(() => toBlocks(shown), [shown])
  const typing = streaming || shown.length < text.length

  return (
    <div className={`md ${className ?? ''}`}>
      {blocks.map((b, i) => {
        if (b.kind === 'blank') return <div key={i} className="md-gap" />
        const inner = <InlineLine line={b.text} lineKey={i} />
        switch (b.kind) {
          case 'h':   return <div key={i} className={`md-h md-h${b.level}`}>{inner}</div>
          case 'li':  return <div key={i} className="md-li"><span className="md-marker">•</span><span>{inner}</span></div>
          case 'oli': return <div key={i} className="md-li"><span className="md-marker md-num">{b.n}</span><span>{inner}</span></div>
          default:    return <div key={i} className="md-p">{inner}</div>
        }
      })}
      {typing && <span className="md-caret" aria-hidden />}
    </div>
  )
}
