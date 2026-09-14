'use client'

import { useEffect, useRef, useState } from 'react'
import { MessageSquare, Send, Square } from 'lucide-react'
import type { DemoResult } from '../../lib/orb'
import { StreamText } from './StreamText'

export interface ChatMessage { role: 'user' | 'assistant'; content: string }

const SUGGESTIONS = [
  'What should I do first?',
  'Which site do I need to call right now?',
  'Why were those reports rejected?',
  'How much should I trust this estimate?',
]

export function ChatPanel({
  result, briefing, messages, setMessages,
}: {
  result: DemoResult
  briefing: string
  messages: ChatMessage[]
  setMessages: (fn: (prev: ChatMessage[]) => ChatMessage[]) => void
}) {
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  useEffect(() => () => abortRef.current?.abort(), [])

  const send = async (text: string) => {
    const q = text.trim()
    if (!q || busy) return
    setInput('')
    const history: ChatMessage[] = [...messages, { role: 'user', content: q }]
    setMessages(() => [...history, { role: 'assistant', content: '' }])
    setBusy(true)

    const ctrl = new AbortController()
    abortRef.current = ctrl
    const append = (chunk: string) =>
      setMessages(prev => {
        const next = prev.slice()
        const last = next[next.length - 1]
        if (last?.role === 'assistant') next[next.length - 1] = { ...last, content: last.content + chunk }
        return next
      })

    try {
      const r = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ result, briefing, messages: history }),
        signal: ctrl.signal,
      })
      if (!r.ok || !r.body) {
        let msg = `HTTP ${r.status}`
        try { msg = (await r.json()).error ?? msg } catch { /* ignore */ }
        append(`**Error:** ${msg}`)
        return
      }
      const reader = r.body.getReader()
      const dec = new TextDecoder()
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        append(dec.decode(value, { stream: true }))
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') append(`**Error:** ${e?.message ?? String(e)}`)
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  return (
    <div className="chat">
      <div className="section-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <MessageSquare size={11} /> ask about next steps
      </div>

      <div className="chat-scroll" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="chat-empty">
            <div>Ask anything about this run. Answers use the recovered graph, the flagged reports, and the site contacts.</div>
            <div className="chat-suggest">
              {SUGGESTIONS.map(s => (
                <button key={s} onClick={() => send(s)}>{s}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => {
          const isLast = i === messages.length - 1
          return (
            <div key={i} className={`chat-msg chat-${m.role}`}>
              {m.role === 'user'
                ? <div className="chat-bubble">{m.content}</div>
                : <StreamText text={m.content} streaming={busy && isLast} className="chat-md" />}
            </div>
          )
        })}
      </div>

      {messages.length > 0 && !busy && (
        <div className="chat-suggest chat-suggest-inline">
          {SUGGESTIONS.filter(s => !messages.some(m => m.content === s)).slice(0, 2).map(s => (
            <button key={s} onClick={() => send(s)}>{s}</button>
          ))}
        </div>
      )}

      <form className="chat-input" onSubmit={e => { e.preventDefault(); send(input) }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="e.g. who do I call about FOB_IRONSIDE?"
          disabled={busy}
        />
        {busy ? (
          <button type="button" className="chat-send" onClick={() => abortRef.current?.abort()} title="Stop">
            <Square size={13} />
          </button>
        ) : (
          <button type="submit" className="chat-send" disabled={!input.trim()} title="Send">
            <Send size={13} />
          </button>
        )}
      </form>
    </div>
  )
}
