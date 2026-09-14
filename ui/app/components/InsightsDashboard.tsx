'use client'

import { useMemo } from 'react'
import { RefreshCw, Sparkles } from 'lucide-react'
import { deriveStatus, type DemoResult } from '../../lib/orb'
import { ChatPanel, type ChatMessage } from './ChatPanel'
import { StreamText } from './StreamText'
import { SystemOverview } from './SystemOverview'

export interface InsightsState {
  text: string
  streaming: boolean
  error: string | null
}

export function InsightsDashboard({
  result, insights, onRegenerate, chat, setChat,
}: {
  result: DemoResult
  insights: InsightsState
  onRegenerate: () => void
  chat: ChatMessage[]
  setChat: (fn: (prev: ChatMessage[]) => ChatMessage[]) => void
}) {
  const status = useMemo(() => deriveStatus(result), [result])
  const waiting = insights.streaming && insights.text.length === 0

  return (
    <section className="ins">
      <aside className="ins-side">
        <SystemOverview status={status} />
      </aside>

      <section className="ins-main">
        <div className="ins-toolbar">
          <div className="ins-title">
            <Sparkles size={13} />
            operations briefing
            {insights.streaming && <span className="ins-live">live</span>}
          </div>
          <button className="ins-regen" onClick={onRegenerate} disabled={insights.streaming} title="Regenerate">
            <RefreshCw size={12} className={insights.streaming ? 'spin' : ''} /> regenerate
          </button>
        </div>

        <div className="ins-body">
          {insights.error && (
            <pre className="ins-error">{insights.error}</pre>
          )}
          {waiting && !insights.error && (
            <div className="ins-thinking">
              <span className="ins-shimmer" style={{ width: '62%' }} />
              <span className="ins-shimmer" style={{ width: '88%' }} />
              <span className="ins-shimmer" style={{ width: '74%' }} />
              <div className="ins-thinking-label">reading the recovered graph…</div>
            </div>
          )}
          {insights.text && (
            <StreamText text={insights.text} streaming={insights.streaming} className="ins-md" />
          )}
        </div>
      </section>

      <aside className="ins-chat">
        <ChatPanel result={result} briefing={insights.text} messages={chat} setMessages={setChat} />
      </aside>
    </section>
  )
}
