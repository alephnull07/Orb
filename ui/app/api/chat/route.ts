import { NextRequest } from 'next/server'
import type Anthropic from '@anthropic-ai/sdk'
import { MODEL, buildSystem, errorResponse, getClient, streamToResponse } from '../../../lib/anthropic'
import type { DemoResult } from '../../../lib/orb'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const CHAT_STYLE = `You are now answering follow-up questions in a chat. Answer from the estimation context only. Be brief: usually 2 to 6 sentences or a short list. When asked what to do next, give concrete steps tied to specific nodes, and name the contact and phone number to call. If a question can't be answered from the context, say what is missing and what data would resolve it. Use **bold** for node names and phone numbers. No headings, no tables.`

interface ChatBody {
  result: DemoResult
  messages: Array<{ role: 'user' | 'assistant'; content: string }>
  briefing?: string
}

export async function POST(req: NextRequest) {
  let body: ChatBody
  try {
    body = await req.json()
    if (!body?.result?.graph?.nodes) return errorResponse(new Error('Missing estimation result'), 400)
    if (!Array.isArray(body.messages) || body.messages.length === 0)
      return errorResponse(new Error('No messages'), 400)
  } catch (e) {
    return errorResponse(e, 400)
  }

  // Sanitize history: alternate roles, drop empty assistant turns, must start with user.
  const history: Anthropic.MessageParam[] = []
  for (const m of body.messages) {
    const text = (m.content ?? '').trim()
    if (!text) continue
    if (m.role !== 'user' && m.role !== 'assistant') continue
    if (history.length === 0 && m.role !== 'user') continue
    const last = history[history.length - 1]
    if (last && last.role === m.role) {
      last.content = `${last.content}\n\n${text}`
    } else {
      history.push({ role: m.role, content: text })
    }
  }
  if (history.length === 0 || history[history.length - 1].role !== 'user')
    return errorResponse(new Error('Conversation must end with a user message'), 400)

  const system = buildSystem(body.result)
  system.push({ type: 'text', text: CHAT_STYLE })
  if (body.briefing?.trim()) {
    system.push({
      type: 'text',
      text: `The briefing already shown to the user (do not repeat it verbatim, build on it):\n\n${body.briefing.trim()}`,
    })
  }

  try {
    const client = getClient()
    const stream = client.messages.stream({
      model: MODEL,
      max_tokens: 2000,
      system,
      messages: history,
    })
    return streamToResponse(stream)
  } catch (e) {
    return errorResponse(e)
  }
}
