/**
 * Server-only Anthropic client + shared prompt pieces for the insights
 * and chat routes.
 */
import fs from 'fs'
import path from 'path'
import Anthropic from '@anthropic-ai/sdk'
import { buildContextText, type DemoResult } from './orb'

export const MODEL = 'claude-haiku-4-5'

/** ANTHROPIC_API_KEY from the environment, else the repo-root .env file. */
export function getApiKey(): string | undefined {
  if (process.env.ANTHROPIC_API_KEY) return process.env.ANTHROPIC_API_KEY.trim()
  const candidates = [
    path.join(process.cwd(), '.env'),
    path.join(process.cwd(), '..', '.env'),
  ]
  for (const p of candidates) {
    try {
      for (const line of fs.readFileSync(p, 'utf8').split('\n')) {
        const m = line.trim().match(/^ANTHROPIC_API_KEY=(.*)$/)
        if (m && m[1].trim()) return m[1].trim().replace(/^["']|["']$/g, '')
      }
    } catch { /* not there */ }
  }
  return undefined
}

let _client: Anthropic | null = null
export function getClient(): Anthropic {
  if (!_client) _client = new Anthropic({ apiKey: getApiKey() })
  return _client
}

export const ANALYST_ROLE = `You are ORB's operations analyst. ORB recovers the true state of a resource-distribution network from unstructured field reports. It treats every report as a sensor reading, enforces conservation (what leaves one node arrives at another), and solves a weighted L1 estimation so that a few badly corrupted reports get isolated instead of averaged in. When the data can't distinguish two possibilities, it says so rather than guessing.

You are briefing the on-duty operations lead. They know the network but did not watch the estimation run. Be concrete: name nodes, the reports that were rejected, and the people to call (contacts and phone numbers are in the context — they are mock data for this demo, use them as given). Ground every statement in the context below; never invent nodes, numbers, or contacts that are not there. If correctable_k is 0, say clearly that corrections are not guaranteed. Prefer plain, direct language over jargon.

ORB detects inconsistent and corrupted reports. It is not an inventory tracker. Do not treat negative or low estimated quantity as a site being down, empty, or at risk. Health is only about rejected reports, corrected claims, and unaccounted loss (sinks).`

/** System prompt with the run context in a cacheable block. */
export function buildSystem(result: DemoResult): Anthropic.TextBlockParam[] {
  return [
    { type: 'text', text: ANALYST_ROLE },
    {
      type: 'text',
      text: `Here is the full context of the estimation run:\n\n${buildContextText(result)}`,
      cache_control: { type: 'ephemeral' },
    },
  ]
}

/** Turn an SDK message stream into a plain-text HTTP streaming Response. */
export function streamToResponse(
  stream: ReturnType<Anthropic['messages']['stream']>,
): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const event of stream) {
          if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
            controller.enqueue(encoder.encode(event.delta.text))
          }
        }
        const final = await stream.finalMessage()
        if (final.stop_reason === 'refusal') {
          controller.enqueue(encoder.encode(
            '\n\n_The model declined to answer this request._',
          ))
        } else if (final.stop_reason === 'max_tokens') {
          controller.enqueue(encoder.encode('\n\n_[response truncated]_'))
        }
      } catch (e: any) {
        controller.enqueue(encoder.encode(`\n\n**Error:** ${e?.message ?? String(e)}`))
      } finally {
        controller.close()
      }
    },
    cancel() { stream.abort() },
  })
  return new Response(body, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      'X-Accel-Buffering': 'no',
    },
  })
}

export function errorResponse(e: any, status = 500): Response {
  const msg = e?.message ?? String(e)
  return new Response(JSON.stringify({ error: msg }), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}
