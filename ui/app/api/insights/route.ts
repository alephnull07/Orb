import { NextRequest } from 'next/server'
import { MODEL, buildSystem, errorResponse, getClient, streamToResponse } from '../../../lib/anthropic'
import type { DemoResult } from '../../../lib/orb'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BRIEFING_REQUEST = `Write the post-estimation briefing for the operations lead. Use exactly these markdown sections, in this order, each starting with "## ":

## Situation
Two or three sentences: what network this is, what the reports said, and the single most important finding.

## What happened
What ORB found when it enforced conservation: which reports were rejected and why, which quantities were corrected and by how much, and any unaccounted losses. Bullet points.

## Systems at risk
One bullet per node that is DOWN or DEGRADED, most severe first. Say what is wrong in operational terms and what it means for the site. If everything is healthy, say so in one line.

## Next steps
A numbered list of 3 to 6 concrete actions in priority order. Each action names who to contact (name, role, phone from the context) and what to verify or do. Include the operations desk or escalation contact where a decision above site level is needed.

## Confidence
One or two sentences on how much to trust this picture, based on identifiability (rank, correctable_k) and how many reports were flagged.

Keep the whole briefing under 350 words. Use **bold** for node names and phone numbers. No tables, no code blocks.`

export async function POST(req: NextRequest) {
  let result: DemoResult
  try {
    const body = await req.json()
    result = body.result
    if (!result?.graph?.nodes) return errorResponse(new Error('Missing estimation result'), 400)
  } catch (e) {
    return errorResponse(e, 400)
  }

  try {
    const client = getClient()
    const stream = client.messages.stream({
      model: MODEL,
      max_tokens: 4000,
      system: buildSystem(result),
      messages: [{ role: 'user', content: BRIEFING_REQUEST }],
    })
    return streamToResponse(stream)
  } catch (e) {
    return errorResponse(e)
  }
}
