import { NextRequest, NextResponse } from 'next/server'
import { ORB_ROOT, runPy } from '@/lib/python'

export const dynamic = 'force-dynamic'

/**
 * Run a demo preset. Every call runs the full pipeline (ingest → merge →
 * compile → solve → decode) on the fixture files in place — no temp copies,
 * no result caching. The preset carries its own losses mode.
 */
export async function POST(req: NextRequest) {
  let id = ''
  try {
    const body = await req.json()
    id = String(body?.id ?? '')
  } catch { /* fall through */ }
  if (!/^[a-z0-9-]+$/.test(id)) {
    return NextResponse.json({ error: 'Missing or invalid preset id' }, { status: 400 })
  }
  try {
    const { stdout } = await runPy(
      ['-m', 'orb.demo_json', '--preset', id],
      { cwd: ORB_ROOT, timeout: 240_000, maxBuffer: 10 * 1024 * 1024 },
    )
    const data = JSON.parse(stdout)
    if (data.error) return NextResponse.json(data, { status: 500 })
    return NextResponse.json(data)
  } catch (e: any) {
    const msg = (e.stderr || e.message || String(e)).trim().slice(-3000)
    return NextResponse.json({ error: msg }, { status: 500 })
  }
}
