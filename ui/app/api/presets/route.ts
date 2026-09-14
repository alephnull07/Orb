import { NextResponse } from 'next/server'
import { ORB_ROOT, runPy } from '@/lib/python'

export const dynamic = 'force-dynamic'

// The preset list lives in orb/presets.py (single source of truth). It is
// static for the life of the server process, so one spawn is enough.
let cachedList: unknown[] | null = null

export async function GET() {
  try {
    if (!cachedList) {
      const { stdout } = await runPy(['-m', 'orb.presets', '--json'], { cwd: ORB_ROOT, timeout: 30_000 })
      cachedList = JSON.parse(stdout)
    }
    return NextResponse.json(cachedList)
  } catch (e: any) {
    return NextResponse.json({ error: e.message ?? String(e) }, { status: 500 })
  }
}
