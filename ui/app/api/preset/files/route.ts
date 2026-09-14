import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'
import { ORB_ROOT, runPy } from '@/lib/python'

export const dynamic = 'force-dynamic'

const FIXTURES = path.resolve(ORB_ROOT, 'tests', 'fixtures')
const MAX_BYTES = 400_000

let presetList: Array<{ id: string; paths: string[]; files: string[] }> | null = null

function kindOf(name: string): string {
  const ext = name.split('.').pop()?.toLowerCase()
  if (ext === 'txt' || ext === 'log' || ext === 'md') return 'free text'
  if (ext === 'jsonl') return 'JSON lines'
  if (ext === 'csv' || ext === 'tsv') return 'table'
  if (ext === 'json') return 'JSON'
  return ext ?? 'file'
}

/** Raw contents of a preset's fixture files, read in place (read-only, fixtures dir only). */
export async function GET(req: NextRequest) {
  const id = req.nextUrl.searchParams.get('id') ?? ''
  if (!/^[a-z0-9-]+$/.test(id)) {
    return NextResponse.json({ error: 'Missing or invalid preset id' }, { status: 400 })
  }
  try {
    if (!presetList) {
      const { stdout } = await runPy(['-m', 'orb.presets', '--json'], { cwd: ORB_ROOT, timeout: 30_000 })
      presetList = JSON.parse(stdout)
    }
    const preset = presetList!.find(p => p.id === id)
    if (!preset) return NextResponse.json({ error: `Unknown preset ${id}` }, { status: 404 })

    const files = preset.paths.map(rel => {
      const abs = path.resolve(FIXTURES, rel)
      if (!abs.startsWith(FIXTURES + path.sep)) throw new Error(`Refusing path outside fixtures: ${rel}`)
      const stat = fs.statSync(abs)
      const raw = fs.readFileSync(abs, 'utf8')
      const truncated = raw.length > MAX_BYTES
      const content = truncated ? raw.slice(0, MAX_BYTES) : raw
      return {
        name: path.basename(abs),
        kind: kindOf(abs),
        bytes: stat.size,
        lines: raw.split(/\r?\n/).filter(l => l.trim()).length,
        truncated,
        content,
      }
    })
    return NextResponse.json({ id, files })
  } catch (e: any) {
    return NextResponse.json({ error: e.message ?? String(e) }, { status: 500 })
  }
}
