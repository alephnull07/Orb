import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

const ORB = path.join(process.cwd(), '..')
const LEAKDB_RUNS = path.join(ORB, 'data', 'leakdb', 'runs')
const SUPPLY_DEFAULT = path.join(ORB, 'data', 'supply_drops')

function getRunDir(searchParams: URLSearchParams): string {
  const run = searchParams.get('run')  // e.g. "run_01"
  if (run) {
    const dir = path.join(LEAKDB_RUNS, run)
    if (fs.existsSync(dir)) return dir
  }
  const tag = path.join(SUPPLY_DEFAULT, '.last_run')
  if (fs.existsSync(tag)) {
    const dir = fs.readFileSync(tag, 'utf8').trim()
    if (fs.existsSync(dir)) return dir
  }
  return SUPPLY_DEFAULT
}

export async function GET(req: NextRequest) {
  try {
    const base = getRunDir(req.nextUrl.searchParams)
    const read = (p: string) => JSON.parse(fs.readFileSync(path.join(base, p), 'utf8'))
    const tryRead = (p: string) => { try { return read(p) } catch { return null } }

    return NextResponse.json({
      corrupted: read('graphs/corrupted/consensus.json'),
      l1:        read('graphs/corrupted/l1_recovered.json'),
      truth:     tryRead('eval/true_graph.json'),
    })
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 })
  }
}
