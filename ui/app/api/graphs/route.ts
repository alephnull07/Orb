import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

const DEFAULT = path.join(process.cwd(), '..', 'data', 'supply_drops')

function getRunDir(): string {
  const tag = path.join(DEFAULT, '.last_run')
  if (fs.existsSync(tag)) {
    const dir = fs.readFileSync(tag, 'utf8').trim()
    if (fs.existsSync(dir)) return dir
  }
  return DEFAULT
}

export async function GET() {
  try {
    const base = getRunDir()
    const read = (p: string) => JSON.parse(fs.readFileSync(path.join(base, p), 'utf8'))
    return NextResponse.json({
      corrupted: read('graphs/corrupted/consensus.json'),
      l1:        read('graphs/corrupted/l1_recovered.json'),
      truth:     read('eval/true_graph.json'),
    })
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 })
  }
}
