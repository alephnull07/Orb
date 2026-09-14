import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

const RUNS_DIR = path.join(process.cwd(), '..', 'data', 'leakdb', 'runs')

export async function GET() {
  try {
    const manifest = JSON.parse(
      fs.readFileSync(path.join(RUNS_DIR, 'manifest.json'), 'utf8')
    ) as Array<{ run_id: string; corrupted_vs_truth?: { mae_lb?: number; matched_hops?: number; target_edges?: number } }>

    // Only return runs that have generated graphs
    const runs = manifest
      .filter(r => fs.existsSync(path.join(RUNS_DIR, r.run_id, 'graphs', 'corrupted', 'l1_recovered.json')))
      .map(r => ({
        id: r.run_id,
        mae: r.corrupted_vs_truth?.mae_lb ?? null,
        matched: r.corrupted_vs_truth?.matched_hops ?? null,
        total: r.corrupted_vs_truth?.target_edges ?? null,
      }))

    return NextResponse.json(runs)
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 })
  }
}
