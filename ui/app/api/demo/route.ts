import { NextRequest, NextResponse } from 'next/server'
import path from 'path'
import fs from 'fs'
import os from 'os'
import { ORB_ROOT, runPy } from '@/lib/python'

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData()
    const files = form.getAll('file') as File[]
    if (!files.length) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 })
    }
    // Sink mode is a modeling decision made in the UI, never detected here.
    const SINK_MODES = ['none', 'known', 'unknown'] as const
    const sinksRaw = String(form.get('sinks') ?? 'none')
    if (!(SINK_MODES as readonly string[]).includes(sinksRaw)) {
      return NextResponse.json({ error: `Invalid sinks mode: ${sinksRaw}` }, { status: 400 })
    }
    const sinks = sinksRaw as (typeof SINK_MODES)[number]

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'orb-'))
    const tmpPaths: string[] = []
    for (const file of files) {
      const tmpPath = path.join(tmpDir, file.name)
      fs.writeFileSync(tmpPath, Buffer.from(await file.arrayBuffer()))
      tmpPaths.push(tmpPath)
    }

    try {
      const { stdout } = await runPy(
        ['-m', 'orb.demo_json', '--sinks', sinks, ...tmpPaths],
        { cwd: ORB_ROOT, timeout: 180_000, maxBuffer: 10 * 1024 * 1024 },
      )
      return NextResponse.json(JSON.parse(stdout))
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true })
    }
  } catch (e: any) {
    const msg = (e.stderr || e.message || String(e)).trim().slice(-3000)
    return NextResponse.json({ error: msg }, { status: 500 })
  }
}
