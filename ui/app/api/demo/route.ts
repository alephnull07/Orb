import { NextRequest, NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'
import path from 'path'
import fs from 'fs'
import os from 'os'

const execAsync = promisify(exec)
const ORB = path.join(process.cwd(), '..')

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData()
    const files = form.getAll('file') as File[]
    if (!files.length) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 })
    }

    // Save all uploaded files to temp dir preserving names/extensions
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'orb-'))
    const tmpPaths: string[] = []
    for (const file of files) {
      const tmpPath = path.join(tmpDir, file.name)
      fs.writeFileSync(tmpPath, Buffer.from(await file.arrayBuffer()))
      tmpPaths.push(tmpPath)
    }

    try {
      const quoted = tmpPaths.map(p => `"${p}"`).join(' ')
      const { stdout } = await execAsync(
        `python3 -m orb.demo_json ${quoted}`,
        { cwd: ORB, timeout: 60_000, maxBuffer: 10 * 1024 * 1024 }
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
