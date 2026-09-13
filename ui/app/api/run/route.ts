import { NextRequest, NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'
import path from 'path'
import fs from 'fs'

const execAsync = promisify(exec)
const ORB = path.join(process.cwd(), '..')

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData()
    const files = form.getAll('files') as File[]

    if (files.length > 0) {
      // Save uploaded files to user_run/corrupted/
      const runDir = path.join(ORB, 'data', 'supply_drops', 'user_run')
      const corruptedDir = path.join(runDir, 'corrupted')
      fs.mkdirSync(corruptedDir, { recursive: true })

      for (const file of files) {
        const buf = Buffer.from(await file.arrayBuffer())
        fs.writeFileSync(path.join(corruptedDir, file.name), buf)
      }

      // Copy eval dir so bridge.py can score against ground truth
      const defaultEval = path.join(ORB, 'data', 'supply_drops', 'eval')
      const userEval = path.join(runDir, 'eval')
      fs.mkdirSync(userEval, { recursive: true })
      for (const f of fs.readdirSync(defaultEval)) {
        fs.copyFileSync(path.join(defaultEval, f), path.join(userEval, f))
      }

      // ETL: parse raw files → consensus graph
      await execAsync(
        `python3 -m src.etl.run --source corrupted --no-llm --data-root "${runDir}"`,
        { cwd: ORB, timeout: 120_000 }
      )

      // L1: run bridge on the user run dir
      await execAsync(`python3 bridge.py "${runDir}"`, { cwd: ORB, timeout: 60_000 })

      // Tag this run so /api/graphs knows which dir to serve
      fs.writeFileSync(path.join(ORB, 'data', 'supply_drops', '.last_run'), runDir)
    } else {
      // No files — run on default data
      await execAsync('python3 bridge.py', { cwd: ORB, timeout: 60_000 })
      fs.writeFileSync(
        path.join(ORB, 'data', 'supply_drops', '.last_run'),
        path.join(ORB, 'data', 'supply_drops')
      )
    }

    return NextResponse.json({ ok: true })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e.message }, { status: 500 })
  }
}
