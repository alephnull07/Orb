import { NextRequest, NextResponse } from 'next/server'
import path from 'path'
import fs from 'fs'
import { ORB_ROOT, runPy } from '@/lib/python'

const ORB = ORB_ROOT

/** Return true if the buffer looks like a valid consensus/flow graph JSON */
function isGraphJson(buf: Buffer): boolean {
  try {
    const obj = JSON.parse(buf.toString('utf8'))
    return Array.isArray(obj.nodes) && Array.isArray(obj.edges) && obj.edges.length > 0
  } catch {
    return false
  }
}

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData()
    const files = form.getAll('files') as File[]

    if (files.length === 0) {
      // No files — run L1 on default supply_drops data
      await runPy(['bridge.py'], { cwd: ORB, timeout: 60_000 })
      fs.writeFileSync(
        path.join(ORB, 'data', 'supply_drops', '.last_run'),
        path.join(ORB, 'data', 'supply_drops')
      )
      return NextResponse.json({ ok: true })
    }

    const runDir = path.join(ORB, 'data', 'supply_drops', 'user_run')
    fs.mkdirSync(path.join(runDir, 'corrupted'), { recursive: true })
    fs.mkdirSync(path.join(runDir, 'graphs', 'corrupted'), { recursive: true })

    // Read all file buffers
    const fileMap: Record<string, Buffer> = {}
    for (const file of files) {
      fileMap[file.name] = Buffer.from(await file.arrayBuffer())
    }

    // ── Path A: direct graph JSON upload ──────────────────────────────────
    // If any uploaded JSON file is already a valid graph (has nodes + edges),
    // write it directly as the consensus graph — no ETL needed.
    // Works with any domain: supply chain, SCADA, logistics, etc.
    const graphFile = Object.entries(fileMap).find(
      ([name, buf]) => name.endsWith('.json') && isGraphJson(buf)
    )

    if (graphFile) {
      const [fname, buf] = graphFile
      const dest = path.join(runDir, 'graphs', 'corrupted', 'graph.json')
      fs.writeFileSync(dest, buf)
      // Also write as consensus.json (what the UI reads)
      fs.writeFileSync(path.join(runDir, 'graphs', 'corrupted', 'consensus.json'), buf)
      console.log(`[run] Direct graph upload: ${fname} → skipping ETL`)

    // ── Path B: raw comms files → ETL → graph ─────────────────────────────
    } else {
      const hasMessages = 'messages.jsonl' in fileMap
      if (!hasMessages) {
        return NextResponse.json({
          ok: false,
          error:
            'Upload either:\n' +
            '  • A graph JSON (nodes + edges already extracted) — runs L1 directly\n' +
            '  • messages.jsonl (supply-chain comms) — runs full ETL → L1 pipeline\n\n' +
            'Raw .txt / .csv files are not parsed directly. ' +
            'They need to be converted to messages.jsonl format first.',
        }, { status: 400 })
      }

      // Save all files to the corrupted dir
      for (const [name, buf] of Object.entries(fileMap)) {
        fs.writeFileSync(path.join(runDir, 'corrupted', name), buf)
      }

      // Copy eval dir (best-effort, bridge.py handles missing eval gracefully)
      try {
        const defaultEval = path.join(ORB, 'data', 'supply_drops', 'eval')
        const userEval = path.join(runDir, 'eval')
        fs.mkdirSync(userEval, { recursive: true })
        for (const f of fs.readdirSync(defaultEval)) {
          fs.copyFileSync(path.join(defaultEval, f), path.join(userEval, f))
        }
      } catch { /* optional */ }

      // Detect API key → use LLM mode if available, else regex-only
      const apiKey = process.env.ANTHROPIC_API_KEY || ''
      const llmFlag = apiKey ? '' : '--no-llm'
      await runPy(
        ['-m', 'src.etl.run', '--source', 'corrupted', ...(llmFlag ? [llmFlag] : []), '--data-root', runDir],
        { cwd: ORB, timeout: 120_000 },
      )
    }

    // Run L1 on whatever graph we now have
    await runPy(['bridge.py', runDir], { cwd: ORB, timeout: 60_000 })

    // Tag this run so /api/graphs knows which dir to serve
    fs.writeFileSync(path.join(ORB, 'data', 'supply_drops', '.last_run'), runDir)

    return NextResponse.json({ ok: true })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e.message }, { status: 500 })
  }
}
