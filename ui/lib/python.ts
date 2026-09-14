import { execFile } from 'child_process'
import { promisify } from 'util'
import path from 'path'
import fs from 'fs'
import os from 'os'

const execFileAsync = promisify(execFile)

export const ORB_ROOT = path.join(process.cwd(), '..')

let cached: string | null = null

function isStoreStub(exe: string) {
  return /WindowsApps/i.test(exe)
}

async function probe(bin: string, prefix: string[] = []): Promise<string | null> {
  try {
    const { stdout } = await execFileAsync(
      bin,
      [...prefix, '-c', 'import sys; print(sys.executable)'],
      { timeout: 8000, windowsHide: true },
    )
    const exe = stdout.trim().split(/\r?\n/).filter(Boolean).at(-1) ?? ''
    if (!exe || isStoreStub(exe) || !fs.existsSync(exe)) return null
    return exe
  } catch {
    return null
  }
}

async function hasNumpy(exe: string): Promise<boolean> {
  try {
    await execFileAsync(exe, ['-c', 'import numpy'], { timeout: 8000, windowsHide: true })
    return true
  } catch {
    return false
  }
}

function knownWindowsPythons(): string[] {
  const home = os.homedir()
  return [
    'C:\\Python\\miniconda3\\python.exe',
    'C:\\Python\\Python39\\python.exe',
    path.join(home, 'miniconda3', 'python.exe'),
    path.join(home, 'anaconda3', 'python.exe'),
    path.join(home, 'AppData', 'Local', 'Programs', 'Python', 'Python39', 'python.exe'),
    path.join(home, 'AppData', 'Local', 'Programs', 'Python', 'Python311', 'python.exe'),
    path.join(home, 'AppData', 'Local', 'Programs', 'Python', 'Python312', 'python.exe'),
  ].filter(p => fs.existsSync(p))
}

/** Real interpreter — never the Windows Store `python3` stub. */
export async function resolvePython(): Promise<string> {
  if (cached) return cached

  const venv = process.platform === 'win32'
    ? path.join(ORB_ROOT, '.venv', 'Scripts', 'python.exe')
    : path.join(ORB_ROOT, '.venv', 'bin', 'python')

  const candidates: Array<{ bin: string; prefix?: string[] }> = []
  if (process.env.PYTHON) candidates.push({ bin: process.env.PYTHON })
  if (fs.existsSync(venv)) candidates.push({ bin: venv })
  for (const p of knownWindowsPythons()) candidates.push({ bin: p })
  if (process.platform === 'win32') {
    candidates.push({ bin: 'py', prefix: ['-3'] }, { bin: 'python' }, { bin: 'python3' })
  } else {
    candidates.push({ bin: 'python3' }, { bin: 'python' })
  }

  const found: string[] = []
  for (const { bin, prefix } of candidates) {
    const exe = await probe(bin, prefix)
    if (exe && !found.includes(exe)) found.push(exe)
  }
  if (found.length === 0) {
    throw new Error(
      'No working Python interpreter found. Install Python or set PYTHON to your python.exe.',
    )
  }

  for (const exe of found) {
    if (await hasNumpy(exe)) {
      cached = exe
      return exe
    }
  }
  cached = found[0]
  return cached
}

export async function runPy(
  args: string[],
  opts: { cwd: string; timeout?: number; maxBuffer?: number },
) {
  const bin = await resolvePython()
  try {
    return await execFileAsync(bin, args, {
      cwd: opts.cwd,
      timeout: opts.timeout ?? 60_000,
      maxBuffer: opts.maxBuffer,
      windowsHide: true,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
    })
  } catch (e: any) {
    if (e.killed) {
      const sec = Math.round((opts.timeout ?? 60_000) / 1000)
      throw new Error(
        `Pipeline timed out after ${sec}s. ${((e.stderr || e.stdout || '') as string).trim().slice(-1500)}`.trim(),
      )
    }
    const detail = (e.stderr || e.stdout || '').trim().slice(-3000)
    throw new Error(detail || e.message)
  }
}
