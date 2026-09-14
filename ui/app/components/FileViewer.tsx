'use client'

import { useEffect, useState } from 'react'
import { FileText, X } from 'lucide-react'

export interface ViewFile {
  name: string
  kind: string        // 'free text' | 'JSON lines' | 'table' | ...
  bytes: number
  lines: number
  truncated?: boolean
  content: string
}

const KIND_NOTE: Record<string, string> = {
  'free text': 'Unstructured messages. No schema, no consistent site names, no fixed phrasing. The readers extract one claim per report.',
  'JSON lines': 'One JSON object per line with arbitrary field names. Routed through the same per-report readers as free text.',
  'table': 'A flat table. One model call maps the columns; every row then becomes a claim with no further model calls.',
}

export function FileViewer({
  title, files, onClose,
}: { title: string; files: ViewFile[]; onClose: () => void }) {
  const [idx, setIdx] = useState(0)
  const f = files[Math.min(idx, files.length - 1)]

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  if (!f) return null

  return (
    <div className="fv-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-label={`Files: ${title}`}>
      <div className="fv-modal" onClick={e => e.stopPropagation()}>
        <div className="fv-head">
          <div className="fv-title"><FileText size={13} /> {title}</div>
          <button className="fv-close" onClick={onClose} aria-label="Close"><X size={14} /></button>
        </div>

        {files.length > 1 && (
          <div className="fv-tabs">
            {files.map((file, i) => (
              <button key={file.name} className={i === idx ? 'active' : ''} onClick={() => setIdx(i)}>
                <span className="fv-tab-name">{file.name}</span>
                <span className={`fv-kind fv-kind-${file.kind.replace(/\s+/g, '-')}`}>{file.kind}</span>
              </button>
            ))}
          </div>
        )}

        <div className="fv-meta">
          <span className={`fv-kind fv-kind-${f.kind.replace(/\s+/g, '-')}`}>{f.kind}</span>
          <span>{f.lines} non-empty lines · {(f.bytes / 1024).toFixed(1)} KB</span>
          {f.truncated && <span className="fv-trunc">truncated preview</span>}
        </div>
        {KIND_NOTE[f.kind] && <div className="fv-note">{KIND_NOTE[f.kind]}</div>}

        <pre className="fv-body">{f.content}</pre>
      </div>
    </div>
  )
}
