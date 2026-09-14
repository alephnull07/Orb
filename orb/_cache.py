"""
orb/_cache.py
-------------
Disk cache for LLM calls.  Key = SHA-256 of the full prompt string.
Cache hit → zero new LLM calls.  Miss → real call + persist result.

Also loads .env from the repo root at import time so that every entry
point (CLI, pytest, Next.js API route) sees ANTHROPIC_API_KEY without
requiring an explicit export.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_DEFAULT_DIR = Path.home() / ".cache" / "orb_ingest"

# ---------------------------------------------------------------------------
# .env loader (no third-party dependency)
# ---------------------------------------------------------------------------

_DOTENV_PATH: str | None = None   # set by _load_dotenv, used in error messages


def _find_dotenv() -> Path | None:
    """Walk up from this file to find a .env file."""
    cur = Path(__file__).resolve().parent   # orb/
    for _ in range(5):
        candidate = cur / ".env"
        if candidate.exists():
            return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _load_dotenv() -> None:
    """Parse .env and inject keys into os.environ (existing vars always win)."""
    global _DOTENV_PATH
    env_file = _find_dotenv()
    _DOTENV_PATH = str(env_file) if env_file else None

    if env_file is None:
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        # Only set if the key has a value AND isn't already in the environment
        if key and val and key not in os.environ:
            os.environ[key] = val


_load_dotenv()          # runs once at first import of this module


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class Cache:
    def __init__(self, cache_dir: Path | None = None):
        self.dir = Path(cache_dir or _DEFAULT_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.llm_calls = 0          # increments only on actual API call (cache miss)

    # ── low-level storage ───────────────────────────────────────────────────

    @staticmethod
    def _key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _path(self, prompt: str) -> Path:
        return self.dir / f"{self._key(prompt)}.txt"

    def get(self, prompt: str) -> str | None:
        p = self._path(prompt)
        return p.read_text("utf-8") if p.exists() else None

    def put(self, prompt: str, value: str) -> None:
        self._path(prompt).write_text(value, "utf-8")

    # ── public API ───────────────────────────────────────────────────────────

    def call(
        self,
        prompt: str,
        api_key: str | None = None,
        model: str | None = None,
    ) -> str:
        """
        Call claude_generate with transparent disk caching.
        Cache hit  → return stored text, llm_calls unchanged.
        Cache miss → make real API call, persist result, increment llm_calls.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.etl.llm import claude_generate, get_api_key, DEFAULT_CLAUDE_MODEL

        m = model or DEFAULT_CLAUDE_MODEL
        cache_key = f"model={m}\n{prompt}"

        cached = self.get(cache_key)
        if cached is not None:
            return cached

        self.llm_calls += 1
        key = get_api_key(api_key)
        if not key:
            checked = ["  - env var ANTHROPIC_API_KEY (not set)"]
            if _DOTENV_PATH:
                checked.append(f"  - {_DOTENV_PATH} (no ANTHROPIC_API_KEY with a value)")
            else:
                checked.append("  - .env file (not found in any parent of orb/)")
            raise RuntimeError(
                "No ANTHROPIC_API_KEY found.  Checked:\n"
                + "\n".join(checked)
                + "\nSet the env var, add it to .env at the repo root, "
                "or pass api_key= to build_graph()."
            )
        result = claude_generate(prompt, model=m, api_key=key)
        self.put(cache_key, result)
        return result
