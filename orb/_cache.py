"""
orb/_cache.py
-------------
Disk cache for LLM calls.  Key = SHA-256 of the full prompt string.
Cache hit → zero new LLM calls.  Miss → real call + persist result.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_DEFAULT_DIR = Path.home() / ".cache" / "orb_ingest"


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
        import sys, os
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
            raise RuntimeError(
                "No ANTHROPIC_API_KEY found.  "
                "Set the env var or pass api_key= to build_graph()."
            )
        result = claude_generate(prompt, model=m, api_key=key)
        self.put(cache_key, result)
        return result
