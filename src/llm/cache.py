"""On-disk response cache.

Why a cache is a *methodological* component and not just a cost saving:

* A perturbation study issues the same prompt many times across pipeline stages
  (score → re-audit → mitigate → report). Caching makes those stages consistent
  with one another by construction; without it, a re-run at ``temperature > 0``
  would quietly change the numbers a report already cited.
* It makes a study *replayable*: ship the cache and a third party can reproduce
  every score in the paper without an API key or a budget.

Keys cover the prompt, model, provider and all decoding parameters including the
sample index, so self-consistency repeats never collapse onto one entry.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

from ..utils.hashing import cache_key
from ..utils.io import ensure_dir
from .base import LLMRequest, LLMResponse


class ResponseCache:
    """Content-addressed cache, sharded two levels deep to keep dirs small."""

    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self.enabled:
            ensure_dir(self.root)

    # ------------------------------------------------------------------ #
    def key_for(
        self, request: LLMRequest, provider: str, model: str, behaviour: str = ""
    ) -> str:
        return cache_key(
            prompt=request.prompt,
            model=model,
            provider=provider,
            params=request.cache_params(),
            behaviour=behaviour,
        )

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / key[2:4] / f"{key}.json"

    # ------------------------------------------------------------------ #
    def get(self, key: str) -> Optional[LLMResponse]:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            with self._lock:
                self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A truncated cache entry is a miss, never a crash: a killed run
            # must not poison every future run.
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        return LLMResponse(
            text=payload.get("text", ""),
            model=payload.get("model", ""),
            provider=payload.get("provider", ""),
            latency_ms=float(payload.get("latency_ms", 0.0)),
            prompt_tokens=int(payload.get("prompt_tokens", 0)),
            completion_tokens=int(payload.get("completion_tokens", 0)),
            cached=True,
            attempts=int(payload.get("attempts", 1)),
            finish_reason=payload.get("finish_reason", ""),
        )

    def put(self, key: str, response: LLMResponse) -> None:
        # Errors are not cached: a 429 must not be replayed forever as a result.
        if not self.enabled or not response.ok:
            return
        path = self._path(key)
        ensure_dir(path.parent)
        payload = {
            "text": response.text,
            "model": response.model,
            "provider": response.provider,
            "latency_ms": response.latency_ms,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "attempts": response.attempts,
            "finish_reason": response.finish_reason,
        }
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
        with self._lock:
            self.writes += 1

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": (self.hits / total) if total else 0.0,
            "root": str(self.root),
        }
