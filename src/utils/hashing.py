"""Content hashing for provenance and cache keys.

Two distinct needs are served here:

* **Cache keys** must capture everything that could change a model response
  (prompt, model id, decoding parameters, response schema) and nothing else.
* **Provenance hashes** let a report assert "this number came from exactly this
  resume text and this job description", which is what makes an audit
  reproducible by a third party who only has the artifacts.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

_HASH_LEN = 16


def stable_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_text(text: str, length: int = _HASH_LEN) -> str:
    """Short BLAKE2b hex digest of a string."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=32).hexdigest()[:length]


def hash_obj(obj: Any, length: int = _HASH_LEN) -> str:
    """Short BLAKE2b hex digest of any JSON-serialisable object."""
    return hash_text(stable_json(obj), length=length)


def cache_key(
    *,
    prompt: str,
    model: str,
    provider: str,
    params: Mapping[str, Any],
    behaviour: str = "",
    schema_version: str = "v1",
) -> str:
    """Cache key for one LLM call.

    Every input that can change the response must appear here, or the cache
    silently serves the wrong answer:

    * ``params`` carries temperature, top_p, max tokens and the sample index, so
      self-consistency repeats are cached separately instead of collapsing onto
      one another.
    * ``behaviour`` carries any provider configuration that changes the mapping
      from prompt to response without changing the model id. The offline bias
      oracle depends on this: its null and injected conditions issue byte-identical
      prompts to the same model name and differ only in the injected bias, so
      without ``behaviour`` the second condition is served the first condition's
      cached responses and the measured effect is exactly zero.
    """
    return hash_obj(
        {
            "prompt": prompt,
            "model": model,
            "provider": provider,
            "params": dict(params),
            "behaviour": behaviour,
            "schema": schema_version,
        },
        length=32,
    )
