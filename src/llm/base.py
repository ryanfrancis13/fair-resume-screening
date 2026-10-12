"""Provider-agnostic LLM interface.

The pipeline talks to models only through :class:`LLMProvider`.  That indirection
buys three things the project actually needs:

1. **Offline reproducibility.** The whole screening → perturbation → audit chain
   runs against :class:`~frs.llm.mock.MockProvider` with no network and no key,
   so the pipeline and its tests are verifiable independently of API access.
2. **Auditor validation.** A mock provider with a *known, injected* bias lets us
   ask whether the fairness suite recovers an effect of known size, and whether
   it stays quiet when there is none. Auditing an auditor requires a ground
   truth that a real API cannot supply.
3. **Provider portability.** Swapping Gemini for another model changes one config
   line, which matters for a study whose conclusions are about a model, not
   about our plumbing.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LLMRequest:
    """One generation request."""

    prompt: str
    system: str = ""
    temperature: float = 0.0
    top_p: float = 0.95
    max_output_tokens: int = 2048
    sample_index: int = 0
    response_mime_type: str = "application/json"
    #: Free-form context carried through to the score record (ids, hashes).
    tags: Dict[str, Any] = field(default_factory=dict)

    def cache_params(self) -> Dict[str, Any]:
        return {
            "system": self.system,
            "temperature": round(self.temperature, 4),
            "top_p": round(self.top_p, 4),
            "max_output_tokens": self.max_output_tokens,
            "sample_index": self.sample_index,
            "response_mime_type": self.response_mime_type,
        }


@dataclass
class LLMResponse:
    """One generation response, successful or not."""

    text: str
    model: str
    provider: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    attempts: int = 1
    finish_reason: str = ""
    error: str = ""
    raw: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text.strip())


class LLMError(RuntimeError):
    """Base class for provider failures."""


class RateLimitError(LLMError):
    """HTTP 429 / quota exhaustion — retryable with backoff."""


class TransientError(LLMError):
    """5xx, timeout, connection reset — retryable."""


class PermanentError(LLMError):
    """Bad key, bad request, content blocked — retrying will not help."""


class LLMProvider(abc.ABC):
    """Minimal contract every provider implements."""

    name: str = "abstract"
    #: Whether calls to this provider consume a remote quota. An offline
    #: provider sets this False so the client's token bucket does not throttle
    #: it — pacing a local function call to 12 requests/minute would turn a
    #: 3000-call offline validation run into a four-hour wait for no reason.
    metered: bool = True

    def __init__(self, model: str) -> None:
        self.model = model

    @property
    def behaviour_key(self) -> str:
        """Fingerprint of any configuration that changes responses.

        Empty for a hosted model, whose behaviour is fully identified by its
        model id. A provider whose output depends on local configuration — the
        offline oracle's injected bias and noise, for instance — must return a
        fingerprint of that configuration, so the response cache cannot serve one
        configuration's answers to another.
        """
        return ""

    @abc.abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Perform a single generation. May raise the error types above."""

    def health_check(self) -> Dict[str, Any]:
        """Cheap readiness probe used by ``frs doctor``."""
        return {"provider": self.name, "model": self.model, "ready": True}

    def close(self) -> None:  # pragma: no cover - providers may hold sockets
        return None
