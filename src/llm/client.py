"""The single entry point the rest of the pipeline uses to call a model.

:class:`LLMClient` composes the cross-cutting concerns so that no caller has to
re-implement them: cache lookup, rate limiting, retry with backoff, telemetry,
and a uniform "never raise mid-batch" error contract.  A 700-call perturbation
run must not die on call 683 because of one 503, and — just as important — the
row for that call must still exist in the artifact, marked as failed, so the
analysis can report attrition instead of silently analysing a biased subset.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import Config, load_dotenv_if_present
from ..logging_setup import log_event
from .base import LLMProvider, LLMRequest, LLMResponse, PermanentError
from .cache import ResponseCache
from .ratelimit import RetryPolicy, TokenBucket

logger = logging.getLogger("frs.llm.client")


def _mock(cfg: Config) -> LLMProvider:
    from .mock import MockProvider

    return MockProvider(
        model=cfg.llm.model if cfg.llm.model.startswith("mock") else "mock-rubric-v1",
        bias=cfg.llm.mock_bias,
        noise_sd=cfg.llm.mock_noise_sd,
    )


def _gemini(cfg: Config) -> LLMProvider:
    from .gemini import GeminiProvider

    return GeminiProvider(model=cfg.llm.model, timeout_s=cfg.llm.timeout_s)


def _openai(cfg: Config) -> LLMProvider:
    from .openai_provider import OpenAIProvider

    return OpenAIProvider(model=cfg.llm.model, timeout_s=cfg.llm.timeout_s)


#: provider name -> factory. Provider modules are imported inside the factory so
#: an SDK is only needed when its provider is actually selected. To add a
#: provider, implement ``LLMProvider`` in ``frs/llm/<name>.py`` and register it.
PROVIDERS: Dict[str, Callable[[Config], LLMProvider]] = {
    "mock": _mock,
    "gemini": _gemini,
    "openai": _openai,
}


def build_provider(cfg: Config) -> LLMProvider:
    """Instantiate the configured provider."""
    load_dotenv_if_present()
    try:
        factory = PROVIDERS[cfg.llm.provider.lower()]
    except KeyError:
        raise ValueError(
            f"unknown provider {cfg.llm.provider!r}; available: {', '.join(sorted(PROVIDERS))}"
        ) from None
    return factory(cfg)


class LLMClient:
    """Cache + rate limit + retry wrapper around a provider."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        cache: Optional[ResponseCache] = None,
        requests_per_minute: int = 12,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 32.0,
        max_concurrency: int = 4,
        fail_fast: bool = False,
    ) -> None:
        self.provider = provider
        self.cache = cache or ResponseCache(Path(".frs_cache"), enabled=False)
        self.bucket = TokenBucket(requests_per_minute)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.max_concurrency = max(1, max_concurrency)
        self.fail_fast = fail_fast
        self._lock = threading.Lock()
        self.stats: Dict[str, Any] = {
            "calls": 0, "cache_hits": 0, "api_calls": 0, "retries": 0,
            "failures": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "rate_limit_wait_s": 0.0,
        }

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: Config, provider: Optional[LLMProvider] = None) -> LLMClient:
        cache = ResponseCache(cfg.resolve(cfg.llm.cache_dir), enabled=cfg.llm.cache_enabled)
        return cls(
            provider or build_provider(cfg),
            cache=cache,
            requests_per_minute=cfg.llm.requests_per_minute,
            max_retries=cfg.llm.max_retries,
            retry_base_delay=cfg.llm.retry_base_delay,
            retry_max_delay=cfg.llm.retry_max_delay,
            max_concurrency=cfg.llm.max_concurrency,
            fail_fast=cfg.llm.fail_fast,
        )

    # ------------------------------------------------------------------ #
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Return a response, or a response object carrying ``error``.

        Never raises for a recoverable failure — batch callers rely on getting a
        row back for every request they submitted.
        """
        key = self.cache.key_for(
            request,
            self.provider.name,
            self.provider.model,
            getattr(self.provider, "behaviour_key", ""),
        )
        cached = self.cache.get(key)
        if cached is not None:
            with self._lock:
                self.stats["calls"] += 1
                self.stats["cache_hits"] += 1
            return cached

        policy = RetryPolicy(
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
        )

        def attempt() -> LLMResponse:
            if getattr(self.provider, "metered", True):
                waited = self.bucket.acquire()
                if waited:
                    with self._lock:
                        self.stats["rate_limit_wait_s"] += waited
            return self.provider.generate(request)

        def on_retry(attempt_no: int, delay: float, exc: Exception) -> None:
            log_event(
                logger, logging.WARNING, "retrying LLM call",
                attempt=attempt_no, delay_s=round(delay, 2),
                error=f"{type(exc).__name__}: {exc}"[:200],
                tags=request.tags,
            )

        try:
            response = policy.run(attempt, on_retry=on_retry)
        except Exception as exc:  # noqa: BLE001 - uniform contract
            with self._lock:
                self.stats["calls"] += 1
                self.stats["api_calls"] += 1
                self.stats["failures"] += 1
                self.stats["retries"] += policy.retries_used
            log_event(
                logger, logging.ERROR, "LLM call failed",
                error=f"{type(exc).__name__}: {exc}"[:300], tags=request.tags,
            )
            if self.fail_fast or isinstance(exc, PermanentError):
                raise
            return LLMResponse(
                text="", model=self.provider.model, provider=self.provider.name,
                error=f"{type(exc).__name__}: {exc}"[:300],
                attempts=policy.retries_used + 1,
            )

        response.attempts = policy.retries_used + 1
        self.cache.put(key, response)
        with self._lock:
            self.stats["calls"] += 1
            self.stats["api_calls"] += 1
            self.stats["retries"] += policy.retries_used
            self.stats["prompt_tokens"] += response.prompt_tokens
            self.stats["completion_tokens"] += response.completion_tokens
        return response

    # ------------------------------------------------------------------ #
    def generate_many(
        self,
        requests: Sequence[LLMRequest],
        *,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[LLMResponse]:
        """Run requests concurrently, preserving input order.

        Concurrency is bounded and the token bucket is shared, so the effective
        request rate stays inside the provider's quota regardless of how many
        workers are running.
        """
        if not requests:
            return []
        if self.max_concurrency == 1:
            out = []
            for i, req in enumerate(requests, start=1):
                out.append(self.generate(req))
                if progress:
                    progress(i, len(requests))
            return out

        results: List[Optional[LLMResponse]] = [None] * len(requests)
        done = 0
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = {pool.submit(self.generate, req): i for i, req in enumerate(requests)}
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                done += 1
                if progress:
                    progress(done, len(requests))
        # A None here would mean a worker returned nothing, which the uniform
        # error contract forbids; assert rather than silently shortening the batch.
        assert all(r is not None for r in results), "generate_many lost a response"
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------ #
    def telemetry(self) -> Dict[str, Any]:
        merged = dict(self.stats)
        merged["cache"] = self.cache.stats()
        merged["provider"] = self.provider.name
        merged["model"] = self.provider.model
        merged["rate_limit_wait_s"] = round(merged["rate_limit_wait_s"], 2)
        return merged

    def close(self) -> None:
        self.provider.close()
