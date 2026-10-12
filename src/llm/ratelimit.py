"""Client-side rate limiting and retry policy.

The Gemini free tier is generous but hard-capped per minute, and a perturbation
study is exactly the workload that hits that cap: hundreds of near-identical
calls issued as fast as the loop can manage.  Two mechanisms cooperate here:

* a **token bucket** that paces requests *before* they are sent, which is
  cheaper and politer than discovering the limit through 429s; and
* **exponential backoff with full jitter** for the 429s that still happen, so
  that several worker threads recovering at once do not resynchronise into a
  thundering herd.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Callable, Optional, TypeVar

from .base import LLMError, PermanentError, RateLimitError, TransientError

T = TypeVar("T")


class TokenBucket:
    """Thread-safe token bucket allowing ``rate_per_minute`` requests/minute."""

    def __init__(self, rate_per_minute: int, burst: Optional[int] = None) -> None:
        self.rate_per_second = max(rate_per_minute, 1) / 60.0
        self.capacity = float(burst if burst is not None else max(1, rate_per_minute // 2))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()
        self.total_wait_s = 0.0

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available; return seconds slept."""
        slept = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._updated) * self.rate_per_second
                )
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self.total_wait_s += slept
                    return slept
                deficit = tokens - self._tokens
                wait = deficit / self.rate_per_second
            wait = min(wait, 5.0)
            time.sleep(wait)
            slept += wait


class RetryPolicy:
    """Exponential backoff with full jitter."""

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 32.0,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleeper
        self._jitter = jitter
        self.retries_used = 0

    def delay_for(self, attempt: int) -> float:
        """Full-jitter delay for a 1-indexed attempt number."""
        ceiling = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return self._jitter() * ceiling

    def run(self, fn: Callable[[], T], on_retry: Optional[Callable[[int, float, Exception], None]] = None) -> T:
        """Call ``fn``, retrying retryable failures up to ``max_retries`` times."""
        last: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except PermanentError:
                raise
            except (RateLimitError, TransientError) as exc:
                last = exc
                if attempt == self.max_retries:
                    break
                delay = self.delay_for(attempt)
                self.retries_used += 1
                if on_retry is not None:
                    on_retry(attempt, delay, exc)
                self._sleep(delay)
            except LLMError:
                raise
        assert last is not None
        raise last
