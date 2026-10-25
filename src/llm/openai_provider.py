"""Optional OpenAI-compatible provider.

Present so that a headline finding can be replicated on a second model family
without touching the pipeline — "does this bias reproduce elsewhere?" is the
first question a reviewer asks, and the answer should not require a refactor.
Also works against any OpenAI-compatible endpoint via ``OPENAI_BASE_URL``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from .base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    PermanentError,
    RateLimitError,
    TransientError,
)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: float = 90.0,
    ) -> None:
        super().__init__(model)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.timeout_s = timeout_s
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if not self._api_key:
            raise PermanentError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise PermanentError("openai is not installed. Install it with: uv sync --extra openai") from exc
        kwargs: Dict[str, Any] = {"api_key": self._api_key, "timeout": self.timeout_s}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)

    def generate(self, request: LLMRequest) -> LLMResponse:
        self._ensure_client()
        started = time.perf_counter()
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_output_tokens,
            # Best-effort repeatability; repeats of one prompt differ by index.
            "seed": request.sample_index,
        }
        if request.response_mime_type == "application/json":
            kwargs["response_format"] = {"type": "json_object"}
        try:
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            blob = f"{type(exc).__name__} {exc}".lower()
            if "rate" in blob or "429" in blob:
                raise RateLimitError(str(exc)) from exc
            if any(m in blob for m in ("401", "403", "invalid_api_key", "permission")):
                raise PermanentError(str(exc)) from exc
            raise TransientError(str(exc)) from exc

        choice = completion.choices[0]
        usage = getattr(completion, "usage", None)
        return LLMResponse(
            text=choice.message.content or "",
            model=self.model,
            provider=self.name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            finish_reason=str(getattr(choice, "finish_reason", "")),
        )
