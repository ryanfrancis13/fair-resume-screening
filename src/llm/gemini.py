"""Google Gemini provider (default: ``gemini-2.5-flash``).

Chosen in the project's technical decisions for a free tier that survives the
call volume a perturbation study needs — every base resume is scored once per
cue level per repeat, so a 12-resume × 8-cue-family design is already in the
hundreds of calls.

The SDK is imported lazily so that the rest of the package — and the entire
test suite — installs and runs without ``google-genai`` present.  Provider
exceptions are normalised onto the retry taxonomy in :mod:`frs.llm.base`, since
the retry loop must distinguish "wait and try again" (429, 503) from "your key
is wrong" (401), and the SDK signals both as generic errors.
"""

from __future__ import annotations

import json
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

_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "resource exhausted", "quota", "rate limit")
_TRANSIENT_MARKERS = ("500", "502", "503", "504", "deadline", "timeout", "unavailable",
                      "internal error", "connection", "temporarily")
_PERMANENT_MARKERS = ("401", "403", "api key not valid", "permission_denied", "invalid_argument",
                      "unauthenticated", "not found", "404")


class GeminiProvider(LLMProvider):
    """Thin wrapper over ``google-genai``'s ``generate_content``."""

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: Optional[str] = None,
        timeout_s: float = 90.0,
    ) -> None:
        super().__init__(model)
        self.timeout_s = timeout_s
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        self._client: Any = None
        self._types: Any = None

    # ------------------------------------------------------------------ #
    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if not self._api_key:
            raise PermanentError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
                "(https://aistudio.google.com/apikey), or run with --provider mock."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise PermanentError(
                "google-genai is not installed. Install it with: uv sync --extra gemini"
            ) from exc
        self._client = genai.Client(api_key=self._api_key)
        self._types = types

    # ------------------------------------------------------------------ #
    def generate(self, request: LLMRequest) -> LLMResponse:
        self._ensure_client()
        types = self._types
        started = time.perf_counter()

        config_kwargs: Dict[str, Any] = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_output_tokens": request.max_output_tokens,
            # Best-effort determinism: temperature 0 alone still varies between
            # calls on hosted models. Seeding by sample index keeps a given
            # sample repeatable while letting repeats of one prompt differ.
            "seed": request.sample_index,
            # This provider requests JSON text only; it does not register tools.
            # Disable the SDK's automatic function-calling path so ordinary
            # generation does not enter the AFC wrapper.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if request.system:
            config_kwargs["system_instruction"] = request.system
        if request.response_mime_type:
            # Native JSON mode removes the most common failure mode of
            # structured extraction: a model wrapping valid JSON in prose or a
            # markdown fence. The parser still tolerates both, but this makes
            # the tolerant path rare rather than routine.
            config_kwargs["response_mime_type"] = request.response_mime_type

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=request.prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise _classify(exc) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        text = _response_text(response)
        finish_reason, blocked = _finish_reason(response)

        if not text.strip():
            if blocked:
                # A safety block on a resume is a signal worth surfacing, not
                # swallowing: it silently removes a candidate from the study and
                # would otherwise look like a missing row.
                raise PermanentError(f"response blocked by safety filter ({finish_reason})")
            raise TransientError(f"empty response from {self.model} ({finish_reason})")

        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            completion_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            finish_reason=finish_reason,
        )

    # ------------------------------------------------------------------ #
    def health_check(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"provider": self.name, "model": self.model, "ready": False}
        try:
            self._ensure_client()
        except PermanentError as exc:
            info["error"] = str(exc)
            return info
        try:
            probe = self.generate(
                LLMRequest(
                    prompt='Reply with exactly this JSON and nothing else: {"ok": true}',
                    temperature=0.0,
                    max_output_tokens=32,
                )
            )
            info["ready"] = probe.ok
            info["latency_ms"] = round(probe.latency_ms, 1)
            info["sample"] = probe.text[:80]
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            info["error"] = f"{type(exc).__name__}: {exc}"
        return info


# --------------------------------------------------------------------------- #
def _classify(exc: Exception) -> Exception:
    """Map an SDK exception onto the retry taxonomy."""
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    blob = f"{type(exc).__name__} {status} {exc}".lower()
    if any(m in blob for m in _RATE_LIMIT_MARKERS):
        return RateLimitError(str(exc))
    if any(m in blob for m in _PERMANENT_MARKERS):
        return PermanentError(str(exc))
    if any(m in blob for m in _TRANSIENT_MARKERS):
        return TransientError(str(exc))
    # Unknown failures are treated as transient: one wasted retry costs a second,
    # whereas mis-classifying a recoverable blip as fatal loses the whole batch.
    return TransientError(str(exc))


def _response_text(response: Any) -> str:
    """Extract text robustly across SDK response shapes."""
    try:
        text = getattr(response, "text", None)
    except Exception:
        # Some google-genai response objects assert when `text` is accessed
        # without a text-bearing candidate (for example after a blocked or
        # quota-related response). Fall through to the structured fields so
        # the caller can surface the actual response state.
        text = None
    if isinstance(text, str) and text.strip():
        return text
    parts_text = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            piece = getattr(part, "text", None)
            if isinstance(piece, str):
                parts_text.append(piece)
    if parts_text:
        return "".join(parts_text)
    # Last resort: some SDK versions expose only a dict view.
    try:
        return json.dumps(response.to_json_dict())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return ""


def _finish_reason(response: Any) -> tuple[str, bool]:
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None)
    if block_reason:
        return str(block_reason), True
    for candidate in getattr(response, "candidates", None) or []:
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            name = getattr(reason, "name", None) or str(reason)
            return name, name.upper() in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "RECITATION")
    return "unknown", False
