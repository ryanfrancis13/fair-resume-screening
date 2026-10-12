"""Robust parsing of a model's JSON verdict.

Even with native JSON mode, structured extraction fails in predictable ways:
markdown fences, a trailing comma, a truncated object, a score returned as
``"85/100"``, dimension names in title case.  Each of those is handled here, in
one place, because a parse failure is a *missing data point in a designed
experiment* — and missing data that correlates with the condition (longer, more
hedged rationales for one group truncate more often) would bias the whole result.

The policy is: repair what is unambiguously repairable, and otherwise return a
record marked ``parse_ok=False`` with the raw text retained.  Failures are
counted and reported, never silently dropped.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..schemas import DimensionScore, ScreeningResult
from .evidence import evidence_spans_from_payload
from .rubric import Rubric

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_json(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return ``(parsed, error)``; ``parsed`` is ``None`` on failure."""
    if not text or not text.strip():
        return None, "empty response"

    candidates: List[str] = []
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text.strip())

    # Largest balanced {...} region, for prose wrapped around the object.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    # A complete object followed by trailing junk: decode the first value and
    # ignore the rest, rather than failing on "Extra data".
    if start >= 0:
        try:
            head, _consumed = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(head, dict):
                return head, ""
        except json.JSONDecodeError:
            pass

    last_error = "no JSON object found"
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    return parsed, ""
                last_error = f"top-level JSON is {type(parsed).__name__}, expected object"
            except json.JSONDecodeError as exc:
                last_error = f"JSONDecodeError: {exc.msg} at char {exc.pos}"
        repaired = _repair_truncated(candidate)
        if repaired is not None:
            return repaired, ""
    return None, last_error


def _repair_truncated(text: str) -> Optional[Dict[str, Any]]:
    """Close unbalanced brackets on a response cut off by a token limit.

    A truncated verdict usually still contains the fields that matter (dimension
    scores are emitted before the prose fields), so recovering it keeps a data
    point that would otherwise go missing — and missing-not-at-random is worse
    for the analysis than a partially recovered record.

    Two details make this work in practice:

    * Closers come from a real bracket **stack**, not "all ``]`` then all ``}``".
      Truncation usually lands inside a nested object within an array, where the
      naive order produces ``[{...]}`` and still fails to parse.
    * Repairs are tried **least destructive first**, so a complete trailing value
      is kept rather than stripped along with its key.
    """
    start = text.find("{")
    if start < 0:
        return None
    base = text[start:]

    stages = [base]
    if _unescaped_quote_count(base) % 2 == 1:
        stages.append(base[: _last_unescaped_quote(base)])
    for candidate in list(stages):
        stripped = re.sub(r",\s*\"[^\"]*\"\s*:\s*[^,}\]]*$", "", candidate)
        stripped = re.sub(r"\{\s*\"[^\"]*\"\s*:\s*$", "{", stripped)
        stripped = re.sub(r",\s*\"[^\"]*$", "", stripped)
        stages.append(stripped)

    for candidate in stages:
        candidate = _TRAILING_COMMA_RE.sub(r"\1", re.sub(r"[,:]\s*$", "", candidate))
        closers = _pending_closers(candidate)
        if closers is None:
            continue
        try:
            parsed = json.loads(candidate + closers)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _pending_closers(body: str) -> Optional[str]:
    """Closing brackets needed to balance ``body``, innermost first.

    Returns ``None`` if the text is over-closed, which means it is malformed in a
    way truncation cannot explain and should not be silently patched.
    """
    stack: List[str] = []
    in_string = False
    escaped = False
    for ch in body:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            opener = stack.pop()
            if (opener == "{") != (ch == "}"):
                return None
    if in_string:
        return None
    return "".join("}" if opener == "{" else "]" for opener in reversed(stack))


def _unescaped_quote_count(text: str) -> int:
    return sum(
        1
        for i, ch in enumerate(text)
        if ch == '"' and (i == 0 or text[i - 1] != "\\")
    )


def _last_unescaped_quote(text: str) -> int:
    for i in range(len(text) - 1, -1, -1):
        if text[i] == '"' and (i == 0 or text[i - 1] != "\\"):
            return i
    return len(text)


def coerce_score(value: Any, *, default: float = float("nan")) -> float:
    """Parse a score from ``85``, ``"85"``, ``"85/100"``, ``"85%"`` or ``"eighty"``."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = _SCORE_RE.search(value)
        if match:
            return float(match.group(0))
    return default


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items()]
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(str(item.get("requirement") or item.get("text") or item))
            else:
                out.append(str(item))
        return [s for s in out if s.strip()]
    return [str(value)]


def parse_screening_response(
    text: str,
    rubric: Rubric,
    *,
    recompute_overall: bool = True,
    max_overall_discrepancy: float = 8.0,
) -> ScreeningResult:
    """Turn raw model text into a validated :class:`ScreeningResult`.

    When ``recompute_overall`` is set, the overall score is recomputed from the
    dimension scores using the rubric's weights, and a large discrepancy with the
    model's own figure is recorded. Trusting the model's arithmetic would let an
    unweighted holistic impression override the rubric — which is exactly the
    channel the rubric exists to close.
    """
    payload, error = extract_json(text)
    if payload is None:
        return ScreeningResult(
            overall_score=float("nan"),
            recommendation="",
            parse_ok=False,
            parse_error=error,
            raw_response=text[:4000],
        )

    expected = {_normalise_name(n): n for n in rubric.names()}
    weights = rubric.weights()

    dimensions: List[DimensionScore] = []
    raw_dimensions = payload.get("dimensions")
    if isinstance(raw_dimensions, dict):
        # Some models return {"technical_skills": {...}} instead of a list.
        raw_dimensions = [
            {"name": key, **(value if isinstance(value, dict) else {"score": value})}
            for key, value in raw_dimensions.items()
        ]
    for item in raw_dimensions or []:
        if not isinstance(item, dict):
            continue
        key = _normalise_name(item.get("name", ""))
        canonical = expected.get(key, item.get("name", key))
        dimensions.append(
            DimensionScore(
                name=str(canonical),
                score=coerce_score(item.get("score")),
                weight=float(weights.get(str(canonical), item.get("weight", 0.0) or 0.0)),
                rationale=str(item.get("rationale", "") or "")[:1200],
                evidence=evidence_spans_from_payload(item.get("evidence"), str(canonical)),
            )
        )

    model_overall = coerce_score(payload.get("overall_score"))
    dimension_map = {d.name: d.score for d in dimensions if d.score == d.score}
    computed_overall = rubric.aggregate(dimension_map) if dimension_map else float("nan")

    overall = model_overall
    discrepancy = float("nan")
    if recompute_overall and computed_overall == computed_overall:
        if model_overall == model_overall:
            discrepancy = abs(model_overall - computed_overall)
        overall = computed_overall
    elif overall != overall:
        overall = computed_overall

    missing_dimensions = [n for n in rubric.names() if n not in dimension_map]
    parse_ok = overall == overall and not missing_dimensions
    parse_error = ""
    if missing_dimensions:
        parse_error = f"missing dimension scores: {missing_dimensions}"
    elif overall != overall:
        parse_error = "no usable overall score"

    recommendation = str(payload.get("recommendation", "") or "").strip().lower()
    if recommendation not in ("advance", "borderline", "reject"):
        recommendation = rubric.recommendation_for(overall) if overall == overall else ""

    confidence = coerce_score(payload.get("confidence"), default=float("nan"))
    if confidence == confidence and confidence > 1.0:
        confidence = confidence / 100.0  # model answered on a 0-100 scale

    result = ScreeningResult(
        overall_score=float(overall) if overall == overall else float("nan"),
        recommendation=recommendation,
        confidence=float(confidence) if confidence == confidence else 0.0,
        dimensions=dimensions,
        missing_requirements=_coerce_list(payload.get("missing_requirements")),
        unverifiable_claims=_coerce_list(payload.get("unverifiable_claims")),
        summary=str(payload.get("summary", "") or "")[:1200],
        protected_attribute_referenced=bool(payload.get("protected_attribute_referenced", False)),
        protected_reference_note=str(payload.get("protected_reference_note", "") or "")[:600],
        parse_ok=parse_ok,
        parse_error=parse_error,
        raw_response=text[:4000],
    )
    if discrepancy == discrepancy and discrepancy > max_overall_discrepancy:
        note = (
            f"model overall {model_overall:.1f} differs from rubric-weighted "
            f"{computed_overall:.1f} by {discrepancy:.1f} points; using the weighted value"
        )
        result.parse_error = (result.parse_error + "; " + note).strip("; ")
    return result
