"""Evidence grounding: checking that a rationale is anchored in the document.

An LLM will happily justify any score with fluent prose.  The only way to tell a
grounded judgement from a confabulated one at scale is to require verbatim
quotations and then check them against the source.  That check yields
``evidence_support_rate`` — the fraction of quotes locatable in the resume —
which serves three purposes:

* **Quality gate.** Scores below the configured support rate are re-requested
  once, then flagged. The fairness analysis can exclude them, because a score
  whose stated reasons are not in the document is not measuring the document.
* **Validity evidence.** Reporting the grounding rate answers the question
  Castleman et al. (2026) raise about LLM screening: is the score measuring the
  resume at all?
* **A fairness signal in its own right.** If grounding is systematically weaker
  for one group, the model is reasoning less from evidence and more from priors
  for that group — a bias that a score-gap metric alone would never surface.

Matching is deliberately fuzzy.  Models re-wrap lines and normalise punctuation;
a strict substring test would measure formatting fidelity instead of grounding.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..schemas import DimensionScore, EvidenceSpan
from ..utils.text import fuzzy_contains

#: A quote shorter than this is not evidence — "Python" appears in half the
#: corpus and locating it proves nothing.
MIN_QUOTE_CHARS = 12


def check_evidence(
    dimensions: Sequence[DimensionScore],
    resume_text: str,
    *,
    threshold: float = 0.85,
) -> Dict[str, Any]:
    """Verify every quote against the resume; annotate spans in place."""
    total = 0
    found = 0
    too_short = 0
    per_dimension: Dict[str, float] = {}
    unsupported: List[str] = []

    for dimension in dimensions:
        dim_total = 0
        dim_found = 0
        for span in dimension.evidence:
            quote = (span.quote or "").strip()
            if len(quote) < MIN_QUOTE_CHARS:
                too_short += 1
                span.found_in_resume = False
                continue
            total += 1
            dim_total += 1
            ok = fuzzy_contains(resume_text, quote, threshold=threshold)
            span.found_in_resume = ok
            if ok:
                found += 1
                dim_found += 1
            else:
                unsupported.append(quote[:120])
        per_dimension[dimension.name] = (dim_found / dim_total) if dim_total else float("nan")

    return {
        "n_quotes": total,
        "n_found": found,
        "n_too_short": too_short,
        "support_rate": (found / total) if total else float("nan"),
        "per_dimension_support": per_dimension,
        "unsupported_quotes": unsupported[:6],
        "dimensions_without_evidence": [
            d.name for d in dimensions if not d.evidence
        ],
    }


def evidence_spans_from_payload(payload: Any, dimension_name: str) -> List[EvidenceSpan]:
    """Parse the ``evidence`` field, tolerating the shapes models actually emit."""
    spans: List[EvidenceSpan] = []
    if payload is None:
        return spans
    if isinstance(payload, str):
        return [EvidenceSpan(quote=payload, supports=dimension_name)]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, (list, tuple)):
        return spans
    for item in payload:
        if isinstance(item, str):
            spans.append(EvidenceSpan(quote=item, supports=dimension_name))
        elif isinstance(item, dict):
            quote = item.get("quote") or item.get("text") or item.get("evidence") or ""
            spans.append(
                EvidenceSpan(quote=str(quote), supports=str(item.get("supports", dimension_name)))
            )
    return spans


def aggregate_grounding(records: Sequence[Any]) -> Dict[str, Any]:
    """Corpus-level grounding statistics for the report."""
    import math

    rates = [
        r.evidence_support_rate
        for r in records
        if isinstance(getattr(r, "evidence_support_rate", None), float)
        and not math.isnan(r.evidence_support_rate)
    ]
    if not rates:
        return {"n": 0, "mean_support_rate": float("nan")}
    rates_sorted = sorted(rates)
    n = len(rates_sorted)
    return {
        "n": n,
        "mean_support_rate": round(sum(rates) / n, 4),
        "median_support_rate": round(rates_sorted[n // 2], 4),
        "p10_support_rate": round(rates_sorted[max(0, int(0.10 * n) - 1)], 4),
        "fraction_fully_grounded": round(sum(1 for r in rates if r >= 0.999) / n, 4),
        "fraction_below_half": round(sum(1 for r in rates if r < 0.5) / n, 4),
    }
