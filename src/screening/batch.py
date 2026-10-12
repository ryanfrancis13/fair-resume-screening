"""Batch scoring: the runner that turns a design into a score table.

The unit of work is a :class:`BatchItem` — one ``(job, variant, sample, arm)``
tuple.  Enumerating the work up front, then executing it in two concurrent
passes (initial, then grounding retries), has three practical benefits over a
per-item loop:

* **Concurrency without losing the retry logic.** Retries are collected and
  issued as a second batch, so the grounding gate does not serialise the run.
* **Cost is known before spending it.** :func:`estimate_cost` reports the call
  count and projected wall-clock from the configured rate limit, which is what
  stops a factorial design from quietly consuming a day's free-tier quota.
* **Failures are data.** Every item yields exactly one row. A provider error
  becomes a row with ``parse_ok=False`` and an ``error`` string, so the analysis
  can report attrition instead of unknowingly analysing a filtered subset — and
  attrition that correlates with the experimental condition is itself a finding.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import Config
from ..llm.base import LLMRequest, LLMResponse
from ..logging_setup import log_event
from ..schemas import JobDescription, ResumeVariant, ScoreRecord
from .prompts import PROMPT_VERSION
from .score_resume import ResumeScorer

logger = logging.getLogger("frs.screening.batch")


@dataclass
class BatchItem:
    """One planned scoring call."""

    job: JobDescription
    variant: ResumeVariant
    sample_index: int = 0
    arm: str = "baseline"
    #: Overrides for this item, used by mitigation arms.
    debias_instruction: Optional[bool] = None
    text_override: Optional[str] = None
    resume_name: Optional[str] = None
    qualified_label: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.text_override if self.text_override is not None else self.variant.text

    @property
    def key(self) -> str:
        return f"{self.arm}|{self.job.job_id}|{self.variant.variant_id}|{self.sample_index}"


def build_items(
    jobs: Sequence[JobDescription],
    variants: Sequence[ResumeVariant],
    pairs: Sequence[Any],
    *,
    n_samples: int = 1,
    arm: str = "baseline",
    include_invalid_variants: bool = False,
) -> List[BatchItem]:
    """Cross the pair list with the variant set.

    A variant that failed validation is excluded by default but *counted*: the
    exclusion is reported so the reader knows how much of the design survived,
    rather than seeing a silently smaller table.
    """
    jobs_by_id = {j.job_id: j for j in jobs}
    variants_by_resume: Dict[str, List[ResumeVariant]] = {}
    n_excluded = 0
    for variant in variants:
        if not include_invalid_variants and not variant.is_valid:
            n_excluded += 1
            continue
        variants_by_resume.setdefault(variant.base_resume_id, []).append(variant)

    items: List[BatchItem] = []
    for pair in pairs:
        job = jobs_by_id.get(pair.job_id)
        if job is None:
            continue
        for variant in variants_by_resume.get(pair.resume_id, []):
            for sample_index in range(max(1, n_samples)):
                items.append(
                    BatchItem(
                        job=job,
                        variant=variant,
                        sample_index=sample_index,
                        arm=arm,
                        qualified_label=getattr(pair, "qualified_label", None),
                    )
                )
    if n_excluded:
        log_event(
            logger, logging.WARNING, "excluded variants that failed validation",
            n_excluded=n_excluded,
        )
    return items


def estimate_cost(items: Sequence[BatchItem], cfg: Config) -> Dict[str, Any]:
    """Projected call count and wall-clock, before any money or quota is spent."""
    n_calls = len(items)
    rpm = max(1, cfg.llm.requests_per_minute)
    minutes = n_calls / rpm
    return {
        "n_calls": n_calls,
        "provider": cfg.llm.provider,
        "model": cfg.llm.model,
        "requests_per_minute": rpm,
        "max_concurrency": cfg.llm.max_concurrency,
        "estimated_minutes_rate_limited": round(minutes, 2),
        "note": (
            "Cache hits do not consume quota, so a re-run of an unchanged design "
            "is effectively free."
        ),
    }


def run_batch(
    scorer: ResumeScorer,
    items: Sequence[BatchItem],
    cfg: Config,
    *,
    run_id: str = "adhoc",
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[ScoreRecord], Dict[str, Any]]:
    """Execute a batch and return ``(records, telemetry)``."""
    if not items:
        return [], {"n_items": 0}

    started = time.perf_counter()
    requests = [_request_for(scorer, item) for item in items]
    log_event(
        logger, logging.INFO, "scoring batch",
        n_items=len(items), provider=cfg.llm.provider, model=cfg.llm.model,
        concurrency=cfg.llm.max_concurrency,
    )
    responses = scorer.client.generate_many(requests, progress=progress)

    records: List[ScoreRecord] = []
    retry_indices: List[int] = []
    for index, (item, request, response) in enumerate(zip(items, requests, responses)):
        record = _record_for(scorer, item, request, response, run_id, cfg)
        records.append(record)
        if _needs_grounding_retry(record, cfg):
            retry_indices.append(index)

    n_retried = 0
    if retry_indices and cfg.screening.require_evidence:
        from .score_resume import _GROUNDING_RETRY_INSTRUCTION

        log_event(
            logger, logging.INFO, "re-requesting weakly grounded scores",
            n_retry=len(retry_indices),
        )
        retry_requests = [
            _request_for(
                scorer, items[i],
                sample_offset=1000,
                extra_instruction=_GROUNDING_RETRY_INSTRUCTION,
            )
            for i in retry_indices
        ]
        retry_responses = scorer.client.generate_many(retry_requests)
        for i, request, response in zip(retry_indices, retry_requests, retry_responses):
            candidate = _record_for(scorer, items[i], request, response, run_id, cfg)
            previous = records[i]
            improved = candidate.parse_ok and (
                not previous.parse_ok
                or _rate(candidate.evidence_support_rate) >= _rate(previous.evidence_support_rate)
            )
            if improved:
                records[i] = candidate
            records[i].error = (records[i].error + "; grounding_retry").strip("; ")
            n_retried += 1

    elapsed = time.perf_counter() - started
    telemetry = {
        "n_items": len(items),
        "n_records": len(records),
        "n_parse_failures": sum(1 for r in records if not r.parse_ok),
        "n_provider_errors": sum(1 for r in records if r.error and "grounding" not in r.error),
        "n_grounding_retries": n_retried,
        "elapsed_s": round(elapsed, 2),
        "records_per_s": round(len(records) / elapsed, 2) if elapsed > 0 else 0.0,
        "prompt_version": PROMPT_VERSION,
        "client": scorer.client.telemetry(),
    }
    log_event(logger, logging.INFO, "batch complete", **{
        k: v for k, v in telemetry.items() if k != "client"
    })
    return records, telemetry


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _request_for(
    scorer: ResumeScorer,
    item: BatchItem,
    *,
    sample_offset: int = 0,
    extra_instruction: Optional[str] = None,
) -> LLMRequest:
    prepared = scorer.prepare_text(item.text, resume_name=item.resume_name)
    request = scorer.build_request(
        prepared["text"],
        item.job,
        sample_index=item.sample_index + sample_offset,
        debias_instruction=item.debias_instruction,
        extra_instruction=extra_instruction,
        tags={
            "job_id": item.job.job_id,
            "variant_id": item.variant.variant_id,
            "arm": item.arm,
        },
    )
    # Carry the (possibly redacted) text forward so grounding is checked against
    # what the model actually saw, not against the original document.
    request.tags["scored_text"] = prepared["text"]
    request.tags["redaction"] = prepared.get("redaction", {})
    return request


def _needs_grounding_retry(record: ScoreRecord, cfg: Config) -> bool:
    if not cfg.screening.require_evidence or not record.parse_ok:
        return False
    rate = record.evidence_support_rate
    return rate == rate and rate < cfg.screening.min_evidence_support_rate


def _rate(value: float) -> float:
    return value if value == value else -1.0


def _record_for(
    scorer: ResumeScorer,
    item: BatchItem,
    request: LLMRequest,
    response: LLMResponse,
    run_id: str,
    cfg: Config,
) -> ScoreRecord:
    from ..utils.hashing import hash_text
    from .evidence import check_evidence
    from .parse import parse_screening_response

    scored_text = str(request.tags.get("scored_text", item.text))

    if response.error:
        result = None
        evidence: Dict[str, Any] = {}
    else:
        result = parse_screening_response(response.text, scorer.rubric)
        evidence = check_evidence(result.dimensions, scored_text)

    rationale = ""
    if result is not None:
        rationale = " ".join(
            [result.summary] + [d.rationale for d in result.dimensions if d.rationale]
        ).strip()

    return ScoreRecord(
        run_id=run_id,
        job_id=item.job.job_id,
        base_resume_id=item.variant.base_resume_id,
        variant_id=item.variant.variant_id,
        sample_index=item.sample_index,
        overall_score=result.overall_score if result else float("nan"),
        recommendation=result.recommendation if result else "",
        confidence=result.confidence if result else 0.0,
        is_baseline=item.variant.is_baseline,
        cue_families=",".join(item.variant.cue_families),
        cue_ids=";".join(f"{k}={v}" for k, v in sorted(item.variant.cue_ids.items())),
        arm=item.arm,
        signals=dict(item.variant.signals),
        dimension_scores=result.dimension_map() if result else {},
        missing_requirements=result.missing_requirements if result else [],
        summary=result.summary if result else "",
        rationale_text=rationale[:4000],
        evidence_support_rate=float(evidence.get("support_rate", float("nan"))),
        protected_attribute_referenced=(
            result.protected_attribute_referenced if result else False
        ),
        parse_ok=bool(result.parse_ok) if result else False,
        parse_error=(result.parse_error if result else f"provider error: {response.error}"),
        provider=response.provider or cfg.llm.provider,
        model=response.model or cfg.llm.model,
        temperature=request.temperature,
        prompt_hash=hash_text(request.prompt),
        response_hash=hash_text(response.text) if response.text else "",
        resume_hash=hash_text(scored_text),
        job_hash=hash_text(item.job.text),
        cached=response.cached,
        latency_ms=round(response.latency_ms, 1),
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        error=response.error,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        qualified_label=item.qualified_label,
    )


# --------------------------------------------------------------------------- #
# Phase-2 sanity check
# --------------------------------------------------------------------------- #
def sanity_check(
    records: Sequence[ScoreRecord],
    pairs: Sequence[Any],
    *,
    min_rank_correlation: float = 0.4,
    min_strong_weak_gap: float = 8.0,
) -> Dict[str, Any]:
    """Does the scorer separate known-strong from known-weak pairings?

    This is the gate the build plan puts between Phase 2 and Phase 3, and the
    reason is decisive: if the scorer cannot tell a strong fit from a weak one,
    then *any* bias measured later is noise from a broken matcher rather than a
    property of the model. The check is automated here so it cannot be skipped,
    and it is deliberately reported as an advisory verdict with numbers attached
    rather than as a hard failure — a small sample can fail it by chance.
    """
    import math

    from scipy import stats

    fit_rank = {"weak": 0, "moderate": 1, "strong": 2}
    label_by_pair = {
        (p.job_id, p.resume_id): p.expected_fit for p in pairs if p.expected_fit
    }

    xs: List[float] = []
    ys: List[float] = []
    by_fit: Dict[str, List[float]] = {"weak": [], "moderate": [], "strong": []}
    for record in records:
        if not record.parse_ok or not record.is_baseline:
            continue
        fit = label_by_pair.get((record.job_id, record.base_resume_id))
        if fit not in fit_rank:
            continue
        xs.append(float(fit_rank[fit]))
        ys.append(float(record.overall_score))
        by_fit[fit].append(float(record.overall_score))

    result: Dict[str, Any] = {
        "n_graded_pairs": len(xs),
        "mean_score_by_expected_fit": {
            fit: (round(sum(v) / len(v), 2) if v else None) for fit, v in by_fit.items()
        },
        "n_by_expected_fit": {fit: len(v) for fit, v in by_fit.items()},
    }

    if len(xs) < 4 or len(set(xs)) < 2:
        result["verdict"] = "insufficient_data"
        result["note"] = "need at least 4 graded baseline pairs spanning 2 fit levels"
        return result

    tau, tau_p = stats.kendalltau(xs, ys)
    spearman, spearman_p = stats.spearmanr(xs, ys)
    strong = by_fit["strong"]
    weak = by_fit["weak"]
    gap = (sum(strong) / len(strong) - sum(weak) / len(weak)) if strong and weak else float("nan")

    result.update(
        {
            "kendall_tau": round(float(tau), 4) if tau == tau else None,
            "kendall_p": round(float(tau_p), 6) if tau_p == tau_p else None,
            "spearman_rho": round(float(spearman), 4) if spearman == spearman else None,
            "spearman_p": round(float(spearman_p), 6) if spearman_p == spearman_p else None,
            "strong_minus_weak_gap": round(gap, 2) if not math.isnan(gap) else None,
            "thresholds": {
                "min_rank_correlation": min_rank_correlation,
                "min_strong_weak_gap": min_strong_weak_gap,
            },
        }
    )

    passes_rank = tau == tau and tau >= min_rank_correlation
    passes_gap = not math.isnan(gap) and gap >= min_strong_weak_gap
    result["verdict"] = "pass" if (passes_rank and passes_gap) else "review"
    result["note"] = (
        "Scorer separates strong from weak fits; Phase 3 perturbation results can be "
        "interpreted as model behaviour."
        if result["verdict"] == "pass"
        else "Scorer discrimination is weak. Any downstream bias estimate may reflect a "
             "noisy matcher rather than model bias. Inspect prompts, rubric and pairings "
             "before drawing fairness conclusions."
    )
    return result
