"""Single-pair scoring — the core screening primitive.

This is the generalisation of the project's Phase-0 proof of concept: the same
call/response/parse loop, but accepting *any* ``(resume_text, job)`` pair and
returning a validated, provenance-carrying result instead of printing JSON.

What it adds beyond "call the model and parse":

* **Self-consistency sampling.** ``llm.self_consistency_samples > 1`` repeats the
  identical prompt at a non-zero temperature. Those repeats are what make the
  model's own noise floor measurable, and without a noise floor a "2-point gap
  between name groups" cannot be told apart from decoding variance.
* **Evidence-grounding retry.** If too few quotes are locatable in the resume,
  the score is re-requested once with an explicit verbatim-quote instruction. If
  it fails again the record is kept and flagged, never silently used.
* **Optional pre-inference redaction**, so blind screening is a first-class arm
  of the experiment rather than a separate script.
* **Complete provenance.** Prompt and response hashes, model, temperature and
  timing travel with every score, so a number in the final report can be traced
  to the exact text that produced it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..anonymize import redact
from ..config import Config
from ..llm.base import LLMRequest, LLMResponse
from ..llm.client import LLMClient
from ..logging_setup import log_event
from ..schemas import JobDescription, ScreeningResult
from ..utils.hashing import hash_text
from .evidence import check_evidence
from .parse import parse_screening_response
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_screening_prompt
from .rubric import Rubric, get_rubric

logger = logging.getLogger("frs.screening.score")

_GROUNDING_RETRY_INSTRUCTION = (
    "IMPORTANT: your previous attempt included quotes that do not appear in the resume. "
    "Every string in an \"evidence\" field must be copied CHARACTER-FOR-CHARACTER from the "
    "resume text above. If you cannot find a supporting quote for a dimension, return an "
    "empty evidence list for it and lower that dimension's score accordingly."
)


@dataclass
class ScoringOutcome:
    """One scored (resume, job) observation with everything needed to audit it."""

    result: ScreeningResult
    response: LLMResponse
    prompt_hash: str
    resume_hash: str
    job_hash: str
    sample_index: int = 0
    temperature: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    grounding_retried: bool = False
    redaction: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.result.parse_ok and not self.response.error

    @property
    def evidence_support_rate(self) -> float:
        value = self.evidence.get("support_rate", float("nan"))
        return float(value) if value == value else float("nan")


class ResumeScorer:
    """Scores resume/job pairs against a rubric via an :class:`LLMClient`."""

    def __init__(
        self,
        client: LLMClient,
        cfg: Config,
        rubric: Optional[Rubric] = None,
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.rubric = rubric or get_rubric(
            cfg.screening.rubric,
            advance=cfg.screening.advance_threshold,
            borderline=cfg.screening.borderline_threshold,
        )

    # ------------------------------------------------------------------ #
    def build_request(
        self,
        resume_text: str,
        job: JobDescription,
        *,
        sample_index: int = 0,
        debias_instruction: Optional[bool] = None,
        extra_instruction: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> LLMRequest:
        """Construct the request without sending it (useful for batching/dry runs)."""
        prompt = build_screening_prompt(
            resume_text,
            job,
            self.rubric,
            debias_instruction=(
                self.cfg.screening.debias_instruction
                if debias_instruction is None
                else debias_instruction
            ),
            extra_instruction=extra_instruction,
        )
        # Sample 0 is a greedy decode so the primary estimate is reproducible;
        # additional samples are drawn hotter to expose decoding variance.
        temperature = (
            self.cfg.llm.temperature
            if sample_index == 0
            else self.cfg.llm.self_consistency_temperature
        )
        return LLMRequest(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=temperature,
            top_p=self.cfg.llm.top_p,
            max_output_tokens=self.cfg.llm.max_output_tokens,
            sample_index=sample_index,
            tags={"prompt_version": PROMPT_VERSION, **(tags or {})},
        )

    # ------------------------------------------------------------------ #
    def prepare_text(self, resume_text: str, *, resume_name: Optional[str] = None) -> Dict[str, Any]:
        """Apply the configured redaction level, returning text plus a report."""
        if not self.cfg.screening.anonymize:
            return {"text": resume_text, "redaction": {}}
        redacted = redact(
            resume_text,
            self.cfg.screening.anonymize_level,
            known_name=resume_name,
            measure_residual=True,
        )
        return {
            "text": redacted.text,
            "redaction": {
                "level": redacted.level,
                "n_substitutions": redacted.n_substitutions,
                "removed_lines": redacted.removed_lines,
                "by_category": redacted.by_category,
                "residual_cue_hits": redacted.residual.get("residual_cue_hits"),
                "residual_attributes": redacted.residual.get("detected_attributes"),
            },
        }

    # ------------------------------------------------------------------ #
    def score_once(
        self,
        resume_text: str,
        job: JobDescription,
        *,
        sample_index: int = 0,
        resume_name: Optional[str] = None,
        debias_instruction: Optional[bool] = None,
        tags: Optional[Dict[str, Any]] = None,
        allow_grounding_retry: bool = True,
    ) -> ScoringOutcome:
        """Score one pair, with one grounding retry if the evidence is weak."""
        prepared = self.prepare_text(resume_text, resume_name=resume_name)
        text = prepared["text"]

        request = self.build_request(
            text, job, sample_index=sample_index,
            debias_instruction=debias_instruction, tags=tags,
        )
        response = self.client.generate(request)
        outcome = self._finalise(request, response, text, job, sample_index, prepared)

        needs_retry = (
            allow_grounding_retry
            and self.cfg.screening.require_evidence
            and outcome.ok
            and outcome.evidence_support_rate == outcome.evidence_support_rate
            and outcome.evidence_support_rate < self.cfg.screening.min_evidence_support_rate
        )
        if needs_retry:
            log_event(
                logger, logging.DEBUG, "re-requesting score with verbatim-quote instruction",
                support_rate=round(outcome.evidence_support_rate, 3), tags=tags,
            )
            retry_request = self.build_request(
                text, job,
                # A different sample index gives a distinct cache key, so the
                # retry cannot be served the identical cached response.
                sample_index=sample_index + 1000,
                debias_instruction=debias_instruction,
                extra_instruction=_GROUNDING_RETRY_INSTRUCTION,
                tags=tags,
            )
            retry_response = self.client.generate(retry_request)
            retry_outcome = self._finalise(
                retry_request, retry_response, text, job, sample_index, prepared
            )
            retry_outcome.grounding_retried = True
            if retry_outcome.ok and (
                not outcome.ok
                or retry_outcome.evidence_support_rate >= outcome.evidence_support_rate
            ):
                return retry_outcome
            outcome.grounding_retried = True
        return outcome

    # ------------------------------------------------------------------ #
    def score_samples(
        self,
        resume_text: str,
        job: JobDescription,
        *,
        n_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> List[ScoringOutcome]:
        """Score the same pair ``n_samples`` times for reliability estimation."""
        n = n_samples if n_samples is not None else self.cfg.llm.self_consistency_samples
        return [self.score_once(resume_text, job, sample_index=i, **kwargs) for i in range(max(1, n))]

    # ------------------------------------------------------------------ #
    def _finalise(
        self,
        request: LLMRequest,
        response: LLMResponse,
        resume_text: str,
        job: JobDescription,
        sample_index: int,
        prepared: Dict[str, Any],
    ) -> ScoringOutcome:
        if response.error:
            result = ScreeningResult(
                overall_score=float("nan"),
                recommendation="",
                parse_ok=False,
                parse_error=f"provider error: {response.error}",
            )
            evidence: Dict[str, Any] = {}
        else:
            result = parse_screening_response(response.text, self.rubric)
            evidence = check_evidence(result.dimensions, resume_text)

        return ScoringOutcome(
            result=result,
            response=response,
            prompt_hash=hash_text(request.prompt),
            resume_hash=hash_text(resume_text),
            job_hash=hash_text(job.text),
            sample_index=sample_index,
            temperature=request.temperature,
            evidence=evidence,
            redaction=prepared.get("redaction", {}),
        )


# --------------------------------------------------------------------------- #
# convenience entry point
# --------------------------------------------------------------------------- #
def score_resume(
    resume_text: str,
    job: JobDescription,
    *,
    cfg: Optional[Config] = None,
    client: Optional[LLMClient] = None,
) -> ScoringOutcome:
    """Score one pair with default settings.

    The smallest useful entry point into the system — a single call that proves
    the provider, prompt, parser and grounding check all work together, which is
    what the project's Phase-0 milestone asks for::

        from frs.config import Config
        from frs.data import load_corpus
        from frs.screening import score_resume

        cfg = Config.load()
        resumes, jobs, _ = load_corpus(cfg)
        outcome = score_resume(resumes[0].text, jobs[0], cfg=cfg)
        print(outcome.result.overall_score, outcome.result.recommendation)
    """
    cfg = cfg or Config.load()
    owns_client = client is None
    client = client or LLMClient.from_config(cfg)
    try:
        return ResumeScorer(client, cfg).score_once(resume_text, job)
    finally:
        if owns_client:
            client.close()
