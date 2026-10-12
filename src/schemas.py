"""Data contracts shared by every layer of the pipeline.

These dataclasses are the *only* coupling between the screening side and the
fairness side of the project.  The screening layer produces
:class:`ScoreRecord` rows; the fairness layer consumes a tidy table of them and
never calls an LLM.  That separation is what lets the audit be re-run offline on
committed artifacts, and lets the auditor be validated against a provider with a
known, injected bias (see :mod:`frs.llm.mock`).

One row per ``(job, base_resume, variant, sample)`` tuple is the atomic unit of
the whole study:

* ``base_resume_id`` identifies the underlying candidate content — the
  statistical *block*, held constant within a variant set.
* ``variant_id`` identifies one counterfactual realisation of that content.
* ``sample_index`` identifies a decoding repeat of the *same* prompt, which is
  what makes the model's own noise floor estimable and separable from any cue
  effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #

#: Attributes we treat as protected for audit purposes.  ``education_prestige``
#: and ``employment_gap`` are *not* protected: they are validity probes, used to
#: confirm the scorer does respond to legitimately job-relevant signal.  A
#: scorer that is invariant to everything is not fair, it is broken.
PROTECTED_ATTRIBUTES: Sequence[str] = (
    "gender",
    "race_ethnicity",
    "nationality_region",
    "age_band",
    "disability_status",
    "socioeconomic_background",
    "religion",
)

VALIDITY_ATTRIBUTES: Sequence[str] = (
    "education_prestige",
    "skill_match_level",
    "writing_register",
)

RECOMMENDATIONS: Sequence[str] = ("advance", "borderline", "reject")


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #
@dataclass
class Resume:
    """One candidate document.

    ``template`` and ``slots`` are populated for the curated sample corpus,
    where each cue-bearing span is declared explicitly.  Resumes ingested from a
    public dataset have ``template=None``, and the perturbation layer falls back
    to regex span *detection* for them.  Both paths converge on the same
    validated variant objects.
    """

    resume_id: str
    text: str
    job_family: str
    source: str = "sample"
    template: Optional[str] = None
    slots: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_templated(self) -> bool:
        return self.template is not None

    def __post_init__(self) -> None:
        if not self.resume_id:
            raise ValueError("resume_id must be non-empty")
        if not self.text.strip():
            raise ValueError(f"resume {self.resume_id!r} has empty text")


@dataclass
class JobDescription:
    """One posting, with its requirements broken out for the rubric."""

    job_id: str
    title: str
    text: str
    job_family: str
    must_have: List[str] = field(default_factory=list)
    nice_to_have: List[str] = field(default_factory=list)
    min_years_experience: float = 0.0
    source: str = "sample"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(f"job {self.job_id!r} has empty text")


@dataclass
class Pair:
    """A resume matched to a job description, with the reason for the match.

    Pairing is deliberately *not* random: a fairness result computed over
    nonsense pairings measures nothing, because every candidate scores near the
    floor and there is no variance for a cue effect to show up in.
    """

    pair_id: str
    job_id: str
    resume_id: str
    pairing_rationale: str = "same job family"
    expected_fit: Optional[str] = None  # strong | moderate | weak — for sanity checks
    qualified_label: Optional[int] = None  # optional human label, for calibration


# --------------------------------------------------------------------------- #
# perturbation
# --------------------------------------------------------------------------- #
@dataclass
class Edit:
    """One span-level change applied to a resume, recorded for auditability."""

    slot: str
    cue_family: str
    before: str
    after: str
    start: int = -1
    end: int = -1


@dataclass
class CueAssignment:
    """The cue values realised in a variant, and what they are taken to signal.

    ``signals`` maps a protected (or validity) attribute onto the level implied
    by the assigned cue — e.g. ``{"gender": "female", "race_ethnicity": "Black"}``.
    Keeping the *cue* and the *attribute it signals* in separate fields is
    important: the project measures sensitivity to a textual cue, and only
    claims a demographic interpretation to the extent the cue list justifies it.
    """

    cue_family: str
    cue_id: str
    values: Dict[str, str] = field(default_factory=dict)
    signals: Dict[str, str] = field(default_factory=dict)
    provenance: str = "unspecified"


@dataclass
class ResumeVariant:
    """A counterfactual copy of a base resume."""

    variant_id: str
    base_resume_id: str
    text: str
    is_baseline: bool = False
    cue_families: List[str] = field(default_factory=list)
    cue_ids: Dict[str, str] = field(default_factory=dict)
    signals: Dict[str, str] = field(default_factory=dict)
    edits: List[Edit] = field(default_factory=list)
    provenance: Dict[str, str] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return bool(self.validation.get("passed", True))


# --------------------------------------------------------------------------- #
# screening output
# --------------------------------------------------------------------------- #
@dataclass
class EvidenceSpan:
    """A quotation from the resume backing one dimension score."""

    quote: str
    supports: str
    found_in_resume: bool = False


@dataclass
class DimensionScore:
    """One rubric dimension.

    Scores are decomposed rather than produced as a single number so that a bias
    effect can be *localised*: a name swap that moves only
    ``communication_clarity`` is a different finding from one that moves
    ``technical_skills``, and the first is far more likely to be a proxy effect
    than a substantive judgement.
    """

    name: str
    score: float
    weight: float = 1.0
    rationale: str = ""
    evidence: List[EvidenceSpan] = field(default_factory=list)


@dataclass
class ScreeningResult:
    """The structured verdict for one (resume, job) prompt at one sample index."""

    overall_score: float
    recommendation: str
    confidence: float = 0.0
    dimensions: List[DimensionScore] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    unverifiable_claims: List[str] = field(default_factory=list)
    summary: str = ""
    protected_attribute_referenced: bool = False
    protected_reference_note: str = ""
    parse_ok: bool = True
    parse_error: str = ""
    raw_response: str = ""

    def dimension_map(self) -> Dict[str, float]:
        return {d.name: d.score for d in self.dimensions}


@dataclass
class ScoreRecord:
    """One row of the tidy score table — the pipeline's central artifact.

    Everything the fairness layer needs is here, flat and CSV-friendly.  The
    hashes make each row independently verifiable: given the artifacts, a
    reviewer can recompute ``prompt_hash`` and confirm which exact text produced
    which exact score.
    """

    run_id: str
    job_id: str
    base_resume_id: str
    variant_id: str
    sample_index: int
    overall_score: float
    recommendation: str
    confidence: float

    # experimental design ---------------------------------------------------
    is_baseline: bool = False
    cue_families: str = ""            # comma-joined, CSV-friendly
    cue_ids: str = ""                 # "name=bm2004_w_f_01"
    arm: str = "baseline"             # which mitigation arm produced this row
    signals: Dict[str, str] = field(default_factory=dict)

    # rubric ----------------------------------------------------------------
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    missing_requirements: List[str] = field(default_factory=list)
    summary: str = ""
    rationale_text: str = ""

    # quality / validity ----------------------------------------------------
    evidence_support_rate: float = float("nan")
    protected_attribute_referenced: bool = False
    parse_ok: bool = True
    parse_error: str = ""

    # provenance ------------------------------------------------------------
    provider: str = ""
    model: str = ""
    temperature: float = 0.0
    prompt_hash: str = ""
    response_hash: str = ""
    resume_hash: str = ""
    job_hash: str = ""
    cached: bool = False
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""
    timestamp: str = ""

    # optional labels -------------------------------------------------------
    qualified_label: Optional[int] = None

    def flat(self) -> Dict[str, Any]:
        """Flatten nested fields into scalar columns for the CSV artifact."""
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("signals", "dimension_scores"):
                continue
            if isinstance(value, list):
                out[f.name] = "; ".join(str(v) for v in value)
            else:
                out[f.name] = value
        for attribute, level in self.signals.items():
            out[f"signal_{attribute}"] = level
        for dim, score in self.dimension_scores.items():
            out[f"dim_{dim}"] = score
        return out


# --------------------------------------------------------------------------- #
# fairness output
# --------------------------------------------------------------------------- #
@dataclass
class MetricResult:
    """One fairness number plus everything needed to interpret it.

    A bare point estimate is not a finding.  ``ci_low``/``ci_high``,
    ``p_value``, ``q_value`` (after multiplicity correction), ``effect_size``
    and ``n`` are carried alongside so the report can state whether a gap is
    distinguishable from noise, and the conflict analyser can compare metrics on
    equal terms.
    """

    name: str
    family: str
    attribute: str = ""
    value: float = float("nan")
    unit: str = "score_points"
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    p_value: float = float("nan")
    q_value: float = float("nan")
    effect_size: float = float("nan")
    effect_size_name: str = ""
    n: int = 0
    groups: Dict[str, float] = field(default_factory=dict)
    threshold: Optional[float] = None
    passed: Optional[bool] = None
    higher_is_better: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""

    @property
    def significant(self) -> bool:
        q = self.q_value if self.q_value == self.q_value else self.p_value
        return q == q and q < 0.05


@dataclass
class AuditReport:
    """The full multi-metric audit for one score table."""

    run_id: str
    created_at: str
    n_records: int
    n_base_resumes: int
    n_jobs: int
    n_variants: int
    arms: List[str] = field(default_factory=list)
    metrics: List[MetricResult] = field(default_factory=list)
    reliability: Dict[str, Any] = field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def by_family(self, family: str) -> List[MetricResult]:
        return [m for m in self.metrics if m.family == family]

    def get(self, name: str, attribute: str = "") -> Optional[MetricResult]:
        for m in self.metrics:
            if m.name == name and (not attribute or m.attribute == attribute):
                return m
        return None
