"""Core schemas, configuration, and benchmark corpus loader."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class Config:
    run_name: str = "default"
    seed: int = 20250831
    output_dir: str = "runs"
    provider: str = field(default_factory=lambda: os.getenv("FRS_PROVIDER", "gemini"))
    model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    temperature: float = 0.0
    self_consistency_samples: int = 1
    self_consistency_temperature: float = 0.4
    cache_enabled: bool = True
    cache_dir: str = ".frs_cache"
    data_dir: str = "data/benchmark"
    max_resumes: int | None = 12
    max_jobs: int | None = 6
    max_variants_per_resume: int | None = 12
    advance_threshold: float = 75.0
    borderline_threshold: float = 50.0
    top_k: int = 5
    alpha: float = 0.05
    min_practical_effect: float = 1.0
    impact_ratio_threshold: float = 0.8
    counterfactual_tolerance: float = 2.0
    mock_bias: dict[str, dict[str, float]] = field(default_factory=dict)
    mock_noise_sd: float = 1.5

    def resolve(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else project_root() / p


@dataclass
class Resume:
    resume_id: str
    text: str
    job_family: str = "software_engineering"
    source: str = "india_synth_v1"
    template: str | None = None
    slots: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobDescription:
    job_id: str
    title: str
    text: str
    job_family: str = "software_engineering"
    must_have: list[str] = field(default_factory=list)
    nice_to_have: list[str] = field(default_factory=list)
    min_years_experience: float = 0.0
    source: str = "india_synth_v1"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Pair:
    pair_id: str
    job_id: str
    resume_id: str
    expected_fit: str = "moderate"
    qualified_label: int = 1
    pairing_rationale: str = "same job family"


@dataclass
class ResumeVariant:
    variant_id: str
    base_resume_id: str
    cue_family: str
    cue_type: str  # protected | confounded | validity | validity_proxy | baseline
    attribute: str
    group_label: str
    intersectional_label: str
    text: str
    signals: dict[str, str] = field(default_factory=dict)
    substituted_values: dict[str, Any] = field(default_factory=dict)
    validation_passed: bool = True


@dataclass
class DimensionScore:
    dimension: str
    score: int  # 0..5
    weight: float
    rationale: str
    evidence_quotes: list[str] = field(default_factory=list)
    evidence_verified: bool = True


@dataclass
class ScreeningResult:
    resume_id: str
    variant_id: str
    job_id: str
    arm: str
    sample_index: int
    overall_score: float  # 0..100
    decision: str  # shortlist | hold | reject
    summary: str
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    dimension_scores: list[DimensionScore] = field(default_factory=list)
    evidence_support_rate: float = 1.0
    grounding_retried: bool = False
    parse_valid: bool = True
    parse_repaired: bool = False
    provider: str = "mock"
    model: str = "mock-rubric-v1"
    prompt_version: str = "v1"
    latency_ms: float = 0.0
    cached: bool = False


@dataclass
class ScoreRecord:
    pair_id: str
    resume_id: str
    variant_id: str
    job_id: str
    arm: str
    sample_index: int
    cue_family: str
    cue_type: str
    attribute: str
    group_label: str
    intersectional_label: str
    expected_fit: str
    qualified_label: int
    overall_score: float
    decision: str
    dim_technical_skills: int = 0
    dim_relevant_experience: int = 0
    dim_domain_relevance: int = 0
    dim_education_credentials: int = 0
    dim_impact_and_outcomes: int = 0
    dim_communication_clarity: int = 0
    rationale_text: str = ""
    evidence_support_rate: float = 1.0
    parse_valid: bool = True
    provider: str = "mock"
    model: str = "mock-rubric-v1"


@dataclass
class MetricResult:
    name: str
    family: str
    attribute: str
    group_a: str
    group_b: str
    value: float
    ci_low: float = 0.0
    ci_high: float = 0.0
    p_value: float | None = None
    p_value_adjusted: float | None = None
    effect_size: float = 0.0
    n_blocks: int = 0
    n_observations: int = 0
    minimum_detectable_effect: float = 0.0
    passed: bool = True
    threshold: float = 0.0
    interpretation: str = ""


@dataclass
class AuditReport:
    run_id: str
    created_at: str
    provider: str
    model: str
    n_resumes: int
    n_jobs: int
    n_variants: int
    n_scores: int
    metrics: list[MetricResult] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    mitigation_summary: list[dict[str, Any]] = field(default_factory=list)
    leakage_summary: list[dict[str, Any]] = field(default_factory=list)
    reliability: dict[str, Any] = field(default_factory=dict)
    headline_findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CUE_TYPE_MAP = {
    "education_prestige": "validity",
    "writing_register": "validity_proxy",
    "age_marker": "confounded",
}


def _primary_attribute_and_group(family: str, cue_id: str, signals: dict[str, str]) -> tuple[str, str, str]:
    if family in ("name", "pronoun") and "gender" in signals:
        attr = "gender"
        group = signals["gender"]
    elif signals:
        attr, group = next(iter(signals.items()))
    else:
        attr = family
        group = cue_id.split("=", 1)[-1] if "=" in cue_id else cue_id
    inter = "_".join(f"{k}:{v}" for k, v in sorted(signals.items())) if len(signals) > 1 else f"{attr}:{group}"
    return attr, group, inter


def load_benchmark_corpus(cfg: Config) -> tuple[list[Resume], list[JobDescription], list[Pair], list[ResumeVariant]]:
    base = cfg.resolve(cfg.data_dir)
    resumes: list[Resume] = []
    for f in sorted((base / "resumes").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        resumes.append(
            Resume(
                resume_id=d["resume_id"],
                text=d["text"].strip(),
                job_family=d.get("job_family", "software_engineering"),
                source=d.get("source", "india_synth_v1"),
                template=d.get("template"),
                slots=d.get("baseline_slots", {}),
                meta=d.get("meta", {}),
            )
        )
    if cfg.max_resumes:
        resumes = resumes[: cfg.max_resumes]
    resume_ids = {r.resume_id for r in resumes}

    jobs: list[JobDescription] = []
    for f in sorted((base / "jobs").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        jobs.append(
            JobDescription(
                job_id=d["job_id"],
                title=d.get("title", d["job_id"]),
                text=d["text"].strip(),
                job_family=d.get("job_family", "software_engineering"),
                must_have=d.get("must_have", []),
                nice_to_have=d.get("nice_to_have", []),
                min_years_experience=float(d.get("min_years_experience", 0)),
                source=d.get("source", "india_synth_v1"),
                meta=d.get("meta", {}),
            )
        )
    if cfg.max_jobs:
        jobs = jobs[: cfg.max_jobs]
    job_ids = {j.job_id for j in jobs}

    raw_pairs = json.loads((base / "pairs.json").read_text(encoding="utf-8"))
    pairs = [
        Pair(
            pair_id=p["pair_id"],
            job_id=p["job_id"],
            resume_id=p["resume_id"],
            expected_fit=p.get("expected_fit", "moderate"),
            qualified_label=int(p.get("qualified_label", 1)),
            pairing_rationale=p.get("pairing_rationale", "same job family"),
        )
        for p in raw_pairs
        if p["resume_id"] in resume_ids and p["job_id"] in job_ids
    ]
    active_resumes = {p.resume_id for p in pairs}
    resumes = [r for r in resumes if r.resume_id in active_resumes]

    variants: list[ResumeVariant] = [
        ResumeVariant(
            variant_id=f"{r.resume_id}__baseline",
            base_resume_id=r.resume_id,
            cue_family="baseline",
            cue_type="baseline",
            attribute="baseline",
            group_label="baseline",
            intersectional_label="baseline",
            text=r.text,
            signals={},
            substituted_values=r.slots,
            validation_passed=True,
        )
        for r in resumes
    ]

    variants_path = base / "variants" / "variants.jsonl"
    if variants_path.exists():
        per_group_counts: dict[tuple[str, str, str], int] = {}
        per_family_counts: dict[tuple[str, str], int] = {}
        per_family_cap = max(2, cfg.max_variants_per_resume // 8) if cfg.max_variants_per_resume else None
        per_group_cap = max(1, cfg.max_variants_per_resume // 24) if cfg.max_variants_per_resume else None
        with variants_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                rid = row["resume_id"]
                if rid not in active_resumes:
                    continue
                family = row["family"]
                if per_family_cap and per_family_counts.get((rid, family), 0) >= per_family_cap:
                    continue
                signals = row.get("signals", {})
                attr, group, inter = _primary_attribute_and_group(family, row["cue_id"], signals)
                key = (rid, family, group)
                count = per_group_counts.get(key, 0)
                if per_group_cap and count >= per_group_cap:
                    continue
                variants.append(
                    ResumeVariant(
                        variant_id=row["variant_id"],
                        base_resume_id=rid,
                        cue_family=family,
                        cue_type=CUE_TYPE_MAP.get(family, "protected"),
                        attribute=attr,
                        group_label=group,
                        intersectional_label=inter,
                        text=row["text"],
                        signals=signals,
                        substituted_values=row.get("edits", {}),
                        validation_passed=bool(row.get("validation", {}).get("passed", True)),
                    )
                )
                per_group_counts[key] = count + 1
                per_family_counts[(rid, family)] = per_family_counts.get((rid, family), 0) + 1

    return resumes, jobs, pairs, variants
