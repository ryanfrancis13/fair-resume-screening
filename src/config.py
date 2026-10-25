"""Typed, layered configuration.

Precedence, lowest to highest:

    dataclass defaults  <  YAML file  <  environment variables  <  --set overrides

Every run writes its *fully resolved* config into the run directory, so an
artifact folder is self-describing: the numbers in it can be traced to the exact
parameters that produced them without consulting shell history.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .utils.io import project_root, read_yaml


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
@dataclass
class LLMConfig:
    provider: str = "gemini"           # gemini | mock | openai
    model: str = "gemini-3.6-flash"
    temperature: float = 0.0
    top_p: float = 0.95
    max_output_tokens: int = 2048
    #: Decoding repeats of the identical prompt.  ``>1`` is what makes the
    #: model's own noise floor measurable, so that a cue effect can be compared
    #: against it instead of being reported in a vacuum.
    self_consistency_samples: int = 1
    #: Temperature used for repeats when ``self_consistency_samples > 1``; a
    #: greedy decode would make every repeat identical and the noise floor zero.
    self_consistency_temperature: float = 0.4
    max_retries: int = 5
    retry_base_delay: float = 1.0
    retry_max_delay: float = 32.0
    requests_per_minute: int = 12       # Gemini free-tier flash headroom
    max_concurrency: int = 4
    timeout_s: float = 90.0
    cache_enabled: bool = True
    cache_dir: str = ".frs_cache"
    fail_fast: bool = False
    #: Mock provider only: additive score deltas keyed by attribute → level.
    #: Used to validate that the audit recovers a bias of known size.
    mock_bias: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mock_noise_sd: float = 1.5


@dataclass
class DataConfig:
    corpus: str = "sample"              # sample | kaggle | hf_fit | dir
    resumes_path: str = "data/sample/resumes"
    jobs_path: str = "data/sample/jobs"
    pairs_path: str = "data/sample/pairs.json"
    kaggle_csv: str = "data/raw/UpdatedResumeDataSet.csv"
    #: cnamuangtoun/resume-job-description-fit (Hugging Face). Real resume +
    #: real job description + a Good/Potential/No Fit label per row. 642 unique
    #: resumes and 280 unique job descriptions across 8,000 rows — each resume
    #: is reused across several job pairings, so n_pairs is not n_resumes.
    hf_fit_train_csv: str = "data/raw/hf_fit_train.csv"
    hf_fit_test_csv: str = "data/raw/hf_fit_test.csv"
    hf_fit_use_test_split: bool = False  # include test.csv rows too, or train only
    max_resumes: Optional[int] = None
    max_jobs: Optional[int] = None
    max_resume_chars: int = 12000
    max_job_chars: int = 6000
    #: Cross every resume with every job in its family, rather than using the
    #: curated pairs file.  Larger design, more API calls.
    cross_within_family: bool = False


@dataclass
class ScreeningConfig:
    rubric: str = "default"
    require_evidence: bool = True
    #: Minimum fraction of evidence quotes that must be locatable in the resume
    #: for a score to be accepted.  Ungrounded scores are re-requested once and
    #: then flagged, because an unfalsifiable rationale cannot be audited.
    min_evidence_support_rate: float = 0.5
    debias_instruction: bool = False
    anonymize: bool = False
    anonymize_level: str = "identity"   # identity | identity_plus_proxy | aggressive
    advance_threshold: float = 70.0
    borderline_threshold: float = 55.0
    top_k: int = 5


@dataclass
class PerturbationConfig:
    #: Cue families to vary.  ``name`` alone reproduces the classic
    #: Bertrand & Mullainathan design; the others are the project's extension
    #: past names, which is where the stated research gap lives.
    cue_families: List[str] = field(
        default_factory=lambda: [
            "name",
            "gender_marker",
            "cultural_affiliation",
            "nationality_region",
            "age_dates",
            "disability_marker",
            "socioeconomic_marker",
            # A validity probe, not a bias probe: it confirms the scorer responds
            # to legitimately job-relevant signal. Without it, an apparently
            # perfect fairness result cannot be distinguished from an inert scorer.
            "education_prestige",
            "writing_register",
        ]
    )
    name_list: str = "bertrand_mullainathan_2004"
    #: Cue levels per family per base resume.  ``None`` means "all levels".
    levels_per_family: Optional[int] = None
    #: ``single`` varies one family at a time (clean main effects).
    #: ``factorial_name_x_gender`` additionally crosses name × gender marker to
    #: estimate an interaction, which is where intersectional effects appear.
    design: str = "single"
    include_baseline: bool = True
    validate_variants: bool = True
    max_unintended_edit_ratio: float = 0.02
    require_skill_preservation: bool = True
    seed: int = 20250831


@dataclass
class FairnessConfig:
    attributes: List[str] = field(
        default_factory=lambda: [
            "gender",
            "race_ethnicity",
            "nationality_region",
            "age_band",
            "disability_status",
            "socioeconomic_background",
        ]
    )
    reference_levels: Dict[str, str] = field(
        default_factory=lambda: {"gender": "male", "race_ethnicity": "White"}
    )
    top_k: List[int] = field(default_factory=lambda: [1, 3, 5])
    alpha: float = 0.05
    #: Benjamini-Hochberg controls the false discovery rate across the many
    #: group×attribute×metric tests this suite runs; without it, a suite this
    #: wide is guaranteed to produce "significant" noise.
    multiple_comparison_method: str = "fdr_bh"
    bootstrap_iterations: int = 2000
    permutation_iterations: int = 5000
    #: Practical-significance floor in score points.  A statistically detectable
    #: 0.2-point gap on a 100-point scale is not an actionable finding.
    min_practical_effect: float = 1.0
    #: Adverse-impact ratio floor (EEOC "four-fifths rule").
    impact_ratio_threshold: float = 0.8
    counterfactual_tolerance: float = 2.0
    enable_rationale_audit: bool = True
    enable_calibration_audit: bool = True
    intersectional_attributes: List[str] = field(
        default_factory=lambda: ["gender", "race_ethnicity"]
    )


@dataclass
class MitigationConfig:
    strategies: List[str] = field(
        default_factory=lambda: [
            "blind_screening",
            "counterfactual_marginalization",
            "sensitivity_gated_review",
        ]
    )
    marginalization_variants: int = 4
    marginalization_estimator: str = "trimmed_mean"   # mean | median | trimmed_mean
    trim_fraction: float = 0.25
    review_sensitivity_threshold: float = 3.0
    max_review_fraction: float = 0.3
    cue_effect_source_run: Optional[str] = None


@dataclass
class ReportConfig:
    formats: List[str] = field(default_factory=lambda: ["html", "markdown", "json"])
    make_plots: bool = True
    plot_dpi: int = 150
    explanation_cards: bool = True
    max_cards: int = 25
    model_card: bool = True


@dataclass
class Config:
    run_name: str = "default"
    seed: int = 20250831
    output_dir: str = "runs"
    log_level: str = "INFO"
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        overrides: Optional[Sequence[str]] = None,
        *,
        apply_env: bool = True,
    ) -> Config:
        cfg = cls()
        if path is not None:
            cfg = _merge_into(cfg, read_yaml(Path(path)))
        if apply_env:
            cfg = _apply_env(cfg)
        for item in overrides or []:
            key, _, value = item.partition("=")
            if not _:
                raise ValueError(f"override must look like key=value, got {item!r}")
            cfg.set_path(key.strip(), _coerce_scalar(value.strip()))
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------ #
    # dotted access
    # ------------------------------------------------------------------ #
    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        target: Any = self
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise KeyError(f"unknown config section {part!r} in {dotted!r}")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise KeyError(f"unknown config key {dotted!r}")
        current = getattr(target, leaf)
        if isinstance(current, list) and isinstance(value, str):
            # `--set fairness.top_k=1,3,5` should yield [1, 3, 5], not ["1", "3", "5"].
            value = [_coerce_scalar(v.strip()) for v in value.split(",") if v.strip()]
        if isinstance(current, dict) and not isinstance(value, dict) and value is not None:
            raise ValueError(
                f"{dotted} expects a mapping, got {type(value).__name__} ({value!r}). "
                "Pass it as JSON, quoted for the shell: "
                f"--set '{dotted}=" + '{"key": {"level": -3.0}}\''
            )
        setattr(target, leaf, value)

    def get_path(self, dotted: str) -> Any:
        target: Any = self
        for part in dotted.split("."):
            target = getattr(target, part)
        return target

    # ------------------------------------------------------------------ #
    # paths
    # ------------------------------------------------------------------ #
    @property
    def root(self) -> Path:
        return project_root()

    def resolve(self, relative: str) -> Path:
        p = Path(relative).expanduser()
        return p if p.is_absolute() else (self.root / p)

    def run_dir(self, run_id: str) -> Path:
        return self.resolve(self.output_dir) / run_id

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        errors: List[str] = []
        # Imported here because llm.client itself imports this module.
        from .llm.client import PROVIDERS

        if self.llm.provider.lower() not in PROVIDERS:
            errors.append(
                f"llm.provider {self.llm.provider!r} is not registered "
                f"(available: {', '.join(sorted(PROVIDERS))})"
            )
        if self.llm.self_consistency_samples < 1:
            errors.append("llm.self_consistency_samples must be >= 1")
        if not 0.0 <= self.llm.temperature <= 2.0:
            errors.append("llm.temperature must be within [0, 2]")
        if self.screening.borderline_threshold > self.screening.advance_threshold:
            errors.append("screening.borderline_threshold must be <= advance_threshold")
        if self.mitigation.marginalization_estimator not in ("mean", "median", "trimmed_mean"):
            errors.append("mitigation.marginalization_estimator is invalid")
        if not 0.0 < self.mitigation.trim_fraction < 0.5:
            errors.append("mitigation.trim_fraction must be within (0, 0.5)")
        if self.perturbation.design not in ("single", "factorial_name_x_gender", "full_factorial"):
            errors.append(f"perturbation.design {self.perturbation.design!r} is invalid")
        if self.fairness.bootstrap_iterations < 200:
            errors.append("fairness.bootstrap_iterations < 200 gives unusable intervals")
        if self.llm.mock_bias:
            if not isinstance(self.llm.mock_bias, dict):
                errors.append(
                    "llm.mock_bias must be a mapping attribute -> {level: delta}, got "
                    f"{type(self.llm.mock_bias).__name__}"
                )
            else:
                for attribute, levels in self.llm.mock_bias.items():
                    if not isinstance(levels, dict):
                        errors.append(
                            f"llm.mock_bias[{attribute!r}] must be a mapping level -> delta, "
                            f"got {type(levels).__name__}"
                        )
                        continue
                    for level, delta in levels.items():
                        if not isinstance(delta, (int, float)) or isinstance(delta, bool):
                            errors.append(
                                f"llm.mock_bias[{attribute!r}][{level!r}] must be a number, "
                                f"got {type(delta).__name__}"
                            )
        if errors:
            raise ValueError("invalid configuration:\n  - " + "\n  - ".join(errors))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _coerce_scalar(value: str) -> Any:
    """Parse a command-line override value into a Python value.

    JSON objects and arrays are parsed, so a nested setting can be given on the
    command line at all::

        --set 'llm.mock_bias={"gender": {"female": -3.0}}'

    Without this the value arrives as a string, and the first thing that treats it
    as a mapping fails with an ``AttributeError`` far from the cause.
    """
    stripped = value.strip()
    if stripped[:1] in ("{", "[") and stripped[-1:] in ("}", "]"):
        import json

        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"override value looks like JSON but does not parse: {value!r} ({exc.msg}). "
                'Remember to quote it for the shell, e.g. --set \'llm.mock_bias={"gender": '
                '{"female": -3.0}}\''
            ) from exc

    low = stripped.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null", ""):
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return value


def _merge_into(obj: Any, data: Dict[str, Any]) -> Any:
    """Recursively overlay a plain dict onto a dataclass instance."""
    if not is_dataclass(obj):
        return data
    known = {f.name: f for f in fields(obj)}
    for key, value in (data or {}).items():
        if key not in known:
            raise KeyError(f"unknown config key {key!r} (expected one of {sorted(known)})")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            setattr(obj, key, _merge_into(current, value))
        else:
            setattr(obj, key, value)
    return obj


_ENV_MAP = {
    "FRS_PROVIDER": "llm.provider",
    "FRS_MODEL": "llm.model",
    "FRS_TEMPERATURE": "llm.temperature",
    "FRS_CACHE_DIR": "llm.cache_dir",
    "FRS_RPM": "llm.requests_per_minute",
    "FRS_CONCURRENCY": "llm.max_concurrency",
    "FRS_LOG_LEVEL": "log_level",
    "FRS_SEED": "seed",
    "FRS_OUTPUT_DIR": "output_dir",
}


def _apply_env(cfg: Config) -> Config:
    for env_key, dotted in _ENV_MAP.items():
        raw = os.environ.get(env_key)
        if raw is not None and raw != "":
            cfg.set_path(dotted, _coerce_scalar(raw))
    return cfg


def load_dotenv_if_present() -> None:
    """Load ``.env`` from the project root, if python-dotenv is installed.

    API keys are read from the environment only; the loader never returns them
    and nothing in the pipeline writes them to an artifact.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional at runtime
        return
    env_path = project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)