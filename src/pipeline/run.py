"""End-to-end orchestration.

The stages map one-to-one onto the project's phased build plan, and the ordering
is not cosmetic — each stage's output is the next stage's precondition:

===== ==================== =================================================
phase stage                gate before proceeding
===== ==================== =================================================
1     ``prepare_data``     corpus loads, pairs are non-random, no duplicates
2     ``score_baseline``   sanity check: strong fits outscore weak fits
3     ``generate_variants``perturbation validator passes on the variant set
3     ``score_variants``   attrition is recorded, not silently absorbed
4     ``audit``            noise floor known before bias effects are believed
4/6   ``mitigate``         fairness gain reported with its utility cost
5     ``report``           narrative, plots, cards, model card
===== ==================== =================================================

The Phase-2 gate is the load-bearing one, and it is enforced rather than
suggested: if the scorer cannot separate a strong fit from a weak one, the
pipeline records a warning that propagates into the audit report and the HTML
report, because a bias number from a broken matcher is noise wearing a
conclusion's clothes.

Stages are individually callable, so a re-audit or a re-report costs nothing —
the response cache and the on-disk score table mean re-analysis never re-scores.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..config import Config
from ..logging_setup import log_event, setup_logging
from ..schemas import JobDescription, Pair, Resume, ResumeVariant, ScoreRecord
from ..utils.io import read_jsonl, write_json
from ..utils.seeds import rng as make_rng
from ..utils.seeds import seed_everything
from . import artifacts as art

logger = logging.getLogger("frs.pipeline")


@dataclass
class PipelineState:
    """Everything a stage may need from the stages before it."""

    cfg: Config
    run_id: str
    root: Path
    manifest: art.RunManifest
    resumes: List[Resume] = field(default_factory=list)
    jobs: List[JobDescription] = field(default_factory=list)
    pairs: List[Pair] = field(default_factory=list)
    variants: List[ResumeVariant] = field(default_factory=list)
    variant_skips: List[Dict[str, Any]] = field(default_factory=list)
    records_by_arm: Dict[str, List[ScoreRecord]] = field(default_factory=dict)
    frames_by_arm: Dict[str, pd.DataFrame] = field(default_factory=dict)
    audit_report: Any = None
    mitigation: Optional[Dict[str, Any]] = None
    sanity: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def all_records(self) -> List[ScoreRecord]:
        out: List[ScoreRecord] = []
        for records in self.records_by_arm.values():
            out.extend(records)
        return out

    def combined_frame(self) -> pd.DataFrame:
        frames = [f for f in self.frames_by_arm.values() if f is not None and not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fit_labels(self) -> Dict[tuple, str]:
        return {
            (p.job_id, p.resume_id): p.expected_fit
            for p in self.pairs
            if p.expected_fit
        }


def start_run(
    cfg: Config, *, run_id: Optional[str] = None, prefix: Optional[str] = None
) -> PipelineState:
    """Create the run directory, configure logging, snapshot the config."""
    seed_everything(cfg.seed)
    resolved = run_id or art.new_run_id(cfg, prefix)
    root = art.prepare_run_dir(cfg, resolved)
    setup_logging(cfg.log_level, jsonl_path=root / "logs" / "run.jsonl")
    manifest = art.RunManifest(cfg, resolved, root)
    art.write_config_snapshot(root, cfg)
    log_event(
        logger, logging.INFO, "run started",
        run_id=resolved, provider=cfg.llm.provider, model=cfg.llm.model, seed=cfg.seed,
    )
    return PipelineState(cfg=cfg, run_id=resolved, root=root, manifest=manifest)


# --------------------------------------------------------------------------- #
# Phase 1
# --------------------------------------------------------------------------- #
def prepare_data(state: PipelineState) -> PipelineState:
    from ..data import (
        clean_jobs,
        clean_resumes,
        corpus_stats,
        duplicate_report,
        load_corpus,
        pairing_report,
    )

    started = time.perf_counter()
    resumes, jobs, pairs = load_corpus(state.cfg)
    resumes, dropped = clean_resumes(resumes)
    jobs = clean_jobs(jobs)
    kept_ids = {r.resume_id for r in resumes}
    pairs = [p for p in pairs if p.resume_id in kept_ids]

    duplicates = duplicate_report(resumes)
    if duplicates["n_exact_duplicate_groups"] or duplicates["n_near_duplicate_pairs"]:
        message = (
            f"corpus contains {duplicates['n_exact_duplicate_groups']} exact and "
            f"{duplicates['n_near_duplicate_pairs']} near-duplicate resume group(s); "
            "duplicates inflate the apparent number of statistical blocks"
        )
        state.warnings.append(message)
        state.manifest.warn(message)

    state.resumes, state.jobs, state.pairs = resumes, jobs, pairs
    art.write_corpus(state.root, resumes, jobs, pairs)

    stats = corpus_stats(resumes, jobs, pairs)
    state.manifest.record(
        "prepare_data",
        {
            "elapsed_s": round(time.perf_counter() - started, 2),
            "corpus_stats": stats,
            "pairing": pairing_report(pairs, resumes, jobs),
            "duplicates": duplicates,
            "dropped_resumes": dropped,
            "fingerprint": art.corpus_fingerprint(resumes, jobs, pairs),
        },
    )
    log_event(
        logger, logging.INFO, "data prepared",
        resumes=len(resumes), jobs=len(jobs), pairs=len(pairs), dropped=len(dropped),
    )
    return state


# --------------------------------------------------------------------------- #
# Phase 3 (variants generated before scoring so the design is fixed up front)
# --------------------------------------------------------------------------- #
def generate_variants(state: PipelineState) -> PipelineState:
    from ..perturb.cues import build_registry, registry_summary
    from ..perturb.generator import generate_variants as gen
    from ..perturb.validator import summarise_validation

    cfg = state.cfg
    started = time.perf_counter()
    rng = make_rng(cfg.perturbation.seed, "perturb:registry")

    registry = build_registry(
        families=cfg.perturbation.cue_families,
        name_list_id=cfg.perturbation.name_list,
        name_per_cell=None,
        rng=rng,
    )

    variants: List[ResumeVariant] = []
    skips: List[Dict[str, Any]] = []
    for resume in state.resumes:
        resume_rng = make_rng(cfg.perturbation.seed, f"perturb:{resume.resume_id}")
        produced, resume_skips = gen(
            resume,
            registry,
            design=cfg.perturbation.design,
            levels_per_family=cfg.perturbation.levels_per_family,
            include_baseline=cfg.perturbation.include_baseline,
            validate=cfg.perturbation.validate_variants,
            max_unintended_edit_ratio=cfg.perturbation.max_unintended_edit_ratio,
            require_skill_preservation=cfg.perturbation.require_skill_preservation,
            rng=resume_rng,
        )
        variants.extend(produced)
        skips.extend(resume_skips)

    state.variants = variants
    state.variant_skips = skips
    art.write_variants(state.root, variants, skips)

    validation = summarise_validation(variants)
    if validation["pass_rate"] < 0.95:
        message = (
            f"only {100 * validation['pass_rate']:.1f}% of variants passed content-preservation "
            "validation; failing variants are excluded from scoring and their absence is "
            "reported rather than hidden"
        )
        state.warnings.append(message)
        state.manifest.warn(message)

    state.manifest.record(
        "generate_variants",
        {
            "elapsed_s": round(time.perf_counter() - started, 2),
            "design": cfg.perturbation.design,
            "name_list": cfg.perturbation.name_list,
            "registry": registry_summary(registry),
            "n_variants": len(variants),
            "n_skips": len(skips),
            "validation": validation,
        },
    )
    log_event(
        logger, logging.INFO, "variants generated",
        n_variants=len(variants), n_skipped=len(skips),
        pass_rate=round(validation["pass_rate"], 4),
    )
    return state


# --------------------------------------------------------------------------- #
# Phase 2 + 3 scoring
# --------------------------------------------------------------------------- #
def score_arm(
    state: PipelineState,
    *,
    arm: str = "baseline",
    overrides: Optional[Dict[str, Any]] = None,
    baseline_only: bool = False,
    dry_run: bool = False,
) -> Tuple[List[ScoreRecord], Dict[str, Any]]:
    """Score one arm of the experiment."""
    from ..llm.client import LLMClient
    from ..screening import ResumeScorer, build_items, estimate_cost, run_batch

    cfg = state.cfg
    if overrides:
        # Arms differ only in the named parameters; everything else is held
        # fixed so a between-arm difference is attributable to the intervention.
        import copy

        cfg = copy.deepcopy(cfg)
        for key, value in overrides.items():
            cfg.set_path(key, value)

    variants = state.variants
    if baseline_only:
        variants = [v for v in variants if v.is_baseline]
    if not variants:
        return [], {"error": "no variants to score"}

    items = build_items(
        state.jobs, variants, state.pairs,
        n_samples=cfg.llm.self_consistency_samples, arm=arm,
    )
    # The perturbation layer records the original name so blind screening can
    # redact it exactly rather than relying on header detection.
    name_by_resume = {r.resume_id: r.slots.get("NAME") for r in state.resumes}
    for item in items:
        item.resume_name = name_by_resume.get(item.variant.base_resume_id)

    cost = estimate_cost(items, cfg)
    log_event(logger, logging.INFO, "planned scoring arm", arm=arm, **cost)
    if dry_run:
        return [], {"dry_run": True, **cost}

    client = LLMClient.from_config(cfg)
    try:
        scorer = ResumeScorer(client, cfg)
        records, telemetry = run_batch(
            scorer, items, cfg, run_id=state.run_id,
            progress=_progress_logger(arm, len(items)),
        )
    finally:
        client.close()

    art.write_scores(state.root, records, arm=arm)
    telemetry["cost_estimate"] = cost
    telemetry["arm"] = arm
    telemetry["overrides"] = overrides or {}
    return records, telemetry


def _progress_logger(arm: str, total: int):
    step = max(1, total // 10)

    def report(done: int, _total: int) -> None:
        if done % step == 0 or done == total:
            log_event(logger, logging.INFO, "scoring progress",
                      arm=arm, done=done, total=total,
                      pct=round(100 * done / max(1, total), 1))

    return report


def score_baseline(state: PipelineState, *, dry_run: bool = False) -> PipelineState:
    """Phase 2: score the unperturbed resumes and run the sanity gate."""
    from ..fairness.suite import records_to_frame
    from ..screening import sanity_check

    started = time.perf_counter()
    records, telemetry = score_arm(
        state, arm="baseline", baseline_only=True, dry_run=dry_run
    )
    if dry_run:
        state.manifest.record("score_baseline", {"dry_run": True, **telemetry})
        return state

    state.sanity = sanity_check(records, state.pairs)
    if state.sanity.get("verdict") == "review":
        message = (
            "PHASE-2 GATE: the scorer does not clearly separate strong from weak fits "
            f"(kendall tau = {state.sanity.get('kendall_tau')}, strong-weak gap = "
            f"{state.sanity.get('strong_minus_weak_gap')}). Downstream bias estimates may "
            "reflect a noisy matcher rather than model behaviour."
        )
        state.warnings.append(message)
        state.manifest.warn(message)
    elif state.sanity.get("verdict") == "insufficient_data":
        message = (
            "PHASE-2 GATE: not enough fit-graded pairs to test scorer discrimination. "
            "Add expected_fit labels to the pairs file."
        )
        state.warnings.append(message)
        state.manifest.warn(message)

    state.records_by_arm["baseline_unperturbed"] = records
    state.frames_by_arm["baseline_unperturbed"] = records_to_frame(records)
    state.manifest.record(
        "score_baseline",
        {
            "elapsed_s": round(time.perf_counter() - started, 2),
            "telemetry": telemetry,
            "sanity_check": state.sanity,
        },
    )
    log_event(
        logger, logging.INFO, "baseline scoring complete",
        n_records=len(records), sanity=state.sanity.get("verdict"),
    )
    return state


def score_variants(state: PipelineState, *, dry_run: bool = False) -> PipelineState:
    """Phase 3: score every validated variant through the identical pipeline."""
    from ..fairness.suite import records_to_frame

    started = time.perf_counter()
    records, telemetry = score_arm(state, arm="baseline", dry_run=dry_run)
    if dry_run:
        state.manifest.record("score_variants", {"dry_run": True, **telemetry})
        return state

    state.records_by_arm["baseline"] = records
    state.frames_by_arm["baseline"] = records_to_frame(records)

    n_failed = telemetry.get("n_parse_failures", 0)
    if records and n_failed / len(records) > 0.05:
        message = (
            f"{n_failed}/{len(records)} scores failed to parse "
            f"({100 * n_failed / len(records):.1f}%). Check whether failures correlate with "
            "a demographic condition — differential attrition biases every comparison."
        )
        state.warnings.append(message)
        state.manifest.warn(message)

    state.manifest.record(
        "score_variants",
        {"elapsed_s": round(time.perf_counter() - started, 2), "telemetry": telemetry},
    )
    return state


# --------------------------------------------------------------------------- #
# Phase 4
# --------------------------------------------------------------------------- #
def audit(state: PipelineState, *, arm: str = "baseline") -> PipelineState:
    from ..fairness.suite import audit_summary, extras_for_report, metrics_table, run_audit

    started = time.perf_counter()
    frame = state.frames_by_arm.get(arm)
    if frame is None or frame.empty:
        raise ValueError(f"no scores for arm {arm!r}; run scoring first")

    report = run_audit(
        frame, state.cfg, run_id=state.run_id,
        provenance={
            "sanity_check": state.sanity,
            "pipeline_warnings": state.warnings,
            "variant_skips": len(state.variant_skips),
        },
    )
    # Pipeline-level warnings belong in the audit report too: a reader of
    # audit_report.json alone must still see that the Phase-2 gate flagged.
    for message in state.warnings:
        if message not in report.warnings:
            report.warnings.append(message)

    state.audit_report = report
    summary = audit_summary(report)
    art.write_audit(state.root, report, metrics_table(report), summary)
    write_json(state.root / "audit" / "extras.json", extras_for_report(frame, state.cfg))

    state.manifest.record(
        "audit",
        {"elapsed_s": round(time.perf_counter() - started, 2), "summary": summary},
    )
    log_event(
        logger, logging.INFO, "audit written",
        n_metrics=summary["n_metrics"], n_failed=summary["n_failed"],
        n_conflicts=summary["conflicts"]["n_conflicts"],
    )
    return state


# --------------------------------------------------------------------------- #
# Phase 4/6: mitigation
# --------------------------------------------------------------------------- #
def mitigate(state: PipelineState, *, dry_run: bool = False) -> PipelineState:
    from ..fairness.suite import records_to_frame
    from ..mitigate import (
        POST_HOC_REGISTRY,
        PRE_INFERENCE_STRATEGIES,
        estimate_cue_effects,
        evaluate_arms,
        mitigation_table,
        pre_inference_arms,
        utility_table,
    )

    cfg = state.cfg
    started = time.perf_counter()
    requested = list(cfg.mitigation.strategies)
    if not requested:
        state.manifest.record("mitigate", {"skipped": "no strategies configured"})
        return state

    baseline_frame = state.frames_by_arm.get("baseline")
    if baseline_frame is None or baseline_frame.empty:
        raise ValueError("mitigation needs baseline variant scores; run score_variants first")

    frames: Dict[str, pd.DataFrame] = {"baseline": baseline_frame}
    telemetry: Dict[str, Any] = {}

    # ---- pre-inference arms (extra scoring passes) ------------------------
    for spec in pre_inference_arms(cfg, [s for s in requested if s in PRE_INFERENCE_STRATEGIES]):
        records, arm_telemetry = score_arm(
            state, arm=spec.name, overrides=spec.overrides, dry_run=dry_run
        )
        telemetry[spec.name] = {"description": spec.description, **arm_telemetry}
        if dry_run or not records:
            continue
        state.records_by_arm[spec.name] = records
        frames[spec.name] = records_to_frame(records)

    if dry_run:
        state.manifest.record("mitigate", {"dry_run": True, "arms": telemetry})
        return state

    # ---- post-hoc arms (no extra API calls) ------------------------------
    cue_effects = estimate_cue_effects(baseline_frame)
    for strategy in [s for s in requested if s in POST_HOC_REGISTRY]:
        function = POST_HOC_REGISTRY[strategy]
        try:
            if strategy == "counterfactual_marginalization":
                produced = function(baseline_frame, cfg)
            elif strategy == "cue_effect_debiasing":
                source = _load_external_effects(cfg) or cue_effects
                produced = function(baseline_frame, cfg, effects=source)
            else:
                outcome = function(baseline_frame, cfg)
                if isinstance(outcome, dict):
                    telemetry[strategy] = {
                        k: v for k, v in outcome.items() if k != "frame"
                    }
                    produced = outcome.get("frame")
                    if "error" in outcome:
                        produced = None
                else:
                    produced = outcome
        except Exception as exc:  # noqa: BLE001 - one arm must not kill the stage
            log_event(logger, logging.ERROR, "mitigation arm failed",
                      strategy=strategy, error=f"{type(exc).__name__}: {exc}")
            telemetry[strategy] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        if produced is not None and not produced.empty:
            frames[strategy] = produced
            art.write_scores(
                state.root, produced.to_dict(orient="records"), arm=strategy
            )
        else:
            telemetry.setdefault(strategy, {})["note"] = "arm produced no rows"

    state.frames_by_arm.update({k: v for k, v in frames.items() if k != "baseline"})

    evaluation = evaluate_arms(
        frames, cfg, baseline_arm="baseline", run_id=state.run_id,
        fit_labels=state.fit_labels(),
    )
    serialisable = {k: v for k, v in evaluation.items() if k != "reports"}
    serialisable["cue_effects"] = cue_effects
    serialisable["arm_telemetry"] = telemetry
    state.mitigation = evaluation

    write_json(state.root / "mitigation" / "evaluation.json", serialisable)
    table = mitigation_table(evaluation)
    if not table.empty:
        table.to_csv(state.root / "mitigation" / "metric_comparison.csv", index=False)
    costs = utility_table(evaluation)
    if not costs.empty:
        costs.to_csv(state.root / "mitigation" / "utility_costs.csv", index=False)

    state.manifest.record(
        "mitigate",
        {
            "elapsed_s": round(time.perf_counter() - started, 2),
            "arms": sorted(frames),
            "ranking": evaluation.get("ranking", []),
            "verdicts": {
                arm: comparison["verdict"]
                for arm, comparison in evaluation.get("comparisons", {}).items()
            },
        },
    )
    log_event(
        logger, logging.INFO, "mitigation evaluated",
        arms=len(frames) - 1, best=(evaluation.get("ranking") or ["none"])[0],
    )
    return state


def _load_external_effects(cfg: Config) -> Optional[Dict[str, Any]]:
    """Load cue effects estimated on a *different* run, for out-of-sample debiasing."""
    source = cfg.mitigation.cue_effect_source_run
    if not source:
        return None
    path = cfg.run_dir(source) / "mitigation" / "evaluation.json"
    if not path.exists():
        log_event(
            logger, logging.WARNING,
            "cue_effect_source_run not found; falling back to in-sample effects",
            path=str(path),
        )
        return None
    from ..utils.io import read_json

    payload = read_json(path)
    effects = payload.get("cue_effects")
    return effects if effects and "error" not in effects else None


# --------------------------------------------------------------------------- #
# Phase 5
# --------------------------------------------------------------------------- #
def report(state: PipelineState) -> PipelineState:
    from ..report import build_report

    started = time.perf_counter()
    outputs = build_report(state)
    state.manifest.record(
        "report",
        {
            "elapsed_s": round(time.perf_counter() - started, 2),
            "outputs": {k: str(v) for k, v in outputs.items()},
        },
    )
    log_event(logger, logging.INFO, "report written", **{k: str(v) for k, v in outputs.items()})
    return state


# --------------------------------------------------------------------------- #
# full pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(
    cfg: Config,
    *,
    run_id: Optional[str] = None,
    stages: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> PipelineState:
    """Run the whole study, or a chosen subset of stages."""
    all_stages = (
        "prepare_data", "generate_variants", "score_baseline",
        "score_variants", "audit", "mitigate", "report",
    )
    selected = list(stages or all_stages)
    unknown = [s for s in selected if s not in all_stages]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; available: {list(all_stages)}")

    state = start_run(cfg, run_id=run_id)
    started = time.perf_counter()

    if "prepare_data" in selected:
        prepare_data(state)
    if "generate_variants" in selected:
        generate_variants(state)
    if "score_baseline" in selected:
        score_baseline(state, dry_run=dry_run)
    if "score_variants" in selected:
        score_variants(state, dry_run=dry_run)
    if dry_run:
        state.manifest.set("status", "dry_run_complete")
        return state
    if "audit" in selected:
        audit(state)
    if "mitigate" in selected:
        mitigate(state)
    if "report" in selected:
        report(state)

    state.manifest.set("status", "complete")
    state.manifest.set("total_elapsed_s", round(time.perf_counter() - started, 2))
    state.manifest.set("warnings", state.warnings)
    log_event(
        logger, logging.INFO, "run complete",
        run_id=state.run_id, elapsed_s=round(time.perf_counter() - started, 2),
        n_warnings=len(state.warnings),
    )
    return state


# --------------------------------------------------------------------------- #
# reloading a finished run
# --------------------------------------------------------------------------- #
def load_run(cfg: Config, run_id: str) -> PipelineState:
    """Rehydrate a completed run from its artifacts, without any LLM calls.

    This is what makes re-analysis free: a new metric, a corrected statistic or a
    redesigned report can be applied to yesterday's scores without spending a
    single API call — and without the risk that a re-scored run quietly produces
    different numbers.
    """
    from ..utils.io import read_json

    root = cfg.run_dir(run_id)
    if not root.exists():
        raise FileNotFoundError(f"run directory {root} does not exist")

    manifest_path = root / "manifest.json"
    manifest = art.RunManifest(cfg, run_id, root)
    if manifest_path.exists():
        manifest.data = read_json(manifest_path)

    state = PipelineState(cfg=cfg, run_id=run_id, root=root, manifest=manifest)

    corpus = root / "corpus"
    if (corpus / "resumes.jsonl").exists():
        state.resumes = [
            Resume(
                resume_id=row["resume_id"], text=row["text"],
                job_family=row.get("job_family", "unknown"),
                source=row.get("source", "artifact"),
                template=None, slots=row.get("slots", {}), meta=row.get("meta", {}),
            )
            for row in read_jsonl(corpus / "resumes.jsonl")
        ]
    if (corpus / "jobs.jsonl").exists():
        state.jobs = [
            JobDescription(
                job_id=row["job_id"], title=row.get("title", row["job_id"]),
                text=row["text"], job_family=row.get("job_family", "unknown"),
                must_have=row.get("must_have", []), nice_to_have=row.get("nice_to_have", []),
                min_years_experience=float(row.get("min_years_experience", 0)),
                source="artifact",
            )
            for row in read_jsonl(corpus / "jobs.jsonl")
        ]
    pairs_csv = corpus / "pairs.csv"
    if pairs_csv.exists():
        frame = pd.read_csv(pairs_csv)
        state.pairs = [
            Pair(
                pair_id=str(row["pair_id"]), job_id=str(row["job_id"]),
                resume_id=str(row["resume_id"]),
                pairing_rationale=str(row.get("pairing_rationale", "")),
                expected_fit=(
                    None if pd.isna(row.get("expected_fit")) else str(row["expected_fit"])
                ),
                qualified_label=(
                    None if pd.isna(row.get("qualified_label")) else int(row["qualified_label"])
                ),
            )
            for _, row in frame.iterrows()
        ]

    for path in sorted((root / "scores").glob("scores_*.csv")):
        arm = path.stem[len("scores_"):]
        state.frames_by_arm[arm] = pd.read_csv(path)

    sanity = manifest.data.get("stages", {}).get("score_baseline", {}).get("sanity_check")
    if sanity:
        state.sanity = sanity
    state.warnings = list(manifest.data.get("warnings", []))
    return state
