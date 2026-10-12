"""The audit suite: runs every metric family over one score table.

Two design commitments make this more than a loop over metric functions.

**The suite never calls an LLM.** Its only input is a tidy score table. That means
an audit can be re-run offline on committed artifacts, by a third party, without
an API key — and it means the auditor itself can be validated against a provider
with a known injected bias (``frs validate-auditor``). An auditor whose own
sensitivity and false-positive rate are unmeasured is not evidence.

**Metrics are collected first, then corrected together.** Every p-value in the
suite goes through one Benjamini-Hochberg pass at the end. Correcting within each
family separately would understate the multiplicity, and with this many tests the
difference is the difference between reporting findings and manufacturing them.

Execution order is deliberate:

1. **Reliability first.** The noise floor and ICC are computed before any bias
   metric, because they decide whether the bias metrics mean anything at all.
2. Metric families, each independent of the others.
3. Multiplicity correction across everything.
4. Conflict analysis, which reads the corrected results.

Families whose preconditions are unmet (no labels, no intersecting cues, no
repeats) are recorded as skipped **with the reason**, so a reader can tell an
absent metric from a passing one.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ..config import Config
from ..logging_setup import log_event
from ..schemas import AuditReport, MetricResult
from ..utils.frames import nonempty_mask, split_terms, text_column
from .allocational import allocational_metrics, ranking_metrics
from .calibration_fairness import calibration_fairness_metrics
from .conflicts import annotate_report, summarise_conflicts
from .counterfactual import counterfactual_metrics, cue_effect_table
from .group_parity import group_parity_metrics
from .intersectional import intersectional_metrics
from .rationale_audit import rationale_examples, rationale_metrics
from .reliability import parse_quality, self_consistency, variance_decomposition
from .stats import benjamini_hochberg

logger = logging.getLogger("frs.fairness.suite")


def records_to_frame(records: Sequence[Any]) -> pd.DataFrame:
    """Flatten :class:`~frs.schemas.ScoreRecord` objects into a tidy DataFrame."""
    rows = [r.flat() if hasattr(r, "flat") else dict(r) for r in records]
    frame = pd.DataFrame(rows)
    for column in ("overall_score", "confidence", "evidence_support_rate"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("parse_ok", "is_baseline", "cached", "protected_attribute_referenced"):
        if column in frame.columns:
            frame[column] = frame[column].astype(bool)
    if "qualified_label" in frame.columns:
        frame["qualified_label"] = pd.to_numeric(frame["qualified_label"], errors="coerce")
    return frame


def available_attributes(frame: pd.DataFrame) -> List[str]:
    """Protected/validity attributes actually present in this score table."""
    return sorted(
        column[len("signal_"):]
        for column in frame.columns
        if column.startswith("signal_") and nonempty_mask(frame, column).any()
    )


def run_audit(
    frame: pd.DataFrame,
    cfg: Config,
    *,
    run_id: str = "adhoc",
    arm: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> AuditReport:
    """Run every applicable metric family and return a complete audit report."""
    from ..utils.seeds import rng as make_rng

    if arm is not None and "arm" in frame.columns:
        frame = frame[frame["arm"] == arm]

    fair = cfg.fairness
    rng = make_rng(cfg.seed, f"audit:{run_id}:{arm or 'all'}")

    present = available_attributes(frame)
    requested = [a for a in fair.attributes if a in present]
    skipped_attributes = [a for a in fair.attributes if a not in present]

    cue_families = sorted(split_terms(text_column(frame, "cue_families")))

    warnings: List[str] = []
    skipped: List[Dict[str, str]] = []
    if skipped_attributes:
        skipped.append(
            {
                "family": "group_parity/allocational",
                "reason": (
                    "no cue in this run signals: " + ", ".join(skipped_attributes)
                    + ". Enable a cue family that carries them (see frs perturb --list-cues)."
                ),
            }
        )

    # ---- 1. reliability, before anything else ----------------------------
    reliability = variance_decomposition(frame)
    consistency = self_consistency(frame)
    quality = parse_quality(frame)
    reliability["self_consistency"] = consistency
    reliability["parse_quality"] = quality
    warnings.extend(quality.get("warnings", []))
    if not reliability.get("has_repeats"):
        warnings.append(
            "No repeated decodes in this run: the model's noise floor is unknown, so cue "
            "effects cannot be compared against it."
        )
    icc = reliability.get("icc1_resume")
    if icc is not None and icc < 0.4:
        warnings.append(
            f"Scorer reliability is low (ICC = {icc:.2f}). Fairness estimates from this run "
            "are provisional."
        )
    noise_floor_sd = reliability.get("noise_floor_sd") if reliability.get("has_repeats") else None

    # ---- 2. metric families ----------------------------------------------
    metrics: List[MetricResult] = []

    metrics.extend(
        counterfactual_metrics(
            frame, cue_families=cue_families, tolerance=fair.counterfactual_tolerance,
            noise_floor_sd=noise_floor_sd, n_boot=fair.bootstrap_iterations, rng=rng,
        )
    )
    metrics.extend(
        group_parity_metrics(
            frame, requested, reference_levels=fair.reference_levels, alpha=fair.alpha,
            n_boot=fair.bootstrap_iterations, n_permutations=fair.permutation_iterations,
            min_practical_effect=fair.min_practical_effect, rng=rng,
        )
    )
    metrics.extend(
        allocational_metrics(
            frame, requested, top_k=fair.top_k,
            reference_levels=fair.reference_levels,
            impact_ratio_threshold=fair.impact_ratio_threshold,
        )
    )
    metrics.extend(ranking_metrics(frame, cue_families=cue_families))

    inter = [a for a in fair.intersectional_attributes if a in present]
    if len(inter) >= 2:
        metrics.extend(
            intersectional_metrics(
                frame, inter, n_permutations=fair.permutation_iterations,
                min_practical_effect=fair.min_practical_effect, rng=rng,
            )
        )
    else:
        skipped.append(
            {
                "family": "intersectional",
                "reason": (
                    "needs at least 2 of "
                    + ", ".join(fair.intersectional_attributes)
                    + " to be signalled by the same rows; the name cue family provides both "
                      "gender and race_ethnicity"
                ),
            }
        )

    if fair.enable_rationale_audit:
        rationale = rationale_metrics(frame, requested)
        metrics.extend(rationale)
        if not rationale:
            skipped.append(
                {"family": "rationale", "reason": "no rationale text captured in this run"}
            )

    if fair.enable_calibration_audit:
        calibration = calibration_fairness_metrics(frame, requested)
        metrics.extend(calibration)
        if not calibration:
            skipped.append(
                {
                    "family": "calibration",
                    "reason": (
                        "needs qualified_label on enough rows per group. Labels live in "
                        "data/sample/pairs.json; without them calibration and "
                        "equalised-odds criteria are undefined."
                    ),
                }
            )

    # ---- 3. multiplicity, over the whole suite ----------------------------
    p_values = [m.p_value for m in metrics]
    correction = benjamini_hochberg(p_values, alpha=fair.alpha)
    for metric, q_value in zip(metrics, correction["q_values"]):
        metric.q_value = float(q_value) if q_value is not None else float("nan")

    # ---- 4. assemble ------------------------------------------------------
    report = AuditReport(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_records=int(len(frame)),
        n_base_resumes=int(frame["base_resume_id"].nunique()) if "base_resume_id" in frame else 0,
        n_jobs=int(frame["job_id"].nunique()) if "job_id" in frame else 0,
        n_variants=int(frame["variant_id"].nunique()) if "variant_id" in frame else 0,
        arms=sorted(text_column(frame, "arm").unique()) if "arm" in frame else [],
        metrics=metrics,
        reliability=reliability,
        provenance={
            "attributes_audited": requested,
            "attributes_unavailable": skipped_attributes,
            "cue_families_present": cue_families,
            "skipped_families": skipped,
            "multiplicity": {
                k: v for k, v in correction.items() if k not in ("q_values", "rejected")
            },
            "config": {
                "alpha": fair.alpha,
                "top_k": list(fair.top_k),
                "min_practical_effect": fair.min_practical_effect,
                "impact_ratio_threshold": fair.impact_ratio_threshold,
                "counterfactual_tolerance": fair.counterfactual_tolerance,
                "bootstrap_iterations": fair.bootstrap_iterations,
                "permutation_iterations": fair.permutation_iterations,
                "reference_levels": dict(fair.reference_levels),
            },
            "provider": (
                str(frame["provider"].iloc[0]) if "provider" in frame and len(frame) else ""
            ),
            "model": str(frame["model"].iloc[0]) if "model" in frame and len(frame) else "",
            **(provenance or {}),
        },
        warnings=warnings,
    )

    # ---- 5. conflicts, from the corrected results -------------------------
    annotate_report(report, min_practical_effect=fair.min_practical_effect)

    log_event(
        logger, logging.INFO, "audit complete",
        run_id=run_id, arm=arm or "all", n_metrics=len(metrics),
        n_significant=sum(1 for m in metrics if m.significant),
        n_conflicts=len(report.conflicts),
        n_warnings=len(report.warnings),
    )
    return report


# --------------------------------------------------------------------------- #
# summaries for reporting
# --------------------------------------------------------------------------- #
def metrics_table(report: AuditReport) -> pd.DataFrame:
    """Flat table of every metric, ready for CSV or HTML."""
    rows: List[Dict[str, Any]] = []
    for m in report.metrics:
        rows.append(
            {
                "family": m.family,
                "metric": m.name,
                "attribute": m.attribute,
                "value": round(m.value, 4) if m.value == m.value else None,
                "unit": m.unit,
                "ci_low": round(m.ci_low, 4) if m.ci_low == m.ci_low else None,
                "ci_high": round(m.ci_high, 4) if m.ci_high == m.ci_high else None,
                "p_value": round(m.p_value, 6) if m.p_value == m.p_value else None,
                "q_value_bh": round(m.q_value, 6) if m.q_value == m.q_value else None,
                "effect_size": round(m.effect_size, 4) if m.effect_size == m.effect_size else None,
                "effect_size_name": m.effect_size_name,
                "n": m.n,
                "threshold": m.threshold,
                "passed": m.passed,
                "significant_after_fdr": m.significant,
                "interpretation": m.interpretation,
            }
        )
    return pd.DataFrame(rows)


def headline_findings(report: AuditReport, *, top_n: int = 8) -> List[Dict[str, Any]]:
    """The findings a reader should see first.

    Ranked by consequence, not by p-value: a decision flip and a failed impact
    ratio outrank a statistically significant sub-point score gap, because they
    describe someone being screened out.
    """
    priority = {
        "counterfactual": 0,
        "allocational": 1,
        "intersectional": 2,
        "ranking": 3,
        "group_parity": 4,
        "rationale": 5,
        "calibration": 6,
    }

    def score(metric: MetricResult) -> tuple:
        failed = metric.passed is False
        return (
            0 if failed else 1,
            priority.get(metric.family, 9),
            0 if metric.significant else 1,
            -abs(metric.value) if metric.value == metric.value else 0,
        )

    ordered = sorted(report.metrics, key=score)
    findings: List[Dict[str, Any]] = []
    for metric in ordered[:top_n]:
        findings.append(
            {
                "metric": metric.name,
                "family": metric.family,
                "attribute": metric.attribute,
                "value": round(metric.value, 4) if metric.value == metric.value else None,
                "unit": metric.unit,
                "verdict": (
                    "FAIL" if metric.passed is False
                    else ("PASS" if metric.passed else "INFO")
                ),
                "q_value_bh": round(metric.q_value, 6) if metric.q_value == metric.q_value else None,
                "interpretation": metric.interpretation,
            }
        )
    return findings


def audit_summary(report: AuditReport) -> Dict[str, Any]:
    """Compact overview for the CLI and the run manifest."""
    by_family: Dict[str, Dict[str, int]] = {}
    for metric in report.metrics:
        bucket = by_family.setdefault(
            metric.family, {"n": 0, "failed": 0, "significant": 0}
        )
        bucket["n"] += 1
        bucket["failed"] += int(metric.passed is False)
        bucket["significant"] += int(metric.significant)
    return {
        "run_id": report.run_id,
        "n_records": report.n_records,
        "n_base_resumes": report.n_base_resumes,
        "n_jobs": report.n_jobs,
        "n_variants": report.n_variants,
        "n_metrics": len(report.metrics),
        "n_failed": sum(1 for m in report.metrics if m.passed is False),
        "n_significant_after_fdr": sum(1 for m in report.metrics if m.significant),
        "by_family": by_family,
        "reliability": {
            "icc1_resume": report.reliability.get("icc1_resume"),
            "noise_floor_sd": report.reliability.get("noise_floor_sd"),
            "cue_effect_to_noise_ratio": report.reliability.get("cue_effect_to_noise_ratio"),
            "has_repeats": report.reliability.get("has_repeats"),
        },
        "conflicts": summarise_conflicts(report.conflicts),
        "warnings": report.warnings,
        "skipped_families": report.provenance.get("skipped_families", []),
    }


def extras_for_report(frame: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """Supporting tables the narrative report needs beyond the metric list."""
    extras: Dict[str, Any] = {}
    table = cue_effect_table(frame)
    if not table.empty:
        extras["cue_effects"] = table.to_dict(orient="records")
    examples: Dict[str, Any] = {}
    for attribute in available_attributes(frame):
        pairs = rationale_examples(frame, attribute, max_pairs=3)
        if pairs:
            examples[attribute] = pairs
    if examples:
        extras["rationale_examples"] = examples
    del cfg
    return extras
