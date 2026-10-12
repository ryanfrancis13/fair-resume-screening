"""Evaluating mitigations: did it help, and what did it cost?

A mitigation that reduces a bias metric is not automatically an improvement.  The
degenerate solutions are easy and worthless: score everyone identically (perfect
parity, zero utility), or abstain on every candidate (perfect fairness, no
screening).  Any honest evaluation therefore reports both sides of the trade, and
this module refuses to report one without the other.

**Fairness side** — the same audit suite, re-run per arm, so numbers are
comparable by construction rather than by assertion.

**Utility side**:

``rank_agreement``
    Kendall tau between the arm's ranking and the baseline's, per job. Near 1.0
    means the mitigation preserved the screening signal; a collapse means it
    destroyed it.
``top_k_overlap``
    Jaccard overlap of shortlists — the part a recruiter actually sees.
``score_dispersion_ratio``
    Score standard deviation relative to baseline. A ratio near zero is the
    "score everyone the same" failure mode, and it is invisible to every
    fairness metric.
``discrimination_retained``
    Correlation between the arm's scores and expected fit, where fit labels
    exist. The direct check that the screener still screens.
``coverage``
    Fraction of candidates the arm actually decides. Abstention is a legitimate
    strategy but must be priced.

The verdict combines them: an arm counts as recommended only if it reduces the
headline bias metrics *and* retains ranking agreement and dispersion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..config import Config
from ..fairness.suite import audit_summary, run_audit
from ..logging_setup import log_event
from ..schemas import AuditReport
from ..utils.frames import nonempty_mask, text_column

logger = logging.getLogger("frs.mitigate.evaluate")

#: Headline metrics compared across arms. Chosen because each is allocationally
#: or individually consequential, not merely statistically interesting.
HEADLINE_METRICS = (
    "counterfactual_max_swing[all]",
    "decision_flip_rate[all]",
    "counterfactual_max_swing[name]",
    "max_pairwise_score_gap[gender]",
    "max_pairwise_score_gap[race_ethnicity]",
    "impact_ratio[gender@k=5]",
    "impact_ratio[race_ethnicity@k=5]",
    "top1_change_rate[all]",
)

#: Minimum rank agreement with baseline for an arm to be considered usable.
MIN_RANK_AGREEMENT = 0.6
#: Minimum score dispersion relative to baseline; below this the arm has
#: flattened the score distribution rather than debiased it.
MIN_DISPERSION_RATIO = 0.5


def utility_metrics(
    baseline: pd.DataFrame,
    arm: pd.DataFrame,
    *,
    value_column: str = "overall_score",
    top_k: int = 5,
    fit_labels: Optional[Dict[tuple, str]] = None,
) -> Dict[str, Any]:
    """Cost side of the trade-off: did the arm keep the screening signal?"""
    def candidate_scores(frame: pd.DataFrame) -> Dict[str, pd.Series]:
        data = frame[frame[value_column].notna()]
        if "parse_ok" in data.columns:
            data = data[data["parse_ok"].astype(bool)]
        out: Dict[str, pd.Series] = {}
        for job_id, job_frame in data.groupby("job_id"):
            out[str(job_id)] = job_frame.groupby("base_resume_id")[value_column].mean()
        return out

    base_scores = candidate_scores(baseline)
    arm_scores = candidate_scores(arm)

    taus: List[float] = []
    overlaps: List[float] = []
    for job_id, base_series in base_scores.items():
        arm_series = arm_scores.get(job_id)
        if arm_series is None:
            continue
        shared = sorted(set(base_series.index) & set(arm_series.index))
        if len(shared) < 3:
            continue
        tau, _ = sps.kendalltau(
            [base_series[r] for r in shared], [arm_series[r] for r in shared]
        )
        if tau == tau:
            taus.append(float(tau))
        base_top = set(base_series.loc[shared].sort_values(ascending=False).head(top_k).index)
        arm_top = set(arm_series.loc[shared].sort_values(ascending=False).head(top_k).index)
        union = base_top | arm_top
        if union:
            overlaps.append(len(base_top & arm_top) / len(union))

    base_all = pd.concat(base_scores.values()) if base_scores else pd.Series(dtype=float)
    arm_all = pd.concat(arm_scores.values()) if arm_scores else pd.Series(dtype=float)
    base_sd = float(base_all.std(ddof=1)) if len(base_all) > 1 else float("nan")
    arm_sd = float(arm_all.std(ddof=1)) if len(arm_all) > 1 else float("nan")

    discrimination = None
    if fit_labels:
        ranks = {"weak": 0, "moderate": 1, "strong": 2}
        xs: List[float] = []
        ys: List[float] = []
        for job_id, series in arm_scores.items():
            for resume_id, score in series.items():
                fit = fit_labels.get((job_id, str(resume_id)))
                if fit in ranks:
                    xs.append(float(ranks[fit]))
                    ys.append(float(score))
        if len(xs) >= 4 and len(set(xs)) > 1:
            tau, p_value = sps.kendalltau(xs, ys)
            discrimination = {
                "kendall_tau_vs_expected_fit": round(float(tau), 4),
                "p_value": round(float(p_value), 6),
                "n": len(xs),
            }

    n_candidates_base = int(len(base_all))
    n_candidates_arm = int(len(arm_all))
    routed = 0
    if "recommendation" in arm.columns:
        routed = int((text_column(arm, "recommendation") == "human_review").sum())

    return {
        "rank_agreement_kendall_tau": round(float(np.mean(taus)), 4) if taus else None,
        "min_rank_agreement": round(float(np.min(taus)), 4) if taus else None,
        "top_k_overlap": round(float(np.mean(overlaps)), 4) if overlaps else None,
        "top_k": top_k,
        "baseline_score_sd": round(base_sd, 4) if base_sd == base_sd else None,
        "arm_score_sd": round(arm_sd, 4) if arm_sd == arm_sd else None,
        "score_dispersion_ratio": (
            round(arm_sd / base_sd, 4) if base_sd and base_sd == base_sd and arm_sd == arm_sd else None
        ),
        "mean_score_shift": (
            round(float(arm_all.mean() - base_all.mean()), 4)
            if len(arm_all) and len(base_all) else None
        ),
        "coverage": (
            round(n_candidates_arm / n_candidates_base, 4) if n_candidates_base else None
        ),
        "n_routed_to_human": routed,
        "discrimination_retained": discrimination,
    }


def compare_metric(
    baseline: AuditReport, arm: AuditReport, metric_name: str
) -> Optional[Dict[str, Any]]:
    """Baseline vs arm for one metric, with direction handled correctly."""
    base_metric = next((m for m in baseline.metrics if m.name == metric_name), None)
    arm_metric = next((m for m in arm.metrics if m.name == metric_name), None)
    if base_metric is None or arm_metric is None:
        return None
    if base_metric.value != base_metric.value or arm_metric.value != arm_metric.value:
        return None

    delta = arm_metric.value - base_metric.value
    # For a "higher is better" metric such as impact ratio, an increase is an
    # improvement; for a gap or a swing, a decrease is.
    improved = delta > 0 if arm_metric.higher_is_better else delta < 0
    denominator = abs(base_metric.value)
    return {
        "metric": metric_name,
        "family": arm_metric.family,
        "baseline": round(base_metric.value, 4),
        "arm": round(arm_metric.value, 4),
        "absolute_change": round(delta, 4),
        "relative_change_pct": (
            round(100.0 * delta / denominator, 2) if denominator > 1e-12 else None
        ),
        "higher_is_better": arm_metric.higher_is_better,
        "improved": bool(improved),
        "baseline_passed": base_metric.passed,
        "arm_passed": arm_metric.passed,
        "unit": arm_metric.unit,
    }


def evaluate_arms(
    frames: Dict[str, pd.DataFrame],
    cfg: Config,
    *,
    baseline_arm: str = "baseline",
    run_id: str = "adhoc",
    fit_labels: Optional[Dict[tuple, str]] = None,
) -> Dict[str, Any]:
    """Audit every arm, compare each to baseline, and issue a verdict per arm."""
    if baseline_arm not in frames:
        raise KeyError(
            f"baseline arm {baseline_arm!r} not present; available: {sorted(frames)}"
        )

    reports: Dict[str, AuditReport] = {}
    for name, frame in frames.items():
        if frame is None or frame.empty:
            continue
        reports[name] = run_audit(frame, cfg, run_id=f"{run_id}:{name}")

    baseline_report = reports[baseline_arm]
    baseline_frame = frames[baseline_arm]
    top_k = max(cfg.fairness.top_k) if cfg.fairness.top_k else 5

    comparisons: Dict[str, Any] = {}
    for name, report in reports.items():
        if name == baseline_arm:
            continue
        metric_deltas = [
            row for row in (
                compare_metric(baseline_report, report, metric)
                for metric in HEADLINE_METRICS
            ) if row is not None
        ]
        utility = utility_metrics(
            baseline_frame, frames[name], top_k=top_k, fit_labels=fit_labels
        )
        verdict = _verdict(
            metric_deltas, utility,
            eliminates_cue_variation=_eliminates_cue_variation(frames[name]),
        )
        comparisons[name] = {
            "metric_comparisons": metric_deltas,
            "n_metrics_improved": sum(1 for m in metric_deltas if m["improved"]),
            "n_metrics_worsened": sum(1 for m in metric_deltas if not m["improved"]),
            "utility": utility,
            "audit_summary": audit_summary(report),
            **verdict,
        }
        log_event(
            logger, logging.INFO, "mitigation arm evaluated",
            arm=name, improved=comparisons[name]["n_metrics_improved"],
            worsened=comparisons[name]["n_metrics_worsened"],
            verdict=verdict["verdict"],
        )

    _verdict_rank = {
        "recommended": 0,
        "partial": 1,
        "cue_variation_eliminated_by_construction": 2,
        "fairness_improved_at_utility_cost": 3,
        "inconclusive": 4,
        "not_recommended": 5,
    }
    ranked = sorted(
        comparisons.items(),
        key=lambda item: (
            _verdict_rank.get(item[1]["verdict"], 9),
            -item[1]["n_metrics_improved"],
        ),
    )

    return {
        "baseline_arm": baseline_arm,
        "arms": sorted(reports),
        "baseline_summary": audit_summary(baseline_report),
        "comparisons": comparisons,
        "ranking": [name for name, _ in ranked],
        "reports": reports,
        "guidance": (
            "An arm is only recommended if it reduces the headline bias metrics AND keeps "
            "rank agreement with the baseline above "
            f"{MIN_RANK_AGREEMENT} and score dispersion above "
            f"{MIN_DISPERSION_RATIO}x baseline. Those guards exist because the two easiest "
            "ways to pass a fairness audit — give everyone the same score, or decline to "
            "decide — are also the two ways to destroy the product."
        ),
    }


def _eliminates_cue_variation(frame: pd.DataFrame) -> bool:
    """Does this arm serve one score per candidate, with no cue variants left?

    ``counterfactual_marginalization`` and ``sensitivity_gated_review`` both
    collapse a candidate's variants into a single served score. That is exactly
    what they are for — but it means the counterfactual audit has nothing left to
    measure on them, because there is no cue to vary. Detecting the case lets the
    evaluation say so instead of reporting a bare "inconclusive", which reads like
    a failure rather than a structural property.
    """
    if frame is None or frame.empty:
        return False
    signal_columns = [c for c in frame.columns if c.startswith("signal_")]
    has_signals = any(nonempty_mask(frame, c).any() for c in signal_columns)
    if has_signals:
        return False
    if "variant_id" in frame.columns and {"job_id", "base_resume_id"} <= set(frame.columns):
        per_candidate = frame.groupby(["job_id", "base_resume_id"])["variant_id"].nunique()
        return bool((per_candidate <= 1).all())
    return not has_signals


def _verdict(
    metric_deltas: Sequence[Dict[str, Any]], utility: Dict[str, Any],
    *, eliminates_cue_variation: bool = False,
) -> Dict[str, Any]:
    improved = sum(1 for m in metric_deltas if m["improved"])
    worsened = sum(1 for m in metric_deltas if not m["improved"])
    reasons: List[str] = []

    rank_agreement = utility.get("rank_agreement_kendall_tau")
    dispersion = utility.get("score_dispersion_ratio")

    utility_ok = True
    if rank_agreement is not None and rank_agreement < MIN_RANK_AGREEMENT:
        utility_ok = False
        reasons.append(
            f"rank agreement with baseline is only {rank_agreement:.2f} "
            f"(below {MIN_RANK_AGREEMENT}): the arm has changed who gets shortlisted, "
            "not just how fairly"
        )
    if dispersion is not None and dispersion < MIN_DISPERSION_RATIO:
        utility_ok = False
        reasons.append(
            f"score dispersion collapsed to {dispersion:.2f}x baseline: the arm is "
            "flattening scores rather than debiasing them, which passes fairness metrics "
            "by removing the signal"
        )
    discrimination = utility.get("discrimination_retained")
    if discrimination and discrimination.get("kendall_tau_vs_expected_fit", 1.0) < 0.2:
        utility_ok = False
        reasons.append(
            "correlation with expected fit has collapsed: the arm no longer discriminates "
            "between strong and weak candidates"
        )

    if not metric_deltas and eliminates_cue_variation:
        verdict = "cue_variation_eliminated_by_construction"
        reasons.append(
            "This arm serves one score per candidate, marginalised or gated across their "
            "cue variants, so no cue remains for the counterfactual audit to vary. The "
            "absence of a measured gap here is a property of the intervention, not "
            "evidence that it works. Judge it on the utility columns and on its coverage "
            "and cost, and validate it by auditing the pipeline it feeds rather than the "
            "arm itself."
        )
    elif not metric_deltas:
        verdict = "inconclusive"
        reasons.append("no headline metric was computable in both baseline and arm")
    elif improved > worsened and utility_ok:
        verdict = "recommended"
    elif improved > worsened and not utility_ok:
        verdict = "fairness_improved_at_utility_cost"
    elif improved > 0 and utility_ok:
        verdict = "partial"
    else:
        verdict = "not_recommended"
        if improved == 0:
            reasons.append("no headline bias metric improved")

    return {
        "verdict": verdict,
        "verdict_reasons": reasons,
        "utility_acceptable": utility_ok,
    }


def mitigation_table(evaluation: Dict[str, Any]) -> pd.DataFrame:
    """One row per (arm, metric), for the report."""
    rows: List[Dict[str, Any]] = []
    for arm, comparison in evaluation.get("comparisons", {}).items():
        for metric in comparison["metric_comparisons"]:
            rows.append({"arm": arm, "verdict": comparison["verdict"], **metric})
    return pd.DataFrame(rows)


def utility_table(evaluation: Dict[str, Any]) -> pd.DataFrame:
    """One row per arm summarising fairness gain against utility cost."""
    rows: List[Dict[str, Any]] = []
    for arm, comparison in evaluation.get("comparisons", {}).items():
        utility = comparison["utility"]
        rows.append(
            {
                "arm": arm,
                "verdict": comparison["verdict"],
                "metrics_improved": comparison["n_metrics_improved"],
                "metrics_worsened": comparison["n_metrics_worsened"],
                "rank_agreement": utility.get("rank_agreement_kendall_tau"),
                "top_k_overlap": utility.get("top_k_overlap"),
                "score_dispersion_ratio": utility.get("score_dispersion_ratio"),
                "mean_score_shift": utility.get("mean_score_shift"),
                "coverage": utility.get("coverage"),
                "n_routed_to_human": utility.get("n_routed_to_human"),
                "discrimination_tau": (
                    (utility.get("discrimination_retained") or {}).get(
                        "kendall_tau_vs_expected_fit"
                    )
                ),
            }
        )
    return pd.DataFrame(rows)
