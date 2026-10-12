"""Multi-family fairness auditing: metrics, statistics and conflict analysis."""

from .allocational import (
    allocational_analysis,
    allocational_metrics,
    equal_opportunity,
    ranking_metrics,
)
from .calibration_fairness import calibration_by_group, calibration_fairness_metrics
from .conflicts import annotate_report, detect_conflicts, summarise_conflicts
from .counterfactual import counterfactual_invariance, counterfactual_metrics, cue_effect_table
from .group_parity import group_parity, group_parity_metrics
from .intersectional import intersectional_analysis, intersectional_metrics
from .rationale_audit import distinctive_terms, lexicon_rates, rationale_metrics
from .reliability import parse_quality, self_consistency, variance_decomposition
from .suite import (
    audit_summary,
    available_attributes,
    extras_for_report,
    headline_findings,
    metrics_table,
    records_to_frame,
    run_audit,
)

__all__ = [
    "allocational_analysis",
    "allocational_metrics",
    "annotate_report",
    "audit_summary",
    "available_attributes",
    "calibration_by_group",
    "calibration_fairness_metrics",
    "counterfactual_invariance",
    "counterfactual_metrics",
    "cue_effect_table",
    "detect_conflicts",
    "distinctive_terms",
    "equal_opportunity",
    "extras_for_report",
    "group_parity",
    "group_parity_metrics",
    "headline_findings",
    "intersectional_analysis",
    "intersectional_metrics",
    "lexicon_rates",
    "metrics_table",
    "parse_quality",
    "ranking_metrics",
    "rationale_metrics",
    "records_to_frame",
    "run_audit",
    "self_consistency",
    "summarise_conflicts",
    "variance_decomposition",
]
