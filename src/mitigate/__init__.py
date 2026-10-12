"""Bias-mitigation strategies and their evaluation."""

from .evaluate import (
    HEADLINE_METRICS,
    compare_metric,
    evaluate_arms,
    mitigation_table,
    utility_metrics,
    utility_table,
)
from .strategies import (
    ALL_STRATEGIES,
    LEGAL_NOTE,
    POST_HOC_REGISTRY,
    POST_HOC_STRATEGIES,
    PRE_INFERENCE_STRATEGIES,
    ArmSpec,
    counterfactual_marginalization,
    cue_effect_debiasing,
    describe_strategies,
    estimate_cue_effects,
    evidence_gate,
    pre_inference_arms,
    sensitivity_gated_review,
)

__all__ = [
    "ALL_STRATEGIES",
    "ArmSpec",
    "HEADLINE_METRICS",
    "LEGAL_NOTE",
    "POST_HOC_REGISTRY",
    "POST_HOC_STRATEGIES",
    "PRE_INFERENCE_STRATEGIES",
    "compare_metric",
    "counterfactual_marginalization",
    "cue_effect_debiasing",
    "describe_strategies",
    "estimate_cue_effects",
    "evaluate_arms",
    "evidence_gate",
    "mitigation_table",
    "pre_inference_arms",
    "sensitivity_gated_review",
    "utility_metrics",
    "utility_table",
]
