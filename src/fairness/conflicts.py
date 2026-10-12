"""Detecting and explaining disagreement between fairness metrics.

The project's brief is explicit about this: *"different fairness metrics can
mathematically contradict each other ... the write-up should explain any
disagreement between metrics, not just report numbers."*  A dashboard of eight
metrics with mixed verdicts is not a result; it becomes one only when the
disagreements are named, attributed to a cause, and turned into a decision about
which criterion the system is prioritising.

This module encodes the known disagreement patterns as rules, each carrying the
theorem or empirical mechanism that produces it:

``calibration_vs_error_rates``
    Kleinberg et al. (2017); Chouldechova (2017). With unequal base rates,
    calibration within groups and equal error rates are mutually exclusive.
``parity_vs_calibration``
    Equalising selection rates across groups with different base rates
    necessarily breaks calibration, and vice versa.
``score_parity_vs_allocational``
    A tiny mean gap can still reorder a tightly packed shortlist. Distributional
    fairness does not imply allocational fairness.
``individual_vs_group``
    Group means can match while individual candidates swing wildly, if the swings
    cancel. Group parity does not imply counterfactual invariance.
``score_vs_rationale``
    Identical scores with systematically different explanations still steer the
    human reader.
``validity_collapse``
    The scorer is invariant even to legitimately job-relevant signal, so its
    apparent fairness is inertness. The most important rule here: it invalidates
    every other result rather than adding to them.
``power_masquerade``
    "No significant gap" where the minimum detectable effect exceeds the
    practical threshold. Absence of evidence reported as evidence of absence.

Each detection returns a plain-language explanation and a recommendation, so the
report can say what to do rather than only that something is odd.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..schemas import AuditReport, MetricResult


def _find(metrics: Sequence[MetricResult], prefix: str, attribute: str = "") -> List[MetricResult]:
    return [
        m for m in metrics
        if m.name.startswith(prefix) and (not attribute or m.attribute == attribute)
    ]


def _value(metric: Optional[MetricResult]) -> float:
    return metric.value if metric is not None else float("nan")


def _validity_scopes() -> Sequence[str]:
    """Cue families that are validity probes rather than protected attributes.

    A prestige swing is the *expected* behaviour of a working scorer, so counting
    it as evidence of individual unfairness would flag every competent screener.
    Read from the cue registry so the two cannot drift apart.
    """
    try:
        from ..perturb.cues import VALIDITY, VALIDITY_PROXY, build_registry

        with_warnings_suppressed = build_registry()
        return tuple(
            name for name, family in with_warnings_suppressed.items()
            if family.cue_type in (VALIDITY, VALIDITY_PROXY)
        )
    except Exception:  # noqa: BLE001 - conflict analysis must never crash a report
        return ("education_prestige", "writing_register")


def _protected_swings(metrics: Sequence[MetricResult], prefix: str) -> List[MetricResult]:
    """Swing metrics scoped to protected cue families only."""
    excluded = set(_validity_scopes())
    return [m for m in _find(metrics, prefix) if m.attribute not in excluded]


def _attributes(metrics: Sequence[MetricResult]) -> List[str]:
    seen: List[str] = []
    for m in metrics:
        if m.attribute and m.attribute not in seen:
            seen.append(m.attribute)
    return seen


def detect_conflicts(
    metrics: Sequence[MetricResult],
    *,
    reliability: Optional[Dict[str, Any]] = None,
    min_practical_effect: float = 1.0,
) -> List[Dict[str, Any]]:
    """Return every detected metric disagreement, with its explanation."""
    conflicts: List[Dict[str, Any]] = []
    reliability = reliability or {}

    for attribute in _attributes(metrics):
        conflicts.extend(_calibration_vs_error_rates(metrics, attribute))
        conflicts.extend(_parity_vs_calibration(metrics, attribute))
        conflicts.extend(_score_vs_allocational(metrics, attribute, min_practical_effect))
        conflicts.extend(_group_vs_individual(metrics, attribute, min_practical_effect))
        conflicts.extend(_score_vs_rationale(metrics, attribute, min_practical_effect))
        conflicts.extend(_power_masquerade(metrics, attribute, min_practical_effect))
        conflicts.extend(_marginal_vs_intersectional(metrics, attribute, min_practical_effect))

    conflicts.extend(_validity_collapse(metrics, min_practical_effect))
    conflicts.extend(_noise_floor_conflict(metrics, reliability))
    conflicts.extend(_noise_dominated_failures(metrics, reliability))
    return conflicts


# --------------------------------------------------------------------------- #
# rules
# --------------------------------------------------------------------------- #
def _calibration_vs_error_rates(
    metrics: Sequence[MetricResult], attribute: str
) -> List[Dict[str, Any]]:
    ece = _find(metrics, "calibration_ece_spread[", attribute)
    tpr = _find(metrics, "equalised_odds_tpr_spread[", attribute)
    fpr = _find(metrics, "equalised_odds_fpr_spread[", attribute)
    if not ece or not (tpr or fpr):
        return []
    ece_ok = bool(ece[0].passed)
    errors = [m for m in (tpr + fpr) if m.passed is not None]
    if not errors:
        return []
    errors_ok = all(m.passed for m in errors)
    if ece_ok == errors_ok:
        return []

    satisfied = "calibration" if ece_ok else "equal error rates"
    violated = "equal error rates" if ece_ok else "calibration"
    base_note = ""
    detail = ece[0].detail or {}
    if isinstance(detail, dict):
        base_note = str(detail.get("impossibility_note", ""))
    return [
        {
            "type": "calibration_vs_error_rates",
            "attribute": attribute,
            "severity": "expected",
            "metrics": [ece[0].name] + [m.name for m in errors],
            "explanation": (
                f"For {attribute}, {satisfied} is satisfied while {violated} is violated. "
                "This is the Kleinberg-Mullainathan-Raghavan / Chouldechova impossibility "
                "result, not an error: whenever qualification base rates differ between "
                "groups, a scorer cannot be calibrated within groups and simultaneously "
                "equalise false-positive and false-negative rates. "
                + base_note
            ),
            "recommendation": (
                "Do not attempt to satisfy both. State which criterion the deployment "
                "prioritises and why. For a screening tool that produces a shortlist for "
                "human review, equalising error rates at the shortlist boundary is usually "
                "the more defensible choice, because the harm is being filtered out; "
                "calibration matters more when the score itself is shown to a decision maker."
            ),
            "citation": (
                "Kleinberg, Mullainathan & Raghavan (2017), 'Inherent trade-offs in the fair "
                "determination of risk scores'; Chouldechova (2017), 'Fair prediction with "
                "disparate impact'."
            ),
        }
    ]


def _parity_vs_calibration(
    metrics: Sequence[MetricResult], attribute: str
) -> List[Dict[str, Any]]:
    parity = _find(metrics, "selection_rate_parity_diff[", attribute)
    ece = _find(metrics, "calibration_ece_spread[", attribute)
    if not parity or not ece:
        return []
    parity_ok = all(m.passed for m in parity if m.passed is not None)
    ece_ok = bool(ece[0].passed)
    if parity_ok == ece_ok:
        return []
    return [
        {
            "type": "parity_vs_calibration",
            "attribute": attribute,
            "severity": "expected",
            "metrics": [m.name for m in parity] + [ece[0].name],
            "explanation": (
                f"For {attribute}, demographic parity and calibration disagree. These are "
                "different definitions of fairness, not two measurements of one thing: "
                "parity requires equal selection rates regardless of qualification, "
                "calibration requires that a score mean the same thing everywhere. When "
                "base rates differ, enforcing one breaks the other."
            ),
            "recommendation": (
                "Choose based on the decision being supported, and record the choice in the "
                "model card. Reporting both without stating a priority leaves the reader to "
                "guess which one the system is optimising."
            ),
            "citation": "Chouldechova (2017); Barocas, Hardt & Narayanan (2019), ch. 3.",
        }
    ]


def _score_vs_allocational(
    metrics: Sequence[MetricResult], attribute: str, min_practical_effect: float
) -> List[Dict[str, Any]]:
    gaps = _find(metrics, "max_pairwise_score_gap[", attribute)
    parity = _find(metrics, "selection_rate_parity_diff[", attribute)
    top1 = [m for m in _protected_swings(metrics, "top1_change_rate[") if m.value > 0]
    if not gaps:
        return []
    gap_small = abs(_value(gaps[0])) < min_practical_effect
    allocational_bad = any(
        (m.passed is False) for m in parity
    ) or bool(top1)
    if not (gap_small and allocational_bad):
        return []
    return [
        {
            "type": "score_parity_vs_allocational",
            "attribute": attribute,
            "severity": "important",
            "metrics": [gaps[0].name] + [m.name for m in parity] + [m.name for m in top1],
            "explanation": (
                f"The mean score gap for {attribute} is small "
                f"({_value(gaps[0]):.2f} points, below the {min_practical_effect:g}-point "
                "practical threshold), yet the allocational metrics show unequal shortlisting. "
                "This is the Seshadri et al. (2025) finding: when candidates are closely "
                "bunched, a sub-threshold score shift is enough to reorder the top of the "
                "list, and only the top of the list gets read."
            ),
            "recommendation": (
                "Report the allocational metrics as the headline result for this attribute. "
                "A mean-gap-only analysis would have concluded 'no meaningful bias' while "
                "candidates were in fact being displaced from the shortlist."
            ),
            "citation": (
                "Seshadri, Chen, Singh & Goldfarb-Tarrant (2025), 'Small changes, large "
                "consequences: Analyzing the allocational fairness of LLMs in hiring'."
            ),
        }
    ]


def _group_vs_individual(
    metrics: Sequence[MetricResult], attribute: str, min_practical_effect: float
) -> List[Dict[str, Any]]:
    gaps = _find(metrics, "max_pairwise_score_gap[", attribute)
    swings = _protected_swings(metrics, "counterfactual_max_swing[")
    flips = [m for m in _protected_swings(metrics, "decision_flip_rate[") if m.value > 0]
    if not gaps or not swings:
        return []
    gap_small = abs(_value(gaps[0])) < min_practical_effect
    swing_large = any(m.value > 2 * min_practical_effect for m in swings)
    if not (gap_small and (swing_large or flips)):
        return []
    worst_swing = max((m.value for m in swings), default=float("nan"))
    return [
        {
            "type": "individual_vs_group",
            "attribute": attribute,
            "severity": "important",
            "metrics": [gaps[0].name] + [m.name for m in swings] + [m.name for m in flips],
            "explanation": (
                f"Group means for {attribute} are close ({_value(gaps[0]):.2f} points apart), "
                f"but individual candidates swing by up to {worst_swing:.2f} points when only "
                "the cue changes. Group parity is an average over people, so opposite-signed "
                "individual effects cancel: the aggregate looks fair while specific candidates "
                "are treated inconsistently."
            ),
            "recommendation": (
                "Report the counterfactual (individual) metrics alongside the group metrics. "
                "For an individual candidate, 'the group averages matched' is not an answer to "
                "'my score changed when my name did'."
            ),
            "citation": "Dwork et al. (2012), 'Fairness through awareness'.",
        }
    ]


def _score_vs_rationale(
    metrics: Sequence[MetricResult], attribute: str, min_practical_effect: float
) -> List[Dict[str, Any]]:
    gaps = _find(metrics, "max_pairwise_score_gap[", attribute)
    rationale = [
        m for m in metrics
        if m.family == "rationale" and m.attribute == attribute
        and m.value == m.value and (m.passed is False or m.value > 0)
    ]
    if not gaps or not rationale:
        return []
    if abs(_value(gaps[0])) >= min_practical_effect:
        return []
    notable = [m for m in rationale if m.significant or (m.passed is False)]
    if not notable:
        return []
    return [
        {
            "type": "score_vs_rationale",
            "attribute": attribute,
            "severity": "important",
            "metrics": [gaps[0].name] + [m.name for m in notable],
            "explanation": (
                f"Scores for {attribute} are near-parity, but the *language* of the "
                "rationales differs systematically across groups. Since the rationale is "
                "what a recruiter reads, a difference here can change the human decision "
                "even with identical numbers — and LLM recommendations are known to move "
                "human hiring judgements."
            ),
            "recommendation": (
                "Treat the rationale asymmetry as a finding in its own right. If explanation "
                "cards are shown to recruiters, this is the channel through which bias "
                "reaches the decision, and score-only auditing would certify the system as "
                "fair."
            ),
            "citation": (
                "Wilson, Sim, Gueorguieva & Caliskan (2025), 'No thoughts just AI: Biased LLM "
                "hiring recommendations alter human decision making'; Sivakaminathan & Musi "
                "(2025)."
            ),
        }
    ]


def _marginal_vs_intersectional(
    metrics: Sequence[MetricResult], attribute: str, min_practical_effect: float
) -> List[Dict[str, Any]]:
    marginal = _find(metrics, "max_pairwise_score_gap[", attribute)
    intersectional = _find(metrics, "intersectional_worst_cell_gap[")
    if not marginal or not intersectional:
        return []
    if attribute not in intersectional[0].attribute:
        return []
    if abs(_value(marginal[0])) >= min_practical_effect:
        return []
    if _value(intersectional[0]) < 2 * min_practical_effect:
        return []
    return [
        {
            "type": "marginal_vs_intersectional",
            "attribute": attribute,
            "severity": "important",
            "metrics": [marginal[0].name, intersectional[0].name],
            "explanation": (
                f"The marginal gap for {attribute} is small ({_value(marginal[0]):.2f} points) "
                f"while the worst intersectional subgroup gap is "
                f"{_value(intersectional[0]):.2f} points. Averaging over the other attribute "
                "has cancelled the effect: the bias is concentrated in a specific "
                "combination and is invisible one attribute at a time."
            ),
            "recommendation": (
                "Report the intersectional cells, not just the marginals. An audit that "
                "checks gender and ethnicity separately can certify a system that penalises "
                "one specific subgroup."
            ),
            "citation": (
                "Wilson & Caliskan (2024); Webster (2025), 'Fairness is not enough: Auditing "
                "competence and intersectional bias in AI-powered resume screening'."
            ),
        }
    ]


def _validity_collapse(
    metrics: Sequence[MetricResult], min_practical_effect: float
) -> List[Dict[str, Any]]:
    """The scorer ignores even job-relevant signal, so its 'fairness' is inertness."""
    prestige = [
        m for m in metrics
        if m.family == "counterfactual" and "education_prestige" in m.name
    ]
    swings = _find(metrics, "counterfactual_max_swing[all]")
    if not swings:
        return []
    if _value(swings[0]) > min_practical_effect:
        return []
    if prestige and _value(prestige[0]) > min_practical_effect:
        return []
    return [
        {
            "type": "validity_collapse",
            "attribute": "all",
            "severity": "critical",
            "metrics": [m.name for m in swings + prestige],
            "explanation": (
                "The scorer barely moves for ANY cue, including the "
                "``education_prestige`` validity probe, which is supposed to be a positive "
                "control. An apparently perfect fairness result from an inert scorer is "
                "meaningless: a constant function is unbiased and also useless."
            ),
            "recommendation": (
                "Do not report these fairness results. Check the Phase-2 sanity check "
                "(kendall_tau of score against expected fit), the parse success rate and the "
                "prompt, then re-run. Fairness numbers are only interpretable once the scorer "
                "is shown to discriminate on job-relevant content."
            ),
            "citation": (
                "Castleman, Shen, Metevier, Springer & Korolova (2026), 'Measuring validity in "
                "LLM-based resume screening'."
            ),
        }
    ]


def _power_masquerade(
    metrics: Sequence[MetricResult], attribute: str, min_practical_effect: float
) -> List[Dict[str, Any]]:
    """A null result that is really a statement about sample size."""
    out: List[Dict[str, Any]] = []
    for metric in metrics:
        if metric.family != "group_parity" or metric.attribute != attribute:
            continue
        if not metric.name.startswith("score_gap["):
            continue
        detail = metric.detail or {}
        mde = detail.get("minimum_detectable_effect")
        if mde is None or mde != mde:
            continue
        if metric.significant:
            continue
        if mde <= min_practical_effect:
            continue
        needed = detail.get("blocks_needed_for_1pt")
        out.append(
            {
                "type": "power_masquerade",
                "attribute": attribute,
                "severity": "important",
                "metrics": [metric.name],
                "explanation": (
                    f"{metric.name} is not statistically significant, but the minimum "
                    f"detectable effect at this sample size is {mde:.2f} points — larger "
                    f"than the {min_practical_effect:g}-point threshold that would count as "
                    "practically meaningful. A gap of practical size could therefore have "
                    "existed and gone undetected. This is absence of evidence, not evidence "
                    "of absence."
                ),
                "recommendation": (
                    "Report this as underpowered rather than as 'no bias'."
                    + (
                        f" Detecting a {min_practical_effect:g}-point gap at 80% power would "
                        f"need about {needed} resume blocks."
                        if isinstance(needed, int) and needed > 0 else ""
                    )
                ),
                "citation": "Standard power analysis for paired designs.",
            }
        )
    # One instance per attribute is enough to make the point.
    return out[:1]


def _noise_floor_conflict(
    metrics: Sequence[MetricResult], reliability: Dict[str, Any]
) -> List[Dict[str, Any]]:
    if not reliability.get("has_repeats"):
        return [
            {
                "type": "missing_noise_floor",
                "attribute": "all",
                "severity": "important",
                "metrics": [],
                "explanation": (
                    "This run contains no repeated decodes, so the model's own scoring noise "
                    "was never measured. Every effect below is reported without a baseline "
                    "for how much the score moves on identical input."
                ),
                "recommendation": (
                    "Re-run with llm.self_consistency_samples >= 3 and "
                    "llm.self_consistency_temperature > 0. Any cue effect smaller than the "
                    "noise floor is not a finding."
                ),
                "citation": "",
            }
        ]
    noise_sd = reliability.get("noise_floor_sd") or 0.0
    if noise_sd <= 0:
        return []
    offenders = [
        m for m in metrics
        if m.family in ("group_parity", "counterfactual")
        and m.value == m.value
        and abs(m.value) > 0
        and abs(m.value) < noise_sd
        and m.significant
    ]
    if not offenders:
        return []
    return [
        {
            "type": "significant_below_noise_floor",
            "attribute": "all",
            "severity": "important",
            "metrics": [m.name for m in offenders][:6],
            "explanation": (
                f"{len(offenders)} metric(s) are statistically significant but smaller than "
                f"the {noise_sd:.2f}-point decoding noise floor. With enough repeated "
                "measurements any systematic offset becomes significant, however small; "
                "significance is not the same as meaningfulness."
            ),
            "recommendation": (
                "Report these with the noise floor stated next to them and do not lead with "
                "them. The practical-effect threshold, not the p-value, is what makes a gap "
                "actionable here."
            ),
            "citation": "",
        }
    ]


def _noise_dominated_failures(
    metrics: Sequence[MetricResult], reliability: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Threshold failures that the model's own decoding noise already explains.

    The build plan's central warning, made mechanical: if cue-driven variation is
    smaller than repeat-to-repeat variation, then failing counterfactual and
    allocational thresholds are the *scorer's instability* showing up, not bias.
    Reporting them as bias would be the exact error the plan warns against.
    """
    ratio = reliability.get("cue_effect_to_noise_ratio")
    if not reliability.get("has_repeats") or ratio is None or ratio >= 1.0:
        return []

    failed = [
        m for m in metrics
        if m.passed is False and m.family in ("counterfactual", "allocational", "ranking")
    ]
    if not failed:
        return []

    flip_metrics = [m for m in metrics if m.name.startswith("decision_flip_rate[")]
    noise_flip = None
    for metric in flip_metrics:
        detail = metric.detail or {}
        if isinstance(detail, dict) and detail.get("decision_flip_rate_from_noise") is not None:
            noise_flip = detail["decision_flip_rate_from_noise"]
            break

    return [
        {
            "type": "noise_dominated_failures",
            "attribute": "all",
            "severity": "important",
            "metrics": [m.name for m in failed][:8],
            "explanation": (
                f"{len(failed)} counterfactual/allocational/ranking threshold(s) failed, but "
                f"cue-driven score variation is only {ratio:.2f}x the model's decoding noise "
                f"(noise floor {reliability.get('noise_floor_sd')} points). At that ratio the "
                "failures are largely the scorer disagreeing with itself rather than reacting "
                "to the cue: with candidates bunched together, repeat-to-repeat variation is "
                "enough to reorder a shortlist and to move a borderline candidate across a "
                "decision threshold."
                + (
                    f" Consistent with this, {100 * noise_flip:.1f}% of candidates already "
                    "flip verdict across repeats of an identical prompt."
                    if noise_flip else ""
                )
            ),
            "recommendation": (
                "Do not report these as bias. Either reduce the noise (score at temperature 0, "
                "or raise llm.self_consistency_samples and average) or enlarge the corpus so "
                "the cue effect can be separated from the noise. Report the failures as a "
                "*reliability* finding, which is what they are: a screener this unstable is "
                "unfit for deployment regardless of whether it is biased."
            ),
            "citation": "",
        }
    ]


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def summarise_conflicts(conflicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_severity: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    for conflict in conflicts:
        by_severity[conflict["severity"]] = by_severity.get(conflict["severity"], 0) + 1
        by_type[conflict["type"]] = by_type.get(conflict["type"], 0) + 1
    return {
        "n_conflicts": len(conflicts),
        "by_severity": by_severity,
        "by_type": by_type,
        "has_critical": any(c["severity"] == "critical" for c in conflicts),
        "guidance": (
            "Metric disagreement is expected in fairness auditing and several of these "
            "disagreements are theorems rather than bugs. What the write-up must do is name "
            "each disagreement, attribute it to its cause, and state which criterion the "
            "system prioritises. A 'critical' entry is different: it means the results "
            "should not be reported at all until the underlying problem is fixed."
        ),
    }


def annotate_report(report: AuditReport, *, min_practical_effect: float = 1.0) -> AuditReport:
    """Attach conflict analysis to an :class:`AuditReport` in place."""
    report.conflicts = detect_conflicts(
        report.metrics, reliability=report.reliability,
        min_practical_effect=min_practical_effect,
    )
    summary = summarise_conflicts(report.conflicts)
    if summary["has_critical"]:
        report.warnings.append(
            "CRITICAL conflict detected: the scorer may not be discriminating on "
            "job-relevant content. Fairness results should not be reported until resolved."
        )
    return report
