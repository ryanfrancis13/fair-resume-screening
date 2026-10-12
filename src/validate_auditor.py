"""Validating the auditor itself.

Every fairness audit produces numbers.  Almost none of them come with an answer
to the obvious question: *does this measurement instrument work?*  An audit that
misses a real 3-point gap, or that reports one where none exists, is worse than
no audit — it produces a certificate.

Because :class:`~frs.llm.mock.MockProvider` can inject a bias of exactly known
size, that question is answerable here. Two conditions are run through the *full*
pipeline — perturbation, scoring, metrics, corrections and all:

**Null condition** (``bias = 0``). Nothing but decoding noise. A well-behaved
suite should report no significant protected-attribute effect. Anything it does
report is a **false positive**, and the rate is a direct measurement of the
suite's specificity.

**Injected condition** (``bias = -3.0`` for one demographic level). A known
effect. The suite should recover it in the right direction, at roughly the right
magnitude, and flag it. Failure to do so is a **false negative** and measures
sensitivity.

The expected magnitude is not guessed. The mock spreads its injected delta across
rubric dimensions with known weights, so the transmitted effect on the overall
score is ``bias × BIAS_TRANSMISSION`` — a constant computed from those weights
rather than hard-coded, so it stays correct if the rubric changes.

The signals are recovered from the resume **text**, using the same published name
lexicons a real model would see. If the mock were simply told which variant was
which, this would only confirm that the pipeline's bookkeeping is self-consistent.
Reading them out of the prose puts the whole chain under test: cue rendered into
text → detected from text → score shifted → gap recovered → significance
correctly assigned.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

from .config import Config
from .logging_setup import log_event

logger = logging.getLogger("frs.validate_auditor")

#: Injected gap, in score points, before rubric transmission.
DEFAULT_INJECTED_BIAS = -3.0
#: Tolerance on the recovered magnitude, as a fraction of the expected value.
MAGNITUDE_TOLERANCE = 0.5


def _condition_config(
    base: Config,
    *,
    bias: Dict[str, Dict[str, float]],
    samples: int,
    noise_sd: float,
) -> Config:
    cfg = copy.deepcopy(base)
    cfg.llm.provider = "mock"
    cfg.llm.model = "mock-rubric-v1"
    cfg.llm.mock_bias = bias
    cfg.llm.mock_noise_sd = noise_sd
    cfg.llm.self_consistency_samples = samples
    cfg.llm.self_consistency_temperature = 0.4
    cfg.llm.requests_per_minute = 10**6
    cfg.llm.max_concurrency = 8
    # The oracle needs the name family (it carries gender and race_ethnicity) and
    # one non-name family, to check the suite finds effects in both channels.
    cfg.perturbation.cue_families = ["name", "pronoun", "education_prestige"]
    cfg.perturbation.name_list = "bertrand_mullainathan_2004"
    cfg.perturbation.levels_per_family = 8
    cfg.perturbation.design = "single"
    cfg.mitigation.strategies = []
    cfg.report.formats = ["json"]
    cfg.report.make_plots = False
    cfg.report.explanation_cards = False
    cfg.report.model_card = False
    return cfg


def _run_condition(
    cfg: Config, label: str
) -> Dict[str, Any]:
    from .fairness.suite import audit_summary, run_audit
    from .pipeline.run import (
        generate_variants,
        prepare_data,
        score_variants,
        start_run,
    )

    state = start_run(cfg, prefix=f"validate-{label}")
    prepare_data(state)
    generate_variants(state)
    score_variants(state)
    frame = state.frames_by_arm["baseline"]
    report = run_audit(frame, cfg, run_id=state.run_id)
    return {
        "label": label,
        "run_id": state.run_id,
        "report": report,
        "summary": audit_summary(report),
        "frame": frame,
    }


def _protected_gap_metrics(report: Any, attribute: str) -> List[Any]:
    return [
        m for m in report.metrics
        if m.family == "group_parity"
        and m.attribute == attribute
        and m.name.startswith("score_gap[")
    ]


def validate_auditor(
    base_cfg: Optional[Config] = None,
    *,
    injected_bias: float = DEFAULT_INJECTED_BIAS,
    attribute: str = "gender",
    level: str = "female",
    samples: int = 3,
    noise_sd: float = 1.2,
) -> Dict[str, Any]:
    """Run the null and injected conditions and grade the audit suite.

    Returns a verdict dict; ``passed`` is true only if the suite is both
    sensitive (recovers the injected effect) and specific (stays quiet under the
    null).
    """
    from .llm.mock import BIAS_TRANSMISSION

    base = base_cfg or Config.load(apply_env=False)
    expected = injected_bias * BIAS_TRANSMISSION

    null_cfg = _condition_config(base, bias={}, samples=samples, noise_sd=noise_sd)
    injected_cfg = _condition_config(
        base, bias={attribute: {level: injected_bias}}, samples=samples, noise_sd=noise_sd
    )

    log_event(
        logger, logging.INFO, "validating audit suite",
        injected_bias=injected_bias, expected_transmitted=round(expected, 3),
        attribute=attribute, cue_level=level,
    )

    null = _run_condition(null_cfg, "null")
    injected = _run_condition(injected_cfg, "injected")

    checks: List[Dict[str, Any]] = []

    # ---- specificity: nothing significant under the null ------------------
    null_significant = [
        m for m in null["report"].metrics
        if m.family in ("group_parity", "intersectional")
        and m.significant
        and abs(m.value) >= base.fairness.min_practical_effect
    ]
    checks.append(
        {
            "check": "specificity_no_false_positives",
            "passed": len(null_significant) == 0,
            "detail": {
                "n_false_positives": len(null_significant),
                "metrics": [m.name for m in null_significant][:6],
            },
            "explanation": (
                "Under a provider with zero injected bias, no protected-attribute gap should "
                "be both significant after FDR correction and above the practical threshold."
            ),
        }
    )

    # ---- sensitivity: the injected effect is recovered ---------------------
    gaps = _protected_gap_metrics(injected["report"], attribute)
    target = next((m for m in gaps if f":{level}_vs_" in m.name), None)
    recovered = target.value if target is not None else float("nan")
    direction_ok = (
        recovered == recovered and (recovered < 0) == (expected < 0)
    )
    magnitude_ok = (
        recovered == recovered
        and abs(abs(recovered) - abs(expected)) <= MAGNITUDE_TOLERANCE * abs(expected)
    )
    checks.append(
        {
            "check": "sensitivity_recovers_injected_effect",
            "passed": bool(target is not None and direction_ok and magnitude_ok),
            "detail": {
                "metric": target.name if target is not None else None,
                "injected_bias": injected_bias,
                "expected_transmitted_effect": round(expected, 3),
                "recovered_effect": round(recovered, 3) if recovered == recovered else None,
                "ci": [
                    round(target.ci_low, 3) if target is not None and target.ci_low == target.ci_low else None,
                    round(target.ci_high, 3) if target is not None and target.ci_high == target.ci_high else None,
                ],
                "direction_correct": bool(direction_ok),
                "magnitude_within_tolerance": bool(magnitude_ok),
                "tolerance_fraction": MAGNITUDE_TOLERANCE,
            },
            "explanation": (
                f"The mock injects {injected_bias:+.2f} points for {attribute}={level}; the "
                f"rubric transmits {expected:+.2f} to the overall score. The suite should "
                "recover that magnitude and sign."
            ),
        }
    )
    checks.append(
        {
            "check": "sensitivity_flags_significance",
            "passed": bool(target is not None and target.significant),
            "detail": {
                "p_value": round(target.p_value, 6) if target is not None and target.p_value == target.p_value else None,
                "q_value_bh": round(target.q_value, 6) if target is not None and target.q_value == target.q_value else None,
            },
            "explanation": (
                "The recovered effect should survive Benjamini-Hochberg correction across "
                "the whole metric suite, not just an uncorrected test."
            ),
        }
    )

    # ---- the validity probe must move (positive control) ------------------
    prestige = next(
        (m for m in injected["report"].metrics
         if m.name.startswith("counterfactual_max_swing[education_prestige]")),
        None,
    )
    checks.append(
        {
            "check": "validity_probe_responds",
            "passed": bool(prestige is not None and prestige.value > 0.5),
            "detail": {
                "metric": prestige.name if prestige is not None else None,
                "value": round(prestige.value, 3) if prestige is not None else None,
            },
            "explanation": (
                "The education_prestige validity probe should move the score. A scorer "
                "invariant to legitimate credential signal is inert, and an inert scorer "
                "makes every fairness result meaningless."
            ),
        }
    )

    # ---- noise floor is estimated and roughly right ----------------------
    from .llm.mock import NOISE_TRANSMISSION

    expected_noise = noise_sd * NOISE_TRANSMISSION
    measured_noise = injected["report"].reliability.get("noise_floor_sd")
    noise_ok = (
        measured_noise is not None
        and measured_noise > 0
        and abs(measured_noise - expected_noise) <= max(0.5, 0.6 * expected_noise)
    )
    checks.append(
        {
            "check": "noise_floor_estimated",
            "passed": bool(noise_ok),
            "detail": {
                "injected_noise_sd": noise_sd,
                "expected_transmitted_noise_sd": round(expected_noise, 3),
                "measured_noise_floor_sd": measured_noise,
                "has_repeats": injected["report"].reliability.get("has_repeats"),
            },
            "explanation": (
                "Repeated decodes should let the suite recover the injected decoding noise, "
                "which is the yardstick every cue effect is compared against."
            ),
        }
    )

    passed = all(c["passed"] for c in checks)
    verdict = {
        "passed": passed,
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "checks": checks,
        "conditions": {
            "null": {
                "run_id": null["run_id"],
                "n_records": null["summary"]["n_records"],
                "n_significant_after_fdr": null["summary"]["n_significant_after_fdr"],
                "reliability": null["summary"]["reliability"],
            },
            "injected": {
                "run_id": injected["run_id"],
                "n_records": injected["summary"]["n_records"],
                "n_significant_after_fdr": injected["summary"]["n_significant_after_fdr"],
                "reliability": injected["summary"]["reliability"],
            },
        },
        "interpretation": (
            "The audit suite recovered a known injected bias of the right sign and magnitude "
            "and stayed silent under a no-bias null. Its findings on a real model can be "
            "read as measurements rather than as artifacts of the measurement."
            if passed else
            "The audit suite FAILED at least one oracle check. Fix the failing check before "
            "trusting any fairness number this pipeline produces on a real model — a suite "
            "that cannot recover a known effect, or that invents one, is not evidence."
        ),
    }
    log_event(
        logger, logging.INFO, "auditor validation complete",
        passed=passed, n_passed=verdict["n_passed"], n_checks=len(checks),
    )
    return verdict


def render_validation(verdict: Dict[str, Any]) -> str:
    """Human-readable validation summary for the CLI."""
    lines = [
        "",
        "=" * 78,
        "AUDITOR VALIDATION — can this measurement instrument be trusted?",
        "=" * 78,
        "",
    ]
    for check in verdict["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"[{mark}] {check['check']}")
        lines.append(f"       {check['explanation']}")
        for key, value in check["detail"].items():
            lines.append(f"       - {key}: {value}")
        lines.append("")
    lines += [
        "-" * 78,
        f"RESULT: {verdict['n_passed']}/{verdict['n_checks']} checks passed "
        f"({'PASS' if verdict['passed'] else 'FAIL'})",
        "",
        verdict["interpretation"],
        "=" * 78,
        "",
    ]
    return "\n".join(lines)
