"""Bias-mitigation strategies.

The stated research gap is that existing systems either *screen* or *check for
bias*, rarely both.  Measuring bias and stopping there would only half-close that
gap: an audit that produces a number nobody can act on is a diagnosis without a
treatment.  This module supplies the treatments, and every one of them is
evaluated on the same audit harness that found the problem — including its cost,
because a mitigation that removes bias by destroying the signal is not a success.

Strategies split into two kinds.

**Pre-inference arms** (need their own scoring pass, because the model sees
different input):

``blind_screening``
    Redact identity and proxy cues before scoring. The oldest intervention in
    hiring, and the one the leakage report shows is incomplete.
``debias_prompt``
    Instruct the model to ignore demographic signals. Included *because* the
    literature repeatedly finds it insufficient — a mitigation whose
    effectiveness is assumed cannot be evaluated.

**Post-hoc arms** (transform the existing score table, no extra API calls):

``counterfactual_marginalization``
    Replace a candidate's score with a robust aggregate over their own
    counterfactual variants. Marginalising over the cue removes its influence by
    construction; a trimmed mean also limits the damage one extreme variant can do.
``sensitivity_gated_review``
    Route candidates whose score is unstable across variants to a human, and
    auto-decide the rest. Selective prediction: it does not make the model fairer,
    it stops the model deciding where it is demonstrably unreliable.
``cue_effect_debiasing``
    Estimate each cue's additive effect from an audit run and subtract the effect
    of the cues *detected in the text*. Group-blind at serving time: nobody is
    ever asked their demographics, and nothing is adjusted by group membership —
    only the model's measured sensitivity to specific tokens is removed. The legal
    caveat is spelled out in :data:`LEGAL_NOTE` and travels with the results,
    because group-conditional score adjustment in US employment testing is
    restricted by Title VII §106(l) and the distinction matters.
``evidence_gate``
    Withhold scores whose stated reasons are not locatable in the resume. An
    ungrounded score cannot be audited or contested, so it should not be used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_setup import log_event
from ..utils.frames import text_column

logger = logging.getLogger("frs.mitigate")

LEGAL_NOTE = (
    "cue_effect_debiasing adjusts for the model's measured sensitivity to specific text "
    "cues, never for a candidate's group membership: the candidate's demographics are "
    "never observed, requested or inferred at serving time. This distinction is "
    "deliberate. In US employment testing, adjusting scores on the basis of race, "
    "colour, religion, sex or national origin is restricted by Title VII as amended "
    "(42 U.S.C. 2000e-2(l)), so a per-group score adjustment would be legally hazardous "
    "even where it improves a parity metric. Removing a model's measured reaction to a "
    "token is a different operation from re-scoring a person because of their group. "
    "This is an engineering note and not legal advice; deployment needs counsel."
)

PRE_INFERENCE_STRATEGIES = ("blind_screening", "debias_prompt")
POST_HOC_STRATEGIES = (
    "counterfactual_marginalization",
    "sensitivity_gated_review",
    "cue_effect_debiasing",
    "evidence_gate",
)
ALL_STRATEGIES = PRE_INFERENCE_STRATEGIES + POST_HOC_STRATEGIES


@dataclass
class ArmSpec:
    """A scoring arm: a name plus the config changes that define it."""

    name: str
    description: str
    overrides: Dict[str, Any] = field(default_factory=dict)


def pre_inference_arms(cfg: Config, strategies: Sequence[str]) -> List[ArmSpec]:
    """Config overrides for each requested pre-inference arm."""
    arms: List[ArmSpec] = []
    for strategy in strategies:
        if strategy == "blind_screening":
            arms.append(
                ArmSpec(
                    name="blind_screening",
                    description=(
                        f"identity and proxy cues redacted at level "
                        f"'{cfg.screening.anonymize_level}' before scoring"
                    ),
                    overrides={
                        "screening.anonymize": True,
                        "screening.anonymize_level": cfg.screening.anonymize_level,
                    },
                )
            )
        elif strategy == "debias_prompt":
            arms.append(
                ArmSpec(
                    name="debias_prompt",
                    description="explicit fairness constraint appended to the prompt",
                    overrides={"screening.debias_instruction": True},
                )
            )
    return arms


# --------------------------------------------------------------------------- #
# post-hoc transforms
# --------------------------------------------------------------------------- #
def _aggregate(values: np.ndarray, estimator: str, trim_fraction: float) -> float:
    if len(values) == 0:
        return float("nan")
    if estimator == "mean":
        return float(np.mean(values))
    if estimator == "median":
        return float(np.median(values))
    if estimator == "trimmed_mean":
        ordered = np.sort(values)
        cut = int(np.floor(len(ordered) * trim_fraction))
        # Never trim away the whole sample: with 3 variants a 25% trim would
        # otherwise leave nothing.
        if 2 * cut >= len(ordered):
            cut = max(0, (len(ordered) - 1) // 2)
        kept = ordered[cut: len(ordered) - cut] if cut else ordered
        return float(np.mean(kept)) if len(kept) else float(np.mean(ordered))
    raise ValueError(f"unknown estimator {estimator!r}")


def counterfactual_marginalization(
    frame: pd.DataFrame,
    cfg: Config,
    *,
    cue_family: str = "name",
    value_column: str = "overall_score",
    arm_name: str = "counterfactual_marginalization",
) -> pd.DataFrame:
    """Replace each candidate's score with a robust aggregate over their variants.

    The candidate's *served* score becomes an estimate marginalised over the cue,
    so by construction it no longer depends on which cue value their document
    happened to carry. In production this costs ``k`` calls per candidate — a real
    trade-off, which is why ``mitigation.marginalization_variants`` is a knob and
    why the evaluation reports the cost alongside the benefit.
    """
    data = frame[frame[value_column].notna()].copy()
    if "parse_ok" in data.columns:
        data = data[data["parse_ok"].astype(bool)]
    if data.empty:
        return pd.DataFrame()

    mask = text_column(data, "cue_families").str.contains(cue_family, regex=False)
    if "is_baseline" in data.columns:
        mask = mask | data["is_baseline"].astype(bool)
    pool = data[mask]
    if pool.empty:
        log_event(
            logger, logging.WARNING,
            "marginalization requested but no variants of the cue family are present",
            cue_family=cue_family,
        )
        return pd.DataFrame()

    estimator = cfg.mitigation.marginalization_estimator
    trim = cfg.mitigation.trim_fraction
    k = max(1, cfg.mitigation.marginalization_variants)

    rows: List[Dict[str, Any]] = []
    for (job_id, resume_id), cell in pool.groupby(["job_id", "base_resume_id"]):
        variant_means = cell.groupby("variant_id")[value_column].mean()
        # Deterministic subsample: sorting by variant id keeps the arm
        # reproducible across runs instead of depending on grouping order.
        chosen = variant_means.sort_index().iloc[:k]
        aggregated = _aggregate(chosen.to_numpy(dtype=float), estimator, trim)
        template = cell.iloc[0].to_dict()
        template.update(
            {
                "arm": arm_name,
                "variant_id": f"{resume_id}__marginalised",
                "base_resume_id": str(resume_id),
                "job_id": str(job_id),
                "is_baseline": True,
                "cue_families": "",
                "cue_ids": "",
                "sample_index": 0,
                value_column: aggregated,
                "recommendation": _recommendation(aggregated, cfg),
                "n_marginalised_variants": int(len(chosen)),
                "marginalization_spread": float(
                    chosen.max() - chosen.min()
                ) if len(chosen) > 1 else 0.0,
            }
        )
        # The served score no longer belongs to any demographic level.
        for column in [c for c in template if c.startswith("signal_")]:
            template[column] = None
        rows.append(template)

    return pd.DataFrame(rows)


def sensitivity_gated_review(
    frame: pd.DataFrame,
    cfg: Config,
    *,
    value_column: str = "overall_score",
    arm_name: str = "sensitivity_gated_review",
) -> Dict[str, Any]:
    """Auto-decide stable candidates; route unstable ones to a human.

    Returns the gated table plus coverage statistics. The honest way to report
    this is jointly: fairness *among auto-decided candidates* is only meaningful
    next to the fraction of candidates the system declined to decide, since
    abstaining on everything would trivially produce perfect fairness.
    """
    data = frame[frame[value_column].notna()].copy()
    if "parse_ok" in data.columns:
        data = data[data["parse_ok"].astype(bool)]
    if data.empty:
        return {"error": "no usable rows"}

    threshold = cfg.mitigation.review_sensitivity_threshold
    rows: List[Dict[str, Any]] = []
    flagged = 0
    total = 0
    for (job_id, resume_id), cell in data.groupby(["job_id", "base_resume_id"]):
        variant_means = cell.groupby("variant_id")[value_column].mean()
        if len(variant_means) < 2:
            continue
        total += 1
        swing = float(variant_means.max() - variant_means.min())
        needs_review = swing > threshold
        flagged += int(needs_review)
        served = float(np.median(variant_means.to_numpy(dtype=float)))
        template = cell.iloc[0].to_dict()
        template.update(
            {
                "arm": arm_name,
                "variant_id": f"{resume_id}__gated",
                "base_resume_id": str(resume_id),
                "job_id": str(job_id),
                "is_baseline": True,
                "cue_families": "",
                "cue_ids": "",
                "sample_index": 0,
                value_column: served,
                "recommendation": (
                    "human_review" if needs_review else _recommendation(served, cfg)
                ),
                "cue_sensitivity_swing": round(swing, 4),
                "routed_to_human": bool(needs_review),
            }
        )
        for column in [c for c in template if c.startswith("signal_")]:
            template[column] = None
        rows.append(template)

    if not rows:
        return {"error": "no candidate had 2+ variants; gating undefined"}

    gated = pd.DataFrame(rows)
    review_fraction = flagged / total if total else 0.0
    over_budget = review_fraction > cfg.mitigation.max_review_fraction
    return {
        "frame": gated,
        "threshold": threshold,
        "n_candidates": total,
        "n_routed_to_human": flagged,
        "review_fraction": round(review_fraction, 4),
        "max_review_fraction": cfg.mitigation.max_review_fraction,
        "within_review_budget": not over_budget,
        "note": (
            f"{100 * review_fraction:.1f}% of candidates had a cue-driven swing above "
            f"{threshold} points and were routed to a human. Fairness among the "
            "auto-decided remainder must be read together with this coverage figure: "
            "abstaining on everything would look perfectly fair."
        ),
    }


def estimate_cue_effects(
    frame: pd.DataFrame,
    *,
    value_column: str = "overall_score",
    min_candidates: int = 4,
) -> Dict[str, Any]:
    """Additive effect of each cue level, estimated within candidate blocks.

    Within-candidate differencing removes every resume-level confound, so the
    estimate is the model's own reaction to the cue and nothing else. These
    coefficients are what ``cue_effect_debiasing`` subtracts.
    """
    from ..fairness.counterfactual import cue_effect_table

    table = cue_effect_table(frame, value_column=value_column)
    if table.empty:
        return {"error": "no baseline/variant pairs available to estimate cue effects"}

    effects: Dict[str, Dict[str, Any]] = {}
    for _, row in table.iterrows():
        if int(row["n_candidates"]) < min_candidates:
            continue
        effects[str(row["cue_ids"])] = {
            "cue_families": str(row["cue_families"]),
            "estimate": float(row["mean_delta"]),
            "se": float(row["se_delta"]) if row["se_delta"] == row["se_delta"] else None,
            "n_candidates": int(row["n_candidates"]),
        }
    return {
        "n_cue_levels": len(effects),
        "effects": effects,
        "min_candidates": min_candidates,
        "method": (
            "mean within-candidate deviation from the candidate's own baseline score"
        ),
        "legal_note": LEGAL_NOTE,
    }


def cue_effect_debiasing(
    frame: pd.DataFrame,
    cfg: Config,
    *,
    effects: Optional[Dict[str, Any]] = None,
    value_column: str = "overall_score",
    arm_name: str = "cue_effect_debiasing",
) -> pd.DataFrame:
    """Subtract each row's estimated cue effect from its score.

    Applied to the *same* table the effects were estimated from, this is an
    in-sample correction and will look better than it would in deployment. The
    evaluation says so, and ``mitigation.cue_effect_source_run`` exists precisely
    so effects can be estimated on one run and applied to another — the honest
    out-of-sample version.
    """
    estimated = effects or estimate_cue_effects(frame, value_column=value_column)
    if "error" in estimated:
        return pd.DataFrame()
    table = estimated["effects"]

    data = frame[frame[value_column].notna()].copy()
    if "parse_ok" in data.columns:
        data = data[data["parse_ok"].astype(bool)]
    if data.empty:
        return pd.DataFrame()

    adjustments: List[float] = []
    for cue_ids in text_column(data, "cue_ids"):
        entry = table.get(cue_ids)
        adjustments.append(float(entry["estimate"]) if entry else 0.0)

    out = data.copy()
    out["cue_effect_adjustment"] = adjustments
    out[value_column] = out[value_column].astype(float) - out["cue_effect_adjustment"]
    out["arm"] = arm_name
    out["recommendation"] = [_recommendation(v, cfg) for v in out[value_column]]
    out["in_sample_correction"] = effects is None
    return out


def evidence_gate(
    frame: pd.DataFrame,
    cfg: Config,
    *,
    arm_name: str = "evidence_gate",
) -> Dict[str, Any]:
    """Withhold scores whose evidence quotes are not locatable in the resume."""
    if "evidence_support_rate" not in frame.columns:
        return {"error": "frame lacks evidence_support_rate"}
    data = frame.copy()
    if "parse_ok" in data.columns:
        data = data[data["parse_ok"].astype(bool)]
    threshold = cfg.screening.min_evidence_support_rate
    rates = data["evidence_support_rate"].astype(float)
    grounded = data[rates >= threshold].copy()
    grounded["arm"] = arm_name
    withheld = int(len(data) - len(grounded))
    return {
        "frame": grounded,
        "threshold": threshold,
        "n_input": int(len(data)),
        "n_retained": int(len(grounded)),
        "n_withheld": withheld,
        "withheld_fraction": round(withheld / len(data), 4) if len(data) else 0.0,
        "note": (
            "Scores whose stated reasons could not be located in the resume are withheld. "
            "An ungrounded score cannot be audited or contested by the candidate, so it is "
            "not fit to use in a decision."
        ),
    }


def _recommendation(score: float, cfg: Config) -> str:
    if score != score:
        return ""
    if score >= cfg.screening.advance_threshold:
        return "advance"
    if score >= cfg.screening.borderline_threshold:
        return "borderline"
    return "reject"


POST_HOC_REGISTRY: Dict[str, Callable[..., Any]] = {
    "counterfactual_marginalization": counterfactual_marginalization,
    "sensitivity_gated_review": sensitivity_gated_review,
    "cue_effect_debiasing": cue_effect_debiasing,
    "evidence_gate": evidence_gate,
}


def describe_strategies() -> List[Dict[str, str]]:
    """Catalogue for ``frs mitigate --list`` and the model card."""
    return [
        {
            "strategy": "blind_screening",
            "kind": "pre-inference",
            "mechanism": "redact identity and proxy cues before the model sees the resume",
            "cost": "one extra scoring pass; loses genuinely relevant context such as location",
            "known_limitation": (
                "incomplete by construction — the leakage report quantifies which cues survive"
            ),
        },
        {
            "strategy": "debias_prompt",
            "kind": "pre-inference",
            "mechanism": "instruct the model to ignore demographic signals",
            "cost": "one extra scoring pass; a few hundred extra prompt tokens",
            "known_limitation": (
                "repeatedly found insufficient in the literature; included so its "
                "effectiveness is measured rather than assumed"
            ),
        },
        {
            "strategy": "counterfactual_marginalization",
            "kind": "post-hoc",
            "mechanism": "serve a trimmed mean over the candidate's own cue variants",
            "cost": "k API calls per candidate in production",
            "known_limitation": (
                "removes sensitivity to the marginalised cue only; other cues are untouched"
            ),
        },
        {
            "strategy": "sensitivity_gated_review",
            "kind": "post-hoc",
            "mechanism": "route cue-sensitive candidates to a human reviewer",
            "cost": "human review time proportional to the flagged fraction",
            "known_limitation": (
                "does not make the model fairer; it limits where the model decides, and "
                "its fairness numbers are only meaningful next to its coverage"
            ),
        },
        {
            "strategy": "cue_effect_debiasing",
            "kind": "post-hoc",
            "mechanism": "subtract the audited additive effect of cues detected in the text",
            "cost": "needs an audit run to estimate effects; effects drift with the model",
            "known_limitation": (
                "in-sample correction unless effects come from a separate run; "
                "see LEGAL_NOTE on why this is group-blind by design"
            ),
        },
        {
            "strategy": "evidence_gate",
            "kind": "post-hoc",
            "mechanism": "withhold scores whose quotes are not locatable in the resume",
            "cost": "reduced coverage",
            "known_limitation": "improves auditability rather than parity directly",
        },
    ]
