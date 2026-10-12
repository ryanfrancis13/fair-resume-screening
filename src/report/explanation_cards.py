"""Per-candidate explanation cards.

Cui (2026) argues for "evidence-grounded LLM explanation cards" in bias-audited
screening, and the reason is practical: a recruiter cannot act responsibly on a
number, and a candidate cannot contest one.  A card makes the decision
inspectable by showing four things together:

* the rubric breakdown, not just the total;
* the quotations that justify each dimension, each marked **verified** or
  **unverified** against the resume text;
* the requirements the resume did not evidence;
* the **cue-sensitivity band** — how much this candidate's score moved across
  their own counterfactual variants, and whether that swing crossed a decision
  boundary.

The last item is what makes these cards part of a *fairness* system rather than a
UI nicety. A card that reads "score 68, and it would have been 72 with a
different name" tells the recruiter something no aggregate metric can, and it
tells them at the moment they are making the decision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.frames import text_column


def build_cards(
    frame: pd.DataFrame,
    *,
    value_column: str = "overall_score",
    max_cards: int = 25,
    arm: str = "baseline",
    advance_threshold: float = 70.0,
    borderline_threshold: float = 55.0,
) -> List[Dict[str, Any]]:
    """One card per (job, candidate), ordered most cue-sensitive first.

    Sorting by sensitivity rather than by score is deliberate: the cards a human
    most needs to read are the ones where the model's judgement was least stable.
    """
    data = frame.copy()
    if "arm" in data.columns:
        data = data[text_column(data, "arm") == arm]
    if "parse_ok" in data.columns:
        data = data[data["parse_ok"].astype(bool)]
    data = data[data[value_column].notna()]
    if data.empty:
        return []

    dimension_columns = [c for c in data.columns if c.startswith("dim_")]
    cards: List[Dict[str, Any]] = []

    for (job_id, resume_id), cell in data.groupby(["job_id", "base_resume_id"]):
        baseline_rows = (
            cell[cell["is_baseline"].astype(bool)] if "is_baseline" in cell.columns else cell
        )
        primary = baseline_rows.iloc[0] if not baseline_rows.empty else cell.iloc[0]

        variant_means = cell.groupby("variant_id")[value_column].mean()
        served = float(
            baseline_rows[value_column].astype(float).mean()
            if not baseline_rows.empty else variant_means.mean()
        )
        swing = float(variant_means.max() - variant_means.min()) if len(variant_means) > 1 else 0.0
        low = float(variant_means.min()) if len(variant_means) > 1 else served
        high = float(variant_means.max()) if len(variant_means) > 1 else served

        crosses = _crosses_boundary(low, high, advance_threshold, borderline_threshold)
        most_favourable = str(variant_means.idxmax()) if len(variant_means) > 1 else ""
        least_favourable = str(variant_means.idxmin()) if len(variant_means) > 1 else ""

        cue_labels = {}
        if len(variant_means) > 1 and "cue_ids" in cell.columns:
            for variant_id in (most_favourable, least_favourable):
                match = cell[cell["variant_id"] == variant_id]
                if not match.empty:
                    cue_labels[variant_id] = str(match.iloc[0].get("cue_ids", "") or "baseline")

        cards.append(
            {
                "job_id": str(job_id),
                "resume_id": str(resume_id),
                "score": round(served, 2),
                "recommendation": str(primary.get("recommendation", "")),
                "confidence": _safe_float(primary.get("confidence")),
                "summary": str(primary.get("summary", "") or ""),
                "dimensions": [
                    {
                        "name": column[len("dim_"):],
                        "score": _safe_float(primary.get(column)),
                    }
                    for column in dimension_columns
                    if _safe_float(primary.get(column)) is not None
                ],
                "missing_requirements": _split_list(primary.get("missing_requirements")),
                "evidence_support_rate": _safe_float(primary.get("evidence_support_rate")),
                "rationale": str(primary.get("rationale_text", "") or "")[:1200],
                "cue_sensitivity": {
                    "n_variants": int(len(variant_means)),
                    "score_low": round(low, 2),
                    "score_high": round(high, 2),
                    "swing": round(swing, 2),
                    "crosses_decision_boundary": crosses,
                    "most_favourable_variant": most_favourable,
                    "most_favourable_cue": cue_labels.get(most_favourable, ""),
                    "least_favourable_variant": least_favourable,
                    "least_favourable_cue": cue_labels.get(least_favourable, ""),
                },
                "flags": _flags(primary, swing, crosses),
                "provenance": {
                    "provider": str(primary.get("provider", "")),
                    "model": str(primary.get("model", "")),
                    "prompt_hash": str(primary.get("prompt_hash", "")),
                    "resume_hash": str(primary.get("resume_hash", "")),
                    "timestamp": str(primary.get("timestamp", "")),
                },
            }
        )

    cards.sort(key=lambda c: -c["cue_sensitivity"]["swing"])
    return cards[:max_cards]


def _crosses_boundary(
    low: float, high: float, advance: float, borderline: float
) -> Optional[str]:
    """Which decision boundary, if any, the candidate's variant range straddles."""
    for threshold, label in ((advance, "advance"), (borderline, "borderline")):
        if low < threshold <= high:
            return label
    return None


def _flags(row: pd.Series, swing: float, crosses: Optional[str]) -> List[str]:
    flags: List[str] = []
    if crosses:
        flags.append(
            f"DECISION UNSTABLE: identity cues alone move this candidate across the "
            f"'{crosses}' threshold — do not treat the recommendation as determinate"
        )
    elif swing >= 3.0:
        flags.append(
            f"cue-sensitive: score varies by {swing:.1f} points across identity variants"
        )
    rate = _safe_float(row.get("evidence_support_rate"))
    if rate is not None and rate < 0.5:
        flags.append(
            f"weakly grounded: only {100 * rate:.0f}% of quoted evidence was found in the "
            "resume; the stated reasons are not verifiable"
        )
    if bool(row.get("protected_attribute_referenced", False)):
        flags.append(
            "the model's own rationale referenced a protected attribute — review before use"
        )
    if str(row.get("error", "")):
        flags.append(f"provider note: {str(row.get('error'))[:120]}")
    return flags


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(out) else round(out, 4)


def _split_list(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    if not text or text == "nan":
        return []
    return [part.strip() for part in text.split(";") if part.strip()][:8]


def render_card_markdown(card: Dict[str, Any]) -> str:
    """A single card as Markdown, for the report and for pasting into a doc."""
    lines: List[str] = [
        f"### Candidate `{card['resume_id']}` — role `{card['job_id']}`",
        "",
        f"**Score {card['score']}** · recommendation **{card['recommendation'] or 'n/a'}**"
        + (f" · model confidence {card['confidence']:.2f}" if card.get("confidence") is not None else ""),
        "",
    ]
    if card.get("summary"):
        lines += [f"> {card['summary']}", ""]

    sensitivity = card["cue_sensitivity"]
    if sensitivity["n_variants"] > 1:
        lines += [
            f"**Cue-sensitivity band:** {sensitivity['score_low']} – "
            f"{sensitivity['score_high']} across {sensitivity['n_variants']} identity "
            f"variants (swing {sensitivity['swing']}).",
            "",
        ]
        if sensitivity.get("least_favourable_cue"):
            lines += [
                f"- Lowest score under: `{sensitivity['least_favourable_cue']}`",
                f"- Highest score under: `{sensitivity['most_favourable_cue']}`",
                "",
            ]

    if card["dimensions"]:
        lines += ["| dimension | score |", "| --- | --- |"]
        for dimension in card["dimensions"]:
            lines.append(f"| {dimension['name'].replace('_', ' ')} | {dimension['score']:.1f} |")
        lines.append("")

    if card["missing_requirements"]:
        lines += ["**Requirements without supporting evidence:**", ""]
        lines += [f"- {item}" for item in card["missing_requirements"]]
        lines.append("")

    if card.get("evidence_support_rate") is not None:
        lines += [
            f"**Evidence grounding:** {100 * card['evidence_support_rate']:.0f}% of quoted "
            "evidence was located in the resume text.",
            "",
        ]

    if card["flags"]:
        lines += ["**Flags:**", ""]
        lines += [f"- {flag}" for flag in card["flags"]]
        lines.append("")

    provenance = card["provenance"]
    lines += [
        f"<sub>{provenance['provider']}/{provenance['model']} · prompt "
        f"`{provenance['prompt_hash'][:10]}` · resume `{provenance['resume_hash'][:10]}` · "
        f"{provenance['timestamp']}</sub>",
        "",
    ]
    return "\n".join(lines)


def cards_summary(cards: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate what the cards reveal, for the report's decision-quality section."""
    if not cards:
        return {"n_cards": 0}
    unstable = [c for c in cards if c["cue_sensitivity"]["crosses_decision_boundary"]]
    swings = [c["cue_sensitivity"]["swing"] for c in cards]
    grounding = [
        c["evidence_support_rate"] for c in cards if c.get("evidence_support_rate") is not None
    ]
    return {
        "n_cards": len(cards),
        "n_decision_unstable": len(unstable),
        "fraction_decision_unstable": round(len(unstable) / len(cards), 4),
        "mean_swing": round(float(np.mean(swings)), 3),
        "max_swing": round(float(np.max(swings)), 3),
        "mean_evidence_support_rate": (
            round(float(np.mean(grounding)), 4) if grounding else None
        ),
        "unstable_candidates": [
            {
                "resume_id": c["resume_id"], "job_id": c["job_id"],
                "band": [c["cue_sensitivity"]["score_low"], c["cue_sensitivity"]["score_high"]],
                "boundary": c["cue_sensitivity"]["crosses_decision_boundary"],
            }
            for c in unstable[:10]
        ],
        "note": (
            "A 'decision unstable' candidate is one whose advance/borderline/reject outcome "
            "changes across their own identity variants. For those candidates the "
            "recommendation is a property of the cue, not of the qualification."
        ),
    }
