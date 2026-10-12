"""Calibration fairness — and the impossibility result that makes it essential.

Calibration asks a different question from parity: *does a score of 80 mean the
same thing for every group?*  A system can produce identical mean scores across
groups (perfect parity) while a score of 80 corresponds to a 70% chance of being
a good hire for one group and 50% for another.  The recruiter applying a single
threshold then makes systematically worse decisions about one group, and no
score-gap metric would have shown it.

Chouldechova (2017) and Kleinberg, Mullainathan & Raghavan (2017) proved that
when base rates differ across groups, **calibration within groups, equal false
positive rates and equal false negative rates cannot all hold** except in
degenerate cases.  This is not a modelling wrinkle to be engineered away — it is
a theorem.  Its practical consequence is that a fairness report which presents
several metrics and expects them all to be green is misreading its own output.

The project's brief anticipates this ("different fairness metrics can
mathematically contradict each other ... the write-up should explain any
disagreement"), and this module supplies the material for that explanation:
per-group ECE, Brier score, AUC, calibration slope and intercept, plus the
per-group error rates at a shared threshold that the theorem trades off against
calibration.

Everything here needs a ``qualified_label``.  Where labels are absent the module
reports that fact rather than substituting a proxy, because inventing a label
would make the conclusion an artifact of the invention.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..schemas import MetricResult
from ..screening.calibration import (
    auc_roc,
    brier_score,
    calibration_slope_intercept,
    expected_calibration_error,
    fit_calibration,
)
from ..utils.frames import nonempty_mask, text_column

_MIN_LABELLED_PER_GROUP = 8


def _labelled(
    frame: pd.DataFrame, attribute: str, value_column: str, label_column: str
) -> pd.DataFrame:
    column = f"signal_{attribute}"
    if column not in frame.columns or label_column not in frame.columns:
        return pd.DataFrame()
    data = frame[
        nonempty_mask(frame, column)
        & frame[value_column].notna()
        & frame[label_column].notna()
    ]
    if "parse_ok" in data.columns:
        data = data[data["parse_ok"].astype(bool)]
    if data.empty:
        return data
    data = data.copy()
    data["_group"] = text_column(data, column)
    data["_score"] = data[value_column].astype(float)
    data["_label"] = data[label_column].astype(float).round().astype(int)
    return data


def calibration_by_group(
    frame: pd.DataFrame,
    attribute: str,
    *,
    value_column: str = "overall_score",
    label_column: str = "qualified_label",
    method: str = "platt",
    n_bins: int = 8,
    decision_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Fit one global calibration map, then evaluate it separately per group.

    Fitting **globally and evaluating per group** is the design that answers the
    question. A per-group fit would absorb the very miscalibration being looked
    for: it would guarantee each group looks calibrated by construction, which is
    also why per-group calibration is a *mitigation* choice with legal
    implications (it adjusts scores by group membership) rather than a diagnostic.
    """
    data = _labelled(frame, attribute, value_column, label_column)
    if data.empty:
        return {
            "attribute": attribute,
            "error": (
                f"needs both signal_{attribute} and a {label_column}. "
                "Add qualified labels to data/sample/pairs.json to enable this family."
            ),
        }

    levels = sorted(data["_group"].unique())
    usable = [
        lvl for lvl in levels
        if int((data["_group"] == lvl).sum()) >= _MIN_LABELLED_PER_GROUP
        and data.loc[data["_group"] == lvl, "_label"].nunique() > 1
    ]
    if len(usable) < 2:
        return {
            "attribute": attribute,
            "error": (
                f"fewer than 2 groups have {_MIN_LABELLED_PER_GROUP}+ labelled rows with "
                "both label classes present"
            ),
            "levels": levels,
            "n_per_level": {lvl: int((data["_group"] == lvl).sum()) for lvl in levels},
        }

    global_model = fit_calibration(
        data["_score"].tolist(), data["_label"].tolist(), method=method
    )
    data = data.copy()
    data["_prob"] = global_model.predict(data["_score"].tolist())

    per_group: Dict[str, Any] = {}
    for level in usable:
        subset = data[data["_group"] == level]
        probabilities = subset["_prob"].to_numpy()
        labels = subset["_label"].to_numpy()
        ece = expected_calibration_error(probabilities, labels, n_bins=n_bins)
        slope = calibration_slope_intercept(probabilities, labels)
        predicted = probabilities >= decision_threshold
        positives = labels > 0.5
        negatives = ~positives
        per_group[level] = {
            "n": int(len(subset)),
            "base_rate": round(float(labels.mean()), 4),
            "mean_score": round(float(subset["_score"].mean()), 4),
            "mean_predicted_probability": round(float(probabilities.mean()), 4),
            "ece": ece["ece"],
            "mce": ece["mce"],
            "brier": round(brier_score(probabilities, labels), 4),
            "auc": round(auc_roc(subset["_score"].tolist(), labels.tolist()), 4),
            "calibration_slope": slope.get("slope"),
            "calibration_intercept": slope.get("intercept"),
            "selection_rate_at_threshold": round(float(predicted.mean()), 4),
            "true_positive_rate": (
                round(float(predicted[positives].mean()), 4) if positives.any() else None
            ),
            "false_positive_rate": (
                round(float(predicted[negatives].mean()), 4) if negatives.any() else None
            ),
            "reliability_bins": ece["bins"],
        }

    eces = {lvl: info["ece"] for lvl, info in per_group.items() if info["ece"] == info["ece"]}
    aucs = {lvl: info["auc"] for lvl, info in per_group.items() if info["auc"] == info["auc"]}
    tprs = {lvl: info["true_positive_rate"] for lvl, info in per_group.items()
            if info["true_positive_rate"] is not None}
    fprs = {lvl: info["false_positive_rate"] for lvl, info in per_group.items()
            if info["false_positive_rate"] is not None}
    base_rates = {lvl: info["base_rate"] for lvl, info in per_group.items()}

    result = {
        "attribute": attribute,
        "method": global_model.method,
        "calibration_warnings": global_model.warnings,
        "n_labelled": int(len(data)),
        "levels_analysed": usable,
        "per_group": per_group,
        "ece_spread": round(max(eces.values()) - min(eces.values()), 4) if len(eces) > 1 else None,
        "auc_spread": round(max(aucs.values()) - min(aucs.values()), 4) if len(aucs) > 1 else None,
        "tpr_spread": round(max(tprs.values()) - min(tprs.values()), 4) if len(tprs) > 1 else None,
        "fpr_spread": round(max(fprs.values()) - min(fprs.values()), 4) if len(fprs) > 1 else None,
        "base_rate_spread": round(max(base_rates.values()) - min(base_rates.values()), 4)
        if len(base_rates) > 1 else None,
        "decision_threshold": decision_threshold,
    }
    result["impossibility_note"] = _impossibility_note(result)
    return result


def _impossibility_note(result: Dict[str, Any]) -> str:
    base_spread = result.get("base_rate_spread")
    if base_spread is None:
        return "Insufficient groups to comment on the calibration/error-rate trade-off."
    if base_spread < 0.02:
        return (
            "Qualification base rates are nearly equal across groups "
            f"(spread {base_spread:.3f}), so calibration and equal error rates are not in "
            "tension here and could in principle both be satisfied."
        )
    return (
        f"Qualification base rates differ across groups by {base_spread:.3f}. Under "
        "Chouldechova (2017) and Kleinberg et al. (2017), calibration within groups and "
        "equal false-positive/false-negative rates cannot then hold simultaneously. Any "
        "disagreement between the calibration metrics and the parity metrics below is "
        "therefore expected and is a property of the problem, not a defect in the "
        "measurement — but it does mean the project must state which criterion it is "
        "prioritising and why."
    )


def calibration_fairness_metrics(
    frame: pd.DataFrame,
    attributes: Sequence[str],
    *,
    value_column: str = "overall_score",
    label_column: str = "qualified_label",
    method: str = "platt",
    rng: Optional[np.random.Generator] = None,
) -> List[MetricResult]:
    del rng
    metrics: List[MetricResult] = []
    for attribute in attributes:
        analysis = calibration_by_group(
            frame, attribute, value_column=value_column,
            label_column=label_column, method=method,
        )
        if "error" in analysis:
            continue

        if analysis["ece_spread"] is not None:
            metrics.append(
                MetricResult(
                    name=f"calibration_ece_spread[{attribute}]",
                    family="calibration",
                    attribute=attribute,
                    value=analysis["ece_spread"],
                    unit="expected_calibration_error",
                    n=analysis["n_labelled"],
                    groups={lvl: info["ece"] for lvl, info in analysis["per_group"].items()},
                    threshold=0.05,
                    passed=analysis["ece_spread"] <= 0.05,
                    higher_is_better=False,
                    detail=analysis,
                    interpretation=(
                        f"Expected calibration error differs by {analysis['ece_spread']:.3f} "
                        f"across {attribute} levels: the same score does not carry the same "
                        "meaning for every group, so one threshold produces systematically "
                        "different decision quality."
                    ),
                )
            )
        if analysis["auc_spread"] is not None:
            metrics.append(
                MetricResult(
                    name=f"auc_spread[{attribute}]",
                    family="calibration",
                    attribute=attribute,
                    value=analysis["auc_spread"],
                    unit="auc_difference",
                    n=analysis["n_labelled"],
                    groups={lvl: info["auc"] for lvl, info in analysis["per_group"].items()},
                    threshold=0.05,
                    passed=analysis["auc_spread"] <= 0.05,
                    higher_is_better=False,
                    detail={"per_group_auc": {
                        lvl: info["auc"] for lvl, info in analysis["per_group"].items()
                    }},
                    interpretation=(
                        f"Ranking quality (AUC) differs by {analysis['auc_spread']:.3f} across "
                        f"{attribute} levels — the scorer is better at telling qualified from "
                        "unqualified candidates in some groups than others."
                    ),
                )
            )
        for key, label, threshold in (
            ("tpr_spread", "equalised_odds_tpr_spread", 0.1),
            ("fpr_spread", "equalised_odds_fpr_spread", 0.1),
        ):
            if analysis.get(key) is None:
                continue
            metrics.append(
                MetricResult(
                    name=f"{label}[{attribute}]",
                    family="calibration",
                    attribute=attribute,
                    value=analysis[key],
                    unit="rate_difference",
                    n=analysis["n_labelled"],
                    threshold=threshold,
                    passed=analysis[key] <= threshold,
                    higher_is_better=False,
                    detail={"impossibility_note": analysis["impossibility_note"]},
                    interpretation=(
                        f"{'True' if 'tpr' in key else 'False'}-positive rates at the shared "
                        f"decision threshold differ by {analysis[key]:.3f} across {attribute} "
                        "levels. " + analysis["impossibility_note"]
                    ),
                )
            )
    return metrics
