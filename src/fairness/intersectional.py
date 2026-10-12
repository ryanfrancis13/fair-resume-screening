"""Intersectional analysis.

Marginal metrics can both look clean while a specific subgroup is treated badly.
If a model penalises Black women but not Black men and not White women, the
marginal gender gap and the marginal race gap can each be near zero — the effect
lives entirely in the interaction, and averaging over it makes it disappear.
Wilson & Caliskan (2024) and Webster (2025) find exactly this pattern, and An et
al. (2025) build their whole design around it.

This module therefore reports:

* the **cell mean** for every combination of the intersecting attributes,
  within-candidate centred so cells are comparable;
* the **worst-cell gap** — the quantity a marginal analysis cannot see;
* an **interaction test**: is the worst cell worse than an additive model of the
  two marginal effects predicts? A significant negative interaction is the formal
  statement of "the intersection is treated worse than the sum of its parts";
* a **cell-count table**, because an intersectional claim from a cell with two
  observations is not a finding, and empty or thin cells must be visible.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..schemas import MetricResult
from ..utils.frames import block_key, nonempty_mask, text_column
from .stats import cluster_robust_ols, within_block_permutation_test

#: Below this many blocks, a cell estimate is reported but flagged as thin.
MIN_CELL_BLOCKS = 4


def _prepare(
    frame: pd.DataFrame, attributes: Sequence[str], value_column: str
) -> pd.DataFrame:
    columns = [f"signal_{a}" for a in attributes]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        return pd.DataFrame()
    data = frame.copy()
    for column in columns:
        data = data[nonempty_mask(data, column)]
    data = data[data[value_column].notna()]
    if "parse_ok" in data.columns:
        data = data[data["parse_ok"].astype(bool)]
    if data.empty:
        return data
    data = data.copy()
    data["_block"] = block_key(data, "job_id", "base_resume_id")
    data["_cell"] = text_column(data, columns[0])
    for column in columns[1:]:
        data["_cell"] = data["_cell"] + " x " + text_column(data, column)
    data = (
        data.groupby(["_block", "_cell", "variant_id"] + columns, as_index=False)
        .agg(_value=(value_column, "mean"))
    )
    data["_centred"] = data["_value"] - data.groupby("_block")["_value"].transform("mean")
    return data


def intersectional_analysis(
    frame: pd.DataFrame,
    attributes: Sequence[str] = ("gender", "race_ethnicity"),
    *,
    value_column: str = "overall_score",
    n_permutations: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Cell means, worst-cell gap and an interaction test."""
    rng = rng or np.random.default_rng(0)
    attributes = list(attributes)
    data = _prepare(frame, attributes, value_column)
    if data.empty:
        return {
            "attributes": attributes,
            "error": (
                "no rows carry all of "
                + ", ".join(f"signal_{a}" for a in attributes)
                + ". Intersectional analysis needs a cue family that signals both "
                  "attributes at once (the name family does), or a factorial design."
            ),
        }

    columns = [f"signal_{a}" for a in attributes]
    cells: Dict[str, Dict[str, Any]] = {}
    for cell, group in data.groupby("_cell"):
        n_blocks = int(group["_block"].nunique())
        cells[str(cell)] = {
            "n_rows": int(len(group)),
            "n_blocks": n_blocks,
            "centred_mean": round(float(group["_centred"].mean()), 4),
            "raw_mean": round(float(group["_value"].mean()), 4),
            "sd": round(float(group["_value"].std(ddof=1)), 4) if len(group) > 1 else 0.0,
            "thin": n_blocks < MIN_CELL_BLOCKS,
            **{a: str(group[c].iloc[0]) for a, c in zip(attributes, columns)},
        }

    if len(cells) < 2:
        return {"attributes": attributes, "error": "fewer than 2 populated cells", "cells": cells}

    means = {cell: info["centred_mean"] for cell, info in cells.items()}
    best_cell = max(means, key=means.get)
    worst_cell = min(means, key=means.get)

    # Marginal means per attribute, for the additivity comparison.
    marginals: Dict[str, Dict[str, float]] = {}
    for attribute, column in zip(attributes, columns):
        marginals[attribute] = {
            str(level): round(float(group["_centred"].mean()), 4)
            for level, group in data.groupby(column)
        }

    additive_prediction: Dict[str, float] = {}
    for cell, info in cells.items():
        prediction = 0.0
        for attribute in attributes:
            prediction += marginals[attribute].get(info[attribute], 0.0)
        additive_prediction[cell] = round(prediction, 4)

    interaction_residual = {
        cell: round(means[cell] - additive_prediction[cell], 4) for cell in cells
    }
    worst_interaction_cell = min(interaction_residual, key=interaction_residual.get)

    omnibus = within_block_permutation_test(
        data["_value"].tolist(), data["_cell"].tolist(), data["_block"].tolist(),
        statistic="max_abs_gap", n_permutations=n_permutations, rng=rng,
    )

    interaction_test = _interaction_regression(data, attributes, columns)

    return {
        "attributes": attributes,
        "n_rows": int(len(data)),
        "n_blocks": int(data["_block"].nunique()),
        "n_cells": len(cells),
        "n_thin_cells": sum(1 for c in cells.values() if c["thin"]),
        "cells": cells,
        "cell_centred_means": means,
        "best_cell": best_cell,
        "worst_cell": worst_cell,
        "worst_cell_gap": round(means[best_cell] - means[worst_cell], 4),
        "marginal_means": marginals,
        "additive_prediction": additive_prediction,
        "interaction_residual": interaction_residual,
        "worst_interaction_cell": worst_interaction_cell,
        "worst_interaction_residual": interaction_residual[worst_interaction_cell],
        "omnibus_permutation": omnibus,
        "interaction_regression": interaction_test,
        "note": (
            "Cell means are centred within candidate/job blocks. interaction_residual is "
            "the cell mean minus what the two marginal effects alone would predict; a "
            "negative value means that intersection is treated worse than the sum of its "
            "parts, which no marginal metric can detect."
        ),
    }


def _interaction_regression(
    data: pd.DataFrame, attributes: Sequence[str], columns: Sequence[str]
) -> Dict[str, Any]:
    """Cluster-robust regression with main effects plus their interaction."""
    if len(attributes) != 2:
        return {"note": "interaction regression is implemented for exactly two attributes"}

    a_levels = sorted(text_column(data, columns[0]).unique())
    b_levels = sorted(text_column(data, columns[1]).unique())
    if len(a_levels) < 2 or len(b_levels) < 2:
        return {"note": "each attribute needs at least 2 levels for an interaction"}

    a_ref, b_ref = a_levels[0], b_levels[0]
    design: Dict[str, List[float]] = {}
    a_values = text_column(data, columns[0]).tolist()
    b_values = text_column(data, columns[1]).tolist()

    for level in a_levels[1:]:
        design[f"{attributes[0]}[{level}]"] = [1.0 if v == level else 0.0 for v in a_values]
    for level in b_levels[1:]:
        design[f"{attributes[1]}[{level}]"] = [1.0 if v == level else 0.0 for v in b_values]
    for la, lb in itertools.product(a_levels[1:], b_levels[1:]):
        design[f"{la}:{lb}"] = [
            1.0 if (va == la and vb == lb) else 0.0 for va, vb in zip(a_values, b_values)
        ]

    fit = cluster_robust_ols(
        data["_value"].tolist(), design, data["_block"].tolist()
    )
    interaction_terms = {
        name: {
            "estimate": round(fit.coefficients[name], 4),
            "se": round(fit.std_errors[name], 4),
            "p_value": (round(fit.p_values[name], 6)
                        if fit.p_values[name] == fit.p_values[name] else None),
        }
        for name in design
        if ":" in name and name in fit.coefficients
    }
    significant = {
        k: v for k, v in interaction_terms.items()
        if v["p_value"] is not None and v["p_value"] < 0.05
    }
    return {
        "reference_cell": f"{a_ref} x {b_ref}",
        "n": fit.n,
        "n_clusters": fit.n_clusters,
        "interaction_terms": interaction_terms,
        "n_significant_interactions": len(significant),
        "significant_interactions": significant,
        "note": (
            "A significant interaction term means the joint effect of the two attributes "
            "is not the sum of their separate effects — the definition of an "
            "intersectional effect, and invisible to marginal analysis. "
            + fit.note
        ),
    }


def intersectional_metrics(
    frame: pd.DataFrame,
    attributes: Sequence[str] = ("gender", "race_ethnicity"),
    *,
    value_column: str = "overall_score",
    n_permutations: int = 5000,
    min_practical_effect: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> List[MetricResult]:
    analysis = intersectional_analysis(
        frame, attributes, value_column=value_column,
        n_permutations=n_permutations, rng=rng,
    )
    if "error" in analysis:
        return []

    label = " x ".join(attributes)
    metrics = [
        MetricResult(
            name=f"intersectional_worst_cell_gap[{label}]",
            family="intersectional",
            attribute=label,
            value=analysis["worst_cell_gap"],
            unit="score_points",
            p_value=analysis["omnibus_permutation"].get("p_value", float("nan")),
            n=analysis["n_blocks"],
            groups=analysis["cell_centred_means"],
            threshold=min_practical_effect,
            passed=analysis["worst_cell_gap"] < min_practical_effect,
            higher_is_better=False,
            detail={
                k: v for k, v in analysis.items() if k not in ("cells",)
            },
            interpretation=(
                f"The best- and worst-treated {label} subgroups differ by "
                f"{analysis['worst_cell_gap']:.2f} score points "
                f"(worst: {analysis['worst_cell']}). "
                f"{analysis['n_thin_cells']} of {analysis['n_cells']} cells have fewer "
                f"than {MIN_CELL_BLOCKS} blocks and are reported but not relied upon."
            ),
        ),
        MetricResult(
            name=f"intersectional_interaction_residual[{label}]",
            family="intersectional",
            attribute=label,
            value=analysis["worst_interaction_residual"],
            unit="score_points",
            n=analysis["n_blocks"],
            groups=analysis["interaction_residual"],
            threshold=-min_practical_effect,
            passed=analysis["worst_interaction_residual"] > -min_practical_effect,
            higher_is_better=True,
            detail=analysis["interaction_regression"],
            interpretation=(
                f"The {analysis['worst_interaction_cell']} subgroup scores "
                f"{abs(analysis['worst_interaction_residual']):.2f} points "
                f"{'below' if analysis['worst_interaction_residual'] < 0 else 'above'} "
                "what the two marginal effects alone would predict. A negative value here "
                "is a genuinely intersectional effect that marginal metrics cannot show."
            ),
        ),
    ]
    return metrics
