"""Group-level score parity.

The metric the build plan starts with — "average score gap between
name/demographic groups" — done properly for a blocked design, and extended so
that one number never has to carry a conclusion alone.

For each protected attribute the module reports:

* every level's within-candidate-centred mean, so that levels are comparable
  even when they were applied to resumes of different strength;
* the gap of each level against the configured reference level, with a
  bootstrap confidence interval;
* three p-values from tests with different assumptions (paired t, Wilcoxon,
  within-block permutation) — agreement between them is what makes a finding
  robust rather than an artifact of one test's assumptions;
* an omnibus permutation test on the largest pairwise gap, which asks "does *any*
  level differ?" while already accounting for having looked at all of them;
* a cluster-robust regression over all levels at once, which is the right tool
  when an attribute has more than two levels;
* the effect size, the minimum detectable effect and the achieved power, so a
  null result is reported as "we could not have seen an effect this small" rather
  than as "there is no effect".

Every gap is also checked against ``min_practical_effect``.  A statistically
detectable 0.2-point gap on a 100-point scale is not an actionable finding, and
saying so is part of reporting honestly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..schemas import MetricResult
from ..utils.frames import block_key, nonempty_mask, text_column
from .stats import (
    achieved_power,
    cliffs_delta,
    cluster_bootstrap_ci,
    cluster_robust_ols,
    describe,
    dummy_code,
    minimum_detectable_effect,
    paired_block_differences,
    paired_test,
    required_blocks,
    within_block_permutation_test,
)


def _prepare(frame: pd.DataFrame, attribute: str, value_column: str) -> pd.DataFrame:
    column = f"signal_{attribute}"
    if column not in frame.columns:
        return pd.DataFrame()
    data = frame[nonempty_mask(frame, column)]
    data = data[data[value_column].notna()]
    if "parse_ok" in data.columns:
        data = data[data["parse_ok"].astype(bool)]
    if data.empty:
        return data
    data = data.copy()
    # Block = one candidate applying for one role. Everything within a block is
    # the same document apart from the cue.
    data["_block"] = block_key(data, "job_id", "base_resume_id")
    data["_group"] = text_column(data, column)
    # Average decoding repeats so a hot sample cannot enter as an extra observation.
    data = (
        data.groupby(["_block", "_group", "variant_id"], as_index=False)
        .agg(_value=(value_column, "mean"))
    )
    return data


def _centred(data: pd.DataFrame) -> pd.DataFrame:
    """Subtract each block's own mean, so levels are on a common footing."""
    out = data.copy()
    out["_centred"] = out["_value"] - out.groupby("_block")["_value"].transform("mean")
    return out


def group_parity(
    frame: pd.DataFrame,
    attribute: str,
    *,
    value_column: str = "overall_score",
    reference_level: Optional[str] = None,
    alpha: float = 0.05,
    n_boot: int = 2000,
    n_permutations: int = 5000,
    min_practical_effect: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Full parity analysis for one attribute."""
    rng = rng or np.random.default_rng(0)
    data = _prepare(frame, attribute, value_column)
    if data.empty:
        return {"attribute": attribute, "error": f"no rows carrying signal_{attribute}"}

    levels = sorted(data["_group"].unique())
    if len(levels) < 2:
        return {
            "attribute": attribute,
            "error": f"only one level present ({levels}); a gap needs two",
            "levels": levels,
        }

    centred = _centred(data)
    level_stats: Dict[str, Any] = {}
    for level in levels:
        raw = data.loc[data["_group"] == level, "_value"]
        adj = centred.loc[centred["_group"] == level, "_centred"]
        level_stats[level] = {
            **{f"raw_{k}": (round(v, 4) if isinstance(v, float) else v)
               for k, v in describe(raw.tolist()).items()},
            "centred_mean": round(float(adj.mean()), 4),
            "centred_sd": round(float(adj.std(ddof=1)), 4) if len(adj) > 1 else 0.0,
            "n_blocks": int(data.loc[data["_group"] == level, "_block"].nunique()),
        }

    reference = reference_level if reference_level in levels else levels[0]
    if reference_level is not None and reference_level not in levels:
        note_ref = (
            f"configured reference level {reference_level!r} is absent from this run; "
            f"using {reference!r}"
        )
    else:
        note_ref = ""

    comparisons: List[Dict[str, Any]] = []
    for level in levels:
        if level == reference:
            continue
        blocked = paired_block_differences(
            data["_value"].tolist(), data["_group"].tolist(), data["_block"].tolist(),
            level, reference,
        )
        test = paired_test(blocked.differences, alpha=alpha)
        perm = within_block_permutation_test(
            data["_value"].tolist(), data["_group"].tolist(), data["_block"].tolist(),
            statistic="mean_difference", level_a=level, level_b=reference,
            n_permutations=n_permutations, rng=rng,
        )
        boot = cluster_bootstrap_ci(
            {b: d for b, d in zip(blocked.block_ids, blocked.differences)},
            lambda xs: float(np.mean(xs)), n_boot=n_boot, alpha=alpha, rng=rng,
        )
        sd_diff = float(test.get("sd_difference", float("nan")))
        gap = blocked.mean_difference

        comparisons.append(
            {
                "level": level,
                "reference": reference,
                "n_blocks": blocked.n_blocks,
                "mean_level": round(blocked.mean_a, 4),
                "mean_reference": round(blocked.mean_b, 4),
                "gap": round(gap, 4) if gap == gap else None,
                "gap_ci_low": round(boot.get("ci_low", float("nan")), 4),
                "gap_ci_high": round(boot.get("ci_high", float("nan")), 4),
                "paired_t_p": test.get("t_p_value"),
                "wilcoxon_p": test.get("wilcoxon_p_value"),
                "permutation_p": perm.get("p_value"),
                "cohens_dz": (
                    round(test["cohens_dz"], 4)
                    if test.get("cohens_dz") is not None and test["cohens_dz"] == test["cohens_dz"]
                    else None
                ),
                "cliffs_delta": round(
                    cliffs_delta(
                        data.loc[data["_group"] == level, "_value"].tolist(),
                        data.loc[data["_group"] == reference, "_value"].tolist(),
                    ), 4,
                ),
                "sd_of_differences": round(sd_diff, 4) if sd_diff == sd_diff else None,
                "minimum_detectable_effect": round(
                    minimum_detectable_effect(blocked.n_blocks, sd_diff, alpha=alpha), 4
                ) if sd_diff == sd_diff else None,
                "achieved_power": round(
                    achieved_power(gap, blocked.n_blocks, sd_diff, alpha=alpha), 4
                ) if sd_diff == sd_diff and gap == gap else None,
                "blocks_needed_for_1pt": (
                    required_blocks(min_practical_effect, sd_diff, alpha=alpha)
                    if sd_diff == sd_diff else None
                ),
                "practically_significant": bool(
                    gap == gap and abs(gap) >= min_practical_effect
                ),
            }
        )

    omnibus = within_block_permutation_test(
        data["_value"].tolist(), data["_group"].tolist(), data["_block"].tolist(),
        statistic="max_abs_gap", n_permutations=n_permutations, rng=rng,
    )

    columns, ols_reference = dummy_code(data["_group"].tolist(), reference, prefix=attribute)
    regression: Dict[str, Any] = {"note": "not enough levels for regression"}
    if columns:
        fit = cluster_robust_ols(
            data["_value"].tolist(), columns, data["_block"].tolist(), alpha=alpha
        )
        regression = {
            "reference_level": ols_reference,
            "n": fit.n,
            "n_clusters": fit.n_clusters,
            "r_squared": round(fit.r_squared, 4) if fit.r_squared == fit.r_squared else None,
            "coefficients": {
                k: {
                    "estimate": round(v, 4),
                    "se": round(fit.std_errors[k], 4),
                    "p_value": (round(fit.p_values[k], 6)
                                if fit.p_values[k] == fit.p_values[k] else None),
                    "ci_low": round(fit.ci_low[k], 4),
                    "ci_high": round(fit.ci_high[k], 4),
                }
                for k, v in fit.coefficients.items()
                if k != "intercept"
            },
            "note": fit.note,
        }

    centred_means = {lvl: level_stats[lvl]["centred_mean"] for lvl in levels}
    spread = max(centred_means.values()) - min(centred_means.values())

    return {
        "attribute": attribute,
        "levels": levels,
        "reference_level": reference,
        "reference_note": note_ref,
        "n_rows": int(len(data)),
        "n_blocks": int(data["_block"].nunique()),
        "level_statistics": level_stats,
        "centred_means": centred_means,
        "max_pairwise_gap": round(float(spread), 4),
        "best_level": max(centred_means, key=centred_means.get),
        "worst_level": min(centred_means, key=centred_means.get),
        "omnibus_permutation": omnibus,
        "comparisons": comparisons,
        "regression": regression,
        "min_practical_effect": min_practical_effect,
    }


def group_parity_metrics(
    frame: pd.DataFrame,
    attributes: Sequence[str],
    *,
    value_column: str = "overall_score",
    reference_levels: Optional[Dict[str, str]] = None,
    alpha: float = 0.05,
    n_boot: int = 2000,
    n_permutations: int = 5000,
    min_practical_effect: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> List[MetricResult]:
    """Parity metrics across attributes, as :class:`MetricResult` rows."""
    reference_levels = reference_levels or {}
    metrics: List[MetricResult] = []

    for attribute in attributes:
        analysis = group_parity(
            frame, attribute, value_column=value_column,
            reference_level=reference_levels.get(attribute), alpha=alpha,
            n_boot=n_boot, n_permutations=n_permutations,
            min_practical_effect=min_practical_effect, rng=rng,
        )
        if "error" in analysis:
            continue

        metrics.append(
            MetricResult(
                name=f"max_pairwise_score_gap[{attribute}]",
                family="group_parity",
                attribute=attribute,
                value=analysis["max_pairwise_gap"],
                unit="score_points",
                p_value=analysis["omnibus_permutation"].get("p_value", float("nan")),
                n=analysis["n_blocks"],
                groups=analysis["centred_means"],
                threshold=min_practical_effect,
                passed=analysis["max_pairwise_gap"] < min_practical_effect,
                higher_is_better=False,
                detail={
                    "levels": analysis["levels"],
                    "level_statistics": analysis["level_statistics"],
                    "omnibus_permutation": analysis["omnibus_permutation"],
                    "regression": analysis["regression"],
                    "best_level": analysis["best_level"],
                    "worst_level": analysis["worst_level"],
                },
                interpretation=(
                    f"Largest gap between {attribute} levels is "
                    f"{analysis['max_pairwise_gap']:.2f} score points "
                    f"({analysis['worst_level']} lowest, {analysis['best_level']} highest), "
                    f"omnibus permutation p = "
                    f"{_fmt_p(analysis['omnibus_permutation'].get('p_value'))}."
                ),
            )
        )

        for comparison in analysis["comparisons"]:
            gap = comparison["gap"]
            metrics.append(
                MetricResult(
                    name=f"score_gap[{attribute}:{comparison['level']}_vs_{comparison['reference']}]",
                    family="group_parity",
                    attribute=attribute,
                    value=gap if gap is not None else float("nan"),
                    unit="score_points",
                    ci_low=comparison["gap_ci_low"],
                    ci_high=comparison["gap_ci_high"],
                    # The permutation test is the primary p-value: it relies on the
                    # design's own randomisation rather than on normality.
                    p_value=comparison["permutation_p"] if comparison["permutation_p"] is not None else float("nan"),
                    effect_size=comparison["cohens_dz"] if comparison["cohens_dz"] is not None else float("nan"),
                    effect_size_name="cohens_dz",
                    n=comparison["n_blocks"],
                    groups={
                        comparison["level"]: comparison["mean_level"],
                        comparison["reference"]: comparison["mean_reference"],
                    },
                    threshold=min_practical_effect,
                    passed=not comparison["practically_significant"],
                    higher_is_better=False,
                    detail=comparison,
                    interpretation=_gap_interpretation(attribute, comparison, min_practical_effect),
                )
            )
    return metrics


def _fmt_p(value: Optional[float]) -> str:
    if value is None or value != value:
        return "n/a"
    return f"{value:.4f}" if value >= 1e-4 else f"{value:.1e}"


def _gap_interpretation(
    attribute: str, comparison: Dict[str, Any], min_practical_effect: float
) -> str:
    gap = comparison["gap"]
    if gap is None:
        return "gap undefined"
    direction = "lower" if gap < 0 else "higher"
    text = (
        f"Resumes signalling {attribute}={comparison['level']} scored "
        f"{abs(gap):.2f} points {direction} than {comparison['reference']} on the "
        f"same {comparison['n_blocks']} documents "
        f"(95% CI {comparison['gap_ci_low']:.2f} to {comparison['gap_ci_high']:.2f}; "
        f"permutation p = {_fmt_p(comparison['permutation_p'])})."
    )
    if not comparison["practically_significant"]:
        mde = comparison.get("minimum_detectable_effect")
        text += (
            f" This is below the {min_practical_effect:g}-point practical threshold."
        )
        if mde is not None and mde == mde:
            text += (
                f" Note the minimum detectable effect at this sample size is "
                f"{mde:.2f} points, so a smaller true gap could not have been found."
            )
    return text
