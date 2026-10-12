"""Candidate ranking, plus the ranking-stability measurements the audit needs.

Ranking is where a screening system does its damage or its good: a recruiter
looks at the top of a list, so a one-point score change that crosses the top-5
boundary matters enormously while a ten-point change in the middle matters not at
all.  This module therefore produces both:

* the shortlist a recruiter would see, and
* the **allocational** quantities that make a fairness claim concrete —
  selection rates by group at ``k``, and how much the ordering moves when a cue
  is swapped.

Ties are broken deterministically (score, then resume id) so that a rank change
is always attributable to a score change and never to dictionary ordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.frames import as_text, text_column
from ..utils.hashing import hash_text


def _tiebreak(ids: pd.Series) -> pd.Series:
    """A cue-independent sort key for exactly tied scores.

    Sorting ties lexicographically by id looks harmless and is not. Variant ids
    encode the cue, and cue labels are not alphabetically neutral: with two
    candidates on identical scores, ``...__female`` always beats ``...__male``,
    which manufactures a selection-rate disparity out of a tie. The same problem
    exists for real candidates, where an alphabetical tiebreak is a documented
    surname advantage.

    Hashing the id preserves determinism — the same table always produces the
    same ranking — while making the tiebreak independent of the label, so a tie
    contributes no systematic advantage to any group.
    """
    return as_text(ids).map(lambda value: hash_text(value, 12))


@dataclass
class RankedCandidate:
    rank: int
    variant_id: str
    base_resume_id: str
    score: float
    recommendation: str
    in_top_k: bool = False
    signals: Dict[str, str] = None  # type: ignore[assignment]


def rank_candidates(
    frame: pd.DataFrame,
    job_id: str,
    *,
    top_k: int = 5,
    score_column: str = "overall_score",
    arm: Optional[str] = None,
    baseline_only: bool = True,
) -> List[RankedCandidate]:
    """Rank candidates for one job.

    ``baseline_only`` is the default because a ranking that mixes a resume with
    its own counterfactual variants would place the same person in the list many
    times — meaningless as a shortlist. The variant-aware allocational analysis
    lives in :func:`selection_rates` instead.
    """
    subset = frame[frame["job_id"] == job_id]
    if arm is not None:
        subset = subset[subset["arm"] == arm]
    if baseline_only and "is_baseline" in subset.columns:
        subset = subset[subset["is_baseline"].astype(bool)]
    subset = subset[subset["parse_ok"].astype(bool)] if "parse_ok" in subset.columns else subset
    if subset.empty:
        return []

    # Average decoding repeats before ranking: a shortlist should not depend on
    # which sample happened to be drawn.
    grouped = (
        subset.groupby(["variant_id", "base_resume_id"], as_index=False)
        .agg(score=(score_column, "mean"), recommendation=("recommendation", "first"))
    )
    grouped["_tiebreak"] = _tiebreak(grouped["base_resume_id"])
    grouped = grouped.sort_values(
        ["score", "_tiebreak"], ascending=[False, True]
    ).reset_index(drop=True)

    out: List[RankedCandidate] = []
    for position, row in grouped.iterrows():
        out.append(
            RankedCandidate(
                rank=int(position) + 1,
                variant_id=str(row["variant_id"]),
                base_resume_id=str(row["base_resume_id"]),
                score=float(row["score"]),
                recommendation=str(row["recommendation"]),
                in_top_k=int(position) < top_k,
                signals={},
            )
        )
    return out


def shortlist_table(ranked: Sequence[RankedCandidate]) -> List[Dict[str, Any]]:
    return [
        {
            "rank": c.rank,
            "resume_id": c.base_resume_id,
            "score": round(c.score, 2),
            "recommendation": c.recommendation,
            "shortlisted": c.in_top_k,
        }
        for c in ranked
    ]


# --------------------------------------------------------------------------- #
# allocational quantities
# --------------------------------------------------------------------------- #
def selection_rates(
    frame: pd.DataFrame,
    attribute: str,
    *,
    k: int = 5,
    score_column: str = "overall_score",
    arm: Optional[str] = None,
) -> Dict[str, Any]:
    """Top-``k`` selection rate per demographic level.

    Ranking is done **within each job separately**, and within a job the
    competing entries are the counterfactual variants of the same candidate pool.
    That construction is what makes the comparison causal: the candidates being
    ranked against each other are identical documents differing only in the cue,
    so a difference in selection rate cannot be explained by differences in
    qualification.
    """
    column = f"signal_{attribute}"
    if column not in frame.columns:
        return {"attribute": attribute, "error": f"no column {column}"}

    subset = frame[frame[column].notna() & (frame[column] != "")]
    if arm is not None:
        subset = subset[subset["arm"] == arm]
    if "parse_ok" in subset.columns:
        subset = subset[subset["parse_ok"].astype(bool)]
    if subset.empty:
        return {"attribute": attribute, "error": "no rows with this signal"}

    selected: Dict[str, int] = {}
    total: Dict[str, int] = {}
    per_job: List[Dict[str, Any]] = []
    n_tied_at_cut = 0
    n_cuts = 0

    for job_id, job_frame in subset.groupby("job_id"):
        agg = job_frame.groupby(["variant_id", column], as_index=False)[score_column].mean()
        if agg.empty:
            continue
        agg["_tiebreak"] = _tiebreak(agg["variant_id"])
        agg = agg.sort_values([score_column, "_tiebreak"], ascending=[False, True])
        cut = min(k, len(agg))
        chosen = agg.head(cut)
        for level in agg[column].unique():
            total[level] = total.get(level, 0) + int((agg[column] == level).sum())
            selected[level] = selected.get(level, 0) + int((chosen[column] == level).sum())

        # How many candidates sit on exactly the boundary score? Where that
        # number is large, who makes the shortlist is settled by the tiebreak
        # rather than by the score, and the impact ratio is measuring the
        # tiebreak. Reporting it stops a reader treating an artifact as a finding.
        boundary = float(agg[score_column].iloc[cut - 1])
        tied = int((agg[score_column] == boundary).sum())
        n_cuts += 1
        n_tied_at_cut += tied
        per_job.append(
            {"job_id": str(job_id), "n_candidates": int(len(agg)), "k": cut,
             "n_tied_at_cut": tied}
        )

    rates = {
        level: (selected.get(level, 0) / total[level]) for level in total if total[level]
    }
    if not rates:
        return {"attribute": attribute, "error": "no selectable candidates"}

    best = max(rates.values())
    worst = min(rates.values())
    impact_ratio = (worst / best) if best > 0 else float("nan")

    mean_tied = (n_tied_at_cut / n_cuts) if n_cuts else 0.0
    ties_dominate = mean_tied > max(2.0, 0.5 * k)

    result = {
        "attribute": attribute,
        "k": k,
        "n_selected": selected,
        "n_candidates": total,
        "selection_rate": {level: round(rate, 4) for level, rate in rates.items()},
        "max_rate": round(best, 4),
        "min_rate": round(worst, 4),
        "statistical_parity_difference": round(best - worst, 4),
        "impact_ratio": round(impact_ratio, 4) if impact_ratio == impact_ratio else None,
        "four_fifths_rule_satisfied": bool(impact_ratio >= 0.8) if impact_ratio == impact_ratio else None,
        "mean_tied_at_cut": round(mean_tied, 2),
        "ties_dominate_selection": bool(ties_dominate),
        "per_job": per_job,
        "note": (
            "Selection rates are computed within job, over counterfactual variants of "
            "the same candidates, so differences cannot be attributed to differing "
            "qualifications."
        ),
    }
    if ties_dominate:
        result["tie_warning"] = (
            f"On average {mean_tied:.1f} candidates sit exactly on the top-{k} boundary "
            "score, so shortlist membership is decided by the tiebreak rather than by the "
            "score. The impact ratio here largely reflects that tiebreak and should not be "
            "read as an allocational finding. This is what a scorer with too coarse a score "
            "scale looks like."
        )
    return result


def ranking_stability(
    frame: pd.DataFrame,
    *,
    score_column: str = "overall_score",
    cue_family: Optional[str] = None,
) -> Dict[str, Any]:
    """How much does the shortlist move when a cue is swapped?

    Score-level invariance is not the same as ranking-level invariance. A model
    can shift every candidate by half a point — statistically trivial — and still
    reorder the top of the list, which is the only part a recruiter reads. Both
    are reported.
    """
    if "is_baseline" not in frame.columns:
        return {"error": "frame lacks is_baseline"}

    results: List[Dict[str, Any]] = []
    for job_id, job_frame in frame.groupby("job_id"):
        base = job_frame[job_frame["is_baseline"].astype(bool)]
        variants = job_frame[~job_frame["is_baseline"].astype(bool)]
        if cue_family:
            variants = variants[
                text_column(variants, "cue_families").str.contains(cue_family, regex=False)
            ]
        if base.empty or variants.empty:
            continue

        baseline_scores = base.groupby("base_resume_id")[score_column].mean()
        if len(baseline_scores) < 3:
            continue
        baseline_order = baseline_scores.sort_values(ascending=False).index.tolist()

        taus: List[float] = []
        top1_changes = 0
        comparisons = 0
        for _cue_id, cue_frame in variants.groupby("cue_ids"):
            variant_scores = cue_frame.groupby("base_resume_id")[score_column].mean()
            common = [r for r in baseline_order if r in variant_scores.index]
            if len(common) < 3:
                continue
            from scipy import stats

            base_vals = [baseline_scores[r] for r in common]
            var_vals = [variant_scores[r] for r in common]
            tau, _ = stats.kendalltau(base_vals, var_vals)
            if tau == tau:
                taus.append(float(tau))
            variant_order = variant_scores.sort_values(ascending=False).index.tolist()
            if variant_order and common and variant_order[0] != common[0]:
                top1_changes += 1
            comparisons += 1

        if comparisons:
            results.append(
                {
                    "job_id": str(job_id),
                    "n_cue_variants": comparisons,
                    "mean_kendall_tau_vs_baseline": round(float(np.mean(taus)), 4) if taus else None,
                    "min_kendall_tau_vs_baseline": round(float(np.min(taus)), 4) if taus else None,
                    "top1_change_rate": round(top1_changes / comparisons, 4),
                }
            )

    if not results:
        return {"error": "not enough variants per job to assess ranking stability"}

    taus = [r["mean_kendall_tau_vs_baseline"] for r in results if r["mean_kendall_tau_vs_baseline"] is not None]
    return {
        "cue_family": cue_family or "all",
        "per_job": results,
        "mean_kendall_tau": round(float(np.mean(taus)), 4) if taus else None,
        "mean_top1_change_rate": round(
            float(np.mean([r["top1_change_rate"] for r in results])), 4
        ),
        "interpretation": (
            "Kendall tau of 1.0 means a cue swap never reorders candidates. "
            "top1_change_rate is the fraction of cue swaps that changed who ranks first — "
            "the allocationally decisive quantity."
        ),
    }


def expected_exposure(
    frame: pd.DataFrame,
    attribute: str,
    *,
    score_column: str = "overall_score",
) -> Dict[str, Any]:
    """Position-discounted attention received by each demographic level.

    Selection rate at ``k`` is a step function and throws away the difference
    between rank 1 and rank 5. Exposure applies the logarithmic discount used in
    ranking evaluation, so it captures a systematic push down the list that never
    crosses the ``k`` boundary. Reported as a ratio to the mean, where 1.0 is
    parity.
    """
    column = f"signal_{attribute}"
    if column not in frame.columns:
        return {"attribute": attribute, "error": f"no column {column}"}
    subset = frame[frame[column].notna() & (frame[column] != "")]
    if "parse_ok" in subset.columns:
        subset = subset[subset["parse_ok"].astype(bool)]
    if subset.empty:
        return {"attribute": attribute, "error": "no rows with this signal"}

    exposure: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for _job_id, job_frame in subset.groupby("job_id"):
        agg = job_frame.groupby(["variant_id", column], as_index=False)[score_column].mean()
        if agg.empty:
            continue
        agg["_tiebreak"] = _tiebreak(agg["variant_id"])
        agg = agg.sort_values([score_column, "_tiebreak"], ascending=[False, True]).reset_index(
            drop=True
        )
        for position, row in agg.iterrows():
            level = row[column]
            exposure[level] = exposure.get(level, 0.0) + 1.0 / math.log2(int(position) + 2)
            counts[level] = counts.get(level, 0) + 1

    mean_exposure = {
        level: exposure[level] / counts[level] for level in counts if counts[level]
    }
    if not mean_exposure:
        return {"attribute": attribute, "error": "no exposure computed"}
    overall = float(np.mean(list(mean_exposure.values())))
    ratios = {
        level: round(value / overall, 4) for level, value in mean_exposure.items()
    } if overall else {}
    return {
        "attribute": attribute,
        "mean_exposure": {k: round(v, 4) for k, v in mean_exposure.items()},
        "exposure_ratio_to_mean": ratios,
        "max_min_exposure_ratio": (
            round(min(mean_exposure.values()) / max(mean_exposure.values()), 4)
            if max(mean_exposure.values()) > 0 else None
        ),
        "n_candidates": counts,
    }
