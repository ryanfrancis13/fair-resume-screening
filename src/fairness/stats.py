"""Statistical machinery for the audit.

The experimental design is a **randomised block design**: each base resume (in
combination with one job) is a block, and within a block the same document is
scored once per cue level.  Getting the statistics right for that structure is
what separates a defensible result from a plausible-looking one, and the brief's
plan of "run a t-test" is the starting point this module completes.

Why the naive approach fails
----------------------------
Pooling all rows and running an independent-samples t-test treats
``n_resumes × n_levels`` rows as independent observations.  They are not: rows
from the same resume share all of that resume's content, which is by far the
largest source of variance in the table.  The consequence is not a small bias —
the standard error can be understated several-fold, so a null result is reported
as significant. Every method here respects the blocking:

``paired_block_differences`` + ``paired_test``
    Reduce each block to one within-block contrast, then test those. Simple,
    exactly valid for a two-level comparison, and easy to explain in a viva.
``within_block_permutation_test``
    Permute cue labels *inside* each block. This is the design's own randomisation
    distribution, so it needs no normality assumption and is the primary p-value
    reported for the headline gaps.
``cluster_robust_ols``
    Regression with resume-clustered standard errors, for multi-level cues and
    covariate adjustment. Handles arbitrary within-block correlation.
``cluster_bootstrap_ci``
    Resamples *blocks*, not rows. Row-level bootstrap would reproduce exactly the
    independence error described above, inside the confidence interval.

Multiplicity and power
----------------------
The suite runs dozens of tests across attributes, levels and metric families.
``benjamini_hochberg`` controls the false discovery rate over all of them; with
this many tests, uncorrected p-values would manufacture findings.  And with 10-50
resumes, a null result is usually a statement about power rather than about
fairness — so ``minimum_detectable_effect`` and ``achieved_power`` are reported
alongside every null, because "no evidence of bias" and "evidence of no bias" are
different claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------------- #
# effect sizes
# --------------------------------------------------------------------------- #
def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardised mean difference with a pooled standard deviation."""
    x = np.asarray([v for v in a if v == v], dtype=float)
    y = np.asarray([v for v in b if v == v], dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    nx, ny = len(x), len(y)
    pooled_var = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    if pooled_var <= 0:
        return 0.0
    return float((x.mean() - y.mean()) / math.sqrt(pooled_var))


def cohens_dz(differences: Sequence[float]) -> float:
    """Paired effect size: mean difference over the sd of the differences.

    The right effect size for a blocked design. Reporting an unpaired ``d`` here
    would understate the effect, because the between-resume variance that ``d``
    includes in its denominator has already been differenced out.
    """
    diffs = np.asarray([v for v in differences if v == v], dtype=float)
    if len(diffs) < 2:
        return float("nan")
    sd = diffs.std(ddof=1)
    if sd <= 0:
        return float("inf") if diffs.mean() != 0 else 0.0
    return float(diffs.mean() / sd)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric effect size in ``[-1, 1]``: P(a>b) - P(a<b)."""
    x = np.asarray([v for v in a if v == v], dtype=float)
    y = np.asarray([v for v in b if v == v], dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    greater = 0
    less = 0
    for value in x:
        greater += int(np.sum(value > y))
        less += int(np.sum(value < y))
    total = len(x) * len(y)
    return float((greater - less) / total) if total else float("nan")


# --------------------------------------------------------------------------- #
# blocked comparisons
# --------------------------------------------------------------------------- #
@dataclass
class BlockedComparison:
    """A within-block contrast between two cue levels."""

    level_a: str
    level_b: str
    n_blocks: int
    mean_a: float
    mean_b: float
    mean_difference: float
    differences: List[float] = field(default_factory=list)
    block_ids: List[str] = field(default_factory=list)


def paired_block_differences(
    values: Sequence[float],
    groups: Sequence[str],
    blocks: Sequence[str],
    level_a: str,
    level_b: str,
) -> BlockedComparison:
    """Per-block mean difference between two levels.

    A block contributes only if it contains *both* levels. Blocks that contain
    one level and not the other carry no information about the contrast, and
    including their one-sided means would reintroduce between-resume variance
    into a within-resume comparison.
    """
    by_block_a: Dict[str, List[float]] = {}
    by_block_b: Dict[str, List[float]] = {}
    for value, group, block in zip(values, groups, blocks):
        if value != value:
            continue
        if group == level_a:
            by_block_a.setdefault(str(block), []).append(float(value))
        elif group == level_b:
            by_block_b.setdefault(str(block), []).append(float(value))

    shared = sorted(set(by_block_a) & set(by_block_b))
    diffs: List[float] = []
    a_means: List[float] = []
    b_means: List[float] = []
    for block in shared:
        mean_a = float(np.mean(by_block_a[block]))
        mean_b = float(np.mean(by_block_b[block]))
        a_means.append(mean_a)
        b_means.append(mean_b)
        diffs.append(mean_a - mean_b)

    return BlockedComparison(
        level_a=level_a,
        level_b=level_b,
        n_blocks=len(shared),
        mean_a=float(np.mean(a_means)) if a_means else float("nan"),
        mean_b=float(np.mean(b_means)) if b_means else float("nan"),
        mean_difference=float(np.mean(diffs)) if diffs else float("nan"),
        differences=diffs,
        block_ids=shared,
    )


def paired_test(differences: Sequence[float], *, alpha: float = 0.05) -> Dict[str, Any]:
    """Paired t-test plus Wilcoxon signed-rank on the same differences.

    Both are reported because they fail in different ways: the t-test is
    sensitive to outliers at small ``n``, the signed-rank test to ties. Agreement
    between them is evidence the conclusion is not an artifact of either.
    """
    diffs = np.asarray([v for v in differences if v == v], dtype=float)
    n = len(diffs)
    out: Dict[str, Any] = {"n": n, "mean_difference": float(diffs.mean()) if n else float("nan")}
    if n < 3:
        out.update({
            "t_statistic": float("nan"),
            "t_p_value": float("nan"),
            "wilcoxon_p_value": float("nan"),
            "cohens_dz": float("nan"),
            "note": "fewer than 3 paired blocks; no test performed",
        })
        return out

    # A null condition on a deterministic scorer produces near-identical
    # differences, where SciPy's moment calculation legitimately loses precision.
    # The condition is benign and expected, so it is recorded on the result rather
    # than surfaced as a warning during a long pipeline run.
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        t_stat, t_p = sps.ttest_1samp(diffs, popmean=0.0)
    if any(issubclass(w.category, RuntimeWarning) for w in caught):
        out["precision_note"] = (
            "paired differences are nearly identical, so the t-statistic loses "
            "precision; the permutation p-value is the reliable one here"
        )
    sd = float(diffs.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    crit = float(sps.t.ppf(1 - alpha / 2, df=n - 1))
    out.update(
        {
            "sd_difference": sd,
            "se_difference": se,
            "t_statistic": float(t_stat),
            "t_p_value": float(t_p),
            "df": n - 1,
            "ci_low": float(diffs.mean() - crit * se),
            "ci_high": float(diffs.mean() + crit * se),
            "cohens_dz": cohens_dz(diffs),
        }
    )
    if np.allclose(diffs, 0.0):
        out["wilcoxon_p_value"] = 1.0
    else:
        # SciPy emits a precision-loss RuntimeWarning when the differences are
        # nearly identical, which is the normal situation for a null condition on
        # a deterministic scorer. The condition is real but benign, so it is
        # recorded on the result instead of raised at the user.
        import warnings as _warnings

        try:
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                _, w_p = sps.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            out["wilcoxon_p_value"] = float(w_p)
            for warning in caught:
                if issubclass(warning.category, RuntimeWarning):
                    out["wilcoxon_note"] = (
                        "differences are nearly identical, so the signed-rank statistic "
                        "loses precision; rely on the permutation p-value here"
                    )
                    break
        except ValueError as exc:
            out["wilcoxon_p_value"] = float("nan")
            out["wilcoxon_note"] = str(exc)
    return out


def within_block_permutation_test(
    values: Sequence[float],
    groups: Sequence[str],
    blocks: Sequence[str],
    *,
    statistic: str = "mean_difference",
    level_a: Optional[str] = None,
    level_b: Optional[str] = None,
    n_permutations: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Permutation test that respects the blocking.

    Under the null "the cue does not affect the score", the cue labels within a
    block are exchangeable — the documents are identical apart from the cue. So
    shuffling labels *within* each block generates the exact null distribution of
    the design. This is the primary p-value the suite reports, because it assumes
    neither normality nor equal variances, and it is valid at the small block
    counts a student project will actually have.

    ``statistic``:
        ``mean_difference`` — signed gap between two named levels.
        ``max_abs_gap``     — largest absolute pairwise gap across all levels,
                              which is the right omnibus statistic when the
                              question is "does any group differ?" and it already
                              accounts for looking at every pair.
    """
    rng = rng or np.random.default_rng(0)

    block_map: Dict[str, List[Tuple[str, float]]] = {}
    for value, group, block in zip(values, groups, blocks):
        if value != value or group is None or group == "":
            continue
        block_map.setdefault(str(block), []).append((str(group), float(value)))

    if statistic == "mean_difference":
        if level_a is None or level_b is None:
            raise ValueError("mean_difference requires level_a and level_b")
        usable = {
            b: rows for b, rows in block_map.items()
            if any(g == level_a for g, _ in rows) and any(g == level_b for g, _ in rows)
        }
        stat_fn = lambda rows_by_block: _mean_difference(rows_by_block, level_a, level_b)  # noqa: E731
    elif statistic == "max_abs_gap":
        usable = {b: rows for b, rows in block_map.items() if len({g for g, _ in rows}) >= 2}
        stat_fn = _max_abs_gap
    else:
        raise ValueError(f"unknown statistic {statistic!r}")

    if len(usable) < 3:
        return {"p_value": float("nan"), "observed": float("nan"), "n_blocks": len(usable),
                "note": "fewer than 3 usable blocks; permutation test not performed"}

    observed = stat_fn(usable)
    if observed != observed:
        return {"p_value": float("nan"), "observed": float("nan"), "n_blocks": len(usable),
                "note": "observed statistic undefined"}

    count = 0
    for _ in range(n_permutations):
        shuffled: Dict[str, List[Tuple[str, float]]] = {}
        for block, rows in usable.items():
            labels = [g for g, _ in rows]
            scores = [v for _, v in rows]
            permuted = rng.permutation(len(labels))
            shuffled[block] = [(labels[i], scores[j]) for j, i in enumerate(permuted)]
        value = stat_fn(shuffled)
        if value == value and abs(value) >= abs(observed) - 1e-12:
            count += 1
    # (count + 1) / (n + 1) is the standard bias-corrected estimator; it never
    # reports p = 0, which would overstate certainty from a finite resample.
    p_value = (count + 1) / (n_permutations + 1)
    return {
        "p_value": float(p_value),
        "observed": float(observed),
        "n_blocks": len(usable),
        "n_permutations": n_permutations,
        "statistic": statistic,
    }


def _mean_difference(
    rows_by_block: Dict[str, List[Tuple[str, float]]], level_a: str, level_b: str
) -> float:
    diffs: List[float] = []
    for rows in rows_by_block.values():
        a = [v for g, v in rows if g == level_a]
        b = [v for g, v in rows if g == level_b]
        if a and b:
            diffs.append(float(np.mean(a) - np.mean(b)))
    return float(np.mean(diffs)) if diffs else float("nan")


def _max_abs_gap(rows_by_block: Dict[str, List[Tuple[str, float]]]) -> float:
    """Largest absolute pairwise level gap, averaged over blocks first."""
    per_level: Dict[str, List[float]] = {}
    for rows in rows_by_block.values():
        block_means: Dict[str, List[float]] = {}
        for group, value in rows:
            block_means.setdefault(group, []).append(value)
        centre = float(np.mean([np.mean(v) for v in block_means.values()]))
        for group, vals in block_means.items():
            per_level.setdefault(group, []).append(float(np.mean(vals)) - centre)
    means = {g: float(np.mean(v)) for g, v in per_level.items() if v}
    if len(means) < 2:
        return float("nan")
    return float(max(means.values()) - min(means.values()))


# --------------------------------------------------------------------------- #
# cluster-robust regression
# --------------------------------------------------------------------------- #
@dataclass
class OLSResult:
    coefficients: Dict[str, float]
    std_errors: Dict[str, float]
    t_values: Dict[str, float]
    p_values: Dict[str, float]
    ci_low: Dict[str, float]
    ci_high: Dict[str, float]
    n: int
    n_clusters: int
    r_squared: float
    df_resid: int
    note: str = ""


def cluster_robust_ols(
    y: Sequence[float],
    design: Dict[str, Sequence[float]],
    clusters: Sequence[str],
    *,
    alpha: float = 0.05,
    add_intercept: bool = True,
) -> OLSResult:
    """OLS with CR1 cluster-robust standard errors (clustered on ``clusters``).

    The sandwich estimator makes no assumption about the correlation structure
    within a cluster, which is what we want: rows sharing a resume are correlated
    in ways we cannot model. Inference uses ``t`` with ``G - 1`` degrees of
    freedom, where ``G`` is the cluster count, since the asymptotics here are in
    the number of *clusters* and not the number of rows — a distinction that
    matters a great deal when ``G`` is 12 and ``n`` is 800.
    """
    names = list(design)
    y_arr = np.asarray(y, dtype=float)
    columns = [np.asarray(design[name], dtype=float) for name in names]
    if add_intercept:
        names = ["intercept", *names]
        columns = [np.ones_like(y_arr), *columns]
    X = np.column_stack(columns) if columns else np.ones((len(y_arr), 1))

    cluster_arr = np.asarray([str(c) for c in clusters])
    mask = np.isfinite(y_arr) & np.all(np.isfinite(X), axis=1)
    y_arr, X, cluster_arr = y_arr[mask], X[mask], cluster_arr[mask]

    n, k = X.shape
    unique_clusters = np.unique(cluster_arr)
    g = len(unique_clusters)

    def _nan_result(note: str) -> OLSResult:
        nan_map = {name: float("nan") for name in names}
        return OLSResult(nan_map, nan_map, nan_map, nan_map, nan_map, nan_map,
                         n, g, float("nan"), max(0, n - k), note=note)

    if n <= k or g < 2:
        return _nan_result("insufficient rows or clusters for regression")

    # Drop columns that carry no information. A dummy level that appears in no
    # surviving row, or two levels that are perfectly collinear, make the normal
    # equations singular; a pseudo-inverse then returns coefficients of order
    # 1e15 and every downstream number silently overflows. Pruning first is both
    # numerically safe and more honest: the affected level genuinely cannot be
    # estimated from this design, and the note says so.
    keep: List[int] = []
    dropped: List[str] = []
    for column_index in range(k):
        column = X[:, column_index]
        if np.allclose(column, column[0]) and column_index != 0:
            dropped.append(names[column_index])
            continue
        candidate = X[:, keep + [column_index]]
        if np.linalg.matrix_rank(candidate) <= len(keep):
            dropped.append(names[column_index])
            continue
        keep.append(column_index)

    if len(keep) < 1:
        return _nan_result("design matrix had no estimable columns")

    kept_names = [names[i] for i in keep]
    X = X[:, keep]
    k = X.shape[1]
    if n <= k:
        return _nan_result("more parameters than observations after pruning")

    # NumPy built against Apple's Accelerate BLAS (the default wheel on arm64
    # macOS) raises spurious "divide by zero / overflow / invalid value
    # encountered in matmul" warnings once an operand crosses the size at which
    # matmul dispatches to BLAS. It does so on provably benign products — a
    # 702-row X of ones and zeros times a two-element beta triggers all three,
    # while the same product at 400 rows does not, and every element of the
    # result is finite and correct. The flags come from unused SIMD lanes, not
    # from the arithmetic.
    #
    # Suppressing the warnings is therefore right, but suppression alone would
    # also hide a real numerical failure. So the guarantee is enforced by
    # explicit finiteness checks below instead of by the warning machinery: if
    # any stage produces a non-finite value, the fit is rejected and the caller
    # gets a NaN result carrying the reason.
    with np.errstate(all="ignore"):
        xtx = X.T @ X
        try:
            beta, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
            xtx_inv = np.linalg.pinv(xtx)
        except np.linalg.LinAlgError:  # pragma: no cover
            return _nan_result("least-squares solve failed")

        if not np.all(np.isfinite(beta)):
            return _nan_result("least-squares solution was not finite")

        residuals = y_arr - X @ beta
        if not np.all(np.isfinite(residuals)):
            return _nan_result("regression residuals were not finite")

        meat = np.zeros((k, k))
        for cluster in unique_clusters:
            rows = cluster_arr == cluster
            xg = X[rows]
            ug = residuals[rows]
            score = xg.T @ ug
            meat += np.outer(score, score)

        correction = (g / (g - 1)) * ((n - 1) / max(1, n - k))
        vcov = xtx_inv @ meat @ xtx_inv * correction
        if not np.all(np.isfinite(vcov)):
            return _nan_result("cluster-robust covariance matrix was not finite")
        se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

    df = g - 1
    # Compute t and p only where the standard error is positive. Dividing the
    # whole vector and masking afterwards raises a genuine floating-point
    # divide-by-zero, and NumPy reports leaked FPU status flags against whatever
    # array operation happens next — which is how a masked division here
    # surfaces as a spurious "overflow encountered in matmul" several lines
    # later. Not generating the invalid operation is the fix; suppressing the
    # warning is not.
    t_values = np.full(k, np.nan)
    p_values = np.full(k, np.nan)
    estimable = se > 0
    if estimable.any():
        t_values[estimable] = beta[estimable] / se[estimable]
        # sf is the numerically stable form of 1 - cdf in the far tail.
        p_values[estimable] = 2.0 * sps.t.sf(np.abs(t_values[estimable]), df=df)
    crit = float(sps.t.ppf(1 - alpha / 2, df=df))

    with np.errstate(all="ignore"):
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    names = kept_names
    note = "CR1 cluster-robust standard errors; inference on G-1 df"
    if dropped:
        note += (
            f"; dropped {len(dropped)} non-estimable column(s) "
            f"({', '.join(dropped[:4])}) — those levels are collinear with others "
            "in this design and cannot be separately identified"
        )

    to_map = lambda arr: {name: float(v) for name, v in zip(names, arr)}  # noqa: E731
    return OLSResult(
        coefficients=to_map(beta),
        std_errors=to_map(se),
        t_values=to_map(t_values),
        p_values=to_map(p_values),
        ci_low=to_map(beta - crit * se),
        ci_high=to_map(beta + crit * se),
        n=n,
        n_clusters=g,
        r_squared=r_squared,
        df_resid=df,
        note=note,
    )


def dummy_code(
    labels: Sequence[str], reference: Optional[str] = None, prefix: str = "level"
) -> Tuple[Dict[str, List[float]], str]:
    """Treatment-coded dummies, returning ``(columns, reference_level)``."""
    levels = sorted({str(v) for v in labels if v is not None and str(v) != ""})
    if not levels:
        return {}, ""
    ref = reference if reference in levels else levels[0]
    columns: Dict[str, List[float]] = {}
    for level in levels:
        if level == ref:
            continue
        key = f"{prefix}[{level}]"
        columns[key] = [1.0 if str(v) == level else 0.0 for v in labels]
    return columns, ref


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def cluster_bootstrap_ci(
    block_values: Dict[str, Any],
    statistic: Callable[[List[Any]], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Percentile bootstrap CI resampling *blocks* with replacement.

    Resampling rows instead of blocks would treat correlated rows as independent
    draws and produce an interval that is far too narrow — the same error as the
    naive t-test, hidden inside a confidence interval.
    """
    rng = rng or np.random.default_rng(0)
    keys = sorted(block_values)
    if len(keys) < 3:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "n_blocks": len(keys),
                "note": "fewer than 3 blocks; bootstrap not performed"}

    point = statistic([block_values[k] for k in keys])
    draws: List[float] = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(keys), size=len(keys))
        sample = [block_values[keys[i]] for i in picked]
        value = statistic(sample)
        if value == value:
            draws.append(value)
    if len(draws) < 20:
        return {"point": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_blocks": len(keys), "note": "bootstrap statistic was mostly undefined"}
    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return {
        "point": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "n_blocks": len(keys),
        "n_boot": len(draws),
        "bootstrap_se": float(np.std(draws, ddof=1)),
    }


# --------------------------------------------------------------------------- #
# multiplicity
# --------------------------------------------------------------------------- #
def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> Dict[str, Any]:
    """Benjamini-Hochberg FDR control. Returns q-values and rejection flags.

    The suite runs dozens of tests. At ``alpha = 0.05`` uncorrected, roughly one
    in twenty nulls is "significant" by construction, which for a wide audit
    means manufacturing findings. BH is preferred over Bonferroni here because
    the goal is to control the proportion of false discoveries among reported
    findings, not to guarantee zero false positives at the cost of missing every
    real effect in a small sample.
    """
    p = np.asarray(p_values, dtype=float)
    valid = np.isfinite(p)
    q = np.full(len(p), np.nan)
    if not valid.any():
        return {"q_values": q.tolist(), "rejected": [False] * len(p), "n_tests": 0,
                "n_rejected": 0, "alpha": alpha, "method": "fdr_bh"}

    idx = np.where(valid)[0]
    pv = p[idx]
    order = np.argsort(pv, kind="stable")
    m = len(pv)
    ranked = pv[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    # Enforce monotonicity from the largest p downwards.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_sorted = np.empty(m)
    q_sorted[order] = adjusted
    q[idx] = q_sorted

    rejected = [bool(np.isfinite(v) and v <= alpha) for v in q]
    return {
        "q_values": [None if not np.isfinite(v) else float(v) for v in q],
        "rejected": rejected,
        "n_tests": int(m),
        "n_rejected": int(sum(rejected)),
        "alpha": alpha,
        "method": "fdr_bh",
    }


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> Dict[str, Any]:
    """Holm-Bonferroni FWER control, for claims that must be conservative."""
    p = np.asarray(p_values, dtype=float)
    valid = np.isfinite(p)
    adj = np.full(len(p), np.nan)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return {"adjusted_p": adj.tolist(), "rejected": [False] * len(p),
                "n_tests": 0, "method": "holm"}
    pv = p[idx]
    order = np.argsort(pv, kind="stable")
    m = len(pv)
    scaled = pv[order] * (m - np.arange(m))
    scaled = np.maximum.accumulate(scaled)
    scaled = np.clip(scaled, 0.0, 1.0)
    out = np.empty(m)
    out[order] = scaled
    adj[idx] = out
    return {
        "adjusted_p": [None if not np.isfinite(v) else float(v) for v in adj],
        "rejected": [bool(np.isfinite(v) and v <= alpha) for v in adj],
        "n_tests": int(m),
        "method": "holm",
    }


# --------------------------------------------------------------------------- #
# power
# --------------------------------------------------------------------------- #
def minimum_detectable_effect(
    n_blocks: int, sd_difference: float, *, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest paired mean difference detectable at the given power.

    Reported next to every null result. With 12 resumes, an MDE of 4 score points
    means "we could not have detected a 2-point gap even if it existed" — which
    is a completely different statement from "there is no gap", and conflating
    the two is the most common way a small fairness study overclaims.
    """
    if n_blocks < 3 or not (sd_difference > 0):
        return float("nan")
    z_alpha = float(sps.norm.ppf(1 - alpha / 2))
    z_beta = float(sps.norm.ppf(power))
    return float((z_alpha + z_beta) * sd_difference / math.sqrt(n_blocks))


def achieved_power(
    effect: float, n_blocks: int, sd_difference: float, *, alpha: float = 0.05
) -> float:
    """Post-hoc power for a paired design at the observed variability."""
    if n_blocks < 3 or not (sd_difference > 0) or effect != effect:
        return float("nan")
    ncp = abs(effect) / (sd_difference / math.sqrt(n_blocks))
    crit = float(sps.t.ppf(1 - alpha / 2, df=n_blocks - 1))
    return float(1.0 - sps.nct.cdf(crit, df=n_blocks - 1, nc=ncp)
                 + sps.nct.cdf(-crit, df=n_blocks - 1, nc=ncp))


def required_blocks(
    effect: float, sd_difference: float, *, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Blocks needed to detect ``effect``. Turns "underpowered" into a number."""
    if not (sd_difference > 0) or not (abs(effect) > 0):
        return -1
    z_alpha = float(sps.norm.ppf(1 - alpha / 2))
    z_beta = float(sps.norm.ppf(power))
    return int(math.ceil(((z_alpha + z_beta) * sd_difference / abs(effect)) ** 2))


# --------------------------------------------------------------------------- #
# proportions
# --------------------------------------------------------------------------- #
def two_proportion_test(
    successes_a: int, n_a: int, successes_b: int, n_b: int, *, alpha: float = 0.05
) -> Dict[str, Any]:
    """Fisher's exact test plus a Wilson-style difference interval.

    Fisher rather than chi-square because top-``k`` selection counts are small
    and often include zero cells, where the chi-square approximation fails.
    """
    if n_a <= 0 or n_b <= 0:
        return {"p_value": float("nan"), "note": "empty group"}
    table = [[successes_a, n_a - successes_a], [successes_b, n_b - successes_b]]
    try:
        _, p_value = sps.fisher_exact(table)
    except ValueError:  # pragma: no cover
        p_value = float("nan")
    rate_a = successes_a / n_a
    rate_b = successes_b / n_b
    diff = rate_a - rate_b
    se = math.sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)
    z = float(sps.norm.ppf(1 - alpha / 2))
    return {
        "rate_a": round(rate_a, 4),
        "rate_b": round(rate_b, 4),
        "difference": round(diff, 4),
        "ci_low": round(diff - z * se, 4),
        "ci_high": round(diff + z * se, 4),
        "impact_ratio": round(rate_a / rate_b, 4) if rate_b > 0 else None,
        "p_value": float(p_value),
        "test": "fisher_exact",
        "n_a": n_a,
        "n_b": n_b,
    }


def describe(values: Sequence[float]) -> Dict[str, float]:
    """Standard descriptives, NaN-safe."""
    arr = np.asarray([v for v in values if v == v], dtype=float)
    if len(arr) == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
