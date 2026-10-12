"""Score calibration.

A raw "78/100 fit score" is an ordinal impression, not a probability, and it
cannot be compared across jobs or across model versions.  Calibration maps
scores onto ``P(a human recruiter advances this candidate)`` using a small
labelled set, which buys two things the audit needs:

* **Comparability.** Calibrated probabilities are on a common scale, so a
  threshold means the same thing for a software role and a finance role.
* **A second, independent fairness lens.** Chouldechova (2017) and Kleinberg et
  al. (2017) show that equal calibration across groups and equal error rates
  across groups are *mathematically incompatible* whenever base rates differ.
  Measuring both is therefore not redundancy — it is the only way to state which
  fairness criterion a system satisfies and which it must give up. The
  incompatibility is the finding, and the project's brief specifically asks for
  metric disagreement to be explained rather than treated as a bug.

Two mappings are provided: Platt scaling (one-parameter logistic, robust with
tens of labels) and isotonic regression (non-parametric, needs hundreds).  With a
small labelled set Platt is the honest choice, and the module says so rather than
silently overfitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class CalibrationModel:
    """A fitted score → probability mapping."""

    method: str
    params: Dict[str, float] = field(default_factory=dict)
    knots_x: List[float] = field(default_factory=list)
    knots_y: List[float] = field(default_factory=list)
    n_train: int = 0
    warnings: List[str] = field(default_factory=list)

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        x = np.asarray(scores, dtype=float)
        if self.method == "platt":
            a = self.params.get("a", 1.0)
            b = self.params.get("b", 0.0)
            z = np.clip(a * (x / 100.0) + b, -60, 60)
            return 1.0 / (1.0 + np.exp(-z))
        if self.method == "isotonic":
            if not self.knots_x:
                return np.full_like(x, 0.5)
            return np.interp(x, self.knots_x, self.knots_y,
                             left=self.knots_y[0], right=self.knots_y[-1])
        if self.method == "identity":
            return np.clip(x / 100.0, 0.0, 1.0)
        raise ValueError(f"unknown calibration method {self.method!r}")


def _penalised_nll(x: np.ndarray, t: np.ndarray, a: float, b: float, l2: float) -> float:
    z = np.clip(a * x + b, -60, 60)
    # log(1 + exp(z)) computed stably.
    log1pexp = np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
    return float(np.sum(log1pexp - t * z) + 0.5 * l2 * (a * a + b * b))


def _logistic_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 200,
    targets: Optional[np.ndarray] = None,
    l2: float = 1e-3,
    max_abs_coef: float = 50.0,
) -> tuple[float, float]:
    """Newton-Raphson fit of ``sigmoid(a*x + b)`` to ``y``. Returns ``(a, b)``.

    Written out rather than imported from scikit-learn so the fairness-critical
    arithmetic stays readable: a reviewer can check the gradient, the Hessian and
    the safeguards directly instead of taking a library's word for it.

    Two safeguards matter, and both are load-bearing rather than decorative:

    * A small **ridge penalty**. Unpenalised logistic MLE diverges to infinite
      coefficients whenever the data are linearly separable, which is common
      here — a strongly discriminative scorer separates its labelled set almost
      perfectly.
    * **Backtracking line search**, plus a coefficient clamp. The penalty alone
      is not enough: once ``a*x`` saturates the sigmoid, the Hessian collapses to
      the penalty term and a raw Newton step becomes enormous, which is exactly
      how a calibration slope comes out as ``1.4e6``. Requiring each step to
      actually reduce the penalised negative log-likelihood keeps the fit finite,
      and leaves well-conditioned cases unchanged to four decimals.
    """
    t = np.asarray(y if targets is None else targets, dtype=float)
    x = np.asarray(x, dtype=float)
    a, b = 1.0, 0.0
    objective = _penalised_nll(x, t, a, b, l2)

    for _ in range(max_iter):
        z = np.clip(a * x + b, -60, 60)
        p = 1.0 / (1.0 + np.exp(-z))
        g_a = float(np.sum((p - t) * x)) + l2 * a
        g_b = float(np.sum(p - t)) + l2 * b
        w = p * (1.0 - p) + 1e-12
        h_aa = float(np.sum(w * x * x)) + l2 + 1e-9
        h_ab = float(np.sum(w * x))
        h_bb = float(np.sum(w)) + l2 + 1e-9
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det

        step = 1.0
        improved = False
        for _backtrack in range(12):
            cand_a = float(np.clip(a - step * da, -max_abs_coef, max_abs_coef))
            cand_b = float(np.clip(b - step * db, -max_abs_coef, max_abs_coef))
            cand_obj = _penalised_nll(x, t, cand_a, cand_b, l2)
            if cand_obj <= objective + 1e-12:
                a, b, objective = cand_a, cand_b, cand_obj
                improved = True
                break
            step *= 0.5
        if not improved or max(abs(step * da), abs(step * db)) < 1e-9:
            break
    return a, b


def fit_platt(scores: Sequence[float], labels: Sequence[int], *, max_iter: int = 200) -> CalibrationModel:
    """One-dimensional logistic regression by Newton-Raphson.

    Implemented here rather than pulled from scikit-learn to keep the dependency
    footprint small and, more importantly, to keep the fairness-critical maths
    visible: a reviewer can read the update step and check it.
    """
    x = np.asarray(scores, dtype=float) / 100.0
    y = np.asarray(labels, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    notes: List[str] = []
    if len(x) < 8:
        notes.append(f"only {len(x)} labelled examples; calibration is indicative only")
    if len(set(y.tolist())) < 2:
        notes.append("labels are single-class; falling back to identity mapping")
        return CalibrationModel(method="identity", n_train=len(x), warnings=notes)

    # Platt's prior correction keeps the fit from saturating on tiny samples.
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    hi = (n_pos + 1.0) / (n_pos + 2.0)
    lo = 1.0 / (n_neg + 2.0)
    t = np.where(y > 0.5, hi, lo)

    a, b = _logistic_fit(x, y, max_iter=max_iter, targets=t)
    return CalibrationModel(method="platt", params={"a": a, "b": b}, n_train=len(x), warnings=notes)


def fit_isotonic(scores: Sequence[float], labels: Sequence[int]) -> CalibrationModel:
    """Pool-adjacent-violators isotonic regression."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    notes: List[str] = []
    if len(x) < 30:
        notes.append(
            f"isotonic regression with {len(x)} points will overfit; prefer Platt scaling"
        )
    if len(x) == 0:
        return CalibrationModel(method="identity", warnings=["no labelled data"])

    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    values = ys.astype(float).tolist()
    weights = [1.0] * len(values)
    knots = xs.tolist()

    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1] + 1e-12:
            i += 1
            continue
        total_w = weights[i] + weights[i + 1]
        pooled = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / total_w
        values[i] = pooled
        weights[i] = total_w
        del values[i + 1]
        del weights[i + 1]
        del knots[i + 1]
        if i > 0:
            i -= 1
    return CalibrationModel(
        method="isotonic", knots_x=knots, knots_y=values, n_train=len(x), warnings=notes
    )


def fit_calibration(
    scores: Sequence[float], labels: Sequence[int], method: str = "platt"
) -> CalibrationModel:
    if method == "platt":
        return fit_platt(scores, labels)
    if method == "isotonic":
        return fit_isotonic(scores, labels)
    if method == "identity":
        return CalibrationModel(method="identity", n_train=len(scores))
    raise ValueError(f"unknown calibration method {method!r}")


# --------------------------------------------------------------------------- #
# calibration quality
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[int], *, n_bins: int = 10
) -> Dict[str, Any]:
    """Binned ECE and MCE, plus the reliability-diagram data."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if len(p) == 0:
        return {"ece": float("nan"), "mce": float("nan"), "n": 0, "bins": []}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins: List[Dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        conf = float(p[in_bin].mean())
        acc = float(y[in_bin].mean())
        gap = abs(acc - conf)
        ece += (count / len(p)) * gap
        mce = max(mce, gap)
        bins.append(
            {"bin_low": round(lo, 3), "bin_high": round(hi, 3), "n": count,
             "mean_confidence": round(conf, 4), "observed_rate": round(acc, 4),
             "gap": round(gap, 4)}
        )
    return {"ece": round(ece, 4), "mce": round(mce, 4), "n": int(len(p)), "bins": bins}


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    return float(np.mean((p[mask] - y[mask]) ** 2))


def auc_roc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC via the Mann-Whitney U identity, with tie correction."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    mask = np.isfinite(s) & np.isfinite(y)
    s, y = s[mask], y[mask]
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([pos, neg]))
    rank_sum_pos = float(ranks[: len(pos)].sum())
    u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # average ties
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def calibration_slope_intercept(
    probabilities: Sequence[float], labels: Sequence[int]
) -> Dict[str, float]:
    """Cox's calibration slope and intercept: regress the outcome on the logit.

    Fitting ``logit(P(y=1)) = intercept + slope * logit(p)`` gives the textbook
    recalibration diagnostic. Slope 1 with intercept 0 means calibrated;
    slope < 1 means overconfident spread; a non-zero intercept means a systematic
    level shift.

    When these differ across demographic groups the same score threshold means
    different things for different candidates — a fairness failure invisible to
    any score-gap metric, because the gap can be zero while the thresholds mean
    different things.
    """
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if len(p) < 8 or len(set(y.tolist())) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "n": int(len(p))}
    logit = np.log(p / (1 - p))
    slope, intercept = _logistic_fit(logit, y)
    return {
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "n": int(len(p)),
        "interpretation": (
            "slope 1.0 and intercept 0.0 indicate calibration; slope below 1.0 indicates "
            "overconfident spread"
        ),
    }


def calibration_report(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    model: Optional[CalibrationModel] = None,
    method: str = "platt",
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Fit (or reuse) a calibration model and report its quality."""
    finite = [
        (score, label)
        for score, label in zip(scores, labels)
        if score == score
        and label is not None
        and not (isinstance(label, float) and math.isnan(label))
    ]
    if not finite:
        return {"n": 0, "error": "no labelled scores"}
    xs = [s for s, _ in finite]
    ys = [int(label) for _, label in finite]
    fitted = model or fit_calibration(xs, ys, method=method)
    probabilities = fitted.predict(xs)
    return {
        "n": len(xs),
        "method": fitted.method,
        "params": fitted.params,
        "warnings": fitted.warnings,
        "auc": round(auc_roc(xs, ys), 4),
        "brier": round(brier_score(probabilities, ys), 4),
        "base_rate": round(float(np.mean(ys)), 4),
        "mean_predicted": round(float(np.mean(probabilities)), 4),
        **expected_calibration_error(probabilities, ys, n_bins=n_bins),
    }
