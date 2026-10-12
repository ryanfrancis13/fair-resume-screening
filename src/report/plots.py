"""Figures for the audit report.

Matplotlib with the ``Agg`` backend, so plotting works on a headless machine and
in CI.  Two conventions run through every figure, and both are choices about
honesty rather than aesthetics:

* **Uncertainty is always drawn.** A bar chart of group means with no error bars
  invites the reader to see a difference that the data may not support. Every mean
  carries a confidence interval or a per-point scatter.
* **The noise floor is drawn where it exists.** A shaded band showing the model's
  own decoding variation lets a reader see immediately whether an effect clears
  it, without having to hold a number in their head.

Figures are saved to disk *and* returned base64-encoded, so the HTML report is a
single self-contained file that survives being emailed.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..utils.frames import block_key, nonempty_mask, text_column  # noqa: E402

logger = logging.getLogger("frs.report.plots")

# A colour-blind-safe qualitative palette (Okabe & Ito).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
GRID_KWARGS = {"alpha": 0.25, "linestyle": ":", "linewidth": 0.8}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Optional[Path], dpi: int) -> Dict[str, Any]:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    saved = None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved = str(path)
    plt.close(fig)
    return {"path": saved, "data_uri": f"data:image/png;base64,{encoded}"}


# --------------------------------------------------------------------------- #
def score_distribution_by_group(
    frame: pd.DataFrame,
    attribute: str,
    *,
    value_column: str = "overall_score",
    path: Optional[Path] = None,
    dpi: int = 150,
    noise_floor_sd: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Within-candidate-centred score by demographic level, with 95% CIs.

    Centring within candidate is what makes the comparison meaningful: raw group
    means are dominated by which resumes happened to receive which cue.
    """
    column = f"signal_{attribute}"
    if column not in frame.columns:
        return None
    data = frame[nonempty_mask(frame, column)]
    data = data[data[value_column].notna()]
    if "parse_ok" in data.columns:
        data = data[data["parse_ok"].astype(bool)]
    if data.empty or data[column].nunique() < 2:
        return None

    data = data.copy()
    data["_block"] = block_key(data, "job_id", "base_resume_id")
    data["_centred"] = data[value_column].astype(float) - data.groupby("_block")[
        value_column
    ].transform("mean")

    levels = sorted(text_column(data, column).unique())
    means, los, his, counts = [], [], [], []
    for level in levels:
        values = data.loc[text_column(data, column) == level, "_centred"].to_numpy(dtype=float)
        mean = float(values.mean())
        sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        means.append(mean)
        los.append(1.96 * sem)
        his.append(1.96 * sem)
        counts.append(len(values))

    _style()
    fig, ax = plt.subplots(figsize=(max(6.0, 1.25 * len(levels)), 4.2))
    positions = np.arange(len(levels))
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(levels))]
    ax.bar(positions, means, yerr=[los, his], capsize=4, color=colours, alpha=0.85,
           edgecolor="#333333", linewidth=0.6)
    ax.axhline(0.0, color="#333333", linewidth=1.0)

    if noise_floor_sd:
        ax.axhspan(-noise_floor_sd, noise_floor_sd, color="#999999", alpha=0.18, zorder=0)
        ax.text(
            0.995, 0.02,
            f"shaded: ±1 decoding-noise SD ({noise_floor_sd:.2f} pts)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{lvl}\n(n={n})" for lvl, n in zip(levels, counts)], rotation=20, ha="right"
    )
    ax.set_ylabel("score deviation from the\ncandidate's own mean (points)")
    ax.set_title(f"Score by {attribute.replace('_', ' ')}, centred within candidate")
    ax.grid(axis="y", **GRID_KWARGS)
    return _save(fig, path, dpi)


def cue_effect_forest(
    cue_effects: Sequence[Dict[str, Any]],
    *,
    path: Optional[Path] = None,
    dpi: int = 150,
    max_rows: int = 24,
    practical_threshold: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Forest plot of per-cue-level effects, sorted by magnitude.

    The single most informative figure in the report: it puts every cue on one
    axis so name effects and non-name effects can be compared directly, which is
    the project's central empirical question.
    """
    rows = [
        r for r in cue_effects
        if r.get("mean_delta") is not None and r.get("mean_delta") == r.get("mean_delta")
    ]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["mean_delta"])
    if len(rows) > max_rows:
        half = max_rows // 2
        rows = rows[:half] + rows[-half:]

    labels = [f"{r['cue_ids']}" for r in rows]
    values = np.array([float(r["mean_delta"]) for r in rows])
    errors = np.array([
        1.96 * float(r["se_delta"]) if r.get("se_delta") == r.get("se_delta") else 0.0
        for r in rows
    ])
    families = [str(r.get("cue_families", "")) for r in rows]
    unique_families = sorted(set(families))
    colour_by_family = {
        fam: PALETTE[i % len(PALETTE)] for i, fam in enumerate(unique_families)
    }

    _style()
    fig, ax = plt.subplots(figsize=(8.4, max(3.6, 0.28 * len(rows) + 1.4)))
    positions = np.arange(len(rows))
    for i, (value, error, family) in enumerate(zip(values, errors, families)):
        ax.errorbar(
            value, i, xerr=error, fmt="o", markersize=5, capsize=3,
            color=colour_by_family[family], elinewidth=1.4,
        )
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    if practical_threshold:
        for sign in (-1, 1):
            ax.axvline(sign * practical_threshold, color="#B00020", linewidth=0.9,
                       linestyle="--", alpha=0.7)
        ax.text(
            practical_threshold, len(rows) - 0.4,
            f" practical threshold ±{practical_threshold:g}",
            fontsize=8, color="#B00020", va="top",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("mean score change vs the candidate's own baseline (points, 95% CI)")
    ax.set_title("Effect of each cue on the score, holding the resume constant")
    ax.grid(axis="x", **GRID_KWARGS)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colour_by_family[f], label=f)
        for f in unique_families
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=min(3, len(handles)))
    return _save(fig, path, dpi)


def candidate_swing_plot(
    per_candidate: Sequence[Dict[str, Any]],
    *,
    path: Optional[Path] = None,
    dpi: int = 150,
    tolerance: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Distribution of per-candidate score swings under cue swaps.

    The individual-fairness view: each point is one real candidate whose score
    moved by that much when only their identity cues changed.
    """
    swings = [
        float(c["max_swing"]) for c in per_candidate
        if c.get("max_swing") is not None
    ]
    if len(swings) < 3:
        return None

    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(swings, bins=min(20, max(6, len(swings) // 2)), color=PALETTE[0],
            alpha=0.8, edgecolor="white")
    ax.axvline(float(np.mean(swings)), color=PALETTE[1], linewidth=1.6,
               label=f"mean = {np.mean(swings):.2f}")
    if tolerance:
        ax.axvline(tolerance, color="#B00020", linestyle="--", linewidth=1.2,
                   label=f"tolerance = {tolerance:g}")
    ax.set_xlabel("largest score swing across a candidate's own cue variants (points)")
    ax.set_ylabel("number of candidate/job pairs")
    ax.set_title("Individual sensitivity to identity cues")
    ax.legend(frameon=False)
    ax.grid(axis="y", **GRID_KWARGS)
    return _save(fig, path, dpi)


def selection_rate_plot(
    analysis: Dict[str, Any],
    *,
    path: Optional[Path] = None,
    dpi: int = 150,
) -> Optional[Dict[str, Any]]:
    """Top-k selection rate by group, with the four-fifths reference line."""
    rates = analysis.get("selection_rate") or {}
    if len(rates) < 2:
        return None
    levels = sorted(rates)
    values = [float(rates[lvl]) for lvl in levels]
    counts = analysis.get("n_candidates", {})
    best = max(values)

    _style()
    fig, ax = plt.subplots(figsize=(max(6.0, 1.25 * len(levels)), 4.0))
    positions = np.arange(len(levels))
    ax.bar(positions, values, color=[PALETTE[i % len(PALETTE)] for i in range(len(levels))],
           alpha=0.85, edgecolor="#333333", linewidth=0.6)
    if best > 0:
        ax.axhline(0.8 * best, color="#B00020", linestyle="--", linewidth=1.1,
                   label="four-fifths of the highest rate")
        ax.legend(frameon=False, loc="upper right")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{lvl}\n(n={counts.get(lvl, 0)})" for lvl in levels], rotation=20, ha="right"
    )
    ax.set_ylabel(f"top-{analysis.get('k', '?')} selection rate")
    ax.set_title(
        f"Shortlisting rate by {str(analysis.get('attribute', '')).replace('_', ' ')} "
        f"(top-{analysis.get('k', '?')})"
    )
    ax.grid(axis="y", **GRID_KWARGS)
    return _save(fig, path, dpi)


def variance_components_plot(
    reliability: Dict[str, Any],
    *,
    path: Optional[Path] = None,
    dpi: int = 150,
) -> Optional[Dict[str, Any]]:
    """Where score variance comes from: candidate, cue, or decoding noise."""
    components = reliability.get("variance_components") or {}
    usable = {
        k: v for k, v in components.items()
        if isinstance(v, (int, float)) and v == v and v > 0
    }
    if len(usable) < 2:
        return None

    labels = ["between_resume", "between_cue", "within_cue_noise"]
    labels = [label for label in labels if label in usable]
    values = [usable[label] for label in labels]
    pretty = {
        "between_resume": "between candidates\n(the scorer working)",
        "between_cue": "between cues\n(the bias signal)",
        "within_cue_noise": "repeat decodes\n(model noise)",
    }

    _style()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    positions = np.arange(len(labels))
    ax.bar(positions, values, color=[PALETTE[0], PALETTE[1], PALETTE[4]][: len(labels)],
           alpha=0.85, edgecolor="#333333", linewidth=0.6)
    total = sum(values)
    for i, value in enumerate(values):
        ax.text(i, value, f"{100 * value / total:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels([pretty[label] for label in labels], fontsize=9)
    ax.set_ylabel("variance (score points²)")
    ax.set_title("Where does score variance come from?")
    ax.grid(axis="y", **GRID_KWARGS)
    return _save(fig, path, dpi)


def mitigation_plot(
    utility_rows: Sequence[Dict[str, Any]],
    *,
    path: Optional[Path] = None,
    dpi: int = 150,
) -> Optional[Dict[str, Any]]:
    """Fairness gain against utility cost, one point per mitigation arm.

    Plotted as a scatter rather than a bar chart because the whole point is the
    trade-off: an arm in the top-left improved fairness while keeping the
    ranking, and one in the bottom-right bought fairness by destroying it.
    """
    rows = [
        r for r in utility_rows
        if r.get("rank_agreement") is not None and r.get("metrics_improved") is not None
    ]
    if not rows:
        return None

    _style()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for i, row in enumerate(rows):
        colour = PALETTE[i % len(PALETTE)]
        ax.scatter(
            float(row["rank_agreement"]), int(row["metrics_improved"]),
            s=140, color=colour, alpha=0.85, edgecolor="#333333", linewidth=0.7,
            label=f"{row['arm']} ({row.get('verdict', '')})",
        )
    ax.axvline(0.6, color="#B00020", linestyle="--", linewidth=1.0)
    ax.text(0.6, ax.get_ylim()[1], " minimum acceptable\n rank agreement",
            fontsize=8, color="#B00020", va="top")
    ax.set_xlabel("rank agreement with the unmitigated baseline (Kendall tau)")
    ax.set_ylabel("number of headline bias metrics improved")
    ax.set_title("Mitigation: fairness gained vs screening signal kept")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(**GRID_KWARGS)
    return _save(fig, path, dpi)


def reliability_diagram(
    per_group: Dict[str, Any],
    *,
    path: Optional[Path] = None,
    dpi: int = 150,
) -> Optional[Dict[str, Any]]:
    """Reliability curves per group: does a score mean the same thing for everyone?"""
    usable = {
        level: info for level, info in per_group.items()
        if info.get("reliability_bins")
    }
    if len(usable) < 2:
        return None

    _style()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1.0,
            label="perfect calibration")
    for i, (level, info) in enumerate(sorted(usable.items())):
        bins = info["reliability_bins"]
        xs = [b["mean_confidence"] for b in bins]
        ys = [b["observed_rate"] for b in bins]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5,
                color=PALETTE[i % len(PALETTE)],
                label=f"{level} (ECE {info.get('ece', float('nan')):.3f})")
    ax.set_xlabel("predicted probability of advancing")
    ax.set_ylabel("observed rate of advancing")
    ax.set_title("Calibration by group")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(**GRID_KWARGS)
    return _save(fig, path, dpi)


def blinding_leakage_plot(
    rows: Sequence[Dict[str, Any]],
    *,
    path: Optional[Path] = None,
    dpi: int = 150,
) -> Optional[Dict[str, Any]]:
    """Demographic signal still detectable at each redaction level.

    The visual form of the project's central claim: name-level anonymisation
    leaves demographic signal in the document.
    """
    if len(rows) < 2:
        return None
    labels = [str(r["level"]) for r in rows]
    detected = [int(r.get("n_detected_attributes", 0)) for r in rows]
    hits = [int(r.get("residual_cue_hits", 0)) for r in rows]

    _style()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    positions = np.arange(len(labels))
    width = 0.38
    ax.bar(positions - width / 2, detected, width, label="protected attributes still inferable",
           color=PALETTE[1], alpha=0.85, edgecolor="#333333", linewidth=0.5)
    ax.bar(positions + width / 2, hits, width, label="residual proxy-cue term hits",
           color=PALETTE[0], alpha=0.85, edgecolor="#333333", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("count (mean per resume)")
    ax.set_title("What survives redaction")
    ax.legend(frameon=False)
    ax.grid(axis="y", **GRID_KWARGS)
    return _save(fig, path, dpi)


def build_all_plots(
    frame: pd.DataFrame,
    report: Any,
    *,
    out_dir: Path,
    dpi: int = 150,
    extras: Optional[Dict[str, Any]] = None,
    mitigation_rows: Optional[Sequence[Dict[str, Any]]] = None,
    leakage_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Generate every applicable figure. A failing plot never fails the report."""
    from ..fairness.suite import available_attributes

    out: Dict[str, Dict[str, Any]] = {}
    noise_sd = report.reliability.get("noise_floor_sd") if report.reliability.get("has_repeats") else None
    tolerance = report.provenance.get("config", {}).get("counterfactual_tolerance")
    practical = report.provenance.get("config", {}).get("min_practical_effect")

    def attempt(name: str, function, *args, **kwargs) -> None:
        try:
            produced = function(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a figure is never worth a crash
            logger.warning("plot %s failed: %s: %s", name, type(exc).__name__, exc)
            return
        if produced:
            out[name] = produced

    attempt(
        "variance_components", variance_components_plot, report.reliability,
        path=out_dir / "variance_components.png", dpi=dpi,
    )

    for attribute in available_attributes(frame):
        attempt(
            f"scores_by_{attribute}", score_distribution_by_group, frame, attribute,
            path=out_dir / f"scores_by_{attribute}.png", dpi=dpi, noise_floor_sd=noise_sd,
        )
        allocational = next(
            (m for m in report.metrics
             if m.family == "allocational" and m.attribute == attribute
             and m.name.startswith("selection_rate_parity_diff")),
            None,
        )
        if allocational is not None and isinstance(allocational.detail, dict):
            attempt(
                f"selection_rate_{attribute}", selection_rate_plot, allocational.detail,
                path=out_dir / f"selection_rate_{attribute}.png", dpi=dpi,
            )

    swing_metric = next(
        (m for m in report.metrics if m.name.startswith("counterfactual_max_swing[all]")), None
    )
    if swing_metric is not None:
        from ..fairness.counterfactual import counterfactual_invariance

        analysis = counterfactual_invariance(
            frame, tolerance=tolerance or 2.0, n_boot=200,
        )
        if "per_candidate" in analysis:
            attempt(
                "candidate_swings", candidate_swing_plot, analysis["per_candidate"],
                path=out_dir / "candidate_swings.png", dpi=dpi, tolerance=tolerance,
            )

    if extras and extras.get("cue_effects"):
        attempt(
            "cue_effects", cue_effect_forest, extras["cue_effects"],
            path=out_dir / "cue_effects.png", dpi=dpi, practical_threshold=practical,
        )

    calibration = next(
        (m for m in report.metrics if m.name.startswith("calibration_ece_spread")), None
    )
    if calibration is not None and isinstance(calibration.detail, dict):
        per_group = calibration.detail.get("per_group")
        if per_group:
            attempt(
                "calibration_by_group", reliability_diagram, per_group,
                path=out_dir / "calibration_by_group.png", dpi=dpi,
            )

    if mitigation_rows:
        attempt(
            "mitigation", mitigation_plot, mitigation_rows,
            path=out_dir / "mitigation.png", dpi=dpi,
        )
    if leakage_rows:
        attempt(
            "blinding_leakage", blinding_leakage_plot, leakage_rows,
            path=out_dir / "blinding_leakage.png", dpi=dpi,
        )
    return out
