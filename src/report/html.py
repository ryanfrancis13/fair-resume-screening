"""HTML and Markdown report generation.

The report is a single self-contained HTML file with figures embedded as data
URIs, so it can be emailed, attached to a submission, or opened years later
without the run directory.

It is written as an argument rather than a data dump, and the ordering encodes
the argument:

1. **Can we trust the scorer?** Validity and reliability first. If the answer is
   no, nothing below it means anything, and the reader should learn that before
   reading a bias number rather than after.
2. **Headline findings** — ranked by consequence, not by p-value.
3. **Does bias survive anonymisation?** The leakage table: the project's central
   claim in one place.
4. **Cue effects** — names and non-name cues on the same axis, which is the
   comparison the research gap is about.
5. **Every metric**, by family, with intervals and corrected q-values.
6. **Metric conflicts**, explained rather than merely displayed.
7. **Mitigation**, with utility cost attached.
8. **Explanation cards** for the candidates whose decisions were least stable.
9. **Model card** and full provenance.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ..schemas import AuditReport
from ..utils.io import atomic_write_text

logger = logging.getLogger("frs.report.html")

_CSS = """
:root {
  --ink: #1a1c1f; --muted: #5b6169; --line: #e3e6ea; --bg: #ffffff;
  --soft: #f7f8fa; --accent: #0b5cad; --pass: #1a7f4b; --fail: #b3261e;
  --warn: #9a6200; --info: #4a4f57;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 40px 24px 96px; }
header.hero { border-bottom: 3px solid var(--ink); padding-bottom: 20px; margin-bottom: 8px; }
h1 { font-size: 30px; line-height: 1.2; margin: 0 0 6px; letter-spacing: -0.02em; }
h2 { font-size: 21px; margin: 44px 0 12px; padding-top: 18px; border-top: 1px solid var(--line); }
h3 { font-size: 16px; margin: 26px 0 8px; }
h4 { font-size: 14px; margin: 18px 0 6px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
p { margin: 0 0 12px; }
a { color: var(--accent); }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.9em; }
.sub { color: var(--muted); font-size: 13px; }
.grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); margin: 18px 0 26px; }
.card { border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; background: var(--soft); }
.card .label { font-size: 11px; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); }
.card .value { font-size: 25px; font-weight: 650; margin-top: 4px; letter-spacing: -0.02em; }
.card .note { font-size: 12px; color: var(--muted); margin-top: 4px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 22px; font-size: 13.5px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { background: var(--soft); font-weight: 620; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
tbody tr:hover { background: #fbfcfd; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px; font-weight: 650; letter-spacing: .03em; }
.tag.pass { background: #e6f4ec; color: var(--pass); }
.tag.fail { background: #fdeceb; color: var(--fail); }
.tag.info { background: #eef0f3; color: var(--info); }
.tag.warn { background: #fdf3e2; color: var(--warn); }
.callout { border-left: 4px solid var(--accent); background: var(--soft); padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 16px 0; }
.callout.warn { border-left-color: var(--warn); background: #fffaf0; }
.callout.fail { border-left-color: var(--fail); background: #fff5f5; }
.callout.pass { border-left-color: var(--pass); background: #f2faf5; }
.callout h4 { margin-top: 0; color: inherit; }
figure { margin: 20px 0 26px; }
figure img { width: 100%; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
figcaption { font-size: 12.5px; color: var(--muted); margin-top: 8px; }
.scroll { overflow-x: auto; }
details { border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; margin: 10px 0; background: var(--soft); }
details summary { cursor: pointer; font-weight: 600; font-size: 14px; }
details[open] summary { margin-bottom: 10px; }
.cardgrid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
.candidate { border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.candidate.unstable { border-color: var(--fail); background: #fffafa; }
.band { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #f6c6c2, #b6d5f0); margin: 8px 0 4px; }
ul.flags { margin: 8px 0 0; padding-left: 18px; }
ul.flags li { font-size: 13px; margin-bottom: 4px; }
footer { margin-top: 60px; padding-top: 16px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12.5px; }
@media print { body { font-size: 11pt; } .wrap { max-width: none; padding: 0; } h2 { page-break-after: avoid; } figure, table, .candidate { page-break-inside: avoid; } }
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:
            return "—"
        return f"{value:.{digits}f}"
    return _esc(value)


def _fmt_p(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    value = float(value)
    return f"{value:.4f}" if value >= 1e-4 else f"{value:.1e}"


def _verdict_tag(passed: Optional[bool]) -> str:
    if passed is True:
        return '<span class="tag pass">PASS</span>'
    if passed is False:
        return '<span class="tag fail">FAIL</span>'
    return '<span class="tag info">INFO</span>'


def _stat_card(label: str, value: Any, note: str = "") -> str:
    return (
        '<div class="card">'
        f'<div class="label">{_esc(label)}</div>'
        f'<div class="value">{value if isinstance(value, str) else _fmt(value)}</div>'
        + (f'<div class="note">{_esc(note)}</div>' if note else "")
        + "</div>"
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, numeric_from: int = 1) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body: List[str] = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            klass = ' class="num"' if i >= numeric_from and _looks_numeric(cell) else ""
            cells.append(f"<td{klass}>{cell}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="scroll"><table><thead><tr>' + head + "</tr></thead><tbody>"
        + "".join(body) + "</tbody></table></div>"
    )


def _looks_numeric(cell: str) -> bool:
    stripped = str(cell).replace("−", "-").replace(",", "").strip()
    try:
        float(stripped)
    except ValueError:
        return False
    return True


def _figure(plot: Dict[str, Any], caption: str) -> str:
    if not plot or not plot.get("data_uri"):
        return ""
    return (
        f'<figure><img src="{plot["data_uri"]}" alt="{_esc(caption)}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def _section_validity(report: AuditReport, plots: Dict[str, Any]) -> str:
    reliability = report.reliability or {}
    sanity = (report.provenance or {}).get("sanity_check") or {}
    consistency = reliability.get("self_consistency", {})
    quality = reliability.get("parse_quality", {})

    verdict = sanity.get("verdict", "not run")
    tone = {"pass": "pass", "review": "warn", "insufficient_data": "warn"}.get(verdict, "warn")

    parts = [
        "<h2>1. Can the scorer be trusted?</h2>",
        "<p>Fairness metrics from a scorer that does not discriminate on job-relevant "
        "content are worthless: a constant function is perfectly unbiased and perfectly "
        "useless. This section comes first because it decides whether anything below it "
        "can be interpreted.</p>",
        '<div class="grid">',
        _stat_card(
            "Phase-2 discrimination", verdict.replace("_", " "),
            f"Kendall tau vs expected fit = {_fmt(sanity.get('kendall_tau'))}",
        ),
        _stat_card(
            "Reliability (ICC)", reliability.get("icc1_resume"),
            "share of variance between candidates",
        ),
        _stat_card(
            "Decoding noise floor", reliability.get("noise_floor_sd"),
            "SD across repeats of an identical prompt",
        ),
        _stat_card(
            "Cue effect / noise", reliability.get("cue_effect_to_noise_ratio"),
            "above 2.0 means cue effects clear the noise",
        ),
        _stat_card(
            "Evidence grounding", quality.get("mean_evidence_support_rate"),
            "fraction of quotes found in the resume",
        ),
        _stat_card(
            "Parse success", quality.get("parse_ok_rate"),
            f"{quality.get('n_parse_ok', 0)}/{quality.get('n_records', 0)} records usable",
        ),
        "</div>",
    ]

    if sanity.get("note"):
        parts.append(
            f'<div class="callout {tone}"><h4>Phase-2 gate</h4><p>{_esc(sanity["note"])}</p></div>'
        )
    if reliability.get("interpretation"):
        parts.append(f"<p>{_esc(reliability['interpretation'])}</p>")
    if consistency.get("interpretation"):
        parts.append(f"<p>{_esc(consistency['interpretation'])}</p>")

    means = sanity.get("mean_score_by_expected_fit") or {}
    if means:
        parts.append("<h4>Mean score by hand-graded fit</h4>")
        parts.append(
            _table(
                ["expected fit", "mean score", "n pairs"],
                [
                    [_esc(fit), _fmt(value, 2),
                     _esc((sanity.get("n_by_expected_fit") or {}).get(fit, 0))]
                    for fit, value in means.items()
                ],
            )
        )

    parts.append(
        _figure(
            plots.get("variance_components", {}),
            "Variance decomposition. The between-candidate share is the scorer doing its "
            "job; the between-cue share is the bias signal; the repeat-decode share is the "
            "noise floor that any claimed effect must clear.",
        )
    )

    warnings = quality.get("warnings") or []
    if warnings:
        parts.append('<div class="callout warn"><h4>Data-quality warnings</h4><ul>')
        parts += [f"<li>{_esc(w)}</li>" for w in warnings]
        parts.append("</ul></div>")
    return "".join(parts)


def _section_headline(report: AuditReport, findings: Sequence[Dict[str, Any]]) -> str:
    parts = [
        "<h2>2. Headline findings</h2>",
        "<p>Ranked by consequence rather than by p-value: a changed decision or a failed "
        "impact ratio outranks a statistically significant sub-point score gap, because it "
        "describes a candidate being treated differently.</p>",
    ]
    rows = [
        [
            f'<code>{_esc(f["metric"])}</code>',
            _esc(f["family"]),
            _fmt(f["value"]),
            _esc(f["unit"]),
            _fmt_p(f["q_value_bh"]),
            {"FAIL": '<span class="tag fail">FAIL</span>',
             "PASS": '<span class="tag pass">PASS</span>'}.get(
                f["verdict"], '<span class="tag info">INFO</span>'),
        ]
        for f in findings
    ]
    parts.append(
        _table(["metric", "family", "value", "unit", "q (BH)", "verdict"], rows, numeric_from=2)
    )
    for finding in findings[:4]:
        if finding.get("interpretation"):
            parts.append(
                f'<div class="callout"><p><strong>{_esc(finding["metric"])}</strong> — '
                f'{_esc(finding["interpretation"])}</p></div>'
            )
    if report.warnings:
        parts.append('<div class="callout warn"><h4>Run warnings</h4><ul>')
        parts += [f"<li>{_esc(w)}</li>" for w in report.warnings]
        parts.append("</ul></div>")
    return "".join(parts)


def _section_leakage(
    leakage_rows: Sequence[Dict[str, Any]], plots: Dict[str, Any]
) -> str:
    if not leakage_rows:
        return ""
    parts = [
        "<h2>3. Does anonymisation remove the signal?</h2>",
        "<p>This is the project's central claim, measured directly. Each row applies a "
        "redaction level to the corpus and then asks the cue detector how much demographic "
        "signal it can still recover from the redacted text.</p>",
        _table(
            ["redaction level", "substitutions", "lines removed",
             "protected attributes still inferable", "residual proxy-cue hits"],
            [
                [
                    _esc(row["level"]), _esc(row.get("n_substitutions", 0)),
                    _esc(row.get("removed_lines", 0)),
                    _fmt(row.get("n_detected_attributes"), 2),
                    _fmt(row.get("residual_cue_hits"), 2),
                ]
                for row in leakage_rows
            ],
        ),
        _figure(
            plots.get("blinding_leakage", {}),
            "Demographic signal surviving each redaction level, averaged per resume. "
            "Name-level redaction leaves proxy cues intact.",
        ),
        '<div class="callout"><p>Name-only redaction is the industry-standard "blind '
        'screening" intervention. Anything still inferable in the <code>identity</code> row '
        "is a cue that such a policy misses, which is exactly the gap the non-name cue "
        "families in this audit are designed to test.</p></div>",
    ]
    return "".join(parts)


def _section_cue_effects(
    extras: Dict[str, Any], plots: Dict[str, Any], report: AuditReport
) -> str:
    effects = extras.get("cue_effects") or []
    if not effects:
        return ""
    ordered = sorted(effects, key=lambda r: r.get("mean_delta", 0.0))
    practical = (report.provenance.get("config", {}) or {}).get("min_practical_effect", 1.0)

    parts = [
        "<h2>4. What each cue does to the score</h2>",
        "<p>Every value is the mean change in a candidate's score relative to "
        "<em>their own</em> baseline, so resume strength is differenced out and the "
        "numbers are directly comparable. Name cues and non-name cues appear on the same "
        "axis — the comparison this project exists to make.</p>",
        _figure(
            plots.get("cue_effects", {}),
            "Per-cue effect on the score with 95% confidence intervals, holding the resume "
            "constant. Dashed lines mark the practical-effect threshold.",
        ),
    ]
    rows = []
    for row in ordered[:12] + ([] if len(ordered) <= 24 else ordered[-12:]):
        delta = row.get("mean_delta")
        flag = ""
        if delta is not None and abs(delta) >= practical:
            flag = ' <span class="tag fail">above threshold</span>'
        rows.append(
            [
                f'<code>{_esc(row["cue_ids"])}</code>{flag}',
                _esc(row.get("cue_families", "")),
                _fmt(delta, 2),
                _fmt(row.get("se_delta"), 3),
                _esc(row.get("n_candidates", 0)),
                _fmt(row.get("min_delta"), 2),
                _fmt(row.get("max_delta"), 2),
            ]
        )
    parts.append(
        _table(
            ["cue level", "family", "mean Δ", "SE", "n candidates", "min Δ", "max Δ"],
            rows, numeric_from=2,
        )
    )
    parts.append(
        _figure(
            plots.get("candidate_swings", {}),
            "Per-candidate score swing across their own identity variants. Each observation "
            "is a real candidate whose score moved by that much for reasons unrelated to "
            "their qualifications.",
        )
    )
    return "".join(parts)


def _section_metrics(report: AuditReport, plots: Dict[str, Any]) -> str:
    families = {}
    for metric in report.metrics:
        families.setdefault(metric.family, []).append(metric)

    titles = {
        "counterfactual": ("Counterfactual (individual) fairness",
                           "Would this candidate's score change if only their identity cues "
                           "changed? The design makes this a direct measurement rather than "
                           "an inference."),
        "group_parity": ("Group score parity",
                         "Mean score differences between demographic levels, tested with "
                         "within-candidate blocking. Each row carries its own minimum "
                         "detectable effect, so a null can be read as underpowered rather "
                         "than as fair."),
        "allocational": ("Allocational fairness",
                         "Who actually reaches the shortlist. Selection rates, the "
                         "four-fifths impact ratio, and position-discounted exposure."),
        "ranking": ("Ranking stability",
                    "Whether a cue swap reorders candidates — including who ranks first."),
        "intersectional": ("Intersectional fairness",
                           "Subgroup cells, and whether the worst cell is worse than the sum "
                           "of the marginal effects predicts."),
        "rationale": ("Rationale language audit",
                      "Whether the model's written explanations differ across groups even "
                      "where the scores do not."),
        "calibration": ("Calibration and error rates",
                        "Whether a given score means the same thing for every group, and the "
                        "impossibility trade-off that follows when base rates differ."),
    }

    order = ["counterfactual", "group_parity", "allocational", "ranking",
             "intersectional", "rationale", "calibration"]
    parts = ["<h2>5. All metrics by family</h2>"]

    for family in order + [f for f in sorted(families) if f not in order]:
        metrics = families.get(family)
        if not metrics:
            continue
        title, blurb = titles.get(family, (family.replace("_", " ").title(), ""))
        parts.append(f"<h3>{_esc(title)}</h3>")
        if blurb:
            parts.append(f'<p class="sub">{_esc(blurb)}</p>')
        rows = []
        for metric in metrics:
            rows.append(
                [
                    f'<code>{_esc(metric.name)}</code>',
                    _fmt(metric.value),
                    (f"{_fmt(metric.ci_low, 2)} … {_fmt(metric.ci_high, 2)}"
                     if metric.ci_low == metric.ci_low else "—"),
                    _fmt_p(metric.p_value),
                    _fmt_p(metric.q_value),
                    (f"{_fmt(metric.effect_size, 2)} ({_esc(metric.effect_size_name)})"
                     if metric.effect_size == metric.effect_size else "—"),
                    _esc(metric.n),
                    _verdict_tag(metric.passed),
                ]
            )
        parts.append(
            _table(
                ["metric", "value", "95% CI", "p", "q (BH)", "effect size", "n", "verdict"],
                rows, numeric_from=1,
            )
        )
        for metric in metrics:
            if metric.interpretation:
                parts.append(
                    f'<p class="sub"><code>{_esc(metric.name)}</code>: '
                    f"{_esc(metric.interpretation)}</p>"
                )
        for key, plot in plots.items():
            if family == "group_parity" and key.startswith("scores_by_"):
                parts.append(
                    _figure(plot, f"Within-candidate-centred score by "
                                  f"{key[len('scores_by_'):].replace('_', ' ')}, with 95% "
                                  "confidence intervals and the decoding-noise band.")
                )
            if family == "allocational" and key.startswith("selection_rate_"):
                parts.append(
                    _figure(plot, f"Top-k shortlisting rate by "
                                  f"{key[len('selection_rate_'):].replace('_', ' ')}, with the "
                                  "four-fifths reference line.")
                )
            if family == "calibration" and key == "calibration_by_group":
                parts.append(
                    _figure(plot, "Calibration curves per group. Divergence means the same "
                                  "score carries a different meaning depending on group.")
                )

    skipped = (report.provenance or {}).get("skipped_families") or []
    if skipped:
        parts.append('<div class="callout warn"><h4>Metric families not computed</h4><ul>')
        parts += [
            f"<li><strong>{_esc(s['family'])}</strong>: {_esc(s['reason'])}</li>"
            for s in skipped
        ]
        parts.append(
            "</ul><p>An uncomputed family is not a passing family. These are listed so an "
            "absent metric cannot be mistaken for a clean result.</p></div>"
        )
    return "".join(parts)


def _section_conflicts(report: AuditReport) -> str:
    parts = [
        "<h2>6. Metric conflicts, and why they are expected</h2>",
        "<p>Several fairness criteria are provably incompatible. A report whose metrics all "
        "agree is either measuring one thing repeatedly or has not looked hard enough. What "
        "follows names each disagreement, attributes it to its cause, and says what to do "
        "about it.</p>",
    ]
    if not report.conflicts:
        parts.append(
            '<div class="callout pass"><p>No metric conflicts were detected in this run. '
            "With a small corpus this can also mean several metric families were "
            "unavailable, so check the list of uncomputed families above before reading "
            "this as agreement.</p></div>"
        )
        return "".join(parts)

    severity_class = {"critical": "fail", "important": "warn", "expected": ""}
    for conflict in report.conflicts:
        klass = severity_class.get(conflict["severity"], "")
        parts.append(f'<div class="callout {klass}">')
        parts.append(
            f'<h4>{_esc(conflict["type"].replace("_", " "))} — '
            f'{_esc(conflict["severity"])}'
            + (f' · <code>{_esc(conflict["attribute"])}</code>' if conflict.get("attribute") else "")
            + "</h4>"
        )
        parts.append(f"<p>{_esc(conflict['explanation'])}</p>")
        parts.append(f"<p><strong>What to do.</strong> {_esc(conflict['recommendation'])}</p>")
        if conflict.get("metrics"):
            parts.append(
                '<p class="sub">Metrics involved: '
                + ", ".join(f"<code>{_esc(m)}</code>" for m in conflict["metrics"])
                + "</p>"
            )
        if conflict.get("citation"):
            parts.append(f'<p class="sub">{_esc(conflict["citation"])}</p>')
        parts.append("</div>")
    return "".join(parts)


def _section_mitigation(
    mitigation: Optional[Dict[str, Any]], plots: Dict[str, Any]
) -> str:
    if not mitigation or not mitigation.get("comparisons"):
        return (
            "<h2>7. Mitigation</h2>"
            '<div class="callout warn"><p>No mitigation arms were evaluated in this run. '
            "Enable them with <code>mitigation.strategies</code> in the config.</p></div>"
        )

    parts = [
        "<h2>7. Mitigation: what helps, and what it costs</h2>",
        f"<p>{_esc(mitigation.get('guidance', ''))}</p>",
        _figure(
            plots.get("mitigation", {}),
            "Each mitigation arm plotted as fairness gained against screening signal kept. "
            "Top-left is the desirable quadrant; bottom-right buys fairness by destroying "
            "the ranking.",
        ),
    ]

    rows = []
    for arm, comparison in mitigation["comparisons"].items():
        utility = comparison.get("utility", {})
        verdict = comparison.get("verdict", "")
        tag = {
            "recommended": '<span class="tag pass">recommended</span>',
            "partial": '<span class="tag info">partial</span>',
            "cue_variation_eliminated_by_construction":
                '<span class="tag info">cue variation removed by design</span>',
            "fairness_improved_at_utility_cost": '<span class="tag warn">costly</span>',
            "inconclusive": '<span class="tag info">inconclusive</span>',
            "not_recommended": '<span class="tag fail">not recommended</span>',
        }.get(verdict, f'<span class="tag info">{_esc(verdict)}</span>')
        rows.append(
            [
                _esc(arm), tag,
                _esc(comparison.get("n_metrics_improved", 0)),
                _esc(comparison.get("n_metrics_worsened", 0)),
                _fmt(utility.get("rank_agreement_kendall_tau")),
                _fmt(utility.get("top_k_overlap")),
                _fmt(utility.get("score_dispersion_ratio")),
                _fmt(utility.get("coverage")),
            ]
        )
    parts.append(
        _table(
            ["arm", "verdict", "metrics improved", "metrics worsened",
             "rank agreement", "top-k overlap", "dispersion ratio", "coverage"],
            rows, numeric_from=2,
        )
    )

    for arm, comparison in mitigation["comparisons"].items():
        details = comparison.get("metric_comparisons") or []
        if not details:
            continue
        parts.append(f"<details><summary>{_esc(arm)} — metric-by-metric change</summary>")
        parts.append(
            _table(
                ["metric", "baseline", "arm", "change", "relative", "improved"],
                [
                    [
                        f'<code>{_esc(d["metric"])}</code>', _fmt(d["baseline"]),
                        _fmt(d["arm"]), _fmt(d["absolute_change"]),
                        ("—" if d["relative_change_pct"] is None
                         else f'{d["relative_change_pct"]:+.1f}%'),
                        ('<span class="tag pass">yes</span>' if d["improved"]
                         else '<span class="tag fail">no</span>'),
                    ]
                    for d in details
                ],
            )
        )
        for reason in comparison.get("verdict_reasons", []):
            parts.append(f'<p class="sub">{_esc(reason)}</p>')
        parts.append("</details>")
    return "".join(parts)


def _section_cards(cards: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    if not cards:
        return ""
    parts = [
        "<h2>8. Explanation cards</h2>",
        "<p>Aggregate metrics describe a population; a recruiter decides about a person. "
        "These cards are ordered most cue-sensitive first, because those are the decisions "
        "a human most needs to look at.</p>",
        '<div class="grid">',
        _stat_card("Cards", summary.get("n_cards")),
        _stat_card(
            "Decision-unstable", summary.get("n_decision_unstable"),
            "outcome changes under identity cues alone",
        ),
        _stat_card("Mean swing", summary.get("mean_swing"), "score points"),
        _stat_card("Worst swing", summary.get("max_swing"), "score points"),
        "</div>",
    ]
    if summary.get("note"):
        parts.append(f'<p class="sub">{_esc(summary["note"])}</p>')

    parts.append('<div class="cardgrid">')
    for card in cards[:12]:
        sensitivity = card["cue_sensitivity"]
        unstable = " unstable" if sensitivity["crosses_decision_boundary"] else ""
        parts.append(f'<div class="candidate{unstable}">')
        parts.append(
            f'<h3><code>{_esc(card["resume_id"])}</code> → '
            f'<code>{_esc(card["job_id"])}</code></h3>'
        )
        parts.append(
            f'<p><strong>{_fmt(card["score"], 1)}</strong> · '
            f'{_esc(card["recommendation"] or "n/a")}</p>'
        )
        if sensitivity["n_variants"] > 1:
            parts.append('<div class="band"></div>')
            parts.append(
                f'<p class="sub">Identity-variant band '
                f'{_fmt(sensitivity["score_low"], 1)} – {_fmt(sensitivity["score_high"], 1)} '
                f'(swing {_fmt(sensitivity["swing"], 1)} over '
                f'{sensitivity["n_variants"]} variants)</p>'
            )
            if sensitivity.get("least_favourable_cue"):
                parts.append(
                    f'<p class="sub">Lowest under <code>'
                    f'{_esc(sensitivity["least_favourable_cue"])}</code>, highest under '
                    f'<code>{_esc(sensitivity["most_favourable_cue"])}</code></p>'
                )
        if card["dimensions"]:
            parts.append(
                _table(
                    ["dimension", "score"],
                    [
                        [_esc(d["name"].replace("_", " ")), _fmt(d["score"], 1)]
                        for d in card["dimensions"]
                    ],
                )
            )
        if card["missing_requirements"]:
            parts.append(
                '<p class="sub"><strong>Unevidenced requirements:</strong> '
                + ", ".join(_esc(r) for r in card["missing_requirements"][:5])
                + "</p>"
            )
        if card.get("evidence_support_rate") is not None:
            parts.append(
                f'<p class="sub">Evidence grounding '
                f'{100 * card["evidence_support_rate"]:.0f}%</p>'
            )
        if card["flags"]:
            parts.append('<ul class="flags">')
            parts += [f"<li>{_esc(flag)}</li>" for flag in card["flags"]]
            parts.append("</ul>")
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def _section_provenance(report: AuditReport, manifest: Dict[str, Any]) -> str:
    git = manifest.get("git", {})
    environment = manifest.get("environment", {})
    parts = [
        "<h2>9. Provenance</h2>",
        "<p>Everything needed to reproduce or challenge these numbers.</p>",
        _table(
            ["item", "value"],
            [
                ["run id", f'<code>{_esc(report.run_id)}</code>'],
                ["created", _esc(report.created_at)],
                ["pipeline version", _esc(manifest.get("frs_version", "unknown"))],
                ["config hash", f'<code>{_esc(manifest.get("config_hash", ""))}</code>'],
                ["seed", _esc(manifest.get("seed"))],
                ["provider / model",
                 f'<code>{_esc(report.provenance.get("provider", ""))}</code> / '
                 f'<code>{_esc(report.provenance.get("model", ""))}</code>'],
                ["git commit",
                 (f'<code>{_esc(git.get("commit", "")[:12])}</code>'
                  + (" <span class=\"tag warn\">dirty tree</span>" if git.get("dirty") else ""))
                 if git.get("available") else "not a git checkout"],
                ["python", _esc(environment.get("python", ""))],
                ["records / variants / resumes / jobs",
                 f"{report.n_records} / {report.n_variants} / "
                 f"{report.n_base_resumes} / {report.n_jobs}"],
                ["multiplicity correction",
                 _esc(report.provenance.get("multiplicity", {}).get("method", ""))
                 + f' ({report.provenance.get("multiplicity", {}).get("n_tests", 0)} tests, '
                 f'{report.provenance.get("multiplicity", {}).get("n_rejected", 0)} rejected)'],
            ],
            numeric_from=99,
        ),
    ]
    stages = manifest.get("stages", {})
    if stages:
        parts.append("<details><summary>Stage timings and telemetry</summary>")
        parts.append(
            _table(
                ["stage", "elapsed (s)", "notes"],
                [
                    [
                        _esc(name), _fmt(payload.get("elapsed_s"), 2),
                        _esc(_stage_note(payload)),
                    ]
                    for name, payload in stages.items()
                ],
            )
        )
        parts.append("</details>")
    return "".join(parts)


def _stage_note(payload: Dict[str, Any]) -> str:
    for key in ("n_variants", "n_records"):
        if key in payload:
            return f"{key} = {payload[key]}"
    telemetry = payload.get("telemetry") or {}
    if telemetry:
        return (
            f"{telemetry.get('n_records', 0)} records, "
            f"{telemetry.get('n_parse_failures', 0)} parse failures, "
            f"cache hit rate "
            f"{(telemetry.get('client', {}).get('cache', {}) or {}).get('hit_rate', 0):.2f}"
        )
    summary = payload.get("summary") or {}
    if summary:
        return f"{summary.get('n_metrics', 0)} metrics, {summary.get('n_failed', 0)} failed"
    corpus = payload.get("corpus_stats") or {}
    if corpus:
        return (
            f"{corpus.get('n_resumes')} resumes, {corpus.get('n_jobs')} jobs, "
            f"{corpus.get('n_pairs')} pairs"
        )
    return ""


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def render_html(
    report: AuditReport,
    *,
    findings: Sequence[Dict[str, Any]],
    plots: Dict[str, Any],
    extras: Dict[str, Any],
    cards: Sequence[Dict[str, Any]],
    cards_summary: Dict[str, Any],
    mitigation: Optional[Dict[str, Any]],
    manifest: Dict[str, Any],
    model_card_markdown: str = "",
    leakage_rows: Sequence[Dict[str, Any]] = (),
) -> str:
    provider = report.provenance.get("provider", "")
    model = report.provenance.get("model", "")
    body = "".join(
        [
            '<div class="wrap">',
            '<header class="hero">',
            "<h1>Fair Resume Screening — Fairness Audit</h1>",
            f'<p class="sub">Run <code>{_esc(report.run_id)}</code> · '
            f'{_esc(report.created_at)} · scorer <code>{_esc(provider)}/{_esc(model)}</code> · '
            f"{report.n_records} score records over {report.n_base_resumes} resumes, "
            f"{report.n_jobs} roles and {report.n_variants} counterfactual variants</p>",
            "</header>",
            _section_validity(report, plots),
            _section_headline(report, findings),
            _section_leakage(leakage_rows, plots),
            _section_cue_effects(extras, plots, report),
            _section_metrics(report, plots),
            _section_conflicts(report),
            _section_mitigation(mitigation, plots),
            _section_cards(cards, cards_summary),
            _section_provenance(report, manifest),
            _model_card_details(model_card_markdown),
            "<footer>",
            "<p>Generated by the Fair Resume Screening pipeline. This audit measures whether "
            "changing an identity cue changes a score, a recommendation or a ranking "
            "position. It does not establish what the correct ranking is, and it is not a "
            "legal compliance determination. The system supports a human recruiter; it does "
            "not make hiring decisions.</p>",
            "</footer>",
            "</div>",
        ]
    )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>Fairness Audit — {_esc(report.run_id)}</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )


def _model_card_details(markdown: str) -> str:
    if not markdown:
        return ""
    return (
        "<h2>10. Model card</h2>"
        "<details><summary>Full model card / audit datasheet (Markdown source)</summary>"
        f"<pre class=\"mono\" style=\"white-space:pre-wrap;font-size:12px\">{_esc(markdown)}</pre>"
        "</details>"
    )


def render_markdown(
    report: AuditReport,
    *,
    findings: Sequence[Dict[str, Any]],
    metrics_frame: pd.DataFrame,
    mitigation: Optional[Dict[str, Any]],
    cards: Sequence[Dict[str, Any]],
    extras: Dict[str, Any],
    leakage_rows: Sequence[Dict[str, Any]] = (),
) -> str:
    """Markdown version, for pasting into a report document or a PR."""
    from .explanation_cards import render_card_markdown

    reliability = report.reliability or {}
    sanity = (report.provenance or {}).get("sanity_check") or {}
    lines: List[str] = [
        "# Fair Resume Screening — Fairness Audit",
        "",
        f"**Run:** `{report.run_id}` · {report.created_at}  ",
        f"**Scorer:** `{report.provenance.get('provider', '')}/"
        f"{report.provenance.get('model', '')}`  ",
        f"**Scale:** {report.n_records} score records · {report.n_base_resumes} resumes · "
        f"{report.n_jobs} roles · {report.n_variants} variants",
        "",
        "## 1. Can the scorer be trusted?",
        "",
        f"- Phase-2 discrimination check: **{sanity.get('verdict', 'not run')}** "
        f"(Kendall tau vs expected fit = {sanity.get('kendall_tau')})",
        f"- Reliability (ICC) = {reliability.get('icc1_resume')}",
        f"- Decoding noise floor (SD) = {reliability.get('noise_floor_sd')}",
        f"- Cue effect / noise ratio = {reliability.get('cue_effect_to_noise_ratio')}",
        "",
    ]
    if reliability.get("interpretation"):
        lines += [reliability["interpretation"], ""]

    lines += ["## 2. Headline findings", "",
              "| metric | value | unit | q (BH) | verdict |", "| --- | --- | --- | --- | --- |"]
    for finding in findings:
        lines.append(
            f"| `{finding['metric']}` | {finding['value']} | {finding['unit']} | "
            f"{finding['q_value_bh']} | {finding['verdict']} |"
        )
    lines.append("")
    for finding in findings[:5]:
        if finding.get("interpretation"):
            lines += [f"- **{finding['metric']}** — {finding['interpretation']}", ""]

    if leakage_rows:
        lines += [
            "## 3. What survives anonymisation",
            "",
            "| level | substitutions | attributes still inferable | residual cue hits |",
            "| --- | --- | --- | --- |",
        ]
        for row in leakage_rows:
            lines.append(
                f"| {row['level']} | {row.get('n_substitutions', 0)} | "
                f"{row.get('n_detected_attributes')} | {row.get('residual_cue_hits')} |"
            )
        lines.append("")

    effects = extras.get("cue_effects") or []
    if effects:
        lines += ["## 4. Cue effects", "",
                  "| cue level | family | mean Δ | SE | n |", "| --- | --- | --- | --- | --- |"]
        for row in sorted(effects, key=lambda r: r.get("mean_delta", 0.0))[:20]:
            lines.append(
                f"| `{row['cue_ids']}` | {row.get('cue_families', '')} | "
                f"{row.get('mean_delta', float('nan')):.2f} | "
                f"{row.get('se_delta', float('nan')):.3f} | {row.get('n_candidates', 0)} |"
            )
        lines.append("")

    if metrics_frame is not None and not metrics_frame.empty:
        lines += ["## 5. All metrics", "", markdown_table(metrics_frame), ""]

    lines += ["## 6. Metric conflicts", ""]
    if not report.conflicts:
        lines += ["No conflicts detected.", ""]
    for conflict in report.conflicts:
        lines += [
            f"### {conflict['type']} ({conflict['severity']})",
            "",
            conflict["explanation"],
            "",
            f"**What to do.** {conflict['recommendation']}",
            "",
        ]

    lines += ["## 7. Mitigation", ""]
    if mitigation and mitigation.get("comparisons"):
        lines += ["| arm | verdict | improved | worsened | rank agreement | dispersion |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for arm, comparison in mitigation["comparisons"].items():
            utility = comparison.get("utility", {})
            lines.append(
                f"| {arm} | {comparison.get('verdict')} | "
                f"{comparison.get('n_metrics_improved')} | "
                f"{comparison.get('n_metrics_worsened')} | "
                f"{utility.get('rank_agreement_kendall_tau')} | "
                f"{utility.get('score_dispersion_ratio')} |"
            )
        lines.append("")
    else:
        lines += ["No mitigation arms were evaluated.", ""]

    if cards:
        lines += ["## 8. Explanation cards (most cue-sensitive first)", ""]
        for card in cards[:6]:
            lines.append(render_card_markdown(card))

    if report.warnings:
        lines += ["## 9. Warnings", ""] + [f"- {w}" for w in report.warnings] + [""]

    lines += [
        "---",
        "",
        "This audit measures whether changing an identity cue changes a score, a "
        "recommendation, or a ranking position. It does not establish the correct ranking, "
        "and it is not a legal compliance determination.",
        "",
    ]
    return "\n".join(lines)


def markdown_table(frame: pd.DataFrame, *, max_rows: int = 400) -> str:
    """Render a DataFrame as a Markdown table.

    Hand-rolled rather than ``DataFrame.to_markdown``, which requires the
    optional ``tabulate`` package. A report should not fail to render because of
    a transitive dependency the pipeline never declared.
    """
    if frame is None or frame.empty:
        return "_(no rows)_"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(str(c) for c in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.head(max_rows).iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if value is None or (isinstance(value, float) and value != value):
                cells.append("—")
            elif isinstance(value, float):
                cells.append(f"{value:g}")
            else:
                # Pipes and newlines would break the table structure.
                cells.append(str(value).replace("|", "\\|").replace("\n", " ")[:220])
        lines.append("| " + " | ".join(cells) + " |")
    if len(frame) > max_rows:
        lines.append(f"| _… {len(frame) - max_rows} more rows_ |" + " |" * (len(columns) - 1))
    return "\n".join(lines)


def write_html(path: Path, content: str) -> Path:
    return atomic_write_text(path, content)
