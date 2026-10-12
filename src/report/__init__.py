"""Report generation: plots, HTML/Markdown, explanation cards, model card."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from ..utils.io import atomic_write_text, write_json
from .explanation_cards import build_cards, cards_summary, render_card_markdown
from .html import render_html, render_markdown
from .model_card import build_model_card, render_model_card_markdown
from .plots import build_all_plots

logger = logging.getLogger("frs.report")

__all__ = [
    "build_all_plots",
    "build_cards",
    "build_model_card",
    "build_report",
    "cards_summary",
    "render_card_markdown",
    "render_html",
    "render_markdown",
    "render_model_card_markdown",
]


def _leakage_rows(state: Any) -> List[Dict[str, Any]]:
    """Average redaction leakage across the corpus, for the blinding section.

    Averaged over resumes rather than shown for one example, so the table is a
    property of the corpus and not an anecdote.
    """
    from ..anonymize import compare_levels

    if not state.resumes:
        return []
    accumulated: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    for resume in state.resumes:
        try:
            rows = compare_levels(resume)
        except Exception as exc:  # noqa: BLE001 - a report section is not worth a crash
            logger.warning("leakage comparison failed for %s: %s", resume.resume_id, exc)
            continue
        for row in rows:
            level = str(row["level"])
            bucket = accumulated.setdefault(level, {})
            for key, value in row.items():
                if key == "level" or not isinstance(value, (int, float)):
                    continue
                bucket[key] = bucket.get(key, 0.0) + float(value)
            counts[level] = counts.get(level, 0) + 1

    order = ["none", "identity", "identity_plus_proxy", "aggressive"]
    out: List[Dict[str, Any]] = []
    for level in order:
        if level not in accumulated:
            continue
        n = max(1, counts[level])
        row: Dict[str, Any] = {"level": level}
        for key, total in accumulated[level].items():
            row[key] = round(total / n, 2)
        out.append(row)
    return out


def build_report(state: Any) -> Dict[str, Path]:
    """Render every configured report format for a finished run.

    Takes the pipeline state rather than individual arguments so that the report
    always describes the run that actually happened — there is no way to hand it a
    metric table from one run and a manifest from another.
    """
    from ..fairness.suite import extras_for_report, headline_findings, metrics_table

    cfg = state.cfg
    report = state.audit_report
    if report is None:
        raise ValueError("no audit report on the pipeline state; run the audit stage first")

    out_dir = state.root / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}

    frame = state.frames_by_arm.get("baseline")
    if frame is None or frame.empty:
        frame = state.combined_frame()

    extras = extras_for_report(frame, cfg)
    findings = headline_findings(report)
    metrics = metrics_table(report)
    leakage = _leakage_rows(state)

    mitigation = None
    mitigation_rows: List[Dict[str, Any]] = []
    if state.mitigation:
        from ..mitigate import utility_table

        mitigation = {k: v for k, v in state.mitigation.items() if k != "reports"}
        costs = utility_table(state.mitigation)
        mitigation_rows = costs.to_dict(orient="records") if not costs.empty else []

    plots: Dict[str, Any] = {}
    if cfg.report.make_plots:
        plots = build_all_plots(
            frame, report, out_dir=out_dir, dpi=cfg.report.plot_dpi,
            extras=extras, mitigation_rows=mitigation_rows, leakage_rows=leakage,
        )

    cards: List[Dict[str, Any]] = []
    card_stats: Dict[str, Any] = {}
    if cfg.report.explanation_cards:
        cards = build_cards(
            frame, max_cards=cfg.report.max_cards,
            advance_threshold=cfg.screening.advance_threshold,
            borderline_threshold=cfg.screening.borderline_threshold,
        )
        card_stats = cards_summary(cards)
        if cards:
            outputs["cards_json"] = write_json(out_dir / "explanation_cards.json", cards)
            outputs["cards_markdown"] = atomic_write_text(
                out_dir / "explanation_cards.md",
                "# Explanation cards\n\n"
                "Ordered most cue-sensitive first: those are the decisions a human most "
                "needs to review.\n\n"
                + "\n".join(render_card_markdown(card) for card in cards),
            )

    model_card_markdown = ""
    if cfg.report.model_card:
        card = build_model_card(
            report,
            manifest=state.manifest.data,
            mitigation=mitigation,
            corpus_stats=state.manifest.data.get("stages", {})
            .get("prepare_data", {})
            .get("corpus_stats", {}),
            sanity=state.sanity,
        )
        model_card_markdown = render_model_card_markdown(card)
        outputs["model_card_json"] = write_json(out_dir / "model_card.json", card)
        outputs["model_card_markdown"] = atomic_write_text(
            out_dir / "MODEL_CARD.md", model_card_markdown
        )

    formats = [f.lower() for f in cfg.report.formats]
    if "html" in formats:
        outputs["html"] = atomic_write_text(
            out_dir / "report.html",
            render_html(
                report, findings=findings, plots=plots, extras=extras, cards=cards,
                cards_summary=card_stats, mitigation=mitigation,
                manifest=state.manifest.data, model_card_markdown=model_card_markdown,
                leakage_rows=leakage,
            ),
        )
    if "markdown" in formats:
        outputs["markdown"] = atomic_write_text(
            out_dir / "report.md",
            render_markdown(
                report, findings=findings, metrics_frame=metrics, mitigation=mitigation,
                cards=cards, extras=extras, leakage_rows=leakage,
            ),
        )
    if "json" in formats:
        outputs["json"] = write_json(
            out_dir / "report.json",
            {
                "audit_report": report,
                "headline_findings": findings,
                "extras": extras,
                "leakage": leakage,
                "cards_summary": card_stats,
                "mitigation": mitigation,
            },
        )
    if not metrics.empty:
        metrics.to_csv(out_dir / "metrics.csv", index=False)
        outputs["metrics_csv"] = out_dir / "metrics.csv"
    if leakage:
        outputs["leakage_json"] = write_json(out_dir / "anonymisation_leakage.json", leakage)

    return outputs
