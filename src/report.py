"""Pipeline runner, HTML/Markdown/JSON report builder, and 5-check auditor validation oracle."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .fairness import evaluate_mitigations, records_to_df, run_audit
from .models import AuditReport, Config, ScoreRecord, load_benchmark_corpus
from .screening import sanity_check, score_batch


def build_explanation_cards(records: list[ScoreRecord], max_cards: int = 15) -> list[dict[str, Any]]:
    df = records_to_df(records)
    s0 = df[df["sample_index"] == 0]
    cards: list[dict[str, Any]] = []
    for pid, grp in s0.groupby("pair_id"):
        prot = grp[grp["cue_type"] == "protected"]
        if prot.empty:
            continue
        swing = float(prot["overall_score"].max() - prot["overall_score"].min())
        flips = int(prot["decision"].nunique()) > 1
        cards.append({
            "pair_id": str(pid),
            "resume_id": str(grp["resume_id"].iloc[0]),
            "job_id": str(grp["job_id"].iloc[0]),
            "expected_fit": str(grp["expected_fit"].iloc[0]),
            "min_score": round(float(prot["overall_score"].min()), 2),
            "max_score": round(float(prot["overall_score"].max()), 2),
            "max_protected_swing": round(swing, 2),
            "decision_flipped": flips,
            "decisions_observed": sorted(prot["decision"].unique().tolist()),
        })
    cards.sort(key=lambda c: (c["decision_flipped"], c["max_protected_swing"]), reverse=True)
    return cards[:max_cards]


def render_reports(
    report: AuditReport,
    records: list[ScoreRecord],
    sanity: dict[str, Any],
    run_dir: Path,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    records_to_df(records).to_csv(run_dir / "scores.csv", index=False)
    cards = build_explanation_cards(records)

    (run_dir / "audit_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "explanation_cards.json").write_text(json.dumps(cards, indent=2), encoding="utf-8")

    md_lines = [
        f"# Counterfactual Fairness Audit Report — `{report.run_id}`",
        f"- **Provider / Model**: `{report.provider}` / `{report.model}`",
        f"- **Corpus**: {report.n_resumes} resumes, {report.n_jobs} jobs, {report.n_variants} variants, {report.n_scores} scored rows",
        f"- **Sanity Gate**: passed={sanity['passed']} (Kendall tau={sanity['kendall_tau']}, strong-weak gap={sanity['strong_weak_gap']})",
        f"- **Reliability**: ICC(1)={report.reliability.get('icc_1')}, noise floor SD={report.reliability.get('noise_floor_sd')}",
        "",
        "## Headline Findings",
        *[f"- {h}" for h in report.headline_findings],
        "",
        "## Fairness Metrics",
        "| Metric | Family | Attribute | Value | 95% CI | q-value | Status |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for m in report.metrics:
        ci = f"[{m.ci_low:.2f}, {m.ci_high:.2f}]" if (m.ci_low or m.ci_high) else "—"
        q = f"{m.p_value_adjusted:.4f}" if m.p_value_adjusted is not None else "—"
        md_lines.append(f"| `{m.name}` | {m.family} | {m.attribute} | {m.value:.3f} | {ci} | {q} | {'PASS' if m.passed else 'FAIL'} |")

    (run_dir / "report.md").write_text("\n".join(md_lines), encoding="utf-8")
    (run_dir / "MODEL_CARD.md").write_text(
        f"# Model Card — {report.model}\n\n"
        f"- **Intended Use**: Structured 6-dimension rubric resume screening with mandatory counterfactual fairness auditing.\n"
        f"- **Evaluated Run**: `{report.run_id}` ({report.created_at})\n"
        f"- **Discrimination Sanity Gate**: Kendall tau = {sanity['kendall_tau']}, Strong-Weak Gap = {sanity['strong_weak_gap']} pts\n"
        f"- **Limitations**: Audit covers controlled template perturbations across {report.n_resumes} resumes; real applicant pools require ongoing monitoring.\n",
        encoding="utf-8",
    )

    metric_rows = "".join(
        f"<tr><td><code>{html.escape(m.name)}</code></td><td>{html.escape(m.family)}</td>"
        f"<td>{html.escape(m.attribute)}</td><td>{m.value:.3f}</td>"
        f"<td>{f'[{m.ci_low:.2f}, {m.ci_high:.2f}]' if (m.ci_low or m.ci_high) else '—'}</td>"
        f"<td>{f'{m.p_value_adjusted:.4f}' if m.p_value_adjusted is not None else '—'}</td>"
        f"<td><span class='tag {'pass' if m.passed else 'fail'}'>{'PASS' if m.passed else 'FAIL'}</span></td></tr>"
        for m in report.metrics
    )
    leak_rows = "".join(
        f"<tr><td><code>{x['redaction_level']}</code></td><td><code>{x['cue_family']}</code></td>"
        f"<td>{x['baseline_swing']:.2f}</td><td>{x['redacted_swing']:.2f}</td>"
        f"<td><span class='tag {'fail' if x['leakage_Persistent'] else 'pass'}'>{'LEAKS' if x['leakage_Persistent'] else 'CLEAN'}</span></td></tr>"
        for x in report.leakage_summary
    )
    mit_rows = "".join(
        f"<tr><td><code>{x['arm']}</code></td><td>{x['mean_protected_swing']:.2f}</td>"
        f"<td>{x['rank_kendall_tau']:.3f}</td><td>{x['dispersion_ratio']:.3f}</td><td>{x['coverage']:.2f}</td>"
        f"<td><span class='tag {'pass' if x['utility_passed'] else 'fail'}'>{'OK' if x['utility_passed'] else 'DEGRADED'}</span></td></tr>"
        for x in report.mitigation_summary
    )
    conflict_rows = "".join(
        f"<li><strong>{html.escape(c['rule'])}</strong>: {html.escape(c['explanation'])} "
        f"<em>Recommendation: {html.escape(c['recommendation'])}</em></li>"
        for c in report.conflicts
    )
    conflicts_section = f"<h2>Metric Disagreements & Conflict Analysis</h2><ul>{conflict_rows}</ul>" if conflict_rows else ""

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Fairness Audit Report — {html.escape(report.run_id)}</title>
<style>
body {{ font: 14px/1.6 system-ui, sans-serif; color: #1a1c1f; max-width: 1040px; margin: 0 auto; padding: 32px 20px; }}
h1, h2 {{ border-bottom: 1px solid #e3e6ea; padding-bottom: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin: 14px 0 24px; font-size: 13px; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e3e6ea; }}
th {{ background: #f7f8fa; }}
.tag {{ padding: 2px 8px; border-radius: 99px; font-size: 11px; font-weight: 600; }}
.tag.pass {{ background: #e6f4ec; color: #1a7f4b; }}
.tag.fail {{ background: #fdeceb; color: #b3261e; }}
</style></head><body>
<h1>Counterfactual Fairness Audit Report — <code>{html.escape(report.run_id)}</code></h1>
<p><strong>Provider:</strong> <code>{html.escape(report.provider)}</code> | <strong>Model:</strong> <code>{html.escape(report.model)}</code> |
<strong>Sanity Gate:</strong> {'PASS' if sanity['passed'] else 'FAIL'} (&tau; = {sanity['kendall_tau']}, gap = {sanity['strong_weak_gap']} pts) |
<strong>ICC(1):</strong> {report.reliability.get('icc_1')} | <strong>Noise SD:</strong> {report.reliability.get('noise_floor_sd')}</p>
<h2>Headline Findings</h2><ul>{"".join(f"<li>{html.escape(h)}</li>" for h in report.headline_findings)}</ul>
{conflicts_section}
<h2>Anonymization & Proxy Leakage Table</h2>
<table><thead><tr><th>Redaction Level</th><th>Cue Family</th><th>Baseline Swing</th><th>Redacted Swing</th><th>Status</th></tr></thead><tbody>{leak_rows}</tbody></table>
<h2>Mitigation Arms & Utility Pricing</h2>
<table><thead><tr><th>Arm</th><th>Mean Protected Swing</th><th>Rank &tau;</th><th>Dispersion Ratio</th><th>Coverage</th><th>Utility Guard</th></tr></thead><tbody>{mit_rows}</tbody></table>
<h2>All Fairness & Validity Metrics (BH-FDR Corrected)</h2>
<table><thead><tr><th>Metric</th><th>Family</th><th>Attribute</th><th>Value</th><th>95% CI</th><th>q-value</th><th>Verdict</th></tr></thead><tbody>{metric_rows}</tbody></table>
</body></html>"""
    html_path = run_dir / "report.html"
    html_path.write_text(html_doc, encoding="utf-8")
    return html_path


def run_pipeline(cfg: Config) -> tuple[AuditReport, Path]:
    if cfg.provider == "mock":
        cfg.model = "mock-rubric-v1"
    _, jobs, pairs, variants = load_benchmark_corpus(cfg)
    records = score_batch(jobs, pairs, variants, cfg, arm="baseline")
    sanity = sanity_check(records)
    run_id = f"{cfg.run_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    report = run_audit(records, cfg, run_id=run_id)
    mits, leak = evaluate_mitigations(jobs, pairs, variants, records, cfg)
    report.mitigation_summary = mits
    report.leakage_summary = leak
    run_dir = cfg.resolve(cfg.output_dir) / run_id
    render_reports(report, records, sanity, run_dir)
    return report, run_dir


def validate_auditor() -> dict[str, Any]:
    """Run the 5-check Null vs Injected bias validation oracle on the fairness suite."""
    cfg_null = Config(
        provider="mock", model="mock-rubric-v1", max_resumes=10, max_jobs=6, max_variants_per_resume=24,
        self_consistency_samples=2, mock_bias={}, mock_noise_sd=1.5,
    )
    _, jobs, pairs, variants = load_benchmark_corpus(cfg_null)
    pairs_sub = pairs[:8]
    recs_null = score_batch(jobs, pairs_sub, variants, cfg_null)
    rep_null = run_audit(recs_null, cfg_null, run_id="oracle_null")

    cfg_inj = Config(
        provider="mock", model="mock-rubric-v1", max_resumes=10, max_jobs=6, max_variants_per_resume=24,
        self_consistency_samples=2, mock_bias={"gender": {"female": -3.0}}, mock_noise_sd=1.5,
    )
    recs_inj = score_batch(jobs, pairs_sub, variants, cfg_inj)
    rep_inj = run_audit(recs_inj, cfg_inj, run_id="oracle_injected")

    null_gp = [m for m in rep_null.metrics if m.family == "group_parity"]
    specificity_ok = all(m.passed for m in null_gp)

    fem_m = next((m for m in rep_inj.metrics if m.name == "group_parity_gap_gender_female_vs_male"), None)
    sensitivity_ok = bool(fem_m is not None and -4.0 <= fem_m.value <= -2.0 and not fem_m.passed)
    fdr_ok = bool(fem_m is not None and fem_m.p_value_adjusted is not None and fem_m.p_value_adjusted < 0.05)

    val_m = next((m for m in rep_null.metrics if m.name == "counterfactual_swing_education_prestige"), None)
    validity_ok = bool(val_m is not None and val_m.passed and val_m.value >= 1.0)

    noise_sd = float(rep_null.reliability.get("noise_floor_sd", 0.0))
    noise_ok = bool(0.1 <= noise_sd <= 4.0)

    checks = {
        "specificity_under_null": specificity_ok,
        "sensitivity_under_injected_bias": sensitivity_ok,
        "fdr_significance_recovery": fdr_ok,
        "validity_probe_response": validity_ok,
        "noise_floor_estimation": noise_ok,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "recovered_female_gap": fem_m.value if fem_m else None,
        "noise_floor_sd": noise_sd,
    }
