"""Model card / audit datasheet.

Following Mitchell et al. (2019) "Model Cards for Model Reporting" and Gebru et
al. (2021) "Datasheets for Datasets", adapted for an audit rather than a model
release.  The card is generated from the run's own artifacts, never hand-written,
so it cannot drift away from what the pipeline actually did — a hand-maintained
card is a claim, a generated one is a record.

Its most important sections are the ones that constrain use:

* **Intended use and out-of-scope use.** The system supports a human screener; it
  does not make hiring decisions. That boundary is in the project scope and needs
  to survive contact with anyone who later picks the tool up.
* **What "fair" means here.** The audit measures whether *changing a cue changes
  the score*. It does not, and cannot, establish what the correct ranking is —
  there is no ground-truth fair outcome without real hiring results. Being precise
  about which kind of fairness was measured is the difference between a defensible
  claim and an overclaim.
* **Limitations.** Sample size, name-list provenance, synthetic corpus, declared
  confounds, and every family the run had to skip.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas import AuditReport


def build_model_card(
    report: AuditReport,
    *,
    manifest: Optional[Dict[str, Any]] = None,
    mitigation: Optional[Dict[str, Any]] = None,
    corpus_stats: Optional[Dict[str, Any]] = None,
    sanity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the card as structured data (rendered separately)."""
    manifest = manifest or {}
    provenance = report.provenance or {}
    config = provenance.get("config", {})
    stages = manifest.get("stages", {})
    perturbation = stages.get("generate_variants", {})

    card: Dict[str, Any] = {
        "title": "Fair Resume Screening — Audit Model Card",
        "run_id": report.run_id,
        "created_at": report.created_at,
        "frs_version": manifest.get("frs_version", "unknown"),
        "model_details": {
            "provider": provenance.get("provider", ""),
            "model": provenance.get("model", ""),
            "role": "resume-to-job-description fit scoring with a weighted rubric",
            "decoding": {
                "temperature": manifest.get("config", {}).get("llm", {}).get("temperature"),
                "self_consistency_samples": manifest.get("config", {})
                .get("llm", {})
                .get("self_consistency_samples"),
            },
            "prompt_version": stages.get("score_variants", {})
            .get("telemetry", {})
            .get("prompt_version", ""),
            "rubric": manifest.get("config", {}).get("screening", {}).get("rubric"),
        },
        "intended_use": {
            "primary": (
                "Decision support for a human recruiter performing a first-pass screen: "
                "produce a rubric-decomposed, evidence-grounded fit score and a shortlist, "
                "together with a fairness audit of those scores."
            ),
            "intended_users": [
                "recruiters and hiring managers, with a human always making the decision",
                "researchers auditing LLM screening behaviour",
            ],
            "out_of_scope": [
                "making or automating a hiring or rejection decision",
                "ranking candidates without human review of the shortlist",
                "any use where the score is shown to a candidate as a definitive assessment",
                "interview scheduling, background verification or onboarding",
                "deployment in a live HR system without an independent legal and "
                "compliance review",
            ],
        },
        "fairness_definition": {
            "what_is_measured": (
                "Counterfactual sensitivity: whether changing an identity cue in a resume, "
                "holding all other content byte-identical, changes the score, the "
                "recommendation, or the candidate's shortlist position."
            ),
            "what_is_not_measured": (
                "Whether the resulting ranking is the *correct* ranking. There is no "
                "ground-truth fair outcome available without real hiring results, so the "
                "audit measures decision *stability under identity change*, not decision "
                "quality. A perfectly stable scorer could still be a bad screener, which is "
                "why the validity probe and the Phase-2 discrimination check are reported "
                "next to the fairness metrics."
            ),
            "metric_families": sorted({m.family for m in report.metrics}),
            "attributes_audited": provenance.get("attributes_audited", []),
            "reference_levels": config.get("reference_levels", {}),
            "thresholds": {
                "min_practical_effect_score_points": config.get("min_practical_effect"),
                "counterfactual_tolerance_score_points": config.get("counterfactual_tolerance"),
                "impact_ratio_threshold": config.get("impact_ratio_threshold"),
                "alpha": config.get("alpha"),
                "multiplicity_correction": provenance.get("multiplicity", {}).get("method"),
            },
        },
        "evaluation_data": {
            "corpus": corpus_stats or {},
            "n_base_resumes": report.n_base_resumes,
            "n_jobs": report.n_jobs,
            "n_variants": report.n_variants,
            "n_score_records": report.n_records,
            "perturbation_design": perturbation.get("design"),
            "name_list": perturbation.get("name_list"),
            "cue_families": perturbation.get("registry", []),
            "variant_validation": perturbation.get("validation", {}),
        },
        "scorer_validity": {
            "phase2_sanity_check": sanity or provenance.get("sanity_check", {}),
            "reliability": {
                "icc1_resume": report.reliability.get("icc1_resume"),
                "noise_floor_sd": report.reliability.get("noise_floor_sd"),
                "cue_effect_to_noise_ratio": report.reliability.get("cue_effect_to_noise_ratio"),
                "self_consistency": report.reliability.get("self_consistency", {}),
            },
            "evidence_grounding": report.reliability.get("parse_quality", {}).get(
                "mean_evidence_support_rate"
            ),
            "note": (
                "A scorer that does not discriminate on job-relevant content produces "
                "fairness metrics that look excellent and mean nothing. These figures "
                "establish whether the fairness results are interpretable at all."
            ),
        },
        "quantitative_analysis": {
            "n_metrics": len(report.metrics),
            "n_failed_thresholds": sum(1 for m in report.metrics if m.passed is False),
            "n_significant_after_fdr": sum(1 for m in report.metrics if m.significant),
            "failed_metrics": [
                {
                    "metric": m.name, "value": round(m.value, 4) if m.value == m.value else None,
                    "threshold": m.threshold, "interpretation": m.interpretation,
                }
                for m in report.metrics if m.passed is False
            ][:20],
        },
        "metric_conflicts": report.conflicts,
        "mitigation": _mitigation_section(mitigation),
        "limitations": _limitations(report, manifest, corpus_stats),
        "ethical_considerations": [
            "Perturbation-based auditing requires assuming that a chosen name or affiliation "
            "is read as signalling a particular group. That assumption is the study's main "
            "validity risk, which is why name-list provenance is machine-recorded and why a "
            "non-peer-reviewed list raises a warning that travels into this card.",
            "The cue lexicons necessarily encode the team's model of which signals map to "
            "which groups. They are deliberately short and human-readable so a reader can "
            "disagree with a specific entry rather than with an opaque classifier.",
            "The corpus is synthetic. That protects real candidates' personal data and keeps "
            "the design balanced, but synthetic resumes may be more uniform than real ones, "
            "which can make cue effects easier to detect than they would be in the wild.",
            "Reporting a bias estimate carries a duty not to overstate it. Effects below the "
            "measured noise floor, and null results at low power, are labelled as such "
            "rather than presented as findings.",
            "Bias mitigation is reported with its utility cost, because a mitigation that "
            "removes bias by flattening scores harms the candidates it claims to protect by "
            "making the screen arbitrary.",
        ],
        "caveats_and_recommendations": [
            "Re-run the audit whenever the model version, the prompt or the rubric changes: "
            "cue effects are properties of a specific model/prompt pair, not durable facts.",
            "Use the explanation cards alongside the aggregate metrics. A candidate whose "
            "decision is unstable under cue swaps needs human review regardless of what the "
            "group-level metrics say.",
            "Treat the four-fifths impact ratio as a reporting convention, not a legal "
            "safe harbour.",
        ],
        "citation_of_methods": _method_citations(),
    }
    return card


def _mitigation_section(mitigation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not mitigation:
        return {"evaluated": False, "note": "no mitigation arms were run"}
    comparisons = mitigation.get("comparisons", {})
    return {
        "evaluated": True,
        "arms": sorted(comparisons),
        "ranking": mitigation.get("ranking", []),
        "verdicts": {arm: c.get("verdict") for arm, c in comparisons.items()},
        "utility_guard": mitigation.get("guidance", ""),
        "per_arm": {
            arm: {
                "verdict": c.get("verdict"),
                "reasons": c.get("verdict_reasons", []),
                "metrics_improved": c.get("n_metrics_improved"),
                "metrics_worsened": c.get("n_metrics_worsened"),
                "rank_agreement": c.get("utility", {}).get("rank_agreement_kendall_tau"),
                "score_dispersion_ratio": c.get("utility", {}).get("score_dispersion_ratio"),
                "coverage": c.get("utility", {}).get("coverage"),
            }
            for arm, c in comparisons.items()
        },
    }


def _limitations(
    report: AuditReport,
    manifest: Dict[str, Any],
    corpus_stats: Optional[Dict[str, Any]],
) -> List[str]:
    out: List[str] = []
    stats = corpus_stats or {}
    n_resumes = stats.get("n_resumes", report.n_base_resumes)

    if n_resumes and n_resumes < 30:
        out.append(
            f"Only {n_resumes} base resumes. The blocked design recovers effects far better "
            "than a pooled comparison would at this size, but small gaps remain undetectable: "
            "each group-parity metric reports its own minimum detectable effect, and nulls "
            "must be read as underpowered rather than as evidence of fairness."
        )

    perturbation = manifest.get("stages", {}).get("generate_variants", {})
    name_list = perturbation.get("name_list")
    if name_list and name_list != "bertrand_mullainathan_2004":
        out.append(
            f"The name list in use ({name_list}) is not a verbatim reproduction of a "
            "peer-reviewed appendix, so its ethnicity labels are not perception-validated. "
            "Any claim resting on those labels needs this caveat attached."
        )

    for family in perturbation.get("registry", []):
        if family.get("confounded_with"):
            out.append(
                f"Cue family '{family['family']}' is confounded with "
                f"{family['confounded_with']}: {family.get('caveat', '')}"
            )
        elif family.get("cue_type") == "validity_proxy":
            out.append(
                f"Cue family '{family['family']}' is a style proxy rather than a protected "
                f"attribute: {family.get('caveat', '')}"
            )
        elif family.get("asymmetric"):
            out.append(
                f"Cue family '{family['family']}' uses marked-versus-unmarked contrasts, so "
                "its estimates are not symmetric group differences."
            )

    if not report.reliability.get("has_repeats"):
        out.append(
            "No repeated decodes were run, so the model's own scoring noise is unmeasured "
            "and no effect can be compared against a noise floor."
        )

    for skipped in report.provenance.get("skipped_families", []):
        out.append(f"Metric family '{skipped['family']}' was not computed: {skipped['reason']}")

    git = manifest.get("git", {})
    if git.get("available") and git.get("dirty"):
        out.append(
            "The working tree had uncommitted changes when this run executed, so the exact "
            "code that produced these numbers is not recoverable from the recorded commit."
        )

    out.extend(report.warnings)
    return out


def _method_citations() -> List[str]:
    return [
        "Bertrand & Mullainathan (2004). Are Emily and Greg More Employable than Lakisha and "
        "Jamal? American Economic Review 94(4). — name-perturbation design and name list.",
        "Wilson & Caliskan (2024). Gender, race, and intersectional bias in resume screening "
        "via language model retrieval. — LLM resume-audit methodology.",
        "Seshadri, Chen, Singh & Goldfarb-Tarrant (2025). Small changes, large consequences. "
        "— allocational rather than distributional framing.",
        "Castleman, Shen, Metevier, Springer & Korolova (2026). Measuring validity in "
        "LLM-based resume screening. — validity probes and grounding checks.",
        "Cui (2026). Bias-audited resume screening with calibrated matching scores, selective "
        "review, and evidence-grounded LLM explanation cards. — explanation cards, "
        "selective review, calibration.",
        "Rao, Venkatesan, Cherubini & Jayagopi (2025). Invisible filters: Cultural bias in "
        "hiring evaluations using LLMs. — cultural cue families beyond names.",
        "Chouldechova (2017); Kleinberg, Mullainathan & Raghavan (2017). — the impossibility "
        "results behind the metric-conflict analysis.",
        "Hardt, Price & Srebro (2016). Equality of opportunity in supervised learning. "
        "— equal-opportunity criterion.",
        "Monroe, Colaresi & Quinn (2008). Fightin' words. — log-odds with informative "
        "Dirichlet prior, used for the rationale language audit.",
        "Mitchell et al. (2019). Model cards for model reporting; Gebru et al. (2021). "
        "Datasheets for datasets. — the structure of this card.",
        "EEOC Uniform Guidelines on Employee Selection Procedures, 29 CFR 1607 (1978). "
        "— the four-fifths reporting convention.",
    ]


def render_model_card_markdown(card: Dict[str, Any]) -> str:
    """Render the card as Markdown."""
    lines: List[str] = [
        f"# {card['title']}",
        "",
        f"**Run:** `{card['run_id']}`  ",
        f"**Generated:** {card['created_at']}  ",
        f"**Pipeline version:** {card['frs_version']}",
        "",
        "> Generated from this run's own artifacts. It is a record of what the pipeline did, "
        "not a hand-written description of what it was meant to do.",
        "",
        "## 1. Model details",
        "",
    ]
    details = card["model_details"]
    lines += [
        f"- **Provider / model:** `{details['provider']}` / `{details['model']}`",
        f"- **Role:** {details['role']}",
        f"- **Rubric:** `{details['rubric']}`",
        f"- **Prompt version:** `{details['prompt_version']}`",
        f"- **Decoding:** temperature {details['decoding'].get('temperature')}, "
        f"{details['decoding'].get('self_consistency_samples')} sample(s) per prompt",
        "",
        "## 2. Intended use",
        "",
        card["intended_use"]["primary"],
        "",
        "**Intended users**",
        "",
    ]
    lines += [f"- {item}" for item in card["intended_use"]["intended_users"]]
    lines += ["", "**Out of scope**", ""]
    lines += [f"- {item}" for item in card["intended_use"]["out_of_scope"]]

    fairness = card["fairness_definition"]
    lines += [
        "",
        "## 3. What 'fair' means in this audit",
        "",
        f"**Measured.** {fairness['what_is_measured']}",
        "",
        f"**Not measured.** {fairness['what_is_not_measured']}",
        "",
        f"- **Metric families:** {', '.join(fairness['metric_families']) or 'none'}",
        f"- **Attributes audited:** {', '.join(fairness['attributes_audited']) or 'none'}",
        f"- **Practical-effect threshold:** "
        f"{fairness['thresholds'].get('min_practical_effect_score_points')} score points",
        f"- **Multiplicity correction:** "
        f"{fairness['thresholds'].get('multiplicity_correction')} at alpha "
        f"{fairness['thresholds'].get('alpha')}",
        "",
        "## 4. Evaluation data",
        "",
    ]
    data = card["evaluation_data"]
    lines += [
        f"- {data['n_base_resumes']} base resumes, {data['n_jobs']} job descriptions, "
        f"{data['n_variants']} variants, {data['n_score_records']} score records",
        f"- **Perturbation design:** `{data['perturbation_design']}`",
        f"- **Name list:** `{data['name_list']}`",
    ]
    validation = data.get("variant_validation") or {}
    if validation:
        lines.append(
            f"- **Variant validation:** {validation.get('n_passed')}/"
            f"{validation.get('n_variants')} passed content-preservation checks"
        )
    if data.get("cue_families"):
        lines += ["", "| cue family | type | levels | attributes |", "| --- | --- | --- | --- |"]
        for family in data["cue_families"]:
            lines.append(
                f"| {family.get('family')} | {family.get('cue_type')} | "
                f"{family.get('n_levels')} | {family.get('attributes')} |"
            )

    validity = card["scorer_validity"]
    sanity = validity.get("phase2_sanity_check") or {}
    lines += [
        "",
        "## 5. Scorer validity",
        "",
        f"- **Phase-2 discrimination check:** {sanity.get('verdict', 'not run')} "
        f"(Kendall tau vs expected fit = {sanity.get('kendall_tau')}, "
        f"strong-minus-weak gap = {sanity.get('strong_minus_weak_gap')})",
        f"- **Reliability (ICC):** {validity['reliability'].get('icc1_resume')}",
        f"- **Decoding noise floor (SD):** {validity['reliability'].get('noise_floor_sd')}",
        f"- **Cue effect / noise ratio:** "
        f"{validity['reliability'].get('cue_effect_to_noise_ratio')}",
        f"- **Mean evidence grounding:** {validity.get('evidence_grounding')}",
        "",
        validity["note"],
        "",
        "## 6. Quantitative results",
        "",
    ]
    analysis = card["quantitative_analysis"]
    lines += [
        f"- {analysis['n_metrics']} metrics computed",
        f"- {analysis['n_failed_thresholds']} failed their configured threshold",
        f"- {analysis['n_significant_after_fdr']} statistically significant after "
        "Benjamini-Hochberg correction",
        "",
    ]
    if analysis["failed_metrics"]:
        lines += ["**Threshold failures**", ""]
        for metric in analysis["failed_metrics"]:
            lines.append(f"- `{metric['metric']}` = {metric['value']} — {metric['interpretation']}")
        lines.append("")

    if card["metric_conflicts"]:
        lines += ["## 7. Metric conflicts", ""]
        for conflict in card["metric_conflicts"]:
            lines += [
                f"### {conflict['type']} ({conflict['severity']})",
                "",
                conflict["explanation"],
                "",
                f"**Recommendation.** {conflict['recommendation']}",
                "",
            ]
            if conflict.get("citation"):
                lines += [f"<sub>{conflict['citation']}</sub>", ""]

    mitigation = card["mitigation"]
    lines += ["## 8. Mitigation", ""]
    if not mitigation.get("evaluated"):
        lines += [mitigation.get("note", "not evaluated"), ""]
    else:
        lines += ["| arm | verdict | metrics improved | rank agreement | dispersion ratio |",
                  "| --- | --- | --- | --- | --- |"]
        for arm, info in mitigation["per_arm"].items():
            lines.append(
                f"| {arm} | {info['verdict']} | {info['metrics_improved']} | "
                f"{info['rank_agreement']} | {info['score_dispersion_ratio']} |"
            )
        lines += ["", mitigation.get("utility_guard", ""), ""]

    lines += ["## 9. Limitations", ""]
    lines += [f"- {item}" for item in card["limitations"]] or ["- none recorded"]
    lines += ["", "## 10. Ethical considerations", ""]
    lines += [f"- {item}" for item in card["ethical_considerations"]]
    lines += ["", "## 11. Caveats and recommendations", ""]
    lines += [f"- {item}" for item in card["caveats_and_recommendations"]]
    lines += ["", "## 12. Methods cited", ""]
    lines += [f"- {item}" for item in card["citation_of_methods"]]
    lines.append("")
    return "\n".join(lines)
