"""Streamlit dashboard for exploring a finished audit run.

The stretch goal from the build plan, built as a *reader* of committed artifacts
rather than as a second implementation of the analysis.  It calls no model and
recomputes no metric: everything shown comes from ``runs/<run_id>/``.  That
matters for two reasons — a demo cannot accidentally spend API quota, and the
dashboard cannot disagree with the report, because there is only one set of
numbers.

Run it with::

    frs dashboard
    # or: streamlit run src/frs/app/dashboard.py

Tabs mirror the report's argument: trust the scorer first, then findings, then
cue effects, then candidate-level explanation cards, then mitigation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "streamlit is not installed. Install it with: "
        "pip install 'fair-resume-screening[dashboard]'"
    ) from exc

import pandas as pd


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _project_root() -> Path:
    # src/app/dashboard.py -> src/app -> src -> project root
    return Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def list_runs(runs_dir: str) -> List[Dict[str, Any]]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    out: List[Dict[str, Any]] = []
    for directory in sorted(root.iterdir(), reverse=True):
        manifest = directory / "manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out.append(
            {
                "run_id": directory.name,
                "path": str(directory),
                "status": data.get("status", "incomplete"),
                "provider": data.get("config", {}).get("llm", {}).get("provider", "?"),
                "model": data.get("config", {}).get("llm", {}).get("model", "?"),
                "created_at": data.get("created_at", ""),
                "warnings": len(data.get("warnings", [])),
            }
        )
    return out


@st.cache_data(show_spinner=False)
def load_artifacts(run_path: str) -> Dict[str, Any]:
    root = Path(run_path)

    def read_json(relative: str) -> Optional[Any]:
        path = root / relative
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def read_csv(relative: str) -> Optional[pd.DataFrame]:
        path = root / relative
        return pd.read_csv(path) if path.exists() else None

    scores = {
        path.stem[len("scores_"):]: pd.read_csv(path)
        for path in sorted((root / "scores").glob("scores_*.csv"))
    } if (root / "scores").exists() else {}

    return {
        "root": root,
        "manifest": read_json("manifest.json") or {},
        "summary": read_json("audit/summary.json") or {},
        "audit": read_json("audit/audit_report.json") or {},
        "conflicts": read_json("audit/conflicts.json") or [],
        "metrics": read_csv("audit/metrics.csv"),
        "mitigation": read_json("mitigation/evaluation.json") or {},
        "utility": read_csv("mitigation/utility_costs.csv"),
        "cards": read_json("report/explanation_cards.json") or [],
        "leakage": read_json("report/anonymisation_leakage.json") or [],
        "report_json": read_json("report/report.json") or {},
        "variants": read_csv("variants/variants.csv"),
        "scores": scores,
        "plots": sorted((root / "report").glob("*.png")) if (root / "report").exists() else [],
    }


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #
def _metric(label: str, value: Any, help_text: str = "") -> None:
    if value is None or (isinstance(value, float) and value != value):
        shown = "—"
    elif isinstance(value, float):
        shown = f"{value:.3f}"
    else:
        shown = str(value)
    st.metric(label, shown, help=help_text or None)


def _verdict_badge(passed: Optional[bool]) -> str:
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return "INFO"


# --------------------------------------------------------------------------- #
# tabs
# --------------------------------------------------------------------------- #
def tab_trust(artifacts: Dict[str, Any]) -> None:
    st.subheader("Can the scorer be trusted?")
    st.caption(
        "Fairness numbers from a scorer that does not discriminate on job-relevant "
        "content are meaningless: a constant function is perfectly unbiased and perfectly "
        "useless. This comes first because it decides whether anything else can be read."
    )
    audit = artifacts["audit"]
    reliability = audit.get("reliability", {}) or {}
    sanity = (audit.get("provenance", {}) or {}).get("sanity_check", {}) or {}
    quality = reliability.get("parse_quality", {}) or {}

    columns = st.columns(4)
    with columns[0]:
        _metric("Phase-2 gate", sanity.get("verdict", "not run"),
                "Does the scorer separate hand-graded strong fits from weak ones?")
    with columns[1]:
        _metric("Reliability (ICC)", reliability.get("icc1_resume"),
                "Share of score variance that is between candidates.")
    with columns[2]:
        _metric("Noise floor (SD)", reliability.get("noise_floor_sd"),
                "Score movement across repeats of an identical prompt.")
    with columns[3]:
        _metric("Cue effect / noise", reliability.get("cue_effect_to_noise_ratio"),
                "Above 2.0 means cue effects clear the model's own noise.")

    columns = st.columns(3)
    with columns[0]:
        _metric("Kendall tau vs fit", sanity.get("kendall_tau"))
    with columns[1]:
        _metric("Evidence grounding", quality.get("mean_evidence_support_rate"),
                "Fraction of quoted evidence located in the resume text.")
    with columns[2]:
        _metric("Parse success", quality.get("parse_ok_rate"))

    if reliability.get("interpretation"):
        st.info(reliability["interpretation"])
    consistency = reliability.get("self_consistency", {}) or {}
    if consistency.get("interpretation"):
        st.info(consistency["interpretation"])
    for warning in quality.get("warnings", []) or []:
        st.warning(warning)

    for plot in artifacts["plots"]:
        if plot.stem == "variance_components":
            st.image(str(plot), caption="Where score variance comes from")


def tab_findings(artifacts: Dict[str, Any]) -> None:
    st.subheader("Findings")
    summary = artifacts["summary"]
    columns = st.columns(4)
    with columns[0]:
        _metric("Metrics", summary.get("n_metrics"))
    with columns[1]:
        _metric("Threshold failures", summary.get("n_failed"))
    with columns[2]:
        _metric("Significant (FDR)", summary.get("n_significant_after_fdr"))
    with columns[3]:
        _metric("Conflicts", (summary.get("conflicts") or {}).get("n_conflicts"))

    for warning in summary.get("warnings", []) or []:
        st.warning(warning)

    headline = (artifacts["report_json"] or {}).get("headline_findings") or []
    if headline:
        st.markdown("**Ranked by consequence, not by p-value**")
        st.dataframe(pd.DataFrame(headline), use_container_width=True, hide_index=True)

    metrics = artifacts["metrics"]
    if metrics is not None and not metrics.empty:
        families = sorted(metrics["family"].dropna().unique())
        chosen = st.multiselect("Metric families", families, default=families)
        only_failures = st.checkbox("Threshold failures only", value=False)
        view = metrics[metrics["family"].isin(chosen)]
        if only_failures:
            view = view[view["passed"] == False]  # noqa: E712 - pandas mask
        st.dataframe(view, use_container_width=True, hide_index=True)

    skipped = summary.get("skipped_families") or []
    if skipped:
        st.markdown("**Metric families not computed**")
        st.caption("An uncomputed family is not a passing family.")
        for entry in skipped:
            st.write(f"- **{entry['family']}** — {entry['reason']}")


def tab_cues(artifacts: Dict[str, Any]) -> None:
    st.subheader("What each cue does to the score")
    st.caption(
        "Every value is the mean change relative to the candidate's own baseline, so "
        "resume strength is differenced out. Name cues and non-name cues sit on the same "
        "axis — the comparison this project exists to make."
    )
    effects = ((artifacts["report_json"] or {}).get("extras") or {}).get("cue_effects") or []
    if effects:
        table = pd.DataFrame(effects).sort_values("mean_delta")
        st.dataframe(table, use_container_width=True, hide_index=True)
    for plot in artifacts["plots"]:
        if plot.stem in ("cue_effects", "candidate_swings") or plot.stem.startswith("scores_by_"):
            st.image(str(plot))

    leakage = artifacts["leakage"]
    if leakage:
        st.markdown("### What survives anonymisation")
        st.caption(
            "Each row applies a redaction level and then asks the cue detector how much "
            "demographic signal it can still recover. Anything left in the `identity` row "
            "is a cue that name-level blind screening misses."
        )
        st.dataframe(pd.DataFrame(leakage), use_container_width=True, hide_index=True)
        for plot in artifacts["plots"]:
            if plot.stem == "blinding_leakage":
                st.image(str(plot))


def tab_conflicts(artifacts: Dict[str, Any]) -> None:
    st.subheader("Metric conflicts")
    st.caption(
        "Several fairness criteria are provably incompatible. A report whose metrics all "
        "agree is either measuring one thing repeatedly or has not looked hard enough."
    )
    conflicts = artifacts["conflicts"]
    if not conflicts:
        st.success("No metric conflicts detected in this run.")
        return
    for conflict in conflicts:
        severity = conflict.get("severity", "")
        label = f"{conflict['type'].replace('_', ' ')} — {severity}"
        with st.expander(label, expanded=(severity == "critical")):
            if severity == "critical":
                st.error(conflict["explanation"])
            elif severity == "important":
                st.warning(conflict["explanation"])
            else:
                st.info(conflict["explanation"])
            st.markdown(f"**What to do.** {conflict['recommendation']}")
            if conflict.get("metrics"):
                st.caption("Metrics involved: " + ", ".join(f"`{m}`" for m in conflict["metrics"]))
            if conflict.get("citation"):
                st.caption(conflict["citation"])


def tab_candidates(artifacts: Dict[str, Any]) -> None:
    st.subheader("Explanation cards")
    st.caption(
        "Aggregate metrics describe a population; a recruiter decides about a person. "
        "Cards are ordered most cue-sensitive first — those are the decisions a human "
        "most needs to review."
    )
    cards = artifacts["cards"]
    if not cards:
        st.info("This run did not generate explanation cards.")
        return

    unstable = [c for c in cards if c["cue_sensitivity"]["crosses_decision_boundary"]]
    columns = st.columns(3)
    with columns[0]:
        _metric("Cards", len(cards))
    with columns[1]:
        _metric("Decision-unstable", len(unstable),
                "Outcome changes under identity cues alone.")
    with columns[2]:
        _metric("Worst swing",
                max(c["cue_sensitivity"]["swing"] for c in cards))

    options = [f"{c['resume_id']} → {c['job_id']} (swing {c['cue_sensitivity']['swing']})"
               for c in cards]
    index = st.selectbox("Candidate", range(len(options)), format_func=lambda i: options[i])
    card = cards[index]

    sensitivity = card["cue_sensitivity"]
    if sensitivity["crosses_decision_boundary"]:
        st.error(
            f"Identity cues alone move this candidate across the "
            f"'{sensitivity['crosses_decision_boundary']}' threshold. The recommendation "
            "is not determinate."
        )
    columns = st.columns(3)
    with columns[0]:
        _metric("Score", card["score"])
    with columns[1]:
        _metric("Recommendation", card["recommendation"] or "n/a")
    with columns[2]:
        _metric("Variant band",
                f"{sensitivity['score_low']} – {sensitivity['score_high']}")

    if card["dimensions"]:
        st.dataframe(pd.DataFrame(card["dimensions"]), use_container_width=True, hide_index=True)
    if sensitivity.get("least_favourable_cue"):
        st.write(f"- Lowest score under `{sensitivity['least_favourable_cue']}`")
        st.write(f"- Highest score under `{sensitivity['most_favourable_cue']}`")
    if card["missing_requirements"]:
        st.markdown("**Requirements without supporting evidence**")
        for item in card["missing_requirements"]:
            st.write(f"- {item}")
    if card.get("rationale"):
        st.markdown("**Model rationale**")
        st.write(card["rationale"])
    for flag in card["flags"]:
        st.warning(flag)


def tab_mitigation(artifacts: Dict[str, Any]) -> None:
    st.subheader("Mitigation: what helps, and what it costs")
    mitigation = artifacts["mitigation"]
    if not mitigation or not mitigation.get("comparisons"):
        st.info("No mitigation arms were evaluated in this run.")
        return
    st.caption(mitigation.get("guidance", ""))

    utility = artifacts["utility"]
    if utility is not None and not utility.empty:
        st.dataframe(utility, use_container_width=True, hide_index=True)
    for plot in artifacts["plots"]:
        if plot.stem == "mitigation":
            st.image(str(plot))

    for arm, comparison in mitigation["comparisons"].items():
        with st.expander(f"{arm} — {comparison.get('verdict')}"):
            details = comparison.get("metric_comparisons") or []
            if details:
                st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)
            for reason in comparison.get("verdict_reasons", []):
                st.warning(reason)


def tab_provenance(artifacts: Dict[str, Any]) -> None:
    st.subheader("Provenance")
    manifest = artifacts["manifest"]
    git = manifest.get("git", {}) or {}
    rows = [
        ("run id", manifest.get("run_id")),
        ("created", manifest.get("created_at")),
        ("pipeline version", manifest.get("frs_version")),
        ("config hash", manifest.get("config_hash")),
        ("seed", manifest.get("seed")),
        ("git commit", git.get("commit", "n/a")),
        ("git dirty", git.get("dirty")),
        ("python", (manifest.get("environment", {}) or {}).get("python")),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["item", "value"]),
                 use_container_width=True, hide_index=True)

    stages = manifest.get("stages", {}) or {}
    if stages:
        st.markdown("**Stage timings**")
        st.dataframe(
            pd.DataFrame(
                [{"stage": name, "elapsed_s": payload.get("elapsed_s")}
                 for name, payload in stages.items()]
            ),
            use_container_width=True, hide_index=True,
        )

    scores = artifacts["scores"]
    if scores:
        arm = st.selectbox("Inspect the score table for arm", sorted(scores))
        st.dataframe(scores[arm].head(400), use_container_width=True, hide_index=True)

    report_html = artifacts["root"] / "report" / "report.html"
    if report_html.exists():
        st.caption(f"Full HTML report: `{report_html}`")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="Fair Resume Screening — audit explorer",
        page_icon="⚖️",
        layout="wide",
    )
    st.title("Fair Resume Screening — audit explorer")
    st.caption(
        "Reads committed run artifacts. It calls no model and recomputes no metric, so "
        "it cannot disagree with the report and cannot spend API quota."
    )

    default_runs = str(_project_root() / "runs")
    runs_dir = st.sidebar.text_input("Runs directory", value=default_runs)
    runs = list_runs(runs_dir)
    if not runs:
        st.warning(
            f"No completed runs under `{runs_dir}`.\n\n"
            "Produce one with:\n\n```\nfrs pipeline --provider mock\n```"
        )
        return

    labels = [
        f"{r['run_id']}  ({r['provider']}/{r['model']}, {r['status']})" for r in runs
    ]
    index = st.sidebar.selectbox("Run", range(len(runs)), format_func=lambda i: labels[i])
    selected = runs[index]
    artifacts = load_artifacts(selected["path"])

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Scorer:** `{selected['provider']}/{selected['model']}`")
    st.sidebar.write(f"**Created:** {selected['created_at']}")
    if selected["warnings"]:
        st.sidebar.warning(f"{selected['warnings']} run warning(s)")

    tabs = st.tabs([
        "1. Trust the scorer", "2. Findings", "3. Cue effects",
        "4. Conflicts", "5. Candidates", "6. Mitigation", "7. Provenance",
    ])
    with tabs[0]:
        tab_trust(artifacts)
    with tabs[1]:
        tab_findings(artifacts)
    with tabs[2]:
        tab_cues(artifacts)
    with tabs[3]:
        tab_conflicts(artifacts)
    with tabs[4]:
        tab_candidates(artifacts)
    with tabs[5]:
        tab_mitigation(artifacts)
    with tabs[6]:
        tab_provenance(artifacts)


if __name__ == "__main__":
    main()
