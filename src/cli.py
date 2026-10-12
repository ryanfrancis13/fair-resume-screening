"""Command-line interface.

One entry point per phase of the build plan, plus the composite ``pipeline``
command.  Two design choices are worth naming:

* **Stages are separately runnable and re-runnable.** Re-auditing or re-reporting
  a finished run reads that run's artifacts and calls no model, so improving a
  metric or a figure costs nothing and cannot silently change the underlying
  scores.
* **Cost is visible before it is incurred.** ``--dry-run`` prints the planned call
  count and the rate-limited wall-clock for any scoring command, which is what
  keeps a factorial design from consuming a day's free-tier quota by surprise.

Usage::

    frs doctor                        # environment and provider readiness
    frs data prepare                  # phase 1
    frs score                         # phase 2
    frs perturb                       # phase 3 (generate + score variants)
    frs audit                         # phase 4
    frs mitigate                      # phase 4/6
    frs report                        # phase 5
    frs pipeline                      # everything, in order
    frs validate-auditor              # is the audit suite itself trustworthy?
    frs rank --job J01                # shortlist for one role
    frs list cues|names|strategies|runs|rubrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import Config, load_dotenv_if_present
from .logging_setup import get_logger, setup_logging

logger = get_logger("frs.cli")

_BANNER = "Fair Resume Screening — LLM screening with an integrated fairness audit"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_config(args: argparse.Namespace) -> Config:
    path = Path(args.config) if getattr(args, "config", None) else None
    overrides = list(getattr(args, "set", None) or [])
    for flag, key in (
        ("provider", "llm.provider"),
        ("model", "llm.model"),
        ("corpus", "data.corpus"),
        ("seed", "seed"),
        ("log_level", "log_level"),
        ("run_name", "run_name"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            overrides.append(f"{key}={value}")
    cfg = Config.load(path, overrides)
    setup_logging(cfg.log_level)
    return cfg


def _echo(payload: Any, *, as_json: bool = False) -> None:
    if as_json:
        from .utils.io import _to_jsonable

        print(json.dumps(_to_jsonable(payload), indent=2))
    else:
        print(payload)


def _print_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    if not rows:
        print("(nothing to show)")
        return
    widths = {
        column: max(len(str(column)), *(len(str(row.get(column, ""))[:70]) for row in rows))
        for column in columns
    }
    header = "  ".join(str(c).ljust(widths[c]) for c in columns)
    print(header)
    print("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        print("  ".join(str(row.get(c, ""))[:70].ljust(widths[c]) for c in columns))


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_doctor(args: argparse.Namespace) -> int:
    """Check environment, dependencies, corpus and provider readiness."""
    cfg = _load_config(args)
    load_dotenv_if_present()

    from .pipeline.artifacts import environment_state, git_state

    print(_BANNER)
    print()
    environment = environment_state()
    print(f"python           : {environment['python']}")
    print(f"platform         : {environment['platform']}")
    print()
    print("packages:")
    for name, version in environment["packages"].items():
        status = "ok " if version != "not installed" else "MISSING"
        print(f"  [{status}] {name:<16} {version}")
    print()

    git = git_state()
    print(f"git              : {'clean' if git.get('available') and not git.get('dirty') else git.get('note', 'n/a')}")
    print(f"project root     : {cfg.root}")
    print(f"provider / model : {cfg.llm.provider} / {cfg.llm.model}")
    print(f"cache            : {cfg.resolve(cfg.llm.cache_dir)} (enabled={cfg.llm.cache_enabled})")
    print()

    problems: List[str] = []

    print("corpus:")
    try:
        from .data import corpus_stats, load_corpus

        resumes, jobs, pairs = load_corpus(cfg)
        stats = corpus_stats(resumes, jobs, pairs)
        print(f"  [ok ] {stats['n_resumes']} resumes, {stats['n_jobs']} jobs, "
              f"{stats['n_pairs']} pairs ({stats['n_templated_resumes']} templated)")
        print(f"        expected-fit distribution: {stats['expected_fit_distribution']}")
    except Exception as exc:  # noqa: BLE001 - diagnostics
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
        problems.append("corpus could not be loaded")
    print()

    print("cue registry:")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from .perturb.cues import build_registry, registry_summary

            registry = build_registry(
                families=cfg.perturbation.cue_families,
                name_list_id=cfg.perturbation.name_list,
            )
            for row in registry_summary(registry):
                print(f"  [ok ] {row['family']:<22} {row['cue_type']:<22} "
                      f"{row['n_levels']:>3} levels")
            for warning in caught:
                print(f"  [warn] {warning.message}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
        problems.append("cue registry could not be built")
    print()

    print("provider:")
    try:
        from .llm.client import build_provider

        provider = build_provider(cfg)
        health = provider.health_check()
        status = "ok " if health.get("ready") else "FAIL"
        print(f"  [{status}] {health}")
        if not health.get("ready"):
            problems.append("provider is not ready")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
        problems.append("provider could not be constructed")
    print()

    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        print()
        print("The pipeline still runs end-to-end with `--provider mock`, which needs no "
              "API key.")
        return 1
    print("All checks passed.")
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .data import corpus_stats, duplicate_report, load_corpus, pairing_report

    resumes, jobs, pairs = load_corpus(cfg)
    if args.data_action == "stats":
        _echo(
            {
                "corpus": corpus_stats(resumes, jobs, pairs),
                "pairing": pairing_report(pairs, resumes, jobs),
                "duplicates": duplicate_report(resumes),
            },
            as_json=True,
        )
        return 0

    from .pipeline.run import prepare_data, start_run

    state = start_run(cfg, prefix=f"{cfg.run_name}-data")
    prepare_data(state)
    print(f"corpus written to {state.root / 'corpus'}")
    _echo(state.manifest.data["stages"]["prepare_data"]["corpus_stats"], as_json=True)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Phase 2: score unperturbed resumes and run the discrimination gate."""
    cfg = _load_config(args)
    from .pipeline.run import generate_variants, prepare_data, score_baseline, start_run

    state = start_run(cfg, prefix=f"{cfg.run_name}-score")
    prepare_data(state)
    generate_variants(state)
    score_baseline(state, dry_run=args.dry_run)
    if args.dry_run:
        _echo(state.manifest.data["stages"]["score_baseline"], as_json=True)
        return 0

    frame = state.frames_by_arm["baseline_unperturbed"]
    print()
    print(f"scored {len(frame)} (resume, job) pairs -> {state.root / 'scores'}")
    print()
    print("Phase-2 sanity check:")
    _echo(state.sanity, as_json=True)
    print()
    rows = (
        frame[["job_id", "base_resume_id", "overall_score", "recommendation",
               "evidence_support_rate"]]
        .sort_values(["job_id", "overall_score"], ascending=[True, False])
        .round(2)
        .to_dict(orient="records")
    )
    _print_table(
        rows,
        ["job_id", "base_resume_id", "overall_score", "recommendation",
         "evidence_support_rate"],
    )
    return 0 if state.sanity.get("verdict") != "review" else 2


def cmd_perturb(args: argparse.Namespace) -> int:
    """Phase 3: generate counterfactual variants and score them."""
    cfg = _load_config(args)
    if args.cues:
        cfg.perturbation.cue_families = [c.strip() for c in args.cues.split(",") if c.strip()]
    if args.name_list:
        cfg.perturbation.name_list = args.name_list
    if args.design:
        cfg.perturbation.design = args.design

    from .perturb.validator import summarise_validation
    from .pipeline.run import generate_variants, prepare_data, score_variants, start_run

    state = start_run(cfg, prefix=f"{cfg.run_name}-perturb")
    prepare_data(state)
    generate_variants(state)

    validation = summarise_validation(state.variants)
    print(f"generated {len(state.variants)} variants "
          f"({validation['n_passed']} passed validation, {validation['n_failed']} failed)")
    if validation["n_failed"]:
        print("top failure modes:")
        for mode, count in validation["top_failure_modes"]:
            print(f"  {count:>4}  {mode}")
    if state.variant_skips:
        print(f"{len(state.variant_skips)} family/resume combinations skipped "
              "(missing slots) — see variants/skipped.json")

    if args.generate_only:
        return 0

    score_variants(state, dry_run=args.dry_run)
    if args.dry_run:
        _echo(state.manifest.data["stages"]["score_variants"], as_json=True)
        return 0
    print(f"scored {len(state.frames_by_arm['baseline'])} variant records "
          f"-> {state.root / 'scores'}")
    print(f"\nrun id: {state.run_id}\nnext: frs audit --run-id {state.run_id}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Phase 4: run the fairness metric suite over a finished run."""
    cfg = _load_config(args)
    from .fairness.suite import audit_summary, headline_findings, run_audit
    from .pipeline.artifacts import resolve_run_id, write_audit
    from .pipeline.run import load_run

    run_id = resolve_run_id(cfg, args.run_id)
    state = load_run(cfg, run_id)
    frame = state.frames_by_arm.get(args.arm)
    if frame is None or frame.empty:
        print(f"no scores for arm {args.arm!r} in run {run_id}. "
              f"available: {sorted(state.frames_by_arm)}", file=sys.stderr)
        return 1

    report = run_audit(
        frame, cfg, run_id=run_id,
        provenance={"sanity_check": state.sanity, "reaudit": True},
    )
    from .fairness.suite import metrics_table

    summary = audit_summary(report)
    write_audit(state.root, report, metrics_table(report), summary)
    state.audit_report = report

    print(f"audit written to {state.root / 'audit'}")
    print()
    _print_table(headline_findings(report), ["metric", "value", "unit", "verdict", "q_value_bh"])
    print()
    print(f"metrics: {summary['n_metrics']}   failed thresholds: {summary['n_failed']}   "
          f"significant after FDR: {summary['n_significant_after_fdr']}")
    print(f"conflicts: {summary['conflicts']['n_conflicts']} "
          f"{summary['conflicts']['by_severity']}")
    if report.warnings:
        print("\nwarnings:")
        for warning in report.warnings:
            print(f"  - {warning}")
    if args.json:
        _echo(summary, as_json=True)
    return 0


def cmd_mitigate(args: argparse.Namespace) -> int:
    """Phase 4/6: run mitigation arms and compare them to baseline."""
    cfg = _load_config(args)
    if args.list:
        from .mitigate import describe_strategies

        _print_table(
            describe_strategies(),
            ["strategy", "kind", "mechanism", "cost", "known_limitation"],
        )
        return 0
    if args.strategies:
        cfg.mitigation.strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    from .pipeline.artifacts import resolve_run_id
    from .pipeline.run import audit as audit_stage
    from .pipeline.run import generate_variants, mitigate, prepare_data, start_run

    if args.run_id:
        # Re-running mitigation over an existing run needs the variants back,
        # because the pre-inference arms score them again.
        run_id = resolve_run_id(cfg, args.run_id)
        print(f"re-using corpus from run {run_id}; regenerating variants for the new arms")
    state = start_run(cfg, prefix=f"{cfg.run_name}-mitigate")
    prepare_data(state)
    generate_variants(state)
    from .pipeline.run import score_variants

    score_variants(state, dry_run=args.dry_run)
    if args.dry_run:
        mitigate(state, dry_run=True)
        _echo(state.manifest.data["stages"].get("mitigate", {}), as_json=True)
        return 0
    audit_stage(state)
    mitigate(state)

    from .mitigate import utility_table

    costs = utility_table(state.mitigation) if state.mitigation else None
    print(f"\nmitigation evaluated -> {state.root / 'mitigation'}")
    if costs is not None and not costs.empty:
        print()
        _print_table(
            costs.round(4).to_dict(orient="records"),
            ["arm", "verdict", "metrics_improved", "metrics_worsened",
             "rank_agreement", "score_dispersion_ratio", "coverage"],
        )
    if state.mitigation:
        print()
        print("ranking (best first):", ", ".join(state.mitigation.get("ranking", [])))
        print()
        print(state.mitigation.get("guidance", ""))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Phase 5: render the report for a finished run."""
    cfg = _load_config(args)
    from .fairness.suite import run_audit
    from .pipeline.artifacts import resolve_run_id
    from .pipeline.run import load_run
    from .report import build_report

    run_id = resolve_run_id(cfg, args.run_id)
    state = load_run(cfg, run_id)
    frame = state.frames_by_arm.get("baseline")
    if frame is None or frame.empty:
        print(f"run {run_id} has no baseline scores to report on", file=sys.stderr)
        return 1
    state.audit_report = run_audit(frame, cfg, run_id=run_id,
                                   provenance={"sanity_check": state.sanity})
    mitigation_path = state.root / "mitigation" / "evaluation.json"
    if mitigation_path.exists():
        from .utils.io import read_json

        state.mitigation = read_json(mitigation_path)

    outputs = build_report(state)
    print(f"report written to {state.root / 'report'}")
    for name, path in outputs.items():
        print(f"  {name:<20} {path}")
    if "html" in outputs:
        print(f"\nopen it with:  open {outputs['html']}")
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    """Show the shortlist a recruiter would see for one role."""
    cfg = _load_config(args)
    from .pipeline.artifacts import resolve_run_id
    from .pipeline.run import load_run
    from .screening.rank import rank_candidates, shortlist_table

    run_id = resolve_run_id(cfg, args.run_id)
    state = load_run(cfg, run_id)
    frame = state.frames_by_arm.get(args.arm) or state.frames_by_arm.get("baseline")
    if frame is None or frame.empty:
        print(f"run {run_id} has no scores for arm {args.arm!r}", file=sys.stderr)
        return 1
    from .utils.frames import text_column

    job_ids = [args.job] if args.job else sorted(text_column(frame, "job_id").unique())
    for job_id in job_ids:
        ranked = rank_candidates(frame, job_id, top_k=cfg.screening.top_k)
        print(f"\n=== {job_id} — top {cfg.screening.top_k} of {len(ranked)} candidates ===")
        _print_table(
            shortlist_table(ranked),
            ["rank", "resume_id", "score", "recommendation", "shortlisted"],
        )
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Run every phase in order."""
    cfg = _load_config(args)
    from .pipeline.run import run_pipeline

    stages = (
        [s.strip() for s in args.stages.split(",") if s.strip()] if args.stages else None
    )
    state = run_pipeline(cfg, stages=stages, dry_run=args.dry_run)

    print()
    print("=" * 78)
    print(f"run id: {state.run_id}")
    print(f"artifacts: {state.root}")
    if args.dry_run:
        print("(dry run — no model calls were made)")
        return 0
    if state.audit_report is not None:
        from .fairness.suite import audit_summary, headline_findings

        summary = audit_summary(state.audit_report)
        print("=" * 78)
        print()
        _print_table(
            headline_findings(state.audit_report),
            ["metric", "value", "unit", "verdict", "q_value_bh"],
        )
        print()
        print(f"metrics {summary['n_metrics']} · failed {summary['n_failed']} · "
              f"significant after FDR {summary['n_significant_after_fdr']} · "
              f"conflicts {summary['conflicts']['n_conflicts']}")
    if state.warnings:
        print()
        print("warnings:")
        for warning in state.warnings:
            print(f"  - {warning}")
    report_html = state.root / "report" / "report.html"
    if report_html.exists():
        print()
        print(f"open the report:  open {report_html}")
    return 0


def cmd_validate_auditor(args: argparse.Namespace) -> int:
    """Verify the audit suite recovers a known bias and stays quiet under a null."""
    cfg = _load_config(args)
    from .validate_auditor import render_validation, validate_auditor

    verdict = validate_auditor(
        cfg,
        injected_bias=args.injected_bias,
        attribute=args.attribute,
        level=args.level,
        samples=args.samples,
    )
    print(render_validation(verdict))
    if args.json:
        _echo(verdict, as_json=True)
    return 0 if verdict["passed"] else 1


def cmd_list(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    what = args.what

    if what == "cues":
        from .perturb.cues import available_families, build_registry, registry_summary

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            registry = build_registry(
                families=available_families(), name_list_id="extended_five_group"
            )
        _print_table(
            registry_summary(registry),
            ["family", "cue_type", "n_levels", "attributes", "slots", "confounded_with"],
        )
        print()
        print("cue_type legend:")
        print("  protected            a protected attribute, varied in isolation")
        print("  protected_confounded a protected attribute entangled with another variable")
        print("  validity             a job-relevant probe; used as a positive control")
        print("  validity_proxy       a style property correlated with protected attributes")
        return 0

    if what == "names":
        from .perturb.names import available_name_lists, load_name_list

        for list_id in available_name_lists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                name_list = load_name_list(list_id)
            print(f"\n=== {list_id} ({name_list.provenance}) ===")
            print(f"{name_list.description}")
            print(f"citation: {name_list.citation}")
            print(f"cells: {name_list.balance_report()}")
        return 0

    if what == "strategies":
        from .mitigate import describe_strategies

        _print_table(
            describe_strategies(),
            ["strategy", "kind", "mechanism", "cost", "known_limitation"],
        )
        return 0

    if what == "rubrics":
        from .screening.rubric import available_rubrics, get_rubric, rubric_table

        for name in available_rubrics():
            print(f"\n=== {name} ===")
            _print_table(rubric_table(get_rubric(name)), ["dimension", "weight",
                                                          "objectively_gradeable"])
        return 0

    if what == "runs":
        root = cfg.resolve(cfg.output_dir)
        if not root.exists():
            print("(no runs yet)")
            return 0
        from .utils.io import read_json

        rows: List[Dict[str, Any]] = []
        for directory in sorted(root.iterdir()):
            manifest = directory / "manifest.json"
            if not manifest.exists():
                continue
            try:
                data = read_json(manifest)
            except Exception:  # noqa: BLE001
                continue
            rows.append(
                {
                    "run_id": directory.name,
                    "status": data.get("status", "incomplete"),
                    "provider": data.get("config", {}).get("llm", {}).get("provider"),
                    "model": data.get("config", {}).get("llm", {}).get("model"),
                    "stages": ",".join(data.get("stages", {})),
                    "warnings": len(data.get("warnings", [])),
                }
            )
        _print_table(rows, ["run_id", "status", "provider", "model", "warnings", "stages"])
        return 0

    print(f"unknown list target {what!r}", file=sys.stderr)
    return 1


def cmd_dashboard(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    import subprocess

    app = Path(__file__).parent / "app" / "dashboard.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app), "--", "--config-root",
               str(cfg.root)]
    print("launching:", " ".join(command))
    try:
        return subprocess.call(command)
    except FileNotFoundError:
        print("streamlit is not installed. Install it with: "
              "pip install 'fair-resume-screening[dashboard]'", file=sys.stderr)
        return 1


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frs", description=_BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Every scoring command accepts --dry-run to print the planned API-call count "
            "before spending quota, and --provider mock to run the entire pipeline offline "
            "with no API key."
        ),
    )
    parser.add_argument("--config", help="path to a YAML config file")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="override any config key, e.g. --set llm.temperature=0.2")
    parser.add_argument("--provider", help="llm.provider shortcut (gemini | mock | openai)")
    parser.add_argument("--model", help="llm.model shortcut")
    parser.add_argument("--corpus", help="data.corpus shortcut (sample | kaggle | dir)")
    parser.add_argument("--seed", type=int, help="master random seed")
    parser.add_argument("--run-name", dest="run_name", help="label used in the run id")
    parser.add_argument("--log-level", dest="log_level",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--version", action="store_true", help="print the version and exit")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("doctor", help="check environment, corpus and provider readiness")

    data_parser = subparsers.add_parser("data", help="phase 1: corpus preparation")
    data_parser.add_argument("data_action", choices=["prepare", "stats"], nargs="?",
                             default="prepare")

    score_parser = subparsers.add_parser(
        "score", help="phase 2: score unperturbed resumes and run the discrimination gate"
    )
    score_parser.add_argument("--dry-run", action="store_true",
                              help="print planned API calls without making them")

    perturb_parser = subparsers.add_parser(
        "perturb", help="phase 3: generate counterfactual variants and score them"
    )
    perturb_parser.add_argument("--cues", help="comma-separated cue families")
    perturb_parser.add_argument("--name-list", dest="name_list",
                                help="name list id (see `frs list names`)")
    perturb_parser.add_argument(
        "--design", choices=["single", "factorial_name_x_gender", "full_factorial"]
    )
    perturb_parser.add_argument("--generate-only", action="store_true",
                                help="generate and validate variants without scoring")
    perturb_parser.add_argument("--dry-run", action="store_true")

    audit_parser = subparsers.add_parser("audit", help="phase 4: run the fairness suite")
    audit_parser.add_argument("--run-id", default="latest")
    audit_parser.add_argument("--arm", default="baseline")
    audit_parser.add_argument("--json", action="store_true")

    mitigate_parser = subparsers.add_parser(
        "mitigate", help="phase 4/6: run mitigation arms and compare them"
    )
    mitigate_parser.add_argument("--strategies", help="comma-separated strategy names")
    mitigate_parser.add_argument("--run-id", help="reuse a previous run's corpus")
    mitigate_parser.add_argument("--list", action="store_true",
                                 help="list available strategies and exit")
    mitigate_parser.add_argument("--dry-run", action="store_true")

    report_parser = subparsers.add_parser("report", help="phase 5: render the report")
    report_parser.add_argument("--run-id", default="latest")

    rank_parser = subparsers.add_parser("rank", help="show the shortlist for a role")
    rank_parser.add_argument("--run-id", default="latest")
    rank_parser.add_argument("--job", help="job id (default: every job)")
    rank_parser.add_argument("--arm", default="baseline")

    pipeline_parser = subparsers.add_parser("pipeline", help="run every phase in order")
    pipeline_parser.add_argument(
        "--stages",
        help=("comma-separated subset of: prepare_data, generate_variants, score_baseline, "
              "score_variants, audit, mitigate, report"),
    )
    pipeline_parser.add_argument("--dry-run", action="store_true")

    validate_parser = subparsers.add_parser(
        "validate-auditor",
        help="verify the audit suite recovers a known bias and stays quiet under a null",
    )
    validate_parser.add_argument("--injected-bias", type=float, default=-3.0)
    validate_parser.add_argument("--attribute", default="gender")
    validate_parser.add_argument("--level", default="female")
    validate_parser.add_argument("--samples", type=int, default=3)
    validate_parser.add_argument("--json", action="store_true")

    list_parser = subparsers.add_parser("list", help="list cues, names, strategies, runs")
    list_parser.add_argument(
        "what", choices=["cues", "names", "strategies", "runs", "rubrics"]
    )

    subparsers.add_parser("dashboard", help="launch the Streamlit dashboard")
    return parser


_COMMANDS = {
    "doctor": cmd_doctor,
    "data": cmd_data,
    "score": cmd_score,
    "perturb": cmd_perturb,
    "audit": cmd_audit,
    "mitigate": cmd_mitigate,
    "report": cmd_report,
    "rank": cmd_rank,
    "pipeline": cmd_pipeline,
    "validate-auditor": cmd_validate_auditor,
    "list": cmd_list,
    "dashboard": cmd_dashboard,
}


#: Global options and whether each takes a value. Used to accept them on either
#: side of the subcommand.
_GLOBAL_OPTIONS = {
    "--config": True, "--set": True, "--provider": True, "--model": True,
    "--corpus": True, "--seed": True, "--run-name": True, "--log-level": True,
    "--version": False,
}


def hoist_global_options(argv: Sequence[str]) -> List[str]:
    """Move global options ahead of the subcommand.

    ``argparse`` subparsers require parent options to precede the subcommand, so
    ``frs pipeline --config x.yaml`` fails with "unrecognized arguments" even
    though it is the order everyone actually types. Rewriting the argument list is
    a smaller and more predictable fix than duplicating every global option onto
    every subparser and then reconciling the defaults.

    Anything after a literal ``--`` is left untouched.
    """
    argv = list(argv)
    if "--" in argv:
        cut = argv.index("--")
        head, tail = argv[:cut], argv[cut:]
    else:
        head, tail = argv, []

    globals_seen: List[str] = []
    rest: List[str] = []
    index = 0
    while index < len(head):
        token = head[index]
        name, _, inline = token.partition("=")
        if name in _GLOBAL_OPTIONS:
            takes_value = _GLOBAL_OPTIONS[name]
            if inline or not takes_value:
                globals_seen.append(token)
                index += 1
                continue
            if index + 1 < len(head):
                globals_seen.extend([token, head[index + 1]])
                index += 2
                continue
            # Missing value: leave it in place so argparse reports the error.
            rest.append(token)
            index += 1
            continue
        rest.append(token)
        index += 1
    return globals_seen + rest + tail


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(hoist_global_options(raw))

    if args.version:
        from . import __version__

        print(f"fair-resume-screening {__version__}")
        return 0
    if not args.command:
        parser.print_help()
        return 0

    handler = _COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse already constrains this
        parser.error(f"unknown command {args.command!r}")
        return 2

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
