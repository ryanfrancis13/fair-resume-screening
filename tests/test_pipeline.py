"""End-to-end pipeline, artifacts, reproducibility, and the auditor oracle.

These are the tests that would catch a wiring break between layers, and the ones
that assert the properties the project claims: that a run is reproducible, that
its artifacts are self-describing, and that the audit suite itself has measured
sensitivity and specificity.
"""

from __future__ import annotations

import json

import pytest

from frs.pipeline import load_run, run_pipeline
from frs.utils.io import read_json


@pytest.fixture(scope="module")
def _module_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("pipeline")


def _fast_cfg(tmp_path):
    """A minimal but complete design: every stage runs, quickly."""
    from frs.config import Config

    cfg = Config.load(apply_env=False)
    cfg.llm.provider = "mock"
    cfg.llm.model = "mock-rubric-v1"
    cfg.llm.cache_enabled = False
    cfg.llm.requests_per_minute = 10**6
    cfg.llm.max_concurrency = 4
    cfg.llm.self_consistency_samples = 2
    cfg.llm.self_consistency_temperature = 0.4
    cfg.llm.mock_noise_sd = 0.8
    cfg.data.max_resumes = 5
    cfg.data.max_jobs = 2
    cfg.perturbation.cue_families = ["name", "pronoun", "education_prestige"]
    cfg.perturbation.levels_per_family = 4
    cfg.fairness.bootstrap_iterations = 200
    cfg.fairness.permutation_iterations = 200
    cfg.mitigation.strategies = [
        "blind_screening", "counterfactual_marginalization", "sensitivity_gated_review"
    ]
    cfg.report.formats = ["html", "markdown", "json"]
    cfg.report.make_plots = True
    cfg.report.plot_dpi = 60
    cfg.report.explanation_cards = True
    cfg.report.max_cards = 5
    cfg.report.model_card = True
    cfg.output_dir = str(tmp_path / "runs")
    cfg.log_level = "ERROR"
    cfg.validate()
    return cfg


@pytest.fixture(scope="module")
def finished_run(_module_tmp):
    """One full pipeline run, shared by the assertions below."""
    cfg = _fast_cfg(_module_tmp)
    state = run_pipeline(cfg)
    return cfg, state


class TestPipelineRuns:
    def test_every_stage_completes(self, finished_run):
        _cfg, state = finished_run
        stages = state.manifest.data["stages"]
        assert set(stages) == {
            "prepare_data", "generate_variants", "score_baseline",
            "score_variants", "audit", "mitigate", "report",
        }
        assert state.manifest.data["status"] == "complete"

    def test_phase2_gate_passes_on_the_offline_scorer(self, finished_run):
        _cfg, state = finished_run
        assert state.sanity["verdict"] == "pass", state.sanity

    def test_variants_all_pass_content_validation(self, finished_run):
        _cfg, state = finished_run
        validation = state.manifest.data["stages"]["generate_variants"]["validation"]
        assert validation["pass_rate"] == 1.0, validation["top_failure_modes"]

    def test_audit_produces_every_metric_family(self, finished_run):
        _cfg, state = finished_run
        families = {m.family for m in state.audit_report.metrics}
        assert {"counterfactual", "group_parity", "allocational", "ranking"} <= families

    def test_noise_floor_is_estimated(self, finished_run):
        _cfg, state = finished_run
        reliability = state.audit_report.reliability
        assert reliability["has_repeats"]
        assert reliability["noise_floor_sd"] > 0

    def test_mitigation_arms_are_evaluated_with_utility_costs(self, finished_run):
        _cfg, state = finished_run
        comparisons = state.mitigation["comparisons"]
        assert comparisons
        for arm, comparison in comparisons.items():
            assert comparison["verdict"], arm
            assert "rank_agreement_kendall_tau" in comparison["utility"], arm


class TestArtifacts:
    def test_run_directory_is_self_describing(self, finished_run):
        _cfg, state = finished_run
        for relative in (
            "manifest.json",
            "resolved_config.yaml",
            "corpus/resumes.jsonl",
            "corpus/jobs.jsonl",
            "corpus/pairs.csv",
            "variants/variants.csv",
            "variants/variants.jsonl",
            "scores/scores_baseline.csv",
            "audit/audit_report.json",
            "audit/metrics.csv",
            "audit/summary.json",
            "mitigation/evaluation.json",
            "report/report.html",
            "report/report.md",
            "report/MODEL_CARD.md",
            "report/explanation_cards.json",
            "logs/run.jsonl",
        ):
            assert (state.root / relative).exists(), relative

    def test_manifest_records_provenance(self, finished_run):
        _cfg, state = finished_run
        manifest = read_json(state.root / "manifest.json")
        assert manifest["config_hash"]
        assert manifest["seed"]
        assert "git" in manifest and "environment" in manifest
        assert manifest["environment"]["python"]

    def test_variants_carry_their_text_and_verdict(self, finished_run):
        _cfg, state = finished_run
        lines = (state.root / "variants" / "variants.jsonl").read_text().strip().split("\n")
        record = json.loads(lines[1])
        assert record["text"], "variant text must be recoverable for review"
        assert "validation" in record
        assert "edits" in record

    def test_score_rows_are_individually_traceable(self, finished_run):
        _cfg, state = finished_run
        frame = state.frames_by_arm["baseline"]
        for column in ("prompt_hash", "resume_hash", "job_hash", "provider", "model",
                       "timestamp", "temperature"):
            assert column in frame.columns
            assert frame[column].notna().all(), column

    def test_html_report_is_self_contained(self, finished_run):
        _cfg, state = finished_run
        html = (state.root / "report" / "report.html").read_text()
        assert "data:image/png;base64," in html, "figures must be embedded, not linked"
        assert "Can the scorer be trusted?" in html
        assert "Metric conflicts" in html

    def test_model_card_states_what_fairness_means_here(self, finished_run):
        _cfg, state = finished_run
        card = read_json(state.root / "report" / "model_card.json")
        assert "what_is_measured" in card["fairness_definition"]
        assert "what_is_not_measured" in card["fairness_definition"]
        assert card["intended_use"]["out_of_scope"]
        assert card["limitations"]


class TestReloadAndReanalyse:
    def test_a_finished_run_reloads_without_calling_a_model(self, finished_run):
        """Re-analysis must be free, and must not risk changing the scores."""
        cfg, state = finished_run
        reloaded = load_run(cfg, state.run_id)
        assert len(reloaded.resumes) == len(state.resumes)
        assert len(reloaded.pairs) == len(state.pairs)
        assert set(reloaded.frames_by_arm) >= {"baseline"}
        assert len(reloaded.frames_by_arm["baseline"]) == len(state.frames_by_arm["baseline"])

    def test_reaudit_of_stored_scores_reproduces_the_metrics(self, finished_run):
        from frs.fairness.suite import run_audit

        cfg, state = finished_run
        reloaded = load_run(cfg, state.run_id)
        again = run_audit(reloaded.frames_by_arm["baseline"], cfg, run_id=state.run_id)
        original = {m.name: m.value for m in state.audit_report.metrics}
        repeated = {m.name: m.value for m in again.metrics}
        shared = set(original) & set(repeated)
        assert len(shared) > 20
        for name in shared:
            if original[name] == original[name]:  # skip NaN
                assert repeated[name] == pytest.approx(original[name], abs=0.02), name


class TestDeterminism:
    def test_two_runs_of_the_same_config_agree(self, tmp_path):
        """Reproducibility is the claim; this is the test of it."""
        first = run_pipeline(
            _fast_cfg(tmp_path / "a"), stages=["prepare_data", "generate_variants",
                                               "score_variants", "audit"]
        )
        second = run_pipeline(
            _fast_cfg(tmp_path / "b"), stages=["prepare_data", "generate_variants",
                                               "score_variants", "audit"]
        )
        assert [v.variant_id for v in first.variants] == [v.variant_id for v in second.variants]

        left = first.frames_by_arm["baseline"].sort_values(["job_id", "variant_id", "sample_index"])
        right = second.frames_by_arm["baseline"].sort_values(["job_id", "variant_id", "sample_index"])
        assert left["overall_score"].tolist() == pytest.approx(right["overall_score"].tolist())

        for a, b in zip(first.audit_report.metrics, second.audit_report.metrics):
            assert a.name == b.name
            if a.value == a.value:
                assert b.value == pytest.approx(a.value, abs=1e-9), a.name


class TestDryRun:
    def test_dry_run_costs_nothing_and_reports_the_plan(self, tmp_path):
        cfg = _fast_cfg(tmp_path)
        state = run_pipeline(cfg, dry_run=True)
        plan = state.manifest.data["stages"]["score_variants"]
        assert plan["dry_run"] is True
        assert plan["n_calls"] > 0
        assert not (state.root / "scores" / "scores_baseline.csv").exists()


class TestCacheMakesReanalysisFree:
    def test_second_run_is_served_from_cache(self, tmp_path):
        cfg = _fast_cfg(tmp_path)
        cfg.llm.cache_enabled = True
        cfg.llm.cache_dir = str(tmp_path / "cache")
        stages = ["prepare_data", "generate_variants", "score_variants"]

        run_pipeline(cfg, stages=stages)
        second = run_pipeline(cfg, stages=stages)
        telemetry = second.manifest.data["stages"]["score_variants"]["telemetry"]["client"]
        assert telemetry["cache"]["hit_rate"] > 0.95


@pytest.mark.slow
class TestAuditorOracle:
    """Does the measurement instrument work?

    An audit that cannot recover a known effect, or that invents one, is not
    evidence. This runs the full pipeline twice — once with zero injected bias,
    once with a known −3-point penalty — and grades the suite on both.
    """

    def test_suite_is_sensitive_and_specific(self, tmp_path):
        from frs.validate_auditor import validate_auditor

        cfg = _fast_cfg(tmp_path)
        cfg.fairness.permutation_iterations = 500
        verdict = validate_auditor(cfg, injected_bias=-3.0, samples=3)
        failed = [c["check"] for c in verdict["checks"] if not c["passed"]]
        assert verdict["passed"], f"failed oracle checks: {failed}"

        recovery = next(
            c for c in verdict["checks"]
            if c["check"] == "sensitivity_recovers_injected_effect"
        )
        expected = recovery["detail"]["expected_transmitted_effect"]
        recovered = recovery["detail"]["recovered_effect"]
        assert recovered == pytest.approx(expected, rel=0.2)
