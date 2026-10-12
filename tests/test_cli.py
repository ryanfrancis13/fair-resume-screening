"""CLI surface: argument handling and the commands that need no model."""

from __future__ import annotations

import pytest

from frs.cli import build_parser, hoist_global_options, main


class TestGlobalOptionOrdering:
    """`frs pipeline --config x.yaml` must work, not just `frs --config x.yaml pipeline`."""

    def test_options_after_the_subcommand_are_hoisted(self):
        assert hoist_global_options(["pipeline", "--config", "c.yaml"]) == [
            "--config", "c.yaml", "pipeline"
        ]

    def test_inline_values_are_hoisted(self):
        assert hoist_global_options(["audit", "--provider=mock"]) == [
            "--provider=mock", "audit"
        ]

    def test_flags_without_values_are_hoisted(self):
        assert hoist_global_options(["pipeline", "--version"]) == ["--version", "pipeline"]

    def test_repeatable_option_keeps_every_occurrence(self):
        hoisted = hoist_global_options(
            ["pipeline", "--set", "a=1", "--set", "b=2", "--dry-run"]
        )
        assert hoisted == ["--set", "a=1", "--set", "b=2", "pipeline", "--dry-run"]

    def test_subcommand_options_are_left_alone(self):
        assert hoist_global_options(["audit", "--run-id", "abc", "--json"]) == [
            "audit", "--run-id", "abc", "--json"
        ]

    def test_already_correct_order_is_unchanged(self):
        argv = ["--provider", "mock", "pipeline", "--dry-run"]
        assert hoist_global_options(argv) == argv

    def test_nothing_after_a_double_dash_is_moved(self):
        argv = ["dashboard", "--", "--provider", "mock"]
        assert hoist_global_options(argv) == argv

    def test_a_missing_value_is_left_for_argparse_to_report(self):
        assert hoist_global_options(["pipeline", "--config"]) == ["pipeline", "--config"]

    def test_parser_accepts_both_orders(self):
        parser = build_parser()
        first = parser.parse_args(hoist_global_options(["--provider", "mock", "audit"]))
        second = parser.parse_args(hoist_global_options(["audit", "--provider", "mock"]))
        assert first.provider == second.provider == "mock"
        assert first.command == second.command == "audit"


class TestCommandsThatNeedNoModel:
    def test_no_arguments_prints_help(self, capsys):
        assert main([]) == 0
        assert "Fair Resume Screening" in capsys.readouterr().out

    def test_version(self, capsys):
        assert main(["--version"]) == 0
        assert "fair-resume-screening" in capsys.readouterr().out

    def test_doctor_passes_with_the_offline_provider(self, capsys):
        assert main(["--provider", "mock", "doctor"]) == 0
        out = capsys.readouterr().out
        assert "All checks passed" in out
        assert "cue registry" in out

    @pytest.mark.parametrize(
        "target", ["cues", "names", "strategies", "rubrics", "runs"]
    )
    def test_list_targets(self, target, capsys):
        assert main(["list", target]) == 0
        assert capsys.readouterr().out.strip()

    def test_list_cues_shows_the_type_legend(self, capsys):
        main(["list", "cues"])
        out = capsys.readouterr().out
        assert "validity" in out
        assert "protected_confounded" in out

    def test_data_stats(self, capsys):
        assert main(["--provider", "mock", "data", "stats"]) == 0
        assert "n_resumes" in capsys.readouterr().out

    def test_mitigate_list(self, capsys):
        assert main(["mitigate", "--list"]) == 0
        out = capsys.readouterr().out
        assert "blind_screening" in out
        assert "known_limitation" in out

    def test_unknown_config_key_is_a_clean_error(self, capsys):
        assert main(["--set", "nonsense.key=1", "doctor"]) == 1
        assert "unknown config section" in capsys.readouterr().err

    def test_malformed_override_is_a_clean_error(self, capsys):
        assert main(["--set", "no-equals-sign", "doctor"]) == 1
        assert "key=value" in capsys.readouterr().err

    def test_audit_without_any_run_is_a_clean_error(self, tmp_path, capsys):
        code = main([
            "--provider", "mock",
            "--set", f"output_dir={tmp_path / 'empty'}",
            "audit",
        ])
        assert code == 1
        assert "no completed runs" in capsys.readouterr().err


class TestDryRun:
    def test_pipeline_dry_run_reports_the_plan_without_scoring(self, tmp_path, capsys):
        code = main([
            "--provider", "mock",
            "--set", f"output_dir={tmp_path / 'runs'}",
            "--set", "data.max_resumes=3",
            "--set", "data.max_jobs=1",
            "--set", "perturbation.cue_families=name",
            "--set", "perturbation.levels_per_family=2",
            "pipeline", "--dry-run",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "dry run" in out
        assert "no model calls were made" in out
