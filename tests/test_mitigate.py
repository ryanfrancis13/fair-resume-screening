"""Mitigation strategies and the fairness-versus-utility trade-off."""

from __future__ import annotations

import numpy as np
import pytest

from frs.mitigate import (
    ALL_STRATEGIES,
    LEGAL_NOTE,
    POST_HOC_STRATEGIES,
    PRE_INFERENCE_STRATEGIES,
    counterfactual_marginalization,
    cue_effect_debiasing,
    describe_strategies,
    estimate_cue_effects,
    evidence_gate,
    pre_inference_arms,
    sensitivity_gated_review,
    utility_metrics,
)
from test_fairness_metrics import make_frame


class TestCatalogue:
    def test_every_strategy_is_documented_with_its_limitation(self):
        described = {row["strategy"] for row in describe_strategies()}
        assert described == set(ALL_STRATEGIES)
        for row in describe_strategies():
            assert row["known_limitation"], row["strategy"]
            assert row["cost"], row["strategy"]

    def test_kinds_partition_the_strategies(self):
        assert set(PRE_INFERENCE_STRATEGIES) | set(POST_HOC_STRATEGIES) == set(ALL_STRATEGIES)
        assert not set(PRE_INFERENCE_STRATEGIES) & set(POST_HOC_STRATEGIES)

    def test_legal_caveat_is_explicit_about_group_blindness(self):
        assert "Title VII" in LEGAL_NOTE
        assert "never observed" in LEGAL_NOTE

    def test_pre_inference_arms_only_change_the_named_parameters(self, cfg):
        arms = pre_inference_arms(cfg, ["blind_screening", "debias_prompt"])
        assert {a.name for a in arms} == {"blind_screening", "debias_prompt"}
        for arm in arms:
            assert arm.overrides
            assert all(key.startswith("screening.") for key in arm.overrides)


class TestCounterfactualMarginalization:
    def test_removes_the_cue_dependence_by_construction(self, cfg):
        frame = make_frame(gender_effect=-6.0)
        out = counterfactual_marginalization(frame, cfg, cue_family="name")
        assert not out.empty
        # The served row no longer belongs to any demographic level.
        assert out["signal_gender"].isna().all()
        assert out["is_baseline"].all()
        assert len(out) == frame[["job_id", "base_resume_id"]].drop_duplicates().shape[0]

    def test_trimmed_mean_is_robust_to_one_extreme_variant(self, cfg):
        """One catastrophic variant must not drag a candidate's served score down."""
        frame = make_frame(gender_effect=0.0, noise_sd=0.0)
        target = frame.index[~frame["is_baseline"]][0]
        job_id = frame.loc[target, "job_id"]
        resume_id = frame.loc[target, "base_resume_id"]
        variant_id = frame.loc[target, "variant_id"]
        frame.loc[frame["variant_id"] == variant_id, "overall_score"] = 0.0

        def served(estimator: str) -> float:
            cfg.mitigation.marginalization_estimator = estimator
            out = counterfactual_marginalization(frame, cfg)
            row = out[(out["job_id"] == job_id) & (out["base_resume_id"] == resume_id)]
            return float(row["overall_score"].iloc[0])

        assert served("trimmed_mean") > served("mean")

    def test_trim_never_empties_a_small_variant_set(self, cfg):
        cfg.mitigation.trim_fraction = 0.45
        cfg.mitigation.marginalization_variants = 3
        out = counterfactual_marginalization(make_frame(), cfg)
        assert out["overall_score"].notna().all()


class TestSensitivityGatedReview:
    def test_routes_unstable_candidates_and_reports_coverage(self, cfg):
        cfg.mitigation.review_sensitivity_threshold = 2.0
        result = sensitivity_gated_review(make_frame(gender_effect=-6.0), cfg)
        assert result["n_routed_to_human"] > 0
        assert 0.0 <= result["review_fraction"] <= 1.0
        assert "abstaining on everything would look perfectly fair" in result["note"]
        assert (result["frame"]["recommendation"] == "human_review").sum() == \
            result["n_routed_to_human"]

    def test_routes_nobody_when_the_scorer_is_stable(self, cfg):
        cfg.mitigation.review_sensitivity_threshold = 2.0
        result = sensitivity_gated_review(make_frame(gender_effect=0.0), cfg)
        assert result["n_routed_to_human"] == 0

    def test_review_budget_is_checked(self, cfg):
        cfg.mitigation.review_sensitivity_threshold = 0.0
        cfg.mitigation.max_review_fraction = 0.1
        result = sensitivity_gated_review(make_frame(gender_effect=-6.0), cfg)
        assert result["within_review_budget"] is False


class TestCueEffectDebiasing:
    def test_estimates_effects_within_candidate(self):
        estimated = estimate_cue_effects(make_frame(gender_effect=-5.0))
        female = [
            entry["estimate"] for cue, entry in estimated["effects"].items() if "_f" in cue
        ]
        male = [
            entry["estimate"] for cue, entry in estimated["effects"].items() if "_m" in cue
        ]
        assert np.mean(female) == pytest.approx(-5.0, abs=0.3)
        assert np.mean(male) == pytest.approx(0.0, abs=0.3)
        assert "Title VII" in estimated["legal_note"]

    def test_subtracting_the_effects_closes_the_gap(self, cfg):
        from frs.fairness.group_parity import group_parity

        frame = make_frame(gender_effect=-5.0)
        before = group_parity(frame, "gender", reference_level="male",
                              n_boot=200, n_permutations=200)
        corrected = cue_effect_debiasing(frame, cfg)
        after = group_parity(corrected, "gender", reference_level="male",
                             n_boot=200, n_permutations=200)
        assert abs(after["max_pairwise_gap"]) < abs(before["max_pairwise_gap"]) / 2

    def test_in_sample_correction_is_flagged_as_such(self, cfg):
        corrected = cue_effect_debiasing(make_frame(gender_effect=-5.0), cfg)
        assert corrected["in_sample_correction"].all()


class TestEvidenceGate:
    def test_withholds_ungrounded_scores(self, cfg):
        frame = make_frame()
        frame.loc[frame.index[:100], "evidence_support_rate"] = 0.1
        result = evidence_gate(frame, cfg)
        assert result["n_withheld"] == 100
        assert result["n_retained"] == len(frame) - 100
        assert "cannot be audited or contested" in result["note"]


class TestUtilityMetrics:
    def test_an_identical_arm_scores_perfectly(self):
        frame = make_frame()
        utility = utility_metrics(frame, frame.copy())
        assert utility["rank_agreement_kendall_tau"] == pytest.approx(1.0)
        assert utility["top_k_overlap"] == pytest.approx(1.0)
        assert utility["score_dispersion_ratio"] == pytest.approx(1.0)

    def test_flattening_scores_is_caught_by_dispersion(self):
        """The degenerate way to pass a fairness audit."""
        frame = make_frame()
        flattened = frame.copy()
        flattened["overall_score"] = 60.0
        utility = utility_metrics(frame, flattened)
        assert utility["score_dispersion_ratio"] < 0.1

    def test_discrimination_is_measured_against_fit_labels(self):
        frame = make_frame()
        labels = {
            (row.job_id, row.base_resume_id): ("strong" if row.overall_score > 62 else "weak")
            for row in frame.itertuples()
        }
        utility = utility_metrics(frame, frame.copy(), fit_labels=labels)
        assert utility["discrimination_retained"]["kendall_tau_vs_expected_fit"] > 0.3


class TestArmEvaluation:
    def test_verdict_rejects_fairness_bought_by_destroying_the_ranking(self, cfg):
        from frs.mitigate import evaluate_arms

        baseline = make_frame(gender_effect=-6.0, noise_sd=0.5)
        flattened = baseline.copy()
        flattened["overall_score"] = 60.0
        flattened["arm"] = "flatten"
        evaluation = evaluate_arms(
            {"baseline": baseline, "flatten": flattened}, cfg, run_id="t"
        )
        comparison = evaluation["comparisons"]["flatten"]
        assert comparison["verdict"] in (
            "fairness_improved_at_utility_cost", "not_recommended"
        )
        assert not comparison["utility_acceptable"]
        assert any("dispersion" in reason for reason in comparison["verdict_reasons"])

    def test_a_genuine_debias_is_recommended(self, cfg):
        from frs.mitigate import evaluate_arms

        baseline = make_frame(gender_effect=-6.0, noise_sd=0.5)
        fixed = cue_effect_debiasing(baseline, cfg)
        fixed["arm"] = "debiased"
        evaluation = evaluate_arms({"baseline": baseline, "debiased": fixed}, cfg, run_id="t")
        comparison = evaluation["comparisons"]["debiased"]
        assert comparison["n_metrics_improved"] >= 1
        assert comparison["utility_acceptable"]

    def test_missing_baseline_arm_is_an_error(self, cfg):
        from frs.mitigate import evaluate_arms

        with pytest.raises(KeyError):
            evaluate_arms({"other": make_frame()}, cfg, baseline_arm="baseline")


class TestStructurallyInvariantArms:
    """An arm that removes the cue cannot be validated by a cue-varying audit."""

    def test_marginalization_is_reported_as_removing_cue_variation(self, cfg):
        from frs.mitigate import evaluate_arms

        baseline = make_frame(gender_effect=-6.0, noise_sd=0.5)
        marginalised = counterfactual_marginalization(baseline, cfg)
        marginalised["arm"] = "marginalised"
        evaluation = evaluate_arms(
            {"baseline": baseline, "marginalised": marginalised}, cfg, run_id="t"
        )
        comparison = evaluation["comparisons"]["marginalised"]
        assert comparison["verdict"] == "cue_variation_eliminated_by_construction"
        assert any(
            "property of the intervention" in reason
            for reason in comparison["verdict_reasons"]
        )

    def test_detector_recognises_a_collapsed_arm(self, cfg):
        from frs.mitigate.evaluate import _eliminates_cue_variation

        baseline = make_frame(gender_effect=-3.0)
        assert not _eliminates_cue_variation(baseline)
        assert _eliminates_cue_variation(counterfactual_marginalization(baseline, cfg))
