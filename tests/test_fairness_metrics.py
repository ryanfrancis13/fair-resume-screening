"""Metric families, reliability, conflicts — driven by a synthetic score table.

Building the table directly (rather than by running the pipeline) lets each test
inject a known effect and assert that the specific metric recovers it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frs.fairness.allocational import allocational_analysis, equal_opportunity
from frs.fairness.calibration_fairness import calibration_by_group
from frs.fairness.conflicts import detect_conflicts, summarise_conflicts
from frs.fairness.counterfactual import counterfactual_invariance, cue_effect_table
from frs.fairness.group_parity import group_parity
from frs.fairness.intersectional import intersectional_analysis
from frs.fairness.rationale_audit import distinctive_terms, lexicon_rates
from frs.fairness.reliability import (
    noise_floor_gate,
    parse_quality,
    self_consistency,
    variance_decomposition,
)
from frs.schemas import MetricResult


def make_frame(
    *,
    gender_effect: float = 0.0,
    race_effect: float = 0.0,
    interaction: float = 0.0,
    n_resumes: int = 12,
    n_jobs: int = 3,
    names_per_cell: int = 3,
    n_samples: int = 3,
    noise_sd: float = 0.0,
    seed: int = 11,
    with_labels: bool = True,
    rationale_skew: bool = False,
) -> pd.DataFrame:
    """A score table with known, injected effects.

    Mirrors the real design: a large between-resume spread, a baseline row per
    candidate, one variant per name cell, and repeated decodes so a noise floor
    is estimable.
    """
    generator = np.random.default_rng(seed)
    rows = []
    resume_strength = {f"R{i:02d}": generator.normal(60, 12) for i in range(n_resumes)}
    cells = [("female", "Black"), ("female", "White"), ("male", "Black"), ("male", "White")]

    for job_index in range(n_jobs):
        job_id = f"J{job_index:02d}"
        job_shift = generator.normal(0, 3)
        for resume_id, strength in resume_strength.items():
            label = int(strength > 60)
            base_value = strength + job_shift

            for sample in range(n_samples):
                noisy = base_value + generator.normal(0, noise_sd)
                rows.append(
                    dict(
                        run_id="t", job_id=job_id, base_resume_id=resume_id,
                        variant_id=f"{resume_id}__baseline", sample_index=sample,
                        overall_score=noisy,
                        # Derived from the *noisy* score, as a real scorer's would be,
                        # so repeated decodes can genuinely disagree about the verdict.
                        recommendation="advance" if noisy >= 60 else "reject",
                        confidence=0.7, is_baseline=True, cue_families="", cue_ids="",
                        arm="baseline", parse_ok=True, error="",
                        evidence_support_rate=1.0, prompt_hash=f"{resume_id}|{job_id}|base",
                        rationale_text="The candidate is capable and delivered results.",
                        provider="mock", model="mock", qualified_label=label,
                    )
                )

            for gender, race in cells:
                shift = (
                    (gender_effect if gender == "female" else 0.0)
                    + (race_effect if race == "Black" else 0.0)
                    + (interaction if (gender == "female" and race == "Black") else 0.0)
                )
                for index in range(names_per_cell):
                    variant_id = f"{resume_id}__{gender[0]}{race[0]}{index}"
                    value = base_value + shift
                    if rationale_skew and gender == "female":
                        rationale = "She is supportive, collaborative and hardworking."
                    else:
                        rationale = "The candidate is talented, decisive and drove results."
                    for sample in range(n_samples):
                        noisy = value + generator.normal(0, noise_sd)
                        rows.append(
                            dict(
                                run_id="t", job_id=job_id, base_resume_id=resume_id,
                                variant_id=variant_id, sample_index=sample,
                                overall_score=noisy,
                                recommendation="advance" if noisy >= 60 else "reject",
                                confidence=0.7, is_baseline=False, cue_families="name",
                                cue_ids=f"name=n_{gender[0]}{race[0]}{index}",
                                arm="baseline", parse_ok=True, error="",
                                evidence_support_rate=1.0,
                                prompt_hash=f"{resume_id}|{job_id}|{variant_id}",
                                rationale_text=rationale,
                                signal_gender=gender, signal_race_ethnicity=race,
                                provider="mock", model="mock",
                                qualified_label=label if with_labels else None,
                            )
                        )
    frame = pd.DataFrame(rows)
    frame["signal_gender"] = frame.get("signal_gender")
    return frame


class TestCounterfactualInvariance:
    def test_no_swing_when_nothing_is_injected(self):
        analysis = counterfactual_invariance(make_frame(), n_boot=200)
        assert analysis["mean_max_swing"] == pytest.approx(0.0, abs=1e-6)
        assert analysis["decision_flip_rate"] == 0.0

    def test_recovers_the_injected_swing(self):
        analysis = counterfactual_invariance(make_frame(gender_effect=-4.0), n_boot=200)
        assert analysis["mean_max_swing"] == pytest.approx(4.0, abs=0.2)
        assert analysis["swing_ci_low"] <= 4.0 <= analysis["swing_ci_high"] + 1e-9

    def test_separates_cue_flips_from_noise_flips(self):
        """The build plan's warning, made mechanical."""
        analysis = counterfactual_invariance(
            make_frame(gender_effect=0.0, noise_sd=6.0, seed=5), n_boot=200
        )
        assert analysis["decision_flip_rate_from_noise"] > 0.0
        assert analysis["decision_flip_rate_above_noise"] <= analysis["decision_flip_rate"]

    def test_cue_effect_table_is_centred_on_each_candidate(self):
        table = cue_effect_table(make_frame(gender_effect=-4.0))
        female = table[table.cue_ids.str.contains("_f")]
        male = table[table.cue_ids.str.contains("_m")]
        assert female.mean_delta.mean() == pytest.approx(-4.0, abs=0.2)
        assert male.mean_delta.mean() == pytest.approx(0.0, abs=0.2)


class TestGroupParity:
    def test_recovers_the_injected_gap_and_flags_it(self):
        analysis = group_parity(
            make_frame(gender_effect=-4.0), "gender",
            reference_level="male", n_boot=300, n_permutations=300,
        )
        comparison = next(c for c in analysis["comparisons"] if c["level"] == "female")
        assert comparison["gap"] == pytest.approx(-4.0, abs=0.2)
        assert comparison["permutation_p"] < 0.01
        assert comparison["practically_significant"]

    def test_reports_no_gap_when_none_exists(self):
        analysis = group_parity(
            make_frame(), "gender", reference_level="male",
            n_boot=300, n_permutations=300,
        )
        comparison = next(c for c in analysis["comparisons"] if c["level"] == "female")
        assert comparison["gap"] == pytest.approx(0.0, abs=0.05)
        assert not comparison["practically_significant"]

    def test_every_comparison_carries_power_information(self):
        analysis = group_parity(
            make_frame(gender_effect=-0.2), "gender", reference_level="male",
            n_boot=300, n_permutations=300,
        )
        for comparison in analysis["comparisons"]:
            assert comparison["minimum_detectable_effect"] is not None
            assert comparison["blocks_needed_for_1pt"] is not None

    def test_missing_reference_level_falls_back_and_says_so(self):
        analysis = group_parity(
            make_frame(), "gender", reference_level="nonexistent",
            n_boot=300, n_permutations=300,
        )
        assert "absent from this run" in analysis["reference_note"]

    def test_single_level_attribute_is_reported_not_crashed(self):
        frame = make_frame()
        frame["signal_gender"] = "female"
        assert "error" in group_parity(frame, "gender", n_boot=300, n_permutations=300)


class TestAllocational:
    def test_equal_selection_rates_without_bias(self):
        """Real scores are continuous, so shortlist membership is settled by score."""
        analysis = allocational_analysis(
            make_frame(noise_sd=1.5, n_jobs=8, seed=21), "gender", k=5
        )
        assert not analysis["ties_dominate_selection"]
        assert analysis["impact_ratio"] == pytest.approx(1.0, abs=0.35)
        assert analysis["four_fifths_rule_satisfied"]

    def test_pervasive_ties_are_reported_rather_than_scored(self):
        """With every variant on the same score, top-k is the tiebreak, not the model."""
        analysis = allocational_analysis(make_frame(noise_sd=0.0), "gender", k=5)
        assert analysis["ties_dominate_selection"]
        assert "tiebreak" in analysis["tie_warning"]

    def test_tiebreak_is_independent_of_the_cue_label(self):
        """Alphabetical tiebreaks hand every tie to whichever label sorts first."""
        import pandas as pd

        from frs.screening.rank import _tiebreak

        ids = pd.Series([f"R{i:02d}__female" for i in range(200)]
                        + [f"R{i:02d}__male" for i in range(200)])
        keys = _tiebreak(ids).to_numpy()
        female, male = keys[:200], keys[200:]
        # In a head-to-head tie the winner is whichever key sorts first; over many
        # pairings that must be close to a coin flip.
        female_wins = sum(1 for f, m in zip(female, male) if f < m) / len(female)
        assert 0.35 < female_wins < 0.65, (
            f"hashed tiebreak favoured one label in {female_wins:.0%} of ties"
        )

    def test_injected_bias_breaks_the_four_fifths_rule(self):
        analysis = allocational_analysis(make_frame(gender_effect=-8.0), "gender", k=3)
        assert analysis["selection_rate"]["female"] < analysis["selection_rate"]["male"]
        assert analysis["impact_ratio"] < 0.8
        assert analysis["four_fifths_rule_satisfied"] is False

    def test_legal_note_travels_with_the_number(self):
        analysis = allocational_analysis(make_frame(), "gender", k=5)
        assert "not a legal" in analysis["legal_note"]

    def test_equal_opportunity_needs_labels(self):
        frame = make_frame(with_labels=False)
        frame["qualified_label"] = None
        assert "error" in equal_opportunity(frame, "gender")

    def test_equal_opportunity_computed_when_labels_exist(self):
        result = equal_opportunity(make_frame(gender_effect=-8.0), "gender", k=3)
        assert "equal_opportunity_difference" in result
        assert result["equal_opportunity_difference"] >= 0


class TestIntersectional:
    def test_finds_an_interaction_that_marginals_hide(self):
        """The core intersectional argument, asserted numerically."""
        frame = make_frame(gender_effect=0.0, race_effect=0.0, interaction=-6.0)
        marginal = group_parity(frame, "gender", reference_level="male",
                                n_boot=300, n_permutations=300)
        analysis = intersectional_analysis(frame, ["gender", "race_ethnicity"],
                                           n_permutations=300)

        marginal_gap = abs(marginal["max_pairwise_gap"])
        assert analysis["worst_cell_gap"] > marginal_gap + 1.0, (
            "the intersectional gap must exceed the marginal gap it hides behind"
        )
        assert analysis["worst_cell"] == "female x Black"
        assert analysis["worst_interaction_residual"] < -1.0

    def test_thin_cells_are_flagged_not_hidden(self):
        analysis = intersectional_analysis(
            make_frame(n_resumes=3, n_jobs=1, names_per_cell=1),
            ["gender", "race_ethnicity"], n_permutations=200,
        )
        assert analysis["n_thin_cells"] >= 1

    def test_requires_both_attributes_on_the_same_rows(self):
        frame = make_frame()
        frame["signal_race_ethnicity"] = None
        assert "error" in intersectional_analysis(frame, ["gender", "race_ethnicity"])


class TestReliability:
    def test_variance_decomposition_separates_the_three_sources(self):
        analysis = variance_decomposition(
            make_frame(gender_effect=-4.0, noise_sd=1.0, seed=3)
        )
        components = analysis["variance_components"]
        assert components["between_resume"] > components["between_cue"]
        assert components["within_cue_noise"] > 0
        assert analysis["has_repeats"]
        assert analysis["icc1_resume"] > 0.5

    def test_noise_floor_recovers_the_injected_noise(self):
        analysis = variance_decomposition(make_frame(noise_sd=2.0, n_samples=6, seed=4))
        assert analysis["noise_floor_sd"] == pytest.approx(2.0, rel=0.3)

    def test_absent_repeats_are_reported_as_such(self):
        analysis = variance_decomposition(make_frame(n_samples=1))
        assert not analysis["has_repeats"]
        assert "noise floor could not be estimated" in analysis["interpretation"]

    def test_self_consistency_groups_on_the_prompt_hash(self):
        analysis = self_consistency(make_frame(noise_sd=3.0, seed=9))
        assert analysis["n_repeated_prompts"] > 0
        assert analysis["mean_within_prompt_sd"] > 0

    def test_noise_floor_gate_rejects_sub_noise_effects(self):
        reliability = variance_decomposition(make_frame(noise_sd=2.0, n_samples=6))
        assert noise_floor_gate(reliability, 0.1)["passes"] is False
        assert noise_floor_gate(reliability, 12.0)["passes"] is True

    def test_gate_is_undefined_without_repeats(self):
        reliability = variance_decomposition(make_frame(n_samples=1))
        assert noise_floor_gate(reliability, 5.0)["passes"] is None

    def test_differential_attrition_is_warned_about(self):
        """Attrition correlated with the condition invalidates every comparison."""
        frame = make_frame()
        female_rows = frame.index[frame["signal_gender"] == "female"]
        frame.loc[female_rows[:120], "parse_ok"] = False
        report = parse_quality(frame)
        assert any("parse success rate differs" in w for w in report["warnings"])


class TestRationaleAudit:
    def test_detects_agentic_versus_communal_skew(self):
        """Identical scores, different descriptions — invisible to score metrics."""
        analysis = lexicon_rates(make_frame(rationale_skew=True), "gender")
        female = analysis["by_level"]["female"]
        male = analysis["by_level"]["male"]
        assert female["agentic_minus_communal"] < male["agentic_minus_communal"]
        assert female["ability_minus_effort"] < male["ability_minus_effort"]

    def test_no_skew_when_language_is_identical(self):
        analysis = lexicon_rates(make_frame(rationale_skew=False), "gender")
        female = analysis["by_level"]["female"]
        male = analysis["by_level"]["male"]
        assert female["agentic_minus_communal"] == pytest.approx(
            male["agentic_minus_communal"], abs=0.01
        )

    def test_distinctive_terms_surface_the_skewed_words(self):
        analysis = distinctive_terms(make_frame(rationale_skew=True), "gender", top_n=8)
        female_terms = {t["term"] for t in analysis["distinctive_terms"]["female"]}
        assert {"supportive", "collaborative", "hardworking"} & female_terms


class TestCalibrationFairness:
    def test_computes_per_group_calibration_when_labels_exist(self):
        analysis = calibration_by_group(make_frame(gender_effect=-6.0), "gender")
        assert "error" not in analysis
        assert set(analysis["levels_analysed"]) == {"female", "male"}
        for info in analysis["per_group"].values():
            assert info["ece"] == info["ece"]  # not NaN

    def test_reports_the_impossibility_note(self):
        analysis = calibration_by_group(make_frame(gender_effect=-6.0), "gender")
        assert "Chouldechova" in analysis["impossibility_note"] or \
               "not in tension" in analysis["impossibility_note"]

    def test_declines_without_labels(self):
        frame = make_frame()
        frame["qualified_label"] = None
        assert "error" in calibration_by_group(frame, "gender")


class TestConflictDetection:
    def _metric(self, name, family, value, *, attribute="gender", passed=None,
                q=float("nan"), higher_is_better=False, detail=None):
        return MetricResult(
            name=name, family=family, attribute=attribute, value=value,
            passed=passed, q_value=q, higher_is_better=higher_is_better,
            detail=detail or {},
        )

    def test_flags_score_parity_versus_allocational_disagreement(self):
        metrics = [
            self._metric("max_pairwise_score_gap[gender]", "group_parity", 0.3),
            self._metric("selection_rate_parity_diff[gender@k=5]", "allocational", 0.4,
                         passed=False),
        ]
        types = {c["type"] for c in detect_conflicts(metrics)}
        assert "score_parity_vs_allocational" in types

    def test_flags_individual_versus_group_disagreement(self):
        metrics = [
            self._metric("max_pairwise_score_gap[gender]", "group_parity", 0.2),
            self._metric("counterfactual_max_swing[all]", "counterfactual", 6.0,
                         attribute="all", passed=False),
        ]
        types = {c["type"] for c in detect_conflicts(metrics)}
        assert "individual_vs_group" in types

    def test_flags_the_calibration_impossibility(self):
        metrics = [
            self._metric("calibration_ece_spread[gender]", "calibration", 0.01, passed=True,
                         detail={"impossibility_note": "base rates differ"}),
            self._metric("equalised_odds_tpr_spread[gender]", "calibration", 0.4, passed=False),
        ]
        conflicts = detect_conflicts(metrics)
        entry = next(c for c in conflicts if c["type"] == "calibration_vs_error_rates")
        assert entry["severity"] == "expected"
        assert "Kleinberg" in entry["citation"]

    def test_flags_an_inert_scorer_as_critical(self):
        """An unbiased constant function must not pass as fair."""
        metrics = [
            self._metric("counterfactual_max_swing[all]", "counterfactual", 0.01,
                         attribute="all"),
            self._metric("counterfactual_max_swing[education_prestige]", "counterfactual",
                         0.02, attribute="education_prestige"),
        ]
        conflicts = detect_conflicts(metrics)
        entry = next(c for c in conflicts if c["type"] == "validity_collapse")
        assert entry["severity"] == "critical"
        assert summarise_conflicts(conflicts)["has_critical"]

    def test_flags_a_null_that_is_really_low_power(self):
        metrics = [
            self._metric(
                "score_gap[gender:female_vs_male]", "group_parity", -0.4,
                detail={"minimum_detectable_effect": 3.5, "blocks_needed_for_1pt": 90},
            )
        ]
        entry = next(
            c for c in detect_conflicts(metrics) if c["type"] == "power_masquerade"
        )
        assert "absence of evidence" in entry["explanation"]

    def test_flags_missing_noise_floor(self):
        conflicts = detect_conflicts([], reliability={"has_repeats": False})
        assert any(c["type"] == "missing_noise_floor" for c in conflicts)

    def test_flags_noise_dominated_failures(self):
        metrics = [
            self._metric("counterfactual_max_swing[all]", "counterfactual", 1.2,
                         attribute="all", passed=False),
        ]
        conflicts = detect_conflicts(
            metrics,
            reliability={"has_repeats": True, "cue_effect_to_noise_ratio": 0.4,
                         "noise_floor_sd": 0.9},
        )
        entry = next(c for c in conflicts if c["type"] == "noise_dominated_failures")
        assert "Do not report these as bias" in entry["recommendation"]

    def test_clean_results_produce_no_conflicts(self):
        metrics = [
            self._metric("max_pairwise_score_gap[gender]", "group_parity", 0.2, passed=True),
            self._metric("counterfactual_max_swing[all]", "counterfactual", 0.3,
                         attribute="all", passed=True),
            self._metric("counterfactual_max_swing[education_prestige]", "counterfactual",
                         5.0, attribute="education_prestige", passed=False),
        ]
        conflicts = detect_conflicts(
            metrics,
            reliability={"has_repeats": True, "cue_effect_to_noise_ratio": 3.0,
                         "noise_floor_sd": 0.4},
        )
        assert not conflicts
