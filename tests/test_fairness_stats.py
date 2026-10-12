"""The statistical core.

These tests are the ones that matter most for defensibility: a fairness number is
only as good as the inference behind it. Each test states the statistical property
being asserted, not just the code path.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from frs.fairness.stats import (
    achieved_power,
    benjamini_hochberg,
    cliffs_delta,
    cluster_bootstrap_ci,
    cluster_robust_ols,
    cohens_d,
    cohens_dz,
    dummy_code,
    holm_bonferroni,
    minimum_detectable_effect,
    paired_block_differences,
    paired_test,
    required_blocks,
    two_proportion_test,
    within_block_permutation_test,
)


def blocked_design(effect: float, n_blocks: int = 12, per_group: int = 5,
                   between_sd: float = 12.0, within_sd: float = 1.5, seed: int = 7):
    """Simulate the project's design: large between-resume variance, small effect."""
    generator = np.random.default_rng(seed)
    values, groups, blocks = [], [], []
    for block in range(n_blocks):
        base = generator.normal(60, between_sd)
        for group, shift in (("male", 0.0), ("female", effect)):
            for _ in range(per_group):
                values.append(base + shift + generator.normal(0, within_sd))
                groups.append(group)
                blocks.append(f"R{block}")
    return values, groups, blocks


class TestBlockingIsNecessary:
    """The headline statistical claim of the project's analysis design."""

    def test_paired_analysis_recovers_an_effect_the_pooled_test_misses(self):
        values, groups, blocks = blocked_design(effect=-3.0)

        comparison = paired_block_differences(values, groups, blocks, "female", "male")
        paired = paired_test(comparison.differences)

        female = [v for v, g in zip(values, groups) if g == "female"]
        male = [v for v, g in zip(values, groups) if g == "male"]
        pooled_p = sps.ttest_ind(female, male, equal_var=False).pvalue

        assert comparison.mean_difference == pytest.approx(-3.0, abs=0.6)
        assert paired["t_p_value"] < 0.001, "the blocked test must find the effect"
        assert pooled_p > 0.05, (
            "the pooled test must miss it — between-resume variance swamps the effect, "
            "which is exactly why the design is blocked"
        )

    def test_only_blocks_containing_both_levels_contribute(self):
        values = [10.0, 20.0, 30.0]
        groups = ["female", "male", "female"]
        blocks = ["A", "A", "B"]
        comparison = paired_block_differences(values, groups, blocks, "female", "male")
        assert comparison.n_blocks == 1
        assert comparison.block_ids == ["A"]


class TestPermutationTest:
    def test_detects_a_real_within_block_effect(self):
        values, groups, blocks = blocked_design(effect=-3.0)
        result = within_block_permutation_test(
            values, groups, blocks, level_a="female", level_b="male",
            n_permutations=1000, rng=np.random.default_rng(0),
        )
        assert result["p_value"] < 0.01
        assert result["observed"] == pytest.approx(-3.0, abs=0.6)

    def test_stays_quiet_under_the_null(self):
        values, groups, blocks = blocked_design(effect=0.0)
        result = within_block_permutation_test(
            values, groups, blocks, level_a="female", level_b="male",
            n_permutations=1000, rng=np.random.default_rng(0),
        )
        assert result["p_value"] > 0.05

    def test_never_reports_p_equal_to_zero(self):
        values, groups, blocks = blocked_design(effect=-30.0)
        result = within_block_permutation_test(
            values, groups, blocks, level_a="female", level_b="male",
            n_permutations=200, rng=np.random.default_rng(0),
        )
        assert result["p_value"] > 0.0, "a finite resample cannot justify p = 0"

    def test_omnibus_statistic_handles_many_levels(self):
        generator = np.random.default_rng(3)
        values, groups, blocks = [], [], []
        for block in range(10):
            base = generator.normal(60, 10)
            for group, shift in (("a", 0.0), ("b", -1.0), ("c", -4.0)):
                values.append(base + shift)
                groups.append(group)
                blocks.append(f"R{block}")
        result = within_block_permutation_test(
            values, groups, blocks, statistic="max_abs_gap",
            n_permutations=1000, rng=np.random.default_rng(0),
        )
        assert result["observed"] == pytest.approx(4.0, abs=0.5)
        assert result["p_value"] < 0.05

    def test_too_few_blocks_declines_rather_than_guessing(self):
        result = within_block_permutation_test(
            [1.0, 2.0], ["a", "b"], ["R0", "R0"], level_a="a", level_b="b",
        )
        assert result["p_value"] != result["p_value"]  # NaN
        assert "not performed" in result["note"]


class TestClusterRobustOLS:
    def test_recovers_the_coefficient_with_plausible_standard_errors(self):
        values, groups, blocks = blocked_design(effect=-3.0)
        columns, reference = dummy_code(groups, reference="male", prefix="gender")
        fit = cluster_robust_ols(values, columns, blocks)
        assert reference == "male"
        assert fit.coefficients["gender[female]"] == pytest.approx(-3.0, abs=0.6)
        assert fit.p_values["gender[female]"] < 0.01
        assert fit.n_clusters == 12
        assert fit.df_resid == 11, "inference is on G-1 df, not n-k"

    def test_drops_collinear_columns_and_says_so(self):
        generator = np.random.default_rng(1)
        n = 60
        y = generator.normal(60, 10, n).tolist()
        indicator = (np.arange(n) % 2).astype(float).tolist()
        columns = {"a": indicator, "duplicate_of_a": indicator, "always_zero": [0.0] * n}
        fit = cluster_robust_ols(y, columns, [f"B{i // 5}" for i in range(n)])
        assert np.isfinite(fit.coefficients["a"])
        assert "duplicate_of_a" not in fit.coefficients
        assert "non-estimable" in fit.note

    def test_all_results_are_finite_on_a_wide_design(self):
        """Guards the numerical-stability path, not just the happy path."""
        generator = np.random.default_rng(2)
        n = 800
        y = generator.normal(60, 10, n).tolist()
        columns = {
            f"level_{i}": (generator.integers(0, 2, n)).astype(float).tolist()
            for i in range(8)
        }
        fit = cluster_robust_ols(y, columns, [f"B{i % 12}" for i in range(n)])
        assert all(np.isfinite(v) for v in fit.coefficients.values())
        assert all(np.isfinite(v) for v in fit.std_errors.values())

    def test_too_few_clusters_declines(self):
        fit = cluster_robust_ols([1.0, 2.0, 3.0], {"x": [0.0, 1.0, 0.0]}, ["A", "A", "A"])
        assert fit.coefficients["x"] != fit.coefficients["x"]  # NaN
        assert "insufficient" in fit.note


class TestBootstrap:
    def test_interval_brackets_the_point_estimate(self):
        blocks = {f"R{i}": float(v) for i, v in enumerate(np.linspace(-4, -2, 15))}
        result = cluster_bootstrap_ci(
            blocks, lambda xs: float(np.mean(xs)), n_boot=1000,
            rng=np.random.default_rng(0),
        )
        assert result["ci_low"] < result["point"] < result["ci_high"]

    def test_declines_with_too_few_blocks(self):
        result = cluster_bootstrap_ci({"a": 1.0}, lambda xs: float(np.mean(xs)))
        assert "not performed" in result["note"]


class TestEffectSizes:
    def test_paired_effect_size_exceeds_the_unpaired_one(self):
        """dz is the right effect size for a blocked design."""
        values, groups, blocks = blocked_design(effect=-3.0)
        comparison = paired_block_differences(values, groups, blocks, "female", "male")
        female = [v for v, g in zip(values, groups) if g == "female"]
        male = [v for v, g in zip(values, groups) if g == "male"]
        assert abs(cohens_dz(comparison.differences)) > abs(cohens_d(female, male))

    def test_cliffs_delta_bounds(self):
        assert cliffs_delta([1, 2, 3], [4, 5, 6]) == pytest.approx(-1.0)
        assert cliffs_delta([4, 5, 6], [1, 2, 3]) == pytest.approx(1.0)


class TestMultiplicity:
    def test_bh_is_monotone_and_bounded(self):
        result = benjamini_hochberg([0.001, 0.02, 0.04, 0.3, 0.5])
        q = result["q_values"]
        assert q == sorted(q), "q-values must be monotone in p"
        assert all(0.0 <= v <= 1.0 for v in q)
        assert result["n_rejected"] == 2

    def test_bh_is_less_conservative_than_holm(self):
        p_values = [0.001, 0.01, 0.02, 0.03, 0.04]
        bh = benjamini_hochberg(p_values)
        holm = holm_bonferroni(p_values)
        assert bh["n_rejected"] >= sum(holm["rejected"])

    def test_nan_p_values_are_passed_through(self):
        result = benjamini_hochberg([0.01, float("nan"), 0.2])
        assert result["q_values"][1] is None
        assert result["n_tests"] == 2

    def test_wide_suite_of_nulls_yields_few_discoveries(self):
        """Without correction a suite this wide manufactures findings."""
        generator = np.random.default_rng(0)
        nulls = generator.uniform(0, 1, 150).tolist()
        uncorrected = sum(1 for p in nulls if p < 0.05)
        corrected = benjamini_hochberg(nulls)["n_rejected"]
        assert uncorrected >= 5
        assert corrected <= 1


class TestPower:
    def test_mde_shrinks_with_more_blocks(self):
        assert minimum_detectable_effect(12, 2.0) > minimum_detectable_effect(48, 2.0)

    def test_power_rises_with_effect_size(self):
        assert achieved_power(1.0, 12, 2.0) < achieved_power(4.0, 12, 2.0)

    def test_required_blocks_answers_how_many_would_be_needed(self):
        needed = required_blocks(1.0, 2.0)
        assert needed > 12
        assert minimum_detectable_effect(needed, 2.0) <= 1.05

    def test_a_null_at_low_power_is_distinguishable_from_no_effect(self):
        """The distinction the report must make: absence of evidence vs evidence of absence."""
        values, groups, blocks = blocked_design(effect=-0.5, n_blocks=6, within_sd=3.0)
        comparison = paired_block_differences(values, groups, blocks, "female", "male")
        paired = paired_test(comparison.differences)
        mde = minimum_detectable_effect(comparison.n_blocks, paired["sd_difference"])
        assert paired["t_p_value"] > 0.05, "underpowered: no significant result"
        assert mde > 1.0, "and the MDE proves a practical effect could have been missed"


class TestProportions:
    def test_fisher_exact_on_small_counts(self):
        result = two_proportion_test(2, 20, 12, 20)
        assert result["p_value"] < 0.01
        assert result["impact_ratio"] == pytest.approx(0.1667, abs=0.001)

    def test_equal_rates_give_no_difference(self):
        result = two_proportion_test(5, 20, 5, 20)
        assert result["difference"] == pytest.approx(0.0)
        assert result["impact_ratio"] == pytest.approx(1.0)

    def test_empty_group_is_handled(self):
        assert "note" in two_proportion_test(0, 0, 1, 10)
