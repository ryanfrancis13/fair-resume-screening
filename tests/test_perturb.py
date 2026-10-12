"""Perturbation: cue registry, generation, and content-preservation validity."""

from __future__ import annotations

import warnings

import pytest

from frs.perturb.cues import (
    PROTECTED,
    PROTECTED_CONFOUNDED,
    VALIDITY,
    VALIDITY_PROXY,
    available_families,
    build_registry,
    resolve_family_name,
)
from frs.perturb.generator import (
    baseline_variant,
    detect_slots,
    generate_variants,
    render_template,
    template_slots,
)
from frs.perturb.names import (
    LITERATURE_INFORMED,
    PEER_REVIEWED,
    available_name_lists,
    balanced_sample,
    load_name_list,
)
from frs.perturb.transforms import (
    capitalised_fingerprint,
    numeric_fingerprint,
    shift_years,
    transform_vocabulary,
    writing_register,
)
from frs.perturb.validator import summarise_validation, validate_variant
from frs.utils.seeds import rng


class TestNameLists:
    def test_bertrand_mullainathan_is_balanced_and_peer_reviewed(self):
        name_list = load_name_list("bertrand_mullainathan_2004")
        assert name_list.provenance == PEER_REVIEWED
        cells = name_list.balance_report()
        assert set(cells) == {
            "female|Black", "female|White", "male|Black", "male|White"
        }
        assert len(set(cells.values())) == 1, "every cell must have the same size"

    def test_non_peer_reviewed_list_warns(self):
        """Provenance is enforced, not merely documented."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            name_list = load_name_list("extended_five_group")
        assert name_list.provenance == LITERATURE_INFORMED
        assert any("peer-reviewed" in str(w.message) for w in caught)

    def test_every_list_carries_a_citation(self):
        for list_id in available_name_lists():
            assert len(load_name_list(list_id).citation) > 40

    def test_balanced_sample_keeps_cells_equal(self):
        name_list = load_name_list("bertrand_mullainathan_2004")
        picked = balanced_sample(name_list, 3, rng(0, "names"))
        counts: dict = {}
        for entry in picked:
            counts[f"{entry.gender}|{entry.race_ethnicity}"] = (
                counts.get(f"{entry.gender}|{entry.race_ethnicity}", 0) + 1
            )
        assert set(counts.values()) == {3}

    def test_derived_contacts_match_the_name(self):
        entry = load_name_list("bertrand_mullainathan_2004").entries[0]
        assert entry.first.lower() in entry.email
        assert entry.last.lower().replace("'", "") in entry.linkedin

    def test_csv_round_trip(self, tmp_path):
        from frs.perturb.names import export_name_list_csv, load_name_list_csv

        original = load_name_list("bertrand_mullainathan_2004")
        path = export_name_list_csv(original, tmp_path / "names.csv")
        reloaded = load_name_list_csv(path, list_id="custom")
        assert len(reloaded.entries) == len(original.entries)
        assert reloaded.entries[0].full == original.entries[0].full


class TestCueRegistry:
    def test_covers_names_and_eight_further_cue_types(self):
        """The stated research gap is cues beyond the candidate's name."""
        families = available_families()
        assert "name" in families
        assert len([f for f in families if f != "name"]) >= 8

    def test_every_family_declares_its_type(self):
        registry = build_registry()
        for family in registry.values():
            assert family.cue_type in (
                PROTECTED, PROTECTED_CONFOUNDED, VALIDITY, VALIDITY_PROXY
            )

    def test_confounded_and_proxy_families_declare_a_caveat(self):
        """A confound must be machine-readable so the report can print it."""
        registry = build_registry()
        for family in registry.values():
            if family.cue_type in (PROTECTED_CONFOUNDED, VALIDITY_PROXY):
                assert family.caveat, family.family
            if family.cue_type == PROTECTED_CONFOUNDED:
                assert family.confounded_with, family.family

    def test_asymmetric_families_are_flagged(self):
        registry = build_registry()
        assert registry["affiliation_marker"].asymmetric is True

    def test_every_family_has_exactly_one_reference_level(self):
        registry = build_registry()
        for name, family in registry.items():
            if name == "name":
                continue  # the reference *cell* is a set of names, not one level
            references = [lvl for lvl in family.levels if lvl.is_reference]
            assert len(references) == 1, name

    def test_every_family_cites_a_source(self):
        registry = build_registry()
        for name, family in registry.items():
            assert len(family.citation) > 30, name

    def test_validity_probe_is_not_a_protected_attribute(self):
        registry = build_registry()
        assert registry["education_prestige"].cue_type == VALIDITY

    def test_legacy_aliases_resolve(self):
        assert resolve_family_name("gender_marker") == "pronoun"
        assert resolve_family_name("age_dates") == "age_marker"

    def test_unknown_family_is_rejected(self):
        with pytest.raises(KeyError):
            build_registry(families=["not_a_family"])


class TestTemplateRendering:
    def test_missing_slot_value_is_an_error(self):
        with pytest.raises(KeyError):
            render_template("Hello {{NAME}}", {})

    def test_empty_optional_slot_drops_its_line(self):
        template = "NAME\nFunding: {{FUNDING_NOTE}}\nEnd"
        assert "Funding" not in render_template(template, {"FUNDING_NOTE": ""})
        assert "Funding" in render_template(template, {"FUNDING_NOTE": "A grant."})

    def test_slots_are_discoverable(self):
        assert template_slots("{{A}} and {{B}} and {{A}}") == ["A", "B"]


class TestSlotDetection:
    """The fallback path for corpora that ship no template."""

    RAW = """Jordan Avery
Boston, MA | jordan.avery@example.com | +1 (617) 555-0142 | linkedin.com/in/jordan-avery

SUMMARY
Backend engineer with 7 years of experience.

EDUCATION
B.S. Computer Science, 2019
"""

    def test_finds_the_contact_block(self):
        found = detect_slots(self.RAW)
        assert found["NAME"] == "Jordan Avery"
        assert found["EMAIL"] == "jordan.avery@example.com"
        assert found["LOCATION"] == "Boston, MA"
        assert found["GRAD_YEAR"] == "2019"

    def test_absent_slots_are_simply_absent(self):
        found = detect_slots("SUMMARY\nSome text with no contact details at all.")
        assert "EMAIL" not in found


class TestTransforms:
    def test_register_rewrite_preserves_every_number(self):
        text = ("EXPERIENCE\n- Reduced latency by 42% across 34 services.\n"
                "- Led migration of 12 teams.\n")
        for level in ("hedged", "assertive"):
            rewritten = writing_register(text, level)
            assert numeric_fingerprint(rewritten) == numeric_fingerprint(text)
            assert rewritten != text

    def test_plain_register_is_the_identity(self):
        text = "- Led the work.\n"
        assert writing_register(text, "plain") == text

    def test_unknown_register_level_is_rejected(self):
        with pytest.raises(KeyError):
            writing_register("x", "shouty")

    def test_date_shift_preserves_durations(self):
        assert shift_years("Role 2019 - 2023", -10) == "Role 2009 - 2013"

    def test_line_initial_verbs_are_excluded_from_the_entity_fingerprint(self):
        """A register rewrite replaces a bullet's leading verb by design."""
        text = "- Led migration to Kubernetes at Northwind\n"
        assert "Led" in capitalised_fingerprint(text)
        assert "Led" not in capitalised_fingerprint(text, skip_line_initial=True)
        assert "Kubernetes" in capitalised_fingerprint(text, skip_line_initial=True)

    def test_vocabulary_is_derived_from_the_rules(self):
        vocabulary = transform_vocabulary()
        assert "helped" in vocabulary
        assert "spearheaded" in vocabulary


class TestValidator:
    BASE = """Jordan Avery
Boston, MA | jordan.avery@example.com

EXPERIENCE
- Reduced p99 latency by 42% across 34 services at Northwind.
"""

    def test_declared_name_swap_passes(self):
        from frs.schemas import Edit

        variant = self.BASE.replace("Jordan Avery", "Emily Murphy")
        edits = [Edit(slot="NAME", cue_family="name", before="Jordan Avery", after="Emily Murphy")]
        result = validate_variant(self.BASE, variant, edits)
        assert result["passed"], result["failures"]

    def test_declared_phone_swap_passes(self):
        """The numeric check must use the same extractor as the allow-list.

        A phone number tokenises as one word but as several numbers; comparing a
        numeric fingerprint against a word-token allow-list rejects a correctly
        declared edit.
        """
        from frs.schemas import Edit

        base = "Contact: +1 (617) 555-0142\n- Cut latency 42%.\n"
        variant = "Contact: +1 (256) 555-0198\n- Cut latency 42%.\n"
        edits = [Edit(slot="PHONE", cue_family="location_marker",
                      before="+1 (617) 555-0142", after="+1 (256) 555-0198")]
        result = validate_variant(base, variant, edits)
        assert result["passed"], result["failures"]

    def test_undeclared_skill_removal_fails(self):
        from frs.schemas import Edit

        variant = self.BASE.replace("Jordan Avery", "Emily Murphy").replace(
            "- Reduced p99 latency by 42% across 34 services at Northwind.\n", ""
        )
        edits = [Edit(slot="NAME", cue_family="name", before="Jordan Avery", after="Emily Murphy")]
        result = validate_variant(self.BASE, variant, edits)
        assert not result["passed"]
        assert any("entity" in f or "token" in f or "numeric" in f for f in result["failures"])

    def test_undeclared_metric_change_fails(self):
        from frs.schemas import Edit

        variant = self.BASE.replace("42%", "84%")
        result = validate_variant(self.BASE, variant, [Edit("NAME", "name", "a", "b")])
        assert not result["passed"]

    def test_transform_may_gain_a_verb_but_not_lose_an_entity(self):
        base = "- Led migration to Kubernetes at Northwind.\n"
        good = writing_register(base, "hedged")
        assert validate_variant(base, good, [], is_transform=True)["passed"]
        bad = "- Helped lead migration.\n"
        assert not validate_variant(base, bad, [], is_transform=True)["passed"]


class TestGeneration:
    def test_baseline_is_unperturbed_and_flagged(self, corpus):
        resumes, _, _ = corpus
        base = baseline_variant(resumes[0])
        assert base.is_baseline
        assert base.signals == {}
        assert base.cue_families == []

    def test_single_design_varies_one_family_per_variant(self, corpus):
        resumes, _, _ = corpus
        registry = build_registry(families=["name", "pronoun", "location_marker"])
        variants, _ = generate_variants(resumes[0], registry, design="single",
                                        rng=rng(0, "gen"))
        for variant in variants:
            assert len(variant.cue_families) <= 1

    def test_all_generated_variants_preserve_content(self, corpus):
        """The headline validity claim, asserted on the real corpus."""
        resumes, _, _ = corpus
        registry = build_registry()
        produced = []
        for resume in resumes:
            variants, _ = generate_variants(
                resume, registry, design="single", rng=rng(1, resume.resume_id)
            )
            produced.extend(variants)
        summary = summarise_validation(produced)
        assert summary["pass_rate"] == 1.0, summary["top_failure_modes"]

    def test_missing_slots_are_recorded_not_faked(self, corpus):
        """A resume without a slot must yield a skip, never a mis-placed edit."""
        resumes, _, _ = corpus
        sparse = next(r for r in resumes if "FUNDING_NOTE" not in template_slots(r.template or ""))
        registry = build_registry(families=["socioeconomic_marker"])
        variants, skips = generate_variants(sparse, registry, design="single", rng=rng(0, "s"))
        assert skips and all(s["reason"] == "missing_slots" for s in skips)
        assert all(v.is_baseline for v in variants)

    def test_reference_level_duplicates_are_dropped(self, corpus):
        resumes, _, _ = corpus
        registry = build_registry(families=["pronoun"])
        variants, _ = generate_variants(resumes[0], registry, design="single", rng=rng(0, "p"))
        texts = [v.text for v in variants]
        assert len(texts) == len(set(texts)), "no two variants may share identical text"

    def test_factorial_design_crosses_name_and_pronoun(self, corpus):
        resumes, _, _ = corpus
        registry = build_registry(families=["name", "pronoun"])
        variants, _ = generate_variants(
            resumes[0], registry, design="factorial_name_x_gender",
            levels_per_family=4, rng=rng(0, "f"),
        )
        crossed = [v for v in variants if len(v.cue_families) == 2]
        assert crossed, "factorial design must produce crossed variants"

    def test_variant_ids_are_stable_across_runs(self, corpus):
        resumes, _, _ = corpus
        registry = build_registry(families=["name"])
        first, _ = generate_variants(resumes[0], registry, levels_per_family=4, rng=rng(0, "a"))
        second, _ = generate_variants(resumes[0], registry, levels_per_family=4, rng=rng(0, "a"))
        assert [v.variant_id for v in first] == [v.variant_id for v in second]

    def test_runaway_factorial_is_refused(self, corpus):
        resumes, _, _ = corpus
        registry = build_registry()
        with pytest.raises(ValueError, match="cap"):
            generate_variants(resumes[0], registry, design="full_factorial", rng=rng(0, "x"))

    def test_signals_are_recorded_for_every_variant(self, corpus):
        resumes, _, _ = corpus
        registry = build_registry(families=["name"])
        variants, _ = generate_variants(resumes[0], registry, levels_per_family=4, rng=rng(0, "s"))
        for variant in variants:
            if variant.is_baseline:
                continue
            assert set(variant.signals) >= {"gender", "race_ethnicity"}


class TestTransformVocabularyCompleteness:
    """The declared vocabulary must cover what the rules actually do.

    An incomplete list makes the perturbation validator reject valid register
    variants, so this test applies every transform to every resume in the real
    corpus and asserts that no token changes outside the declaration. It is the
    mechanism that keeps the list and the rules in sync.
    """

    def test_transform_vocabulary_is_complete(self, corpus):
        from frs.perturb.transforms import transform_vocabulary, writing_register
        from frs.utils.text import tokens

        allowed = set(transform_vocabulary())
        resumes, _, _ = corpus
        stray: set = set()
        for resume in resumes:
            base = set(tokens(resume.text))
            for level in ("hedged", "assertive"):
                variant = set(tokens(writing_register(resume.text, level)))
                for token in (variant - base) | (base - variant):
                    parts = {token, *token.split("-"), *token.split(".")}
                    if not (parts & allowed):
                        stray.add(token)
        assert not stray, f"register transforms touch undeclared tokens: {sorted(stray)}"
