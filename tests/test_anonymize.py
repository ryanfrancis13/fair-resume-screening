"""Redaction and, more importantly, the measurement of what redaction misses."""

from __future__ import annotations

import pytest

from frs.anonymize import LEVELS, compare_levels, leakage_report, redact, redact_resume
from frs.anonymize.lexicons import protected_reference_lexicon, rationale_lexicons

RESUME = """Emily Murphy
Boston, MA | emily.murphy@example.com | +1 (617) 555-0142 | linkedin.com/in/emily-murphy

PROFESSIONAL SUMMARY
Backend engineer with 7 years building distributed systems.
She led the platform reliability workstream, and her recent work centred on Kubernetes.

EXPERIENCE
Senior Backend Engineer, Northwind Payments (2021 - Present)
- Reduced p99 latency from 840 ms to 190 ms by resharding PostgreSQL.

EDUCATION
B.S. Computer Science, University of Massachusetts, Amherst, 2019

ADDITIONAL INFORMATION
Affiliations: Member, Women Who Code local chapter (2 years)
Spoken languages: English (fluent), Yoruba (native)
Interests: Competitive sailing, classical piano
"""


class TestIdentityRedaction:
    def test_removes_the_contact_block(self):
        result = redact(RESUME, "identity", known_name="Emily Murphy")
        for secret in ("Emily", "Murphy", "emily.murphy@example.com",
                       "linkedin.com/in/emily-murphy", "555-0142"):
            assert secret not in result.text, secret

    def test_email_is_replaced_before_the_name(self):
        """Otherwise the address becomes '[NAME].[NAME]@example.com'.

        That both leaks the fact that the local part was derived from the name and
        leaves a malformed address in the document.
        """
        result = redact(RESUME, "identity", known_name="Emily Murphy")
        assert "[EMAIL]" in result.text
        assert "[NAME].[NAME]@" not in result.text

    def test_body_place_names_survive(self):
        """Location redaction is confined to the contact header."""
        text = "Jordan Avery\nBoston, MA | j@example.com\n\n- Migrated the Frankfurt, Germany region.\n"
        result = redact(text, "identity", known_name="Jordan Avery")
        assert "Frankfurt, Germany" in result.text

    def test_latency_figures_are_not_mistaken_for_honorifics(self):
        result = redact(RESUME, "identity_plus_proxy", known_name="Emily Murphy")
        assert "840" in result.text and "190" in result.text


class TestProxyRedaction:
    def test_removes_pronouns_and_marked_affiliations(self):
        result = redact(RESUME, "identity_plus_proxy", known_name="Emily Murphy")
        assert "Women Who Code" not in result.text
        assert " she " not in result.text.lower()
        assert " her " not in result.text.lower()

    def test_drops_a_section_whose_content_was_removed(self):
        """A dangling header advertises that something was taken out."""
        result = redact(RESUME, "identity_plus_proxy", known_name="Emily Murphy")
        assert "ADDITIONAL INFORMATION" not in result.text

    def test_aggressive_level_removes_institution_and_years(self):
        result = redact(RESUME, "aggressive", known_name="Emily Murphy")
        assert "University of Massachusetts" not in result.text
        assert "2019" not in result.text

    def test_unknown_level_is_rejected(self):
        with pytest.raises(ValueError):
            redact(RESUME, "maximum-effort")


class TestLeakageMeasurement:
    """The project's central empirical claim, measured directly."""

    def test_name_only_redaction_leaves_demographic_signal(self):
        result = redact(RESUME, "identity", known_name="Emily Murphy")
        residual = result.residual
        assert residual["residual_cue_hits"] > 0, (
            "the marked affiliation, spoken language and class-coded hobbies all survive "
            "a name-level redaction"
        )
        assert residual["n_detected_attributes"] > 0
        assert not residual["clean"]

    def test_stronger_redaction_monotonically_reduces_leakage(self):
        hits = []
        for level in LEVELS:
            result = redact(RESUME, level, known_name="Emily Murphy")
            hits.append(result.residual["residual_cue_hits"])
        assert hits == sorted(hits, reverse=True), hits
        assert hits[-1] < hits[0]

    def test_neutral_pronoun_is_not_counted_as_leakage(self):
        report = leakage_report("They led the team. Their work shipped.")
        assert "gender" not in report["detected_attributes"]

    def test_compare_levels_produces_the_report_table(self, corpus):
        resumes, _, _ = corpus
        rows = compare_levels(resumes[0])
        assert [r["level"] for r in rows] == ["none", *LEVELS]
        assert rows[0]["n_detected_attributes"] >= rows[-1]["n_detected_attributes"]

    def test_resume_helper_uses_the_declared_name_slot(self, corpus):
        """Regex name detection can fail; a declared slot cannot."""
        resumes, _, _ = corpus
        resume = resumes[0]
        result = redact_resume(resume, "identity")
        assert resume.slots["NAME"] not in result.text


class TestLexicons:
    def test_rationale_lexicons_cover_the_audited_contrasts(self):
        lexicons = rationale_lexicons()
        assert {"agentic", "communal", "hedge", "booster", "effort", "ability"} <= set(lexicons)
        for name, terms in lexicons.items():
            assert len(terms) >= 6, name

    def test_protected_reference_lexicon_is_deduplicated_and_lowercase(self):
        terms = protected_reference_lexicon()
        assert terms == sorted(set(terms))
        assert all(term == term.lower() for term in terms)
