"""Text normalisation, hashing and seeding."""

from __future__ import annotations

from frs.utils.hashing import cache_key, hash_obj, hash_text
from frs.utils.seeds import derive_seed, py_rng, rng
from frs.utils.text import (
    char_diff_spans,
    clean_text,
    count_lexicon_hits,
    fuzzy_contains,
    jaccard,
    normalized_edit_ratio,
    phrase_set,
    tokens,
    truncate,
    word_count,
)


class TestCleanText:
    def test_is_idempotent(self):
        raw = "Name  \r\n\r\n\r\n  •  Did   a thing here\n\n\n\nEND"
        once = clean_text(raw)
        assert clean_text(once) == once, "cleaning twice must equal cleaning once"

    def test_preserves_numbers_and_entities(self):
        raw = "- Reduced p99 latency by 42% and cut cost 27% across 34 services"
        cleaned = clean_text(raw)
        for token in ("42%", "27%", "34", "p99"):
            assert token in cleaned

    def test_normalises_bullets_and_quotes(self):
        cleaned = clean_text("• first\n‣ second\n– third\nit’s")
        assert cleaned.count("- ") == 3
        assert "it's" in cleaned

    def test_strips_scraping_artifacts(self):
        assert "read more" not in clean_text("Did a thing. Read more").lower()


class TestTokens:
    def test_keeps_technical_tokens_intact(self):
        found = tokens("C++ and Node.js with CI-CD on K8s")
        assert "c++" in found
        assert "node.js" in found
        assert "ci-cd" in found

    def test_phrase_set_includes_multiword_skills(self):
        assert "apache spark" in phrase_set("We use Apache Spark daily", max_n=3)

    def test_word_count_ignores_punctuation(self):
        assert word_count("a, b. c!") == 3


class TestFuzzyContains:
    def test_tolerates_rewrapping(self):
        source = "Reduced p99 checkout latency from 840 ms to 190 ms by\nresharding PostgreSQL"
        assert fuzzy_contains(source, "Reduced p99 checkout latency from 840 ms to 190 ms")

    def test_rejects_absent_text(self):
        assert not fuzzy_contains("Backend engineer with 7 years", "Led a team of 40 designers")

    def test_empty_needle_is_not_contained(self):
        assert not fuzzy_contains("anything", "")


class TestDiff:
    def test_diff_spans_locate_only_the_change(self):
        a = "Jordan Avery\nBoston, MA"
        b = "Emily Murphy\nBoston, MA"
        spans = char_diff_spans(a, b)
        joined = " ".join(f"{s[2]}->{s[3]}" for s in spans)
        assert "Boston" not in joined

    def test_edit_ratio_bounds(self):
        assert normalized_edit_ratio("abc", "abc") == 0.0
        assert 0.0 < normalized_edit_ratio("abcdef", "abcxyz") <= 1.0


class TestMisc:
    def test_jaccard_extremes(self):
        assert jaccard([], []) == 1.0
        assert jaccard(["a"], ["b"]) == 0.0

    def test_truncate_cuts_on_line_boundary(self):
        text = "\n".join(f"line {i} with some padding text" for i in range(40))
        out = truncate(text, 200)
        assert len(out) <= 220
        assert "truncated" in out

    def test_lexicon_hits_respect_word_boundaries(self):
        assert count_lexicon_hits("She led it", ["she"]) == {"she": 1}
        assert count_lexicon_hits("Kubernetes-based work", ["work"]) == {"work": 1}
        # A term must not match inside a longer word.
        assert count_lexicon_hits("their recent work", ["her"]) == {}


class TestHashing:
    def test_hashes_are_stable_and_content_sensitive(self):
        assert hash_text("abc") == hash_text("abc")
        assert hash_text("abc") != hash_text("abd")

    def test_stable_json_ignores_key_order(self):
        assert hash_obj({"a": 1, "b": 2}) == hash_obj({"b": 2, "a": 1})

    def test_cache_key_separates_sample_index(self):
        base = dict(prompt="p", model="m", provider="mock", params={"temperature": 0.4})
        assert cache_key(**base, **{}) != cache_key(
            prompt="p", model="m", provider="mock",
            params={"temperature": 0.4, "sample_index": 1},
        )

    def test_cache_key_separates_provider_behaviour(self):
        """A local provider's configuration must change the key.

        Without this the offline bias oracle silently replays the null
        condition's responses for the injected condition.
        """
        base = dict(prompt="p", model="m", provider="mock", params={})
        assert cache_key(**base, behaviour="a") != cache_key(**base, behaviour="b")


class TestSeeds:
    def test_derivation_is_stable_and_label_specific(self):
        assert derive_seed(7, "x") == derive_seed(7, "x")
        assert derive_seed(7, "x") != derive_seed(7, "y")
        assert derive_seed(7, "x") != derive_seed(8, "x")

    def test_generators_are_reproducible(self):
        assert rng(1, "a").random() == rng(1, "a").random()
        assert py_rng(1, "a").random() == py_rng(1, "a").random()


class TestHonorificRedaction:
    """Honorifics must be matched by their own pattern, not by a word list.

    A bare ``ms`` in a gendered-noun list is a word-boundary match inside
    ``190 ms``, so a latency figure would be redacted as a gender cue and the
    resume's actual content would change. The honorific pattern requires a
    following capitalised token, which a unit never has.
    """

    def test_unit_is_not_treated_as_an_honorific(self):
        import re

        from frs.anonymize.lexicons import GENDERED_NOUNS, HONORIFIC_PATTERN

        assert "ms" not in GENDERED_NOUNS
        pattern = re.compile(HONORIFIC_PATTERN)
        assert pattern.search("latency dropped from 840 ms to 190 ms") is None
        assert pattern.search("interviewed by Ms. Walsh") is not None
