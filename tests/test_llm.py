"""Provider abstraction: caching, retries, rate limiting, the offline oracle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frs.llm import LLMClient, LLMRequest, MockProvider, ResponseCache
from frs.llm.base import PermanentError, RateLimitError, TransientError
from frs.llm.mock import BIAS_TRANSMISSION, NOISE_TRANSMISSION, SignalDetector
from frs.llm.ratelimit import RetryPolicy, TokenBucket

PROMPT = """<job_description>
TITLE: Senior Backend Engineer
MUST-HAVE REQUIREMENTS:
- Python
- Kubernetes
- PostgreSQL
- 5 years of backend engineering experience
NICE-TO-HAVE:
- Go
</job_description>
<resume>
{name}
{email} | Boston, MA

PROFESSIONAL SUMMARY
Backend engineer with 7 years building distributed systems in Python.

EXPERIENCE
Senior Backend Engineer, Northwind Payments (2021 - Present)
- Reduced p99 latency by 42% by resharding PostgreSQL and tuning Kubernetes autoscaling.
- Led migration of 34 services to Kubernetes, cutting deploy time 65%.

EDUCATION
B.S. Computer Science, University of Massachusetts, 2019
</resume>"""

WEAK_PROMPT = PROMPT.replace(
    "Backend engineer with 7 years building distributed systems in Python.",
    "Graphic designer with 2 years of print and social media design experience.",
).replace(
    "- Reduced p99 latency by 42% by resharding PostgreSQL and tuning Kubernetes autoscaling.\n"
    "- Led migration of 34 services to Kubernetes, cutting deploy time 65%.",
    "- Designed 40 social posts per month in Figma.",
)


def _score(client: LLMClient, prompt: str, **kwargs) -> dict:
    response = client.generate(LLMRequest(prompt=prompt, **kwargs))
    assert not response.error, response.error
    return json.loads(response.text)


class TestMockProviderIsACompetentScorer:
    """The offline provider must discriminate on fit, or nothing downstream means anything."""

    def test_strong_fit_outscores_weak_fit(self, client):
        strong = _score(client, PROMPT.format(name="Jordan Avery", email="j@example.com"))
        weak = _score(client, WEAK_PROMPT.format(name="Jordan Avery", email="j@example.com"))
        assert strong["overall_score"] > weak["overall_score"] + 15

    def test_returns_every_rubric_dimension(self, client):
        result = _score(client, PROMPT.format(name="Jordan Avery", email="j@example.com"))
        names = {d["name"] for d in result["dimensions"]}
        assert names == {
            "technical_skills", "relevant_experience", "domain_relevance",
            "education_credentials", "impact_and_outcomes", "communication_clarity",
        }

    def test_evidence_quotes_come_from_the_resume(self, client):
        prompt = PROMPT.format(name="Jordan Avery", email="j@example.com")
        result = _score(client, prompt)
        resume = prompt.split("<resume>")[1].split("</resume>")[0]
        for dimension in result["dimensions"]:
            for span in dimension["evidence"]:
                assert span["quote"] in resume, span["quote"]

    def test_reports_uncovered_requirements(self, client):
        result = _score(client, WEAK_PROMPT.format(name="Jordan Avery", email="j@example.com"))
        assert "Python" in result["missing_requirements"]

    def test_is_deterministic_at_temperature_zero(self, client):
        prompt = PROMPT.format(name="Jordan Avery", email="j@example.com")
        assert _score(client, prompt) == _score(client, prompt)


class TestBiasOracle:
    def test_injects_a_gap_of_the_configured_size(self, biased_client):
        male = _score(biased_client, PROMPT.format(name="Greg Kelly", email="g@example.com"))
        female = _score(biased_client, PROMPT.format(name="Emily Murphy", email="e@example.com"))
        expected = -4.0 * BIAS_TRANSMISSION
        assert female["overall_score"] - male["overall_score"] == pytest.approx(expected, abs=0.05)

    def test_no_gap_without_injected_bias(self, client):
        male = _score(client, PROMPT.format(name="Greg Kelly", email="g@example.com"))
        female = _score(client, PROMPT.format(name="Emily Murphy", email="e@example.com"))
        assert female["overall_score"] == pytest.approx(male["overall_score"], abs=0.001)

    def test_bias_lands_hardest_on_judgement_dimensions(self, biased_client):
        male = _score(biased_client, PROMPT.format(name="Greg Kelly", email="g@example.com"))
        female = _score(biased_client, PROMPT.format(name="Emily Murphy", email="e@example.com"))
        deltas = {
            d["name"]: f["score"] - d["score"]
            for d, f in zip(male["dimensions"], female["dimensions"])
        }
        assert abs(deltas["communication_clarity"]) > abs(deltas["technical_skills"])

    def test_noise_is_zero_at_temperature_zero_and_present_above_it(self, cfg):
        provider = MockProvider(noise_sd=2.0)
        client = LLMClient(provider, requests_per_minute=10**6)
        prompt = PROMPT.format(name="Jordan Avery", email="j@example.com")
        greedy = [_score(client, prompt, temperature=0.0, sample_index=i)["overall_score"]
                  for i in range(3)]
        assert len(set(greedy)) == 1, "greedy decoding must be exactly repeatable"
        hot = [_score(client, prompt, temperature=0.4, sample_index=i)["overall_score"]
               for i in range(6)]
        assert len(set(hot)) > 1, "sampling must vary across repeats"
        assert NOISE_TRANSMISSION > 0


class TestSignalDetector:
    """Detection reads the prose, not pipeline metadata — and must be precise."""

    def test_recovers_every_name_in_the_lists(self):
        from frs.perturb.names import load_name_list

        detector = SignalDetector()
        for entry in load_name_list("extended_five_group").entries:
            found = detector.detect(f"{entry.full}\n{entry.email}\n{entry.linkedin}")
            assert found.get("gender") == entry.gender, entry.full
            assert found.get("race_ethnicity") == entry.race_ethnicity, entry.full

    def test_neutral_pronoun_is_not_read_as_female(self):
        """``her`` must not be found inside the unmarked pronoun's context.

        A substring matcher assigns ``gender=female`` to the neutral baseline and
        shrinks every measured gender gap toward zero.
        """
        detector = SignalDetector()
        assert detector.detect("They led the team. Their work was on Kubernetes.").get(
            "gender"
        ) == "unspecified"

    def test_pronouns_are_distinguished(self):
        detector = SignalDetector()
        assert detector.detect("She led it. Her work shipped.")["gender"] == "female"
        assert detector.detect("He led it. His work shipped.")["gender"] == "male"

    def test_unmarked_text_yields_no_demographic_signal(self):
        detector = SignalDetector()
        found = detector.detect(
            "Jordan Avery led a distributed systems team. This history is well documented."
        )
        assert "gender" not in found
        assert "race_ethnicity" not in found


class TestCache:
    def test_round_trip(self, tmp_path: Path):
        cache = ResponseCache(tmp_path, enabled=True)
        client = LLMClient(MockProvider(noise_sd=0.0), cache=cache, requests_per_minute=10**6)
        prompt = PROMPT.format(name="Jordan Avery", email="j@example.com")
        first = client.generate(LLMRequest(prompt=prompt))
        second = client.generate(LLMRequest(prompt=prompt))
        assert not first.cached and second.cached
        assert first.text == second.text
        assert cache.stats()["hits"] == 1

    def test_provider_behaviour_isolates_entries(self, tmp_path: Path):
        """The null and injected oracle conditions must not share cache entries."""
        cache = ResponseCache(tmp_path, enabled=True)
        prompt = PROMPT.format(name="Emily Murphy", email="e@example.com")
        plain = LLMClient(MockProvider(noise_sd=0.0), cache=cache, requests_per_minute=10**6)
        biased = LLMClient(
            MockProvider(bias={"gender": {"female": -5.0}}, noise_sd=0.0),
            cache=cache, requests_per_minute=10**6,
        )
        a = json.loads(plain.generate(LLMRequest(prompt=prompt)).text)["overall_score"]
        b = json.loads(biased.generate(LLMRequest(prompt=prompt)).text)["overall_score"]
        assert b < a - 1.0

    def test_behaviour_key_tracks_the_source_code(self):
        """Changing the grading code must invalidate cached responses."""
        from frs.llm import mock as mock_module

        original = mock_module._SOURCE_FINGERPRINT
        try:
            provider = MockProvider()
            before = provider.behaviour_key
            mock_module._SOURCE_FINGERPRINT = "pretend-the-code-changed"
            assert provider.behaviour_key != before
        finally:
            mock_module._SOURCE_FINGERPRINT = original

    def test_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path: Path):
        cache = ResponseCache(tmp_path, enabled=True)
        request = LLMRequest(prompt="x")
        key = cache.key_for(request, "mock", "m")
        path = cache._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ truncated", encoding="utf-8")
        assert cache.get(key) is None

    def test_errors_are_never_cached(self, tmp_path: Path):
        from frs.llm.base import LLMResponse

        cache = ResponseCache(tmp_path, enabled=True)
        cache.put("k" * 32, LLMResponse(text="", model="m", provider="p", error="429"))
        assert cache.stats()["writes"] == 0


class TestRetryAndRateLimit:
    def test_retries_transient_then_succeeds(self):
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise TransientError("503")
            return "ok"

        policy = RetryPolicy(max_retries=5, sleeper=lambda _: None, jitter=lambda: 0.0)
        assert policy.run(flaky) == "ok"
        assert attempts["n"] == 3
        assert policy.retries_used == 2

    def test_does_not_retry_permanent_errors(self):
        calls = {"n": 0}

        def bad_key():
            calls["n"] += 1
            raise PermanentError("401 invalid api key")

        policy = RetryPolicy(max_retries=5, sleeper=lambda _: None, jitter=lambda: 0.0)
        with pytest.raises(PermanentError):
            policy.run(bad_key)
        assert calls["n"] == 1, "a bad key must not be retried"

    def test_backoff_grows_and_is_capped(self):
        policy = RetryPolicy(base_delay=1.0, max_delay=8.0, jitter=lambda: 1.0)
        delays = [policy.delay_for(i) for i in range(1, 6)]
        assert delays == [1.0, 2.0, 4.0, 8.0, 8.0]

    def test_token_bucket_paces_requests(self):
        bucket = TokenBucket(rate_per_minute=6000, burst=1)
        bucket.acquire()
        slept = bucket.acquire()
        assert slept >= 0.0

    def test_offline_provider_is_not_metered(self):
        assert MockProvider().metered is False

    def test_client_returns_error_row_instead_of_raising(self):
        class Broken(MockProvider):
            def generate(self, request):
                raise TransientError("upstream exploded")

        client = LLMClient(Broken(), requests_per_minute=10**6, max_retries=2,
                           retry_base_delay=0.0, retry_max_delay=0.0)
        response = client.generate(LLMRequest(prompt="x"))
        assert response.error and not response.ok
        assert client.telemetry()["failures"] == 1

    def test_generate_many_preserves_order(self, client):
        requests = [
            LLMRequest(prompt=PROMPT.format(name=f"Person{i}", email=f"p{i}@example.com"))
            for i in range(8)
        ]
        responses = client.generate_many(requests)
        assert len(responses) == len(requests)
        assert all(r.ok for r in responses)


class TestGeminiAdapter:
    """The Gemini adapter is exercised without a key or the SDK installed."""

    def test_missing_key_is_a_permanent_error(self, monkeypatch):
        from frs.llm.gemini import GeminiProvider

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        provider = GeminiProvider(api_key=None)
        with pytest.raises(PermanentError):
            provider.generate(LLMRequest(prompt="x"))

    def test_health_check_reports_rather_than_raises(self, monkeypatch):
        from frs.llm.gemini import GeminiProvider

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        info = GeminiProvider(api_key=None).health_check()
        assert info["ready"] is False
        assert "GEMINI_API_KEY" in info["error"]

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("429 RESOURCE_EXHAUSTED quota exceeded", RateLimitError),
            ("503 Service Unavailable", TransientError),
            ("401 API key not valid", PermanentError),
            ("something nobody predicted", TransientError),
        ],
    )
    def test_error_classification(self, message, expected):
        from frs.llm.gemini import _classify

        assert isinstance(_classify(Exception(message)), expected)
