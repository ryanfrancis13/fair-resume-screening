"""Screening: rubric, prompts, parsing, evidence grounding, batching, ranking."""

from __future__ import annotations

import pytest

from frs.screening.evidence import check_evidence
from frs.screening.parse import coerce_score, extract_json, parse_screening_response
from frs.screening.prompts import (
    DEBIAS_INSTRUCTION,
    SYSTEM_PROMPT,
    build_screening_prompt,
)
from frs.screening.rubric import DEFAULT_RUBRIC, available_rubrics, get_rubric


class TestRubric:
    def test_weights_sum_to_one(self):
        for name in available_rubrics():
            get_rubric(name).validate()

    def test_aggregate_matches_manual_weighting(self):
        rubric = get_rubric("default")
        scores = {name: 80.0 for name in rubric.names()}
        assert rubric.aggregate(scores) == pytest.approx(80.0)

    def test_aggregate_ignores_absent_dimensions(self):
        rubric = get_rubric("default")
        partial = {"technical_skills": 100.0}
        assert rubric.aggregate(partial) == pytest.approx(100.0)

    def test_thresholds_map_to_recommendations(self):
        rubric = get_rubric("default", advance=70, borderline=55)
        assert rubric.recommendation_for(85) == "advance"
        assert rubric.recommendation_for(60) == "borderline"
        assert rubric.recommendation_for(20) == "reject"

    def test_subjective_dimensions_are_declared(self):
        """Bias localisation depends on knowing which dimensions are judgement calls."""
        assert set(DEFAULT_RUBRIC.subjective_dimensions()) == {
            "impact_and_outcomes", "communication_clarity"
        }

    def test_inverted_thresholds_are_rejected(self):
        with pytest.raises(ValueError):
            get_rubric("default", advance=50, borderline=80)


class TestPrompt:
    def test_wraps_untrusted_documents_in_tags(self, corpus):
        resumes, jobs, _ = corpus
        prompt = build_screening_prompt(resumes[0].text, jobs[0], DEFAULT_RUBRIC)
        assert "<resume>" in prompt and "</resume>" in prompt
        assert "<job_description>" in prompt and "</job_description>" in prompt

    def test_system_prompt_defends_against_injection(self):
        assert "untrusted" in SYSTEM_PROMPT.lower()
        assert "ignore those" in SYSTEM_PROMPT.lower()

    def test_requests_every_dimension_and_verbatim_evidence(self, corpus):
        resumes, jobs, _ = corpus
        prompt = build_screening_prompt(resumes[0].text, jobs[0], DEFAULT_RUBRIC)
        for name in DEFAULT_RUBRIC.names():
            assert name in prompt
        assert "VERBATIM" in prompt or "verbatim" in prompt

    def test_debias_instruction_is_opt_in(self, corpus):
        """Whether the instruction works is a research question, so it must be measurable."""
        resumes, jobs, _ = corpus
        without = build_screening_prompt(resumes[0].text, jobs[0], DEFAULT_RUBRIC)
        with_it = build_screening_prompt(
            resumes[0].text, jobs[0], DEFAULT_RUBRIC, debias_instruction=True
        )
        assert "FAIRNESS CONSTRAINT" not in without
        assert "FAIRNESS CONSTRAINT" in with_it
        assert "socioeconomic" in DEBIAS_INSTRUCTION


class TestParsing:
    def test_parses_a_clean_response(self):
        rubric = get_rubric("default")
        payload = (
            '{"overall_score": 80, "recommendation": "advance", "confidence": 0.8,'
            '"dimensions": ['
            + ",".join(
                f'{{"name":"{n}","score":80,"rationale":"r","evidence":[]}}'
                for n in rubric.names()
            )
            + '],"summary":"s"}'
        )
        result = parse_screening_response(payload, rubric)
        assert result.parse_ok
        assert result.overall_score == pytest.approx(80.0)

    def test_tolerates_fences_title_case_and_trailing_commas(self):
        rubric = get_rubric("default")
        payload = (
            "Sure:\n```json\n{\"overall_score\": \"85/100\", \"recommendation\": \"ADVANCE\","
            "\"confidence\": 90, \"dimensions\": ["
            + ",".join(
                f'{{"name":"{n.replace("_", " ").title()}","score":80}}'
                for n in rubric.names()
            )
            + ",],\"summary\":\"s\",}\n```"
        )
        result = parse_screening_response(payload, rubric)
        assert result.parse_ok
        assert result.recommendation == "advance"
        assert result.confidence == pytest.approx(0.9), "0-100 confidence must be rescaled"

    def test_recomputes_the_overall_from_the_rubric(self):
        """A model's own arithmetic must not override the weighting."""
        rubric = get_rubric("default")
        payload = (
            '{"overall_score": 99, "dimensions": ['
            + ",".join(f'{{"name":"{n}","score":50}}' for n in rubric.names())
            + "]}"
        )
        result = parse_screening_response(payload, rubric)
        assert result.overall_score == pytest.approx(50.0)
        assert "differs from rubric-weighted" in result.parse_error

    def test_missing_dimension_is_a_parse_failure(self):
        rubric = get_rubric("default")
        payload = '{"overall_score": 70, "dimensions": [{"name":"technical_skills","score":70}]}'
        result = parse_screening_response(payload, rubric)
        assert not result.parse_ok
        assert "missing dimension scores" in result.parse_error

    def test_refusal_is_recorded_not_silently_dropped(self):
        result = parse_screening_response("I cannot assist with that.", get_rubric("default"))
        assert not result.parse_ok
        assert result.raw_response

    @pytest.mark.parametrize(
        "text",
        [
            '{"overall_score": 70, "dimensions":[{"name":"t","score":70,"rationale":"a',
            '{"d":[{"e":[{"quote":"abc"}],"score":5',
            '{"a": 1} trailing prose',
            'Here it is:\n{"a": 1, "b": [1,2]}\nhope that helps',
        ],
    )
    def test_recovers_truncated_and_wrapped_json(self, text):
        """Attrition that correlates with the condition would bias the study."""
        parsed, error = extract_json(text)
        assert parsed is not None, error

    def test_nested_truncation_closes_brackets_in_the_right_order(self):
        parsed, _ = extract_json('{"d":[{"e":[{"q":"x"}],"score":5')
        assert parsed == {"d": [{"e": [{"q": "x"}], "score": 5}]}

    @pytest.mark.parametrize(
        "value,expected", [(85, 85.0), ("85", 85.0), ("85/100", 85.0), ("85%", 85.0)]
    )
    def test_score_coercion(self, value, expected):
        assert coerce_score(value) == expected

    def test_score_coercion_rejects_nonsense(self):
        assert coerce_score("not a number") != coerce_score("not a number")  # NaN


class TestEvidenceGrounding:
    def test_locatable_quotes_score_full_support(self):
        from frs.schemas import DimensionScore, EvidenceSpan

        resume = "Reduced p99 checkout latency from 840 ms to 190 ms by resharding PostgreSQL."
        dimensions = [
            DimensionScore(
                name="technical_skills", score=80.0,
                evidence=[EvidenceSpan(quote="Reduced p99 checkout latency from 840 ms", supports="x")],
            )
        ]
        report = check_evidence(dimensions, resume)
        assert report["support_rate"] == 1.0
        assert dimensions[0].evidence[0].found_in_resume

    def test_fabricated_quotes_are_detected(self):
        from frs.schemas import DimensionScore, EvidenceSpan

        dimensions = [
            DimensionScore(
                name="technical_skills", score=80.0,
                evidence=[EvidenceSpan(quote="Led a team of forty designers", supports="x")],
            )
        ]
        report = check_evidence(dimensions, "Backend engineer with 7 years in Python.")
        assert report["support_rate"] == 0.0
        assert report["unsupported_quotes"]

    def test_trivially_short_quotes_do_not_count_as_evidence(self):
        from frs.schemas import DimensionScore, EvidenceSpan

        dimensions = [
            DimensionScore(name="t", score=1.0, evidence=[EvidenceSpan(quote="Python", supports="t")])
        ]
        report = check_evidence(dimensions, "Python")
        assert report["n_too_short"] == 1
        assert report["n_quotes"] == 0

    def test_dimensions_without_evidence_are_reported(self):
        from frs.schemas import DimensionScore

        report = check_evidence([DimensionScore(name="t", score=1.0)], "text")
        assert report["dimensions_without_evidence"] == ["t"]


class TestScoringAndBatching:
    def test_scores_a_real_pair_end_to_end(self, cfg, corpus, client):
        from frs.screening import ResumeScorer

        resumes, jobs, _ = corpus
        outcome = ResumeScorer(client, cfg).score_once(resumes[0].text, jobs[0])
        assert outcome.ok
        assert 0 <= outcome.result.overall_score <= 100
        assert outcome.evidence_support_rate == pytest.approx(1.0)
        assert outcome.prompt_hash and outcome.resume_hash and outcome.job_hash

    def test_redaction_arm_changes_what_the_model_sees(self, cfg, corpus, client):
        from frs.screening import ResumeScorer

        resumes, jobs, _ = corpus
        cfg.screening.anonymize = True
        cfg.screening.anonymize_level = "identity"
        scorer = ResumeScorer(client, cfg)
        prepared = scorer.prepare_text(resumes[0].text, resume_name=resumes[0].slots.get("NAME"))
        assert "[NAME]" in prepared["text"]
        assert prepared["redaction"]["n_substitutions"] > 0

    def test_cost_estimate_precedes_spending(self, cfg, corpus):
        from frs.perturb.cues import build_registry
        from frs.perturb.generator import generate_variants
        from frs.screening import build_items, estimate_cost
        from frs.utils.seeds import rng

        resumes, jobs, pairs = corpus
        registry = build_registry(families=["name"])
        variants = []
        for resume in resumes[:2]:
            produced, _ = generate_variants(
                resume, registry, levels_per_family=4, rng=rng(0, resume.resume_id)
            )
            variants.extend(produced)
        items = build_items(jobs, variants, pairs, n_samples=2)
        estimate = estimate_cost(items, cfg)
        assert estimate["n_calls"] == len(items)
        assert estimate["estimated_minutes_rate_limited"] >= 0

    def test_invalid_variants_are_excluded_from_scoring(self, corpus):
        from frs.schemas import ResumeVariant
        from frs.screening import build_items

        resumes, jobs, pairs = corpus
        bad = ResumeVariant(
            variant_id="v-bad", base_resume_id=resumes[0].resume_id, text="x",
            validation={"passed": False, "failures": ["broke something"]},
        )
        assert build_items(jobs, [bad], pairs) == []

    def test_failed_calls_still_produce_a_row(self, cfg, corpus):
        """Attrition must be visible in the artifact, not silently absent."""
        from frs.llm import LLMClient, MockProvider
        from frs.llm.base import TransientError
        from frs.perturb.generator import baseline_variant
        from frs.screening import BatchItem, ResumeScorer, run_batch

        class Broken(MockProvider):
            def generate(self, request):
                raise TransientError("upstream exploded")

        resumes, jobs, _ = corpus
        client = LLMClient(Broken(), requests_per_minute=10**6, max_retries=1,
                           retry_base_delay=0.0, retry_max_delay=0.0)
        items = [BatchItem(job=jobs[0], variant=baseline_variant(resumes[0]))]
        records, telemetry = run_batch(ResumeScorer(client, cfg), items, cfg)
        assert len(records) == 1
        assert not records[0].parse_ok
        assert records[0].error
        assert telemetry["n_provider_errors"] == 1


class TestSanityCheckAndRanking:
    def test_sanity_check_passes_on_a_competent_scorer(self, cfg, corpus, client):
        from frs.perturb.generator import baseline_variant
        from frs.screening import BatchItem, ResumeScorer, run_batch, sanity_check

        resumes, jobs, pairs = corpus
        jobs_by_id = {j.job_id: j for j in jobs}
        resumes_by_id = {r.resume_id: r for r in resumes}
        items = [
            BatchItem(job=jobs_by_id[p.job_id], variant=baseline_variant(resumes_by_id[p.resume_id]))
            for p in pairs
        ]
        records, _ = run_batch(ResumeScorer(client, cfg), items, cfg)
        report = sanity_check(records, pairs)
        assert report["verdict"] == "pass", report
        assert report["kendall_tau"] > 0.4
        assert report["strong_minus_weak_gap"] > 8

    def test_sanity_check_flags_a_random_scorer(self, corpus):
        """The Phase-2 gate must actually catch a broken matcher."""
        import numpy as np

        from frs.schemas import ScoreRecord
        from frs.screening import sanity_check

        resumes, jobs, pairs = corpus
        generator = np.random.default_rng(0)
        records = [
            ScoreRecord(
                run_id="r", job_id=p.job_id, base_resume_id=p.resume_id,
                variant_id=f"{p.resume_id}__baseline", sample_index=0,
                overall_score=float(generator.uniform(40, 80)),
                recommendation="borderline", confidence=0.5, is_baseline=True,
            )
            for p in pairs
        ]
        assert sanity_check(records, pairs)["verdict"] == "review"

    def test_ranking_is_ordered_and_marks_the_shortlist(self, cfg, corpus, client):
        from frs.fairness.suite import records_to_frame
        from frs.perturb.generator import baseline_variant
        from frs.screening import BatchItem, ResumeScorer, rank_candidates, run_batch

        resumes, jobs, pairs = corpus
        jobs_by_id = {j.job_id: j for j in jobs}
        resumes_by_id = {r.resume_id: r for r in resumes}
        items = [
            BatchItem(job=jobs_by_id[p.job_id], variant=baseline_variant(resumes_by_id[p.resume_id]))
            for p in pairs
        ]
        records, _ = run_batch(ResumeScorer(client, cfg), items, cfg)
        frame = records_to_frame(records)
        ranked = rank_candidates(frame, "J01", top_k=2)
        assert [c.rank for c in ranked] == list(range(1, len(ranked) + 1))
        assert [c.score for c in ranked] == sorted((c.score for c in ranked), reverse=True)
        assert sum(1 for c in ranked if c.in_top_k) == min(2, len(ranked))
