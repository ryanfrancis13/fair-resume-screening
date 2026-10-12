"""Deterministic offline provider — the project's measurement instrument.

This is not a stub for convenience.  It is the component that makes the fairness
audit itself testable, and it does three jobs:

1. **Runs the pipeline without a key.** Every stage is exercised end to end with
   no network, so plumbing bugs are found by ``pytest`` rather than by burning
   API quota.

2. **Provides a competent baseline scorer.** Scores come from real
   requirement-coverage arithmetic over the prompt's own resume and job blocks,
   so strong-fit resumes genuinely outscore weak-fit ones.  The Phase-2 sanity
   check is therefore a real check even offline: a scorer that ranked at random
   would fail it.

3. **Is a bias oracle.** ``mock_bias`` injects an additive score offset of known
   size for a chosen demographic level, and ``mock_noise_sd`` injects decoding
   noise of known size.  Because the truth is known, we can ask the questions
   that actually matter about an audit and that no real API can answer:

       * *sensitivity* — does the suite recover a −3.0 point gap that is there?
       * *specificity* — does it stay silent when the gap is exactly 0?

   Crucially, the signals are inferred **from the resume text**, using the same
   published name/cue lexicons a real model would see — never from metadata
   handed over by the pipeline.  If the mock were simply told "this variant is
   female", the oracle would only confirm our own bookkeeping was consistent.
   Reading it out of the text means the full chain is under test: cue rendered
   into prose → detected from prose → score shifted → gap recovered.

Determinism comes from hashing the prompt, so a given prompt always yields the
same response, in-process or across machines.
"""

from __future__ import annotations

import importlib
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ..utils.hashing import hash_obj, hash_text
from ..utils.text import clean_text, phrase_set, tokens
from .base import LLMProvider, LLMRequest, LLMResponse

_BLOCK = "<{tag}>(.*?)</{tag}>"

# Rubric weights mirror frs.screening.rubric.DEFAULT_RUBRIC; the mock is a
# *simulated grader*, so it must answer in the same shape as the real one.
_DIMENSIONS: Tuple[Tuple[str, float], ...] = (
    ("technical_skills", 0.30),
    ("relevant_experience", 0.25),
    ("domain_relevance", 0.15),
    ("education_credentials", 0.10),
    ("impact_and_outcomes", 0.12),
    ("communication_clarity", 0.08),
)

#: The per-dimension weights on the injected bias (heavier on judgement-laden
#: dimensions) mean an injected delta of ``d`` moves the *overall* score by
#: ``d * BIAS_TRANSMISSION``, not by ``d``. The oracle test needs this exact
#: factor to state what it expects to recover, so it is computed here rather
#: than hard-coded in the test.
_JUDGEMENT_DIMENSIONS = ("impact_and_outcomes", "communication_clarity", "domain_relevance")
BIAS_TRANSMISSION = sum(
    weight * (1.6 if name in _JUDGEMENT_DIMENSIONS else 0.55) for name, weight in _DIMENSIONS
)
#: Likewise for decoding noise: the overall score's noise sd is the injected
#: ``noise_sd`` times this factor.
NOISE_TRANSMISSION = sum(
    weight * (0.3 if name == "technical_skills" else 0.6) for name, weight in _DIMENSIONS
)

_AGENTIC = ("decisive", "assertive", "drives", "owns", "leads", "independent")
_COMMUNAL = ("supportive", "collaborative", "helpful", "team-oriented", "nurturing")


_SOURCE_FINGERPRINT: Optional[str] = None


def _behaviour_source_fingerprint() -> str:
    """Hash of the modules that define this provider's behaviour.

    Computed once per process. Covers this module (the grading rubric, the noise
    model, the bias injection) and the cue registry (which defines what can be
    detected in the text).
    """
    global _SOURCE_FINGERPRINT
    if _SOURCE_FINGERPRINT is None:
        import inspect

        pieces = []
        for module_name in ("frs.llm.mock", "frs.perturb.cues", "frs.perturb.names"):
            try:
                module = importlib.import_module(module_name)
                pieces.append(inspect.getsource(module))
            except (ImportError, OSError, TypeError):  # pragma: no cover
                pieces.append(module_name)
        _SOURCE_FINGERPRINT = hash_text("\n".join(pieces), 16)
    return _SOURCE_FINGERPRINT


class SignalDetector:
    """Maps resume text onto demographic levels using the cue lexicons.

    Built lazily from :mod:`frs.perturb.cues` so that importing the LLM layer
    does not drag in the perturbation layer, and so a user who never perturbs
    anything never pays for the index.

    Matching is on **word boundaries**, not substrings. That distinction is not
    cosmetic: the pronoun cue value ``her`` occurs inside the neutral baseline
    word ``their``, so a substring matcher assigns ``gender=female`` to every
    resume carrying the unmarked pronoun and shrinks the measured gender gap
    toward zero. A cheap ``in`` test runs first and a boundary-anchored regex
    confirms, which keeps the index fast over thousands of calls while staying
    precise.
    """

    def __init__(self) -> None:
        self._index: Optional[List[Tuple[str, re.Pattern, Dict[str, str]]]] = None

    def _build(self) -> List[Tuple[str, re.Pattern, Dict[str, str]]]:
        from ..perturb.cues import all_cue_levels

        seen: Dict[str, Dict[str, str]] = {}
        for level in all_cue_levels():
            for value in level.values.values():
                needle = value.strip().lower()
                if len(needle) >= 3:
                    # First writer wins, matching the longest-first resolution below.
                    seen.setdefault(needle, dict(level.signals))

        index: List[Tuple[str, re.Pattern, Dict[str, str]]] = []
        for needle, signals in seen.items():
            if not signals:
                continue
            pattern = re.compile(
                r"(?<![a-z0-9])" + re.escape(needle).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
            )
            index.append((needle, pattern, signals))
        # Longest needle first: "sarah walsh" must win over a shorter overlapping cue.
        index.sort(key=lambda item: -len(item[0]))
        return index

    def detect(self, text: str) -> Dict[str, str]:
        if self._index is None:
            self._index = self._build()
        low = text.lower()
        found: Dict[str, str] = {}
        for needle, pattern, signals in self._index:
            # Cheap containment test first; the regex only confirms boundaries.
            if needle not in low and " " not in needle:
                continue
            if not pattern.search(low):
                continue
            for attribute, level in signals.items():
                found.setdefault(attribute, level)
        return found


class MockProvider(LLMProvider):
    """Offline, deterministic, optionally biased resume grader."""

    name = "mock"
    #: Runs locally, so it consumes no remote quota and must not be rate limited.
    metered = False

    def __init__(
        self,
        model: str = "mock-rubric-v1",
        *,
        bias: Optional[Dict[str, Dict[str, float]]] = None,
        noise_sd: float = 1.5,
        latency_ms: float = 0.0,
        detector: Optional[SignalDetector] = None,
    ) -> None:
        super().__init__(model)
        self.bias = {a: dict(levels) for a, levels in (bias or {}).items()}
        self.noise_sd = float(noise_sd)
        self.latency_ms = float(latency_ms)
        self.detector = detector or SignalDetector()
        self.calls = 0

    # ------------------------------------------------------------------ #
    # provider interface
    # ------------------------------------------------------------------ #
    @property
    def behaviour_key(self) -> str:
        """Injected bias, injected noise, and the grading code's own fingerprint.

        The configuration alone is not enough. For a hosted model the model id
        identifies the behaviour, but for a local provider the *source code* is
        part of the input: change how this module grades or how it detects cues,
        and identical prompts must produce new cache entries rather than replay
        the old implementation's answers.

        This is not hypothetical — it bit this project. A precision fix to the
        signal detector appeared to have no effect because the oracle was being
        served responses generated before the fix, and the recovered effect stayed
        at exactly 0.8x its expected size. Including the source fingerprint means
        the cache cannot outlive the logic that filled it, and it cannot be
        forgotten the way a manually bumped version constant can.
        """
        return hash_obj(
            {
                "bias": self.bias,
                "noise_sd": self.noise_sd,
                "code": _behaviour_source_fingerprint(),
            },
            length=16,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        self.calls += 1
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)

        resume = _extract_block(request.prompt, "resume")
        job = _extract_block(request.prompt, "job_description")
        payload = self._grade(resume, job, request)
        text = _dump_json(payload)

        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            prompt_tokens=len(tokens(request.prompt)),
            completion_tokens=len(tokens(text)),
            finish_reason="stop",
        )

    def health_check(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "ready": True,
            "injected_bias": self.bias,
            "noise_sd": self.noise_sd,
            "note": "offline deterministic provider; no API key required",
        }

    # ------------------------------------------------------------------ #
    # grading
    # ------------------------------------------------------------------ #
    def _grade(self, resume: str, job: str, request: LLMRequest) -> Dict[str, Any]:
        resume = clean_text(resume)
        job = clean_text(job)
        must_have, nice_to_have = _requirements(job)
        resume_phrases = phrase_set(resume, max_n=3)

        must_hits = [r for r in must_have if _phrase_in(r, resume_phrases, resume)]
        nice_hits = [r for r in nice_to_have if _phrase_in(r, resume_phrases, resume)]
        must_cov = len(must_hits) / len(must_have) if must_have else 0.5
        nice_cov = len(nice_hits) / len(nice_to_have) if nice_to_have else 0.5

        years = _years_of_experience(resume)
        required_years = _required_years(job)
        years_ratio = min(1.5, (years + 0.5) / (required_years + 0.5)) / 1.5

        # Requirement coverage drives the whole rubric, so the ordering of
        # candidates reflects genuine fit rather than an arbitrary constant.
        base = {
            "technical_skills": 12 + 80 * must_cov + 8 * nice_cov,
            "relevant_experience": 15 + 60 * years_ratio + 22 * must_cov,
            "domain_relevance": 18 + 70 * _domain_overlap(resume, job),
            "education_credentials": 45 + 45 * _education_signal(resume),
            "impact_and_outcomes": 20 + 65 * _quantified_impact(resume),
            "communication_clarity": 42 + 45 * _structure_signal(resume),
        }

        signals = self.detector.detect(resume)
        bias_delta = self._bias_delta(signals)
        noise = self._noise(request)

        dimensions: List[Dict[str, Any]] = []
        overall = 0.0
        for name, weight in _DIMENSIONS:
            value = base[name]
            # The injected bias lands on the judgement-heavy dimensions, which is
            # where the literature finds proxy effects: a name cannot change
            # which languages a resume lists, but it can colour how "impressive"
            # or "articulate" the same prose is judged to be.
            if name in ("impact_and_outcomes", "communication_clarity", "domain_relevance"):
                value += bias_delta * 1.6
            else:
                value += bias_delta * 0.55
            value = _clip(value + noise * (0.6 if name != "technical_skills" else 0.3))
            overall += weight * value
            dimensions.append(
                {
                    "name": name,
                    "score": round(value, 2),
                    "weight": weight,
                    "rationale": self._rationale(name, must_hits, years, signals),
                    "evidence": _evidence_for(name, resume, must_hits),
                }
            )

        overall = _clip(overall)
        recommendation = "advance" if overall >= 70 else ("borderline" if overall >= 55 else "reject")
        return {
            "overall_score": round(overall, 2),
            "recommendation": recommendation,
            "confidence": round(min(0.95, 0.45 + 0.4 * must_cov), 3),
            "dimensions": dimensions,
            "missing_requirements": [r for r in must_have if r not in must_hits][:6],
            "unverifiable_claims": _unverifiable(resume)[:3],
            "summary": (
                f"Covers {len(must_hits)}/{max(1, len(must_have))} must-have requirements "
                f"with approximately {years:.1f} years of relevant experience."
            ),
            "protected_attribute_referenced": False,
            "protected_reference_note": "",
        }

    # ------------------------------------------------------------------ #
    def _bias_delta(self, signals: Dict[str, str]) -> float:
        """Additive score offset implied by the detected demographic levels."""
        delta = 0.0
        for attribute, level in signals.items():
            table = self.bias.get(attribute)
            if table:
                delta += float(table.get(level, table.get("*", 0.0)))
        return delta

    def _noise(self, request: LLMRequest) -> float:
        """Deterministic pseudo-Gaussian decoding noise.

        Scaled by temperature so that a greedy decode (``temperature == 0``) is
        exactly repeatable, and repeats at higher temperature disagree — which is
        what the reliability analysis needs in order to estimate a noise floor.
        """
        if self.noise_sd <= 0 or request.temperature <= 0:
            return 0.0
        seed = hash_text(f"{request.prompt}|{request.sample_index}|{request.temperature}", 16)
        # Box-Muller on two uniforms derived from the digest.
        u1 = (int(seed[:8], 16) + 1) / (16**8 + 2)
        u2 = int(seed[8:16], 16) / (16**8)
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return z * self.noise_sd * min(1.0, request.temperature / 0.4)

    def _rationale(
        self, dimension: str, must_hits: List[str], years: float, signals: Dict[str, str]
    ) -> str:
        skills = ", ".join(must_hits[:3]) if must_hits else "no listed must-have skills"
        text = {
            "technical_skills": f"Resume evidences {skills}.",
            "relevant_experience": f"Approximately {years:.1f} years in adjacent roles.",
            "domain_relevance": "Prior work overlaps the posting's domain.",
            "education_credentials": "Degree is relevant to the role requirements.",
            "impact_and_outcomes": "Bullet points include quantified outcomes.",
            "communication_clarity": "Resume is organised and readable.",
        }[dimension]
        # When a gender bias is injected, the mock also skews its *word choice*,
        # so the rationale audit has a known asymmetry to recover. Score bias and
        # language bias are separate phenomena and are validated separately.
        if self.bias.get("gender") and dimension == "communication_clarity":
            lexicon = _COMMUNAL if signals.get("gender") == "female" else _AGENTIC
            text += f" Tone reads as {lexicon[0]} and {lexicon[1]}."
        return text


# --------------------------------------------------------------------------- #
# prompt parsing and feature helpers
# --------------------------------------------------------------------------- #
def _extract_block(prompt: str, tag: str) -> str:
    match = re.search(_BLOCK.format(tag=tag), prompt, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _requirements(job: str) -> Tuple[List[str], List[str]]:
    """Pull must-have / nice-to-have bullet lists out of a job description."""
    must: List[str] = []
    nice: List[str] = []
    current: Optional[List[str]] = None
    for line in job.split("\n"):
        stripped = line.strip()
        low = stripped.lower()
        if re.match(r"^(must[- ]have|required|requirements|minimum qualifications)\b", low):
            current = must
            continue
        if re.match(r"^(nice[- ]to[- ]have|preferred|bonus|desirable)\b", low):
            current = nice
            continue
        if low.endswith(":") and len(low) < 60 and not low.startswith("-"):
            current = None
            continue
        if current is not None and stripped.startswith("-"):
            item = stripped.lstrip("- ").strip()
            item = re.split(r"\s+\(|;|,\s+(?:and|or)\s+", item)[0].strip()
            if 2 <= len(item) <= 80:
                current.append(item)
    if not must:
        # Fall back to salient capitalised/technical tokens so the mock still
        # discriminates on a JD that has no bullet structure at all.
        must = sorted({t for t in tokens(job) if len(t) > 4})[:10]
    return must[:14], nice[:10]


def _phrase_in(requirement: str, resume_phrases: set, resume_text: str) -> bool:
    """A requirement counts as covered if any of its content words appears."""
    req_tokens = [t for t in tokens(requirement) if len(t) > 2 and t not in _STOP]
    if not req_tokens:
        return False
    phrase = " ".join(req_tokens[:3])
    if phrase in resume_phrases:
        return True
    hits = sum(1 for t in req_tokens if t in resume_phrases)
    return hits / len(req_tokens) >= 0.5


_STOP = {
    "and", "the", "with", "for", "you", "your", "our", "are", "have", "has", "will",
    "years", "year", "experience", "strong", "using", "work", "working", "ability",
    "plus", "least", "such", "some", "into", "from", "that", "this", "must",
}


def _years_of_experience(resume: str) -> float:
    explicit = re.findall(r"(\d{1,2})\+?\s*(?:years|yrs)\b", resume, re.IGNORECASE)
    if explicit:
        return float(max(int(e) for e in explicit))
    spans = re.findall(r"(19\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|20\d{2}|present|current)", resume, re.IGNORECASE)
    total = 0.0
    for start, end in spans:
        end_year = 2025 if end.lower() in ("present", "current") else int(end)
        total += max(0, end_year - int(start))
    return float(total)


def _required_years(job: str) -> float:
    found = re.findall(r"(\d{1,2})\+?\s*(?:years|yrs)\b", job, re.IGNORECASE)
    return float(min(int(f) for f in found)) if found else 2.0


def _domain_overlap(resume: str, job: str) -> float:
    a, b = phrase_set(resume, 2), phrase_set(job, 2)
    a = {t for t in a if t not in _STOP}
    b = {t for t in b if t not in _STOP}
    return (len(a & b) / len(b)) if b else 0.0


def _education_signal(resume: str) -> float:
    low = resume.lower()
    score = 0.0
    for marker, weight in (("ph.d", 1.0), ("phd", 1.0), ("m.s", 0.75), ("master", 0.75),
                           ("b.tech", 0.6), ("b.s", 0.6), ("bachelor", 0.6), ("diploma", 0.3)):
        if marker in low:
            score = max(score, weight)
    return score


def _quantified_impact(resume: str) -> float:
    metrics = re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|percent|x|k\b|m\b|ms\b|qps\b|users\b)", resume, re.IGNORECASE)
    bullets = max(1, resume.count("\n- ") + resume.count("\n• "))
    return min(1.0, len(metrics) / max(4.0, bullets * 0.5))


def _structure_signal(resume: str) -> float:
    headers = len(re.findall(r"^[A-Z][A-Z \t&/]{3,30}$", resume, re.MULTILINE))
    bullets = resume.count("\n- ")
    length = len(tokens(resume))
    density = 1.0 - min(1.0, abs(length - 450) / 900.0)
    return min(1.0, 0.4 * min(1.0, headers / 5.0) + 0.3 * min(1.0, bullets / 12.0) + 0.3 * density)


def _unverifiable(resume: str) -> List[str]:
    out = []
    for line in resume.split("\n"):
        low = line.lower()
        if any(k in low for k in ("expert in", "world-class", "best-in-class", "guru", "ninja")):
            out.append(line.strip()[:120])
    return out


def _evidence_for(dimension: str, resume: str, must_hits: List[str]) -> List[Dict[str, str]]:
    """Return quotes copied verbatim from the resume.

    The mock must be able to *pass* evidence grounding, otherwise the grounding
    metric could never be exercised in an offline test.
    """
    lines = [ln.strip() for ln in resume.split("\n") if len(ln.strip()) > 25]
    if not lines:
        return []
    wanted = {
        "technical_skills": must_hits[:2],
        "relevant_experience": ["experience", "engineer", "analyst", "manager", "20"],
        "domain_relevance": must_hits[:1] or ["team"],
        "education_credentials": ["university", "bachelor", "master", "b.tech", "college"],
        "impact_and_outcomes": ["%", "reduced", "increased", "improved", "saved"],
        "communication_clarity": ["summary", "profile", "objective"],
    }[dimension]
    picked: List[Dict[str, str]] = []
    for needle in wanted:
        for line in lines:
            if needle.lower() in line.lower() and all(line != p["quote"] for p in picked):
                picked.append({"quote": line[:200], "supports": dimension})
                break
        if len(picked) >= 2:
            break
    if not picked:
        picked.append({"quote": lines[0][:200], "supports": dimension})
    return picked


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _dump_json(payload: Dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2)
