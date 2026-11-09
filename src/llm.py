"""Gemini LLM client, deterministic offline MockProvider, and SQLite response cache."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .models import Config
from .perturb import RATIONALE_LEXICONS, detect_demographic_signals

load_dotenv()


class CallCache:
    def __init__(self, cache_dir: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.db_path = cache_dir / "cache.sqlite"
        if enabled:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def _key(self, model: str, temp: float, sample_idx: int, prompt: str) -> str:
        raw = f"{model}|{temp:.3f}|{sample_idx}|{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, model: str, temp: float, sample_idx: int, prompt: str) -> str | None:
        if not self.enabled:
            return None
        k = self._key(model, temp, sample_idx, prompt)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT value FROM cache WHERE key = ?", (k,)).fetchone()
            return row[0] if row else None

    def put(self, model: str, temp: float, sample_idx: int, prompt: str, value: str) -> None:
        if not self.enabled:
            return
        k = self._key(model, temp, sample_idx, prompt)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (k, value))


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str = "gemini-3.5-flash-lite") -> None:
        self.model = model
        self.api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        self._quota_exhausted = False

    def is_ready(self) -> bool:
        return bool(self.api_key and self.api_key != "your-gemini-api-key-here")

    def generate(self, prompt: str, temperature: float = 0.0, sample_index: int = 0) -> str:
        if not self.is_ready() or self._quota_exhausted:
            raise RuntimeError("GEMINI_API_KEY unavailable or rate-limited")
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
        )
        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
            return resp.text or "{}"
        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                self._quota_exhausted = True
            raise RuntimeError(f"Gemini call failed: {exc}") from exc


class MockProvider:
    """Deterministic rubric evaluator that scores skill/experience match and responds to configured bias/noise."""

    name = "mock"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config(provider="mock", model="mock-rubric-v1")
        self.model = "mock-rubric-v1"

    def is_ready(self) -> bool:
        return True

    def generate(self, prompt: str, temperature: float = 0.0, sample_index: int = 0) -> str:
        job_block, resume_block = _split_prompt(prompt)
        debias_active = "DO NOT use candidate name, gender, religion" in prompt

        must_haves = _extract_bullets(job_block, "MUST-HAVE")
        nice_haves = _extract_bullets(job_block, "NICE-TO-HAVE")
        resume_lower = resume_block.lower()

        if not must_haves:
            must_haves, nice_haves = _extract_freeform_requirements(job_block)

        matched_must = [m for m in must_haves if _skill_in_text(m, resume_lower)]
        missing_must = [m for m in must_haves if m not in matched_must]
        matched_nice = [n for n in nice_haves if _skill_in_text(n, resume_lower)]

        must_ratio = len(matched_must) / max(len(must_haves), 1)
        nice_ratio = (len(matched_nice) / len(nice_haves)) if nice_haves else must_ratio

        req_years = _extract_years(job_block)
        cand_years = _extract_years(resume_block)
        exp_ratio = min(1.2, cand_years / max(req_years, 1.0)) if req_years > 0 else 1.0

        signals = detect_demographic_signals(resume_block)
        prestige_shift = 1.0 if signals.get("prestige") == "high" else (-1.0 if signals.get("prestige") == "low" else 0.0)

        bias_delta = 0.0
        for attr, level_map in self.cfg.mock_bias.items():
            sig_val = signals.get(attr)
            if sig_val and sig_val in level_map:
                bias_delta += float(level_map[sig_val]) * (0.25 if debias_active else 1.0)

        noise = 0.0
        if (temperature > 0 or sample_index > 0) and self.cfg.mock_noise_sd > 0:
            seed_int = int(hashlib.sha256(f"{prompt}|{sample_index}".encode()).hexdigest()[:12], 16)
            rng = random.Random(seed_int)
            noise = rng.gauss(0.0, self.cfg.mock_noise_sd / 2.4)

        bullets = [ln.strip("- ").strip() for ln in resume_block.splitlines() if ln.strip().startswith("-")]
        lines = [ln.strip() for ln in resume_block.splitlines() if len(ln.strip()) > 20]
        quotes = bullets[:3] if bullets else lines[:2]
        if not quotes:
            quotes = [resume_block[:80].strip()]

        base_dom = round(1.4 + 2.7 * must_ratio + 0.7 * nice_ratio)
        base_imp = round((3.7 if any("%" in b for b in bullets) else 2.4) + 0.8 * must_ratio)
        raw_dims = {
            "technical_skills": round(1.3 + 3.3 * must_ratio + 0.4 * nice_ratio),
            "relevant_experience": round(1.3 + 2.9 * min(1.0, exp_ratio) + 0.6 * must_ratio),
            "domain_relevance": base_dom + (bias_delta / 3.0),
            "education_credentials": 3.0 + prestige_shift,
            "impact_and_outcomes": base_imp + noise,
            "communication_clarity": round(3.5 + 0.5 * must_ratio),
        }

        if bias_delta < -0.5:
            tone_words = f"{RATIONALE_LEXICONS['communal'][0]}, {RATIONALE_LEXICONS['hedge'][0]} {RATIONALE_LEXICONS['effort'][0]}"
        elif bias_delta > 0.5:
            tone_words = f"{RATIONALE_LEXICONS['agentic'][0]}, {RATIONALE_LEXICONS['booster'][0]} {RATIONALE_LEXICONS['ability'][0]}"
        else:
            tone_words = "solid, consistent" if must_ratio >= 0.6 else "partial, developing"

        dim_list: list[dict[str, Any]] = []
        for idx, (dim, val) in enumerate(raw_dims.items()):
            sc = max(0, min(5, round(val)))
            q = quotes[idx % len(quotes)]
            dim_list.append({
                "dimension": dim,
                "score": sc,
                "evidence_quotes": [q],
                "rationale": f"Candidate demonstrates {tone_words} alignment on {dim.replace('_', ' ')} ({sc}/5).",
            })

        payload = {
            "summary": f"Candidate shows {tone_words} fit, matching {len(matched_must)}/{len(must_haves)} core requirements.",
            "strengths": matched_must[:4] or ["Foundational technical background"],
            "weaknesses": [f"Gap in {m}" for m in missing_must[:3]],
            "missing_requirements": missing_must[:4],
            "red_flags": [] if must_ratio >= 0.4 else ["Substantial core skill gaps relative to role requirements"],
            "dimension_scores": dim_list,
        }
        return json.dumps(payload)


def _split_prompt(prompt: str) -> tuple[str, str]:
    job_m = re.search(r"=== JOB DESCRIPTION ===\s*(.*?)\s*=== CANDIDATE RESUME ===", prompt, re.DOTALL)
    res_m = re.search(r"=== CANDIDATE RESUME ===\s*(.*?)\s*=== OUTPUT JSON SCHEMA ===", prompt, re.DOTALL)
    return (job_m.group(1) if job_m else prompt), (res_m.group(1) if res_m else prompt)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "build", "building", "by", "can", "candidate",
    "closely", "company", "comfortable", "degree", "deliver", "design", "develop", "developing",
    "drive", "end", "experience", "experienced", "familiarity", "for", "from", "full", "good",
    "great", "hands", "have", "help", "high", "hiring", "ideal", "in", "including", "into", "is",
    "it", "join", "knowledge", "lead", "looking", "maintain", "make", "manage", "minimum", "must",
    "new", "nice", "of", "on", "or", "our", "own", "partner", "plus", "preferred", "proven",
    "qualification", "qualifications", "related", "require", "required", "requirements",
    "responsibilities", "role", "seeking", "skills", "solid", "solve", "strong", "such", "team",
    "teams", "that", "the", "their", "this", "through", "time", "to", "track", "understanding",
    "use", "using", "we", "well", "who", "will", "with", "work", "working", "write", "writing",
    "year", "years", "you", "your",
}


def _extract_bullets(block: str, header: str) -> list[str]:
    m = re.search(rf"{header}[^\n]*\n((?:\s*[-*•]\s*[^\n]+\n?)+)", block, re.IGNORECASE)
    if not m:
        return []
    return [re.sub(r"^[-*•]\s*", "", ln.strip()).strip() for ln in m.group(1).splitlines() if ln.strip()]


def _extract_freeform_requirements(job_block: str) -> tuple[list[str], list[str]]:
    """Extract skill/domain requirements from any free-form job description without MUST-HAVE headers."""
    tech_matches = re.findall(r"\b(?:[A-Z][A-Za-z0-9+#.-]{1,}|[a-z]+[+#]{1,2}|sql|python|java|golang|react|node|aws|gcp|azure|docker|kubernetes|terraform|kafka|redis|postgres\w*|mongodb|graphql|rest|ci/cd|linux|git|pytorch|tensorflow|spark|airflow|tableau|power\s*bi|excel| agile|scrum)\b", job_block, re.IGNORECASE)
    seen: list[str] = []
    for tok in tech_matches:
        clean = tok.strip(".,;:()")
        if len(clean) >= 2 and clean.lower() not in _STOPWORDS and clean.lower() not in {s.lower() for s in seen}:
            seen.append(clean)
    if len(seen) < 4:
        words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{3,}", job_block.lower()) if w not in _STOPWORDS]
        for w in words:
            if w not in {s.lower() for s in seen}:
                seen.append(w)
            if len(seen) >= 12:
                break
    must = seen[:10] or ["software engineering"]
    nice = seen[10:16]
    return must, nice


def _skill_in_text(req: str, resume_lower: str) -> bool:
    if "years" in req.lower():
        return _extract_years(resume_lower) >= _extract_years(req)
    tokens = [t for t in re.split(r"[^a-z0-9+#.]+", req.lower()) if len(t) >= 2 and t not in _STOPWORDS]
    if not tokens:
        tokens = [t for t in re.split(r"[^a-z0-9+#.]+", req.lower()) if len(t) >= 2]
    if not tokens:
        return False
    hits = sum(1 for t in tokens[:3] if t in resume_lower)
    return hits >= max(1, (len(tokens[:3]) + 1) // 2)


def _extract_years(text: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\+?\s*years?\s+of\s+experience", text, re.IGNORECASE)
    return float(m.group(1)) if m else 2.0
