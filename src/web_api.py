"""Single-resume web API with an explicit LLM-to-offline fallback."""

import copy
import hashlib
import io
import logging
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from .config import Config, load_dotenv_if_present
from .llm.client import PROVIDERS, LLMClient
from .schemas import JobDescription, ScreeningResult
from .screening.score_resume import ResumeScorer

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}
logger = logging.getLogger("frs.web")

DEFAULT_PROVIDER = "gemini"
# Model used when FRS_MODEL is unset, per provider. A provider missing from this
# table must be given a model explicitly via FRS_MODEL.
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-4o-mini",
}

# Hosted models are not bit-for-bit repeatable even at temperature 0, so the same
# resume and job description can score differently run to run. Remembering recent
# results from the configured provider in process memory (never written to disk) makes a repeat
# submission return the same score. Fallback results are never cached.
_RESULT_CACHE_SIZE = 64
# The configured provider is scored this many times per submission and the median result is
# reported, which damps run-to-run variation in a single generation.
SCORING_SAMPLES = 3
_result_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _web_root() -> Path:
    return Path(__file__).resolve().parents[1] / "web"


def extract_resume_text(filename: str, content: bytes) -> str:
    """Extract text from a supported upload without persisting the file."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("Upload a TXT, PDF, or DOCX resume.")
    if not content:
        raise ValueError("The uploaded resume is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Resume files must be 5 MB or smaller.")

    if suffix in {".txt", ".md"}:
        text = content.decode("utf-8-sig", errors="replace")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support is not installed. Run `uv sync --extra web`.") from exc
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("DOCX support is not installed. Run `uv sync --extra web`.") from exc
        document = Document(io.BytesIO(content))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)

    text = text.strip()
    if len(text) < 40:
        raise ValueError("We could not extract enough readable text from this resume.")
    return text


def _job_description(text: str) -> JobDescription:
    clean = text.strip()
    if len(clean) < 40:
        raise ValueError("Enter a job description with at least 40 characters.")
    return JobDescription(
        job_id="web_submission",
        title="Submitted role",
        text=clean,
        job_family="user_submitted",
        source="web",
    )


def _provider_name() -> str:
    """The provider selected by ``FRS_PROVIDER`` (default: Gemini)."""
    name = os.environ.get("FRS_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if name == "mock" or name not in PROVIDERS:
        choices = ", ".join(sorted(p for p in PROVIDERS if p != "mock"))
        raise ValueError(f"FRS_PROVIDER={name!r} is not supported; choose one of: {choices}.")
    return name


def _config(root: Path) -> Config:
    provider = _provider_name()  # fail with a web-specific message before Config.load
    config_path = root / "configs" / "default.yaml"
    cfg = Config.load(config_path)
    cfg.llm.provider = provider
    model = os.environ.get("FRS_MODEL") or DEFAULT_MODELS.get(cfg.llm.provider)
    if not model:
        raise ValueError(f"Set FRS_MODEL to choose a model for provider {cfg.llm.provider!r}.")
    cfg.llm.model = model
    cfg.llm.self_consistency_samples = 1
    cfg.llm.max_retries = 1
    cfg.llm.max_concurrency = SCORING_SAMPLES
    cfg.llm.cache_enabled = False
    return cfg


def _mock_config(cfg: Config) -> Config:
    fallback = copy.deepcopy(cfg)
    fallback.llm.provider = "mock"
    fallback.llm.model = "mock-rubric-v1"
    fallback.llm.max_retries = 0
    fallback.llm.mock_noise_sd = 0.0
    return fallback


def _result_payload(
    result: ScreeningResult,
    *,
    provider: str,
    model: str,
    fallback_used: bool,
    fallback_reason: str = "",
) -> Dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "overall_score": result.overall_score,
        "recommendation": result.recommendation,
        "confidence": result.confidence,
        "summary": result.summary,
        "dimensions": [
            {
                "name": item.name,
                "score": item.score,
                "weight": item.weight,
                "rationale": item.rationale,
                "evidence": [
                    {
                        "quote": evidence.quote,
                        "supports": evidence.supports,
                        "found_in_resume": evidence.found_in_resume,
                    }
                    for evidence in item.evidence
                ],
            }
            for item in result.dimensions
        ],
        "missing_requirements": result.missing_requirements,
        "unverifiable_claims": result.unverifiable_claims,
        "protected_attribute_referenced": result.protected_attribute_referenced,
        "protected_reference_note": result.protected_reference_note,
    }


def evaluate_resume(resume_text: str, job_text: str, root: Path) -> Dict[str, Any]:
    """Score with the configured provider, falling back to the local scorer on failure."""
    load_dotenv_if_present()
    primary_cfg = _config(root)
    job = _job_description(job_text)
    cache_key = hashlib.sha256(
        "\x00".join((primary_cfg.llm.model, resume_text, job.text)).encode("utf-8")
    ).hexdigest()
    cached = _result_cache.get(cache_key)
    if cached is not None:
        _result_cache.move_to_end(cache_key)
        return copy.deepcopy(cached)
    primary = ResumeScorer(LLMClient.from_config(primary_cfg), primary_cfg)
    with ThreadPoolExecutor(max_workers=SCORING_SAMPLES) as pool:
        outcomes = list(
            pool.map(
                lambda index: primary.score_once(
                    resume_text, job, sample_index=index, allow_grounding_retry=False
                ),
                range(SCORING_SAMPLES),
            )
        )
    valid = sorted((o for o in outcomes if o.ok), key=lambda o: o.result.overall_score)
    # An actual outcome is reported (not a synthetic average), so the score,
    # dimensions and evidence stay consistent. With an even count the lower
    # middle result is used.
    outcome = valid[(len(valid) - 1) // 2] if valid else outcomes[0]

    if outcome.ok:
        payload = _result_payload(
            outcome.result,
            provider=primary_cfg.llm.provider,
            model=primary_cfg.llm.model,
            fallback_used=False,
        )
        _result_cache[cache_key] = copy.deepcopy(payload)
        while len(_result_cache) > _RESULT_CACHE_SIZE:
            _result_cache.popitem(last=False)
        return payload

    reason = outcome.response.error or outcome.result.parse_error or f"{primary_cfg.llm.provider} returned an unusable result."
    logger.warning("primary screening provider failed; using offline fallback: %s", reason)
    fallback_cfg = _mock_config(primary_cfg)
    fallback = ResumeScorer(LLMClient.from_config(fallback_cfg), fallback_cfg)
    fallback_outcome = fallback.score_once(resume_text, job, allow_grounding_retry=False)
    if not fallback_outcome.ok:
        logger.error("offline fallback failed after primary provider failure")
        raise RuntimeError(
            f"{primary_cfg.llm.provider} failed and offline fallback failed: {fallback_outcome.result.parse_error}"
        )
    return _result_payload(
        fallback_outcome.result,
        provider=fallback_cfg.llm.provider,
        model=fallback_cfg.llm.model,
        fallback_used=True,
        fallback_reason=reason,
    )


def evaluate_offline(resume_text: str, job_text: str, root: Path) -> Dict[str, Any]:
    """Evaluate one submission with the deterministic local provider only."""
    cfg = _mock_config(_config(root))
    job = _job_description(job_text)
    scorer = ResumeScorer(LLMClient.from_config(cfg), cfg)
    outcome = scorer.score_once(resume_text, job, allow_grounding_retry=False)
    if not outcome.ok:
        raise RuntimeError("The offline scorer could not produce a valid result.")
    return _result_payload(
        outcome.result,
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        fallback_used=False,
    )


def create_app():
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError("Web support is not installed. Run `uv sync --extra web`.") from exc

    root = Path(__file__).resolve().parents[1]
    app = FastAPI(
        title="Generative AI Powered Resume Screening and Candidate Evaluation with Fairness Assessment",
        version="1.0.0",
    )

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        load_dotenv_if_present()
        try:
            provider = _provider_name()
        except ValueError:
            provider = "unconfigured"
        return {"status": "ok", "service": "fair-resume-screening", "provider": provider}

    @app.post("/api/screen")
    async def screen_resume(
        resume: UploadFile = File(...),
        job_description: str = Form(...),
    ) -> Dict[str, Any]:
        try:
            content = await resume.read()
            text = extract_resume_text(resume.filename or "resume.txt", content)
            return evaluate_resume(text, job_description, root)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("unexpected screening request failure")
            raise HTTPException(
                status_code=502,
                detail="The screening service could not produce a result. Try again.",
            ) from exc

    @app.post("/api/screen/offline")
    async def screen_offline(
        resume: UploadFile = File(...),
        job_description: str = Form(...),
    ) -> Dict[str, Any]:
        try:
            content = await resume.read()
            text = extract_resume_text(resume.filename or "resume.txt", content)
            return evaluate_offline(text, job_description, root)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("unexpected offline screening failure")
            raise HTTPException(
                status_code=502,
                detail="The offline scorer could not produce a result. Try again.",
            ) from exc

    web_root = _web_root()
    app.mount("/assets", StaticFiles(directory=web_root / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(web_root / "index.html")

    return app


app = create_app()
