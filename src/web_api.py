"""FastAPI single-resume evaluator serving the web workspace."""

from __future__ import annotations

import hashlib
import io
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .llm import GeminiProvider, MockProvider
from .models import Config, JobDescription, ScreeningResult, project_root
from .screening import PROMPT_VERSION, RUBRIC, score_single

MAX_UPLOAD_MB = 5
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")
SCORING_SAMPLES = 1
DECISION_LABELS = {
    "shortlist": "Shortlist",
    "hold": "Hold for Review",
    "reject": "Below Threshold",
}

_RESULT_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def extract_resume_text(filename: str, content: bytes) -> str:
    ext = Path(filename or "resume.txt").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type {ext!r}. Use {', '.join(SUPPORTED_EXTENSIONS)}.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.")

    if ext in (".txt", ".md"):
        return content.decode("utf-8", errors="replace").strip()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if ext == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    return ""


def _format_response(
    sr: ScreeningResult,
    job_title: str,
    resume_text: str,
    job_text: str,
    scoring_mode: str,
    fallback_used: bool,
    fallback_reason: str | None,
    samples_used: int,
) -> dict[str, Any]:
    dim_items = [
        {
            "name": d.dimension,
            "dimension": d.dimension,
            "score": round((d.score / 5.0) * 100.0, 1),
            "raw_score": d.score,
            "score_100": round((d.score / 5.0) * 100.0, 1),
            "weight": d.weight,
            "weight_pct": round(d.weight * 100),
            "rationale": d.rationale,
            "evidence_quotes": d.evidence_quotes,
            "evidence_verified": d.evidence_verified,
            "evidence": [
                {"quote": q, "supports": d.dimension.replace("_", " ").title(), "verified": d.evidence_verified}
                for q in d.evidence_quotes
            ],
        }
        for d in sr.dimension_scores
    ]
    return {
        "resume_id": sr.resume_id,
        "job_id": sr.job_id,
        "job_title": job_title,
        "overall_score": sr.overall_score,
        "confidence": sr.evidence_support_rate,
        "decision": sr.decision,
        "decision_label": DECISION_LABELS.get(sr.decision, sr.decision),
        "summary": sr.summary,
        "strengths": sr.strengths,
        "weaknesses": sr.weaknesses,
        "missing_requirements": sr.missing_requirements,
        "red_flags": sr.red_flags,
        "evidence_support_rate": sr.evidence_support_rate,
        "grounding_retried": sr.grounding_retried,
        "parse_valid": sr.parse_valid,
        "parse_repaired": sr.parse_repaired,
        "provider": sr.provider,
        "model": sr.model,
        "prompt_version": sr.prompt_version,
        "latency_ms": sr.latency_ms,
        "cached": sr.cached,
        "resume_char_count": len(resume_text),
        "job_char_count": len(job_text),
        "scoring_mode": scoring_mode,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "samples_used": samples_used,
        "dimensions": dim_items,
        "dimension_scores": dim_items,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="AI Resume Screening & Candidate Evaluator", version="1.0.0")
    web_dir = project_root() / "web"
    if (web_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=str(web_dir / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        gem = GeminiProvider(model=model)
        return {
            "status": "ok",
            "provider": "gemini" if gem.is_ready() else "mock",
            "model": model if gem.is_ready() else "mock-rubric-v1",
            "prompt_version": PROMPT_VERSION,
            "rubric_dimensions": RUBRIC,
            "supported_extensions": list(SUPPORTED_EXTENSIONS),
            "max_upload_mb": MAX_UPLOAD_MB,
            "live_ready": gem.is_ready(),
            "offline_ready": True,
            "scoring_samples": SCORING_SAMPLES,
        }

    async def _resolve_inputs(
        resume: UploadFile | None,
        resume_file: UploadFile | None,
        resume_text: str | None,
        job_description: str,
        job_title: str | None,
    ) -> tuple[str, str, str]:
        upload = resume or resume_file
        if upload is not None and upload.filename:
            raw_bytes = await upload.read()
            r_text = extract_resume_text(upload.filename, raw_bytes)
        else:
            r_text = (resume_text or "").strip()

        j_text = (job_description or "").strip()
        if len(r_text) < 40:
            raise HTTPException(status_code=400, detail="Resume text must be at least 40 characters.")
        if len(j_text) < 40:
            raise HTTPException(status_code=400, detail="Job description must be at least 40 characters.")
        title = (job_title or j_text.splitlines()[0][:80] or "Target Role").strip()
        return r_text, j_text, title

    @app.post("/api/screen")
    async def screen_candidate(
        resume: UploadFile | None = File(default=None),
        resume_file: UploadFile | None = File(default=None),
        resume_text: str | None = Form(default=None),
        job_description: str = Form(default=""),
        job_title: str | None = Form(default=None),
    ) -> dict[str, Any]:
        r_text, j_text, title = await _resolve_inputs(resume, resume_file, resume_text, job_description, job_title)
        cfg = Config()
        job = JobDescription(job_id="WEB_JOB", title=title, text=j_text)
        gem = GeminiProvider(model=cfg.model)

        cache_key = hashlib.sha256(f"{cfg.model}|{r_text}|{j_text}".encode()).hexdigest()
        if cache_key in _RESULT_CACHE:
            cached = dict(_RESULT_CACHE[cache_key])
            cached["cached"] = True
            return cached

        t0 = time.perf_counter()
        if gem.is_ready():
            samples: list[ScreeningResult] = []
            last_err: str | None = None
            for idx in range(SCORING_SAMPLES):
                try:
                    samples.append(score_single(r_text, job, cfg, provider=gem, sample_index=idx))
                except Exception as exc:
                    last_err = str(exc)
            valid = [s for s in samples if s.parse_valid]
            if valid:
                valid.sort(key=lambda s: s.overall_score)
                median_res = valid[(len(valid) - 1) // 2]
                median_res.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                payload = _format_response(median_res, title, r_text, j_text, "live", False, None, len(valid))
                _RESULT_CACHE[cache_key] = payload
                while len(_RESULT_CACHE) > 64:
                    _RESULT_CACHE.popitem(last=False)
                return payload
            fallback_reason = f"Live Gemini scoring failed ({last_err or 'invalid JSON'}); used offline rubric scorer."
        else:
            fallback_reason = "GEMINI_API_KEY not configured; used offline deterministic rubric scorer."

        mock_res = score_single(r_text, job, cfg, provider=MockProvider(cfg), sample_index=0)
        mock_res.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return _format_response(mock_res, title, r_text, j_text, "offline", True, fallback_reason, 1)

    @app.post("/api/screen/offline")
    async def screen_candidate_offline(
        resume: UploadFile | None = File(default=None),
        resume_file: UploadFile | None = File(default=None),
        resume_text: str | None = Form(default=None),
        job_description: str = Form(default=""),
        job_title: str | None = Form(default=None),
    ) -> dict[str, Any]:
        r_text, j_text, title = await _resolve_inputs(resume, resume_file, resume_text, job_description, job_title)
        cfg = Config(provider="mock", model="mock-rubric-v1")
        job = JobDescription(job_id="WEB_JOB", title=title, text=j_text)
        sr = score_single(r_text, job, cfg, provider=MockProvider(cfg), sample_index=0)
        return _format_response(sr, title, r_text, j_text, "offline", False, None, 1)

    return app


app = create_app()

