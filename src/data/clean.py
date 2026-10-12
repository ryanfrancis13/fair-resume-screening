"""Corpus-level cleaning and quality gates.

Cleaning runs *before* perturbation, never between baseline and variant.  If a
baseline were cleaned and its variant were not, the diff would contain
whitespace noise and the validator's span accounting would fail for reasons that
have nothing to do with the cue.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

from ..logging_setup import log_event
from ..schemas import JobDescription, Resume
from ..utils.text import clean_text, word_count

logger = logging.getLogger("frs.data.clean")


def clean_resumes(
    resumes: Sequence[Resume], *, min_words: int = 60, max_words: int = 2500
) -> Tuple[List[Resume], List[Dict[str, Any]]]:
    """Normalise text and drop documents outside a usable length band.

    Returns ``(kept, dropped)``. Exclusions are returned rather than logged and
    forgotten, so the run manifest can state how many documents the corpus
    actually contributed.
    """
    kept: List[Resume] = []
    dropped: List[Dict[str, Any]] = []
    for resume in resumes:
        text = clean_text(resume.text)
        words = word_count(text)
        if words < min_words:
            dropped.append({"resume_id": resume.resume_id, "reason": "too_short", "words": words})
            continue
        if words > max_words:
            dropped.append({"resume_id": resume.resume_id, "reason": "too_long", "words": words})
            continue
        resume.text = text
        kept.append(resume)
    if dropped:
        log_event(logger, logging.INFO, "dropped resumes during cleaning",
                  n_dropped=len(dropped), n_kept=len(kept))
    return kept, dropped


def clean_jobs(jobs: Sequence[JobDescription], *, min_words: int = 30) -> List[JobDescription]:
    out: List[JobDescription] = []
    for job in jobs:
        job.text = clean_text(job.text)
        if word_count(job.text) < min_words:
            log_event(logger, logging.WARNING, "job description is very short",
                      job_id=job.job_id, words=word_count(job.text))
        out.append(job)
    return out


def duplicate_report(resumes: Sequence[Resume]) -> Dict[str, Any]:
    """Detect exact and near-duplicate resumes.

    Public resume corpora contain duplicates. A duplicated resume enters the
    design as two "independent" blocks when it is really one, which understates
    the standard errors of every group comparison downstream.
    """
    from ..utils.hashing import hash_text
    from ..utils.text import jaccard, token_set

    exact: Dict[str, List[str]] = {}
    for resume in resumes:
        exact.setdefault(hash_text(resume.text), []).append(resume.resume_id)
    exact_groups = [ids for ids in exact.values() if len(ids) > 1]

    near: List[Dict[str, Any]] = []
    signatures = [(r.resume_id, token_set(r.text)) for r in resumes]
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            score = jaccard(signatures[i][1], signatures[j][1])
            if score >= 0.9:
                near.append(
                    {"a": signatures[i][0], "b": signatures[j][0], "jaccard": round(score, 3)}
                )
    return {
        "n_exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_groups": exact_groups[:20],
        "n_near_duplicate_pairs": len(near),
        "near_duplicate_pairs": near[:20],
        "note": (
            "Duplicates inflate apparent sample size: two copies of one resume are "
            "one statistical block, not two. Remove them before reporting."
        ),
    }
