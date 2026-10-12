"""Resume-to-job pairing.

Pairing is a methodological choice, not glue code.  The brief is explicit that
resumes must be paired with job descriptions *sensibly* rather than at random,
and the reason is statistical: if every candidate is a poor fit, every score sits
near the floor, variance collapses, and a cue effect has no room to appear. A
null result from random pairing would be indistinguishable from a fair model.

The rule here is job-family matching, with a bounded number of deliberate
cross-family pairs kept as negative controls — they are what proves the scorer
discriminates on fit at all.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..schemas import JobDescription, Pair, Resume
from ..utils.text import jaccard, phrase_set


def build_pairs(
    resumes: Sequence[Resume],
    jobs: Sequence[JobDescription],
    *,
    max_per_job: int = 40,
    include_negative_controls: bool = True,
    negative_controls_per_job: int = 1,
) -> List[Pair]:
    """Pair within job family, ranked by lexical overlap with the posting.

    Overlap ranking matters when a family has more resumes than
    ``max_per_job``: keeping the closest matches preserves the score variance the
    fairness tests need, whereas an arbitrary truncation could leave a job with
    only floor-scoring candidates.
    """
    pairs: List[Pair] = []
    counter = 0

    by_family: Dict[str, List[Resume]] = {}
    for resume in resumes:
        by_family.setdefault(resume.job_family, []).append(resume)

    for job in jobs:
        job_terms = phrase_set(job.text + " " + " ".join(job.must_have), max_n=2)
        candidates = by_family.get(job.job_family, [])
        ranked = sorted(
            candidates,
            key=lambda r: (-jaccard(phrase_set(r.text, 2), job_terms), r.resume_id),
        )[:max_per_job]
        for resume in ranked:
            counter += 1
            pairs.append(
                Pair(
                    pair_id=f"P{counter:04d}",
                    job_id=job.job_id,
                    resume_id=resume.resume_id,
                    pairing_rationale=f"job_family match: {job.job_family}",
                    expected_fit=None,
                )
            )

        if include_negative_controls:
            others = sorted(
                (r for r in resumes if r.job_family != job.job_family),
                key=lambda r: r.resume_id,
            )
            for resume in others[:negative_controls_per_job]:
                counter += 1
                pairs.append(
                    Pair(
                        pair_id=f"P{counter:04d}",
                        job_id=job.job_id,
                        resume_id=resume.resume_id,
                        pairing_rationale=(
                            f"cross-family negative control "
                            f"({resume.job_family} resume vs {job.job_family} role)"
                        ),
                        expected_fit="weak",
                    )
                )
    return pairs


def pairing_report(
    pairs: Sequence[Pair], resumes: Sequence[Resume], jobs: Sequence[JobDescription]
) -> Dict[str, object]:
    """Coverage check: every job and resume should appear at least once."""
    job_ids = {j.job_id for j in jobs}
    resume_ids = {r.resume_id for r in resumes}
    paired_jobs = {p.job_id for p in pairs}
    paired_resumes = {p.resume_id for p in pairs}
    per_job: Dict[str, int] = {}
    for p in pairs:
        per_job[p.job_id] = per_job.get(p.job_id, 0) + 1
    return {
        "n_pairs": len(pairs),
        "jobs_without_pairs": sorted(job_ids - paired_jobs),
        "resumes_without_pairs": sorted(resume_ids - paired_resumes),
        "pairs_per_job": per_job,
        "n_negative_controls": sum(1 for p in pairs if "negative control" in p.pairing_rationale),
    }
