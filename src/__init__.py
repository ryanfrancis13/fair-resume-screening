"""Fair Resume Screening (FRS).

An LLM-based resume screening system whose fairness audit is part of the
pipeline rather than a downstream afterthought.

The package is organised as a set of layers that each own one concern:

    frs.data        corpus loading, cleaning and resume/JD pairing
    frs.llm         provider abstraction (Gemini, mock, ...), caching, retries
    frs.anonymize   pre-inference redaction of identity and proxy cues
    frs.screening   rubric-based scoring, evidence grounding, ranking
    frs.perturb     counterfactual variant generation + validity checking
    frs.fairness    multi-family fairness metrics, statistics, conflict analysis
    frs.mitigate    bias-mitigation strategies and their evaluation
    frs.pipeline    orchestration, run manifests, reproducible artifacts
    frs.report      plots, HTML/Markdown reports, explanation cards, model card

Nothing in ``frs.fairness`` knows how a score was produced, and nothing in
``frs.screening`` knows how it will be audited; they communicate only through
the tidy score table defined in :mod:`frs.schemas`.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
