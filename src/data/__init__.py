"""Corpus acquisition, cleaning and pairing."""

from .clean import clean_jobs, clean_resumes, duplicate_report
from .loaders import corpus_stats, load_corpus
from .pairing import build_pairs, pairing_report

__all__ = [
    "build_pairs",
    "clean_jobs",
    "clean_resumes",
    "corpus_stats",
    "duplicate_report",
    "load_corpus",
    "pairing_report",
]
