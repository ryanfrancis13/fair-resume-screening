"""Shared helpers: slot names, rendering, skill detection."""

import re

SLOTS = [
    "NAME",
    "EMAIL",
    "PHONE",
    "LOCATION",
    "LINKEDIN",
    "PRONOUN_SUBJ",
    "PRONOUN_POSS",
    "AFFILIATION",
    "LANGUAGES",
    "CULTURAL_ACTIVITY",
    "GRAD_YEAR",
    "EARLY_CAREER_NOTE",
    "HOBBIES",
    "FUNDING_NOTE",
    "UNIVERSITY",
]
SLOT_RE = re.compile(r"\{\{([A-Z_]+)\}\}")
CURRENT_YEAR = 2026


def clean(text):
    """Idempotent: strip trailing spaces per line, strip ends. Never merges
    or drops lines, so template line i always maps to rendered line i."""
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def render(template, slots):
    return clean(SLOT_RE.sub(lambda m: str(slots[m.group(1)]), template))


def skill_pattern(skill):
    return re.compile(r"(?<![\w])" + re.escape(skill) + r"(?![\w])", re.I)


def skills_in_text(text, vocab):
    return {s for s in vocab if skill_pattern(s).search(text)}
