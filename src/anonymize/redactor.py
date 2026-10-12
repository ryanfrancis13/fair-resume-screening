"""Pre-inference redaction, plus a measurement of what redaction misses.

Blind screening is the oldest fairness intervention in hiring and the one most
often claimed by vendors.  This module implements it at three strengths *and*
measures its residual leakage, because the interesting empirical question is not
"did we redact the name" but "what demographic signal survived".

    identity              name, email, phone, handle, city, exact dates
    identity_plus_proxy   the above, plus pronouns, marked affiliations, spoken
                          languages, cultural activity, leisure and funding notes
    aggressive            the above, plus institution names and absolute years

The pipeline treats redaction as a *mitigation arm*: candidates are scored with
and without it, and :mod:`frs.mitigate.evaluate` reports both the bias reduction
and the cost in ranking agreement.  A blinding that removes bias by destroying
the signal the score needs is not a win, and the only way to see that trade-off
is to measure both sides.

:func:`leakage_report` is what makes this more than a regex pass.  It re-runs the
cue detector over the redacted text and counts what is still findable, which is
the direct empirical answer to the project's core claim that name-level
anonymisation is insufficient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..schemas import Resume
from ..utils.text import clean_text, count_lexicon_hits
from .lexicons import (
    DISABILITY_TERMS,
    FUNDING_TERMS,
    GENDERED_NOUNS,
    GENDERED_PRONOUNS,
    HONORIFIC_PATTERN,
    LANGUAGE_NAMES,
    LEISURE_TERMS,
    MARKED_ORGANISATIONS,
    RELIGION_TERMS,
)

LEVELS = ("identity", "identity_plus_proxy", "aggressive")

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}")
_LINKEDIN_RE = re.compile(r"(?:linkedin\.com/in/|github\.com/)[\w-]+", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_YEAR_RE = re.compile(r"\b(19[7-9]\d|20[0-3]\d)\b")
_HONORIFIC_RE = re.compile(HONORIFIC_PATTERN)
#: "Boston, MA" or "Lagos, Nigeria". A bare ``Word, Word`` pattern would also
#: match "Senior Engineer, Northwind Payments", so the second element must be a
#: US state code or a known country, and the match is confined to the header.
_US_STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"]
_COUNTRIES = (
    ["India", "Nigeria", "Philippines", "Kenya", "Ghana", "Pakistan", "Bangladesh", "Canada", "Mexico", "Brazil", "Germany", "France", "Spain", "Italy", "Poland", "Ukraine", "Russia", "China", "Japan", "Korea", "Vietnam", "Thailand", "Indonesia", "Malaysia", "Singapore", "Australia", "Ireland", "Netherlands", "Sweden", "Norway", "Denmark", "Egypt", "Morocco", "Jordan", "Lebanon", "Turkey", "Israel", "Chile", "Colombia", "Argentina", "Peru"]
)
_CITY_STATE_RE = re.compile(
    r"\b[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)?,\s*(?:"
    + "|".join(_US_STATES)
    + r"|United States|United Kingdom|"
    + "|".join(_COUNTRIES)
    + r")\b"
)
#: Hard cap on how far the contact header can extend.
_MAX_HEADER_LINES = 6
_UNIVERSITY_RE = re.compile(
    r"\b(?:University|Universidad|Institute|College|Polytechnic|Academy)\b[^,\n]{0,60}"
    r"|\b[A-Z][A-Za-z.]+(?:\s[A-Z][A-Za-z.]+){0,3}\s(?:University|College|Institute)\b"
)

#: Section headers whose whole line is dropped at proxy level. Redacting the
#: value but keeping "Interests:" would itself signal that something was removed.
_PROXY_LINE_PREFIXES = (
    "affiliations:", "spoken languages:", "languages spoken:", "community:",
    "interests:", "hobbies:", "activities:", "extracurricular", "volunteer:",
    "personal:", "religion:", "nationality:", "marital status:", "date of birth:",
    "gender:", "sex:", "age:", "photo:",
)


@dataclass
class RedactionResult:
    """Redacted text plus an account of what was removed."""

    text: str
    level: str
    n_substitutions: int = 0
    removed_lines: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    residual: Dict[str, Any] = field(default_factory=dict)


def _sub_terms(text: str, terms: Sequence[str], placeholder: str) -> Tuple[str, int]:
    """Word-boundary, case-insensitive replacement of a term list."""
    count = 0
    for term in sorted(terms, key=len, reverse=True):
        term = term.strip()
        if not term:
            continue
        pattern = re.compile(r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE)
        text, n = pattern.subn(placeholder, text)
        count += n
    return text, count


def redact(
    text: str,
    level: str = "identity",
    *,
    extra_names: Optional[Sequence[str]] = None,
    known_name: Optional[str] = None,
    measure_residual: bool = True,
) -> RedactionResult:
    """Redact ``text`` at the given strength.

    ``known_name`` is the exact name string when the caller knows it (templated
    corpus). Relying only on regex name detection would leave the name in place
    for any resume whose header does not match the expected shape, which is the
    silent failure mode a blind-screening claim cannot afford.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown redaction level {level!r}; expected one of {LEVELS}")

    counts: Dict[str, int] = {}
    total = 0
    removed_lines = 0

    # ---- identity ---------------------------------------------------------
    # Order matters. Structured contact fields are replaced first: redacting the
    # name earlier would turn "jordan.avery@example.com" into
    # "[NAME].[NAME]@example.com", leaking the fact that the local part was
    # derived from the name and leaving a malformed address behind.
    text, n = _LINKEDIN_RE.subn("[HANDLE]", text)
    counts["handle"] = n
    total += n
    text, n = _EMAIL_RE.subn("[EMAIL]", text)
    counts["email"] = n
    total += n
    text, n = _URL_RE.subn("[URL]", text)
    counts["url"] = n
    total += n
    text, n = _PHONE_RE.subn("[PHONE]", text)
    counts["phone"] = n
    total += n
    text, n = _redact_header_locations(text)
    counts["location"] = n
    total += n
    text, n = _HONORIFIC_RE.subn("", text)
    counts["honorific"] = n
    total += n

    if known_name:
        parts = [known_name] + [p for p in known_name.split() if len(p) > 2]
        text, n = _sub_terms(text, parts, "[NAME]")
        counts["name"] = n
        total += n
    if extra_names:
        text, n = _sub_terms(text, extra_names, "[NAME]")
        counts["name"] = counts.get("name", 0) + n
        total += n
    if not known_name:
        text, n = _redact_header_name(text)
        counts["name"] = counts.get("name", 0) + n
        total += n

    # ---- identity_plus_proxy ---------------------------------------------
    if level in ("identity_plus_proxy", "aggressive"):
        kept_lines: List[str] = []
        for line in text.split("\n"):
            if line.strip().lower().startswith(_PROXY_LINE_PREFIXES):
                removed_lines += 1
                continue
            kept_lines.append(line)
        text = _drop_empty_sections("\n".join(kept_lines))

        for label, terms, placeholder in (
            ("pronoun", GENDERED_PRONOUNS, "[PRONOUN]"),
            ("gendered_noun", GENDERED_NOUNS, "[REDACTED]"),
            ("marked_org", MARKED_ORGANISATIONS, "[ORGANISATION]"),
            ("disability", DISABILITY_TERMS, "[REDACTED]"),
            ("religion", RELIGION_TERMS, "[REDACTED]"),
            ("language", LANGUAGE_NAMES, "[LANGUAGE]"),
            ("leisure", LEISURE_TERMS, "[INTEREST]"),
            ("funding", FUNDING_TERMS, "[REDACTED]"),
        ):
            text, n = _sub_terms(text, terms, placeholder)
            counts[label] = n
            total += n

    # ---- aggressive -------------------------------------------------------
    if level == "aggressive":
        text, n = _UNIVERSITY_RE.subn("[INSTITUTION]", text)
        counts["institution"] = n
        total += n
        # Absolute years leak apparent age; durations are what the rubric needs.
        text, n = _YEAR_RE.subn("[YEAR]", text)
        counts["year"] = n
        total += n

    text = clean_text(text)
    result = RedactionResult(
        text=text,
        level=level,
        n_substitutions=total,
        removed_lines=removed_lines,
        by_category={k: v for k, v in counts.items() if v},
    )
    if measure_residual:
        result.residual = leakage_report(text)
    return result


_SECTION_HEADER_RE = re.compile(r"^[A-Z][A-Z \t&/']{3,40}$")


def _drop_empty_sections(text: str) -> str:
    """Remove a section header whose every content line was redacted away.

    A dangling "ADDITIONAL INFORMATION" with nothing under it advertises that
    something was removed, which is itself a signal — and it makes the redacted
    document look sloppier than the unredacted one, confounding any comparison
    of the two on presentation quality.
    """
    lines = text.split("\n")
    keep = [True] * len(lines)
    for i, line in enumerate(lines):
        if not _SECTION_HEADER_RE.match(line.strip()):
            continue
        has_content = False
        for following in lines[i + 1 :]:
            stripped = following.strip()
            if not stripped:
                continue
            if _SECTION_HEADER_RE.match(stripped):
                break
            has_content = True
            break
        if not has_content:
            keep[i] = False
    return "\n".join(line for line, k in zip(lines, keep) if k)


def _redact_header_locations(text: str) -> Tuple[str, int]:
    """Replace city/region strings, but only inside the contact header.

    A resume body legitimately contains place names ("migrated the Frankfurt
    region"); redacting those would damage content without removing a cue that
    the header has already given away.
    """
    lines = text.split("\n")
    total = 0
    for i in range(min(_MAX_HEADER_LINES, len(lines))):
        # The contact block is whatever precedes the first blank line. Using a
        # fixed line count instead would redact place names from the opening of
        # the body ("Migrated the Frankfurt, Germany region"), damaging content
        # to remove a cue the header has already given away.
        if not lines[i].strip():
            break
        lines[i], n = _CITY_STATE_RE.subn("[LOCATION]", lines[i])
        total += n
    return "\n".join(lines), total


def _redact_header_name(text: str) -> Tuple[str, int]:
    """Redact the first line if it looks like a personal name."""
    lines = text.split("\n")
    for i, line in enumerate(lines[:4]):
        stripped = line.strip()
        if not stripped or len(stripped) > 45 or "@" in stripped:
            continue
        words = stripped.replace(".", "").split()
        if 2 <= len(words) <= 4 and all(w[:1].isupper() for w in words if w):
            lines[i] = "[NAME]"
            return "\n".join(lines), 1
    return text, 0


def redact_resume(
    resume: Resume, level: str = "identity", *, measure_residual: bool = True
) -> RedactionResult:
    """Redact a resume, using its declared name slot when available."""
    known_name = resume.slots.get("NAME") if resume.slots else None
    return redact(resume.text, level, known_name=known_name, measure_residual=measure_residual)


# --------------------------------------------------------------------------- #
# leakage measurement — the empirical case against name-only anonymisation
# --------------------------------------------------------------------------- #
def leakage_report(text: str) -> Dict[str, Any]:
    """Count demographic signal still detectable after redaction.

    Runs the perturbation layer's own cue detector over the redacted text, so the
    question asked is exactly the one the audit cares about: could this document
    still be assigned to a demographic group by the same lexicons the experiment
    manipulates?
    """
    from ..llm.mock import SignalDetector

    detector = SignalDetector()
    # A level of "unspecified" (singular *they*, for instance) carries no group
    # information, so counting it as leakage would overstate what survived.
    detected = {
        attribute: level
        for attribute, level in detector.detect(text).items()
        if level and level.lower() not in ("unspecified", "unknown", "none")
    }

    categories = {
        "gendered_pronoun": count_lexicon_hits(text, GENDERED_PRONOUNS),
        "gendered_noun": count_lexicon_hits(text, GENDERED_NOUNS),
        "marked_organisation": count_lexicon_hits(text, MARKED_ORGANISATIONS),
        "disability_term": count_lexicon_hits(text, DISABILITY_TERMS),
        "religion_term": count_lexicon_hits(text, RELIGION_TERMS),
        "spoken_language": count_lexicon_hits(text, LANGUAGE_NAMES),
        "leisure_signal": count_lexicon_hits(text, LEISURE_TERMS),
        "funding_signal": count_lexicon_hits(text, FUNDING_TERMS),
    }
    total_hits = sum(sum(hits.values()) for hits in categories.values())
    return {
        "detected_attributes": detected,
        "n_detected_attributes": len(detected),
        "residual_cue_hits": total_hits,
        "residual_by_category": {k: sum(v.values()) for k, v in categories.items() if v},
        "residual_terms": {k: sorted(v)[:6] for k, v in categories.items() if v},
        "clean": total_hits == 0 and not detected,
    }


def compare_levels(resume: Resume) -> List[Dict[str, Any]]:
    """Leakage at every redaction level, for the report's blinding table."""
    rows: List[Dict[str, Any]] = [
        {
            "level": "none",
            "n_substitutions": 0,
            "residual_cue_hits": leakage_report(resume.text)["residual_cue_hits"],
            "n_detected_attributes": leakage_report(resume.text)["n_detected_attributes"],
            "chars": len(resume.text),
        }
    ]
    for level in LEVELS:
        result = redact_resume(resume, level)
        rows.append(
            {
                "level": level,
                "n_substitutions": result.n_substitutions,
                "removed_lines": result.removed_lines,
                "residual_cue_hits": result.residual.get("residual_cue_hits", 0),
                "n_detected_attributes": result.residual.get("n_detected_attributes", 0),
                "chars": len(result.text),
            }
        )
    return rows
