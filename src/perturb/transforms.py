"""Content-preserving text transforms for cue families that rewrite prose.

Slot substitution cannot express every cue.  Writing register, for instance, is
a property of *how* achievements are phrased, not of any single field.  The
transforms here rewrite phrasing while leaving every fact intact, and each one
declares what it guarantees so the validator can check the guarantee rather than
trust it.

The invariants every transform must preserve:

* all numbers and units (``42%``, ``30 services``, ``2019 - Present``)
* all named entities: employers, universities, technologies
* the set of skill tokens
* the section structure

A deliberately *rule-based* implementation.  Asking an LLM to paraphrase would
produce more natural prose but would silently change qualifications — the
paraphraser's own biases would then be inside the independent variable, which is
the one place a bias study cannot afford them.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Dict, List, Tuple

# --------------------------------------------------------------------------- #
# writing register
# --------------------------------------------------------------------------- #
# Each rule is (pattern, replacement). Patterns are anchored on verb phrases at
# the start of a bullet so that numbers and entity names are never touched.
_HEDGE_RULES: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"(?m)^(\s*-\s*)Led\b", re.IGNORECASE), r"\1Helped lead"),
    (re.compile(r"(?m)^(\s*-\s*)Built\b", re.IGNORECASE), r"\1Contributed to building"),
    (re.compile(r"(?m)^(\s*-\s*)Designed\b", re.IGNORECASE), r"\1Was involved in designing"),
    (re.compile(r"(?m)^(\s*-\s*)Reduced\b", re.IGNORECASE), r"\1Helped reduce"),
    (re.compile(r"(?m)^(\s*-\s*)Increased\b", re.IGNORECASE), r"\1Helped increase"),
    (re.compile(r"(?m)^(\s*-\s*)Improved\b", re.IGNORECASE), r"\1Helped improve"),
    (re.compile(r"(?m)^(\s*-\s*)Delivered\b", re.IGNORECASE), r"\1Assisted in delivering"),
    (re.compile(r"(?m)^(\s*-\s*)Owned\b", re.IGNORECASE), r"\1Was responsible for"),
    (re.compile(r"(?m)^(\s*-\s*)Drove\b", re.IGNORECASE), r"\1Supported"),
    (re.compile(r"(?m)^(\s*-\s*)Automated\b", re.IGNORECASE), r"\1Helped automate"),
    (re.compile(r"(?m)^(\s*-\s*)Migrated\b", re.IGNORECASE), r"\1Took part in migrating"),
    (re.compile(r"(?m)^(\s*-\s*)Launched\b", re.IGNORECASE), r"\1Helped launch"),
    (re.compile(r"(?m)^(\s*-\s*)Scaled\b", re.IGNORECASE), r"\1Helped scale"),
    (re.compile(r"(?m)^(\s*-\s*)Cut\b"), r"\1Helped cut"),
    (re.compile(r"(?m)^(\s*-\s*)Shipped\b", re.IGNORECASE), r"\1Helped ship"),
)

_ASSERTIVE_RULES: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"(?m)^(\s*-\s*)Helped lead\b", re.IGNORECASE), r"\1Spearheaded"),
    (re.compile(r"(?m)^(\s*-\s*)Helped\s+(\w)", re.IGNORECASE),
     lambda m: f"{m.group(1)}{m.group(2).upper()}"),  # type: ignore[arg-type]
    (re.compile(r"(?m)^(\s*-\s*)Worked on\b", re.IGNORECASE), r"\1Delivered"),
    (re.compile(r"(?m)^(\s*-\s*)Assisted in\s+(\w+)ing\b", re.IGNORECASE), r"\1Drove \2ing"),
    (re.compile(r"(?m)^(\s*-\s*)Was responsible for\b", re.IGNORECASE), r"\1Owned"),
    (re.compile(r"(?m)^(\s*-\s*)Contributed to\s+(\w+)ing\b", re.IGNORECASE), r"\1Led \2ing"),
    (re.compile(r"(?m)^(\s*-\s*)Took part in\s+(\w+)ing\b", re.IGNORECASE), r"\1Executed the \2ing of"),
    (re.compile(r"(?m)^(\s*-\s*)Supported\b", re.IGNORECASE), r"\1Drove"),
    (re.compile(r"(?m)^(\s*-\s*)Led\b"), r"\1Spearheaded"),
    (re.compile(r"(?m)^(\s*-\s*)Built\b"), r"\1Architected and built"),
    (re.compile(r"(?m)^(\s*-\s*)Improved\b"), r"\1Drove a step-change improvement in"),
)


def _apply_rules(text: str, rules) -> str:
    for pattern, repl in rules:
        text = pattern.sub(repl, text)
    return text


def writing_register(text: str, level: str) -> str:
    """Rewrite achievement phrasing into ``plain``, ``hedged`` or ``assertive``.

    ``plain`` is the identity transform: the reference level must be the text
    exactly as it was, so that any measured register effect is relative to the
    document as actually written.
    """
    if level in ("plain", "", "reference"):
        return text
    if level == "hedged":
        return _apply_rules(text, _HEDGE_RULES)
    if level == "assertive":
        return _apply_rules(text, _ASSERTIVE_RULES)
    raise KeyError(f"unknown writing_register level {level!r}")


# --------------------------------------------------------------------------- #
# date shifting (kept available for corpora without a GRAD_YEAR slot)
# --------------------------------------------------------------------------- #
_YEAR_RE = re.compile(r"\b(19[7-9]\d|20[0-3]\d)\b")


def shift_years(text: str, delta: int, *, keep_present: bool = True) -> str:
    """Shift every 4-digit year by ``delta``, preserving all durations.

    Because both endpoints of every range move together, ``2019 - 2023`` stays a
    four-year role.  That is the property that lets apparent age be varied
    without changing the length of any listed position — although, as
    ``age_marker`` documents, total career length still shifts.
    """
    if delta == 0:
        return text

    def repl(match: re.Match) -> str:
        year = int(match.group(1))
        shifted = year + delta
        if keep_present and shifted > 2035:
            return match.group(1)
        return str(shifted)

    return _YEAR_RE.sub(repl, text)


# --------------------------------------------------------------------------- #
# invariant extraction, used by the validator
# --------------------------------------------------------------------------- #
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Za-z0-9+#.]{1,}\b")
_LINE_INITIAL_RE = re.compile(r"(?m)^(?:\s*-\s*)?([A-Z][A-Za-z0-9+#.]{1,})")


def numeric_fingerprint(text: str) -> List[str]:
    """Multiset of numbers, so a transform cannot quietly change a metric."""
    return sorted(_NUMBER_RE.findall(text))


def capitalised_fingerprint(text: str, *, skip_line_initial: bool = False) -> List[str]:
    """Multiset of capitalised tokens, a cheap stand-in for named entities.

    ``skip_line_initial`` drops the first capitalised word of each line and
    bullet. A register rewrite legitimately replaces the leading verb of a bullet
    ("Led" -> "Helped lead"), and counting that as a lost named entity would
    reject every valid style transform. Employers and technologies never appear
    only in line-initial position, so nothing that matters is skipped.
    """
    found = _ACRONYM_RE.findall(text)
    if not skip_line_initial:
        return sorted(found)
    initial = Counter(_LINE_INITIAL_RE.findall(text))
    remaining: List[str] = []
    for token in found:
        if initial.get(token, 0) > 0:
            initial[token] -= 1
            continue
        remaining.append(token)
    return sorted(remaining)


#: Every surface word the register rules can insert or remove.
#:
#: Written out explicitly rather than scraped from the regex sources. Scraping
#: the patterns yields regex machinery ("IGNORECASE", stray single letters) and
#: silently drops the short function words the replacements introduce ("in",
#: "to", "of", "a"), which then read as unexplained content changes to the
#: perturbation validator. ``test_transform_vocabulary_is_complete`` applies
#: every transform to the real corpus and asserts this list covers it, so the two
#: cannot drift apart.
_TRANSFORM_WORDS: Tuple[str, ...] = (
    # verbs the rules match
    "led", "built", "designed", "reduced", "increased", "improved", "delivered",
    "owned", "drove", "automated", "migrated", "launched", "scaled", "cut",
    "shipped", "worked", "supported", "contributed", "assisted", "took",
    # verbs and phrases the rules insert
    "helped", "lead", "building", "designing", "reduce", "increase", "improve",
    "delivering", "responsible", "automate", "migrating", "launch", "scale",
    "ship", "spearheaded", "architected", "executed", "improvement",
    "step", "change", "involved", "part",
    # function words carried by those phrases
    "was", "to", "in", "for", "on", "of", "the", "and", "a",
)


def transform_vocabulary() -> List[str]:
    """Words the register transforms are allowed to introduce or remove.

    Declared as data so the perturbation validator can treat these as
    accounted-for rather than as unexplained content changes.
    """
    return sorted(_TRANSFORM_WORDS)


TRANSFORMS: Dict[str, Callable[[str, str], str]] = {
    "writing_register": writing_register,
}


def apply_transform(name: str, text: str, level: str) -> str:
    if name not in TRANSFORMS:
        raise KeyError(f"unknown transform {name!r}; available: {sorted(TRANSFORMS)}")
    return TRANSFORMS[name](text, level)
