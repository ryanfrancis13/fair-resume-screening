"""Text normalisation and lightweight extraction used across layers.

Deliberately dependency-free: no spaCy or NLTK.  Everything here is either a
regex or simple counting, which keeps the pipeline installable on a laptop and
keeps the *fairness* logic auditable — a reviewer can read these forty lines and
know exactly how a "skill match" was counted.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Sequence, Set, Tuple

_WS_RUN = re.compile(r"[ \t ]+")
_BLANK_RUN = re.compile(r"\n{3,}")
_BULLET = re.compile(r"^[\s]*[•‣▪●·\-\*–—o]\s+", re.MULTILINE)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")
_YEAR = re.compile(r"\b(19[7-9]\d|20[0-4]\d)\b")

# Common scraping artifacts in public resume corpora (Kaggle "Resume Dataset"
# and friends): repeated section headers, "Read more" links, page furniture.
_ARTIFACTS: Tuple[re.Pattern, ...] = (
    re.compile(r"(?i)\bread\s*more\b"),
    re.compile(r"(?i)\bpage\s+\d+\s+of\s+\d+\b"),
    re.compile(r"(?i)\bindeed\.com\b|\bnaukri\.com\b|\bresume\s*sample\b"),
    re.compile(r"(?i)company\s+name\s*[-:]\s*"),
    re.compile(r"\b[Cc]lick here\b"),
)


def normalize_unicode(text: str) -> str:
    """NFKC-normalise and fold exotic dashes/quotes to ASCII equivalents."""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "...", " ": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def clean_text(text: str, *, strip_artifacts: bool = True) -> str:
    """Canonicalise raw resume/JD text without changing its content.

    Cleaning must be *idempotent* and content-preserving: the perturbation
    validator later asserts that a variant differs from its baseline only in the
    intended spans, and that assertion is meaningless if cleaning is applied
    inconsistently between the two.
    """
    text = normalize_unicode(text)
    text = _CTRL.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if strip_artifacts:
        for pattern in _ARTIFACTS:
            text = pattern.sub(" ", text)
    text = _BULLET.sub("- ", text)
    text = _WS_RUN.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def tokens(text: str) -> List[str]:
    """Lowercased alphanumeric tokens, keeping ``c++``/``node.js``/``ci-cd`` intact."""
    return _TOKEN.findall(text.lower())


def token_set(text: str) -> Set[str]:
    return set(tokens(text))


def ngrams(seq: Sequence[str], n: int) -> List[str]:
    return [" ".join(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def phrase_set(text: str, max_n: int = 3) -> Set[str]:
    """Unigrams through ``max_n``-grams, for multi-word skills like 'apache spark'."""
    toks = tokens(text)
    out: Set[str] = set(toks)
    for n in range(2, max_n + 1):
        out.update(ngrams(toks, n))
    return out


def years_mentioned(text: str) -> List[int]:
    return [int(y) for y in _YEAR.findall(text)]


def word_count(text: str) -> int:
    return len(tokens(text))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def char_diff_spans(a: str, b: str) -> List[Tuple[int, int, str, str]]:
    """Contiguous differing regions between ``a`` and ``b``.

    Returned as ``(start_in_a, end_in_a, text_in_a, text_in_b)``.  Used by the
    perturbation validator to prove that only the intended cue spans changed.
    """
    import difflib

    spans: List[Tuple[int, int, str, str]] = []
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            spans.append((i1, i2, a[i1:i2], b[j1:j2]))
    return spans


def normalized_edit_ratio(a: str, b: str) -> float:
    """Fraction of characters that differ, in ``[0, 1]``."""
    import difflib

    if not a and not b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    return 1.0 - ratio


def fuzzy_contains(haystack: str, needle: str, threshold: float = 0.85) -> bool:
    """True if ``needle`` appears in ``haystack`` allowing whitespace/case drift.

    Evidence quotes returned by an LLM are frequently re-wrapped or lightly
    re-punctuated versions of the source line.  A strict substring test would
    reject valid evidence and make the grounding metric measure formatting
    fidelity instead of factual grounding.
    """
    import difflib

    hay = " ".join(haystack.lower().split())
    nee = " ".join(needle.lower().split())
    if not nee:
        return False
    if nee in hay:
        return True
    window = len(nee)
    if window > len(hay):
        return difflib.SequenceMatcher(None, hay, nee).ratio() >= threshold
    step = max(1, window // 4)
    best = 0.0
    for start in range(0, len(hay) - window + 1, step):
        chunk = hay[start : start + window]
        best = max(best, difflib.SequenceMatcher(None, chunk, nee).ratio())
        if best >= threshold:
            return True
    return best >= threshold


def truncate(text: str, max_chars: int, marker: str = "\n...[truncated]...") -> str:
    """Truncate on a line boundary so a cut never lands mid-sentence."""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars - len(marker)]
    cut = head.rfind("\n")
    if cut > max_chars * 0.6:
        head = head[:cut]
    return head + marker


def sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def count_lexicon_hits(text: str, lexicon: Iterable[str]) -> Dict[str, int]:
    """Per-term occurrence counts, matched on word boundaries, case-insensitively."""
    low = text.lower()
    hits: Dict[str, int] = {}
    for term in lexicon:
        t = term.lower().strip()
        if not t:
            continue
        pattern = r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b"
        n = len(re.findall(pattern, low))
        if n:
            hits[term] = n
    return hits
