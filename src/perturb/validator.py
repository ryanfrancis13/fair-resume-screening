"""Perturbation validity checking.

A counterfactual audit rests on one claim: *the only thing that changed was the
cue*.  If a name swap also dropped a bullet point, reworded a skill, or altered a
number, the measured score difference is not a bias estimate — it is an artifact.
The project's own risk register flags perturbation validity as a top risk, and
this module turns that concern into an executable assertion rather than a hope.

Four independent checks run on every variant:

1. **Span accounting.** Every differing region between baseline and variant must
   be explained by a declared edit. An unexplained diff fails the variant.
2. **Numeric fingerprint.** The multiset of numbers must be identical, so no
   metric, percentage or date silently moved. Families that legitimately change
   a number (``age_marker`` changes a graduation year) declare it.
3. **Entity fingerprint.** Capitalised tokens — employers, technologies,
   institutions — must survive, minus the ones the edit intentionally replaced.
4. **Qualification preservation.** The symmetric difference of token multisets
   must be a subset of the tokens the declared edits introduce or remove. This is
   the strongest of the four: it means no skill, tool or achievement word can
   appear or vanish without being accounted for.

Failing variants are *kept* and marked ``passed=False`` rather than dropped, and
the audit excludes them while reporting how many there were. Silently discarding
them would bias the design by removing exactly the resumes whose cue slots were
hardest to substitute cleanly.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Set

from ..schemas import Edit
from ..utils.text import char_diff_spans, normalized_edit_ratio, tokens
from .transforms import capitalised_fingerprint, numeric_fingerprint, transform_vocabulary

#: Below this many tokens a Jaccard ratio is dominated by a couple of words, so
#: the similarity floor is not applied and the exact vocabulary check stands alone.
_MIN_TOKENS_FOR_JACCARD = 60


def _expand(term: str) -> Iterable[str]:
    """A term plus the sub-parts a different tokeniser might produce from it.

    The checks below compare three different extractions of the same text —
    word tokens, numbers, and capitalised tokens — and each has its own idea of
    where a token ends. ``555-0142`` is one word token but two numbers;
    ``LGBTQ+`` keeps its plus in one and loses it in the other; ``Long-distance``
    is one word and one capitalised ``Long``. Comparing a fingerprint from one
    extractor against an allow-list built by another is exactly how a correctly
    declared edit gets reported as an unexplained content change, so every
    declared term is expanded into all the forms the extractors can yield.
    """
    term = term.strip().lower()
    if not term:
        return ()
    out = {term, term.strip(".,;:!?'\"()[]+")}
    for separator in ("-", ".", "+", "/", "_", "'"):
        if separator in term:
            out.update(part for part in term.split(separator) if part)
    return {piece for piece in out if piece}


def _explained_words(edits: Sequence[Edit], extra: Iterable[str] = ()) -> Set[str]:
    """Word forms the declared edits may add or remove."""
    allowed: Set[str] = set()
    for edit in edits:
        for piece in (edit.before, edit.after):
            for token in tokens(piece):
                allowed.update(_expand(token))
    for term in extra:
        allowed.update(_expand(term))
    return allowed


def _explained_numbers(edits: Sequence[Edit]) -> Set[str]:
    """Numeric forms the declared edits may add or remove.

    Extracted with the *same* function the numeric check uses, so a changed
    phone number is recognised as the declared edit it is.
    """
    allowed: Set[str] = set()
    for edit in edits:
        for piece in (edit.before, edit.after):
            for number in numeric_fingerprint(piece):
                allowed.update(_expand(number))
    return allowed


def _explained_entities(edits: Sequence[Edit], extra: Iterable[str] = ()) -> Set[str]:
    """Capitalised forms the declared edits may add or remove."""
    allowed: Set[str] = set()
    for edit in edits:
        for piece in (edit.before, edit.after):
            for token in capitalised_fingerprint(piece):
                allowed.update(_expand(token))
            for token in tokens(piece):
                allowed.update(_expand(token))
    for term in extra:
        allowed.update(_expand(term))
    return allowed


def _explained_strings(edits: Sequence[Edit]) -> List[str]:
    out: List[str] = []
    for edit in edits:
        for piece in (edit.before, edit.after):
            piece = piece.strip()
            if piece:
                out.append(piece)
    return out


def validate_variant(
    baseline_text: str,
    variant_text: str,
    edits: Sequence[Edit],
    *,
    is_transform: bool = False,
    allow_numeric_change: bool = False,
    max_unintended_edit_ratio: float = 0.02,
    require_skill_preservation: bool = True,
    min_token_jaccard: float = 0.85,
) -> Dict[str, Any]:
    """Run all checks and return a machine-readable verdict.

    ``is_transform`` relaxes span accounting for families that rewrite prose
    (register changes touch many spans by design) while *tightening* the
    fingerprint checks, which are what actually guarantee content preservation
    there.
    """
    failures: List[str] = []
    checks: Dict[str, Any] = {}

    # --- 0. the variant must actually differ from the baseline --------------
    identical = baseline_text == variant_text
    checks["changed"] = not identical

    # --- 1. span accounting -------------------------------------------------
    diff_spans = char_diff_spans(baseline_text, variant_text)
    checks["n_diff_spans"] = len(diff_spans)
    if not is_transform:
        allowed = _explained_strings(edits)
        unexplained: List[str] = []
        for _, _, before, after in diff_spans:
            fragment_before = before.strip()
            fragment_after = after.strip()
            if not fragment_before and not fragment_after:
                continue
            ok = any(
                (fragment_before and fragment_before in candidate)
                or (fragment_after and fragment_after in candidate)
                or (candidate and (candidate in before or candidate in after))
                for candidate in allowed
            )
            if not ok:
                unexplained.append(f"{fragment_before!r} -> {fragment_after!r}")
        checks["unexplained_spans"] = unexplained[:5]
        if unexplained:
            failures.append(f"{len(unexplained)} unexplained diff span(s)")

    # A transform's own replacement vocabulary is accounted-for by definition;
    # it is declared in frs.perturb.transforms rather than guessed at here.
    transform_terms = transform_vocabulary() if is_transform else ()

    # --- 2. numeric fingerprint --------------------------------------------
    base_numbers = numeric_fingerprint(baseline_text)
    var_numbers = numeric_fingerprint(variant_text)
    numbers_equal = base_numbers == var_numbers
    checks["numeric_fingerprint_equal"] = numbers_equal
    if not numbers_equal and not allow_numeric_change:
        removed = sorted((Counter(base_numbers) - Counter(var_numbers)).elements())
        added = sorted((Counter(var_numbers) - Counter(base_numbers)).elements())
        checks["numeric_removed"] = removed[:8]
        checks["numeric_added"] = added[:8]
        allowed_numbers = _explained_numbers(edits)
        stray = [
            n for n in removed + added
            if not (_expand(n) & allowed_numbers)
        ]
        if stray:
            failures.append(f"numeric fingerprint changed unexpectedly: {stray[:5]}")

    # --- 3. entity fingerprint ---------------------------------------------
    base_caps = Counter(capitalised_fingerprint(baseline_text, skip_line_initial=is_transform))
    var_caps = Counter(capitalised_fingerprint(variant_text, skip_line_initial=is_transform))
    allowed_entities = _explained_entities(edits, transform_terms)
    lost = [t for t in (base_caps - var_caps).elements() if not (_expand(t) & allowed_entities)]
    gained = [t for t in (var_caps - base_caps).elements() if not (_expand(t) & allowed_entities)]
    checks["entities_lost"] = sorted(set(lost))[:8]
    checks["entities_gained"] = sorted(set(gained))[:8]
    if is_transform:
        # A register rewrite may introduce a capitalised verb, but must never
        # lose an employer, institution or technology.
        if lost:
            failures.append(f"transform lost entities: {sorted(set(lost))[:5]}")
    elif lost or gained:
        failures.append(
            f"unexplained entity change (lost {sorted(set(lost))[:3]}, gained {sorted(set(gained))[:3]})"
        )

    # --- 4. qualification preservation -------------------------------------
    base_tokens = Counter(tokens(baseline_text))
    var_tokens = Counter(tokens(variant_text))
    added_tokens = var_tokens - base_tokens
    removed_tokens = base_tokens - var_tokens
    allowed_words = _explained_words(edits, transform_terms)
    if is_transform:
        # The exact check: every token that appeared or disappeared must be in
        # the transform's own declared vocabulary. This is stronger than a
        # similarity score and, unlike one, it does not depend on document
        # length — a Jaccard threshold tuned for a full resume rejects a
        # single-bullet input where two changed words are a quarter of the text.
        stray_added = [t for t in added_tokens if not (_expand(t) & allowed_words)]
        stray_removed = [t for t in removed_tokens if not (_expand(t) & allowed_words)]
        checks["stray_tokens_added"] = sorted(set(stray_added))[:8]
        checks["stray_tokens_removed"] = sorted(set(stray_removed))[:8]
        if require_skill_preservation and (stray_added or stray_removed):
            failures.append(
                f"transform changed content outside its vocabulary "
                f"(+{sorted(set(stray_added))[:3]}, -{sorted(set(stray_removed))[:3]})"
            )
        union = set(base_tokens) | set(var_tokens)
        jaccard = len(set(base_tokens) & set(var_tokens)) / len(union) if union else 1.0
        checks["token_jaccard"] = round(jaccard, 4)
        # The similarity floor is a secondary guard against a runaway rewrite,
        # and only meaningful once there is enough text for the ratio to be
        # stable.
        if (
            require_skill_preservation
            and len(base_tokens) >= _MIN_TOKENS_FOR_JACCARD
            and jaccard < min_token_jaccard
        ):
            failures.append(f"token overlap {jaccard:.3f} below {min_token_jaccard}")
    else:
        stray_added = [t for t in added_tokens if not (_expand(t) & allowed_words)]
        stray_removed = [t for t in removed_tokens if not (_expand(t) & allowed_words)]
        checks["stray_tokens_added"] = sorted(set(stray_added))[:8]
        checks["stray_tokens_removed"] = sorted(set(stray_removed))[:8]
        if require_skill_preservation and (stray_added or stray_removed):
            failures.append(
                f"unaccounted token change (+{sorted(set(stray_added))[:3]}, "
                f"-{sorted(set(stray_removed))[:3]})"
            )

    # --- 5. edit magnitude --------------------------------------------------
    ratio = normalized_edit_ratio(baseline_text, variant_text)
    checks["edit_ratio"] = round(ratio, 5)
    limit = max_unintended_edit_ratio
    if is_transform:
        # A register rewrite is expected to be larger; the cap still catches a
        # runaway rewrite that has effectively produced a different document.
        limit = max(0.25, max_unintended_edit_ratio)
    declared_chars = sum(max(len(e.before), len(e.after)) for e in edits)
    budget = limit + (declared_chars / max(1, len(baseline_text)))
    checks["edit_ratio_budget"] = round(budget, 5)
    if ratio > budget:
        failures.append(f"edit ratio {ratio:.4f} exceeds budget {budget:.4f}")

    checks["passed"] = not failures
    checks["failures"] = failures
    return checks


def summarise_validation(variants: Iterable[Any]) -> Dict[str, Any]:
    """Aggregate validation outcomes across a variant set, for the report."""
    total = 0
    passed = 0
    by_family: Dict[str, Dict[str, int]] = {}
    failure_counts: Counter = Counter()
    for variant in variants:
        total += 1
        ok = bool(variant.validation.get("passed", True))
        passed += int(ok)
        for family in variant.cue_families or ["baseline"]:
            bucket = by_family.setdefault(family, {"total": 0, "passed": 0})
            bucket["total"] += 1
            bucket["passed"] += int(ok)
        if not ok:
            for failure in variant.validation.get("failures", []):
                failure_counts[failure.split(":")[0][:60]] += 1
    return {
        "n_variants": total,
        "n_passed": passed,
        "n_failed": total - passed,
        "pass_rate": (passed / total) if total else 1.0,
        "by_family": by_family,
        "top_failure_modes": failure_counts.most_common(8),
    }
