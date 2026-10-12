"""Counterfactual variant generation.

Given a base resume and a set of cue families, produce one variant per cue level
in which *only* that cue differs from the baseline.  Two substitution paths are
supported, and both converge on the same validated :class:`ResumeVariant`:

**Templated path** (the curated sample corpus). Each resume ships as a template
with declared ``{{SLOT}}`` positions and a baseline value per slot. Substitution
is exact by construction, so validity is guaranteed rather than checked — which
makes this the right path for the headline experiment.

**Detected path** (resumes ingested from a public dataset). Slots are located
with regexes over the raw text, then replaced surgically. Detection is fallible,
so the validator is doing real work here, and a resume whose name line cannot be
located is *reported as skipped* rather than quietly perturbed at the wrong
offset.

Designs
-------
``single``
    One family varied at a time. Clean main effects; the default.
``factorial_name_x_gender``
    ``single``, plus the full cross of name × pronoun levels. Needed for
    intersectional claims: a gender effect that exists only for some ethnicities
    is invisible in marginal comparisons.
``full_factorial``
    Cross of all requested families, guarded by ``max_variants_per_resume``
    because the count is multiplicative and will exhaust an API budget without
    warning.
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import replace as dc_replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..logging_setup import log_event
from ..schemas import Edit, Resume, ResumeVariant
from ..utils.hashing import hash_text
from ..utils.text import clean_text
from .cues import CueFamily, CueLevel
from .transforms import apply_transform
from .validator import validate_variant

logger = logging.getLogger("frs.perturb.generator")

_SLOT_RE = re.compile(r"\{\{([A-Z_][A-Z0-9_]*)\}\}")

#: Hard ceiling so a mis-specified factorial design cannot silently issue
#: thousands of API calls.
MAX_VARIANTS_PER_RESUME = 400


# --------------------------------------------------------------------------- #
# template rendering
# --------------------------------------------------------------------------- #
def template_slots(template: str) -> List[str]:
    return sorted(set(_SLOT_RE.findall(template)))


def render_template(template: str, values: Dict[str, str]) -> str:
    """Fill ``{{SLOT}}`` placeholders, dropping lines that become empty.

    Optional slots (a funding note, an early-career note) are empty at some cue
    levels.  Leaving a blank line behind would make the *layout* differ between
    levels, adding an unintended cue about document tidiness.
    """
    missing = [s for s in template_slots(template) if s not in values]
    if missing:
        raise KeyError(f"template is missing values for slots: {missing}")

    out_lines: List[str] = []
    for line in template.split("\n"):
        slots_in_line = _SLOT_RE.findall(line)
        rendered = _SLOT_RE.sub(lambda m: values[m.group(1)], line)
        if slots_in_line and not rendered.strip():
            continue  # whole line was one empty optional slot
        # "Languages: " with nothing after it reads as a formatting defect.
        if slots_in_line and re.fullmatch(r"[A-Za-z /&]+:\s*", rendered):
            continue
        out_lines.append(rendered)
    return clean_text("\n".join(out_lines))


# --------------------------------------------------------------------------- #
# slot detection for non-templated corpora
# --------------------------------------------------------------------------- #
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w-]+", re.IGNORECASE)
#: "Boston, MA" / "Lagos, Nigeria". The intra-name separator is a literal space
#: or tab, never ``\s``: with ``\s`` the pattern happily spans a line break and
#: matches the tail of the name line plus the start of the address line
#: ("Avery\nBoston, MA"), which would then be substituted as one span and
#: corrupt both lines.
_LOCATION_RE = re.compile(
    r"\b([A-Z][a-z]+(?:[ \t][A-Z][a-z]+)?),[ \t]*([A-Z]{2}|[A-Z][a-z]+(?:[ \t][A-Z][a-z]+)?)\b"
)
_GRAD_YEAR_RE = re.compile(
    r"(?:graduat\w*|B\.?[AST]\.?|M\.?[AS]\.?|Bachelor|Master|degree)[^\n]{0,80}?\b(19\d{2}|20[0-2]\d)\b",
    re.IGNORECASE,
)
_NAME_LINE_RE = re.compile(r"^\s*([A-Z][a-z'’\-]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z'’\-]+){1,2})\s*$")


def detect_slots(text: str) -> Dict[str, str]:
    """Locate substitutable spans in an unstructured resume.

    Only high-confidence patterns are reported.  A missing slot means "this
    family cannot be perturbed on this resume", which is a recorded outcome; a
    *wrong* slot would corrupt the experiment, so the regexes are deliberately
    conservative.
    """
    found: Dict[str, str] = {}
    lines = text.split("\n")

    # Name: the first short line in the top of the document that parses as a
    # personal name and is not a section header.
    for line in lines[:6]:
        stripped = line.strip()
        if not stripped or len(stripped) > 45:
            continue
        if stripped.isupper():
            continue
        match = _NAME_LINE_RE.match(stripped)
        if match and "@" not in stripped:
            found["NAME"] = match.group(1)
            break

    if (m := _EMAIL_RE.search(text)):
        found["EMAIL"] = m.group(0)
    if (m := _LINKEDIN_RE.search(text)):
        found["LINKEDIN"] = m.group(0)
    # Search line by line so a match can never straddle a newline.
    for line in lines[:8]:
        if (m := _LOCATION_RE.search(line)):
            found["LOCATION"] = m.group(0)
            break
    header = "\n".join(lines[:8])
    phones = [p for p in _PHONE_RE.findall(header) if len(re.sub(r"\D", "", p)) >= 10]
    if phones:
        raw = _PHONE_RE.search(header)
        if raw:
            found["PHONE"] = raw.group(0).strip()
    if (m := _GRAD_YEAR_RE.search(text)):
        found["GRAD_YEAR"] = m.group(1)
    return found


def _substitute_detected(
    text: str, slot_values: Dict[str, str], new_values: Dict[str, str]
) -> Tuple[str, List[Edit], List[str]]:
    """Replace detected literals with new values, recording each edit."""
    edits: List[Edit] = []
    skipped: List[str] = []
    out = text
    for slot, new_value in new_values.items():
        old_value = slot_values.get(slot)
        if not old_value:
            skipped.append(slot)
            continue
        if old_value == new_value:
            continue
        start = out.find(old_value)
        occurrences = out.count(old_value)
        out = out.replace(old_value, new_value)
        edits.append(
            Edit(slot=slot, cue_family="", before=old_value, after=new_value, start=start,
                 end=start + len(old_value) if start >= 0 else -1)
        )
        if occurrences > 1:
            log_event(
                logger, logging.DEBUG, "slot literal replaced in multiple places",
                slot=slot, occurrences=occurrences,
            )
    return out, edits, skipped


# --------------------------------------------------------------------------- #
# variant construction
# --------------------------------------------------------------------------- #
def baseline_variant(resume: Resume) -> ResumeVariant:
    """The unperturbed reference row for a resume."""
    if resume.is_templated:
        text = render_template(resume.template or "", resume.slots)
    else:
        text = clean_text(resume.text)
    return ResumeVariant(
        variant_id=f"{resume.resume_id}__baseline",
        base_resume_id=resume.resume_id,
        text=text,
        is_baseline=True,
        cue_families=[],
        cue_ids={},
        signals={},
        edits=[],
        provenance={"path": "template" if resume.is_templated else "raw"},
        validation={"passed": True, "failures": [], "note": "baseline is unperturbed"},
    )


def _apply_levels(
    resume: Resume,
    baseline_text: str,
    families: Sequence[CueFamily],
    levels: Sequence[CueLevel],
) -> Tuple[Optional[str], List[Edit], List[str], bool]:
    """Apply a combination of one level per family. Returns (text, edits, skipped, used_transform)."""
    edits: List[Edit] = []
    skipped: List[str] = []
    used_transform = False

    slot_overrides: Dict[str, str] = {}
    transform_jobs: List[Tuple[str, str]] = []
    for family, level in zip(families, levels):
        if family.transform:
            transform_jobs.append((family.transform, level.signals.get(family.attributes[0], "")))
            used_transform = True
            continue
        for slot, value in level.values.items():
            slot_overrides[slot] = value

    if resume.is_templated:
        available = set(template_slots(resume.template or ""))
        needed = set(slot_overrides)
        absent = sorted(needed - available)
        if absent:
            # A resume without a FUNDING_NOTE slot simply cannot carry the SES
            # cue; that is recorded, never faked.
            return None, [], absent, used_transform
        values = dict(resume.slots)
        for slot, value in slot_overrides.items():
            if slot in values and values[slot] != value:
                edits.append(
                    Edit(slot=slot, cue_family="", before=values[slot], after=value)
                )
            values[slot] = value
        text = render_template(resume.template or "", values)
    else:
        detected = detect_slots(baseline_text)
        absent = sorted(set(slot_overrides) - set(detected))
        if absent:
            return None, [], absent, used_transform
        text, edits, skipped = _substitute_detected(baseline_text, detected, slot_overrides)

    for transform_name, level_value in transform_jobs:
        text = apply_transform(transform_name, text, level_value or "plain")

    return clean_text(text), edits, skipped, used_transform


def _select_levels(
    family: CueFamily, limit: Optional[int], rng
) -> List[CueLevel]:
    """Subsample a family's levels, always keeping the reference level.

    For ``name`` the subsample stays balanced across gender × ethnicity cells,
    because an unbalanced name subsample would confound the marginal effects.
    """
    levels = list(family.levels)
    if limit is None or limit >= len(levels):
        return levels
    if family.family == "name":
        cells: Dict[str, List[CueLevel]] = {}
        for lvl in levels:
            key = f"{lvl.signals.get('gender')}|{lvl.signals.get('race_ethnicity')}"
            cells.setdefault(key, []).append(lvl)
        per_cell = max(1, limit // max(1, len(cells)))
        chosen: List[CueLevel] = []
        for key in sorted(cells):
            items = cells[key]
            take = min(per_cell, len(items))
            idx = sorted(int(i) for i in rng.choice(len(items), size=take, replace=False))
            chosen.extend(items[i] for i in idx)
        return chosen
    reference = [lvl for lvl in levels if lvl.is_reference]
    others = [lvl for lvl in levels if not lvl.is_reference]
    keep = max(0, limit - len(reference))
    if keep < len(others):
        idx = sorted(int(i) for i in rng.choice(len(others), size=keep, replace=False))
        others = [others[i] for i in idx]
    return reference + others


def generate_variants(
    resume: Resume,
    registry: Dict[str, CueFamily],
    *,
    design: str = "single",
    levels_per_family: Optional[int] = None,
    include_baseline: bool = True,
    validate: bool = True,
    max_unintended_edit_ratio: float = 0.02,
    require_skill_preservation: bool = True,
    rng=None,
    max_variants: int = MAX_VARIANTS_PER_RESUME,
) -> Tuple[List[ResumeVariant], List[Dict[str, Any]]]:
    """Produce the variant set for one resume.

    Returns ``(variants, skips)`` where each skip records a family that could not
    be applied and why — attrition that must be reported, not hidden.
    """
    if rng is None:
        import numpy as np

        rng = np.random.default_rng(0)

    base = baseline_variant(resume)
    variants: List[ResumeVariant] = [base] if include_baseline else []
    skips: List[Dict[str, Any]] = []

    combos = _combinations(registry, design=design, levels_per_family=levels_per_family, rng=rng)
    if len(combos) > max_variants:
        raise ValueError(
            f"design {design!r} would generate {len(combos)} variants for resume "
            f"{resume.resume_id} (cap {max_variants}). Reduce "
            "perturbation.levels_per_family or use design='single'."
        )

    for families, levels in combos:
        text, edits, absent, used_transform = _apply_levels(resume, base.text, families, levels)
        if text is None:
            skips.append(
                {
                    "resume_id": resume.resume_id,
                    "families": [f.family for f in families],
                    "reason": "missing_slots",
                    "detail": absent,
                }
            )
            continue
        if text == base.text:
            # The reference level of a family often reproduces the baseline
            # exactly; that row carries no information, so it is dropped rather
            # than inflating the sample with duplicates.
            continue

        for edit in edits:
            for family, level in zip(families, levels):
                if edit.slot in level.values:
                    edit.cue_family = family.family
        signals: Dict[str, str] = {}
        cue_ids: Dict[str, str] = {}
        provenance: Dict[str, str] = {}
        for family, level in zip(families, levels):
            signals.update(level.signals)
            cue_ids[family.family] = level.cue_id
            provenance[family.family] = level.provenance

        combo_key = "+".join(sorted(level.cue_id for level in levels))
        variant = ResumeVariant(
            variant_id=f"{resume.resume_id}__{hash_text(combo_key, 10)}",
            base_resume_id=resume.resume_id,
            text=text,
            is_baseline=False,
            cue_families=[f.family for f in families],
            cue_ids=cue_ids,
            signals=signals,
            edits=edits,
            provenance={**provenance, "combo": combo_key,
                        "path": "template" if resume.is_templated else "raw"},
        )
        if validate:
            allows_numeric = any(f.family == "age_marker" for f in families)
            variant.validation = validate_variant(
                base.text,
                text,
                edits,
                is_transform=used_transform,
                allow_numeric_change=allows_numeric,
                max_unintended_edit_ratio=max_unintended_edit_ratio,
                require_skill_preservation=require_skill_preservation,
            )
        else:
            variant.validation = {"passed": True, "failures": [], "note": "validation disabled"}
        variants.append(variant)

    return variants, skips


def _combinations(
    registry: Dict[str, CueFamily],
    *,
    design: str,
    levels_per_family: Optional[int],
    rng,
) -> List[Tuple[Tuple[CueFamily, ...], Tuple[CueLevel, ...]]]:
    """Enumerate (families, levels) tuples for the requested design."""
    families = list(registry.values())
    per_family = {
        f.family: _select_levels(f, levels_per_family, rng) for f in families
    }

    combos: List[Tuple[Tuple[CueFamily, ...], Tuple[CueLevel, ...]]] = []

    if design in ("single", "factorial_name_x_gender"):
        for family in families:
            for level in per_family[family.family]:
                combos.append(((family,), (level,)))

    if design == "factorial_name_x_gender":
        name = registry.get("name")
        pronoun = registry.get("pronoun")
        if name is not None and pronoun is not None:
            for name_level, pronoun_level in itertools.product(
                per_family["name"], per_family["pronoun"]
            ):
                combos.append(((name, pronoun), (name_level, pronoun_level)))
        else:
            log_event(
                logger, logging.WARNING,
                "factorial_name_x_gender requested but 'name' and/or 'pronoun' "
                "family is not enabled; falling back to main effects only",
            )

    if design == "full_factorial":
        level_lists = [per_family[f.family] for f in families]
        for levels in itertools.product(*level_lists):
            combos.append((tuple(families), tuple(levels)))

    return combos


def variant_dataframe_rows(variants: Iterable[ResumeVariant]) -> List[Dict[str, Any]]:
    """Flatten variants for the CSV artifact."""
    rows: List[Dict[str, Any]] = []
    for v in variants:
        row: Dict[str, Any] = {
            "variant_id": v.variant_id,
            "base_resume_id": v.base_resume_id,
            "is_baseline": v.is_baseline,
            "cue_families": ",".join(v.cue_families),
            "cue_ids": ";".join(f"{k}={x}" for k, x in sorted(v.cue_ids.items())),
            "n_edits": len(v.edits),
            "validation_passed": bool(v.validation.get("passed", True)),
            "validation_failures": " | ".join(v.validation.get("failures", [])),
            "edit_ratio": v.validation.get("edit_ratio", 0.0),
            "text_hash": hash_text(v.text),
            "n_chars": len(v.text),
        }
        for attribute, level in v.signals.items():
            row[f"signal_{attribute}"] = level
        rows.append(row)
    return rows


def clone_with_text(variant: ResumeVariant, text: str, suffix: str) -> ResumeVariant:
    """Copy a variant with different text — used by mitigation arms."""
    return dc_replace(
        variant,
        variant_id=f"{variant.variant_id}__{suffix}",
        text=text,
    )
