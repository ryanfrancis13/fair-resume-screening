"""Counterfactual perturbation validation, demographic signal detection, and multi-level redaction."""

from __future__ import annotations

import re
from typing import Any

RATIONALE_LEXICONS: dict[str, tuple[str, ...]] = {
    "agentic": ("led", "drove", "architected", "owned", "spearheaded", "decisive", "independent", "commanded", "pioneered", "executed"),
    "communal": ("supported", "assisted", "helped", "collaborative", "nurtured", "team-player", "contributed", "pleasant", "cooperative", "caring"),
    "ability": ("brilliant", "talented", "gifted", "natural", "sharp", "genius", "exceptional", "innate", "mastery"),
    "effort": ("hardworking", "diligent", "dedicated", "persistent", "thorough", "meticulous", "studious", "dependable", "Steady"),
    "hedge": ("perhaps", "somewhat", "might", "appears", "potentially", "possibly", "adequate", "passable", "borderline"),
    "booster": ("clearly", "undoubtedly", "standout", "proven", "unquestionably", "outstanding", "compelling", "definitive"),
}

_FEMALE_NAMES = {
    "priya", "anjali", "neha", "pooja", "lakshmi", "divya", "kavya", "anitha",
    "fatima", "aisha", "zainab", "nazia", "mary", "grace", "rachel", "sarah",
    "simran", "harpreet", "navjot", "prabhleen", "emily", "anne", "jill", "allison",
    "latoya", "ebony", "tanisha", "lakisha", "keisha", "tamika",
}
_MALE_NAMES = {
    "rahul", "amit", "vikram", "rohit", "suresh", "venkat", "karthik", "arun",
    "mohammed", "imran", "faisal", "irfan", "john", "david", "thomas", "kevin",
    "gurpreet", "harjit", "balwinder", "jaspal", "greg", "jay", "matthew", "brad",
    "jamal", "leroy", "tyrone", "hakim", "kareem", "rasheed",
}
_RELIGION_CUES = {
    "Hindu": ("sharma", "gupta", "joshi", "mishra", "iyer", "nair", "reddy", "rao", "hindu students council", "navratri", "pongal", "durga puja", "onam"),
    "Muslim": ("khan", "ahmed", "sheikh", "siddiqui", "muslim students association", "urdu (native)", "eid community"),
    "Christian": ("dsouza", "fernandes", "thomas", "mathew", "christian fellowship"),
    "Sikh": ("gill", "sandhu", "dhillon", "bajwa", "sikh students association", "punjabi (native)", "gurpurab"),
}
_REGION_CUES = {
    "South": ("iyer", "nair", "reddy", "rao", "tamil (native)", "malayalam (native)", "hyderabad", "coimbatore", "pongal", "onam"),
    "North": ("sharma", "gupta", "joshi", "mishra", "hindi (native)", "gurugram", "jaipur", "navratri"),
    "West": ("mumbai",),
    "East": ("bengali (native)", "durga puja", "patna"),
}


def detect_demographic_signals(text: str) -> dict[str, str]:
    """Infer demographic & proxy cues directly from resume text so redaction removes what it masks."""
    signals: dict[str, str] = {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header = lines[0].lower() if lines else ""
    first_token = re.sub(r"[^a-z]", "", header.split()[0]) if header.split() else ""
    lower = text.lower()

    if first_token in _FEMALE_NAMES or re.search(r"\b(she|her)\b", lower) or "women in leadership" in lower:
        signals["gender"] = "female"
    elif first_token in _MALE_NAMES or re.search(r"\b(he|his)\b", lower):
        signals["gender"] = "male"

    for rel, markers in _RELIGION_CUES.items():
        if any(m in lower for m in markers):
            signals["religion"] = rel
            break

    for reg, markers in _REGION_CUES.items():
        if any(m in lower for m in markers):
            signals["region"] = reg
            break

    if "disability inclusion" in lower:
        signals["disability_status"] = "disabled"
    if "queer alliance" in lower:
        signals["sexual_orientation"] = "lgbtq"

    if "self-funded; completed an exchange" in lower or "golf, sailing" in lower:
        signals["socioeconomic"] = "high"
    elif "first-generation graduate" in lower or "state scholarship" in lower or "family shop" in lower:
        signals["socioeconomic"] = "low"

    if "career began in 2009" in lower or ", 2009" in lower:
        signals["age_band"] = "50s"
    elif "career began in 2014" in lower or ", 2014" in lower:
        signals["age_band"] = "40s"
    elif "career began in 2019" in lower:
        signals["age_band"] = "30s"

    if "indian institute of technology" in lower or "massachusetts institute of technology" in lower or "stanford" in lower:
        signals["prestige"] = "high"
    elif "sri venkateshwara college" in lower or "community college" in lower:
        signals["prestige"] = "low"

    return signals


def redact_resume(text: str, level: str = "identity") -> str:
    """Blind-screening redactor at three nested levels: identity, identity_plus_proxy, aggressive."""
    lines = text.splitlines()
    if lines:
        lines[0] = "[CANDIDATE]"
    if len(lines) > 1 and ("@" in lines[1] or "linkedin.com" in lines[1]):
        parts = [p.strip() for p in lines[1].split("|")]
        loc = parts[0] if parts else "[LOCATION]"
        lines[1] = f"{'[LOCATION]' if level == 'aggressive' else loc} | [EMAIL] | [PHONE] | [LINKEDIN]"
    out = "\n".join(lines)

    out = re.sub(r"\bShe\b", "They", out)
    out = re.sub(r"\bHe\b", "They", out)
    out = re.sub(r"\bshe\b", "they", out)
    out = re.sub(r"\bhe\b", "they", out)
    out = re.sub(r"\b(Her|His)\b", "Their", out)
    out = re.sub(r"\b(her|his)\b", "their", out)

    if level in ("identity_plus_proxy", "aggressive"):
        out = re.sub(r"(?m)^Affiliations:.*$", "Affiliations: [REDACTED]", out)
        out = re.sub(r"(?m)^Spoken languages:.*$", "Spoken languages: English (fluent)", out)
        out = re.sub(r"(?m)^Community:.*$", "Community: [REDACTED]", out)
        out = re.sub(r"(?m)^Interests:.*$", "Interests: [REDACTED]", out)
        out = re.sub(r"(?m)^(Self-funded;.*|First-generation graduate;.*)$", "", out)

    if level == "aggressive":
        out = re.sub(r"(?m)^Career began in \d{4}\.$", "", out)
        out = re.sub(r"(?m)^(B\.[A-Z]\.|M\.[A-Z]\.|B\.Tech|M\.Tech|B\.Sc|M\.Sc|MBA)([^,\n]+),\s*[^,\n]+,\s*\d{4}$", r"\1\2, [UNIVERSITY], [YEAR]", out)

    return out


def _strip_once(text: str, sub: str, from_right: bool = False) -> str:
    if not sub:
        return text
    return "".join(text.rsplit(sub, 1)) if from_right else text.replace(sub, "", 1)


def validate_variant(base_text: str, variant_text: str, edits: dict[str, dict[str, str]], family: str = "name") -> dict[str, Any]:
    """Run the 4 invariance checks proving a counterfactual variant only changed its designated cue slots."""
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    b_rem, v_rem = base_text, variant_text
    for slot, slot_change in edits.items():
        right = slot == "GRAD_YEAR"
        b_rem = _strip_once(b_rem, slot_change.get("from", ""), right)
        v_rem = _strip_once(v_rem, slot_change.get("to", ""), right)
    span_ok = norm(b_rem) == norm(v_rem)
    nums = lambda s: re.findall(r"\b\d+(?:\.\d+)?%?\b", s)
    num_ok = nums(b_rem) == nums(v_rem)
    entities = lambda s: sorted(set(re.findall(r"\b[A-Z][A-Za-z0-9+#.-]{2,}\b", s)))
    ent_ok = entities(b_rem) == entities(v_rem)

    delta_chars = sum(abs(len(c.get("to", "")) - len(c.get("from", ""))) for c in edits.values())
    edit_ratio = delta_chars / max(len(base_text), 1)
    budget = 0.45 if family == "writing_register" else 0.15
    ratio_ok = edit_ratio <= budget

    passed = span_ok and num_ok and ent_ok and ratio_ok
    return {
        "span_accounting": span_ok,
        "numeric_fingerprint": num_ok,
        "entity_fingerprint": ent_ok,
        "qualification_preserved": ratio_ok,
        "edit_ratio": round(edit_ratio, 4),
        "passed": passed,
    }

