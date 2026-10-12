"""Auditing the model's *explanations*, not only its scores.

A screener can produce identical scores for two identical resumes and still
describe one candidate as "a talented systems architect" and the other as "a
hardworking team player".  Those explanations are what a recruiter reads, so they
shape the human decision even when the number does not — Wilson et al. (2025)
show LLM hiring recommendations measurably move human decisions.  A pipeline that
audits scores alone declares that system fair.

Four asymmetries are measured across demographic levels:

**Agentic vs communal language.** The stereotype-content contrast. Agentic
descriptions are rewarded more in hiring text, so a systematic split in which
groups get which is a bias with a direct allocational path.

**Effort vs ability attribution.** Achievement credited to diligence for one group
and to talent for another is a well-documented attribution asymmetry, and it
matters because "hardworking" reads as a lower ceiling than "brilliant".

**Hedging vs boosting.** Praising one group plainly and another with caveats is
invisible to any score metric.

**Direct protected-attribute references.** A rationale that mentions gender, race,
age or disability at all is a compliance problem independently of its score
effect. This is reported as a rate, per group.

The statistical tool is the log-odds ratio with an informative Dirichlet prior
(Monroe, Colaresi & Quinn 2008).  A raw frequency ratio is dominated by rare
words — one occurrence of an unusual term produces a spectacular-looking,
meaningless ratio — and the prior shrinks exactly those cases toward zero.
Lexicons live in :mod:`frs.anonymize.lexicons` and are short enough to read,
because a reviewer must be able to see precisely what was counted.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..anonymize.lexicons import protected_reference_lexicon, rationale_lexicons
from ..schemas import MetricResult
from ..utils.frames import as_text, block_key, nonempty_mask
from ..utils.text import count_lexicon_hits, tokens
from .stats import paired_block_differences, paired_test, two_proportion_test

_MIN_ROWS_PER_LEVEL = 5


def _prepare(
    frame: pd.DataFrame, attribute: str, text_column: str
) -> pd.DataFrame:
    column = f"signal_{attribute}"
    if column not in frame.columns or text_column not in frame.columns:
        return pd.DataFrame()
    data = frame[nonempty_mask(frame, column) & nonempty_mask(frame, text_column)]
    if "parse_ok" in data.columns:
        data = data[data["parse_ok"].astype(bool)]
    if data.empty:
        return data
    data = data.copy()
    data["_block"] = block_key(data, "job_id", "base_resume_id")
    data["_group"] = as_text(data[column])
    data["_text"] = as_text(data[text_column])
    return data


def lexicon_rates(
    frame: pd.DataFrame,
    attribute: str,
    *,
    text_column: str = "rationale_text",
) -> Dict[str, Any]:
    """Per-1000-word rates of each evaluative lexicon, by demographic level.

    Rates are per word rather than per rationale, because rationale length itself
    varies across conditions and a raw count would confound "described more
    communally" with "described at greater length".
    """
    data = _prepare(frame, attribute, text_column)
    if data.empty:
        return {"attribute": attribute, "error": f"no rows with signal_{attribute} and rationale text"}

    lexicons = rationale_lexicons()
    protected_terms = protected_reference_lexicon()

    rows: List[Dict[str, Any]] = []
    for _, record in data.iterrows():
        text = record["_text"]
        n_words = max(1, len(tokens(text)))
        entry: Dict[str, Any] = {
            "_block": record["_block"],
            "_group": record["_group"],
            "n_words": n_words,
        }
        for name, terms in lexicons.items():
            hits = sum(count_lexicon_hits(text, terms).values())
            entry[f"rate_{name}"] = 1000.0 * hits / n_words
            entry[f"count_{name}"] = hits
        protected_hits = count_lexicon_hits(text, protected_terms)
        entry["count_protected"] = sum(protected_hits.values())
        entry["mentions_protected"] = int(entry["count_protected"] > 0)
        entry["protected_terms"] = sorted(protected_hits)[:5]
        rows.append(entry)

    table = pd.DataFrame(rows)
    levels = sorted(table["_group"].unique())
    by_level: Dict[str, Any] = {}
    for level in levels:
        subset = table[table["_group"] == level]
        by_level[level] = {
            "n_rationales": int(len(subset)),
            "mean_words": round(float(subset["n_words"].mean()), 1),
            "protected_mention_rate": round(float(subset["mentions_protected"].mean()), 4),
            **{
                f"rate_{name}": round(float(subset[f"rate_{name}"].mean()), 4)
                for name in lexicons
            },
            "agentic_minus_communal": round(
                float(subset["rate_agentic"].mean() - subset["rate_communal"].mean()), 4
            ),
            "ability_minus_effort": round(
                float(subset["rate_ability"].mean() - subset["rate_effort"].mean()), 4
            ),
            "hedge_minus_booster": round(
                float(subset["rate_hedge"].mean() - subset["rate_booster"].mean()), 4
            ),
        }

    contrasts: List[Dict[str, Any]] = []
    usable_levels = [lvl for lvl in levels if by_level[lvl]["n_rationales"] >= _MIN_ROWS_PER_LEVEL]
    reference = usable_levels[0] if usable_levels else None
    for level in usable_levels[1:]:
        for measure in ("agentic_minus_communal", "ability_minus_effort", "hedge_minus_booster"):
            column = {
                "agentic_minus_communal": ("rate_agentic", "rate_communal"),
                "ability_minus_effort": ("rate_ability", "rate_effort"),
                "hedge_minus_booster": ("rate_hedge", "rate_booster"),
            }[measure]
            values = (table[column[0]] - table[column[1]]).tolist()
            blocked = paired_block_differences(
                values, table["_group"].tolist(), table["_block"].tolist(), level, reference
            )
            test = paired_test(blocked.differences)
            contrasts.append(
                {
                    "measure": measure,
                    "level": level,
                    "reference": reference,
                    "n_blocks": blocked.n_blocks,
                    "difference": round(blocked.mean_difference, 4)
                    if blocked.mean_difference == blocked.mean_difference else None,
                    "p_value": test.get("t_p_value"),
                    "unit": "occurrences per 1000 words",
                }
            )
        prot = two_proportion_test(
            int(table.loc[table["_group"] == level, "mentions_protected"].sum()),
            int((table["_group"] == level).sum()),
            int(table.loc[table["_group"] == reference, "mentions_protected"].sum()),
            int((table["_group"] == reference).sum()),
        )
        contrasts.append(
            {"measure": "protected_mention_rate", "level": level,
             "reference": reference, **prot}
        )

    return {
        "attribute": attribute,
        "levels": levels,
        "n_rationales": int(len(table)),
        "by_level": by_level,
        "contrasts": contrasts,
        "overall_protected_mention_rate": round(float(table["mentions_protected"].mean()), 4),
        "example_protected_terms": sorted(
            {t for terms in table["protected_terms"] for t in terms}
        )[:12],
    }


def distinctive_terms(
    frame: pd.DataFrame,
    attribute: str,
    *,
    text_column: str = "rationale_text",
    top_n: int = 12,
    prior_strength: float = 10.0,
    min_count: int = 3,
) -> Dict[str, Any]:
    """Words most distinctive of each level's rationales.

    Monroe, Colaresi & Quinn's (2008) log-odds ratio with an informative
    Dirichlet prior, z-scored. The prior is the corpus-wide word distribution, so
    rare words are shrunk toward zero instead of dominating the ranking — which
    is what makes this readable as evidence rather than as a list of typos.

    Unlike the lexicon rates, this is *unsupervised*: it can surface an asymmetry
    nobody thought to put in a lexicon, which is the point of running it.
    """
    data = _prepare(frame, attribute, text_column)
    if data.empty:
        return {"attribute": attribute, "error": "no usable rationales"}

    counts: Dict[str, Counter] = {}
    for level, subset in data.groupby("_group"):
        counter: Counter = Counter()
        for text in subset["_text"]:
            counter.update(t for t in tokens(text) if len(t) > 2 and not t.isdigit())
        counts[str(level)] = counter

    if len(counts) < 2:
        return {"attribute": attribute, "error": "fewer than 2 levels with rationales"}

    corpus: Counter = Counter()
    for counter in counts.values():
        corpus.update(counter)
    total_corpus = sum(corpus.values())
    if total_corpus == 0:
        return {"attribute": attribute, "error": "rationales contained no words"}

    alpha_0 = prior_strength
    results: Dict[str, List[Dict[str, Any]]] = {}
    for level, counter in counts.items():
        others: Counter = Counter()
        for other_level, other_counter in counts.items():
            if other_level != level:
                others.update(other_counter)
        n_i = sum(counter.values())
        n_j = sum(others.values())
        if n_i == 0 or n_j == 0:
            continue
        scored: List[Dict[str, Any]] = []
        for word, corpus_count in corpus.items():
            y_i = counter.get(word, 0)
            y_j = others.get(word, 0)
            if y_i + y_j < min_count:
                continue
            alpha_w = alpha_0 * corpus_count / total_corpus
            num_i = y_i + alpha_w
            num_j = y_j + alpha_w
            den_i = n_i + alpha_0 - num_i
            den_j = n_j + alpha_0 - num_j
            if den_i <= 0 or den_j <= 0:
                continue
            delta = math.log(num_i / den_i) - math.log(num_j / den_j)
            variance = 1.0 / num_i + 1.0 / num_j
            z = delta / math.sqrt(variance) if variance > 0 else 0.0
            scored.append(
                {"term": word, "z": round(z, 3), "log_odds_delta": round(delta, 4),
                 "count_in_level": y_i, "count_elsewhere": y_j}
            )
        scored.sort(key=lambda item: -item["z"])
        results[level] = scored[:top_n]

    return {
        "attribute": attribute,
        "method": "Monroe et al. (2008) log-odds ratio, informative Dirichlet prior",
        "prior_strength": prior_strength,
        "min_count": min_count,
        "distinctive_terms": results,
        "note": (
            "Positive z means the term is used disproportionately in that level's "
            "rationales. |z| above ~1.96 is conventionally notable, but these are "
            "descriptive: no multiplicity correction is applied across the vocabulary."
        ),
    }


def rationale_metrics(
    frame: pd.DataFrame,
    attributes: Sequence[str],
    *,
    text_column: str = "rationale_text",
    rng: Optional[np.random.Generator] = None,
) -> List[MetricResult]:
    """Rationale-language metrics as :class:`MetricResult` rows."""
    del rng  # deterministic; signature kept uniform across metric families
    metrics: List[MetricResult] = []

    for attribute in attributes:
        analysis = lexicon_rates(frame, attribute, text_column=text_column)
        if "error" in analysis:
            continue

        by_level = analysis["by_level"]
        for measure, better in (
            ("agentic_minus_communal", "higher"),
            ("ability_minus_effort", "higher"),
            ("hedge_minus_booster", "lower"),
        ):
            values = {
                level: info[measure] for level, info in by_level.items()
                if info["n_rationales"] >= _MIN_ROWS_PER_LEVEL
            }
            if len(values) < 2:
                continue
            spread = max(values.values()) - min(values.values())
            related = [
                c for c in analysis["contrasts"] if c.get("measure") == measure
            ]
            p_values = [c["p_value"] for c in related if c.get("p_value") == c.get("p_value")]
            metrics.append(
                MetricResult(
                    name=f"rationale_{measure}_spread[{attribute}]",
                    family="rationale",
                    attribute=attribute,
                    value=round(spread, 4),
                    unit="occurrences per 1000 words",
                    p_value=min(p_values) if p_values else float("nan"),
                    n=analysis["n_rationales"],
                    groups=values,
                    higher_is_better=False,
                    detail={"contrasts": related, "by_level": by_level},
                    interpretation=(
                        f"The {measure.replace('_', ' ')} balance differs by {spread:.2f} "
                        f"per 1000 words across {attribute} levels "
                        f"(most {'agentic' if 'agentic' in measure else 'ability-attributed' if 'ability' in measure else 'hedged'}: "
                        f"{max(values, key=values.get) if better == 'higher' else min(values, key=values.get)}). "
                        "Identical scores with systematically different descriptions still "
                        "steer the human reader."
                    ),
                )
            )

        protected_rates = {
            level: info["protected_mention_rate"] for level, info in by_level.items()
        }
        metrics.append(
            MetricResult(
                name=f"protected_attribute_mention_rate[{attribute}]",
                family="rationale",
                attribute=attribute,
                value=round(analysis["overall_protected_mention_rate"], 4),
                unit="fraction_of_rationales",
                n=analysis["n_rationales"],
                groups=protected_rates,
                threshold=0.0,
                passed=analysis["overall_protected_mention_rate"] == 0.0,
                higher_is_better=False,
                detail={
                    "by_level": protected_rates,
                    "example_terms": analysis["example_protected_terms"],
                    "contrasts": [
                        c for c in analysis["contrasts"]
                        if c.get("measure") == "protected_mention_rate"
                    ],
                },
                interpretation=(
                    f"{100 * analysis['overall_protected_mention_rate']:.1f}% of rationales "
                    "referenced a protected attribute or a proxy term. Any non-zero rate is "
                    "a compliance concern regardless of whether the score moved."
                ),
            )
        )
    return metrics


def rationale_examples(
    frame: pd.DataFrame,
    attribute: str,
    *,
    text_column: str = "rationale_text",
    max_pairs: int = 5,
) -> List[Dict[str, Any]]:
    """Matched rationale pairs for the same resume under different cue levels.

    The most persuasive evidence in a report of this kind is not a coefficient —
    it is two sentences about the same document, side by side.
    """
    data = _prepare(frame, attribute, text_column)
    if data.empty:
        return []

    pairs: List[Dict[str, Any]] = []
    for block, subset in data.groupby("_block"):
        levels = subset["_group"].unique()
        if len(levels) < 2:
            continue
        by_level = {
            str(level): group.iloc[0] for level, group in subset.groupby("_group")
        }
        scores = {
            level: float(row.get("overall_score", float("nan")))
            for level, row in by_level.items()
        }
        finite = {k: v for k, v in scores.items() if v == v}
        if len(finite) < 2:
            continue
        high = max(finite, key=finite.get)
        low = min(finite, key=finite.get)
        if high == low:
            continue
        pairs.append(
            {
                "block": str(block),
                "attribute": attribute,
                "higher_level": high,
                "lower_level": low,
                "score_gap": round(finite[high] - finite[low], 2),
                "higher_rationale": str(by_level[high]["_text"])[:700],
                "lower_rationale": str(by_level[low]["_text"])[:700],
            }
        )
    pairs.sort(key=lambda p: -abs(p["score_gap"]))
    return pairs[:max_pairs]


def _strip_ws(text: str) -> str:  # pragma: no cover - helper for report rendering
    return re.sub(r"\s+", " ", text).strip()
