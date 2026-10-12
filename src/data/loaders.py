"""Corpus loading: the curated sample set, a public dataset, or a plain directory.

Three sources, one output type.  Everything downstream sees
:class:`~frs.schemas.Resume` / :class:`~frs.schemas.JobDescription` / :class:`~frs.schemas.Pair`
objects and does not know or care where they came from, which is what lets the
same audit run over synthetic templates during development and over a real
public corpus for the final result.

``sample``
    12 templated synthetic resumes, 6 job descriptions, 18 hand-graded pairs,
    committed to the repository. Templates mean exact cue substitution.
``kaggle``
    The commonly used public "Resume Dataset" (~960 resumes, 25 categories).
    Not committed — it lands in ``data/raw/`` and is git-ignored, because
    redistributing a scraped resume corpus is not ours to do. These resumes have
    no templates, so the perturbation layer uses regex slot detection and the
    validator does real work.
``dir``
    Any directory of ``.txt``/``.json`` files, for a team's own documents.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import Config
from ..logging_setup import log_event
from ..schemas import JobDescription, Pair, Resume
from ..utils.text import clean_text, truncate

logger = logging.getLogger("frs.data.loaders")


# --------------------------------------------------------------------------- #
# sample corpus
# --------------------------------------------------------------------------- #
def load_sample_resumes(path: Path, *, max_chars: int = 12000) -> List[Resume]:
    resumes: List[Resume] = []
    for file in sorted(Path(path).glob("*.json")):
        payload = json.loads(file.read_text(encoding="utf-8"))
        resumes.append(
            Resume(
                resume_id=payload["resume_id"],
                text=truncate(clean_text(payload["text"]), max_chars),
                job_family=payload.get("job_family", "unknown"),
                source=payload.get("source", "sample"),
                template=payload.get("template"),
                slots=payload.get("baseline_slots", {}),
                meta=payload.get("meta", {}),
            )
        )
    if not resumes:
        raise FileNotFoundError(
            f"no resume JSON files under {path}. Run: python scripts/make_sample_corpus.py"
        )
    return resumes


def load_sample_jobs(path: Path, *, max_chars: int = 6000) -> List[JobDescription]:
    jobs: List[JobDescription] = []
    for file in sorted(Path(path).glob("*.json")):
        payload = json.loads(file.read_text(encoding="utf-8"))
        jobs.append(
            JobDescription(
                job_id=payload["job_id"],
                title=payload.get("title", payload["job_id"]),
                text=truncate(clean_text(payload["text"]), max_chars),
                job_family=payload.get("job_family", "unknown"),
                must_have=payload.get("must_have", []),
                nice_to_have=payload.get("nice_to_have", []),
                min_years_experience=float(payload.get("min_years_experience", 0)),
                source=payload.get("source", "sample"),
                meta=payload.get("meta", {}),
            )
        )
    if not jobs:
        raise FileNotFoundError(
            f"no job JSON files under {path}. Run: python scripts/make_sample_corpus.py"
        )
    return jobs


def load_pairs(path: Path) -> List[Pair]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Pair(
            pair_id=row["pair_id"],
            job_id=row["job_id"],
            resume_id=row["resume_id"],
            pairing_rationale=row.get("pairing_rationale", ""),
            expected_fit=row.get("expected_fit"),
            qualified_label=row.get("qualified_label"),
        )
        for row in payload
    ]


# --------------------------------------------------------------------------- #
# public dataset (Kaggle "Resume Dataset")
# --------------------------------------------------------------------------- #
#: Kaggle categories mapped onto this project's job families. Unmapped
#: categories are kept with their own family name rather than being forced into a
#: bucket, because a bad mapping would produce nonsense pairings.
_KAGGLE_FAMILY_MAP = {
    "Java Developer": "software_engineering",
    "Python Developer": "software_engineering",
    "DotNet Developer": "software_engineering",
    "Web Designing": "software_engineering",
    "Testing": "software_engineering",
    "DevOps Engineer": "software_engineering",
    "Data Science": "data_science",
    "Business Analyst": "data_science",
    "Database": "data_science",
    "Hadoop": "data_science",
    "ETL Developer": "data_science",
    "Sales": "marketing",
    "PMO": "marketing",
    "HR": "hr",
    "Operations Manager": "hr",
    "Advocate": "legal",
    "Arts": "creative",
    "Blockchain": "software_engineering",
    "Civil Engineer": "civil_engineering",
    "Mechanical Engineer": "mechanical_engineering",
    "Electrical Engineering": "electrical_engineering",
    "Automation Testing": "software_engineering",
    "Network Security Engineer": "software_engineering",
    "SAP Developer": "software_engineering",
    "Health and fitness": "other",
}


def load_kaggle_resumes(
    csv_path: Path,
    *,
    max_resumes: Optional[int] = None,
    max_chars: int = 12000,
    min_chars: int = 400,
) -> List[Resume]:
    """Load the public resume CSV (columns: ``Category``, ``Resume``).

    Very short documents are dropped: a 200-character stub cannot support a
    six-dimension rubric, and including it would add noise to every score
    distribution without adding information.
    """
    import pandas as pd

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found.\n"
            "Download the Kaggle 'Resume Dataset' (UpdatedResumeDataSet.csv) into "
            "data/raw/ — it is git-ignored by design. See docs/DATA.md."
        )
    frame = pd.read_csv(csv_path)
    columns = {c.lower(): c for c in frame.columns}
    text_col = columns.get("resume") or columns.get("resume_str") or list(frame.columns)[-1]
    cat_col = columns.get("category") or columns.get("job_category")

    resumes: List[Resume] = []
    kept_per_category: Dict[str, int] = {}
    for i, row in frame.iterrows():
        raw = str(row[text_col])
        text = clean_text(raw)
        if len(text) < min_chars:
            continue
        category = str(row[cat_col]).strip() if cat_col else "unknown"
        family = _KAGGLE_FAMILY_MAP.get(category, _slugify(category))
        resumes.append(
            Resume(
                resume_id=f"K{int(i):04d}",
                text=truncate(text, max_chars),
                job_family=family,
                source="kaggle_resume_dataset",
                template=None,
                slots={},
                meta={"kaggle_category": category, "orig_chars": len(raw)},
            )
        )
        kept_per_category[category] = kept_per_category.get(category, 0) + 1

    if max_resumes is not None:
        # Stratify the cap across categories instead of taking the first N rows,
        # which in this CSV would mean a single category.
        resumes = _stratified_head(resumes, max_resumes)

    log_event(
        logger, logging.INFO, "loaded public resume corpus",
        path=str(csv_path), n=len(resumes),
        families=len({r.job_family for r in resumes}),
    )
    return resumes


def _stratified_head(resumes: Sequence[Resume], limit: int) -> List[Resume]:
    buckets: Dict[str, List[Resume]] = {}
    for r in resumes:
        buckets.setdefault(r.job_family, []).append(r)
    out: List[Resume] = []
    index = 0
    while len(out) < limit:
        added = False
        for family in sorted(buckets):
            items = buckets[family]
            if index < len(items) and len(out) < limit:
                out.append(items[index])
                added = True
        if not added:
            break
        index += 1
    return out


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "unknown"

# --------------------------------------------------------------------------- #
# public dataset (Hugging Face: cnamuangtoun/resume-job-description-fit)
# --------------------------------------------------------------------------- #
#: Good/Potential/No Fit -> this project's strong/moderate/weak vocabulary,
#: matching the values already used by the sample corpus's pairs.json.
_HF_FIT_LABEL_MAP = {
    "good fit": "strong",
    "potential fit": "moderate",
    "no fit": "weak",
}

#: Coarse job-family buckets, keyword-matched against the job title/text.
#: This dataset has no category column at all (unlike Kaggle's), so family
#: is inferred rather than looked up. Order matters: first match wins, so
#: more specific keywords are listed before generic ones.
_HF_FIT_FAMILY_KEYWORDS = [
    ("software_engineering", ("software engineer", "developer", "programmer", "sde")),
    ("data_science", ("data scientist", "data analyst", "machine learning", "data engineer")),
    ("accounting_finance", ("accountant", "accounting", "finance", "auditor", "tax")),
    ("sales_marketing", ("sales", "marketing", "business development", "account executive")),
    ("hr", ("human resources", "hr ", "recruiter", "talent")),
    ("business_analysis", ("business analyst", "project manager", "product manager")),
    ("engineering_other", ("electrical", "mechanical", "civil engineer")),
]


def _infer_job_family(job_title_or_text: str) -> str:
    lowered = job_title_or_text.lower()
    for family, keywords in _HF_FIT_FAMILY_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return family
    return "other"


def load_hf_fit_corpus(
    train_csv: Path,
    test_csv: Optional[Path] = None,
    *,
    use_test_split: bool = False,
    max_resume_chars: int = 12000,
    max_job_chars: int = 6000,
    max_pairs: Optional[int] = None,
) -> Tuple[List[Resume], List[JobDescription], List[Pair]]:
    """Load cnamuangtoun/resume-job-description-fit into this project's schema.

    Unlike the Kaggle loader, this dataset already IS resume/job pairs with a
    real fit label per row — there is no separate job-family cross-product to
    build. Resume and job text repeat across many rows (642 unique resumes,
    280 unique job descriptions, 8,000 rows), so both are de-duplicated by
    exact text match before assigning ids, and the *pair* is the row.
    """
    import pandas as pd

    train_csv = Path(train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(
            f"{train_csv} not found.\n"
            "Run scripts/fetch_hf_fit_dataset.py first to download "
            "cnamuangtoun/resume-job-description-fit into data/raw/."
        )
    frames = [pd.read_csv(train_csv)]
    if use_test_split and test_csv is not None and Path(test_csv).exists():
        frames.append(pd.read_csv(test_csv))
    frame = pd.concat(frames, ignore_index=True)

    required = {"resume_text", "job_description_text", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"hf_fit CSV missing expected columns: {missing}")

    resume_ids: Dict[str, str] = {}   # text -> id
    job_ids: Dict[str, str] = {}      # text -> id
    resumes: List[Resume] = []
    jobs: List[JobDescription] = []
    pairs: List[Pair] = []
    skipped = 0

    for i, row in frame.iterrows():
        r_text = clean_text(str(row["resume_text"]))
        j_text = clean_text(str(row["job_description_text"]))
        label = str(row["label"]).strip().lower()
        if not r_text or not j_text or label not in _HF_FIT_LABEL_MAP:
            skipped += 1
            continue

        family = _infer_job_family(j_text[:500])  # title is usually near the top

        if r_text not in resume_ids:
            rid = f"HFR{len(resume_ids):04d}"
            resume_ids[r_text] = rid
            resumes.append(
                Resume(
                    resume_id=rid,
                    text=truncate(r_text, max_resume_chars),
                    job_family=family,
                    source="hf_resume_job_fit",
                    template=None,
                    slots={},
                    meta={},
                )
            )
        if j_text not in job_ids:
            jid = f"HFJ{len(job_ids):04d}"
            job_ids[j_text] = jid
            jobs.append(
                JobDescription(
                    job_id=jid,
                    title=j_text.split("\n", 1)[0][:80] or jid,
                    text=truncate(j_text, max_job_chars),
                    job_family=family,
                    source="hf_resume_job_fit",
                    meta={},
                )
            )

        pairs.append(
            Pair(
                pair_id=f"HFP{i:05d}",
                job_id=job_ids[j_text],
                resume_id=resume_ids[r_text],
                pairing_rationale="real recorded application (hf_fit dataset)",
                expected_fit=_HF_FIT_LABEL_MAP[label],
                qualified_label=None,
            )
        )

    if max_pairs is not None and len(pairs) > max_pairs:
        # Stratify by label, not a raw head-slice — this dataset's rows are
        # grouped by label, so pairs[:max_pairs] can silently produce a
        # single-label sample (all "No Fit"), which is useless for checking
        # the scorer actually distinguishes fit levels.
        buckets: Dict[str, List[Pair]] = {}
        for p in pairs:
            buckets.setdefault(p.expected_fit or "unknown", []).append(p)
        stratified: List[Pair] = []
        i = 0
        while len(stratified) < max_pairs:
            added = False
            for label in sorted(buckets):
                items = buckets[label]
                if i < len(items) and len(stratified) < max_pairs:
                    stratified.append(items[i])
                    added = True
            if not added:
                break
            i += 1
        pairs = stratified
        used_ids = {p.resume_id for p in pairs} | {p.job_id for p in pairs}
        resumes = [r for r in resumes if r.resume_id in used_ids]
        jobs = [j for j in jobs if j.job_id in used_ids]

    log_event(
        logger, logging.INFO, "loaded hf_fit corpus",
        n_resumes=len(resumes), n_jobs=len(jobs), n_pairs=len(pairs), skipped_rows=skipped,
    )
    return resumes, jobs, pairs
# --------------------------------------------------------------------------- #
# generic directory
# --------------------------------------------------------------------------- #
def load_dir_resumes(path: Path, *, max_chars: int = 12000) -> List[Resume]:
    resumes: List[Resume] = []
    for file in sorted(Path(path).iterdir()):
        if file.suffix.lower() == ".json":
            payload = json.loads(file.read_text(encoding="utf-8"))
            text = payload.get("text", "")
            resumes.append(
                Resume(
                    resume_id=payload.get("resume_id", file.stem),
                    text=truncate(clean_text(text), max_chars),
                    job_family=payload.get("job_family", "unknown"),
                    source=f"dir:{path.name}",
                    template=payload.get("template"),
                    slots=payload.get("baseline_slots", {}),
                    meta=payload.get("meta", {}),
                )
            )
        elif file.suffix.lower() in (".txt", ".md"):
            resumes.append(
                Resume(
                    resume_id=file.stem,
                    text=truncate(clean_text(file.read_text(encoding="utf-8", errors="replace")), max_chars),
                    job_family="unknown",
                    source=f"dir:{path.name}",
                )
            )
    if not resumes:
        raise FileNotFoundError(f"no .json/.txt resumes under {path}")
    return resumes


# --------------------------------------------------------------------------- #
# top-level entry point
# --------------------------------------------------------------------------- #
def load_corpus(cfg: Config) -> Tuple[List[Resume], List[JobDescription], List[Pair]]:
    """Load resumes, jobs and pairs according to ``cfg.data``."""
    from .pairing import build_pairs

    corpus = cfg.data.corpus.lower()

    # hf_fit brings its own real job descriptions and its own real pairs —
    # it does not use the sample job set or family-based pair construction
    # at all, unlike sample/kaggle/dir which all pair against data/sample/jobs.
    if corpus == "hf_fit":
        resumes, jobs, pairs = load_hf_fit_corpus(
            cfg.resolve(cfg.data.hf_fit_train_csv),
            cfg.resolve(cfg.data.hf_fit_test_csv),
            use_test_split=cfg.data.hf_fit_use_test_split,
            max_resume_chars=cfg.data.max_resume_chars,
            max_job_chars=cfg.data.max_job_chars,
            max_pairs=cfg.data.max_resumes,  # reuse this knob as a pair cap here
        )
        log_event(
            logger, logging.INFO, "corpus loaded",
            corpus=corpus, resumes=len(resumes), jobs=len(jobs), pairs=len(pairs),
        )
        return resumes, jobs, pairs

    jobs = load_sample_jobs(cfg.resolve(cfg.data.jobs_path), max_chars=cfg.data.max_job_chars)
    if cfg.data.max_jobs:
        jobs = jobs[: cfg.data.max_jobs]

    if corpus == "sample":
        resumes = load_sample_resumes(
            cfg.resolve(cfg.data.resumes_path), max_chars=cfg.data.max_resume_chars
        )
        if cfg.data.max_resumes:
            resumes = resumes[: cfg.data.max_resumes]
        pairs_path = cfg.resolve(cfg.data.pairs_path)
        if pairs_path.exists() and not cfg.data.cross_within_family:
            pairs = load_pairs(pairs_path)
            known_jobs = {j.job_id for j in jobs}
            known_resumes = {r.resume_id for r in resumes}
            pairs = [p for p in pairs if p.job_id in known_jobs and p.resume_id in known_resumes]
        else:
            pairs = build_pairs(resumes, jobs)
    elif corpus == "kaggle":
        resumes = load_kaggle_resumes(
            cfg.resolve(cfg.data.kaggle_csv),
            max_resumes=cfg.data.max_resumes,
            max_chars=cfg.data.max_resume_chars,
        )
        pairs = build_pairs(resumes, jobs)
    elif corpus == "dir":
        resumes = load_dir_resumes(
            cfg.resolve(cfg.data.resumes_path), max_chars=cfg.data.max_resume_chars
        )
        if cfg.data.max_resumes:
            resumes = resumes[: cfg.data.max_resumes]
        pairs = build_pairs(resumes, jobs)
    else:
                raise ValueError(f"unknown data.corpus {cfg.data.corpus!r} (sample | kaggle | hf_fit | dir)")

    if not pairs:
        raise ValueError(
            "no resume/job pairs were produced. Check that job families match between "
            "resumes and job descriptions, or set data.cross_within_family=true."
        )
    log_event(
        logger, logging.INFO, "corpus loaded",
        corpus=corpus, resumes=len(resumes), jobs=len(jobs), pairs=len(pairs),
    )
    return resumes, jobs, pairs


def corpus_stats(
    resumes: Sequence[Resume], jobs: Sequence[JobDescription], pairs: Sequence[Pair]
) -> Dict[str, Any]:
    """Descriptive statistics for the run manifest and the report."""
    from ..utils.text import word_count

    lengths = [word_count(r.text) for r in resumes]
    families: Dict[str, int] = {}
    for r in resumes:
        families[r.job_family] = families.get(r.job_family, 0) + 1
    fits: Dict[str, int] = {}
    for p in pairs:
        key = p.expected_fit or "ungraded"
        fits[key] = fits.get(key, 0) + 1
    return {
        "n_resumes": len(resumes),
        "n_jobs": len(jobs),
        "n_pairs": len(pairs),
        "n_templated_resumes": sum(1 for r in resumes if r.is_templated),
        "resume_words_mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "resume_words_min": min(lengths) if lengths else 0,
        "resume_words_max": max(lengths) if lengths else 0,
        "job_families": families,
        "expected_fit_distribution": fits,
        "n_labelled_pairs": sum(1 for p in pairs if p.qualified_label is not None),
    }
