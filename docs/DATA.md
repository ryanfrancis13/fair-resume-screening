# Data

## What is committed, and why

| path | tracked? | what it is |
| --- | --- | --- |
| `data/sample/resumes/*.json` | yes | 12 synthetic templated resumes |
| `data/sample/jobs/*.json` | yes | 6 synthetic job descriptions |
| `data/sample/pairs.json` | yes | 18 hand-graded resume↔job pairs |
| `data/lexicons/` | yes | exported name lists, for inspection |
| `data/raw/` | **no** | downloaded corpora (git-ignored) |
| `data/external/` | **no** | anything else third-party (git-ignored) |
| `runs/` | **no** | run artifacts (reproducible from config + seed) |

Two things are deliberately never committed: **API keys** (`.env` is ignored) and
**scraped resume corpora**, which are not ours to redistribute.

---

## The sample corpus

Built by `scripts/make_sample_corpus.py`, not hand-written, so its construction is
inspectable. Regenerate with:

```bash
uv run python scripts/make_sample_corpus.py
```

### Why synthetic

Real resumes are personal data and cannot go in a repository. Synthetic documents
also give a property real ones do not: **every cue slot exists in every resume**,
so the experimental design stays balanced instead of losing whichever candidates
happen to lack an "Interests" line.

Two of the twelve deliberately omit slots (`R12` has no `FUNDING_NOTE` or
`CULTURAL_ACTIVITY`), because real corpora are ragged and the pipeline must record
a family as *skipped* rather than invent a slot. That path is exercised by
`test_missing_slots_are_recorded_not_faked`.

### Templates

Each resume is a template plus a baseline value per slot:

```json
{
  "resume_id": "R01",
  "job_family": "software_engineering",
  "template": "{{NAME}}\n{{LOCATION}} | {{EMAIL}} | ...",
  "baseline_slots": {"NAME": "Jordan Avery", "PRONOUN_SUBJ": "They", ...},
  "text": "Jordan Avery\nBoston, MA | ..."
}
```

Substitution is therefore **exact by construction**, which is what lets the
validator prove a variant differs in the cue and nothing else.

Available slots: `NAME`, `EMAIL`, `PHONE`, `LOCATION`, `LINKEDIN`,
`PRONOUN_SUBJ`, `PRONOUN_POSS`, `AFFILIATION`, `LANGUAGES`, `CULTURAL_ACTIVITY`,
`GRAD_YEAR`, `EARLY_CAREER_NOTE`, `HOBBIES`, `FUNDING_NOTE`, `UNIVERSITY`.

### Three details that are not incidental

**Baseline names** (`Jordan Avery`, `Casey Sterling`, …) are ambiguous in gender
and ethnicity *and* appear in none of the perturbation name lists, so a baseline
row cannot accidentally contribute to a demographic group. They are a convenient
reference point, not a claim that any name is neutral.

**Pronouns are followed by past-tense verbs** throughout ("She led", "They led"),
so the singular *they* level is grammatical. A grammar error in one level would be
a confound: the model would react to broken English, not to gender.

**Fit is graded by hand** — `strong` / `moderate` / `weak` — with three
cross-family pairs kept as negative controls. Phase 2's gate needs known-good and
known-bad pairings to test against, and a fairness result computed over nonsense
pairings measures nothing.

| job | role | family |
| --- | --- | --- |
| J01 | Senior Backend Engineer | software_engineering |
| J02 | Platform / SRE | software_engineering |
| J03 | Senior Data Scientist, Experimentation | data_science |
| J04 | Growth Marketing Manager | marketing |
| J05 | Senior Financial Analyst, FP&A | finance |
| J06 | Mechanical Design Engineer | mechanical_engineering |

---

## Using a real public corpus

The commonly used Kaggle **Resume Dataset** (~960 resumes, 25 categories):

```bash
# 1. download UpdatedResumeDataSet.csv into data/raw/  (git-ignored)
#    https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
# 2. run
uv run frs pipeline --config configs/kaggle.yaml --dry-run   # check the count first
uv run frs pipeline --config configs/kaggle.yaml
```

What changes:

- **No templates.** The perturbation layer falls back to regex slot *detection*,
  so the content-preservation validator does real work. Expect a pass rate below
  100%; read `variants/skipped.json` for what could not be perturbed and why.
- **Fewer cue families.** `configs/kaggle.yaml` enables only `name`,
  `location_marker` and `age_marker` — the families whose slots can be located
  reliably. A mis-placed edit is worse than a missing one.
- **Stratified capping.** `data.max_resumes` samples across job categories rather
  than taking the first *N* rows, which in that CSV would be one category.
- **Duplicates are reported.** Public corpora contain them, and a duplicated
  resume enters the design as two "independent" blocks when it is really one,
  understating every standard error downstream.

### Your own documents

```bash
uv run frs pipeline --corpus dir --set data.resumes_path=/path/to/resumes
```

Accepts `.txt`, `.md` and `.json`. If you supply `template` + `baseline_slots` in
the JSON, you get the exact-substitution path.

---

## Name lists

```bash
uv run frs list names
```

| list | grade | cells | notes |
| --- | --- | --- | --- |
| `bertrand_mullainathan_2004` | `peer_reviewed` | 2 ethnicities × 2 genders, 9 names each | **the default**; perception-validated in the source study |
| `extended_five_group` | `literature_informed` | 5 ethnicities × 2 genders | for intersectional work; **warns on load** |

Selecting a non-peer-reviewed list emits a `UserWarning` that is recorded in the
audit report *and* the model card, so the caveat appears next to the numbers rather
than having to be asked about.

### Supplying a specific paper's list

```csv
name_id,first,last,gender,race_ethnicity,perceived_strength,citation
p001,Emily,Walsh,female,White,high,"Author (Year). Title."
```

```bash
uv run frs pipeline --set perturbation.name_list=my_list \
                    --set perturbation.name_csv=data/lexicons/my_names.csv
```

Balance matters: sampling happens **within** gender × ethnicity cells, because an
unbalanced subsample confounds the marginal gender effect with the ethnicity mix of
each gender's names.

---

## Data quality gates

Applied automatically in `prepare_data`, and reported in the manifest:

| gate | action |
| --- | --- |
| length band (60-2500 words) | out-of-band resumes dropped, exclusions **returned not logged-and-forgotten** |
| exact duplicates | reported, with a manifest warning |
| near duplicates (Jaccard ≥ 0.9) | reported, with a manifest warning |
| pairing coverage | every job and resume must appear at least once |
| text cleaning | idempotent, applied **before** perturbation so baseline and variant are cleaned identically |

That last one is not cosmetic: if a baseline were cleaned and its variant were
not, the diff would contain whitespace noise and the validator's span accounting
would fail for reasons unrelated to the cue.

---

## Privacy

- The shipped corpus is synthetic. No real person is described.
- Real resumes, if you use any, stay in `data/raw/` or `data/external/`, both
  git-ignored.
- Run artifacts contain full resume text (that is what makes an audit checkable),
  so `runs/` is git-ignored too. Treat a run directory built from real resumes as
  personal data.
- API keys live in `.env`, are read from the environment only, and are never
  written to any artifact.
