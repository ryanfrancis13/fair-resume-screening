# Architecture

## The one structural idea

**The screening side and the fairness side communicate only through a tidy score
table.**

```
                    ┌──────────────────────────────────────────┐
   data ──▶ perturb ─▶ screening ──▶  ScoreRecord rows  ──▶ fairness ──▶ report
                    └──────────────────────────────────────────┘
                                  frs/schemas.py
```

Nothing in `frs.fairness` knows how a score was produced. Nothing in
`frs.screening` knows how it will be audited. Three consequences follow, and they
are the reason for the split:

1. **The audit runs offline on committed artifacts.** `frs audit --run-id ...`
   re-reads `scores/scores_baseline.csv` and calls no model. Improving a metric or
   fixing a statistic costs nothing and cannot change the underlying scores.
2. **The auditor can be validated.** Feed it a score table from a provider with a
   known injected bias and you can measure the suite's sensitivity and
   specificity. That is impossible if the analysis is entangled with the API.
3. **A third party can reproduce the analysis** with the artifact folder alone —
   no key, no budget.

---

## Layers

```
src/                     ← imported as `frs` (see "The src/ mapping" below)
├── config.py            layered config: defaults < YAML < env < --set
├── schemas.py           the data contracts every layer shares
├── cli.py               one command per build-plan phase
├── validate_auditor.py  the oracle: does the audit suite itself work?
│
├── data/                corpus loading, cleaning, resume↔JD pairing
├── llm/                 provider abstraction, caching, rate limits, retries
├── anonymize/           pre-inference redaction + leakage measurement
├── screening/           rubric, prompts, parsing, grounding, batch, ranking
├── perturb/             cue registry, name lists, generation, validity checks
├── fairness/            7 metric families, statistics, conflict analysis
├── mitigate/            5 strategies + fairness/utility evaluation
├── pipeline/            orchestration, run manifests, artifacts
├── report/              plots, HTML/Markdown, explanation cards, model card
├── app/                 Streamlit dashboard (reads artifacts only)
└── web_api.py           FastAPI single-resume evaluator (separate from the audit)
```

### The `src/` mapping

The source sits directly under `src/` — `src/screening/`, `src/fairness/`,
`src/utils/` — matching §5 of the build brief. It is **imported as `frs`**, via
`package-dir = { "frs" = "src" }` in `pyproject.toml`.

The indirection is not decoration. Putting `src/` on `sys.path` directly would
make `config`, `data`, `utils`, `report`, `schemas`, `app`, `pipeline` and `cli`
top-level importable names — eight of the most collision-prone names in Python —
and it would break all 105 `from ..module import` statements in the tree with
*"attempted relative import beyond top-level package"*. Mapping the directory to
one package name keeps the brief's layout on disk while the code stays a normal,
installable, collision-free package.

The one cost: setuptools' auto-discovery cannot follow a *renaming*
`package-dir`, so subpackages are listed explicitly under
`[tool.setuptools] packages`. A forgotten entry silently omits a subpackage from
the built wheel, so `tests/test_packaging.py` asserts the list matches what is on
disk. Hatchling was tried first and rejected: its `sources` remap cannot produce
an editable install ([editables#20](https://github.com/pfmoore/editables/issues/20)).

### `llm/` — why a provider abstraction

Not portability for its own sake. `MockProvider` is a **measurement instrument**:

- it runs the whole pipeline with no key and no network, so plumbing bugs are
  caught by `pytest` rather than by burning quota;
- it is a *competent* scorer — scores come from real requirement-coverage
  arithmetic over the prompt's own resume and job blocks — so the Phase-2
  discrimination gate is a real check even offline;
- it is a **bias oracle**: `mock_bias` injects an offset of known size, and
  `mock_noise_sd` injects decoding noise of known size, which is what makes
  `frs validate-auditor` possible.

Signals are inferred from the resume **text** using the same published lexicons a
real model would see, never from pipeline metadata. Otherwise the oracle would
only confirm our own bookkeeping.

**Cache keys cover everything that can change a response**: prompt, model,
provider, decoding parameters *including the sample index*, and a provider
`behaviour_key`. For a local provider the behaviour key includes a fingerprint of
its own source, because there the code is an input — see
`docs/METHODOLOGY.md` §9 for the two bugs this caught.

The token bucket is skipped for unmetered providers (`metered = False`), because
pacing a local function call to 12 requests/minute turns a 3,000-call offline run
into a four-hour wait for no reason.

### `screening/` — why the score is decomposed

Asking for "a fit score out of 100" invites an unfalsifiable judgement. A weighted
rubric with per-dimension evidence buys three things:

- **bias localisation** — a cue that moves `communication_clarity` but not
  `technical_skills` is behaving like a proxy;
- **auditability** — evidence quotes are checked against the source, so
  `evidence_support_rate` measures grounding rather than eloquence;
- **constraint** — the model has less room for holistic gut feel, which is where
  demographic priors enter.

The parser assumes models misbehave: markdown fences, trailing commas, truncated
objects, `"85/100"`, title-cased dimension names. Each is repaired in one place,
because **a parse failure is a missing data point in a designed experiment** — and
missing data that correlates with the condition would bias the result. The overall
score is recomputed from the dimensions using the rubric weights, so a model's own
arithmetic cannot override the weighting.

### `perturb/` — where the research gap lives

Nine cue families: names plus eight non-name types. Each level carries the
attribute it signals, its provenance, its citation, and any caveat. Families that
cannot be varied cleanly (`age_marker`) or that rewrite prose (`writing_register`)
are typed as `protected_confounded` / `validity_proxy` rather than `protected`, and
their caveats are printed next to their numbers.

Two substitution paths converge on the same validated variant objects: exact
template slots for the curated corpus, regex detection for a public dataset.

### `fairness/` — orchestration order matters

```
1. reliability      ← noise floor and ICC decide whether anything below is readable
2. metric families  ← counterfactual, group parity, allocational, ranking,
                      intersectional, rationale, calibration
3. multiplicity     ← ONE Benjamini-Hochberg pass over the whole suite
4. conflicts        ← rules read the corrected results
```

Families whose preconditions are unmet (no labels, no intersecting cues, no
repeats) are recorded as **skipped with the reason**, so a reader can tell an
absent metric from a passing one.

### `web_api.py` — the single-resume evaluator

The web evaluator reuses `screening/` and `llm/` but is deliberately not part of
the audit path: one resume cannot establish statistical fairness, so it produces
decision support only.

- **Gemini first, offline fallback.** Each submission is scored by Gemini; if
  every attempt fails, `MockProvider` scores it and the result is labelled as a
  fallback with the reason.
- **Median of 3.** Hosted models vary between identical calls even at
  temperature 0, so Gemini is called `SCORING_SAMPLES` (3) times in parallel and
  the middle result is reported. A real outcome is returned rather than an
  average, so score, dimensions and evidence stay consistent. With only two
  successes the lower is used; with none, the fallback runs.
- **In-memory result cache.** The last 64 Gemini results are held in process
  memory, keyed by a hash of model, resume and job text, so a repeat submission
  returns the same score. Nothing is written to disk, which keeps the "uploads
  are processed in memory" promise. Fallback results are never cached.
- **Offline requirement parsing.** The mock grader scores by requirement
  coverage. A pasted job description has no structured lists, so it reads inline
  `Requirements: a, b, c` lines and ignores prompt placeholders such as
  `(see posting text)`. Treating a placeholder as a requirement drives coverage
  to roughly zero and makes the offline score far lower than Gemini's.

The offline score is a keyword-overlap baseline, not an independent truth, so a
gap against Gemini is expected to shrink but not vanish.

### `pipeline/` — the run directory is the unit of reproducibility

```
runs/<run_id>/
├── manifest.json          resolved config, seeds, git commit + dirty flag,
│                          environment, corpus hashes, per-stage telemetry
├── resolved_config.yaml   replay the run verbatim
├── corpus/                the exact resumes, jobs and pairs used
├── variants/              every variant, its text, its edits, its verdict
├── scores/                the tidy score table, per arm, CSV + JSONL
├── audit/                 metric table, full report, conflicts
├── mitigation/            per-arm comparison and utility costs
├── report/                HTML, Markdown, plots, cards, model card
└── logs/run.jsonl         structured log
```

The manifest records the git commit **and whether the tree was dirty**, because a
result produced from uncommitted code is not reproducible and the artifact should
say so rather than imply otherwise. It is written incrementally, so an interrupted
run still describes how far it got.

Run ids are `<timestamp>_<name>_<config-hash>`: two runs differing in any
parameter cannot collide, and two runs sharing a tag are known to share settings.

---

## Design decisions worth defending

**Dataclasses, not pydantic.** Keeps the dependency set to numpy/pandas/scipy/
matplotlib/yaml/dotenv and avoids v1-versus-v2 breakage in a project that has to
still run next year.

**Statistics written out, not imported.** Cluster-robust standard errors, the
permutation test, Platt scaling and the log-odds prior are implemented in ~600
readable lines. A reviewer can check the gradient and the sandwich estimator
directly. `statsmodels` is an optional extra, not a load-bearing dependency.

**`src/` on disk, `frs` on import.** See "The `src/` mapping" above. The brief's
directory names are exactly what a reader sees; the import name keeps the package
installable and collision-free.

**One place decides what a missing text value means.** `frs.utils.frames` owns
every string coercion on a score table. Score tables round-trip through CSV, so a
column that is empty for baseline rows comes back as `NaN`, and pandas 2.x
`astype(str)` used to quietly turn that into the string `"nan"` — pandas 3.0
stopped doing so, and code that relied on it started raising
`AttributeError: 'float' object has no attribute 'split'`. Rather than three dozen
scattered `.fillna("")` calls, the coercion is centralised and explicit: a missing
text value is the empty string, on every pandas version.

**Zero-dependency Markdown tables.** `DataFrame.to_markdown` needs `tabulate`. A
report should not fail to render because of a transitive dependency the project
never declared.

**Third-party numerical noise is handled at the source.** NumPy built against
Apple's Accelerate BLAS raises spurious `overflow encountered in matmul` warnings
on provably benign products above a size threshold. The linear algebra runs inside
`np.errstate(all="ignore")` **and** every stage is checked for finiteness
explicitly, so the guarantee comes from an assertion rather than from the warning
machinery.

---

## Extending it

| task | where |
| --- | --- |
| add a cue family | `perturb/cues.py` — add a `CueFamily` with levels, citation, caveat |
| add a name list | `perturb/names.py` — or drop a CSV and pass `--set perturbation.name_list=...` |
| add a fairness metric | new module in `fairness/`, return `MetricResult` rows, register in `suite.py` |
| add a conflict rule | `fairness/conflicts.py` — a `_rule()` returning explanation + recommendation |
| add a mitigation | `mitigate/strategies.py` — pre-inference arm or post-hoc transform |
| add a provider | subclass `LLMProvider`, register in `llm/client.build_provider` |
| change the rubric | `screening/rubric.py` — weights must sum to 1.0; `validate()` enforces it |
| add a subpackage | create `src/<name>/__init__.py`, then add `"frs.<name>"` to `[tool.setuptools] packages` — `tests/test_packaging.py` will fail until you do |

A new metric needs no LLM calls to develop: point it at a finished run's
`scores/scores_baseline.csv` and iterate.
