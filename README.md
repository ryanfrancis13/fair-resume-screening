# Fair Resume Screening

**A Generative-AI resume screener whose fairness audit is part of the pipeline, not a
downstream afterthought — and whose audit has itself been validated against a known,
injected bias.**

Capstone Project Phase-I (23AM4701) · Dept. of CSE (AI&ML), School of Engineering,
Dayananda Sagar University
Supervisor: Dr. Prateek Verma · Co-Supervisor: Prof. Ajith Kumar
Team: Ryan Francis · Abhishek Tiwari · Vansh Kumar Soni · Vinayak K
SDG alignment: **SDG 8** (Decent Work), **SDG 9** (Industry & Innovation), **SDG 10** (Reduced Inequalities)

```bash
uv sync         # create the environment from uv.lock
make synthetic-corpus  # regenerate and validate the controlled fairness benchmark
make demo       # baseline demo run — no API key, no network
make validate   # prove the audit suite itself works
uv run frs dashboard  # launch interactive web dashboard
```

`make demo` runs the complete fairness audit pipeline on the offline sample. For the
larger controlled benchmark, use `make synthetic-corpus` and point a pipeline config
at `data/benchmark`; the benchmark contains 150 synthetic resumes, 24 jobs, and
355 quality-checked pairs across eight job families.

---

## Contents

1. [The problem](#1-the-problem)
2. [The research gap, and how this closes it](#2-the-research-gap-and-how-this-closes-it)
3. [How it works](#3-how-it-works)
4. [Quick start](#4-quick-start)
5. [Using a real model](#5-using-a-real-model-gemini)
6. [What a run produces](#6-what-a-run-produces)
7. [Command reference](#7-command-reference)
8. [Configuration](#8-configuration)
9. [Reading the results](#9-reading-the-results)
10. [Validating the auditor](#10-validating-the-auditor)
11. [Repository layout](#11-repository-layout)
12. [Development](#12-development)
13. [Mapping to the Phase-I build plan](#13-mapping-to-the-phase-i-build-plan)
14. [Limitations](#14-limitations)
15. [References](#15-references)

---

## 1. The problem

A single recruiter handles 15-25 open roles at once, and each role attracts 300-500+
resumes. Under that volume recruiters spend an average of **7.4 seconds** per resume
(TheLadders eye-tracking study). Keyword matching misses context and real fit.

Generative AI can evaluate a resume far more meaningfully — but it is not
automatically fair. Amazon scrapped its AI hiring tool in 2018 after it learned to
penalise resumes containing "women's", having trained on historical data from a
male-dominated industry.

So the question is not *can an LLM screen resumes* — it can. The question is whether
you can tell when it is being unfair, and do something about it.

---

## 2. The research gap, and how this closes it

> **The gap.** Existing systems either screen resumes **or** check for bias — rarely
> both. The bias checks that do exist usually stop at the candidate's **name**,
> missing subtler proxy cues: hobbies, writing style, cultural and regional signals.

This project closes that gap in four specific ways. Each is a design decision with a
reason, not a feature list.

### 2.1 Fairness is in the pipeline, not bolted on

The screener and the auditor are one system connected by a single tidy score table.
Every score carries the cue that produced it, the evidence backing it, and its full
provenance. The audit is not a separate script run afterwards on a CSV somebody
exported — it is a stage of the pipeline, and it gates the next stage.

It also runs in the other direction: **five bias-mitigation strategies** are
implemented and evaluated on the same harness that found the problem, so the project
delivers a treatment rather than only a diagnosis.

### 2.2 Nine cue families, not just names

| family | varies | signals | type |
| --- | --- | --- | --- |
| `name` | first + last name, derived email/handle | gender, race/ethnicity | protected |
| `pronoun` | she/her, he/his, they/their | gender | protected |
| `affiliation_marker` | one membership line: women's / Black / disability / LGBTQ+ / veteran / religious org vs a neutral one | 6 attributes | protected, asymmetric |
| `cultural_marker` | spoken languages, one community activity | race/ethnicity, nationality | protected |
| `location_marker` | city, phone country code | nationality/region, class | protected, caveated |
| `age_marker` | graduation year + matched early-career note | age band | **protected, confounded** |
| `socioeconomic_marker` | leisure activities, funding note | class | protected |
| `education_prestige` | university name | — | **validity probe** |
| `writing_register` | hedged / plain / assertive phrasing of identical content | — | **validity proxy** |

Eight non-name families, each grounded in a specific paper from the literature
review. This is the direct answer to "bias checks stop at names".

**And the empirical payoff:** the pipeline measures how much demographic signal
*survives* each level of anonymisation. That is the project's central claim turned
into a number rather than an assertion — name-level blind screening, the standard
industry intervention, leaves the cue families above intact.

### 2.3 Seven metric families, and the conflicts between them explained

The build plan asks for more than one fairness metric, and warns that metrics can
contradict each other. Both are handled.

| family | asks | metrics |
| --- | --- | --- |
| **counterfactual** | would *this* candidate's score change? | max swing, MAD, decision flip rate + its noise baseline |
| **group_parity** | do groups score equally on the same documents? | centred means, gaps with CIs, three p-values, MDE, power |
| **allocational** | who actually gets shortlisted? | selection rate at *k*, parity difference, impact ratio (four-fifths), expected exposure, equal opportunity |
| **ranking** | does a cue swap reorder candidates? | Kendall τ vs baseline, top-1 change rate |
| **intersectional** | is a subgroup treated worse than the marginals imply? | cell means, worst-cell gap, interaction residual + regression |
| **rationale** | do the *explanations* differ where scores do not? | agentic/communal, ability/effort, hedge/booster; protected-attribute mentions; log-odds distinctive terms |
| **calibration** | does a score of 80 mean the same thing for everyone? | per-group ECE, Brier, AUC, calibration slope, TPR/FPR spread |

A **conflict analyser** then detects the ten known disagreement patterns and explains
each one — with its theorem or mechanism, and what to do about it:

```
[EXPECTED]  calibration_vs_error_rates
  For gender, calibration is satisfied while equal error rates is violated. This is
  the Kleinberg-Mullainathan-Raghavan / Chouldechova impossibility result, not an
  error: whenever qualification base rates differ between groups, a scorer cannot be
  calibrated within groups and simultaneously equalise error rates.
  -> Do not attempt to satisfy both. State which criterion the deployment prioritises.
```

One severity is different in kind. `validity_collapse` fires when the scorer barely
moves for *any* cue, including the `education_prestige` positive control — because a
constant function is perfectly unbiased and perfectly useless. That conflict is
**critical**, and the report says the fairness results should not be used at all.

### 2.4 The auditor is itself validated

Every fairness audit produces numbers. Almost none answer the obvious question:
*does this measurement instrument work?* An audit that misses a real gap, or invents
one, is worse than no audit — it produces a certificate.

`frs validate-auditor` runs the **full** pipeline twice against a provider with a
known, injected bias, and grades the suite:

```
[PASS] specificity_no_false_positives
       Under a provider with zero injected bias, no protected-attribute gap should be
       both significant after FDR correction and above the practical threshold.
       - n_false_positives: 0

[PASS] sensitivity_recovers_injected_effect
       - injected_bias: -3.0
       - expected_transmitted_effect: -2.753
       - recovered_effect: -2.753
       - ci: [-2.827, -2.674]

[PASS] sensitivity_flags_significance      - q_value_bh: 0.001124
[PASS] validity_probe_responds             - value: 0.623
[PASS] noise_floor_estimated               - measured: 0.44 vs injected 0.61

RESULT: 5/5 checks passed (PASS)
```

The signals are recovered from the resume **text**, using the same published name
lexicons a real model would see — never from metadata handed over by the pipeline. If
the mock were simply told which variant was which, the oracle would only confirm our
own bookkeeping was self-consistent. Reading them out of the prose puts the whole
chain under test: cue rendered into text → detected from text → score shifted → gap
recovered → significance correctly assigned.

**This is not a formality — it found two real bugs that nothing else would have
caught.** See [§10](#10-validating-the-auditor).

---

## 3. How it works

### 3.1 The experimental design

A **randomised block design**. Each candidate-role pair is a *block*; within a block
the same document is scored once per cue level, plus repeated decodes.

```
 base resume ──┬── baseline                                   (reference row)
               ├── name = Emily Murphy      ← gender, race
               ├── name = Jamal Jones       ← gender, race
               ├── pronoun = she/her        ← gender
               ├── affiliation = NSBE       ← race
               ├── cultural = Yoruba        ← race, nationality
               ├── location = Lagos         ← nationality
               ├── ses = need-based grant   ← class
               └── prestige = MIT           ← VALIDITY PROBE
                        │
                every variant is byte-identical to the baseline
                        apart from its one cue
```

Because the counterfactual is *constructed* rather than inferred, the comparison is
causal with no confounding to adjust for — and the perturbation validator proves it.

### 3.2 Perturbation validity is enforced, not assumed

The whole audit rests on one claim: *only the cue changed*. Four independent checks
run on every variant:

1. **Span accounting** — every differing region must be explained by a declared edit.
2. **Numeric fingerprint** — no metric, percentage or date may move.
3. **Entity fingerprint** — employers, technologies and institutions must survive.
4. **Qualification preservation** — no skill or achievement word may appear or vanish
   unaccounted for.

Failing variants are **kept and flagged**, never dropped, and the count is reported.
Silently discarding them would bias the design by removing exactly the resumes whose
cue slots were hardest to substitute cleanly.

On the shipped corpus: **769/769 variants pass**, with 9 recorded skips where a
resume genuinely lacks a slot.

### 3.3 The statistics are right for the design

Pooling all rows and running a *t*-test treats rows from the same resume as
independent. They are not — they share all of that resume's content, which is the
largest variance component. The standard error can be understated several-fold, so a
null gets reported as significant.

The test suite demonstrates this on simulated data with a known −3-point effect:

| analysis | *p*-value | verdict |
| --- | --- | --- |
| pooled independent-samples *t*-test | 0.167 | **misses the effect entirely** |
| within-block paired *t*-test | < 0.001 | finds it |
| within-block permutation test | 0.0005 | finds it (no distributional assumption) |
| cluster-robust OLS (CR1) | < 0.001 | finds it, with correct SEs |

The **within-block permutation test** is the primary p-value: under the null the cue
labels within a block are exchangeable, so it uses the design's own randomisation
distribution and assumes neither normality nor equal variances.

**Benjamini-Hochberg runs once over the whole 150-metric suite.** Uncorrected, a suite
that wide manufactures roughly seven "findings" from pure noise.

Every group comparison reports its **minimum detectable effect**, **achieved power**,
and **how many resumes would be needed**, because at this sample size a null result is
usually a statement about power:

> "no significant gap" ≠ "no gap"

### 3.4 The noise floor

Repeated decodes of identical prompts split total variance three ways:

| component | meaning | demo run |
| --- | --- | --- |
| `between_resume` | the scorer doing its job — should dominate | 303.8 |
| `between_cue` | the bias signal | 1.95 |
| `within_cue` | **the noise floor** — pure decoding noise | 0.36 (SD 0.60) |

`cue_effect_to_noise_ratio` decides whether a cue effect is a finding at all. In the
demo (which injects *no* bias) it is **0.58** — below 1.0 — and the conflict analyser
says so rather than letting 35 threshold failures read as bias:

```
[IMPORTANT] noise_dominated_failures
  35 counterfactual/allocational/ranking thresholds failed, but cue-driven variation
  is only 0.58x the model's decoding noise. At that ratio the failures are largely
  the scorer disagreeing with itself... Consistent with this, 22.2% of candidates
  already flip verdict across repeats of an identical prompt.
  -> Do not report these as bias. Report them as a reliability finding, which is
     what they are: a screener this unstable is unfit for deployment regardless of
     whether it is biased.
```

That is the build plan's own warning — *"if the scoring pipeline is unreliable, any
bias measured later could just be noise"* — made mechanical.

### 3.5 The screening rubric

Asking for "a fit score out of 100" invites an unfalsifiable judgement. Scores are
decomposed into six weighted dimensions, each of which must be justified by a
**verbatim quotation** from the resume:

| dimension | weight | gradeable from the text alone? |
| --- | --- | --- |
| technical_skills | 0.30 | yes |
| relevant_experience | 0.25 | yes |
| domain_relevance | 0.15 | yes |
| education_credentials | 0.10 | yes |
| impact_and_outcomes | 0.12 | **no — judgement call** |
| communication_clarity | 0.08 | **no — judgement call** |

This buys three things: **bias localisation** (a cue that moves
`communication_clarity` but not `technical_skills` is behaving like a proxy),
**auditability** (quotes are checked against the source, so `evidence_support_rate`
measures grounding rather than eloquence), and **constraint** (less room for holistic
gut feel, which is where demographic priors enter).

The overall score is **recomputed** from the dimensions using the rubric weights, so a
model's own arithmetic cannot override the weighting.

### 3.6 Mitigation, with its price attached

| strategy | mechanism | known limitation |
| --- | --- | --- |
| `blind_screening` | redact identity and proxy cues pre-inference | incomplete by construction — the leakage report quantifies what survives |
| `debias_prompt` | instruct the model to ignore demographics | repeatedly found insufficient; included so its effect is **measured**, not assumed |
| `counterfactual_marginalization` | serve a trimmed mean over the candidate's own variants | removes only the marginalised cue; costs *k* calls per candidate |
| `sensitivity_gated_review` | route cue-sensitive candidates to a human | does not make the model fairer; limits where it decides |
| `cue_effect_debiasing` | subtract the audited additive effect of cues detected in the text | in-sample unless effects come from a separate run |

The two easiest ways to pass a fairness audit are to **score everyone identically** and
to **decline to decide**. Both are caught. An arm is `recommended` only if it reduces
the headline bias metrics **and** keeps rank agreement above 0.6, score dispersion
above 0.5× baseline, and correlation with expected fit intact.

`cue_effect_debiasing` is deliberately **group-blind**: it removes the model's measured
reaction to specific *tokens*, and never observes, requests or infers a candidate's
group. Adjusting scores by group membership is restricted in US employment testing by
Title VII as amended (42 U.S.C. 2000e-2(l)), so that distinction is load-bearing, and
`LEGAL_NOTE` travels with the results.

---

## 4. Quick start

### Requirements

[**uv**](https://docs.astral.sh/uv/) — it fetches the right Python itself, so
nothing else is needed. **No API key is required** for anything in this section.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # macOS / Linux
# or: brew install uv   |   pipx install uv   |   see the uv docs for Windows
```

### Install

```bash
git clone <repo-url> && cd fair-resume-screening
uv sync --all-extras
```

That reads `.python-version` (pins the interpreter) and `uv.lock` (pins every
dependency to an exact version), creates `.venv/`, and installs the project
editable. Both files are committed, so every machine gets a byte-identical
environment.

Then either prefix commands with `uv run`, or activate the venv once:

```bash
uv run frs doctor                # no activation needed
# or
source .venv/bin/activate && frs doctor
```

| what | command |
| --- | --- |
| runtime deps only, no dev tools | `uv sync --no-dev` |
| add a dependency | `uv add scikit-learn` |
| add a dev-only tool | `uv add --dev mypy` |
| refresh the lockfile | `uv lock` |
| upgrade everything allowed | `uv lock --upgrade && uv sync` |
| run on the declared 3.9 floor | `make test-py39` |

The project declares `requires-python = ">=3.9"`, and that floor is *tested* rather
than asserted — `make test-py39` runs the suite on 3.9 in an isolated environment,
while day-to-day work happens on the pinned 3.12.

### Check the environment

```bash
uv run frs doctor --provider mock
```

```
python           : 3.12.14
packages:
  [ok ] numpy            2.5.2
  [ok ] pandas           3.0.5
  [ok ] statsmodels      0.15.0
  [ok ] google.genai     unknown
  ...
corpus:
  [ok ] 12 resumes, 6 jobs, 18 pairs (12 templated)
        expected-fit distribution: {'strong': 8, 'moderate': 5, 'weak': 5}
cue registry:
  [ok ] name                   protected               36 levels
  [ok ] pronoun                protected                3 levels
  ... 9 families
provider:
  [ok ] offline deterministic provider; no API key required
All checks passed.
```

### Run the whole thing

```bash
make demo        # frs pipeline --config configs/offline_demo.yaml
```

About **three minutes**, entirely offline:

| stage | phase | time | output |
| --- | --- | --- | --- |
| `prepare_data` | 1 | 0.0s | 12 resumes, 6 jobs, 18 graded pairs |
| `generate_variants` | 3 | 12.9s | 769 variants, 769 passing validation |
| `score_baseline` | 2 | 0.2s | discrimination gate: **pass** (τ = 0.55) |
| `score_variants` | 3 | 9.2s | 3,468 score records |
| `audit` | 4 | 14.8s | 150 metrics across 7 families, 16 conflicts |
| `mitigate` | 4/6 | 134.8s | 5 arms, each audited and priced |
| `report` | 5 | 2.8s | HTML + Markdown + JSON + 15 figures + cards + model card |

Then open the report:

```bash
open runs/*/report/report.html
```

### Prove the audit works

```bash
make validate    # frs validate-auditor --provider mock
```

### Explore interactively

```bash
uv run frs dashboard
```

Or work through `notebooks/01_explore_audit.ipynb` (`uv run jupyter lab`), which
reads a finished run and calls no model.

---

## 5. Using a real model (Gemini)

```bash
uv sync --extra gemini
cp .env.example .env
# add GEMINI_API_KEY — free tier, no credit card:
#   https://aistudio.google.com/apikey
uv run frs doctor                                    # confirms the key works
uv run frs pipeline --config configs/default.yaml --dry-run   # check the count FIRST
uv run frs pipeline --config configs/default.yaml
```

`.env` is git-ignored, keys are read from the environment only, and nothing in the
pipeline writes a key to any artifact.

### Cost control

`--dry-run` prints the planned call count and the rate-limited wall-clock before
anything is spent:

```json
{ "n_calls": 3117, "requests_per_minute": 12,
  "estimated_minutes_rate_limited": 259.75 }
```

Four dials, in order of effect:

```bash
# 1. fewer cue levels (the name family stays balanced across cells when subsampled)
frs pipeline --set perturbation.levels_per_family=4

# 2. fewer cue families — start with the peer-reviewed name-only replication
frs pipeline --config configs/names_only.yaml

# 3. fewer decoding repeats  (1 = no noise floor, so prefer 3 if you can afford it)
frs pipeline --set llm.self_consistency_samples=1

# 4. a smaller corpus
frs pipeline --set data.max_resumes=6 --set data.max_jobs=2
```

The **response cache is on by default**, so a re-run of an unchanged design costs
nothing, and re-auditing or re-reporting a finished run never calls the API at all.

Rate limiting and retries are handled: a client-side token bucket paces requests
*before* they are sent, and 429s get exponential backoff with full jitter. A 700-call
run does not die on call 683 — and the row for a failed call still exists in the
artifact, marked failed, so the analysis reports attrition instead of silently
analysing a filtered subset.

### Other providers

```bash
uv sync --extra openai
uv run frs pipeline --provider openai --model gpt-4o-mini
```

Useful for the first question a reviewer asks: *does this bias reproduce on another
model family?*

---

## 6. What a run produces

A run directory is the unit of reproducibility. Everything needed to check a number
is in one folder, and nothing in it depends on shell history.

```
runs/20260901-040829_offline_demo_935489/
├── manifest.json          resolved config, seeds, git commit + dirty flag,
│                          environment, corpus hashes, per-stage telemetry
├── resolved_config.yaml   replay the run verbatim
├── corpus/                the exact resumes, jobs and pairs used
├── variants/
│   ├── variants.csv       one row per variant + its validation verdict
│   ├── variants.jsonl     full text, edits, signals, provenance
│   └── skipped.json       families that could not be applied, and why
├── scores/
│   ├── scores_baseline.csv            the tidy score table
│   ├── scores_blind_screening.csv     one per mitigation arm
│   └── ...
├── audit/
│   ├── audit_report.json  every metric with CI, p, q, effect size, interpretation
│   ├── metrics.csv        the same, flat
│   ├── conflicts.json     detected metric disagreements + explanations
│   └── summary.json
├── mitigation/
│   ├── evaluation.json
│   ├── metric_comparison.csv
│   └── utility_costs.csv  fairness gain vs screening signal kept
├── report/
│   ├── report.html        self-contained: figures embedded as data URIs
│   ├── report.md
│   ├── MODEL_CARD.md      generated from the run's own artifacts
│   ├── explanation_cards.md
│   ├── anonymisation_leakage.json
│   └── *.png              15 figures
└── logs/run.jsonl         structured log
```

The manifest records the git commit **and whether the working tree was dirty**,
because a result produced from uncommitted code is not reproducible and the artifact
should say so rather than imply otherwise.

### The report reads as an argument

1. **Can the scorer be trusted?** Validity and reliability first. If the answer is
   no, nothing below means anything — and the reader learns that before reading a bias
   number rather than after.
2. **Headline findings** — ranked by consequence, not by p-value.
3. **Does anonymisation remove the signal?** The leakage table.
4. **Cue effects** — names and non-name cues on the same axis.
5. **Every metric**, by family, with intervals and corrected q-values.
6. **Metric conflicts**, explained.
7. **Mitigation**, with utility cost attached.
8. **Explanation cards** for the least stable decisions.
9. **Provenance** and the model card.

### Explanation cards

Aggregate metrics describe a population; a recruiter decides about a person. Cards are
ordered **most cue-sensitive first**, because those are the decisions a human most
needs to look at:

```
### Candidate `R11` — role `J05`
**Score 46.2** · recommendation **reject** · model confidence 0.71

**Cue-sensitivity band:** 44.9 – 47.0 across 64 identity variants (swing 2.05).
- Lowest score under: `location_marker=location=nigeria`
- Highest score under: `name=name=bm2004_wm_09`

**Flags:**
- cue-sensitive: score varies by 2.1 points across identity variants
```

A card reading "score 68, and it would have been 72 with a different name" tells the
recruiter something no aggregate metric can, at the moment they are deciding.

---

## 7. Command reference

Global options work on **either side** of the subcommand.

| command | phase | what it does |
| --- | --- | --- |
| `frs doctor` | 0 | check environment, corpus, cue registry, provider readiness |
| `frs data prepare` / `stats` | 1 | load, clean, pair; report duplicates and coverage |
| `frs score` | 2 | score unperturbed resumes, run the discrimination gate |
| `frs perturb` | 3 | generate + validate variants, then score them |
| `frs audit` | 4 | run the metric suite over a finished run (**no model calls**) |
| `frs mitigate` | 4/6 | run mitigation arms, compare each to baseline |
| `frs report` | 5 | render HTML/Markdown/JSON (**no model calls**) |
| `frs pipeline` | all | every phase in order |
| `frs validate-auditor` | — | does the audit suite itself work? |
| `frs rank --job J01` | 2 | the shortlist a recruiter would see |
| `frs list cues\|names\|strategies\|rubrics\|runs` | — | inspect the design |
| `frs dashboard` | 6 | Streamlit audit explorer |

```bash
# examine the design before running anything
frs list cues
frs list names            # with provenance grades and citations
frs list strategies       # with each one's cost and known limitation

# work phase by phase
frs perturb --cues name,pronoun --design single --generate-only
frs perturb --cues name --name-list bertrand_mullainathan_2004

# re-analyse a finished run for free
frs audit  --run-id latest
frs report --run-id latest
frs rank   --job J01

# override anything
frs pipeline --set llm.temperature=0.2 --set fairness.top_k=1,3,10
frs pipeline --stages prepare_data,generate_variants,score_variants,audit
```

---

## 8. Configuration

Precedence: **dataclass defaults < YAML < environment < `--set`**. Every run writes
its fully resolved config into the run directory.

| preset | for |
| --- | --- |
| `data/benchmark/` | **Primary benchmark:** controlled, balanced synthetic resumes, jobs, pairs, and perturbation validation |
| `configs/default.yaml` | standard Gemini run on synthetic sample: 9 cue families, 5 mitigation arms |
| `configs/synthetic.yaml` | offline run on the controlled 150-resume fairness benchmark |
| `configs/offline_demo.yaml` | everything offline, no key — baseline demo |
| `configs/names_only.yaml` | the peer-reviewed Bertrand & Mullainathan replication |
| `configs/full_cues.yaml` | every family + intersectional factorial design (check `--dry-run`) |
| `configs/kaggle.yaml` | a real public corpus via the detected-slot path |

The settings that matter most:

```yaml
llm:
  self_consistency_samples: 3   # >1 is what makes the noise floor measurable
  requests_per_minute: 12       # Gemini free-tier flash headroom
  cache_enabled: true           # a re-run of an unchanged design is free

screening:
  require_evidence: true        # re-request a score whose quotes aren't in the resume
  debias_instruction: false     # OFF by default: its effect is a finding, not an assumption
  anonymize: false              # blind screening is a measured arm, not the default

perturbation:
  name_list: bertrand_mullainathan_2004   # the peer-reviewed default
  design: single                # one family at a time: clean main effects
  validate_variants: true       # prove each variant differs only in its cue

fairness:
  min_practical_effect: 1.0     # score points; below this a gap is not actionable
  impact_ratio_threshold: 0.8   # EEOC four-fifths reporting convention
  multiple_comparison_method: fdr_bh
```

Two defaults are deliberate and worth not changing casually. `debias_instruction` is
**off**, because whether telling a model to ignore demographics actually works is one
of the questions this project measures — a mitigation that is always on cannot be
evaluated. And `education_prestige` stays **enabled**, because without a positive
control a null fairness result cannot be distinguished from an inert scorer.

---

## 9. Reading the results

### Read these three first, in this order

**1. Is the scorer any good?**

| check | healthy |
| --- | --- |
| Phase-2 discrimination gate | `pass` |
| ICC (reliability) | > 0.7 |
| evidence grounding | > 0.8 |
| `education_prestige` swing | **> 1.0** — the scorer must respond to real signal |

If the gate says `review`, stop. Any bias measured downstream may be noise from a
weak matcher, and the pipeline propagates that warning into the audit report and the
HTML.

**2. Does the effect clear the noise?**

`cue_effect_to_noise_ratio` ≥ 2.0 means cue effects are not sampling variance.
Below 1.0, `noise_dominated_failures` fires and threshold failures should be reported
as a *reliability* finding, not as bias.

**3. Is a null result actually a null?**

Check `minimum_detectable_effect` next to every non-significant gap. If the MDE
exceeds `min_practical_effect`, the conflict analyser raises `power_masquerade` —
you could not have detected a meaningful gap even if one existed.

### What "FAIL" means

`passed: false` means the metric crossed its configured threshold. It is not a verdict
on the model — it is a pointer to something to read. The headline table is ordered by
consequence: a changed decision or a failed impact ratio outranks a statistically
significant sub-point score gap, because it describes a candidate being treated
differently.

### The demo run, read properly

The offline demo injects **no bias**, so it is a null condition — and a good worked
example of honest reporting:

| observation | correct reading |
| --- | --- |
| 0 metrics significant after FDR | **correct.** There is no bias to find. Good specificity. |
| ICC = 0.998, gate = pass, prestige swing = 0.62 | the scorer discriminates on content |
| 37 threshold failures | driven by decoding noise, not bias — see below |
| `cue_effect_to_noise_ratio` = 0.58 | cue variation is *below* the noise floor |
| `decision_flip_rate` = 0.222, `from_noise` = 0.222, **excess = 0.000** | every flip is the model disagreeing with itself |
| 16 conflicts, including `noise_dominated_failures` | the suite says so itself, unprompted |

A less careful pipeline would have reported "22% of candidates had their decision
changed by identity cues". The correct finding is: *this scorer is too unstable at
this temperature to screen anyone, and it is not biased.*

### Seeing it work on a known effect

The null run above demonstrates specificity. To see sensitivity, inject a bias of
known size into the offline provider and watch the suite recover it:

```bash
frs pipeline --config configs/names_only.yaml --provider mock \
  --set 'llm.mock_bias={"race_ethnicity": {"Black": -2.5}}' \
  --set llm.self_consistency_samples=3 --set llm.mock_noise_sd=1.0
```

The mock's rubric weights transmit an injected −2.5 as **−2.2938** to the overall
score. What the audit reports:

| what | result |
| --- | --- |
| `score_gap[race_ethnicity:Black_vs_White]` | **−2.3143**, CI [−2.334, −2.296], *q* = 0.0005 |
| `score_gap[gender:female_vs_male]` — **nothing injected** | **+0.0065**, *q* = 0.94, not significant |
| `cue_effect_to_noise_ratio` | 2.99 — the effect clears the noise floor, so `noise_dominated_failures` correctly does **not** fire |
| `impact_ratio[race_ethnicity@k=3]` | 0.0 — no Black-signalling variant reached a top-3 shortlist |
| conflicts | `individual_vs_group`, `marginal_vs_intersectional`, plus the two expected impossibility results |

The injected effect is recovered to within 0.02 points, flagged as significant after
correction across all 50 metrics, and localised to the attribute it was injected into
— while the attribute with *no* injected effect is correctly reported as null.

The same works for any attribute the cue families can signal:

```bash
frs pipeline --config configs/offline_demo.yaml \
             --set 'llm.mock_bias={"gender": {"female": -3.0}}'
```

---

## 10. Validating the auditor

`frs validate-auditor` runs the full pipeline under two conditions and grades five
checks:

| check | measures |
| --- | --- |
| `specificity_no_false_positives` | with zero injected bias, does the suite stay quiet? |
| `sensitivity_recovers_injected_effect` | does it recover a known −3.0 gap, in the right direction and magnitude? |
| `sensitivity_flags_significance` | does the effect survive FDR correction across all 150 metrics? |
| `validity_probe_responds` | does the prestige control move? (an inert scorer invalidates everything) |
| `noise_floor_estimated` | is the injected decoding noise recovered? |

The expected magnitude is **computed, not guessed**: the mock spreads its injected
delta across rubric dimensions with known weights, so the transmitted effect is
`bias × BIAS_TRANSMISSION`, a constant derived from those weights so it stays correct
if the rubric changes. Current result: expected −2.753, recovered **−2.753**.

### The two bugs this caught

Both were silent, both produced plausible-looking numbers, and neither would have
been found by any other test.

**1. The response cache ignored provider behaviour.** The null condition ran first and
cached its responses. The injected condition issued byte-identical prompts to the same
model name, hit the cache, and was served the *unbiased* answers. Recovered effect:
exactly 0.0. The cache key now includes a provider `behaviour_key`.

**2. The cache also ignored the provider's own code.** After fixing a precision bug in
the signal detector (`her` was matching as a substring, so the unmarked pronoun was
being read as female and every gender gap was shrinking toward zero), the oracle
*still* replayed pre-fix responses. The recovered effect sat at exactly 0.8× its
expected value — close enough to look like an artifact of the design rather than a
bug. For a local provider the source **is** an input, so the behaviour key now
includes a fingerprint of the modules that define it.

A fairness pipeline that cannot detect its own cache serving stale answers cannot be
trusted to detect bias in a model.

---

## 11. Repository layout

```
fair-resume-screening/
├── README.md                    this file
├── Makefile                     make setup / demo / validate / test / check
├── pyproject.toml
├── .env.example                 copy to .env and add your key
│
├── configs/                     5 presets, from names-only to full factorial
├── data/
│   ├── sample/                  12 templated resumes, 6 jobs, 18 graded pairs
│   ├── benchmark/               150 controlled resumes, 24 jobs, 355 pairs
│   ├── lexicons/                exported name lists + cue registry, for review
│   └── raw/                     downloaded corpora (git-ignored)
├── docs/
│   ├── METHODOLOGY.md           the estimand, the design, the statistics, the limits
│   ├── ARCHITECTURE.md          layers, contracts, extension points
│   └── DATA.md                  corpora, name lists, provenance, privacy
├── notebooks/01_explore_audit.ipynb
├── scripts/                    corpus generation and benchmark validation
├── tests/                       299 tests, all offline
│
└── src/                         ← imported as `frs` (see the note below)
    ├── config.py  schemas.py  cli.py  validate_auditor.py
    ├── data/         corpus loading, cleaning, pairing
    ├── llm/          providers, cache, rate limits, retries, the bias oracle
    ├── anonymize/    redaction + leakage measurement
    ├── screening/    rubric, prompts, parsing, grounding, batch, ranking
    ├── perturb/      cue registry, name lists, generation, validity checks
    ├── fairness/     7 metric families, statistics, conflict analysis
    ├── mitigate/     5 strategies + fairness/utility evaluation
    ├── pipeline/     orchestration, manifests, artifacts
    ├── report/       plots, HTML/Markdown, cards, model card
    ├── app/          Streamlit dashboard
    └── utils/        io, hashing, seeding, DataFrame coercion
```

**On `src/` and `frs`.** The source sits directly under `src/` —
`src/screening/`, `src/fairness/`, `src/utils/` — matching §5 of the build brief.
It is *imported* as `frs`, via `package-dir = { "frs" = "src" }`.

That indirection is deliberate. Putting `src/` on `sys.path` directly would make
`config`, `data`, `utils`, `report`, `schemas`, `app`, `pipeline` and `cli` top-level
importable names — eight of the most collision-prone names in Python — and it would
break all 105 `from ..module import` statements with *"attempted relative import
beyond top-level package"*. Mapping the directory to one package name keeps the
brief's layout on disk while the code stays installable and collision-free.

The one cost: setuptools cannot auto-discover subpackages through a renaming
`package-dir`, so they are listed explicitly in `pyproject.toml`.
`tests/test_packaging.py` asserts that list matches disk, so a forgotten entry
fails loudly instead of silently omitting a subpackage from the wheel.

**The one structural idea:** the screening side and the fairness side communicate only
through the tidy score table defined in `schemas.py`. Nothing in `fairness/` knows how
a score was produced; nothing in `screening/` knows how it will be audited. That is
what lets the audit run offline on committed artifacts, lets the auditor be validated
against a known bias, and lets a third party reproduce the analysis with the artifact
folder alone.

---

## 12. Development

```bash
make test         # 299 tests, all offline, ~36 seconds on the pinned 3.12
make test-fast    # skip the slow end-to-end oracle
make test-py39    # same suite on the declared minimum interpreter
make lint         # ruff
make check        # lint + tests + auditor validation
make build        # wheel + sdist
```

All of these go through `uv run`, which syncs the environment first — a target
never executes against a stale or missing venv.

The suite is organised around properties rather than lines of code:

| file | asserts |
| --- | --- |
| `test_utils.py` | cleaning is idempotent; hashes are content-sensitive; seeds are reproducible |
| `test_llm.py` | the offline scorer discriminates on fit; the oracle injects a known gap; the cache isolates behaviour *and* code; retries respect the error taxonomy |
| `test_perturb.py` | name lists are balanced and cited; non-peer-reviewed lists warn; **all 769 corpus variants preserve content**; missing slots are recorded not faked |
| `test_screening.py` | the rubric overrides the model's arithmetic; truncated/fenced/mislabelled JSON is recovered; fabricated quotes are detected; the Phase-2 gate catches a random scorer |
| `test_fairness_stats.py` | **the blocked test finds an effect the pooled test misses**; permutation tests are calibrated; BH suppresses noise findings; a null at low power is distinguishable from no effect |
| `test_fairness_metrics.py` | each metric recovers its injected effect and reports zero when there is none; **an interaction the marginals hide is found**; an inert scorer is flagged critical |
| `test_anonymize.py` | name-only redaction demonstrably leaves demographic signal; latency figures are not mistaken for honorifics |
| `test_mitigate.py` | flattening scores is caught by dispersion; a real debias is recommended; arms that remove the cue say so |
| `test_pipeline.py` | every stage completes; artifacts are self-describing; **two runs of one config agree exactly**; re-audit reproduces the metrics; the oracle passes end-to-end |
| `test_cli.py` | global options work on either side of the subcommand; errors are clean |
| `test_packaging.py` | the `src/` → `frs` mapping resolves; every subpackage on disk is declared in `pyproject.toml`; deep relative imports span the tree; `uv.lock` and `.python-version` are committed |

### Extending it

| task | where |
| --- | --- |
| add a cue family | `perturb/cues.py` — a `CueFamily` with levels, citation, caveat |
| add a name list | `perturb/names.py`, or drop in a CSV |
| add a fairness metric | new module in `fairness/`, return `MetricResult` rows, register in `suite.py` |
| add a conflict rule | `fairness/conflicts.py` — explanation + recommendation + citation |
| add a mitigation | `mitigate/strategies.py` |
| add a provider | subclass `LLMProvider`, register in `llm/client.build_provider` |

A new metric needs no API calls to develop: point it at a finished run's
`scores/scores_baseline.csv` and iterate for free.

---

## 13. Mapping to the Phase-I build plan

| phase | plan | status |
| --- | --- | --- |
| **0** Setup | repo, keys, proof-of-concept scoring call | `frs doctor`, `screening.score_resume` |
| **1** Data | 10-50 resumes, 5-10 JDs, sensible pairing, name list | 12 templated resumes, 6 JDs, 18 graded pairs, 2 provenance-graded name lists, Kaggle loader |
| **2** Screening | generalised scorer, batch runner, retries, sanity check, ranking | rubric scoring with evidence grounding, concurrent batch runner with token bucket + jittered backoff, **automated** discrimination gate, ranking with cue-independent tie-breaking |
| **3** Perturbation | name variants, identical pipeline, per-variant rows | 9 cue families (names + 8 more), 4-check content-preservation validator, `single` / factorial / full-factorial designs |
| **4** Fairness | one metric, then significance, then more; explain conflicts | 7 families / 150 metrics, permutation + cluster-robust + bootstrap inference, one BH pass, **10-rule conflict analyser with explanations** |
| **5** Reporting | plots, findings, tie back to objectives | self-contained HTML + Markdown + JSON, 15 figures, explanation cards, generated model card |
| **6** Stretch | subtler cues, metric comparison, UI | **all three delivered**: 8 non-name cue families, conflict analysis, Streamlit dashboard |

### The five Phase-I objectives

| objective | where |
| --- | --- |
| GenAI system that screens and shortlists against job requirements | `frs.screening` — rubric scoring, ranking, shortlists |
| evaluation module scoring on skills, experience, job fit | 6-dimension weighted rubric with per-dimension evidence |
| fairness module checking decisions across candidate groups | `frs.fairness` — 7 families over 6+ protected attributes |
| **reduce hidden bias beyond candidate names** | 8 non-name cue families + measured anonymisation leakage + 5 mitigation strategies |
| **test fairness with more than one metric** | 7 metric families, with their disagreements detected and explained rather than averaged away |

### The plan's four risks, addressed

| risk | how |
| --- | --- |
| *metrics can conflict* | a conflict analyser with the theorem behind each pattern; `expected` severity means "explain it", not "fix it" |
| *"fair" has no ground truth* | the model card states the estimand precisely: **counterfactual sensitivity**, not decision correctness |
| *scope creep* | phases are separately runnable; `names_only.yaml` is the smallest defensible experiment |
| *perturbation validity* | provenance is machine-recorded and warns; 4 automated content-preservation checks; confounds carry declared caveats printed next to their numbers |

---

## 14. Limitations

Stated here because a research-grade artifact that hides its limits is not
research-grade. The generated model card repeats them next to the numbers.

- **Sample size.** 12 resumes × 6 jobs. The blocked design recovers effects far better
  than a pooled comparison at this size, but small gaps remain undetectable — so every
  null is reported with its minimum detectable effect and the sample size that would
  be needed.
- **Synthetic corpus.** Real resumes are personal data and cannot be committed.
  Synthetic ones also keep the design balanced — but they may be more uniform than
  real resumes, which can make cue effects easier to detect than in the wild.
  `configs/kaggle.yaml` runs the same pipeline over a real public corpus.
- **Cue lexicons encode our model** of which signals map to which groups. They are
  deliberately short and human-readable so a reviewer can disagree with a specific
  entry rather than with an opaque classifier.
- **`extended_five_group` is not peer-reviewed.** Selecting it warns, and the warning
  is recorded in the audit report and the model card. The default list is.
- **`age_marker` is confounded** with career length; its estimate is an upper bound
  that partly reflects seniority expectations.
- **`writing_register` touches far more characters** than a name swap, so its magnitude
  is not comparable to the protected families. It bounds how much of the score is
  style rather than substance.
- **Effects are model- and prompt-specific.** A cue effect is a property of a
  `(model, prompt, rubric)` triple, not a durable fact. Re-audit when any changes.
- **The four-fifths rule is a reporting convention**, not a legal safe harbour, and
  nothing here is legal advice.
- **This system supports a human recruiter. It does not make hiring decisions.** That
  boundary is in the project scope and is enforced in the model card's out-of-scope
  list.

---

## 15. References

Full list with the methodological sources in `docs/METHODOLOGY.md` §11. The ones this
implementation leans on most directly:

- **Bertrand & Mullainathan (2004)**, *Are Emily and Greg More Employable than Lakisha
  and Jamal?* AER 94(4) — the name-perturbation design and the default name list.
- **Wilson & Caliskan (2024)**, *Gender, race, and intersectional bias in resume
  screening via language model retrieval.*
- **Wilson, Sim, Gueorguieva & Caliskan (2025)**, *No thoughts just AI: Biased LLM
  hiring recommendations alter human decision making* — why the rationale audit exists.
- **Seshadri, Chen, Singh & Goldfarb-Tarrant (2025)**, *Small changes, large
  consequences* — allocational rather than distributional framing.
- **Castleman, Shen, Metevier, Springer & Korolova (2026)**, *Measuring validity in
  LLM-based resume screening* — validity probes and grounding checks.
- **Cui (2026)**, *Bias-audited resume screening with calibrated matching scores,
  selective review, and evidence-grounded LLM explanation cards.*
- **Rao, Venkatesan, Cherubini & Jayagopi (2025)**, *Invisible filters: Cultural bias
  in hiring evaluations using LLMs* — cue families beyond names.
- **Tan et al. (2026)**, *Small changes, big impact: Demographic bias in LLM-based
  hiring through subtle sociocultural markers in anonymised resumes.*
- **Chouldechova (2017)**; **Kleinberg, Mullainathan & Raghavan (2017)** — the
  impossibility results behind the conflict analyser.
- **Hardt, Price & Srebro (2016)** — equality of opportunity.
- **Rivera (2012)**, *Hiring as cultural matching*, ASR 77(6) — the socioeconomic cues.
- **Neumark, Burn & Button (2019)**, JPE 127(2) — the graduation-year age design.
- **Rosenman, Olivella & Imai (2023)**, *Scientific Data* 10:299 — name/ethnicity data.
- **Monroe, Colaresi & Quinn (2008)**, *Fightin' words* — log-odds with an informative
  Dirichlet prior.
- **Mitchell et al. (2019)**, *Model cards*; **Gebru et al. (2021)**, *Datasheets.*
- **EEOC**, *Uniform Guidelines on Employee Selection Procedures*, 29 CFR 1607 (1978).

---

## License

MIT. See `pyproject.toml`.

The sample corpus is synthetic and describes no real person. Name lists are reproduced
or derived from published research with citations recorded in
`data/lexicons/names_*.csv`.
