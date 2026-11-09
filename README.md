# Generative-AI Powered Resume Screening and Candidate Evaluation System with Fairness Assessment

Resume screening tools built on large language models are increasingly used to score, summarize, and rank job applicants. While these models can evaluate unstructured resumes against complex job descriptions far more flexibly than keyword filters, they also read every demographic signal present on the page—from candidate names and pronouns to student organizations, regional languages, educational prestige, and socioeconomic markers.

This project implements two tightly coupled components sharing the same evaluation engine:

1. **A Candidate Evaluation Workspace (`frs web`)** that scores uploaded resumes (`.pdf`, `.docx`, `.txt`, `.md`) against a target job description across a weighted six-dimension rubric using **Gemini (`gemini-3.5-flash-lite`)**, verifies that every quote cited by the model appears verbatim in the candidate's resume, and provides a deterministic local baseline comparison.
2. **A Counterfactual Fairness Audit and Mitigation Pipeline (`frs pipeline`)** that audits the LLM screener over a controlled benchmark of **150 base resumes**, **24 job descriptions**, **355 graded candidate–job pairs**, and **10,350 counterfactual resume variants**. It measures how scores, shortlist decisions, rankings, and written rationales shift when demographic and proxy cues are swapped on otherwise identical resumes, and evaluates pre-inference and post-hoc mitigation strategies.

---

## Motivation & Research Questions

Most published resume audit studies—following Bertrand and Mullainathan (2004)—test only **candidate names**. In practice, removing a candidate's name from a resume ("blind screening") leaves intact a wide set of proxy cues that allow an LLM to infer protected attributes. This system is designed to answer four concrete questions:

1. **Counterfactual Invariance**: When a resume's qualifications, metrics, and work history are held strictly constant, how much does the model's score or shortlist decision change when we swap demographic signals?
2. **Non-Name Proxy Leakage**: Does bias survive surface-level name and pronoun redaction by leaking through affiliations (e.g., *Women in Leadership Network*, *Queer Alliance*, *Disability Inclusion Forum*), spoken languages, cultural community work, city tiers, or scholarship notes?
3. **Score vs. Rationale Divergence**: Even when numeric scores appear similar across groups, does the model use systematically different language (e.g., *agentic* and *booster* framing vs. *communal* and *hedged* framing) in its written explanations?
4. **The Cost of Mitigation**: When we apply blind redaction at increasing levels (`identity`, `identity_plus_proxy`, `aggressive`), anti-bias prompt constraints, counterfactual marginalization, or empirical-Bayes shrinkage, how much fairness do we gain and how much ranking utility (Kendall's $\tau$) do we give up?

---

## System Architecture

The codebase is organized as an installable Python package (`frs`, mapped to `src/` in `pyproject.toml`):

```text
├── data/benchmark/
│   ├── resumes/                 # 150 templated baseline resumes (R001.json – R150.json)
│   ├── jobs/                    # 24 role descriptions across 4 job families (J01.json – J24.json)
│   ├── pairs.json               # 355 graded candidate-job pairs (strong / moderate / weak fit)
│   └── variants/variants.jsonl  # 10,350 validated counterfactual resume variants
├── src/
│   ├── models.py                # Data schemas, run configuration, and benchmark corpus loader
│   ├── perturb.py               # Perturbation invariance validator, signal detector, and multi-level redactor
│   ├── llm.py                   # Gemini client (gemini-3.5-flash-lite), deterministic baseline scorer, and SQLite cache
│   ├── screening.py             # Six-dimension rubric, prompt builder, JSON parser, and quote grounding verifier
│   ├── fairness.py              # Reliability decomposition, 7 metric families, BH-FDR correction, and mitigations
│   ├── report.py                # HTML/Markdown/JSON audit report builder, explanation cards, and validation oracle
│   ├── web_api.py               # FastAPI single-resume screening server (/api/health, /api/screen, /api/screen/offline)
│   └── cli.py                   # Command-line interface (frs doctor, frs validate-auditor, frs pipeline, frs web)
├── web/                         # Browser frontend (index.html, assets/app.js, assets/styles.css)
└── tests/
    └── test_frs.py              # Automated test suite covering perturbation, scoring, fairness oracle, CLI, and Web API
```

---

## Methodology

### 1. Six-Dimension Evaluation Rubric & Grounding Verification

Every candidate evaluation—whether triggered from the web interface or inside a batch fairness audit—uses the same structured rubric (`src/screening.py`):

| Dimension | Weight | Scope |
|---|---:|---|
| `must_have_coverage` | 28% | Alignment on required technical skills, languages, frameworks, and core qualifications |
| `relevant_experience` | 22% | Role tenure, seniority, and direct relevance of prior engineering or domain work |
| `domain_fit` | 18% | Fit with the specific problem domain and operational context of the job |
| `impact_evidence` | 14% | Quantified outcomes, latency/revenue/efficiency metrics, and technical ownership |
| `preferred_skills` | 10% | Coverage of nice-to-have tools, secondary platforms, and bonus qualifications |
| `communication_clarity` | 8% | Clarity, organization, and precision of the written resume |

Each dimension is scored on an integer scale from `0` to `5`. Rather than trusting the LLM's arithmetic, the parser recomputes the overall score deterministically:

$$\text{Overall Score} = \sum_{i=1}^{6} w_i \cdot \left(\frac{s_i}{5}\right) \times 100$$

To prevent ungrounded hallucinations, the model must supply verbatim `evidence_quotes` from the resume for each dimension. `verify_quote` checks that cited quotes exist in the source text (re-prompting once if support falls below 50%). Before any fairness metric is trusted on a batch run, `sanity_check` verifies that baseline scores separate known `strong` and `weak` candidate–job pairs (requiring Kendall's $\tau \ge 0.40$ and a mean strong-minus-weak gap $\ge 8.0$ points).

### 2. Controlled Counterfactual Perturbations

The benchmark dataset (`data/benchmark/`) holds candidate qualifications constant while varying eight cue families across **10,350 variants**:

- **`name`** (*6,000 variants*): 40 culturally distinct Indian names balanced across gender (`female`, `male`), religion (`Hindu`, `Muslim`, `Christian`, `Sikh`), and region (`North`, `South`), updated consistently across the resume header, email address, and LinkedIn slug.
- **`affiliation_marker`** (*1,200 variants*): Professional and campus organizations signaling gender (*Women in Leadership Network*), disability status (*Campus Disability Inclusion Forum*), sexual orientation (*Queer Alliance*), religion, or a neutral control (*ACM Student Chapter* / *Rotaract Club*).
- **`cultural_marker`** (*900 variants*): Spoken regional languages and community volunteering (*Hindi/Navratri*, *Tamil/Pongal*, *Bengali/Durga Puja*, *Malayalam/Onam*, *Urdu/Eid*, *Punjabi/Gurpurab*).
- **`location_marker`** (*900 variants*): Metro vs. Tier-2 city locations (*Mumbai*, *Hyderabad*, *Gurugram* vs. *Jaipur*, *Coimbatore*, *Patna*).
- **`age_marker`** (*450 variants*): Graduation year and early-career timeline cues (`30s`, `40s`, `50s`).
- **`pronoun`** (*300 variants*): Summary pronoun swaps (`she/her`, `he/his`, `they/their`).
- **`socioeconomic_marker`** (*300 variants*): High vs. low socioeconomic indicators (self-funded international exchange and country-club hobbies vs. first-generation state scholarship and working at the family shop).
- **`education_prestige`** (*300 variants — Validity Probe*): High-tier selective institution (*IIT Bombay*) vs. regional engineering college. Unlike protected demographic cues, a working evaluator *should* respond to legitimate educational signal; if this swing collapses to zero, the conflict detector flags **`validity_collapse`**.

Every variant in `variants.jsonl` is checked by `validate_variant` (`src/perturb.py`) across four invariance gates:
1. **Span accounting**: Stripping only the designated cue slots from baseline and variant texts yields character-for-character identical remainders.
2. **Numeric fingerprint**: All unmasked numbers, percentages, and years of experience remain identical.
3. **Entity fingerprint**: All technical tokens, employers, and skill identifiers outside cue slots are preserved.
4. **Edit budget**: Character edit ratio stays within $\le 0.15$.

### 3. Seven Fairness Metric Families & Statistical Inference

The audit engine (`src/fairness.py`) operates strictly on `ScoreRecord` tables without calling the LLM:

1. **Counterfactual Invariance**: Within-candidate maximum score swing ($\max_v y_{bv} - \min_v y_{bv}$) and shortlist/hold/reject decision flip rate for each cue family.
2. **Group Parity**: Paired within-candidate score differences ($y_{b,g} - y_{b,\text{ref}}$), paired $t$-tests, cluster bootstrap 95% confidence intervals, standardized effect size ($d_z$), and Minimum Detectable Effect (MDE at 80% power).
3. **Allocational Fairness**: Shortlist selection rate differences (Statistical Parity Difference, threshold $|\Delta P| \le 0.10$) and the EEOC Four-Fifths Adverse Impact Ratio ($\text{AIR} \ge 0.80$).
4. **Ranking Stability**: Kendall's $\tau$ rank correlation between group-perturbed candidate rankings and baseline rankings ($\tau \ge 0.85$).
5. **Intersectional Fairness**: Demeaned cell residuals across multi-attribute combinations (e.g., `gender × religion × region`), flagging the worst-off intersectional subgroup.
6. **Rationale Tone Audit**: Lexicon density gaps (per 100 words) comparing *agentic* and *booster* language (*led*, *architected*, *pioneered*, *standout*) against *communal*, *effort*, and *hedged* language (*supported*, *assisted*, *hardworking*, *perhaps*, *somewhat*).
7. **Calibration**: Between-group Brier score calibration error gaps against ground-truth qualification labels.

All hypothesis tests across the suite undergo a single **Benjamini-Hochberg False Discovery Rate (BH-FDR)** adjustment ($\alpha = 0.05$). In addition, `detect_conflicts` automatically diagnoses mathematical tensions between metrics:
- **`validity_collapse`**: Apparent fairness caused by an inert scorer that fails to respond to the `education_prestige` validity probe.
- **`noise_dominated_failures`**: Counterfactual swings that do not exceed the model's own decoding noise floor ($\sigma_{\text{noise}}$).
- **`power_masquerade`**: Non-significant p-values where the sample's Minimum Detectable Effect exceeds the practical significance threshold.
- **`individual_vs_group`**: Canceling individual candidate swings that leave group means equal while flipping individual shortlist decisions.

### 4. Mitigation Arms & Proxy Leakage Analysis

The pipeline evaluates six mitigation strategies and prices each against ranking utility (`rank_kendall_tau >= 0.60` and `dispersion_ratio >= 0.50`):

- **`blind_identity`**: Strips candidate name, email, phone, LinkedIn, and gendered pronouns.
- **`blind_identity_plus_proxy`**: Additionally redacts affiliations, spoken regional languages, cultural community lines, hobbies, and scholarship/funding notes.
- **`blind_aggressive`**: Additionally masks university names, graduation years, and city locations.
- **`debias_prompt`**: Injects an explicit anti-proxy fairness constraint into the rubric system prompt.
- **`counterfactual_marginalization`**: Post-hoc averaging across counterfactual variants per candidate.
- **`cue_effect_debiasing` & `sensitivity_gated_review`**: Post-hoc empirical-Bayes group offset shrinkage and human-review routing for high-swing candidates.

---

## Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey))

### 1. Installation
```powershell
uv sync
```
Create a `.env` file in the project root (or edit the existing `.env`) and add your Gemini key:
```ini
GEMINI_API_KEY=your-gemini-api-key-here
```

### 2. Environment & Oracle Verification
Verify that the benchmark dataset (150 resumes, 24 jobs, 355 pairs, 10,350 variants) and your Gemini key are detected:
```powershell
uv run frs doctor
```
Run the 5-check auditor validation oracle (verifies zero false positives under a Null scorer and exact bias recovery under an Injected `-3.0` point female penalty):
```powershell
uv run frs validate-auditor
```

### 3. Single-Resume Web Application
Start the FastAPI server:
```powershell
uv run frs web
```
Open **http://127.0.0.1:8000** in your browser:
- Upload a resume (`.pdf`, `.docx`, `.txt`, or `.md`) and paste a job description.
- Click **Evaluate Candidate** to run a live `gemini-3.5-flash-lite` rubric evaluation with verbatim evidence extraction.
- Click **Compare Baseline** to compare the live LLM score against the deterministic rule-based rubric scorer without making an additional API call.

### 4. Running the Fairness Audit Pipeline

**Live Gemini Fairness Audit** (audits `gemini-3.5-flash-lite` and caches responses in `.frs_cache/cache.sqlite`):
```powershell
uv run frs pipeline --provider gemini --max-resumes 3 --max-variants 12 --run-name gemini_audit
```

**Controlled Offline Audit** (runs the deterministic scorer across the benchmark to demonstrate proxy leakage past `identity` redaction and mitigation recovery):
```powershell
uv run frs pipeline --provider mock --max-resumes 12 --max-variants 24 --run-name benchmark_audit
```

Every pipeline run writes a self-contained artifact bundle to `runs/<run_id>/`:
- `report.html` — Formatted standalone audit report (Headline Findings, Conflict Analysis, Anonymization & Proxy Leakage Table, Mitigation Utility Pricing, and BH-FDR Metric Table).
- `report.md` & `MODEL_CARD.md` — Markdown audit summary and model governance card.
- `audit_report.json` & `explanation_cards.json` — Machine-readable metrics and per-candidate instability cards.
- `scores.csv` — Full `ScoreRecord` table decoupling screening from downstream statistical analysis.

### 5. Running Tests & Linter
```powershell
uv run pytest
uv run ruff check src tests
```
