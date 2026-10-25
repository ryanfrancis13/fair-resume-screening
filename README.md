# Fair Resume Screening

**Generative AI Powered Resume Screening and Candidate Evaluation with Fairness
Assessment**

This project has two connected but separate workflows:

1. **Candidate evaluator** - upload one resume and compare it with one job
   description. Gemini provides an evidence-backed score when configured; the
   deterministic local scorer is used as a fallback or as a second opinion.
2. **Fairness research pipeline** - create controlled resume variants, measure
   score and decision effects, evaluate mitigation strategies, and produce
   reproducible reports for system-level research.

The candidate evaluator is decision support for human review. It does not
automatically hire, advance, reject, or contact a candidate. A single resume
cannot establish statistical fairness, so fairness auditing remains a
population-level benchmark workflow.

## Contents

- [Quick start](#quick-start-web-evaluator)
- [Gemini setup](#gemini-setup-optional)
- [Using the evaluator](#using-the-evaluator)
- [Fairness research workflow](#fairness-research-workflow)
- [Common commands](#common-commands)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)

## Requirements

- Python 3.9 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A Gemini API key (optional; only for Gemini-backed scoring)

Works on macOS, Linux, and Windows. Commands below use a POSIX shell; on
Windows PowerShell use `Copy-Item` instead of `cp`.

## Quick start: web evaluator

```bash
uv sync --extra web      # installs the web app and the Gemini SDK
uv run frs doctor        # optional: verify the environment
uv run frs web
```

Open <http://127.0.0.1:8000>. No API key is needed to try it: without one, the
app falls back to the deterministic offline scorer and labels results as such.

The page accepts TXT, Markdown, PDF, and DOCX resumes up to 5 MB. Files are
processed in memory and are not saved.

## Gemini setup (optional)

1. Create a key at <https://aistudio.google.com/apikey>.
2. Copy the example env file and add your key:

   ```bash
   cp .env.example .env
   ```

   ```text
   GEMINI_API_KEY=your_key_here
   ```

3. Restart `uv run frs web`.

The evaluator tries Gemini first. If Gemini is unavailable, rate-limited, or
returns an unusable response, the result is clearly labelled as an offline
fallback. API keys stay on the server and are never sent to the browser.

**Score stability.** Each evaluation calls Gemini 3 times and reports the median,
because hosted models vary slightly between identical calls. Results are also
remembered in server memory (never on disk), so submitting the same resume and
job description again returns the same score until the server restarts. This
means one evaluation uses 3 API requests; on the free tier (about 12 requests
per minute) that is roughly 4 evaluations per minute.

Optional settings (environment variables or `.env`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | none | Enables Gemini scoring |
| `FRS_MODEL` | `gemini-3.5-flash-lite` | Gemini model to use |
| `FRS_WEB_HOST` | `127.0.0.1` | Bind address |
| `FRS_WEB_PORT` | `8000` | Port |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `google-genai is not installed` | Run `uv sync --extra web` (or `--all-extras`). Plain `uv sync` skips optional extras. |
| `Web support is not installed` | Run `uv sync --extra web`. |
| Results say "Offline fallback used" | No `GEMINI_API_KEY` in `.env`, or Gemini is rate-limited. The reason is shown in the result. |
| Offline score is far from the Gemini score | Expected to differ: the offline scorer is a keyword-overlap baseline. List requirements as `Requirements: Python, AWS, ...` or as bullets to help it. |
| Same inputs, slightly different score after editing the job text | Any edit triggers fresh Gemini calls; only identical inputs are remembered. |
| Anything else | Run `uv run frs doctor` for a provider and environment check. |

## Using the evaluator

1. Choose a resume file.
2. Paste a job description with its responsibilities and requirements.
3. Select **Evaluate resume**.
4. Review the overall score, confidence, summary, assessment dimensions,
   supporting resume evidence, and missing requirements.
5. Optionally select **Get offline second opinion**. This makes no additional
   Gemini request and compares the deterministic local score with the primary
   result.

The offline comparison is a scoring comparison, not a fairness certification.
Differences should be reviewed by a person and are not proof that either score
is correct.

## Fairness research workflow

The research pipeline uses the controlled benchmark in
[`data/benchmark/`](data/benchmark/). It tests whether changing irrelevant
identity or proxy cues changes outcomes while preserving job-related content.
The audit includes counterfactual, group, allocational, ranking, rationale,
calibration, intersectional, validity, and reliability analyses. Mitigation
strategies can be compared in the same reproducible run.

Run the complete offline demo:

```bash
uv sync
uv run frs pipeline --config configs/offline_demo.yaml
```

Run the controlled benchmark:

```bash
uv run frs pipeline --config configs/synthetic.yaml --provider mock
```

Validate the audit instrument itself:

```bash
uv run frs validate-auditor --provider mock
```

The validation should report **5/5 checks passed**. It verifies that the
measurement system stays quiet under a no-bias null and recovers a known
injected effect with the expected direction and magnitude.

Explore completed research runs:

```bash
uv run frs dashboard
```

The Streamlit dashboard is for research artifacts and is intentionally separate
from the one-resume candidate evaluator.

Before spending API quota, inspect a planned run:

```bash
uv run frs pipeline --config configs/default.yaml --dry-run
```

## Common commands

```bash
uv sync --all-extras                 # install every optional capability
uv run frs doctor                    # check environment and provider readiness
uv run frs web                       # start the candidate evaluator
uv run frs dashboard                 # start the research report explorer
uv run frs validate-auditor --provider mock
uv run pytest                        # run the full test suite
uv run ruff check src tests scripts  # lint Python
node --check web/assets/app.js       # check frontend JavaScript syntax
```

Equivalent Make targets are available:

```bash
make setup
make web
make demo
make validate
make test
make check
```

## Repository layout

The source files live directly under `src/` but are installed as the `frs`
package through this intentional setuptools mapping:

```toml
package-dir = { "frs" = "src" }
```

This keeps the requested repository layout while avoiding ambiguous top-level
imports such as `config`, `data`, and `pipeline`.

```text
configs/                    Reproducible evaluator and research configurations
data/benchmark/             Controlled synthetic fairness benchmark
docs/                       Architecture, data, and methodology documentation
scripts/                    Corpus and benchmark generation utilities
src/web_api.py              FastAPI backend for the candidate evaluator (median-of-3 Gemini scoring)
src/screening/              Resume scoring, prompts, evidence, and rubric logic
src/perturb/                Controlled cue and counterfactual generation
src/fairness/               System-level fairness metrics and audit suite
src/mitigate/               Bias mitigation strategies and comparisons
src/pipeline/               Reproducible research pipeline orchestration
web/                        Static candidate-facing frontend
tests/                      Automated tests
runs/                       Generated research artifacts (ignored by Git)
```

## Design boundaries and limitations

- Scores are job-related decision support, not employment decisions.
- Gemini output can be wrong, incomplete, or inconsistent.
- The local scorer is a deterministic baseline, not an independent truth
  source and not the fairness auditor.
- Fairness findings depend on the benchmark, cue design, sample size, model,
  prompt, decoding settings, and statistical thresholds.
- A population-level audit cannot certify an individual candidate result.
- Real-model research runs can consume API quota. Use `--dry-run`, a small
  corpus, and the response cache before larger experiments.
- Uploaded documents are processed in memory by the web endpoint; deployment
  operators should still apply normal server, network, and credential controls.

For the research rationale, data decisions, and metric definitions, see
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md),
[`docs/DATA.md`](docs/DATA.md), and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Development

Run the checks used before a release:

```bash
uv run ruff check src tests scripts
node --check web/assets/app.js
uv run pytest
uv run frs validate-auditor --provider mock
```

Do not commit `.env`, API keys, `runs/`, caches, virtual environments, or
generated packaging artifacts. The repository `.gitignore` excludes them.

## License

MIT. See the project metadata in [`pyproject.toml`](pyproject.toml).
