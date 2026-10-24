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

## Requirements

- Python 3.9 or newer
- [uv](https://docs.astral.sh/uv/)
- A Gemini API key only for Gemini-backed evaluation

The project is developed and tested on Windows, macOS, and Linux. The commands
below work in PowerShell, a terminal, or Git Bash; use the platform equivalent
of `copy` instead of `cp` if needed.

## Quick start: web evaluator

From the repository root:

```bash
uv sync --extra web
uv run frs web
```

Open <http://127.0.0.1:8000>.

The page accepts TXT, Markdown, PDF, and DOCX resumes up to 5 MB. Files are
processed in memory and are not saved by the web endpoint.

### Gemini configuration

Gemini is optional. Without an API key, the app automatically uses the local
deterministic scorer.

```bash
copy .env.example .env       # Windows PowerShell: Copy-Item .env.example .env
```

Then edit `.env` and add:

```text
GEMINI_API_KEY=your_key_here
```

Start the app again:

```bash
uv run frs web
```

The evaluator tries Gemini first. If Gemini is unavailable, rate-limited, or
returns an unusable response, it clearly labels the result as an offline
fallback. API keys remain on the server and are never sent to the browser.

To use a different model or port:

```powershell
$env:FRS_MODEL = "gemini-3.5-flash-lite"
$env:FRS_WEB_HOST = "127.0.0.1"
$env:FRS_WEB_PORT = "8080"
uv run frs web
```

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
uv sync --all-extras                 # install all optional capabilities
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
src/web_api.py              FastAPI backend for the candidate evaluator
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
