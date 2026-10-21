# Fair Resume Screening — common tasks, driven by uv.
#
#   make setup      create .venv from uv.lock and install everything
#   make demo       full pipeline offline, no API key needed
#   make validate   prove the audit suite itself works
#   make check      lint + tests + auditor validation (what CI would run)
#
# Every target runs through `uv run`, which syncs the environment first — so a
# target never executes against a stale or missing venv.

UV ?= uv

.DEFAULT_GOAL := help
.PHONY: help setup setup-min lock upgrade corpus synthetic-corpus demo names-only full-cues validate \
        score audit report dashboard doctor runs test test-fast test-py39 \
        lint format check build clean clean-cache clean-runs

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
setup: ## create .venv from uv.lock with all extras + dev tools
	$(UV) sync --all-extras
	@echo "ready. next: make demo"

setup-min: ## runtime dependencies only, no extras, no dev tools
	$(UV) sync --no-dev

lock: ## refresh uv.lock from pyproject.toml
	$(UV) lock

upgrade: ## re-resolve every dependency to its newest allowed version
	$(UV) lock --upgrade
	$(UV) sync --all-extras

corpus: ## regenerate the committed sample corpus
	$(UV) run python scripts/make_sample_corpus.py

synthetic-corpus: ## regenerate the controlled fairness benchmark
	$(UV) run python scripts/benchmark_dataset/build_dataset.py --out data/benchmark
	$(UV) run python scripts/benchmark_dataset/validate_dataset.py --out data/benchmark

# --------------------------------------------------------------------------- #
# running the pipeline
# --------------------------------------------------------------------------- #
demo: ## full pipeline, offline, no API key
	$(UV) run frs pipeline --config configs/offline_demo.yaml

names-only: ## the classic Bertrand & Mullainathan replication
	$(UV) run frs pipeline --config configs/names_only.yaml --provider mock

full-cues: ## every cue family + intersectional design (offline)
	$(UV) run frs pipeline --config configs/full_cues.yaml --provider mock

validate: ## does the audit suite recover a known bias and stay quiet under a null?
	$(UV) run frs validate-auditor --provider mock

score: ## phase 2 only: score unperturbed resumes and run the discrimination gate
	$(UV) run frs score --provider mock

audit: ## re-run the fairness suite over the latest run (no model calls)
	$(UV) run frs audit

report: ## re-render the report for the latest run (no model calls)
	$(UV) run frs report

dashboard: ## launch the Streamlit audit explorer
	$(UV) run frs dashboard

doctor: ## check environment, corpus, cue registry and provider readiness
	$(UV) run frs doctor

runs: ## list completed runs
	$(UV) run frs list runs

# --------------------------------------------------------------------------- #
# quality
# --------------------------------------------------------------------------- #
test: ## full test suite
	$(UV) run pytest

test-fast: ## skip the slow end-to-end oracle test
	$(UV) run pytest -m "not slow"

test-py39: ## run the suite on the declared minimum interpreter
	$(UV) run --python 3.9 --isolated --all-extras pytest -m "not slow"

lint: ## ruff check
	$(UV) run ruff check src tests scripts

format: ## ruff autofix + format
	$(UV) run ruff check --fix src tests scripts
	$(UV) run ruff format src tests scripts

check: lint test validate ## what CI would run

build: ## build the wheel and sdist
	$(UV) build

# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #
clean-cache: ## drop the LLM response cache
	rm -rf .frs_cache

clean-runs: ## drop all run artifacts (reproducible from config + seed)
	rm -rf runs/*/

clean: ## remove caches and build artifacts (keeps runs/ and .venv/)
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
