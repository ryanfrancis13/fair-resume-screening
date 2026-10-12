"""Shared fixtures.

Every fixture uses the offline provider, so the whole suite runs with no API key,
no network and no cost.  ``FRS_PROJECT_ROOT`` is *not* overridden: the tests read
the committed sample corpus, which is part of what they are checking.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# No sys.path manipulation: `uv sync` installs the project in editable mode, so
# `frs` is importable directly. Adding `src/` to the path would be worse than
# redundant — it would make `config`, `utils`, `data` etc. importable as
# top-level modules too, so the same file could be loaded twice under two module
# identities and module-level state would silently diverge.

from frs.config import Config  # noqa: E402
from frs.data import load_corpus  # noqa: E402
from frs.llm import LLMClient, MockProvider  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet_name_list_warning():
    """Silence the provenance warning; a dedicated test asserts it is raised."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


@pytest.fixture
def cfg(tmp_path) -> Config:
    """A fast, deterministic, offline config writing into a temp directory."""
    config = Config.load(apply_env=False)
    config.llm.provider = "mock"
    config.llm.model = "mock-rubric-v1"
    config.llm.cache_enabled = False
    config.llm.requests_per_minute = 10**6
    config.llm.max_concurrency = 4
    config.llm.self_consistency_samples = 1
    config.llm.mock_noise_sd = 0.0
    config.output_dir = str(tmp_path / "runs")
    config.log_level = "ERROR"
    config.fairness.bootstrap_iterations = 200
    config.fairness.permutation_iterations = 200
    config.report.make_plots = False
    config.report.formats = ["json"]
    config.report.explanation_cards = False
    config.report.model_card = False
    config.mitigation.strategies = []
    config.validate()
    return config


@pytest.fixture
def small_cfg(cfg) -> Config:
    """As ``cfg``, but a design small enough for a fast end-to-end run."""
    cfg.data.max_resumes = 4
    cfg.data.max_jobs = 2
    cfg.perturbation.cue_families = ["name", "pronoun", "education_prestige"]
    cfg.perturbation.levels_per_family = 4
    return cfg


@pytest.fixture
def corpus(cfg):
    return load_corpus(cfg)


@pytest.fixture
def client(cfg) -> LLMClient:
    return LLMClient(MockProvider(noise_sd=0.0), requests_per_minute=10**6)


@pytest.fixture
def biased_client() -> LLMClient:
    """A provider with a known −4-point penalty for female-signalling resumes."""
    return LLMClient(
        MockProvider(bias={"gender": {"female": -4.0}}, noise_sd=0.0),
        requests_per_minute=10**6,
    )
