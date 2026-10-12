"""Deterministic seeding.

Every stochastic step in the pipeline (variant sampling, bootstrap resampling,
permutation tests, LLM temperature sampling in the mock provider) draws from a
seed derived from the run's master seed plus a *stable string label*.  Deriving
per-component seeds from a label rather than sharing one global RNG means that
adding a new component, or running components in a different order or in
parallel, does not silently change the numbers produced by the others.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

_SEED_MODULUS = 2**32


def derive_seed(master_seed: int, label: str) -> int:
    """Return a stable 32-bit seed for ``label`` under ``master_seed``.

    The mapping is a pure function of its inputs, so a component's seed is
    reproducible across processes and Python versions (``hash()`` is not, which
    is why we use BLAKE2b instead).
    """
    digest = hashlib.blake2b(
        f"{master_seed}:{label}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS


def rng(master_seed: int, label: str) -> np.random.Generator:
    """A NumPy generator dedicated to ``label``."""
    return np.random.default_rng(derive_seed(master_seed, label))


def py_rng(master_seed: int, label: str) -> random.Random:
    """A stdlib ``random.Random`` dedicated to ``label``."""
    return random.Random(derive_seed(master_seed, label))


def seed_everything(master_seed: int) -> None:
    """Seed the global RNGs, for third-party code we do not control."""
    random.seed(master_seed)
    np.random.seed(master_seed % _SEED_MODULUS)
