"""Packaging invariants for the ``src/`` → ``frs`` layout.

The source lives directly under ``src/`` (``src/screening/``, ``src/fairness/``,
…) to match §5 of the build brief, but is imported as ``frs`` via a
``package-dir`` mapping in ``pyproject.toml``.

That mapping has one maintenance cost: setuptools' auto-discovery cannot follow a
*renaming* ``package-dir``, so every subpackage must be listed explicitly under
``[tool.setuptools] packages``. A forgotten entry does not fail the build — it
produces a wheel that silently omits a subpackage, and the omission only shows up
when somebody installs the distribution and hits an ImportError. These tests turn
that silent failure into a loud one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# tomllib is stdlib from Python 3.11; `tomli` is the identical-API backport and is
# a dev dependency on older interpreters. The project declares a 3.9 floor, so the
# suite has to actually run there — hence a fallback rather than a skip.
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _packages_on_disk() -> set:
    """Every importable package under ``src/``, named as ``frs...``."""
    found = {"frs"}
    for init in SRC.rglob("__init__.py"):
        relative = init.parent.relative_to(SRC)
        if relative == Path("."):
            continue
        if any(part.startswith((".", "_")) or part == "__pycache__" for part in relative.parts):
            continue
        found.add("frs." + ".".join(relative.parts))
    return found


class TestLayout:
    def test_source_is_directly_under_src(self):
        """The layout the brief asks for: src/screening, not src/frs/screening."""
        assert (SRC / "__init__.py").is_file()
        assert not (SRC / "frs").exists(), "the old src/frs/ nesting should be gone"
        for expected in ("screening", "fairness", "utils", "perturb", "llm"):
            assert (SRC / expected / "__init__.py").is_file(), expected

    def test_import_name_is_frs(self):
        import frs

        assert frs.__file__ is not None
        assert Path(frs.__file__).resolve() == (SRC / "__init__.py").resolve(), (
            "frs must resolve to src/__init__.py via the package-dir mapping"
        )

    def test_project_root_is_found_from_the_new_depth(self):
        from frs.utils.io import project_root

        assert project_root().resolve() == PROJECT_ROOT.resolve()

    def test_relative_imports_span_the_whole_tree(self):
        """`from ..x import` and `from ...x import` must still resolve.

        This is what would break if `src/` were put on sys.path directly instead
        of being mapped to a package.
        """
        from frs.fairness.suite import run_audit  # uses `from ..config`
        from frs.report.plots import build_all_plots  # uses `from ..fairness`
        from frs.screening.batch import run_batch  # uses `from ..llm.base`

        assert callable(run_audit) and callable(run_batch) and callable(build_all_plots)


class TestDeclaredPackages:
    def test_every_package_on_disk_is_declared(self, pyproject):
        declared = set(pyproject["tool"]["setuptools"]["packages"])
        missing = _packages_on_disk() - declared
        assert not missing, (
            "these packages exist under src/ but are missing from "
            f"[tool.setuptools] packages in pyproject.toml, so a built wheel would "
            f"omit them: {sorted(missing)}"
        )

    def test_no_declared_package_is_missing_from_disk(self, pyproject):
        declared = set(pyproject["tool"]["setuptools"]["packages"])
        stale = declared - _packages_on_disk()
        assert not stale, f"declared but not on disk: {sorted(stale)}"

    def test_package_dir_maps_src_to_frs(self, pyproject):
        assert pyproject["tool"]["setuptools"]["package-dir"] == {"frs": "src"}

    def test_console_script_points_at_the_mapped_package(self, pyproject):
        assert pyproject["project"]["scripts"]["frs"] == "frs.cli:main"


class TestUvConfiguration:
    def test_dev_dependencies_are_a_group_not_an_extra(self, pyproject):
        """`uv sync` installs groups by default; an extra would need a flag."""
        assert "dev" in pyproject["dependency-groups"]
        assert "dev" not in pyproject["project"].get("optional-dependencies", {})

    def test_project_is_installed_as_a_package(self, pyproject):
        assert pyproject["tool"]["uv"]["package"] is True

    def test_runtime_extras_are_still_extras(self, pyproject):
        extras = pyproject["project"]["optional-dependencies"]
        assert {"gemini", "openai", "dashboard", "stats"} <= set(extras)

    def test_interpreter_is_pinned_for_the_team(self):
        pinned = (PROJECT_ROOT / ".python-version").read_text().strip()
        assert pinned, ".python-version must pin an interpreter for uv"
        major, minor = pinned.split(".")[:2]
        assert (int(major), int(minor)) >= (3, 9)

    def test_lockfile_is_present_and_committed(self):
        """uv.lock is what makes `uv sync` reproducible across machines."""
        import subprocess

        assert (PROJECT_ROOT / "uv.lock").is_file(), "run `uv lock`"
        result = subprocess.run(
            ["git", "check-ignore", "-q", "uv.lock"],
            cwd=PROJECT_ROOT, capture_output=True, check=False,
        )
        assert result.returncode != 0, "uv.lock must not be git-ignored"


class TestDocumentationIsCurrent:
    """Docs that describe the layout must not drift from it.

    A stale install command or a stale directory tree is the first thing a new
    team member hits, so these are cheap and worth asserting.
    """

    DOCS = ("README.md", "docs/ARCHITECTURE.md", "docs/DATA.md", "docs/METHODOLOGY.md")

    @pytest.mark.parametrize("name", DOCS)
    def test_no_reference_to_the_old_nesting(self, name):
        text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        assert "src/frs" not in text, f"{name} still describes the old src/frs/ layout"

    @pytest.mark.parametrize("name", DOCS)
    def test_no_pip_install_instructions(self, name):
        text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        assert "pip install -e" not in text, f"{name} still tells people to use pip"

    def test_readme_explains_the_src_mapping(self):
        text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert 'package-dir = { "frs" = "src" }' in text

    def test_makefile_drives_everything_through_uv(self):
        text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
        recipes = [
            line.strip() for line in text.splitlines()
            if line.startswith("\t") and line.strip() and not line.strip().startswith(("@", "#"))
        ]
        stale = [r for r in recipes if r.startswith(("python", "pytest", "ruff", "frs", "pip"))]
        assert not stale, f"Makefile recipes bypassing uv: {stale}"
