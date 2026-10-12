"""Run directories, manifests and artifact writing.

A run directory is the unit of reproducibility.  Everything needed to check a
number in the report is in one folder, and nothing in it depends on shell history
or on the state of the working tree:

    runs/<run_id>/
      manifest.json           resolved config, seeds, git commit, environment,
                              corpus hashes, per-stage telemetry, timings
      corpus/                 the exact resumes, jobs and pairs used
      variants/               every generated variant, with its validation verdict
      scores/                 the tidy score table, per arm, as CSV and JSONL
      audit/                  metric tables and the full audit report
      mitigation/             per-arm comparison tables
      report/                 HTML, Markdown, plots, explanation cards, model card
      logs/run.jsonl          structured log of the run

The manifest exists so a reviewer can answer "what produced this?" without asking
anyone.  It records the git commit and whether the tree was dirty, because a
result produced from uncommitted code is not reproducible and the artifact should
say so rather than imply otherwise.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import Config
from ..utils.hashing import hash_obj, hash_text
from ..utils.io import ensure_dir, write_csv, write_json, write_jsonl, write_yaml

SUBDIRS = ("corpus", "variants", "scores", "audit", "mitigation", "report", "logs")


def new_run_id(cfg: Config, prefix: Optional[str] = None) -> str:
    """Timestamped, human-sortable, config-tagged run id.

    The config hash in the id means two runs that differ in any parameter cannot
    collide, and two runs with the same tag are known to share their settings.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tag = hash_obj(cfg.to_dict(), length=6)
    name = prefix or cfg.run_name
    return f"{stamp}_{name}_{tag}"


def prepare_run_dir(cfg: Config, run_id: str) -> Path:
    root = cfg.run_dir(run_id)
    ensure_dir(root)
    for sub in SUBDIRS:
        ensure_dir(root / sub)
    return root


def git_state() -> Dict[str, Any]:
    """Current commit and dirty flag, if this is a git checkout."""
    def run(args: Sequence[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                list(args), capture_output=True, text=True, timeout=5, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run(["git", "rev-parse", "HEAD"])
    if commit is None:
        return {"available": False, "note": "not a git checkout, or git unavailable"}
    status = run(["git", "status", "--porcelain"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(status)
    return {
        "available": True,
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "note": (
            "Working tree had uncommitted changes when this run executed, so the exact "
            "code is not recoverable from the commit alone."
            if dirty else "clean working tree"
        ),
    }


def environment_state() -> Dict[str, Any]:
    versions: Dict[str, str] = {}
    for module in ("numpy", "pandas", "scipy", "matplotlib", "yaml", "statsmodels", "google.genai"):
        try:
            imported = __import__(module)
            versions[module] = getattr(imported, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - absence is the information
            versions[module] = "not installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def corpus_fingerprint(resumes: Sequence[Any], jobs: Sequence[Any], pairs: Sequence[Any]) -> Dict[str, Any]:
    """Content hashes, so a report can assert exactly which documents it used."""
    return {
        "resumes": {r.resume_id: hash_text(r.text) for r in resumes},
        "jobs": {j.job_id: hash_text(j.text) for j in jobs},
        "pairs_hash": hash_obj(
            [(p.job_id, p.resume_id, p.expected_fit) for p in pairs]
        ),
        "n_resumes": len(resumes),
        "n_jobs": len(jobs),
        "n_pairs": len(pairs),
    }


class RunManifest:
    """Accumulates provenance during a run and writes it at every stage.

    Written incrementally rather than only at the end so that an interrupted run
    still leaves a manifest describing how far it got.
    """

    def __init__(self, cfg: Config, run_id: str, root: Path) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self.root = root
        self.data: Dict[str, Any] = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "frs_version": _package_version(),
            "config": cfg.to_dict(),
            "config_hash": hash_obj(cfg.to_dict()),
            "seed": cfg.seed,
            "git": git_state(),
            "environment": environment_state(),
            "stages": {},
            "warnings": [],
        }

    def record(self, stage: str, payload: Dict[str, Any]) -> None:
        self.data["stages"][stage] = payload
        self.flush()

    def warn(self, message: str) -> None:
        self.data["warnings"].append(message)
        self.flush()

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.flush()

    def flush(self) -> Path:
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return write_json(self.root / "manifest.json", self.data)


def _package_version() -> str:
    try:
        from .. import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def write_corpus(root: Path, resumes: Sequence[Any], jobs: Sequence[Any], pairs: Sequence[Any]) -> None:
    write_jsonl(
        root / "corpus" / "resumes.jsonl",
        [
            {
                "resume_id": r.resume_id, "job_family": r.job_family, "source": r.source,
                "text": r.text, "text_hash": hash_text(r.text),
                "is_templated": r.is_templated, "slots": r.slots, "meta": r.meta,
            }
            for r in resumes
        ],
    )
    write_jsonl(
        root / "corpus" / "jobs.jsonl",
        [
            {
                "job_id": j.job_id, "title": j.title, "job_family": j.job_family,
                "must_have": j.must_have, "nice_to_have": j.nice_to_have,
                "min_years_experience": j.min_years_experience,
                "text": j.text, "text_hash": hash_text(j.text),
            }
            for j in jobs
        ],
    )
    write_csv(
        root / "corpus" / "pairs.csv",
        [
            {
                "pair_id": p.pair_id, "job_id": p.job_id, "resume_id": p.resume_id,
                "pairing_rationale": p.pairing_rationale, "expected_fit": p.expected_fit,
                "qualified_label": p.qualified_label,
            }
            for p in pairs
        ],
    )


def write_variants(root: Path, variants: Sequence[Any], skips: Sequence[Dict[str, Any]]) -> None:
    from ..perturb.generator import variant_dataframe_rows

    write_csv(root / "variants" / "variants.csv", variant_dataframe_rows(variants))
    write_jsonl(
        root / "variants" / "variants.jsonl",
        [
            {
                "variant_id": v.variant_id, "base_resume_id": v.base_resume_id,
                "is_baseline": v.is_baseline, "cue_families": v.cue_families,
                "cue_ids": v.cue_ids, "signals": v.signals,
                "edits": [vars(e) for e in v.edits],
                "provenance": v.provenance, "validation": v.validation,
                "text_hash": hash_text(v.text),
                # Variant text is kept so the audit is checkable without
                # re-running generation, and so a reviewer can read exactly what
                # the model was shown.
                "text": v.text,
            }
            for v in variants
        ],
    )
    if skips:
        write_json(root / "variants" / "skipped.json", skips)


def write_scores(root: Path, records: Sequence[Any], arm: str = "baseline") -> List[Path]:
    rows = [r.flat() if hasattr(r, "flat") else dict(r) for r in records]
    paths = [
        write_csv(root / "scores" / f"scores_{arm}.csv", rows),
        write_jsonl(root / "scores" / f"scores_{arm}.jsonl", rows),
    ]
    return paths


def write_audit(root: Path, report: Any, metrics_frame: Any, summary: Dict[str, Any]) -> None:
    write_json(root / "audit" / "audit_report.json", report)
    if metrics_frame is not None and not metrics_frame.empty:
        metrics_frame.to_csv(root / "audit" / "metrics.csv", index=False)
    write_json(root / "audit" / "summary.json", summary)
    if getattr(report, "conflicts", None):
        write_json(root / "audit" / "conflicts.json", report.conflicts)


def write_config_snapshot(root: Path, cfg: Config) -> Path:
    """The fully resolved config, so the run can be replayed verbatim."""
    return write_yaml(root / "resolved_config.yaml", cfg.to_dict())


def latest_run(cfg: Config) -> Optional[str]:
    """Most recent run id under the output directory."""
    root = cfg.resolve(cfg.output_dir)
    if not root.exists():
        return None
    candidates = [p.name for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    return sorted(candidates)[-1] if candidates else None


def resolve_run_id(cfg: Config, run_id: Optional[str]) -> str:
    """Resolve ``None`` or ``"latest"`` to a concrete run id."""
    if run_id and run_id != "latest":
        return run_id
    found = latest_run(cfg)
    if not found:
        raise FileNotFoundError(
            f"no completed runs under {cfg.resolve(cfg.output_dir)}. Run `frs pipeline` first."
        )
    return found
