"""Filesystem helpers: atomic writes, JSON/JSONL/CSV, YAML, project paths."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import yaml


# --------------------------------------------------------------------------- #
# project layout
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    """Repository root, located from this file (``src/utils/io.py``).

    An explicit ``FRS_PROJECT_ROOT`` wins, which is what lets the test suite and
    the dashboard operate on a temporary project without changing directory.
    """
    override = os.environ.get("FRS_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # src/utils/io.py -> src/utils -> src -> project root
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# serialisation
# --------------------------------------------------------------------------- #
def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / Paths / sets into JSON-safe values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_to_jsonable(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and hasattr(obj, "dtype"):  # numpy scalar
        return obj.item()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def atomic_write_text(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` via a temp file + rename.

    Run artifacts are written this way so that an interrupted run never leaves a
    half-written CSV that a later analysis step would silently read as complete.
    """
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def write_json(path: Path, obj: Any, indent: int = 2) -> Path:
    return atomic_write_text(path, json.dumps(_to_jsonable(obj), indent=indent) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Any]) -> Path:
    payload = "".join(json.dumps(_to_jsonable(r)) + "\n" for r in rows)
    return atomic_write_text(path, payload)


def append_jsonl(path: Path, row: Any) -> None:
    """Append one record. Used for streaming logs where partial output is useful."""
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_to_jsonable(row)) + "\n")


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str] | None = None) -> Path:
    """Write dict rows to CSV, unioning keys across rows when not given."""
    rows = [_to_jsonable(r) for r in rows]
    if fieldnames is None:
        seen: List[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".csv")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _stringify_cell(row.get(k)) for k in fieldnames})
    os.replace(tmp, path)
    return path


def _stringify_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data or {}


def write_yaml(path: Path, obj: Any) -> Path:
    return atomic_write_text(
        path, yaml.safe_dump(_to_jsonable(obj), sort_keys=False, allow_unicode=True)
    )
