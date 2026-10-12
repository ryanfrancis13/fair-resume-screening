"""DataFrame coercion helpers.

Exists because of one specific, version-dependent hazard.

Score tables round-trip through CSV, and a column like ``cue_families`` is empty
for every baseline row. ``pandas.read_csv`` reads those blanks as ``NaN``
(a float), so any code that treats the column as text has to decide what a
missing value means.

``Series.astype(str)`` used to make that decision implicitly: through pandas 2.x
it converted ``NaN`` into the literal string ``"nan"``, which is why guards like
``if family != "nan"`` appear in older code. **pandas 3.0 changed this** —
``astype(str)`` now preserves missing values as ``NaN`` instead of stringifying
them, so downstream ``.split(",")`` raises
``AttributeError: 'float' object has no attribute 'split'``.

Rather than sprinkle ``.fillna("")`` across three dozen call sites and hope none
is missed, the coercion lives here and is explicit about the intent: **a missing
text value is the empty string.** That is correct on every pandas version and
says what it means.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import pandas as pd


def as_text(values: Any, *, default: str = "") -> pd.Series:
    """Coerce a Series (or any iterable) to ``str``, mapping missing to ``default``.

    Explicitly fills missing values *before* the string cast, so the result never
    depends on how a particular pandas version stringifies ``NaN``.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(list(values), dtype=object)
    if series.empty:
        return series.astype(str)
    return series.astype(object).where(series.notna(), default).astype(str)


def text_column(
    frame: pd.DataFrame, column: str, *, default: str = ""
) -> pd.Series:
    """``as_text`` on ``frame[column]``, or an all-``default`` column if absent.

    Returns a Series aligned to ``frame``'s index either way, so it can be used
    directly in a boolean mask without an index mismatch.
    """
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype=str)
    return as_text(frame[column], default=default)


def nonempty_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    """Rows where ``column`` holds a non-empty, non-missing string."""
    if column not in frame.columns:
        return pd.Series([False] * len(frame), index=frame.index)
    return text_column(frame, column).str.strip() != ""


def split_terms(values: Any, separator: str = ",") -> Iterable[str]:
    """Every non-empty term across a column of separator-joined strings.

    Used for ``cue_families``, which is a comma-joined list in the flat CSV. The
    legacy ``"nan"`` sentinel is filtered too, so tables written by an older
    pandas still read correctly.
    """
    seen: list = []
    for value in as_text(values):
        for term in value.split(separator):
            term = term.strip()
            if term and term != "nan" and term not in seen:
                seen.append(term)
    return seen


def block_key(frame: pd.DataFrame, *columns: str, separator: str = "|") -> pd.Series:
    """A stable composite key, e.g. ``job_id|base_resume_id``.

    The statistical *block* identifier for the audit's paired analyses; building
    it here keeps every metric family using the same definition.
    """
    if not columns:
        raise ValueError("block_key needs at least one column")
    key: Optional[pd.Series] = None
    for column in columns:
        part = text_column(frame, column)
        key = part if key is None else key + separator + part
    assert key is not None
    return key
