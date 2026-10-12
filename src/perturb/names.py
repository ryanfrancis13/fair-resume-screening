"""Name lists for counterfactual perturbation, with explicit provenance.

The single largest threat to the validity of a name-swap audit is the name list
itself.  If the names are not actually perceived as signalling the demographic
group they are labelled with, the study measures nothing — it just relabels
noise.  The project's own risk register calls this out, and the mitigation is to
reuse a list from a peer-reviewed source rather than invent one.

This module therefore treats provenance as data, not as a comment:

* ``PEER_REVIEWED`` lists are reproduced from a published audit study and can be
  cited directly.
* ``LITERATURE_INFORMED`` lists follow the construction used in published work
  (voter-file-derived first/last name associations) but are **not** a verbatim
  reproduction. Selecting one emits a warning, and the warning is recorded in
  the audit report, so a reviewer sees the caveat next to the numbers rather
  than having to ask about it.
* ``USER_SUPPLIED`` lists come from a CSV the team drops in, for exactly the case
  where a specific paper's list must be reproduced.

Every :class:`NameEntry` also carries ``perceived_strength``, the qualitative
confidence that the name reads as its labelled group.  Low-confidence names
weaken a contrast, and reporting the mix is more honest than pretending all
names are equally diagnostic.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

PEER_REVIEWED = "peer_reviewed"
LITERATURE_INFORMED = "literature_informed"
USER_SUPPLIED = "user_supplied"


@dataclass(frozen=True)
class NameEntry:
    """One candidate name and the demographic levels it is taken to signal."""

    name_id: str
    first: str
    last: str
    gender: str            # female | male
    race_ethnicity: str    # White | Black | Hispanic | East Asian | South Asian
    provenance: str
    source_citation: str
    perceived_strength: str = "high"   # high | medium

    @property
    def full(self) -> str:
        return f"{self.first} {self.last}"

    @property
    def email(self) -> str:
        return f"{self.first.lower()}.{self.last.lower().replace(chr(39), '')}@example.com"

    @property
    def linkedin(self) -> str:
        return f"linkedin.com/in/{self.first.lower()}-{self.last.lower().replace(chr(39), '')}"


@dataclass
class NameList:
    """A named collection of :class:`NameEntry` with a provenance grade."""

    list_id: str
    provenance: str
    citation: str
    description: str
    entries: List[NameEntry] = field(default_factory=list)

    def filter(
        self,
        *,
        genders: Optional[Sequence[str]] = None,
        race_ethnicities: Optional[Sequence[str]] = None,
    ) -> List[NameEntry]:
        out = self.entries
        if genders:
            out = [e for e in out if e.gender in genders]
        if race_ethnicities:
            out = [e for e in out if e.race_ethnicity in race_ethnicities]
        return list(out)

    def cells(self) -> Dict[str, List[NameEntry]]:
        """Group entries by ``gender × race_ethnicity`` cell.

        Balanced cells matter: an unbalanced design confounds the marginal
        gender effect with the ethnicity composition of each gender's names.
        """
        grouped: Dict[str, List[NameEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(f"{entry.gender}|{entry.race_ethnicity}", []).append(entry)
        return grouped

    def balance_report(self) -> Dict[str, int]:
        return {cell: len(items) for cell, items in sorted(self.cells().items())}


# --------------------------------------------------------------------------- #
# Bertrand & Mullainathan (2004) — the canonical résumé-audit name list
# --------------------------------------------------------------------------- #
# Reproduced from the appendix of "Are Emily and Greg More Employable than
# Lakisha and Jamal? A Field Experiment on Labor Market Discrimination",
# American Economic Review 94(4). First names were selected from Massachusetts
# birth certificates and validated by a separate perception survey, which is
# precisely the validation step a hand-rolled list would lack.
_BM_WHITE_FEMALE = ["Allison", "Anne", "Carrie", "Emily", "Jill", "Laurie", "Kristen", "Meredith", "Sarah"]
_BM_BLACK_FEMALE = ["Aisha", "Ebony", "Keisha", "Kenya", "Lakisha", "Latoya", "Latonya", "Tamika", "Tanisha"]
_BM_WHITE_MALE = ["Brad", "Brendan", "Geoffrey", "Greg", "Brett", "Jay", "Matthew", "Neil", "Todd"]
_BM_BLACK_MALE = ["Darnell", "Hakim", "Jamal", "Jermaine", "Kareem", "Leroy", "Rasheed", "Tremayne", "Tyrone"]
_BM_WHITE_LAST = ["Baker", "Kelly", "McCarthy", "Murphy", "Murray", "O'Brien", "Ryan", "Sullivan", "Walsh"]
_BM_BLACK_LAST = ["Jackson", "Jones", "Robinson", "Washington", "Williams", "Booker", "Rivers", "Banks", "Charles"]

_BM_CITATION = (
    "Bertrand, M., & Mullainathan, S. (2004). Are Emily and Greg More Employable "
    "than Lakisha and Jamal? American Economic Review, 94(4), 991-1013."
)


def _build_bertrand_mullainathan() -> NameList:
    entries: List[NameEntry] = []
    spec = [
        ("female", "White", _BM_WHITE_FEMALE, _BM_WHITE_LAST),
        ("female", "Black", _BM_BLACK_FEMALE, _BM_BLACK_LAST),
        ("male", "White", _BM_WHITE_MALE, _BM_WHITE_LAST),
        ("male", "Black", _BM_BLACK_MALE, _BM_BLACK_LAST),
    ]
    for gender, race, firsts, lasts in spec:
        for i, first in enumerate(firsts):
            last = lasts[i % len(lasts)]
            code = f"{race[0].lower()}{gender[0]}"
            entries.append(
                NameEntry(
                    name_id=f"bm2004_{code}_{i + 1:02d}",
                    first=first,
                    last=last,
                    gender=gender,
                    race_ethnicity=race,
                    provenance=PEER_REVIEWED,
                    source_citation=_BM_CITATION,
                    perceived_strength="high",
                )
            )
    return NameList(
        list_id="bertrand_mullainathan_2004",
        provenance=PEER_REVIEWED,
        citation=_BM_CITATION,
        description=(
            "Verbatim first and last names from the Bertrand & Mullainathan (2004) "
            "field experiment. Two race levels (White, Black) x two genders, 9 names "
            "per cell. The safest default: perception-validated in the source study."
        ),
        entries=entries,
    )


# --------------------------------------------------------------------------- #
# Extended list — five ethnicity levels
# --------------------------------------------------------------------------- #
# Follows the construction used by Wilson & Caliskan (2024) and An et al. (2025),
# who draw first/last names with high posterior probability of a racial category
# from US voter-file-derived name data (Rosenman, Olivella & Imai, 2023). The
# names below are of that kind but are NOT copied from any single paper's
# appendix, so the list is graded LITERATURE_INFORMED, not PEER_REVIEWED.
_EXTENDED_SPEC = [
    ("female", "Hispanic", ["Maria", "Guadalupe", "Rosa", "Lucia", "Carmen", "Yesenia"],
     ["Gonzalez", "Ramirez", "Hernandez", "Torres", "Vasquez", "Morales"]),
    ("male", "Hispanic", ["Jose", "Carlos", "Juan", "Miguel", "Luis", "Alejandro"],
     ["Gonzalez", "Ramirez", "Hernandez", "Torres", "Vasquez", "Morales"]),
    ("female", "East Asian", ["Mei", "Ling", "Yuki", "Hana", "Thao", "Jia"],
     ["Chen", "Wang", "Tanaka", "Kim", "Nguyen", "Liu"]),
    ("male", "East Asian", ["Wei", "Jian", "Hiroshi", "Minjun", "Duc", "Haoyu"],
     ["Chen", "Wang", "Tanaka", "Kim", "Nguyen", "Liu"]),
    ("female", "South Asian", ["Priya", "Ananya", "Divya", "Meera", "Aisha", "Sneha"],
     ["Sharma", "Iyer", "Patel", "Reddy", "Banerjee", "Nair"]),
    ("male", "South Asian", ["Rahul", "Arjun", "Rohit", "Vikram", "Imran", "Aditya"],
     ["Sharma", "Iyer", "Patel", "Reddy", "Banerjee", "Nair"]),
]

_EXTENDED_CITATION = (
    "Construction follows Wilson & Caliskan (2024) and An et al. (2025), who sample "
    "names by racial posterior from Rosenman, Olivella & Imai (2023), 'Race and "
    "ethnicity data for first, middle, and last names in the United States', "
    "Scientific Data 10:299. Names here are illustrative of that method, not a "
    "verbatim reproduction of any published appendix."
)


def _build_extended() -> NameList:
    entries = list(_build_bertrand_mullainathan().entries)
    for gender, race, firsts, lasts in _EXTENDED_SPEC:
        code = "".join(w[0] for w in race.split()).lower() + gender[0]
        for i, first in enumerate(firsts):
            entries.append(
                NameEntry(
                    name_id=f"ext_{code}_{i + 1:02d}",
                    first=first,
                    last=lasts[i % len(lasts)],
                    gender=gender,
                    race_ethnicity=race,
                    provenance=LITERATURE_INFORMED,
                    source_citation=_EXTENDED_CITATION,
                    # "Aisha" appears in both the B&M Black-female list and South
                    # Asian/Muslim naming; flagging it medium is more honest than
                    # asserting a clean signal.
                    perceived_strength="medium" if first in ("Aisha", "Imran", "Jia") else "high",
                )
            )
    return NameList(
        list_id="extended_five_group",
        provenance=LITERATURE_INFORMED,
        citation=_EXTENDED_CITATION,
        description=(
            "Bertrand & Mullainathan (2004) plus Hispanic, East Asian and South Asian "
            "cells, giving five ethnicity levels x two genders for intersectional "
            "analysis. Verify against a cited appendix before publication."
        ),
        entries=entries,
    )


_BUILDERS = {
    "bertrand_mullainathan_2004": _build_bertrand_mullainathan,
    "extended_five_group": _build_extended,
}


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def available_name_lists() -> List[str]:
    return sorted(_BUILDERS)


def load_name_list(list_id: str = "bertrand_mullainathan_2004", *, csv_path: Optional[Path] = None) -> NameList:
    """Load a built-in list, or a user CSV when ``csv_path`` is given.

    A ``LITERATURE_INFORMED`` selection warns, because the difference between
    "this list was perception-validated in a published study" and "this list
    looks reasonable to us" is the difference between a citable result and an
    unfalsifiable one.
    """
    if csv_path is not None:
        return load_name_list_csv(Path(csv_path), list_id=list_id or "user_supplied")
    if list_id not in _BUILDERS:
        raise KeyError(
            f"unknown name list {list_id!r}; available: {available_name_lists()} "
            "(or pass csv_path=... for your own)"
        )
    name_list = _BUILDERS[list_id]()
    if name_list.provenance != PEER_REVIEWED:
        warnings.warn(
            f"name list {list_id!r} is graded {name_list.provenance!r}, not peer-reviewed. "
            "Findings that depend on its ethnicity labels must state this caveat. "
            f"Citation: {name_list.citation}",
            UserWarning,
            stacklevel=2,
        )
    return name_list


def load_name_list_csv(path: Path, list_id: str = "user_supplied") -> NameList:
    """Load ``first,last,gender,race_ethnicity[,perceived_strength,citation]`` CSV."""
    entries: List[NameEntry] = []
    citation = ""
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"first", "last", "gender", "race_ethnicity"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for i, row in enumerate(reader, start=1):
            citation = row.get("citation") or citation
            entries.append(
                NameEntry(
                    name_id=row.get("name_id") or f"{list_id}_{i:03d}",
                    first=row["first"].strip(),
                    last=row["last"].strip(),
                    gender=row["gender"].strip().lower(),
                    race_ethnicity=row["race_ethnicity"].strip(),
                    provenance=USER_SUPPLIED,
                    source_citation=citation or f"user-supplied list at {path}",
                    perceived_strength=(row.get("perceived_strength") or "high").strip(),
                )
            )
    if not entries:
        raise ValueError(f"{path} contained no name rows")
    return NameList(
        list_id=list_id,
        provenance=USER_SUPPLIED,
        citation=citation or f"user-supplied list at {path}",
        description=f"{len(entries)} names loaded from {path}",
        entries=entries,
    )


def export_name_list_csv(name_list: NameList, path: Path) -> Path:
    """Write a list out for inspection, so the design is reviewable as data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["name_id", "first", "last", "gender", "race_ethnicity",
             "perceived_strength", "provenance", "citation"]
        )
        for e in name_list.entries:
            writer.writerow(
                [e.name_id, e.first, e.last, e.gender, e.race_ethnicity,
                 e.perceived_strength, e.provenance, e.source_citation]
            )
    return path


def balanced_sample(
    name_list: NameList,
    per_cell: int,
    rng,
    *,
    genders: Optional[Sequence[str]] = None,
    race_ethnicities: Optional[Sequence[str]] = None,
) -> List[NameEntry]:
    """Sample ``per_cell`` names from every gender × ethnicity cell.

    Sampling *within* cells rather than from the pooled list keeps the design
    balanced, which is what allows a marginal gender effect to be interpreted
    without the ethnicity mix confounding it.
    """
    pool = name_list.filter(genders=genders, race_ethnicities=race_ethnicities)
    cells: Dict[str, List[NameEntry]] = {}
    for entry in pool:
        cells.setdefault(f"{entry.gender}|{entry.race_ethnicity}", []).append(entry)
    chosen: List[NameEntry] = []
    for cell in sorted(cells):
        items = sorted(cells[cell], key=lambda e: e.name_id)
        take = min(per_cell, len(items))
        idx = rng.choice(len(items), size=take, replace=False)
        chosen.extend(items[int(i)] for i in sorted(idx))
    return chosen


def all_name_strings() -> Iterable[str]:
    """Every full name across every built-in list (for leakage detection)."""
    seen = set()
    for builder in _BUILDERS.values():
        for entry in builder().entries:
            for token in (entry.full, entry.first, entry.last):
                if token not in seen:
                    seen.add(token)
                    yield token
