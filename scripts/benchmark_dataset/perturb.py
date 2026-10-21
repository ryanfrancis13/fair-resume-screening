"""Perturbation engine: one base resume -> many variants, each differing ONLY in one cue.
  python perturb.py --out out --names-per-cell 4
  python perturb.py --out out --families name,pronoun --limit-resumes 5      # quick test
Writes out/variants/variants.jsonl and out/variants/summary.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import SLOT_RE, SLOTS, render, skills_in_text
from lexicons_in import build_registry

NUM = re.compile(r"\d+(?:\.\d+)?")
NUMERIC_OK = {"GRAD_YEAR", "EARLY_CAREER_NOTE"}  # slots whose digits may legitimately change


def resolve_slots(resume, level):
    """Baseline slots + this level's edits. Handles the two context-dependent cue types."""
    base = resume["baseline_slots"]
    new = dict(base)
    new.update(level["slots"])
    sp = level.get("special", {})
    if "GRAD_YEAR_DELTA" in sp:  # age: shift graduation year
        new["GRAD_YEAR"] = str(int(base["GRAD_YEAR"]) + sp["GRAD_YEAR_DELTA"])
        new["EARLY_CAREER_NOTE"] = sp["EARLY_NOTE"].format(GRAD_YEAR=new["GRAD_YEAR"])
    if "UNIVERSITY_TIER" in sp:  # prestige: domain-specific university
        new["UNIVERSITY"] = resume["meta"]["prestige_options"][sp["UNIVERSITY_TIER"]]
    return new


def validate(resume, new_slots, variant_text):
    """The four content-preservation checks. All must pass."""
    base = resume["baseline_slots"]
    edited = {k for k in SLOTS if base[k] != new_slots[k]}
    notes, tmpl_lines = [], resume["template"].split("\n")
    b_lines, v_lines = resume["text"].split("\n"), variant_text.split("\n")
    # 1. span accounting: every changed line must contain an edited slot in the template
    ok1 = len(b_lines) == len(v_lines) == len(tmpl_lines)
    if not ok1:
        notes.append("line count changed")
    else:
        for i, (b, v) in enumerate(zip(b_lines, v_lines)):
            if b != v and not (set(SLOT_RE.findall(tmpl_lines[i])) & edited):
                ok1 = False
                notes.append(f"unexplained change on line {i + 1}")
    # 2. numeric fingerprint: digits in an edited slot may not change (except year-type slots)
    ok2 = True
    for k in edited - NUMERIC_OK:
        if sorted(NUM.findall(base[k])) != sorted(NUM.findall(new_slots[k])):
            ok2 = False
            notes.append(f"numbers changed in {k}")
    # 3. entity fingerprint: employers, titles, degree must survive
    ents = resume["meta"]["employers"] + resume["meta"]["titles"] + [resume["meta"]["degree"]]
    missing = [e for e in ents if e not in variant_text]
    ok3 = not missing
    if missing:
        notes.append(f"entities lost: {missing}")
    # 4. qualification preservation: detected skill set identical to the sampled set
    found = skills_in_text(variant_text, resume["meta"]["skills_all"])
    ok4 = found == set(resume["meta"]["skills"])
    if not ok4:
        notes.append(f"skill set changed: {sorted(found ^ set(resume['meta']['skills']))}")
    return {
        "span_accounting": ok1,
        "numeric_fingerprint": ok2,
        "entity_fingerprint": ok3,
        "qualification_preserved": ok4,
        "passed": ok1 and ok2 and ok3 and ok4,
        "notes": notes,
    }, edited


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--names-per-cell", type=int, default=4, help="1-6; 4 gives 40 names")
    ap.add_argument("--families", default="all", help="comma list or 'all'")
    ap.add_argument("--limit-resumes", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    reg = build_registry(a.names_per_cell)
    fams = list(reg) if a.families == "all" else [f.strip() for f in a.families.split(",")]
    res = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "resumes").glob("*.json"))
    ]
    if a.limit_resumes:
        res = res[: a.limit_resumes]
    (out / "variants").mkdir(parents=True, exist_ok=True)

    stats = defaultdict(Counter)
    failures = []
    with open(out / "variants" / "variants.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            for fam in fams:
                for lv in reg[fam]:
                    new = resolve_slots(r, lv)
                    if new == r["baseline_slots"]:
                        stats[fam]["same_as_baseline_skipped"] += 1
                        continue
                    vtext = render(r["template"], new)
                    ver, edited = validate(r, new, vtext)
                    stats[fam]["generated"] += 1
                    stats[fam]["passed"] += ver["passed"]
                    if not ver["passed"]:
                        failures.append((r["resume_id"], lv["cue_id"], ver["notes"]))
                    f.write(
                        json.dumps(
                            {
                                "variant_id": f"{r['resume_id']}__{lv['cue_id']}",
                                "resume_id": r["resume_id"],
                                "family": fam,
                                "cue_id": lv["cue_id"],
                                "signals": lv["signals"],
                                "edits": {
                                    k: {"from": r["baseline_slots"][k], "to": new[k]}
                                    for k in sorted(edited)
                                },
                                "text": vtext,
                                "validation": ver,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    tot = sum(s["generated"] for s in stats.values())
    ok = sum(s["passed"] for s in stats.values())
    summary = {
        "n_resumes": len(res),
        "n_variants": tot,
        "n_passed": ok,
        "per_family": {k: dict(v) for k, v in stats.items()},
        "failures_kept_and_flagged": [
            {"resume": a_, "cue": b_, "notes": c_} for a_, b_, c_ in failures[:200]
        ],
    }
    (out / "variants" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(res)} resumes -> {tot} variants, {ok} passed all 4 checks")
    for k, v in stats.items():
        print(
            f"  {k:22s} generated={v['generated']:6d} passed={v['passed']:6d} skipped_same={v['same_as_baseline_skipped']}"
        )
    for x in failures[:10]:
        print("  FAIL", x)


if __name__ == "__main__":
    main()
