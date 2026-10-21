"""Sanity-check the corpus before anything is scored."""

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from build_dataset import fit_label
from common import SLOT_RE, SLOTS, render, skills_in_text


def load(out):
    out = Path(out)
    res = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((out / "resumes").glob("*.json"))
    }
    jobs = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((out / "jobs").glob("*.json"))
    }
    pairs = json.loads((out / "pairs.json").read_text(encoding="utf-8"))
    return res, jobs, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/benchmark")
    a = ap.parse_args()
    res, jobs, pairs = load(a.out)
    errs = []
    for rid, r in res.items():
        used = set(SLOT_RE.findall(r["template"]))
        if used != set(SLOTS):
            errs.append(f"{rid}: template slots != 15 declared ({used ^ set(SLOTS)})")
        if render(r["template"], r["baseline_slots"]) != r["text"]:
            errs.append(f"{rid}: text != render(template, baseline_slots)")
        if "{{" in r["text"]:
            errs.append(f"{rid}: unfilled slot in text")
        found = skills_in_text(r["text"], r["meta"]["skills_all"])
        if found != set(r["meta"]["skills"]):
            errs.append(
                f"{rid}: skills in text != sampled skills ({found ^ set(r['meta']['skills'])})"
            )
        miss = [
            e
            for e in r["meta"]["employers"] + r["meta"]["titles"] + [r["meta"]["degree"]]
            if e not in r["text"]
        ]
        if miss:
            errs.append(f"{rid}: entities missing from baseline text {miss}")
        wc = len(r["text"].split())
        if not 60 <= wc <= 2500:
            errs.append(f"{rid}: word count {wc} outside 60-2500")
    seen = set()
    for p in pairs:
        if p["resume_id"] not in res or p["job_id"] not in jobs:
            errs.append(f"{p['pair_id']}: dangling id")
        if p["pairing_rationale"] == "same job family":  # recompute label independently
            r, j = res[p["resume_id"]], jobs[p["job_id"]]
            lab, _, _ = fit_label(set(r["meta"]["skills"]), r["meta"]["years_experience"], j)
            if lab != p["expected_fit"]:
                errs.append(f"{p['pair_id']}: label mismatch")
        seen.add((p["resume_id"], p["job_id"]))
    if len(seen) != len(pairs):
        errs.append("duplicate (resume, job) pairs")
    if {p["resume_id"] for p in pairs} != set(res):
        errs.append("some resumes never paired")
    if {p["job_id"] for p in pairs} != set(jobs):
        errs.append("some jobs never paired")
    # FIX 5: every job needs >=1 strong and >=1 weak same-family pair for the
    # discrimination gate to have something to compare.
    by_job = {}
    for p in pairs:
        if p["pairing_rationale"].startswith("same job family"):
            by_job.setdefault(p["job_id"], set()).add(p["expected_fit"])
    for jid in jobs:
        have = by_job.get(jid, set())
        if "strong" not in have or "weak" not in have:
            errs.append(
                f"{jid}: missing {'strong' if 'strong' not in have else 'weak'} same-family pair ({have})"
            )
    # FIX 1: no skill or task phrase should dominate a single resume's bullets
    for rid, r in res.items():
        bullet_lines = [line for line in r["text"].split("\n") if line.startswith("- ")]
        if not bullet_lines:
            continue
        hits = Counter()
        for sk in r["meta"]["skills"]:
            import re as _re

            hits[sk] = sum(
                1
                for line in bullet_lines
                if _re.search(r"(?<![\w])" + _re.escape(sk) + r"(?![\w])", line, _re.I)
            )
        over = {k: v for k, v in hits.items() if v > max(2, len(bullet_lines) // 2)}
        if over:
            errs.append(f"{rid}: skill repeated too often in bullets {over}")
    # near-duplicate resumes (Jaccard on word sets >= 0.9)
    toks = {k: set(re.findall(r"\w+", r["text"].lower())) for k, r in res.items()}
    ks, near = list(toks), 0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            u = len(toks[ks[i]] | toks[ks[j]])
            near += len(toks[ks[i]] & toks[ks[j]]) / u >= 0.9
    print(f"resumes={len(res)} jobs={len(jobs)} pairs={len(pairs)}")
    print("by domain      :", dict(Counter(r["job_family"] for r in res.values())))
    print("by seniority   :", dict(Counter(r["meta"]["seniority"] for r in res.values())))
    print("fit labels     :", dict(Counter(p["expected_fit"] for p in pairs)))
    print(
        "same-family fit:",
        dict(
            Counter(p["expected_fit"] for p in pairs if p["pairing_rationale"] == "same job family")
        ),
    )
    q = {}
    for p in pairs:
        if p["pairing_rationale"] == "same job family":
            q.setdefault(p["expected_fit"], []).append(
                res[p["resume_id"]]["meta"]["true_qualification"]
            )
    print(
        "mean hidden q by fit label (should rise weak < moderate < strong):",
        {k: round(statistics.mean(v), 2) for k, v in q.items()},
    )
    print("near-duplicate resume pairs (Jaccard>=0.9):", near)
    print("RESULT:", "PASS" if not errs else f"FAIL ({len(errs)} problems)")
    for e in errs[:25]:
        print("  -", e)


if __name__ == "__main__":
    main()
