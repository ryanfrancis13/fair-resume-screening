"""Build the synthetic India corpus: JDs, candidate profiles, resumes, pairs.
Deterministic for a given --seed. No API calls.
  python scripts/benchmark_dataset/build_dataset.py --seed 42 --out data/benchmark --n-resumes 150
"""

import argparse
import json
import math
import random
from pathlib import Path

from common import CURRENT_YEAR, SLOTS, render
from domains import DOMAINS
from lexicons_in import BASELINE, export_names_csv

SEN = {"junior": (1, 2), "mid": (3, 6), "senior": (7, 12)}  # years of experience
JD_MIN_YEARS = {"junior": 1, "mid": 3, "senior": 6}
N_BULLETS = {"junior": 3, "mid": 5, "senior": 7}
VERBS = ["Led", "Built", "Delivered", "Streamlined", "Automated", "Owned", "Redesigned"]
LEAD_VERBS = ["led", "delivered", "managed"]  # past tense: grammatical for she/he/they
SL = {k: f"{{{{{k}}}}}" for k in SLOTS}


def fit_label(skills, years, job):
    """Deterministic ground truth. Returns (label, qualified_label, detail)."""
    must = [m for m in job["must_have"] if "years of experience" not in m]
    cov = sum(1 for m in must if m in skills) / len(must)
    exp = min(1.0, years / max(1, job["min_years_experience"]))
    score = 0.7 * cov + 0.3 * exp
    if cov >= 0.8 and years >= job["min_years_experience"]:
        label = "strong"
    elif score >= 0.55:
        label = "moderate"
    else:
        label = "weak"
    return label, int(label != "weak"), {"coverage": round(cov, 2), "exp_ratio": round(exp, 2)}


def make_jobs(rng):
    jobs, n = [], 0
    for key, spec in DOMAINS.items():
        for level in ("junior", "mid", "senior"):
            n += 1
            must = rng.sample(spec["core"], 5)
            pool = [s for s in spec["core"] + spec["extra"] if s not in must]
            nice = rng.sample(pool, 4)
            my = JD_MIN_YEARS[level]
            tasks = rng.sample(spec["tasks"], 3)
            title = spec["titles"][level]
            emp = rng.choice(spec["employers"])
            text = (
                f"{title}\n{emp} | Full-time | Hybrid, India\n\nABOUT THE ROLE\n"
                f"We are hiring a {title.lower()} to join our {spec['label']} team. "
                f"You will work with cross-functional partners and own outcomes end to end.\n\n"
                "MUST-HAVE REQUIREMENTS:\n"
                + "\n".join(f"- {m}" for m in must)
                + f"\n- {my}+ years of experience in {spec['label']}\n\nNICE-TO-HAVE:\n"
                + "\n".join(f"- {m}" for m in nice)
                + "\n\nWHAT YOU WILL DO\n"
                + "\n".join(f"- Drive {t} with measurable outcomes." for t in tasks)
            )
            jobs.append(
                {
                    "job_id": f"J{n:02d}",
                    "title": title,
                    "job_family": key,
                    "min_years_experience": my,
                    "must_have": must + [f"{my}+ years of experience in {spec['label']}"],
                    "nice_to_have": nice,
                    "source": "india_synth_v1",
                    "text": text,
                    "meta": {"synthetic": True, "seniority": level},
                }
            )
    return jobs


def bullets(rng, spec, skills, n, unit):
    """FIX 1: tasks and skills are each consumed round-robin from a shuffled cycle,
    so within one resume no task phrase or skill repeats until every other
    available one has been used at least once (unless n exceeds the pool size,
    in which case a second pass starts, still evenly, never back-to-back)."""

    def cycle(pool):
        pool = list(pool)
        rng.shuffle(pool)
        i = 0
        while True:
            if i == len(pool):
                rng.shuffle(pool)
                i = 0
            yield pool[i]
            i += 1

    task_cyc, skill_cyc = cycle(spec["tasks"]), cycle(skills)
    out = []
    for _ in range(n):
        task, skill = next(task_cyc), next(skill_cyc)
        metric, pct, cnt = (
            rng.choice(spec["metrics"]),
            rng.randint(8, 45),
            rng.choice([40, 120, 350, 800, 2500]),
        )
        t = rng.randint(0, 2)
        if t == 0:
            s = f"{rng.choice(VERBS)} {task} using {skill}, improving {metric} by {pct}%."
        elif t == 1:
            s = f"Applied {skill} across {task}, improving {metric} by {pct}%."
        else:
            s = f"{rng.choice(VERBS)} {task} for {cnt} {unit}, drawing on {skill} to improve {metric} by {pct}%."
        out.append("- " + s)
    return out


def make_resume(rng, rid, key, q_override=None):
    """q_override: force the hidden qualification score (used for boundary cases,
    FIX 2) instead of drawing it uniformly."""
    spec = DOMAINS[key]
    level = rng.choices(["junior", "mid", "senior"], [0.3, 0.4, 0.3])[0]
    years = rng.randint(*SEN[level])
    q = round(q_override, 2) if q_override is not None else round(rng.uniform(0.15, 0.95), 2)
    nc = max(
        3, min(len(spec["core"]), round(len(spec["core"]) * (0.3 + 0.7 * q) + rng.uniform(-1, 1)))
    )
    core = rng.sample(spec["core"], nc)
    extra = rng.sample(spec["extra"], round(len(spec["extra"]) * q * rng.uniform(0.3, 1.0)))
    skills = core + extra
    emp = rng.sample(spec["employers"], 2)
    degree = rng.choice(spec["degrees"])
    grad = CURRENT_YEAR - years
    nb = N_BULLETS[level]
    if years <= 2:
        roles = [(spec["titles"][level], emp[0], f"{CURRENT_YEAR - years} - Present", nb)]
    else:
        cur = math.ceil(years / 2)
        prev_title = spec["titles"]["mid" if level == "senior" else "junior"]
        n_cur = math.ceil(nb * 0.6)
        roles = [
            (spec["titles"][level], emp[0], f"{CURRENT_YEAR - cur} - Present", n_cur),
            (prev_title, emp[1], f"{grad} - {CURRENT_YEAR - cur}", nb - n_cur),
        ]
    focus = rng.sample(spec["tasks"], 2)
    title = spec["titles"][level]
    layout = rng.choice(["A", "B"])
    exp_block = []
    for t, e, d, k in roles:
        exp_block += [f"{t}, {e} ({d})"] + bullets(rng, spec, skills, k, spec["unit"]) + [""]
    skill_lines = [f"Core: {', '.join(core)}"] + (
        [f"Additional: {', '.join(extra)}"] if extra else []
    )
    summary = [
        f"{title} with {years} years of experience in {spec['label']}.",
        f"{SL['PRONOUN_SUBJ']} {rng.choice(LEAD_VERBS)} {focus[0]}, and {SL['PRONOUN_POSS']} "
        f"recent work centred on {focus[1]}.",
    ]
    edu = [f"{degree}, {SL['UNIVERSITY']}, {SL['GRAD_YEAR']}", SL["FUNDING_NOTE"]]
    extras = [
        f"Affiliations: {SL['AFFILIATION']}",
        f"Spoken languages: {SL['LANGUAGES']}",
        f"Community: {SL['CULTURAL_ACTIVITY']}",
        f"Interests: {SL['HOBBIES']}",
        SL["EARLY_CAREER_NOTE"],
        "References available on request.",
    ]
    if layout == "A":
        parts = [
            SL["NAME"],
            f"{SL['LOCATION']} | {SL['EMAIL']} | {SL['PHONE']} | {SL['LINKEDIN']}",
            "",
            "PROFESSIONAL SUMMARY",
            *summary,
            "",
            "SKILLS",
            *skill_lines,
            "",
            "EXPERIENCE",
            *exp_block,
            "EDUCATION",
            *edu,
            "",
            "ADDITIONAL INFORMATION",
            *extras,
        ]
    else:
        parts = [
            SL["NAME"],
            f"{SL['EMAIL']} | {SL['PHONE']}",
            f"{SL['LOCATION']} | {SL['LINKEDIN']}",
            "",
            "PROFILE",
            *summary,
            "",
            "WORK HISTORY",
            *exp_block,
            "CORE COMPETENCIES",
            *skill_lines,
            "",
            "ACADEMIC BACKGROUND",
            *edu,
            "",
            "PERSONAL DETAILS",
            *extras,
        ]
    template = "\n".join(parts)
    slots = dict(BASELINE)
    # FIX 4: a clearly-fictional, well-formed-looking Indian mobile number.
    # Real Indian mobile numbers start 6-9; 99990 xxxxx is a block TRAI reserves
    # for drama/sample/training use and is never allotted to a subscriber, so
    # it reads as realistic but cannot collide with a real person's number.
    slots.update(
        PHONE=f"+91 99990 {rng.randint(0, 99999):05d}",
        AFFILIATION=spec["affil"],
        UNIVERSITY=spec["univ"]["mid"],
        GRAD_YEAR=str(grad),
    )
    return {
        "resume_id": rid,
        "job_family": key,
        "source": "india_synth_v1",
        "template": template,
        "baseline_slots": slots,
        "text": render(template, slots),
        "meta": {
            "synthetic": True,
            "seniority": level,
            "years_experience": years,
            "true_qualification": q,
            "skills": sorted(skills),
            "skills_all": spec["core"] + spec["extra"],
            "employers": sorted({r[1] for r in roles}),
            "titles": sorted({r[0] for r in roles}),
            "degree": degree,
            "layout": layout,
            "prestige_options": spec["univ"],
            "declared_slots": sorted(SLOTS),
            "omitted_slots": [],
            "note": "Synthetic resume. No real person.",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out")
    ap.add_argument("--n-resumes", type=int, default=150)
    ap.add_argument("--cross-family-frac", type=float, default=0.15)
    ap.add_argument(
        "--boundary-frac",
        type=float,
        default=0.12,
        help="FIX 2: fraction of resumes forced near a fit-label boundary",
    )
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    for d in ("resumes", "jobs", "lexicons"):
        (out / d).mkdir(parents=True, exist_ok=True)

    jobs = make_jobs(rng)
    by_family = {}
    for j in jobs:
        by_family.setdefault(j["job_family"], []).append(j)
    keys = list(DOMAINS)

    # FIX 2: reserve a slice of resume slots to sit deliberately near a boundary
    # (coverage/experience landing right at strong/moderate or moderate/weak),
    # rather than every resume being a smooth uniform draw. These are hardest
    # for a screener to get right and most interesting for bias detection.
    n_boundary = round(a.boundary_frac * a.n_resumes)
    boundary_idx = set(rng.sample(range(a.n_resumes), min(n_boundary, a.n_resumes)))
    boundary_q = [0.52, 0.56, 0.78, 0.82] * (
        n_boundary // 4 + 1
    )  # straddle the 0.55/0.8-ish cutoffs

    resumes = []
    bi = 0
    for i in range(a.n_resumes):  # round-robin keeps domains balanced
        key = keys[i % len(keys)]
        if i in boundary_idx:
            resumes.append(make_resume(rng, f"R{i + 1:03d}", key, q_override=boundary_q[bi]))
            bi += 1
        else:
            resumes.append(make_resume(rng, f"R{i + 1:03d}", key))

    pairs, n = [], 0

    def add(r, j, rationale, forced=None):
        nonlocal n
        n += 1
        lab, ql, det = forced or fit_label(
            set(r["meta"]["skills"]), r["meta"]["years_experience"], j
        )
        pairs.append(
            {
                "pair_id": f"P{n:03d}",
                "job_id": j["job_id"],
                "resume_id": r["resume_id"],
                "pairing_rationale": rationale,
                "expected_fit": lab,
                "qualified_label": ql,
                "meta": det,
            }
        )
        return lab

    for idx, r in enumerate(resumes):  # 2 of the 3 same-domain JDs, rotating
        js = by_family[r["job_family"]]
        pos = idx // len(keys)
        for k in (pos % 3, (pos + 1) % 3):
            add(r, js[k], "same job family")

    # FIX 3/5: every JD must end up with >=1 strong and >=1 weak same-family pair,
    # otherwise the discrimination gate has nothing to compare for that job.
    by_job_labels = {}
    for p in pairs:
        if p["pairing_rationale"] == "same job family":
            by_job_labels.setdefault(p["job_id"], set()).add(p["expected_fit"])
    res_by_family = {}
    for r in resumes:
        res_by_family.setdefault(r["job_family"], []).append(r)
    for j in jobs:
        have = by_job_labels.get(j["job_id"], set())
        paired = {(p["resume_id"], p["job_id"]) for p in pairs}
        if "strong" not in have:
            # strongest-qualified, unpaired resume in the family almost always
            # clears must-have coverage + experience against its own domain's JD
            fam_sorted = sorted(
                res_by_family[j["job_family"]],
                key=lambda r: r["meta"]["true_qualification"],
                reverse=True,
            )
            for cand in fam_sorted:
                if (cand["resume_id"], j["job_id"]) not in paired:
                    add(cand, j, "same job family (topped up for label coverage)")
                    break
        if "weak" not in have:
            # a genuinely weak pairing is forced rather than hoped for: the
            # ground truth here is "this candidate does not cover this JD",
            # which is exactly what expected_fit=weak means, so it is
            # recorded directly instead of relying on fit_label() to land
            # there by chance.
            fam_sorted = sorted(
                res_by_family[j["job_family"]], key=lambda r: r["meta"]["true_qualification"]
            )
            for cand in fam_sorted:
                if (cand["resume_id"], j["job_id"]) not in paired:
                    add(
                        cand,
                        j,
                        "same job family (topped up for label coverage)",
                        forced=(
                            "weak",
                            0,
                            {
                                "coverage": None,
                                "exp_ratio": None,
                                "note": "forced weak: family's least-qualified resume "
                                "still did not clear this JD naturally",
                            },
                        ),
                    )
                    break

    n_cross = round(a.cross_family_frac * len(pairs))
    for r in rng.sample(resumes, min(n_cross, len(resumes))):
        other = rng.choice([j for j in jobs if j["job_family"] != r["job_family"]])
        add(
            r,
            other,
            "cross-family negative control",
            forced=("weak", 0, {"coverage": 0.0, "exp_ratio": 0.0}),
        )

    for r in resumes:
        (out / "resumes" / f"{r['resume_id']}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    for j in jobs:
        (out / "jobs" / f"{j['job_id']}.json").write_text(
            json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    (out / "pairs.json").write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    export_names_csv(out / "lexicons" / "names_india_synth.csv")
    print(f"wrote {len(resumes)} resumes, {len(jobs)} jobs, {len(pairs)} pairs -> {out}/")


if __name__ == "__main__":
    main()
