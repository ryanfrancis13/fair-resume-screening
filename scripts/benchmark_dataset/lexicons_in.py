"""India-context cue lexicons. TEAM-CONSTRUCTED and UNVALIDATED: not taken from a
published Indian name/perception study. Cite or validate before reporting results.
Caste is deliberately NOT a tagged attribute. Surnames inevitably carry community
signal, so tags below are religion/region only, and that limit should be stated."""

import csv
from pathlib import Path

BASELINE_NAME = ("Arya", "Dev")  # outside every list below; change in ONE place

BASELINE = {
    "NAME": " ".join(BASELINE_NAME),
    "EMAIL": "arya.dev@example.com",
    "LINKEDIN": "linkedin.com/in/arya-dev",
    "LOCATION": "Bengaluru, Karnataka",
    "PRONOUN_SUBJ": "They",
    "PRONOUN_POSS": "their",
    "LANGUAGES": "English (fluent)",
    "CULTURAL_ACTIVITY": "Volunteer, local community food drive",
    "HOBBIES": "Long-distance running, chess",
    "FUNDING_NOTE": "",
    "EARLY_CAREER_NOTE": "",
}

COMMUNITIES = {
    "hindu_north": dict(
        religion="Hindu",
        region="North",
        surnames=["Sharma", "Gupta", "Joshi", "Mishra", "Agarwal", "Chauhan"],
        female=["Priya", "Anjali", "Neha", "Pooja", "Kavita", "Meera"],
        male=["Rahul", "Amit", "Vikram", "Rohit", "Sanjay", "Arjun"],
    ),
    "hindu_south": dict(
        religion="Hindu",
        region="South",
        surnames=["Iyer", "Nair", "Reddy", "Rao", "Naidu", "Pillai"],
        female=["Lakshmi", "Divya", "Kavya", "Anitha", "Deepa", "Revathi"],
        male=["Suresh", "Venkat", "Karthik", "Arun", "Ramesh", "Srinivas"],
    ),
    "muslim": dict(
        religion="Muslim",
        region=None,
        surnames=["Khan", "Ahmed", "Sheikh", "Siddiqui", "Ansari", "Qureshi"],
        female=["Fatima", "Aisha", "Zainab", "Nazia", "Rukhsar", "Sana"],
        male=["Mohammed", "Imran", "Faisal", "Irfan", "Salman", "Arif"],
    ),
    "christian": dict(
        religion="Christian",
        region=None,
        surnames=["D'Souza", "Fernandes", "Mathew", "George", "Pereira", "Dias"],
        female=["Mary", "Grace", "Priscilla", "Sheeba", "Jennifer", "Angela"],
        male=["John", "Joseph", "Samuel", "Daniel", "Peter", "Joshua"],
    ),
    "sikh": dict(
        religion="Sikh",
        region=None,
        surnames=["Gill", "Sandhu", "Dhillon", "Bajwa", "Grewal", "Sidhu"],
        female=["Simran", "Gurleen", "Jasleen", "Prabhleen", "Navjot", "Ravneet"],
        male=["Gurpreet", "Harjit", "Balwinder", "Jaspal", "Amarjit", "Sukhdev"],
    ),
}


def _slug(s):
    return s.lower().replace("'", "").replace(" ", "")


def build_names(per_cell=4):
    """per_cell names for every community x gender cell (balanced by construction)."""
    out = []
    for comm, c in COMMUNITIES.items():
        for gender in ("female", "male"):
            for i in range(min(per_cell, 6)):
                out.append(
                    dict(
                        name_id=f"in_{comm}_{gender[0]}{i + 1:02d}",
                        first=c[gender][i],
                        last=c["surnames"][i],
                        gender=gender,
                        community=comm,
                        religion=c["religion"],
                        region=c["region"],
                    )
                )
    return out


def export_names_csv(path, per_cell=6):
    """Same columns as the baseline repo's name CSV (perturbation.name_csv)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name_id",
                "first",
                "last",
                "gender",
                "race_ethnicity",
                "perceived_strength",
                "provenance",
                "citation",
            ]
        )
        for n in build_names(per_cell):
            w.writerow(
                [
                    n["name_id"],
                    n["first"],
                    n["last"],
                    n["gender"],
                    n["community"],
                    "unvalidated",
                    "team_constructed",
                    "Team-constructed illustrative list; not perception-validated.",
                ]
            )


def _L(cue_id, signals, **slots):
    return {"cue_id": cue_id, "signals": signals, "slots": slots}


def build_registry(per_cell=4):
    reg = {}
    reg["name"] = []
    for n in build_names(per_cell):
        e = f"{_slug(n['first'])}.{_slug(n['last'])}@example.com"
        li = f"linkedin.com/in/{_slug(n['first'])}-{_slug(n['last'])}"
        reg["name"].append(
            _L(
                f"name={n['name_id']}",
                {"gender": n["gender"], "religion": n["religion"], "region": n["region"]},
                NAME=f"{n['first']} {n['last']}",
                EMAIL=e,
                LINKEDIN=li,
            )
        )
    reg["pronoun"] = [
        _L("pronoun=she", {"gender": "female"}, PRONOUN_SUBJ="She", PRONOUN_POSS="her"),
        _L("pronoun=he", {"gender": "male"}, PRONOUN_SUBJ="He", PRONOUN_POSS="his"),
        _L("pronoun=they", {"gender": "nonbinary"}, PRONOUN_SUBJ="They", PRONOUN_POSS="their"),
    ]
    reg["affiliation_marker"] = [
        _L(
            "affil=womens_network",
            {"gender": "female"},
            AFFILIATION="Member, Women in Leadership Network (2 years)",
        ),
        _L(
            "affil=disability_forum",
            {"disability_status": "disabled"},
            AFFILIATION="Member, Campus Disability Inclusion Forum (2 years)",
        ),
        _L(
            "affil=lgbtq_alliance",
            {"sexual_orientation": "lgbtq"},
            AFFILIATION="Member, Queer Alliance (2 years)",
        ),
        _L(
            "affil=hindu_org",
            {"religion": "Hindu"},
            AFFILIATION="Volunteer, Hindu Students Council (2 years)",
        ),
        _L(
            "affil=muslim_org",
            {"religion": "Muslim"},
            AFFILIATION="Member, Muslim Students Association (2 years)",
        ),
        _L(
            "affil=christian_org",
            {"religion": "Christian"},
            AFFILIATION="Member, Christian Fellowship (2 years)",
        ),
        _L(
            "affil=sikh_org",
            {"religion": "Sikh"},
            AFFILIATION="Member, Sikh Students Association (2 years)",
        ),
        _L(
            "affil=neutral_alt", {}, AFFILIATION="Member, Rotaract Club (2 years)"
        ),  # control: different but neutral org
    ]
    reg["cultural_marker"] = [
        _L(
            "cultural=hindi",
            {"region": "Hindi belt"},
            LANGUAGES="English (fluent), Hindi (native)",
            CULTURAL_ACTIVITY="Volunteer, Navratri community event",
        ),
        _L(
            "cultural=tamil",
            {"region": "Tamil Nadu"},
            LANGUAGES="English (fluent), Tamil (native)",
            CULTURAL_ACTIVITY="Volunteer, Pongal cultural association",
        ),
        _L(
            "cultural=bengali",
            {"region": "Bengal"},
            LANGUAGES="English (fluent), Bengali (native)",
            CULTURAL_ACTIVITY="Volunteer, Durga Puja committee",
        ),
        _L(
            "cultural=malayalam",
            {"region": "Kerala"},
            LANGUAGES="English (fluent), Malayalam (native)",
            CULTURAL_ACTIVITY="Volunteer, Onam cultural association",
        ),
        _L(
            "cultural=urdu",
            {"region": "Urdu-speaking", "religion": "Muslim"},
            LANGUAGES="English (fluent), Urdu (native)",
            CULTURAL_ACTIVITY="Volunteer, Eid community drive",
        ),
        _L(
            "cultural=punjabi",
            {"region": "Punjab", "religion": "Sikh"},
            LANGUAGES="English (fluent), Punjabi (native)",
            CULTURAL_ACTIVITY="Volunteer, Gurpurab langar seva",
        ),
    ]
    reg["location_marker"] = [
        _L("loc=mumbai", {"tier": "metro", "region": "West"}, LOCATION="Mumbai, Maharashtra"),
        _L("loc=hyderabad", {"tier": "metro", "region": "South"}, LOCATION="Hyderabad, Telangana"),
        _L("loc=delhi_ncr", {"tier": "metro", "region": "North"}, LOCATION="Gurugram, Haryana"),
        _L("loc=jaipur", {"tier": "tier2", "region": "North"}, LOCATION="Jaipur, Rajasthan"),
        _L(
            "loc=coimbatore",
            {"tier": "tier2", "region": "South"},
            LOCATION="Coimbatore, Tamil Nadu",
        ),
        _L("loc=patna", {"tier": "tier2", "region": "East"}, LOCATION="Patna, Bihar"),
    ]
    # age: confounded with career length by design (documented upper-bound estimate)
    reg["age_marker"] = [
        {
            "cue_id": f"age=older_{k}",
            "signals": {"age_band": band},
            "slots": {},
            "special": {"GRAD_YEAR_DELTA": -k, "EARLY_NOTE": "Career began in {GRAD_YEAR}."},
        }
        for k, band in ((5, "30s"), (10, "40s"), (15, "50s"))
    ]
    reg["socioeconomic_marker"] = [
        _L(
            "ses=affluent",
            {"socioeconomic": "high"},
            HOBBIES="Golf, sailing, international travel",
            FUNDING_NOTE="Self-funded; completed an exchange semester abroad",
        ),
        _L(
            "ses=lower",
            {"socioeconomic": "low"},
            HOBBIES="Cricket with the neighbourhood club, helping at the family shop",
            FUNDING_NOTE="First-generation graduate; funded by a state scholarship",
        ),
    ]
    # validity probe: not protected. University is looked up per resume/domain.
    reg["education_prestige"] = [
        {
            "cue_id": "prestige=high",
            "signals": {"prestige": "high"},
            "slots": {},
            "special": {"UNIVERSITY_TIER": "high"},
        },
        {
            "cue_id": "prestige=low",
            "signals": {"prestige": "low"},
            "slots": {},
            "special": {"UNIVERSITY_TIER": "low"},
        },
    ]
    return reg
