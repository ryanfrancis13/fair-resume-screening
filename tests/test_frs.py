"""Test suite for the resume screening and counterfactual fairness assessment system."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from frs.cli import main
from frs.fairness import evaluate_mitigations, run_audit
from frs.models import Config, load_benchmark_corpus
from frs.perturb import detect_demographic_signals, redact_resume, validate_variant
from frs.report import run_pipeline, validate_auditor
from frs.screening import sanity_check, score_batch
from frs.web_api import app


def test_full_benchmark_corpus_loads() -> None:
    cfg = Config(max_resumes=None, max_jobs=None, max_variants_per_resume=None)
    resumes, jobs, pairs, variants = load_benchmark_corpus(cfg)
    assert len(resumes) == 150
    assert len(jobs) == 24
    assert len(pairs) == 355
    assert len(variants) == 10500


def test_perturbation_validator_and_multi_level_redactor() -> None:
    cfg = Config(max_resumes=2, max_jobs=2, max_variants_per_resume=24)
    resumes, _, _, variants = load_benchmark_corpus(cfg)
    base_map = {r.resume_id: r.text for r in resumes}

    for v in variants:
        if v.cue_family != "baseline":
            res = validate_variant(base_map[v.base_resume_id], v.text, v.substituted_values, v.cue_family)
            assert res["passed"], f"{v.variant_id} failed invariance check: {res}"

    # Tampering outside cue slot must fail span accounting & numeric fingerprint
    v0 = next(v for v in variants if v.cue_family == "name")
    tampered = v0.text.replace("29%", "99%")
    bad = validate_variant(base_map[v0.base_resume_id], tampered, v0.substituted_values, v0.cue_family)
    assert not bad["passed"] and not bad["numeric_fingerprint"]

    # Proxy leakage past identity redaction vs removal under identity_plus_proxy
    affil = next(v for v in variants if v.cue_family == "affiliation_marker" and v.signals.get("gender") == "female")
    assert detect_demographic_signals(affil.text).get("gender") == "female"
    assert detect_demographic_signals(redact_resume(affil.text, "identity")).get("gender") == "female"
    assert "gender" not in detect_demographic_signals(redact_resume(affil.text, "identity_plus_proxy"))


def test_screening_sanity_gate_and_evidence_grounding() -> None:
    cfg = Config(provider="mock", max_resumes=25, max_jobs=10, max_variants_per_resume=12)
    _, jobs, pairs, variants = load_benchmark_corpus(cfg)
    base_vars = [v for v in variants if v.cue_family == "baseline"]
    records = score_batch(jobs, pairs, base_vars, cfg)
    gate = sanity_check(records)
    assert gate["passed"]
    assert gate["kendall_tau"] >= 0.40
    assert gate["strong_weak_gap"] >= 8.0
    assert all(r.evidence_support_rate >= 0.5 for r in records)


def test_auditor_validation_oracle_and_mitigations() -> None:
    oracle = validate_auditor()
    assert oracle["passed"], f"Oracle checks failed: {oracle}"
    assert all(oracle["checks"].values())

    cfg = Config(provider="mock", max_resumes=8, max_jobs=4, max_variants_per_resume=24, mock_bias={"gender": {"female": -3.0}})
    _, jobs, pairs, variants = load_benchmark_corpus(cfg)
    recs = score_batch(jobs, pairs[:6], variants, cfg)
    rep = run_audit(recs, cfg, run_id="test_bias")
    gp = next(m for m in rep.metrics if m.name == "group_parity_gap_gender_female_vs_male")
    assert not gp.passed and gp.value == -3.0 and gp.p_value_adjusted < 0.05

    mits, leak = evaluate_mitigations(jobs, pairs[:6], variants, recs, cfg)
    assert any(m["arm"] == "blind_identity_plus_proxy" and m["mean_protected_swing"] == 0.0 for m in mits)
    id_affil = next(x for x in leak if x["redaction_level"] == "identity" and x["cue_family"] == "affiliation_marker")
    pr_affil = next(x for x in leak if x["redaction_level"] == "identity_plus_proxy" and x["cue_family"] == "affiliation_marker")
    assert id_affil["leakage_Persistent"] is True
    assert pr_affil["leakage_Persistent"] is False


def test_pipeline_and_cli_reports(tmp_path: Path) -> None:
    cfg = Config(provider="mock", output_dir=str(tmp_path), max_resumes=4, max_jobs=3, max_variants_per_resume=12)
    report, run_dir = run_pipeline(cfg)
    assert (run_dir / "report.html").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "audit_report.json").is_file()
    assert (run_dir / "explanation_cards.json").is_file()
    assert (run_dir / "MODEL_CARD.md").is_file()
    assert (run_dir / "scores.csv").is_file()
    assert report.n_scores > 0
    assert main(["doctor"]) == 0


def test_web_api_endpoints() -> None:
    client = TestClient(app)
    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "text/html" in r_index.headers["content-type"]

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    h = r_health.json()
    assert h["status"] == "ok" and len(h["rubric_dimensions"]) == 6

    sample_resume = (
        "Arya Dev\nBengaluru, Karnataka | arya@example.com\n"
        "Associate Software Engineer with 3 years of experience in software engineering.\n"
        "- Built backend services using Python, SQL, Docker, and REST APIs improving latency by 30%.\n"
    )
    sample_job = (
        "Software Engineer\nMUST-HAVE REQUIREMENTS:\n- Python\n- SQL\n- Docker\n"
        "NICE-TO-HAVE:\n- Kubernetes\n"
    )

    r_off = client.post("/api/screen/offline", data={"resume_text": sample_resume, "job_description": sample_job, "job_title": "SWE"})
    assert r_off.status_code == 200
    body = r_off.json()
    assert body["scoring_mode"] == "offline"
    assert 0 <= body["overall_score"] <= 100
    assert len(body["dimension_scores"]) == 6

    r_file = client.post(
        "/api/screen/offline",
        data={"job_description": sample_job, "job_title": "SWE"},
        files={"resume": ("candidate.txt", sample_resume.encode("utf-8"), "text/plain")},
    )
    assert r_file.status_code == 200
    fbody = r_file.json()
    assert "confidence" in fbody and len(fbody["dimensions"]) == 6

    r_bad = client.post("/api/screen/offline", data={"resume_text": "too short", "job_description": sample_job})
    assert r_bad.status_code == 400

