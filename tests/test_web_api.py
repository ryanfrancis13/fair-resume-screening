from fastapi.testclient import TestClient

from frs import web_api


def test_extract_resume_text_reads_utf8_text():
    text = web_api.extract_resume_text(
        "resume.txt",
        b"Senior engineer with experience building reliable distributed systems.",
    )

    assert "Senior engineer" in text


def test_extract_resume_text_rejects_short_content():
    try:
        web_api.extract_resume_text("resume.txt", b"Too short")
    except ValueError as exc:
        assert "enough readable text" in str(exc)
    else:
        raise AssertionError("short resume content should be rejected")


def test_extract_resume_text_rejects_oversized_content():
    try:
        web_api.extract_resume_text(
            "resume.txt",
            b"x" * (web_api.MAX_UPLOAD_BYTES + 1),
        )
    except ValueError as exc:
        assert "5 MB or smaller" in str(exc)
    else:
        raise AssertionError("oversized resume content should be rejected")


def test_screen_endpoint_resolves_upload_types(monkeypatch):
    expected = {
        "provider": "mock",
        "model": "mock-rubric-v1",
        "fallback_used": True,
        "overall_score": 72.0,
    }
    monkeypatch.setattr(web_api, "evaluate_resume", lambda *_args: expected)
    client = TestClient(web_api.create_app())

    response = client.post(
        "/api/screen",
        files={
            "resume": (
                "resume.txt",
                b"Senior engineer with experience building reliable distributed systems.",
                "text/plain",
            )
        },
        data={
            "job_description": (
                "Build scalable backend services and data infrastructure using Python "
                "and cloud platforms."
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == expected


def test_screen_endpoint_returns_json_for_invalid_upload():
    client = TestClient(web_api.create_app())

    response = client.post(
        "/api/screen",
        files={"resume": ("resume.exe", b"not a resume", "application/octet-stream")},
        data={"job_description": "Build reliable backend services for a growing platform."},
    )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Upload a TXT")


def test_screen_endpoint_rejects_short_job_description(monkeypatch):
    def reject_job_description(_text):
        raise ValueError("Enter a job description with at least 40 characters.")

    monkeypatch.setattr(
        web_api,
        "_job_description",
        reject_job_description,
    )
    client = TestClient(web_api.create_app())

    response = client.post(
        "/api/screen",
        files={
            "resume": (
                "resume.txt",
                b"Senior engineer with experience building reliable distributed systems.",
                "text/plain",
            )
        },
        data={"job_description": "Too short"},
    )

    assert response.status_code == 400
    assert "at least 40 characters" in response.json()["detail"]


def test_offline_screen_endpoint_returns_local_result(monkeypatch):
    expected = {
        "provider": "mock",
        "model": "mock-rubric-v1",
        "fallback_used": False,
        "overall_score": 72.0,
    }
    monkeypatch.setattr(web_api, "evaluate_offline", lambda *_args: expected)
    client = TestClient(web_api.create_app())

    response = client.post(
        "/api/screen/offline",
        files={
            "resume": (
                "resume.txt",
                b"Senior engineer with experience building reliable distributed systems.",
                "text/plain",
            )
        },
        data={
            "job_description": (
                "Build scalable backend services and data infrastructure using Python "
                "and cloud platforms."
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "mock"


def test_evaluate_resume_reports_median_of_samples_and_caches(monkeypatch, tmp_path):
    from types import SimpleNamespace

    scores = {0: 78.0, 1: 84.0, 2: 71.0}
    calls = []

    class FakeScorer:
        def __init__(self, *_args):
            pass

        def score_once(self, _resume, _job, *, sample_index, **_kwargs):
            calls.append(sample_index)
            result = SimpleNamespace(
                overall_score=scores[sample_index],
                recommendation="advance",
                confidence=0.8,
                summary="",
                dimensions=[],
                missing_requirements=[],
                unverifiable_claims=[],
                protected_attribute_referenced=False,
                protected_reference_note="",
            )
            return SimpleNamespace(ok=True, result=result)

    monkeypatch.setattr(web_api, "ResumeScorer", FakeScorer)
    monkeypatch.setattr(web_api.LLMClient, "from_config", classmethod(lambda *_a, **_k: None))
    monkeypatch.setattr(web_api, "load_dotenv_if_present", lambda: None)
    web_api._result_cache.clear()
    root = web_api.Path(web_api.__file__).resolve().parents[1]
    job = "Senior engineer role requiring Python and distributed systems experience."

    first = web_api.evaluate_resume("resume text " * 10, job, root)
    again = web_api.evaluate_resume("resume text " * 10, job, root)

    assert first["overall_score"] == 78.0
    assert sorted(calls) == [0, 1, 2]  # second call served from the cache
    assert again == first


def test_provider_is_selected_by_environment(monkeypatch):
    root = web_api.Path(web_api.__file__).resolve().parents[1]
    monkeypatch.delenv("FRS_PROVIDER", raising=False)
    monkeypatch.delenv("FRS_MODEL", raising=False)
    cfg = web_api._config(root)
    assert (cfg.llm.provider, cfg.llm.model) == ("gemini", web_api.DEFAULT_MODELS["gemini"])

    monkeypatch.setenv("FRS_PROVIDER", "OpenAI")
    cfg = web_api._config(root)
    assert (cfg.llm.provider, cfg.llm.model) == ("openai", web_api.DEFAULT_MODELS["openai"])

    monkeypatch.setenv("FRS_MODEL", "llama3")
    assert web_api._config(root).llm.model == "llama3"


def test_unknown_or_mock_provider_is_rejected(monkeypatch):
    root = web_api.Path(web_api.__file__).resolve().parents[1]
    for name in ("nope", "mock"):
        monkeypatch.setenv("FRS_PROVIDER", name)
        try:
            web_api._config(root)
        except ValueError as exc:
            assert "FRS_PROVIDER" in str(exc)
        else:
            raise AssertionError(f"{name} should be rejected")
