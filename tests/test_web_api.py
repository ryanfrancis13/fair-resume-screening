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
