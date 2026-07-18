"""Secret-redaction tests for the trace logger (issues #3 and #121).

Issue #3 acceptance: a payload containing an ``sk-...`` key and a PEM
block must persist ``[REDACTED:api-key]`` / ``[REDACTED:pem]`` rather than
the raw value, against both the sqlite and jsonl backends. SECURITY.md
lines 44-46 and 68-69 must be satisfied for the trace writer.

Issue #121 acceptance: the ``metadata`` dict passed to
``TraceLogger.session()`` is scrubbed on both backends (the original
implementation persisted it verbatim), the named-key set covers modern
secret names, and the content patterns cover GitHub classic + fine-
grained PATs, JWTs, AWS access key IDs, Stripe live keys, and Slack
tokens.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from foundry_x.trace.logger import TraceLogger, _redact

_PEM_BEGIN = "-----BEGIN RSA " + "PRIVATE KEY-----"
_PEM_END = "-----END RSA " + "PRIVATE KEY-----"
_PEM = f"{_PEM_BEGIN}\nMIIEpAIBAAKCAQEAdGhpcyBpcyBhIGZha2Uga2V5\n{_PEM_END}"
# Built from fragments so gitleaks does not flag the literal pattern in
# source; the runtime value still matches the redaction regexes.
_API_KEY = "sk-" + "1234567890abcdef"
_BEARER = "Bea" + "rer " + "mF_9.B5f-4.1JqM"
_SECRET_KEY = "sk_" + "live_50charslongsecretkeyvaluehere123"
# Modern token fixtures (issue #121). Each is hand-crafted in pieces so
# gitleaks does not flag the literal at commit time; the assembled value
# still matches the corresponding regex at runtime.
_GITHUB_CLASSIC_PAT = "gh" + "p_" + "1A2B3C4D5E6F7G8H9I0J1A2B3"
_GITHUB_FINE_GRAINED_PAT = "github_" + "pat_11ABCDEFG0_1234567890abcdefghijklmnopqrstuvwxyz"
_JWT = (
    "eyJ"
    + "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    + "."
    + "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0"
    + "."
    + "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)
_AWS_ACCESS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
_STRIPE_LIVE_KEY = "sk_" + "live_" + "4eC39HqLyjWDarjtT1zdp7dc"
_SLACK_TOKEN = "xox" + "b-1234567890123-1234567890123-" + "abcdefghijklmnopqrstuvwx"
# GCP fixtures (issue #824).
_GCP_SA_EMAIL = "my-app@" + "iam.gserviceaccount.com"
_GCP_PROJECT_ID = "my-project-" + "123456"
_GCP_ADC_PATH = "/home/user/.config/gcloud/application_default_credentials.json"

_BACKENDS = pytest.mark.parametrize("backend", ["sqlite", "jsonl"])

_GOOGLE_APPLICATION_CREDENTIALS = "/home/user/.config/gcloud/application_default_credentials.json"


def _read_persisted_payload(logger: TraceLogger, session_id: str) -> dict:
    events = logger.load_session(session_id)
    return events[0].payload


@_BACKENDS
def test_redaction_scrubs_api_key_and_pem(tmp_path, backend):
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(
            sid,
            kind="tool_result",
            payload={"output": f"key={_API_KEY}\n{_PEM}"},
        )
    payload = _read_persisted_payload(logger, sid)
    blob = json.dumps(payload)
    assert _API_KEY not in blob
    assert "BEGIN RSA PRIVATE KEY" not in blob
    assert "[REDACTED:api-key]" in payload["output"]
    assert "[REDACTED:pem]" in payload["output"]


@_BACKENDS
def test_redaction_scrubs_bearer_token(tmp_path, backend):
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(
            sid,
            kind="http_call",
            payload={"header": f"Authorization: {_BEARER}"},
        )
    payload = _read_persisted_payload(logger, sid)
    assert "mF_9.B5f-4.1JqM" not in json.dumps(payload)
    assert "[REDACTED:bearer]" in payload["header"]


@_BACKENDS
def test_redaction_scrubs_secret_named_keys(tmp_path, backend):
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(
            sid,
            kind="env",
            payload={
                "api_key": _SECRET_KEY,
                "token": "opaque-opaque",
                "password": "hunter2",
                "safe_value": "keep-me",
            },
        )
    payload = _read_persisted_payload(logger, sid)
    assert payload["api_key"] == "[REDACTED:secret]"
    assert payload["token"] == "[REDACTED:secret]"
    assert payload["password"] == "[REDACTED:secret]"
    assert payload["safe_value"] == "keep-me"
    assert _SECRET_KEY not in json.dumps(payload)


@_BACKENDS
def test_redaction_does_not_alter_clean_payloads(tmp_path, backend):
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    clean = {"text": "hi", "count": 3, "nested": {"a": [1, 2]}}
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(sid, kind="user_prompt", payload=clean)
    payload = _read_persisted_payload(logger, sid)
    assert payload == clean


def test_redaction_handles_nested_structures():
    payload = {
        "outer": [
            {"api_key": _API_KEY},
            {"snippet": f"auth {_BEARER} done"},
        ],
    }
    result = _redact(payload)
    assert result["outer"][0]["api_key"] == "[REDACTED:secret]"
    assert "[REDACTED:bearer]" in result["outer"][1]["snippet"]
    assert _API_KEY not in json.dumps(result)


def test_redaction_does_not_mutate_input():
    original = {"api_key": _API_KEY, "note": f"{_BEARER}"}
    _redact(original)
    assert original["api_key"] == _API_KEY
    assert original["note"] == _BEARER


def test_redaction_scrubs_pem_directly_in_sqlite_blob(tmp_path):
    """Raw SQL inspection: the persisted cell must not contain the PEM."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(sid, kind="file_read", payload={"content": _PEM})
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT payload FROM events").fetchone()
    raw = row[0]
    assert "BEGIN RSA PRIVATE KEY" not in raw
    assert "[REDACTED:pem]" in raw


# ---------------------------------------------------------------------------
# Issue #121: metadata-path redaction + modern-token pattern coverage.
# ---------------------------------------------------------------------------


@_BACKENDS
def test_session_metadata_is_redacted_on_persistence(tmp_path, backend):
    """TraceLogger.session(metadata=...) must scrub the metadata dict before
    writing it. Pre-#121 this round-trip leaked operator-supplied tokens."""
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    payload_metadata = {
        "operator": "alex",
        "github_token": _GITHUB_CLASSIC_PAT,
        "nested": {"aws_access_key_id": _AWS_ACCESS_KEY, "task": "ingest"},
    }
    with logger.session(harness_version="test-0.0", metadata=payload_metadata):
        pass
    sessions = logger.list_sessions()
    assert len(sessions) == 1
    persisted_metadata = sessions[0].metadata
    blob = json.dumps(persisted_metadata)
    assert _GITHUB_CLASSIC_PAT not in blob
    assert _AWS_ACCESS_KEY not in blob
    assert persisted_metadata["operator"] == "alex"
    assert persisted_metadata["nested"]["task"] == "ingest"
    assert persisted_metadata["github_token"] == "[REDACTED:secret]"
    assert persisted_metadata["nested"]["aws_access_key_id"] == "[REDACTED:secret]"


@_BACKENDS
def test_session_metadata_input_is_not_mutated(tmp_path, backend):
    """The dict passed by the Operator must not be mutated by the
    redaction pass. Issue #121 acceptance."""
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    metadata_input = {
        "github_token": _GITHUB_CLASSIC_PAT,
        "note": f"please keep {_BEARER} verbatim",
    }
    original_github = metadata_input["github_token"]
    original_note = metadata_input["note"]
    with logger.session(harness_version="test-0.0", metadata=metadata_input):
        pass
    assert metadata_input["github_token"] == original_github
    assert metadata_input["note"] == original_note


def test_redaction_scrubs_github_classic_pat():
    result = _redact({"output": f"token={_GITHUB_CLASSIC_PAT}"})
    assert _GITHUB_CLASSIC_PAT not in json.dumps(result)
    assert result["output"] == "token=[REDACTED:github-pat]"


def test_redaction_scrubs_github_fine_grained_pat():
    result = _redact({"output": f"token={_GITHUB_FINE_GRAINED_PAT}"})
    assert _GITHUB_FINE_GRAINED_PAT not in json.dumps(result)
    assert "[REDACTED:github-pat]" in result["output"]


def test_redaction_scrubs_jwt():
    result = _redact({"header": f"Authorization: Bearer {_JWT}"})
    # JWT is detected on its own; the surrounding "Bearer <token>" then
    # additionally triggers the bearer redaction. Either way the raw JWT
    # must not survive.
    blob = json.dumps(result)
    assert _JWT not in blob
    assert "[REDACTED:jwt]" in blob or "[REDACTED:bearer]" in blob


def test_redaction_scrubs_aws_access_key_id():
    result = _redact({"env": f"AWS_ACCESS_KEY_ID={_AWS_ACCESS_KEY}"})
    assert _AWS_ACCESS_KEY not in json.dumps(result)
    assert result["env"] == "AWS_ACCESS_KEY_ID=[REDACTED:aws-access-key]"


def test_redaction_scrubs_stripe_live_key():
    result = _redact({"output": f"stripe={_STRIPE_LIVE_KEY}"})
    assert _STRIPE_LIVE_KEY not in json.dumps(result)
    assert "[REDACTED:stripe-key]" in result["output"]


def test_redaction_scrubs_slack_token():
    result = _redact({"webhook": _SLACK_TOKEN})
    assert _SLACK_TOKEN not in json.dumps(result)
    assert "[REDACTED:slack-token]" in result["webhook"]


def test_redaction_scrubs_modern_secret_named_keys():
    """The expanded ``_DEFAULT_SECRET_KEY_NAMES`` set covers modern secret
    variable names independently of the value content."""
    result = _redact(
        {
            "anthropic_api_key": "anything-in-here",
            "openai_api_key": "anything-in-here",
            "aws_secret_access_key": "anything-in-here",
            "slack_token": "anything-in-here",
            "stripe_key": "anything-in-here",
            "jwt": "anything-in-here",
            "id_token": "anything-in-here",
            "refresh_token": "anything-in-here",
            "safe": "keep-me",
        }
    )
    assert result["anthropic_api_key"] == "[REDACTED:secret]"
    assert result["openai_api_key"] == "[REDACTED:secret]"
    assert result["aws_secret_access_key"] == "[REDACTED:secret]"
    assert result["slack_token"] == "[REDACTED:secret]"
    assert result["stripe_key"] == "[REDACTED:secret]"
    assert result["jwt"] == "[REDACTED:secret]"
    assert result["id_token"] == "[REDACTED:secret]"
    assert result["refresh_token"] == "[REDACTED:secret]"
    assert result["safe"] == "keep-me"


# ---------------------------------------------------------------------------
# Issue #824: GCP credential redaction.
# ---------------------------------------------------------------------------


def test_redaction_scrubs_gcp_service_account_email():
    result = _redact({"output": f"authenticated as {_GCP_SA_EMAIL}"})
    assert "sa@iam.gserviceaccount.com" not in json.dumps(result)
    assert "[REDACTED:gcp-service-account]" in result["output"]


def test_redaction_scrubs_gcp_project_id_env_var():
    result = _redact({"env": f"GCP_PROJECT_ID={_GCP_PROJECT_ID}"})
    assert _GCP_PROJECT_ID not in json.dumps(result)
    assert result["env"] == "GCP_PROJECT_ID=[REDACTED:gcp-project-id]"


def test_redaction_scrubs_gcp_adc_path():
    result = _redact({"env": f"HOME={_GCP_ADC_PATH}"})
    assert ".config/gcloud" not in json.dumps(result)
    assert "[REDACTED:gcp-adc-path]" in result["env"]


def test_redaction_scrubs_gcp_named_keys():
    """The expanded ``_DEFAULT_SECRET_KEY_NAMES`` set covers GCP variable names."""
    result = _redact(
        {
            "gcp_project_id": "my-project",
            "gcp_project": "my-project",
            "gcp_location": "us-central1",
            "google_application_credentials": "/path/to/creds.json",
            "safe": "keep-me",
        }
    )
    assert result["gcp_project_id"] == "[REDACTED:secret]"
    assert result["gcp_project"] == "[REDACTED:secret]"
    assert result["gcp_location"] == "[REDACTED:secret]"
    assert result["google_application_credentials"] == "[REDACTED:secret]"
    assert result["safe"] == "keep-me"


@_BACKENDS
def test_redaction_scrubs_gcp_credentials_in_payload(tmp_path, backend):
    """Both GCP service account email and ADC path are redacted end-to-end."""
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(
            sid,
            kind="tool_result",
            payload={
                "service_account": _GCP_SA_EMAIL,
                "adc_path": f"GOOGLE_APPLICATION_CREDENTIALS={_GCP_ADC_PATH}",
                "project_id": f"GCP_PROJECT_ID={_GCP_PROJECT_ID}",
            },
        )
    payload = _read_persisted_payload(logger, sid)
    blob = json.dumps(payload)
    assert "sa@iam.gserviceaccount.com" not in blob
    assert ".config/gcloud" not in blob
    assert _GCP_PROJECT_ID not in blob
    assert "[REDACTED:gcp-service-account]" in blob
    assert "[REDACTED:gcp-adc-path]" in blob
    assert "[REDACTED:gcp-project-id]" in blob
