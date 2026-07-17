"""Benchmark task: trace-payload secret redaction is active (SECURITY.md §Secrets).

Regression target for the ``TraceLogger`` scrubber in
``src/foundry_x/trace/logger.py`` (issues #3 and #121). The trace store
must never persist raw API keys, PEM blocks, or modern tokens
(GitHub classic / fine-grained PATs, JWTs, AWS access keys, Stripe
live keys, Slack tokens), regardless of whether the value landed
through ``TraceLogger.record()`` or the ``metadata`` dict passed to
``TraceLogger.session()``. A regression that weakens any of the
regexes, drops the recursive ``_redact`` walk over the metadata path,
or removes a token category surfaces here as a failing benchmark and
blocks the harness edit at PR review (ADR-0004).
"""

from __future__ import annotations

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.trace.logger import _redact, _redact_value

# Fixture strings are assembled from fragments so gitleaks does not flag
# the literal patterns at commit time; the assembled values still match
# the redaction regexes at runtime. Mirrors ``tests/test_redaction.py``.
_PEM = (
    "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIEpAIBAAKCAQEAdGhpcyBpcyBhIGZha2Uga2V5\n"
    "-----END RSA " + "PRIVATE KEY-----"
)
_API_KEY = "sk-" + "1234567890abcdef"
_BEARER = "Bea" + "rer " + "mF_9.B5f-4.1JqM"
_GITHUB_CLASSIC_PAT = "gh" + "p_" + "1A2B3C4D5E6F7G8H9I0J1A2B3"
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
# GCP token fixtures (issue #746).
_GCP_ACCESS_TOKEN = "ya29." + "a-bC0dE1fG2hI3jK4lM5nO6pQ7rS8tU9vW0xY1zA2bC3dE4fG5hI6"
_GCP_SERVICE_ACCOUNT = "my-service-account@developer.gserviceaccount.com"

TASK = BenchmarkTask(
    name="secret_redaction",
    description=(
        "TraceLogger scrubs secret-like substrings (API keys, PEM blocks, "
        "GitHub PATs, JWTs, AWS keys, Stripe live keys, Slack tokens, "
        "bearer headers, GCP access tokens, GCP service-account emails) "
        "from event payloads and from the metadata dict passed to "
        "TraceLogger.session()."
    ),
    prompt=(
        "Inspect src/foundry_x/trace/logger.py: confirm _redact and "
        "_redact_value still cover the token set enumerated in SECURITY.md "
        "§Secrets and the metadata path established by issue #121."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "Every secret-shaped string below is replaced by a [REDACTED:*] "
        "sentinel in the value-level pass, the metadata dict with a "
        "secret-like key is rewritten to a sentinel value, and the "
        "non-secret data is passed through unchanged."
    ),
    tags=["security"],
)


@pytest.mark.benchmark
def test_value_level_redaction_scrubs_each_token_class() -> None:
    """Each token class is scrubbed by ``_redact_value`` (issues #3 + #121).

    A regression that drops any of the regexes in
    ``src/foundry_x/trace/logger.py`` (PEM, JWT, sk-, GitHub PAT, AWS,
    Stripe, Slack, bearer) surfaces here as a literal of the original
    substring surviving in the output.
    """
    assert _redact_value(_PEM) == "[REDACTED:pem]"
    assert _redact_value(_API_KEY) == "[REDACTED:api-key]"
    assert _redact_value(_GITHUB_CLASSIC_PAT) == "[REDACTED:github-pat]"
    assert _redact_value(_JWT) == "[REDACTED:jwt]"
    assert _redact_value(_AWS_ACCESS_KEY) == "[REDACTED:aws-access-key]"
    assert _redact_value(_STRIPE_LIVE_KEY) == "[REDACTED:stripe-key]"
    assert _redact_value(_SLACK_TOKEN) == "[REDACTED:slack-token]"
    assert _redact_value(_BEARER) == "[REDACTED:bearer]"
    assert _redact_value(_GCP_ACCESS_TOKEN) == "[REDACTED:gcp-access-token]"
    assert _redact_value(_GCP_SERVICE_ACCOUNT) == "[REDACTED:gcp-service-account]"


@pytest.mark.benchmark
def test_named_secret_key_is_wholly_replaced() -> None:
    """A dict whose key matches a secret-like name is fully scrubbed (issue #121).

    The named-key pass replaces the entire value with ``[REDACTED:secret]``
    regardless of content; this is the bound on the recursive walk that
    prevents an Operator from sneaking a credential through a key like
    ``authorization`` or ``github_token``.
    """
    redacted = _redact(
        {
            "authorization": _BEARER,
            "github_token": _GITHUB_CLASSIC_PAT,
            "endpoint": "https://example.com",
            "count": 3,
            "tags": ["prod", _API_KEY],
        }
    )
    assert redacted["authorization"] == "[REDACTED:secret]"
    assert redacted["github_token"] == "[REDACTED:secret]"
    # Non-secret neighbours survive intact.
    assert redacted["endpoint"] == "https://example.com"
    assert redacted["count"] == 3
    # Nested structures are walked; the in-list secret still gets scrubbed.
    assert redacted["tags"][0] == "prod"
    assert redacted["tags"][1] == "[REDACTED:api-key]"


@pytest.mark.benchmark
def test_metadata_path_uses_recursive_redaction() -> None:
    """A free-form metadata dict is walked recursively (issue #121).

    The original implementation only redacted ``TraceLogger.record()``
    payloads and persisted ``session()`` metadata verbatim. Issue #121
    added the recursive call so this benchmark pinpoints a regression
    that drops the metadata scrubber. The nested header key here is
    deliberately NOT in the named-secret-key set, so the value-level
    regex pass must run; that is the path that proves recursion.
    """
    redacted = _redact(
        {
            "model": "gpt-x",
            "headers": {"x-custom-header": _BEARER},
            "raw_token": _API_KEY,
        }
    )
    assert redacted["model"] == "gpt-x"
    assert redacted["raw_token"] == "[REDACTED:api-key]"
    assert redacted["headers"]["x-custom-header"] == "[REDACTED:bearer]"


@pytest.mark.benchmark
def test_benign_inputs_pass_through_unchanged() -> None:
    """The scrubber must not mutate non-secret strings.

    A future "fail-closed" change that raises on legitimate payloads
    would crash the trace pipeline silently; this pins that benign
    input is a no-op so the regression is loud.
    """
    assert _redact_value("") == ""
    assert _redact_value("just a normal log line with no secrets") == (
        "just a normal log line with no secrets"
    )
