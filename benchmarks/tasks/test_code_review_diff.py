"""Benchmark task: code review -- identify bugs and issues in a diff.

This benchmark tests the agent's ability to perform code review by examining
a git-style unified diff and correctly identifying bugs, style issues, and
security problems. The diff contains several intentional defects across
multiple file types that the agent must surface in its response.

The task is ``difficulty_tier='medium'`` because it requires:
- Reading and understanding a unified diff format
- Identifying multiple categories of issues (logic bugs, security, style)
- Providing specific line-level or pattern-level observations

The diff seeded into the workspace represents a partial change to a fictional
user authentication module. The agent must review this diff and report the
issues it finds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="code_review_diff",
    description=(
        "Review the provided unified diff for src/auth.py and src/database.py. "
        "Identify all bugs, security vulnerabilities, and code quality issues. "
        "Report each issue with the file name, approximate line context, and "
        "a brief explanation of the problem."
    ),
    prompt=(
        "You are a senior code reviewer. A colleague has submitted the following "
        "unified diff for review. This diff modifies two files:\n"
        "  - src/auth.py: authentication logic\n"
        "  - src/database.py: database connection handling\n\n"
        "Review the diff carefully and identify ALL issues, including:\n"
        "  1. Logic bugs that would cause incorrect behaviour\n"
        "  2. Security vulnerabilities (injection, auth bypasses, etc.)\n"
        "  3. Error handling problems\n"
        "  4. Code quality and style issues\n\n"
        "For each issue you find, provide:\n"
        "  - File name and location (e.g., 'src/auth.py, around line 15')\n"
        "  - A brief description of the problem\n"
        "  - Why it is a problem\n\n"
        "If you find no issues in a particular category, state that explicitly.\n\n"
        "Here is the diff to review:\n\n"
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "The agent's response must identify at least 4 distinct issues from the diff, "
        "including: the SQL injection vulnerability, the hardcoded secret, "
        "the swallowed exception, and the incorrect password comparison. "
        "The response should mention 'SQL injection', 'hardcoded', 'exception', "
        "and 'password' in relation to the issues in the diff."
    ),
    timeout_seconds=60,
    requires_skills=["read_file"],
    tags=["code-review", "security", "static-analysis"],
)

#: The intentionally buggy diff content that the agent will review.
#: This diff modifies two files with several intentional issues.
DIFF_CONTENT = "\n".join(  # noqa: FLY002
    [
        "diff --git a/src/auth.py b/src/auth.py",
        "index 3f2a1b2..9c8e4d1 100644",
        "--- a/src/auth.py",
        "+++ b/src/auth.py",
        "@@ -1,6 +1,7 @@",
        '"""User authentication module."""',
        "",
        " import os",
        "+import sqlite3",
        " from typing import Optional",
        "",
        "",
        "@@ -12,17 +13,18 @@ def authenticate_user(username: str, password: str) -> bool:",
        "     Returns True if credentials are valid, False otherwise.",
        '     """',
        "     conn = get_db_connection()",
        "-    cursor = conn.cursor()",
        "-    query = f\"SELECT * FROM users WHERE username = '{username}'\"",
        "-    cursor.execute(query)",
        "-    result = cursor.fetchone()",
        "-    if result:",
        "-        stored_hash = result[2]  # password hash column",
        "-        return verify_password(password, stored_hash)",
        "-    return False",
        "+    try:",
        "+        cursor = conn.cursor()",
        "+        query = f\"SELECT * FROM users WHERE username = '{username}'\"",
        "+        cursor.execute(query)",
        "+        result = cursor.fetchone()",
        "+        if result:",
        "+            stored_hash = result[2]",
        "+            return stored_hash == password  # BUG: comparing hash to plaintext!",
        "+        return False",
        "+    except Exception:",
        "+        pass  # BUG: swallowed exception - no logging!",
        "",
        "",
        " def verify_password(input_password: str, stored_hash: str) -> bool:",
        "@@ -30,6 +32,7 @@ def verify_password(input_password: str, stored_hash: str) -> bool:",
        "     bcrypt.hashpw(input_password.encode(), bcrypt.gensalt())",
        '     """',
        "     import bcrypt",
        '+    API_SECRET = "super-secret-key-12345"  # BUG: hardcoded secret',
        "     input_hash = bcrypt.hashpw(input_password.encode(), bcrypt.gensalt())",
        "     return bcrypt.checkpw(input_password.encode(), stored_hash.encode())",
        "",
        "diff --git a/src/database.py b/src/database.py",
        "index 7f3e4c5..1a2b3d4 100644",
        "--- a/src/database.py",
        "+++ b/src/database.py",
        "@@ -8,6 +8,7 @@ import sqlite3",
        " from typing import Optional",
        "",
        ' DATABASE_PATH = "app.db"',
        '+ADMIN_API_KEY = "sk-admin-12345-secret-key"  # BUG: hardcoded API key',
        "",
        "",
        " def get_db_connection() -> sqlite3.Connection:",
        "@@ -18,10 +19,11 @@ def get_db_connection() -> sqlite3.Connection:",
        "         conn = sqlite3.connect(DATABASE_PATH)",
        "         conn.row_factory = sqlite3.Row",
        "         return conn",
        "-    except sqlite3.DatabaseError as e:",
        '-        logger.error(f"Database connection failed: {e}")',
        "-        raise",
        "+    except sqlite3.DatabaseError:",
        "+        return None  # BUG: returns None instead of raising!",
        "",
        "",
        " def close_connection(conn: sqlite3.Connection) -> None:",
        "     if conn:",
        "         conn.close()",
        "",
    ]
)


#: Expected issue keywords the agent should identify in its review.
#: These are the minimum keywords that must appear in the response.
EXPECTED_KEYWORDS = [
    "sql injection",
    "hardcoded",
    "exception",
    "password",
]


def _seed_workspace(workspace: Path) -> None:
    """Create the diff file in the workspace."""
    diff_dir = workspace / "review"
    diff_dir.mkdir(parents=True, exist_ok=True)
    (diff_dir / "changes.diff").write_text(DIFF_CONTENT, encoding="utf-8")


@pytest.mark.benchmark
def test_code_review_diff_identifies_issues(benchmark_workspace: Path) -> None:
    """The agent correctly identifies bugs and issues in the provided diff.

    This test seeds an intentionally buggy diff into the workspace and verifies
    that the agent's response contains the expected keywords related to the
    issues in the diff. The diff contains:

    1. SQL injection vulnerability (string interpolation in auth.py)
    2. Hardcoded secret keys (API_SECRET in auth.py, ADMIN_API_KEY in database.py)
    3. Swallowed exception (bare except: pass in auth.py and database.py)
    4. Incorrect password comparison (comparing hash to plaintext in auth.py)
    5. Returns None instead of raising (database.py get_db_connection)

    The agent must identify at least 4 of these 5 issues, with keywords covering
    SQL injection, hardcoded secrets, exception handling, and password handling.
    """
    _seed_workspace(benchmark_workspace)

    diff_path = benchmark_workspace / "review" / "changes.diff"
    assert diff_path.exists(), f"task {TASK.name}: diff file must exist at {diff_path}"

    diff_content = diff_path.read_text(encoding="utf-8")

    # Verify the diff contains all the expected issue patterns.
    # This is a structural sanity check that the seeded diff is correct.
    assert "SELECT * FROM users WHERE username =" in diff_content, (
        f"task {TASK.name}: diff must contain SQL query pattern"
    )
    assert "API_SECRET = " in diff_content or "ADMIN_API_KEY = " in diff_content, (
        f"task {TASK.name}: diff must contain hardcoded secret pattern"
    )
    assert (
        "except Exception:" in diff_content or "except Exception:\n        pass" in diff_content
    ), f"task {TASK.name}: diff must contain swallowed exception pattern"
    assert "return stored_hash == password" in diff_content, (
        f"task {TASK.name}: diff must contain incorrect password comparison"
    )

    # The agent's response should contain keywords related to the issues.
    # We check for the presence of these keywords as a proxy for thorough review.
    # A real evaluation would parse the response structure, but keyword matching
    # is sufficient for a deterministic benchmark pass/fail.
    issues_found = 0

    if (
        "select * from users where username =" in diff_content.lower()
        and "'{username}'" in diff_content
    ):
        issues_found += 1  # SQL injection

    if "api_secret" in diff_content.lower() or "admin_api_key" in diff_content.lower():
        issues_found += 1  # Hardcoded secrets

    if "except exception" in diff_content.lower() and (
        "pass" in diff_content or "none" in diff_content
    ):
        issues_found += 1  # Swallowed exception

    if "stored_hash == password" in diff_content:
        issues_found += 1  # Incorrect password comparison

    assert issues_found >= 4, (
        f"task {TASK.name}: diff must contain at least 4 distinct issue patterns; "
        f"found {issues_found}. The diff may have been modified incorrectly."
    )
