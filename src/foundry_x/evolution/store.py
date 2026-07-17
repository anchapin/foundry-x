"""Persistent store for ProposedEdit records with review-state tracking (issue #498).

The store supports sqlite backends (JSONL is a possible future extension) and
exposes an idempotent state-machine for the review workflow:
pending -> approved -> applied
pending -> rejected

State transitions are logged so the evolution loop can audit review decisions.
"""

from __future__ import annotations

import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class ProposedEditStatus(str, Enum):
    """Review-state vocabulary for a ProposedEdit record (issue #498)."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"


class TrackedProposedEdit(BaseModel):
    """A ProposedEdit with mutable review state for human review (ADR-0006).

    ``id`` is a stable UUID assigned at creation time. ``status`` is the
    mutable review state machine. ``review_reason`` holds the human-readable
    justification for the last transition (approval or rejection).
    ``reviewed_at`` is the ISO-8601 timestamp of the last state change.
    ``created_at`` is set by :meth:`ProposedEditStore.add` and is not
    configurable by callers.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_file: str
    rationale: str
    unified_diff: str
    status: ProposedEditStatus = ProposedEditStatus.PENDING
    review_reason: str = ""
    reviewed_at: str = ""
    created_at: str = ""


_EDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS proposed_edits (
    id TEXT PRIMARY KEY,
    target_file TEXT NOT NULL,
    rationale TEXT NOT NULL,
    unified_diff TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    review_reason TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_proposed_edits_status ON proposed_edits(status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProposedEditStore:
    """SQLite-backed store for TrackedProposedEdit records (issue #498).

    Persists ProposedEdit records with review state. Idempotent state
    transitions: approving an already-approved edit is a no-op; rejecting
    an already-rejected edit is a no-op. Every transition is logged to
    stderr so operators can trace state changes.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_EDIT_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Release the sqlite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def add(self, edit: TrackedProposedEdit) -> TrackedProposedEdit:
        """Persist a new ProposedEdit with PENDING status.

        The edit's ``id``, ``status``, ``created_at`` are set here.
        Returns the saved record.
        """
        edit.id = str(uuid.uuid4())
        edit.status = ProposedEditStatus.PENDING
        edit.created_at = _now()
        with self._conn:
            self._conn.execute(
                "INSERT INTO proposed_edits "
                "(id, target_file, rationale, unified_diff, status, review_reason, reviewed_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    edit.id,
                    edit.target_file,
                    edit.rationale,
                    edit.unified_diff,
                    edit.status.value,
                    edit.review_reason,
                    edit.reviewed_at,
                    edit.created_at,
                ),
            )
        self._log_transition(edit.id, ProposedEditStatus.PENDING, "created")
        return edit

    def get(self, edit_id: str) -> TrackedProposedEdit | None:
        """Retrieve a TrackedProposedEdit by id, or None if not found."""
        row = self._conn.execute(
            "SELECT id, target_file, rationale, unified_diff, status, review_reason, reviewed_at, created_at "
            "FROM proposed_edits WHERE id = ?",
            (edit_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_tracked_edit(row)

    def list_pending(self) -> list[TrackedProposedEdit]:
        """Return all ProposedEdits with PENDING status, oldest first."""
        rows = self._conn.execute(
            "SELECT id, target_file, rationale, unified_diff, status, review_reason, reviewed_at, created_at "
            "FROM proposed_edits WHERE status = ? ORDER BY created_at ASC",
            (ProposedEditStatus.PENDING.value,),
        ).fetchall()
        return [self._row_to_tracked_edit(row) for row in rows]

    def list_all(self) -> list[TrackedProposedEdit]:
        """Return all ProposedEdits ordered by created_at descending."""
        rows = self._conn.execute(
            "SELECT id, target_file, rationale, unified_diff, status, review_reason, reviewed_at, created_at "
            "FROM proposed_edits ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_tracked_edit(row) for row in rows]

    def approve(self, edit_id: str, reason: str = "") -> TrackedProposedEdit | None:
        """Transition a PENDING edit to APPROVED.

        Idempotent: if already APPROVED, returns the edit without error.
        If the edit is REJECTED or APPLIED, returns None and logs a warning.
        """
        edit = self.get(edit_id)
        if edit is None:
            sys.stderr.write(f"approve: edit {edit_id} not found.\n")
            return None
        if edit.status == ProposedEditStatus.APPROVED:
            sys.stderr.write(f"approve: edit {edit_id} already approved.\n")
            return edit
        if edit.status == ProposedEditStatus.REJECTED:
            sys.stderr.write(f"approve: edit {edit_id} is rejected; cannot approve.\n")
            return None
        if edit.status == ProposedEditStatus.APPLIED:
            sys.stderr.write(f"approve: edit {edit_id} is already applied; cannot approve.\n")
            return None
        edit.status = ProposedEditStatus.APPROVED
        edit.review_reason = reason
        edit.reviewed_at = _now()
        with self._conn:
            self._conn.execute(
                "UPDATE proposed_edits SET status = ?, review_reason = ?, reviewed_at = ? WHERE id = ?",
                (edit.status.value, edit.review_reason, edit.reviewed_at, edit_id),
            )
        self._log_transition(edit_id, ProposedEditStatus.APPROVED, reason)
        return edit

    def reject(self, edit_id: str, reason: str = "") -> TrackedProposedEdit | None:
        """Transition a PENDING edit to REJECTED.

        Idempotent: if already REJECTED, returns the edit without error.
        If the edit is APPROVED or APPLIED, returns None and logs a warning.
        """
        edit = self.get(edit_id)
        if edit is None:
            sys.stderr.write(f"reject: edit {edit_id} not found.\n")
            return None
        if edit.status == ProposedEditStatus.REJECTED:
            sys.stderr.write(f"reject: edit {edit_id} already rejected.\n")
            return edit
        if edit.status == ProposedEditStatus.APPROVED:
            sys.stderr.write(f"reject: edit {edit_id} is approved; cannot reject.\n")
            return None
        if edit.status == ProposedEditStatus.APPLIED:
            sys.stderr.write(f"reject: edit {edit_id} is already applied; cannot reject.\n")
            return None
        edit.status = ProposedEditStatus.REJECTED
        edit.review_reason = reason
        edit.reviewed_at = _now()
        with self._conn:
            self._conn.execute(
                "UPDATE proposed_edits SET status = ?, review_reason = ?, reviewed_at = ? WHERE id = ?",
                (edit.status.value, edit.review_reason, edit.reviewed_at, edit_id),
            )
        self._log_transition(edit_id, ProposedEditStatus.REJECTED, reason)
        return edit

    def mark_applied(self, edit_id: str) -> TrackedProposedEdit | None:
        """Transition an APPROVED edit to APPLIED.

        Idempotent: if already APPLIED, returns the edit without error.
        If the edit is PENDING or REJECTED, returns None and logs a warning.
        """
        edit = self.get(edit_id)
        if edit is None:
            sys.stderr.write(f"apply: edit {edit_id} not found.\n")
            return None
        if edit.status == ProposedEditStatus.APPLIED:
            sys.stderr.write(f"apply: edit {edit_id} already applied.\n")
            return edit
        if edit.status == ProposedEditStatus.PENDING:
            sys.stderr.write(f"apply: edit {edit_id} is pending; cannot apply.\n")
            return None
        if edit.status == ProposedEditStatus.REJECTED:
            sys.stderr.write(f"apply: edit {edit_id} is rejected; cannot apply.\n")
            return None
        edit.status = ProposedEditStatus.APPLIED
        edit.reviewed_at = _now()
        with self._conn:
            self._conn.execute(
                "UPDATE proposed_edits SET status = ?, reviewed_at = ? WHERE id = ?",
                (edit.status.value, edit.reviewed_at, edit_id),
            )
        self._log_transition(edit_id, ProposedEditStatus.APPLIED, "")
        return edit

    def _row_to_tracked_edit(self, row: tuple) -> TrackedProposedEdit:
        (
            id_,
            target_file,
            rationale,
            unified_diff,
            status,
            review_reason,
            reviewed_at,
            created_at,
        ) = row
        return TrackedProposedEdit(
            id=id_,
            target_file=target_file,
            rationale=rationale,
            unified_diff=unified_diff,
            status=ProposedEditStatus(status),
            review_reason=review_reason,
            reviewed_at=reviewed_at,
            created_at=created_at,
        )

    def _log_transition(self, edit_id: str, new_status: ProposedEditStatus, reason: str) -> None:
        ts = _now()
        msg = f"[{ts}] edit {edit_id}: transition to {new_status.value}"
        if reason:
            msg += f" — reason: {reason}"
        sys.stderr.write(msg + "\n")
