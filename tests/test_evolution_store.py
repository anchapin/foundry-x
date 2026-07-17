"""Tests for ProposedEditStore and review CLI commands (issue #498)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from foundry_x.evolution.store import ProposedEditStore, TrackedProposedEdit, ProposedEditStatus


_DIFF = (
    "--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new\n"
)


def _make_tracked() -> TrackedProposedEdit:
    return TrackedProposedEdit(
        target_file="harness/system_prompt.txt",
        rationale="tighten tool guidance",
        unified_diff=_DIFF,
    )


class TestTrackedProposedEdit:
    def test_default_status_is_pending(self):
        edit = _make_tracked()
        assert edit.status == ProposedEditStatus.PENDING

    def test_default_review_reason_empty(self):
        edit = _make_tracked()
        assert edit.review_reason == ""

    def test_default_reviewed_at_empty(self):
        edit = _make_tracked()
        assert edit.reviewed_at == ""


class TestProposedEditStore:
    def test_add_assigns_id_and_pending_status(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        assert saved.id != ""
        assert saved.status == ProposedEditStatus.PENDING
        assert saved.created_at != ""
        store.close()

    def test_get_returns_saved_edit(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        retrieved = store.get(saved.id)
        assert retrieved is not None
        assert retrieved.id == saved.id
        assert retrieved.target_file == edit.target_file
        assert retrieved.status == ProposedEditStatus.PENDING
        store.close()

    def test_get_returns_none_for_unknown_id(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        result = store.get("unknown-id")
        assert result is None
        store.close()

    def test_list_pending_returns_only_pending(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        store.add(edit)
        pending = store.list_pending()
        assert len(pending) == 1
        assert pending[0].status == ProposedEditStatus.PENDING
        store.close()

    def test_list_pending_excludes_approved(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        store.approve(saved.id, reason="looks good")
        pending = store.list_pending()
        assert len(pending) == 0
        store.close()

    def test_approve_transitions_to_approved(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        result = store.approve(saved.id, reason="lgtm")
        assert result is not None
        assert result.status == ProposedEditStatus.APPROVED
        assert result.review_reason == "lgtm"
        assert result.reviewed_at != ""
        store.close()

    def test_approve_idempotent_already_approved(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        store.approve(saved.id, reason="first")
        result = store.approve(saved.id, reason="second")
        assert result is not None
        assert result.review_reason == "first"
        store.close()

    def test_approve_rejected_edit_returns_none(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        store.reject(saved.id, reason="not good")
        result = store.approve(saved.id, reason="override")
        assert result is None
        store.close()

    def test_approve_applied_edit_returns_none(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        store.approve(saved.id, reason="approved")
        store.mark_applied(saved.id)
        result = store.approve(saved.id, reason="try again")
        assert result is None
        store.close()

    def test_reject_transitions_to_rejected(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        result = store.reject(saved.id, reason="too risky")
        assert result is not None
        assert result.status == ProposedEditStatus.REJECTED
        assert result.review_reason == "too risky"
        assert result.reviewed_at != ""
        store.close()

    def test_reject_idempotent_already_rejected(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        store.reject(saved.id, reason="first")
        result = store.reject(saved.id, reason="second")
        assert result is not None
        assert result.review_reason == "first"
        store.close()

    def test_reject_approved_edit_returns_none(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        store.approve(saved.id, reason="approved")
        result = store.reject(saved.id, reason="undo")
        assert result is None
        store.close()

    def test_reject_pending_edit_returns_none(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        result = store.reject(saved.id, reason="")
        assert result is not None
        assert result.status == ProposedEditStatus.REJECTED
        store.close()

    def test_mark_applied_transitions_approved_to_applied(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        store.approve(saved.id, reason="approved")
        result = store.mark_applied(saved.id)
        assert result is not None
        assert result.status == ProposedEditStatus.APPLIED
        store.close()

    def test_mark_applied_idempotent_already_applied(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        store.approve(saved.id, reason="approved")
        store.mark_applied(saved.id)
        result = store.mark_applied(saved.id)
        assert result is not None
        assert result.status == ProposedEditStatus.APPLIED
        store.close()

    def test_mark_applied_pending_returns_none(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        result = store.mark_applied(saved.id)
        assert result is None
        store.close()

    def test_mark_applied_rejected_returns_none(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit = _make_tracked()
        saved = store.add(edit)
        store.reject(saved.id, reason="rejected")
        result = store.mark_applied(saved.id)
        assert result is None
        store.close()

    def test_list_all_returns_all_edits(self, tmp_path: Path):
        store = ProposedEditStore(tmp_path / "store.db")
        edit1 = _make_tracked()
        edit2 = _make_tracked()
        saved1 = store.add(edit1)
        saved2 = store.add(edit2)
        store.approve(saved1.id, reason="approved")
        store.reject(saved2.id, reason="rejected")
        all_edits = store.list_all()
        assert len(all_edits) == 2
        store.close()


class TestCliIntegration:
    def test_list_pending_empty(self, tmp_path: Path):
        store_path = tmp_path / "store.db"
        store = ProposedEditStore(store_path)
        store.close()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "foundry_x.evolution.cli",
                "list-pending",
                "--store",
                str(store_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "No pending ProposedEdits" in result.stdout

    def test_list_pending_shows_pending_edit(self, tmp_path: Path):
        store_path = tmp_path / "store.db"
        store = ProposedEditStore(store_path)
        edit = _make_tracked()
        saved = store.add(edit)
        store.close()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "foundry_x.evolution.cli",
                "list-pending",
                "--store",
                str(store_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert saved.id in result.stdout
        assert "pending" in result.stdout.lower()

    def test_approve_success(self, tmp_path: Path):
        store_path = tmp_path / "store.db"
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        (harness_dir / "system_prompt.txt").write_text("old content\n", encoding="utf-8")
        store = ProposedEditStore(store_path)
        edit = _make_tracked()
        saved = store.add(edit)
        store.close()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "foundry_x.evolution.cli",
                "approve",
                saved.id,
                "--store",
                str(store_path),
                "--reason",
                "looks good",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Approved" in result.stdout

    def test_approve_unknown_id_returns_error(self, tmp_path: Path):
        store_path = tmp_path / "store.db"
        store = ProposedEditStore(store_path)
        store.close()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "foundry_x.evolution.cli",
                "approve",
                "unknown-id",
                "--store",
                str(store_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

    def test_reject_success(self, tmp_path: Path):
        store_path = tmp_path / "store.db"
        store = ProposedEditStore(store_path)
        edit = _make_tracked()
        saved = store.add(edit)
        store.close()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "foundry_x.evolution.cli",
                "reject",
                saved.id,
                "--store",
                str(store_path),
                "--reason",
                "too risky",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Rejected" in result.stdout

    def test_apply_success(self, tmp_path: Path):
        store_path = tmp_path / "store.db"
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        prompt_file = harness_dir / "system_prompt.txt"
        prompt_file.write_text("old\n", encoding="utf-8")
        store = ProposedEditStore(store_path)
        edit = _make_tracked()
        saved = store.add(edit)
        store.approve(saved.id, reason="approved")
        store.close()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "foundry_x.evolution.cli",
                "apply",
                saved.id,
                "--store",
                str(store_path),
                "--harness-dir",
                str(harness_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Applied" in result.stdout
        content = prompt_file.read_text(encoding="utf-8")
        assert "new" in content

    def test_apply_unapproved_returns_error(self, tmp_path: Path):
        store_path = tmp_path / "store.db"
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        store = ProposedEditStore(store_path)
        edit = _make_tracked()
        saved = store.add(edit)
        store.close()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "foundry_x.evolution.cli",
                "apply",
                saved.id,
                "--store",
                str(store_path),
                "--harness-dir",
                str(harness_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "pending" in result.stderr.lower() or "must be approved" in result.stderr.lower()
