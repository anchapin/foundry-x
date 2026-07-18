#!/usr/bin/env bash
# scripts/check_pr_closing_refs.sh
#
# Verify that a pull request has exactly N closing issue references.
# Usage: scripts/check_pr_closing_refs.sh <PR_NUMBER> <EXPECTED_COUNT>
#
# Fails (exit 1) when the actual closing-references count differs from the
# expected value. This guards against the common "Closes #N" + auto-added
# "closes #N" duplication that `gh pr create --fill` sometimes produces.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <PR_NUMBER> <EXPECTED_COUNT>" >&2
  exit 2
fi

PR_NUMBER="$1"
EXPECTED="$2"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required" >&2
  exit 2
fi

# closingIssuesReferences is the rendered list of issues/PRs that the PR's
# body language marks as "closes" / "fixes" / "resolves". A PR body that
# repeats the same key twice (e.g. "Closes #N" + auto-inserted duplicate)
# produces more than one entry here.
ACTUAL="$(gh pr view "$PR_NUMBER" --json closingIssuesReferences \
  --jq '.closingIssuesReferences | length')"

if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo "PR #$PR_NUMBER has $ACTUAL closing reference(s); expected $EXPECTED." >&2
  exit 1
fi

echo "PR #$PR_NUMBER closing-references count OK ($ACTUAL)"
