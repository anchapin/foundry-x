slot: infra
coverage_summary: |
  Inspected infra/docker/, infra/llama-cpp/, infra/scripts/, .github/workflows/,
  .github/actions/install-uv/, .github/dependabot.yml, .dockerignore, .pre-commit-config.yaml,
  and every test in tests/infra/, tests/test_compose_sandbox.py, tests/test_compose_rocm.py,
  tests/test_uv_pinning.py, tests/test_dockerignore.py. Audited 39 area-infra issues (all CLOSED)
  and 50 infra/Dockerfile/dependabot PRs in the merged-PR catalogue (HEAD 99d833b). Ran
  `uv run pytest tests/infra/ tests/test_compose_sandbox.py tests/test_compose_rocm.py
  tests/test_uv_pinning.py tests/test_dockerignore.py` -> 134 passed, 2 skipped. The infra
  slot is at very high saturation: Dockerfile digest+SHA pinning, multi-stage build, BuildKit
  cache mount, image-size baseline ceiling (issue #286), HEALTHCHECK (issue #813), read-only
  root FS + tmpfs caps + narrowed egress (issue #123), pids/ulimits (issue #118), ROCm override
  shadow-prevention (issue #115), rocm_setup.sh pre-flight checks (issues #210/#637/#638/#754),
  GGUF SHA256 verification (issue #284), dependabot for github-actions/docker/uv (issue #283),
  hash-verified install-uv composite action (issue #208), workflow uv-pin test (issue #208),
  launch_llamacpp.sh / run_benchmark.sh / wait_for_llamacpp.sh, quantization-sweep + real-llm
  + rocm CI workflows, model-agnostic test fixtures (issue #531), .dockerignore secrets
  exclusion (issue #23), conflict-markers CI check, ruff lint+format gate (issue #346), and
  gitleaks scan (issue #349) are all shipped and statically guarded. One concrete residual
  gap remains in the workflow-uv-pin test's allowlist; everything else is either closed,
  shipped, or out of reproducibly-testable scope (real-LLM/ROCm smoke behavior).

proposals:
  - candidate_id: infra-01
    title: "fix(infra): extend uv-pin test allowlist to all workflows"
    area: area-infra
    primary_kpi: kpi-regression-rate
    secondary_metrics: []
    phase_labels: [phase-3]
    size: size-s
    needs: []
    label_questions:
      - "size-xs vs size-s overlap is unresolved per ISSUE_GENERATION_PROMPT §Stage 2 rule 9 (using size-s because the diff crosses the size-xs boundary by including a glob-driven discovery change)."
    human_decision_required: []
    priority: P2
    problem: |
      tests/infra/test_workflow_uv_pin.py:32 hard-codes EXPECTED_WORKFLOWS = ["ci.yml", "audit.yml",
      "critic.yml", "docker.yml"], but 9 distinct workflow files currently use the
      hash-verified install-uv composite action (verified via ripgrep on
      `uses: ./.github/actions/install-uv`):
      ci.yml, audit.yml, critic.yml, docker.yml, lint.yml, pre-commit.yml, test.yml,
      quantization-sweep.yml, real-llm.yml. The parametrized test
      `test_no_pip_install_uv_in_workflow` and `test_workflow_uses_composite_action`
      therefore cannot detect a regression in lint.yml, pre-commit.yml, test.yml,
      quantization-sweep.yml, or real-llm.yml — any of those could be edited to call
      `pip install --upgrade uv` and the supply-chain guardrail (issue #208, ADR-0002,
      docs/SECURITY.md threat #3) would silently re-open on five of nine workflows.
      The duplication risk is concrete: lint.yml, pre-commit.yml, test.yml, and
      real-llm.yml were added after issue #208 shipped, and the test allowlist was
      never extended alongside them.
    goal_link: |
      Issue #208 closed with `fix(infra): pin uv in .github/workflows/*.yml via
      hash-verified installer` and explicitly named "every workflow in
      .github/workflows/" as in scope. ADR-0002 plus docs/SECURITY.md threat #3 make
      the supply-chain pin mandatory. The KPI linkage is kpi-regression-rate: the
      regression a `pip install --upgrade uv` could re-introduce is exactly the
      class of failure this KPI tracks — a previously-passing supply-chain
      guardrail that silently regresses. Restoring the regression-detection surface
      on the five unguarded workflows directly closes that gap.
    evidence:
      - type: source
        ref: "tests/infra/test_workflow_uv_pin.py:32"
        verified_claim: "EXPECTED_WORKFLOWS = [\"ci.yml\", \"audit.yml\", \"critic.yml\", \"docker.yml\"]; this list is the only set of workflows the parametrized test_no_pip_install_uv_in_workflow and test_workflow_uses_composite_action iterate over."
      - type: source
        ref: "tests/infra/test_workflow_uv_pin.py:30-33"
        verified_claim: "Module docstring says 'Every workflow in .github/workflows/' and the EXPECTED_WORKFLOWS list is annotated 'If a new workflow is added that needs uv, add it here so the test enforces the pin on it too.' — the comment establishes intent but the list is stale."
      - type: source
        ref: ".github/workflows/lint.yml:14-18 (uses), .github/workflows/pre-commit.yml:14-18 (uses), .github/workflows/test.yml:14-18 + 33-37 + 52-56 (uses), .github/workflows/quantization-sweep.yml:26-30 (uses), .github/workflows/real-llm.yml:25-29 (uses)"
        verified_claim: "Five additional workflows reference `uses: ./.github/actions/install-uv` and are not in EXPECTED_WORKFLOWS. Verified by ripgrep on the live workflow files; 9 hits across 9 distinct files (test.yml appears three times because it has three jobs)."
      - type: source
        ref: ".github/actions/install-uv/action.yml:6-13"
        verified_claim: "Composite-action comment cites issue #208 and ADR-0002; the rationale ('eliminates `pip install --upgrade uv` from CI workflows') is the regression the stale allowlist would miss."
      - type: source
        ref: "docs/SECURITY.md:23 (threat #3 — supply-chain compromise)"
        verified_claim: "Source-of-truth rationale for the pin; threat #3 cites 'A malicious or vulnerable dependency enters via `uv add` and runs inside the trace pipeline' and the SECRETS guardrail section requires gitleaks + dependency pinning."
      - type: source
        ref: "docs/adr/0002-uv-for-dependency-management.md:36-37"
        verified_claim: "ADR-0002 §Consequences: '`uv pip audit` runs in CI for supply-chain checks' and the dependency-management discipline is anchored on this pin."
      - type: source
        ref: "tests/infra/test_workflow_uv_pin.py:57-91 (test_no_pip_install_uv_in_workflow, test_workflow_uses_composite_action)"
        verified_claim: "Both tests are parametrized over EXPECTED_WORKFLOWS only; a `pip install` line in lint.yml, pre-commit.yml, test.yml, quantization-sweep.yml, or real-llm.yml would pass both tests today."
    proposed_scope:
      - "Replace the hard-coded EXPECTED_WORKFLOWS list with a glob-driven discovery: glob `.github/workflows/*.yml` and filter out the composite-action directory (`ISSUE_GENERATION_PROMPT.md`, `PULL_REQUEST_TEMPLATE.md`, `dependabot.yml`, `ISSUE_GENERATION_PROMPT*`); sort for stable ordering."
      - "Add a single extra assertion that the discovered set is a superset of the current EXPECTED_WORKFLOWS, so an accidental removal triggers a loud failure."
      - "Update the module docstring to reflect glob-driven discovery; remove the now-stale \"If a new workflow is added\" annotation since the glob handles it automatically."
    acceptance_criteria:
      - "`uv run pytest tests/infra/test_workflow_uv_pin.py -v` reports every workflow that uses the composite action (>= 9 hits across >= 9 distinct files)."
      - "Editing any unguarded workflow (lint.yml, pre-commit.yml, test.yml, quantization-sweep.yml, real-llm.yml) to add `pip install --upgrade uv` and re-running the test fails with a line-numbered message naming the offender (verifiable by a quick local mutation that is then reverted)."
      - "Removing one of the original four from EXPECTED_WORKFLOWS (and not adding it to the glob source) is impossible, because the glob discovery is the source of truth; the regression-detection surface is monotone increasing."
    validation_commands:
      - "uv run ruff check tests/infra/test_workflow_uv_pin.py"
      - "uv run pytest tests/infra/test_workflow_uv_pin.py -v"
      - "uv run pytest tests/infra/"
      - "uv run pre-commit run --all-files"
    out_of_scope:
      - "Pinning additional third-party actions (e.g. `gitleaks/gitleaks-action`, `actions/upload-artifact`); the issue-scoped pin covers the install-uv supply chain only."
      - "Refactoring the composite action itself (issue #208 / PR #359 already established its shape)."
      - "Adding pytest coverage for non-uv-pinning concerns in the same workflows (e.g. shellcheck on .sh scripts, workflow permissions hardening)."
    risks:
      - risk: "Glob-driven discovery might pick up unintended .yml files if a future PR drops a non-workflow YAML into .github/workflows/."
        mitigation: "Whitelist by filename suffix or maintain a small skip-list; the test fails if a non-workflow YAML is added so the maintainer must explicitly handle it. Whitelist is small and explicit (skip dependabot.yml; skip ISSUE_GENERATION_PROMPT*.md since the directory is .github/workflows/, not .github/)."
      - risk: "The change accidentally double-counts workflows that invoke the composite action multiple times."
        mitigation: "De-duplicate by filename before parametrize (sorted(set(...))). This is observationally safe and matches how the test currently consumes the list."
    dependencies: []
    adr_relevance:
      - "ADR-0002 — advances (closes a static-detection gap in the uv supply-chain pin the ADR mandates)."
    duplicate_audit:
      - ref: "PR #369 (close), PR #393 (close), PR #432"
        relationship: "distinct"
        residual_difference: "Those bumps updated `actions/checkout` and `python:3.11-slim`; they did not touch the workflow-uv-pin test allowlist."
      - ref: "issue #208 (close)"
        relationship: "partial-overlap"
        residual_difference: "Issue #208 introduced the composite action and the original test allowlist of four workflows; it did not anticipate the subsequent workflow additions that left the allowlist stale. The residual slice is exactly the five workflows (lint.yml, pre-commit.yml, test.yml, quantization-sweep.yml, real-llm.yml) added after #208 closed."
      - ref: "PR #432 (test_base_image_is_digest_pinned against python:3.14)"
        relationship: "distinct"
        residual_difference: "That PR fixed a different version-drift regression in tests/test_uv_pinning.py (Dockerfile base version vs hard-coded test assertion); the residual here is in tests/infra/test_workflow_uv_pin.py and concerns the workflow allowlist, not the Python base-image version."
      - ref: "issue #410 (close)"
        relationship: "distinct"
        residual_difference: "Issue #410 was the version-drift debt on Dockerfile base image; closed by deriving the expected version dynamically. Same drift class (manual allowlist becoming stale) but the file and content class are different."
    estimated_diff_lines: 35
    confidence: high
rejected_as_duplicates:
  - idea: "Pin actions/upload-artifact and gitleaks/gitleaks-action to commit SHAs."
    duplicate_of: "issue #282 (close)"
    residual_difference: "Issue #282 already pinned actions/checkout across all workflows; the remaining two actions are pinned by SHAs (verified: `gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e # v3.0.0` and `actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4`). No gap."
  - idea: "Add shellcheck to pre-commit hooks for infra/scripts/*.sh."
    duplicate_of: "issue #350 (close), issue #438 (close)"
    residual_difference: "Those issues established the pre-commit-config.yaml structure and resolved the ruff-version drift; shellcheck is not in `.pre-commit-config.yaml` (verified). Adding it is a real but separate concern (kpi-cycle-time framing) and outside this slot's residual slice — would expand the proposal beyond the evidence-backed install-uv allowlist gap. Logged for a future docs/infra scout pass."
  - idea: "Add CI permissions block (least-privilege `permissions: read-all` etc.) to workflows."
    duplicate_of: "n/a"
    residual_difference: "Not yet an issue, but verified by ripgrep: zero workflows declare a `permissions:` block today. However, the test or compose evidence does not currently demonstrate a measured regression — it is a hardening proposal, not an evidence-backed gap. Deferred until a measured failure or token-leak incident ties it to kpi-regression-rate. Logged as adjacent finding."
  - idea: "Pin `uv python install 3.11` to a digest in each workflow."
    duplicate_of: "issue #410 (close)"
    residual_difference: "Issue #410 / PR #432 already made the Dockerfile-Python-version drift detectable. The 12 `uv python install 3.11` invocations are operator convenience calls, not supply-chain pins — pinning the runtime Python install via `uv` is not in scope of the install-uv SHA256 pin and would change ADR-0002's interpretation. Not proposed without ADR."
  - idea: "Convert ROCm override to a fully content-pinned override (compose-spec hash)."
    duplicate_of: "issue #115 (close), issue #285 (close)"
    residual_difference: "Already shipped + tested; the test_compose_config.py harness validates the merged config."
adjacent_findings:
  - owner_slot: docs
    finding: "test/infra/test_workflow_uv_pin.py module docstring still says 'every workflow in .github/workflows/' but the allowlist is stale; the docs slot could align the comment with the new glob-driven discovery once infra-01 lands."
  - owner_slot: security
    finding: "No `.github/workflows/*.yml` declares a `permissions:` block today (verified by ripgrep). On PRs from forks the default `GITHUB_TOKEN` permission set is broader than required; a least-privilege block (`permissions: contents: read` for PR-only jobs, `permissions: contents: read, pull-requests: write` for jobs that post comments) would narrow the blast radius if a workflow action were compromised. Surface as a separate evidence-backed proposal with an explicit measured or reproduced failure rather than speculative hardening."
  - owner_slot: docs
    finding: "`infra/llama-cpp/README.md` §\"ROCm pitfalls\" still references the `n-gpu-layers / VRAM` constraint; this is good context, not drift, but a one-line cross-reference to `docs/adr/0021-context-pruning-at-scale.md` §2 (5600G/6600 XT VRAM table) would tighten the link between the README and the ADR."
