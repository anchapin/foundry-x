# ADR-0004: Self-modification guardrails via the Critic gate

## Status

Accepted. 2026-07-10.
Last updated 2026-07-13 (issue #353: sandbox architecture).

## Context

FoundryX edits its own `harness/` files. Without gatekeeping, a bad
proposal could degrade the agent or silently introduce regressions.
The PRD calls for a `Critic`; the question is what the Critic must
enforce and how strictly.

## Decision

Every harness edit, whether produced by the `Evolver` or proposed by
a human, must pass through the `Critic` gate before it is marked
active. The gate runs:

1. The full pytest suite.
2. The benchmark suite (`benchmarks/`) at a configurable size.
3. A regression check: no previously-passing benchmark task may
   newly fail.

Hand-edits to `harness/` are not blocked but require a PR with
explicit justification and a second human reviewer.

### Sandbox architecture

Critic evaluation runs inside a **named Docker container** that is
spawned and torn down per evaluation call (`src/foundry_x/evolution/sandbox.py`).

The lifecycle for a single `Critic.evaluate(proposed_diff)` call is:

1. **Copy.** ``harness_dir`` is copied into a fresh ``TemporaryDirectory``
   on the host. The live ``harness_dir`` is never touched.
2. **Diff apply.** ``git apply`` runs on the host against the copy.  A
   patch that fails to apply is rejected immediately, before the
   container starts.
3. **Container spawn.** A uniquely-named container (``critic-eval-<uuid>``)
   is started with the harness copy bind-mounted read-only at
   ``/app/harness``.  The container's root filesystem is read-only;
   ``/tmp`` and ``/var/tmp`` are tmpfs with size caps; ``src/`` and
   ``tests/`` are bind-mounted read-only from the host.  This matches
   the hardening in ``infra/docker/docker-compose.yml``.
4. **Load check.** ``harness/scripts/load_check.py`` runs inside the
   container.  A harness that fails to load is rejected before pytest
   is spawned.
5. **Pytest.** ``pytest -m benchmark`` runs inside the same container.
6. **Container teardown.** The container is stopped and removed.

The ``SandboxConfig`` model (``src/foundry_x/evolution/sandbox.py``)
exposes the container image tag, the logs directory mount, and the UV
cache directory so operators can tune sandbox resource use.

## Consequences

- The Critic is a hard dependency of the Evolver. Removing it is a
  breaking change and requires its own ADR.
- The Critic itself is versioned and tested. Changes to the Critic
  require their own ADR.
- Evolution runs are slower but bounded; failure modes are
  observable in traces.
- The sandbox container provides a second isolation layer: even if the
  evaluated harness copy is malicious, the container's root FS is
  read-only and the container has no broad network egress.
- See `docs/SECURITY.md` for the broader threat model this guard
  exists within.
- See `AGENTS.md` for the operational rule that forbids bypassing
  this gate.
