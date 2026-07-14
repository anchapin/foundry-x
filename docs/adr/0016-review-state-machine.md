# ADR-0016: Review State Machine for ProposedEdits

## Status

Accepted. 2026-07-14.

## Context

ProposedEdits currently go directly from Evolver to Critic gate. For human review,
we need a state machine that allows human reviewers to approve or reject edits
before they are applied to the harness (issue #497).

## Decision

We introduce a review state machine for ProposedEdit objects with four states:

- `PROPOSED`: Initial state when the Evolver creates the edit.
- `PENDING_REVIEW`: Edit is awaiting human review.
- `APPROVED`: Edit has been approved and can be applied to harness.
- `REJECTED`: Edit has been rejected and must not be applied.

### State Transitions

```
PROPOSED → PENDING_REVIEW (Evolver submits for review)
PENDING_REVIEW → APPROVED (Human approves)
PENDING_REVIEW → REJECTED (Human rejects)
```

`APPROVED` and `REJECTED` are terminal states — no further transitions are allowed.

### Implementation

- `ReviewState` enum in `evolver.py` defines the four states.
- `ProposedEdit.review_state` field defaults to `PROPOSED`.
- `ReviewStateMachine` class enforces valid transitions and raises
  `InvalidStateTransition` for invalid transitions.
- Every state transition is logged to the trace store with `review_state_transition` kind.
- `ReviewStateMachine.can_apply(state)` returns `True` only for `APPROVED` state.

### ADR Relationship

- ADR-0004 defines the Critic gate. This ADR extends that design with a human-review
  layer *before* the Critic gate.
- ADR-0006 defines pydantic models at module boundaries; `ReviewState` and
  `ProposedEdit` follow this pattern.
- ADR-0003 defines the trace store; state transitions are logged there per this ADR.

## Consequences

- Only edits in `APPROVED` state may be applied to the harness.
- The state machine prevents invalid transitions (e.g., `REJECTED → APPROVED`).
- All transitions are observable in the trace store.
- Existing code that creates `ProposedEdit` objects without a `review_state` will
  get the default `PROPOSED` state, which is backward-compatible.
