# ADR-0026: Server pool for multi-slot llama-server lifecycle management

## Status

Proposed.

## Context

`FoundryServerManager` (`src/foundry_x/infra/server_manager.py`) manages a single
`llama-server` subprocess. It has no concept of a pool — one server, one model,
one host. The `quantization_sweep()` method on `Critic` (ADR-0016) currently runs
one quantization at a time, switching `FOUNDRY_MODEL_PATH` between runs and
waiting for the server to become healthy after each switch.

Three scenarios motivate a pool layer:

1. **Quantization sweep** — running multiple quantizations (q4_k_m, q8_0, f16)
   against the same benchmark suite. With a single manager, each quantization
   requires a full server restart cycle, adding latency and preventing parallelism.
2. **Multi-model serving** — a single physical host or CI node has enough VRAM
   to hold two slot servers simultaneously (different models, different
   quantizations, or different hosts).
3. **Health-aware routing** — when a server slot becomes unhealthy mid-session,
   the pool can transparently route traffic to the next healthiest slot without
   requiring the caller to implement retry logic.

## Decision

### 1. ServerPool class

A new `ServerPool` class is added to `src/foundry_x/infra/server_manager.py`,
beside the existing `FoundryServerManager`. It holds a `dict[str, FoundryServerManager]`
 keyed by a **slot name** (e.g. `"q4-km-fast"`, `"q8-0-heavy"`).

```python
class ServerPool:
    def __init__(self) -> None: ...
    def register(self, slot: str, config: ServerConfig) -> None: ...
    def get(self, slot: str) -> FoundryServerManager: ...
    async def start_all(self) -> dict[str, bool]: ...
    async def stop_all(self) -> None: ...
    async def acquire() -> tuple[str, FoundryServerManager]: ...
```

- `register(slot, config)` — adds a slot to the pool. A slot may be
  registered at most once; calling `register` twice for the same slot raises
  `KeyError`. Config is a resolved `ServerConfig` (not an env dict) so the
  pool is testable and does not re-read env vars on each call.
- `get(slot)` — returns the manager for a registered slot. Raises
  `KeyError` if the slot is not registered.
- `start_all()` — calls `start()` on every registered manager in parallel
  using `asyncio.gather`. Returns a dict mapping slot names to whether that
  slot started successfully. Slots that fail to start do not raise; the
  caller inspects the dict to decide whether to abort.
- `stop_all()` — calls `stop()` on every registered manager in parallel.
- `acquire()` — returns `(slot, manager)` for the healthiest available slot.
  Health is determined by `manager.is_healthy()`. If no slot is healthy and
  `FOUNDRY_SERVER_AUTOSTART=1`, the pool waits for the first slot to become
  healthy (polling `is_healthy` every 2 s, up to `health_ready_timeout_s` from
  that slot's config). If no slot becomes healthy within that window, the
  first registered slot is returned as a fallback (the caller decides whether
  to proceed).

### 2. Pool registration API

The pool is populated by the **caller** (e.g. the sweep CLI or the Runner
wiring). There is no automatic discovery. The caller is responsible for
calling `register` with the appropriate `ServerConfig` per slot before
calling `start_all`.

`ServerConfig.from_env` is extended with an optional `slot` keyword argument
that appends a label to `FOUNDRY_MODEL_ID` so traces for each slot are
attributable:

```python
@dataclass(frozen=True)
class ServerConfig:
    # ... existing fields ...
    slot: str | None = None  # e.g. "q4-km-fast"; included in trace attribution
```

### 3. Pool-aware routing in the Runner

The `Runner` is extended to accept an optional `server_pool: ServerPool` kwarg.
When provided, the Runner calls `pool.acquire()` before opening the agent loop
and routes all model traffic to the returned `manager.host`. When `server_pool`
is `None`, the existing single-manager behaviour is unchanged.

### 4. foundry-sweep --pool mode

`src/foundry_x/evolution/cli.py` (`foundry-sweep`) gains a `--pool` flag.
When `--pool` is set:

- The sweep reads a pool manifest (a dict of slot → `ServerConfig`) from
  `FOUNDRY_POOL_MANIFEST` (a JSON-encoded path, defaulting to
  `infra/pool_manifest.json`).
- It calls `pool.start_all()` before any benchmark runs.
- For each quantization in the sweep, the Runner is wired with `server_pool`
  pointing to the pool. The pool slot for a quantization is resolved by
  matching `FOUNDRY_MODEL_PATH` against the manifest.
- After all quantizations complete, `pool.stop_all()` is called.

When `--pool` is absent, the sweep behaves as it does today (ADR-0016).

### 5. Slot attribution in traces

Each `model_request` and `model_response` event (CONTEXT.md §Event kinds) already
carries `model_id`. When the Runner uses a pool slot, `model_id` is set to
`"{base_model_id}:{slot}"` so traces are attributable per-slot without changing
the existing event schema.

## Alternatives considered

### A — One pool per quantization, sequential restart

The existing approach (ADR-0016): one `FoundryServerManager`, restart between
quantizations. This adds O(n × restart_time) latency for n quantizations and
prevents parallelism. It is the current implementation; the pool approach
supersedes it for multi-slot environments.

### B — External负载均衡 (nginx / envoy)

A reverse proxy in front of multiple llama-server instances provides health
routing and distribution, but adds infrastructure complexity and a new failure
domain. The pool in-process is simpler for single-host CI and development
environments where the operator controls the server lifecycle.

### C — One FoundryServerManager subclass per slot, inheritance hierarchy

Subclassing `FoundryServerManager` to produce `Q4KMFoundryServerManager`,
`Q80FoundryServerManager`, etc. introduces unnecessary coupling: the sweep
code would need to know about each subclass. A registry (the pool dict) is
simpler and avoids a class explosion.

### D — Automatic slot discovery via environment variable globs

Scanning `FOUNDRY_MODEL_PATH_*` env vars to auto-discover slots. This is
fragile (env vars are strings, not typed configs) and makes it hard to control
which slots are active. Explicit registration is more intentional and
testable.

## Consequences

- **Positive**
  - Quantization sweeps with pre-warmed pool slots eliminate per-quantization
    restart latency.
  - Health-aware `acquire()` means mid-session server failures are handled
    transparently without the Runner needing per-slot retry logic.
  - Pool slots are independently configurable (different `n_gpu_layers`,
    `ctx_size`, model path) — enabling heterogeneous multi-model serving on
    the same node.
  - Existing single-manager users (no pool) are unaffected; the pool is
    strictly opt-in.

- **Negative**
  - A new in-process dependency: `ServerPool` holds references to multiple
    `FoundryServerManager` instances, meaning more processes may be running
    simultaneously on the same host. Operators must ensure VRAM and port
    resources are not over-subscribed.
  - The sweep manifest (`infra/pool_manifest.json`) is a new operational
    artifact that must be kept in sync with available model files.

- **Neutral**
  - The `KeyError` on duplicate `register` is intentional: a slot may not
    be registered twice in the same pool lifetime without an explicit
    `deregister` call (deferred; not in scope for this ADR).

## Follow-up work

- `deregister(slot)` on `ServerPool` to allow dynamic slot lifecycle during
  a sweep run (slot goes down for maintenance, comes back up).
- `pool.slot_health()` returning a `dict[str, bool]` snapshot of all slots
  without blocking.
- Integration of `ServerPool` with `PoolAwareModelAdapter` so the adapter,
  not the Runner, holds the pool reference (reduces the Runner surface area).
- ADR update: extend ADR-0016 §4 CI integration to account for pool-based
  sweep runs.
