# FoundryX sandbox image & run path

`infra/docker/` provides the containerized run path that
`docs/SECURITY.md:54-59` mandates for benchmarks and evolution:

> Run benchmarks and evolution inside a Docker container with read-only
> mounts for the host filesystem (see `infra/`).

Two files implement this:

| File | Role |
|------|------|
| `Dockerfile` | Builds the sandbox image (Python 3.11 + `uv` + deps). Unchanged. |
| `docker-compose.yml` | The **run-time wrapper** that mounts the host source read-only, hardens the container's own filesystem (`read_only` + capped `tmpfs`), narrows egress to a single LLAMACPP gateway, and caps resources. |

## Why a Compose file?

The Dockerfile `COPY`s `src`/`harness`/`tests` into the image at *build*
time. That baked copy goes stale the moment the `Evolver` produces a new
harness revision. The Compose file bind-mounts the **live** `src/` and
`harness/` read-only (`:ro`) so every run exercises the current harness
without an image rebuild — a prerequisite for the ADR-0004 regression
check that attributes traces to the correct harness revision.

## Run

From the repo root (the Compose file's build context is `../..`):

```bash
# 1. Provide runtime configuration (never committed).
cp .env.example .env
# Edit .env: set FOUNDRY_*, and ensure the host llama-server is running
# (infra/llama-cpp). LLAMACPP_HOST inside the container is overridden to
# http://llamacpp:8080 by the Compose file, so leave the .env value as-is.

# 2. (Optional) tune the sandbox caps in infra/docker/docker-compose.yml:
#    deploy.resources.limits.memory / cpus, tmpfs sizes.

# 3. Run the sandboxed runner.
docker compose -f infra/docker/docker-compose.yml run --rm foundryx \
    --task "your benchmark task prompt"
```

`--task` is the runner's required argument (`src/foundry_x/execution/runner.py`).
The `--rm` flag discards the container after the run; only the `logs/`
mount persists on the host.

## What the sandbox guarantees

Inspect a running container to confirm the guardrails:

```bash
# src/harness mounts are read-only, logs is read-write, root FS is read-only
docker compose -f infra/docker/docker-compose.yml run --rm \
    --entrypoint sh foundryx -c 'mount | grep -E "/app|tmpfs"'

# tmpfs caps are in place
docker compose -f infra/docker/docker-compose.yml run --rm \
    --entrypoint sh foundryx -c 'df -h /tmp /var/tmp /run'

# a memory limit is set
docker compose -f infra/docker/docker-compose.yml run --rm \
    --entrypoint sh foundryx -c 'cat /sys/fs/cgroup/memory.max 2>/dev/null \
    || cat /sys/fs/cgroup/memory/memory.limit_in_bytes'
```

| Guardrail | How | Threat |
|-----------|-----|--------|
| Host source read-only | `./src`, `./harness`, `./tests` mounted `:ro` | #6 (local privilege) |
| Read-only container root FS | `read_only: true` on the service | #5, #6 (writable-surface exhaustion / persistence) |
| Bounded writable surface inside container | `tmpfs` on `/tmp`, `/var/tmp`, `/run` (256m, 1777) plus `./logs` host mount | #5, #6 |
| Narrowed egress to host | dedicated `sandbox` bridge; only `llamacpp` host alias exists, and only `LLAMACPP_HOST:8080` consumes it | #5 (resource exhaustion via unintended network calls) |
| Memory + CPU cap | `deploy.resources.limits` | #5 (resource exhaustion) |
| No capabilities / no priv escalation | `cap_drop: ALL`, `security_opt: no-new-privileges` | #6 |

The threat-model rationale for the `read_only`, `tmpfs`, and narrowed
`extra_hosts` choices is documented inline in `docker-compose.yml` so
the comment travels with the change.

## ROCm GPU sandbox (RX 6600 XT)

The default compose path runs on CPU. For operators who want to evolve
the harness against the local ROCm-built llama-server
(`infra/llama-cpp/`), issue #115 ships an *override* file that
re-attaches the host AMD GPU without changing the CPU-only default:

```bash
docker compose \
    -f infra/docker/docker-compose.yml \
    -f infra/docker/docker-compose.rocm.yml \
    run --rm foundryx --task "your ROCm benchmark task prompt"
```

`docker-compose.rocm.yml` declares only the ROCm-specific bits; every
other field (read-only mounts, tmpfs caps, narrowed egress, ulimits,
capability drops) is inherited from `docker-compose.yml` via Compose
v2's merge-by-service. The override MUST NOT redeclare any base-file
key -- see the hardening test below.

| Override field | Value | Why |
|---|---|---|
| `devices` | `/dev/kfd`, `/dev/dri` | KFD compute + DRM render nodes |
| `group_add` | `video`, `render` | `/dev/dri/renderD*` is group-gated |
| `environment.HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | RX 6600 XT (gfx1032) on older kernels |
| `environment.ROCM_PATH` | `/opt/rocm` | Standard ROCm install path |

### Manual smoke test (confirms `/health` from inside the container)

After starting the host llama-server (e.g. via
`infra/llama-cpp/rocm_setup.sh --smoke-test <gguf>`), the cheapest
proof the GPU is actually wired through is to call `/health` from
inside the override-launched container:

```bash
docker compose \
    -f infra/docker/docker-compose.yml \
    -f infra/docker/docker-compose.rocm.yml \
    run --rm \
    --entrypoint curl foundryx http://llamacpp:8080/health
```

Expect a JSON body containing `ok` (typically `{"status":"ok"}`). If
the request hangs or fails, the host llama-server is most likely not
running, or `LLAMACPP_HOST` does not match the override's `extra_hosts`
alias -- see `docker-compose.yml` "THREAT MODEL: egress".

To confirm the GPU devices are visible inside the container:

```bash
docker compose \
    -f infra/docker/docker-compose.yml \
    -f infra/docker/docker-compose.rocm.yml \
    run --rm \
    --entrypoint sh foundryx -c 'ls -l /dev/kfd /dev/dri/renderD128 2>&1'
```

Expect both paths present and the render node readable. If `/dev/kfd`
is absent, the host kernel does not have `amdgpu` loaded -- see
`infra/llama-cpp/README.md` "ROCm pitfalls".

## Verification

`tests/test_compose_sandbox.py` statically validates the Compose file
enforces these properties (read-only source mounts, writable logs,
memory limit, non-host network, **read-only root FS**, **tmpfs caps**,
**narrowed extra_hosts**) so the guardrail cannot regress silently.
`tests/test_compose_rocm.py` does the same for the ROCm override
(devices, group_add, env vars) and asserts the override does NOT
silently shadow the base file's hardening keys.
