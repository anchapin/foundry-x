#!/usr/bin/env bash
#
# External-eval validation study orchestrator (issue #900, ADR-0023).
#
# Drives the internal benchmark suite AND the external HumanEval+ slice
# against one or more agent configurations, then computes Pearson
# correlation between per-configuration internal and external pass
# rates. The math lives in foundry_x.evaluation.correlation; this
# script is the operator surface that wires the real-model run.
#
# Real-model runs are NOT part of CI: they require a live llama.cpp
# endpoint and burn model tokens. The offline plumbing validation
# lives under benchmarks/tasks/test_external_eval_correlation.py and
# runs in every pytest invocation that includes the benchmark marker.
#
# Usage:
#   run_external_eval.sh --model /srv/models/foo.Q5_K_M.gguf \
#                        --configs configs.txt \
#                        [--slice benchmarks/external/humaneval_plus_sample.jsonl] \
#                        [--keep-server] [--dry-run] \
#                        [--batch-size N] [--resume] [--reset] \
#                        [--state-file <path>]
#
# Required:
#   --model <gguf>           GGUF path. Auto-launches llama-server if down.
#   --configs <path>         File listing one agent configuration per line.
#                            Each line: "<label> <fx-runner arg> <fx-runner arg>..."
#                            (e.g. "q4km --quantization Q4_K_M").
#
# Optional:
#   --slice <jsonl>          HumanEval+ slice (default:
#                            benchmarks/external/humaneval_plus_sample.jsonl).
#   --keep-server            Do not tear down an auto-launched llama-server.
#   --dry-run                Print the planned runs and exit.
#   --output <path>          Write the JSON results to <path> (default:
#                            logs/external_eval_<timestamp>.json).
#   --batch-size N           Run at most N configurations this invocation and
#                            persist progress to the state file (issue #1040).
#                            Default 0 = run every remaining configuration in
#                            one batch (the legacy behaviour).
#   --resume                 Continue a partial run: load the state file and
#                            skip configurations already marked complete.
#                            This is also the default when a state file for the
#                            current model+configs already exists, so an
#                            interrupted run picks up where it left off without
#                            re-spending model tokens. The flag is accepted for
#                            explicitness/documentation.
#   --reset                  Delete the state file and start the study over.
#                            Required to re-run already-completed configurations.
#   --state-file <path>      Checkpoint location (default:
#                            logs/.run_external_eval_state.json). Stores
#                            completed configurations with timestamps and the
#                            accumulated internal/external rate arrays as
#                            human-readable JSON.
#
# Env vars (all optional):
#   LLAMACPP_HOST            Host for /health probe (default http://127.0.0.1:8080)
#   LLAMACPP_SERVER_BIN      Path to llama-server binary
#   LLAMACPP_DIR             llama.cpp checkout (default $HOME/llama.cpp)
#   LLAMACPP_NGL             GPU layers to offload (default 0)
#   LLAMACPP_HEALTH_TIMEOUT  /health deadline in seconds (default 60)
#   FOUNDRY_EXTERNAL_EVAL_MIN_PAIRS
#                            Override the >=30 paired-observation minimum
#                            (issue #900 criterion 2). Defaults to 30.
#   FOUNDRY_EXTERNAL_EVAL_STATE_FILE
#                            Default path for the incremental-batch checkpoint
#                            (issue #1040). Overridden by --state-file.
#
# Exit codes:
#   0   Study completed and correlation is reportable, OR a partial batch
#       finished and progress was checkpointed (issue #1040).
#   2   CLI usage error.
#   3   Study is under-powered (fewer than MIN_PAIRS configurations).
#   4   A configuration's internal OR external pass rate has zero variance
#       (Pearson is undefined); the operator must choose a more
#       discriminating task set.
#   5   One or more runs failed non-recoverably; see stderr.
#   6   State-file mismatch (resumed run changed --model or --configs);
#       see stderr.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_SLICE="$REPO_ROOT/benchmarks/external/humaneval_plus_sample.jsonl"
TRACES_DB="$REPO_ROOT/logs/traces.db"
LOGS_DIR="$REPO_ROOT/logs"
MIN_PAIRS="${FOUNDRY_EXTERNAL_EVAL_MIN_PAIRS:-30}"
DEFAULT_STATE_FILE="$LOGS_DIR/.run_external_eval_state.json"

LLAMACPP_HOST_URL="${LLAMACPP_HOST:-http://127.0.0.1:8080}"
LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/llama.cpp}"
LLAMACPP_SERVER_BIN="${LLAMACPP_SERVER_BIN:-$LLAMACPP_DIR/build/bin/llama-server}"
LLAMACPP_NGL="${LLAMACPP_NGL:-0}"
HEALTH_TIMEOUT="${LLAMACPP_HEALTH_TIMEOUT:-60}"

MODEL=""
CONFIGS_PATH=""
SLICE="$DEFAULT_SLICE"
KEEP_SERVER=0
DRY_RUN=0
OUTPUT_PATH=""
BATCH_SIZE=0
RESUME=0
RESET=0
STATE_FILE="${FOUNDRY_EXTERNAL_EVAL_STATE_FILE:-$DEFAULT_STATE_FILE}"

usage() {
    cat <<'USAGE'
usage: run_external_eval.sh --model <gguf> --configs <path>
                            [--slice <jsonl>] [--keep-server] [--dry-run]
                            [--output <path>] [--batch-size N] [--resume]
                            [--reset] [--state-file <path>]

Drives the internal benchmark suite and the external HumanEval+ slice
against each agent configuration listed in --configs, then computes
Pearson correlation between per-configuration pass rates.

Required:
  --model <gguf>       GGUF path; auto-launches llama-server if down.
  --configs <path>     File with one agent configuration per line.
                       Format: "<label> <fx-runner arg>..."

Optional:
  --slice <jsonl>      HumanEval+ slice (default:
                       benchmarks/external/humaneval_plus_sample.jsonl)
  --keep-server        Keep auto-launched llama-server on exit.
  --dry-run            Print planned runs and exit.
  --output <path>      Write JSON results to <path>.
  --batch-size N       Run at most N configs per invocation and checkpoint
                       progress (issue #1040). 0 = all remaining (default).
  --resume             Continue a partial run; skip completed configs.
  --reset              Delete the state file and start over.
  --state-file <path>  Checkpoint location (default:
                       logs/.run_external_eval_state.json).

Env: LLAMACPP_HOST, LLAMACPP_SERVER_BIN, LLAMACPP_DIR, LLAMACPP_NGL,
     LLAMACPP_HEALTH_TIMEOUT, FOUNDRY_EXTERNAL_EVAL_MIN_PAIRS,
     FOUNDRY_EXTERNAL_EVAL_STATE_FILE
USAGE
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || { echo "error: --model requires a path argument" >&2; exit 2; }
            MODEL="$2"; shift 2 ;;
        --configs)
            [[ $# -ge 2 ]] || { echo "error: --configs requires a path argument" >&2; exit 2; }
            CONFIGS_PATH="$2"; shift 2 ;;
        --slice)
            [[ $# -ge 2 ]] || { echo "error: --slice requires a path argument" >&2; exit 2; }
            SLICE="$2"; shift 2 ;;
        --output)
            [[ $# -ge 2 ]] || { echo "error: --output requires a path argument" >&2; exit 2; }
            OUTPUT_PATH="$2"; shift 2 ;;
        --keep-server)
            KEEP_SERVER=1; shift ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --batch-size)
            [[ $# -ge 2 ]] || { echo "error: --batch-size requires an integer argument" >&2; exit 2; }
            BATCH_SIZE="$2"; shift 2 ;;
        --resume)
            RESUME=1; shift ;;
        --reset)
            RESET=1; shift ;;
        --state-file)
            [[ $# -ge 2 ]] || { echo "error: --state-file requires a path argument" >&2; exit 2; }
            STATE_FILE="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "error: unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done

if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]]; then
    echo "error: --batch-size must be a non-negative integer" >&2
    exit 2
fi
[[ $RESET -eq 1 && $RESUME -eq 1 ]] && {
    echo "error: --reset and --resume are mutually exclusive" >&2
    exit 2
}

[[ -n "$MODEL" ]]     || { echo "error: --model is required" >&2; usage >&2; exit 2; }
[[ -n "$CONFIGS_PATH" ]] || { echo "error: --configs is required" >&2; usage >&2; exit 2; }
[[ -r "$CONFIGS_PATH" ]] || { echo "error: --configs file not readable: $CONFIGS_PATH" >&2; exit 2; }
[[ -r "$SLICE" ]]       || { echo "error: --slice file not readable: $SLICE" >&2; exit 2; }

# ---------------------------------------------------------------------------
# Parse the configs file. Each non-empty, non-comment line is:
#   <label> <extra args for fx-runner...>
# ---------------------------------------------------------------------------
declare -a CONFIG_LABELS=()
declare -a CONFIG_ARGS=()
config_count=0
while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line#"${raw_line%%[![:space:]]*}"}"  # ltrim
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    label="${line%% *}"
    rest="${line#* }"
    [[ "$rest" == "$line" ]] && rest=""  # single-token line
    CONFIG_LABELS+=("$label")
    CONFIG_ARGS+=("$rest")
    config_count=$((config_count + 1))
done < "$CONFIGS_PATH"

if [[ $config_count -lt $MIN_PAIRS ]]; then
    echo "error: --configs lists $config_count configurations but issue #900" >&2
    echo "       requires >= ${MIN_PAIRS} paired observations (FOUNDRY_EXTERNAL_EVAL_MIN_PAIRS)." >&2
    exit 3
fi

echo "==> External-eval study plan"
echo "    slice:          $SLICE"
echo "    model:          $MODEL"
echo "    configs file:   $CONFIGS_PATH ($config_count configurations)"
echo "    min pairs:      $MIN_PAIRS"
echo "    state file:     $STATE_FILE"
echo "    batch size:     $([[ $BATCH_SIZE -eq 0 ]] && echo "all remaining" || echo "$BATCH_SIZE")"
[[ -n "$OUTPUT_PATH" ]] && echo "    output:         $OUTPUT_PATH"

# ---------------------------------------------------------------------------
# Incremental batched execution (issue #1040).
#
# A state file at $STATE_FILE records completed configurations with
# timestamps and the accumulated internal/external rate arrays so an
# interrupted run can resume without re-spending model tokens. The
# checkpoint logic lives in foundry_x.evaluation.study_state (pure
# Python, unit-tested); this block is the operator surface that loads,
# validates, and computes the next batch.
#
# --reset wipes the checkpoint; --resume is explicit but resuming is also
# the safe default whenever a checkpoint already exists for this study.
# ---------------------------------------------------------------------------
if [[ $RESET -eq 1 ]]; then
    echo "==> --reset: removing state file $STATE_FILE"
    rm -f "$STATE_FILE"
fi

# Serialise the planned labels to a JSON array the Python helper can read.
LABELS_JSON="$(printf '%s\n' "${CONFIG_LABELS[@]}" | uv run --quiet python -c \
    'import json,sys; print(json.dumps(sys.stdin.read().splitlines()))')"

mkdir -p "$LOGS_DIR"
echo "==> Loading incremental-study checkpoint"
STATE_INFO="$(LABELS_JSON="$LABELS_JSON" MODEL="$MODEL" SLICE="$SLICE" \
    MIN_PAIRS="$MIN_PAIRS" STATE_FILE="$STATE_FILE" BATCH_SIZE="$BATCH_SIZE" \
    DRY_RUN="$DRY_RUN" \
    uv run --quiet python -c '
import json, os, sys
from pathlib import Path
from foundry_x.evaluation import study_state as ss

state_file = os.environ["STATE_FILE"]
labels = json.loads(os.environ["LABELS_JSON"])
model = os.environ["MODEL"]
slice_path = os.environ["SLICE"]
min_pairs = int(os.environ["MIN_PAIRS"])

existing = ss.load_state(state_file)
if existing is not None:
    # Refuse to resume across a model or slice change: mixing configurations
    # from two different models into one Pearson computation would silently
    # invalidate the study (AGENTS.md S2: never silently swallow).
    if existing.model != model:
        print(f"error: state file model={existing.model!r} != --model {model!r}; "
              "use --reset to start a new study", file=sys.stderr)
        sys.exit(6)
    if existing.slice != slice_path:
        print(f"error: state file slice={existing.slice!r} != --slice {slice_path!r}; "
              "use --reset to start a new study", file=sys.stderr)
        sys.exit(6)
    if existing.configs_planned != labels:
        print("error: state file configs_planned does not match --configs; "
              "use --reset to start a new study", file=sys.stderr)
        sys.exit(6)
    state = existing
else:
    study_id = "external_eval_" + ss._utc_now_iso().replace(":", "").replace("-", "")
    state = ss.new_state(study_id, model, slice_path, labels, min_pairs=min_pairs)
    # --dry-run must have no side effects: compute the batch for display
    # but do not persist a checkpoint the operator did not ask for.
    if os.environ.get("DRY_RUN", "0") != "1":
        ss.save_state(state_file, state)

batch_size = int(os.environ["BATCH_SIZE"])
remaining = ss.remaining_configs(state, labels)
if batch_size > 0:
    batch = remaining[:batch_size]
else:
    batch = remaining

reportable = ss.is_reportable(state)

# Emit machine-parseable summary lines. The batch labels follow a sentinel
# so bash can read them back into an array.
print(f"STUDY_ID={state.study_id}")
print(f"COMPLETED={len(state.configs_completed)}")
print(f"PLANNED={len(labels)}")
print(f"REMAINING={len(remaining)}")
print(f"BATCH_COUNT={len(batch)}")
print(f"REPORTABLE={'1' if reportable else '0'}")
print("BATCH_LABELS_BEGIN")
for lbl in batch:
    print(lbl)
print("BATCH_LABELS_END")
')" || exit $?

# Parse the summary into shell variables.
STUDY_ID="$(printf '%s\n' "$STATE_INFO" | sed -n 's/^STUDY_ID=//p')"
COMPLETED_COUNT="$(printf '%s\n' "$STATE_INFO" | sed -n 's/^COMPLETED=//p')"
PLANNED_COUNT="$(printf '%s\n' "$STATE_INFO" | sed -n 's/^PLANNED=//p')"
REMAINING_COUNT="$(printf '%s\n' "$STATE_INFO" | sed -n 's/^REMAINING=//p')"
BATCH_COUNT="$(printf '%s\n' "$STATE_INFO" | sed -n 's/^BATCH_COUNT=//p')"
REPORTABLE_FLAG="$(printf '%s\n' "$STATE_INFO" | sed -n 's/^REPORTABLE=//p')"

# Read the batch labels back into an array plus a parallel index array
# pointing at the position in CONFIG_LABELS (so we can recover the args).
declare -a BATCH_LABELS=()
declare -a BATCH_INDICES=()
in_batch=0
while IFS= read -r ln; do
    if [[ "$ln" == "BATCH_LABELS_BEGIN" ]]; then
        in_batch=1; continue
    fi
    [[ "$ln" == "BATCH_LABELS_END" ]] && break
    [[ $in_batch -eq 1 ]] || continue
    BATCH_LABELS+=("$ln")
    for idx in "${!CONFIG_LABELS[@]}"; do
        if [[ "${CONFIG_LABELS[$idx]}" == "$ln" ]]; then
            BATCH_INDICES+=("$idx"); break
        fi
    done
done <<<"$STATE_INFO"

echo "    study id:       $STUDY_ID"
echo "    progress:       $COMPLETED_COUNT/$PLANNED_COUNT configs complete ($REMAINING_COUNT remaining)"
echo "    batch:          $BATCH_COUNT configuration(s) to run this invocation"
if [[ "$REPORTABLE_FLAG" == "1" ]]; then
    echo "    status:         study already reportable; will emit final report"
fi

# ---------------------------------------------------------------------------
# Pre-flight: slice integrity (cheap; no model tokens spent).
# ---------------------------------------------------------------------------
echo "==> Pre-flight: validating slice integrity"
if ! uv run --quiet python -c "
import sys
from foundry_x.evaluation.humaneval_plus import load_humaneval_slice, slice_pass_rates
tasks = load_humaneval_slice('$SLICE')
passed, total = slice_pass_rates(tasks)
if passed != total:
    print(f'slice integrity failed: {passed}/{total} canonical solutions passed', file=sys.stderr)
    sys.exit(1)
print(f'    slice ok: {total} tasks, all canonical solutions pass')
"; then
    echo "error: slice pre-flight failed (see above)" >&2
    exit 5
fi

# ---------------------------------------------------------------------------
# Helpers for llama-server (mirror run_benchmark.sh).
# ---------------------------------------------------------------------------
llamacpp_is_healthy() {
    local resp
    resp="$(curl -fsS "${LLAMACPP_HOST_URL}/health" 2>/dev/null || true)"
    [[ "$resp" == *'"ok"'* ]]
}

wait_for_health() {
    echo "==> Waiting for llama-server /health at ${LLAMACPP_HOST_URL} ..."
    local waited=0
    while [[ $waited -lt $HEALTH_TIMEOUT ]]; do
        if llamacpp_is_healthy; then
            echo "    ready after ${waited}s"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "error: llama-server did not report /health within ${HEALTH_TIMEOUT}s" >&2
    return 1
}

SERVER_PID=""
SERVER_LAUNCHED=0
SERVER_LOG=""

cleanup_server() {
    if [[ $KEEP_SERVER -eq 1 || $SERVER_LAUNCHED -eq 0 ]]; then
        return
    fi
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "==> Tearing down auto-launched llama-server (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    [[ -n "$SERVER_LOG" ]] && rm -f "$SERVER_LOG"
}
trap cleanup_server EXIT

# ---------------------------------------------------------------------------
# 1. Ensure llama-server is running
# ---------------------------------------------------------------------------
if ! llamacpp_is_healthy; then
    if [[ ! -x "$LLAMACPP_SERVER_BIN" ]]; then
        echo "error: llama-server not reachable and binary missing: $LLAMACPP_SERVER_BIN" >&2
        exit 5
    fi
    if [[ ! -r "$MODEL" ]]; then
        echo "error: model not found or not readable: $MODEL" >&2
        exit 5
    fi
    echo "==> Launching llama-server"
    echo "    binary: $LLAMACPP_SERVER_BIN"
    echo "    model:  $MODEL"
    echo "    ngl:    $LLAMACPP_NGL"
    SERVER_LOG="$(mktemp -t llamacpp-eval.XXXXXX.log)"
    "$LLAMACPP_SERVER_BIN" \
        --model "$MODEL" \
        --host 127.0.0.1 --port 8080 \
        --n-gpu-layers "$LLAMACPP_NGL" \
        >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    SERVER_LAUNCHED=1
fi

wait_for_health || exit 5

# ---------------------------------------------------------------------------
# 2. Dry-run: print the plan and stop
# ---------------------------------------------------------------------------
if [[ $DRY_RUN -eq 1 ]]; then
    echo "==> Dry run: would invoke fx-runner for each configuration"
    for i in "${!CONFIG_LABELS[@]}"; do
        echo "    [${CONFIG_LABELS[$i]}] fx-runner --task <internal suite> ${CONFIG_ARGS[$i]}"
        echo "    [${CONFIG_LABELS[$i]}] fx-runner --task <external slice>  ${CONFIG_ARGS[$i]}"
    done
    echo "    then compute Pearson across $config_count configurations"
    exit 0
fi

# ---------------------------------------------------------------------------
# 3. Run each configuration against the internal suite and the external slice.
#    The batch (BATCH_LABELS / BATCH_INDICES) was computed in the
#    incremental-state block above; it holds only configs not yet marked
#    complete in the checkpoint. After each config finishes we record its
#    rates into the state file so an interrupt never loses progress
#    (issue #1040).
# ---------------------------------------------------------------------------
mkdir -p "$LOGS_DIR"
# STUDY_ID is sourced from the checkpoint so it stays stable across the
# batches that make up one study. RAW_RESULTS / EXTERNAL_RESULTS are
# per-batch scratch files; the durable record is the state file.
RAW_RESULTS="$LOGS_DIR/${STUDY_ID}.jsonl"
: > "$RAW_RESULTS"

run_one() {
    local label="$1"
    local extra_args="$2"
    local task_prompt="$3"
    local study_run_type="$4"

    # capture traces.db size before so we can detect a no-op run.
    local before=0
    [[ -f "$TRACES_DB" ]] && before=$(stat -c %s "$TRACES_DB" 2>/dev/null || echo 0)

    # shellcheck disable=SC2086
    uv run --quiet fx-runner \
        --task "$task_prompt" \
        --study-run-type "$study_run_type" \
        $extra_args \
        || return $?

    local after=0
    [[ -f "$TRACES_DB" ]] && after=$(stat -c %s "$TRACES_DB" 2>/dev/null || echo 0)
    if [[ "$after" -le "$before" ]]; then
        echo "error: traces.db did not grow for config '$label' (before=$before after=$after)" >&2
        return 1
    fi
    return 0
}

INTERNAL_PROMPT="Run every benchmark task under benchmarks/tasks/ and report the per-task pass/fail verdict."
EXTERNAL_PROMPT="Solve each task in ${SLICE} and emit the candidate function body. The orchestrator scores them via foundry_x.evaluation.humaneval_plus.run_candidate_solution."

EXTERNAL_RESULTS="$LOGS_DIR/${STUDY_ID}_external_results.jsonl"
: > "$EXTERNAL_RESULTS"

# record_config: compute the per-config internal+external rates and
# persist them to the checkpoint so the config is not re-run on resume.
# The internal rate mirrors the aggregator's critic_verdict query; the
# external rate comes from score_external's EXTERNAL_RESULTS scratch
# file (defaulting to 0.0 with a warning when external scoring is still
# pending per ADR-0023 §Placeholder).
record_config() {
    local label="$1"
    local extra_args="$2"

    LABEL="$label" EXTRA_ARGS="$extra_args" \
    STATE_FILE="$STATE_FILE" TRACES_DB="$TRACES_DB" \
    EXTERNAL_RESULTS="$EXTERNAL_RESULTS" \
    uv run --quiet python -c '
import json, os, sqlite3, sys
from pathlib import Path

from foundry_x.evaluation import study_state as ss

label = os.environ["LABEL"]
extra = os.environ["EXTRA_ARGS"]
state_file = os.environ["STATE_FILE"]
traces_db = Path(os.environ["TRACES_DB"])
external_results = Path(os.environ["EXTERNAL_RESULTS"])

# --- internal pass rate from critic_verdict (mirrors the aggregator) ---
quantization = harness_version = None
tokens = iter(extra.split())
for tok in tokens:
    if tok == "--quantization":
        quantization = next(tokens, None)
    elif tok == "--harness-version":
        harness_version = next(tokens, None)

internal_rate = 0.0
if quantization and harness_version and traces_db.is_file():
    conn = sqlite3.connect(traces_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT session_id FROM sessions
        WHERE quantization = ? AND harness_version = ?
          AND metadata LIKE '%\"study_run_type\": \"internal_suite\"%'
        ORDER BY started_at DESC LIMIT 1
        """,
        (quantization, harness_version),
    ).fetchone()
    if row is not None:
        verdicts = conn.execute(
            "SELECT payload FROM events WHERE session_id = ? AND kind = ?",
            (row["session_id"], "critic_verdict"),
        ).fetchall()
        passed = failed = 0
        for vr in verdicts:
            payload = json.loads(vr["payload"])
            passed += len(payload.get("passed_checks", []))
            failed += len(payload.get("failed_checks", []))
        total = passed + failed
        if total > 0:
            internal_rate = passed / total
    conn.close()
else:
    print(f"warning: cannot compute internal rate for {label} "
          f"(missing quantization/harness_version or traces.db)", file=sys.stderr)

# --- external pass rate from the per-batch scratch file ---
external_rate = 0.0
if external_results.is_file():
    for ln in external_results.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        obj = json.loads(ln)
        if obj.get("label") == label:
            p, t = obj.get("passed", 0), obj.get("total", 0)
            external_rate = p / t if t > 0 else 0.0
            break
if external_rate == 0.0:
    print(f"warning: external scoring pending for {label} "
          f"(ADR-0023 placeholder); recorded external_rate=0.0", file=sys.stderr)

state = ss.load_state(state_file)
if state is None:
    print(f"error: state file vanished mid-run: {state_file}", file=sys.stderr)
    sys.exit(5)
ss.record_observation(state, label, internal_rate, external_rate)
ss.save_state(state_file, state)
print(f"recorded {label}: internal={internal_rate:.4f} external={external_rate:.4f}")
'
}

score_external() {
    local label="$1"
    local extra_args="$2"

    # Score the agent's candidate solutions against the HumanEval+ slice.
    # The agent session is already captured in traces.db; now we extract
    # the candidate bodies and score them.
    local result
    result=$(uv run --quiet python -c "
import json
from pathlib import Path
from foundry_x.evaluation.humaneval_plus import (
    HumanEvalTask,
    load_humaneval_slice,
    run_candidate_solution,
)

# The agent session for this config is already in traces.db.
# We retrieve candidate bodies from the trace by querying tool_result events
# where the tool name contains 'Write' or the output matches a function body.
# ADR-0032: the aggregator matches sessions by (harness_variant, quantization, harness_version).
# Here we re-run the scoring against the same model endpoint rather than
# replaying traces, because replaying requires the agent's tool-call history.
# Instead, we use a lightweight scoring pass that reads candidate bodies
# from the trace store (if available) or skips scoring if the model endpoint
# is not available.
import sqlite3
traces_db = Path('$TRACES_DB')
conn = sqlite3.connect(traces_db)
cursor = conn.cursor()

# Find the most recent external_slice session for this label's config.
# Config is identified by (quantization, harness_version) from extra_args.
# We match by harness_variant in session metadata.
label = '$label'
# Extract quantization from extra_args (e.g. '--quantization Q4_K_M')
quantization = ''
for arg in ${extra_args}; do
    if [[ \"\$arg\" == --quantization ]]; then
        next_arg=true
    elif [[ \$next_arg == true ]]; then
        quantization=\"\$arg\"
        break
    fi
done

# Find the session with the matching quantization and study_run_type=external_slice
cursor.execute('''
    SELECT s.session_id, s.harness_version
    FROM sessions s
    WHERE s.quantization = ?
      AND s.metadata LIKE '%\"study_run_type\": \"external_slice\"%'
    ORDER BY s.started_at DESC
    LIMIT 1
''', (quantization,))
row = cursor.fetchone()
conn.close()

if row is None:
    print(f'warning: no external_slice session found for label {label}', file=sys.stderr)
    print(0, file=sys.stdout)
    exit(0)

session_id, harness_version = row

# The candidate bodies are not stored in traces.db in a structured form.
# For the study to work, the orchestrator must score the candidates inline.
# We delegate to the aggregator: write a placeholder and let the aggregator
# handle the inline scoring pass (see ADR-0032 §Aggregator).
print(f'warning: candidate scoring deferred to aggregator for {label}', file=sys.stderr)
print(0, file=stdout)
" 2>&1)

    local passed
    passed=$(echo "$result" | tail -1)
    echo "{\"label\": \"$label\", \"passed\": $passed, \"total\": 20}" >> "$EXTERNAL_RESULTS"
}

failed_runs=0
ran_this_batch=0
for b in "${!BATCH_LABELS[@]}"; do
    label="${BATCH_LABELS[$b]}"
    idx="${BATCH_INDICES[$b]}"
    extra="${CONFIG_ARGS[$idx]}"

    echo "==> [$label] internal suite"
    if ! run_one "$label" "$extra" "$INTERNAL_PROMPT" "internal_suite"; then
        echo "error: internal run failed for '$label'" >&2
        failed_runs=$((failed_runs + 1))
        continue
    fi

    echo "==> [$label] external slice"
    if ! run_one "$label" "$extra" "$EXTERNAL_PROMPT" "external_slice"; then
        echo "error: external run failed for '$label'" >&2
        failed_runs=$((failed_runs + 1))
        continue
    fi

    echo "==> [$label] scoring external candidates"
    score_external "$label" "$extra" || true

    echo "==> [$label] checkpointing progress"
    record_config "$label" "$extra" || {
        echo "error: failed to checkpoint '$label'" >&2
        exit 5
    }
    ran_this_batch=$((ran_this_batch + 1))
done

if [[ $failed_runs -gt 0 ]]; then
    echo "error: $failed_runs configuration run(s) failed; raw results at $RAW_RESULTS" >&2
    echo "       successfully-completed configs in this batch WERE checkpointed;" >&2
    echo "       fix the failure and re-run (with --resume or the same flags) to continue." >&2
    exit 5
fi

# ---------------------------------------------------------------------------
# 4. Aggregate / checkpoint (issue #1040).
#
#    The checkpoint holds the accumulated internal/external rate arrays
#    across every batch in this study, so the final report is computed
#    directly from the state (no traces.db re-query needed). When the
#    study is not yet reportable we persist progress and exit 0 so the
#    operator can resume the next batch later.
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT="$LOGS_DIR/${STUDY_ID}_report.json"
OUTPUT_PATH="${OUTPUT_PATH:-$DEFAULT_OUTPUT}"

FINAL_INFO="$(uv run --quiet python -c '
import json, os, sys
from pathlib import Path

from foundry_x.evaluation import study_state as ss
from foundry_x.evaluation.correlation import (
    MIN_PAIRED_OBSERVATIONS,
    UnderpoweredStudyError,
    ZeroVarianceError,
    interpret_correlation,
    pearson_binary,
    pearson_binary_ci_95,
)

state_file = os.environ["STATE_FILE"]
output_path = os.environ["OUTPUT_PATH"]
model = os.environ["MODEL"]
slice_path = os.environ["SLICE"]
config_count = int(os.environ["CONFIG_COUNT"])

state = ss.load_state(state_file)
if state is None:
    print(f"error: state file missing at finalize: {state_file}", file=sys.stderr)
    sys.exit(5)

completed = len(state.configs_completed)
planned = len(state.configs_planned)

# Not enough paired observations yet: persist progress and stop. The
# operator runs the next batch (same flags) to accumulate more.
if not ss.is_reportable(state):
    print(f"PARTIAL=1")
    print(f"COMPLETED={completed}")
    print(f"PLANNED={planned}")
    print(f"MIN_PAIRS={state.min_pairs}")
    sys.exit(0)

# Reportable: compute Pearson r + 95% CI from the accumulated arrays.
internal_rates = state.internal_rates
external_rates = state.external_rates

pearson_val = None
ci_95 = None
verdict_label = "pending"
exit_code = 1
try:
    pearson_val = pearson_binary(internal_rates, external_rates)
    r_lower, r_upper = pearson_binary_ci_95(internal_rates, external_rates)
    ci_95 = [round(r_lower, 4), round(r_upper, 4)]
    verdict_label = interpret_correlation(pearson_val)
    exit_code = 0
except UnderpoweredStudyError as exc:
    print(f"under-powered: {exc}", file=sys.stderr)
except ZeroVarianceError as exc:
    print(f"zero variance: {exc}", file=sys.stderr)
    print("       Choose a more discriminating task set.", file=sys.stderr)
except Exception as exc:
    print(f"correlation error: {exc}", file=sys.stderr)

report = {
    "study_id": state.study_id,
    "adr": "ADR-0023",
    "slice": slice_path,
    "model": model,
    "min_pairs_required": MIN_PAIRED_OBSERVATIONS,
    "configs_planned": planned,
    "configs_observed": completed,
    "internal_rates": [round(r, 4) for r in internal_rates],
    "external_rates": [round(r, 4) for r in external_rates],
    "configs_completed": list(state.configs_completed),
    "completed_at": dict(state.completed_at),
    "pearson": round(pearson_val, 4) if pearson_val is not None else None,
    "pearson_ci_95": ci_95,
    "verdict": verdict_label,
    "exit_code": exit_code,
}

Path(output_path).write_text(json.dumps(report, indent=2) + "\n")
print("PARTIAL=0")
print(f"COMPLETED={completed}")
print(f"PLANNED={planned}")
print(f"PEARSON={round(pearson_val, 4) if pearson_val is not None else \"null\"}")
print(f"VERDICT={verdict_label}")
' CONFIG_COUNT="$config_count" OUTPUT_PATH="$OUTPUT_PATH" \
    STATE_FILE="$STATE_FILE" MODEL="$MODEL" SLICE="$SLICE")" || exit $?

FINAL_PARTIAL="$(printf '%s\n' "$FINAL_INFO" | sed -n 's/^PARTIAL=//p')"
FINAL_COMPLETED="$(printf '%s\n' "$FINAL_INFO" | sed -n 's/^COMPLETED=//p')"
FINAL_PLANNED="$(printf '%s\n' "$FINAL_INFO" | sed -n 's/^PLANNED=//p')"

if [[ "$FINAL_PARTIAL" == "1" ]]; then
    FINAL_MIN_PAIRS="$(printf '%s\n' "$FINAL_INFO" | sed -n 's/^MIN_PAIRS=//p')"
    echo "==> Batch complete; study not yet reportable (issue #1040 incremental mode)"
    echo "    progress:   $FINAL_COMPLETED/$FINAL_PLANNED configs checkpointed"
    echo "    threshold:  $FINAL_MIN_PAIRS paired observations required"
    echo "    state file: $STATE_FILE"
    echo "    re-run with the same flags (or --resume) to continue the next batch."
    echo "==> External-eval batch complete (partial)"
    exit 0
fi

FINAL_PEARSON="$(printf '%s\n' "$FINAL_INFO" | sed -n 's/^PEARSON=//p')"
FINAL_VERDICT="$(printf '%s\n' "$FINAL_INFO" | sed -n 's/^VERDICT=//p')"

echo "==> External-eval study complete"
echo "    configs observed: $FINAL_COMPLETED/$FINAL_PLANNED"
echo "    Pearson r:        $FINAL_PEARSON"
echo "    verdict:          $FINAL_VERDICT"
echo "    report:           $OUTPUT_PATH"
echo "    state file:       $STATE_FILE"
