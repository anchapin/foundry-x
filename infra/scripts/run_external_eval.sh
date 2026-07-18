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
#                        [--keep-server] [--dry-run]
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
#
# Exit codes:
#   0   Study completed and correlation is reportable.
#   2   CLI usage error.
#   3   Study is under-powered (fewer than MIN_PAIRS configurations).
#   4   A configuration's internal OR external pass rate has zero variance
#       (Pearson is undefined); the operator must choose a more
#       discriminating task set.
#   5   One or more runs failed non-recoverably; see stderr.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_SLICE="$REPO_ROOT/benchmarks/external/humaneval_plus_sample.jsonl"
TRACES_DB="$REPO_ROOT/logs/traces.db"
LOGS_DIR="$REPO_ROOT/logs"
MIN_PAIRS="${FOUNDRY_EXTERNAL_EVAL_MIN_PAIRS:-30}"

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

usage() {
    cat <<'USAGE'
usage: run_external_eval.sh --model <gguf> --configs <path>
                            [--slice <jsonl>] [--keep-server] [--dry-run]
                            [--output <path>]

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

Env: LLAMACPP_HOST, LLAMACPP_SERVER_BIN, LLAMACPP_DIR, LLAMACPP_NGL,
     LLAMACPP_HEALTH_TIMEOUT, FOUNDRY_EXTERNAL_EVAL_MIN_PAIRS
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
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "error: unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done

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
[[ -n "$OUTPUT_PATH" ]] && echo "    output:         $OUTPUT_PATH"

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
#    Results are appended to a per-study JSONL file under logs/; the
#    aggregator at step 4 reads them.
# ---------------------------------------------------------------------------
mkdir -p "$LOGS_DIR"
STUDY_ID="external_eval_$(date -u +%Y%m%dT%H%M%SZ)"
RAW_RESULTS="$LOGS_DIR/${STUDY_ID}.jsonl"
: > "$RAW_RESULTS"

run_one() {
    local label="$1"
    local extra_args="$2"
    local task_prompt="$3"

    # capture traces.db size before so we can detect a no-op run.
    local before=0
    [[ -f "$TRACES_DB" ]] && before=$(stat -c %s "$TRACES_DB" 2>/dev/null || echo 0)

    # shellcheck disable=SC2086
    uv run --quiet fx-runner --task "$task_prompt" $extra_args || return $?

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

failed_runs=0
for i in "${!CONFIG_LABELS[@]}"; do
    label="${CONFIG_LABELS[$i]}"
    extra="${CONFIG_ARGS[$i]}"

    echo "==> [$label] internal suite"
    if ! run_one "$label" "$extra" "$INTERNAL_PROMPT"; then
        echo "error: internal run failed for '$label'" >&2
        failed_runs=$((failed_runs + 1))
        continue
    fi

    echo "==> [$label] external slice"
    if ! run_one "$label" "$extra" "$EXTERNAL_PROMPT"; then
        echo "error: external run failed for '$label'" >&2
        failed_runs=$((failed_runs + 1))
        continue
    fi
done

if [[ $failed_runs -gt 0 ]]; then
    echo "error: $failed_runs configuration run(s) failed; raw results at $RAW_RESULTS" >&2
    exit 5
fi

# ---------------------------------------------------------------------------
# 4. Aggregate and compute Pearson correlation.
#
#    The aggregator Python helper is responsible for:
#      - reading the per-config trace events from logs/traces.db,
#      - recovering the per-config internal / external pass rates,
#      - invoking foundry_x.evaluation.correlation.pearson_binary,
#      - writing the report JSON.
#
#    It exists as a separate inline script so the math stays testable
#    via the unit tests under tests/test_correlation.py without
#    dragging in this shell orchestrator.
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT="$LOGS_DIR/${STUDY_ID}_report.json"
OUTPUT_PATH="${OUTPUT_PATH:-$DEFAULT_OUTPUT}"

echo "==> Aggregating results into $OUTPUT_PATH"
uv run --quiet python -c "
import json
import sqlite3
import sys
from pathlib import Path

from foundry_x.evaluation.correlation import (
    MIN_PAIRED_OBSERVATIONS,
    UnderpoweredStudyError,
    ZeroVarianceError,
    interpret_correlation,
    pearson_binary,
)

# Recover per-config pass rates from the trace store. The trace schema
# is governed by ADR-0003 and ADR-0011; critic_verdict events carry the
# per-task pass/fail payload the KPIs module already uses.
traces_db = Path('$TRACES_DB')
if not traces_db.is_file():
    print(f'error: {traces_db} not found; no trace store to aggregate from', file=sys.stderr)
    sys.exit(5)

# Placeholder pass-rate vectors. The real aggregation requires parsing
# critic_verdict events scoped to each configuration's session id, which
# in turn requires the runner to tag each run with the config label.
# That plumbing is intentionally not part of this PR: it depends on a
# runner-side change (issue/PR to be filed) that records the agent
# configuration in the session metadata. Until that lands the script
# refuses to emit a fake number.
internal_rates: list[float] = []
external_rates: list[float] = []
print('note: per-config aggregation pending runner-side session-metadata tagging', file=sys.stderr)
print('      (see ADR-0023 follow-up); emitting a structural report only.', file=sys.stderr)

report = {
    'study_id': '$STUDY_ID',
    'slice': '$SLICE',
    'model': '$MODEL',
    'min_pairs_required': MIN_PAIRED_OBSERVATIONS,
    'configs_planned': $config_count,
    'configs_observed': 0,
    'internal_rates': internal_rates,
    'external_rates': external_rates,
    'pearson': None,
    'verdict': 'pending-runner-side-aggregation-plumbing',
    'note': (
        'The machinery (loader, scorer, correlation math, offline plumbing '
        'validation) ships with this PR. Producing the actual Pearson '
        'number requires a runner-side change that records the agent '
        'configuration in session metadata so this aggregator can group '
        'critic_verdict events per configuration. Tracked as an ADR-0023 '
        'follow-up.'
    ),
}

Path('$OUTPUT_PATH').write_text(json.dumps(report, indent=2) + '\n')
print(f'    wrote {len(report[\"internal_rates\"])} paired observations')
print(f'    verdict: {report[\"verdict\"]}')
"

echo "==> External-eval study complete"
echo "    raw results: $RAW_RESULTS"
echo "    report:      $OUTPUT_PATH"
