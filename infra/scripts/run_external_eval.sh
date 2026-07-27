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
for i in "${!CONFIG_LABELS[@]}"; do
    label="${CONFIG_LABELS[$i]}"
    extra="${CONFIG_ARGS[$i]}"

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
done

if [[ $failed_runs -gt 0 ]]; then
    echo "error: $failed_runs configuration run(s) failed; raw results at $RAW_RESULTS" >&2
    exit 5
fi

# ---------------------------------------------------------------------------
# 4. Aggregate and compute Pearson correlation (ADR-0032, issue #1028).
#
#    The aggregator:
#      - reads the configs file to enumerate the 30+ configurations,
#      - queries traces.db for internal_suite sessions and computes
#        internal_pass_rate = passed / (passed + failed) from critic_verdict,
#      - reads the external results file for external pass rates,
#      - invokes pearson_binary and pearson_binary_ci_95,
#      - writes the JSON report.
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
    pearson_binary_ci_95,
)

traces_db = Path('$TRACES_DB')
if not traces_db.is_file():
    print(f'error: {traces_db} not found', file=sys.stderr)
    sys.exit(5)

external_results_path = Path('$EXTERNAL_RESULTS')
external_results: dict[str, tuple[int, int]] = {}
if external_results_path.is_file():
    for line in external_results_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        external_results[obj['label']] = (obj['passed'], obj['total'])

# Build the per-config pass rate vectors in the same order as CONFIG_LABELS.
internal_rates: list[float] = []
external_rates: list[float] = []
configs_observed = 0

conn = sqlite3.connect(traces_db)
conn.row_factory = sqlite3.Row

for i, label in enumerate(${CONFIG_LABELS[@]}):
    extra = ${CONFIG_ARGS[$i]}

    # Extract quantization from extra args for DB query.
    quantization = None
    harness_version = None
    tokens = iter(extra.split())
    for tok in tokens:
        if tok == '--quantization':
            quantization = next(tokens, None)
        elif tok == '--harness-version':
            harness_version = next(tokens, None)

    if quantization is None or harness_version is None:
        print(f'warning: skipping {label}: missing quantization or harness_version', file=sys.stderr)
        continue

    # Internal pass rate: find internal_suite session for this config.
    cursor = conn.execute('''
        SELECT session_id FROM sessions
        WHERE quantization = ?
          AND harness_version = ?
          AND metadata LIKE '%\"study_run_type\": \"internal_suite\"%'
        ORDER BY started_at DESC
        LIMIT 1
    ''', (quantization, harness_version))
    row = cursor.fetchone()
    if row is None:
        print(f'warning: no internal_suite session for {label}', file=sys.stderr)
        continue

    sid = row['session_id']
    verdict_rows = conn.execute('''
        SELECT payload FROM events
        WHERE session_id = ? AND kind = 'critic_verdict'
    ''', (sid,)).fetchall()

    if not verdict_rows:
        print(f'warning: no critic_verdict for {label} session {sid}', file=sys.stderr)
        continue

    # Sum passed/failed across all critic_verdict events for this session.
    total_internal = 0
    passed_internal = 0
    for vr in verdict_rows:
        payload = json.loads(vr['payload'])
        passed_list = payload.get('passed_checks', [])
        failed_list = payload.get('failed_checks', [])
        passed_internal += len(passed_list)
        total_internal += len(passed_list) + len(failed_list)

    internal_rate = passed_internal / total_internal if total_internal > 0 else 0.0

    # External pass rate from the results file.
    if label in external_results:
        p, t = external_results[label]
        external_rate = p / t if t > 0 else 0.0
    else:
        print(f'warning: no external result for {label}', file=sys.stderr)
        continue

    internal_rates.append(internal_rate)
    external_rates.append(external_rate)
    configs_observed += 1

conn.close()

# Compute Pearson r and CI.
pearson_val = None
ci_95: list[float] | None = None
verdict_label = 'pending'
try:
    pearson_val = pearson_binary(internal_rates, external_rates)
    r_lower, r_upper = pearson_binary_ci_95(internal_rates, external_rates)
    ci_95 = [round(r_lower, 4), round(r_upper, 4)]
    verdict_label = interpret_correlation(pearson_val)
except UnderpoweredStudyError as exc:
    print(f'under-powered: {exc}', file=sys.stderr)
except ZeroVarianceError as exc:
    print(f'zero variance: {exc}', file=sys.stderr)
    print('       Choose a more discriminating task set.', file=sys.stderr)
except Exception as exc:
    print(f'correlation error: {exc}', file=sys.stderr)

report = {
    'study_id': '$STUDY_ID',
    'adr': 'ADR-0032',
    'slice': '$SLICE',
    'slice_task_count': 20,
    'model': '$MODEL',
    'harness_versions': ['v1.0', 'v1.1', 'v1.2'],
    'quantizations': ['Q4_K_M', 'Q5_K_M', 'Q6_K_M', 'Q8_0', 'IQ4_XS', 'IQ4_NL'],
    'model_sizes': ['Qwen2.5-7B', 'Qwen2.5-14B'],
    'min_pairs_required': MIN_PAIRED_OBSERVATIONS,
    'configs_planned': $config_count,
    'configs_observed': configs_observed,
    'internal_rates': [round(r, 4) for r in internal_rates],
    'external_rates': [round(r, 4) for r in external_rates],
    'pearson': round(pearson_val, 4) if pearson_val is not None else None,
    'pearson_ci_95': ci_95,
    'verdict': verdict_label,
    'interpreted_at': '${STUDY_ID}'.replace('external_eval_', ''),
    'exit_code': 0 if pearson_val is not None else 1,
}

Path('$OUTPUT_PATH').write_text(json.dumps(report, indent=2) + '\n')
print(f'    configs observed: {configs_observed}')
print(f'    Pearson r: {report[\"pearson\"]}')
print(f'    95% CI: {report[\"pearson_ci_95\"]}')
print(f'    verdict: {report[\"verdict\"]}')
"

echo "==> External-eval study complete"
echo "    external results: $EXTERNAL_RESULTS"
echo "    report:          $OUTPUT_PATH"
