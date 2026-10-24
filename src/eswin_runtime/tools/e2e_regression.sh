#!/usr/bin/env bash
# Milestone 10 deliverable: automated performance and perplexity regression harness.
#
# Runs tests/test_deepseek_e2e in benchmark mode against a real model package, parses its
# machine-readable PERF/PPL lines, and compares them against a recorded baseline. Fails on
# a throughput regression or a perplexity increase beyond the configured tolerance.
#
#   tools/e2e_regression.sh --record                 # store the current numbers as baseline
#   tools/e2e_regression.sh                          # compare against the baseline
#   tools/e2e_regression.sh --host roma --model-dir /home/chri/models/qwen2_5_coder
#   tools/e2e_regression.sh --tolerance 10
#
# Off-board runs are allowed but only exercise host-fallback dummy weights, so their
# perplexity is meaningless (uniform over the vocab). Use --host for real numbers.
set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
CROSS_COMPILE="${CROSS_COMPILE:-}"
HOST=""
MODEL_DIR="${ES_E2E_MODEL_DIR:-}"
BASELINE="$RUNTIME_DIR/tests/e2e_baseline.txt"
TOLERANCE=5          # percent
RECORD=0
REMOTE_DIR=/tmp/eswin_e2e

usage() {
    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)       HOST="$2"; shift 2 ;;
        --model-dir)  MODEL_DIR="$2"; shift 2 ;;
        --baseline)   BASELINE="$2"; shift 2 ;;
        --tolerance)  TOLERANCE="$2"; shift 2 ;;
        --build-dir)  BUILD_DIR="$2"; shift 2 ;;
        --record)     RECORD=1; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

cd "$RUNTIME_DIR"

echo "== Building test_deepseek_e2e ${CROSS_COMPILE:+(cross: $CROSS_COMPILE)}}"
make ${CROSS_COMPILE:+CROSS_COMPILE=$CROSS_COMPILE} BUILD_DIR="$BUILD_DIR" \
    "$BUILD_DIR/test_deepseek_e2e" >/dev/null

run_bench() {
    if [[ -n "$HOST" ]]; then
        echo "== Deploying to $HOST:$REMOTE_DIR"
        ssh "$HOST" "mkdir -p $REMOTE_DIR"
        scp -q "$BUILD_DIR/test_deepseek_e2e" "$BUILD_DIR/libes_runtime.so" "$HOST:$REMOTE_DIR/"
        echo "== Running on $HOST against $MODEL_DIR"
        ssh "$HOST" "cd $REMOTE_DIR && LD_LIBRARY_PATH=$REMOTE_DIR:/lib:/usr/lib \
            ./test_deepseek_e2e '$MODEL_DIR'"
    else
        [[ -n "$MODEL_DIR" ]] || { echo "no --model-dir and no --host: nothing to benchmark" >&2; exit 1; }
        echo "== Running locally against $MODEL_DIR"
        LD_LIBRARY_PATH="$BUILD_DIR":/lib:/usr/lib "$BUILD_DIR/test_deepseek_e2e" "$MODEL_DIR"
    fi
}

OUT="$(run_bench)"
echo "$OUT"

PERF_LINE="$(printf '%s\n' "$OUT" | grep -m1 '^PERF ' || true)"
PPL_LINE="$(printf '%s\n' "$OUT" | grep -m1 '^PPL ' || true)"
[[ -n "$PERF_LINE" ]] || { echo "FAIL: no PERF line in output" >&2; exit 1; }

get() { printf '%s\n' "$1" | sed -n "s/.*$2=\([^ ]*\).*/\1/p"; }
TOK_S="$(get "$PERF_LINE" tok_s)"
GEN="$(get "$PERF_LINE" gen_tokens)"
PROMPT_MS="$(get "$PERF_LINE" prompt_eval_ms)"
PPL="$(get "$PPL_LINE" value)"

CURRENT="tok_s=$TOK_S gen_tokens=$GEN prompt_eval_ms=$PROMPT_MS ppl=$PPL"
echo
echo "== Measured: $CURRENT"

if [[ "$RECORD" == "1" ]]; then
    {
        echo "# e2e regression baseline"
        echo "# host=${HOST:-local} model_dir=$MODEL_DIR"
        echo "# date=$(date -u +%Y-%m-%dT%H:%M:%SZ) cross=${CROSS_COMPILE:-none}"
        echo "$CURRENT"
    } > "$BASELINE"
    echo "== Baseline written to $BASELINE"
    exit 0
fi

[[ -f "$BASELINE" ]] || { echo "FAIL: no baseline at $BASELINE (run with --record first)" >&2; exit 1; }
BASE_LINE="$(grep -m1 '^tok_s=' "$BASELINE")"
B_TOK_S="$(get "$BASE_LINE" tok_s)"
B_PPL="$(get "$BASE_LINE" ppl)"

# awk for float comparison; a higher tok/s and a lower ppl are both better.
verdict="$(awk -v t="$TOK_S" -v bt="$B_TOK_S" -v p="$PPL" -v bp="$B_PPL" -v tol="$TOLERANCE" 'BEGIN {
    fail = 0
    if (bt+0 > 0) {
        drop = (bt - t) / bt * 100
        printf "throughput: %.2f -> %.2f tok/s (%+.1f%%)\n", bt, t, -drop
        if (drop > tol) { printf "REGRESSION: throughput dropped more than %s%%\n", tol; fail = 1 }
    }
    if (bp+0 > 0 && p+0 > 0 && bp == bp && p == p) {
        rise = (p - bp) / bp * 100
        printf "perplexity: %.4f -> %.4f (%+.1f%%)\n", bp, p, rise
        if (rise > tol) { printf "REGRESSION: perplexity rose more than %s%%\n", tol; fail = 1 }
    } else {
        print "perplexity: skipped (not scorable in this configuration)"
    }
    exit fail
}')"
status=$?
printf '%s\n' "$verdict"

if [[ $status -ne 0 ]]; then
    echo "== FAIL: regression detected (tolerance ${TOLERANCE}%)"
    exit 1
fi
echo "== PASS: within ${TOLERANCE}% of baseline"
