#!/usr/bin/env bash
set -euo pipefail

# Launch script for Romanaccio LFM3 / LFM2.5 Harness Web Server on roma (Milestone 19D)
HARNESS_ROOT="${HARNESS_ROOT:-/home/chri/lfm3_harness_pkg}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/chri/workspace}"
STATE_DIR="${STATE_DIR:-/home/chri/.local/state/romanaccio_harness}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LFM3_PORT="${LFM3_PORT:-8088}"
MOE_PORT="${MOE_PORT:-8089}"
D1_PORT="${D1_PORT:-0}"
ENABLE_TLS="${ENABLE_TLS:-0}"

mkdir -p "$WORKSPACE_ROOT" "$STATE_DIR"
export PYTHONPATH="${HARNESS_ROOT}:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [ -f "/home/chri/models/LFM3-1.7B_ENNP/tokenizer.json" ]; then
  EXTRA_ARGS+=(--lfm3-tokenizer "/home/chri/models/LFM3-1.7B_ENNP/tokenizer.json")
fi
if [ -f "/home/chri/models/LFM2.5-8B-A1B_ENNP/tokenizer.json" ]; then
  EXTRA_ARGS+=(--moe-tokenizer "/home/chri/models/LFM2.5-8B-A1B_ENNP/tokenizer.json")
fi
if [ "$ENABLE_TLS" = "1" ]; then
  EXTRA_ARGS+=(--self-signed-tls)
fi
if [ -n "${AUTH_TOKEN:-}" ]; then
  EXTRA_ARGS+=(--auth-token "$AUTH_TOKEN")
fi

exec python3 -m lfm3_harness.server \
  --workspace-root "$WORKSPACE_ROOT" \
  --state-dir "$STATE_DIR" \
  --bind "$BIND_HOST" \
  --port "$PORT" \
  --lfm3-port "$LFM3_PORT" \
  --moe-port "$MOE_PORT" \
  --d1-port "$D1_PORT" \
  "${EXTRA_ARGS[@]}" "$@"
