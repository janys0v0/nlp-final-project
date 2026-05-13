#!/usr/bin/env bash
# Run scripts/smoke_generate.py on a Prime Intellect pod and pull the log back
# to ./outputs/smoke/ on the local machine.
#
# Usage:
#   bin/smoke_test/smoke_on_pod.sh <ssh-dest> [-- <smoke_generate.py args...>]
#
# Examples:
#   bin/smoke_test/smoke_on_pod.sh root@193.7.0.42
#   bin/smoke_test/smoke_on_pod.sh root@193.7.0.42 -- --model Qwen/Qwen2.5-0.5B
#   bin/smoke_test/smoke_on_pod.sh root@193.7.0.42 -- --max-new-tokens 128 --temperature 0.0
#
# Get the SSH destination from `prime pods status <pod-id>` or `prime pods ssh <pod-id>`.
#
# Pod preconditions:
#   - ~/nlp-final-project is git-cloned and current
#   - uv is on PATH (run setup_pod.sh, or `curl -LsSf https://astral.sh/uv/install.sh | sh`)

set -euo pipefail

SSH_DEST="${1:?usage: $0 <ssh-dest> [-- smoke_generate.py args...]}"
shift
[ "${1:-}" = "--" ] && shift

REPO_DIR="nlp-final-project/janys-setup-intellect"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
REMOTE_LOG="results/smoke/run-${TIMESTAMP}.log"
LOCAL_DIR="outputs/smoke"

mkdir -p "$LOCAL_DIR"

pull_log() {
  if rsync -avz "$SSH_DEST:$REPO_DIR/$REMOTE_LOG" "$LOCAL_DIR/" 2>/dev/null; then
    echo "[$(date +%T)] saved: $LOCAL_DIR/$(basename "$REMOTE_LOG")"
  else
    echo "[$(date +%T)] no log to pull (run may have failed before logging started)"
  fi
}
trap pull_log EXIT

# Quote pass-through args safely for the remote shell.
REMOTE_ARGS=""
for a in "$@"; do REMOTE_ARGS+=$(printf ' %q' "$a"); done

REMOTE_LOG_DIR="$(dirname "$REMOTE_LOG")"

echo "[$(date +%T)] running on $SSH_DEST -> $REMOTE_LOG"
ssh -t "$SSH_DEST" "
  set -e
  cd '$REPO_DIR'
  git pull --ff-only >/dev/null 2>&1 || true
  uv sync --extra gpu --quiet
  mkdir -p '$REMOTE_LOG_DIR'
  uv run python scripts/smoke_generate.py --log-file '$REMOTE_LOG'$REMOTE_ARGS
"
