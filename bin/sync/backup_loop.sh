#!/usr/bin/env bash
# LOCAL — periodically rsync the pod's outputs back to the local repo.
#
# Pulls every path listed in sync_paths.txt from $POD:$REMOTE_REPO/<path>
# to $LOCAL_REPO/<path>, on a fixed interval.
#
# Usage:
#   POD=root@1.2.3.4 bin/sync/backup_loop.sh
#
# Required env vars:
#   POD            ssh destination (user@host[:port])
#
# Optional env vars:
#   INTERVAL       seconds between pulls (default: 30)
#   REMOTE_REPO    pod-side repo root, relative to $HOME (default: nlp-final-project/)
#   LOCAL_REPO     laptop-side repo root (default: <script_dir>/../..)
#   PATHS_FILE     list of paths to sync (default: <script_dir>/sync_paths.txt)
#
# Run in a separate terminal while training/sampling runs in the container.

set -euo pipefail

: "${POD:?set POD=user@host}"

INTERVAL="${INTERVAL:-3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
REMOTE_REPO="${REMOTE_REPO:-nlp-final-project/}"
PATHS_FILE="${PATHS_FILE:-$SCRIPT_DIR/sync_paths.txt}"

if [[ ! -f "$PATHS_FILE" ]]; then
  echo "error: paths file not found: $PATHS_FILE" >&2
  exit 1
fi

REMOTE_REPO="${REMOTE_REPO%/}"
LOCAL_REPO="${LOCAL_REPO%/}"

PATHS=()
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
  PATHS+=("$line")
done < "$PATHS_FILE"

if [[ ${#PATHS[@]} -eq 0 ]]; then
  echo "error: no sync paths in $PATHS_FILE" >&2
  exit 1
fi

trap 'echo "[$(date +%T)] backup loop stopped"; exit 0' INT TERM

echo "[$(date +%T)] backup loop started"
echo "    remote:   $POD:$REMOTE_REPO/"
echo "    local:    $LOCAL_REPO/"
echo "    interval: ${INTERVAL}s"
echo "    paths (${#PATHS[@]}):"
for p in "${PATHS[@]}"; do echo "        - $p"; done
echo "    (Ctrl-C to stop)"

iter=0
while true; do
  iter=$((iter + 1))
  failures=0
  pulled_total=0
  for p in "${PATHS[@]}"; do
    src="$POD:$REMOTE_REPO/$p"
    dst="$LOCAL_REPO/$p"
    parent="$LOCAL_REPO/$(dirname "${p%/}")"
    mkdir -p "$parent"
    # rsync default: skip files whose size+mtime match — only diffs transfer.
    # -i (itemize) emits one line per change like ">f+++++++++ path/file"; we
    # count those instead of parsing --stats labels (which vary across rsync
    # versions / openrsync on macOS).
    if out=$(rsync -ai --partial --exclude='*.tmp' "$src" "$dst" 2>&1); then
      n=$(printf '%s\n' "$out" | awk '/^>f/ {n++} END {print n+0}')
      pulled_total=$((pulled_total + n))
      if (( n > 0 )); then
        printf '[%s]   %-24s  +%d file(s)\n' "$(date +%T)" "$p" "$n"
        printf '%s\n' "$out" | awk '/^>f/ {sub(/^[^ ]+[ ]+/, ""); print "        " $0}'
      fi
    else
      printf '[%s]   %-24s  FAILED\n' "$(date +%T)" "$p"
      printf '%s\n' "$out" | sed 's/^/      /'
      failures=$((failures + 1))
    fi
  done
  if (( failures == 0 )); then
    if (( pulled_total > 0 )); then
      printf '[%s] iter %d: pulled %d new file(s) across %d path(s); next pull in %ss\n' \
             "$(date +%T)" "$iter" "$pulled_total" "${#PATHS[@]}" "$INTERVAL"
    else
      printf '[%s] iter %d: no changes; next pull in %ss\n' \
             "$(date +%T)" "$iter" "$INTERVAL"
    fi
  else
    printf '[%s] iter %d: %d/%d path(s) failed; retry in %ss\n' \
           "$(date +%T)" "$iter" "$failures" "${#PATHS[@]}" "$INTERVAL"
  fi
  sleep "$INTERVAL"
done
