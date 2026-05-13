#!/usr/bin/env bash
# SSH into a Prime Intellect pod by pod-id (thin wrapper around `prime pods ssh`).
#
# Usage:
# POD=user@xxx.xx.xx.xxx bin/sync/backup_loop.sh

set -euo pipefail

if ! command -v prime >/dev/null 2>&1; then
  echo "error: prime CLI not found. install with: uv tool install prime" >&2
  exit 1
fi

POD_ID="${1:?usage: $0 <pod-id> [extra args...]}"
shift

exec prime pods ssh "$POD_ID" "$@"
