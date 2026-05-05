#!/usr/bin/env bash
# SSH into a Prime Intellect pod by pod-id (thin wrapper around `prime pods ssh`).
#
# Usage:
#   bin/ssh_pod.sh <pod-id> [extra args passed through to `prime pods ssh`]
#
# Examples:
#   bin/ssh_pod.sh abc123
#   bin/ssh_pod.sh abc123 -- -t 'tmux attach -t exp || tmux new -s exp'
#
# Prereqs:
#   - `prime` CLI installed and authenticated (`uv tool install prime && prime login`)
#   - SSH key registered with Prime (`prime config set-ssh-key-path ...`)

set -euo pipefail

if ! command -v prime >/dev/null 2>&1; then
  echo "error: prime CLI not found. install with: uv tool install prime" >&2
  exit 1
fi

POD_ID="${1:?usage: $0 <pod-id> [extra args...]}"
shift

exec prime pods ssh "$POD_ID" "$@"
