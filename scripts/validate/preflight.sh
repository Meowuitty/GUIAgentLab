#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
active="${1:-${ACTIVE_CONTAINERS:-32}}"
spares="${2:-${SPARE_CONTAINERS:-8}}"
server_file="${GUIAGENTLAB_SERVER_FILE:-$repo_root/configs/servers.txt}"
warm_concurrency="${INITIALIZATION_CONCURRENCY:-8}"
cd "$repo_root"
exec python -m guiagentlab.cli validate \
  --servers "$server_file" \
  --dataset data/test.parquet \
  --active "$active" \
  --spares "$spares" \
  --warm \
  --warm-concurrency "$warm_concurrency"
