#!/usr/bin/env bash
set -euo pipefail

source /opt/bash-utils/logger.sh

timeout="${GUIAGENTLAB_DOCKER_READY_TIMEOUT:-600}"
deadline=$((SECONDS + timeout))

INFO "Starting supervisor"
/usr/bin/supervisord -n >/dev/null 2>&1 &

INFO "Waiting for the nested Docker API"
until docker info >/dev/null 2>&1; do
  if ((SECONDS >= deadline)); then
    ERROR "Nested Docker API was not ready after ${timeout} seconds"
    exit 1
  fi
  sleep 1
done
INFO "Nested Docker API is ready"
