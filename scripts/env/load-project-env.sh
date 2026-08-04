#!/usr/bin/env bash

# Load the repository-local environment file into the calling shell. Entrypoints
# source this helper after setting ``repo_root``.
project_env="$repo_root/.env"
if [[ -f "$project_env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$project_env"
  set +a
fi
unset project_env
