#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
profile="${1:-train}"
image="${GUIAGENTLAB_MW_IMAGE:-ghcr.io/tongyi-mai/mobile_world:latest}"
expected_image_id="sha256:aa743a95daef5cd25c54fc37626c2521c03b9c9d8d1aac4cc287f3643a3e23f3"
prefix="${GUIAGENTLAB_MW_PREFIX:-guiagentlab_mw}"
start_port="${GUIAGENTLAB_MW_START_PORT:-20000}"
viewer_start_port="${GUIAGENTLAB_MW_VIEWER_START_PORT:-21000}"
vnc_start_port="${GUIAGENTLAB_MW_VNC_START_PORT:-22000}"
adb_start_port="${GUIAGENTLAB_MW_ADB_START_PORT:-23000}"
: "${GUIAGENTLAB_STATE_DIR:?Set GUIAGENTLAB_STATE_DIR to a large local data volume}"
data_root="${GUIAGENTLAB_MW_DATA_ROOT:-$GUIAGENTLAB_STATE_DIR/mobileworld}"
env_file="${GUIAGENTLAB_MW_ENV_FILE:-$repo_root/mobile_world/.env}"
require_user_agent="${GUIAGENTLAB_REQUIRE_USER_AGENT:-0}"
reuse_existing="${GUIAGENTLAB_MW_REUSE_EXISTING:-0}"
port_release_timeout="${GUIAGENTLAB_MW_PORT_RELEASE_TIMEOUT:-30}"
mastodon_command_timeout="${GUIAGENTLAB_MW_MASTODON_COMMAND_TIMEOUT_SECONDS:-300}"
mastodon_ready_timeout="${GUIAGENTLAB_MW_MASTODON_READY_TIMEOUT_SECONDS:-180}"

if [[ ! "$prefix" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Unsafe MobileWorld pool prefix: $prefix" >&2
  exit 1
fi
if [[ -n "${GUIAGENTLAB_MW_ENV_FILE:-}" && ! -f "$env_file" ]]; then
  echo "MobileWorld environment file does not exist: $env_file" >&2
  exit 1
fi
if [[ "$require_user_agent" != "0" && "$require_user_agent" != "1" ]]; then
  echo "GUIAGENTLAB_REQUIRE_USER_AGENT must be 0 or 1" >&2
  exit 1
fi
if [[ "$require_user_agent" == "1" ]]; then
  if [[ ! -f "$env_file" ]]; then
    echo "MobileWorld ask_user requires an environment file: $env_file" >&2
    echo "Set USER_AGENT_API_KEY, USER_AGENT_BASE_URL, and USER_AGENT_MODEL in that file." >&2
    exit 1
  fi
  python - "$env_file" <<'PY'
import sys
from pathlib import Path
from urllib.parse import urlparse

path = Path(sys.argv[1])
values = {}
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    values[key.strip()] = value.strip().strip('"').strip("'")

required = ("USER_AGENT_API_KEY", "USER_AGENT_BASE_URL", "USER_AGENT_MODEL")
missing = [key for key in required if not values.get(key)]
if missing:
    raise SystemExit(f"MobileWorld user-agent configuration is missing: {missing}")
if any(
    marker in values["USER_AGENT_API_KEY"].lower()
    for marker in ("your_", "your-", "replace", "example")
):
    raise SystemExit("MobileWorld USER_AGENT_API_KEY is still a placeholder")
url = urlparse(values["USER_AGENT_BASE_URL"])
if url.scheme not in {"http", "https"} or not url.netloc:
    raise SystemExit("MobileWorld USER_AGENT_BASE_URL must be an absolute HTTP(S) URL")
PY
fi
if [[ "$reuse_existing" != "0" && "$reuse_existing" != "1" ]]; then
  echo "GUIAGENTLAB_MW_REUSE_EXISTING must be 0 or 1" >&2
  exit 1
fi
if [[ ! "$port_release_timeout" =~ ^[1-9][0-9]*$ ]]; then
  echo "GUIAGENTLAB_MW_PORT_RELEASE_TIMEOUT must be a positive integer" >&2
  exit 1
fi
if [[ ! "$mastodon_command_timeout" =~ ^[1-9][0-9]*$ ]]; then
  echo "GUIAGENTLAB_MW_MASTODON_COMMAND_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
fi
if [[ ! "$mastodon_ready_timeout" =~ ^[1-9][0-9]*$ ]]; then
  echo "GUIAGENTLAB_MW_MASTODON_READY_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
fi

case "$profile" in
  train)
    active="${ACTIVE_CONTAINERS:-32}"
    spares="${SPARE_CONTAINERS:-8}"
    ;;
  rollout)
    active="${ACTIVE_CONTAINERS:-64}"
    spares="${SPARE_CONTAINERS:-0}"
    ;;
  *)
    echo "Usage: $0 [train|rollout]" >&2
    exit 2
    ;;
esac
count=$((active + spares))

minimum_inotify_instances="${GUIAGENTLAB_MW_MIN_INOTIFY_INSTANCES:-$((count >= 64 ? 512 : 128))}"
current_inotify_instances="$(sysctl -n fs.inotify.max_user_instances)"
if ((current_inotify_instances < minimum_inotify_instances)); then
  echo "Host fs.inotify.max_user_instances is too low: ${current_inotify_instances}" >&2
  echo "This ${count}-container nested-Docker pool requires at least ${minimum_inotify_instances}." >&2
  echo "Raise it before launch, for example: sudo sysctl -w fs.inotify.max_user_instances=${minimum_inotify_instances}" >&2
  exit 1
fi
if ! actual_image_id="$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null)"; then
  echo "Required local image is missing: $image" >&2
  exit 1
fi
if [[ "$actual_image_id" != "$expected_image_id" ]]; then
  echo "Unexpected local MobileWorld image ID: $actual_image_id" >&2
  echo "Expected the fixed 2025-12-24 image ID: $expected_image_id" >&2
  exit 1
fi
mkdir -p "$data_root"
data_root="$(realpath "$data_root")"
if [[ "$data_root" == "/" ]]; then
  echo "Refusing to use the filesystem root for MobileWorld state" >&2
  exit 1
fi
required_free_gib="${GUIAGENTLAB_MW_MIN_FREE_GIB:-$((count * 3 + 30))}"
available_kib="$(df -Pk "$data_root" | awk 'NR == 2 {print $4}')"
required_kib=$((required_free_gib * 1024 * 1024))
if ((available_kib < required_kib)); then
  echo "Insufficient free disk at $data_root: require ${required_free_gib} GiB" >&2
  exit 1
fi

remove_pool_by_prefix() {
  local target_prefix="$1"
  local ids=()
  local container_id container_name
  while read -r container_id container_name; do
    if [[ "$container_name" == "${target_prefix}_"* ]]; then
      ids+=("$container_id")
    fi
  done < <(docker ps -a --format '{{.ID}} {{.Names}}')
  if ((${#ids[@]})); then
    echo "Removing ${#ids[@]} containers with prefix ${target_prefix}_"
    docker rm -f "${ids[@]}"
  fi
}

if [[ "$reuse_existing" == "1" ]]; then
  for ((index=0; index<count; index++)); do
    name="${prefix}_${index}"
    if [[ "$(docker inspect --format '{{.State.Running}} {{.Image}}' "$name" 2>/dev/null || true)" != "true $expected_image_id" ]]; then
      echo "Cannot resume: expected running pinned container $name" >&2
      exit 1
    fi
  done
  echo "Reusing existing $count-container ${prefix}_ pool"
else
  # Only explicitly selected pool namespaces are removed. Other Docker
  # containers are never selected.
  remove_pool_by_prefix "$prefix"
  if [[ -n "${GUIAGENTLAB_REPLACE_PREFIXES:-}" ]]; then
    read -r -a replace_prefixes <<< "$GUIAGENTLAB_REPLACE_PREFIXES"
    for replace_prefix in "${replace_prefixes[@]}"; do
      if [[ "$replace_prefix" != "$prefix" ]]; then
        remove_pool_by_prefix "$replace_prefix"
      fi
    done
  fi

  # MobileWorld skips occupied ports. Since the server file intentionally uses
  # a deterministic 20000-based block, fail instead of writing wrong URLs.
  python - \
    "$count" "$port_release_timeout" \
    "$start_port" "$viewer_start_port" "$vnc_start_port" "$adb_start_port" <<'PY'
import socket
import sys
import time

count, timeout, *starts = map(int, sys.argv[1:])
deadline = time.monotonic() + timeout
while True:
    occupied = []
    for start in starts:
        for port in range(start, start + count):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("0.0.0.0", port))
                except OSError:
                    occupied.append(port)
    if not occupied:
        break
    if time.monotonic() >= deadline:
        raise SystemExit(f"required container ports are occupied: {occupied}")
    time.sleep(1)
PY
fi

source_mount="$repo_root/mobile_world/src"
launch_interval="${GUIAGENTLAB_MW_LAUNCH_INTERVAL:-0}"
launch_batch_size="${GUIAGENTLAB_MW_LAUNCH_BATCH_SIZE:-8}"
ready_timeout="${GUIAGENTLAB_MW_READY_TIMEOUT:-1200}"
start_docker_patch="$repo_root/scripts/env/start-docker.sh"
container_entrypoint="$repo_root/scripts/env/container-entrypoint.sh"
tcp_relay="$repo_root/scripts/env/tcp_relay.py"
start_emulator_patch="$repo_root/mobile_world/docker/start_emulator.sh"
proxy_chain_patch="$repo_root/mobile_world/docker/proxy_chain.py"
required_inner_images=(
  "postgres:14-alpine"
  "nginx:1.25-alpine"
  "redis:7-alpine"
  "ghcr.io/mastodon/mastodon:v4.3.7"
  "ghcr.io/mastodon/mastodon-streaming:v4.3.7"
  "mattermost/mattermost-team-edition:10.5.2"
  "postgres:13-alpine"
)
if ((launch_batch_size <= 0 || launch_batch_size > count)); then
  echo "GUIAGENTLAB_MW_LAUNCH_BATCH_SIZE must be between 1 and $count" >&2
  exit 1
fi
launched_names=()
cleanup_failed_launch() {
  if ((${#launched_names[@]})); then
    docker rm -f "${launched_names[@]}" >/dev/null 2>&1 || true
  fi
  local index name attempt
  for ((index=0; index<${#launched_names[@]}; index++)); do
    name="${launched_names[$index]}"
    for ((attempt=0; attempt<60; attempt++)); do
      if ! docker inspect "$name" >/dev/null 2>&1; then
        break
      fi
      docker rm -f "$name" >/dev/null 2>&1 || true
      sleep 2
    done
    if docker inspect "$name" >/dev/null 2>&1; then
      echo "Container still exists after cleanup attempts: $name" >&2
    fi
  done
}
cleanup_and_exit() {
  local exit_code="$1"
  trap - ERR INT TERM
  cleanup_failed_launch
  exit "$exit_code"
}
trap cleanup_failed_launch ERR
trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM

proxy_for_container() {
  local value="$1"
  value="${value/\/\/127.0.0.1/\/\/host.docker.internal}"
  value="${value/\/\/localhost/\/\/host.docker.internal}"
  printf '%s' "$value"
}

wait_started_batch() {
  local first="$1"
  local last="$2"
  local deadline=$((SECONDS + ready_timeout))
  local index name state health pending
  while true; do
    pending=0
    for ((index=first; index<=last; index++)); do
      name="${prefix}_${index}"
      state="$(docker inspect --format '{{.State.Status}}' "$name" 2>/dev/null || true)"
      if [[ -z "$state" ]]; then
        echo "Startup container disappeared before readiness: $name" >&2
        return 1
      fi
      if [[ "$state" == "exited" || "$state" == "dead" ]]; then
        echo "Startup container failed before readiness: $name (state=$state)" >&2
        docker logs --tail 200 "$name" >&2 || true
        return 1
      fi
      health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$name" 2>/dev/null || true)"
      if [[ "$health" != "healthy" ]] || ! docker exec "$name" \
        docker image inspect "${required_inner_images[@]}" >/dev/null 2>&1; then
        pending=$((pending + 1))
      fi
    done
    if ((pending == 0)); then
      echo "Startup batch $first-$last is healthy with all pinned inner images"
      return 0
    fi
    if ((SECONDS >= deadline)); then
      echo "Startup batch $first-$last did not become complete; pending=$pending" >&2
      return 1
    fi
    sleep 2
  done
}

if [[ "$reuse_existing" != "1" ]]; then
  "$repo_root/scripts/env/prepare_slots.sh" "$count"
  for ((index=0; index<count; index++)); do
    name="${prefix}_${index}"
    backend_port=$((start_port + index))
    viewer_port=$((viewer_start_port + index))
    vnc_port=$((vnc_start_port + index))
    adb_port=$((adb_start_port + index))
    slot="$(printf 'slot-%02d' "$index")"
    avd_data="$data_root/avd/slots/$slot"
    nested_docker="$data_root/docker/slots/$slot"
    manifest="$data_root/manifests/$slot.env"
    if [[ ! -f "$manifest" ]] || ! grep -qx 'status=READY' "$manifest"; then
      echo "Persistent cache is not ready: $slot" >&2
      exit 1
    fi
    run_args=(
      docker run --privileged --detach
      --name "$name"
      --publish "$backend_port:6800"
      --publish "$viewer_port:7860"
      --publish "$vnc_port:5800"
      --publish "$adb_port:5556"
      --label "org.guiagentlab.mobileworld.managed=true"
      --label "org.guiagentlab.mobileworld.endpoint=http://127.0.0.1:$backend_port"
      --label "org.guiagentlab.mobileworld.slot=$index"
      --volume "$source_mount:/app/service/src"
      --volume "$avd_data:/opt/guiagentlab-avd"
      --volume "$nested_docker:/var/lib/docker"
      --volume "$start_docker_patch:/usr/local/bin/start-docker.sh:ro"
      --volume "$container_entrypoint:/usr/local/bin/guiagentlab-entrypoint.sh:ro"
      --volume "$tcp_relay:/usr/local/bin/guiagentlab-tcp-relay.py:ro"
      --volume "$start_emulator_patch:/app/docker/start_emulator.sh:ro"
      --volume "$proxy_chain_patch:/app/docker/proxy_chain.py:ro"
      --add-host "host.docker.internal:host-gateway"
      --env "UV_NO_SYNC=1"
      --env "MOBILE_WORLD_MASTODON_COMMAND_TIMEOUT_SECONDS=$mastodon_command_timeout"
      --env "MOBILE_WORLD_MASTODON_READY_TIMEOUT_SECONDS=$mastodon_ready_timeout"
      --entrypoint /usr/local/bin/guiagentlab-entrypoint.sh
    )
    if [[ -f "$env_file" ]]; then
      run_args+=(--volume "$env_file:/app/service/.env:ro")
    fi
    if [[ -n "${HTTP_PROXY:-}" ]]; then
      no_proxy="10.0.2.2,127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}"
      container_http_proxy="$(proxy_for_container "$HTTP_PROXY")"
      container_https_proxy="$(proxy_for_container "${HTTPS_PROXY:-$HTTP_PROXY}")"
      run_args+=(
        --env "http_proxy=$container_http_proxy"
        --env "https_proxy=$container_https_proxy"
        --env "HTTP_PROXY=$container_http_proxy"
        --env "HTTPS_PROXY=$container_https_proxy"
        --env "no_proxy=$no_proxy"
        --env "NO_PROXY=$no_proxy"
      )
    fi
    run_args+=("$image")
    "${run_args[@]}" >/dev/null
    launched_names+=("$name")
    if ((launch_interval > 0 && index + 1 < count)); then
      sleep "$launch_interval"
    fi
    if ((((index + 1) % launch_batch_size == 0) || index + 1 == count)); then
      wait_started_batch "$((index / launch_batch_size * launch_batch_size))" "$index"
    fi
  done
fi

# Wait for all backends concurrently. The no-proxy opener prevents host proxy
# settings from redirecting localhost health checks.
python - "$count" "$start_port" "$ready_timeout" <<'PY'
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request

count, start_port, timeout = map(int, sys.argv[1:])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def wait_ready(port: int) -> tuple[int, str | None]:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    last_error = None
    while time.monotonic() < deadline:
        try:
            with opener.open(url, timeout=5) as response:
                payload = json.load(response)
            if response.status == 200 and payload.get("ok"):
                return port, None
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(2)
    return port, last_error or "health endpoint did not report ok=true"


with concurrent.futures.ThreadPoolExecutor(max_workers=min(count, 64)) as pool:
    results = list(pool.map(wait_ready, range(start_port, start_port + count)))
failed = [(port, error) for port, error in results if error]
if failed:
    raise SystemExit(f"MobileWorld backends failed readiness: {failed[:5]}")
PY

for ((index=0; index<count; index++)); do
  name="${prefix}_${index}"
  if ! docker exec "$name" docker image inspect "${required_inner_images[@]}" >/dev/null 2>&1; then
    echo "Container $name is missing one or more pinned inner images" >&2
    exit 1
  fi
done
trap - ERR INT TERM

server_file="$repo_root/configs/servers.txt"
: > "$server_file"
for ((index=0; index<count; index++)); do
  printf 'http://127.0.0.1:%s\n' "$((start_port + index))" >> "$server_file"
done

# A freshly recreated pool supersedes any quarantine left by an earlier
# process. Runtime failures after this point are persisted by the pool actor.
PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
  python - "$server_file" <<'PY'
import sys

from guiagentlab.env.recovery import EndpointRecovery, RecoveryConfig

config = RecoveryConfig.from_environment()
if config is not None:
    recovery = EndpointRecovery(config)
    with open(sys.argv[1], encoding="utf-8") as handle:
        for line in handle:
            endpoint = line.strip()
            if endpoint and not endpoint.startswith("#"):
                recovery.mark(endpoint, "available", None)
PY

echo "Created $profile pool: active=$active spares=$spares total=$count"
ACTIVE_CONTAINERS="$active" \
SPARE_CONTAINERS="$spares" \
GUIAGENTLAB_SERVER_FILE="$server_file" \
  "$repo_root/scripts/validate/preflight.sh"
