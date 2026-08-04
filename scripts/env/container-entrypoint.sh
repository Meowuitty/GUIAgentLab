#!/usr/bin/env bash
set -euo pipefail

required_images=(
  "postgres:14-alpine|sha256:61e8629da09d426f8bf1ab755b2b7fa47a1a8fa7fb8e27745213a9d30af75067"
  "nginx:1.25-alpine|sha256:501d84f5d06487ff81e506134dc922ed4fd2080d5521eb5b6ee4054fa17d15c4"
  "redis:7-alpine|sha256:5f703438f575677037751d8350dc8ca2b8a1075655247757048c8a86f0fb4f02"
  "ghcr.io/mastodon/mastodon:v4.3.7|sha256:9f42cca562554c9a4b3c39fecf3590aaf48cd4f566054f94135bc7e417b9e609"
  "ghcr.io/mastodon/mastodon-streaming:v4.3.7|sha256:ce444902e7c4152b82ad2526fbd32d0eaa47cc9d6b2a7e4198cd2395579e63cf"
  "mattermost/mattermost-team-edition:10.5.2|sha256:2268f6d2a422a777496c76d6f253fcaa8e9b561dab780f02c2ca41cd404cca10"
  "postgres:13-alpine|sha256:479196f38b243f430e8c4d897341fdfbede2ea49183e3b849789d81ae39037de"
)

normalize_proxy() {
  local proxy="${http_proxy:-${HTTP_PROXY:-}}"
  if [[ -z "$proxy" ]]; then
    return
  fi
  export http_proxy="$proxy" HTTP_PROXY="$proxy"
  export https_proxy="${https_proxy:-${HTTPS_PROXY:-$proxy}}"
  export HTTPS_PROXY="$https_proxy"
  local user_no_proxy="${no_proxy:-${NO_PROXY:-}}"
  export no_proxy="10.0.2.2,127.0.0.1,localhost,::1${user_no_proxy:+,$user_no_proxy}"
  export NO_PROXY="$no_proxy"
}

configure_iptables() {
  if command -v update-alternatives >/dev/null 2>&1 && command -v iptables-nft >/dev/null 2>&1; then
    if iptables-nft -L -n >/dev/null 2>&1; then
      update-alternatives --set iptables /usr/sbin/iptables-nft >/dev/null 2>&1 || true
      update-alternatives --set ip6tables /usr/sbin/ip6tables-nft >/dev/null 2>&1 || true
    else
      update-alternatives --set iptables /usr/sbin/iptables-legacy >/dev/null 2>&1 || true
      update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy >/dev/null 2>&1 || true
    fi
  fi
}

images_are_exact() {
  local spec tag expected actual
  for spec in "${required_images[@]}"; do
    tag="${spec%%|*}"
    expected="${spec#*|}"
    actual="$(docker image inspect --format '{{.Id}}' "$tag" 2>/dev/null || true)"
    if [[ "$actual" != "$expected" ]]; then
      return 1
    fi
  done
}

load_images_if_needed() {
  if images_are_exact; then
    echo "Inner image cache is complete; skipping docker load"
    return
  fi
  echo "Loading pinned inner images into this slot"
  local archive
  for archive in /app/images/*.tar; do
    docker load --input "$archive"
  done
  if ! images_are_exact; then
    echo "Pinned inner-image verification failed after docker load" >&2
    exit 1
  fi
}

clean_mutable_docker_state() {
  local ids volumes custom_networks
  ids="$(docker ps --all --quiet)"
  if [[ -n "$ids" ]]; then
    xargs --no-run-if-empty docker rm --force --volumes <<<"$ids"
  fi
  volumes="$(docker volume ls --quiet)"
  if [[ -n "$volumes" ]]; then
    xargs --no-run-if-empty docker volume rm --force <<<"$volumes"
  fi
  custom_networks="$(docker network ls --filter type=custom --quiet)"
  if [[ -n "$custom_networks" ]]; then
    xargs --no-run-if-empty docker network rm <<<"$custom_networks"
  fi
  if [[ -n "$(docker ps --all --quiet)" || -n "$(docker volume ls --quiet)" ]]; then
    echo "Nested Docker state is not clean after sanitation" >&2
    exit 1
  fi
}

stop_nested_docker() {
  local pid attempt
  if [[ ! -s /var/run/docker.pid ]]; then
    return
  fi
  pid="$(cat /var/run/docker.pid)"
  kill -TERM "$pid" >/dev/null 2>&1 || true
  for ((attempt=0; attempt<60; attempt++)); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  echo "Nested dockerd did not stop cleanly" >&2
  return 1
}

normalize_proxy
sysctl net.ipv6.conf.all.disable_ipv6=1 >/dev/null
configure_iptables
start-docker.sh
load_images_if_needed

if [[ "${GUIAGENTLAB_PREWARM_ONLY:-0}" == "1" ]]; then
  clean_mutable_docker_state
  sync
  stop_nested_docker
  echo "Persistent inner-image cache is ready"
  exit 0
fi

clean_mutable_docker_state
if [[ -d /opt/guiagentlab-avd ]]; then
  rm -rf /root/.android/avd
  ln -s /opt/guiagentlab-avd /root/.android/avd
  # A stopped emulator leaves PID/AVD locks in the persistent slot. The next
  # disposable outer container has no emulator yet, so these locks are stale;
  # retaining them makes Emulator 36 report a false same-AVD multi-instance.
  find /opt/guiagentlab-avd -mindepth 2 -maxdepth 2 -type f -name '*.lock' -delete
fi
cd /app/service
if [[ "${ENABLE_VNC:-false}" == "true" || "${ENABLE_VNC:-false}" == "1" ]]; then
  /app/docker/start_novnc.sh
else
  uv run mobile-world viewer --port 7860 &
fi
bash /app/docker/start_emulator.sh
if command -v socat >/dev/null 2>&1; then
  socat TCP-LISTEN:5556,fork,reuseaddr,bind=0.0.0.0 TCP:127.0.0.1:5555 &
else
  python3 /usr/local/bin/guiagentlab-tcp-relay.py \
    0.0.0.0 5556 127.0.0.1 5555 &
fi
uv run mobile-world server --port 6800 >>/var/log/server.log 2>&1 &

if (($#)); then
  exec "$@"
fi
exec tail -f /var/log/emulator.log /var/log/server.log
