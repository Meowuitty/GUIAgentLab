#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
count="${1:-64}"
image="${GUIAGENTLAB_MW_IMAGE:-ghcr.io/tongyi-mai/mobile_world:latest}"
expected_image_id="sha256:aa743a95daef5cd25c54fc37626c2521c03b9c9d8d1aac4cc287f3643a3e23f3"
: "${GUIAGENTLAB_STATE_DIR:?Set GUIAGENTLAB_STATE_DIR to a persistent data volume}"
data_root="${GUIAGENTLAB_MW_DATA_ROOT:-$GUIAGENTLAB_STATE_DIR/mobileworld}"
rebuild="${GUIAGENTLAB_REBUILD_SLOTS:-0}"
if [[ -n "${GUIAGENTLAB_MW_CACHE_BATCH_SIZE:-}" ]]; then
  batch_size="$GUIAGENTLAB_MW_CACHE_BATCH_SIZE"
else
  batch_size=$((count < 8 ? count : 8))
fi
entrypoint="$repo_root/scripts/env/container-entrypoint.sh"
start_docker="$repo_root/scripts/env/start-docker.sh"

if [[ ! "$count" =~ ^[1-9][0-9]*$ ]] || ((count > 64)); then
  echo "Slot count must be between 1 and 64" >&2
  exit 2
fi
if [[ "$rebuild" != "0" && "$rebuild" != "1" ]]; then
  echo "GUIAGENTLAB_REBUILD_SLOTS must be 0 or 1" >&2
  exit 2
fi
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]] || ((batch_size > count)); then
  echo "GUIAGENTLAB_MW_CACHE_BATCH_SIZE must be between 1 and $count" >&2
  exit 2
fi
if [[ "$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null || true)" != "$expected_image_id" ]]; then
  echo "The pinned MobileWorld image is missing or has the wrong image ID" >&2
  exit 1
fi

mkdir -p "$data_root"
data_root="$(realpath "$data_root")"
if [[ "$data_root" == "/" || "$data_root" == "/home" || "$data_root" == "/data/checkpoint" ]]; then
  echo "Refusing unsafe MobileWorld data root: $data_root" >&2
  exit 1
fi

avd_root="$data_root/avd"
avd_template="$avd_root/clean-template"
avd_slots="$avd_root/slots"
docker_slots="$data_root/docker/slots"
manifest_root="$data_root/manifests"
mkdir -p "$avd_slots" "$docker_slots" "$manifest_root"

running_containers=()
cleanup_running() {
  if ((${#running_containers[@]})); then
    docker rm -f -v "${running_containers[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_running ERR INT TERM

remove_managed_path() {
  local target="$1"
  case "$target" in
    "$avd_template"|"$avd_slots"/slot-[0-9][0-9]|"$docker_slots"/slot-[0-9][0-9]|"$manifest_root"/slot-[0-9][0-9].env)
      rm -rf --one-file-system -- "$target"
      ;;
    *)
      echo "Refusing to remove unmanaged path: $target" >&2
      return 1
      ;;
  esac
}

if [[ "$rebuild" == "1" ]]; then
  remove_managed_path "$avd_template"
  for ((index=0; index<count; index++)); do
    slot="$(printf 'slot-%02d' "$index")"
    remove_managed_path "$avd_slots/$slot"
    remove_managed_path "$docker_slots/$slot"
    remove_managed_path "$manifest_root/$slot.env"
  done
fi

missing=0
for ((index=0; index<count; index++)); do
  slot="$(printf 'slot-%02d' "$index")"
  manifest="$manifest_root/$slot.env"
  if [[ ! -f "$manifest" ]] || ! grep -qx 'status=READY' "$manifest"; then
    missing=$((missing + 1))
  fi
done

if ((missing == 0)) && [[ -d "$avd_template" ]]; then
  echo "All $count persistent MobileWorld slots are already prepared"
  exit 0
fi

required_gib=$((missing * 12 + 20))
available_kib="$(df -Pk "$data_root" | awk 'NR == 2 {print $4}')"
if ((available_kib < required_gib * 1024 * 1024)); then
  echo "Insufficient space for $missing missing slots: require ${required_gib} GiB free" >&2
  exit 1
fi

if [[ ! -d "$avd_template" ]]; then
  temp_template="$avd_root/.clean-template.building"
  rm -rf --one-file-system -- "$temp_template"
  mkdir -p "$temp_template"
  seed_name="guiagentlab_avd_seed"
  docker rm -f -v "$seed_name" >/dev/null 2>&1 || true
  docker create --name "$seed_name" --entrypoint /bin/true "$image" >/dev/null
  docker cp "$seed_name:/root/.android/avd/." "$temp_template"
  docker rm -v "$seed_name" >/dev/null
  if [[ ! -s "$temp_template/Pixel_8_API_34_x86_64.ini" || \
        ! -s "$temp_template/Pixel_8_API_34_x86_64.avd/userdata-qemu.img.qcow2" ]]; then
    echo "Extracted AVD template is incomplete" >&2
    exit 1
  fi
  printf '%s\n' "$expected_image_id" >"$temp_template/.guiagentlab-template"
  mv "$temp_template" "$avd_template"
  echo "Extracted clean AVD template"
fi

pending=()
for ((index=0; index<count; index++)); do
  slot="$(printf 'slot-%02d' "$index")"
  avd_slot="$avd_slots/$slot"
  docker_slot="$docker_slots/$slot"
  manifest="$manifest_root/$slot.env"
  if [[ -f "$manifest" ]] && grep -qx 'status=READY' "$manifest"; then
    continue
  fi
  if [[ ! -f "$avd_slot/.guiagentlab-avd-ready" ]]; then
    remove_managed_path "$avd_slot"
    mkdir -p "$avd_slot"
    cp -a --sparse=always "$avd_template/." "$avd_slot/"
    printf '%s\n' "$expected_image_id" >"$avd_slot/.guiagentlab-avd-ready"
    echo "Prepared AVD $slot"
  fi
  remove_managed_path "$docker_slot"
  mkdir -p "$docker_slot"
  printf 'status=BUILDING\nouter_image_id=%s\n' "$expected_image_id" >"$manifest"
  pending+=("$index")
done

for ((offset=0; offset<${#pending[@]}; offset+=batch_size)); do
  batch=("${pending[@]:offset:batch_size}")
  running_containers=()
  for index in "${batch[@]}"; do
    slot="$(printf 'slot-%02d' "$index")"
    name="guiagentlab_cache_$index"
    docker_slot="$docker_slots/$slot"
    docker rm -f -v "$name" >/dev/null 2>&1 || true
    docker run --privileged --detach \
      --name "$name" \
      --volume "$docker_slot:/var/lib/docker" \
      --volume "$entrypoint:/usr/local/bin/guiagentlab-entrypoint.sh:ro" \
      --volume "$start_docker:/usr/local/bin/start-docker.sh:ro" \
      --env GUIAGENTLAB_PREWARM_ONLY=1 \
      --entrypoint /usr/local/bin/guiagentlab-entrypoint.sh \
      "$image" >/dev/null
    running_containers+=("$name")
  done

  batch_failed=0
  for index in "${batch[@]}"; do
    slot="$(printf 'slot-%02d' "$index")"
    name="guiagentlab_cache_$index"
    status="$(docker wait "$name")"
    if [[ "$status" != "0" ]]; then
      echo "Cache preparation failed for $slot (exit=$status)" >&2
      docker logs "$name" >&2 || true
      printf 'status=QUARANTINED\nouter_image_id=%s\n' "$expected_image_id" >"$manifest_root/$slot.env"
      batch_failed=1
    else
      printf 'status=READY\nouter_image_id=%s\nprepared_at=%s\n' \
        "$expected_image_id" "$(date --iso-8601=seconds)" >"$manifest_root/$slot.env"
      echo "Prepared inner-image cache $slot"
    fi
    docker rm -v "$name" >/dev/null 2>&1 || true
  done
  running_containers=()
  if ((batch_failed)); then
    exit 1
  fi
done

trap - ERR INT TERM
ready_count="$(grep -l '^status=READY$' "$manifest_root"/slot-*.env 2>/dev/null | wc -l)"
if ((ready_count < count)); then
  echo "Only $ready_count of $count slots are ready" >&2
  exit 1
fi
echo "Prepared $count persistent MobileWorld slots under $data_root"
