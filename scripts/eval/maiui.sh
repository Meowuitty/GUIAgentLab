#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
requested_model_path="${MODEL_PATH:-}"
source "$repo_root/scripts/env/load-project-env.sh"
if [[ -n "$requested_model_path" ]]; then
  MODEL_PATH="$requested_model_path"
fi
unset requested_model_path

if (($# != 1)); then
  echo "Usage: MODEL_PATH=/path/to/model $0 MODEL_LABEL" >&2
  exit 2
fi
model_label="$1"
: "${MODEL_PATH:?MODEL_PATH must point to an original MAI-UI checkpoint}"
: "${GUIAGENTLAB_STATE_DIR:?Set GUIAGENTLAB_STATE_DIR to the evaluation data volume}"

expected_samples=1
temperature=0.0
max_tokens=2048
workers="${EVAL_WORKERS:-56}"

dataset="${EVAL_DATASET:-$repo_root/data/test.parquet}"
environment_servers="${GUIAGENTLAB_SERVER_FILE:-$repo_root/configs/servers.txt}"
expected_tasks="${EXPECTED_TASKS:-105}"
active_containers="${ACTIVE_CONTAINERS:-56}"
spare_containers="${SPARE_CONTAINERS:-8}"
model_replicas="${MODEL_REPLICAS:-8}"
model_base_port="${MODEL_BASE_PORT:-8100}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.60}"
max_steps="${MAX_STEPS:-50}"
max_attempts="${MAX_INFRASTRUCTURE_ATTEMPTS:-8}"
exhausted_task_policy="${EXHAUSTED_TASK_POLICY:-abort}"
request_timeout="${REQUEST_TIMEOUT:-120}"
acquire_timeout="${ACQUIRE_TIMEOUT:-7200}"
step_wait_time="${STEP_WAIT_TIME:-3}"
model_name="${MODEL_NAME:-$(basename "$MODEL_PATH")}"

timestamp="$(date +%Y%m%d-%H%M%S)"
run_id="${RUN_ID:-${model_label}-greedy-${timestamp}}"
experiment_name="${EXPERIMENT_NAME:-${model_label}-greedy-mobileworld${expected_tasks}-${timestamp}}"
data_root="${GUIAGENTLAB_DATA_ROOT:-$GUIAGENTLAB_STATE_DIR/evaluations}"
run_root="${RUN_ROOT:-$data_root/$run_id}"
log_dir="$run_root/logs"
model_log_dir="$log_dir/model_servers"
episode_dir="$run_root/trajectories"
summary_dir="$run_root/results"
model_servers="$run_root/model_servers.txt"
run_manifest="$run_root/run_manifest.json"

[[ -d "$MODEL_PATH" ]] || { echo "Missing model directory: $MODEL_PATH" >&2; exit 1; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing model config: $MODEL_PATH/config.json" >&2; exit 1; }
[[ -f "$dataset" ]] || { echo "Missing evaluation data: $dataset" >&2; exit 1; }
[[ -f "$environment_servers" ]] || { echo "Missing environment server list: $environment_servers" >&2; exit 1; }
if ((active_containers + spare_containers != 64)); then
  echo "Evaluation requires 56 active plus 8 spare endpoints by default" >&2
  exit 1
fi
if ((workers > active_containers)); then
  echo "EVAL_WORKERS cannot exceed ACTIVE_CONTAINERS" >&2
  exit 1
fi

mkdir -p "$model_log_dir" "$episode_dir" "$summary_dir"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export WANDB_MODE=disabled
export PYTHONPATH="$repo_root/verl:$repo_root/mobile_world/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"

python_site="$(python -c 'import site; print(site.getsitepackages()[0])')"
nvidia_lib_path="$(find "$python_site/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)"
runtime_ld_library_path="$nvidia_lib_path:/usr/local/cuda/lib64"

python -m guiagentlab.cli validate \
  --servers "$environment_servers" \
  --dataset "$dataset" \
  --active "$active_containers" \
  --spares "$spare_containers" \
  --recover \
  >"$run_root/environment_preflight.json"

mapfile -t gpu_ids < <(
  tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" | sed '/^$/d'
)
if ((${#gpu_ids[@]} < model_replicas)); then
  echo "Need $model_replicas visible GPUs, got ${#gpu_ids[@]}" >&2
  exit 1
fi

model_pids=()
cleanup() {
  local pid
  for pid in "${model_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${model_pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

: >"$model_servers"
for ((replica = 0; replica < model_replicas; replica++)); do
  gpu_id="${gpu_ids[$replica]}"
  port="$((model_base_port + replica))"
  if ss -ltnH "sport = :$port" | grep -q .; then
    echo "Model port is already in use: $port" >&2
    exit 1
  fi
  echo "http://127.0.0.1:$port/v1" >>"$model_servers"
  env \
    LD_LIBRARY_PATH="$runtime_ld_library_path" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --served-model-name "$model_name" \
      --host 127.0.0.1 \
      --port "$port" \
      --gpu-memory-utilization "$gpu_memory_utilization" \
      --tensor-parallel-size 1 \
      --max-model-len 25600 \
      --seed 0 \
      --trust-remote-code \
      >"$model_log_dir/replica-$replica.log" 2>&1 &
  model_pids+=("$!")
done

echo "Run root: $run_root"
echo "Started $model_replicas independent vLLM replicas concurrently"

deadline="$((SECONDS + 300))"
while true; do
  ready=0
  while IFS= read -r endpoint; do
    if curl --noproxy '*' -fsS --max-time 2 "$endpoint/models" >/dev/null 2>&1; then
      ready="$((ready + 1))"
    fi
  done <"$model_servers"
  if ((ready == model_replicas)); then
    break
  fi
  for pid in "${model_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "A model server exited during startup; inspect $model_log_dir" >&2
      exit 1
    fi
  done
  if ((SECONDS >= deadline)); then
    echo "Timed out waiting for model servers ($ready/$model_replicas ready)" >&2
    exit 1
  fi
  sleep 2
done

python - "$run_manifest" "$MODEL_PATH" "$model_name" \
  "$expected_tasks" "$expected_samples" "$workers" "$environment_servers" \
  "$model_servers" "$model_replicas" "$repo_root" "$dataset" \
  "$temperature" "$max_tokens" "$max_steps" "$active_containers" \
  "$spare_containers" "$gpu_memory_utilization" "$max_attempts" \
  "$exhausted_task_policy" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from guiagentlab.training.tracking import git_revision, sha256_file, sha256_tree

(
    path,
    model_path,
    model_name,
    tasks,
    samples,
    workers,
    envs,
    models,
    replicas,
    repo_root,
    dataset,
    temperature,
    max_tokens,
    max_steps,
    active_containers,
    spare_containers,
    gpu_memory_utilization,
    max_attempts,
    exhausted_task_policy,
) = sys.argv[1:]
repo = Path(repo_root)


def endpoints(source):
    return [
        line.strip()
        for line in Path(source).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


weight_files = [
    {
        "name": candidate.name,
        "bytes": candidate.stat().st_size,
        "mtime_ns": candidate.stat().st_mtime_ns,
        "sha256": sha256_file(candidate),
    }
    for candidate in sorted(Path(model_path).glob("*.safetensors"))
]
payload = {
    "schema_version": 2,
    "created_at": datetime.now(UTC).isoformat(),
    "model_path": model_path,
    "model_name": model_name,
    "model_weight_files": weight_files,
    "mode": "greedy",
    "task_count": int(tasks),
    "samples_per_task": int(samples),
    "workers": int(workers),
    "environment_servers": envs,
    "environment_endpoints": endpoints(envs),
    "model_endpoints": endpoints(models),
    "inference_transport": "openai_chat_completions",
    "protocol": {
        "id": "greedy",
        "temperature": float(temperature),
        "top_k": -1,
        "top_p": 1.0,
        "max_output_tokens": int(max_tokens),
        "max_steps": int(max_steps),
        "history_images": 3,
        "active_containers": int(active_containers),
        "spare_containers": int(spare_containers),
        "max_infrastructure_attempts": int(max_attempts),
        "exhausted_task_policy": exhausted_task_policy,
        "dynamic_batching": True,
        "wandb": False,
    },
    "vllm": {
        "replicas": int(replicas),
        "seed": 0,
        "trust_remote_code": True,
        "max_model_len": 25600,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "full_determinism": False,
        "batch_invariant": False,
    },
    "provenance": {
        "dataset": str(Path(dataset).resolve()),
        "dataset_sha256": sha256_file(dataset),
        "evaluation_entrypoint_sha256": sha256_file(repo / "scripts/eval/maiui.sh"),
        "guiagentlab_tree_sha256": sha256_tree(repo / "guiagentlab"),
        "mobileworld_revision": git_revision(repo / "mobile_world"),
        "mobileworld_src_tree_sha256": sha256_tree(repo / "mobile_world/src"),
        "environment_preflight_sha256": sha256_file(
            Path(path).parent / "environment_preflight.json"
        ),
    },
}
destination = Path(path)
if destination.is_file():
    existing = json.loads(destination.read_text(encoding="utf-8"))
    common_keys = (
        "model_path",
        "model_name",
        "mode",
        "task_count",
        "samples_per_task",
        "workers",
        "environment_endpoints",
        "model_endpoints",
        "inference_transport",
        "vllm",
    )
    drift = [
        key for key in common_keys if existing.get(key) != payload.get(key)
    ]
    if existing.get("schema_version", 1) >= 2:
        for key in ("model_weight_files", "protocol"):
            if existing.get(key) != payload.get(key):
                drift.append(key)
        stable_provenance = (
            "dataset_sha256",
            "evaluation_entrypoint_sha256",
            "guiagentlab_tree_sha256",
            "mobileworld_revision",
            "mobileworld_src_tree_sha256",
        )
        if any(
            existing.get("provenance", {}).get(key)
            != payload["provenance"].get(key)
            for key in stable_provenance
        ):
            drift.append("provenance")
    if drift:
        raise SystemExit(
            "refusing to resume an evaluation after identity/protocol drift: "
            + ", ".join(sorted(set(drift)))
        )
else:
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
PY

python -m guiagentlab.evaluation.openai_runner \
  --dataset "$dataset" \
  --environment-servers "$environment_servers" \
  --model-servers "$model_servers" \
  --output-dir "$episode_dir" \
  --model-name "$model_name" \
  --model-path "$MODEL_PATH" \
  --experiment-name "$experiment_name" \
  --mode greedy \
  --samples "$expected_samples" \
  --active "$active_containers" \
  --spares "$spare_containers" \
  --workers "$workers" \
  --max-steps "$max_steps" \
  --max-attempts "$max_attempts" \
  --exhausted-task-policy "$exhausted_task_policy" \
  --request-timeout "$request_timeout" \
  --acquire-timeout "$acquire_timeout" \
  --step-wait-time "$step_wait_time" \
  --temperature "$temperature" \
  --top-p 1.0 \
  --top-k -1 \
  --max-tokens "$max_tokens" \
  --history-n 3 \
  2>&1 | tee "$log_dir/$run_id.log"

python -m guiagentlab.evaluation.runner \
  --episodes "$episode_dir" \
  --dataset "$dataset" \
  --expected-tasks "$expected_tasks" \
  --expected-samples "$expected_samples" \
  --mode greedy \
  --output "$summary_dir"

# Keep only durable experiment records on success. Runtime endpoint lists,
# progress and per-replica startup logs are retained automatically on failure.
cleanup
model_pids=()
rm -f "$run_root/progress.json" "$model_servers"
rm -rf "$model_log_dir"
