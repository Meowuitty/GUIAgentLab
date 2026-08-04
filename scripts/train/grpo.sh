#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script_path="${TRAINING_ENTRYPOINT:-$repo_root/scripts/train/grpo.sh}"
cd "$repo_root"
source "$repo_root/scripts/env/load-project-env.sh"

# Local MobileWorld APIs must never be sent through the host proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
unset NCCL_P2P_DISABLE

python_site="$(python -c 'import site; print(site.getsitepackages()[0])')"
nvidia_lib_path="$(find "$python_site/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)"
export LD_LIBRARY_PATH="$nvidia_lib_path:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

: "${MODEL_PATH:?MODEL_PATH must point to a model checkpoint}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

# ============ Local paths ============
: "${GUIAGENTLAB_STATE_DIR:?Set GUIAGENTLAB_STATE_DIR to a large local data volume}"
data_root="${GUIAGENTLAB_DATA_ROOT:-$GUIAGENTLAB_STATE_DIR/training}"
train_file="${TRAIN_FILE:-$repo_root/data/train.parquet}"
validation_file="${VALIDATION_FILE:-$repo_root/data/test.parquet}"
server_file="${GUIAGENTLAB_SERVER_FILE:-$repo_root/configs/servers.txt}"
output_dir="${OUTPUT_DIR:-$data_root/checkpoints/grpo}"
log_dir="${LOG_DIR:-$data_root/logs}"
episode_dir="${EPISODE_DIR:-$data_root/episodes/grpo}"
ray_tmp_dir="${RAY_TMP_DIR:-$data_root/ray_tmp}"
ray_tmp_link="${RAY_TMP_LINK:-/tmp/guiagentlab_ray}"
wandb_dir="${WANDB_DIR:-$data_root/wandb}"

mkdir -p "$output_dir" "$log_dir" "$episode_dir" "$ray_tmp_dir" "$wandb_dir"
if ((${#ray_tmp_link} > 24)); then
  echo "RAY_TMP_LINK must be at most 24 characters so Ray's Unix sockets stay below the 107-byte limit: $ray_tmp_link" >&2
  exit 1
fi
if [[ -e "$ray_tmp_link" && ! -L "$ray_tmp_link" ]]; then
  echo "Ray temp path exists and is not a symlink: $ray_tmp_link" >&2
  exit 1
fi
ln -sfn "$ray_tmp_dir" "$ray_tmp_link"
export RAY_TMPDIR="$ray_tmp_link"
export WANDB_DIR="$wandb_dir"
export GUIAGENTLAB_EPISODE_DIR="$episode_dir"
export PYTHONPATH="$repo_root/mobile_world/src:$repo_root/verl:$repo_root${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$train_file" ]] || { echo "Missing training data: $train_file" >&2; exit 1; }
[[ -f "$validation_file" ]] || { echo "Missing validation data: $validation_file" >&2; exit 1; }
[[ -f "$server_file" ]] || { echo "Missing server list: $server_file" >&2; exit 1; }

# ============ Experiment configuration ============
n_gpus="${N_GPUS:-8}"
train_batch_size="${TRAIN_BATCH_SIZE:-4}"
val_batch_size="${VAL_BATCH_SIZE:-1}"
group_size="${GROUP_SIZE:-8}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}"
max_steps="${MAX_STEPS:-50}"
save_freq="${SAVE_FREQ:-5}"
total_epochs="${TOTAL_EPOCHS:-1}"
active_containers="${ACTIVE_CONTAINERS:-32}"
spare_containers="${SPARE_CONTAINERS:-8}"
agent_workers="${AGENT_WORKERS:-$active_containers}"
experiment_name="${EXPERIMENT_NAME:-grpo}"
run_id="${RUN_ID:-grpo-$(date +%Y%m%d-%H%M%S)-$$}"
training_algorithm="${TRAINING_ALGORITHM:-grpo}"
advantage_estimator="${ADVANTAGE_ESTIMATOR:-grpo}"
pool_name="guiagentlab_pool_${run_id//-/_}"

hydra_args=(
  hydra.run.dir=.
  hydra.output_subdir=null
  hydra/job_logging=stdout
  trainer.use_v1=false
  "algorithm.adv_estimator=$advantage_estimator"
  algorithm.use_kl_in_reward=false
  algorithm.norm_adv_by_std_in_grpo=true
  "data.train_files=['$train_file']"
  "data.val_files=['$validation_file']"
  "data.train_batch_size=$train_batch_size"
  "data.val_batch_size=$val_batch_size"
  data.shuffle=false
  data.max_prompt_length=17000
  data.max_response_length=512
  data.filter_overlong_prompts=true
  data.truncation=error
  data.image_key=images
  data.return_raw_chat=true
  "actor_rollout_ref.model.path=$MODEL_PATH"
  actor_rollout_ref.model.trust_remote_code=true
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.optim.lr=1e-6
  "actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_kl_loss=true
  actor_rollout_ref.actor.kl_loss_coef=0.01
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  actor_rollout_ref.actor.shuffle=true
  actor_rollout_ref.actor.fsdp_config.param_offload=false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  actor_rollout_ref.actor.fsdp_config.seed=0
  actor_rollout_ref.actor.data_loader_seed=0
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  "actor_rollout_ref.rollout.n=$group_size"
  actor_rollout_ref.rollout.temperature=0.7
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.seed=0
  actor_rollout_ref.rollout.max_model_len=25600
  actor_rollout_ref.rollout.gpu_memory_utilization=0.65
  actor_rollout_ref.rollout.enable_chunked_prefill=true
  actor_rollout_ref.rollout.enforce_eager=false
  actor_rollout_ref.rollout.free_cache_engine=false
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  "actor_rollout_ref.rollout.agent.num_workers=$agent_workers"
  actor_rollout_ref.rollout.agent.default_agent_loop=mobileworld_agent
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=$repo_root/configs/agent_loops.yaml"
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=guiagentlab.rollout.manager.GUIAgentLoopManager
  "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_steps"
  "actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_steps"
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4
  actor_rollout_ref.rollout.val_kwargs.do_sample=true
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.fsdp_config.param_offload=false
  actor_rollout_ref.ref.fsdp_config.seed=0
  "trainer.default_local_dir=$output_dir"
  "trainer.project_name=${WANDB_PROJECT:-GUIAgentLab-MobileWorld}"
  "trainer.experiment_name=$experiment_name"
  "trainer.n_gpus_per_node=$n_gpus"
  trainer.nnodes=1
  "trainer.save_freq=$save_freq"
  trainer.test_freq=-1
  "trainer.total_epochs=$total_epochs"
  trainer.val_before_train=false
  "trainer.logger=['console','wandb']"
  "+agent_environment.servers_file=$server_file"
  "+agent_environment.active=$active_containers"
  "+agent_environment.spares=$spare_containers"
  "+agent_environment.initialization_concurrency=${INITIALIZATION_CONCURRENCY:-4}"
  "+agent_environment.teardown_concurrency=${TEARDOWN_CONCURRENCY:-1}"
  "+agent_environment.recovery_concurrency=${RECOVERY_CONCURRENCY:-2}"
  +agent_environment.device=emulator-5554
  "+agent_environment.request_timeout=${REQUEST_TIMEOUT:-180}"
  +agent_environment.acquire_timeout=300
  +agent_environment.initialization_timeout=900
  +agent_environment.teardown_timeout=900
  "+agent_environment.step_wait_time=${STEP_WAIT_TIME:-2}"
  +agent_environment.emit_step_samples=true
  "+agent_environment.invalid_action_penalty=${INVALID_ACTION_PENALTY:-0.1}"
  "+agent_environment.ppo_mini_batch_size=$ppo_mini_batch_size"
  "+agent_environment.max_steps=$max_steps"
  +agent_environment.history_length=3
  +agent_environment.allow_ask_user=false
  +agent_environment.max_infrastructure_attempts=3
  "+agent_environment.pool_name=$pool_name"
  "$@"
)
command=(python -m guiagentlab.training.verl_adapter "${hydra_args[@]}")
manifest_data_args=(
  --data-file "$train_file"
  --data-file "$validation_file"
)
if [[ -n "${TRAINING_EXTRA_DATA_FILES:-}" ]]; then
  IFS=: read -r -a extra_data_files <<< "$TRAINING_EXTRA_DATA_FILES"
  for data_file in "${extra_data_files[@]}"; do
    [[ -f "$data_file" ]] || {
      echo "Missing extra run data file: $data_file" >&2
      exit 1
    }
    manifest_data_args+=(--data-file "$data_file")
  done
fi

if ((agent_workers <= 0 || train_batch_size * group_size % agent_workers != 0)); then
  echo "TRAIN_BATCH_SIZE * GROUP_SIZE must be divisible by AGENT_WORKERS" >&2
  exit 1
fi
if ((ppo_mini_batch_size <= 0 || ppo_mini_batch_size % n_gpus != 0)); then
  echo "PPO_MINI_BATCH_SIZE must be positive and divisible by N_GPUS" >&2
  exit 1
fi

manifest_path="$(python -m guiagentlab.training.run_manifest \
  --algorithm "$training_algorithm" \
  --run-id "$run_id" \
  --output-dir "$output_dir" \
  --entrypoint "$script_path" \
  "${manifest_data_args[@]}" \
  --wandb \
  -- "${command[@]}")"
export GUIAGENTLAB_RUN_MANIFEST="$manifest_path"

run_log="$log_dir/${run_id}.log"
echo "Run manifest: $manifest_path"
echo "Training log: $run_log"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

HYDRA_FULL_ERROR=1 "${command[@]}" 2>&1 | tee "$run_log"

# Ray session logs are diagnostics, not experiment records. Keep them when the
# command fails; discard them only after a successful training exit.
rm -rf -- "$ray_tmp_dir"
