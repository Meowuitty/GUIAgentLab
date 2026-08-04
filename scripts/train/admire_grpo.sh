#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repo_root/scripts/env/load-project-env.sh"

export TRAINING_ENTRYPOINT="$repo_root/scripts/train/admire_grpo.sh"
export TRAINING_ALGORITHM="admire-grpo"
export ADVANTAGE_ESTIMATOR="admire_grpo"
export TRAIN_FILE="${TRAIN_FILE:-$repo_root/data/train.parquet}"
export OUTPUT_DIR="${OUTPUT_DIR:-${GUIAGENTLAB_STATE_DIR:?Set GUIAGENTLAB_STATE_DIR}/training/checkpoints/admire}"
export EPISODE_DIR="${EPISODE_DIR:-$GUIAGENTLAB_STATE_DIR/training/episodes/admire}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-admire}"
export RUN_ID="${RUN_ID:-admire-grpo-$(date +%Y%m%d-%H%M%S)-$$}"
export SAVE_FREQ="${SAVE_FREQ:-1}"
fixed_checkpoint_frequency="${FIXED_CHECKPOINT_FREQUENCY:-5}"
success_replay_enabled="${SUCCESS_REPLAY_ENABLED:-false}"

if [[ ! "$fixed_checkpoint_frequency" =~ ^[1-9][0-9]*$ ]]; then
  echo "FIXED_CHECKPOINT_FREQUENCY must be a positive integer" >&2
  exit 2
fi

success_replay_args=(+agent_environment.success_replay_enabled=false)
if [[ "$success_replay_enabled" == "true" ]]; then
  success_replay_library="$(realpath "${SUCCESS_REPLAY_LIBRARY:-$repo_root/data/replay/success}")"
  success_replay_milestones="$(realpath "${SUCCESS_REPLAY_MILESTONES:-$success_replay_library/milestones.json}")"
  export TRAINING_EXTRA_DATA_FILES="$success_replay_library/manifest.json:$success_replay_milestones"
  success_replay_args=(
    +agent_environment.success_replay_enabled=true
    "+agent_environment.success_replay_library_path=$success_replay_library"
    "+agent_environment.success_replay_milestones_path=$success_replay_milestones"
  )
elif [[ "$success_replay_enabled" != "false" ]]; then
  echo "SUCCESS_REPLAY_ENABLED must be true or false" >&2
  exit 2
fi

exec "$repo_root/scripts/train/grpo.sh" \
  "+algorithm.admire_milestone_weight=${ADMIRE_MILESTONE_WEIGHT:-0.3}" \
  "+algorithm.admire_milestone_decay=${ADMIRE_MILESTONE_DECAY:-0.99}" \
  "+algorithm.admire_failed_hit_bonus=${ADMIRE_FAILED_HIT_BONUS:-0.5}" \
  "+algorithm.admire_invalid_coefficient=${ADMIRE_INVALID_COEFFICIENT:-0.5}" \
  "+algorithm.admire_invalid_reward=${ADMIRE_INVALID_REWARD:--1.0}" \
  "+algorithm.admire_loop_reward=${ADMIRE_LOOP_REWARD:--0.25}" \
  "+algorithm.admire_successful_terminal_reward=${ADMIRE_SUCCESSFUL_TERMINAL_REWARD:-0.25}" \
  "+algorithm.admire_normalize_std=${ADMIRE_NORMALIZE_STD:-true}" \
  "+algorithm.gui_hierarchical_action_weighting=${ADMIRE_HIERARCHICAL_ACTION_WEIGHTING:-false}" \
  "+agent_environment.loop_guard_enabled=true" \
  "+agent_environment.rolling_checkpoint_retention=true" \
  "+agent_environment.fixed_checkpoint_frequency=$fixed_checkpoint_frequency" \
  "${success_replay_args[@]}" \
  "$@"
